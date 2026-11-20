use super::{
    MetricTag, STATUS_INVALID_CONFIG, STATUS_INVALID_K, STATUS_INVALID_LENGTH,
    STATUS_INVALID_METRIC, STATUS_INVALID_SHAPE, STATUS_NONFINITE_INPUT, STATUS_OK,
    UmapCapiDenseEmbedding, catch_status, input_slice, map_core_error, map_embedding_error,
    slice_size_is_representable, validate_dense_input, validate_required_pointer,
};
use umap_core::embedding::{DenseEmbedding, DenseEmbeddingFitError, EmbeddingLayoutConfig};
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::neighbors::DistanceMetric;
use umap_core::update::{DenseUpdateConfig, DenseUpdateError, update_dense_embedding};

/// Layout and fuzzy graph settings for appending dense feature rows.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDenseUpdateConfig {
    /// Effective neighbor count retained from the fit.
    pub n_neighbors: usize,
    /// Number of optimization epochs. Zero returns the initialized layout.
    pub n_epochs: usize,
    /// Fitted positive curve coefficient.
    pub a: f64,
    /// Fitted positive curve exponent.
    pub b: f64,
    /// Non-negative learning rate retained from the fit.
    pub learning_rate: f64,
    /// Non-negative update repulsion strength.
    pub repulsion_strength: f64,
    /// Positive negative-sample rate for the update.
    pub negative_sample_rate: f64,
    /// Non-negative local connectivity.
    pub local_connectivity: f64,
    /// Fuzzy union-to-intersection mix in the inclusive range zero to one.
    pub set_op_mix_ratio: f64,
    /// Seed for update initialization and negative sampling.
    pub transform_seed: u32,
}

fn map_update_error(error: DenseUpdateError) -> u32 {
    match error {
        DenseUpdateError::InvalidDataShape { .. } | DenseUpdateError::LayoutWidth { .. } => {
            STATUS_INVALID_SHAPE
        }
        DenseUpdateError::Graph(error) => map_core_error(error),
        DenseUpdateError::CoincidentNonFiniteLayout { .. } => STATUS_NONFINITE_INPUT,
        DenseUpdateError::Initialization(error) => {
            map_embedding_error(DenseEmbeddingFitError::Initialization(error))
        }
        DenseUpdateError::Optimization(error) => {
            map_embedding_error(DenseEmbeddingFitError::Optimize(error))
        }
    }
}

/// Append dense Euclidean features to a fitted layout and return a new owning handle.
///
/// Supply the combined feature matrix with fitted rows first. The existing
/// handle and all input buffers remain read-only. This call writes the output
/// slot only after the updated coordinates and graph are available.
///
/// # Safety
///
/// `fitted` must be a live handle returned by a dense embedding fit or update.
/// `data` and `config` must point to aligned, initialized storage for
/// `data_len` floats and one configuration, respectively. All inputs must
/// remain readable and unchanged for the duration of the call. `out_embedding`
/// must be a writable aligned pointer slot disjoint from the input storage.
/// Release each live handle with [`super::umap_capi_dense_embedding_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_update_dense_embedding(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    fitted: *const UmapCapiDenseEmbedding,
    config: *const UmapCapiDenseUpdateConfig,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> u32 {
    // SAFETY: This wrapper forwards the caller's storage contract unchanged.
    unsafe {
        umap_capi_update_dense_embedding_with_metric(
            data,
            data_len,
            n_rows,
            n_features,
            MetricTag::Euclidean as u32,
            fitted,
            config,
            out_embedding,
        )
    }
}

/// Append rows using the same dense metric as the original fit.
///
/// The combined graph is rebuilt without a distance cutoff.
///
/// # Safety
///
/// The storage and handle requirements match [the Euclidean update](umap_capi_update_dense_embedding).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn umap_capi_update_dense_embedding_with_metric(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    metric: u32,
    fitted: *const UmapCapiDenseEmbedding,
    config: *const UmapCapiDenseUpdateConfig,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> u32 {
    catch_status(|| {
        let Some(metric) = MetricTag::parse(metric) else {
            return STATUS_INVALID_METRIC;
        };
        // SAFETY: The ABI caller guarantees the documented storage lifetimes.
        unsafe {
            update_dense_impl(
                data,
                data_len,
                n_rows,
                n_features,
                metric.into_core(),
                fitted,
                config,
                out_embedding,
            )
        }
        .map_or_else(|status| status, |()| STATUS_OK)
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn update_dense_impl(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    metric: DistanceMetric,
    fitted: *const UmapCapiDenseEmbedding,
    config: *const UmapCapiDenseUpdateConfig,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> Result<(), u32> {
    validate_required_pointer(out_embedding)?;
    validate_required_pointer(config)?;
    let fitted = fitted.cast::<DenseEmbedding>();
    validate_required_pointer(fitted)?;
    validate_dense_input(data, data_len, n_rows, n_features)?;

    // SAFETY: Required pointers are non-null and aligned. The ABI caller
    // guarantees that the fit or update handle and configuration are live.
    let (fitted, config) = unsafe { (&*fitted, config.read()) };
    let (n_existing, n_components) = fitted.shape;
    if n_existing >= n_rows || n_features == 0 {
        return Err(STATUS_INVALID_SHAPE);
    }
    if !(1..=n_existing).contains(&config.n_neighbors) {
        return Err(STATUS_INVALID_K);
    }
    let coordinates_len = n_rows
        .checked_mul(n_components)
        .ok_or(STATUS_INVALID_SHAPE)?;
    if !slice_size_is_representable::<f32>(coordinates_len) {
        return Err(STATUS_INVALID_LENGTH);
    }
    let graph_options = GraphOptions::new(config.local_connectivity, config.set_op_mix_ratio)
        .map_err(|_| STATUS_INVALID_CONFIG)?;
    let core_config = DenseUpdateConfig {
        n_neighbors: config.n_neighbors,
        metric,
        n_components,
        graph_options: (graph_options != GraphOptions::default()).then_some(graph_options),
        transform_seed: config.transform_seed,
    };
    let layout = EmbeddingLayoutConfig {
        n_components,
        a: config.a,
        b: config.b,
        learning_rate: config.learning_rate,
        n_epochs: config.n_epochs,
        repulsion_strength: config.repulsion_strength,
        negative_sample_rate: config.negative_sample_rate,
    };
    // SAFETY: Length, pointer, and shape have been validated. The caller
    // guarantees that data_len contiguous values can be read.
    let data = unsafe { input_slice(data, data_len) };
    let result = update_dense_embedding(
        data,
        n_rows,
        n_features,
        n_existing,
        &fitted.coordinates,
        &core_config,
        &layout,
    )
    .map_err(map_update_error)?;
    let handle = Box::into_raw(Box::new(result)).cast::<UmapCapiDenseEmbedding>();

    // SAFETY: The caller guarantees one writable output slot, and all input
    // reads have finished.
    unsafe { out_embedding.write(handle) };
    Ok(())
}
