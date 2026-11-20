use super::{
    MetricTag, STATUS_INTERNAL_ERROR, STATUS_INVALID_CONFIG, STATUS_INVALID_K,
    STATUS_INVALID_LENGTH, STATUS_INVALID_METRIC, STATUS_INVALID_POINTER, STATUS_INVALID_SHAPE,
    STATUS_NONFINITE_INPUT, STATUS_OK, catch_status, input_slice, map_knn_error,
    optional_slice_pointer, pointer_is_aligned, slice_size_is_representable, validate_dense_input,
    validate_required_pointer,
};
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::layout::LayoutError;
use umap_core::optimize::OptimizeError;
use umap_core::transform::{
    CooGraph, DenseEmbeddingTransformConfig, DenseEmbeddingTransformError,
    DenseGraphTransformError, TransformInitializationError, transform_dense_embedding_with_options,
    transform_dense_graph_with_options,
};

/// Use the query-count-dependent transform epoch default.
pub const FIT_EPOCHS_DEFAULT: u32 = 0;
/// Divide the caller's fit epoch count by three, rounding down.
pub const FIT_EPOCHS_EXPLICIT: u32 = 1;

/// Borrowed row-major feature matrices for either dense transform.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDenseTransformInput {
    /// Number of fitted rows and graph training columns.
    pub n_training_rows: usize,
    /// Number of held-out rows.
    pub n_query_rows: usize,
    /// Positive feature count shared by both matrices.
    pub n_features: usize,
    /// Element count of `training`.
    pub training_len: usize,
    /// Element count of `queries`.
    pub queries_len: usize,
    /// Read-only fitted feature rows.
    pub training: *const f32,
    /// Read-only query feature rows, optionally null for zero rows.
    pub queries: *const f32,
}

/// Fixed-layout settings for the serial dense embedding transform.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDenseTransformConfig {
    /// Effective number of neighbors, in `1..=n_training_rows`.
    pub n_neighbors: usize,
    /// Positive width of both coordinate matrices.
    pub n_components: usize,
    /// Explicit fit epoch count, or zero when using the default.
    pub fit_epochs: usize,
    /// Finite positive curve coefficient.
    pub a: f64,
    /// Finite positive curve exponent.
    pub b: f64,
    /// Finite non-negative fit learning rate, scaled by one quarter in the transform.
    pub learning_rate: f64,
    /// Finite non-negative repulsion strength.
    pub repulsion_strength: f64,
    /// Finite positive negative-sample rate.
    pub negative_sample_rate: f64,
    /// Finite non-negative training connectivity.
    pub local_connectivity: f64,
    /// Fresh transform seed, with valid values in `0..=UINT32_MAX`.
    pub transform_seed: u64,
    /// Feature metric tag from the C header.
    pub metric: u32,
    /// [`FIT_EPOCHS_DEFAULT`] or [`FIT_EPOCHS_EXPLICIT`].
    pub fit_epochs_mode: u32,
    /// Positive cutoff, or positive infinity for no cutoff.
    pub disconnection_distance: f32,
}

/// Opaque owning result of [`umap_capi_transform_dense_embedding`].
#[repr(C)]
pub struct UmapCapiDenseTransform {
    _private: [u8; 0],
}

/// Caller-owned view of the transformed coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UmapCapiDenseTransformView {
    /// Number of query rows.
    pub n_rows: usize,
    /// Number of coordinates per row.
    pub n_components: usize,
    /// Number of floats in `coordinates`.
    pub coordinates_len: usize,
    /// Borrowed row-major coordinates, or null for zero rows.
    pub coordinates: *const f32,
}

/// Opaque owning result of [`umap_capi_transform_dense_graph`].
#[repr(C)]
pub struct UmapCapiCooGraph {
    _private: [u8; 0],
}

/// Caller-owned view of a directed query-to-training COO graph.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UmapCapiCooView {
    /// Number of query rows.
    pub n_rows: usize,
    /// Number of training columns.
    pub n_cols: usize,
    /// Number of elements in each of the three arrays, including stored zeros.
    pub nnz: usize,
    /// Borrowed query indices, or null for zero stored entries.
    pub row: *const usize,
    /// Borrowed training indices, or null for zero stored entries.
    pub col: *const usize,
    /// Borrowed membership weights, or null for zero stored entries.
    pub data: *const f32,
}

struct OwnedTransform {
    n_rows: usize,
    n_components: usize,
    coordinates: Vec<f32>,
}

fn catch_result(function: impl FnOnce() -> Result<(), u32>) -> u32 {
    catch_status(|| function().map_or_else(|status| status, |()| STATUS_OK))
}

unsafe fn read_transform_input(
    input: *const UmapCapiDenseTransformInput,
) -> Result<UmapCapiDenseTransformInput, u32> {
    validate_required_pointer(input)?;
    // SAFETY: The caller supplies a readable initialized input descriptor.
    let input = unsafe { input.read() };
    if input.n_features == 0 {
        return Err(STATUS_INVALID_SHAPE);
    }
    validate_dense_input(
        input.training,
        input.training_len,
        input.n_training_rows,
        input.n_features,
    )?;
    validate_dense_input(
        input.queries,
        input.queries_len,
        input.n_query_rows,
        input.n_features,
    )?;
    Ok(input)
}

fn validate_output_shape<T>(n_rows: usize, n_cols: usize) -> Result<(), u32> {
    let len = n_rows.checked_mul(n_cols).ok_or(STATUS_INVALID_SHAPE)?;
    if !slice_size_is_representable::<T>(len) {
        return Err(STATUS_INVALID_LENGTH);
    }
    Ok(())
}

fn transform_graph_settings(
    input: &UmapCapiDenseTransformInput,
    n_neighbors: usize,
    metric: u32,
    local_connectivity: f64,
) -> Result<(MetricTag, GraphOptions), u32> {
    let metric = MetricTag::parse(metric).ok_or(STATUS_INVALID_METRIC)?;
    if !(1..=input.n_training_rows).contains(&n_neighbors) {
        return Err(STATUS_INVALID_K);
    }
    validate_output_shape::<usize>(input.n_query_rows, n_neighbors)?;
    let options = GraphOptions::new(local_connectivity, 1.0).map_err(|_| STATUS_INVALID_CONFIG)?;
    Ok((metric, options))
}

fn map_graph_error(error: DenseGraphTransformError) -> u32 {
    match error {
        DenseGraphTransformError::Training(error) | DenseGraphTransformError::Queries(error) => {
            map_knn_error(error)
        }
        DenseGraphTransformError::NoFeatures => STATUS_INVALID_SHAPE,
        DenseGraphTransformError::NonFiniteDistance { .. } => STATUS_NONFINITE_INPUT,
    }
}

fn map_transform_error(error: DenseEmbeddingTransformError) -> u32 {
    use DenseEmbeddingTransformError as Error;
    match error {
        Error::Graph(error) => map_graph_error(error),
        Error::Initialization(TransformInitializationError::InvalidFittedShape { .. }) => {
            STATUS_INVALID_LENGTH
        }
        Error::Initialization(TransformInitializationError::OutputShapeOverflow { .. }) => {
            STATUS_INVALID_SHAPE
        }
        Error::InvalidDisconnectionDistance(_)
        | Error::Optimize(OptimizeError::Layout(LayoutError::InvalidParameter { .. }))
        | Error::Optimize(OptimizeError::Sampling(_)) => STATUS_INVALID_CONFIG,
        Error::Optimize(OptimizeError::Layout(LayoutError::NonFiniteValue { .. })) => {
            STATUS_NONFINITE_INPUT
        }
        _ => STATUS_INTERNAL_ERROR,
    }
}

/// Compute held-out coordinates using a read-only fitted layout.
///
/// # Safety
///
/// `input` and `config` must point to aligned, readable, initialized descriptors.
/// Their feature buffers and `fitted_coordinates` must hold the declared lengths
/// and stay unchanged for the call. `out_transform` must point to an aligned,
/// writable handle slot disjoint from the inputs. Release a successful result
/// exactly once using [`umap_capi_dense_transform_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_transform_dense_embedding(
    input: *const UmapCapiDenseTransformInput,
    fitted_coordinates: *const f32,
    fitted_len: usize,
    config: *const UmapCapiDenseTransformConfig,
    out_transform: *mut *mut UmapCapiDenseTransform,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_transform)?;
        validate_required_pointer(config)?;
        // SAFETY: The caller supplies initialized input and config descriptors.
        let input = unsafe { read_transform_input(input)? };
        let config = unsafe { config.read() };
        let (metric, options) = transform_graph_settings(
            &input,
            config.n_neighbors,
            config.metric,
            config.local_connectivity,
        )?;
        if config.n_components == 0 {
            return Err(STATUS_INVALID_SHAPE);
        }
        validate_dense_input(
            fitted_coordinates,
            fitted_len,
            input.n_training_rows,
            config.n_components,
        )?;
        validate_output_shape::<f32>(input.n_query_rows, config.n_components)?;

        let fit_epochs = match config.fit_epochs_mode {
            FIT_EPOCHS_DEFAULT if config.fit_epochs == 0 => None,
            FIT_EPOCHS_EXPLICIT => Some(config.fit_epochs),
            _ => return Err(STATUS_INVALID_CONFIG),
        };
        let core_config = DenseEmbeddingTransformConfig {
            n_neighbors: config.n_neighbors,
            metric: metric.into_core(),
            n_components: config.n_components,
            a: config.a,
            b: config.b,
            learning_rate: config.learning_rate,
            fit_epochs,
            repulsion_strength: config.repulsion_strength,
            negative_sample_rate: config.negative_sample_rate,
            transform_seed: u32::try_from(config.transform_seed)
                .map_err(|_| STATUS_INVALID_CONFIG)?,
            disconnection_distance: Some(config.disconnection_distance),
        };
        // SAFETY: The pointers and lengths were checked above, and the caller
        // guarantees readable storage for the three arrays.
        let coordinates = unsafe {
            transform_dense_embedding_with_options(
                input_slice(input.training, input.training_len),
                input.n_training_rows,
                input_slice(input.queries, input.queries_len),
                input.n_query_rows,
                input.n_features,
                input_slice(fitted_coordinates, fitted_len),
                &core_config,
                options,
            )
        }
        .map_err(map_transform_error)?;
        let result = Box::new(OwnedTransform {
            n_rows: input.n_query_rows,
            n_components: config.n_components,
            coordinates,
        });
        // SAFETY: The checked output slot is writable and disjoint from the inputs.
        unsafe { out_transform.write(Box::into_raw(result).cast()) };
        Ok(())
    })
}

/// Borrow the transformed coordinates while their handle is live.
///
/// # Safety
///
/// `transform` must be a live handle returned by [`umap_capi_transform_dense_embedding`].
/// `out_view` must point to an aligned, writable view outside the handle's storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_transform_view(
    transform: *const UmapCapiDenseTransform,
    out_view: *mut UmapCapiDenseTransformView,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_view)?;
        let transform = transform.cast::<OwnedTransform>();
        validate_required_pointer(transform)?;
        // SAFETY: The caller supplies a live handle and writable view storage.
        let transform = unsafe { &*transform };
        unsafe {
            out_view.write(UmapCapiDenseTransformView {
                n_rows: transform.n_rows,
                n_components: transform.n_components,
                coordinates_len: transform.coordinates.len(),
                coordinates: optional_slice_pointer(&transform.coordinates),
            })
        };
        Ok(())
    })
}

/// Release a transform handle and invalidate its borrowed coordinates.
///
/// # Safety
///
/// A non-null handle must be a live owning result from [`umap_capi_transform_dense_embedding`].
/// All reads of the handle and borrowed coordinates must finish before release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_transform_release(
    transform: *mut UmapCapiDenseTransform,
) -> u32 {
    catch_result(|| {
        let transform = transform.cast::<OwnedTransform>();
        if !transform.is_null() {
            if !pointer_is_aligned(transform) {
                return Err(STATUS_INVALID_POINTER);
            }
            // SAFETY: The caller returns the live allocation from the transform.
            unsafe { drop(Box::from_raw(transform)) };
        }
        Ok(())
    })
}

/// Construct the directed query-to-training graph with no distance cutoff.
///
/// # Safety
///
/// `input` must be aligned and initialized, and its feature buffers must hold the
/// declared lengths and stay unchanged during the call. `out_graph` must be an
/// aligned writable handle slot disjoint from the inputs. Release a successful
/// result exactly once using [`umap_capi_coo_graph_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_transform_dense_graph(
    input: *const UmapCapiDenseTransformInput,
    n_neighbors: usize,
    metric: u32,
    local_connectivity: f64,
    out_graph: *mut *mut UmapCapiCooGraph,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_graph)?;
        // SAFETY: The caller supplies an initialized input descriptor.
        let input = unsafe { read_transform_input(input)? };
        let (metric, options) =
            transform_graph_settings(&input, n_neighbors, metric, local_connectivity)?;
        // SAFETY: The pointers and lengths were checked above, and the caller
        // guarantees readable storage for both matrices.
        let graph = unsafe {
            transform_dense_graph_with_options(
                input_slice(input.training, input.training_len),
                input.n_training_rows,
                input_slice(input.queries, input.queries_len),
                input.n_query_rows,
                input.n_features,
                n_neighbors,
                metric.into_core(),
                options,
            )
        }
        .map_err(map_graph_error)?;
        // SAFETY: The checked output slot is writable and disjoint from the input.
        unsafe { out_graph.write(Box::into_raw(Box::new(graph)).cast()) };
        Ok(())
    })
}

/// Borrow the three COO arrays while their graph handle is live.
///
/// # Safety
///
/// `graph` must be a live handle returned by [`umap_capi_transform_dense_graph`].
/// `out_view` must point to an aligned, writable view outside the handle's storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_coo_graph_view(
    graph: *const UmapCapiCooGraph,
    out_view: *mut UmapCapiCooView,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_view)?;
        let graph = graph.cast::<CooGraph>();
        validate_required_pointer(graph)?;
        // SAFETY: The caller supplies a live handle and writable view storage.
        let graph = unsafe { &*graph };
        unsafe {
            out_view.write(UmapCapiCooView {
                n_rows: graph.n_rows,
                n_cols: graph.n_cols,
                nnz: graph.data.len(),
                row: optional_slice_pointer(&graph.row),
                col: optional_slice_pointer(&graph.col),
                data: optional_slice_pointer(&graph.data),
            })
        };
        Ok(())
    })
}

/// Release a directed graph handle and invalidate its borrowed COO arrays.
///
/// # Safety
///
/// A non-null handle must be a live owning result from [`umap_capi_transform_dense_graph`].
/// All reads of the handle and borrowed arrays must finish before release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_coo_graph_release(graph: *mut UmapCapiCooGraph) -> u32 {
    catch_result(|| {
        let graph = graph.cast::<CooGraph>();
        if !graph.is_null() {
            if !pointer_is_aligned(graph) {
                return Err(STATUS_INVALID_POINTER);
            }
            // SAFETY: The caller returns the live allocation from the graph transform.
            unsafe { drop(Box::from_raw(graph)) };
        }
        Ok(())
    })
}
