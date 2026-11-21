use super::{
    MetricTag, STATUS_INTERNAL_ERROR, STATUS_INVALID_CONFIG, STATUS_INVALID_K,
    STATUS_INVALID_METRIC, STATUS_INVALID_POINTER, STATUS_INVALID_SHAPE, STATUS_NONFINITE_INPUT,
    UmapCapiDenseTransformConfig, catch_result, input_slice, map_knn_error, pointer_is_aligned,
    validate_dense_input, validate_output_shape, validate_required_pointer,
};
use crate::transform::parse_fit_epochs;
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::inverse::{
    DenseInverseError, DenseInverseInput, DenseInverseLayout, DenseInverseOutput,
    InverseLocalScale, InverseTriangulation, dense_inverse_epochs,
    fit_dense_inverse_scales_with_metric, fit_inverse_scales_from_ranked_distances,
    inverse_transform_dense_with_metric,
};
use umap_core::layout::LayoutError;
use umap_core::neighbors::DistanceMetric;
use umap_core::sampling::EdgeSamplingCursor;

/// An exterior query, encoded as `SIZE_MAX` in the C location array.
pub const INVERSE_EXTERIOR: usize = usize::MAX;

/// Read-only dense matrices and caller-provided Delaunay geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDenseInverseInput {
    /// Number of rows in both the training and fitted embedding matrices.
    pub n_training_rows: usize,
    /// Number of queries and simplex locations.
    pub n_query_rows: usize,
    /// Width of the training matrix and the reconstruction.
    pub n_features: usize,
    /// Element count of `training`.
    pub training_len: usize,
    /// Element count of `embedding`.
    pub embedding_len: usize,
    /// Element count of `queries`.
    pub queries_len: usize,
    /// Number of vertex indices in `simplices`.
    pub simplices_len: usize,
    /// Element count of `query_simplices`.
    pub query_simplices_len: usize,
    /// Original row-major feature matrix.
    pub training: *const f32,
    /// Corresponding row-major fitted coordinates.
    pub embedding: *const f32,
    /// Row-major queries in the fitted coordinate space.
    pub queries: *const f32,
    /// Row-major simplex vertices, with `n_components + 1` indices per simplex.
    pub simplices: *const usize,
    /// One simplex index or [`INVERSE_EXTERIOR`] per query.
    pub query_simplices: *const usize,
}

/// One fitted inverse scale pair in original feature space.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiInverseLocalScale {
    /// Finite non-negative distance offset.
    pub rho: f32,
    /// Finite positive exponential scale.
    pub sigma: f32,
}

/// Optional metadata from the fit. Supply exactly one of the two arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UmapCapiInverseScales {
    /// Number of scale pairs in `fitted`.
    pub fitted_len: usize,
    /// Number of distances in `ranked_distances`.
    pub ranked_len: usize,
    /// Supplied row width of `ranked_distances`.
    pub ranked_width: usize,
    /// One fitted scale pair per training row.
    pub fitted: *const UmapCapiInverseLocalScale,
    /// Fitted row-major neighbor distances, including the self position and cutoff decisions.
    pub ranked_distances: *const f32,
}

/// Opaque owning result of [`umap_capi_inverse_transform_dense`].
#[repr(C)]
pub struct UmapCapiDenseInverse {
    _private: [u8; 0],
}

/// Read-only feature reconstructions borrowed from an owning inverse handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UmapCapiDenseInverseView {
    /// Number of query rows.
    pub n_rows: usize,
    /// Number of original features per row.
    pub n_features: usize,
    /// Number of floats in `coordinates`.
    pub coordinates_len: usize,
    /// Number of queries marked exterior by the caller.
    pub exterior_queries: usize,
    /// Borrowed row-major feature reconstructions.
    pub coordinates: *const f32,
}

struct OwnedInverse {
    n_rows: usize,
    n_features: usize,
    output: DenseInverseOutput,
}

fn validate_inverse_input(
    input: &UmapCapiDenseInverseInput,
    n_components: usize,
) -> Result<(), u32> {
    let simplex_width = n_components.checked_add(1).ok_or(STATUS_INVALID_SHAPE)?;
    if n_components < 2
        || input.n_training_rows <= simplex_width
        || input.n_features == 0
        || input.n_query_rows == 0
        || input.simplices_len == 0
    {
        return Err(STATUS_INVALID_SHAPE);
    }
    validate_dense_input(
        input.training,
        input.training_len,
        input.n_training_rows,
        input.n_features,
    )?;
    validate_dense_input(
        input.embedding,
        input.embedding_len,
        input.n_training_rows,
        n_components,
    )?;
    validate_dense_input(
        input.queries,
        input.queries_len,
        input.n_query_rows,
        n_components,
    )?;
    validate_dense_input(
        input.simplices,
        input.simplices_len,
        input.simplices_len / simplex_width,
        simplex_width,
    )?;
    validate_dense_input(
        input.query_simplices,
        input.query_simplices_len,
        input.n_query_rows,
        1,
    )?;
    validate_output_shape::<Option<usize>>(input.n_query_rows, 1)?;
    validate_output_shape::<f32>(input.n_query_rows, input.n_features)
}

unsafe fn read_scales(
    source: *const UmapCapiInverseScales,
    n_training_rows: usize,
    n_neighbors: usize,
) -> Result<UmapCapiInverseScales, u32> {
    if source.is_null() {
        return Ok(UmapCapiInverseScales::default());
    }
    validate_required_pointer(source)?;
    // SAFETY: The caller supplies an initialized scale descriptor.
    let source = unsafe { source.read() };
    match (source.fitted.is_null(), source.ranked_distances.is_null()) {
        (false, true) if source.ranked_len == 0 && source.ranked_width == 0 => {
            validate_dense_input(source.fitted, source.fitted_len, n_training_rows, 1)?;
        }
        (true, false) if source.fitted_len == 0 => {
            if n_neighbors > source.ranked_width {
                return Err(STATUS_INVALID_K);
            }
            validate_dense_input(
                source.ranked_distances,
                source.ranked_len,
                n_training_rows,
                source.ranked_width,
            )?;
        }
        _ => return Err(STATUS_INVALID_CONFIG),
    }
    Ok(source)
}

unsafe fn active_scales(
    source: UmapCapiInverseScales,
    training: &[f32],
    n_features: usize,
    config: &UmapCapiDenseTransformConfig,
    metric: DistanceMetric,
    options: GraphOptions,
) -> Result<Vec<InverseLocalScale>, DenseInverseError> {
    if !source.fitted.is_null() {
        // SAFETY: The caller supplies readable scale pairs with checked shape.
        let fitted = unsafe { input_slice(source.fitted, source.fitted_len) };
        return Ok(fitted
            .iter()
            .map(|scale| InverseLocalScale {
                rho: scale.rho,
                sigma: scale.sigma,
            })
            .collect());
    }
    if !source.ranked_distances.is_null() {
        // SAFETY: The caller supplies readable distances with checked shape.
        let ranked = unsafe { input_slice(source.ranked_distances, source.ranked_len) };
        return fit_inverse_scales_from_ranked_distances(
            ranked,
            source.ranked_width,
            config.n_neighbors,
            options,
        );
    }
    let cutoff = config.disconnection_distance;
    fit_dense_inverse_scales_with_metric(
        training,
        n_features,
        config.n_neighbors,
        metric,
        cutoff.is_finite().then_some(cutoff),
        options,
    )
}

fn map_inverse_error(error: DenseInverseError) -> u32 {
    match error {
        DenseInverseError::Training
        | DenseInverseError::Embedding
        | DenseInverseError::Queries
        | DenseInverseError::Layout(LayoutError::NonFiniteValue { .. }) => STATUS_NONFINITE_INPUT,
        DenseInverseError::Triangulation => STATUS_INVALID_SHAPE,
        DenseInverseError::Neighbors(error) => map_knn_error(error),
        DenseInverseError::LocalScales
        | DenseInverseError::InvalidGradient
        | DenseInverseError::UnsupportedMetric(_)
        | DenseInverseError::NoMembership { .. }
        | DenseInverseError::Sampling(_)
        | DenseInverseError::Layout(LayoutError::InvalidParameter { .. }) => STATUS_INVALID_CONFIG,
        DenseInverseError::Layout(_) => STATUS_INTERNAL_ERROR,
    }
}

/// Reconstruct dense features using supplied Delaunay simplices and query locations.
///
/// # Safety
///
/// `input` and `config` must be aligned and point to initialized descriptors.
/// A non-null `scales` must point to an aligned initialized descriptor. Validated
/// arrays must be readable for their declared lengths and stay unchanged for the
/// call. `out_inverse` must be writable and aligned, outside input storage.
/// Release a successful result exactly once with [`umap_capi_dense_inverse_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_inverse_transform_dense(
    input: *const UmapCapiDenseInverseInput,
    config: *const UmapCapiDenseTransformConfig,
    scales: *const UmapCapiInverseScales,
    out_inverse: *mut *mut UmapCapiDenseInverse,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_inverse)?;
        validate_required_pointer(input)?;
        validate_required_pointer(config)?;
        // SAFETY: The caller supplies initialized descriptors.
        let (input, config) = unsafe { (input.read(), config.read()) };
        validate_inverse_input(&input, config.n_components)?;
        if !(1..=input.n_training_rows).contains(&config.n_neighbors) {
            return Err(STATUS_INVALID_K);
        }
        let metric = MetricTag::parse(config.metric)
            .ok_or(STATUS_INVALID_METRIC)?
            .into_core();
        let options =
            GraphOptions::new(config.local_connectivity, 1.0).map_err(|_| STATUS_INVALID_CONFIG)?;
        if scales.is_null()
            && (config.disconnection_distance.is_nan() || config.disconnection_distance <= 0.0)
        {
            return Err(STATUS_INVALID_CONFIG);
        }
        let fit_epochs = parse_fit_epochs(config.fit_epochs_mode, config.fit_epochs)?;
        let layout = DenseInverseLayout {
            n_epochs: dense_inverse_epochs(fit_epochs, input.n_query_rows),
            initial_alpha: config.learning_rate / 4.0,
            repulsion_strength: config.repulsion_strength,
            negative_sample_rate: config.negative_sample_rate,
            seed: u32::try_from(config.transform_seed).map_err(|_| STATUS_INVALID_CONFIG)?,
        };
        if !config.learning_rate.is_finite()
            || config.learning_rate < 0.0
            || EdgeSamplingCursor::from_positive_interval(1.0, layout.negative_sample_rate).is_err()
        {
            return Err(STATUS_INVALID_CONFIG);
        }

        // SAFETY: A non-null scale descriptor is initialized by the caller.
        let scales = unsafe { read_scales(scales, input.n_training_rows, config.n_neighbors)? };
        // SAFETY: All pointer and length checks passed, and the caller supplies readable arrays.
        let (training, embedding, queries, simplices, locations) = unsafe {
            (
                input_slice(input.training, input.training_len),
                input_slice(input.embedding, input.embedding_len),
                input_slice(input.queries, input.queries_len),
                input_slice(input.simplices, input.simplices_len),
                input_slice(input.query_simplices, input.query_simplices_len),
            )
        };
        let locations: Vec<_> = locations
            .iter()
            .map(|&i| (i != INVERSE_EXTERIOR).then_some(i))
            .collect();
        // SAFETY: The optional metadata arrays were validated by read_scales.
        let scales = if layout.n_epochs > 1 && layout.initial_alpha != 0.0 {
            unsafe { active_scales(scales, training, input.n_features, &config, metric, options) }
                .map_err(map_inverse_error)?
        } else {
            Vec::new()
        };
        let core_input = DenseInverseInput {
            training,
            embedding,
            queries,
            n_features: input.n_features,
            n_components: config.n_components,
            triangulation: InverseTriangulation {
                simplices,
                query_simplices: &locations,
            },
        };
        let output = inverse_transform_dense_with_metric(
            &core_input,
            &scales,
            config.a,
            config.b,
            layout,
            metric,
        )
        .map_err(map_inverse_error)?;
        let result = Box::new(OwnedInverse {
            n_rows: input.n_query_rows,
            n_features: input.n_features,
            output,
        });
        // SAFETY: The checked output slot is writable and disjoint from the inputs.
        unsafe { out_inverse.write(Box::into_raw(result).cast()) };
        Ok(())
    })
}

/// Borrow the reconstructed features and exterior query count from a live inverse handle.
///
/// # Safety
///
/// `inverse` must be a live result of [`umap_capi_inverse_transform_dense`].
/// `out_view` must point to an aligned, writable view outside the handle's storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_inverse_view(
    inverse: *const UmapCapiDenseInverse,
    out_view: *mut UmapCapiDenseInverseView,
) -> u32 {
    catch_result(|| {
        validate_required_pointer(out_view)?;
        let inverse = inverse.cast::<OwnedInverse>();
        validate_required_pointer(inverse)?;
        // SAFETY: The caller supplies a live handle and writable view storage.
        let inverse = unsafe { &*inverse };
        unsafe {
            out_view.write(UmapCapiDenseInverseView {
                n_rows: inverse.n_rows,
                n_features: inverse.n_features,
                coordinates_len: inverse.output.coordinates.len(),
                exterior_queries: inverse.output.exterior_queries,
                coordinates: inverse.output.coordinates.as_ptr(),
            })
        };
        Ok(())
    })
}

/// Release an owning inverse handle and invalidate its coordinate view.
///
/// # Safety
///
/// A non-null handle must be a live result of [`umap_capi_inverse_transform_dense`].
/// All reads of the handle and borrowed coordinates must finish before release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_inverse_release(
    inverse: *mut UmapCapiDenseInverse,
) -> u32 {
    catch_result(|| {
        let inverse = inverse.cast::<OwnedInverse>();
        if !inverse.is_null() {
            if !pointer_is_aligned(inverse) {
                return Err(STATUS_INVALID_POINTER);
            }
            // SAFETY: The caller returns the live allocation from the inverse transform.
            unsafe { drop(Box::from_raw(inverse)) };
        }
        Ok(())
    })
}
