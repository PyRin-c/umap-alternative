//! C ABI for native dense UMAP graph and embedding fits, transforms, and updates.
//!
//! Opaque handles own their result arrays. Callers must keep a handle alive
//! while reading its borrowed views and release it exactly once. The public
//! declarations for C consumers live in `include/umap_capi.h`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

use umap_core::density::DensityError;
use umap_core::densmap::{DensmapError, DensmapOptions, fit_dense_densmap_seeded};
use umap_core::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingFitError, DenseEmbeddingInit,
    fit_dense_embedding_seeded,
};
use umap_core::fit::{DenseGraphFitError, fit_dense_graph_with_options};
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};
use umap_core::initialization::InitializationError;
use umap_core::layout::LayoutError;
use umap_core::neighbors::{DistanceMetric, KnnError};
use umap_core::optimize::OptimizeError;
use umap_core::sampling::{EdgeSamplingCursor, SamplingError};

mod transform;
mod update;
pub use transform::*;
pub use update::*;

/// Successful C ABI operation.
pub const STATUS_OK: u32 = 0;
/// A required pointer argument was null.
pub const STATUS_NULL_ARGUMENT: u32 = 1;
/// A non-null pointer was not aligned for its pointee type.
pub const STATUS_INVALID_POINTER: u32 = 2;
/// A supplied element length did not fit the requested shape or address space.
pub const STATUS_INVALID_LENGTH: u32 = 3;
/// Matrix dimensions could not be represented as a valid input or output shape.
pub const STATUS_INVALID_SHAPE: u32 = 4;
/// The requested neighbor count was outside `1..=n_rows`.
pub const STATUS_INVALID_K: u32 = 5;
/// The metric tag is not one of the documented values.
pub const STATUS_INVALID_METRIC: u32 = 6;
/// A feature or initialization value was not finite.
pub const STATUS_NONFINITE_INPUT: u32 = 7;
/// A handle argument was not a live handle returned by this ABI.
pub const STATUS_INVALID_HANDLE: u32 = 8;
/// A Rust panic was contained at the ABI boundary.
pub const STATUS_PANIC: u32 = 9;
/// An unexpected core error was returned.
pub const STATUS_INTERNAL_ERROR: u32 = 10;
/// A graph, initialization, or optimizer setting was invalid.
pub const STATUS_INVALID_CONFIG: u32 = 11;

/// Generate initial coordinates from the fit seed.
pub const INIT_RANDOM: u32 = 0;
/// Normalize caller-supplied initial coordinates.
pub const INIT_EXPLICIT: u32 = 1;

const VERSION_NUL: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Opaque handle type corresponding to `umap_capi_csr_graph_t` in the header.
///
/// The value is only produced by [`umap_capi_fit_dense_graph`].
#[repr(C)]
pub struct UmapCapiCsrGraph {
    _private: [u8; 0],
}

/// Caller-owned destination for the borrowed arrays in an opaque CSR handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UmapCapiCsrView {
    /// Number of graph rows.
    pub n_rows: usize,
    /// Number of graph columns.
    pub n_cols: usize,
    /// Number of entries in `indptr` (always `n_rows + 1`).
    pub indptr_len: usize,
    /// Number of entries in `indices`.
    pub indices_len: usize,
    /// Number of entries in `data`.
    pub data_len: usize,
    /// Borrowed row offsets, or null when the length is zero.
    pub indptr: *const usize,
    /// Borrowed column indices, or null when the length is zero.
    pub indices: *const usize,
    /// Borrowed membership weights, or null when the length is zero.
    pub data: *const f32,
}

/// Opaque owning result of a dense embedding fit or update.
#[repr(C)]
pub struct UmapCapiDenseEmbedding {
    _private: [u8; 0],
}

/// Fixed-layout configuration for a seeded dense embedding fit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDenseEmbeddingConfig {
    /// Number of neighbors retained per row, in `1..=n_rows`.
    pub n_neighbors: usize,
    /// Positive output component count.
    pub n_components: usize,
    /// Number of optimizer epochs. Zero returns the normalized initialization.
    pub n_epochs: usize,
    /// Finite positive curve coefficient.
    pub a: f64,
    /// Finite positive curve exponent.
    pub b: f64,
    /// Finite non-negative initial learning rate.
    pub learning_rate: f64,
    /// Finite non-negative repulsion strength.
    pub repulsion_strength: f64,
    /// Finite positive negative-sample rate.
    pub negative_sample_rate: f64,
    /// Numeric distance metric tag from the C header.
    pub metric: u32,
    /// [`INIT_RANDOM`] or [`INIT_EXPLICIT`].
    pub init: u32,
    /// Seed for random initialization and sampling, or sampling alone for explicit initialization.
    pub seed: u32,
    /// Number of explicit coordinates, zero for random initialization.
    pub init_len: usize,
    /// Row-major explicit coordinates, null for random initialization.
    pub init_coordinates: *const f32,
}

/// Density correlation settings for a seeded densMAP fit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UmapCapiDensmapOptions {
    /// Finite nonnegative weight for the density objective.
    pub dens_lambda: f64,
    /// Fraction of epochs eligible for density updates, from zero to one.
    pub dens_frac: f64,
    /// Finite nonnegative variance shift.
    pub dens_var_shift: f64,
}

/// Caller-owned destination for a result's borrowed coordinates and CSR graph.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UmapCapiDenseEmbeddingView {
    /// Number of output rows.
    pub n_rows: usize,
    /// Number of coordinates per row.
    pub n_components: usize,
    /// Number of entries in `coordinates`.
    pub coordinates_len: usize,
    /// Borrowed row-major coordinates.
    pub coordinates: *const f32,
    /// Borrowed CSR view of the fitted graph.
    pub graph: UmapCapiCsrView,
}

struct OwnedGraph {
    graph: CsrGraph,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum MetricTag {
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
}

impl MetricTag {
    fn parse(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Euclidean),
            1 => Some(Self::Manhattan),
            2 => Some(Self::Chebyshev),
            _ => None,
        }
    }

    fn into_core(self) -> DistanceMetric {
        match self {
            Self::Euclidean => DistanceMetric::Euclidean,
            Self::Manhattan => DistanceMetric::Manhattan,
            Self::Chebyshev => DistanceMetric::Chebyshev,
        }
    }
}

fn pointer_is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize).is_multiple_of(align_of::<T>())
}

fn slice_size_is_representable<T>(length: usize) -> bool {
    length <= isize::MAX as usize / size_of::<T>()
}

fn optional_slice_pointer<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr()
    }
}

fn csr_view(graph: &CsrGraph) -> UmapCapiCsrView {
    UmapCapiCsrView {
        n_rows: graph.n_rows,
        n_cols: graph.n_cols,
        indptr_len: graph.indptr.len(),
        indices_len: graph.indices.len(),
        data_len: graph.data.len(),
        indptr: optional_slice_pointer(&graph.indptr),
        indices: optional_slice_pointer(&graph.indices),
        data: optional_slice_pointer(&graph.data),
    }
}

fn validate_required_pointer<T>(pointer: *const T) -> Result<(), u32> {
    if pointer.is_null() {
        return Err(STATUS_NULL_ARGUMENT);
    }
    if !pointer_is_aligned(pointer) {
        return Err(STATUS_INVALID_POINTER);
    }
    Ok(())
}

fn validate_dense_input(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
) -> Result<(), u32> {
    if !data.is_null() && !pointer_is_aligned(data) {
        return Err(STATUS_INVALID_POINTER);
    }
    if data_len != 0 && data.is_null() {
        return Err(STATUS_NULL_ARGUMENT);
    }
    if !slice_size_is_representable::<f32>(data_len) {
        return Err(STATUS_INVALID_LENGTH);
    }
    let expected_len = n_rows.checked_mul(n_features).ok_or(STATUS_INVALID_SHAPE)?;
    if expected_len != data_len {
        return Err(STATUS_INVALID_LENGTH);
    }
    Ok(())
}

// The caller must validate nullness, alignment, length and address-space size
// and guarantee readable storage for `len` contiguous values.
unsafe fn input_slice<'a>(data: *const f32, len: usize) -> &'a [f32] {
    if len == 0 {
        &[]
    } else {
        // SAFETY: The caller guarantees the preconditions above.
        unsafe { slice::from_raw_parts(data, len) }
    }
}

fn as_owned_graph<'a>(handle: *const UmapCapiCsrGraph) -> Result<&'a OwnedGraph, u32> {
    if handle.is_null() {
        return Err(STATUS_NULL_ARGUMENT);
    }
    if !pointer_is_aligned(handle.cast::<OwnedGraph>()) {
        return Err(STATUS_INVALID_POINTER);
    }

    // SAFETY: The ABI contract requires a non-null handle to be a live pointer
    // previously returned by a dense graph fit. The alignment was
    // checked above.  The returned reference is borrowed only for this call.
    Ok(unsafe { &*handle.cast::<OwnedGraph>() })
}

#[allow(clippy::too_many_arguments)]
fn fit_dense_graph_impl(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: u32,
    options: Option<GraphOptions>,
    out_graph: *mut *mut UmapCapiCsrGraph,
) -> u32 {
    if let Err(status) = validate_required_pointer(out_graph) {
        return status;
    }
    if let Err(status) = validate_dense_input(data, data_len, n_rows, n_features) {
        return status;
    }

    let Some(metric) = MetricTag::parse(metric) else {
        return STATUS_INVALID_METRIC;
    };

    // SAFETY: The pointer and dimensions were checked above. The caller
    // promises readable storage for `data_len` contiguous values.
    let data = unsafe { input_slice(data, data_len) };

    let graph = match fit_dense_graph_with_options(
        data,
        n_rows,
        n_features,
        k,
        metric.into_core(),
        options,
    ) {
        Ok(graph) => graph,
        Err(error) => return map_core_error(error),
    };

    let handle = Box::into_raw(Box::new(OwnedGraph { graph })).cast::<UmapCapiCsrGraph>();

    // SAFETY: `out_graph` is non-null and aligned, and the caller promises one
    // writable handle slot.  No output is written before all input checks and
    // core fitting have succeeded.
    unsafe { out_graph.write(handle) };
    STATUS_OK
}

fn map_core_error(error: DenseGraphFitError) -> u32 {
    match error {
        DenseGraphFitError::Knn(error) => map_knn_error(error),
        DenseGraphFitError::Graph(_) => STATUS_INTERNAL_ERROR,
    }
}

fn map_knn_error(error: KnnError) -> u32 {
    match error {
        KnnError::InvalidShape { .. } => STATUS_INVALID_LENGTH,
        KnnError::InvalidK { .. } => STATUS_INVALID_K,
        KnnError::NonFiniteValue { .. } => STATUS_NONFINITE_INPUT,
    }
}

fn map_embedding_error(error: DenseEmbeddingFitError) -> u32 {
    match error {
        DenseEmbeddingFitError::Graph(error) => map_core_error(error),
        DenseEmbeddingFitError::Initialization(InitializationError::InvalidShape { .. }) => {
            STATUS_INVALID_LENGTH
        }
        DenseEmbeddingFitError::Initialization(InitializationError::ShapeOverflow { .. }) => {
            STATUS_INVALID_SHAPE
        }
        DenseEmbeddingFitError::Initialization(InitializationError::NonFiniteValue { .. })
        | DenseEmbeddingFitError::Optimize(OptimizeError::Layout(LayoutError::NonFiniteValue {
            ..
        })) => STATUS_NONFINITE_INPUT,
        DenseEmbeddingFitError::Initialization(InitializationError::ZeroColumnSpan { .. })
        | DenseEmbeddingFitError::Optimize(OptimizeError::Layout(
            LayoutError::InvalidParameter { .. },
        ))
        | DenseEmbeddingFitError::Optimize(OptimizeError::Sampling(
            SamplingError::InvalidNegativeSampleRate { .. }
            | SamplingError::InvalidNegativeInterval { .. }
            | SamplingError::NegativeScheduleOverflow { .. },
        )) => STATUS_INVALID_CONFIG,
        DenseEmbeddingFitError::Optimize(_) => STATUS_INTERNAL_ERROR,
    }
}

fn map_densmap_error(error: DensmapError) -> u32 {
    match error {
        DensmapError::InvalidOption { .. } => STATUS_INVALID_CONFIG,
        DensmapError::Preparation(error) => map_embedding_error(error),
        DensmapError::OriginalRadii(DensityError::Matrix(error)) => map_knn_error(error),
        DensmapError::OriginalRadii(DensityError::InvalidDistance { distance, .. })
            if !distance.is_finite() =>
        {
            STATUS_NONFINITE_INPUT
        }
        DensmapError::OriginalRadii(_) => STATUS_INTERNAL_ERROR,
        DensmapError::Optimize(error) => map_embedding_error(error.into()),
    }
}

unsafe fn fit_dense_embedding_impl(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    config: *const UmapCapiDenseEmbeddingConfig,
    density: Option<DensmapOptions>,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> Result<(), u32> {
    validate_required_pointer(out_embedding)?;
    validate_required_pointer(config)?;
    validate_dense_input(data, data_len, n_rows, n_features)?;

    // SAFETY: The pointer was checked above. The caller promises storage for
    // one initialized configuration. The tags are integers and checked below.
    let config = unsafe { config.read() };
    let metric = MetricTag::parse(config.metric).ok_or(STATUS_INVALID_METRIC)?;
    if !(1..=n_rows).contains(&config.n_neighbors) {
        return Err(STATUS_INVALID_K);
    }
    let coordinates_len = n_rows
        .checked_mul(config.n_components)
        .filter(|&len| len != 0)
        .ok_or(STATUS_INVALID_SHAPE)?;
    if !slice_size_is_representable::<f32>(coordinates_len) {
        return Err(STATUS_INVALID_LENGTH);
    }
    if ![config.a, config.b]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
        || ![config.learning_rate, config.repulsion_strength]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
        || EdgeSamplingCursor::from_positive_interval(1.0, config.negative_sample_rate).is_err()
    {
        return Err(STATUS_INVALID_CONFIG);
    }

    let init = match config.init {
        INIT_RANDOM if config.init_coordinates.is_null() && config.init_len == 0 => {
            DenseEmbeddingInit::random(config.seed)
        }
        INIT_EXPLICIT => {
            validate_required_pointer(config.init_coordinates)?;
            if config.init_len != coordinates_len {
                return Err(STATUS_INVALID_LENGTH);
            }
            // SAFETY: The pointer is aligned and non-null. The validated shape
            // determines the checked length, and the caller promises that
            // many readable coordinates.
            let coordinates = unsafe { input_slice(config.init_coordinates, config.init_len) };
            DenseEmbeddingInit::explicit(coordinates)
        }
        _ => return Err(STATUS_INVALID_CONFIG),
    };
    let core_config = DenseEmbeddingConfig::<f64, f64> {
        n_neighbors: config.n_neighbors,
        metric: metric.into_core(),
        n_components: config.n_components,
        a: config.a,
        b: config.b,
        learning_rate: config.learning_rate,
        n_epochs: config.n_epochs,
        repulsion_strength: config.repulsion_strength,
        negative_sample_rate: config.negative_sample_rate,
        init,
    };
    // SAFETY: The pointer and dimensions were checked above. The caller
    // promises readable storage for `data_len` contiguous values.
    let data = unsafe { input_slice(data, data_len) };
    let embedding = if let Some(options) = density {
        fit_dense_densmap_seeded(data, n_rows, n_features, &core_config, config.seed, options)
            .map_err(map_densmap_error)?
    } else {
        fit_dense_embedding_seeded(data, n_rows, n_features, &core_config, config.seed)
            .map_err(map_embedding_error)?
    };
    let handle = Box::into_raw(Box::new(embedding)).cast::<UmapCapiDenseEmbedding>();

    // SAFETY: The output pointer was checked above, and the caller promises
    // writable storage for one handle. The fit has succeeded.
    unsafe { out_embedding.write(handle) };
    Ok(())
}

fn catch_status<F>(function: F) -> u32
where
    F: FnOnce() -> u32,
{
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

/// Return the package version as a static NUL-terminated string.
#[unsafe(no_mangle)]
pub extern "C" fn umap_capi_version() -> *const std::ffi::c_char {
    VERSION_NUL.as_ptr().cast()
}

/// Fit a dense row-major matrix into an owning opaque CSR graph handle.
///
/// # Safety
///
/// `out_graph` must be non-null, aligned for a pointer, and point to writable
/// storage for one handle.  `data` must be aligned for `f32` and point to
/// `data_len` readable contiguous values unless `data_len` is zero, in which
/// case it may be null.  The dimensions must describe exactly that many
/// values.  `metric` must be one of the documented numeric tags.  The caller
/// must release a successful output exactly once with
/// [`umap_capi_csr_graph_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_fit_dense_graph(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: u32,
    out_graph: *mut *mut UmapCapiCsrGraph,
) -> u32 {
    catch_status(|| {
        fit_dense_graph_impl(
            data, data_len, n_rows, n_features, k, metric, None, out_graph,
        )
    })
}

/// Fit a dense CSR graph with explicit connectivity and fuzzy set mixing.
///
/// Invalid graph settings return [`STATUS_INVALID_CONFIG`] and leave the
/// output slot untouched.
///
/// # Safety
///
/// The data and output slot must meet the same requirements as
/// [`umap_capi_fit_dense_graph`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_fit_dense_graph_with_options(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: u32,
    local_connectivity: f64,
    set_op_mix_ratio: f64,
    out_graph: *mut *mut UmapCapiCsrGraph,
) -> u32 {
    catch_status(|| {
        let Ok(options) = GraphOptions::new(local_connectivity, set_op_mix_ratio) else {
            return STATUS_INVALID_CONFIG;
        };
        fit_dense_graph_impl(
            data,
            data_len,
            n_rows,
            n_features,
            k,
            metric,
            Some(options),
            out_graph,
        )
    })
}

/// Fill a caller-owned view with borrowed pointers into an opaque CSR handle.
///
/// # Safety
///
/// `graph` must be a live, aligned handle returned by
/// [`umap_capi_fit_dense_graph`] or [`umap_capi_fit_dense_graph_with_options`].
/// `out_view` must be non-null, aligned for
/// [`UmapCapiCsrView`], and point to writable storage for one complete view.
/// The returned pointers must not be read after the handle is released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_csr_graph_view(
    graph: *const UmapCapiCsrGraph,
    out_view: *mut UmapCapiCsrView,
) -> u32 {
    catch_status(|| {
        if out_view.is_null() {
            return STATUS_NULL_ARGUMENT;
        }
        if !pointer_is_aligned(out_view) {
            return STATUS_INVALID_POINTER;
        }
        let graph = match as_owned_graph(graph) {
            Ok(graph) => graph,
            Err(status) => return status,
        };

        // SAFETY: The pointer is non-null and aligned, and the caller
        // promises writable storage for one `UmapCapiCsrView`.
        unsafe { out_view.write(csr_view(&graph.graph)) };
        STATUS_OK
    })
}

/// Release an owning opaque CSR handle.
///
/// # Safety
///
/// `graph` may be null, which is a no-op.  Otherwise it must be a live,
/// aligned handle returned by [`umap_capi_fit_dense_graph`] and must not have
/// been released previously.  All view pointers borrowed from it become
/// invalid when this function returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_csr_graph_release(graph: *mut UmapCapiCsrGraph) -> u32 {
    catch_status(|| {
        if graph.is_null() {
            return STATUS_OK;
        }
        if !pointer_is_aligned(graph.cast::<OwnedGraph>()) {
            return STATUS_INVALID_POINTER;
        }

        // SAFETY: The ABI contract requires a live handle returned by this
        // crate.  `Box::from_raw` takes back exactly the allocation produced by
        // `Box::into_raw` in `umap_capi_fit_dense_graph`.
        unsafe { drop(Box::from_raw(graph.cast::<OwnedGraph>())) };
        STATUS_OK
    })
}

/// Fit a seeded dense embedding and its fuzzy CSR graph into one owning handle.
///
/// On error, the output slot is left untouched. The fit retains no pointers
/// to the configuration or input buffers.
///
/// # Safety
///
/// `data` must point to `data_len` readable aligned values, or be null if
/// `data_len` is zero. `config` must point to one readable, initialized,
/// aligned configuration. With explicit initialization, its coordinates must
/// point to `init_len` readable aligned values. Input storage must remain
/// readable and unmodified for the call. `out_embedding` must point to one
/// writable aligned handle slot disjoint from input storage. Release a
/// successful output exactly once with [`umap_capi_dense_embedding_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_fit_dense_embedding_seeded(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    config: *const UmapCapiDenseEmbeddingConfig,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> u32 {
    catch_status(|| {
        // SAFETY: The caller guarantees valid storage as documented above.
        unsafe {
            fit_dense_embedding_impl(
                data,
                data_len,
                n_rows,
                n_features,
                config,
                None,
                out_embedding,
            )
        }
        .map_or_else(|status| status, |()| STATUS_OK)
    })
}

/// Fit a seeded dense embedding with the densMAP density objective.
///
/// Returns the fitted CSR and coordinates through the existing embedding handle.
/// Failure leaves the output slot untouched.
///
/// # Safety
///
/// Follow the storage and handle rules of [`umap_capi_fit_dense_embedding_seeded`].
/// `density` must also point to one readable, aligned configuration whose
/// storage stays valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_fit_dense_densmap_seeded(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    config: *const UmapCapiDenseEmbeddingConfig,
    density: *const UmapCapiDensmapOptions,
    out_embedding: *mut *mut UmapCapiDenseEmbedding,
) -> u32 {
    catch_status(|| {
        if let Err(status) = validate_required_pointer(density) {
            return status;
        }
        // SAFETY: The pointer is aligned and the caller supplies initialized storage.
        let density = unsafe { density.read() };
        let density = DensmapOptions {
            dens_lambda: density.dens_lambda,
            dens_frac: density.dens_frac,
            dens_var_shift: density.dens_var_shift,
        };
        // SAFETY: The caller supplies the readable input and writable output slots.
        unsafe {
            fit_dense_embedding_impl(
                data,
                data_len,
                n_rows,
                n_features,
                config,
                Some(density),
                out_embedding,
            )
        }
        .map_or_else(|status| status, |()| STATUS_OK)
    })
}

/// Fill a caller-owned view of the coordinates and graph in one embedding handle.
///
/// # Safety
///
/// `embedding` must be a live handle returned by a dense embedding fit or
/// update. `out_view` must point to writable
/// aligned storage for one [`UmapCapiDenseEmbeddingView`] outside the handle's
/// storage. The borrowed arrays must only be read while the handle is live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_embedding_view(
    embedding: *const UmapCapiDenseEmbedding,
    out_view: *mut UmapCapiDenseEmbeddingView,
) -> u32 {
    catch_status(|| {
        if let Err(status) = validate_required_pointer(out_view) {
            return status;
        }
        let embedding = embedding.cast::<DenseEmbedding>();
        if let Err(status) = validate_required_pointer(embedding) {
            return status;
        }

        // SAFETY: The pointer was checked above, and a non-null handle must
        // refer to a live DenseEmbedding allocated by this crate.
        let embedding = unsafe { &*embedding };
        let (n_rows, n_components) = embedding.shape;
        let view = UmapCapiDenseEmbeddingView {
            n_rows,
            n_components,
            coordinates_len: embedding.coordinates.len(),
            coordinates: optional_slice_pointer(&embedding.coordinates),
            graph: csr_view(&embedding.graph),
        };

        // SAFETY: The pointer was checked above, and the caller promises
        // writable storage for one embedding view.
        unsafe { out_view.write(view) };
        STATUS_OK
    })
}

/// Release the embedding, its graph, and all associated arrays.
///
/// # Safety
///
/// `embedding` may be null, which is a no-op. Otherwise it must be a live
/// handle returned by a dense embedding fit or update. Access to the
/// handle and its borrowed arrays must end before release. The handle may
/// only be released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn umap_capi_dense_embedding_release(
    embedding: *mut UmapCapiDenseEmbedding,
) -> u32 {
    catch_status(|| {
        if embedding.is_null() {
            return STATUS_OK;
        }
        let embedding = embedding.cast::<DenseEmbedding>();
        if !pointer_is_aligned(embedding) {
            return STATUS_INVALID_POINTER;
        }

        // SAFETY: The caller returns ownership of the exact allocation from
        // `Box::into_raw` in a dense embedding fit or update.
        unsafe { drop(Box::from_raw(embedding)) };
        STATUS_OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIES_DATA: [f32; 16] = [
        0.0, 0.0, // 0
        0.0, 0.0, // 1
        1.0, 0.0, // 2
        1.0, 0.0, // 3
        0.0, 1.0, // 4
        0.0, 1.0, // 5
        1.0, 1.0, // 6
        1.0, 1.0, // 7
    ];

    #[test]
    fn version_is_static_and_nul_terminated() {
        let pointer = umap_capi_version();
        assert!(!pointer.is_null());
        // SAFETY: `umap_capi_version` returns a static NUL-terminated string.
        let version = unsafe { std::ffi::CStr::from_ptr(pointer) };
        assert_eq!(
            version.to_str().expect("version is UTF-8"),
            env!("CARGO_PKG_VERSION")
        );
    }

    #[test]
    fn fits_and_exposes_borrowed_csr_view() {
        let mut handle = std::ptr::null_mut();
        // SAFETY: all pointers refer to valid writable/readable Rust storage.
        let status = unsafe {
            umap_capi_fit_dense_graph(
                TIES_DATA.as_ptr(),
                TIES_DATA.len(),
                8,
                2,
                4,
                MetricTag::Euclidean as u32,
                &mut handle,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert!(!handle.is_null());

        let mut view = UmapCapiCsrView::default();
        // SAFETY: `handle` is live and `view` is writable Rust storage.
        let status = unsafe { umap_capi_csr_graph_view(handle, &mut view) };
        assert_eq!(status, STATUS_OK);
        assert_eq!((view.n_rows, view.n_cols), (8, 8));
        assert_eq!(view.indptr_len, 9);
        assert_eq!(view.indices_len, 32);
        assert_eq!(view.data_len, 32);

        // SAFETY: lengths and pointers came from the live handle's view.
        let indptr = unsafe { slice::from_raw_parts(view.indptr, view.indptr_len) };
        // SAFETY: lengths and pointers came from the live handle's view.
        let indices = unsafe { slice::from_raw_parts(view.indices, view.indices_len) };
        // SAFETY: lengths and pointers came from the live handle's view.
        let data = unsafe { slice::from_raw_parts(view.data, view.data_len) };
        assert_eq!(indptr, &[0, 5, 10, 15, 20, 23, 26, 29, 32]);
        assert_eq!(indices[0..5], [1, 2, 3, 4, 5]);
        assert!(data.iter().all(|weight| *weight == 1.0));

        // SAFETY: `handle` is live and is released exactly once.
        assert_eq!(unsafe { umap_capi_csr_graph_release(handle) }, STATUS_OK);
    }

    #[test]
    fn rejects_invalid_inputs_without_touching_output() {
        let mut handle = std::ptr::null_mut();
        // SAFETY: `&mut handle` is a valid output slot; the null input is
        // intentionally paired with a non-zero length to exercise validation.
        let status = unsafe {
            umap_capi_fit_dense_graph(
                std::ptr::null(),
                1,
                1,
                1,
                1,
                MetricTag::Euclidean as u32,
                &mut handle,
            )
        };
        assert_eq!(status, STATUS_NULL_ARGUMENT);
        assert!(handle.is_null());

        let sentinel = std::ptr::dangling_mut::<UmapCapiCsrGraph>();
        handle = sentinel;
        // SAFETY: the output slot is valid; dimensions intentionally disagree
        // with the explicit input length, so the data pointer is not read.
        let status = unsafe {
            umap_capi_fit_dense_graph(
                [1.0_f32, 2.0].as_ptr(),
                2,
                2,
                2,
                1,
                MetricTag::Euclidean as u32,
                &mut handle,
            )
        };
        assert_eq!(status, STATUS_INVALID_LENGTH);
        assert_eq!(handle, sentinel);

        // SAFETY: null output is explicitly tested and no dereference occurs.
        let status = unsafe {
            umap_capi_fit_dense_graph(
                std::ptr::null(),
                0,
                0,
                0,
                1,
                MetricTag::Euclidean as u32,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, STATUS_NULL_ARGUMENT);
    }

    #[test]
    fn supports_empty_feature_rows_without_a_data_pointer() {
        let mut handle = std::ptr::null_mut();
        // SAFETY: null is valid for the explicitly empty input; output is
        // writable Rust storage.
        let status = unsafe {
            umap_capi_fit_dense_graph(
                std::ptr::null(),
                0,
                1,
                0,
                1,
                MetricTag::Euclidean as u32,
                &mut handle,
            )
        };
        assert_eq!(status, STATUS_OK);

        let mut view = UmapCapiCsrView::default();
        // SAFETY: handle is live and view is writable.
        assert_eq!(
            unsafe { umap_capi_csr_graph_view(handle, &mut view) },
            STATUS_OK
        );
        assert_eq!(view.indptr_len, 2);
        assert_eq!(view.indices_len, 0);
        assert_eq!(view.data_len, 0);
        assert!(!view.indptr.is_null());
        assert!(view.indices.is_null());
        assert!(view.data.is_null());

        // SAFETY: handle is live and released exactly once.
        assert_eq!(unsafe { umap_capi_csr_graph_release(handle) }, STATUS_OK);
    }

    #[test]
    fn rejects_unknown_metric_before_fitting() {
        let mut handle = std::ptr::null_mut();
        // SAFETY: all non-null pointers refer to valid storage.
        let status = unsafe {
            umap_capi_fit_dense_graph([0.0_f32].as_ptr(), 1, 1, 1, 1, u32::MAX, &mut handle)
        };
        assert_eq!(status, STATUS_INVALID_METRIC);
        assert!(handle.is_null());
    }
}
