//! Minimal C ABI for the native dense UMAP graph primitive.
//!
//! The ABI intentionally exposes one owning opaque CSR handle and one borrowed
//! view.  The handle owns all CSR arrays; callers must keep it alive while
//! reading a view and release it exactly once.  The public declarations for C
//! consumers live in `include/umap_capi.h`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

use umap_core::fit::DenseGraphFitError;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::DistanceMetric;

/// Successful C ABI operation.
pub const STATUS_OK: u32 = 0;
/// A required pointer argument was null.
pub const STATUS_NULL_ARGUMENT: u32 = 1;
/// A non-null pointer was not aligned for its pointee type.
pub const STATUS_INVALID_POINTER: u32 = 2;
/// A supplied element length did not fit the requested shape or address space.
pub const STATUS_INVALID_LENGTH: u32 = 3;
/// Matrix dimensions could not be represented as a valid dense input shape.
pub const STATUS_INVALID_SHAPE: u32 = 4;
/// The requested neighbor count was outside `1..=n_rows`.
pub const STATUS_INVALID_K: u32 = 5;
/// The metric tag is not one of the documented values.
pub const STATUS_INVALID_METRIC: u32 = 6;
/// A matrix element was not finite.
pub const STATUS_NONFINITE_INPUT: u32 = 7;
/// A handle argument was not a live handle returned by this ABI.
pub const STATUS_INVALID_HANDLE: u32 = 8;
/// A Rust panic was contained at the ABI boundary.
pub const STATUS_PANIC: u32 = 9;
/// An unexpected core graph error was returned.
pub const STATUS_INTERNAL_ERROR: u32 = 10;

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

fn as_owned_graph<'a>(handle: *const UmapCapiCsrGraph) -> Result<&'a OwnedGraph, u32> {
    if handle.is_null() {
        return Err(STATUS_NULL_ARGUMENT);
    }
    if !pointer_is_aligned(handle.cast::<OwnedGraph>()) {
        return Err(STATUS_INVALID_POINTER);
    }

    // SAFETY: The ABI contract requires a non-null handle to be a live pointer
    // previously returned by `umap_capi_fit_dense_graph`; the alignment was
    // checked above.  The returned reference is borrowed only for this call.
    Ok(unsafe { &*handle.cast::<OwnedGraph>() })
}

fn fit_dense_graph_impl(
    data: *const f32,
    data_len: usize,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: u32,
    out_graph: *mut *mut UmapCapiCsrGraph,
) -> u32 {
    if out_graph.is_null() {
        return STATUS_NULL_ARGUMENT;
    }
    if !pointer_is_aligned(out_graph) {
        return STATUS_INVALID_POINTER;
    }
    if !data.is_null() && !pointer_is_aligned(data) {
        return STATUS_INVALID_POINTER;
    }
    if data_len != 0 {
        if data.is_null() {
            return STATUS_NULL_ARGUMENT;
        }
        if !slice_size_is_representable::<f32>(data_len) {
            return STATUS_INVALID_LENGTH;
        }
    }

    let Some(expected_len) = n_rows.checked_mul(n_features) else {
        return STATUS_INVALID_SHAPE;
    };
    if expected_len != data_len {
        return STATUS_INVALID_LENGTH;
    }
    if !slice_size_is_representable::<f32>(expected_len) {
        return STATUS_INVALID_LENGTH;
    }

    let Some(metric) = MetricTag::parse(metric) else {
        return STATUS_INVALID_METRIC;
    };

    // A null pointer is accepted only for the empty matrix.  Avoid calling
    // `from_raw_parts` for that case because even a zero-length slice requires
    // a non-null, correctly aligned pointer.
    let data = if data_len == 0 {
        &[]
    } else {
        // SAFETY: Nullness, alignment, element count, and address-space size
        // were validated above.  The caller additionally promises that this
        // region is readable for `data_len` contiguous f32 values.
        unsafe { slice::from_raw_parts(data, data_len) }
    };

    let graph =
        match umap_core::fit::fit_dense_graph(data, n_rows, n_features, k, metric.into_core()) {
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
        DenseGraphFitError::Knn(error) => match error {
            umap_core::neighbors::KnnError::InvalidShape { .. } => STATUS_INVALID_LENGTH,
            umap_core::neighbors::KnnError::InvalidK { .. } => STATUS_INVALID_K,
            umap_core::neighbors::KnnError::NonFiniteValue { .. } => STATUS_NONFINITE_INPUT,
        },
        DenseGraphFitError::Graph(_) => STATUS_INTERNAL_ERROR,
    }
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
    catch_status(|| fit_dense_graph_impl(data, data_len, n_rows, n_features, k, metric, out_graph))
}

/// Fill a caller-owned view with borrowed pointers into an opaque CSR handle.
///
/// # Safety
///
/// `graph` must be a live, aligned handle returned by
/// [`umap_capi_fit_dense_graph`].  `out_view` must be non-null, aligned for
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

        let indptr = &graph.graph.indptr;
        let indices = &graph.graph.indices;
        let data = &graph.graph.data;
        let view = UmapCapiCsrView {
            n_rows: graph.graph.n_rows,
            n_cols: graph.graph.n_cols,
            indptr_len: indptr.len(),
            indices_len: indices.len(),
            data_len: data.len(),
            indptr: optional_slice_pointer(indptr),
            indices: optional_slice_pointer(indices),
            data: optional_slice_pointer(data),
        };

        // SAFETY: The pointer is non-null and aligned, and the caller
        // promises writable storage for one `UmapCapiCsrView`.
        unsafe { out_view.write(view) };
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
