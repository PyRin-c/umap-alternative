/*
 * Stable C-facing declarations for the UMAP dense graph primitive.
 *
 * The ABI uses uint32_t status and metric values and size_t dimensions.  The
 * numeric values below are part of the ABI; callers should not depend on the
 * compiler's representation of a C enum.
 */
#ifndef UMAP_CAPI_H
#define UMAP_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A handle returned by umap_capi_fit_dense_graph.  Its representation is
 * private and must not be allocated, copied, or inspected by callers. */
typedef struct umap_capi_csr_graph umap_capi_csr_graph_t;

typedef uint32_t umap_capi_status_t;
enum {
    UMAP_CAPI_STATUS_OK = 0u,
    UMAP_CAPI_STATUS_NULL_ARGUMENT = 1u,
    UMAP_CAPI_STATUS_INVALID_POINTER = 2u,
    UMAP_CAPI_STATUS_INVALID_LENGTH = 3u,
    UMAP_CAPI_STATUS_INVALID_SHAPE = 4u,
    UMAP_CAPI_STATUS_INVALID_K = 5u,
    UMAP_CAPI_STATUS_INVALID_METRIC = 6u,
    UMAP_CAPI_STATUS_NONFINITE_INPUT = 7u,
    UMAP_CAPI_STATUS_INVALID_HANDLE = 8u,
    UMAP_CAPI_STATUS_PANIC = 9u,
    UMAP_CAPI_STATUS_INTERNAL_ERROR = 10u
};

/* Values accepted by umap_capi_fit_dense_graph's metric argument. */
typedef uint32_t umap_capi_metric_t;
enum {
    UMAP_CAPI_METRIC_EUCLIDEAN = 0u,
    UMAP_CAPI_METRIC_MANHATTAN = 1u,
    UMAP_CAPI_METRIC_CHEBYSHEV = 2u
};

/*
 * Arrays in this view are borrowed from the graph handle.  indptr has
 * indptr_len entries; indices and data each have their corresponding length.
 * For a graph produced by this API, n_cols equals n_rows and
 * indices_len equals data_len.  The view and every pointer in it remain valid
 * only while graph is alive; release the handle only after the caller has
 * finished reading these arrays.  Zero-length arrays are represented by NULL
 * pointers.  The pointers are read-only and must not be freed or written.
 */
typedef struct umap_capi_csr_view {
    size_t n_rows;
    size_t n_cols;
    size_t indptr_len;
    size_t indices_len;
    size_t data_len;
    const size_t *indptr;
    const size_t *indices;
    const float *data;
} umap_capi_csr_view_t;

/*
 * Return the package version as a static NUL-terminated UTF-8 string.  The
 * returned pointer is never NULL and must not be freed.
 */
const char *umap_capi_version(void);

/*
 * Fit a fuzzy CSR graph from a contiguous row-major f32 matrix.
 *
 * data must either be NULL when data_len is zero or point to data_len readable
 * float values.  It must be aligned for float.  data_len must exactly equal
 * n_rows * n_features (with overflow rejected); n_rows, n_features, and k are
 * expressed in elements, not bytes.  k must be in 1..=n_rows.  The metric
 * argument must be one of UMAP_CAPI_METRIC_*.  out_graph must be a non-NULL,
 * aligned, writable pointer to one handle slot.  On success, the slot receives
 * an owning handle; on failure it is left untouched.  The caller owns that
 * handle and must release it exactly once with
 * umap_capi_csr_graph_release.
 */
umap_capi_status_t umap_capi_fit_dense_graph(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    size_t k,
    umap_capi_metric_t metric,
    umap_capi_csr_graph_t **out_graph);

/*
 * Fill a caller-provided view of graph.  graph must be a live handle returned
 * by umap_capi_fit_dense_graph and out_view must be a non-NULL, aligned,
 * writable pointer to a complete umap_capi_csr_view_t.  The output borrows the
 * handle's storage; it must not outlive the handle.
 */
umap_capi_status_t umap_capi_csr_graph_view(
    const umap_capi_csr_graph_t *graph,
    umap_capi_csr_view_t *out_view);

/*
 * Release an owning graph handle.  NULL is accepted as a no-op.  A non-NULL
 * pointer must be a live handle returned by umap_capi_fit_dense_graph and must
 * not be used again after this call.  Any borrowed view pointers become
 * invalid when this function returns.
 */
umap_capi_status_t umap_capi_csr_graph_release(umap_capi_csr_graph_t *graph);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UMAP_CAPI_H */
