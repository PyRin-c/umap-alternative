/*
 * Stable C-facing declarations for dense UMAP graphs and seeded embeddings.
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

/* A handle returned by either dense graph fit.  Its representation is
 * private and must not be allocated, copied, or inspected by callers. */
typedef struct umap_capi_csr_graph umap_capi_csr_graph_t;

/* An owning result returned by umap_capi_fit_dense_embedding_seeded. */
typedef struct umap_capi_dense_embedding umap_capi_dense_embedding_t;

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
    UMAP_CAPI_STATUS_INTERNAL_ERROR = 10u,
    UMAP_CAPI_STATUS_INVALID_CONFIG = 11u
};

/* Distance metric for either dense fit. */
typedef uint32_t umap_capi_metric_t;
enum {
    UMAP_CAPI_METRIC_EUCLIDEAN = 0u,
    UMAP_CAPI_METRIC_MANHATTAN = 1u,
    UMAP_CAPI_METRIC_CHEBYSHEV = 2u
};

/*
 * Arrays in this view are borrowed from the owning handle.  indptr has
 * indptr_len entries; indices and data each have their corresponding length.
 * For a graph produced by this API, n_cols equals n_rows and
 * indices_len equals data_len.  The arrays remain valid while the owning
 * handle is alive.  Zero-length arrays are represented by NULL pointers.
 * The arrays are read-only and must not be freed or written.
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

typedef uint32_t umap_capi_embedding_init_t;
enum {
    UMAP_CAPI_INIT_RANDOM = 0u,
    UMAP_CAPI_INIT_EXPLICIT = 1u
};

/*
 * The layout of this configuration is fixed for the seeded fit entry point.
 * n_neighbors is in 1..=n_rows, n_components is positive, and n_epochs may be
 * zero.  a and b are positive curve parameters.  learning_rate and
 * repulsion_strength are non-negative, and negative_sample_rate is positive.
 * All doubles must be finite and the negative-sample rate must yield a
 * representable schedule.  metric is a UMAP_CAPI_METRIC_* tag.
 *
 * For UMAP_CAPI_INIT_RANDOM, init_coordinates must be NULL and init_len zero.
 * seed initializes the generator for both the random layout and subsequent
 * sampling.  For UMAP_CAPI_INIT_EXPLICIT, init_coordinates points to init_len
 * aligned, readable, finite floats in row-major order with
 * n_rows * n_components elements.  Each column must have a nonzero span.
 * The fit copies and normalizes the coordinates and uses seed for sampling.
 * All uint32_t seed values are accepted for either mode.
 */
typedef struct umap_capi_dense_embedding_config {
    size_t n_neighbors;
    size_t n_components;
    size_t n_epochs;
    double a;
    double b;
    double learning_rate;
    double repulsion_strength;
    double negative_sample_rate;
    umap_capi_metric_t metric;
    umap_capi_embedding_init_t init;
    uint32_t seed;
    size_t init_len;
    const float *init_coordinates;
} umap_capi_dense_embedding_config_t;

/*
 * Coordinates are read-only row-major floats with n_rows * n_components
 * entries.  The coordinate and CSR arrays are all borrowed from the same
 * embedding handle and are valid until it is released.  The graph is square
 * with n_rows rows.  Release the embedding only after all readers finish.
 */
typedef struct umap_capi_dense_embedding_view {
    size_t n_rows;
    size_t n_components;
    size_t coordinates_len;
    const float *coordinates;
    umap_capi_csr_view_t graph;
} umap_capi_dense_embedding_view_t;

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
 * Fit with explicit fuzzy graph settings.  The data and output ownership
 * follow umap_capi_fit_dense_graph.  local_connectivity must be finite and
 * nonnegative.  set_op_mix_ratio ranges from 0 (intersection) to 1 (union).
 * Invalid settings return INVALID_CONFIG.  Zero memberships are omitted.
 */
umap_capi_status_t umap_capi_fit_dense_graph_with_options(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    size_t k,
    umap_capi_metric_t metric,
    double local_connectivity,
    double set_op_mix_ratio,
    umap_capi_csr_graph_t **out_graph);

/*
 * Fill a caller-provided view of graph.  graph must be a live handle returned
 * by a dense graph fit and out_view must be a non-NULL, aligned,
 * writable pointer to a complete umap_capi_csr_view_t.  The output borrows the
 * handle's storage; it must not outlive the handle.
 */
umap_capi_status_t umap_capi_csr_graph_view(
    const umap_capi_csr_graph_t *graph,
    umap_capi_csr_view_t *out_view);

/*
 * Release an owning graph handle.  NULL is accepted as a no-op.  A non-NULL
 * pointer must be a live handle returned by a dense graph fit and must
 * not be used again after this call.  Any borrowed view pointers become
 * invalid when this function returns.
 */
umap_capi_status_t umap_capi_csr_graph_release(umap_capi_csr_graph_t *graph);

/*
 * Fit a seeded embedding and its CSR graph from a finite row-major matrix.
 * data_len is an element count and must equal n_rows * n_features.  data may
 * be NULL when data_len is zero.  All non-NULL pointers must be aligned for
 * their types.  config must point to one fully initialized settings struct.
 * The input storage must be readable and unchanged throughout the call and
 * may be reused or freed after it returns.  Zero epochs returns the normalized
 * initial coordinates, and one epoch returns the same coordinates because
 * scheduling starts with epoch zero before any updates are due.
 *
 * out_embedding must point to one writable handle slot outside the input
 * storage.  On success it receives an owning handle.  Every error leaves the
 * slot untouched.  Shape multiplication overflow and zero n_components
 * return INVALID_SHAPE.  A mismatched length or unaddressable float buffer
 * returns INVALID_LENGTH.  Invalid initialization tags, unexpected random
 * initialization fields, degenerate columns and invalid optimizer settings
 * return INVALID_CONFIG.
 */
umap_capi_status_t umap_capi_fit_dense_embedding_seeded(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    const umap_capi_dense_embedding_config_t *config,
    umap_capi_dense_embedding_t **out_embedding);

/*
 * Fill a caller-owned view, leaving it untouched on error.  embedding must
 * be a live handle returned by umap_capi_fit_dense_embedding_seeded.  out_view
 * must point to one aligned writable view outside the handle's storage.
 */
umap_capi_status_t umap_capi_dense_embedding_view(
    const umap_capi_dense_embedding_t *embedding,
    umap_capi_dense_embedding_view_t *out_view);

/*
 * Release the fit handle exactly once.  This releases both the coordinates
 * and graph and invalidates every borrowed pointer.  NULL is a no-op.  Any
 * non-NULL argument must be a live embedding handle returned by the fit.
 */
umap_capi_status_t umap_capi_dense_embedding_release(
    umap_capi_dense_embedding_t *embedding);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UMAP_CAPI_H */
