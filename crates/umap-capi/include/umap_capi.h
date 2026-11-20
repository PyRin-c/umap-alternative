/*
 * Stable C-facing declarations for dense UMAP graph and embedding fits, transforms, and updates.
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

/* An owning result returned by a dense embedding fit or update. */
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
    UMAP_CAPI_METRIC_CHEBYSHEV = 2u,
    UMAP_CAPI_METRIC_COSINE = 3u,
    UMAP_CAPI_METRIC_CORRELATION = 4u
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

typedef struct umap_capi_densmap_options {
    double dens_lambda;
    double dens_frac;
    double dens_var_shift;
} umap_capi_densmap_options_t;

/*
 * Use the effective fitted neighbor count, curve parameters and learning
 * rate. n_epochs is the update epoch count, with zero selecting initialization.
 * Repulsion and the negative-sample rate apply to this update. Use 1.0 for
 * both graph options to select the default connectivity and set operation.
 */
typedef struct umap_capi_dense_update_config {
    size_t n_neighbors;
    size_t n_epochs;
    double a;
    double b;
    double learning_rate;
    double repulsion_strength;
    double negative_sample_rate;
    double local_connectivity;
    double set_op_mix_ratio;
    uint32_t transform_seed;
} umap_capi_dense_update_config_t;

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
 * Fit a seeded dense embedding with the density objective. The storage and
 * handle rules of umap_capi_fit_dense_embedding_seeded apply. density points
 * to initialized, aligned storage. dens_frac is in [0, 1], and dens_lambda
 * and dens_var_shift are nonnegative. All three parameters must be finite.
 */
umap_capi_status_t umap_capi_fit_dense_densmap_seeded(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    const umap_capi_dense_embedding_config_t *config,
    const umap_capi_densmap_options_t *density,
    umap_capi_dense_embedding_t **out_embedding);

/*
 * Append rows to a dense Euclidean embedding. Pass the original feature
 * rows in fitted order followed by at least one new row. data_len must equal
 * n_rows * n_features. fitted must be a live handle from a dense fit or
 * update. The fitted handle supplies the original row count and coordinate
 * width. n_neighbors must be between one and the original row count.
 *
 * All input pointers must refer to aligned readable storage throughout
 * the call. out_embedding must point to an aligned writable slot outside
 * that storage. The existing handle and input arrays are read-only. Success
 * returns a separate owning handle. Failure leaves the output slot untouched.
 */
umap_capi_status_t umap_capi_update_dense_embedding(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    const umap_capi_dense_embedding_t *fitted,
    const umap_capi_dense_update_config_t *config,
    umap_capi_dense_embedding_t **out_embedding);

/*
 * Same storage and ownership rules as umap_capi_update_dense_embedding.
 * Pass the UMAP_CAPI_METRIC_* tag used to fit the original rows. The updated
 * graph is rebuilt without a distance cutoff.
 */
umap_capi_status_t umap_capi_update_dense_embedding_with_metric(
    const float *data,
    size_t data_len,
    size_t n_rows,
    size_t n_features,
    umap_capi_metric_t metric,
    const umap_capi_dense_embedding_t *fitted,
    const umap_capi_dense_update_config_t *config,
    umap_capi_dense_embedding_t **out_embedding);

/*
 * Fill a caller-owned view, leaving it untouched on error. embedding must
 * be a live handle returned by a dense fit or update. out_view must point
 * to one aligned writable view outside the handle's storage.
 */
umap_capi_status_t umap_capi_dense_embedding_view(
    const umap_capi_dense_embedding_t *embedding,
    umap_capi_dense_embedding_view_t *out_view);

/*
 * Release the embedding handle exactly once. This releases both the coordinates
 * and graph and invalidates every borrowed pointer.  NULL is a no-op.  Any
 * non-NULL argument must be a live handle returned by a dense fit or update.
 */
umap_capi_status_t umap_capi_dense_embedding_release(
    umap_capi_dense_embedding_t *embedding);

/*
 * Both transforms borrow finite row-major feature matrices with the same
 * positive n_features. Each length must equal its row count times n_features
 * in floats. An array with zero length may be NULL. Zero queries are allowed.
 * The descriptor and all input buffers must remain readable and unchanged
 * throughout the call and may be reused or freed afterward.
 */
typedef struct umap_capi_dense_transform_input {
    size_t n_training_rows;
    size_t n_query_rows;
    size_t n_features;
    size_t training_len;
    size_t queries_len;
    const float *training;
    const float *queries;
} umap_capi_dense_transform_input_t;

/*
 * Borrow a row-major n_query_rows x n_training_rows distance matrix with
 * finite nonnegative values. A supplied ranking has n_query_rows * k
 * distinct in-range columns, ordered by nondecreasing distance in each row.
 * NULL ranked_columns with zero length selects native ranking, which orders
 * equal distances by increasing column index.
 */
typedef struct umap_capi_precomputed_transform_input {
    size_t n_training_rows;
    size_t n_query_rows;
    size_t distances_len;
    const float *distances;
    size_t ranked_columns_len;
    const size_t *ranked_columns;
} umap_capi_precomputed_transform_input_t;

typedef uint32_t umap_capi_fit_epochs_mode_t;
enum {
    UMAP_CAPI_FIT_EPOCHS_DEFAULT = 0u,
    UMAP_CAPI_FIT_EPOCHS_EXPLICIT = 1u
};

/*
 * Use the effective fit neighbor count (1..=n_training_rows) and positive
 * n_components. All doubles must be finite. The curve parameters a and b
 * and the negative_sample_rate must be positive. The rate must yield a
 * representable schedule. The learning_rate, repulsion_strength, and
 * local_connectivity must be nonnegative. The transform starts at one quarter
 * of learning_rate and uses max(0, local_connectivity - 1) for query membership.
 * metric controls search over features. Layout distances are Euclidean.
 *
 * With FIT_EPOCHS_DEFAULT, set fit_epochs to zero. The optimizer uses 100
 * epochs for up to 10,000 query rows or 30 for larger batches. With
 * FIT_EPOCHS_EXPLICIT it uses fit_epochs / 3, rounding down. Explicit zero
 * returns initialization without updates.
 *
 * transform_seed is independent of the fit seed. Each call starts a new
 * random stream. Values above UINT32_MAX return INVALID_CONFIG.
 * disconnection_distance is a positive float cutoff. Positive infinity keeps
 * all selected neighbors. A neighbor at or above the cutoff is excluded.
 */
typedef struct umap_capi_dense_transform_config {
    size_t n_neighbors;
    size_t n_components;
    size_t fit_epochs;
    double a;
    double b;
    double learning_rate;
    double repulsion_strength;
    double negative_sample_rate;
    double local_connectivity;
    uint64_t transform_seed;
    umap_capi_metric_t metric;
    umap_capi_fit_epochs_mode_t fit_epochs_mode;
    float disconnection_distance;
} umap_capi_dense_transform_config_t;

/* Owning result types for transformed coordinates and directed COO graphs. */
typedef struct umap_capi_dense_transform umap_capi_dense_transform_t;
typedef struct umap_capi_coo_graph umap_capi_coo_graph_t;

/* Read-only row-major coordinates borrowed from a transform handle. */
typedef struct umap_capi_dense_transform_view {
    size_t n_rows;
    size_t n_components;
    size_t coordinates_len;
    const float *coordinates;
} umap_capi_dense_transform_view_t;

/*
 * Read-only arrays borrowed from a graph handle. Each array has nnz entries
 * (including any zero weights), and all three pointers are NULL if nnz is 0.
 * Rows index queries and columns index training rows. The storage order
 * follows query index and the neighbor ranking for that query. Positions cut
 * by a disconnection distance store zero at coordinate (0, 0).
 */
typedef struct umap_capi_coo_view {
    size_t n_rows;
    size_t n_cols;
    size_t nnz;
    const size_t *row;
    const size_t *col;
    const float *data;
} umap_capi_coo_view_t;

/*
 * Transform held-out queries using fixed fitted coordinates. fitted_len must
 * equal n_training_rows * n_components and fitted_coordinates must point to
 * those readable row-major floats, unchanged throughout the call. Fitted
 * NaNs may propagate at initialization and such rows are skipped during
 * updates. A query with no positive memberships yields NaN coordinates.
 * A zero-query result has NULL coordinates. The caller may supply coordinates
 * borrowed from a live fitted handle. The result owns its coordinate buffer,
 * so the caller may reuse or release inputs once this function returns.
 *
 * All non-NULL pointers must be aligned for their types. input and config
 * must point to initialized descriptors. out_transform must point to a
 * writable handle slot outside all input storage. Success writes a new
 * owning handle to the slot. Every error leaves the slot untouched.
 * Zero features, zero components, and multiplication overflow return
 * INVALID_SHAPE. A mismatched length or an array exceeding the addressable
 * range returns INVALID_LENGTH. Nullness, alignment, and numeric shapes are
 * checked before the transform reads any of the arrays.
 */
umap_capi_status_t umap_capi_transform_dense_embedding(
    const umap_capi_dense_transform_input_t *input,
    const float *fitted_coordinates,
    size_t fitted_len,
    const umap_capi_dense_transform_config_t *config,
    umap_capi_dense_transform_t **out_transform);

/*
 * Construct the directed query-to-training graph, retaining all selected
 * neighbors. n_neighbors is in 1..=n_training_rows. local_connectivity must
 * be finite and nonnegative and has the same meaning as above.
 * out_graph follows the pointer, shape, and ownership contract above.
 */
umap_capi_status_t umap_capi_transform_dense_graph(
    const umap_capi_dense_transform_input_t *input,
    size_t n_neighbors,
    umap_capi_metric_t metric,
    double local_connectivity,
    umap_capi_coo_graph_t **out_graph);

/*
 * Precomputed variants use the same output handles, borrowing rules and
 * error behavior. The embedding transform uses config's layout settings and
 * disconnection distance. The graph transform uses the explicit inclusive
 * cutoff, with positive infinity retaining every selected distance.
 */
umap_capi_status_t umap_capi_transform_precomputed_embedding(
    const umap_capi_precomputed_transform_input_t *input,
    const float *fitted_coordinates,
    size_t fitted_len,
    const umap_capi_dense_transform_config_t *config,
    umap_capi_dense_transform_t **out_transform);

umap_capi_status_t umap_capi_transform_precomputed_graph(
    const umap_capi_precomputed_transform_input_t *input,
    size_t n_neighbors,
    double local_connectivity,
    float disconnection_distance,
    umap_capi_coo_graph_t **out_graph);

/*
 * Read from a live handle into an aligned writable view outside the handle's
 * storage. Errors leave the view untouched. The view's arrays are valid only
 * while its handle is alive. Release each owning handle exactly once after
 * all readers finish. Both release functions accept NULL as a no-op.
 */
umap_capi_status_t umap_capi_dense_transform_view(
    const umap_capi_dense_transform_t *transform,
    umap_capi_dense_transform_view_t *out_view);

umap_capi_status_t umap_capi_dense_transform_release(
    umap_capi_dense_transform_t *transform);

umap_capi_status_t umap_capi_coo_graph_view(
    const umap_capi_coo_graph_t *graph,
    umap_capi_coo_view_t *out_view);

umap_capi_status_t umap_capi_coo_graph_release(umap_capi_coo_graph_t *graph);

/* Query location outside the fitted embedding's convex hull. */
#define UMAP_CAPI_INVERSE_EXTERIOR SIZE_MAX

/*
 * Supply three finite row-major matrices. The training matrix has
 * n_training_rows by n_features elements. The fitted embedding and queries
 * have n_training_rows and n_query_rows rows respectively, both with width
 * config->n_components. Feature width and query count must be positive.
 * The embedding width must be at least two, and the training row count
 * must be at least n_components + 2.
 *
 * Supply the Delaunay tessellation of the fitted embedding and each query's
 * location in it. simplices has one row per simplex, each with n_components + 1
 * distinct zero-based training vertex indices. At least one simplex is
 * required. Preserve the simplex's vertex order because its first vertex
 * seeds the search. query_simplices has one zero-based simplex index or
 * UMAP_CAPI_INVERSE_EXTERIOR per query. Exterior queries seed the search at
 * the first nearest fitted vertex by squared float32 distance. The caller
 * determines each location and uses the same coordinate space for all geometry.
 *
 * Every length is an element count, and all arrays are read-only for the call.
 */
typedef struct umap_capi_dense_inverse_input {
    size_t n_training_rows;
    size_t n_query_rows;
    size_t n_features;
    size_t training_len;
    size_t embedding_len;
    size_t queries_len;
    size_t simplices_len;
    size_t query_simplices_len;
    const float *training;
    const float *embedding;
    const float *queries;
    const size_t *simplices;
    const size_t *query_simplices;
} umap_capi_dense_inverse_input_t;

/* Original-space fitted local offset and exponential scale for one row. */
typedef struct umap_capi_inverse_local_scale {
    float rho;
    float sigma;
} umap_capi_inverse_local_scale_t;

/*
 * A non-NULL descriptor supplies exactly one scale source. Set the unused
 * pointer and its associated lengths and width to zero.
 *
 * fitted supplies one pair per training row, with finite rho >= 0 and
 * finite sigma > 0. ranked_distances supplies the original fit-time ranked
 * feature distances in row-major order, with self first and positive infinity
 * for positions removed by a cutoff. Keep the full supplied width and order.
 * ranked_len equals n_training_rows * ranked_width. ranked_width can exceed
 * config->n_neighbors, which is the effective smoothing target.
 */
typedef struct umap_capi_inverse_scales {
    size_t fitted_len;
    size_t ranked_len;
    size_t ranked_width;
    const umap_capi_inverse_local_scale_t *fitted;
    const float *ranked_distances;
} umap_capi_inverse_scales_t;

typedef struct umap_capi_dense_inverse umap_capi_dense_inverse_t;

/* Read-only feature reconstruction and exterior count from an inverse handle. */
typedef struct umap_capi_dense_inverse_view {
    size_t n_rows;
    size_t n_features;
    size_t coordinates_len;
    size_t exterior_queries;
    const float *coordinates;
} umap_capi_dense_inverse_view_t;

/*
 * Reconstruct dense features using the fitted curve, metric, initial learning
 * rate, local connectivity, and effective neighbor count in config. All five
 * UMAP_CAPI_METRIC_* tags select original-space distances and gradients.
 * Epoch modes and transform_seed follow the dense transform settings. Inverse
 * optimization starts at learning_rate / 4 and uses scales only with at least
 * two inverse epochs and a nonzero initial rate.
 *
 * Pass NULL scales to derive scales from exhaustive fitted feature neighbors.
 * Derivation uses local_connectivity and omits distances >=
 * disconnection_distance. Use positive infinity for an unbounded cutoff.
 * Caller-supplied metadata already incorporates any cutoff used by the fit.
 * Ranked distances are calibrated with the fitted local_connectivity.
 *
 * All descriptors and buffers must be aligned and readable. out_inverse must
 * be an aligned, writable handle slot disjoint from the input storage. Success
 * writes an owning handle and every error leaves the slot untouched. Zero
 * queries, multiplication overflow, and invalid simplex indices or duplicate
 * simplex vertices return INVALID_SHAPE. Length mismatches and unaddressable
 * buffers return INVALID_LENGTH. An active query with all weights underflowed
 * to zero returns INVALID_CONFIG.
 */
umap_capi_status_t umap_capi_inverse_transform_dense(
    const umap_capi_dense_inverse_input_t *input,
    const umap_capi_dense_transform_config_t *config,
    const umap_capi_inverse_scales_t *scales,
    umap_capi_dense_inverse_t **out_inverse);

/*
 * Fill a writable aligned view outside the live handle's storage. On error
 * the view is untouched. The view borrows n_rows * n_features row-major floats
 * until release. All input storage may be reused when the transform returns.
 */
umap_capi_status_t umap_capi_dense_inverse_view(
    const umap_capi_dense_inverse_t *inverse,
    umap_capi_dense_inverse_view_t *out_view);

/* Release exactly once after all readers finish. NULL is a no-op. */
umap_capi_status_t umap_capi_dense_inverse_release(umap_capi_dense_inverse_t *inverse);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UMAP_CAPI_H */
