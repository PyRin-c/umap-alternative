/* Compile with a C11 compiler and the crate's include directory. */
#include "umap_capi.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define MISALIGNED(type, pointer) ((type *)((uintptr_t)(pointer) + 1u))
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

_Static_assert(sizeof(float) == sizeof(uint32_t), "expected 32-bit float");

static void check_graph_only(void) {
    const float points[] = {0.0f, 0.0f, 1.0f, 0.0f};
    umap_capi_csr_graph_t *graph = NULL;
    umap_capi_status_t status = umap_capi_fit_dense_graph(
        points,
        COUNT(points),
        2,
        2,
        2,
        UMAP_CAPI_METRIC_EUCLIDEAN,
        &graph);
    CHECK(status == UMAP_CAPI_STATUS_OK && graph != NULL);

    umap_capi_csr_view_t view = {0};
    status = umap_capi_csr_graph_view(graph, &view);
    CHECK(status == UMAP_CAPI_STATUS_OK && view.n_rows == 2 && view.n_cols == 2);
    CHECK(view.indptr_len == 3 && view.indices_len == 2 && view.data_len == 2);
    CHECK(view.indptr && view.indices && view.data);
    CHECK(view.indptr[0] == 0 && view.indptr[1] == 1 && view.indptr[2] == 2);
    CHECK(view.indices[0] == 1 && view.indices[1] == 0);
    CHECK(view.data[0] == 1.0f && view.data[1] == 1.0f);
    CHECK(umap_capi_csr_graph_release(graph) == UMAP_CAPI_STATUS_OK);
}

static void check_graph_options(void) {
    const float points[] = {0.0f, 1.0f, 3.0f, 7.0f};
    const size_t indptr[] = {0, 1, 2, 2, 2};
    umap_capi_csr_graph_t *graph = NULL;
    CHECK(umap_capi_fit_dense_graph_with_options(
        points, COUNT(points), 4, 1, 2, UMAP_CAPI_METRIC_EUCLIDEAN, 1.0, 0.0, &graph
    ) == UMAP_CAPI_STATUS_OK);
    umap_capi_csr_view_t view = {0};
    CHECK(umap_capi_csr_graph_view(graph, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 4 && view.n_cols == 4);
    CHECK(view.indptr_len == COUNT(indptr) && view.indices_len == 2 && view.data_len == 2);
    CHECK(memcmp(view.indptr, indptr, sizeof indptr) == 0);
    CHECK(view.indices[0] == 1 && view.indices[1] == 0);
    CHECK(view.data[0] == 1.0f && view.data[1] == 1.0f);

    umap_capi_csr_graph_t *original = graph;
    const double invalid[][2] = {{-1.0, 1.0}, {INFINITY, 0.0}, {1.0, NAN}, {1.0, 1.0000000000000002}};
    for (size_t index = 0; index < COUNT(invalid); ++index) {
        CHECK(umap_capi_fit_dense_graph_with_options(
            points, COUNT(points), 4, 1, 2, UMAP_CAPI_METRIC_EUCLIDEAN,
            invalid[index][0], invalid[index][1], &graph
        ) == UMAP_CAPI_STATUS_INVALID_CONFIG);
        CHECK(graph == original);
    }
    CHECK(umap_capi_csr_graph_release(graph) == UMAP_CAPI_STATUS_OK);
}

/* weighted_explicit_min_seed, weighted_random_max_seed and weighted_k4 in
 * reference/fixtures/umap-learn-0.6.0-seeded-fit-windows-x86_64.json. */
static const float weighted[] = {
    0.0f, 0.0f, 0.0f,
    0.2f, 0.07f, 0.1f,
    0.25f, 0.89f, 0.13f,
    0.76f, 0.1f, 0.83f,
    0.99f, 0.95f, 0.37f,
    1.9f, 1.31f, 0.33f,
    2.61f, 1.7f, 0.9f,
    3.3f, 1.2f, 1.1f,
};
static const float explicit_init[] = {
    0.0f, 6.0f, 1.0f, 3.0f, 4.0f, 0.0f, 2.0f, 1.0f,
    5.0f, 4.0f, 6.0f, 5.0f, 3.0f, 2.0f, 7.0f, 7.0f,
};
static const size_t weighted_indptr[] = {0, 3, 6, 9, 12, 17, 20, 23, 26};
static const size_t weighted_indices[] = {
    1, 2, 3, 0, 2, 3, 0, 1, 4, 0, 1, 4, 2, 3, 5, 6, 7, 4, 6, 7, 4, 5, 7, 4, 5, 6,
};
static const uint32_t weighted_bits[] = {
    0x3f800000u, 0x3f2ca7e0u, 0x3f2222c4u, 0x3f800000u,
    0x3f5d1e62u, 0x3f800000u, 0x3f2ca7e0u, 0x3f5d1e62u,
    0x3f800000u, 0x3f2222c4u, 0x3f800000u, 0x3f557ca1u,
    0x3f800000u, 0x3f557ca1u, 0x3f800000u, 0x3e3949b1u,
    0x3ebd961au, 0x3f800000u, 0x3f7d9e38u, 0x3f261529u,
    0x3e3949b1u, 0x3f7d9e38u, 0x3f800000u, 0x3ebd961au,
    0x3f261529u, 0x3f800000u,
};
static const umap_capi_dense_embedding_config_t explicit_config = {
    .n_neighbors = 4,
    .n_components = 2,
    .a = 1.7,
    .b = 0.9,
    .learning_rate = 0.13,
    .repulsion_strength = 1.0,
    .negative_sample_rate = 2.0,
    .metric = UMAP_CAPI_METRIC_EUCLIDEAN,
    .init = UMAP_CAPI_INIT_EXPLICIT,
    .seed = 0,
    .init_len = COUNT(explicit_init),
    .init_coordinates = explicit_init,
};
static const umap_capi_dense_embedding_config_t random_config = {
    .n_neighbors = 4,
    .n_components = 2,
    .a = 1.7,
    .b = 0.9,
    .learning_rate = 0.13,
    .repulsion_strength = 0.37,
    .negative_sample_rate = 5.0,
    .metric = UMAP_CAPI_METRIC_EUCLIDEAN,
    .init = UMAP_CAPI_INIT_RANDOM,
    .seed = UINT32_MAX,
};

static void check_bits(const float *actual, const uint32_t *expected, size_t len) {
    CHECK(actual != NULL);
    for (size_t i = 0; i < len; ++i) {
        uint32_t bits;
        memcpy(&bits, &actual[i], sizeof(bits));
        if (bits != expected[i]) {
            fprintf(stderr, "index %zu: expected %08" PRIx32 ", got %08" PRIx32 "\n",
                i, expected[i], bits);
        }
        CHECK(bits == expected[i]);
    }
}

static umap_capi_dense_embedding_t *fit_weighted(
    umap_capi_dense_embedding_config_t config, size_t epochs) {
    float points[COUNT(weighted)];
    float init[COUNT(explicit_init)];
    memcpy(points, weighted, sizeof(points));
    memcpy(init, explicit_init, sizeof(init));
    if (config.init == UMAP_CAPI_INIT_EXPLICIT) {
        config.init_coordinates = init;
    }
    config.n_epochs = epochs;
    umap_capi_dense_embedding_t *fit = NULL;
    CHECK(umap_capi_fit_dense_embedding_seeded(
        points, COUNT(points), 8, 3, &config, &fit) == UMAP_CAPI_STATUS_OK);
    CHECK(fit != NULL);
    CHECK(memcmp(points, weighted, sizeof(points)) == 0);
    CHECK(memcmp(init, explicit_init, sizeof(init)) == 0);
    return fit;
}

static void check_checkpoint(
    umap_capi_dense_embedding_config_t config, size_t epochs, const uint32_t *expected) {
    /* The input buffers and config have gone out of scope. */
    umap_capi_dense_embedding_t *fit = fit_weighted(config, epochs);
    umap_capi_dense_embedding_view_t view = {0};
    CHECK(umap_capi_dense_embedding_view(fit, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 8 && view.n_components == 2 && view.coordinates_len == 16);
    check_bits(view.coordinates, expected, view.coordinates_len);

    CHECK(view.graph.n_rows == 8 && view.graph.n_cols == 8);
    CHECK(view.graph.indptr_len == COUNT(weighted_indptr));
    CHECK(view.graph.indices_len == COUNT(weighted_indices));
    CHECK(view.graph.data_len == COUNT(weighted_bits));
    CHECK(view.graph.indptr && view.graph.indices);
    for (size_t i = 0; i < view.graph.indptr_len; ++i) {
        CHECK(view.graph.indptr[i] == weighted_indptr[i]);
    }
    for (size_t i = 0; i < view.graph.indices_len; ++i) {
        CHECK(view.graph.indices[i] == weighted_indices[i]);
    }
    check_bits(view.graph.data, weighted_bits, view.graph.data_len);
    CHECK(umap_capi_dense_embedding_release(fit) == UMAP_CAPI_STATUS_OK);
}

static void check_seeded_checkpoints(void) {
    static const uint32_t explicit_0_and_1[] = {
        0x00000000u, 0x41092492u, 0x3fb6db6eu, 0x40892492u,
        0x40b6db6eu, 0x00000000u, 0x4036db6eu, 0x3fb6db6eu,
        0x40e49249u, 0x40b6db6eu, 0x41092492u, 0x40e49249u,
        0x40892492u, 0x4036db6eu, 0x41200000u, 0x41200000u,
    };
    static const uint32_t explicit_30[] = {
        0x3fc3ca28u, 0x40af848bu, 0x4014ac0cu, 0x40810560u,
        0x409cd7eeu, 0x40204f64u, 0x4023deb6u, 0x4076edb2u,
        0x40dce68fu, 0x40a27b20u, 0x40cd2c5eu, 0x40ad4e8fu,
        0x40c57dd4u, 0x40a52aedu, 0x41051dcfu, 0x40f39287u,
    };
    static const uint32_t random_0_and_1[] = {
        0x3f530ef7u, 0x4114e661u, 0x40ff45cdu, 0x40f65a2bu,
        0x00000000u, 0x41200000u, 0x40be6ed0u, 0x00000000u,
        0x41138e41u, 0x409e6a8cu, 0x41200000u, 0x4050690cu,
        0x404ac6c3u, 0x3eddae27u, 0x41178df9u, 0x4103b3c0u,
    };
    static const uint32_t random_500[] = {
        0x412e3a08u, 0x40419400u, 0x4133ba74u, 0x4047729du,
        0x412b8a82u, 0x402508edu, 0x4134ad9fu, 0x402bb84bu,
        0x412c7f09u, 0x40001653u, 0x41286fb8u, 0x3fbe8ae2u,
        0x4125a3aau, 0x3f8ebb54u, 0x412b25f3u, 0x3f8c8055u,
    };
    check_checkpoint(explicit_config, 0, explicit_0_and_1);
    check_checkpoint(explicit_config, 1, explicit_0_and_1);
    check_checkpoint(explicit_config, 30, explicit_30);
    check_checkpoint(random_config, 0, random_0_and_1);
    check_checkpoint(random_config, 1, random_0_and_1);
    check_checkpoint(random_config, 500, random_500);
}

static void check_fit_error(
    umap_capi_status_t expected, const float *data, size_t len,
    size_t rows, size_t features, const umap_capi_dense_embedding_config_t *config,
    umap_capi_dense_embedding_t *sentinel) {
    umap_capi_dense_embedding_t *out = sentinel;
    umap_capi_status_t status = umap_capi_fit_dense_embedding_seeded(
        data, len, rows, features, config, &out);
    if (status != expected) {
        fprintf(stderr, "expected fit status %" PRIu32 ", got %" PRIu32 "\n", expected, status);
    }
    CHECK(status == expected);
    CHECK(out == sentinel);
}

static void check_rejected_pointers_and_shapes(umap_capi_dense_embedding_t *fit) {
    umap_capi_dense_embedding_t *slot = fit;
    CHECK(umap_capi_fit_dense_embedding_seeded(
        weighted, COUNT(weighted), 8, 3, &random_config, NULL) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_fit_dense_embedding_seeded(
        weighted, COUNT(weighted), 8, 3, &random_config,
        MISALIGNED(umap_capi_dense_embedding_t *, &slot)) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(slot == fit);
    check_fit_error(UMAP_CAPI_STATUS_NULL_ARGUMENT,
        NULL, COUNT(weighted), 8, 3, &random_config, fit);
    check_fit_error(UMAP_CAPI_STATUS_NULL_ARGUMENT,
        weighted, COUNT(weighted), 8, 3, NULL, fit);
    check_fit_error(UMAP_CAPI_STATUS_INVALID_POINTER,
        MISALIGNED(const float, weighted), COUNT(weighted), 8, 3, &random_config, fit);
    check_fit_error(UMAP_CAPI_STATUS_INVALID_POINTER,
        weighted, COUNT(weighted), 8, 3,
        MISALIGNED(const umap_capi_dense_embedding_config_t, &random_config), fit);
    check_fit_error(UMAP_CAPI_STATUS_INVALID_LENGTH,
        weighted, COUNT(weighted) - 1, 8, 3, &random_config, fit);
    check_fit_error(UMAP_CAPI_STATUS_INVALID_SHAPE,
        weighted, COUNT(weighted), SIZE_MAX, 3, &random_config, fit);
    const size_t oversized = (size_t)PTRDIFF_MAX / sizeof(float) + 1;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_LENGTH,
        weighted, oversized, oversized, 1, &random_config, fit);

    umap_capi_dense_embedding_config_t config = random_config;
    config.n_components = 0;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_SHAPE, weighted, COUNT(weighted), 8, 3, &config, fit);
    config.n_components = SIZE_MAX;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_SHAPE, weighted, COUNT(weighted), 8, 3, &config, fit);
    config.n_components = oversized / 8 + 1;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_LENGTH, weighted, COUNT(weighted), 8, 3, &config, fit);

    config = explicit_config;
    config.init_coordinates = NULL;
    check_fit_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, weighted, COUNT(weighted), 8, 3, &config, fit);
    config.init_coordinates = MISALIGNED(const float, explicit_init);
    check_fit_error(UMAP_CAPI_STATUS_INVALID_POINTER, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = explicit_config;
    config.init_len--;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_LENGTH, weighted, COUNT(weighted), 8, 3, &config, fit);
    config.init_len = oversized;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_LENGTH, weighted, COUNT(weighted), 8, 3, &config, fit);
}

static void check_rejected_config(umap_capi_dense_embedding_t *fit) {
    umap_capi_dense_embedding_config_t config = random_config;
    config.a = 0.0;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.b = NAN;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.learning_rate = -1.0;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.repulsion_strength = INFINITY;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.negative_sample_rate = 0.0;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);

    config = random_config;
    config.n_neighbors = 9;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_K, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.metric = UINT32_MAX;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_METRIC, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = random_config;
    config.init = UINT32_MAX;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    config = explicit_config;
    config.init = UMAP_CAPI_INIT_RANDOM;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);

    float nonfinite_data[COUNT(weighted)] = {NAN};
    check_fit_error(UMAP_CAPI_STATUS_NONFINITE_INPUT,
        nonfinite_data, COUNT(nonfinite_data), 8, 3, &random_config, fit);
    float bad_init[COUNT(explicit_init)] = {0};
    config = explicit_config;
    config.init_coordinates = bad_init;
    check_fit_error(UMAP_CAPI_STATUS_INVALID_CONFIG, weighted, COUNT(weighted), 8, 3, &config, fit);
    bad_init[0] = NAN;
    check_fit_error(UMAP_CAPI_STATUS_NONFINITE_INPUT, weighted, COUNT(weighted), 8, 3, &config, fit);
}

static void check_boundaries(void) {
    umap_capi_dense_embedding_t *fit = fit_weighted(random_config, 0);
    check_rejected_pointers_and_shapes(fit);
    check_rejected_config(fit);

    umap_capi_dense_embedding_view_t view = {0};
    view.n_rows = SIZE_MAX;
    CHECK(umap_capi_dense_embedding_view(fit, NULL) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_embedding_view(
        fit, MISALIGNED(umap_capi_dense_embedding_view_t, &view)) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_dense_embedding_view(NULL, &view) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_embedding_view(
        MISALIGNED(const umap_capi_dense_embedding_t, fit), &view) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(view.n_rows == SIZE_MAX && view.coordinates == NULL && view.graph.indptr == NULL);
    CHECK(umap_capi_dense_embedding_release(NULL) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_dense_embedding_release(
        MISALIGNED(umap_capi_dense_embedding_t, fit)) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_dense_embedding_view(fit, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_dense_embedding_release(fit) == UMAP_CAPI_STATUS_OK);
}

static void check_empty_graph(void) {
    const float points[] = {0.0f, 1.0f};
    umap_capi_dense_embedding_config_t config = random_config;
    config.n_neighbors = 1;
    config.n_components = 1;
    config.n_epochs = 3;
    config.learning_rate = 0.0;
    config.repulsion_strength = 0.0;
    umap_capi_dense_embedding_t *fit = NULL;
    CHECK(umap_capi_fit_dense_embedding_seeded(
        points, COUNT(points), 2, 1, &config, &fit) == UMAP_CAPI_STATUS_OK);
    umap_capi_dense_embedding_view_t view = {0};
    CHECK(umap_capi_dense_embedding_view(fit, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 2 && view.n_components == 1 && view.coordinates_len == 2);
    CHECK(view.graph.indptr_len == 3 && view.graph.indices_len == 0 && view.graph.data_len == 0);
    CHECK(view.graph.indptr && !view.graph.indices && !view.graph.data);
    CHECK(umap_capi_dense_embedding_release(fit) == UMAP_CAPI_STATUS_OK);

    fit = NULL;
    config.n_neighbors = 2;
    CHECK(umap_capi_fit_dense_embedding_seeded(
        NULL, 0, 2, 0, &config, &fit) == UMAP_CAPI_STATUS_OK);
    CHECK(fit != NULL);
    CHECK(umap_capi_dense_embedding_release(fit) == UMAP_CAPI_STATUS_OK);
}

/* Inputs and curve of the 45-epoch public transform checkpoint in
 * umap-learn-0.6.0-transform-windows-x86_64.json and the default curve in
 * umap-learn-0.6.0-curve-windows-x86_64.json. */
static const float transform_training[] = {
    0.0f, 0.0f, 0.1f, 0.9f, 0.1f, 0.2f, 0.3f, 1.0f, 0.4f, 1.4f, 0.9f, 0.3f,
    0.2f, 0.6f, 1.1f, 1.0f, 0.2f, 1.3f, 1.6f, 1.3f, 1.0f, 0.7f, 1.7f, 0.8f,
};
static const float transform_queries[] = {0.4f, 0.3f, 0.5f, 1.2f, 0.8f, 1.1f, 0.4f, 1.4f, 0.6f};
static const umap_capi_dense_transform_input_t transform_input = {
    .n_training_rows = 8,
    .n_query_rows = 3,
    .n_features = 3,
    .training_len = COUNT(transform_training),
    .queries_len = COUNT(transform_queries),
    .training = transform_training,
    .queries = transform_queries,
};
static const umap_capi_dense_transform_config_t transform_config = {
    .n_neighbors = 4,
    .n_components = 2,
    .fit_epochs = 45,
    .a = 1.5769434607659205,
    .b = 0.89506087820881486,
    .learning_rate = 1.0,
    .repulsion_strength = 1.0,
    .negative_sample_rate = 5.0,
    .local_connectivity = 1.0,
    .transform_seed = 24,
    .metric = UMAP_CAPI_METRIC_EUCLIDEAN,
    .fit_epochs_mode = UMAP_CAPI_FIT_EPOCHS_EXPLICIT,
    .disconnection_distance = INFINITY,
};

static umap_capi_dense_transform_t *run_transform(
    const umap_capi_dense_transform_input_t *input, const float *fitted, size_t fitted_len,
    const umap_capi_dense_transform_config_t *config) {
    umap_capi_dense_transform_t *transform = NULL;
    CHECK(umap_capi_transform_dense_embedding(input, fitted, fitted_len, config, &transform)
        == UMAP_CAPI_STATUS_OK);
    CHECK(transform != NULL);
    return transform;
}

static umap_capi_dense_transform_view_t read_transform(
    const umap_capi_dense_transform_t *transform, size_t rows, size_t components) {
    umap_capi_dense_transform_view_t view = {0};
    CHECK(umap_capi_dense_transform_view(transform, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == rows && view.n_components == components);
    CHECK(view.coordinates_len == rows * components);
    CHECK((view.coordinates == NULL) == (rows == 0));
    return view;
}

static void check_optimized_transform(void) {
    static const uint32_t fitted_bits[] = {
        0x40941e62u, 0x40b77443u, 0x40a71eecu, 0x40a3f28bu,
        0x40a72047u, 0x40d339d5u, 0x409011dbu, 0x40f230e6u,
        0x407059fau, 0x40bb193au, 0x4046dd9du, 0x40adab19u,
        0x40a75088u, 0x4101ba5bu, 0x40b1b81bu, 0x40f17d9du,
    };
    static const uint32_t expected[] = {
        0x40abebcbu, 0x408603f5u, 0x404014d3u, 0x40e7dc58u, 0x40569faau, 0x40f7f148u,
    };
    umap_capi_dense_embedding_config_t config = random_config;
    config.n_epochs = transform_config.fit_epochs;
    config.a = transform_config.a;
    config.b = transform_config.b;
    config.learning_rate = transform_config.learning_rate;
    config.repulsion_strength = transform_config.repulsion_strength;
    config.seed = 42;
    umap_capi_dense_embedding_t *fit = NULL;
    CHECK(umap_capi_fit_dense_embedding_seeded(transform_training, COUNT(transform_training),
        8, 3, &config, &fit) == UMAP_CAPI_STATUS_OK);
    umap_capi_dense_embedding_view_t fitted = {0};
    CHECK(umap_capi_dense_embedding_view(fit, &fitted) == UMAP_CAPI_STATUS_OK);
    CHECK(fitted.coordinates_len == COUNT(fitted_bits));
    check_bits(fitted.coordinates, fitted_bits, COUNT(fitted_bits));

    umap_capi_dense_transform_t *first = run_transform(
        &transform_input, fitted.coordinates, fitted.coordinates_len, &transform_config);
    umap_capi_dense_transform_t *second = run_transform(
        &transform_input, fitted.coordinates, fitted.coordinates_len, &transform_config);
    check_bits(fitted.coordinates, fitted_bits, COUNT(fitted_bits));
    CHECK(umap_capi_dense_embedding_release(fit) == UMAP_CAPI_STATUS_OK);
    check_bits(read_transform(first, 3, 2).coordinates, expected, COUNT(expected));
    check_bits(read_transform(second, 3, 2).coordinates, expected, COUNT(expected));
    CHECK(umap_capi_dense_transform_release(first) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_dense_transform_release(second) == UMAP_CAPI_STATUS_OK);
}

static void check_transform_connectivity(void) {
    /* Embedding case with 18 fit epochs and connectivity 2 in
     * umap-learn-0.6.0-estimator-graph-options-x86_64.json. */
    static const float training_values[] = {
        0.0f, 0.0f, 0.25f, 0.75f, 0.5f, 1.5f, 1.0f, 0.125f,
        1.75f, 1.25f, 2.5f, 0.25f, 3.125f, 1.0f, 3.5f, 0.125f,
    };
    static const float query_values[] = {0.125f, 0.5f, 1.25f, 1.125f, 2.125f, 0.625f};
    static const uint32_t fitted_bits[] = {
        0x4065651au, 0x410f5995u, 0x40f60a85u, 0x40b3240fu,
        0x400596b4u, 0x3ed8e7c3u, 0x40196a40u, 0x410ad456u,
        0x40c96c92u, 0x40d51f14u, 0x3ed159a3u, 0x41184626u,
        0x411245c8u, 0x3f7120c5u, 0x404ffa25u, 0x3fd1470du,
    };
    static const uint32_t expected[] = {
        0x4102cd33u, 0x40a31023u, 0x40bc58f5u, 0x40e22a77u, 0x3f3de528u, 0x41132e79u,
    };
    umap_capi_dense_transform_t *transform;
    {
        float training[COUNT(training_values)], queries[COUNT(query_values)], fitted[COUNT(fitted_bits)];
        memcpy(training, training_values, sizeof(training));
        memcpy(queries, query_values, sizeof(queries));
        memcpy(fitted, fitted_bits, sizeof(fitted));
        const umap_capi_dense_transform_input_t input = {
            8, 3, 2, COUNT(training), COUNT(queries), training, queries,
        };
        umap_capi_dense_transform_config_t config = transform_config;
        config.fit_epochs = 18;
        config.a = 1.7;
        config.b = 0.9;
        config.learning_rate = 0.13;
        config.local_connectivity = 2.0;
        config.transform_seed = 42;
        transform = run_transform(&input, fitted, COUNT(fitted), &config);
        CHECK(memcmp(training, training_values, sizeof(training)) == 0);
        CHECK(memcmp(queries, query_values, sizeof(queries)) == 0);
        check_bits(fitted, fitted_bits, COUNT(fitted_bits));
    }
    check_bits(read_transform(transform, 3, 2).coordinates, expected, COUNT(expected));
    CHECK(umap_capi_dense_transform_release(transform) == UMAP_CAPI_STATUS_OK);
}

/* Zero-epoch fitted coordinates shared by several checkpoints in
 * umap-learn-0.6.0-transform-embedding-windows-x86_64.json. */
static const uint32_t zero_fit_bits[] = {
    0x408b83a7u, 0x411c39fdu, 0x410c3425u, 0x40ae09d0u,
    0x3fd58791u, 0x00000000u, 0x3eec7d35u, 0x410b9ba3u,
    0x40e4d20bu, 0x40d90e43u, 0x00000000u, 0x41200000u,
    0x41200000u, 0x3f31385bu, 0x3ffe3772u, 0x3eac6cc3u,
};

static void check_transform_epochs(void) {
    /* The first three rows of fit_0, and default_10000 and default_10001. */
    static const uint32_t expected[][6] = {
        {0x40b0c49au, 0x40b16dbau, 0x408c27ebu, 0x40cf79d1u, 0x402f83c9u, 0x4059c472u},
        {0x40dd2a72u, 0x40b37ac1u, 0x4025d7eau, 0x40f67c49u, 0x40092d94u, 0x3fcffb7cu},
        {0x40c67404u, 0x40bd8669u, 0x4085c39cu, 0x40dbf2feu, 0x400e53a2u, 0x3fccb263u},
    };
    static const size_t rows[] = {3, 10000, 10001};
    float fitted[COUNT(zero_fit_bits)];
    memcpy(fitted, zero_fit_bits, sizeof(fitted));
    umap_capi_dense_transform_config_t config = transform_config;
    config.a = 1.7;
    config.b = 0.9;
    config.fit_epochs = 0;
    for (size_t i = 0; i < COUNT(rows); ++i) {
        config.fit_epochs_mode = i == 0 ? UMAP_CAPI_FIT_EPOCHS_EXPLICIT : UMAP_CAPI_FIT_EPOCHS_DEFAULT;
        umap_capi_dense_transform_input_t input = transform_input;
        input.n_query_rows = rows[i];
        input.queries_len = input.n_query_rows * input.n_features;
        float *queries = malloc(input.queries_len * sizeof(*queries));
        CHECK(queries != NULL);
        for (size_t j = 0; j < input.queries_len; ++j) {
            queries[j] = transform_queries[j % COUNT(transform_queries)];
        }
        input.queries = queries;
        umap_capi_dense_transform_t *transform = run_transform(&input, fitted, COUNT(fitted), &config);
        free(queries);
        const umap_capi_dense_transform_view_t view = read_transform(transform, rows[i], 2);
        for (size_t row = 0; row < rows[i]; ++row) {
            check_bits(view.coordinates + 2 * row, expected[i] + 2 * (row % 3), 2);
        }
        CHECK(umap_capi_dense_transform_release(transform) == UMAP_CAPI_STATUS_OK);
    }
}

static void check_transform_cutoff(void) {
    /* cutoff_at and cutoff_above exercise strict comparison and fitted NaNs. */
    const float training[] = {0.0f, 0.1f, 0.2f, 0.3f, 1.5f, 1.6f, 1.7f, 1.8f, 5.0f};
    const float queries[] = {-1.0f, 5.0f, 15.0f, 1.51f, 0.43f};
    const umap_capi_dense_transform_input_t input = {
        COUNT(training), COUNT(queries), 1, COUNT(training), COUNT(queries), training, queries,
    };
    static const uint32_t expected[][10] = {
        {0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u,
         0x7fc00000u, 0x40aa548cu, 0x40aea5a1u, 0x40552cfeu, 0x40c97beau},
        {0x40807ee1u, 0x41111635u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u,
         0x7fc00000u, 0x40aa548cu, 0x40aea5a1u, 0x40552cfeu, 0x40c97beau},
    };
    static const float cutoffs[] = {1.0f, 1.00000011920928955078125f};
    static const uint32_t disconnected_tail[] = {0x7fc00000u, 0x7fc00000u};
    float fitted[2 * COUNT(training)];
    memcpy(fitted, zero_fit_bits, sizeof(zero_fit_bits));
    memcpy(fitted + COUNT(zero_fit_bits), disconnected_tail, sizeof(disconnected_tail));
    umap_capi_dense_transform_config_t config = transform_config;
    config.a = 1.7;
    config.b = 0.9;
    for (size_t i = 0; i < COUNT(cutoffs); ++i) {
        config.disconnection_distance = cutoffs[i];
        umap_capi_dense_transform_t *transform = run_transform(&input, fitted, COUNT(fitted), &config);
        check_bits(read_transform(transform, COUNT(queries), 2).coordinates, expected[i], COUNT(expected[i]));
        CHECK(umap_capi_dense_transform_release(transform) == UMAP_CAPI_STATUS_OK);
    }
    check_bits(fitted, zero_fit_bits, COUNT(zero_fit_bits));
    check_bits(fitted + COUNT(zero_fit_bits), disconnected_tail, COUNT(disconnected_tail));
}

static void check_transform_error(
    umap_capi_status_t expected, const umap_capi_dense_transform_input_t *input,
    const float *fitted, size_t fitted_len, const umap_capi_dense_transform_config_t *config,
    umap_capi_dense_transform_t *sentinel) {
    umap_capi_dense_transform_t *out = sentinel;
    CHECK(umap_capi_transform_dense_embedding(input, fitted, fitted_len, config, &out) == expected);
    CHECK(out == sentinel);
}

static void check_transform_boundaries(void) {
    float fitted[COUNT(zero_fit_bits)];
    memcpy(fitted, zero_fit_bits, sizeof(fitted));
    umap_capi_dense_transform_config_t config = transform_config;
    config.transform_seed = 0;
    umap_capi_dense_transform_t *first = run_transform(&transform_input, fitted, COUNT(fitted), &config);
    config.transform_seed = UINT32_MAX;
    umap_capi_dense_transform_t *second = run_transform(&transform_input, fitted, COUNT(fitted), &config);
    const umap_capi_dense_transform_view_t a = read_transform(first, 3, 2);
    const umap_capi_dense_transform_view_t b = read_transform(second, 3, 2);
    CHECK(memcmp(a.coordinates, b.coordinates, a.coordinates_len * sizeof(float)) != 0);
    CHECK(umap_capi_dense_transform_release(first) == UMAP_CAPI_STATUS_OK);

    const uint64_t invalid_seeds[] = {(uint64_t)UINT32_MAX + 1u, UINT64_MAX};
    for (size_t i = 0; i < COUNT(invalid_seeds); ++i) {
        config.transform_seed = invalid_seeds[i];
        check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input,
            fitted, COUNT(fitted), &config, second);
    }
    config = transform_config;
    config.fit_epochs_mode = UINT32_MAX;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input, fitted, COUNT(fitted), &config, second);
    config.fit_epochs_mode = UMAP_CAPI_FIT_EPOCHS_DEFAULT;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input, fitted, COUNT(fitted), &config, second);
    const float invalid_cutoffs[] = {0.0f, -1.0f, -INFINITY, NAN};
    for (size_t i = 0; i < COUNT(invalid_cutoffs); ++i) {
        config = transform_config;
        config.disconnection_distance = invalid_cutoffs[i];
        check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input,
            fitted, COUNT(fitted), &config, second);
    }
    config = transform_config;
    config.local_connectivity = NAN;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input, fitted, COUNT(fitted), &config, second);
    config = transform_config;
    config.negative_sample_rate = 0.0;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &transform_input, fitted, COUNT(fitted), &config, second);
    config = transform_config;
    config.n_components = SIZE_MAX;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &transform_input, fitted, COUNT(fitted), &config, second);
    config.n_components = 0;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &transform_input, fitted, COUNT(fitted), &config, second);

    check_transform_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, NULL, fitted, COUNT(fitted), &transform_config, second);
    check_transform_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, &transform_input, fitted, COUNT(fitted), NULL, second);
    check_transform_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, &transform_input, NULL, COUNT(fitted), &transform_config, second);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_POINTER, &transform_input,
        MISALIGNED(const float, fitted), COUNT(fitted), &transform_config, second);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_POINTER,
        MISALIGNED(const umap_capi_dense_transform_input_t, &transform_input),
        fitted, COUNT(fitted), &transform_config, second);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_POINTER, &transform_input, fitted, COUNT(fitted),
        MISALIGNED(const umap_capi_dense_transform_config_t, &transform_config), second);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &transform_input,
        fitted, COUNT(fitted) - 1, &transform_config, second);

    umap_capi_dense_transform_input_t input = transform_input;
    input.n_training_rows = SIZE_MAX;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, fitted, COUNT(fitted), &transform_config, second);
    /* The feature shapes fit, but the output row/column product overflows. */
    const float wide_fitted[32] = {0};
    config = transform_config;
    config.n_neighbors = 1;
    config.n_components = COUNT(wide_fitted);
    input.n_training_rows = input.training_len = input.n_features = 1;
    input.n_query_rows = input.queries_len = SIZE_MAX / COUNT(wide_fitted) + 1;
    check_transform_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, wide_fitted, COUNT(wide_fitted), &config, second);
    input = transform_input;
    input.training = MISALIGNED(const float, transform_training);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_POINTER, &input, fitted, COUNT(fitted), &transform_config, second);
    input = transform_input;
    input.queries = NULL;
    check_transform_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, &input, fitted, COUNT(fitted), &transform_config, second);
    input.queries = MISALIGNED(const float, transform_queries);
    check_transform_error(UMAP_CAPI_STATUS_INVALID_POINTER, &input, fitted, COUNT(fitted), &transform_config, second);
    const float nonfinite_query[COUNT(transform_queries)] = {NAN};
    input.queries = nonfinite_query;
    check_transform_error(UMAP_CAPI_STATUS_NONFINITE_INPUT, &input, fitted, COUNT(fitted), &transform_config, second);

    umap_capi_dense_transform_t *slot = second;
    CHECK(umap_capi_transform_dense_embedding(&transform_input, fitted, COUNT(fitted),
        &transform_config, NULL) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_transform_dense_embedding(&transform_input, fitted, COUNT(fitted),
        &transform_config, MISALIGNED(umap_capi_dense_transform_t *, &slot)) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(slot == second);
    umap_capi_dense_transform_view_t view = {.n_rows = SIZE_MAX};
    CHECK(umap_capi_dense_transform_view(NULL, &view) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_transform_view(second, NULL) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_transform_view(second, MISALIGNED(umap_capi_dense_transform_view_t, &view))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_dense_transform_view(MISALIGNED(const umap_capi_dense_transform_t, second), &view)
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(view.n_rows == SIZE_MAX && view.coordinates == NULL);
    CHECK(umap_capi_dense_transform_release(MISALIGNED(umap_capi_dense_transform_t, second))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_dense_transform_release(second) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_dense_transform_release(NULL) == UMAP_CAPI_STATUS_OK);

    input.n_query_rows = 0;
    input.queries_len = 0;
    input.queries = NULL;
    umap_capi_dense_transform_t *empty = run_transform(&input, fitted, COUNT(fitted), &transform_config);
    read_transform(empty, 0, 2);
    CHECK(umap_capi_dense_transform_release(empty) == UMAP_CAPI_STATUS_OK);
}

static void check_directed_graph_transform(void) {
    /* Graph mode of umap-learn-0.6.0-transform-windows-x86_64.json. */
    static const size_t rows[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    static const size_t cols[] = {1, 0, 4, 2, 6, 5, 3, 4, 2, 7, 4, 3};
    static const uint32_t bits[] = {
        0x3f31b07fu, 0x3f2f31d5u, 0x3f291bdeu, 0x3f27b2b6u,
        0x3f3ab9e6u, 0x3f395844u, 0x3f2ad7c9u, 0x3f1bd014u,
        0x3f4d5cecu, 0x3f4c4cf4u, 0x3f20fe79u, 0x3f12b50du,
    };
    umap_capi_coo_graph_t *graph = NULL;
    CHECK(umap_capi_transform_dense_graph(&transform_input, 4, UMAP_CAPI_METRIC_EUCLIDEAN, 1.0, &graph)
        == UMAP_CAPI_STATUS_OK);
    CHECK(graph != NULL);
    umap_capi_coo_graph_t *sentinel = graph;
    CHECK(umap_capi_transform_dense_graph(&transform_input, 9, UMAP_CAPI_METRIC_EUCLIDEAN, 1.0, &graph)
        == UMAP_CAPI_STATUS_INVALID_K);
    CHECK(umap_capi_transform_dense_graph(&transform_input, 4, UINT32_MAX, 1.0, &graph)
        == UMAP_CAPI_STATUS_INVALID_METRIC);
    CHECK(umap_capi_transform_dense_graph(&transform_input, 4, UMAP_CAPI_METRIC_EUCLIDEAN, -1.0, &graph)
        == UMAP_CAPI_STATUS_INVALID_CONFIG);
    CHECK(graph == sentinel);
    umap_capi_coo_view_t view = {.n_rows = SIZE_MAX};
    CHECK(umap_capi_coo_graph_view(NULL, &view) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_coo_graph_view(graph, MISALIGNED(umap_capi_coo_view_t, &view))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(view.n_rows == SIZE_MAX && view.data == NULL);
    CHECK(umap_capi_coo_graph_view(graph, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 3 && view.n_cols == 8 && view.nnz == COUNT(bits));
    CHECK(view.row && view.col);
    CHECK(memcmp(view.row, rows, sizeof(rows)) == 0);
    CHECK(memcmp(view.col, cols, sizeof(cols)) == 0);
    check_bits(view.data, bits, COUNT(bits));
    CHECK(umap_capi_coo_graph_release(MISALIGNED(umap_capi_coo_graph_t, graph))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_coo_graph_release(graph) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_coo_graph_release(NULL) == UMAP_CAPI_STATUS_OK);

    /* Duplicate-query probe from reference/transform_reference.md. */
    const float training[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 10.0f, 20.0f, 21.0f, 25.0f, 30.0f};
    const float queries[] = {0.0f, 0.0f};
    umap_capi_dense_transform_input_t input = {
        COUNT(training), COUNT(queries), 1, COUNT(training), COUNT(queries), training, queries,
    };
    CHECK(umap_capi_transform_dense_graph(&input, 5, UMAP_CAPI_METRIC_EUCLIDEAN, 1.0, &graph)
        == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_coo_graph_view(graph, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 2 && view.n_cols == COUNT(training) && view.nnz == 10);
    for (size_t entry = 0; entry < view.nnz; ++entry) {
        CHECK(view.row[entry] == entry / 5 && view.col[entry] == entry % 5);
        CHECK(view.data[entry] == (entry % 5 == 4 ? 0.0f : 1.0f));
    }
    CHECK(umap_capi_coo_graph_release(graph) == UMAP_CAPI_STATUS_OK);

    input.n_query_rows = 0;
    input.queries_len = 0;
    input.queries = NULL;
    CHECK(umap_capi_transform_dense_graph(&input, 5, UMAP_CAPI_METRIC_EUCLIDEAN, 1.0, &graph)
        == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_coo_graph_view(graph, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 0 && view.n_cols == COUNT(training) && view.nnz == 0);
    CHECK(view.row == NULL && view.col == NULL && view.data == NULL);
    CHECK(umap_capi_coo_graph_release(graph) == UMAP_CAPI_STATUS_OK);
}

static void check_dense_update(void) {
    /* Close-pair jitter and optimizer checkpoints from the public update-layout fixture. */
    const float combined[] = {
        0.0f, 0.0f, 0.0f,
        0.2f, 0.07f, 0.1f,
        0.25f, 0.89f, 0.13f,
        0.76f, 0.1f, 0.83f,
        0.99f, 0.95f, 0.37f,
        1.9f, 1.31f, 0.33f,
        2.61f, 1.7f, 0.9f,
        3.3f, 1.2f, 1.1f,
        0.31f, 0.9f, 0.2f,
        0.305f, 0.91f, 0.205f,
    };
    const uint32_t initial[] = {
        1082309325, 1091744602, 1087988334, 1088210070, 1069873797, 1060110722,
        1081684873, 1091066003, 1082368327, 1089405340, 0, 1092616192, 1092616192,
        0, 1081437117, 1058502120, 1077207337, 1082295373, 1077196900, 1082301952,
    };
    const uint32_t optimized[] = {
        1082513110, 1091659266, 1087546951, 1088518834, 1069873797, 1060110722,
        1081982604, 1090794034, 1082097466, 1089268539, 1034155853, 1092564056,
        1092538578, 1005109337, 1081743926, 1058384350, 1077300694, 1082434718,
        1077284236, 1082411564,
    };
    const size_t indptr[] = {0, 3, 7, 12, 15, 22, 25, 28, 31, 37, 40};
    const size_t columns[] = {
        1, 2, 8, 0, 2, 3, 8, 0, 1, 4, 8, 9, 1, 4, 8, 2, 3, 5, 6, 7,
        8, 9, 4, 6, 7, 4, 5, 7, 4, 5, 6, 0, 1, 2, 3, 4, 9, 2, 4, 8,
    };
    const uint32_t weights[] = {
        1065353216, 1057122905, 1056647969, 1065353216, 1057067398, 1065353216,
        1056759340, 1057122905, 1057067398, 1037220775, 1065353215, 1065308232,
        1065353216, 1059695368, 1051502904, 1037220775, 1059695368, 1065353216,
        1043941809, 1052612122, 1065353216, 1064125748, 1065353216, 1065197112,
        1059460393, 1043941809, 1065197112, 1065353216, 1052612122, 1059460393,
        1065353216, 1056647969, 1056759340, 1065353215, 1051502904, 1065353216,
        1065353216, 1065308232, 1064125748, 1065353216,
    };
    const umap_capi_dense_embedding_config_t fit_config = {
        .n_neighbors = 4, .n_components = 2, .n_epochs = 30,
        .a = 1.7, .b = 0.9, .learning_rate = 0.13,
        .repulsion_strength = 1.0, .negative_sample_rate = 2.0,
        .metric = UMAP_CAPI_METRIC_EUCLIDEAN, .init = UMAP_CAPI_INIT_RANDOM, .seed = 42,
    };
    umap_capi_dense_update_config_t update_config = {
        .n_neighbors = 4, .a = 1.7, .b = 0.9, .learning_rate = 0.13,
        .repulsion_strength = 1.0, .negative_sample_rate = 2.0,
        .local_connectivity = 1.0, .set_op_mix_ratio = 1.0, .transform_seed = 42,
    };
    umap_capi_dense_embedding_t *fitted = NULL;
    CHECK(umap_capi_fit_dense_embedding_seeded(combined, 24, 8, 3, &fit_config, &fitted)
        == UMAP_CAPI_STATUS_OK);
    umap_capi_dense_embedding_view_t original = {0};
    CHECK(umap_capi_dense_embedding_view(fitted, &original) == UMAP_CAPI_STATUS_OK);
    CHECK(original.coordinates_len == 16);
    float before[16];
    memcpy(before, original.coordinates, sizeof before);

    for (size_t epochs = 0; epochs <= 2; ++epochs) {
        update_config.n_epochs = epochs;
        umap_capi_dense_embedding_t *updated = NULL;
        CHECK(umap_capi_update_dense_embedding(
            combined, COUNT(combined), 10, 3, fitted, &update_config, &updated
        ) == UMAP_CAPI_STATUS_OK);
        CHECK(updated != NULL && updated != fitted);
        umap_capi_dense_embedding_view_t view = {0};
        CHECK(umap_capi_dense_embedding_view(updated, &view) == UMAP_CAPI_STATUS_OK);
        CHECK(view.n_rows == 10 && view.n_components == 2 && view.coordinates_len == COUNT(initial));
        check_bits(view.coordinates, epochs == 2 ? optimized : initial, COUNT(initial));
        CHECK(view.graph.n_rows == 10 && view.graph.n_cols == 10);
        CHECK(view.graph.indptr_len == COUNT(indptr));
        CHECK(view.graph.indices_len == COUNT(columns) && view.graph.data_len == COUNT(weights));
        CHECK(memcmp(view.graph.indptr, indptr, sizeof indptr) == 0);
        CHECK(memcmp(view.graph.indices, columns, sizeof columns) == 0);
        check_bits(view.graph.data, weights, COUNT(weights));
        CHECK(umap_capi_dense_embedding_release(updated) == UMAP_CAPI_STATUS_OK);
    }

    umap_capi_dense_embedding_t *unchanged = fitted;
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined), 10, 3, fitted, NULL, &unchanged
    ) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(unchanged == fitted);
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined), SIZE_MAX, 3, fitted, &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_SHAPE);
    CHECK(unchanged == fitted);
    CHECK(umap_capi_update_dense_embedding(
        combined, 24, 8, 3, fitted, &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_SHAPE);
    CHECK(unchanged == fitted);
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined) - 1, 10, 3, fitted, &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_LENGTH);
    CHECK(unchanged == fitted);
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined), 10, 3, MISALIGNED(umap_capi_dense_embedding_t, fitted),
        &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(unchanged == fitted);
    update_config.n_neighbors = 9;
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined), 10, 3, fitted, &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_K);
    CHECK(unchanged == fitted);
    update_config.n_neighbors = 4;
    update_config.set_op_mix_ratio = 1.0000000000000002;
    CHECK(umap_capi_update_dense_embedding(
        combined, COUNT(combined), 10, 3, fitted, &update_config, &unchanged
    ) == UMAP_CAPI_STATUS_INVALID_CONFIG);
    CHECK(unchanged == fitted && memcmp(original.coordinates, before, sizeof before) == 0);
    CHECK(umap_capi_dense_embedding_release(fitted) == UMAP_CAPI_STATUS_OK);
}

int main(void) {
    check_graph_only();
    check_graph_options();
    check_seeded_checkpoints();
    check_boundaries();
    check_empty_graph();
    check_optimized_transform();
    check_transform_connectivity();
    check_transform_epochs();
    check_transform_cutoff();
    check_transform_boundaries();
    check_directed_graph_transform();
    check_dense_update();
    return EXIT_SUCCESS;
}
