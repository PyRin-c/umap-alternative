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

int main(void) {
    check_graph_only();
    check_seeded_checkpoints();
    check_boundaries();
    check_empty_graph();
    return EXIT_SUCCESS;
}
