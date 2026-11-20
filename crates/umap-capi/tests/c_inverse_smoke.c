/* Compile as C11 against the public header and the cdylib. */
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

static void check_bits(const float *actual, const uint32_t *expected, size_t len) {
    CHECK(actual != NULL);
    for (size_t i = 0; i < len; ++i) {
        uint32_t bits;
        memcpy(&bits, actual + i, sizeof(bits));
        if (bits != expected[i]) {
            fprintf(stderr, "index %zu: expected %08" PRIx32 ", got %08" PRIx32 "\n",
                i, expected[i], bits);
        }
        CHECK(bits == expected[i]);
    }
}

static umap_capi_dense_inverse_t *run_inverse(const umap_capi_dense_inverse_input_t *input,
    const umap_capi_dense_transform_config_t *config, const umap_capi_inverse_scales_t *scales) {
    umap_capi_dense_inverse_t *handle = NULL;
    CHECK(umap_capi_inverse_transform_dense(input, config, scales, &handle) == UMAP_CAPI_STATUS_OK);
    CHECK(handle != NULL);
    return handle;
}

static void check_result(umap_capi_dense_inverse_t *handle, size_t exterior, const uint32_t expected[9]) {
    umap_capi_dense_inverse_view_t view = {0};
    CHECK(umap_capi_dense_inverse_view(handle, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.n_rows == 3 && view.n_features == 3 && view.coordinates_len == 9);
    CHECK(view.exterior_queries == exterior);
    check_bits(view.coordinates, expected, view.coordinates_len);
    CHECK(umap_capi_dense_inverse_release(handle) == UMAP_CAPI_STATUS_OK);
}

/* Geometry and features from umap-learn-0.6.0-inverse-metrics-x86_64.json.
 * The simplex order and locations come from public SciPy Delaunay with QJ. */
static const float training[] = {
    0.12f, 0.07f, 1.2f, 1.0f, 4.0f, 2.0f, 0.78f, 1.2f, 3.1f, 2.2f, -3.0f, 0.5f,
    -0.1f, 0.7f, 6.0f, 0.3f, -0.5f, -0.9f, 5.0f, 6.0f, 7.0f,
};
static const float embedding[] = {
    2.0f, 3.6363637447357178f, 4.0f, 0.0f, 6.0f, 6.363636493682861f, 9.0f,
    0.9090909361839294f, 7.0f, 10.0f, 10.0f, 3.6363637447357178f, 0.0f, 5.454545497894287f,
};
static const float queries[] = {3.0f, 4.0f, 5.0f, 5.0f, 12.0f, -1.0f};
static const size_t simplices[] = {2, 4, 6, 4, 2, 5, 2, 3, 5, 3, 2, 1, 1, 0, 6, 0, 2, 6, 2, 0, 1};
static const size_t locations[] = {6, 6, UMAP_CAPI_INVERSE_EXTERIOR};
static const umap_capi_dense_inverse_input_t metric_input = {
    7, 3, 3, COUNT(training), COUNT(embedding), COUNT(queries), COUNT(simplices), COUNT(locations),
    training, embedding, queries, simplices, locations,
};
static const umap_capi_dense_transform_config_t metric_config = {
    .n_neighbors = 4,
    .n_components = 2,
    .fit_epochs = 30,
    .a = 1.2,
    .b = 0.9,
    .learning_rate = 1.0,
    .repulsion_strength = 1.0,
    .negative_sample_rate = 5.0,
    .local_connectivity = 1.0,
    .transform_seed = UINT32_MAX,
    .metric = UMAP_CAPI_METRIC_EUCLIDEAN,
    .fit_epochs_mode = UMAP_CAPI_FIT_EPOCHS_EXPLICIT,
    .disconnection_distance = INFINITY,
};

static void check_original_euclidean(void) {
    /* original_interior and original_exterior in umap-learn-0.6.0-inverse-windows-x86_64.json. */
    static const float features[] = {
        0.0f, 0.0f, 0.1f, 0.9f, 0.1f, 0.2f, 0.3f, 1.0f, 0.4f, 1.4f, 0.9f, 0.3f,
        0.2f, 0.6f, 1.1f, 1.0f, 0.2f, 1.3f, 1.6f, 1.3f, 1.0f, 0.7f, 1.7f, 0.8f,
    };
    static const uint32_t fitted_bits[] = {
        0x40941e62u, 0x40b77443u, 0x40a71eecu, 0x40a3f28bu,
        0x40a72047u, 0x40d339d5u, 0x409011dbu, 0x40f230e6u,
        0x407059fau, 0x40bb193au, 0x4046dd9du, 0x40adab19u,
        0x40a75088u, 0x4101ba5bu, 0x40b1b81bu, 0x40f17d9du,
    };
    static const float probe_queries[][6] = {
        {4.5f, 5.7f, 4.7f, 6.5f, 5.25f, 7.0f},
        {5.372533321380615f, 4.18798303604126f, 3.0012710094451904f,
         7.245647430419922f, 3.3534951210021973f, 7.748203277587891f},
    };
    static const size_t triangles[] = {1, 0, 5, 2, 1, 7, 2, 0, 1, 6, 3, 7, 3, 2, 7, 0, 4, 5, 4, 3, 5, 2, 4, 0, 3, 4, 2};
    static const size_t found[][3] = {
        {0, 8, 4}, {UMAP_CAPI_INVERSE_EXTERIOR, UMAP_CAPI_INVERSE_EXTERIOR, UMAP_CAPI_INVERSE_EXTERIOR},
    };
    static const uint32_t expected[][9] = {
        {0x3e826e40u, 0x3e3c737au, 0x3ec8a01fu, 0x3ee9cc96u, 0x3f2542e9u, 0x3e8e0422u,
         0x3f30ebaau, 0x3f9b0907u, 0x3f02bbceu},
        {0x3f5842e0u, 0x3d4e35a4u, 0xbf3e1e4du, 0x3f1b9dcau, 0x3f50b1cau, 0xbf45e8abu,
         0x3e4c1b1au, 0x3fe34fcbu, 0xbef2d558u},
        {0xbf537ea0u, 0xbf1e48f4u, 0xbf1817eeu, 0x400420f1u, 0x3dfa9538u, 0x3f87a2efu,
         0xbe562f69u, 0x40086c91u, 0x3fc82169u},
    };
    umap_capi_dense_inverse_t *results[COUNT(expected)];
    {
        float x[COUNT(features)], fitted[COUNT(fitted_bits)], q[COUNT(probe_queries[0])];
        size_t simplex_copy[COUNT(triangles)], located[COUNT(found[0])];
        memcpy(x, features, sizeof(x));
        memcpy(fitted, fitted_bits, sizeof(fitted));
        memcpy(simplex_copy, triangles, sizeof(simplex_copy));
        umap_capi_dense_inverse_input_t input = {
            8, 3, 3, COUNT(x), COUNT(fitted), COUNT(q), COUNT(simplex_copy), COUNT(located),
            x, fitted, q, simplex_copy, located,
        };
        umap_capi_dense_transform_config_t config = metric_config;
        config.a = 1.5769434607659205;
        config.b = 0.89506087820881486;
        config.transform_seed = 24;
        for (size_t i = 0; i < COUNT(results); ++i) {
            const size_t geometry = i == 2 ? 1 : 0;
            config.fit_epochs = i == 0 ? 0 : 45;
            memcpy(q, probe_queries[geometry], sizeof(q));
            memcpy(located, found[geometry], sizeof(located));
            results[i] = run_inverse(&input, &config, NULL);
            CHECK(memcmp(q, probe_queries[geometry], sizeof(q)) == 0);
            CHECK(memcmp(located, found[geometry], sizeof(located)) == 0);
        }
        CHECK(memcmp(x, features, sizeof(x)) == 0);
        CHECK(memcmp(simplex_copy, triangles, sizeof(simplex_copy)) == 0);
        check_bits(fitted, fitted_bits, COUNT(fitted_bits));
    }
    for (size_t i = 0; i < COUNT(results); ++i) {
        check_result(results[i], i == 2 ? 3 : 0, expected[i]);
    }
}

static void check_metrics(void) {
    /* Positive repulsion, 30 fit epochs, from the public metric fixture. */
    static const struct {
        umap_capi_metric_t metric;
        float cutoff;
        uint32_t bits[9];
    } cases[] = {
        {UMAP_CAPI_METRIC_MANHATTAN, INFINITY,
         {0xc0298402u, 0x3f196114u, 0x3ffe7d97u, 0x3f9b2d6fu, 0x3ff1a0a6u, 0xbe60487cu,
          0x4005541du, 0xbff4a5ffu, 0x3f5c3e89u}},
        {UMAP_CAPI_METRIC_CHEBYSHEV, INFINITY,
         {0x40592dd8u, 0x3e933f40u, 0x3fb34eabu, 0x3ee1fe94u, 0x401cc200u, 0x3f50e639u,
          0x3fb9f8ecu, 0xc03599aeu, 0x3e8be32du}},
        {UMAP_CAPI_METRIC_COSINE, 2.0f,
         {0xbf9a1f57u, 0xbde76d8eu, 0x407cbe63u, 0xbf140185u, 0x3ffe982du, 0x40a6c8f2u,
          0x3f2a2500u, 0xbf77cbfbu, 0x3f6d9f9du}},
        {UMAP_CAPI_METRIC_CORRELATION, 2.0f,
         {0x3fbc2eadu, 0x3eeeda01u, 0x40b99802u, 0x3f233ba8u, 0x3ffd2b27u, 0x40a86d3du,
          0x3f40604eu, 0xbfecb1f6u, 0x3f857cd6u}},
    };
    for (size_t i = 0; i < COUNT(cases); ++i) {
        umap_capi_dense_transform_config_t config = metric_config;
        config.metric = cases[i].metric;
        config.disconnection_distance = cases[i].cutoff;
        check_result(run_inverse(&metric_input, &config, NULL), 1, cases[i].bits);
    }
}

static void check_ranked_scales(void) {
    /* Reproduce the input distances of supplied_width and supplied_cutoff.
     * The full supplied width stays distinct from the smoothing target k=4. */
    static const uint32_t wide_bits[] = {
        0x00000000u, 0x3f4ed024u, 0x3f5a8631u, 0x3fb2b9bbu, 0x3fc275e5u,
        0x00000000u, 0x3f8edac6u, 0x3fc275e5u, 0x3ffb0d7du, 0x3fffb33du,
        0x00000000u, 0x3f5a8631u, 0x3f8edac6u, 0x3f9177eeu, 0x3fcf172bu,
        0x00000000u, 0x3fa2d323u, 0x3fb2b9bbu, 0x3ff36a68u, 0x40262609u,
        0x00000000u, 0x3f9177eeu, 0x3fe583a9u, 0x3ffb0d7du, 0x40261d65u,
        0x00000000u, 0x3f4ed024u, 0x3fa2d323u, 0x3fcf172bu, 0x3fffb33du,
        0x00000000u, 0x401ed9abu, 0x402fc671u, 0x40314c1au, 0x4063e4ceu,
    };
    static const uint32_t cutoff_bits[] = {
        0x00000000u, 0x3f4ed024u, 0x3f5a8631u, 0x3fb2b9bbu,
        0x00000000u, 0x3f8edac6u, 0x3fc275e5u, 0x7f800000u,
        0x00000000u, 0x3f5a8631u, 0x3f8edac6u, 0x3f9177eeu,
        0x00000000u, 0x3fa2d323u, 0x3fb2b9bbu, 0x7f800000u,
        0x00000000u, 0x3f9177eeu, 0x7f800000u, 0x7f800000u,
        0x00000000u, 0x3f4ed024u, 0x3fa2d323u, 0x3fcf172bu,
        0x00000000u, 0x7f800000u, 0x7f800000u, 0x7f800000u,
    };
    static const uint32_t expected[][9] = {
        {0x3dfe9941u, 0x3c613233u, 0x3f9128dbu, 0x3f482d0fu, 0x3e84cda6u, 0x40454bfau,
         0x3ffb9459u, 0xbfe53606u, 0x3f4931fbu},
        {0xc089b6f8u, 0xc06e94f3u, 0xc024bc2au, 0x3f812059u, 0xbf400712u, 0x3fb8fb57u,
         0x3fc45bd9u, 0xbfae7081u, 0x3eb00e33u},
    };
    float wide[COUNT(wide_bits)], cutoff[COUNT(cutoff_bits)];
    memcpy(wide, wide_bits, sizeof(wide));
    memcpy(cutoff, cutoff_bits, sizeof(cutoff));
    const umap_capi_inverse_scales_t sources[] = {
        {.ranked_len = COUNT(wide), .ranked_width = 5, .ranked_distances = wide},
        {.ranked_len = COUNT(cutoff), .ranked_width = 4, .ranked_distances = cutoff},
    };
    const double connectivity[] = {1.3, 1.0};
    for (size_t i = 0; i < COUNT(sources); ++i) {
        umap_capi_dense_transform_config_t config = metric_config;
        config.local_connectivity = connectivity[i];
        check_result(run_inverse(&metric_input, &config, &sources[i]), 1, expected[i]);
    }
    check_bits(wide, wide_bits, COUNT(wide_bits));
    check_bits(cutoff, cutoff_bits, COUNT(cutoff_bits));
}

static void check_fitted_scales(void) {
    /* Local scales from the Rust scale fitter for the finite_cutoff case. */
    static const umap_capi_inverse_local_scale_t pairs[] = {
        {2.183414698e0f, 8.123474121e-1f}, {3.016355276e0f, 1.844674407e19f},
        {2.307054281e0f, 1.062591553e0f}, {3.438022614e0f, 6.553600000e4f},
        {3.071547031e0f, 1.844674407e19f}, {2.183414698e0f, 1.310720000e5f},
        {0.0f, 1.844674407e19f},
    };
    static const uint32_t expected[] = {
        0xc1237271u, 0xc11486f4u, 0xc0e50af3u, 0xc08a5f27u, 0xc05aac1eu, 0xc03495b1u,
        0x4023efa8u, 0xc0181a82u, 0x3ea4e31au,
    };
    umap_capi_dense_transform_config_t config = metric_config;
    config.disconnection_distance = 4.0f;
    check_result(run_inverse(&metric_input, &config, NULL), 1, expected);

    umap_capi_inverse_local_scale_t mutable_pairs[COUNT(pairs)];
    memcpy(mutable_pairs, pairs, sizeof(mutable_pairs));
    const umap_capi_inverse_scales_t source = {.fitted_len = COUNT(mutable_pairs), .fitted = mutable_pairs};
    check_result(run_inverse(&metric_input, &metric_config, &source), 1, expected);
    CHECK(memcmp(mutable_pairs, pairs, sizeof(mutable_pairs)) == 0);
}

static void check_error(umap_capi_status_t expected, const umap_capi_dense_inverse_input_t *input,
    const umap_capi_dense_transform_config_t *config, const umap_capi_inverse_scales_t *scales,
    umap_capi_dense_inverse_t *sentinel) {
    umap_capi_dense_inverse_t *output = sentinel;
    const umap_capi_status_t status = umap_capi_inverse_transform_dense(input, config, scales, &output);
    if (status != expected) {
        fprintf(stderr, "expected status %" PRIu32 ", got %" PRIu32 "\n", expected, status);
    }
    CHECK(status == expected && output == sentinel);
}

static void check_inverse_boundaries(void) {
    umap_capi_dense_inverse_t *sentinel = run_inverse(&metric_input, &metric_config, NULL);
    umap_capi_dense_inverse_input_t input = metric_input;
    input.n_query_rows = 0;
    input.queries = NULL;
    input.queries_len = 0;
    input.query_simplices = NULL;
    input.query_simplices_len = 0;
    check_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, &metric_config, NULL, sentinel);

    input = metric_input;
    input.embedding_len--;
    check_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &input, &metric_config, NULL, sentinel);
    input = metric_input;
    input.simplices_len--;
    check_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &input, &metric_config, NULL, sentinel);
    input.simplices_len = (size_t)PTRDIFF_MAX / sizeof(size_t) + 1;
    check_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &input, &metric_config, NULL, sentinel);
    input = metric_input;
    input.query_simplices_len--;
    check_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &input, &metric_config, NULL, sentinel);
    input = metric_input;
    input.n_training_rows = SIZE_MAX;
    check_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, &metric_config, NULL, sentinel);

    const size_t bad_location[] = {COUNT(simplices) / 3, 0, UMAP_CAPI_INVERSE_EXTERIOR};
    input = metric_input;
    input.query_simplices = bad_location;
    check_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, &metric_config, NULL, sentinel);
    size_t bad_simplex[] = {0, 1, 7};
    input = metric_input;
    input.simplices = bad_simplex;
    input.simplices_len = COUNT(bad_simplex);
    const size_t simplex_zero[] = {0, 0, 0};
    input.query_simplices = simplex_zero;
    check_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, &metric_config, NULL, sentinel);
    bad_simplex[2] = 1;
    check_error(UMAP_CAPI_STATUS_INVALID_SHAPE, &input, &metric_config, NULL, sentinel);

    input = metric_input;
    input.queries = NULL;
    check_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, &input, &metric_config, NULL, sentinel);
    input = metric_input;
    input.simplices = MISALIGNED(const size_t, simplices);
    check_error(UMAP_CAPI_STATUS_INVALID_POINTER, &input, &metric_config, NULL, sentinel);
    const float nonfinite_queries[COUNT(queries)] = {NAN};
    input = metric_input;
    input.queries = nonfinite_queries;
    check_error(UMAP_CAPI_STATUS_NONFINITE_INPUT, &input, &metric_config, NULL, sentinel);

    umap_capi_dense_transform_config_t config = metric_config;
    config.metric = UINT32_MAX;
    check_error(UMAP_CAPI_STATUS_INVALID_METRIC, &metric_input, &config, NULL, sentinel);
    config = metric_config;
    config.transform_seed = (uint64_t)UINT32_MAX + 1u;
    check_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &metric_input, &config, NULL, sentinel);
    config = metric_config;
    config.disconnection_distance = 0.0f;
    check_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &metric_input, &config, NULL, sentinel);

    umap_capi_inverse_local_scale_t bad_pairs[COUNT(training) / 3] = {{0.0f, 0.0f}};
    umap_capi_inverse_scales_t source = {.fitted_len = COUNT(bad_pairs), .fitted = bad_pairs};
    check_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &metric_input, &metric_config, &source, sentinel);
    source.fitted_len--;
    check_error(UMAP_CAPI_STATUS_INVALID_LENGTH, &metric_input, &metric_config, &source, sentinel);
    const float ranked[COUNT(training)] = {0};
    source = (umap_capi_inverse_scales_t){.ranked_len = COUNT(ranked), .ranked_width = 3, .ranked_distances = ranked};
    check_error(UMAP_CAPI_STATUS_INVALID_K, &metric_input, &metric_config, &source, sentinel);
    source.fitted = bad_pairs;
    source.fitted_len = COUNT(bad_pairs);
    check_error(UMAP_CAPI_STATUS_INVALID_CONFIG, &metric_input, &metric_config, &source, sentinel);

    CHECK(umap_capi_inverse_transform_dense(&metric_input, &metric_config, NULL, NULL)
        == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    check_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, NULL, &metric_config, NULL, sentinel);
    check_error(UMAP_CAPI_STATUS_NULL_ARGUMENT, &metric_input, NULL, NULL, sentinel);
    check_error(UMAP_CAPI_STATUS_INVALID_POINTER, &metric_input,
        MISALIGNED(const umap_capi_dense_transform_config_t, &metric_config), NULL, sentinel);
    umap_capi_dense_inverse_t *slot = sentinel;
    CHECK(umap_capi_inverse_transform_dense(&metric_input, &metric_config, NULL,
        MISALIGNED(umap_capi_dense_inverse_t *, &slot)) == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(slot == sentinel);

    umap_capi_dense_inverse_view_t view = {.n_rows = SIZE_MAX};
    CHECK(umap_capi_dense_inverse_view(NULL, &view) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_inverse_view(sentinel, NULL) == UMAP_CAPI_STATUS_NULL_ARGUMENT);
    CHECK(umap_capi_dense_inverse_view(sentinel, MISALIGNED(umap_capi_dense_inverse_view_t, &view))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(view.n_rows == SIZE_MAX && view.coordinates == NULL);
    CHECK(umap_capi_dense_inverse_view(sentinel, &view) == UMAP_CAPI_STATUS_OK);
    CHECK(view.exterior_queries == 1);
    CHECK(umap_capi_dense_inverse_release(MISALIGNED(umap_capi_dense_inverse_t, sentinel))
        == UMAP_CAPI_STATUS_INVALID_POINTER);
    CHECK(umap_capi_dense_inverse_release(sentinel) == UMAP_CAPI_STATUS_OK);
    CHECK(umap_capi_dense_inverse_release(NULL) == UMAP_CAPI_STATUS_OK);
}

int main(void) {
    check_original_euclidean();
    check_metrics();
    check_ranked_scales();
    check_fitted_scales();
    check_inverse_boundaries();
    return EXIT_SUCCESS;
}
