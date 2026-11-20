import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from umap_alternative import UMAP, _native


FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json"
    ).read_text(encoding="utf-8")
)


def fixture_input(name):
    return np.array(FIXTURE["inputs"][name]["bits_uint32"], dtype=np.uint32).view(np.float32)


TRAINING = fixture_input("training")
QUERIES = fixture_input("transform_queries")


def graph_model(**overrides):
    keys = (
        "n_components",
        "n_neighbors",
        "metric",
        "init",
        "random_state",
        "n_jobs",
        "n_epochs",
        "compatibility_layout",
        "set_op_mix_ratio",
        "local_connectivity",
    )
    parameters = {key: FIXTURE["parameters"][key] for key in keys}
    return UMAP(transform_mode="graph", **(parameters | overrides))


def assert_coo(graph, shape, row, col, bits):
    assert isinstance(graph, coo_matrix)
    assert graph.format == "coo"
    assert graph.dtype == np.dtype(np.float32)
    assert graph.shape == tuple(shape)
    assert graph.nnz == len(bits)
    np.testing.assert_array_equal(graph.row, row)
    np.testing.assert_array_equal(graph.col, col)
    np.testing.assert_array_equal(graph.data.view(np.uint32), bits)


def test_held_out_dense_graph_matches_public_coo_structure_and_bits():
    expected = FIXTURE["cases"]["graph"]["transform"]
    model = graph_model().fit(TRAINING)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = model.transform(QUERIES)

    expected_graph = expected["output"]
    assert type(output).__name__ == expected_graph["type"]
    assert output.format == expected_graph["format"]
    assert output.dtype.name == expected_graph["dtype"]
    assert output.nnz == expected_graph["nnz"]
    assert_coo(
        output,
        expected_graph["shape"],
        expected_graph["row"],
        expected_graph["col"],
        expected_graph["data_bits_uint32"],
    )
    assert [(warning.category.__name__, str(warning.message)) for warning in caught] == [
        (warning["category"], warning["message"]) for warning in expected["warnings"]
    ]


@pytest.mark.parametrize(
    "query",
    [TRAINING.copy(), TRAINING.tolist(), np.asfortranarray(TRAINING.astype(np.float64))],
    ids=["copy", "list", "fortran-float64"],
)
def test_fitted_data_fast_path_returns_csr_by_identity(query):
    model = graph_model().fit(TRAINING)

    assert isinstance(model.graph_, csr_matrix)
    assert model.transform(query) is model.graph_


def test_fast_path_distinguishes_signed_zero_and_does_not_alias_training_input():
    training = TRAINING.copy()
    model = graph_model().fit(training)
    training[0, 0] = -0.0

    assert model.transform(TRAINING) is model.graph_
    output = model.transform(training)
    assert isinstance(output, coo_matrix)
    assert output.shape == (8, 8)
    assert output.nnz == 32
    assert output.col[0] == 0
    assert output.data[0] == 1


@pytest.mark.parametrize("n_queries", [1, 2])
def test_one_query_and_duplicate_query_coordinates_keep_stored_zero(n_queries):
    training = np.array([0, 0, 0, 0, 0.1, 10, 20, 21, 25, 30], dtype=np.float32)[:, None]
    model = graph_model(n_neighbors=5).fit(training)

    output = model.transform(np.zeros((n_queries, 1), dtype=np.float64))

    assert_coo(
        output,
        (n_queries, 10),
        np.repeat(np.arange(n_queries), 5),
        [0, 1, 2, 3, 4] * n_queries,
        [1065353216, 1065353216, 1065353216, 1065353216, 0] * n_queries,
    )
    assert output.count_nonzero() == 4 * n_queries


def test_transform_uses_effective_neighbor_count_after_truncation():
    with pytest.warns(UserWarning):
        model = graph_model(n_neighbors=10).fit(TRAINING[:2])

    assert model.n_neighbors_ == 1
    assert_coo(model.transform(QUERIES[:1]), (1, 2), [0], [1], [1057633385])


def test_unfitted_transform_reports_the_public_error_before_input_validation():
    with pytest.raises(AttributeError, match="^'UMAP' object has no attribute '_raw_data'$"):
        graph_model().transform(QUERIES[:0])


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (QUERIES[:0], r"Found array with 0 sample\(s\) \(shape=\(0, 3\)\)"),
        (QUERIES[:, :0], r"Found array with 0 feature\(s\) \(shape=\(3, 0\)\)"),
        (QUERIES[0], "Expected 2D array, got 1D array instead"),
        (QUERIES[None, :, :], "Found array with dim 3, while dim <= 2 is required"),
        (QUERIES[:, :2], "X has 2 features, but UMAP is expecting 3 features as input"),
        (np.pad(QUERIES, ((0, 0), (0, 1))), "X has 4 features, but UMAP is expecting 3 features"),
        ([[0, 0, np.nan]], r"Input contains NaN\."),
        ([[0, 0, np.inf]], "Input contains infinity or a value too large"),
        ([[0, 0, np.finfo(np.float32).max]], "distance from query row 0 .* is not finite"),
    ],
)
def test_transform_rejects_invalid_dense_queries(queries, message):
    with pytest.raises(ValueError, match=message):
        graph_model().fit(TRAINING).transform(queries)


def test_transform_rejects_float32_cast_overflow():
    model = graph_model().fit(TRAINING)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast"):
        with pytest.raises(ValueError, match="Input contains infinity or a value too large"):
            model.transform([[0, 0, 1e100]])


def test_transform_stays_consistent_across_failed_and_successful_refits():
    model = graph_model().fit(TRAINING)
    original_graph = model.graph_
    original_result = model.transform(QUERIES)
    replacement = TRAINING[:5, :2]
    new_queries = QUERIES[:, :2]

    model.set_params(metric="linfty", n_neighbors=3)
    with pytest.raises(ValueError, match="Length of x = 5, length of y = 4"):
        model.fit(replacement, y=[0, 0, 1, 1])

    assert model.transform(TRAINING) is original_graph
    assert model.n_features_in_ == 3
    assert model.n_neighbors_ == 4
    assert_coo(
        model.transform(QUERIES), original_result.shape,
        original_result.row, original_result.col, original_result.data.view(np.uint32),
    )

    assert model.fit(replacement) is model
    assert model.transform(replacement) is model.graph_
    assert model.graph_ is not original_graph
    assert model.n_features_in_ == 2
    assert model.n_neighbors_ == 3
    fresh = graph_model(metric="chebyshev", n_neighbors=3).fit(replacement).transform(new_queries)
    assert fresh.shape == (3, 5)
    assert_coo(model.transform(new_queries), fresh.shape, fresh.row, fresh.col, fresh.data.view(np.uint32))
    with pytest.raises(ValueError, match="X has 3 features, but UMAP is expecting 2 features"):
        model.transform(TRAINING)


def test_failed_first_fit_does_not_enable_transform():
    model = graph_model()
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit([[0, 0, np.nan]])
    with pytest.raises(AttributeError, match="has no attribute '_raw_data'"):
        model.transform(QUERIES)


def test_singleton_refit_replaces_the_previous_transform_state():
    model = graph_model().fit(TRAINING)
    model.fit(TRAINING[:1])

    assert not hasattr(model, "graph_")
    for queries in (TRAINING[:1], QUERIES):
        with pytest.raises(ValueError, match="Transform unavailable .* only a single data sample"):
            model.transform(queries)


def test_transform_rejects_sparse_feature_queries():
    model = graph_model().fit(TRAINING)
    with pytest.raises(TypeError, match="sparse input is not supported"):
        model.transform(csr_matrix(QUERIES))

def test_native_transform_translates_core_shape_errors():
    with pytest.raises(ValueError, match="invalid query input: matrix shape 1x2 does not match"):
        _native._transform_dense_graph([0, 0, 1, 1], 2, [0], 1, 2, 1, "l2")
