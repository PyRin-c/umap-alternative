import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from umap_alternative import UMAP


BASELINE = np.asarray(
    [[0.0, 0.0], [1.0, 0.1], [0.2, 1.3], [1.7, 2.1]], dtype=np.float32
)
EXPECTED_INDPTR = np.array([0, 2, 5, 8, 10], dtype=np.int64)
EXPECTED_INDICES = np.array([1, 2, 0, 2, 3, 0, 1, 3, 1, 2], dtype=np.int64)
EXPECTED_BITS = np.array(
    [
        1065353216,
        1065353216,
        1065353216,
        1062463265,
        1058390034,
        1065353216,
        1062463265,
        1065353216,
        1058390034,
        1065353216,
    ],
    dtype=np.uint32,
)
def graph_model(**kwargs):
    parameters = dict(
        n_components=2,
        n_neighbors=3,
        metric="euclidean",
        init="random",
        random_state=42,
        n_jobs=1,
        n_epochs=30,
        compatibility_layout=True,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        transform_mode="graph",
    )
    parameters.update(kwargs)
    return UMAP(**parameters)


def assert_baseline_graph(graph):
    assert isinstance(graph, csr_matrix)
    assert graph.shape == (4, 4)
    assert graph.format == "csr"
    assert graph.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(graph.indptr, EXPECTED_INDPTR)
    np.testing.assert_array_equal(graph.indices, EXPECTED_INDICES)
    actual_bits = graph.data.view(np.uint32)
    for position, (actual, expected) in enumerate(zip(actual_bits, EXPECTED_BITS)):
        assert int(actual) == int(expected), (position, int(actual), int(expected))


def test_graph_fit_is_sklearn_shaped_and_matches_public_csr_contract():
    model = graph_model()

    returned = model.fit(BASELINE)

    assert isinstance(model, BaseEstimator)
    assert returned is model
    assert_baseline_graph(model.graph_)


def test_graph_fit_transform_returns_the_graph_object_by_identity():
    model = graph_model()

    returned = model.fit_transform(BASELINE)

    assert returned is model.graph_
    assert_baseline_graph(returned)


def test_float64_is_cast_to_float32_before_native_fit():
    returned = graph_model().fit_transform(BASELINE.astype(np.float64))

    assert_baseline_graph(returned)


def test_large_neighbor_count_truncates_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = graph_model(n_neighbors=10).fit_transform(BASELINE)

    assert_baseline_graph(returned)
    assert [(type(item.message).__name__, str(item.message)) for item in caught] == [
        (
            "UserWarning",
            "n_neighbors is larger than the dataset size; truncating to "
            "X.shape[0] - 1",
        )
    ]


def test_two_rows_produces_empty_graph_and_public_warnings():
    data = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = graph_model().fit_transform(data)

    assert returned.shape == (2, 2)
    assert returned.nnz == 0
    assert returned.indptr.tolist() == [0, 0, 0]
    assert returned.indices.tolist() == []
    assert returned.data.dtype == np.dtype(np.float32)
    assert [str(item.message) for item in caught] == [
        "n_neighbors is larger than the dataset size; truncating to "
        "X.shape[0] - 1",
        "A large number of your vertices were disconnected from the manifold.\n"
        "Disconnection_distance = inf has removed 0 edges.\n"
        "It has fully disconnected 2 vertices.\n"
        "You might consider using find_disconnected_points() to find and remove "
        "these points from your data.\n"
        "Use umap.utils.disconnected_vertices() to identify them.",
    ]


def test_one_row_fit_has_no_graph_but_fit_transform_matches_public_failure():
    data = np.asarray([[0.0, 0.0]], dtype=np.float32)
    model = graph_model()

    assert model.fit(data) is model
    assert not hasattr(model, "graph_")
    with pytest.raises(AttributeError, match="'UMAP' object has no attribute 'graph_'"):
        graph_model().fit_transform(data)


def test_validation_matches_public_graph_contract():
    with pytest.raises(ValueError, match="n_neighbors must be greater than 1"):
        graph_model(n_neighbors=1).fit_transform(BASELINE)
    with pytest.raises(ValueError, match="metric is neither callable nor a recognised string"):
        graph_model(metric="definitely-not-a-metric").fit_transform(BASELINE)
    with pytest.raises(ValueError, match="Input contains NaN"):
        graph_model().fit_transform(
            np.asarray(
                [[0.0, 0.0], [1.0, 0.1], [0.2, 1.3], [np.nan, 2.1]],
                dtype=np.float32,
            )
        )


def test_unknown_mode_fails_explicitly():
    model = UMAP(transform_mode="unknown")
    with pytest.raises(NotImplementedError, match="supported transform_mode"):
        model.fit(BASELINE)
    with pytest.raises(NotImplementedError, match="supported transform_mode"):
        model.transform(BASELINE)


def test_non_default_graph_mix_parameters_fail_explicitly():
    with pytest.raises(NotImplementedError, match="set_op_mix_ratio"):
        graph_model(set_op_mix_ratio=0.5).fit(BASELINE)
    with pytest.raises(NotImplementedError, match="local_connectivity"):
        graph_model(local_connectivity=2.0).fit(BASELINE)
