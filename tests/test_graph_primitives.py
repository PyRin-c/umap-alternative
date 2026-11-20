import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from umap_alternative import UMAP, umap_, utils


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = json.loads(
    (_ROOT / "reference/fixtures/umap-learn-0.6.0-graph-primitives-x86_64.json")
    .read_text(encoding="utf-8")
)


def _decode(data):
    if isinstance(data, list):
        return tuple(_decode(element) for element in data)
    dtype = np.dtype(data["dtype"])
    wire_dtype = f"uint{dtype.itemsize * 8}" if dtype.kind == "f" else dtype
    return np.asarray(data["contents"], dtype=wire_dtype).view(dtype).reshape(data["shape"])


def _assert_result(value, expected):
    if expected is None:
        assert value is None
    elif isinstance(expected, list):
        assert len(value) == len(expected)
        for element, item in zip(value, expected):
            _assert_result(element, item)
    elif "format" in expected:
        assert value.format == expected["format"]
        assert value.shape == tuple(expected["shape"])
        graph = value.tocoo()
        np.testing.assert_array_equal(graph.row, expected["rows"])
        np.testing.assert_array_equal(graph.col, expected["columns"])
        _assert_result(graph.data, expected["data"])
    else:
        wanted = _decode(expected)
        assert value.shape == wanted.shape and value.dtype == wanted.dtype
        if wanted.dtype.kind == "f":
            unsigned = f"uint{wanted.dtype.itemsize * 8}"
            np.testing.assert_array_equal(value.view(unsigned), wanted.view(unsigned))
        else:
            np.testing.assert_array_equal(value, wanted)


def _assert_public_call(case, invoke):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = invoke()
    assert [notice.category.__name__ for notice in seen] == case["warnings"]
    _assert_result(result, case["result"])


@pytest.mark.parametrize("case", _FIXTURE["smooth"], ids=lambda case: case["name"])
def test_public_smooth_knn(case):
    _assert_public_call(
        case, lambda: umap_.smooth_knn_dist(_decode(case["distances"]), **case["options"])
    )


@pytest.mark.parametrize("case", _FIXTURE["memberships"], ids=lambda case: case["name"])
def test_public_membership_strengths(case):
    _assert_public_call(
        case, lambda: umap_.compute_membership_strengths(*_decode(case["input"]), **case["options"])
    )


@pytest.mark.parametrize("case", _FIXTURE["graphs"], ids=lambda case: case["name"])
def test_public_fuzzy_outputs(case):
    indices, distances = _decode(case["input"])
    _assert_public_call(
        case, lambda: umap_.fuzzy_simplicial_set(
            np.zeros((len(indices), 1), np.float32), case["n_neighbors"], np.random.RandomState(123),
            "euclidean", knn_indices=indices, knn_dists=distances, **case["options"]
        )
    )


@pytest.mark.parametrize("case", _FIXTURE["utilities"], ids=lambda case: case["name"])
def test_public_neighbor_utilities(case):
    function = case["function"]
    values = _decode(case["input"])
    if function == "nearest_neighbors":
        _assert_public_call(
            case, lambda: umap_.nearest_neighbors(
                values, case["n_neighbors"], "precomputed", {}, False, np.random.RandomState(34)
            )
        )
    elif function == "fast_knn_indices":
        _assert_public_call(case, lambda: utils.fast_knn_indices(values, case["n_neighbors"]))
    elif function == "submatrix":
        _assert_public_call(case, lambda: utils.submatrix(*values, 2))
    elif function == "average_nn_distance":
        _assert_public_call(case, lambda: utils.average_nn_distance(values))
    else:
        _assert_public_call(case, lambda: umap_.make_epochs_per_sample(values, 9))


def test_graph_options_through_public_helper():
    fixture = json.loads(
        (_ROOT / "crates/umap-core/tests/fixtures/graph_options.json").read_text(encoding="utf-8")
    )
    for dataset in fixture["datasets"]:
        indices = np.asarray(dataset["neighbors"], dtype=np.int32)
        distances = np.asarray(dataset["distance_bits"], dtype=np.uint32).view(np.float32)
        features = np.zeros((indices.shape[0], 1), dtype=np.float32)
        for profile in dataset["profiles"]:
            for expected in profile["graphs"]:
                graph, sigmas, rhos = umap_.fuzzy_simplicial_set(
                    features, indices.shape[1], np.random.RandomState(42), "euclidean",
                    knn_indices=indices, knn_dists=distances, local_connectivity=profile["connectivity"],
                    set_op_mix_ratio=expected["mix"],
                )
                np.testing.assert_array_equal(sigmas.view(np.uint32), profile["sigma_bits"])
                np.testing.assert_array_equal(rhos.view(np.uint32), profile["rho_bits"])
                np.testing.assert_array_equal(graph.indptr, expected["indptr"])
                np.testing.assert_array_equal(graph.indices, expected["indices"])
                weights = expected["data_bits"]
                if sys.platform == "linux":
                    weights = expected.get("linux_x86_64_data_bits", weights)
                np.testing.assert_array_equal(graph.data.view(np.uint32), weights)


def test_fuzzy_graph_uses_precomputed_distance_ranks():
    matrix = np.asarray([[0, .7, 2], [.7, 0, 1.2], [2, 1.2, 0]], np.float32)
    computed = umap_.fuzzy_simplicial_set(matrix, 2, np.random.RandomState(42), "precomputed")
    indices = np.asarray([[0, 1], [1, 0], [2, 1]], np.int32)
    distances = np.asarray([[0, .7], [0, .7], [0, 1.2]], np.float32)
    supplied = umap_.fuzzy_simplicial_set(
        matrix, 2, np.random.RandomState(42), "precomputed",
        knn_indices=indices, knn_dists=distances,
    )
    np.testing.assert_array_equal(computed[0].toarray(), supplied[0].toarray())
    np.testing.assert_array_equal(computed[1:], supplied[1:])


def test_membership_rejects_incompatible_types_and_shapes():
    indices = np.zeros((2, 3), dtype=np.int32)
    distances = np.zeros((2, 3), dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)
    with pytest.raises(TypeError, match="float32"):
        umap_.compute_membership_strengths(indices, distances.astype(np.float64), scales, scales)
    with pytest.raises(ValueError, match="shape"):
        umap_.compute_membership_strengths(indices, distances, scales[:1], scales)


def test_disconnected_vertices_uses_weight_sum_and_validates_fit():
    model = UMAP(
        random_state=42, n_jobs=1, n_neighbors=2, compatibility_layout=True,
        init="random", n_epochs=0, transform_mode="graph",
    ).fit(np.arange(8, dtype=np.float32).reshape(-1, 1))
    model.graph_ = csr_matrix(([0, 1, -1, 1], ([0, 1, 1, 2], [1, 0, 2, 0])), shape=(8, 8))
    np.testing.assert_array_equal(
        utils.disconnected_vertices(model), [True, True, False, True, True, True, True, True]
    )
    with pytest.raises(TypeError):
        utils.disconnected_vertices(object())
    with pytest.raises(NotFittedError):
        utils.disconnected_vertices(UMAP())


@pytest.mark.parametrize("convert", [np.asarray, csr_matrix])
def test_disconnected_vertices_restores_unique_rows(convert):
    x = convert(np.array([[0], [0], [.5], [.75], [9], [9]], np.float32))
    with pytest.warns(UserWarning, match="disconnected"):
        model = UMAP(
            unique=True, n_neighbors=2, n_jobs=1, random_state=42, init="random",
            transform_mode="graph", disconnection_distance=1.0, n_epochs=0, compatibility_layout=True,
        ).fit(x)
    assert model.graph_.shape == (4, 4)
    np.testing.assert_array_equal(utils.disconnected_vertices(model), [False, False, False, False, True, True])


def test_disconnected_vertices_keeps_unique_fit_map_after_update():
    features = np.asarray([[7, 3], [1, 0], [7, 3], [0, 0], [2, 4], [1, 0], [4, 3], [0, 0],
                           [3, 7], [3, 5]], np.float32)
    model = UMAP(
        unique=True, n_neighbors=3, n_jobs=1, random_state=42, init="random",
        a=1.2, b=.8, n_epochs=0, compatibility_layout=True,
    ).fit(features)
    model.update(np.asarray([[1.5, 2], [0, 0], [7, 3]], np.float32))
    assert model.graph_.shape == (13, 13)
    assert utils.disconnected_vertices(model).shape == (10,)
