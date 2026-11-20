import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def load_fixture(name, platform="windows"):
    return json.loads(
        (FIXTURES / f"umap-learn-0.6.0-{name}-{platform}-x86_64.json").read_text(encoding="utf-8")
    )


PYTHON_FIXTURE = load_fixture(
    "supplied-neighbors-python", "linux" if sys.platform == "linux" else "windows"
)
GRAPH_CASES = [
    pytest.param(fixture, case, id=name)
    for fixture in (
        load_fixture("precomputed-neighbors"), load_fixture("supplied-neighbors-held-out"),
        PYTHON_FIXTURE,
    )
    for name, case in fixture["cases"].items()
]
BASE = PYTHON_FIXTURE["base_parameters"]
TRAINING = np.asarray(PYTHON_FIXTURE["training"], dtype=np.float32)
HELD_OUT = np.asarray(PYTHON_FIXTURE["held_out"], dtype=np.float32)


def array_pair():
    return (
        np.asarray(PYTHON_FIXTURE["layout_indices"], dtype=np.int32),
        np.asarray(PYTHON_FIXTURE["layout_distance_bits"], dtype=np.uint32).view(np.float32),
    )


def assert_graph(graph, expected):
    assert graph.format == expected["format"]
    assert graph.shape == tuple(expected["shape"])
    assert graph.dtype == np.float32
    for name in ("indptr", "indices") if graph.format == "csr" else ("row", "col"):
        np.testing.assert_array_equal(getattr(graph, name), expected[name])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["weight_bits"])


def assert_warnings(captured, expected):
    assert [
        {"type": item.category.__name__, "message": str(item.message)} for item in captured
    ] == expected


@pytest.mark.parametrize(("fixture", "case"), GRAPH_CASES)
def test_supplied_graph_fixture(fixture, case):
    training = np.asarray(fixture["training"], dtype=np.float32)
    queries = np.asarray(fixture["held_out"], dtype=np.float32)
    parameters = fixture["base_parameters"] | case["overrides"]
    if "indices" in case:
        indices = np.asarray(case["indices"], dtype=np.int32)
        distances = np.asarray(case["distance_bits"], dtype=np.uint32).view(np.float32)
        parameters["precomputed_knn"] = (indices, distances, None)
    model = UMAP(**parameters)
    expected_fit = case["fit_transform"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected_fit:
            with pytest.raises(ValueError) as error:
                model.fit_transform(training)
            assert str(error.value) == expected_fit["error"]["message"]
            assert not hasattr(model, "graph_")
        else:
            graph = model.fit_transform(training)
            assert graph is model.graph_
            assert_graph(graph, expected_fit["graph"])
            assert model.n_neighbors_ == min(parameters["n_neighbors"], len(training) - 1)
            assert model.transform(training.astype(np.float64)) is graph
    assert_warnings(caught, expected_fit["warnings"])
    if "indices" in case:
        np.testing.assert_array_equal(indices, case.get("indices_after_fit", case["indices"]))
        np.testing.assert_array_equal(
            distances.view(np.uint32), case.get("distance_bits_after_fit", case["distance_bits"])
        )
    if "transform_held_out" in case:
        expected_transform = case["transform_held_out"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in expected_transform:
                with pytest.raises(NotImplementedError) as error:
                    model.transform(queries)
                assert str(error.value) == expected_transform["error"]["message"]
            else:
                assert_graph(model.transform(queries), expected_transform["graph"])
        assert_warnings(caught, expected_transform["warnings"])


@pytest.mark.parametrize("name", PYTHON_FIXTURE["layouts"])
def test_supplied_layout_uses_fitted_graph(name):
    case = PYTHON_FIXTURE["layouts"][name]
    init = case["init"]
    if not isinstance(init, str):
        init = np.asarray(init, dtype=np.float32)
    model = UMAP(**(BASE | {
        "init": init, "n_epochs": case["n_epochs"], "transform_mode": "embedding",
        "disconnection_distance": case["disconnection_distance"], "precomputed_knn": array_pair(),
    }))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coordinates = model.fit_transform(TRAINING)
    assert_warnings(caught, case["warnings"])
    assert_graph(model.graph_, case["graph"])
    assert coordinates is model.embedding_
    assert model.transform(TRAINING.copy()) is coordinates
    with pytest.raises(NotImplementedError, match="No search index available"):
        model.transform(HELD_OUT)
    expected = np.asarray(case["coordinates_bits"], dtype=np.uint32).view(np.float32)
    np.testing.assert_allclose(coordinates, expected, rtol=0, atol=2e-4, equal_nan=True)


@pytest.mark.parametrize(
    ("neighbors", "error", "message", "warning"),
    [
        (None, TypeError, "not subscriptable", None),
        ((), IndexError, "tuple index out of range", None),
        ((None,), IndexError, "tuple index out of range", None),
        ((None, [[0]]), ValueError, r"precomputed_knn\[0\] must be ndarray", None),
        ((np.empty((8, 4)), [[0]]), ValueError, r"precomputed_knn\[1\] must be ndarray", None),
        ((np.empty(8), np.empty(8)), IndexError, "tuple index out of range", "NNDescent"),
        ((np.empty((8, 4, 1)), np.empty((8, 4, 1))), ValueError, "two-dimensional", "NNDescent"),
    ],
)
def test_supplied_tuple_validation(neighbors, error, message, warning):
    model = UMAP(**BASE, precomputed_knn=neighbors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error, match=message):
            model.fit(TRAINING)
    if warning is None:
        assert not caught
    else:
        assert len(caught) == 1
        assert warning in str(caught[0].message)
    assert not hasattr(model, "graph_")


@pytest.mark.parametrize("as_list", [False, True])
def test_two_supplied_arrays_and_exact_input_transform(as_list):
    pair = array_pair()
    parameters = list(pair) if as_list else pair
    model = clone(UMAP(**BASE, precomputed_knn=parameters))
    with pytest.warns(UserWarning, match="NNDescent"):
        graph = model.fit_transform(TRAINING)
    assert model.transform(TRAINING.copy()) is graph
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.transform([[np.nan]])
    with pytest.raises(NotImplementedError, match="No search index available"):
        model.transform(np.zeros((2, 2), dtype=np.float32))
    model.set_params(precomputed_knn=(pair[0], None, None)).fit(TRAINING)
    assert model.transform(HELD_OUT).shape == (3, 8)


@pytest.mark.parametrize("cutoff", ["1", np.float32(1), np.int64(1)])
def test_invalid_cutoff_is_rejected_before_neighbor_warnings_or_mutation(cutoff):
    indices, distances = array_pair()
    initial_indices, initial_distances = array_pair()
    model = UMAP(**BASE, precomputed_knn=(indices, distances), disconnection_distance=cutoff)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="disconnection_distance must either be None or a numeric"):
            model.fit(TRAINING)
    assert not caught
    np.testing.assert_array_equal(indices, initial_indices)
    np.testing.assert_array_equal(distances, initial_distances)


@pytest.mark.parametrize("cutoff", [None, 1.0])
@pytest.mark.parametrize("read_only", [0, 1])
def test_cutoff_writes_before_readonly_error_and_preserves_previous_fit(cutoff, read_only):
    model = UMAP(**BASE).fit(TRAINING)
    graph = model.graph_
    pair = array_pair()
    pair[read_only].setflags(write=False)
    model.set_params(precomputed_knn=pair, disconnection_distance=cutoff)
    with pytest.warns(UserWarning, match="NNDescent"):
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            model.fit(TRAINING)
    indices, distances = array_pair()
    if read_only == 1 and cutoff is not None:
        indices[distances >= cutoff] = -1
    np.testing.assert_array_equal(pair[0], indices)
    np.testing.assert_array_equal(pair[1], distances)
    assert model.graph_ is graph
    assert model.transform(TRAINING.copy()) is graph
    with pytest.raises(NotImplementedError, match="No search index available"):
        model.transform(HELD_OUT)


def test_bad_supplied_shapes_leave_previous_graph_and_update_held_out_training():
    model = UMAP(**BASE).fit(TRAINING)
    graph = model.graph_
    shifted = TRAINING + np.float32(50)
    shifted_model = UMAP(**BASE).fit(shifted)
    indices, distances = array_pair()
    model.set_params(precomputed_knn=(indices, distances[:, :-1]))
    with pytest.raises(ValueError, match="numpy arrays of the same size"):
        model.fit(shifted)

    assert model.graph_ is graph
    assert model.transform(TRAINING.copy()) is graph
    held_out = model.transform(HELD_OUT)
    expected = shifted_model.transform(HELD_OUT)
    np.testing.assert_array_equal(held_out.row, expected.row)
    np.testing.assert_array_equal(held_out.col, expected.col)
    np.testing.assert_array_equal(held_out.data.view(np.uint32), expected.data.view(np.uint32))


def test_embedding_error_retains_supplied_graph_and_previous_embedding():
    model = UMAP(**(BASE | {"transform_mode": "embedding"})).fit(TRAINING)
    old_graph, old_embedding = model.graph_, model.embedding_
    indices, distances = array_pair()
    model.set_params(precomputed_knn=(indices, distances), disconnection_distance=0.0)
    with pytest.warns(UserWarning) as caught:
        with pytest.raises(ValueError, match="zero-size array"):
            model.fit(TRAINING)
    assert [str(item.message).splitlines()[0] for item in caught] == [
        "precomputed_knn[2] (knn_search_index) is not an NNDescent object: "
        "transforming new data with transform will be unavailable.",
        "A large number of your vertices were disconnected from the manifold.",
    ]
    np.testing.assert_array_equal(indices, -np.ones_like(indices))
    assert np.all(np.isposinf(distances))
    assert model.graph_ is not old_graph
    assert model.graph_.nnz == 0
    assert model.embedding_ is old_embedding
    assert model.transform(TRAINING.copy()) is old_embedding
    with pytest.raises(NotImplementedError, match="No search index available"):
        model.transform(HELD_OUT)


def test_cutoff_masks_out_of_range_index_before_native_graph_validation():
    indices, distances = array_pair()
    indices[0, 3] = len(TRAINING)
    model = UMAP(**BASE, precomputed_knn=(indices, distances))
    with pytest.warns(UserWarning, match="NNDescent"):
        with pytest.raises(ValueError, match="row 0, column 3 has invalid pair"):
            model.fit(TRAINING)
    assert not hasattr(model, "graph_")

    model.set_params(disconnection_distance=1.0)
    with pytest.warns(UserWarning) as caught:
        graph = model.fit_transform(TRAINING)
    assert len(caught) == 2
    assert indices[0, 3] == -1
    assert distances[0, 3] == np.inf
    assert graph.nnz == 6


def test_supplied_graph_uses_arrays_with_a_precomputed_feature_metric():
    features = np.abs(TRAINING - TRAINING.T)
    model = UMAP(**BASE, metric="precomputed", precomputed_knn=array_pair())
    with pytest.warns(UserWarning) as caught:
        graph = model.fit_transform(features)
    assert len(caught) == 2
    expected = load_fixture("precomputed-neighbors")["cases"]["override_distances"]
    assert_graph(graph, expected["fit_transform"]["graph"])
    assert model.transform(features.copy()) is graph


def test_singleton_validates_supplied_shape_without_masking():
    indices = np.zeros((1, 4), dtype=np.int32)
    distances = np.zeros((1, 4), dtype=np.float32)
    model = UMAP(**BASE, precomputed_knn=(indices, distances), disconnection_distance=0)
    with pytest.warns(UserWarning, match="NNDescent") as caught:
        assert model.fit(TRAINING[:1]) is model
    assert len(caught) == 1
    assert not hasattr(model, "graph_")
    np.testing.assert_array_equal(indices, np.zeros_like(indices))
    np.testing.assert_array_equal(distances, np.zeros_like(distances))
    with pytest.raises(ValueError, match="only a single data sample"):
        model.transform(TRAINING[:1])
