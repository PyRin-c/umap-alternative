import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import euclidean_distances

from umap_alternative import UMAP


PARITY_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "reference/fixtures/umap-learn-0.6.0-distance-matrix-parity-windows-x86_64.json"
    ).read_text(encoding="utf-8")
)
PRECOMPUTED_WARNING = "using precomputed metric; inverse_transform will be unavailable"


def precomputed_model(**kwargs):
    parameters = PARITY_FIXTURE["base_parameters"] | kwargs
    return UMAP(metric="precomputed", **parameters)


@pytest.mark.parametrize("case_name", PARITY_FIXTURE["cases"])
@pytest.mark.parametrize(
    "method", ["f32_pure", "f64_ordered", "sklearn_euclidean_distances"]
)
def test_precomputed_csr_bits_match_public_distance_fixture(case_name, method):
    case = PARITY_FIXTURE["cases"][case_name]
    points = np.asarray(case["input"]["values"], dtype=np.float32)
    if method == "sklearn_euclidean_distances":
        distances = euclidean_distances(points)
    else:
        if method == "f64_ordered":
            points = points.astype(np.float64)
        delta = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((delta * delta).sum(axis=2))
        if method == "f64_ordered":
            # Let the estimator perform the fixture's float32 conversion.
            distances = np.asfortranarray(distances)

    model = precomputed_model()
    with pytest.warns(UserWarning, match=PRECOMPUTED_WARNING):
        graph = model.fit_transform(distances)

    expected = case["distance_matrices"][method]["precomputed_graph"]
    assert isinstance(graph, csr_matrix)
    assert graph is model.graph_
    assert graph.dtype == np.dtype(np.float32)
    assert graph.shape == tuple(expected["shape"])
    np.testing.assert_array_equal(graph.indptr, expected["indptr"])
    np.testing.assert_array_equal(graph.indices, expected["indices"])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["data_bits_uint32"])


def test_precomputed_fit_uses_asymmetric_distances_in_row_order():
    # Both averaging and transposing these rows change the selected support.
    distances = [[0, 1, 9], [2, 0, 1], [3, 9, 0]]
    model = precomputed_model(n_neighbors=2)

    with pytest.warns(UserWarning, match=PRECOMPUTED_WARNING):
        returned = model.fit(distances)

    assert returned is model
    assert model.n_features_in_ == 3
    assert model.n_neighbors_ == 2
    assert model.graph_.shape == (3, 3)
    assert model.graph_.indptr.tolist() == [0, 2, 4, 6]
    assert model.graph_.indices.tolist() == [1, 2, 0, 2, 0, 1]
    assert model.graph_.data.tolist() == [1.0] * 6


def test_precomputed_two_rows_truncates_neighbors_and_warns():
    model = precomputed_model(n_neighbors=2)

    with pytest.warns(UserWarning) as caught:
        graph = model.fit_transform([[0, 1], [1, 0]])

    assert graph.shape == (2, 2)
    assert graph.nnz == 0
    assert model.n_neighbors_ == 1
    assert [str(item.message).splitlines()[0] for item in caught] == [
        PRECOMPUTED_WARNING,
        "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1",
        "A large number of your vertices were disconnected from the manifold.",
    ]


def test_precomputed_singleton_clears_previous_graph():
    model = precomputed_model(n_neighbors=2)
    with pytest.warns(UserWarning, match=PRECOMPUTED_WARNING):
        model.fit([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

    with pytest.warns(UserWarning, match=PRECOMPUTED_WARNING):
        assert model.fit([[0]]) is model

    assert model.n_features_in_ == 1
    assert not hasattr(model, "graph_")
    assert not hasattr(model, "n_neighbors_")
    with pytest.warns(UserWarning, match=PRECOMPUTED_WARNING):
        with pytest.raises(AttributeError, match="'UMAP' object has no attribute 'graph_'"):
            model.fit_transform([[0]])


@pytest.mark.filterwarnings(f"ignore:{PRECOMPUTED_WARNING}:UserWarning")
@pytest.mark.parametrize(
    ("distances", "error", "message"),
    [
        ([[0, 1]], ValueError, "must be a square matrix"),
        ([[0, 1], [1, 0], [1, 1]], ValueError, "must be a square matrix"),
        ([[0, np.nan], [1, 0]], ValueError, "Input contains NaN"),
        ([[0, np.inf], [1, 0]], ValueError, "Input contains infinity"),
        ([[0, -1, 3], [1, 0, 2], [3, 2, 0]], ValueError, "invalid distance -1"),
        ([[0, 1, 3], [1, 0.5, 2], [3, 2, 0]], ValueError, "self distance.*expected zero"),
        ([[0.5]], ValueError, "self distance.*expected zero"),
        (csr_matrix(np.eye(3)), ValueError, "Non-zero distances from samples to themselves"),
        (np.zeros((4, 4)), ValueError, "row 2 does not include itself"),
    ],
)
def test_precomputed_rejects_inputs_outside_the_supported_graph_path(distances, error, message):
    with pytest.raises(error, match=message):
        precomputed_model(n_neighbors=2).fit(distances)


def test_precomputed_requires_the_public_metric_name():
    with pytest.raises(ValueError, match="metric is neither callable nor a recognised string"):
        UMAP(metric="Precomputed", transform_mode="graph").fit([[0]])
