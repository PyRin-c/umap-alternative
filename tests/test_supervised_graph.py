import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from umap_alternative import UMAP


FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "fixtures"
        / "umap-learn-0.6.0-supervised-dense-graph-windows-x86_64.json"
    ).read_text(encoding="utf-8")
)
CASES = FIXTURE["cases"]
X = np.asarray(FIXTURE["input"]["values"], dtype=FIXTURE["input"]["dtype"])
PARTIAL_LABELS = np.asarray(CASES["semisupervised"]["y"]["values"], dtype=np.int64)


def graph_model(**kwargs):
    parameters = FIXTURE["base_parameters"].copy()
    # This estimator always uses exact neighbors.
    force_approximation = parameters.pop("force_approximation_algorithm")
    assert force_approximation is False
    return UMAP(**(parameters | kwargs))


def assert_csr_matches(graph, expected):
    assert isinstance(graph, csr_matrix)
    assert graph.format == expected["format"]
    assert graph.shape == tuple(expected["shape"])
    assert graph.nnz == expected["nnz"]
    assert graph.dtype.name == expected["dtype"]
    assert graph.indptr.dtype.name == expected["indptr_dtype"]
    assert graph.indices.dtype.name == expected["indices_dtype"]
    np.testing.assert_array_equal(graph.indptr, expected["indptr"])
    np.testing.assert_array_equal(graph.indices, expected["indices"])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["data_bits_uint32"])


@pytest.mark.parametrize("case_name", CASES)
def test_dense_supervised_graph_matches_public_csr_bits(case_name):
    case = CASES[case_name]
    labels = case["y"]
    y = None if labels is None else np.asarray(labels["values"], dtype=labels["dtype"])
    model = graph_model(target_weight=case["target_weight"])

    graph = model.fit_transform(X, y=y)

    assert graph is model.graph_
    assert_csr_matches(graph, case["expected"])


def test_target_defaults_clone_and_unsupervised_refit():
    model = clone(UMAP(transform_mode="graph", n_neighbors=4))
    assert model.target_metric == "categorical"
    assert model.target_weight == 0.5
    assert model.fit(X, PARTIAL_LABELS) is model
    assert_csr_matches(model.graph_, CASES["semisupervised"]["expected"])

    model.set_params(target_metric="euclidean", target_weight=np.nan)
    assert_csr_matches(model.fit_transform(X), CASES["unsupervised"]["expected"])


@pytest.mark.parametrize(
    "labels",
    [
        PARTIAL_LABELS.astype(np.int8),
        PARTIAL_LABELS.astype(np.int32)[:, None],
        PARTIAL_LABELS.tolist(),
        np.array([0.1, 0.2, 0.1, -1, -1, 0.2, 0.1, 0.2]),
        np.where(PARTIAL_LABELS == 10, np.iinfo(np.int64).min, PARTIAL_LABELS),
    ],
    ids=["int8", "column", "list", "fractional", "negative-class"],
)
def test_numeric_labels_preserve_class_equality_and_the_unlabeled_sentinel(labels):
    graph = graph_model().fit_transform(X, y=labels)

    assert_csr_matches(graph, CASES["semisupervised"]["expected"])


def test_uint64_classes_keep_their_identity_at_the_top_of_the_range():
    low = 2**64 - 2
    high = 2**64 - 1
    labels = np.array([low, high] * 4, dtype=np.uint64)

    graph = graph_model().fit_transform(X, y=labels)

    assert_csr_matches(graph, CASES["categorical"]["expected"])


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        (PARTIAL_LABELS[:-1], ValueError, "Length of x = 8, length of y = 7"),
        (np.zeros(9), ValueError, "Length of x = 8, length of y = 9"),
        (PARTIAL_LABELS[None, :], ValueError, "Length of x = 8, length of y = 1"),
        (np.tile(PARTIAL_LABELS[:, None], (1, 2)), ValueError, "one label per row"),
        (10, TypeError, "has no len"),
        (np.full(8, "class"), ValueError, "not compatible with arrays of bytes/strings"),
        (np.full(8, 1j), ValueError, "Complex data not supported"),
        (np.full(8, np.nan), ValueError, "Input contains NaN"),
        (np.full(8, np.inf), ValueError, "Input contains infinity"),
        (csr_matrix(PARTIAL_LABELS[:, None]), TypeError, "sparse array length is ambiguous"),
    ],
)
def test_categorical_target_validation(labels, error, message):
    with pytest.raises(error, match=message):
        graph_model().fit(X, y=labels)


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_categorical_target_weight_requires_the_validated_core_range(weight):
    with pytest.raises(ValueError, match=r"target weight .* must be in 0\.\.=1"):
        graph_model(target_weight=weight).fit(X, y=PARTIAL_LABELS)


def test_categorical_target_weight_rejects_text():
    with pytest.raises(TypeError):
        graph_model(target_weight="0.5").fit(X, y=PARTIAL_LABELS)


def test_categorical_singleton_clears_a_previous_graph():
    model = graph_model().fit(X, y=PARTIAL_LABELS)

    assert model.fit(X[:1], y=PARTIAL_LABELS) is model
    assert not hasattr(model, "graph_")
    assert not hasattr(model, "n_neighbors_")
    with pytest.raises(AttributeError, match="'UMAP' object has no attribute 'graph_'"):
        model.fit_transform(X[:1], y=PARTIAL_LABELS[:1])


def test_supervised_fit_rejects_other_target_metrics_and_unsupported_feature_inputs():
    with pytest.raises(NotImplementedError, match="target_metric='categorical'"):
        graph_model(target_metric="euclidean").fit(X, y=PARTIAL_LABELS)
    with pytest.raises(TypeError, match="sparse input is not supported"):
        graph_model().fit(csr_matrix(X), y=PARTIAL_LABELS)
    with pytest.raises(NotImplementedError, match="metric='precomputed'"):
        graph_model(metric="precomputed").fit(np.eye(len(X)), y=PARTIAL_LABELS)
