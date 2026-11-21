import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

import umap_alternative
from umap_alternative import AlignedUMAP, ParametricUMAP, UMAP, distances, umap_


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "umap-learn-0.6.0-configured-metrics-x86_64.json"
CASES = json.loads(FIXTURE.read_bytes())
TRAINING = np.asarray(CASES["training"], dtype=np.float32)
QUERIES = np.asarray(CASES["queries"], dtype=np.float32)
OPTIONS = CASES["options"]
BASE = {name: value for name, value in OPTIONS.items() if name not in ("metric", "metric_kwds")}
WEIGHTS = np.array([2, 3], dtype=np.float32)

pytestmark = [
    pytest.mark.filterwarnings("ignore:gradient function is not yet implemented"),
    pytest.mark.filterwarnings("ignore:n_jobs value -1 overridden"),
]


def assert_sparse_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.format == expected.format
    for axis in ("indices", "indptr") if actual.format == "csr" else ("row", "col"):
        np.testing.assert_array_equal(getattr(actual, axis), getattr(expected, axis))
    np.testing.assert_array_equal(actual.data.view(np.uint32), expected.data.view(np.uint32))


def assert_public_graph(actual, expected):
    assert actual.shape == tuple(expected["shape"])
    assert actual.format == expected["format"]
    for axis in ("indices", "indptr") if actual.format == "csr" else ("row", "col"):
        np.testing.assert_array_equal(getattr(actual, axis), expected[axis])
    np.testing.assert_array_equal(actual.data.view(np.uint32), expected["bits"])


def weighted_l1_with_gradient(x, y, weights):
    difference = x - y
    return np.abs(difference) @ weights, np.sign(difference) * weights


@pytest.mark.parametrize("case", CASES["distances"], ids=lambda case: case["id"])
def test_public_configured_distance(case):
    expected = np.uint64(int(case["bits"], 16)).view(np.float64)
    x, y = (np.asarray(case[side], dtype=case["dtype"]) for side in ("left", "right"))
    actual = distances.named_distances[case["metric"]](x, y, **case["kwds"])
    if expected == 0.0:
        assert actual == 0.0
    else:
        assert actual == pytest.approx(expected, rel=1e-6, abs=0.0)


@pytest.mark.parametrize(("metric", "x", "y", "kwds", "message"), [
    ("minkowski", [1, 2], [2, 3], {"p": 0}, "positive"),
    ("minkowski", [1, 2], [2, 3], {"p": np.inf}, "finite"),
    ("minkowski", [1, np.nan], [2, 3], {}, "finite"),
    ("weighted_minkowski", [1, 2], [2, 3], {"w": [1]}, "shape"),
    ("weighted_minkowski", [1, 2], [2, 3], {"w": [1, -1]}, "nonnegative"),
    ("standardised_euclidean", [1, 2], [2, 3], {"sigma": [1, 0]}, "positive"),
    ("mahalanobis", [1, 2], [2, 3], {"vinv": [1, 2, 3, 4]}, "shape"),
    ("haversine", [1], [2], {}, "latitude"),
    ("poincare", [1, 0], [0, 0], {}, "undefined"),
    ("hellinger", [-1, 1], [0, 1], {}, "undefined"),
    ("symmetric_kl", [0, 1], [1, 0], {"z": 0}, "positive"),
])
def test_configured_distance_invalid_domain(metric, x, y, kwds, message):
    with pytest.raises(ValueError, match=message):
        distances.named_distances[metric](x, y, **kwds)


def test_configured_estimator_against_public_graph_and_layout():
    graph_model = UMAP(**OPTIONS, transform_mode="graph")
    assert_public_graph(graph_model.fit_transform(TRAINING), CASES["estimator"]["graph"]["fit_graph"])
    assert_public_graph(graph_model.transform(QUERIES), CASES["estimator"]["graph"]["transform"])

    expected = CASES["estimator"]["embedding"]
    model = UMAP(**OPTIONS)
    embedding = model.fit_transform(TRAINING)
    assert_public_graph(model.graph_, expected["fit_graph"])
    np.testing.assert_array_equal(embedding.view(np.uint32), expected["embedding"])
    np.testing.assert_array_equal(model.transform(QUERIES).view(np.uint32), expected["transform"])
    assert model.update(QUERIES) is None
    assert_public_graph(model.graph_, expected["updated_graph"])
    np.testing.assert_array_equal(model.embedding_.view(np.uint32), expected["updated_embedding"])


@pytest.mark.parametrize("use_callback", [False, True], ids=["weighted", "callback"])
@pytest.mark.parametrize("mode", ["graph", "embedding"])
def test_configured_estimator_routes_feature_ranks(use_callback, mode):
    weights = WEIGHTS.copy()
    if use_callback:
        weights.setflags(write=False)
        metric, kwargs = weighted_l1_with_gradient, {"weights": weights}
    else:
        metric, kwargs = "wminkowski", {"p": 1.0, "w": weights}

    model = UMAP(**BASE, metric=metric, metric_kwds=kwargs, transform_mode=mode)
    expected = UMAP(**BASE, metric="manhattan", transform_mode=mode).fit(TRAINING * WEIGHTS)
    actual = model.fit_transform(TRAINING)
    assert_sparse_equal(model.graph_, expected.graph_)
    if not use_callback:
        weights[:] = 0
    if mode == "graph":
        assert actual is model.graph_
        assert_sparse_equal(model.transform(QUERIES), expected.transform(QUERIES * WEIGHTS))
        assert model.update(QUERIES) is model
        whole = np.concatenate((TRAINING, QUERIES)) * WEIGHTS
        expected.fit(whole)
        assert_sparse_equal(model.graph_, expected.graph_)
    else:
        np.testing.assert_array_equal(actual, expected.embedding_)
        assert model.transform(TRAINING) is model.embedding_
        np.testing.assert_array_equal(model.transform(QUERIES), expected.transform(QUERIES * WEIGHTS))
        assert model.update(QUERIES) is None
        expected.update(QUERIES * WEIGHTS)
        assert_sparse_equal(model.graph_, expected.graph_)
        np.testing.assert_array_equal(model.embedding_, expected.embedding_)


def test_configured_graph_reuses_categorical_target_processing():
    y = np.array([0, 0, 1, 0, 1, -1, 0, 1, 1])
    model = UMAP(**OPTIONS, transform_mode="graph", target_weight=1.0).fit(TRAINING, y)
    expected = UMAP(**BASE, transform_mode="graph", metric="manhattan", target_weight=1.0).fit(TRAINING, y)
    assert_sparse_equal(model.graph_, expected.graph_)
    with pytest.raises(ValueError, match="supervised"):
        model.update(QUERIES)


@pytest.mark.parametrize("sparse", [False, True])
def test_low_level_neighbors_and_fuzzy_graph_from_feature_rows(sparse):
    data = csr_matrix(TRAINING) if sparse else TRAINING
    metric = "euclidean" if sparse else "minkowski"
    kwargs = {} if sparse else {"p": 1.0}
    matrix_metric = "euclidean" if sparse else "cityblock"
    k = 4
    indices, ranks, index = umap_.nearest_neighbors(data, k, metric, kwargs, False, 42)
    expected = cdist(TRAINING, TRAINING, metric=matrix_metric)
    order = np.argsort(expected, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(indices, order)
    np.testing.assert_allclose(ranks, np.take_along_axis(expected, order, axis=1), rtol=2e-7)

    query_matrix = cdist(QUERIES, TRAINING, metric=matrix_metric)
    order = np.argsort(query_matrix.astype(np.float32), axis=1)[:, :k]
    query_indices, query_ranks = index.query(QUERIES, k)
    np.testing.assert_array_equal(query_indices, order)
    np.testing.assert_allclose(query_ranks, np.take_along_axis(query_matrix, order, axis=1), rtol=2e-7)

    graph, sigmas, rhos = umap_.fuzzy_simplicial_set(data, k, 42, metric, kwargs)
    supplied, expected_sigmas, expected_rhos = umap_.fuzzy_simplicial_set(
        data, k, 42, metric, kwargs, knn_indices=indices, knn_dists=ranks,
    )
    assert_sparse_equal(graph, supplied)
    np.testing.assert_array_equal(sigmas, expected_sigmas)
    np.testing.assert_array_equal(rhos, expected_rhos)
    _, _, graph_only_index = umap_.nearest_neighbors(data, k, metric, kwargs, False, 42, graph_only=True)
    assert graph_only_index is None


def test_supplied_exact_index_drives_the_embedding_and_survives_pickle():
    k = BASE["n_neighbors"]
    indices, ranks, search = umap_.nearest_neighbors(TRAINING, k, "minkowski", {"p": 1.0}, False, 19)
    model = UMAP(**BASE, precomputed_knn=(indices, ranks, search)).fit(TRAINING)
    expected = UMAP(**OPTIONS).fit(TRAINING)
    assert_sparse_equal(model.graph_, expected.graph_)
    np.testing.assert_array_equal(model.transform(QUERIES), expected.transform(QUERIES))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).transform(QUERIES), model.transform(QUERIES))


@pytest.mark.parametrize("bad_result", [False, True])
def test_external_index_validation_and_cutoff(monkeypatch, bad_result):
    class SearchIndex:
        def query(self, X, k):
            np.testing.assert_array_equal(X, QUERIES)
            assert k == 4
            columns = [2, 0, 3, len(TRAINING) if bad_result else 5]
            return np.tile(np.array(columns, dtype=np.uint32), (len(X), 1)), np.tile(
                np.array([.25, 1.0, 5.0, np.inf], dtype=np.float32), (len(X), 1)
            )

    monkeypatch.setitem(sys.modules, "pynndescent", SimpleNamespace(NNDescent=SearchIndex))
    columns, ranks, _ = umap_.nearest_neighbors(TRAINING, 4, "euclidean", {}, False, 19)
    model = UMAP(
        **BASE, transform_mode="graph", disconnection_distance=5.0,
        precomputed_knn=(columns, ranks, SearchIndex()),
    ).fit(TRAINING)
    with pytest.raises(ValueError, match="features"):
        model.transform(QUERIES[:, :1])
    if bad_result:
        with pytest.raises(ValueError, match="invalid neighbor ranks"):
            model.transform(QUERIES)
    else:
        graph = model.transform(QUERIES)
        positive = graph.data > 0
        assert sorted(zip(graph.row[positive].tolist(), graph.col[positive].tolist())) == [
            (0, 0), (0, 2), (1, 0), (1, 2),
        ]


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation"])
@pytest.mark.parametrize("mode", ["graph", "embedding"])
def test_forced_native_search_queries_and_pickle(metric, mode):
    rng = np.random.default_rng(157)
    training = rng.normal(size=(83, 5)).astype(np.float32)
    queries = rng.normal(size=(4, 5)).astype(np.float32)
    model = UMAP(**BASE, metric=metric, transform_mode=mode, force_approximation_algorithm=True).fit(training)
    assert model.graph_.nnz > 0
    index = model._feature_search_index
    columns, ranks = index.query(queries, model.n_neighbors_)
    assert (columns >= 0).all()
    expected_distances = np.take_along_axis(cdist(queries, training, metric=metric), columns, axis=1)
    np.testing.assert_allclose(ranks, expected_distances, rtol=2e-6, atol=1e-7)
    assert np.all(np.diff(ranks, axis=1) >= 0)
    first = model.transform(queries)
    restored = pickle.loads(pickle.dumps(model))
    second = restored.transform(queries)
    if mode == "graph":
        assert first.shape == (len(queries), len(training))
        assert first.nnz > 0
        assert_sparse_equal(first, second)
    else:
        assert first.shape == (len(queries), 2)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


def test_forced_native_search_rejects_unsupported_sparse_data():
    with pytest.raises(NotImplementedError, match="requires dense"):
        UMAP(**BASE, force_approximation_algorithm=True).fit(csr_matrix(TRAINING))


@pytest.mark.parametrize("init", ["random", "spectral"])
def test_randomstate_and_global_rng_follow_the_same_initialization_path(init):
    opts = OPTIONS | {"init": init}
    manual = np.random.RandomState(19)
    expected = UMAP(**(opts | {"random_state": manual})).fit_transform(TRAINING)
    assert manual.get_state()[2] != np.random.RandomState(19).get_state()[2]
    saved = np.random.get_state()
    try:
        np.random.seed(19)
        actual = UMAP(**(opts | {"random_state": None})).fit_transform(TRAINING)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.random.get_state()[1], manual.get_state()[1])
        assert np.random.get_state()[2:] == manual.get_state()[2:]
    finally:
        np.random.set_state(saved)


def test_singleton_fit_leaves_supplied_rng_unadvanced():
    manual = np.random.RandomState(19)
    state = manual.get_state()
    model = UMAP(**(OPTIONS | {"random_state": manual}))
    np.testing.assert_array_equal(model.fit_transform(TRAINING[:1]), [[0, 0]])
    np.testing.assert_array_equal(manual.get_state()[1], state[1])
    assert manual.get_state()[2:] == state[2:]


@pytest.mark.parametrize("force", [False, True])
def test_aligned_graph_uses_configured_search(force):
    data = [TRAINING, TRAINING * np.array([.5, 2.0], dtype=np.float32)]
    relation = {i: i for i in range(len(TRAINING))}
    search = {"metric": "euclidean", "force_approximation_algorithm": True} if force else {
        "metric": "minkowski", "metric_kwds": {"p": 1.0},
    }
    model = AlignedUMAP(
        n_neighbors=4, n_epochs=0, compatibility_layout=True, init="random",
        random_state=19, **search,
    ).fit(data, relations=[relation])
    for mapper, features in zip(model.mappers_, data, strict=True):
        expected = UMAP(**BASE, transform_mode="graph", **search).set_params(n_epochs=0).fit(features)
        weights = expected.graph_.data
        weights[weights < weights.max() / 500] = 0
        assert_sparse_equal(mapper.graph_, expected.graph_)


@pytest.mark.parametrize("force", [False, True])
def test_parametric_graph_uses_configured_search_without_keras(monkeypatch, force):
    search = {"metric": "euclidean", "force_approximation_algorithm": True} if force else {
        "metric": "minkowski", "metric_kwds": {"p": 1.0},
    }
    expected = UMAP(**BASE, transform_mode="graph", **search).fit(TRAINING)

    class IdentityEncoder:
        def predict(self, X, batch_size, verbose):
            return X.copy()

    def train(model, features, head, tail, counts, total, a, b):
        np.testing.assert_array_equal(features, TRAINING)
        assert model.graph_.nnz > 0
        np.testing.assert_array_equal(
            np.asarray(model.graph_[head, tail]).ravel(),
            np.asarray(expected.graph_[head, tail]).ravel(),
        )
        assert (counts > 0).all()
        return IdentityEncoder(), None, None, {}

    monkeypatch.setattr(umap_alternative, "_parametric_keras", SimpleNamespace(train=train), raising=False)
    model = ParametricUMAP(**BASE, **search)
    np.testing.assert_array_equal(model.fit_transform(TRAINING), TRAINING)
