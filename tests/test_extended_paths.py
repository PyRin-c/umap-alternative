import pickle
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import umap_alternative
from umap_alternative import AlignedUMAP, ParametricUMAP, UMAP, _native, _recursive, load_ParametricUMAP
from umap_alternative._feature_search import ApproximateFeatureIndex


OPTIONS = {
    "n_neighbors": 4, "compatibility_layout": True, "random_state": 19,
    "transform_seed": 31, "n_jobs": 1, "n_epochs": 24, "init": "random",
    "a": 1.576943460, "b": .895060879,
}
TRAINING = np.asarray([
    [0, 0], [0, 1], [1, 3], [4, 4], [2, 7], [6, 6], [7, 2], [11, 3], [9, 8],
], dtype=np.float32)
QUERY = np.asarray([[.25, 2.75], [7.75, 4.5]], dtype=np.float32)
METRIC = {"metric": "minkowski", "metric_kwds": {"p": 1.}}

pytestmark = [
    pytest.mark.filterwarnings("ignore:gradient function is not yet implemented for bit_"),
    pytest.mark.filterwarnings("ignore:n_jobs value -1 overridden"),
]


def assert_same_graph(left, right):
    a, b = left.tocsr(), right.tocsr()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.data, b.data)


def test_automatic_search_uses_the_distinct_row_boundary():
    assert not _native._uses_approximate_search(4095, False)
    assert _native._uses_approximate_search(4096, False)
    data = np.random.default_rng(719).standard_normal((4096, 3), dtype=np.float32)
    below_boundary = UMAP(**OPTIONS, transform_mode="graph").fit(data[:-1])
    assert below_boundary._feature_search_index is None
    assert below_boundary.graph_.shape == (4095, 4095)
    graph_model = UMAP(**OPTIONS, transform_mode="graph").fit(data)
    assert graph_model._feature_search_index is not None
    held = graph_model.transform(data[::2048] + .015)
    assert held.shape == (2, len(data))
    np.testing.assert_array_equal(np.bincount(held.row), [4, 4])
    collapsed = UMAP(**OPTIONS, transform_mode="graph", unique=True).fit(np.resize(TRAINING, (4096, 2)))
    assert collapsed._feature_search_index is None
    assert collapsed.graph_.shape == (len(TRAINING), len(TRAINING))


def test_approximate_query_uses_its_requested_width():
    training = np.random.RandomState(42).normal(size=(96, 4)).astype(np.float32)
    query = np.random.RandomState(15).normal(size=(41, 4))[40:].astype(np.float32)
    index = ApproximateFeatureIndex(training, 5, "euclidean", 42)

    narrow, distances = index.query(query, k=1)
    broad, _ = index.query(query, k=5)
    # NNDescent visits different candidates at these widths.
    np.testing.assert_array_equal(narrow, [[36]])
    np.testing.assert_array_equal(broad, [[19, 36, 86, 5, 15]])
    assert distances.shape == (1, 1)


@pytest.mark.parametrize("mode", ["graph", "embedding"])
def test_configured_csr_fit_transform_and_update_match_dense(mode):
    options = OPTIONS | METRIC | {"transform_mode": mode}
    dense = UMAP(**options).fit(TRAINING)
    sparse = UMAP(**options).fit(csr_matrix(TRAINING))
    assert_same_graph(sparse.graph_, dense.graph_)
    if mode == "graph":
        assert_same_graph(sparse.transform(csr_matrix(QUERY)), dense.transform(QUERY))
    else:
        np.testing.assert_array_equal(sparse.embedding_, dense.embedding_)
        np.testing.assert_array_equal(sparse.transform(csr_matrix(QUERY)), dense.transform(QUERY))
    dense.update(QUERY)
    sparse.update(csr_matrix(QUERY))
    assert_same_graph(sparse.graph_, dense.graph_)
    if mode == "embedding":
        np.testing.assert_array_equal(sparse.embedding_, dense.embedding_)


@pytest.mark.parametrize("density", [False, True])
def test_configured_csr_density_uses_unsquared_edge_distances(density):
    options = OPTIONS | METRIC | {"densmap": density, "output_dens": True}
    dense = UMAP(**options).fit_transform(TRAINING)
    sparse = UMAP(**options).fit_transform(csr_matrix(TRAINING))
    for actual, expected in zip(sparse, dense, strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
    assert sparse[1].shape == sparse[2].shape == (len(TRAINING),)


@pytest.mark.parametrize("name", ["bit_hamming", "bit_jaccard"])
def test_packed_features_can_report_original_and_embedding_radii(name):
    packed = np.asarray([[128 + index, 128 + index * 3] for index in range(9)], dtype=np.uint8)
    embedding, original, mapped = UMAP(**OPTIONS, metric=name, output_dens=True).fit_transform(packed)
    assert embedding.shape == (9, 2)
    assert original.shape == mapped.shape == (9,)
    assert np.isfinite(original).all()
    assert np.isfinite(mapped).all()


def test_configured_inverse_uses_numerical_or_provided_feature_gradient():
    called = []

    def l1_gradient(x, y):
        delta = x - y
        called.append(np.any(delta != 0))
        return np.abs(delta).sum(), np.sign(delta)

    numerical = UMAP(**OPTIONS, **METRIC).fit(TRAINING)
    analytic = UMAP(**OPTIONS, metric=l1_gradient).fit(TRAINING)
    np.testing.assert_array_equal(numerical.embedding_, analytic.embedding_)
    points = numerical.embedding_[0:2] * .75 + numerical.embedding_[2:4] * .25
    called.clear()
    recovered = analytic.inverse_transform(points)
    assert recovered.shape == (2, TRAINING.shape[1])
    assert np.isfinite(recovered).all()
    assert any(called)
    np.testing.assert_allclose(numerical.inverse_transform(points), recovered, atol=2e-4, rtol=0)


@pytest.mark.parametrize("weighted,bits", [
    (False, [[1057376572, 1073424220], [3216803114, 1071774026], [3216803114, 1071774026],
             [3171089836, 1043876914], [3171089836, 1043876914], [3171089836, 1043876914],
             [3219376185, 3216504904], [3219376185, 3216504904], [3208573610, 3217160826],
             [1049843258, 3217816750], [1036522478, 3214538823], [3183292709, 3209684928],
             [1053500432, 1074413969], [1053500432, 1074413969], [1073001940, 3209228424],
             [1073001940, 3209228424], [1073001940, 3209228424]]),
    (True, [[1057329332, 1073402496], [3216826734, 1071753827], [3216826733, 1071753827],
            [3171845682, 1043813577], [3171845682, 1043813577], [3171845682, 1043813577],
            [3219399805, 3216500292], [3219399805, 3216500292], [3207674928, 3217191354],
            [1049748777, 3217810927], [1035002257, 3214389390], [3183670632, 3209685391],
            [1053405951, 1074402340], [1053405951, 1074402340], [1072978320, 3209229308],
            [1072978320, 3209229308], [1072978320, 3209229308]]),
])
def test_recursive_anchor_can_use_a_subset(weighted, bits):
    features = np.random.RandomState(13).normal(size=(17, 4)).astype(np.float32)
    edges = np.linspace(.1, 1, 16, dtype=np.float32) if weighted else np.ones(16, dtype=np.float32)
    graph = csr_matrix(np.diag(edges, 1) + np.diag(edges, -1))
    rng = np.random.RandomState(19)
    actual = _recursive._modern_recursive_initialization(
        graph, features, 2, rng, {"base_init_threshold": 17, "n_embedding_epochs": 0, "anchor_sample_size": 7},
        4, "transition", None,
    )
    assert rng.get_state()[2] == 134
    np.testing.assert_allclose(actual, np.asarray(bits, dtype=np.uint32).view(np.float32), atol=2e-6, rtol=0)


def test_aligned_can_use_continuous_target_neighbor_counts():
    y = np.linspace(0, 1, len(TRAINING), dtype=np.float32)
    model = AlignedUMAP(
        n_epochs=0, n_neighbors=4, random_state=19, compatibility_layout=True,
        target_metric="euclidean", target_n_neighbors=[2, 3],
    ).fit([TRAINING, TRAINING + 1], y=[y, y], relations=[dict(enumerate(range(len(TRAINING))))])
    assert [mapper.target_n_neighbors for mapper in model.mappers_] == [2, 3]
    assert all(mapper.graph_.nnz > 0 for mapper in model.mappers_)


class Network:
    def __init__(self, scale):
        self.scale = scale

    def predict(self, X, **kwargs):
        return X * self.scale

    def save(self, path):
        path.write_bytes(pickle.dumps(self))


class TrainingModel(Network):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


def test_parametric_multirow_save_and_load_preserve_both_networks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    def train(model, *_):
        encoder, decoder = Network(2), Network(.5)
        return encoder, decoder, TrainingModel(encoder, decoder), {"loss": [.1]}

    keras = SimpleNamespace(models=SimpleNamespace(load_model=lambda path, **_: pickle.loads(path.read_bytes())))
    backend = SimpleNamespace(train=train, keras=keras)
    monkeypatch.setitem(sys.modules, "umap_alternative._parametric_keras", backend)
    monkeypatch.setattr(umap_alternative, "_parametric_keras", backend, raising=False)
    model = ParametricUMAP(**OPTIONS, parametric_reconstruction=True).fit(TRAINING)
    model.save(tmp_path, verbose=False, exclude_raw_data=True)
    restored = load_ParametricUMAP(tmp_path, verbose=False)
    assert restored._graph_transform_fit[0].shape == (0, TRAINING.shape[1])
    assert restored.history_ == model.history_
    assert restored.encoder_ is restored.parametric_model_.encoder
    assert restored.decoder_ is restored.parametric_model_.decoder
    np.testing.assert_array_equal(restored.transform(QUERY), model.transform(QUERY))
    np.testing.assert_array_equal(restored.inverse_transform(QUERY), model.inverse_transform(QUERY))
