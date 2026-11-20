import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from umap_alternative import UMAP, _umap


FIXTURES = Path(__file__).resolve().parents[1] / "reference/fixtures"
HANDOFF = json.loads(
    (FIXTURES / "umap-learn-0.6.0-spectral-handoff-windows-x86_64.json").read_text(encoding="utf-8")
)
REFERENCE = json.loads(
    (FIXTURES / (
        "umap-learn-0.6.0-spectral-fit-ubuntu-x86_64.json"
        if platform.system() == "Linux"
        else "umap-learn-0.6.0-spectral-fit-windows-x86_64.json"
    )).read_text(encoding="utf-8")
)
DISCONNECTED = json.loads(
    (FIXTURES / (
        "umap-learn-0.6.0-disconnected-spectral-ubuntu-x86_64.json"
        if platform.system() == "Linux"
        else "umap-learn-0.6.0-disconnected-spectral-windows-x86_64.json"
    )).read_text(encoding="utf-8")
)
CENTROID_LAYOUT_RESIDUALS = {"chebyshev-d2-e0-s24", "mixed-d2-e0-s7", "wide-d2-e0-s0"}
WEIGHTED = np.asarray(HANDOFF["input"], dtype=np.float32)
BASE = HANDOFF["settings"]


def assert_graph_bits(graph, n_rows, expected):
    assert isinstance(graph, csr_matrix)
    assert graph.shape == (n_rows, n_rows)
    assert graph.dtype == np.float32
    np.testing.assert_array_equal(graph.indptr, expected["indptr"])
    np.testing.assert_array_equal(graph.indices, expected["indices"])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["bits"])


def assert_fit_bits(model, data, method, graph, bits):
    result = getattr(model, method)(data)
    assert result is (model if method == "fit" else model.embedding_)
    assert model.n_features_in_ == data.shape[1]
    assert model.n_neighbors_ == min(model.n_neighbors, len(data) - 1)
    assert model.transform(data.astype(np.float64)) is model.embedding_
    assert model.embedding_.shape == (len(data), model.n_components)
    assert model.embedding_.dtype == np.float32
    np.testing.assert_array_equal(model.embedding_.view(np.uint32).ravel(), bits)
    assert_graph_bits(model.graph_, len(data), graph)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize(
    "seed,epochs,bits",
    [
        pytest.param(case["seed"], int(epochs), bits, id=f"{case['seed']}-{epochs}")
        for case in HANDOFF["cases"]
        for epochs, bits in case["checkpoints"].items()
    ],
)
def test_spectral_fit_matches_public_handoff(seed, epochs, bits, method):
    model = UMAP(**BASE, random_state=np.uint32(seed), n_epochs=epochs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_fit_bits(model, WEIGHTED, method, HANDOFF["graph"], bits)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_spectral_fit_matches_public_graph_embedding_and_rng(case, method, monkeypatch):
    initializer = _umap._spectral_initialization
    streams = []

    def capture_rng(graph, components, rng):
        streams.append(rng)
        return initializer(graph, components, rng)

    monkeypatch.setattr(_umap, "_spectral_initialization", capture_rng)
    params = case["params"]
    data = np.asarray(REFERENCE["inputs"][case["input"]], dtype=np.float32)
    model = UMAP(**params, random_state=case["seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_fit_bits(model, data, method, REFERENCE["graphs"][case["graph"]], case["bits"])
    if "compatibility_layout" in params:
        assert not caught
    else:
        assert len(caught) == 1 and caught[0].category is FutureWarning
        assert "compatibility_layout was not set" in str(caught[0].message)
        assert model.init == "spectral"
        assert model.n_epochs is None
        assert model.compatibility_layout is None

    [rng] = streams
    algorithm, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
    expected = case["post_fit_rng"]
    assert algorithm == "MT19937"
    assert (cursor, has_gaussian, cached_gaussian) == (
        expected["cursor"], expected["has_gaussian"], expected["cached_gaussian"],
    )
    np.testing.assert_array_equal(
        rng.randint(0, 2**32, size=4, dtype=np.uint32), expected["next_words"]
    )


def test_spectral_fit_builds_graph_once_and_prunes_before_initializing(monkeypatch):
    data = np.asarray(REFERENCE["inputs"]["weak"], dtype=np.float32)
    fit_graph = _umap._native._fit_dense_graph
    initializer = _umap._spectral_initialization
    builds = 0

    def count_graph(*args, **kwargs):
        nonlocal builds
        builds += 1
        return fit_graph(*args, **kwargs)

    def check_positive_graph(graph, components, rng):
        assert np.all(graph.data >= np.float32(1 / 11))
        assert graph.nnz == REFERENCE["graphs"]["weak-unpruned"]["indptr"][-1] - 4
        return initializer(graph, components, rng)

    monkeypatch.setattr(_umap._native, "_fit_dense_graph", count_graph)
    monkeypatch.setattr(_umap, "_spectral_initialization", check_positive_graph)
    params = BASE | dict(n_neighbors=5, n_epochs=11, random_state=0, learning_rate=0)
    model = UMAP(**params).fit(data)
    assert builds == 1
    assert np.count_nonzero(model.graph_.data == 0) == 4
    unpruned = UMAP(**params, transform_mode="graph").fit_transform(data)
    assert_graph_bits(unpruned, len(data), REFERENCE["graphs"]["weak-unpruned"])


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_spectral_fit_singleton_uses_the_existing_zero_layout(method):
    model = UMAP(compatibility_layout=True, n_components=3)
    result = getattr(model, method)(WEIGHTED[:1])
    assert model.embedding_.dtype == np.float64
    np.testing.assert_array_equal(model.embedding_, [[0, 0, 0]])
    assert not hasattr(model, "graph_")
    if method == "fit":
        assert result is model
    else:
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, model.embedding_)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_spectral_fit_empty_graph_warns_and_preserves_previous_fit(method):
    model = UMAP(**BASE, n_epochs=0, random_state=0).fit(WEIGHTED)
    graph, embedding = model.graph_, model.embedding_
    model.set_params(compatibility_layout=None, n_jobs=-1)
    with pytest.warns((FutureWarning, UserWarning)) as caught:
        with pytest.raises(ValueError, match="zero-size array"):
            getattr(model, method)(WEIGHTED[:2])
    assert len(caught) == 4
    for warning, expected in zip(
        caught, (
            "compatibility_layout was not set", "n_jobs value -1 overridden to 1",
            "truncating", "fully disconnected 2 vertices",
        ), strict=True,
    ):
        assert expected in str(warning.message)
    assert model.embedding_ is embedding
    assert model.graph_ is graph
    assert model.n_neighbors_ == BASE["n_neighbors"]
    assert model.transform(WEIGHTED.copy()) is embedding


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("components", [len(WEIGHTED), len(WEIGHTED) + 1])
def test_spectral_fit_rejects_dimensions_outside_connected_solver(method, components):
    model = UMAP(**(BASE | {"n_components": components}), n_epochs=0, random_state=0)
    with pytest.raises(ValueError, match="1 <= n_components < n_rows"):
        getattr(model, method)(WEIGHTED)
    assert not hasattr(model, "embedding_")
    assert not hasattr(model, "graph_")


@pytest.mark.parametrize("name", DISCONNECTED["cases"])
def test_disconnected_spectral_graph_layout_and_rng(name, monkeypatch):
    case = DISCONNECTED["cases"][name]
    features = np.asarray(DISCONNECTED["inputs"][case["input"]], dtype=np.float32)
    graph = DISCONNECTED["graphs"][case["graph"]]
    expected_warnings = case["warnings"]
    if case["input"] in DISCONNECTED["explicit_labels"]:
        # These probes supply neighbor lists to the reference estimator.
        # Replay the public graph at the dense graph builder boundary.
        sizes = (len(features),) * 2
        values = np.asarray(graph["bits"], dtype=np.uint32).view(np.float32).tolist()

        def explicit_graph(*args, graph_options=None):
            assert graph_options is None
            return sizes, graph["indptr"], graph["indices"], values

        monkeypatch.setattr(_umap._native, "_fit_dense_graph", explicit_graph)
        expected_warnings = [
            message for message in expected_warnings
            if not message.startswith(("precomputed_knn[2]", "A few of your vertices"))
        ]

    streams = []
    fit_layout = _umap._fit_spectral_embedding

    def fit_layout_with_rng(graph, features, metric, rng, **options):
        streams.append(rng)
        return fit_layout(graph, features, metric, rng, **options)

    monkeypatch.setattr(_umap, "_fit_spectral_embedding", fit_layout_with_rng)
    model = UMAP(**DISCONNECTED["settings"])
    for method in ("fit", "fit_transform"):
        model.set_params(**case["params"], random_state=case["seed"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = getattr(model, method)(features)
        assert [str(w.message).splitlines()[0] for w in caught] == expected_warnings
        assert fitted is (model if method == "fit" else model.embedding_)
        assert model.transform(features.copy()) is model.embedding_
        assert_graph_bits(model.graph_, len(features), graph)

        positive = model.graph_.copy()
        positive.eliminate_zeros()
        _, labels = connected_components(positive, directed=False)
        np.testing.assert_array_equal(np.bincount(labels), case["component_sizes"])
        embedding = model.embedding_
        assert embedding.shape == (len(features), case["params"]["n_components"])
        assert embedding.dtype == np.float32
        isolated = np.diff(positive.indptr) == 0
        assert np.isnan(embedding[isolated]).all()
        assert np.isfinite(embedding[~isolated]).all()
        if name not in CENTROID_LAYOUT_RESIDUALS:
            np.testing.assert_array_equal(embedding.view(np.uint32).ravel(), case["bits"])

        rng = streams[-1]
        algorithm, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
        expected = case["post_fit_rng"]
        assert algorithm == "MT19937"
        assert (cursor, has_gaussian, cached_gaussian) == (
            expected["cursor"], expected["has_gaussian"], expected["cached_gaussian"],
        )
        np.testing.assert_array_equal(
            rng.randint(0, 2**32, size=4, dtype=np.uint32), expected["next_words"]
        )


def test_spectral_fit_native_failure_preserves_previous_fit():
    model = UMAP(**BASE, n_epochs=0, random_state=0).fit(WEIGHTED)
    graph, embedding = model.graph_, model.embedding_
    model.set_params(negative_sample_rate=np.nextafter(0.0, 1.0))
    with pytest.raises(ValueError, match="negative sample interval"):
        model.fit_transform(WEIGHTED[::-1])
    assert model.graph_ is graph
    assert model.transform(WEIGHTED.copy()) is embedding
