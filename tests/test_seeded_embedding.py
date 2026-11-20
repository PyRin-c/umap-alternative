import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference/fixtures"
SEEDED_FIT = json.loads(
    (FIXTURES / "umap-learn-0.6.0-seeded-fit-windows-x86_64.json").read_text(
        encoding="utf-8"
    )
)
WEIGHTED = np.asarray(SEEDED_FIT["inputs"]["weighted"], dtype=np.float32)
EXPLICIT = np.asarray(SEEDED_FIT["cases"][0]["init"], dtype=np.float32)
CHECKPOINTS = [
    pytest.param(case, int(epoch), bits, id=f"{case['name']}-{epoch}")
    for case in SEEDED_FIT["cases"]
    for epoch, bits in case["checkpoints"].items()
]


def embedding_model(**overrides):
    settings = dict(
        init="random",
        random_state=0,
        compatibility_layout=True,
        n_neighbors=4,
        n_jobs=1,
        n_epochs=2,
    )
    return UMAP(**(settings | overrides))


def assert_embedding_bits(embedding, bits):
    assert embedding.dtype == np.float32
    np.testing.assert_array_equal(embedding.view(np.uint32), np.asarray(bits, dtype=np.uint32))


def assert_graph_bits(graph, shape, indptr, indices, bits):
    assert isinstance(graph, csr_matrix)
    assert graph.shape == shape
    assert graph.dtype == np.float32
    np.testing.assert_array_equal(graph.indptr, indptr)
    np.testing.assert_array_equal(graph.indices, indices)
    np.testing.assert_array_equal(graph.data.view(np.uint32), np.asarray(bits, dtype=np.uint32))


@pytest.mark.parametrize("case,epoch,bits", CHECKPOINTS)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_seeded_embedding_matches_public_fixture(case, epoch, bits, method):
    data = np.asarray(SEEDED_FIT["inputs"][case["input"]], dtype=np.float32)
    initial = case["init"]
    if isinstance(initial, list):
        initial = np.asarray(initial, dtype=np.float32)
    model = UMAP(**case["params"], n_epochs=epoch, init=initial)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = getattr(model, method)(data)
    assert result is (model if method == "fit" else model.embedding_)

    expected = SEEDED_FIT["graphs"][case["graph"]]
    assert_graph_bits(
        model.graph_,
        (len(data), len(data)),
        expected["indptr"],
        expected["indices"],
        expected["bits"],
    )
    assert_embedding_bits(model.embedding_, bits)
    assert model.n_features_in_ == data.shape[1]
    assert model.n_neighbors_ == case["params"]["n_neighbors"]
    if isinstance(initial, np.ndarray):
        np.testing.assert_array_equal(initial, case["init"])


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_default_curve_matches_public_fit(method):
    fixture = json.loads(
        (FIXTURES / "umap-learn-0.6.0-transform-windows-x86_64.json").read_text(
            encoding="utf-8"
        )
    )
    data = np.asarray(fixture["inputs"]["training"]["bits_uint32"], dtype=np.uint32)
    data = data.view(np.float32)
    params = dict(fixture["parameters"])
    for key in (
        "output_metric", "transform_queue_size", "transform_seed", "force_approximation_algorithm"
    ):
        del params[key]
    model = UMAP(**params)
    result = getattr(model, method)(data)
    assert result is (model if method == "fit" else model.embedding_)

    graph = fixture["cases"]["graph"]["training"]["graph"]
    assert_graph_bits(
        model.graph_,
        tuple(graph["shape"]),
        graph["indptr"],
        graph["indices"],
        graph["data_bits_uint32"],
    )
    expected = fixture["cases"]["embedding"]["training"]["embedding"]
    assert_embedding_bits(model.embedding_, expected["bits_uint32"])


@pytest.mark.parametrize("case", SEEDED_FIT["cases"][:2], ids=lambda case: case["name"])
def test_small_data_default_epochs_match_500(case):
    initial = case["init"]
    if isinstance(initial, list):
        initial = np.asarray(initial, dtype=np.float32)
    model = UMAP(**case["params"], init=initial)
    assert_embedding_bits(model.fit_transform(WEIGHTED), case["checkpoints"]["500"])
    assert model.n_epochs is None


@pytest.mark.parametrize("seed", [np.uint32(0), np.uint32(2**32 - 1)])
def test_numpy_integer_seed_edges(seed):
    expected = embedding_model(random_state=int(seed)).fit_transform(WEIGHTED)
    assert_embedding_bits(
        embedding_model(random_state=seed).fit_transform(WEIGHTED),
        expected.view(np.uint32),
    )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_unsigned_32_bit_range(seed):
    with pytest.raises(ValueError, match=r"Seed must be between 0 and 2\*\*32 - 1"):
        embedding_model(random_state=seed).fit(WEIGHTED)


@pytest.mark.parametrize("epochs", [-1, 0.0, 2.5])
def test_epochs_must_be_nonnegative_integers(epochs):
    with pytest.raises(ValueError, match="n_epochs"):
        embedding_model(n_epochs=epochs).fit(WEIGHTED)


@pytest.mark.parametrize("components", [0, -1, 2.5, "2"])
def test_invalid_embedding_dimensions(components):
    with pytest.raises(ValueError, match="n_components"):
        embedding_model(n_components=components).fit(WEIGHTED)


@pytest.mark.parametrize(
    "initial,match",
    [
        (EXPLICIT[:-1], "same number of rows"),
        (EXPLICIT[:, :1], "must match n_components"),
        (EXPLICIT[:, 0], "Expected 2D array"),
        (EXPLICIT[:, :, None], "dim 3"),
    ],
)
def test_invalid_explicit_init_shape(initial, match):
    with pytest.raises(ValueError, match=match):
        embedding_model(init=initial).fit(WEIGHTED)


@pytest.mark.parametrize("min_dist", [0.0, 0.1, 1.0])
def test_min_dist_at_or_below_spread(min_dist):
    embedding = embedding_model(min_dist=min_dist, spread=1.0).fit_transform(WEIGHTED)
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize("min_dist", [np.nextafter(0.0, -np.inf), np.nextafter(1.0, np.inf)])
def test_min_dist_outside_spread_bounds(min_dist):
    with pytest.raises(ValueError, match="min_dist"):
        embedding_model(min_dist=min_dist, spread=1.0).fit(WEIGHTED)


@pytest.mark.parametrize("overrides", [{"a": 7.0}, {"b": 7.0}])
def test_partial_curve_pair_refits_both_coefficients(overrides):
    expected = embedding_model().fit_transform(WEIGHTED)
    model = embedding_model(**overrides)
    assert_embedding_bits(model.fit_transform(WEIGHTED), expected.view(np.uint32))
    assert all(model.get_params()[key] == value for key, value in overrides.items())


@pytest.mark.parametrize(
    "data",
    [WEIGHTED[:0], WEIGHTED[:, :0], WEIGHTED[0], WEIGHTED[:, :, None]],
)
def test_invalid_feature_shape(data):
    with pytest.raises(ValueError):
        embedding_model().fit(data)


def test_singleton_embedding_is_zero_and_removes_prior_graph():
    model = embedding_model(n_components=3).fit(WEIGHTED)
    returned = model.fit_transform(WEIGHTED[:1])
    assert_embedding_bits(returned, [[0, 0, 0]])
    assert model.embedding_.dtype == np.float64
    np.testing.assert_array_equal(model.embedding_.view(np.uint64), [[0, 0, 0]])
    assert returned is not model.embedding_
    assert not hasattr(model, "graph_")
    assert not hasattr(model, "n_neighbors_")
    with pytest.raises(ValueError, match="only a single data sample"):
        model.transform(WEIGHTED[:1])


def test_two_rows_warn_and_reject_empty_embedding_graph():
    with pytest.warns(UserWarning) as caught:
        with pytest.raises(ValueError, match="zero-size array"):
            embedding_model().fit(WEIGHTED[:2])
    assert "truncating" in str(caught[0].message)
    assert "fully disconnected 2 vertices" in str(caught[1].message)


def test_refit_state_is_committed_after_validation_and_tracks_mode():
    model = embedding_model().fit(WEIGHTED)
    original_graph, original_embedding = model.graph_, model.embedding_
    invalid = WEIGHTED.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(invalid)
    assert model.graph_ is original_graph
    assert model.transform(WEIGHTED.astype(np.float64)) is original_embedding

    with pytest.raises(NotImplementedError, match="out-of-sample embedding transform"):
        model.transform(WEIGHTED[1:])
    with pytest.raises(NotImplementedError, match="inverse_transform"):
        model.inverse_transform(original_embedding)

    assert model.set_params(transform_mode="graph").fit(WEIGHTED) is model
    assert model.transform(WEIGHTED.copy()) is model.graph_
    assert not hasattr(model, "embedding_")

    changed = WEIGHTED[::-1, :2]
    assert model.set_params(transform_mode="embedding").fit(changed) is model
    assert model.n_features_in_ == 2
    assert model.transform(changed) is model.embedding_
    assert model.embedding_ is not original_embedding


@pytest.mark.parametrize("optimizer", [None, "compatibility"])
def test_compatibility_none_warns_and_selects_same_seeded_layout(optimizer):
    expected = embedding_model().fit_transform(WEIGHTED)
    with pytest.warns(FutureWarning, match="compatibility_layout was not set"):
        actual = embedding_model(
            compatibility_layout=None, optimizer=optimizer
        ).fit_transform(WEIGHTED)
    assert_embedding_bits(actual, expected.view(np.uint32))


def test_seeded_fit_warns_when_n_jobs_is_overridden():
    expected = embedding_model().fit_transform(WEIGHTED)
    model = embedding_model(n_jobs=-1)
    with pytest.warns(UserWarning, match="n_jobs value -1 overridden to 1"):
        actual = model.fit_transform(WEIGHTED)
    assert_embedding_bits(actual, expected.view(np.uint32))
    assert model.n_jobs == -1


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_modern_optimizer_cannot_use_seeded_compatibility_path(optimizer):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotImplementedError, match=optimizer):
            embedding_model(compatibility_layout=None, optimizer=optimizer).fit(WEIGHTED)
        with pytest.raises(ValueError, match="requires compatibility_layout=False"):
            embedding_model(optimizer=optimizer).fit(WEIGHTED)


@pytest.mark.parametrize(
    "overrides,exception,match",
    [
        ({"compatibility_layout": False}, NotImplementedError, "compatibility_layout=False"),
        ({"optimizer": "sgd"}, ValueError, "Unknown optimizer"),
        ({"init": "spectral"}, NotImplementedError, "spectral"),
        ({"init": EXPLICIT.tolist()}, ValueError, "string or ndarray"),
        ({"init": EXPLICIT[[0, 1, 1, 3, 4, 5, 6, 7]]}, NotImplementedError, "repeated rows"),
        ({"n_epochs": [0, 2]}, NotImplementedError, "scalar n_epochs"),
        ({"metric": "precomputed"}, NotImplementedError, "precomputed"),
    ],
)
def test_unsupported_embedding_features_fail_explicitly(overrides, exception, match):
    with pytest.raises(exception, match=match):
        embedding_model(**overrides).fit(WEIGHTED)


def test_embedding_rejects_sparse_features_and_supervision():
    with pytest.raises(TypeError, match="sparse input"):
        embedding_model().fit(csr_matrix(WEIGHTED))
    with pytest.raises(NotImplementedError, match="supervised embedding"):
        embedding_model().fit(WEIGHTED, y=np.zeros(len(WEIGHTED)))


def test_public_embedding_parameters_are_cloneable():
    model = embedding_model(learning_rate=0.13, repulsion_strength=0.37, a=1.7, b=0.9)
    assert clone(model).get_params() == model.get_params()
