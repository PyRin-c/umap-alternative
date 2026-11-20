import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from umap_alternative import UMAP, _native


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
CASES = json.loads(
    (FIXTURES / "umap-learn-0.6.0-transform-embedding-windows-x86_64.json").read_text(
        encoding="utf-8"
    )
)["cases"]
LAYOUT_OPTIONS = (
    "n_components", "metric", "a", "b", "learning_rate",
    "repulsion_strength", "negative_sample_rate", "transform_seed",
)


def from_bits(values):
    return np.array(values, dtype=np.uint32).view(np.float32)


def assert_bits(actual, expected):
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected)


def queries_and_expected(case):
    queries = from_bits(case["queries"])
    rows = case["checked_query_rows"] or len(queries)
    return (
        np.resize(queries, (rows, queries.shape[1])),
        np.resize(case["output"], (rows, case["n_components"])),
    )


def estimator(case, **overrides):
    settings = dict(
        **{key: case[key] for key in LAYOUT_OPTIONS},
        n_neighbors=case["n_neighbors"],
        n_epochs=case["training_epochs"],
        init="random",
        random_state=case["fit_seed"],
        n_jobs=1,
        compatibility_layout=True,
    )
    return UMAP(**(settings | overrides))


@pytest.mark.parametrize("name", CASES)
def test_native_binding_matches_public_fixed_tail_fixture(name):
    case = CASES[name]
    training = from_bits(case["training"])
    queries, expected = queries_and_expected(case)
    output = _native._transform_dense_embedding(
        training.ravel().tolist(), len(training), queries.ravel().tolist(), len(queries),
        training.shape[1], from_bits(case["fitted"]).ravel().tolist(),
        k=case["n_neighbors"], fit_epochs=case["fit_epochs"],
        disconnection_distance=case["disconnection_distance"],
        **{key: case[key] for key in LAYOUT_OPTIONS},
    )
    assert_bits(np.array(output, dtype=np.float32).reshape(expected.shape), expected)


@pytest.mark.parametrize("name", [
    name for name, case in CASES.items()
    if case["init"] == "random" and case["disconnection_distance"] is None
])
def test_estimator_fit_then_transform_matches_public_fixture(name):
    case = CASES[name]
    training = from_bits(case["training"])
    queries, expected = queries_and_expected(case)
    model = estimator(case).fit(training)
    assert_bits(model.embedding_, case["fitted"])
    model.set_params(n_epochs=case["fit_epochs"])
    assert_bits(model.transform(queries), expected)
    assert_bits(model.transform(queries.copy()), expected)


def test_default_curve_fit_and_transform_match_public_fixture():
    fixture = json.loads(
        (FIXTURES / "umap-learn-0.6.0-transform-windows-x86_64.json").read_text(encoding="utf-8")
    )
    model = estimator(CASES["fit_18"], n_epochs=45, a=None, b=None)
    training = from_bits(fixture["inputs"]["training"]["bits_uint32"])
    queries = from_bits(fixture["inputs"]["transform_queries"]["bits_uint32"])
    model.fit(training)
    expected = fixture["cases"]["embedding"]
    assert_bits(model.embedding_, expected["training"]["embedding"]["bits_uint32"])
    assert_bits(model.transform(queries), expected["transform"]["output"]["bits_uint32"])


def test_transform_after_spectral_fit_matches_public_probe():
    case = CASES["fit_18"]
    training, queries = from_bits(case["training"]), from_bits(case["queries"])
    model = estimator(case, n_epochs=0, init="spectral").fit(training)
    assert model.transform(training) is model.embedding_
    model.set_params(n_epochs=45)
    assert_bits(model.transform(queries), [
        [1091468397, 1085771710], [1083348299, 1077916613], [1084563906, 1089811230],
        [1091643596, 1082018868], [1091468397, 1085771710],
    ])


def test_transform_reads_fitted_geometry_and_current_schedule():
    case = CASES["fit_18"]
    training, queries = from_bits(case["training"]), from_bits(case["queries"])
    model = estimator(case).fit(training)
    model.set_params(
        n_neighbors=2, n_components=3, random_state=7, metric="chebyshev",
        learning_rate=0.1, a=0.7, b=0.5, min_dist=0.9, spread=2,
    )
    assert_bits(model.transform(queries), case["output"])
    model.set_params(n_epochs=0)
    assert_bits(model.transform(queries), [
        [1079473542, 1084432434], [1083168577, 1084055592], [1076473203, 1084183210],
        [1077513049, 1084210005], [1079473542, 1084432434],
    ])
    model.set_params(n_epochs=18, transform_seed=np.uint32(0))
    assert_bits(model.transform(queries), [
        [1082524232, 1084767832], [1082658371, 1083502550], [1071365212, 1085784351],
        [1075526161, 1079404591], [1082524232, 1084767832],
    ])
    model.set_params(transform_seed=np.uint32(2**32 - 1))
    assert_bits(model.transform(queries), CASES["different_transform_seed"]["output"])
    model.set_params(transform_seed=case["transform_seed"], repulsion_strength=0)
    assert_bits(model.transform(queries), [
        [1078081698, 1084371018], [1084132865, 1083742266], [1073749032, 1083418031],
        [1077662257, 1084289214], [1078081698, 1084371018],
    ])
    model.set_params(repulsion_strength=1, negative_sample_rate=1)
    assert_bits(model.transform(queries), [
        [1081371125, 1085157011], [1083136559, 1083306457], [1073269582, 1083651560],
        [1079002609, 1084585879], [1081371125, 1085157011],
    ])
    assert_bits(model.embedding_, case["fitted"])


def test_transform_keeps_fitted_state_through_errors_and_refit():
    case = CASES["fit_18"]
    training, queries = from_bits(case["training"]), from_bits(case["queries"])
    model = estimator(case).fit(training)
    original_embedding, original_graph = model.embedding_, model.graph_
    graph_snapshot = original_graph.copy()
    assert model.transform(training) is original_embedding
    assert model.transform(np.asfortranarray(training, dtype=np.float64)) is original_embedding
    model.transform(queries)[:] = np.nan
    snapshot = training.copy()
    training[:] = np.nan
    assert model.transform(snapshot) is original_embedding

    for invalid in (
        queries[:0], queries[:, :0], queries[:, :-1], queries[0],
        np.full_like(queries, np.inf), np.full_like(queries, np.nan),
    ):
        with pytest.raises(ValueError):
            model.transform(invalid)
    with pytest.raises(TypeError, match="sparse input"):
        model.transform(csr_matrix(queries))
    with pytest.raises(ValueError, match="distance.*not finite"):
        model.transform(np.full_like(queries, np.finfo(np.float32).max))
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(training)
    invalid_init = np.column_stack((np.arange(len(snapshot)), np.zeros(len(snapshot))))
    model.set_params(learning_rate=0.7, a=0.3, b=0.8, init=invalid_init)
    with pytest.raises(ValueError, match="layout column 1 has zero span"):
        model.fit(snapshot)
    assert model.graph_ is original_graph
    assert model.embedding_ is original_embedding
    assert model.transform(snapshot) is original_embedding
    assert_bits(model.transform(queries), case["output"])
    assert_bits(model.embedding_, case["fitted"])
    assert_bits(model.graph_.data, graph_snapshot.data.view(np.uint32))
    np.testing.assert_array_equal(model.graph_.indices, graph_snapshot.indices)
    np.testing.assert_array_equal(model.graph_.indptr, graph_snapshot.indptr)

    replacement = snapshot[:, :2].copy()
    settings = dict(
        n_neighbors=3, metric="chebyshev", a=0.8, b=1.1, learning_rate=0.2, init="random"
    )
    model.set_params(**settings).fit(replacement)
    fresh = estimator(case, **settings).fit(replacement)
    assert model.embedding_ is not original_embedding
    assert model.transform(replacement) is model.embedding_
    assert_bits(model.transform(queries[:, :2]), fresh.transform(queries[:, :2]).view(np.uint32))

    model.set_params(transform_mode="graph")
    assert model.transform(replacement) is model.graph_
    graph = model.transform(queries[:, :2])
    assert graph.shape == (len(queries), len(replacement))
    assert graph.format == "coo"


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_transform_seed_validation_is_on_held_out_calls(seed):
    case = CASES["fit_18"]
    training, queries = from_bits(case["training"]), from_bits(case["queries"])
    model = estimator(case, transform_seed=seed)
    assert clone(model).get_params()["transform_seed"] == seed
    with pytest.raises(AttributeError, match="_raw_data"):
        model.transform(queries[:0])
    model.fit(training)
    assert model.transform(training) is model.embedding_
    with pytest.raises(ValueError, match=r"Seed must be between 0 and 2\*\*32 - 1"):
        model.transform(queries)
    model.set_params(transform_seed=case["transform_seed"])
    assert_bits(model.transform(queries), case["output"])
