import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial import QhullError

from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
REFERENCE = json.loads(
    (FIXTURES / "umap-learn-0.6.0-inverse-windows-x86_64.json").read_text(encoding="utf-8")
)
ORIGINAL = json.loads(
    (FIXTURES / "umap-learn-0.6.0-transform-windows-x86_64.json").read_text(encoding="utf-8")
)
METRICS = json.loads(
    (FIXTURES / "umap-learn-0.6.0-inverse-metrics-x86_64.json").read_text(encoding="utf-8")
)


def estimator(**overrides):
    settings = dict(
        n_neighbors=4, init="random", compatibility_layout=True, n_epochs=0,
        random_state=42, transform_seed=24, n_jobs=1, a=1.7, b=0.9,
    )
    return UMAP(**(settings | overrides))


def assert_bits(output, expected):
    assert type(output) is np.ndarray
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output.view(np.uint32), expected)


def from_record(record):
    return np.asarray(record["bits_uint32"], dtype=np.uint32).view(np.float32)


def reconstruct(model, queries):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = model.inverse_transform(queries)
    return output, caught


@pytest.mark.parametrize("name", METRICS["cases"])
def test_dense_inverse_uses_the_fitted_metric_and_scales(name):
    case = METRICS["cases"][name]
    training = np.asarray(case["training"], dtype=np.float32)
    options = METRICS["parameters"] | case["settings"]
    options["init"] = np.asarray(METRICS["explicit"], dtype=np.float32)
    if case["supplied"] is not None:
        values = training.astype(np.float64)
        d = np.linalg.norm(values[:, None] - values[None, :], axis=-1).astype(np.float32)
        indices = np.argsort(d, axis=1, kind="stable")[:, :case["supplied"]].astype(np.int32)
        distances = np.take_along_axis(d, indices, axis=1) * np.float32(0.37)
        options["precomputed_knn"] = (indices, distances, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = UMAP(**options).fit(training)
    if case["supplied"] is not None:
        distances.fill(123.0)
    queries = np.asarray(METRICS["queries"], dtype=np.float32)
    model.embedding_ = np.asarray(METRICS["embedding"], dtype=np.float32)
    for checkpoint in case["checks"]:
        model.set_params(n_epochs=checkpoint["fit_epochs"], repulsion_strength=checkpoint["strength"])
        output, caught = reconstruct(model, queries)
        assert [warning.category.__name__ for warning in caught] == checkpoint["warnings"]
        if name == "correlation" and checkpoint["fit_epochs"] == 30 and checkpoint["strength"] == 0.0:
            # Two coordinates in this attraction-only checkpoint differ by one ULP.
            expected = np.asarray(checkpoint["bits"], dtype=np.uint32).view(np.float32)
            np.testing.assert_array_max_ulp(output, expected, maxulp=1)
        else:
            assert_bits(output, checkpoint["bits"])


@pytest.mark.parametrize("case", REFERENCE["cases"] + REFERENCE["tie_diagnostics"], ids=lambda case: case["name"])
def test_inverse_uses_public_embedding_and_matches_locked_checkpoints(case):
    n_features, n_components = case["n_features"], case["n_components"]
    training = np.asarray(case["training"], dtype=np.float32).reshape(-1, n_features)
    queries = np.asarray(case["queries"], dtype=np.float32).reshape(-1, n_components)
    embedding = np.asarray(case["embedding"], dtype=np.float32).reshape(-1, n_components)
    model = estimator(
        n_neighbors=case["n_neighbors"], n_components=n_components,
        a=case["a"], b=case["b"], learning_rate=4 * case["initial_alpha"],
        transform_seed=case["transform_seed"],
    ).fit(training)
    model.embedding_ = embedding.copy()
    shape = len(queries), n_features
    for checkpoint in case["checkpoints"]:
        model.set_params(
            n_epochs=checkpoint["fit_epochs"],
            repulsion_strength=checkpoint["repulsion_strength"],
            negative_sample_rate=checkpoint["negative_sample_rate"],
        )
        output, caught = reconstruct(model, queries)
        assert output.shape == shape
        assert_bits(output, np.asarray(checkpoint["bits"], dtype=np.uint32).reshape(shape))
        # The Rust exterior reduction does not emit NumPy arithmetic warnings.
        assert [str(w.message) for w in caught] == [
            message for message in checkpoint["warnings"]
            if message != "overflow encountered in multiply"
        ]
        assert all(w.category is UserWarning for w in caught)
    np.testing.assert_array_equal(model.embedding_, embedding)


def test_fit_transform_and_inverse_match_the_original_45_epoch_records():
    inputs = ORIGINAL["inputs"]
    case = ORIGINAL["cases"]["embedding"]
    model = estimator(n_epochs=45, a=None, b=None).fit(from_record(inputs["training"]))
    assert_bits(model.embedding_, case["training"]["embedding"]["bits_uint32"])

    interior = from_record(inputs["inverse_queries"])
    exterior = model.transform(from_record(inputs["transform_queries"]))
    assert_bits(exterior, case["transform"]["output"]["bits_uint32"])
    for queries, key in (
        (interior, "inverse_transform"),
        (exterior, "inverse_transform_roundtrip"),
    ):
        output, caught = reconstruct(model, queries)
        assert output.shape == (3, 3)
        assert_bits(output, case[key]["output"]["bits_uint32"])
        assert [
            {"category": w.category.__name__, "message": str(w.message)} for w in caught
        ] == case[key]["warnings"]


@pytest.mark.parametrize("case", REFERENCE["default_schedule"], ids=lambda case: case["query_rows"])
def test_inverse_default_schedule_depends_on_query_count(case):
    training = np.array([[0, 0], [1.2, 1.5], [-0.1, 0.6], [1.6, -0.8], [-0.2, 0.1]], np.float32)
    model = estimator(n_neighbors=3, repulsion_strength=0).fit(training)
    model.embedding_ = np.array([[0, 0], [2, 0.1], [0.2, 1], [1, 1.1], [-0.3, -0.1]], np.float32)
    queries = np.broadcast_to(np.array([0.2, 0.2], np.float32), (case["query_rows"], 2))
    model.set_params(n_epochs=None)
    assert_bits(
        model.inverse_transform(queries),
        np.broadcast_to(case["first_row_bits"]["None"], queries.shape),
    )


def test_inverse_public_boundaries():
    training = np.asarray(REFERENCE["cases"][0]["training"], dtype=np.float32).reshape(-1, 3)
    query = [[0, 1]]
    model = estimator().fit(training)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precomputed = estimator(metric="precomputed", transform_mode="graph").fit(
            np.linalg.norm(training[:, None] - training, axis=-1)
        )
    too_few = estimator(n_components=len(training)).fit(training)
    too_few.embedding_ = np.eye(len(training), dtype=np.float32)
    cases = {
        "unfitted": (estimator(), query, AttributeError),
        "empty": (model, np.empty((0, 2), np.float32), ValueError),
        "wrong_width": (model, [[0]], ValueError),
        "nonfinite_query": (model, [[0, np.nan]], ValueError),
        "sparse_query": (model, csr_matrix(query), TypeError),
        "sparse_training": (
            estimator(transform_mode="graph").fit(csr_matrix(training)), query, ValueError,
        ),
        "graph": (estimator(transform_mode="graph").fit(training), query, ValueError),
        "precomputed": (precomputed, query, ValueError),
        "single_row": (estimator().fit(training[:1]), query, QhullError),
        "too_few_for_simplex": (too_few, np.zeros((1, len(training))), QhullError),
        "one_component": (estimator(n_components=1).fit(training), [[0]], ValueError),
    }
    for name, (fitted, invalid, error_type) in cases.items():
        with pytest.raises(error_type) as caught:
            fitted.inverse_transform(invalid)
        assert type(caught.value).__name__ == REFERENCE["boundaries"][name]["error"]
        assert str(caught.value).splitlines()[0] == REFERENCE["boundaries"][name]["message"]

def test_held_out_inverse_epochs_seed_and_fitted_state():
    # Separate public umap-learn 0.6.0 probe with nondefault fitted local connectivity.
    rng = np.random.RandomState(163)
    training = rng.uniform(-3, 3, (12, 3)).astype(np.float32)
    embedding = rng.normal(size=(12, 2)).astype(np.float32)
    queries = np.array([embedding[0], embedding.mean(axis=0), [5, 3], [-5, -5]], np.float32)
    model = estimator(local_connectivity=0.6, learning_rate=0.76).fit(training)
    model.embedding_ = embedding.copy()
    fitted, graph = model.embedding_, model.graph_
    graph_snapshot, query_snapshot = graph.copy(), queries.copy()
    rng_state = np.random.get_state()

    # A reconstruction uses the estimator's fitted copy of the training matrix.
    training.fill(np.nan)
    initialization = [[1065138542, 1058159409, 3212483023], [1060020234, 1058580523, 3205620708]]
    exterior = [[3206913168, 3206113268, 3214195140], [1074861887, 1070817142, 3206906374]]
    seed_zero = [[1079438427, 1057620438, 3196909848], [3183149761, 1077155736, 3148758481]]
    for epochs, seed, interior in (
        (0, 24, initialization),
        (5, 24, initialization),
        (6, 24, [[1059744313, 1055802203, 3206812955], initialization[1]]),
        (45, 0, seed_zero),
        (45, 2**32 - 1, [[1064912891, 1076090621, 3207156498], [3192989576, 1077518568, 3196905233]]),
        ([0, 9], 0, [[1070393300, 1058217692, 3206766653], [1045304484, 1066592010, 1051659118]]),
    ):
        model.set_params(n_epochs=epochs, transform_seed=np.uint32(seed))
        output, caught = reconstruct(model, queries)
        assert [str(w.message) for w in caught] == [
            "2 inverse-transform point(s) lie outside the embedding convex hull; "
            "using nearest-vertex extrapolation."
        ]
        assert_bits(output, interior + exterior)
        output.fill(np.nan)

    model.set_params(
        n_epochs=45, transform_seed=0, n_components=3, n_neighbors=2,
        learning_rate=0.015, a=3.7, b=1.4, local_connectivity=0, metric="chebyshev",
    )
    assert_bits(reconstruct(model, queries.astype(np.float64, order="F"))[0], seed_zero + exterior)
    model.set_params(transform_seed=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"Seed must be between 0 and 2\*\*32 - 1"):
            model.inverse_transform(queries)

    assert model.embedding_ is fitted
    assert model.graph_ is graph
    np.testing.assert_array_equal(fitted, embedding)
    np.testing.assert_array_equal(graph.data, graph_snapshot.data)
    np.testing.assert_array_equal(graph.indices, graph_snapshot.indices)
    np.testing.assert_array_equal(graph.indptr, graph_snapshot.indptr)
    np.testing.assert_array_equal(queries, query_snapshot)
    after = np.random.get_state()
    assert (after[0], *after[2:]) == (rng_state[0], *rng_state[2:])
    np.testing.assert_array_equal(after[1], rng_state[1])


def test_held_out_inverse_after_a_distinct_fit_with_truncated_neighbors():
    training = np.random.RandomState(163).uniform(-3, 3, (12, 3)).astype(np.float32)
    with pytest.warns(UserWarning, match="truncating"):
        model = estimator(
            n_neighbors=15, n_epochs=18, learning_rate=0.65, random_state=67, transform_seed=11,
        ).fit(training)
    assert model.n_neighbors_ == 11
    queries = [[4.1, 4.8], [3.2, 5.7], [18, -2]]
    with pytest.warns(UserWarning, match="^3 inverse-transform point"):
        output = model.inverse_transform(queries)
    assert_bits(output, [
        [3222266875, 3208666148, 3218384816],
        [1054828185, 1071115746, 3208638501],
        [1075510116, 1072968982, 3219947337],
    ])
