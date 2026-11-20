import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from umap_alternative import UMAP, _umap
from test_supervised_graph import assert_csr_matches


FIXTURES = Path(__file__).resolve().parents[1] / "reference/fixtures"
OS = "ubuntu" if platform.system() == "Linux" else "windows"


def load_fixture(stage):
    path = FIXTURES / f"umap-learn-0.6.0-supervised-{stage}-{OS}-x86_64.json"
    return json.loads(path.read_text(encoding="utf-8"))


CATEGORICAL = load_fixture("embedding")
CONTINUOUS = load_fixture("continuous")
SUPPORTED_TARGET_METRICS = {"euclidean", "manhattan", "chebyshev"}
GRAPH_RESIDUALS = {"euclidean_holdout": 1.2e-6, "chebyshev_holdout": 6.6e-5}


def assert_array_bits(array, expected):
    assert array.shape == tuple(expected["shape"])
    assert array.dtype.name == expected["dtype"]
    bits_dtype = np.uint32 if array.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(array.ravel().view(bits_dtype), expected["bits"])


def exercise_fit(fixture, name, method, monkeypatch):
    case = fixture["cases"][name]
    params = fixture["base_parameters"] | case["options"]
    params.pop("force_approximation_algorithm")
    if isinstance(params["init"], list):
        params["init"] = np.asarray(params["init"], dtype=np.float32)
    data = np.asarray(case.get("data", fixture["data"]), dtype=np.float32)[:case["data_rows"]]
    y = case["labels"]
    model = UMAP(**params)
    layout = _umap._fit_graph_embedding
    streams = []

    def capture_rng(graph, features, metric, rng, initial, **options):
        result = layout(graph, features, metric, rng, initial, **options)
        streams.append(rng)
        return result

    monkeypatch.setattr(_umap, "_fit_graph_embedding", capture_rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(model, method)
        if case["error"] is None:
            result = fit(data, y=y)
        else:
            with pytest.raises(ValueError) as failure:
                fit(data, y=y)
            assert type(failure.value).__name__ == case["error"]
    assert [[item.category.__name__, str(item.message)] for item in caught] == case["warnings"]

    graph = case["graph"]
    if graph is None:
        assert not hasattr(model, "graph_")
    else:
        assert_csr_matches(model.graph_, graph, atol=GRAPH_RESIDUALS.get(name, 0.0))
        assert model.n_neighbors_ == min(model.n_neighbors, len(data) - 1)
    expected = case["embedding"]
    if expected is None:
        assert not hasattr(model, "embedding_")
    else:
        if "repeat_embeddings" in case or name == "euclidean_holdout":
            assert model.embedding_.shape == tuple(expected["shape"])
            assert model.embedding_.dtype.name == expected["dtype"]
            assert np.isfinite(model.embedding_).all()
        else:
            assert_array_bits(model.embedding_, expected)

    if case["error"] is not None:
        assert not streams
        return
    assert model.n_features_in_ == data.shape[1]
    if method == "fit":
        assert result is model
    else:
        if "density" in case:
            result, *radii = result
            for actual, expected_radii in zip(radii, case["density"], strict=True):
                assert_array_bits(actual, expected_radii)
        assert result.dtype.name == case["fit_transform_dtype"]
        assert (result is getattr(model, "embedding_", None)) == case["fit_transform_returns_embedding"]
    if "checkpoints" in case:
        for actual, snapshot in zip(model.embedding_list_, case["checkpoints"], strict=True):
            assert_array_bits(actual, snapshot)
    if case["same_input_transform_returns_fitted_output"]:
        fitted_output = model.embedding_ if expected is not None else model.graph_
        assert model.transform(data.copy()) is fitted_output
    if case["supervised_update_error"] == "ValueError":
        with pytest.raises(ValueError, match="Updating supervised models"):
            model.update(data[:1])
    if (
        expected is not None and len(data) > 1 and "density" not in case
        and (y is not None or params["init"] == "spectral")
    ):
        [rng] = streams
        _, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
        assert [cursor, has_gaussian, cached_gaussian] == case["rng"][:3]
        np.testing.assert_array_equal(rng.randint(0, 2**32, 4, dtype=np.uint32), case["rng"][3])


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("name", CATEGORICAL["cases"])
def test_categorical_embedding_matches_public_fit(name, method, monkeypatch):
    exercise_fit(CATEGORICAL, name, method, monkeypatch)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("name", [
    name for name, case in CONTINUOUS["cases"].items()
    if case["options"]["target_metric"] in SUPPORTED_TARGET_METRICS and case["error"] != "SystemError"
])
def test_continuous_target_matches_public_fit(name, method, monkeypatch):
    exercise_fit(CONTINUOUS, name, method, monkeypatch)


def test_supervised_refit_preserves_previous_fit_on_failure_and_clears_supervision_on_success():
    X = np.asarray(CATEGORICAL["data"], dtype=np.float32)
    labels = CATEGORICAL["cases"]["semisupervised_random"]["labels"]
    model = UMAP(random_state=42, compatibility_layout=True, n_jobs=1, n_neighbors=4,
                 init="random", n_epochs=30).fit(X, y=labels)
    graph, embedding = model.graph_, model.embedding_
    model.set_params(target_weight=1.0)
    with pytest.raises(ValueError, match="zero-size array"):
        model.fit(X, y=np.arange(len(X)))
    assert model.graph_ is graph
    assert model.transform(X) is embedding

    plain = model.set_params(target_metric="cosine", target_weight=np.nan, target_n_neighbors=0)
    baseline = clone(plain).set_params(target_metric="categorical", target_weight=0.5)
    np.testing.assert_array_equal(plain.fit_transform(X), baseline.fit_transform(X))
    assert not model._supervised_fit


def test_continuous_density_and_unverified_target_metric_raise_explicitly():
    X = np.asarray(CONTINUOUS["data"], dtype=np.float32)
    target = np.arange(len(X), dtype=np.float32)
    model = UMAP(random_state=42, compatibility_layout=True, n_jobs=1, n_neighbors=4,
                 target_metric="euclidean", output_dens=True)
    with pytest.raises(NotImplementedError, match="density fitting with continuous targets"):
        model.fit(X, y=target)
    with pytest.raises(NotImplementedError, match="target_metric='cosine'"):
        model.set_params(target_metric="cosine", output_dens=False).fit(X, y=target)
