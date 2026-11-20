"""The Python native bridge preserves the coordinate and Gaussian RNG handoff."""

import json
from pathlib import Path

import numpy as np
import pytest

from umap_alternative import UMAP, _native


FIXTURE = Path(__file__).parents[1] / "reference" / "fixtures" / "umap-learn-0.6.0-modern-init-ubuntu-x86_64.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASES = {item["name"]: item for item in REFERENCE["cases"]}
PAIRS = np.array([[0, 0], [1, 0], [0, 6], [3, 10]], dtype=np.float32)


def floats(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32).tolist()


@pytest.mark.parametrize("case_name", ["random-13x2-per_axis", "duplicates-gaussian-cache"])
def test_modern_initialization_and_rng_state(case_name):
    case = CASES[case_name]
    graph = REFERENCE["graphs"][case["graph"]]
    shape = (graph["n_rows"], graph["n_rows"])
    components = (shape, graph["indptr"], graph["indices"], floats(graph["weights_bits"]))
    rng = np.random.RandomState(case["seed"])
    if "rng_start" in case:
        rng.standard_normal()
    algorithm, words, position, has_gaussian, gaussian = rng.get_state()
    coords, (words, position, gaussian) = _native._initialize_modern_layout(
        components,
        (words.tolist(), position, gaussian if has_gaussian else None),
        n_components=case["n_components"],
        init=case["init"],
        init_rescale=case["method"],
        init_scale_factor=case["factor"],
        coordinates=floats(case["initial_bits"]) if case["initial_bits"] else None,
    )
    np.testing.assert_array_equal(np.array(coords, dtype=np.float32).view(np.uint32), case["embedding_bits"])

    # These three words belong to the optimizer, after the native stage.
    rng.set_state((algorithm, np.array(words, dtype=np.uint32), position, gaussian is not None, gaussian or 0.0))
    rng.randint(-2**31 + 1, 2**31 - 2, 3)
    _, _, position, has_gaussian, cached = rng.get_state()
    assert position == case["rng"]["position"]
    assert (cached if has_gaussian else None) == case["rng"]["gaussian"]
    assert rng.randint(2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]


@pytest.mark.parametrize("case_name", [
    "pair-isotropic-1.0",
    "pair-graph_edge-0.73",
    "flat-per_axis",
    "duplicates-gaussian-cache",
    "positive-1e-40",
    "positive-1e+39",
    "random-13x2-per_axis",
    "random-13x2-isotropic",
])
@pytest.mark.parametrize("optimizer,compatibility", [
    ("momentum", None), ("adam", None), (None, False), ("compatibility", False),
])
def test_zero_epoch_modern_estimator_matches_public_fit(case_name, optimizer, compatibility):
    case = CASES[case_name]
    n_components = case["n_components"]
    data = PAIRS if case["init"] == "explicit" else (
        np.random.RandomState(7234 + 13).uniform(-100, 100, (13, 3)).astype(np.float32)
    )
    rng = np.random.RandomState(case["seed"])
    if "rng_start" in case:
        rng.standard_normal()
    initial = case["init"] if case["init"] == "random" else (
        np.asarray(case["initial_bits"], dtype=np.uint32).view(np.float32).reshape(-1, n_components)
    )
    model = UMAP(
        n_neighbors=2, n_components=n_components, init=initial,
        random_state=rng, n_jobs=1, a=1.0, b=1.0, n_epochs=[0, 0],
        negative_sample_rate=0, compatibility_layout=compatibility, optimizer=optimizer,
        init_rescale=case["method"], init_scale_factor=case["factor"],
    )
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        result = model.fit_transform(data)
    assert result is model.embedding_
    assert model.transform(data.copy()) is result
    expected_graph = REFERENCE["graphs"][case["graph"]]
    np.testing.assert_array_equal(model.graph_.indptr, expected_graph["indptr"])
    np.testing.assert_array_equal(model.graph_.indices, expected_graph["indices"])
    np.testing.assert_array_equal(model.graph_.data.view(np.uint32), expected_graph["weights_bits"])
    np.testing.assert_array_equal(result.view(np.uint32).ravel(), case["embedding_bits"])
    assert len(model.embedding_list_) == 1
    np.testing.assert_array_equal(model.embedding_list_[0].view(np.uint32), result.view(np.uint32))
    _, _, position, has_gaussian, cached = rng.get_state()
    assert position == case["rng"]["position"]
    assert (cached if has_gaussian else None) == case["rng"]["gaussian"]
    assert rng.randint(2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]


def test_zero_epoch_modern_fit_masks_isolates_after_rescaling():
    model = UMAP(
        n_neighbors=2, init="random", n_epochs=0, random_state=42, n_jobs=1,
        a=1.0, b=1.0, compatibility_layout=False, disconnection_distance=1.5,
    )
    with pytest.warns(UserWarning, match="disconnected"):
        assert model.fit(PAIRS) is model
    np.testing.assert_array_equal(np.diff(model.graph_.indptr), [1, 1, 0, 0])
    np.testing.assert_array_equal(model.embedding_[:2].view(np.uint32), [
        [1039753016, 1062745316],
        [1066212819, 3187180078],
    ])
    assert np.isnan(model.embedding_[2:]).all()
    assert not hasattr(model, "embedding_list_")


@pytest.mark.parametrize("options, message", [
    ({"negative_sample_rate": -1}, "negative_sample_rate"),
    ({"init_rescale": "unknown"}, "init_rescale"),
    ({"init_scale_factor": 0}, "init_scale_factor"),
    ({"init_scale_factor": -1}, "init_scale_factor"),
    ({"init_scale_factor": np.inf}, "init_scale_factor"),
    ({"init_scale_factor": np.nan}, "init_scale_factor"),
    ({"init": PAIRS[:3]}, "same number of rows"),
    ({"init": PAIRS[:, :1]}, "match n_components"),
])
def test_invalid_zero_epoch_modern_fit_does_not_consume_randomness(options, message):
    rng = np.random.RandomState(3)
    model = UMAP(**(dict(
        n_neighbors=2, init="random", n_epochs=0, random_state=rng, n_jobs=1,
        a=1.0, b=1.0, compatibility_layout=False,
    ) | options))
    with pytest.raises(ValueError, match=message):
        model.fit(PAIRS)
    assert rng.get_state()[2] == 624
    assert not hasattr(model, "embedding_")
