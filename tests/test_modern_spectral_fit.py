"""Connected modern spectral fits against platform-specific public wheels."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from umap_alternative import UMAP


PLATFORM = "windows" if sys.platform == "win32" else "ubuntu"
FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-spectral-fit-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def rng_checkpoint(rng):
    _, _, cursor, has_gaussian, gaussian = rng.get_state()
    return {
        "cursor": cursor,
        "gaussian_bits": int(np.float64(gaussian).view(np.uint64)) if has_gaussian else None,
    }


class SpectralRandomState(np.random.RandomState):
    def __init__(self, seed, jitter):
        super().__init__(seed)
        self.jitter = jitter
        self.events = []

    def normal(self, *, scale, size):
        before = rng_checkpoint(self)
        values = super().normal(scale=scale, size=size)
        self.events.append({
            "method": "normal", "scale": scale, "shape": list(values.shape),
            "before": before, "after": rng_checkpoint(self),
        })
        return np.zeros_like(values) if self.jitter == "zero" else values

    def uniform(self, *, low, high, size):
        before = rng_checkpoint(self)
        values = super().uniform(low=low, high=high, size=size)
        self.events.append({
            "method": "uniform", "bounds": [low, high], "shape": list(values.shape),
            "before": before, "after": rng_checkpoint(self),
        })
        return values


def assert_bits(values, expected):
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values.view(np.uint32).ravel(), expected)


def check_graph(graph, expected):
    np.testing.assert_array_equal(graph.indptr, expected["indptr"])
    np.testing.assert_array_equal(graph.indices, expected["indices"])
    assert_bits(graph.data, expected["weights_bits"])


def fit_case(case, method):
    rng = SpectralRandomState(case["seed"], case.get("jitter"))
    if case.get("warm_gaussian"):
        rng.standard_normal()
    assert rng_checkpoint(rng) == case["rng_start"]
    model = UMAP(**case["options"], random_state=rng, n_jobs=1)
    data = np.asarray(case["data"], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = getattr(model, method)(data)
    assert fitted is (model if method == "fit" else model.embedding_)
    warned = any("Spectral initialisation needs more samples" in str(w.message) for w in caught)
    assert warned == case["rank_warning"]
    check_graph(model.graph_, case["graph"])
    assert rng.events == case["initialization_draws"]
    assert rng_checkpoint(rng) == {
        key: case["rng"][key] for key in ("cursor", "gaussian_bits")
    }
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    return model


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_connected_modern_spectral_fit(case, method):
    model = fit_case(case, method)
    assert_bits(model.embedding_, case["embedding_bits"])
    if case["history_bits"] is None:
        assert not hasattr(model, "embedding_list_")
    else:
        for observed, expected in zip(model.embedding_list_, case["history_bits"], strict=True):
            assert_bits(observed, expected)


def test_symmetric_spectral_fit_reproduces_the_repeated_eigenspace():
    case = REFERENCE["rank_degenerate"]
    model = fit_case(case, "fit")
    actual = np.linalg.qr(model.embedding_.astype(np.float64))[0]
    observed = np.asarray(case["embedding_bits"], dtype=np.uint32).view(np.float32).reshape(-1, 2)
    expected = np.linalg.qr(observed.astype(np.float64))[0]
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, rtol=0, atol=2e-6)


def test_disconnected_modern_spectral_stops_before_using_caller_rng():
    points = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], dtype=np.float32)
    rng = np.random.RandomState(11)
    model = UMAP(
        optimizer="adam", n_neighbors=3, n_epochs=2, random_state=rng,
        n_jobs=1, disconnection_distance=3.0,
    )
    with pytest.raises(NotImplementedError, match="modern spectral initialization requires a connected graph"):
        model.fit(points)
    assert rng_checkpoint(rng) == {"cursor": 624, "gaussian_bits": None}
