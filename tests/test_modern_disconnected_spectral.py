"""Disconnected spectral placement observed through public modern fits."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from test_modern_spectral_fit import PLATFORM, assert_bits, check_graph, rng_checkpoint
from umap_alternative import UMAP


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-disconnected-spectral-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


class TracedRandomState(np.random.RandomState):
    def __init__(self, seed, jitter):
        super().__init__(seed)
        self.jitter = jitter
        self.draws = []

    def uniform(self, low=0.0, high=1.0, size=None):
        before = rng_checkpoint(self)
        values = super().uniform(low, high, size)
        self.draws.append(["uniform", list(values.shape), [low, high], before, rng_checkpoint(self)])
        return values

    def normal(self, loc=0.0, scale=1.0, size=None):
        before = rng_checkpoint(self)
        values = super().normal(loc, scale, size)
        self.draws.append(["normal", list(values.shape), [loc, scale], before, rng_checkpoint(self)])
        return np.zeros_like(values) if self.jitter == "zero" else values


def fit_case(case, method):
    rng = TracedRandomState(case["seed"], case.get("jitter"))
    if case.get("warm_gaussian"):
        rng.standard_normal()
    assert rng_checkpoint(rng) == case["rng_start"]
    model = UMAP(**case["options"], init="spectral", random_state=rng, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(model, method)(np.array(case["data"], dtype=np.float32))
    assert result is (model if method == "fit" else model.embedding_)
    assert not any("Spectral initialisation needs more samples" in str(w.message) for w in caught)
    assert any("Graph is not fully connected" in str(w.message) for w in caught) == case["affinity_warning"]
    check_graph(model.graph_, case["graph"])
    sampled = model.graph_.copy()
    epochs = model.n_epochs
    budget = max(epochs) if isinstance(epochs, list) else epochs
    sampled.data[sampled.data < np.float32(sampled.data.max() / (budget if budget > 10 else 500))] = 0
    sampled.eliminate_zeros()
    np.testing.assert_array_equal(connected_components(sampled, directed=False)[1], case["sampled_labels"])
    assert rng.draws == case["initialization_draws"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    return model


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_disconnected_modern_spectral_fit(case, method):
    model = fit_case(case, method)
    assert_bits(model.embedding_, case["embedding_bits"])
    if case["history_bits"] is None:
        assert not hasattr(model, "embedding_list_")
    else:
        for layout, expected in zip(model.embedding_list_, case["history_bits"], strict=True):
            assert_bits(layout, expected)


@pytest.mark.parametrize("case", REFERENCE["residuals"], ids=lambda case: case["name"])
def test_degenerate_disconnected_modern_spectral_fit(case):
    model = fit_case(case, "fit")
    assert np.isfinite(model.embedding_).all()
    cycle = np.asarray(case["sampled_labels"]) == 0
    angles = np.arange(np.count_nonzero(cycle)) * (2 * np.pi / np.count_nonzero(cycle))
    harmonics = np.column_stack([np.sin(angles), np.cos(angles)])
    basis = np.linalg.qr(harmonics)[0]

    observations = [np.asarray(words, dtype=np.uint32).view(np.float32).reshape(-1, 2)
                    for words in case["observations"]]
    for layout in [model.embedding_, *observations]:
        coordinates = layout[cycle].astype(np.float64)
        projected = np.linalg.qr(coordinates - coordinates.mean(axis=0))[0]
        np.testing.assert_allclose(projected @ projected.T, basis @ basis.T, rtol=0, atol=2e-6)


def test_all_isolated_modern_spectral_fit_preserves_caller_rng():
    case = REFERENCE["all_isolated"]
    rng = TracedRandomState(case["seed"], None)
    rng.standard_normal()
    checkpoint = {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng_checkpoint(rng) == checkpoint
    model = UMAP(**case["options"], init="spectral", random_state=rng, n_jobs=1)
    with pytest.warns(UserWarning, match="fully disconnected"), pytest.raises(
        ValueError,
    ) as caught:
        model.fit(np.array(case["data"], dtype=np.float32))
    assert str(caught.value) == case["error"]
    assert not rng.draws
    assert rng_checkpoint(rng) == checkpoint
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
