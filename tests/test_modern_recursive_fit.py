"""Recursive initialization observed through the pinned public estimator."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from test_modern_spectral_fit import PLATFORM, check_graph, rng_checkpoint
from umap_alternative import UMAP


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-fit-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def make_data(spec):
    if isinstance(spec, list):
        return np.asarray(spec, dtype=np.float32)
    rng = np.random.RandomState(spec["seed"])
    values = rng.uniform(-5, 7, spec["shape"]).astype(np.float32)
    return pairwise_distances(values) if spec.get("precomputed") else values


class TracedRandomState(np.random.RandomState):
    def __init__(self, seed):
        super().__init__(seed)
        self.normal_calls = []

    def normal(self, *args, **kwds):
        start = rng_checkpoint(self)
        values = super().normal(*args, **kwds)
        self.normal_calls.append(["normal", list(values.shape), start, rng_checkpoint(self)])
        return values


def fit_case(case, method):
    rng = TracedRandomState(case["seed"])
    if case["warm_gaussian"]:
        rng.standard_normal()
    assert rng_checkpoint(rng) == case["rng_start"]
    data = make_data(case["data"])
    model = UMAP(**case["options"], random_state=rng, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = getattr(model, method)(data.copy())
    assert fitted is (model if method == "fit" else model.embedding_)
    check_graph(model.graph_, case["graph"])
    assert rng.normal_calls == [call for call in case["rng_calls"] if call[0] == "normal"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    return model, data


def recorded_layout(words, shape):
    return np.asarray(words, dtype=np.uint32).view(np.float32).reshape(shape)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_modern_recursive_fit(case, method):
    model, data = fit_case(case, method)
    if case["name"].startswith("repeated-axis-"):
        # The equal PCA singular values determine a subspace, with a basis
        # that can rotate with the sketch and the LAPACK implementation.
        basis = np.linalg.qr(data.astype(np.float64))[0]
        observed = recorded_layout(case["embedding_bits"], model.embedding_.shape)
        for layout in (model.embedding_, observed):
            centered = layout.astype(np.float64) - layout.mean(axis=0, dtype=np.float64)
            selected = np.linalg.qr(centered)[0]
            np.testing.assert_allclose(selected @ selected.T, basis @ basis.T, rtol=0, atol=2e-6)
        return
    expected = recorded_layout(case["embedding_bits"], model.embedding_.shape)
    np.testing.assert_allclose(model.embedding_, expected, rtol=0, atol=6e-6)
    for epoch, actual, words in zip(model.n_epochs, model.embedding_list_, case["history_bits"], strict=True):
        expected = recorded_layout(words, actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-6 if epoch == 0 else 3e-6)


@pytest.mark.parametrize("ratio,weights,message", [
    (5, "transition", "recursive_coarsening_ratio"),
    (2.0, "transition", "recursive_coarsening_ratio"),
    (3, "none", "coarse_layout_weights"),
])
def test_invalid_recursive_coarsening_options_preserve_rng(ratio, weights, message):
    rng = TracedRandomState(81)
    model = UMAP(
        init="recursive", optimizer="adam", n_neighbors=3, n_epochs=[0], n_jobs=1, random_state=rng,
        recursive_coarsening_ratio=ratio, coarse_layout_weights=weights,
    )
    with pytest.raises(ValueError, match=message):
        model.fit(make_data(REFERENCE["cases"][0]["data"]))
    assert not rng.normal_calls
    assert rng_checkpoint(rng) == {"cursor": 624, "gaussian_bits": None}
