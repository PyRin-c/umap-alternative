"""Public seeded modern fit and fit_transform replay."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from umap_alternative import UMAP


PLATFORM = "windows" if sys.platform == "win32" else "ubuntu"
FIXTURE = (
    Path(__file__).parents[1] / f"reference/fixtures/umap-learn-0.6.0-modern-fit-{PLATFORM}-x86_64.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


class TiedProjectionRandomState(np.random.RandomState):
    first = True

    def randn(self, width):
        direction = super().randn(width)
        if self.first:
            direction[:] = np.arange(width) % 4
            self.first = False
        return direction


def assert_bits(array, expected):
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array.view(np.uint32).ravel(), expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_seeded_modern_fit_matches_public_wheel(case, method):
    rng_type = TiedProjectionRandomState if case.get("tie") else np.random.RandomState
    rng = rng_type(case["seed"])
    if case.get("warm_gaussian"):
        rng.standard_normal()
    options = case["options"].copy()
    if not isinstance(options["init"], str):
        options["init"] = np.asarray(options["init"], dtype=np.float32)
    data = np.asarray(case["data"], dtype=np.float32)
    model = UMAP(**options, random_state=rng, n_jobs=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="using precomputed metric", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*disconnected.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
        result = getattr(model, method)(data)

    assert result is (model if method == "fit" else model.embedding_)
    assert model.embedding_.shape == (len(data), options.get("n_components", 2))
    np.testing.assert_array_equal(model.graph_.indptr, case["indptr"])
    np.testing.assert_array_equal(model.graph_.indices, case["indices"])
    assert_bits(model.graph_.data, case["weights_bits"])
    assert_bits(model.embedding_, case["embedding_bits"])
    if case["history_bits"] is None:
        assert not hasattr(model, "embedding_list_")
    else:
        for observed, expected in zip(model.embedding_list_, case["history_bits"], strict=True):
            assert not np.shares_memory(observed, model.embedding_)
            assert_bits(observed, expected)

    _, _, cursor, has_gaussian, gaussian = rng.get_state()
    gaussian_bits = int(np.float64(gaussian).view(np.uint64)) if has_gaussian else None
    assert cursor == case["rng"]["cursor"]
    assert gaussian_bits == case["rng"]["gaussian_bits"]
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]


def test_modern_fit_matches_public_wheel_with_integer_seed():
    case = next(case for case in CASES if case["name"] == "default-modern-adam")
    data = np.asarray(case["data"], dtype=np.float32)
    model = UMAP(**case["options"], random_state=case["seed"], n_jobs=1)
    result = model.fit_transform(data)
    assert_bits(result, case["embedding_bits"])
    assert model.transform(data.copy()) is result
