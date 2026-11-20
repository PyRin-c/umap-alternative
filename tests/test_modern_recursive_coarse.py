"""Public-wheel recursive optimizer traces and fit checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest

from test_modern_recursive_fit import PLATFORM, TracedRandomState, fit_case, recorded_layout, rng_checkpoint


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-coarse-{PLATFORM}-x86_64.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
RESIDUALS = {
    "ratio-three-stars-positive-only-eleven",
}
COORDINATE_BOUNDS = {
    "four-level-budget-3": 3e-5,
    "four-level-positive-only": 2e-5,
    "ratio-three-stars-budget-None": 2e-5,
    "two-level-strong_to_one": 2e-5,
    "three-level-budget-4": 2e-5,
    "three-level-budget-6": 1e-4,
    "three-level-coarse-eleven": 4e-5,
}


class CoarseRandomState(TracedRandomState):
    def __init__(self, seed):
        super().__init__(seed)
        self.calls = []

    def normal(self, *args, **kwds):
        values = super().normal(*args, **kwds)
        self.calls.append(self.normal_calls[-1])
        return values

    def randint(self, *args, **kwds):
        start = rng_checkpoint(self)
        values = super().randint(*args, **kwds)
        self.calls.append(["randint", list(values.shape), start, rng_checkpoint(self)])
        return values

    def shuffle(self, values):
        start = rng_checkpoint(self)
        super().shuffle(values)
        self.calls.append(["shuffle", list(values.shape), start, rng_checkpoint(self)])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_coarse_rng_trace(case, method):
    rng = CoarseRandomState(case["seed"])
    fit_case(case, method, rng)
    expected = case["rng_calls"]
    # The last triplet seeds the outer fit, whose native full-range path
    # restores RandomState directly. Compare all calls through coarsening.
    outer_start = max(index for index, call in enumerate(expected) if call[0] == "randint")
    assert rng.calls[:outer_start] == expected[:outer_start]


@pytest.mark.parametrize("case", [
    pytest.param(case, id=case["name"], marks=pytest.mark.xfail(
        raises=AssertionError, strict=True, reason="recursive coarse coordinate residual",
    )) if case["name"] in RESIDUALS else pytest.param(case, id=case["name"])
    for case in CASES
])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_coarse_coordinates(case, method):
    model, _ = fit_case(case, method)
    bound = COORDINATE_BOUNDS.get(case["name"], 5e-6)
    expected = recorded_layout(case["embedding_bits"], model.embedding_.shape)
    np.testing.assert_allclose(model.embedding_, expected, rtol=0, atol=bound)
    for actual, bits in zip(model.embedding_list_, case["history_bits"], strict=True):
        expected = recorded_layout(bits, actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)
