"""Public-wheel recursive optimizer traces and fit checkpoints."""

import json
from pathlib import Path

import pytest

from test_modern_recursive_fit import PLATFORM, TracedRandomState, check_layout_bits, fit_case, rng_checkpoint


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-coarse-{PLATFORM}-x86_64.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


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


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_coarse_coordinates(case, method):
    model, _ = fit_case(case, method)
    check_layout_bits(model.embedding_, case["embedding_bits"])
    for actual, bits in zip(model.embedding_list_, case["history_bits"], strict=True):
        check_layout_bits(actual, bits)
