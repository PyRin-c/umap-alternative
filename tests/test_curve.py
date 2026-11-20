import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy
from scipy.optimize import OptimizeWarning

from umap_alternative._curve import _resolve_curve_params


RUNTIME = {
    "platform": platform.system(),
    "machine": platform.machine(),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
}
REFERENCE_PLATFORM = "ubuntu" if RUNTIME["platform"] == "Linux" else "windows"
REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference/fixtures"
    / f"umap-learn-0.6.0-curve-{REFERENCE_PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "sample",
    REFERENCE["samples"],
    ids=[f"{sample['spread']}-{sample['min_dist']}" for sample in REFERENCE["samples"]],
)
def test_curve_matches_public_reference(sample):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _resolve_curve_params(sample["spread"], sample["min_dist"])

    expected = np.array(sample["bits"], dtype=np.uint64)
    if all(REFERENCE["environment"][key] == value for key, value in RUNTIME.items()):
        actual_bits = np.asarray(actual, dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(actual_bits, expected)
        assert [item.category.__name__ for item in caught] == sample["warnings"]
    else:
        # NumPy and SciPy builds can change the optimizer's rounding and iterations.
        np.testing.assert_allclose(actual, expected.view(np.float64), rtol=1e-6, atol=0)


def test_curve_defaults_and_real_numpy_scalars():
    assert _resolve_curve_params() == _resolve_curve_params(1.0, 0.1)
    assert _resolve_curve_params(np.int64(1), np.float32(0.125)) == (
        _resolve_curve_params(1.0, 0.125)
    )


@pytest.mark.parametrize("a,b", [(7.0, None), (None, 7.0)])
def test_curve_refits_both_coefficients_when_either_is_missing(a, b):
    assert _resolve_curve_params(0.5, 0.2, a, b) == _resolve_curve_params(0.5, 0.2)


def test_curve_uses_complete_explicit_pair_without_fitting():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_curve_params(1e-10, 0.0, a=0.75, b=1.25) == (0.75, 1.25)


def test_curve_propagates_optimizer_warning():
    with pytest.warns(OptimizeWarning):
        assert _resolve_curve_params(1e-10, 0.0) == (1.0, 1.0)


@pytest.mark.parametrize(
    "min_dist", [math.nextafter(0.0, -math.inf), math.nextafter(1.0, math.inf)]
)
def test_curve_rejects_first_float_outside_min_dist_bounds(min_dist):
    with pytest.raises(ValueError, match="min_dist must be between 0 and spread"):
        _resolve_curve_params(1.0, min_dist)


@pytest.mark.parametrize("spread", [0.0, -1.0])
def test_curve_rejects_nonpositive_spread(spread):
    with pytest.raises(ValueError, match="spread must be positive"):
        _resolve_curve_params(spread, 0.0)


def test_curve_rejects_grid_endpoint_overflow():
    with pytest.raises(ValueError, match=r"3 \* spread must be finite"):
        _resolve_curve_params(sys.float_info.max, 0.0)


@pytest.mark.parametrize(
    "overrides,parameter",
    [
        ({"spread": math.inf}, "spread"),
        ({"spread": 10**400}, "spread"),
        ({"min_dist": math.nan}, "min_dist"),
        ({"a": math.nan, "b": 1.0}, "a"),
        ({"a": 1.0, "b": -math.inf}, "b"),
    ],
)
def test_curve_rejects_nonfinite_parameters(overrides, parameter):
    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        _resolve_curve_params(**overrides)


@pytest.mark.parametrize(
    "overrides,parameter",
    [
        ({"spread": "1.0"}, "spread"),
        ({"min_dist": np.array(0.1)}, "min_dist"),
        ({"min_dist": True}, "min_dist"),
        ({"a": 1.0 + 0.0j, "b": 1.0}, "a"),
        ({"a": 1.0, "b": np.bool_(True)}, "b"),
    ],
)
def test_curve_rejects_nonreal_scalars(overrides, parameter):
    with pytest.raises(TypeError, match=f"{parameter} must be a real scalar"):
        _resolve_curve_params(**overrides)


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, -1.0)])
def test_curve_rejects_nonpositive_explicit_coefficients(a, b):
    with pytest.raises(ValueError, match="a and b must be positive"):
        _resolve_curve_params(a=a, b=b)
