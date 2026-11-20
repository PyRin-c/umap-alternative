"""Resolve the low-dimensional embedding curve coefficients."""

from math import isfinite
from numbers import Real

import numpy as np
from scipy.optimize import curve_fit


def _finite_float(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite in float64") from error
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite in float64")
    return converted


def _membership(distance: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + a * distance ** (2.0 * b))


def find_ab_params(spread, min_dist):
    """Fit the embedding curve coefficients at float64 precision."""
    distance = np.linspace(0.0, 3.0 * spread, 300, dtype=np.float64)
    target = np.exp(-np.maximum(distance - min_dist, 0.0) / spread)
    (a, b), _ = curve_fit(_membership, distance, target)
    return a, b


def _resolve_curve_params(
    spread: float = 1.0,
    min_dist: float = 0.1,
    a: float | None = None,
    b: float | None = None,
) -> tuple[float, float]:
    """Resolve both coefficients at float64 precision.

    Require positive spread and 0 <= min_dist <= spread. A complete explicit
    pair must be finite and positive. If either coefficient is None, fit both
    with SciPy and propagate its warnings and errors to the caller.
    """
    spread = _finite_float(spread, "spread")
    min_dist = _finite_float(min_dist, "min_dist")
    if spread <= 0.0:
        raise ValueError("spread must be positive")
    if not 0.0 <= min_dist <= spread:
        raise ValueError("min_dist must be between 0 and spread, inclusive")

    if a is not None and b is not None:
        a = _finite_float(a, "a")
        b = _finite_float(b, "b")
        if a <= 0.0 or b <= 0.0:
            raise ValueError("a and b must be positive")
        return a, b

    grid_end = 3.0 * spread
    if not isfinite(grid_end):
        raise ValueError("3 * spread must be finite in float64 to fit the curve")
    a, b = find_ab_params(spread, min_dist)
    return float(a), float(b)
