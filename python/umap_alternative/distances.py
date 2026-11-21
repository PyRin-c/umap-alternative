"""Named distances used by the public graph helpers."""

import numpy as np
from scipy import stats

from . import _native


DISCRETE_METRICS = ("categorical", "hierarchical_categorical", "ordinal", "count", "string", "myers")


def categorical_distance(x, y):
    """Return zero for equal labels and one for different labels."""
    return 0.0 if x == y else 1.0


def hierarchical_categorical_distance(x, y, cat_hierarchy=[{}]):
    """Return the fraction of hierarchy levels before the first shared category."""
    depth = len(cat_hierarchy)
    for level, categories in enumerate(cat_hierarchy):
        if categories[x] == categories[y]:
            return level / depth
    return 1.0


def ordinal_distance(x, y, support_size=1.0):
    """Measure the absolute difference between ordinal values."""
    left, right = np.asarray(x), np.asarray(y)
    if left.ndim or right.ndim:
        raise TypeError("ordinal inputs must be scalar values")
    if support_size == 0:
        raise ZeroDivisionError("ordinal support is zero")
    if left.dtype.kind in "iub" and right.dtype.kind in "iub":
        difference = int(left) - int(right)
        if left.dtype.kind == right.dtype.kind == "u":
            difference &= (1 << 64) - 1
        else:
            difference = (difference + (1 << 63)) % (1 << 64) - (1 << 63)
        return abs(difference) / float(support_size)
    dtype = np.result_type(left.dtype, right.dtype)
    difference = abs(left.astype(dtype)[()] - right.astype(dtype)[()])
    if dtype == np.float32 and np.asarray(support_size).dtype == np.float32:
        return difference / np.float32(support_size)
    return float(difference) / float(support_size)


def count_distance(x, y, poisson_lambda=1.0, normalisation=1.0):
    """Return the normalized signed Poisson distance between two counts."""
    if np.ndim(x) or np.ndim(y):
        raise TypeError("count inputs must be scalar values")
    if normalisation == 0:
        raise ZeroDivisionError("count normalisation is zero")
    if not np.isfinite(x) or not np.isfinite(y):
        return 0.0 / float(normalisation)
    scale = float(poisson_lambda), np.asarray(poisson_lambda).dtype == np.float32
    return _native._discrete_count_distance(int(x), int(y), scale) / float(normalisation)


def _edit_distance(x, y, normalisation, max_distance, cap):
    if not isinstance(x, str) or not isinstance(y, str):
        raise TypeError("edit distance inputs must be strings")
    if normalisation == 0:
        raise ZeroDivisionError("edit distance normalisation is zero")
    codes_x, codes_y = list(map(ord, x)), list(map(ord, y))
    return _native._discrete_edit_distance(codes_x, codes_y, max_distance, cap) / float(normalisation)


def levenshtein(x, y, normalisation=1.0, max_distance=20):
    """Measure Unicode edit distance using a row cutoff."""
    return _edit_distance(x, y, normalisation, max_distance, False)


def levenshtein_myers_ascii(x, y, normalisation=1.0, max_distance=20):
    """Measure short edit distances with a capped word-sized bit vector."""
    return _edit_distance(x, y, normalisation, max_distance, True)


def get_discrete_params(data, metric):
    """Estimate the metric parameters used for ordinal, count and string targets."""
    if metric == "ordinal":
        return {"support_size": (np.max(data) - np.min(data)) / 2.0}
    if metric == "count":
        low, high, mean = stats.tmin(data), stats.tmax(data), stats.tmean(data)
        return {"poisson_lambda": mean, "normalisation": count_distance(low, high, mean) / 2.0}
    if metric == "string":
        bound = stats.tmax([len(value) for value in data]) / 3.0
        return {"normalisation": bound, "max_dist": bound}
    return {}


def _magnitude(x, y, name):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise TypeError("distance inputs must be one-dimensional vectors")
    if x.dtype.kind in "iub" and y.dtype.kind in "iub":
        dtype = np.dtype(np.int64 if "i" in (x.dtype.kind, y.dtype.kind) else np.uint64)
    else:
        dtype = np.result_type(x.dtype, y.dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("distance vectors must have real numeric dtypes")
    return _native._discrete_magnitude_distance(
        x.astype(dtype, copy=False).tolist(), y.astype(dtype, copy=False).tolist(),
        name, f"{dtype.kind}{8 * dtype.itemsize}",
    )


def euclidean(x, y):
    return _magnitude(x, y, "euclidean")


def manhattan(x, y):
    return _magnitude(x, y, "manhattan")


def chebyshev(x, y):
    return _magnitude(x, y, "chebyshev")


def hamming(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape:
        raise TypeError("Hamming inputs must be equally sized vectors")
    return np.count_nonzero(x != y) / len(x)


named_distances = {
    "categorical": categorical_distance, "hierarchical_categorical": hierarchical_categorical_distance,
    "ordinal": ordinal_distance, "count": count_distance, "string": levenshtein,
    "myers": levenshtein_myers_ascii,
    "euclidean": euclidean, "l2": euclidean,
    "manhattan": manhattan, "taxicab": manhattan, "l1": manhattan,
    "chebyshev": chebyshev, "linfinity": chebyshev, "linfty": chebyshev, "linf": chebyshev,
    "hamming": hamming,
}

__all__ = [
    "DISCRETE_METRICS", "categorical_distance", "chebyshev", "count_distance", "euclidean",
    "get_discrete_params", "hamming", "hierarchical_categorical_distance", "levenshtein",
    "levenshtein_myers_ascii", "manhattan", "named_distances", "ordinal_distance",
]
