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


def _feature_pair(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise TypeError("distance inputs must be one-dimensional vectors")
    if y.size < x.size:
        raise ValueError("the second distance vector is shorter than the first")
    for row in (x, y):
        if row.dtype.kind not in "biuf":
            raise TypeError("distance vectors must have real numeric dtypes")
        if row.dtype.kind == "f" and row.dtype.itemsize not in (4, 8):
            raise NotImplementedError(row.dtype.name)
    return x, y[:x.size]


def _metric_row(row):
    dtype = row.dtype.newbyteorder("=")
    if dtype.kind != "f":
        dtype = np.dtype(np.uint64 if dtype.kind == "u" else np.int64)
    return row.astype(dtype, copy=False).tobytes(), f"{dtype.kind}{8 * dtype.itemsize}"


def _numeric_distance(x, y, name):
    x, y = _feature_pair(x, y)
    if name == "correlation" and not x.size:
        raise ZeroDivisionError("correlation of empty vectors")
    if name in ("canberra", "braycurtis"):
        dtype = np.result_type(x.dtype, y.dtype)
        if "f" in (x.dtype.kind, y.dtype.kind):
            x, y = x.astype(dtype, copy=False), y.astype(dtype, copy=False)
    left, left_dtype = _metric_row(x)
    right, right_dtype = _metric_row(y)
    return _native._numeric_row_distance(left, right, name, left_dtype, right_dtype)


def canberra(x, y):
    """Sum the absolute differences divided by the pairwise absolute sums."""
    return _numeric_distance(x, y, "canberra")


def bray_curtis(x, y):
    """Divide the sum of absolute differences by the sum of absolute pair sums."""
    return _numeric_distance(x, y, "braycurtis")


def cosine(x, y):
    """Return one minus the cosine similarity of the two feature vectors."""
    return _numeric_distance(x, y, "cosine")


def correlation(x, y):
    """Return cosine distance after centering each feature vector."""
    return _numeric_distance(x, y, "correlation")


def hamming(x, y):
    """Return the fraction of coordinates with different numeric values."""
    x, y = _feature_pair(x, y)
    if not x.size:
        raise ZeroDivisionError("Hamming distance of empty vectors")
    if {x.dtype.kind, y.dtype.kind} == {"i", "u"}:
        x, y = x.astype(np.float64), y.astype(np.float64)
    return int(np.count_nonzero(x != y)) / len(x)


def _support_distance(x, y, name):
    x, y = _feature_pair(x, y)
    if not x.size and name in ("matching", "rogerstanimoto", "sokalmichener"):
        raise ZeroDivisionError("support distance denominator is zero")
    return _native._support_distance((x != 0).tobytes(), (y != 0).tobytes(), name)


def matching(x, y):
    return _support_distance(x, y, "matching")


def jaccard(x, y):
    return _support_distance(x, y, "jaccard")


def dice(x, y):
    return _support_distance(x, y, "dice")


def kulsinski(x, y):
    return _support_distance(x, y, "kulsinski")


def rogers_tanimoto(x, y):
    return _support_distance(x, y, "rogerstanimoto")


def russellrao(x, y):
    return _support_distance(x, y, "russellrao")


def sokal_sneath(x, y):
    return _support_distance(x, y, "sokalsneath")


def sokal_michener(x, y):
    return _support_distance(x, y, "sokalmichener")


def yule(x, y):
    return _support_distance(x, y, "yule")


_CONFIGURABLE_METRICS = {
    "minkowski": "minkowski",
    "poincare": "poincare",
    "seuclidean": "standardised_euclidean",
    "standardised_euclidean": "standardised_euclidean",
    "wminkowski": "weighted_minkowski",
    "weighted_minkowski": "weighted_minkowski",
    "mahalanobis": "mahalanobis",
    "hellinger": "hellinger",
    "softmax_hellinger": "softmax_hellinger",
    "haversine": "haversine",
    "symmetric_kl": "symmetric_kl",
}


def _metric_parameters(name, width, kwargs):
    kwargs = dict(kwargs)
    parameter, values = 0.0, []
    if name in ("minkowski", "weighted_minkowski"):
        parameter = float(kwargs.pop("p", 2))
    if name == "symmetric_kl":
        parameter = float(kwargs.pop("z", 1e-11))
    if name in ("weighted_minkowski", "standardised_euclidean", "mahalanobis"):
        key = {"weighted_minkowski": "w", "standardised_euclidean": "sigma", "mahalanobis": "vinv"}[name]
        values = kwargs.pop(key, None)
        if values is None:
            values = np.eye(width) if name == "mahalanobis" else np.ones(width)
        shape = (width, width) if name == "mahalanobis" else (width,)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        values = values.ravel().tolist()
    if kwargs:
        raise TypeError(f"{name} got an unexpected metric parameter {next(iter(kwargs))!r}")
    return parameter, values


def _configured_distance(x, y, name, kwargs):
    x, y = _feature_pair(x, y)
    parameter, values = _metric_parameters(name, len(x), kwargs)
    return _native._feature_metric_distance(name, parameter, values, x.tolist(), y.tolist())


def minkowski(x, y, p=2):
    return _configured_distance(x, y, "minkowski", {"p": p})


def weighted_minkowski(x, y, w=None, p=2):
    return _configured_distance(x, y, "weighted_minkowski", {"w": w, "p": p})


def standardised_euclidean(x, y, sigma=None):
    return _configured_distance(x, y, "standardised_euclidean", {"sigma": sigma})


def mahalanobis(x, y, vinv=None):
    return _configured_distance(x, y, "mahalanobis", {"vinv": vinv})


def haversine(x, y):
    return _configured_distance(x, y, "haversine", {})


def poincare(u, v):
    return _configured_distance(u, v, "poincare", {})


def hellinger(x, y):
    return _configured_distance(x, y, "hellinger", {})


def softmax_hellinger(x, y):
    return _configured_distance(x, y, "softmax_hellinger", {})


def symmetric_kl(x, y, z=1e-11):
    return _configured_distance(x, y, "symmetric_kl", {"z": z})


named_distances = {
    "euclidean": euclidean, "l2": euclidean,
    "manhattan": manhattan, "taxicab": manhattan, "l1": manhattan,
    "chebyshev": chebyshev, "linfinity": chebyshev, "linfty": chebyshev, "linf": chebyshev,
    "minkowski": minkowski, "poincare": poincare,
    "seuclidean": standardised_euclidean, "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski, "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis, "canberra": canberra, "cosine": cosine, "correlation": correlation,
    "hellinger": hellinger, "softmax_hellinger": softmax_hellinger, "haversine": haversine,
    "braycurtis": bray_curtis, "symmetric_kl": symmetric_kl,
    "hamming": hamming, "jaccard": jaccard, "dice": dice, "matching": matching,
    "kulsinski": kulsinski, "rogerstanimoto": rogers_tanimoto, "russellrao": russellrao,
    "sokalsneath": sokal_sneath, "sokalmichener": sokal_michener, "yule": yule,
    "categorical": categorical_distance, "ordinal": ordinal_distance,
    "hierarchical_categorical": hierarchical_categorical_distance, "count": count_distance,
    "string": levenshtein, "myers": levenshtein_myers_ascii,
}

__all__ = [
    "DISCRETE_METRICS", "bray_curtis", "canberra", "categorical_distance", "chebyshev",
    "correlation", "cosine", "count_distance", "dice", "euclidean", "get_discrete_params",
    "hamming", "haversine", "hellinger", "hierarchical_categorical_distance", "jaccard", "kulsinski",
    "levenshtein", "levenshtein_myers_ascii", "mahalanobis", "manhattan", "matching", "minkowski",
    "named_distances", "ordinal_distance", "poincare", "rogers_tanimoto", "russellrao",
    "softmax_hellinger", "sokal_michener", "sokal_sneath", "standardised_euclidean", "symmetric_kl",
    "weighted_minkowski", "yule",
]
