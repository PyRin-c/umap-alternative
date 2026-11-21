"""Rank configured feature distances for the native graph and layout paths."""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.validation import check_array

from . import _native
from .distances import _CONFIGURABLE_METRICS, _metric_parameters, named_distances


_PAIRWISE_ALIASES = {
    "l1": "manhattan",
    "l2": "euclidean",
    "taxicab": "manhattan",
    "linfinity": "chebyshev",
    "linfty": "chebyshev",
    "linf": "chebyshev",
    "max": "chebyshev",
}
_PAIRWISE_METRICS = {
    "euclidean", "manhattan", "cityblock", "chebyshev", "cosine", "correlation",
    "canberra", "braycurtis", "hamming",
}

def rank_tied_dense_distances(distances, k):
    return np.argsort(np.asarray(distances, dtype=np.float32))[:k].tolist()


class FeatureSearch:
    def __init__(self, metric, metric_kwds=None):
        self.metric = metric
        self.kwds = {} if metric_kwds is None else dict(metric_kwds)
        if isinstance(metric, str):
            self.name = _CONFIGURABLE_METRICS.get(metric)
            pairwise = _PAIRWISE_ALIASES.get(metric, metric)
            self.function = pairwise if pairwise in _PAIRWISE_METRICS else named_distances.get(metric, metric)
            # Fitted named metrics keep their parameter arrays across user mutations.
            for key in ("w", "sigma", "vinv"):
                if self.kwds.get(key) is not None:
                    self.kwds[key] = np.array(self.kwds[key], dtype=np.float64, copy=True)
        else:
            self.name = None
            self.function = metric

    def __call__(self, x, y):
        if issparse(x):
            x = x.toarray().ravel()
        if issparse(y):
            y = y.toarray().ravel()
        value = self.function(x, y, **self.kwds)
        return float(value[0] if isinstance(value, tuple) else value)

    def distance_gradient(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        value = self.function(x, y, **self.kwds)
        if isinstance(value, tuple):
            return float(value[0]), np.asarray(value[1], dtype=np.float64).tolist()
        center = float(value)
        gradient = np.zeros_like(x)
        for axis in range(len(x)):
            step = np.cbrt(np.finfo(np.float64).eps) * max(1.0, abs(x[axis]))
            shifted = x.copy()
            shifted[axis] += step
            try:
                upper = self(shifted, y)
            except ValueError:
                upper = np.nan
            shifted[axis] = x[axis] - step
            try:
                lower = self(shifted, y)
            except ValueError:
                lower = np.nan
            if np.isfinite(upper) and np.isfinite(lower):
                gradient[axis] = (upper - lower) / (2 * step)
            elif np.isfinite(upper):
                gradient[axis] = (upper - center) / step
            elif np.isfinite(lower):
                gradient[axis] = (center - lower) / step
        return center, gradient.tolist()

    def rank(self, training, k, *, queries=None, cutoff=np.inf):
        n_rows, width = training.shape
        if not 1 <= k <= n_rows:
            raise ValueError("neighbor count is out of range")
        if queries is not None and queries.shape[1] != width:
            raise ValueError(f"expected {width} query features, got {queries.shape[1]}")
        if np.isnan(cutoff) or cutoff < 0:
            raise ValueError("disconnection distance must be nonnegative")
        if self.name is not None and not (issparse(training) or issparse(queries)):
            parameter, values = _metric_parameters(self.name, width, self.kwds)
            training_bytes = np.asarray(training, dtype=np.float64, order="C").tobytes()
            query_bytes = None if queries is None else np.asarray(queries, dtype=np.float64, order="C").tobytes()
            columns, distances, removed = _native._rank_configured_features(
                self.name, parameter, values, training_bytes, query_bytes, width, k, cutoff,
                None if queries is None else rank_tied_dense_distances,
            )
            return (
                np.asarray(columns, dtype=np.int32).reshape(-1, k),
                np.asarray(distances, dtype=np.float32).reshape(-1, k),
                removed,
            )
        return self._rank_chunks(training, k, queries, cutoff)

    def _rank_chunks(self, training, k, queries, cutoff):
        training_rows = queries is None
        queries = training if training_rows else queries
        columns = np.empty((queries.shape[0], k), dtype=np.int32)
        distances = np.empty_like(columns, dtype=np.float32)
        removed = start = 0
        metric = self if callable(self.function) else self.function
        kwargs = {} if metric is self else self.kwds
        for block in pairwise_distances_chunked(queries, training, metric=metric, **kwargs):
            stop = start + block.shape[0]
            diagonal = (np.arange(stop - start), np.arange(start, stop))
            if np.isnan(block).any() or np.any(block < 0):
                raise ValueError("feature distances must be nonnegative and must not be NaN")
            if training_rows:
                block[diagonal] = 0.0
            cut = (block >= cutoff) | (block > np.finfo(np.float32).max)
            removed += np.count_nonzero(cut)
            block[cut] = np.inf
            if training_rows and cutoff > 0:
                block[diagonal] = -1.0
            order = "stable" if training_rows else "quicksort"
            nearest = np.argsort(block, axis=1, kind=order)[:, :k]
            selected = np.take_along_axis(block, nearest, axis=1)
            if training_rows and cutoff > 0:
                selected[:, 0] = 0.0
            distances[start:stop] = selected
            columns[start:stop] = np.where(np.isfinite(distances[start:stop]), nearest, -1)
            start = stop
        return columns, distances, int(removed)


class ExactFeatureIndex:
    def __init__(self, training, search):
        self.training = training.copy()
        self.search = search

    def query(self, X, k=10, epsilon=0.1):
        """Return exact neighbors for the supplied feature rows."""
        queries = check_array(X, dtype=np.float32, accept_sparse="csr")
        indices, distances, _ = self.search.rank(self.training, k, queries=queries)
        return indices, distances


class ApproximateFeatureIndex:
    def __init__(self, training, k, metric, seed, query_rows=None, *, rng=None):
        self.training = training.astype(np.float32, copy=False).tobytes(order="C")
        self.shape = training.shape
        self.k = k
        self.metric = metric
        self.seed = seed
        self.query_rows = query_rows
        self._index = None
        if rng is not None:
            algorithm, words, cursor, has_gaussian, cached_gaussian = rng.get_state()
            self._rng_state = (words.tolist(), cursor)
            words, cursor = self._native_index().rng_after()
            rng.set_state(
                (algorithm, np.asarray(words, dtype=np.uint32), cursor, has_gaussian, cached_gaussian)
            )

    def __getstate__(self):
        return {**self.__dict__, "_index": None}

    def _native_index(self):
        if self._index is None:
            self._index = _native.NativeApproximateIndex(
                self.training, self.shape, self.k, self.metric, self.seed,
                rng_state=getattr(self, "_rng_state", None),
            )
        return self._index

    def training_ranks(self):
        return self._as_arrays(*self._native_index().training_ranks(), self.k)

    def training_graph(self, graph_options):
        return self._native_index().training_graph(graph_options)

    def query(self, X, k):
        queries = check_array(X, dtype=np.float32, order="C")
        if queries.shape[1] != self.shape[1] or not 1 <= k <= self.k:
            raise ValueError("approximate query shape or neighbor count differs from the fitted index")
        indices, distances = self._as_arrays(*self._native_index().query(queries.tobytes(), len(queries), k), k)
        query_rows = getattr(self, "query_rows", None)
        if query_rows is not None:
            indices = np.where(indices >= 0, query_rows[indices], -1)
        return indices, distances

    @staticmethod
    def _as_arrays(indices, distances, width):
        return (
            np.asarray(indices, dtype=np.int32).reshape(-1, width),
            np.asarray(distances, dtype=np.float32).reshape(-1, width),
        )
