"""Trustworthiness curves for UMAP embeddings and separately ranked inputs."""

from operator import index

import numpy as np
from sklearn.neighbors import KDTree

from . import _native


_SOURCE_METRICS = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "minkowski": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "chebyshev": "chebyshev",
}


def _max_k(value):
    k = index(value)
    if k < -1:
        raise ValueError("max_k must be at least -1")
    return k


def _indices(values, name):
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a two-dimensional integer array")
    return values


def _source_matrix(source):
    if not isinstance(source, np.ndarray) or source.ndim != 2:
        raise TypeError("source must be a two-dimensional NumPy array")
    return source


def trustworthiness_vector_bulk(indices_source, indices_embedded, max_k):
    """Score embedded neighbors against the ordered source sample indices.

    Source rows include the entire ranking, with index zero as its first
    position. Embedded rows supply at least ``max_k`` ordered neighbors.
    The result has a zero placeholder followed by the scores for sizes 1
    through ``max_k``.
    """
    source = _indices(indices_source, "indices_source")
    embedded = _indices(indices_embedded, "indices_embedded")
    k = _max_k(max_k)
    if len(source) != len(embedded) or k > embedded.shape[1]:
        raise ValueError("source and embedded rankings have incompatible shapes")
    if k == -1:
        return np.empty(0, dtype=np.float64)
    scores = _native._trustworthiness_ranked(source.tolist(), embedded[:, :k].tolist(), k)
    return np.asarray(scores, dtype=np.float64)


def make_trustworthiness_calculator(metric):
    """Build a scorer using a callable source distance and ranked embedded neighbors."""
    def trustworthiness(source, indices_embedded, max_k):
        source = _source_matrix(source)
        embedded = _indices(indices_embedded, "indices_embedded")
        k = _max_k(max_k)
        if len(source) != len(embedded) or k > embedded.shape[1]:
            raise ValueError("source and embedded rankings have incompatible shapes")
        if k == -1:
            return np.empty(0, dtype=np.float64)
        if not callable(metric):
            raise TypeError("metric must be a callable source distance")
        rows = ([metric(point, other) for other in source] for point in source)
        scores = _native._trustworthiness_distances(rows, embedded[:, :k].tolist(), k)
        return np.asarray(scores, dtype=np.float64)

    return trustworthiness


def trustworthiness_vector(source, embedding, max_k, metric="euclidean"):
    """Score how well the embedding preserves the source neighborhoods.

    Both spaces use ``metric`` to rank neighbors. The vector starts with a
    score of one for zero neighbors, then contains the scores for each size
    through ``max_k``. Different feature counts are supported across the two spaces.
    """
    k = _max_k(max_k)
    tree = KDTree(embedding, metric=metric)
    source_metric = _SOURCE_METRICS[metric]
    source = _source_matrix(source)
    if len(source) != len(embedding):
        raise ValueError("source and embedding must have the same number of samples")
    if k == -1:
        return np.empty(0, dtype=np.float64)
    embedded = tree.query(embedding, k=k + 1, return_distance=False)[:, 1:]
    if source.dtype.kind in "ib":
        dtype = "i64"
    elif source.dtype.kind == "u":
        dtype = "u64"
    elif source.dtype.kind == "f" and source.dtype.itemsize in (4, 8):
        dtype = f"f{source.dtype.itemsize * 8}"
    else:
        raise TypeError("unsupported source dtype for trustworthiness")
    scores = _native._trustworthiness_dense(
        source.ravel().tolist(), source.shape[1], embedded.tolist(), k, source_metric, dtype
    )
    return np.asarray(scores, dtype=np.float64)


__all__ = [
    "make_trustworthiness_calculator",
    "trustworthiness_vector",
    "trustworthiness_vector_bulk",
]
