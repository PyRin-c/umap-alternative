"""Utilities for UMAP inputs."""

from operator import index
import warnings

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_is_fitted

from . import _native


def fast_knn_indices(X, n_neighbors):
    """Return the first neighbor indices in ascending distance order."""
    distances = np.asarray(X)
    k = index(n_neighbors)
    if distances.ndim != 2 or k < 0:
        raise ValueError("expected a distance matrix and a nonnegative neighbor count")
    if distances.shape[0] == 0:
        return np.empty((0, k), dtype=np.int32)
    if k > distances.shape[1]:
        raise SystemError("neighbor count exceeds the distance row width")
    return np.argsort(distances, axis=1, kind="stable")[:, :k].astype(np.int32)


def submatrix(dmat, indices_col, n_neighbors):
    """Gather the requested columns in each row of a matrix."""
    matrix = np.asarray(dmat)
    columns = np.asarray(indices_col)
    k = index(n_neighbors)
    if (
        matrix.ndim != 2 or columns.ndim != 2 or matrix.shape[0] != columns.shape[0]
        or not 0 <= k <= columns.shape[1]
    ):
        raise ValueError("matrix, indices and neighbor count have incompatible shapes")
    return np.take_along_axis(matrix, columns[:, :k], axis=1)


def average_nn_distance(dist_matrix):
    """Return the float64 mean of the distances in each row."""
    distances = np.asarray(dist_matrix)
    if distances.ndim != 2:
        raise ValueError("dist_matrix must be two-dimensional")
    if distances.size == 0:
        return np.empty(0, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        averages = np.mean(distances, axis=1, dtype=np.float64)
    if np.isnan(averages).any():
        warnings.warn(
            "Embedding contains disconnected vertices which will be ignored."
            "Use umap.utils.disconnected_vertices() to identify them.",
            UserWarning,
            stacklevel=2,
        )
    return averages


def disconnected_vertices(model):
    """Return the boolean mask of fitted graph rows whose weight sum is zero."""
    from ._umap import UMAP

    if not isinstance(model, UMAP):
        raise TypeError("model must be a fitted UMAP estimator")
    check_is_fitted(model)
    disconnected = np.asarray(model.graph_.sum(axis=1)).ravel() == 0
    if model.unique:
        inverse = getattr(model, "_fit_inverse_rows", None)
        if inverse is None:
            unique_fit = getattr(model, "_inverse_unique_fit", None)
            if unique_fit is not None:
                inverse = unique_fit[1]
            elif getattr(model, "_graph_transform_fit", None) is not None:
                training = model._graph_transform_fit[0]
                if training.shape[0] > disconnected.size:
                    if issparse(training):
                        inverse, = csr_unique(training, return_index=False, return_counts=False)
                    else:
                        _, inverse = np.unique(training, axis=0, return_inverse=True)
        if inverse is not None:
            disconnected = disconnected[inverse]
    return disconnected


def csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True):
    """Return the optional indices, inverse map, and counts of distinct CSR rows.

    Row keys consist of the stored column indices followed by the stored
    values. Repeated indices are coalesced, and explicit zeros remain entries.
    """
    csr = csr_matrix(matrix.tocsr(), copy=True)
    csr.check_format(full_check=True)
    csr.sum_duplicates()
    flags = (return_index, return_inverse, return_counts)
    if not any(flags):
        return np.empty(0, dtype=object)
    if csr.dtype == np.float32 and np.isfinite(csr.data).all():
        groups = _native._unique_csr_rows(
            (csr.shape, csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist())
        )
        return tuple(np.asarray(group, dtype=np.intp) for group, flag in zip(groups, flags) if flag)
    # Preserve integers and wider float values when building the comparison keys.
    rows = csr.tolil()
    keys = np.empty(csr.shape[0], dtype=object)
    keys[:] = [tuple(columns) + tuple(values) for columns, values in zip(rows.rows, rows.data)]
    return np.unique(
        keys, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts
    )[1:]


__all__ = ["average_nn_distance", "csr_unique", "disconnected_vertices", "fast_knn_indices", "submatrix"]
