"""Low-level UMAP graph operations and the estimator."""

from operator import index

import numpy as np
from scipy.sparse import coo_matrix, diags
from sklearn.preprocessing import normalize

from . import _native
from ._curve import find_ab_params
from ._umap import UMAP
from .sparse import general_sset_intersection, general_sset_union
from .utils import fast_knn_indices, submatrix


def _distance_buffer(values):
    return values.ravel().tolist(), values.dtype == np.dtype(np.float32)


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Return the float32 scales and offsets for the ranked distance rows."""
    distances = np.asarray(distances)
    if distances.ndim != 2:
        raise ValueError("distances must be two-dimensional")
    sigmas, rhos = _native._smooth_knn_distances(
        _distance_buffer(distances), distances.shape,
        (k, max(index(n_iter), 0), local_connectivity, bandwidth),
    )
    return np.asarray(sigmas, dtype=np.float32), np.asarray(rhos, dtype=np.float32)


def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, return_dists=False, bipartite=False,
):
    """Return flat directed row, column, weight, and optional distance arrays."""
    indices = np.asarray(knn_indices)
    distances = np.asarray(knn_dists)
    sigmas = np.asarray(sigmas)
    rhos = np.asarray(rhos)
    if indices.ndim != 2 or indices.shape != distances.shape or sigmas.ndim != 1 or rhos.ndim != 1:
        raise ValueError("near-neighbor and scale arrays have incompatible shapes")
    if any(values.dtype != np.float32 for values in (distances, sigmas, rhos)):
        raise TypeError("distances, sigmas and rhos must be float32")
    rows, columns, weights, distances = _native._compute_knn_memberships(
        indices.ravel().tolist(), distances.ravel().tolist(), sigmas.tolist(),
        rhos.tolist(), indices.shape, return_dists, bipartite,
    )
    return (
        np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
        None if distances is None else np.asarray(distances, dtype=np.float32),
    )


def nearest_neighbors(
    X, n_neighbors, metric, metric_kwds, angular, random_state, low_memory=True,
    use_pynndescent=True, n_jobs=-1, compatibility_layout=True, verbose=False,
    nearest_neighbor_backend="auto", graph_only=False,
):
    """Return neighbor arrays for a precomputed dense distance matrix."""
    if metric != "precomputed":
        raise NotImplementedError("nearest_neighbors currently requires precomputed distances")
    matrix = np.asarray(X)
    indices = fast_knn_indices(matrix, n_neighbors)
    distances = submatrix(matrix, indices, n_neighbors)
    indices[distances == np.inf] = -1
    return indices, distances, None


def fuzzy_simplicial_set(
    X, n_neighbors, random_state, metric, metric_kwds={}, knn_indices=None, knn_dists=None,
    angular=False, set_op_mix_ratio=1.0, local_connectivity=1.0, apply_set_operations=True,
    verbose=False, return_dists=None, compatibility_layout=True,
):
    """Build a fuzzy graph and its scales from ranked neighbors."""
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state,
            compatibility_layout=compatibility_layout, verbose=verbose,
        )
    knn_dists = np.asarray(knn_dists, dtype=np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors, local_connectivity=local_connectivity)
    rows, columns, weights, distances = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=bool(return_dists)
    )
    shape = (X.shape[0], X.shape[0])
    graph = coo_matrix((weights, (rows, columns)), shape=shape)
    graph.eliminate_zeros()
    if apply_set_operations:
        transpose = graph.T
        intersection = graph.multiply(transpose)
        graph = set_op_mix_ratio * (graph + transpose - intersection) + (1.0 - set_op_mix_ratio) * intersection
        graph.eliminate_zeros()
    if return_dists is None:
        return graph, sigmas, rhos
    distance_graph = None
    if distances is not None:
        distance_graph = coo_matrix((distances, (rows, columns)), shape=shape)
        distance_graph = distance_graph.maximum(distance_graph.T).todok()
    return graph, sigmas, rhos, distance_graph


def make_epochs_per_sample(weights, n_epochs):
    """Convert weights to the positive sampling interval for each edge."""
    weights = np.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    result = np.full(weights.shape, -1.0, dtype=np.float64)
    if weights.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = n_epochs * (weights / weights.max())
        active = samples > 0
        result[active] = n_epochs / samples[active].astype(np.float64)
    return result


def general_simplicial_set_union(simplicial_set1, simplicial_set2):
    """Return the directed fuzzy union on the support of the summed graphs."""
    result = (simplicial_set1 + simplicial_set2).tocoo()
    left, right = simplicial_set1.tocsr(), simplicial_set2.tocsr()
    general_sset_union(
        left.indptr, left.indices, left.data, right.indptr, right.indices, right.data,
        result.row, result.col, result.data,
    )
    return result


def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight=0.5, right_complement=False):
    """Return the directed weighted intersection, optionally with the right complement.

    Taking the complement uses the left graph's COO storage and updates the
    left graph when the conversion shares its data buffer.
    """
    result = simplicial_set1.tocoo() if right_complement else (simplicial_set1 + simplicial_set2).tocoo()
    left, right = simplicial_set1.tocsr(), simplicial_set2.tocsr()
    general_sset_intersection(
        left.indptr, left.indices, left.data, right.indptr, right.indices, right.data,
        result.row, result.col, result.data, right_complement, weight,
    )
    return result


def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Normalize row maxima and take a fuzzy union with the transpose."""
    result = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        result.data[:] = _native._reset_membership_scales(result.indptr.tolist(), result.data.tolist())
    transpose = result.T
    result = result + transpose - result.multiply(transpose)
    result.eliminate_zeros()
    return result


def init_transform(indices, weights, embedding):
    """Sum fitted coordinates using the supplied weights in rank order."""
    indices, weights, embedding = map(np.asarray, (indices, weights, embedding))
    if indices.ndim != 2 or indices.shape != weights.shape or embedding.ndim != 2:
        raise ValueError("incompatible transform initialization shapes")
    if indices.dtype.kind not in "iu":
        raise TypeError("neighbor indices must be integers")
    if np.float16 in (weights.dtype, embedding.dtype):
        raise NotImplementedError("float16 ranked initialization is unsupported")
    kinds = weights.dtype.kind, embedding.dtype.kind
    if all(kind in "iub" for kind in kinds):
        dtype = np.dtype(np.int64 if "i" in kinds else np.uint64)
    else:
        dtype = np.result_type(weights.dtype, embedding.dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("weights and embedding must have real numeric dtypes")
    coordinates = _native._initialize_ranked_transform(
        (indices.ravel().tolist(), indices.shape),
        weights.astype(dtype, copy=False).ravel().tolist(),
        embedding.astype(dtype, copy=False).ravel().tolist(),
        embedding.shape, f"{dtype.kind}{8 * dtype.itemsize}",
    )
    return np.asarray(coordinates, dtype=np.float32).reshape(indices.shape[0], embedding.shape[1])


def init_graph_transform(graph, embedding):
    """Initialize coordinates from a compressed sparse membership graph."""
    offsets, columns, weights = graph.indptr, graph.indices, graph.data
    n_rows, n_columns = graph.shape
    embedding = np.asarray(embedding)
    if offsets.size != n_rows + 1:
        raise IndexError("compressed graph axes do not align with query rows")
    if embedding.ndim != 2:
        raise ValueError("embedding must have two dimensions")
    if (
        offsets[0] != 0 or offsets[-1] != weights.size or columns.size != weights.size
        or np.any(offsets[1:] < offsets[:-1]) or np.any(columns < 0)
        or np.any(columns >= (n_rows if graph.format == "csc" else n_columns))
    ):
        raise ValueError("invalid compressed graph storage")

    product_type = np.result_type(weights.dtype, embedding.dtype)
    if (
        graph.format == "csr" and graph.has_canonical_format and weights.dtype in (np.float32, np.float64)
        and product_type in (np.float32, np.float64)
        and np.all((weights >= np.finfo(weights.dtype).tiny) & (weights <= 1.0))
    ):
        coordinates = _native._initialize_csr_transform(
            (offsets.tolist(), columns.tolist(), weights.tolist()), n_columns,
            (embedding.ravel().tolist(), embedding.shape),
            (weights.dtype == np.float32, product_type == np.float32),
        )
        return np.asarray(coordinates, dtype=np.float32).reshape(n_rows, embedding.shape[1])

    # The sparse product determines duplicate folding, dtype promotion and the
    # NumPy warnings for zero or nonfinite row sums on these inputs.
    coordinates = np.full((n_rows, embedding.shape[1]), np.nan, dtype=np.float32)
    weighted = offsets[1:] != offsets[:-1]
    unit_entries = np.flatnonzero(weights == 1.0)
    unit_rows = np.searchsorted(offsets, unit_entries, side="right") - 1
    direct_rows, first_unit = np.unique(unit_rows, return_index=True)
    coordinates[direct_rows] = embedding[columns[unit_entries[first_unit]]]
    weighted[direct_rows] = False
    if weighted.any():
        selected = graph[weighted]
        factors = 1.0 / np.asarray(selected.sum(axis=1)).ravel()
        coordinates[weighted] = (diags(factors, dtype=factors.dtype) @ selected) @ embedding
    return coordinates


__all__ = [
    "UMAP", "compute_membership_strengths", "find_ab_params", "fuzzy_simplicial_set",
    "general_simplicial_set_intersection", "general_simplicial_set_union", "init_graph_transform",
    "init_transform", "make_epochs_per_sample", "nearest_neighbors", "reset_local_connectivity",
    "smooth_knn_dist",
]
