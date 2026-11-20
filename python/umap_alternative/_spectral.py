"""Spectral initialization for fuzzy graphs."""

import warnings
from math import isqrt

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances


def _spectral_initialization(
    graph: csr_matrix, n_components: int, rng: np.random.RandomState
) -> np.ndarray:
    """Return float32 coordinates before layout normalization and advance ``rng``.

    The graph must be a symmetric CSR fuzzy graph with positive stored weights
    and one connected component. Requesting all nontrivial eigenvectors uses a
    dense solve. Solver errors propagate and leave the supplied RNG untouched.
    """
    n_rows, n_columns = graph.shape
    if n_rows != n_columns:
        raise ValueError("spectral initialization requires a square graph")
    if not 1 <= n_components < n_rows:
        raise ValueError("spectral initialization requires 1 <= n_components < n_rows")
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise NotImplementedError("spectral initialization requires a connected graph")

    selected = _spectral_eigenvectors(graph, n_components, rng)
    return _scale_and_perturb(selected, rng)


def _normalized_adjacency(graph):
    graph = graph.astype(np.float32, copy=False)
    degrees = np.asarray(graph.sum(axis=0)).ravel()
    inverse_sqrt_degrees = diags(1.0 / np.sqrt(degrees))
    return inverse_sqrt_degrees @ graph @ inverse_sqrt_degrees


def _spectral_eigenvectors(graph, n_components, rng):
    n_rows = graph.shape[0]
    # Keep the weighted adjacency product in float32 and promote at subtraction.
    laplacian = eye(n_rows, dtype=np.float64, format="csr") - _normalized_adjacency(graph)

    n_eigenpairs = n_components + 1
    if n_eigenpairs == n_rows:
        eigenvalues, eigenvectors = eigh(laplacian.toarray())
    else:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=n_eigenpairs,
            which="SM",
            v0=np.ones(n_rows, dtype=np.float64),
            ncv=max(2 * n_eigenpairs + 1, isqrt(n_rows)),
            tol=1e-4,
        )

    # Account for the normal starting block before drawing the coordinate noise.
    rng.standard_normal((n_rows, n_eigenpairs))
    return eigenvectors[:, np.argsort(eigenvalues)[1:n_eigenpairs]]


def _modern_spectral_initialization(graph, n_components, rng):
    """Return the connected modern layout before centering and rescaling."""
    n_rows = graph.shape[0]
    if n_components >= n_rows:
        warnings.warn(
            "Spectral initialisation needs more samples than n_components.\n"
            "Falling back to random initialisation!",
            UserWarning,
            stacklevel=4,
        )
        coordinates = rng.uniform(low=-10.0, high=10.0, size=(n_rows, n_components))
    else:
        if connected_components(graph, directed=False, return_labels=False) != 1:
            raise NotImplementedError("modern spectral initialization requires a connected graph")
        adjacency = _normalized_adjacency(graph)
        count = n_components + 1
        if count == n_rows:
            laplacian = eye(n_rows, dtype=np.float64, format="csr") - adjacency
            eigenvalues, vectors = eigh(laplacian.toarray())
            order = np.argsort(eigenvalues)
        else:
            eigenvalues, vectors = eigsh(
                adjacency, k=count, which="LA", v0=np.ones(n_rows, dtype=np.float32),
                ncv=max(2 * count + 1, isqrt(n_rows)), tol=1e-4,
            )
            order = np.argsort(eigenvalues)[::-1]
        coordinates = vectors[:, order[1:count]]
    scale = 40.0 / np.max(np.abs(coordinates))
    coordinates = np.ascontiguousarray(scale * coordinates, dtype=np.float32)
    coordinates += rng.normal(scale=0.01, size=coordinates.shape).astype(np.float32)
    return coordinates


def _scale_and_perturb(coordinates, rng):
    scale = 10.0 / np.max(np.abs(coordinates))
    coordinates = np.ascontiguousarray(scale * coordinates, dtype=np.float32)
    coordinates += (1e-4 * rng.standard_normal(coordinates.shape)).astype(np.float32)
    return coordinates


def _disconnected_spectral_initialization(graph, features, metric, n_components, rng, labels):
    """Place the components and perturb the assembled float32 layout once."""
    coordinates = _disconnected_spectral_coordinates(graph, features, metric, n_components, rng, labels)
    return _scale_and_perturb(coordinates, rng)


def _disconnected_spectral_coordinates(graph, features, metric, n_components, rng, labels):
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(labels))[:-1])
    count = len(groups)
    if count <= 2 * n_components:
        axes = np.eye((count + 1) // 2, n_components)
        centers = np.concatenate((axes, -axes))[:count]
    else:
        centroids = np.array(
            [np.asarray(features[rows].mean(axis=0)).ravel() for rows in groups],
            dtype=np.float64,
        )
        distances = pairwise_distances(centroids, metric=metric)
        centers = SpectralEmbedding(
            n_components=n_components, affinity="precomputed", random_state=rng
        ).fit_transform(np.exp(-(distances**2)))
        centers /= centers.max()

    coordinates = np.empty((graph.shape[0], n_components), dtype=np.float32)
    min_spectral_size = max(2 * n_components, n_components + 2)
    for rows, center in zip(groups, centers):
        # A separate distance query for each center preserves the observed
        # rounding at the zero-distance cutoff.
        distances = pairwise_distances(center.reshape(1, -1), centers)
        radius = distances[distances > 0].min() / 2
        if len(rows) < min_spectral_size:
            local = rng.uniform(-radius, radius, (len(rows), n_components))
        else:
            local = _spectral_eigenvectors(graph[rows][:, rows], n_components, rng)
            local *= radius / np.max(np.abs(local))
        coordinates[rows] = center + local
    return coordinates
