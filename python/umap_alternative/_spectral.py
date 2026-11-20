"""Spectral initialization for connected fuzzy graphs."""

from math import isqrt

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


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

    graph = graph.astype(np.float32, copy=False)
    degrees = np.asarray(graph.sum(axis=0)).ravel()
    inverse_sqrt_degrees = diags(1.0 / np.sqrt(degrees))
    # Keep the weighted adjacency product in float32 and promote at subtraction.
    laplacian = eye(n_rows, dtype=np.float64, format="csr") - (
        inverse_sqrt_degrees @ graph @ inverse_sqrt_degrees
    )

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

    selected = eigenvectors[:, np.argsort(eigenvalues)[1:n_eigenpairs]]
    scale = 10.0 / np.max(np.abs(selected))
    coordinates = np.ascontiguousarray(scale * selected, dtype=np.float32)

    # Account for the normal starting block before drawing the coordinate noise.
    rng.standard_normal((n_rows, n_eigenpairs))
    coordinates += (1e-4 * rng.standard_normal((n_rows, n_components))).astype(np.float32)
    return coordinates
