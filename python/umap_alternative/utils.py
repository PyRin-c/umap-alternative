"""Utilities for UMAP inputs."""

import numpy as np
from scipy.sparse import csr_matrix

from . import _native


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


__all__ = ["csr_unique"]
