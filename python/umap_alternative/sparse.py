"""Low-level operators on sparse fuzzy membership buffers."""

import numpy as np

from . import _native


def _operand(indptr, indices, data, output):
    buffers = tuple(np.asarray(buffer) for buffer in (indptr, indices, data))
    if any(buffer.ndim != 1 for buffer in buffers):
        raise ValueError("CSR membership buffers must be one-dimensional")
    data = buffers[2]
    aliases = np.shares_memory(data, output)
    if aliases and (
        data.shape != output.shape or data.dtype != output.dtype or data.strides != output.strides
        or data.ctypes.data != output.ctypes.data
    ):
        raise ValueError("operand and output need the same value buffer when they overlap")
    return tuple(buffer.tolist() for buffer in buffers), aliases


def _combine(left, right, rows, columns, result, intersection=None):
    if not isinstance(result, np.ndarray) or result.ndim != 1:
        raise TypeError("result_val must be a one-dimensional NumPy array")
    rows, columns = np.asarray(rows), np.asarray(columns)
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError("COO membership indices must be one-dimensional")
    left, aliases_left = _operand(*left, result)
    right, aliases_right = _operand(*right, result)
    if result.dtype == np.float32:
        storage = "f32"
    elif result.dtype.kind in "iu":
        storage = "int"
    elif result.dtype.kind == "b":
        storage = "bool"
    else:
        storage = "f64"
    result[:] = _native._combine_memberships(
        left, right, (rows.tolist(), columns.tolist()), (result.tolist(), storage),
        (aliases_left, aliases_right), intersection,
    )


def general_sset_union(
    indptr1, indices1, data1, indptr2, indices2, data2, result_row, result_col, result_val,
):
    """Evaluate the fuzzy union at the supplied COO coordinates in place."""
    _combine(
        (indptr1, indices1, data1), (indptr2, indices2, data2), result_row, result_col, result_val,
    )


def general_sset_intersection(
    indptr1, indices1, data1, indptr2, indices2, data2, result_row, result_col, result_val,
    right_complement=False, mix_weight=0.5,
):
    """Evaluate the weighted fuzzy intersection at the supplied COO coordinates in place."""
    _combine(
        (indptr1, indices1, data1), (indptr2, indices2, data2), result_row, result_col, result_val,
        (right_complement, mix_weight),
    )


__all__ = ["general_sset_intersection", "general_sset_union"]
