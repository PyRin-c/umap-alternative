"""Observe directed graph operations on the pinned public umap-learn wheel."""

import argparse
from base64 import b64encode
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse as scipy_sparse
from umap import sparse, umap_


def array(values):
    values = np.asarray(values)
    return {"dtype": values.dtype.str, "bytes": b64encode(values.tobytes()).decode("ascii")}


def graph(matrix):
    indices = ("row", "col") if matrix.format == "coo" else ("indptr", "indices")
    return {
        "type": type(matrix).__name__, "shape": matrix.shape,
        "indices": [array(getattr(matrix, name).astype(np.int64)) for name in indices],
        "data": array(matrix.data),
    }


def observe(operation, left, right, options):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            if operation == "union":
                result = umap_.general_simplicial_set_union(left, right)
            elif operation == "intersection":
                result = umap_.general_simplicial_set_intersection(left, right, **options)
            else:
                result = umap_.reset_local_connectivity(left, **options)
            observed = {"result": graph(result), "returns_left": result is left,
                        "shares_left": bool(np.shares_memory(result.data, left.data))}
        except Exception as error:
            observed = {"error": type(error).__name__}
    observed["warnings"] = [type(item.message).__name__ for item in seen]
    return observed


def generate():
    assert version("umap-learn") == "0.6.0"
    cases = []
    inputs = {}

    def capture(name, operation, label, args, options=None):
        options = options or {}
        before = [graph(arg) if arg is not None else None for arg in args]
        inputs[label] = before
        observed = observe(operation, *args, options)
        for name_of_operand, arg, original in zip(("left", "right"), args, before):
            after = None if arg is None else graph(arg)
            if after != original:
                observed[name_of_operand] = after
        cases.append({"name": name, "operation": operation, "input": label, "options": options,
                      "same_operand": args[0] is args[1], "observed": observed})

    def operands(fmt, dtype, right_dtype=None):
        left = (([0, 0, 0, 1, 2, 2, 2, 3], [2, 1, 3, 0, 1, 3, 3, 2]),
                [0.25, 0.5, 0, 0.2, 0.1, 0.3, 0.1, 0.9])
        right = (([0, 0, 1, 1, 2, 3], [1, 3, 1, 2, 1, 2]), [0.3, 0.8, 0.6, 0.7, 0.82, 1])
        return [getattr(scipy_sparse, fmt)(scipy_sparse.coo_matrix((np.array(val, dt), ij), shape=(5, 5)))
                for (ij, val), dt in zip((left, right), (dtype, right_dtype or dtype))]

    for fmt, dtype, right_dtype in (
        ("coo_matrix", "float32", None), ("csr_matrix", "float32", None),
        ("csc_matrix", "float32", None), ("coo_array", "float32", None),
        ("csr_array", "float64", None), ("csr_matrix", "float32", "float64"),
        ("csr_matrix", "int32", None),
    ):
        label = f"{fmt}-{dtype}-{right_dtype or dtype}"
        capture(f"{label}-union", "union", label, operands(fmt, dtype, right_dtype))
        for mix in (-1.0, 0.0, 0.25, float(np.nextafter(0.5, 0.0)), 0.5,
                    float(np.nextafter(0.5, 1.0)), 0.75, 1.0, 2.0):
            for complement in (False, True):
                if mix == 2 and dtype == "int32":
                    continue
                capture(f"{label}-intersection-{mix}-{complement}", "intersection", label,
                        operands(fmt, dtype, right_dtype), {"weight": mix, "right_complement": complement})
        for retune in (False, True):
            capture(f"{label}-reset-{retune}", "reset", f"{label}-reset",
                    (operands(fmt, dtype, right_dtype)[0], None), {"reset_local_metric": retune})

    for dtype in (np.float32, np.float64):
        for n in (3, 4, 8, 15, 16, 30):
            source = scipy_sparse.coo_matrix(
                (np.geomspace(0.007, 1.0, n).astype(dtype), (np.zeros(n, int), np.arange(1, n + 1))),
                shape=(n + 1, n + 1),
            )
            label = f"reset-degree-{n}-{dtype.__name__}"
            capture(label, "reset", label, (source, None), {"reset_local_metric": True})
        for same in (False, True):
            for complement in (False, True):
                left = scipy_sparse.csr_matrix((np.array([.7, .1, .3, .2], dtype), [2, 1, 2, 1],
                                                [0, 4, 4, 4]), shape=(3, 3))
                right = left if same else scipy_sparse.csr_matrix([[.12, .4, .6], [0, 0, 0], [0, 0, 0]], dtype=dtype)
                label = f"unsorted-duplicates-{dtype.__name__}-{same}-{complement}"
                capture(label, "intersection", label, (left, right), {"right_complement": complement})

    empty = scipy_sparse.csr_matrix((3, 3), dtype=np.float32)
    identity = scipy_sparse.eye(3, dtype=np.float32, format="csr")
    for operation in ("union", "intersection"):
        capture(f"{operation}-empty", operation, "empty-operand", (empty, identity))
        capture(f"{operation}-incompatible", operation, "incompatible",
                (identity, scipy_sparse.eye(4, dtype=np.float32, format="csr")))
    capture("reset-empty", "reset", "empty-reset", (empty, None), {"reset_local_metric": True})
    for label, weights in (("stored-zeros", [0, 0]), ("nan", [np.nan, 1]), ("infinity", [np.inf, 1]),
                           ("negative", [1, -.25]), ("negative-row", [-1, -.25])):
        for retune in (False, True):
            source = scipy_sparse.csr_matrix((np.array(weights, np.float32), [1, 2], [0, 2, 2, 2]), shape=(3, 3))
            capture(f"reset-{label}-{retune}", "reset", f"reset-{label}", (source, None),
                    {"reset_local_metric": retune})

    raw = []
    for dt in (np.float32, np.float64):
        left = scipy_sparse.csr_matrix((np.array([0, 2e-10, .3], dt), [0, 1, 3], [0, 3, 3]), shape=(2, 5))
        right = scipy_sparse.csr_matrix((np.array([0, .1, .5], dt), [0, 2, 3], [0, 3, 3]), shape=(2, 5))
        label = f"buffer-{dt.__name__}"
        inputs[label] = [graph(left), graph(right)]
        for op, options in (("union", {}), ("intersection", {}), ("intersection", {"right_complement": True})):
            result = np.arange(21, 26, dtype=np.float64)
            before = array(result)
            rows = np.zeros(result.size, dtype=np.int32)
            columns = np.arange(result.size, dtype=np.int32)
            function = getattr(sparse, f"general_sset_{op}")
            value = function(left.indptr, left.indices, left.data, right.indptr, right.indices, right.data,
                             rows, columns, result, **options)
            assert value is None
            raw.append({"name": f"{label}-{op}-{bool(options)}", "input": label, "operation": op,
                        "options": options, "rows": array(rows), "columns": array(columns),
                        "before": before, "after": array(result)})
    return {"cases": cases, "inputs": inputs, "raw": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contents = json.dumps(generate(), sort_keys=True, separators=(",", ":")) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != contents:
            raise SystemExit("graph operator fixture differs")
    else:
        args.output.write_bytes(contents.encode("utf-8"))


if __name__ == "__main__":
    main()
