"""Capture public neighbor stages and utilities from the pinned UMAP wheel."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from umap import umap_, utils


def encode(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return [encode(element) for element in value]
    if issparse(value):
        coo = value.tocoo()
        return {
            "format": value.format, "shape": list(value.shape),
            "rows": coo.row.tolist(), "columns": coo.col.tolist(), "data": encode(coo.data),
        }
    dtype = value.dtype
    contents = value.view(f"uint{dtype.itemsize * 8}") if dtype.kind == "f" else value
    return {"dtype": dtype.name, "shape": list(value.shape), "contents": contents.ravel().tolist()}


def record(function, args, options, name):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = function(*args, **options)
    return {
        "name": name, "options": options, "result": encode(result),
        "warnings": [notice.category.__name__ for notice in seen],
    }


def capture():
    base = np.asarray([[0., .6, 2.3, 3.], [0, 0, 1, 1.8], [0, .7, .9, 1.7]], np.float32)
    graph_indices = np.asarray([[0, 1, 2, -1], [1, 0, 2, 2], [2, 1, 0, -1]], np.int32)
    sigmas = np.asarray([.7, 0, 1.1], np.float32)
    rhos = np.asarray([.3, 0, 1.7], np.float32)
    cases = {"smooth": [], "memberships": [], "graphs": [], "utilities": []}
    for dtype in (np.float32, np.float64):
        profiles = [
            ("default", {"k": 4}),
            ("custom", {"k": 3.2, "bandwidth": 1.3, "local_connectivity": .5, "n_iter": 12}),
            ("zero_iterations", {"k": 4, "n_iter": 0}),
            ("zero_target", {"k": 1}),
            ("negative_infinite_target", {"k": 0}),
            ("rho_zero", {"k": 4, "local_connectivity": 0}),
            ("rho_maximum", {"k": 4, "local_connectivity": 12}),
        ]
        for name, parameters in profiles:
            d = base.astype(dtype)
            case = record(umap_.smooth_knn_dist, (d,), parameters, f"{dtype.__name__}_{name}")
            cases["smooth"].append(case | {"distances": encode(d)})
    for name, data in [
        ("infinite", [[0, 1, np.inf], [0, 2, 3]]),
        ("all_missing_row", [[0, np.inf, np.inf], [0, 2, 3]]),
        ("nan", [[0, np.nan, 1], [0, 2, 3]]),
        ("negative", [[0, 1, -1], [0, 2, 3]]),
        ("unsorted", [[0, 3, 1], [0, 2, 3]]),
        ("empty_width", np.zeros((3, 0), np.float32)),
        ("empty_rows", np.zeros((0, 3), np.float32)),
        ("self_only", np.zeros((2, 1), np.float32)),
    ]:
        d = np.asarray(data, np.float32)
        case = record(umap_.smooth_knn_dist, (d,), {"k": 3, "local_connectivity": 1.5}, name)
        cases["smooth"].append(case | {"distances": encode(d)})
    for bipartite in (False, True):
        for ret in (False, True):
            args = (graph_indices, base, sigmas, rhos)
            case = record(
                umap_.compute_membership_strengths, args, {"bipartite": bipartite, "return_dists": ret},
                f"bipartite_{bipartite}_dists_{ret}",
            )
            cases["memberships"].append(case | {"input": encode(args)})
    for apply in (False, True):
        for ret in (None, False, True):
            options = {"apply_set_operations": apply, "return_dists": ret}
            args = (np.zeros((3, 1), np.float32), 3, np.random.RandomState(123), "euclidean")
            case = record(
                umap_.fuzzy_simplicial_set, args,
                options | {"knn_indices": graph_indices, "knn_dists": base},
                f"apply_{apply}_dists_{ret}",
            )
            del case["options"]["knn_indices"], case["options"]["knn_dists"]
            cases["graphs"].append(case | {"input": encode((graph_indices, base)), "n_neighbors": 3})
    matrix = np.asarray(
        [[0, 1, 2, np.inf], [np.inf, -0., 2, 2], [np.inf, 4, 0, -1], [np.inf, 4, 3, 0]], np.float64
    )
    for k in (0, 2, 3, 4):
        args = (matrix, k, "precomputed", {}, False, np.random.RandomState(34))
        case = record(umap_.nearest_neighbors, args, {}, f"precomputed_{k}")
        cases["utilities"].append(case | {
            "function": "nearest_neighbors", "input": encode(matrix), "n_neighbors": k,
        })
    sorting = (np.arange(64) * 17 % 5).reshape(1, -1).astype(np.float32)
    for k in (0, 7, 64):
        case = record(utils.fast_knn_indices, (sorting, k), {}, f"stable_ties_{k}")
        cases["utilities"].append(case | {
            "function": "fast_knn_indices", "input": encode(sorting), "n_neighbors": k,
        })
    gather = np.asarray([[3, 1], [2, 2], [1, 0], [3, 2]], np.int32)
    case = record(utils.submatrix, (matrix, gather, 2), {}, "submatrix")
    cases["utilities"].append(case | {"function": "submatrix", "input": encode((matrix, gather))})
    for name, data in [
        ("average_finite", [[1.2345678, 9.876543, 10.876543]]),
        ("average_nan", [[1, np.nan, 2]]),
        ("average_positive_infinity", [[0, 2, np.inf], [np.inf, np.inf, np.inf], [3, -3, 0]]),
        ("average_negative_infinity", [[1, -np.inf, 2]]),
        ("average_empty", np.empty((2, 0), np.float32)),
    ]:
        values = np.asarray(data, np.float32)
        case = record(utils.average_nn_distance, (values,), {}, name)
        cases["utilities"].append(case | {"function": "average_nn_distance", "input": encode(values)})
    for dtype in (np.float32, np.float64):
        for name, data in [
            ("positive", [0, .1, .33, 1, 2]), ("zero", [0, 0]), ("negative", [-2, -1, -3]),
            ("empty", []),
        ]:
            values = np.asarray(data, dtype)
            case = record(umap_.make_epochs_per_sample, (values, 9), {}, f"epochs_{dtype.__name__}_{name}")
            cases["utilities"].append(case | {
                "function": "make_epochs_per_sample", "input": encode(values),
            })
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = capture()
    document = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, separators=(",", ":"), allow_nan=False)}'
        for key, value in fixture.items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != document:
            parser.exit(1, f"{args.output} differs from the regenerated fixture\n")
    else:
        args.output.write_text(document, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream JIT diagnostics can contain source excerpts.
        raise SystemExit(type(error).__name__) from None
