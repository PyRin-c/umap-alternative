"""Capture public UMAP 0.6.0 transforms after sparse Euclidean fitting."""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
from umap import UMAP


OPTIONS = dict(
    metric="euclidean", n_neighbors=4, n_components=2, n_jobs=1, random_state=42,
    compatibility_layout=True, init="random", a=1.7, b=.9, learning_rate=.13,
    negative_sample_rate=2, n_epochs=6, transform_seed=42,
)


def values(array):
    return {
        "shape": list(array.shape), "dtype": array.dtype.name,
        "bits": array.view(f"uint{array.itemsize * 8}").ravel().tolist(),
    }


def matrix_record(matrix):
    if not issparse(matrix):
        return values(matrix)
    record = {"format": matrix.format, "shape": list(matrix.shape), "data": values(matrix.data)}
    if matrix.format == "coo":
        return record | {"row": matrix.row.tolist(), "col": matrix.col.tolist()}
    return record | {"indptr": matrix.indptr.tolist(), "indices": matrix.indices.tolist()}


def observe(call, fitted):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except ValueError as error:
            record = {"error": type(error).__name__, "message": str(error)}
        except Exception as error:
            # A JIT failure may include upstream source in its message.
            record = {"error": type(error).__name__}
        else:
            record = {"identity": result is fitted}
            if result is not fitted:
                record["result"] = matrix_record(result)
    return record | {"warnings": [[item.category.__name__, str(item.message)] for item in caught]}


def inputs():
    path = Path(__file__).with_name("fixtures") / "umap-learn-0.6.0-sparse-graph-windows-x86_64.json"
    source = json.loads(path.read_text(encoding="utf-8"))["inputs"]["features"]
    base = csr_matrix((
        np.asarray(source["data_bits_uint32"], dtype=np.uint32).view(np.float32),
        source["indices"], source["indptr"],
    ), shape=source["shape"])
    dense = base.toarray()
    batch = np.stack((dense[1], np.zeros_like(dense[0]), dense[0] * np.float32(.9),
                      dense[-1] + np.float32(.03), np.zeros_like(dense[0])))
    csr = csr_matrix(batch)
    split = csr_matrix((np.repeat(csr.data / 2, 2), np.repeat(csr.indices, 2), csr.indptr * 2),
                       shape=csr.shape)
    unsorted = csr.copy()
    for start, end in zip(unsorted.indptr[:-1], unsorted.indptr[1:]):
        unsorted.data[start:end] = unsorted.data[start:end][::-1]
        unsorted.indices[start:end] = unsorted.indices[start:end][::-1]
    cutoff = np.array([0, .125, .5, .8125, 1.21875, 1.7, 2.125, 9.25], np.float32)
    tied = np.array([-1.375, -.875, -.5, -.1875, 0, .125, .5, .78125, 1.0625, 8], np.float32)
    angles = np.resize(np.array([2, -3, 1, -2, 0, 1, -1, 2], np.float32), 17)
    tie_points = np.column_stack((
        angles, (np.arange(17, dtype=np.float32) + 1) * np.float32(.0001) * (np.abs(angles) != 1),
    ))
    cancellation = csr_matrix((
        np.array([1e8, 1, -1e8, 2, 4], dtype=np.float64), [0] * 5, [0, 3, 4, 5],
    ), shape=(3, 1))
    wide = csr_matrix((cutoff[1:], (np.arange(1, cutoff.size), [4096] * (cutoff.size - 1))),
                      shape=(cutoff.size, 4097))
    return {
        "csr": base, "csc": base.tocsc(), "coo": base.tocoo(),
        "cutoff": csr_matrix(cutoff.reshape(-1, 1)),
        "cutoff_ties": csr_matrix(tied.reshape(-1, 1)),
        "wide_ties": csr_matrix(tie_points),
        "cancellation": csr_matrix(np.array([[0], [2], [4], [7], [10]], np.float32)),
        "wide_features": wide,
        "width_residual": csr_matrix(np.array(
            [[0, 5, 1], [1, 3, 0], [3, 0, 2], [5, 0, 9], [7, 1, 1], [9, 0, 0]], np.float32,
        )),
    }, {
        "dense": batch, "csr": csr, "csc": csr.tocsc(), "coo": split.tocoo(),
        "split_csr": split, "unsorted": unsorted,
        "float64_dense": batch.astype(np.float64), "float64_csr": csr.astype(np.float64),
        "empty_rows": csr_matrix((2, base.shape[1]), dtype=np.float32),
        "empty_batch": csr[:0], "no_features": csr[:, :0], "narrow": csr[:, :-1],
        "fitted_csr": base.copy(), "fitted_csc": base.tocsc(),
        "fitted_coo": base.tocoo(), "fitted_dense": dense, "fitted_f64": base.astype(np.float64),
        "cutoff": csr_matrix(np.array([[.3125], [.5], [1], [20], [9.25], [0], [.625]], np.float32)),
        "all_cut": csr_matrix(np.array([[20]], np.float32)),
        "exact_cutoff_ties": csr_matrix(np.array([[0], [.3125], [20]], np.float32)),
        "wide_ties": np.array([[0, 0], [0, 1], [0, -1]], np.float32),
        "cancel_csr": cancellation, "cancel_csc": cancellation.tocsc(),
        "cancel_coo": cancellation.tocoo(),
        "wide_features": csr_matrix((np.array([.3, 1.4], np.float32), ([0, 2], [4096, 4096])),
                                    shape=(3, 4097)),
        "width_residual": csr_matrix(np.array(
            [[0, 2, 0, -7], [1, 1, 2, 8], [7, 0, 1, 0], [0, 0, 0, 4]], np.float32,
        )),
    }


def build():
    training, queries = inputs()
    cases = {}

    def capture(name, source, mode, calls, *, fit_mode=None, **overrides):
        fit_mode = fit_mode or mode
        model = UMAP(**(OPTIONS | overrides), transform_mode=fit_mode)
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(training[source].copy())
        model.set_params(transform_mode=mode)
        fitted = model.graph_ if mode == "graph" else model.embedding_
        results = {"training": source, "mode": mode, "fit_mode": fit_mode, "options": overrides,
                   "fitted": matrix_record(fitted),
                   "fit_warnings": [[w.category.__name__, str(w.message)] for w in fit_warnings],
                   "transforms": {}}
        for label, query, parameters in calls:
            original = {key: getattr(model, key) for key in parameters}
            model.set_params(**parameters)
            result = observe(lambda: model.transform(queries[query].copy()), fitted)
            results["transforms"][label] = result | {"query": query, "parameters": parameters}
            model.set_params(**original)
        cases[name] = results

    def calls(names):
        return [(name, name, {}) for name in names]

    formats = (
        "dense", "csr", "csc", "coo", "split_csr", "unsorted", "float64_dense",
        "float64_csr", "empty_rows", "empty_batch", "no_features", "narrow",
        "fitted_csr", "fitted_csc", "fitted_coo", "fitted_dense", "fitted_f64",
    )
    for mode in ("graph", "embedding"):
        capture(f"{mode}_csr", "csr", mode, calls(formats))
        for source in ("csc", "coo"):
            capture(f"{mode}_{source}", source, mode, calls(("csr", source, "fitted_csr")))
        capture(f"{mode}_cancellation", "cancellation", mode,
                calls(("cancel_csr", "cancel_csc", "cancel_coo")), n_neighbors=3)
        capture(f"{mode}_ties", "wide_ties", mode, calls(("wide_ties",)), fit_mode="embedding")
        capture(f"{mode}_wide_features", "wide_features", mode, calls(("wide_features",)))
        capture(f"{mode}_cutoff_ties", "cutoff_ties", mode,
                calls(("exact_cutoff_ties",)), disconnection_distance=.5)
        capture(f"{mode}_cutoff_ties_nextafter", "cutoff_ties", mode, calls(("exact_cutoff_ties",)),
                disconnection_distance=float(np.nextafter(np.float32(.5), np.float32(1))))

    capture("graph_cutoff", "cutoff", "graph", calls(("cutoff", "all_cut")),
            disconnection_distance=.5)
    capture("graph_all_cut", "cutoff", "graph", calls(("cutoff",)), disconnection_distance=0.)
    cutoff_calls = calls(("cutoff", "all_cut"))
    cutoff_calls += [
        ("zero_budget", "cutoff", {"n_epochs": 0}),
        ("zero_budget_all_cut", "all_cut", {"n_epochs": 0}),
        ("positive_budget", "cutoff", {"n_epochs": 30}),
        ("current_cutoff_ignored", "cutoff", {"disconnection_distance": None}),
    ]
    capture("embedding_cutoff", "cutoff", "embedding", cutoff_calls, disconnection_distance=.5)
    budgets = [(f"epochs_{epochs}", "csr", {"n_epochs": epochs})
               for epochs in (None, 0, 1, 2, 3, 5, 6, 9, 30, [0, 2, 6])]
    budgets += [(f"seed_{seed}_{epochs}", "csr", {"transform_seed": seed, "n_epochs": epochs})
                for seed in (0, 2**32 - 1) for epochs in (6, 30)]
    budgets.append(("fitted_bypass", "fitted_csr", {"transform_seed": -1, "n_epochs": 0}))
    capture("budgets", "csr", "embedding", budgets)
    capture("wider_query", "width_residual", "graph", calls(("width_residual",)), n_neighbors=3)
    capture("graph_only_ties", "wide_ties", "graph", calls(("wide_ties",)))
    residuals = {name: cases.pop(name) for name in ("wider_query", "graph_only_ties")}
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                      "scikit-learn", "pynndescent")},
        "options": OPTIONS,
        "training": {name: matrix_record(matrix) for name, matrix in training.items()},
        "queries": {name: matrix_record(matrix) for name, matrix in queries.items()},
        "cases": cases,
        "residuals": residuals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = build()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(data, allow_nan=False)}' for key, data in record.items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, f"{args.output} differs from the regenerated fixture")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
