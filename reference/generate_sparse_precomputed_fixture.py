"""Capture public 0.6.0 CSR precomputed fits and held-out transforms."""

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
from scipy.sparse import csr_matrix
from umap import UMAP


OPTIONS = dict(
    metric="precomputed", n_neighbors=4, n_components=2, n_jobs=1, random_state=42,
    compatibility_layout=True, init="random", a=1.7, b=.9, learning_rate=.13,
    negative_sample_rate=2, n_epochs=6,
)


def csr(rows, n_cols=8):
    entries = [entry for row in rows for entry in row]
    return csr_matrix((
        np.asarray([distance for _, distance in entries], dtype=np.float32),
        np.asarray([column for column, _ in entries], dtype=np.int32),
        np.r_[0, np.cumsum([len(row) for row in rows])],
    ), shape=(len(rows), n_cols))


def bits(values):
    return values.view(np.uint32).ravel().tolist()


def encoded(matrix):
    record = {"shape": list(matrix.shape)}
    if isinstance(matrix, np.ndarray):
        return record | {"kind": "dense", "bits": bits(matrix)}
    record["bits"] = bits(matrix.data)
    if matrix.format == "csr":
        return record | {"kind": "csr", "indptr": matrix.indptr.tolist(),
                         "indices": matrix.indices.tolist()}
    return record | {"kind": "coo", "row": matrix.row.tolist(), "col": matrix.col.tolist()}


def observe(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except (AssertionError, ValueError) as error:
            outcome = {"error": type(error).__name__, "message": str(error)}
        except Exception as error:
            # JIT diagnostics may contain implementation details.
            outcome = {"error": type(error).__name__}
        else:
            outcome = {"result": encoded(result)}
    return outcome | {"warnings": [[item.category.__name__, str(item.message)] for item in caught]}


def build():
    axis = np.array([0, .12, .35, .52, .83, 1.5, 1.71, 2.], dtype=np.float32)
    dense = np.abs(axis[:, None] - axis[None, :])
    rows = [list(enumerate(row)) for row in dense]
    tied = np.ones_like(dense)
    np.fill_diagonal(tied, 0)
    duplicate = [row.copy() for row in rows]
    duplicate[0].insert(0, (1, .05))
    duplicate[1].append((0, .05))
    duplicate[1].extend([(1, .05), (1, -.05)])
    asymmetric_values = dense.copy()
    asymmetric_values[0, 1] = 8
    asymmetric_support = [row.copy() for row in rows]
    asymmetric_support[0].pop(1)
    asymmetric_support[3].pop(0)
    zero = dense.copy()
    zero[0, 1] = 0
    negative = dense.copy()
    negative[0, 1] = -.1
    diagonal = dense.copy()
    diagonal[0, 0] = .1

    def ring(offsets):
        return csr([[(j, dense[i, j]) for j in ((i + p) % 8 for p in offsets)] for i in range(8)])

    training = {
        "dense": dense,
        "full": csr(rows),
        "implicit_diagonal": csr_matrix(dense),
        "k_plus_one": ring([0, 1, -1, 2, -2]),
        "k": ring([1, -1, 2, -2]),
        "k_minus_one": ring([0, 1, -1]),
        "asymmetric_values": csr([list(enumerate(row)) for row in asymmetric_values]),
        "asymmetric_support": csr(asymmetric_support),
        "unbalanced_counts": csr([rows[0][:1] + rows[0][2:], *rows[1:]]),
        "nonzero_diagonal": csr([list(enumerate(row)) for row in diagonal]),
        "explicit_zero": csr([list(enumerate(row)) for row in zero]),
        "negative": csr([list(enumerate(row)) for row in negative]),
        "duplicates": csr(duplicate),
        "unsorted_ties": csr([list(enumerate(row))[::-1] for row in tied]),
        "wide": csr(rows, n_cols=9),
    }
    regular = np.abs(np.array([.1, 1.4], dtype=np.float32)[:, None] - axis[None, :])
    row = [(7, .3), (4, 0), (6, .1), (3, .2), (5, .7), (2, .2), (1, .2), (0, 2)]
    queries = {
        "dense": regular,
        "full": csr([list(enumerate(row)) for row in regular]),
        "k_plus_one": csr([row[:5]]),
        "k": csr([row[:4]]),
        "k_minus_one": csr([row[:3]]),
        "unsorted_ties": csr([row, [(i, 1) for i in [6, 1, 4, 2, 0, 7, 5, 3]]]),
        "duplicates": csr([[(7, .3), (7, 0), (6, .1), (3, .2), (5, .7)]]),
        "one_column": csr([[(7, value) for value in [0, .1, .2, .3, .7]]]),
        "cutoff": csr([[(7, .01), (0, .69), (1, .7), (2, .72), (3, 2)],
                       [(7, .7), (0, .69), (1, .7), (2, .72), (3, 2)]]),
        "wrong_width": csr([row[:5]], n_cols=9),
        "negative": csr([row[:4] + [(5, -.3)]]),
        "short_negative": csr([[(1, -.2)]]),
        "empty_row": csr([[]]),
    }
    cases = {}

    def record(name, dataset, mode, query_names=(), **options):
        values = training[dataset].copy()
        model = UMAP(**(OPTIONS | options), transform_mode=mode)
        fit = observe(lambda: model.fit_transform(values))
        result = {"training": dataset, "mode": mode, "options": options, "fit": fit}
        if encoded(values) != encoded(training[dataset]):
            result["training_after"] = encoded(values)
        if "result" in fit:
            if mode == "embedding":
                result["graph"] = encoded(model.graph_)
            expected = model.graph_ if mode == "graph" else model.embedding_
            result["identity"] = model.transform(values.copy()) is expected
            result["queries"] = {query: observe(lambda: model.transform(queries[query].copy()))
                                 for query in query_names}
        cases[name] = result

    for name in training:
        query_names = ()
        if name in ("dense", "implicit_diagonal"):
            query_names = queries
        elif name == "wide":
            query_names = ("k_plus_one", "wrong_width")
        record(f"{name}_graph", name, "graph", query_names)
    for name in ("dense", "full", "implicit_diagonal", "k_plus_one", "duplicates"):
        record(f"{name}_embedding", name, "embedding",
               ("dense", "full", "k_plus_one", "unsorted_ties", "duplicates", "one_column"))
    record("cutoff_graph", "full", "graph", ("cutoff", "k_plus_one"), disconnection_distance=.7)
    record("cutoff_embedding", "full", "embedding", ("cutoff",), disconnection_distance=.7)
    record("duplicates_zero_epochs", "implicit_diagonal", "embedding",
           ("duplicates", "one_column"), n_epochs=0)
    record("duplicates_thirty_epochs", "implicit_diagonal", "embedding",
           ("duplicates", "one_column"), n_epochs=30)
    record("partially_disconnected", "full", "graph", disconnection_distance=.3)
    record("graph_options", "implicit_diagonal", "graph", ("k_plus_one",),
           disconnection_distance=.7, local_connectivity=2.5, set_op_mix_ratio=.4)
    record("zero_cutoff", "full", "graph", ("cutoff",), disconnection_distance=0.)
    record("truncated", "full", "graph", n_neighbors=15)
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                      "scikit-learn", "pynndescent")},
        "options": OPTIONS,
        "training": {name: encoded(values) for name, values in training.items()},
        "queries": {name: encoded(values) for name, values in queries.items()},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = build()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value)}' for key, value in results.items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, f"{args.output} differs from the regenerated fixture\n")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
