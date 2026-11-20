"""Capture sparse cosine results from public UMAP 0.6.0 fit and transform."""

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

from generate_sparse_lp_transform_fixture import encode, warning_records


def split_csr(matrix):
    return csr_matrix((np.repeat(matrix.data / 2, 2), np.repeat(matrix.indices, 2),
                       matrix.indptr * 2), shape=matrix.shape)


def build():
    rng = np.random.RandomState(904)
    values = rng.normal(size=(12, 7)).astype(np.float32)
    values[rng.rand(*values.shape) < .52] = 0
    values[2] = 0
    values[-1] = values[1]
    base = csr_matrix(values)
    wide = csr_matrix(np.eye(24, dtype=np.float32))
    overflow = csr_matrix(np.array([
        [1e20, 1e20, 0], [1e-26, 0, 0], [0, 0, 0], [-1, 0, 0],
        [1, 0, 0], [0, 1, 1], [0, -1, 1], [1, 1, 1],
    ], np.float32))
    boundary = csr_matrix(np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [1, 1, 0], [-1, -1, 0], [0, 0, 0], [0, 0, 0],
    ], np.float32))
    antipodes = csr_matrix(np.array([
        [1, 0], [-1, 0], [1, 0], [-1, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
    ], np.float32))
    training = {"base": base, "split": split_csr(base), "csc": base.tocsc(),
                "split_csc": split_csr(base).tocsc(), "split_coo": split_csr(base).tocoo(),
                "ties": wide, "overflow": overflow, "cutoff": boundary, "antipodes": antipodes,
                "zero": csr_matrix((8, 5), dtype=np.float32),
                "singleton": csr_matrix([[2., 0.]], dtype=np.float32)}
    batch = rng.normal(size=(4, 7)).astype(np.float32)
    batch[rng.rand(*batch.shape) < .52] = 0
    batch[1] = 0
    batch[3] = batch[0]
    q = csr_matrix(batch)
    cutoff = csr_matrix(np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], np.float32))
    narrow = csr_matrix(np.array([[0, .5, -.72], [1, 0, 0], [.4, .23, .51], [0, 0, 0]], np.float32))
    signed_zero = csr_matrix((np.array([-0.0, 0.0], np.float32), [1, 6], [0, 2, 2]), shape=(2, 7))
    reversed_query = q.copy()
    for first, last in zip(q.indptr[:-1], q.indptr[1:]):
        reversed_query.indices[first:last] = q.indices[first:last][::-1]
        reversed_query.data[first:last] = q.data[first:last][::-1]
    queries = {
        "dense": batch, "csr": q, "csc": q.tocsc(), "coo": split_csr(q).tocoo(),
        "f64": q.astype(np.float64), "dense_f64": batch.astype(np.float64),
        "split": split_csr(q), "split_csc": split_csr(q).tocsc(),
        "narrow": narrow, "narrow_dense": narrow.toarray(), "narrow_split": split_csr(narrow),
        "width_plus_one": csr_matrix((3, 8)), "signed_zero": signed_zero, "reversed": reversed_query,
        "zero_rows": csr_matrix((0, 7), dtype=np.float32), "zero_width": csr_matrix((2, 0)),
        "fit": base, "fit_dense": values, "fit_coo": base.tocoo(),
        "fit_csc": base.tocsc(), "fit_split": training["split"],
        "fit_zero": training["zero"], "f64_fit": base.astype(np.float64),
        "ties": csr_matrix((2, 24), dtype=np.float32), "ties_narrow": csr_matrix((2, 5)),
        "cutoff": cutoff, "overflow": overflow[[0, 1, 2, 3]],
        "antipodes": csr_matrix([[1., 0.], [0., 0.], [-1., 0.]], dtype=np.float32),
        "zero": csr_matrix((2, 5), dtype=np.float32), "singleton": training["singleton"],
    }
    options = {"metric": "cosine", "n_neighbors": 4, "n_components": 2,
               "random_state": 19, "transform_seed": 0, "n_epochs": 30,
               "init": "random", "n_jobs": 1, "compatibility_layout": True,
               "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(name, source, mode, query_names, *, extra=(), **overrides):
        model = UMAP(**(options | overrides), transform_mode=mode)
        result = {"training": source, "mode": mode, "fit_mode": mode,
                  "options": overrides, "transforms": {}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X = training[source].copy()
            try:
                model.fit(X)
            except Exception as error:
                raise SystemExit(f"{name} fit: {type(error).__name__}") from None
        fitted = model.graph_ if mode == "graph" else model.embedding_
        result["fitted"] = encode(fitted)
        result["fit_warnings"] = warning_records(caught)
        if mode == "embedding" and hasattr(model, "graph_"):
            result["fitted_graph"] = encode(model.graph_)
        calls = [(query, query, {}) for query in query_names] + list(extra)
        for label, query, changes in calls:
            original = {key: getattr(model, key) for key in changes}
            model.set_params(**changes)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    transformed = model.transform(queries[query].copy())
                except Exception as error:
                    call = {"error": type(error).__name__}
                else:
                    call = {"identity": transformed is fitted}
                    if transformed is not fitted:
                        call["result"] = encode(transformed)
            result["transforms"][label] = call | {
                "query": query, "parameters": changes, "warnings": warning_records(caught),
            }
            model.set_params(**original)
        cases[name] = result

    for mode in ("graph", "embedding"):
        capture(f"{mode}_formats", "base", mode,
                ("dense", "csr", "csc", "coo", "split", "split_csc", "narrow", "narrow_dense",
                 "narrow_split", "signed_zero", "reversed",
                 "f64", "dense_f64", "fit", "fit_dense", "fit_coo", "fit_csc", "f64_fit",
                 "width_plus_one", "zero_rows", "zero_width"))
        capture(f"{mode}_split", "split", mode, ("csr", "split", "fit_split", "fit", "narrow_split"))
        capture(f"{mode}_csc", "csc", mode, ("csr", "fit", "fit_csc"))
        capture(f"{mode}_split_csc", "split_csc", mode, ("split_csc", "csr"))
        capture(f"{mode}_split_coo", "split_coo", mode, ("split", "fit_coo"))
        capture(f"{mode}_ties", "ties", mode, ("ties", "ties_narrow"), n_epochs=0)
        capture(f"{mode}_overflow", "overflow", mode, ("overflow",), n_epochs=1, n_neighbors=3)
        capture(f"{mode}_zero", "zero", mode, ("zero", "fit_zero"), n_epochs=0, n_neighbors=2)
        for cutoff_value in (None, 1.0, 1.5, 2.0, 2.25):
            capture(f"{mode}_cutoff_{cutoff_value}", "cutoff", mode, ("cutoff",),
                    disconnection_distance=cutoff_value, n_epochs=9, n_neighbors=7)
        for cutoff_value in (None, 2.25):
            capture(f"{mode}_antipodes_{cutoff_value}", "antipodes", mode, ("antipodes",),
                    disconnection_distance=cutoff_value, n_epochs=9, n_neighbors=7)
    capture("embedding_singleton", "singleton", "embedding", ("singleton",))
    for epochs in (0, 1, 3):
        capture(f"embedding_epochs_{epochs}", "base", "embedding", ("csr",), n_epochs=epochs,
                extra=(("positive", "csr", {"n_epochs": 30, "transform_seed": 2**32 - 1}),))
    capture("embedding_default_epochs", "base", "embedding", ("csr",), n_epochs=None)
    capture("embedding_spectral", "base", "embedding", ("csr",), n_epochs=9, init="spectral")
    residuals = {}
    for mode in ("graph", "embedding"):
        name = f"{mode}_formats"
        residuals[name] = cases[name]["transforms"].pop("width_plus_one")
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": options,
        "training": {name: encode(matrix) for name, matrix in training.items()},
        "queries": {name: encode(matrix) for name, matrix in queries.items()},
        "cases": cases,
        "wider_query_residuals": residuals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in build().items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, "Regenerated fixture differs")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # JIT errors can include source excerpts in their diagnostic text.
        raise SystemExit(type(error).__name__) from None
