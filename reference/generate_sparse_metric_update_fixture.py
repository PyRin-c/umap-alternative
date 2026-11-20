"""Capture public sparse Manhattan, Chebyshev, and cosine update results."""

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
from scipy.sparse import csr_matrix, vstack
from umap import UMAP

from generate_sparse_lp_transform_fixture import encode, warning_records
from generate_sparse_update_fixture import repeated_columns, state


def transform(model, matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.transform(matrix.copy())
        except Exception as error:
            return {"error": type(error).__name__}
    if result is getattr(model, "embedding_", None):
        return {"identity": "embedding_"}
    if result is getattr(model, "graph_", None):
        return {"identity": "graph_"}
    return {"result": encode(result)}


def build(seed):
    rng = np.random.RandomState(seed)
    values = rng.normal(size=(9, 7)).astype(np.float32)
    values[rng.rand(*values.shape) < .45] = 0
    values[3] = 0
    base = csr_matrix(values)
    batch = rng.normal(size=(3, 7)).astype(np.float32)
    batch[rng.rand(*batch.shape) < .53] = 0
    batch = csr_matrix(batch)
    axes = csr_matrix(np.array([[1, 0], [-1, 0], [1, 0], [-1, 0], [1, 0], [-1, 0],
                                [0, 1], [0, -1], [1, 0]], np.float32))
    eye = csr_matrix(np.eye(9, dtype=np.float32))
    discrete = csr_matrix(rng.choice(np.array([-1, 0, 1], np.float32), (28, 6)))
    eps = np.float32(2**-25)
    precision = csr_matrix(np.array([[1, eps, 0], [1, 0, 0], [1, 2*eps, 0], [1.06, 0, 0],
                                     [1.125, 0, 0], [1.18, 0, .05], [1.35, .01, 0],
                                     [1.4, 0, 0], [0, 0, 0]], np.float32))
    chebyshev_precision = csr_matrix(np.array([[-2*eps, 0], [0, -eps], [-1, -1], [-.5, -1],
                                               [-2, -1], [-3, -2], [-4, -1], [-2, -3], [1, 1]],
                                              np.float32))
    cancel = csr_matrix((np.array([1e8, 1, -1e8, 2, -3], np.float64),
                         [0, 0, 0, 1, 6], [0, 3, 5]), shape=(2, 7))
    inputs = {
        "base": base, "split": repeated_columns(base), "one": batch[:1], "last": batch[2:],
        "two": batch[:2], "split_two": repeated_columns(batch[:2]), "two_dense": batch[:2].toarray(),
        "two_csc": batch[:2].tocsc(), "two_coo": repeated_columns(batch[:2]).tocoo(),
        "cancel": cancel, "cancel_coo": cancel.tocoo(), "repeat": base[1:2],
        "zero": csr_matrix((1, 7), dtype=np.float32),
        "stored_zero": csr_matrix((np.array([-0., 0.], np.float32), [0, 6], [0, 2]), shape=(1, 7)),
        "empty": csr_matrix((0, 7), dtype=np.float32), "wide": csr_matrix((1, 8), dtype=np.float32),
        "ties": eye[:8], "ties_append": eye[8:], "axes": axes[:8], "axes_append": axes[8:],
        "discrete": discrete[:24], "discrete_append": discrete[24:],
        "precision": precision[:8], "precision_append": precision[8:],
        "chebyshev_precision": chebyshev_precision[:8],
        "chebyshev_precision_append": chebyshev_precision[8:],
    }
    for name in ("split", "split_two"):
        canonical = inputs[name].copy()
        canonical.sum_duplicates()
        inputs[f"{name}_canonical"] = canonical
    options = {"n_neighbors": 4, "n_components": 2, "random_state": 41, "transform_seed": 0,
               "n_epochs": 0, "init": "random", "n_jobs": 1, "compatibility_layout": True,
               "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(metric, name, source, calls, **overrides):
        settings = overrides | {"metric": metric}
        model = UMAP(**(options | settings))
        training = inputs[source].copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.fit(training)
            except Exception as error:
                raise SystemExit(f"{metric}/{name} fit: {type(error).__name__}") from None
        record = {"training": source, "options": settings, "fit": state(model),
                  "fit_warnings": warning_records(caught),
                  "fit_input_mutated": encode(training) != encode(inputs[source]), "steps": []}
        if record["fit_input_mutated"]:
            record["fit_input_slots"] = [int(training.nnz), training.data.size]
        original = inputs[source].copy()
        stacked = csr_matrix(original)
        for batch_name, parameters in calls:
            saved = {key: getattr(model, key) for key in parameters}
            model.set_params(**parameters)
            before = state(model)
            old_graph = getattr(model, "graph_", None)
            old_embedding = getattr(model, "embedding_", None)
            batch = inputs[batch_name].copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = model.update(batch)
                except Exception as error:
                    status = type(error).__name__
                else:
                    assert result is None
                    status = "ok"
            step = {"batch": batch_name, "options": parameters, "status": status,
                    "warnings": warning_records(caught), "after": state(model),
                    "graph_replaced": getattr(model, "graph_", None) is not old_graph,
                    "embedding_replaced": getattr(model, "embedding_", None) is not old_embedding,
                    "previous_graph_unchanged": old_graph is None or encode(old_graph) == before["graph_"],
                    "previous_embedding_unchanged": old_embedding is None or encode(old_embedding) == before["embedding_"],
                    "batch_unchanged": encode(batch) == encode(inputs[batch_name]),
                    "transform": {"original": transform(model, original)}}
            if status == "ok":
                stacked = vstack((stacked, csr_matrix(batch)), format="csr")
                step["transform"]["combined"] = transform(model, stacked)
                if not stacked.has_canonical_format:
                    canonical = stacked.copy()
                    canonical.sum_duplicates()
                    step["transform"]["canonical"] = transform(model, canonical)
                step["transform"]["held_out"] = transform(model, original[:1])
                if name == "cutoff":
                    model.set_params(transform_mode="graph")
                    step["held_out_graph"] = transform(model, original[:1])
                    model.set_params(transform_mode="embedding")
            model.set_params(**saved)
            record["steps"].append(step)
        cases[f"{metric}_{name}"] = record

    def call(name, **options):
        return name, options

    for metric in ("manhattan", "chebyshev", "cosine"):
        capture(metric, "epochs", "base", [call("one"), call("last", n_epochs=1),
                                            call("two", n_epochs=9, transform_seed=2**32 - 1)])
        capture(metric, "warm", "base", [call("two", n_epochs=6, n_neighbors=12),
                                          call("repeat", n_epochs=18)], n_epochs=6, n_neighbors=11)
        capture(metric, "ties", "ties", [call("ties_append", n_epochs=6)], n_neighbors=3)
        capture(metric, "discrete", "discrete", [call("discrete_append", n_epochs=6)], n_neighbors=5)
        capture(metric, "split", "base", [call("split_two", n_epochs=6), call("repeat")])
        capture(metric, "split_fit", "split", [call("two"), call("split_two", n_epochs=6)])
        capture(metric, "zeros", "base", [call("zero"), call("stored_zero", n_epochs=6)])
        capture(metric, "formats", "base", [call("two_csc"), call("two_coo", n_epochs=6), call("two_dense")])
        capture(metric, "cancel", "base", [call("cancel"), call("cancel_coo", n_epochs=6)])
        capture(metric, "graph", "base", [call("one"), call("last")], transform_mode="graph")
        capture(metric, "errors", "base", [call("empty"), call("wide"), call("one", transform_seed=-1),
                                           call("one", n_epochs=-1), call("last")])
        capture(metric, "cutoff", "axes", [call("axes_append", n_epochs=6)],
                n_neighbors=7, disconnection_distance=2.0 if metric == "cosine" else 1.5)
    capture("manhattan", "precision", "precision", [call("precision_append")], n_neighbors=2)
    capture("chebyshev", "precision", "chebyshev_precision",
            [call("chebyshev_precision_append")], n_neighbors=2)
    capture("manhattan", "canonical_fit", "split_canonical", [call("two"), call("split_two_canonical", n_epochs=6)])
    capture("manhattan", "canonical_append", "base", [call("split_two_canonical", n_epochs=6), call("repeat")])
    residuals = {"manhattan_split_fit": cases.pop("manhattan_split_fit"),
                 "manhattan_split_repeat": cases["manhattan_split"]["steps"].pop()}
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": options,
        "seed": seed,
        "inputs": {name: encode(matrix) for name, matrix in inputs.items()},
        "cases": cases,
        "residuals": residuals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=565)
    args = parser.parse_args()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in build(args.seed).items()
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
        # JIT exceptions can include source excerpts.
        raise SystemExit(type(error).__name__) from None
