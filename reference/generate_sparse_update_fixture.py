"""Record sparse Euclidean updates through the pinned public UMAP wheel."""

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
from scipy.sparse import csr_array, csr_matrix, vstack
from umap import UMAP

from generate_sparse_lp_transform_fixture import encode, warning_records


def repeated_columns(matrix):
    data = np.empty(matrix.data.size * 2, dtype=matrix.dtype)
    data[::2] = matrix.data / 4
    data[1::2] = matrix.data - data[::2]
    return csr_matrix((data, np.repeat(matrix.indices, 2), matrix.indptr * 2), shape=matrix.shape)


def state(model):
    return {name: encode(getattr(model, name)) if hasattr(model, name) else None
            for name in ("graph_", "embedding_")}


def build():
    rng = np.random.RandomState(1353)
    values = rng.normal(size=(9, 6)).astype(np.float32)
    values[rng.rand(*values.shape) < .45] = 0
    values[3] = 0
    base = csr_matrix(values)
    other = rng.normal(size=(3, 6)).astype(np.float32)
    other[rng.rand(*other.shape) < .5] = 0
    batch = csr_matrix(other)
    closer = csr_matrix(np.array([[0, 0, .965, 0, .225, 0], [0, 0, .966, 0, .226, 0]], np.float32))
    ties = csr_matrix(np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [2, 0],
                                [0, 2], [0, 0], [2, 0], [0, 0], [0, 0], [0, 0]], np.float32))
    cutoff = csr_matrix(np.array([[0, 0], [.1, 0], [.2, 0], [.3, 0], [.4, 0],
                                  [.5, 0], [9, 0], [.7, 0]], np.float32))
    discrete = csr_matrix(rng.choice(np.array([-1, 0, 1], np.float32), (30, 6)))
    eye = csr_matrix(np.eye(9, dtype=np.float32))
    columns = np.array([0, 200_000, 7, 33, 200_000, 7, 39, 41, 200_000])
    wide = csr_matrix((np.array([.5, 1.5, -.5, 1.25, .2, 2., .3, 2.57, 2.], np.float32),
                       (np.arange(columns.size), columns)), shape=(columns.size, 200_001))
    inputs = {
        "base": base, "base_csc": base.tocsc(), "base_coo": base.tocoo(),
        "base_split": repeated_columns(base), "base_f64": base.astype(np.float64),
        "pair": batch[:2], "last": batch[2:], "one": batch[:1], "closer": closer,
        "pair_csc": batch[:2].tocsc(), "pair_coo": batch[:2].tocoo(), "pair_dense": other[:2],
        "pair_array": csr_array(batch[:2]),
        "pair_split": repeated_columns(batch[:2]), "duplicate_row": base[1:2],
        "zero_row": csr_matrix((1, 6), dtype=np.float32),
        "stored_zero_row": csr_matrix((np.array([-0., 0.], np.float32), [0, 5], [0, 2]), shape=(1, 6)),
        "empty": csr_matrix((0, 6), dtype=np.float32), "zero_features": csr_matrix((1, 0)),
        "narrow": batch[:, :5], "wider": csr_matrix((2, 7)),
        "nan": csr_matrix((np.array([np.nan], np.float32), [0], [0, 1]), shape=(1, 6)),
        "singleton": base[:1], "two": base[:2], "ties": ties, "cutoff": cutoff,
        "tie_append": ties[[0, 2, 4, 7]], "cutoff_append": cutoff[[6, 2]],
        "cutoff_one": csr_matrix([[0.25, 0.]], dtype=np.float32),
        "discrete": discrete[:24], "discrete_append": discrete[24:],
        "identity": eye[:8], "identity_append": eye[8:],
        "wide": wide[:8], "wide_append": wide[8:],
        "two_equal": base[[1, 1]], "all_zero": csr_matrix((8, 6), dtype=np.float32),
    }
    options = {"metric": "euclidean", "n_neighbors": 4, "n_components": 2,
               "random_state": 31, "transform_seed": 0, "n_epochs": 0,
               "init": "random", "n_jobs": 1, "compatibility_layout": True,
               "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(name, training, calls, **overrides):
        model = UMAP(**(options | overrides))
        initial = inputs[training].copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.fit(initial)
            except Exception as error:
                fit_error = type(error).__name__
            else:
                fit_error = None
        record = {"training": training, "options": overrides, "fit_error": fit_error,
                  "fit_warnings": warning_records(caught), "fit": state(model), "steps": []}
        for batch_name, parameters, kwargs in calls:
            original = {key: getattr(model, key) for key in parameters}
            model.set_params(**parameters)
            old_graph = getattr(model, "graph_", None)
            old_embedding = getattr(model, "embedding_", None)
            before = state(model)
            appended = inputs[batch_name].copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = model.update(appended, **kwargs)
                except Exception as error:
                    status = type(error).__name__
                else:
                    if result is not None:
                        raise AssertionError("update return")
                    status = "ok"
            step = {"batch": batch_name, "options": parameters, "kwargs": kwargs,
                    "status": status, "warnings": warning_records(caught), "after": state(model),
                    "graph_replaced": getattr(model, "graph_", None) is not old_graph,
                    "embedding_replaced": getattr(model, "embedding_", None) is not old_embedding,
                    "previous_graph_unchanged": old_graph is None or encode(old_graph) == before["graph_"],
                    "previous_embedding_unchanged": old_embedding is None or encode(old_embedding) == before["embedding_"],
                    "batch_unchanged": encode(appended) == encode(inputs[batch_name])}
            if fit_error is None and training != "singleton":
                potential = {"old_input": initial}
                if appended.shape[1] == initial.shape[1] and appended.shape[0]:
                    potential["combined_input"] = vstack((initial, csr_matrix(appended)), format="csr")
                step["transform"] = {}
                for label, candidate in potential.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            result = model.transform(candidate.copy())
                        except Exception as error:
                            outcome = type(error).__name__
                        else:
                            if result is getattr(model, "embedding_", None):
                                outcome = "embedding"
                            elif result is getattr(model, "graph_", None):
                                outcome = "graph"
                            else:
                                outcome = list(result.shape)
                    step["transform"][label] = outcome
                if status == "ok":
                    initial = potential["combined_input"]
            record["steps"].append(step)
            model.set_params(**original)
        cases[name] = record

    def call(name, *, kwargs=None, **options):
        return name, options, kwargs or {}

    for epochs in (None, 0, 1, 2, 6, 11, 18):
        capture(f"epochs_{epochs}", "base", [call("one", n_epochs=epochs)])
    capture("warm", "base", [call("pair", n_epochs=6, transform_seed=2**32 - 1),
                             call("last", n_epochs=9, repulsion_strength=.7, negative_sample_rate=4)], n_epochs=9)
    capture("near_pair", "base", [call("closer"), call("closer", n_epochs=9)], n_neighbors=2)
    for name in ("base_csc", "base_coo", "base_f64", "base_split"):
        capture(name, name, [call("pair"), call("pair_split", n_epochs=6)])
    for name in ("pair_csc", "pair_coo", "pair_dense", "pair_array", "pair_split", "duplicate_row",
                 "zero_row", "stored_zero_row"):
        capture(name, "base", [call(name, n_epochs=6)])
    for k in (2, 3, 4, 6, 12, 19):
        capture(f"ties_{k}", "ties", [call("tie_append", n_epochs=6, n_neighbors=7)], n_neighbors=k)
    for k in (3, 6):
        capture(f"discrete_{k}", "discrete", [call("discrete_append", n_epochs=6)], n_neighbors=k)
    capture("identity_ties", "identity", [call("identity_append", n_epochs=6)], n_neighbors=3)
    capture("wide", "wide", [call("wide_append", n_epochs=6)], n_neighbors=3)
    capture("graph_options", "base", [call("pair", n_epochs=6, local_connectivity=.67,
                                          set_op_mix_ratio=.37, a=1.7, learning_rate=.4)])
    for k in (8, 9, 10, 11):
        capture(f"effective_neighbors_{k}", "base", [call("pair", n_epochs=6, n_neighbors=12)], n_neighbors=k)
    for name in ("empty", "zero_features", "narrow", "wider", "nan"):
        capture(name, "base", [call(name), call("one", n_epochs=3)])
    capture("nan_allowed", "base", [call("nan", kwargs={"ensure_all_finite": False}), call("one")])
    capture("graph_only", "base", [call("one"), call("last")], transform_mode="graph")
    capture("singleton", "singleton", [call("one")])
    capture("two_rows", "two", [call("one")])
    capture("two_equal", "two_equal", [call("duplicate_row", n_epochs=6), call("one")])
    capture("all_zero", "all_zero", [call("stored_zero_row", n_epochs=6)], n_neighbors=2)
    capture("cutoff_one", "cutoff", [call("cutoff_one", n_epochs=6)], disconnection_distance=.19)
    capture("cutoff_tie", "cutoff", [call("cutoff_append", n_epochs=6)], disconnection_distance=.19, n_neighbors=2)
    capture("epochs_list", "base", [call("one", n_epochs=[0, 1]), call("last")])
    capture("epochs_list_positive", "base", [call("one", n_epochs=[0, 6])])
    capture("negative_epochs", "base", [call("one", n_epochs=-1), call("last")])
    capture("noninteger_epochs", "base", [call("one", n_epochs=1.5), call("last")])
    capture("bad_seed", "base", [call("one", transform_seed=-1), call("last")])
    residuals = {"nan_allowed": cases.pop("nan_allowed")}
    for name in ("cutoff_tie", "negative_epochs", "noninteger_epochs"):
        for step in cases[name]["steps"]:
            step["stale_layout_transform"] = step["transform"].pop("combined_input")
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": options,
        "inputs": {name: encode(matrix) | {"type": type(matrix).__name__}
                   for name, matrix in inputs.items()},
        "cases": cases,
        "residuals": residuals,
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
        # Keep JIT diagnostics from exposing upstream source excerpts.
        raise SystemExit(type(error).__name__) from None
