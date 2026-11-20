"""Capture packed-feature updates from the locked public wheels."""

import argparse
import os
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import csr_matrix, issparse
from umap import UMAP

from generate_boolean_update_fixture import observe
from generate_packed_metric_fixture import write_fixture
from generate_sparse_lp_transform_fixture import encode
from generate_sparse_update_fixture import state


def byte_view(rows, fill):
    if fill is None:
        return rows.copy()
    storage = np.full(rows.size + 16, fill, dtype=np.uint8)
    view = storage[:rows.size].reshape(rows.shape)
    view[:] = rows
    return view


def append_candidate(training, batch):
    if (issparse(batch) or batch.ndim != 2 or batch.shape[1] != training.shape[1]
            or not np.isfinite(batch).all()):
        return None
    return np.concatenate((training, batch.astype(np.uint8)), axis=0)


def build():
    base = np.array([[0, 3, 128], [1, 1, 129], [2, 2, 130], [4, 3, 128], [15, 0, 192],
                     [3, 1, 224], [7, 2, 248], [128, 128, 128], [255, 16, 240], [5, 0, 128]], np.uint8)
    query = np.array([[0, 0, 0], [1, 3, 128], [2, 7, 145], [1, 0, 240]], np.uint8)
    axis = np.array([[0], [0], [1], [2], [3], [7], [4], [8], [128], [255]], np.uint8)
    rng = np.random.RandomState(294)
    ties = rng.choice(np.array([0, 1, 2, 4, 8, 128], np.uint8), (29, 3))
    inputs = {
        "base": base, "signed_train": base.view(np.int8),
        "float_train": base.astype(np.float64) + .75,
        "first": query[:1], "second": query[1:2], "last": query[2:],
        "repeat": base[[0, 0]], "query": query, "axis": axis,
        "axis_append": np.array([[1], [128]], np.uint8),
        "axis_query": np.array([[0], [1], [3], [255]], np.uint8),
        "disjoint": np.array([[16], [32]], np.uint8),
        "ties": ties[:24], "ties_append": ties[24:],
        "signed": query[2:].view(np.int8),
        "integers": np.array([[-1, 256, 300], [129, 384, -1]], np.int32),
        "float64": np.array([[-1.75, 256.75, 300.9], [129.9, 384.5, -1.5]], np.float64),
        "float16": query[2:].astype(np.float16) + .75,
        "uint64": np.array([[2**33 + 31, 2**24 + 5, 2**53 + 11]], np.uint64),
        "boolean": query[:2].astype(bool),
        "sparse": csr_matrix(query[:2]), "csc": csr_matrix(query[:2]).tocsc(),
        "short": query[:, :2], "wide": np.zeros((1, 4), np.uint8),
        "empty": query[:0], "vector": query[0], "empty_width": query[:, :0],
        "nan": np.array([[0., np.nan, np.inf]], np.float64),
        "singleton": base[:1],
    }
    options = {"n_neighbors": 4, "n_components": 2, "n_epochs": 0, "init": "random",
               "random_state": 41, "transform_seed": 0, "n_jobs": 1,
               "compatibility_layout": True, "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(metric, name, calls, *, train="base", query="query", storage=None, **parameters):
        settings = {"metric": metric, "disconnection_distance": "inf"} | parameters
        kwargs = options | settings
        if isinstance(kwargs.get("disconnection_distance"), str):
            kwargs["disconnection_distance"] = float(kwargs["disconnection_distance"])
        model = UMAP(**kwargs)
        original = byte_view(inputs[train], storage)
        query_data = byte_view(inputs[query], storage)
        record = {"training": train, "query": query, "storage": storage, "options": settings,
                  "fit": observe(lambda: model.fit(original)), "initial": state(model), "steps": []}
        cases[f"{metric}_{name}"] = record
        if "error" in record["fit"]:
            return
        training = original.astype(np.uint8)
        for batch_name, finite, overrides in calls:
            batch = byte_view(inputs[batch_name], storage)
            before = state(model)
            previous = {attr: getattr(model, attr, None) for attr in before}
            saved = {key: getattr(model, key) for key in overrides}
            model.set_params(**overrides)
            update = observe(lambda: model.update(batch, ensure_all_finite=finite), expect_none=True)
            after = state(model)
            candidate = append_candidate(training, batch)
            queries = {"original": original, "held_out": query_data}
            if batch.ndim == 2 and not issparse(batch) and batch.shape[1] == training.shape[1]:
                queries["batch"] = batch
            if candidate is not None:
                queries["combined"] = candidate
                if batch.dtype.kind == "f":
                    queries["mixed_combined"] = np.concatenate((training, batch), axis=0)
            if query_data.shape[1] > 1 and "error" not in update:
                queries["prefix"] = query_data[:, :1]
            results = {label: observe(lambda: model.transform(value), model)
                       for label, value in queries.items()}
            if (metric, name, batch_name) == ("bit_jaccard", "epochs", "first"):
                # The one-row zero query has returned finite coordinates and NaNs.
                del results["batch"]
            mode = model.transform_mode
            model.set_params(transform_mode="graph")
            graph_query = observe(lambda: model.transform(query_data), model)
            model.set_params(transform_mode=mode)
            outcome = {
                "batch": batch_name, "finite": finite, "options": overrides, "update": update,
                "after": after, "replaced": {attr: getattr(model, attr, None) is not old
                                             for attr, old in previous.items()},
                "prior_unchanged": all(old is None or encode(old) == before[attr]
                                       for attr, old in previous.items()),
                "input_unchanged": encode(batch) == encode(inputs[batch_name]),
                "transforms": results, "held_out_graph": graph_query,
            }
            if ("n_epochs" in overrides and after["embedding_"] is not None
                    and after["graph_"]["shape"][0] > after["embedding_"]["shape"][0]):
                results["retained"] = observe(lambda: model.transform(original[:1]), model)
                model.set_params(n_epochs=0)
                outcome["after_epoch_reset"] = observe(lambda: model.transform(original[:1]), model)
            model.set_params(**saved)
            record["steps"].append(outcome)
            if candidate is not None and after["graph_"] is not None and (
                after["graph_"]["shape"][0] == candidate.shape[0]
            ):
                training = candidate
        record["training_unchanged"] = encode(original) == encode(inputs[train])

    def step(batch, *, finite=True, **options):
        return batch, finite, options

    for metric in ("bit_hamming", "bit_jaccard"):
        capture(metric, "epochs", [step("first"), step("second", n_epochs=1),
                                   step("last", n_epochs=9, transform_seed=2**32-1),
                                   step("repeat", n_epochs=6)])
        capture(metric, "cap", [step("last", n_neighbors=3, n_epochs=6),
                                step("second", n_epochs=6)], n_neighbors=12, n_epochs=6)
        capture(metric, "mixed", [step("integers"), step("float64", n_epochs=6), step("uint64"),
                                  step("float16", n_epochs=1), step("signed"), step("boolean")],
                train="float_train")
        capture(metric, "ties", [step("ties_append", n_epochs=6)], train="ties", n_neighbors=5)
        capture(metric, "checkpoints", [step("second", n_epochs=[0, 3]), step("last", n_epochs=None)],
                train="signed_train")
        capture(metric, "invalid", [step("short"), step("wide"), step("empty"), step("empty_width"),
                                     step("vector"), step("sparse"), step("csc"), step("nan"),
                                     step("nan", finite=False), step("first", transform_seed=-1),
                                     step("second")])
        capture(metric, "partial", [step("first", n_epochs=-1), step("second")])
        capture(metric, "epochs_type", [step("first", n_epochs=.25), step("second")])
        capture(metric, "isolated", [step("first"), step("repeat", n_epochs=6)],
                disconnection_distance=3.0 if metric == "bit_hamming" else None)
        capture(metric, "graph", [step("first"), step("second")], transform_mode="graph")
        capture(metric, "singleton", [step("first")], train="singleton")
        capture(metric, "unique", [step("axis_append"), step("disjoint", n_epochs=6)],
                train="axis", query="axis_query", unique=True)
        for fill in (0, 255):
            capture(metric, f"storage_{fill}", [step("second", n_epochs=6), step("last")], storage=fill)
        boundary = np.float32(1 if metric == "bit_hamming" else .6931471824645996)
        for label, threshold in (("equal", boundary),
                                 ("next64", np.nextafter(np.float64(boundary), np.inf)),
                                 ("next32", np.nextafter(boundary, np.float32(np.inf)))):
            capture(metric, f"cutoff_{label}", [step("axis_append", disconnection_distance=10.0),
                                                step("disjoint", n_epochs=9)],
                    train="axis", query="axis_query", disconnection_distance=float(threshold))
    return {"packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                        "scikit-learn", "pynndescent")},
            "options": options, "inputs": {name: encode(values) for name, values in inputs.items()},
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_fixture(build(), args.output, check=args.check)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
