"""Record public 0.6.0 fits and transforms of rectangular precomputed distances."""

import argparse
import json
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from umap import UMAP


OPTIONS = dict(
    metric="precomputed", n_neighbors=4, n_components=2, n_jobs=1, random_state=42,
    compatibility_layout=True, init="random", a=1.7, b=0.9, learning_rate=.13,
    negative_sample_rate=2,
)


def matrix(values):
    values = np.asarray(values, dtype=np.float32)
    return {"shape": list(values.shape), "bits": values.view(np.uint32).ravel().tolist()}


def distances(left, right):
    delta = left[:, None, :].astype(np.float64) - right[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32)


def inputs():
    training = np.array([
        [0, 0], [.2, .07], [.25, .89], [.76, .1], [.99, .95], [1.9, 1.31],
        [2.61, 1.7], [3.3, 1.2],
    ], dtype=np.float32)
    fitted = distances(training, training)
    regular = distances(np.array([[.26, .24], [3.08, 1.31]], np.float32), training)
    tied = np.array([[4, 3, 0, 3, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], np.float32)
    combined = np.vstack((regular, fitted[3:4], tied, tied[:1]))
    feature = np.array([0, .1, .32, .6, .91, 1.4, 1.8, 2], dtype=np.float32)
    cutoff = np.array([
        [0, .7, np.nextafter(np.float32(1), np.float32(0)), 1, 2, 3, 4, 5],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    ], dtype=np.float32)
    data = {
        "base": (fitted, combined),
        "asymmetric": (fitted * np.linspace(.5, 1.7, 8, dtype=np.float32)[:, None], combined),
        "cutoff": (np.abs(feature[:, None] - feature[None, :]), cutoff),
    }
    for width in (17, 65):
        axis = np.linspace(0, 1, width, dtype=np.float32) ** 3
        ties = np.array([
            np.ones(width),
            np.resize([2, 1, 1, 3, 2], width),
            np.resize([4, 3, 0, 3, 0, 2, 0, 0, 4], width),
        ], dtype=np.float32)
        data[f"ties_{width}"] = (np.abs(axis[:, None] - axis[None, :]), ties)
    return data


def notes(caught):
    return [[item.category.__name__, str(item.message)] for item in caught]


def capture(dataset, samples, mode, epochs, **overrides):
    training, queries = samples[dataset]
    model = UMAP(**(OPTIONS | overrides), transform_mode=mode, n_epochs=epochs)
    training_input = training.copy()
    with warnings.catch_warnings(record=True) as fit_messages:
        warnings.simplefilter("always")
        model.fit(training_input)
    with warnings.catch_warnings(record=True) as identity_messages:
        warnings.simplefilter("always")
        identity = model.transform(training.copy())
    expected_identity = model.graph_ if mode == "graph" else model.embedding_
    queries_input = queries.copy()
    with warnings.catch_warnings(record=True) as transform_messages:
        warnings.simplefilter("always")
        transformed = model.transform(queries_input)
    fit = model.graph_.tocsr()
    record = {
        "dataset": dataset, "mode": mode, "epochs": epochs, "options": overrides,
        "fit_warnings": notes(fit_messages),
        "transform_warnings": notes(transform_messages),
        "identity": identity is expected_identity,
        "identity_warnings": notes(identity_messages),
        "queries_preserved": np.array_equal(queries_input.view(np.uint32), queries.view(np.uint32)),
        "graph": {"indptr": fit.indptr.tolist(), "indices": fit.indices.tolist(),
                  "data": matrix(fit.data)},
    }
    if not np.array_equal(training_input.view(np.uint32), training.view(np.uint32)):
        record["training_after"] = matrix(training_input)
    if mode == "graph":
        record["result"] = {
            "shape": list(transformed.shape), "row": transformed.row.tolist(),
            "column": transformed.col.tolist(), "data": matrix(transformed.data),
        }
    else:
        record["fitted"] = matrix(model.embedding_)
        record["result"] = matrix(transformed)
    return record


def capture_input_failures(training, queries):
    negative = queries.copy()
    negative[0, 0] = -np.nextafter(np.float32(0), np.float32(1))
    cases = {
        "negative": negative, "short": queries[:, :-1],
        "wide": np.pad(queries, ((0, 0), (0, 1))),
        "empty": queries[:0], "nan": np.full_like(queries, np.nan),
        "infinite": np.full_like(queries, np.inf),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UMAP(**OPTIONS, transform_mode="graph", n_epochs=6).fit(training.copy())
    expected = {}
    for name, values in cases.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.transform(values)
            except (AssertionError, ValueError) as error:
                expected[name] = {"type": type(error).__name__, "message": str(error),
                                  "warnings": notes(caught)}
            else:
                raise AssertionError(f"{name} did not raise")
    return expected


def build():
    samples = inputs()
    cases = {}
    for epochs in (0, 1, 2, 3, 6, 30, [0, 2, 6]):
        cases[f"embedding_{epochs}"] = capture("base", samples, "embedding", epochs)
    for mode in ("graph", "embedding"):
        for k in (2, 4, 15):
            cases[f"asymmetric_{mode}_{k}"] = capture(
                "asymmetric", samples, mode, 6, n_neighbors=k,
            )
        for cutoff in (.7, 1.0):
            cases[f"cutoff_{mode}_{cutoff}"] = capture(
                "cutoff", samples, mode, 6, disconnection_distance=cutoff,
            )
    for connectivity in (0.0, 1.5, 2.5):
        cases[f"connectivity_{connectivity}"] = capture(
            "base", samples, "graph", 6, local_connectivity=connectivity,
        )
    for dataset in ("ties_17", "ties_65"):
        cases[dataset] = capture(dataset, samples, "graph", 6)
    cases["cutoff_embedding_zero_epochs"] = capture(
        "cutoff", samples, "embedding", 0, disconnection_distance=1.0,
    )
    cases["cutoff_graph_all_disconnected"] = capture(
        "cutoff", samples, "graph", 6, disconnection_distance=0.0,
    )
    return {
        "system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                      "scikit-learn", "pynndescent")},
        "options": OPTIONS,
        "inputs": {key: {"training": matrix(training), "queries": matrix(queries)}
                   for key, (training, queries) in samples.items()},
        "cases": cases, "errors": capture_input_failures(*samples["base"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = build()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in results.items()
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
        # Numba can include upstream implementation details in diagnostics.
        raise SystemExit(type(error).__name__) from None
