"""Record modern layout epochs using UMAP.fit and its fitted attributes."""

import argparse
import json
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


PAIRS = [
    [0, 1, 70, 80],
    [1, 0, 90, 100],
    [70, 90, 0, 2],
    [80, 100, 2, 0],
]
CYCLE = [
    [0, 1, 3, 1],
    [1, 0, 1, 3],
    [3, 1, 0, 1],
    [1, 3, 1, 0],
]
PAIR_INIT = [[0, 0], [1, 0], [0, 6], [3, 10]]
PAIR_AXIS_INIT = [[0, 0], [1, 2], [0, 6], [3, 10]]
PAIR_NEG_INIT = [[0, 0], [1, 0], [4, -2], [4, 2]]
DENSE_PAIRS = [[0], [1], [70], [72]]
WEIGHTED_DENSE = [[0, 0], [1.2, 0.13], [0.35, 2.3], [3.45, 0.78]]
WEIGHTED_INIT = [[-3], [0.5], [2.8], [6.4]]
CYCLE_INIT = [[0], [1], [2], [5]]
CYCLE_2D_INIT = [[0, 0], [1, 1], [0.125, 0], [2, 2]]
SQUARE_INIT = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
COMMON = {
    "metric": "precomputed",
    "compatibility_layout": False,
    "init_rescale": "graph_edge",
    "init_scale_factor": 2.0,
    "n_neighbors": 2,
    "local_connectivity": 1.0,
    "learning_rate": 1.0,
    "a": 1.0,
    "b": 1.0,
    "repulsion_strength": 0.0,
    "exclude_graph_neighbors": True,
    "negative_sample_rate": 0,
    "n_jobs": 1,
    "random_state": 42,
}


def bits(values):
    assert values.dtype == np.float32
    return np.ascontiguousarray(values).view(np.uint32).ravel().tolist()


def fit(distances, coordinates, **parameters):
    config = {**COMMON, **parameters}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return UMAP(
            **config, init=np.array(coordinates, dtype=np.float32),
            n_components=len(coordinates[0]),
        ).fit(np.array(distances, dtype=np.float32))


def graph_bits(estimator):
    graph = estimator.graph_
    assert graph.format == "csr"
    return {
        "shape": list(graph.shape),
        "indptr": graph.indptr.tolist(),
        "indices": graph.indices.tolist(),
        "weights_bits": bits(graph.data),
    }


def record(distances, coordinates, **parameters):
    runs = []
    for _ in range(3):
        estimator = fit(distances, coordinates, **parameters)
        run = {
            "graph": graph_bits(estimator),
            "final_bits": bits(estimator.embedding_),
        }
        if isinstance(parameters["n_epochs"], (list, tuple)):
            run["checkpoint_bits"] = [bits(e) for e in estimator.embedding_list_]
        runs.append(run)
    return {
        "parameters": parameters,
        **runs[0],
        "repeat_bits_equal": all(run == runs[0] for run in runs),
        "different_repeats": [run for run in runs[1:] if run != runs[0]],
    }


def make_fixture():
    cases = []
    inputs = {
        "pairs": {"data": PAIRS, "init": PAIR_INIT},
        "pairs-axis": {"data": PAIRS, "init": PAIR_AXIS_INIT},
        "pairs-negative": {"data": PAIRS, "init": PAIR_NEG_INIT},
        "dense-pairs": {"data": DENSE_PAIRS, "init": PAIR_INIT},
        "dense-weighted": {"data": WEIGHTED_DENSE, "init": WEIGHTED_INIT},
        "cycle": {"data": CYCLE, "init": CYCLE_INIT},
        "cycle-2d": {"data": CYCLE, "init": CYCLE_2D_INIT},
        "square": {"data": CYCLE, "init": SQUARE_INIT},
    }

    def add(name, optimizer, graph="pairs", **config):
        data = inputs[graph]
        cases.append({
            "name": name,
            "graph_input": graph,
            **record(data["data"], data["init"], optimizer=optimizer, **config),
        })

    for optimizer in ("momentum", "adam"):
        all_epochs = [0, 1, 2, 3, 4, 5]
        add("pairs", optimizer, n_epochs=all_epochs)
        add("dense-pairs", optimizer, "dense-pairs", n_epochs=all_epochs, metric="euclidean")
        add("dense-weighted", optimizer, "dense-weighted", n_epochs=list(range(11)),
            n_neighbors=3, metric="euclidean")
        add("pairs-curve", optimizer, n_epochs=all_epochs, a=1.7, b=0.9, init_scale_factor=0.125)
        add("pairs-strong", optimizer, n_epochs=all_epochs, a=1e5, init_scale_factor=0.03125)
        add("pairs-weak", optimizer, n_epochs=all_epochs, init_scale_factor=8192.0)
        add("pairs-zero-rate", optimizer, n_epochs=all_epochs, learning_rate=0.0)
        add("pairs-collision", optimizer, n_epochs=all_epochs, init_scale_factor=1.0,
            learning_rate=6.25 if optimizer == "momentum" else 1.1972542693826895)
        single_epochs = [0, 1, 2, 3, 4] if optimizer == "momentum" else all_epochs
        add("pairs-single-negative", optimizer, "pairs-negative", n_epochs=single_epochs,
            init_scale_factor=1.0, negative_sample_rate=1, repulsion_strength=1.0,
            learning_rate=1.0 if optimizer == "momentum" else 1.1972542693826895)
        add("pairs-sparse-report", optimizer, n_epochs=[7, 2, 0, 2, 5])
        add("pairs-small-rate", optimizer, n_epochs=all_epochs, learning_rate=0.125,
            init_scale_factor=16.0)
        for epochs in (0, 1, 2):
            add("pairs-scalar-epochs", optimizer, n_epochs=epochs)
        cycle = {"n_neighbors": 3, "local_connectivity": 2.0, "init_scale_factor": 1.0}
        add("cycle-attraction", optimizer, "cycle", n_epochs=all_epochs, **cycle)
        add("square-repulsion-single", optimizer, "square", n_epochs=[0, 1, 2],
            negative_sample_rate=3, repulsion_strength=0.5, **cycle)
        add("cycle-cancellation", optimizer, "cycle", n_epochs=[0, 1, 2],
            negative_sample_rate=3, repulsion_strength=0.5, **cycle)
        add("cycle-repulsion", optimizer, "cycle", n_epochs=all_epochs,
            negative_sample_rate=5, repulsion_strength=1.0, **cycle)
        add("square-repulsion", optimizer, "square", n_epochs=all_epochs,
            negative_sample_rate=5, repulsion_strength=1.0, **cycle)
        add("cycle-repulsion-2d", optimizer, "cycle-2d", n_epochs=all_epochs,
            negative_sample_rate=5, repulsion_strength=1.0, **cycle)

    add("adam-switch-cap", "adam", n_epochs=[0, 1, 98, 99, 100, 101, 200, 201, 202],
        init_scale_factor=8192.0)
    add("adam-axis-collision", "adam", "pairs-axis", n_epochs=[0, 1, 2, 3, 4, 5],
        init_scale_factor=5.0 ** 0.5, learning_rate=1.1980142426752638)
    add("modern-default", None, n_epochs=[0, 1, 2])
    for optimizer in ("momentum", "adam"):
        add("modern-opt-in", optimizer, n_epochs=[0, 1, 2], compatibility_layout=None)

    seed_controls = [
        record(CYCLE, CYCLE_INIT, optimizer=optimizer, n_epochs=[0, 1, 2, 5],
               n_neighbors=3, local_connectivity=2.0, init_scale_factor=1.0,
               negative_sample_rate=5, repulsion_strength=1.0,
               random_state=seed, exclude_graph_neighbors=exclude)
        for optimizer in ("momentum", "adam")
        for seed in (42, 43)
        for exclude in (True, False)
    ]

    invalid = []
    for optimizer in ("momentum", "adam"):
        for options in (
            {"compatibility_layout": True},
            {"n_epochs": []}, {"n_epochs": [-1, 2]}, {"n_epochs": 2.0},
            {"learning_rate": -0.5}, {"negative_sample_rate": -1}, {"repulsion_strength": -0.5},
        ):
            parameters = {"optimizer": optimizer, "n_epochs": 2, **options}
            try:
                fit(PAIRS, PAIR_INIT, **parameters)
            except ValueError as error:
                invalid.append({"parameters": parameters, "error": type(error).__name__})
            else:
                raise AssertionError(f"expected a ValueError for {parameters}")

    return {
        "operation": "UMAP.fit",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in (
                "umap-learn", "numpy", "scipy", "numba", "llvmlite", "scikit-learn"
            )},
            "threads": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"
            )},
        },
        "common_parameters": COMMON,
        "inputs": inputs,
        "cases": cases,
        "invalid": invalid,
        "seed_controls": seed_controls,
    }


if __name__ == "__main__":
    sys.tracebacklimit = 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(make_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n")
