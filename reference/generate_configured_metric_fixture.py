"""Record configured distances and an estimator path through the pinned public API."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import warnings

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from umap import UMAP, distances


METRIC_CASES = (
    ("minkowski", {}, [.1, .2], [.4, .3]),
    ("minkowski", {"p": 3.0}, [.1, .2, .6], [.4, .3, .1]),
    ("minkowski", {"p": .5}, [0, 0], [1, 4]),
    ("wminkowski", {}, [.1, .2], [.4, .3]),
    ("weighted_minkowski", {"w": [2., 3., 0.], "p": 3.0}, [.1, .2, .6], [.4, .3, .1]),
    ("wminkowski", {"w": [0., 0.], "p": 2.0}, [1, 2], [3, 4]),
    ("seuclidean", {}, [.1, .2], [.4, .3]),
    ("standardised_euclidean", {"sigma": [4., 9., 1.]}, [.1, .2, .6], [.4, .3, .1]),
    ("mahalanobis", {}, [.1, .2], [.4, .3]),
    ("mahalanobis", {"vinv": [[4., 1.], [1., 2.]]}, [.1, .2], [.4, .3]),
    ("haversine", {}, [.1, .2], [.4, .3]),
    ("haversine", {}, [0, 0], [0, np.pi]),
    ("poincare", {}, [.1, .2], [.4, .3]),
    ("poincare", {}, [0, 0], [.999, 0]),
    ("hellinger", {}, [.1, .2], [.4, .3]),
    ("hellinger", {}, [0, 0], [0, 0]),
    ("hellinger", {}, [0, 0], [0, .3]),
    ("hellinger", {}, [.3, 0], [0, .3]),
    ("softmax_hellinger", {}, [.1, .2], [.4, .3]),
    ("softmax_hellinger", {}, [0, 0], [0, 0]),
    ("symmetric_kl", {}, [.1, .2], [.4, .3]),
    ("symmetric_kl", {"z": .05}, [.3, 0], [0, .3]),
)

TRAINING = [[0, 0], [0, 1], [1, 3], [4, 4], [2, 7], [6, 6], [7, 2], [11, 3], [9, 8]]
QUERIES = [[.25, 2.75], [7.75, 4.5]]
OPTIONS = {
    "metric": "minkowski", "metric_kwds": {"p": 1.0}, "n_neighbors": 4,
    "random_state": 19, "transform_seed": 31, "n_jobs": 1, "n_epochs": 24,
    "init": "random", "compatibility_layout": True,
    "a": 1.576943460, "b": .895060879,
}


def array_bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()


def sparse_values(graph):
    axes = ("indptr", "indices") if graph.format == "csr" else ("row", "col")
    return {
        "shape": list(graph.shape),
        "format": graph.format,
        **{name: getattr(graph, name).tolist() for name in axes},
        "bits": array_bits(graph.data),
    }


def generate():
    assert version("umap-learn") == "0.6.0"
    cases = []
    for i, (name, options, x, y) in enumerate(METRIC_CASES):
        kwargs = {key: np.asarray(value) if isinstance(value, list) else value for key, value in options.items()}
        for dtype in (np.float32, np.float64):
            value = distances.named_distances[name](np.asarray(x, dtype), np.asarray(y, dtype), **kwargs)
            cases.append({
                "id": f"{name}-{i}-{dtype.__name__}", "metric": name, "dtype": dtype.__name__,
                "left": x, "right": y, "kwds": options,
                "bits": f"{np.float64(value).view(np.uint64).item():016x}",
            })
    train, queries = (np.asarray(rows, dtype=np.float32) for rows in (TRAINING, QUERIES))
    estimator = {}
    for mode in ("graph", "embedding"):
        model = UMAP(**OPTIONS, transform_mode=mode)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fit = model.fit_transform(train.copy())
            held = model.transform(queries.copy())
        estimator[mode] = {
            "fit_graph": sparse_values(model.graph_),
            "transform": sparse_values(held) if mode == "graph" else array_bits(held),
        }
        if mode == "embedding":
            estimator[mode]["embedding"] = array_bits(fit)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                model.update(queries.copy())
            estimator[mode]["updated_graph"] = sparse_values(model.graph_)
            estimator[mode]["updated_embedding"] = array_bits(model.embedding_)
    return {"distances": cases, "estimator": estimator, "training": TRAINING, "queries": QUERIES, "options": OPTIONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = (json.dumps(generate(), indent=2, allow_nan=False) + "\n").encode()
    if args.check:
        if args.output.read_bytes() != data:
            raise SystemExit(f"fixture differs from {args.output}")
    else:
        args.output.write_bytes(data)


if __name__ == "__main__":
    main()
