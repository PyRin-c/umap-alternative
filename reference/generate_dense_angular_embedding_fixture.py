"""Record dense angular embeddings from the public umap-learn wheel."""

import argparse
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


GENERIC = [
    [0.12, 0.07, 1.2], [1, 4, 2], [0.78, 1.2, 3.1], [2.2, -3, 0.5],
    [-0.1, 0.7, 6], [0.3, -0.5, -0.9], [5, 6, 7],
]
QUERIES = [[0, 0, 0], [2, 2, 2], [1, 0.11, 0.75], [1, 0, -1]]
REPEATED = [
    [0, 0, 0], [0, 0, 0], [1, 1, 1], [-2, -2, -2], [1, 0, -1],
    [-1, 0, 1], [1, 0, -1], [2, 4, 1], [0.3, -0.5, 0.7],
]


def bits(array):
    assert array.dtype == np.float32
    return array.view(np.uint32).tolist()


def cases():
    yield "random_zero", GENERIC, QUERIES, {"n_epochs": 0}
    yield "random_two", GENERIC, QUERIES, {"n_epochs": 2, "transform_seed": 0}
    yield "random_thirty", GENERIC, QUERIES, {"n_epochs": 30}
    explicit = {
        "n_epochs": 9, "a": 1.2, "b": 0.9, "transform_seed": 2**32 - 1,
        "init": [[-2, 1], [0, -3], [2, 4], [5, -2], [3, 8], [6, 1], [-4, 3]],
    }
    yield "explicit", GENERIC, QUERIES, explicit
    yield "explicit_single_query", GENERIC, QUERIES[2:3], explicit
    yield "spectral", GENERIC, QUERIES, {"n_epochs": 9, "init": "spectral"}
    yield "distance_ties", REPEATED, QUERIES, {"n_epochs": 9}
    yield "connectivity_mix", GENERIC, QUERIES, {
        "n_epochs": 9, "local_connectivity": 1.5, "set_op_mix_ratio": 0.4,
    }
    yield "cutoff_at_two", [
        [1, 0], [-1, 0], [1, 0], [-1, 0],
    ], [[1, 0], [-1, 0], [0, 0]], {"n_epochs": 9, "n_neighbors": 3}
    yield "isolated_rows", [
        [1, 0, -1], [1.1, 0.05, -1], [1, -0.1, -0.8], [0, 0, 0],
        [-1, 0, 1], [-1.1, -0.05, 1], [-1, 0.1, 0.8], [2, 2, 2],
    ], [[1, 0, -1], [-1, 0, 1], [1, 1, -1]], {
        "n_epochs": 9, "disconnection_distance": 0.15,
    }


def observe(training, queries, settings):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        model = UMAP(**settings)
        fitted = model.fit_transform(training.copy())
        assert model.transform(training.copy()) is model.embedding_
        transformed = model.transform(queries.copy())
    graph = model.graph_
    return {
        "indptr": graph.indptr.tolist(),
        "indices": graph.indices.tolist(),
        "weights": bits(graph.data),
        "fitted": bits(fitted),
        "transformed": bits(transformed),
        "warnings": [warning.category.__name__ for warning in observed],
    }


def generate():
    assert version("umap-learn") == "0.6.0"
    output = {}
    for name, training, queries, options in cases():
        x = np.asarray(training, dtype=np.float32)
        q = np.asarray(queries, dtype=np.float32)
        settings = dict(
            n_neighbors=4, random_state=42, n_jobs=1, compatibility_layout=True,
            init="random",
        ) | options
        if isinstance(settings["init"], list):
            settings["init"] = np.asarray(settings["init"], dtype=np.float32)
        observed = {}
        for metric in ("cosine", "correlation"):
            try:
                first = observe(x, q, settings | {"metric": metric})
                second = observe(x, q, settings | {"metric": metric})
                assert first == second
            except Exception as exc:
                raise SystemExit(f"{name}/{metric}: {type(exc).__name__}") from None
            observed[metric] = first
        output[name] = {
            "training": training, "queries": queries, "options": options, "metrics": observed,
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = json.dumps(generate(), separators=(",", ":")) + "\n"
    if args.check:
        if args.output.read_bytes() != result.encode():
            raise SystemExit("angular embedding fixture differs")
    else:
        args.output.write_bytes(result.encode())
