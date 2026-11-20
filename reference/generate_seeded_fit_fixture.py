"""Record serial dense fit checkpoints through the public UMAP estimator."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import warnings

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from umap import UMAP


def snapshot(data, initial, settings):
    records = []
    for _ in range(2):
        init = initial.copy() if isinstance(initial, np.ndarray) else initial
        model = UMAP(**settings, init=init)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            embedding = model.fit_transform(data.copy())
        assert not captured, (settings, captured)
        graph = model.graph_
        assert graph.shape == (len(data), len(data))
        assert graph.data.dtype == np.float32
        assert embedding.shape == (len(data), settings["n_components"])
        assert embedding.dtype == np.float32
        records.append(
            (
                {
                    "indptr": graph.indptr.tolist(),
                    "indices": graph.indices.tolist(),
                    "bits": graph.data.view(np.uint32).tolist(),
                },
                embedding.view(np.uint32).tolist(),
            )
        )
    assert records[0] == records[1], settings
    return records[0]


def generate():
    inputs = {
        "weighted": np.array(
            [
                [0, 0, 0],
                [0.2, 0.07, 0.1],
                [0.25, 0.89, 0.13],
                [0.76, 0.1, 0.83],
                [0.99, 0.95, 0.37],
                [1.9, 1.31, 0.33],
                [2.61, 1.7, 0.9],
                [3.3, 1.2, 1.1],
            ],
            dtype=np.float32,
        ),
        "isolated": np.array(
            [[0, 0], [0.125, 0], [5, 4], [5.25, 4], [9, 9], [9.5, 9]],
            dtype=np.float32,
        ),
    }
    explicit = np.array(
        [[0, 6], [1, 3], [4, 0], [2, 1], [5, 4], [6, 5], [3, 2], [7, 7]],
        dtype=np.float32,
    )
    common_epochs = [0, 1, 2, 3, 11, 30, 500]
    scenarios = [
        ("weighted_explicit_min_seed", "weighted", explicit, 0, 4, 2, 2, 1, common_epochs),
        ("weighted_random_max_seed", "weighted", "random", 2**32 - 1, 4, 2, 5, 0.37, common_epochs),
        ("weighted_random_mid_seed", "weighted", "random", 42, 7, 3, 3, 1, [0, 2, 3, 10, 30]),
        ("isolated_random", "isolated", "random", 7, 2, 1, 2, 1, [0, 1, 2, 3, 30]),
    ]

    graphs = {}
    cases = []
    for name, input_key, initial, seed, k, dim, rate, repulsion, epochs in scenarios:
        settings = {
            "n_neighbors": k,
            "n_components": dim,
            "metric": "euclidean",
            "n_jobs": 1,
            "random_state": seed,
            "compatibility_layout": True,
            "learning_rate": 0.13,
            "repulsion_strength": repulsion,
            "negative_sample_rate": rate,
            "a": 1.7,
            "b": 0.9,
        }
        graph_key = f"{input_key}_k{k}"
        checkpoints = {}
        for epoch in epochs:
            graph, embedding = snapshot(inputs[input_key], initial, dict(settings, n_epochs=epoch))
            assert graph == graphs.setdefault(graph_key, graph), (name, epoch, "graph changed")
            checkpoints[str(epoch)] = embedding
        if 1 in epochs:
            assert checkpoints["0"] == checkpoints["1"], name
        if 500 in epochs:
            assert snapshot(inputs[input_key], initial, settings) == (
                graphs[graph_key], checkpoints["500"]
            )
        cases.append(
            {
                "name": name,
                "input": input_key,
                "graph": graph_key,
                "params": settings,
                "init": initial.tolist() if isinstance(initial, np.ndarray) else initial,
                "checkpoints": checkpoints,
            }
        )

    for bad_seed in [-1, 2**32]:
        try:
            UMAP(
                n_neighbors=4, n_epochs=0, init="random", n_jobs=1, random_state=bad_seed,
                compatibility_layout=True,
            ).fit(inputs["weighted"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad seed {bad_seed}")

    return {
        "versions": {
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "umap-learn")
        },
        "inputs": {key: value.tolist() for key, value in inputs.items()},
        "graphs": graphs,
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(generate(), handle, indent=2)
        handle.write("\n")
