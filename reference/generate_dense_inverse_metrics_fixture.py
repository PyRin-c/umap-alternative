"""Capture metric and fitted-scale controls from the public inverse estimator."""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1"
)

import numpy as np
from umap import UMAP


BASE = dict(
    n_neighbors=4, n_jobs=1, random_state=42, compatibility_layout=True,
    n_epochs=0, a=1.2, b=0.9, transform_seed=2**32 - 1,
)
TRAINING = [
    [0.12, 0.07, 1.2], [1, 4, 2], [0.78, 1.2, 3.1], [2.2, -3, 0.5],
    [-0.1, 0.7, 6], [0.3, -0.5, -0.9], [5, 6, 7],
]
EXPLICIT = [[-2, 1], [0, -3], [2, 4], [5, -2], [3, 8], [6, 1], [-4, 3]]
QUERIES = [[3, 4], [5, 5], [12, -1]]
POINTS = np.asarray(EXPLICIT, dtype=np.float32)
FIXED = 10.0 * (POINTS - POINTS.min(axis=0)) / np.ptp(POINTS, axis=0)
SCHEDULES = [(0, 0.0), (6, 0.0), (30, 0.0), (6, 1.0), (30, 1.0)]


def cases():
    for metric in ("manhattan", "chebyshev", "cosine", "correlation"):
        yield metric, TRAINING, {"metric": metric}, None
    yield "finite_cutoff", TRAINING, {"disconnection_distance": 4.0}, None
    yield "finite_cutoff_no_loss", TRAINING, {"disconnection_distance": 50.0}, None
    yield "supplied_width", TRAINING, {"local_connectivity": 1.3}, 5
    yield "supplied_cutoff", TRAINING, {"disconnection_distance": 1.7}, 4
    yield "cosine_zero", [[0, 0, 0], *TRAINING[1:]], {"metric": "cosine"}, None
    yield "correlation_constant", [[1, 1, 1], *TRAINING[1:]], {"metric": "correlation"}, None


def supplied_neighbors(training, width):
    values = training.astype(np.float64)
    distances = np.linalg.norm(values[:, None] - values[None, :], axis=-1).astype(np.float32)
    columns = np.argsort(distances, axis=1, kind="stable")[:, :width].astype(np.int32)
    # A nontrivial scale distinguishes the supplied graph from recomputation.
    return columns, np.take_along_axis(distances, columns, axis=1) * np.float32(0.37), None


def capture(rows, overrides, supplied):
    x = np.asarray(rows, dtype=np.float32)
    settings = BASE | overrides | dict(init=np.asarray(EXPLICIT, dtype=np.float32))
    if supplied is not None:
        settings["precomputed_knn"] = supplied_neighbors(x, supplied)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UMAP(**settings).fit(x)
    if supplied is not None:
        settings["precomputed_knn"][1].fill(123.0)
    # Use fixed finite geometry to isolate inverse behavior, including fits
    # whose original coordinates contained a disconnected row.
    model.embedding_ = np.asarray(FIXED, dtype=np.float32)
    query_matrix = np.asarray(QUERIES, dtype=np.float32)
    snapshots = []
    for epochs, strength in SCHEDULES:
        model.set_params(n_epochs=epochs, repulsion_strength=strength)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = model.inverse_transform(query_matrix)
        assert result.dtype == np.float32
        snapshots.append({
            "fit_epochs": epochs, "strength": strength, "bits": result.view(np.uint32).tolist(),
            "warnings": [warning.category.__name__ for warning in seen],
        })
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert version("umap-learn") == "0.6.0"
    results = {}
    for name, training, settings, supplied in cases():
        try:
            checkpoints = capture(training, settings, supplied)
            assert checkpoints == capture(training, settings, supplied)
        except Exception as error:
            raise SystemExit(f"{name}: {type(error).__name__}") from None
        results[name] = dict(training=training, settings=settings, supplied=supplied, checks=checkpoints)
    fixture = {
        "parameters": BASE, "explicit": EXPLICIT, "embedding": np.asarray(FIXED, dtype=np.float32).tolist(),
        "queries": QUERIES, "cases": results,
    }
    output = json.dumps(fixture, separators=(",", ":")) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            raise SystemExit("inverse metrics fixture differs")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
