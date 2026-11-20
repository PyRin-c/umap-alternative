"""Capture dense graph-mode UMAP with categorical and partially labeled targets."""

from __future__ import annotations

import argparse
import json
import os
import platform
from importlib.metadata import distributions, version
from pathlib import Path
from tempfile import TemporaryDirectory


DATA_ROWS = [
    [0.0, 0.0],
    [1.0, 0.125],
    [0.25, 1.25],
    [1.75, 2.125],
    [3.0, 1.375],
    [3.5, 3.25],
    [4.25, 0.375],
    [4.875, 2.25],
]
CATEGORICAL_LABELS = [10, 20, 10, 20, 10, 20, 10, 20]
PARTIAL_LABELS = [10, 20, 10, -1, -1, 20, 10, 20]
ALL_UNLABELED = [-1] * len(DATA_ROWS)
CASES = (
    ("unsupervised", None, 0.5),
    ("categorical", CATEGORICAL_LABELS, 0.5),
    ("categorical_zero_weight", CATEGORICAL_LABELS, 0.0),
    ("categorical_full_weight", CATEGORICAL_LABELS, 1.0),
    ("semisupervised", PARTIAL_LABELS, 0.5),
    ("semisupervised_lower_weight", PARTIAL_LABELS, 0.25),
    ("semisupervised_full_weight", PARTIAL_LABELS, 1.0),
    ("all_unlabeled", ALL_UNLABELED, 0.5),
)
BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 4,
    "metric": "euclidean",
    "init": "random",
    "random_state": 42,
    "n_jobs": 1,
    "n_epochs": 30,
    "compatibility_layout": True,
    "transform_mode": "graph",
    "force_approximation_algorithm": False,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
    "target_metric": "categorical",
}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def csr_record(graph) -> dict:
    """Record the public CSR arrays in their observed order."""
    if graph.format != "csr" or graph.dtype.name != "float32":
        raise RuntimeError("Expected a float32 CSR graph")
    return {
        "type": type(graph).__name__,
        "format": graph.format,
        "dtype": graph.dtype.name,
        "shape": list(graph.shape),
        "nnz": graph.nnz,
        "indptr_dtype": graph.indptr.dtype.name,
        "indptr": graph.indptr.tolist(),
        "indices_dtype": graph.indices.dtype.name,
        "indices": graph.indices.tolist(),
        "data_bits_uint32": graph.data.view("uint32").tolist(),
    }


def make_fixture() -> dict:
    # Import after the caller sets thread limits and the temporary Numba cache.
    import numpy as np
    from umap import UMAP

    data = np.asarray(DATA_ROWS, dtype=np.float32)
    cases = {}
    for name, labels, target_weight in CASES:
        records = []
        for _ in range(2):
            y = None if labels is None else np.array(labels, dtype=np.int64)
            model = UMAP(**BASE_PARAMETERS, target_weight=target_weight)
            graph = model.fit_transform(data.copy(), y=y)
            if graph is not model.graph_:
                raise RuntimeError(f"{name}: fit_transform did not return graph_")
            records.append(csr_record(graph))
        first, repeated = records
        if first != repeated:
            raise RuntimeError(f"{name}: independent fits differed")
        cases[name] = {
            "target_weight": target_weight,
            "y": (
                None
                if y is None
                else {"dtype": y.dtype.name, "shape": list(y.shape), "values": labels}
            ),
            "expected": first,
        }

    return {
        "operation": "UMAP(transform_mode='graph').fit_transform(X, y=y)",
        "environment": {
            "platform": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": dict(
                sorted((item.metadata["Name"], item.version) for item in distributions())
            ),
            "thread_environment": THREAD_ENVIRONMENT,
        },
        "base_parameters": BASE_PARAMETERS,
        "input": {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": data.tolist(),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if version("umap-learn") != "0.6.0":
        parser.error("The reference environment requires umap-learn==0.6.0")

    os.environ.update(THREAD_ENVIRONMENT)
    with TemporaryDirectory(prefix="umap-supervised-graph-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        fixture = make_fixture()
    args.output.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
