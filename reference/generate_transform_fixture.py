"""Capture fixed-seed UMAP fit, transform, and inverse_transform public outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from umap import UMAP


TRAINING_ROWS = [
    [0.0, 0.0, 0.1],
    [0.9, 0.1, 0.2],
    [0.3, 1.0, 0.4],
    [1.4, 0.9, 0.3],
    [0.2, 0.6, 1.1],
    [1.0, 0.2, 1.3],
    [1.6, 1.3, 1.0],
    [0.7, 1.7, 0.8],
]
TRANSFORM_QUERIES = [[0.4, 0.3, 0.5], [1.2, 0.8, 1.1], [0.4, 1.4, 0.6]]
INVERSE_QUERIES = [[4.5, 5.7], [4.7, 6.5], [5.25, 7.0]]
PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 4,
    "metric": "euclidean",
    "output_metric": "euclidean",
    "init": "random",
    "n_epochs": 45,
    "learning_rate": 1.0,
    "min_dist": 0.1,
    "spread": 1.0,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
    "repulsion_strength": 1.0,
    "negative_sample_rate": 5,
    "transform_queue_size": 4.0,
    "random_state": 42,
    "transform_seed": 24,
    "n_jobs": 1,
    "compatibility_layout": True,
    "force_approximation_algorithm": False,
}
THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def array_record(values: np.ndarray) -> dict:
    if values.dtype != np.dtype(np.float32) or not np.isfinite(values).all():
        raise AssertionError("expected finite float32 values")
    return {
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        "bits_uint32": values.view(np.uint32).tolist(),
    }


def sparse_record(graph: csr_matrix | coo_matrix) -> dict:
    record = {
        "type": type(graph).__name__,
        "format": graph.format,
        "dtype": graph.dtype.name,
        "shape": list(graph.shape),
        "nnz": graph.nnz,
        "data_bits_uint32": array_record(graph.data)["bits_uint32"],
    }
    if isinstance(graph, csr_matrix):
        record.update(indptr=graph.indptr.tolist(), indices=graph.indices.tolist())
    elif isinstance(graph, coo_matrix):
        record.update(row=graph.row.tolist(), col=graph.col.tolist())
    else:
        raise AssertionError(f"unexpected sparse format: {graph.format}")
    return record


def observe(method, values):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = method(values.copy())
    return output, [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]


def fit(training: np.ndarray, mode: str) -> tuple[UMAP, dict]:
    model = UMAP(transform_mode=mode, **PARAMETERS)
    fitted, fit_warnings = observe(model.fit, training)
    if fitted is not model:
        raise AssertionError("fit did not return its estimator")
    checkpoint = {"returns_estimator": True, "warnings": fit_warnings}
    if mode == "embedding":
        checkpoint["embedding"] = array_record(model.embedding_)
    else:
        checkpoint["graph"] = sparse_record(model.graph_)
    return model, checkpoint


def embedding_checkpoints(training, transform_queries, inverse_queries):
    def checkpoint(method, values, input_key):
        output, call_warnings = observe(method, values)
        return output, {
            "input": input_key,
            "output": array_record(output),
            "warnings": call_warnings,
        }

    model, training_record = fit(training, "embedding")
    transformed, transform_record = checkpoint(
        model.transform, transform_queries, "inputs.transform_queries"
    )
    _, inverse_record = checkpoint(
        model.inverse_transform, inverse_queries, "inputs.inverse_queries"
    )
    _, roundtrip_record = checkpoint(
        model.inverse_transform, transformed, "cases.embedding.transform.output"
    )
    first = {
        "training": training_record,
        "transform": transform_record,
        "inverse_transform": inverse_record,
        "inverse_transform_roundtrip": roundtrip_record,
    }

    # Use a fresh fit and the original inverse inputs. Run both inverse calls
    # before any transform on the second estimator.
    repeated_model, repeated_training = fit(training, "embedding")
    _, repeated_inverse = checkpoint(
        repeated_model.inverse_transform, inverse_queries, "inputs.inverse_queries"
    )
    _, repeated_roundtrip = checkpoint(
        repeated_model.inverse_transform,
        transformed,
        "cases.embedding.transform.output",
    )
    _, repeated_transform = checkpoint(
        repeated_model.transform, transform_queries, "inputs.transform_queries"
    )
    repeated = {
        "training": repeated_training,
        "transform": repeated_transform,
        "inverse_transform": repeated_inverse,
        "inverse_transform_roundtrip": repeated_roundtrip,
    }
    return record_repetition("embedding", first, repeated)


def graph_checkpoints(training, transform_queries):
    def run():
        model, training_record = fit(training, "graph")
        transformed, transform_warnings = observe(model.transform, transform_queries)
        return {
            "training": training_record,
            "transform": {
                "input": "inputs.transform_queries",
                "output": sparse_record(transformed),
                "warnings": transform_warnings,
            },
        }

    return record_repetition("graph", run(), run())


def record_repetition(mode: str, first: dict, repeated: dict) -> dict:
    matches = {name: record == repeated[name] for name, record in first.items()}
    if not all(matches.values()):
        raise AssertionError(f"{mode} repeat changed: {matches}")
    return {"transform_mode": mode, **first, "fresh_estimator_repeat_identical": matches}


def make_fixture() -> dict:
    packages = {
        name: version(name)
        for name in (
            "umap-learn",
            "numpy",
            "scipy",
            "scikit-learn",
            "numba",
            "llvmlite",
            "pynndescent",
        )
    }
    if packages["umap-learn"] != "0.6.0":
        raise RuntimeError("this fixture requires the published umap-learn 0.6.0 wheel")

    training = np.asarray(TRAINING_ROWS, dtype=np.float32)
    transform_queries = np.asarray(TRANSFORM_QUERIES, dtype=np.float32)
    inverse_queries = np.asarray(INVERSE_QUERIES, dtype=np.float32)
    return {
        "operation": "public UMAP.fit, transform, and inverse_transform on dense input",
        "environment": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": packages,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_LIMITS},
        },
        "parameters": PARAMETERS,
        "inputs": {
            "training": array_record(training),
            "transform_queries": array_record(transform_queries),
            "inverse_queries": array_record(inverse_queries),
        },
        "cases": {
            "embedding": embedding_checkpoints(training, transform_queries, inverse_queries),
            "graph": graph_checkpoints(training, transform_queries),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(make_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
