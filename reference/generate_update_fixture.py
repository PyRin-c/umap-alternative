"""Capture UMAP.update via the pinned wheel's public estimator interface."""

from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
os.environ.update(THREAD_LIMITS)

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import euclidean_distances
from umap import UMAP


PYTHON_VERSION = "3.13.12"
PACKAGE_VERSIONS = {
    "umap-learn": "0.6.0",
    "numpy": "2.5.5",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "numba": "0.68.1",
    "llvmlite": "0.50.0",
    "pynndescent": "0.7.0",
}

BASE_PARAMETERS = {
    "n_neighbors": 4,
    "n_components": 2,
    "metric": "euclidean",
    "init": "random",
    "n_epochs": 30,
    "learning_rate": 0.13,
    "a": 1.7,
    "b": 0.9,
    "negative_sample_rate": 2,
    "compatibility_layout": True,
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
}

TRAINING = np.array(
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
)
NEAR_PAIR = np.array([[0.31, 0.9, 0.2], [0.305, 0.91, 0.205]], dtype=np.float32)
INPUTS = {
    "training": TRAINING,
    "one_training_row": TRAINING[:1],
    "two_training_rows": TRAINING[:2],
    "training_one_feature": TRAINING[:, :1],
    "near_one": NEAR_PAIR[:1],
    "near_second": NEAR_PAIR[1:],
    "near_pair": NEAR_PAIR,
    "near_pair_reversed": NEAR_PAIR[::-1],
    "near_pair_one_feature": NEAR_PAIR[:, :1],
    "near_pair_distinct_anchors": np.array(
        [[0.822, 0.336, 0.461], [0.703, 0.297, 0.512]], dtype=np.float32
    ),
    "far_pair": np.array([[4.1, 2.2, 1.3], [4.7, 2.0, 1.7]], dtype=np.float32),
    "duplicate": TRAINING[:1],
    "empty": np.empty((0, 3), dtype=np.float32),
    "fewer_features": np.zeros((1, 2), dtype=np.float32),
    "more_features": np.zeros((1, 4), dtype=np.float32),
    "zero_features": np.empty((1, 0), dtype=np.float32),
    "one_dimensional": NEAR_PAIR[0],
    "nan": np.array([[1, np.nan, 2]], dtype=np.float32),
    "training_distances": euclidean_distances(TRAINING),
    "update_distances": euclidean_distances(NEAR_PAIR[:1], TRAINING),
}
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)


@dataclass(frozen=True)
class Case:
    updates: tuple[str, ...] = ("near_one",)
    fit_input: str | None = "training"
    parameters: dict = field(default_factory=dict)
    sparse: bool = False
    supervised: bool = False
    compare_combined_fit: bool = False


CASES = {
    "dense_one": Case(compare_combined_fit=True),
    "dense_far_pair": Case(updates=("far_pair",), compare_combined_fit=True),
    "dense_duplicate": Case(updates=("duplicate",), compare_combined_fit=True),
    "dense_initialization_one": Case(parameters={"n_epochs": 0}, compare_combined_fit=True),
    "dense_initialization_pair": Case(
        updates=("near_pair",), parameters={"n_epochs": 0}, compare_combined_fit=True
    ),
    "dense_initialization_pair_reversed": Case(
        updates=("near_pair_reversed",),
        parameters={"n_epochs": 0},
        compare_combined_fit=True,
    ),
    "dense_initialization_pair_distinct_anchors": Case(
        updates=("near_pair_distinct_anchors",),
        parameters={"n_epochs": 0},
        compare_combined_fit=True,
    ),
    "dense_initialization_far_pair_other_fit_seed": Case(
        updates=("far_pair",), parameters={"n_epochs": 0, "random_state": 17}
    ),
    "dense_initialization_pair_other_transform_seed": Case(
        updates=("near_pair",), parameters={"n_epochs": 0, "transform_seed": 24}
    ),
    "dense_initialization_sequential": Case(
        updates=("near_one", "near_second"),
        parameters={"n_epochs": 0},
        compare_combined_fit=True,
    ),
    "dense_one_one_epoch": Case(parameters={"n_epochs": 1}),
    "dense_pair_one_epoch": Case(updates=("near_pair",), parameters={"n_epochs": 1}),
    "dense_pair_two_epochs": Case(updates=("near_pair",), parameters={"n_epochs": 2}),
    "sparse_one": Case(sparse=True, compare_combined_fit=True),
    "sparse_initialization_pair": Case(
        updates=("near_pair",),
        parameters={"n_epochs": 0},
        sparse=True,
        compare_combined_fit=True,
    ),
    "one_feature": Case(
        updates=("near_pair_one_feature",),
        fit_input="training_one_feature",
        parameters={"n_epochs": 0},
        compare_combined_fit=True,
    ),
    "neighbors_one": Case(updates=(), parameters={"n_neighbors": 1, "n_epochs": 0}),
    "neighbors_two": Case(parameters={"n_neighbors": 2, "n_epochs": 0}, compare_combined_fit=True),
    "neighbors_two_near_pair": Case(
        updates=("near_pair",),
        parameters={"n_neighbors": 2, "n_epochs": 0},
        compare_combined_fit=True,
    ),
    "neighbors_equal_training_count": Case(
        parameters={"n_neighbors": 8, "n_epochs": 0}, compare_combined_fit=True
    ),
    "neighbors_exceed_combined_count": Case(
        parameters={"n_neighbors": 12, "n_epochs": 0}, compare_combined_fit=True
    ),
    "neighbors_exceed_combined_count_pair": Case(
        updates=("near_pair",),
        parameters={"n_neighbors": 12, "n_epochs": 0},
        compare_combined_fit=True,
    ),
    "two_training_rows": Case(
        fit_input="two_training_rows", parameters={"n_epochs": 0}, compare_combined_fit=True
    ),
    "one_training_row": Case(
        fit_input="one_training_row", parameters={"n_epochs": 0}, compare_combined_fit=True
    ),
    "empty_update": Case(updates=("empty",)),
    "fewer_features": Case(updates=("fewer_features",)),
    "more_features": Case(updates=("more_features",)),
    "zero_features": Case(updates=("zero_features",)),
    "one_dimensional": Case(updates=("one_dimensional",)),
    "nan_update": Case(updates=("nan",)),
    "supervised": Case(supervised=True),
    "precomputed": Case(
        fit_input="training_distances",
        updates=("update_distances",),
        parameters={"metric": "precomputed"},
    ),
    "graph_only": Case(parameters={"transform_mode": "graph"}, compare_combined_fit=True),
    "before_fit": Case(fit_input=None),
}


def array_words(values: np.ndarray) -> tuple[str, list]:
    if values.dtype not in (np.float32, np.float64):
        raise AssertionError(f"unexpected floating-point dtype {values.dtype}")
    width = 8 * values.itemsize
    return f"bits_uint{width}", values.view(f"uint{width}").tolist()


def array_record(values: np.ndarray) -> dict:
    key, words = array_words(values)
    return {
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        key: words,
    }


def graph_record(graph: csr_matrix | coo_matrix) -> dict:
    key, words = array_words(graph.data)
    record = {
        "type": type(graph).__name__,
        "format": graph.format,
        "dtype": graph.dtype.name,
        "shape": list(graph.shape),
        "nnz": graph.nnz,
        f"data_{key}": words,
    }
    if isinstance(graph, csr_matrix):
        record.update(indptr=graph.indptr.tolist(), indices=graph.indices.tolist())
    elif isinstance(graph, coo_matrix):
        record.update(row=graph.row.tolist(), col=graph.col.tolist())
    else:
        raise AssertionError(f"unexpected sparse format {graph.format}")
    return record


def snapshot(model: UMAP) -> dict:
    return {
        "embedding": array_record(model.embedding_) if hasattr(model, "embedding_") else None,
        "graph": graph_record(model.graph_) if hasattr(model, "graph_") else None,
    }


def observe(method, rows, estimator: UMAP, **kwargs) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            returned = method(rows, **kwargs)
        except Exception as error:
            result = {
                "status": "error",
                "exception_class": type(error).__name__,
                "exception_message": str(error),
            }
        else:
            result = {
                "status": "ok",
                "return_type": type(returned).__name__,
                "returns_estimator": returned is estimator,
            }
    result["warnings"] = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    result["state"] = snapshot(estimator)
    return result


def input_matrix(rows: np.ndarray, sparse: bool) -> np.ndarray | csr_matrix:
    return csr_matrix(rows) if sparse else rows.copy()


def compare_fresh_graph(rows, parameters: dict, sparse: bool, graph: dict | None) -> dict:
    fresh = UMAP(**parameters)
    fitted = observe(fresh.fit, input_matrix(rows, sparse), fresh)
    fresh_graph = fitted["state"]["graph"]
    comparison = {
        "n_neighbors": parameters["n_neighbors"],
        "status": fitted["status"],
        "warnings": fitted["warnings"],
        "graph": fresh_graph,
        "same_graph_bits_and_order": graph is not None and graph == fresh_graph,
    }
    if fitted["status"] == "error":
        comparison["exception_class"] = fitted["exception_class"]
        comparison["exception_message"] = fitted["exception_message"]
    return comparison


def capture(case: Case) -> dict:
    parameters = BASE_PARAMETERS | case.parameters
    model = UMAP(**parameters)
    initial_state = snapshot(model)
    result = {
        "fit_input": case.fit_input,
        "fit_labels": "labels" if case.supervised else None,
        "input_format": "csr" if case.sparse else "dense",
        "parameter_overrides": case.parameters,
        "initial_state": initial_state,
        "fit": None,
        "updates": [],
    }
    combined = []
    before_update = initial_state
    if case.fit_input is not None:
        rows = INPUTS[case.fit_input]
        combined.append(rows)
        fitted = observe(
            model.fit,
            input_matrix(rows, case.sparse),
            model,
            y=LABELS.copy() if case.supervised else None,
        )
        result["fit"] = fitted
        if fitted["status"] == "error":
            return result
        before_update = fitted["state"]

    for key in case.updates:
        rows = INPUTS[key]
        updated = observe(model.update, input_matrix(rows, case.sparse), model)
        updated["input"] = key
        state = updated["state"]
        updated["public_state_changed"] = {
            name: before_update[name] != state[name] for name in state
        }
        if case.compare_combined_fit:
            combined.append(rows)
            all_rows = np.concatenate(combined)
            updated["combined_fit"] = compare_fresh_graph(
                all_rows, parameters, case.sparse, state["graph"]
            )
            clipped_neighbors = len(combined[0]) - 1
            if 2 <= clipped_neighbors < parameters["n_neighbors"]:
                clipped_parameters = parameters | {"n_neighbors": clipped_neighbors}
                updated["combined_fit_with_training_neighbor_cap"] = compare_fresh_graph(
                    all_rows, clipped_parameters, case.sparse, state["graph"]
                )
        result["updates"].append(updated)
        before_update = state
    return result


def make_fixture() -> dict:
    packages = {name: version(name) for name in PACKAGE_VERSIONS}
    if platform.python_version() != PYTHON_VERSION or packages != PACKAGE_VERSIONS:
        raise RuntimeError("run with the locked reference environment")

    observations = {}
    for name, case in CASES.items():
        first = capture(case)
        if capture(case) != first:
            raise AssertionError(f"{name}: repeated fit and update checkpoints differ")
        observations[name] = first | {"fresh_estimator_repeat_identical": True}

    return {
        "operation": "Public UMAP.fit and UMAP.update checkpoints",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": packages,
            "thread_environment": THREAD_LIMITS,
        },
        "base_parameters": BASE_PARAMETERS,
        "inputs": {name: array_record(rows) for name, rows in INPUTS.items()},
        "labels": LABELS.tolist(),
        "cases": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(make_fixture(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
