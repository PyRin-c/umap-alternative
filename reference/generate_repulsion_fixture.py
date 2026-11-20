"""Record deterministic public UMAP embedding checkpoints for optimizer knobs."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


DATA_ROWS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.5],
]
INIT_ROWS = [
    [-3.0, -2.0],
    [-1.0, 1.0],
    [2.0, -3.0],
    [4.0, 2.0],
    [7.0, 0.0],
]
CHECKPOINTS = (0, 2, 3)
BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 3,
    "metric": "euclidean",
    "a": 1.0,
    "b": 1.0,
    "init": "explicit_float32_array",
    "random_state": 42,
    "n_jobs": 1,
    "compatibility_layout": True,
    "transform_mode": "embedding",
}
SCENARIOS = {
    "repulsion_0_nrate_1": {"repulsion_strength": 0.0, "negative_sample_rate": 1},
    "repulsion_1_nrate_1": {"repulsion_strength": 1.0, "negative_sample_rate": 1},
    "repulsion_1_nrate_2": {"repulsion_strength": 1.0, "negative_sample_rate": 2},
}


def float32_bits(array: np.ndarray) -> list[list[int]]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(contiguous.shape).tolist()


def graph_record(graph) -> dict:
    data = np.ascontiguousarray(graph.data, dtype=np.float32)
    return {
        "format": graph.format,
        "dtype": np.dtype(graph.dtype).name,
        "shape": list(graph.shape),
        "indptr": np.asarray(graph.indptr).tolist(),
        "indices": np.asarray(graph.indices).tolist(),
        "data_bits_uint32": data.view(np.uint32).reshape(-1).tolist(),
    }


def warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {"class": type(item.message).__name__, "message": str(item.message)}
        for item in caught
    ]


def run_checkpoint(data: np.ndarray, init: np.ndarray, scenario: str, n_epochs: int) -> dict:
    parameters = dict(BASE_PARAMETERS)
    parameters.update(SCENARIOS[scenario])
    parameters.update({"n_epochs": n_epochs, "init": init})
    with warnings.catch_warnings(record=True) as first_warnings:
        warnings.simplefilter("always")
        first_model = UMAP(**parameters)
        first_embedding = first_model.fit_transform(data)
    with warnings.catch_warnings(record=True) as second_warnings:
        warnings.simplefilter("always")
        second_model = UMAP(**parameters)
        second_embedding = second_model.fit_transform(data)

    first_graph = graph_record(first_model.graph_)
    second_graph = graph_record(second_model.graph_)
    first_bits = float32_bits(first_embedding)
    second_bits = float32_bits(second_embedding)
    assert first_graph == second_graph
    assert first_bits == second_bits
    return {
        "n_epochs": n_epochs,
        "parameters": {
            **{key: value for key, value in parameters.items() if key != "init"},
            "init": "explicit_float32_array",
        },
        "warnings_first_run": warning_records(first_warnings),
        "warnings_second_run": warning_records(second_warnings),
        "graph": first_graph,
        "embedding": {
            "dtype": np.dtype(first_embedding.dtype).name,
            "shape": list(first_embedding.shape),
            "bits_uint32": first_bits,
        },
    }


def first_embedding_difference(left: dict, right: dict) -> dict:
    for left_checkpoint, right_checkpoint in zip(left["checkpoints"], right["checkpoints"]):
        left_bits = left_checkpoint["embedding"]["bits_uint32"]
        right_bits = right_checkpoint["embedding"]["bits_uint32"]
        if left_bits != right_bits:
            mismatch_positions = []
            for row, (left_row, right_row) in enumerate(zip(left_bits, right_bits)):
                for column, (left_value, right_value) in enumerate(zip(left_row, right_row)):
                    if left_value != right_value:
                        mismatch_positions.append([row, column, left_value, right_value])
            return {
                "first_different_n_epochs": left_checkpoint["n_epochs"],
                "mismatch_positions": mismatch_positions,
            }
    return {"first_different_n_epochs": None, "mismatch_positions": []}


def package_versions() -> dict[str, str]:
    return {
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


def make_fixture() -> dict:
    data = np.asarray(DATA_ROWS, dtype=np.float32)
    init = np.asarray(INIT_ROWS, dtype=np.float32)
    assert data.shape == init.shape == (5, 2)
    assert np.unique(init, axis=0).shape[0] == len(init)
    scenarios = {
        name: {
            "settings": values,
            "checkpoints": [
                run_checkpoint(data, init, name, n_epochs) for n_epochs in CHECKPOINTS
            ],
        }
        for name, values in SCENARIOS.items()
    }
    comparisons = {
        "repulsion_strength_0_vs_1_at_negative_sample_rate_1": {
            "left": "repulsion_0_nrate_1",
            "right": "repulsion_1_nrate_1",
            **first_embedding_difference(
                scenarios["repulsion_0_nrate_1"], scenarios["repulsion_1_nrate_1"]
            ),
        },
        "negative_sample_rate_1_vs_2_at_repulsion_strength_1": {
            "left": "repulsion_1_nrate_1",
            "right": "repulsion_1_nrate_2",
            **first_embedding_difference(
                scenarios["repulsion_1_nrate_1"], scenarios["repulsion_1_nrate_2"]
            ),
        },
    }
    assert comparisons["repulsion_strength_0_vs_1_at_negative_sample_rate_1"][
        "first_different_n_epochs"
    ] == 3
    assert comparisons["negative_sample_rate_1_vs_2_at_repulsion_strength_1"][
        "first_different_n_epochs"
    ] == 2
    return {
        "operation": "UMAP.fit_transform optimizer checkpoint comparison",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "base_parameters": BASE_PARAMETERS,
        "checkpoints_n_epochs": list(CHECKPOINTS),
        "input": {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": DATA_ROWS,
            "init_dtype": init.dtype.name,
            "init_values": INIT_ROWS,
        },
        "scenarios": scenarios,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(make_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
