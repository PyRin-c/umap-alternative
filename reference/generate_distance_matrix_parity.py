"""Compare public UMAP Euclidean graphs with public precomputed distances.

The diagnostic nextafter cases intentionally perturb only the connected-dense
distance entries (1, 5) and (5, 1).  They localize float32 sensitivity; they
are not replacement fixture inputs or a production workaround.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import euclidean_distances
from umap import UMAP


CASES = {
    "connected-dense": [
        [0.00, 0.00],
        [0.12, 0.03],
        [0.24, 0.08],
        [0.05, 0.18],
        [0.17, 0.22],
        [0.29, 0.18],
        [0.11, 0.35],
    ],
    "ties-duplicates": [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ],
}
BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 4,
    "init": "random",
    "random_state": 42,
    "n_jobs": 1,
    "n_epochs": 30,
    "compatibility_layout": True,
    "transform_mode": "graph",
    "set_op_mix_ratio": 1.0,
}
EDGE_PAIRS = ((1, 5), (5, 1))


def float32_bits(array: np.ndarray) -> list[int]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(-1).tolist()


def scalar_bits(value: np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).reshape(1).view(np.uint32)[0])


def csr_record(graph) -> dict:
    return {
        "type": type(graph).__name__,
        "format": graph.format,
        "dtype": np.dtype(graph.dtype).name,
        "shape": list(graph.shape),
        "indptr": np.asarray(graph.indptr).tolist(),
        "indices": np.asarray(graph.indices).tolist(),
        "data_bits_uint32": float32_bits(graph.data),
    }


def fit_graph(values: np.ndarray, metric: str) -> dict:
    """Fit twice through the public API and assert deterministic CSR output."""

    parameters = dict(BASE_PARAMETERS)
    parameters["metric"] = metric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = UMAP(**parameters).fit_transform(values)
        second = UMAP(**parameters).fit_transform(values)
    first_record = csr_record(first)
    second_record = csr_record(second)
    assert first_record == second_record
    return first_record


def f64_ordered_distances(data: np.ndarray) -> np.ndarray:
    result = np.zeros((len(data), len(data)), dtype=np.float32)
    for i in range(len(data)):
        for j in range(len(data)):
            dx = np.float64(data[i, 0]) - np.float64(data[j, 0])
            dy = np.float64(data[i, 1]) - np.float64(data[j, 1])
            result[i, j] = np.float32(np.sqrt((dx * dx) + (dy * dy)))
    return result


def f32_pure_distances(data: np.ndarray) -> np.ndarray:
    result = np.zeros((len(data), len(data)), dtype=np.float32)
    for i in range(len(data)):
        for j in range(len(data)):
            dx = np.float32(data[i, 0] - data[j, 0])
            dy = np.float32(data[i, 1] - data[j, 1])
            dx2 = np.float32(dx * dx)
            dy2 = np.float32(dy * dy)
            result[i, j] = np.float32(np.sqrt(np.float32(dx2 + dy2)))
    return result


def distance_record(matrix: np.ndarray, row: int = 5) -> dict:
    local = [
        {
            "column": column,
            "value": float(matrix[row, column]),
            "bits_uint32": scalar_bits(matrix[row, column]),
        }
        for column in range(len(matrix))
        if column != row
    ]
    local.sort(key=lambda item: (item["value"], item["column"]))
    return {
        "dtype": np.dtype(matrix.dtype).name,
        "shape": list(matrix.shape),
        "edge_distance_bits": {
            f"{i},{j}": scalar_bits(matrix[i, j]) for i, j in EDGE_PAIRS
        },
        "row_5_local_nearest": local,
    }


def bit_mismatches(dense: dict, precomputed: dict) -> list[dict]:
    return [
        {"csr_position": index, "dense": left, "precomputed": right}
        for index, (left, right) in enumerate(
            zip(dense["data_bits_uint32"], precomputed["data_bits_uint32"])
        )
        if left != right
    ]


def compare_graphs(dense: dict, precomputed: dict) -> dict:
    structure_match = dense["shape"] == precomputed["shape"] and dense[
        "indptr"
    ] == precomputed["indptr"] and dense["indices"] == precomputed["indices"]
    return {
        "structure_match": structure_match,
        "data_bit_mismatches": bit_mismatches(dense, precomputed),
        "full_csr_bits_match": structure_match
        and dense["data_bits_uint32"] == precomputed["data_bits_uint32"],
    }


def diagnostic_ulp_cases(data: np.ndarray, dense_graph: dict, base_matrix: np.ndarray) -> list[dict]:
    cases = []
    for direction, target in (("down", 0.0), ("up", np.inf)):
        for steps in (0, 1, 2):
            matrix = base_matrix.copy()
            for _ in range(steps):
                for row, column in EDGE_PAIRS:
                    matrix[row, column] = np.nextafter(
                        matrix[row, column], target, dtype=np.float32
                    )
            graph = fit_graph(matrix, "precomputed")
            cases.append(
                {
                    "diagnostic_only": True,
                    "direction": direction,
                    "steps": steps,
                    "edge_distance_bits": {
                        f"{i},{j}": scalar_bits(matrix[i, j])
                        for i, j in EDGE_PAIRS
                    },
                    "graph_edge_bits": {
                        "1,5": graph["data_bits_uint32"][6],
                        "5,1": graph["data_bits_uint32"][19],
                    },
                    "comparison": compare_graphs(dense_graph, graph),
                }
            )
    return cases


def make_case(name: str, rows: list[list[float]]) -> dict:
    data = np.asarray(rows, dtype=np.float32)
    dense_graph = fit_graph(data, "euclidean")
    matrices = {
        "f64_ordered": f64_ordered_distances(data),
        "f32_pure": f32_pure_distances(data),
        "sklearn_euclidean_distances": np.ascontiguousarray(
            euclidean_distances(data), dtype=np.float32
        ),
    }
    matrix_results = {}
    for matrix_name, matrix in matrices.items():
        precomputed_graph = fit_graph(matrix, "precomputed")
        matrix_results[matrix_name] = {
            "distance_matrix": distance_record(matrix),
            "precomputed_graph": precomputed_graph,
            "comparison": compare_graphs(dense_graph, precomputed_graph),
        }
    result = {
        "input": {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": rows,
        },
        "dense_euclidean_graph": dense_graph,
        "distance_matrices": matrix_results,
    }
    if name == "connected-dense":
        result["diagnostic_nextafter_only_at_1_5_and_5_1"] = diagnostic_ulp_cases(
            data, dense_graph, matrices["f64_ordered"]
        )
    return result


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
    return {
        "operation": "public UMAP graph-mode Euclidean/precomputed parity",
        "distance_methods": {
            "f64_ordered": "ordered float64 dx*dx + dy*dy, sqrt, then float32 cast",
            "f32_pure": "float32 subtraction, products, accumulation, sqrt",
            "sklearn_euclidean_distances": "public sklearn.metrics.pairwise.euclidean_distances, then float32 cast",
        },
        "diagnostic_note": "nextafter cases alter only (1,5) and (5,1) to localize sensitivity; they are not production inputs or a hardcoding recommendation",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "base_parameters": BASE_PARAMETERS,
        "cases": {name: make_case(name, rows) for name, rows in CASES.items()},
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
