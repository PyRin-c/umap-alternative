"""Record a public-wheel attraction probe with two independent edges.

The probe deliberately uses only the public ``umap-learn`` API:
``UMAP(...)``, ``fit_transform``, and the fitted estimator's public ``graph_``
CSR matrix.  It does not inspect UMAP source, private attributes, or compiled
implementation details.

The input is a symmetric, finite precomputed distance matrix.  With
``n_neighbors=2`` the public graph contains exactly two disconnected mutual
pairs, ``0 <-> 1`` and ``2 <-> 3``.  Keeping the pair weights at one makes
the layout checkpoints an attraction arithmetic discriminator independent of
fuzzy-graph weight rounding.  A large finite cross-pair distance is used
because the public precomputed path rejects infinity before graph fitting.

Run from the repository root with the locked reference environment, for
example::

    $env:UV_PROJECT_ENVIRONMENT = "C:\\temp\\umap-alternative-reference-edge"
    $env:OMP_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:NUMBA_NUM_THREADS = "1"
    uv run --project reference --locked python reference/generate_isolated_edge_arithmetic.py `
      --output reference/fixtures/umap-learn-0.6.0-isolated-edge-arithmetic-windows-x86_64.json

Each checkpoint is a fresh public ``fit_transform`` call.  The generator
repeats every call and records exact float32 bit patterns for both the public
CSR graph and the embedding.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


SEED = 42
CHECKPOINTS = (0, 2, 3)

# The finite cross-pair distance leaves only one non-self neighbor in each
# row after the public n_neighbors=2 precomputed search.  The matrix is
# symmetric and has no ties among the selected pair and cross-pair entries.
DISTANCE_MATRIX_ROWS = [
    [0.0, 1.25, 100.0, 100.0],
    [1.25, 0.0, 100.0, 100.0],
    [100.0, 100.0, 0.0, 2.5],
    [100.0, 100.0, 2.5, 0.0],
]

# Unequal ranges in the two columns make compatibility-layout normalization
# observable and give the arithmetic probe both a very large and a very small
# initial coordinate range.  Values are converted to float32 before every
# public fit call.
INIT_ROWS = [
    [-12345.678, -0.00123],
    [0.1257, 1.2345],
    [9876.543, 100.0001],
    [54321.09, 1000.0007],
]

COMMON_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 2,
    "metric": "precomputed",
    "min_dist": 0.1,
    "spread": 1.0,
    "negative_sample_rate": 1,
    "repulsion_strength": 0.0,
    "n_jobs": 1,
    "random_state": SEED,
    "compatibility_layout": True,
    "transform_mode": "embedding",
}

CASES = (
    {
        "name": "a1-b1-lr1",
        "a": 1.0,
        "b": 1.0,
        "learning_rate": 1.0,
    },
    {
        "name": "a17-b09-lr1",
        "a": 1.7,
        "b": 0.9,
        "learning_rate": 1.0,
    },
    {
        "name": "a1-b1-lr325",
        "a": 1.0,
        "b": 1.0,
        "learning_rate": 3.25,
    },
)


def float32_bits(array: np.ndarray) -> list:
    """Return row-major IEEE-754 bits after an explicit float32 conversion."""

    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(contiguous.shape).tolist()


def graph_record(graph) -> dict:
    """Record only public CSR fields from ``UMAP.graph_``."""

    assert graph.format == "csr"
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


def fit_once(
    distance_matrix: np.ndarray, init: np.ndarray, case: dict, n_epochs: int
) -> tuple[np.ndarray, dict, list[dict[str, str]]]:
    parameters = dict(COMMON_PARAMETERS)
    parameters.update(
        {
            "a": case["a"],
            "b": case["b"],
            "learning_rate": case["learning_rate"],
            "n_epochs": n_epochs,
            "init": init,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = UMAP(**parameters)
        embedding = model.fit_transform(distance_matrix)
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    assert embedding.shape == (4, 2)
    assert np.isfinite(embedding).all()
    graph = graph_record(model.graph_)
    return embedding, graph, warning_records(caught)


def run_checkpoint(
    distance_matrix: np.ndarray, init: np.ndarray, case: dict, n_epochs: int
) -> dict:
    """Run one public checkpoint and a fresh deterministic repeat."""

    first_embedding, first_graph, first_warnings = fit_once(
        distance_matrix, init.copy(), case, n_epochs
    )
    second_embedding, second_graph, second_warnings = fit_once(
        distance_matrix, init.copy(), case, n_epochs
    )
    first_embedding_bits = float32_bits(first_embedding)
    second_embedding_bits = float32_bits(second_embedding)
    assert first_graph == second_graph
    assert first_embedding_bits == second_embedding_bits
    assert first_warnings == second_warnings
    return {
        "n_epochs": n_epochs,
        "graph": first_graph,
        "embedding": {
            "dtype": first_embedding.dtype.name,
            "shape": list(first_embedding.shape),
            "bits_uint32": first_embedding_bits,
        },
        "repeat_bits_equal": True,
        "warnings_first_run": first_warnings,
        "warnings_second_run": second_warnings,
    }


def graph_isolated_pairs(graph: dict) -> bool:
    return graph == {
        "format": "csr",
        "dtype": "float32",
        "shape": [4, 4],
        "indptr": [0, 1, 2, 3, 4],
        "indices": [1, 0, 3, 2],
        "data_bits_uint32": [0x3F800000] * 4,
    }


def first_bit_difference(left: list, right: list) -> dict:
    for row, (left_row, right_row) in enumerate(zip(left, right)):
        for column, (left_value, right_value) in enumerate(zip(left_row, right_row)):
            if left_value != right_value:
                return {
                    "row": row,
                    "column": column,
                    "left_uint32": left_value,
                    "right_uint32": right_value,
                }
    return {}


def make_fixture() -> dict:
    distance_matrix = np.asarray(DISTANCE_MATRIX_ROWS, dtype=np.float32)
    init = np.asarray(INIT_ROWS, dtype=np.float32)
    assert distance_matrix.shape == (4, 4)
    assert init.shape == (4, 2)
    assert np.array_equal(distance_matrix, distance_matrix.T)
    assert np.all(np.isfinite(distance_matrix))
    assert np.all(np.diag(distance_matrix) == 0.0)
    assert np.all(np.isfinite(init))
    assert np.unique(init, axis=0).shape[0] == len(init)

    scenarios = {}
    for case in CASES:
        checkpoints = [
            run_checkpoint(distance_matrix, init, case, n_epochs)
            for n_epochs in CHECKPOINTS
        ]
        assert all(graph_isolated_pairs(checkpoint["graph"]) for checkpoint in checkpoints)
        scenarios[case["name"]] = {
            "settings": {
                "a": case["a"],
                "b": case["b"],
                "learning_rate": case["learning_rate"],
            },
            "checkpoints": checkpoints,
        }

    base_checkpoints = scenarios[CASES[0]["name"]]["checkpoints"]
    comparisons = {}
    for case in CASES[1:]:
        other_checkpoints = scenarios[case["name"]]["checkpoints"]
        comparisons[f"{CASES[0]['name']}_vs_{case['name']}"] = {
            "graph_bits_match_at_all_checkpoints": all(
                left["graph"] == right["graph"]
                for left, right in zip(base_checkpoints, other_checkpoints, strict=True)
            ),
            "first_embedding_bit_difference_by_checkpoint": [
                {
                    "n_epochs": left["n_epochs"],
                    "difference": first_bit_difference(
                        left["embedding"]["bits_uint32"],
                        right["embedding"]["bits_uint32"],
                    ),
                }
                for left, right in zip(base_checkpoints, other_checkpoints, strict=True)
            ],
        }

    return {
        "operation": "public UMAP.fit_transform isolated-edge attraction checkpoints",
        "protocol": {
            "public_api_only": True,
            "fit_transform_call": "UMAP(**parameters).fit_transform(float32 precomputed distance matrix)",
            "fresh_call_per_checkpoint": True,
            "repeat_call_per_checkpoint": True,
            "n_epochs_interpretation": "total epochs for that fresh call, not a resumable checkpoint",
            "graph_interpretation": "n_neighbors=2 emits exactly 0<->1 and 2<->3 after omitting self entries",
            "precomputed_cross_distance_note": "finite 100.0 cross-pair entries are required because public precomputed validation rejects infinity",
            "normalization_interpretation": "compatibility_layout=True normalizes the explicit float32 init independently per coordinate to [0, 10] at n_epochs=0",
        },
        "environment": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "input": {
            "distance_matrix": {
                "dtype": distance_matrix.dtype.name,
                "shape": list(distance_matrix.shape),
                "values": DISTANCE_MATRIX_ROWS,
                "bits_uint32": float32_bits(distance_matrix),
            },
            "init": {
                "dtype": init.dtype.name,
                "shape": list(init.shape),
                "values": INIT_ROWS,
                "bits_uint32": float32_bits(init),
            },
        },
        "parameters": {
            **COMMON_PARAMETERS,
            "init": "explicit_float32_array",
            "checkpoints_n_epochs": list(CHECKPOINTS),
        },
        "scenarios": scenarios,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this path.")
    args = parser.parse_args()
    fixture = json.dumps(make_fixture(), indent=2, sort_keys=True)
    if args.output is None:
        print(fixture)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(fixture + "\n", encoding="utf-8", newline="\n")
        print(args.output)


if __name__ == "__main__":
    main()
