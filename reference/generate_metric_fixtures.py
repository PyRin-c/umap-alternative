"""Record public graph-mode UMAP fixtures for Manhattan and Chebyshev metrics.

The generator uses only the public ``UMAP`` constructor,
``fit_transform``, and the returned SciPy CSR matrix.  Each metric case is
run twice and its shape, CSR structure, and float32 data bits are compared
before the JSON fixture is written.

Run with the pinned environment from the repository root::

    uv run --project reference --locked python \
        reference/generate_metric_fixtures.py \
        --output reference/fixtures/umap-learn-0.6.0-metric-graph-windows-x86_64.json
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


DATA_ROWS = [
    [0.0, 0.0],
    [5.0, 0.0],
    [3.0, 3.0],
    [0.0, 4.25],
    [4.0, 4.0],
    [7.0, 2.0],
]
METRICS = ("manhattan", "chebyshev")
BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 3,
    "init": "random",
    "random_state": 42,
    "n_jobs": 1,
    "n_epochs": 30,
    "compatibility_layout": True,
    "transform_mode": "graph",
    "set_op_mix_ratio": 1.0,
}


def float32_bits(array: np.ndarray) -> list[int]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(-1).tolist()


def csr_record(graph) -> dict:
    """Record only public fields of the graph-mode CSR result."""

    assert graph.format == "csr"
    data = np.ascontiguousarray(graph.data, dtype=np.float32)
    assert data.dtype == np.dtype(np.float32)
    return {
        "format": graph.format,
        "dtype": data.dtype.name,
        "shape": list(graph.shape),
        "indptr": np.asarray(graph.indptr).tolist(),
        "indices": np.asarray(graph.indices).tolist(),
        "data_bits_uint32": float32_bits(data),
    }


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


def ordering_witness(data: np.ndarray) -> dict:
    """Show one row whose metric neighbor ordering is intentionally distinct."""

    row = data[0]
    orders = {}
    for metric in METRICS:
        deltas = np.abs(data - row)
        distances = (
            np.sum(deltas, axis=1)
            if metric == "manhattan"
            else np.max(deltas, axis=1)
        )
        order = np.argsort(distances, kind="stable").tolist()
        orders[metric] = {
            "columns_by_including_self": order,
            "distances_float32": [float(distances[column]) for column in order],
        }
    assert orders["manhattan"]["columns_by_including_self"] != orders[
        "chebyshev"
    ]["columns_by_including_self"]
    return {"row": 0, "orders": orders}


def run_case(data: np.ndarray, metric: str) -> dict:
    parameters = dict(BASE_PARAMETERS)
    parameters["metric"] = metric

    first = UMAP(**parameters).fit_transform(data)
    second = UMAP(**parameters).fit_transform(data)
    first_record = csr_record(first)
    second_record = csr_record(second)

    np.testing.assert_array_equal(first_record["indptr"], second_record["indptr"])
    np.testing.assert_array_equal(first_record["indices"], second_record["indices"])
    np.testing.assert_array_equal(
        first_record["data_bits_uint32"], second_record["data_bits_uint32"]
    )
    return {"metric": metric, "parameters": parameters, "expected": first_record}


def make_fixture() -> dict:
    data = np.asarray(DATA_ROWS, dtype=np.float32)
    cases = [run_case(data, metric) for metric in METRICS]
    assert cases[0]["expected"]["indices"] != cases[1]["expected"]["indices"]
    return {
        "operation": "UMAP(transform_mode='graph').fit_transform",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "input": {"dtype": data.dtype.name, "shape": list(data.shape), "values": DATA_ROWS},
        "ordering_witness": ordering_witness(data),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this path.")
    args = parser.parse_args()
    fixture = json.dumps(make_fixture(), indent=2)
    if args.output is None:
        print(fixture)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(fixture + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
