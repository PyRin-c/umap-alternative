"""Record the output of a public UMAP call on a synthetic dataset."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


GRAPH_CASES = {
    "connected-dense": {
        "values": [
            [0.00, 0.00],
            [0.12, 0.03],
            [0.24, 0.08],
            [0.05, 0.18],
            [0.17, 0.22],
            [0.29, 0.18],
            [0.11, 0.35],
        ],
        "parameters": {
            "n_components": 2,
            "n_neighbors": 4,
            "n_epochs": 30,
            "metric": "euclidean",
            "init": "random",
            "random_state": 42,
            "n_jobs": 1,
            "compatibility_layout": True,
        },
    },
    "ties-duplicates": {
        "values": [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        "parameters": {
            "n_components": 2,
            "n_neighbors": 4,
            "n_epochs": 30,
            "metric": "euclidean",
            "init": "random",
            "random_state": 42,
            "n_jobs": 1,
            "compatibility_layout": True,
        },
    },
}


def make_fixture():
    rows = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.1],
        [0.5, 0.0, 0.2],
        [0.5, 0.5, 0.3],
        [2.0, 2.0, 1.0],
        [2.0, 2.5, 1.1],
        [2.5, 2.0, 1.2],
        [2.5, 2.5, 1.3],
    ]
    data = np.array(rows, dtype=np.float32)
    parameters = {
        "n_components": 2,
        "n_neighbors": 3,
        "n_epochs": 30,
        "metric": "euclidean",
        "init": "random",
        "random_state": 42,
        "n_jobs": 1,
        "compatibility_layout": True,
    }

    first = UMAP(**parameters).fit_transform(data)
    second = UMAP(**parameters).fit_transform(data)
    assert first.shape == (len(data), parameters["n_components"])
    assert first.dtype == second.dtype == np.dtype(np.float32)
    assert np.isfinite(first).all()
    first_bits = first.view(np.uint32)
    np.testing.assert_array_equal(first_bits, second.view(np.uint32), strict=True)

    return {
        "operation": "UMAP.fit_transform",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": {
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
            },
        },
        "input": {"dtype": data.dtype.name, "values": rows},
        "parameters": parameters,
        "expected": {
            "dtype": first.dtype.name,
            "shape": list(first.shape),
            "bits_uint32": first_bits.tolist(),
        },
    }


def make_graph_fixture(case):
    """Record public UMAP graph_ and embedding_ outputs for a named case."""
    definition = GRAPH_CASES[case]
    rows = definition["values"]
    parameters = definition["parameters"]
    data = np.array(rows, dtype=np.float32)

    first = UMAP(**parameters).fit(data)
    second = UMAP(**parameters).fit(data)
    graph = first.graph_
    second_graph = second.graph_
    embedding = first.embedding_
    second_embedding = second.embedding_

    assert graph.format == second_graph.format == "csr"
    assert graph.data.dtype == second_graph.data.dtype == np.dtype(np.float32)
    assert embedding.dtype == second_embedding.dtype == np.dtype(np.float32)
    assert np.isfinite(embedding).all()
    assert np.isfinite(graph.data).all()
    np.testing.assert_array_equal(graph.indptr, second_graph.indptr, strict=True)
    np.testing.assert_array_equal(graph.indices, second_graph.indices, strict=True)
    np.testing.assert_array_equal(
        graph.data.view(np.uint32), second_graph.data.view(np.uint32), strict=True
    )
    np.testing.assert_array_equal(
        embedding.view(np.uint32), second_embedding.view(np.uint32), strict=True
    )

    return {
        "operation": "UMAP.fit",
        "case": case,
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": {
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
            },
        },
        "input": {"dtype": data.dtype.name, "values": rows},
        "parameters": parameters,
        "expected": {
            "graph": {
                "format": graph.format,
                "dtype": graph.data.dtype.name,
                "shape": list(graph.shape),
                "indptr": graph.indptr.tolist(),
                "indices": graph.indices.tolist(),
                "data_bits_uint32": graph.data.view(np.uint32).tolist(),
            },
            "embedding": {
                "dtype": embedding.dtype.name,
                "shape": list(embedding.shape),
                "bits_uint32": embedding.view(np.uint32).tolist(),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this path.")
    parser.add_argument(
        "--case",
        choices=("legacy", *GRAPH_CASES),
        default="legacy",
        help="Fixture case to record; the default preserves the original 8-row case.",
    )
    args = parser.parse_args()
    result = make_fixture() if args.case == "legacy" else make_graph_fixture(args.case)
    fixture = json.dumps(result, indent=2)
    if args.output is None:
        print(fixture)
    else:
        args.output.write_text(fixture + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
