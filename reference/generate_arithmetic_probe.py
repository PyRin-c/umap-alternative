"""Record tiny public UMAP checkpoints for optimizer arithmetic probes.

This generator intentionally uses only the public ``umap-learn`` API:
``UMAP(...)``, ``fit_transform``, and the returned public ``graph_`` CSR
matrix.  It does not inspect UMAP source, private attributes, or compiled
implementation details.

The two cases differ only in the explicit ``a``/``b`` curve parameters.  Each
case records the public graph and fresh ``n_epochs=0`` and ``n_epochs=2``
outputs as exact float32 bit patterns.

Run from the repository root with the locked reference environment, for
example::

    $env:OMP_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:NUMBA_NUM_THREADS = "1"
    uv run --project reference --locked python reference/generate_arithmetic_probe.py \
        --output-dir reference/fixtures
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


SEED = 42
CHECKPOINTS = (0, 2)

# Four distinct float32 points have all three other points in each selected
# neighborhood.  Their pairwise distances are unequal, while the resulting
# ten-edge CSR graph remains small enough to inspect by hand.
DATA_ROWS = [
    [0.0, 0.0],
    [1.2, 0.13],
    [0.35, 2.3],
    [3.45, 0.78],
]

# An explicit one-dimensional layout with unequal spacing makes the public
# compatibility normalization and the first optimizer updates observable.
INIT_ROWS = [
    [-3.0],
    [0.5],
    [2.8],
    [6.4],
]

COMMON_PARAMETERS = {
    "n_components": 1,
    "n_neighbors": 3,
    "metric": "euclidean",
    "min_dist": 0.1,
    "spread": 1.0,
    "learning_rate": 1.0,
    "negative_sample_rate": 5,
    "repulsion_strength": 0.0,
    "n_jobs": 1,
    "random_state": SEED,
    "compatibility_layout": True,
}

CASES = (
    {
        "name": "base-a1-b1",
        "filename": "umap-learn-0.6.0-arithmetic-4x1-base-windows-x86_64.json",
        "a": 1.0,
        "b": 1.0,
    },
    {
        "name": "explicit-a1.7-b0.9",
        "filename": "umap-learn-0.6.0-arithmetic-4x1-ab17-09-windows-x86_64.json",
        "a": 1.7,
        "b": 0.9,
    },
)


def float32_bits(array: np.ndarray) -> list[list[int]]:
    """Return a row-major float32 array's IEEE-754 bits."""

    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(contiguous.shape).tolist()


def graph_record(graph) -> dict:
    """Record only public CSR fields from ``UMAP.graph_``."""

    assert graph.format == "csr"
    data = np.ascontiguousarray(graph.data, dtype=np.float32)
    return {
        "format": graph.format,
        "dtype": data.dtype.name,
        "shape": list(graph.shape),
        "indptr": np.asarray(graph.indptr).tolist(),
        "indices": np.asarray(graph.indices).tolist(),
        "data_bits_uint32": data.view(np.uint32).reshape(-1).tolist(),
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


def run_checkpoint(
    data: np.ndarray, init: np.ndarray, case: dict, n_epochs: int
) -> dict:
    """Run and deterministically repeat one public fit_transform checkpoint."""

    parameters = dict(COMMON_PARAMETERS)
    parameters.update(
        {
            "a": case["a"],
            "b": case["b"],
            "n_epochs": n_epochs,
            "init": init,
        }
    )

    first_model = UMAP(**parameters)
    first_embedding = first_model.fit_transform(data)
    second_model = UMAP(**parameters)
    second_embedding = second_model.fit_transform(data)

    assert first_embedding.shape == (len(data), COMMON_PARAMETERS["n_components"])
    assert first_embedding.dtype == second_embedding.dtype == np.dtype(np.float32)
    assert np.isfinite(first_embedding).all()
    assert float32_bits(first_embedding) == float32_bits(second_embedding)
    assert graph_record(first_model.graph_) == graph_record(second_model.graph_)

    return {
        "n_epochs": n_epochs,
        "embedding": {
            "dtype": first_embedding.dtype.name,
            "shape": list(first_embedding.shape),
            "bits_uint32": float32_bits(first_embedding),
        },
        "graph": graph_record(first_model.graph_),
        "repeat_bits_equal": True,
    }


def make_fixture(case: dict) -> dict:
    data = np.asarray(DATA_ROWS, dtype=np.float32)
    init = np.asarray(INIT_ROWS, dtype=np.float32)
    assert data.shape == (4, 2)
    assert init.shape == (4, 1)
    assert np.unique(data, axis=0).shape[0] == len(data)
    assert np.unique(init, axis=0).shape[0] == len(init)

    parameters = {
        **COMMON_PARAMETERS,
        "a": case["a"],
        "b": case["b"],
        "init": "explicit_float32_array",
        "checkpoints_n_epochs": list(CHECKPOINTS),
    }
    return {
        "operation": "UMAP.fit_transform",
        "probe": "optimizer-arithmetic-discriminator-4x1",
        "case": case["name"],
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "input": {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": DATA_ROWS,
            "bits_uint32": float32_bits(data),
            "init_dtype": init.dtype.name,
            "init_shape": list(init.shape),
            "init_values": INIT_ROWS,
            "init_bits_uint32": float32_bits(init),
        },
        "parameters": parameters,
        "expected": {
            "checkpoints": [
                run_checkpoint(data, init, case, n_epochs)
                for n_epochs in CHECKPOINTS
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write one JSON fixture per case to this directory.",
    )
    args = parser.parse_args()

    fixtures = [make_fixture(case) for case in CASES]
    if args.output_dir is None:
        print(json.dumps({"fixtures": fixtures}, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case, fixture in zip(CASES, fixtures, strict=True):
        path = args.output_dir / case["filename"]
        path.write_text(
            json.dumps(fixture, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(path)


if __name__ == "__main__":
    main()
