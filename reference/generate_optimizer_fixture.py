"""Record public UMAP optimizer checkpoints on a tiny dense data set.

This probe deliberately stays on the public ``umap-learn`` API.  Each
checkpoint is a fresh ``UMAP(...).fit_transform`` call, and the fitted
estimator's public ``graph_`` CSR matrix is recorded alongside the embedding.

Run with the pinned reference environment from the repository root, for
example::

    uv run --project reference --locked python \
        reference/generate_optimizer_fixture.py \
        --output reference/fixtures/umap-learn-0.6.0-optimizer-windows-x86_64.json
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
CHECKPOINTS = (0, 2, 3)

# Five nearby, distinct points keep the public neighbor graph small enough to
# inspect by hand while avoiding duplicate rows.  The array is explicitly
# float32 because the target API and the recorded bit patterns are float32.
DATA_ROWS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.5],
]

# A distinct, explicit initialization is passed to every public fit call.
# compatibility_layout=True normalizes this initial layout to [0, 10] per
# coordinate before the optimizer runs, making the zero-epoch checkpoint a
# useful public control output.
INIT_ROWS = [
    [-3.0, -2.0],
    [-1.0, 1.0],
    [2.0, -3.0],
    [4.0, 2.0],
    [7.0, 0.0],
]

BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 3,
    "metric": "euclidean",
    "a": 1.0,
    "b": 1.0,
    "repulsion_strength": 0.0,
    "n_jobs": 1,
    "random_state": SEED,
    "compatibility_layout": True,
}


def float32_bits(array: np.ndarray) -> list[list[int]]:
    """Return a float32 array's row-major IEEE-754 bits."""

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


def run_checkpoint(data: np.ndarray, init: np.ndarray, n_epochs: int) -> dict:
    """Run and deterministically repeat one public fit_transform checkpoint."""

    parameters = dict(BASE_PARAMETERS)
    parameters.update({"n_epochs": n_epochs, "init": init})

    first_model = UMAP(**parameters)
    first_embedding = first_model.fit_transform(data)
    second_model = UMAP(**parameters)
    second_embedding = second_model.fit_transform(data)

    assert first_embedding.shape == (len(data), BASE_PARAMETERS["n_components"])
    assert first_embedding.dtype == second_embedding.dtype == np.dtype(np.float32)
    assert np.isfinite(first_embedding).all()
    np.testing.assert_array_equal(
        float32_bits(first_embedding), float32_bits(second_embedding), strict=True
    )
    np.testing.assert_array_equal(
        graph_record(first_model.graph_), graph_record(second_model.graph_), strict=True
    )

    return {
        "n_epochs": n_epochs,
        "embedding": {
            "dtype": first_embedding.dtype.name,
            "shape": list(first_embedding.shape),
            "bits_uint32": float32_bits(first_embedding),
        },
        "graph": graph_record(first_model.graph_),
    }


def make_fixture() -> dict:
    data = np.asarray(DATA_ROWS, dtype=np.float32)
    init = np.asarray(INIT_ROWS, dtype=np.float32)
    assert data.shape == init.shape == (5, 2)
    assert np.unique(init, axis=0).shape[0] == len(init)

    return {
        "operation": "UMAP.fit_transform",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "input": {
            "dtype": data.dtype.name,
            "values": DATA_ROWS,
            "init_dtype": init.dtype.name,
            "init_values": INIT_ROWS,
        },
        "parameters": {
            **BASE_PARAMETERS,
            "init": "explicit_float32_array",
            "checkpoints_n_epochs": list(CHECKPOINTS),
        },
        "expected": {
            "checkpoints": [
                run_checkpoint(data, init, n_epochs) for n_epochs in CHECKPOINTS
            ]
        },
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
