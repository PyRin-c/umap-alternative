"""Record the output of a public UMAP call on a synthetic dataset."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this path.")
    args = parser.parse_args()
    fixture = json.dumps(make_fixture(), indent=2)
    if args.output is None:
        print(fixture)
    else:
        args.output.write_text(fixture + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
