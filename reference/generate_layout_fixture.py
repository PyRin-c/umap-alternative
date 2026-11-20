"""Record public UMAP layout checkpoints and a public NumPy RNG fixture.

This probe intentionally uses only public APIs: ``UMAP(...)``,
``fit_transform``, and the returned NumPy array.  It does not inspect UMAP
source, private attributes, or compiled implementation details.

Run from the repository root with the pinned environment, for example:

    $env:OMP_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    uv run --project reference --locked python reference/generate_layout_fixture.py \
        --output reference/fixtures/umap-learn-0.6.0-layout-windows-x86_64.json

The input is deliberately small and dense so Euclidean neighbor search is
unambiguous.  ``n_epochs`` is a public *total run length*, so each checkpoint
is a fresh ``fit_transform`` call; it is not an API for resuming an optimizer
or for reading an intermediate state from one call.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


SEED = 42
CHECKPOINTS = (0, 1, 2, 3, 5, 10)

# Eight distinct points on a compact integer grid.  The array is dense,
# float32, and has no duplicate rows or distance ties that are relevant to
# the selected four-neighbor graph.
DATA_ROWS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [0.0, 2.0],
    [2.0, 2.0],
    [1.0, 2.0],
]

# Unequal ranges make per-coordinate normalization observable rather than
# accidentally looking like one global min/max operation.
EXPLICIT_INIT_ROWS = [
    [-4.0, -2.0],
    [0.0, 1.0],
    [3.0, 4.0],
    [8.0, 2.0],
    [10.0, 9.0],
    [17.0, 6.0],
    [20.0, 15.0],
    [30.0, 25.0],
]

BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 4,
    "metric": "euclidean",
    "random_state": SEED,
    "n_jobs": 1,
    "compatibility_layout": True,
}


def bits(array: np.ndarray) -> list[list[int]]:
    """Return an output array as nested row-major float32 bit patterns."""

    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(contiguous.shape).tolist()


def flat_bits(array: np.ndarray) -> list[int]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(-1).tolist()


def array_record(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
        "bits_uint32": bits(contiguous),
        "min": [float(value) for value in np.min(contiguous, axis=0)],
        "max": [float(value) for value in np.max(contiguous, axis=0)],
    }


def normalization_record(array: np.ndarray) -> dict:
    """Record the independently column-normalized [0, 10] control result."""

    initial = np.ascontiguousarray(array, dtype=np.float32)
    normalized = (
        10.0
        * (initial - initial.min(axis=0))
        / (initial.max(axis=0) - initial.min(axis=0))
    ).astype(np.float32)
    return {
        "formula": "(10 * (init - init.min(axis=0)) / (init.max(axis=0) - init.min(axis=0))).astype(float32)",
        "expected": array_record(normalized),
    }


def run_checkpoint(data: np.ndarray, init: str | np.ndarray, n_epochs: int) -> dict:
    """Run one public fit_transform call and a fresh deterministic repeat."""

    parameters = dict(BASE_PARAMETERS)
    parameters.update({"n_epochs": n_epochs, "init": init})
    warning_records: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = UMAP(**parameters).fit_transform(data)
    for warning in caught:
        warning_records.append(
            {"category": warning.category.__name__, "message": str(warning.message)}
        )

    # Pass a fresh explicit array to avoid any possibility of a callee
    # modifying the object supplied for the next public call.
    repeat_init = init.copy() if isinstance(init, np.ndarray) else init
    second = UMAP(**{**parameters, "init": repeat_init}).fit_transform(data)
    first_bits = flat_bits(first)
    second_bits = flat_bits(second)
    if first_bits != second_bits:
        raise AssertionError(f"seeded repeat changed at n_epochs={n_epochs}")
    if first.dtype != np.dtype(np.float32):
        raise AssertionError(f"unexpected UMAP output dtype: {first.dtype}")
    if not np.isfinite(first).all():
        raise AssertionError(f"non-finite UMAP output at n_epochs={n_epochs}")

    return {
        "n_epochs": n_epochs,
        "output": array_record(first),
        "repeat_bits_equal": True,
        "warnings": warning_records,
    }


def random_state_fixture() -> dict:
    """Capture only documented NumPy RandomState calls for RNG parity work."""

    state = np.random.RandomState(SEED)
    state_name, state_words, state_position, has_gauss, cached_gaussian = (
        state.get_state()
    )

    raw_rng = np.random.RandomState(SEED)
    raw_uint32 = raw_rng.randint(0, 2**32, size=16, dtype=np.uint32)

    sample_rng = np.random.RandomState(SEED)
    sample = sample_rng.random_sample(16).astype(np.float32)

    uniform_rng = np.random.RandomState(SEED)
    uniform = uniform_rng.uniform(-10.0, 10.0, size=(len(DATA_ROWS), 2)).astype(
        np.float32
    )
    normalized = (
        10.0 * (uniform - uniform.min(axis=0)) / (uniform.max(axis=0) - uniform.min(axis=0))
    ).astype(np.float32)

    return {
        "constructor": "np.random.RandomState(42)",
        "get_state_public_result": {
            "algorithm": state_name,
            "state_words_dtype": np.asarray(state_words).dtype.name,
            "state_words_count": int(len(state_words)),
            "state_words_prefix_16_uint32": np.asarray(state_words[:16], dtype=np.uint32).tolist(),
            "position": int(state_position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "randint_public_call": {
            "call": "RandomState(42).randint(0, 2**32, size=16, dtype=np.uint32)",
            "dtype": raw_uint32.dtype.name,
            "raw_uint32": raw_uint32.tolist(),
        },
        "random_sample_public_call": {
            "call": "RandomState(42).random_sample(16).astype(np.float32)",
            "dtype": sample.dtype.name,
            "shape": list(sample.shape),
            "bits_uint32": flat_bits(sample),
        },
        "uniform_public_call": {
            "call": "RandomState(42).uniform(-10.0, 10.0, size=(8, 2)).astype(np.float32)",
            "dtype": uniform.dtype.name,
            "shape": list(uniform.shape),
            "bits_uint32": bits(uniform),
            "normalized_columnwise_0_10_bits_uint32": bits(normalized),
        },
    }


def package_versions() -> dict[str, str]:
    names = (
        "umap-learn",
        "numpy",
        "scipy",
        "scikit-learn",
        "numba",
        "llvmlite",
        "pynndescent",
    )
    return {name: version(name) for name in names}


def make_fixture() -> dict:
    data = np.asarray(DATA_ROWS, dtype=np.float32)
    explicit_init = np.asarray(EXPLICIT_INIT_ROWS, dtype=np.float32)

    random_checkpoints = [
        run_checkpoint(data, "random", n_epochs) for n_epochs in CHECKPOINTS
    ]
    explicit_checkpoints = [
        run_checkpoint(data, explicit_init.copy(), n_epochs)
        for n_epochs in CHECKPOINTS
    ]

    # With this small exact-neighbor case, the public random initializer at
    # n_epochs=0 is exactly the normalized public RandomState uniform vector.
    random_uniform = np.random.RandomState(SEED).uniform(
        -10.0, 10.0, size=data.shape
    ).astype(np.float32)
    observed_random_zero = random_checkpoints[0]["output"]["bits_uint32"]
    expected_random_zero = bits(
        10.0
        * (random_uniform - random_uniform.min(axis=0))
        / (random_uniform.max(axis=0) - random_uniform.min(axis=0))
    )
    if observed_random_zero != expected_random_zero:
        raise AssertionError("random n_epochs=0 did not match public RNG normalization")

    observed_explicit_zero = explicit_checkpoints[0]["output"]["bits_uint32"]
    expected_explicit_zero = normalization_record(explicit_init)["expected"]["bits_uint32"]
    if observed_explicit_zero != expected_explicit_zero:
        raise AssertionError("explicit n_epochs=0 did not match column normalization")

    return {
        "operation": "public UMAP.fit_transform layout checkpoints",
        "protocol": {
            "public_api_only": True,
            "fit_transform_call": "UMAP(**parameters).fit_transform(float32 dense array)",
            "fresh_call_per_checkpoint": True,
            "n_epochs_interpretation": "total epochs for that call, not a resumable checkpoint",
            "normalization_interpretation": "with compatibility_layout=True, n_epochs=0 exposes a finite float32 initialization normalized independently per coordinate to [0, 10] for this probe",
            "platform_scope": f"This fixture was generated on {platform.system()} {platform.machine()}; no unrun platform is inferred from it.",
        },
        "environment": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
        "input": {
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "values": DATA_ROWS,
        },
        "parameters": {
            **BASE_PARAMETERS,
            "n_epochs": list(CHECKPOINTS),
        },
        "cases": {
            "random_init": {
                "init": "random",
                "seed": SEED,
                "checkpoints": random_checkpoints,
                "n_epochs_0_matches_public_randomstate_normalization": True,
            },
            "explicit_init": {
                "init": {
                    "dtype": explicit_init.dtype.name,
                    "shape": list(explicit_init.shape),
                    "values": EXPLICIT_INIT_ROWS,
                    "bits_uint32": bits(explicit_init),
                },
                "normalization_control": normalization_record(explicit_init),
                "checkpoints": explicit_checkpoints,
                "n_epochs_0_matches_column_normalization": True,
            },
        },
        "numpy_randomstate_fixture": random_state_fixture(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.dumps(make_fixture(), indent=2)
    args.output.write_text(fixture + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
