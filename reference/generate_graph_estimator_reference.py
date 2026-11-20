"""Capture the public graph-mode UMAP estimator contract."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 3,
    "metric": "euclidean",
    "init": "random",
    "random_state": 42,
    "n_jobs": 1,
    "n_epochs": 30,
    "compatibility_layout": True,
    "transform_mode": "graph",
}

DATA = {
    "baseline_float32": np.asarray(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.3], [1.7, 2.1]], dtype=np.float32
    ),
    "float64_same_values": np.asarray(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.3], [1.7, 2.1]], dtype=np.float64
    ),
    "one_sample_float32": np.asarray([[0.0, 0.0]], dtype=np.float32),
    "two_samples_float32": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    "nan_float32": np.asarray(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.3], [np.nan, 2.1]], dtype=np.float32
    ),
}


def parameters(**overrides) -> dict:
    result = dict(BASE_PARAMETERS)
    result.update(overrides)
    return result


def warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {"class": type(item.message).__name__, "message": str(item.message)}
        for item in caught
    ]


def float32_bits(array: np.ndarray) -> list[int]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous.view(np.uint32).reshape(-1).tolist()


def csr_record(graph) -> dict:
    """Record public CSR fields from graph_ or graph-mode fit_transform."""

    return {
        "type": type(graph).__name__,
        "format": graph.format,
        "dtype": np.dtype(graph.dtype).name,
        "shape": list(graph.shape),
        "indptr": np.asarray(graph.indptr).tolist(),
        "indices": np.asarray(graph.indices).tolist(),
        "data_bits_uint32": float32_bits(graph.data),
    }


def run_fit(data: np.ndarray, fit_parameters: dict) -> dict:
    model = UMAP(**fit_parameters)
    result = {"operation": "fit", "parameters": fit_parameters}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            returned = model.fit(data)
        except Exception as error:
            result.update(
                {
                    "status": "error",
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                }
            )
        else:
            result.update(
                {
                    "status": "ok",
                    "return_type": type(returned).__name__,
                    "return_is_estimator": returned is model,
                    "model_has_graph": hasattr(model, "graph_"),
                }
            )
            if hasattr(model, "graph_"):
                result["graph"] = csr_record(model.graph_)
        result["warnings"] = warning_records(caught)
    return result


def run_fit_transform(data: np.ndarray, fit_parameters: dict) -> dict:
    model = UMAP(**fit_parameters)
    result = {"operation": "fit_transform", "parameters": fit_parameters}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            returned = model.fit_transform(data)
        except Exception as error:
            result.update(
                {
                    "status": "error",
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                    "model_has_graph": hasattr(model, "graph_"),
                }
            )
        else:
            result.update(
                {
                    "status": "ok",
                    "return_type": type(returned).__name__,
                    "model_has_graph": hasattr(model, "graph_"),
                    "output_is_graph": hasattr(model, "graph_")
                    and returned is model.graph_,
                    "output": csr_record(returned),
                }
            )
            if hasattr(model, "graph_"):
                result["graph_is_output"] = model.graph_ is returned
        result["warnings"] = warning_records(caught)
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


def make_observations() -> dict:
    baseline = DATA["baseline_float32"]
    float64 = DATA["float64_same_values"]
    observations = {
        "baseline_fit": run_fit(baseline, parameters()),
        "baseline_fit_transform": run_fit_transform(baseline, parameters()),
        "n_neighbors_1": run_fit_transform(
            baseline, parameters(n_neighbors=1)
        ),
        "n_neighbors_larger_than_sample_count": run_fit_transform(
            baseline, parameters(n_neighbors=10)
        ),
        "one_sample_fit": run_fit(DATA["one_sample_float32"], parameters()),
        "one_sample_fit_transform": run_fit_transform(
            DATA["one_sample_float32"], parameters()
        ),
        "two_samples_fit": run_fit(DATA["two_samples_float32"], parameters()),
        "two_samples_fit_transform": run_fit_transform(
            DATA["two_samples_float32"], parameters()
        ),
        "invalid_metric": run_fit_transform(
            baseline, parameters(metric="definitely-not-a-metric")
        ),
        "nan_default_validation": run_fit_transform(
            DATA["nan_float32"], parameters()
        ),
        "float64_input": run_fit_transform(float64, parameters()),
    }
    baseline_bits = observations["baseline_fit_transform"]["output"][
        "data_bits_uint32"
    ]
    float64_bits = observations["float64_input"]["output"]["data_bits_uint32"]
    observations["float64_input"]["same_data_bits_as_float32_baseline"] = (
        baseline_bits == float64_bits
    )
    return observations


def make_capture() -> dict:
    observations = make_observations()
    return {
        "operation": "UMAP(transform_mode='graph') public estimator contract",
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "packages": package_versions(),
        },
        "base_parameters": BASE_PARAMETERS,
        "inputs": {
            name: {"dtype": value.dtype.name, "shape": list(value.shape), "values": value.tolist()}
            for name, value in DATA.items()
        },
        "observations": observations,
    }


def make_document(capture: dict) -> str:
    return (
        "# Public graph-estimator reference\n\n"
        "Generated exclusively through the public `umap-learn` wheel API; no "
        "source, private attributes, or tests are used. Each successful call "
        "records public CSR fields and float32 IEEE-754 data bits.\n\n"
        "Reproduce with the pinned environment:\n\n"
        "```powershell\n"
        "uv run --project reference --locked python "
        "reference/generate_graph_estimator_reference.py "
        "--output reference/graph_estimator_reference.md "
        "--json-output reference/graph_estimator_reference.json\n"
        "```\n\n"
        "The complete machine-readable capture follows.\n\n"
        "```json\n"
        + json.dumps(capture, indent=2, sort_keys=True)
        + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = make_capture()
    args.output.write_text(make_document(capture), encoding="utf-8", newline="\n")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(capture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
