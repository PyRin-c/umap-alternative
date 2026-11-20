"""Record UMAP 0.6.0 behavior when passing explicit nearest-neighbor arrays."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import tomllib
import warnings
from importlib.metadata import distributions
from pathlib import Path


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "umap-learn-0.6.0-precomputed-neighbors-windows-x86_64.json"
)
RUST_HELD_OUT_FIXTURE = FIXTURE.with_name(
    "umap-learn-0.6.0-supplied-neighbors-held-out-windows-x86_64.json"
)
PYTHON_FIXTURE = FIXTURE.with_name(
    "umap-learn-0.6.0-supplied-neighbors-python-windows-x86_64.json"
)
PYTHON_LINUX_FIXTURE = PYTHON_FIXTURE.with_name(
    "umap-learn-0.6.0-supplied-neighbors-python-linux-x86_64.json"
)
PARAMETERS = {
    "n_neighbors": 4,
    "n_components": 2,
    "n_epochs": 0,
    "random_state": 42,
    "n_jobs": 1,
    "compatibility_layout": True,
    "init": "random",
    "transform_mode": "graph",
}
THREADS = {
    name: "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
}


def pinned_environment():
    lock = tomllib.loads(Path(__file__).with_name("uv.lock").read_text(encoding="utf-8"))
    required = {
        item["name"]: item["version"] for item in lock["package"] if "registry" in item["source"]
    }
    versions = {item.metadata["Name"].lower(): item.version for item in distributions()}
    if platform.system() != "Windows":
        del required["colorama"]
    if versions != required or f"=={platform.python_version()}" != lock["requires-python"]:
        raise RuntimeError("Use the Python and packages pinned in reference/uv.lock")
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": dict(sorted(versions.items())),
        "threads": THREADS,
    }


def capture(rust_held_out=False, python_estimator=False):
    import numpy as np
    from umap import UMAP

    training = np.array([0, 0.75, 1.75, 3, 4.5, 6, 8, 8.75], dtype=np.float32).reshape(-1, 1)
    held_out = np.array([0.125, 4.125, 8.875], dtype=np.float32).reshape(-1, 1)
    full_distances = np.abs(training - training.T)
    ordered = np.argsort(full_distances, axis=1, kind="stable").astype(np.int32)
    ordered_distances = np.take_along_axis(full_distances, ordered, axis=1)
    k = PARAMETERS["n_neighbors"]
    indices = ordered[:, :k].copy()
    distances = ordered_distances[:, :k].copy()

    overridden = distances.copy()
    overridden[0, 1:] = (0.5, 0.75, 1.0)
    missing_indices, missing_distances = indices.copy(), distances.copy()
    missing_indices[0, 2:] = -1
    missing_distances[0, 2:] = np.inf
    empty_indices, empty_distances = missing_indices.copy(), missing_distances.copy()
    empty_indices[0, 1] = -1
    empty_distances[0, 1] = np.inf
    unordered_indices, unordered_distances = indices.copy(), overridden.copy()
    unordered_indices[0, [0, 1]] = unordered_indices[0, [1, 0]]
    unordered_distances[0, [0, 1]] = unordered_distances[0, [1, 0]]
    no_self_indices, no_self_distances = indices.copy(), distances.copy()
    no_self_indices[0] = ordered[0, 1 : k + 1]
    no_self_distances[0] = ordered_distances[0, 1 : k + 1]
    repeated_indices, repeated_distances = indices.copy(), distances.copy()
    repeated_indices[0, 3] = repeated_indices[0, 2]
    repeated_distances[0, 3] = repeated_distances[0, 2]
    unsorted_distances = overridden.copy()
    unsorted_distances[0, [1, 3]] = unsorted_distances[0, [3, 1]]

    def graph_record(graph):
        result = {
            "format": graph.format,
            "shape": list(graph.shape),
            "weight_bits": graph.data.view(np.uint32).tolist(),
        }
        if graph.format == "csr":
            result.update(indptr=graph.indptr.tolist(), indices=graph.indices.tolist())
        else:
            assert graph.format == "coo"
            result.update(row=graph.row.tolist(), col=graph.col.tolist())
        return result

    def observe(operation):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                graph = operation()
            except Exception as error:
                result = {"error": {"type": type(error).__name__, "message": str(error)}}
            else:
                result = {"graph": graph_record(graph)}
        result["warnings"] = [
            {"type": item.category.__name__, "message": str(item.message)} for item in captured
        ]
        return result

    def fit_case(pair=None, **overrides):
        result = {"overrides": overrides.copy()}
        if pair is not None:
            supplied_indices, supplied_distances = (array.copy() for array in pair)
            result["indices"] = supplied_indices.tolist()
            result["distance_bits"] = supplied_distances.view(np.uint32).tolist()
            overrides["precomputed_knn"] = (supplied_indices, supplied_distances, None)
        model = UMAP(**(PARAMETERS | overrides))
        result["fit_transform"] = observe(lambda: model.fit_transform(training))
        if pair is not None:
            result["mutated_indices"] = not np.array_equal(supplied_indices, pair[0])
            result["mutated_distances"] = not np.array_equal(supplied_distances, pair[1])
            if result["mutated_indices"]:
                result["indices_after_fit"] = supplied_indices.tolist()
            if result["mutated_distances"]:
                result["distance_bits_after_fit"] = supplied_distances.view(np.uint32).tolist()
        if "graph" in result["fit_transform"]:
            result["transform_held_out"] = observe(lambda: model.transform(held_out))
        return result

    if python_estimator:
        explicit = np.array(
            [[0, 0], [0, 1], [3, 2], [5, -1], [0.25, 0.25], [-1, 5], [2, -1], [1, 5]],
            dtype=np.float32,
        )
        layouts = {}
        for name, init, epochs, cutoff in (
            ("random_initial", "random", 0, None),
            ("random_optimized", "random", 25, None),
            ("explicit_initial", explicit, 0, None),
            ("explicit_optimized", explicit, 25, None),
            ("spectral_connected", "spectral", 6, None),
            ("spectral_disconnected", "spectral", 6, 1.75),
            ("random_isolated", "random", 25, 1.0),
        ):
            i, d = indices.copy(), overridden.copy()
            model = UMAP(**(PARAMETERS | {
                "transform_mode": "embedding", "init": init, "n_epochs": epochs,
                "disconnection_distance": cutoff, "precomputed_knn": (i, d, None),
            }))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                coordinates = model.fit_transform(training)
            assert model.transform(training.copy()) is model.embedding_
            layouts[name] = {
                "init": init if isinstance(init, str) else init.tolist(),
                "n_epochs": epochs,
                "disconnection_distance": cutoff,
                "graph": graph_record(model.graph_),
                "coordinates_bits": coordinates.view(np.uint32).tolist(),
                "warnings": [
                    {"type": item.category.__name__, "message": str(item.message)}
                    for item in captured
                ],
            }
        return {
            "training": training.tolist(), "held_out": held_out.tolist(),
            "layout_indices": indices.tolist(), "layout_distance_bits": overridden.view(np.uint32).tolist(),
            "layouts": layouts,
            "cases": {
                "width_fallback_at_cutoff": fit_case(
                    (indices[:, :-1], distances[:, :-1]), disconnection_distance=1.0
                ),
            },
        }

    if rust_held_out:

        def supplied(width, rows=()):
            pair = (ordered[:, :width].copy(), ordered_distances[:, :width].copy())
            for row, indices, distances in rows:
                pair[0][row] = indices
                pair[1][row] = distances
            return pair

        wide = supplied(6)
        wider_than_training = tuple(
            np.column_stack((array, array[:, 1])) for array in supplied(8)
        )
        adjacent = (
            np.nextafter(np.float32(1), np.float32(0)),
            np.float32(1),
            np.nextafter(np.float32(1), np.float32(2)),
        )
        duplicates = supplied(
            6,
            [
                (0, [0, 1, 1, 1, 2, 2], [0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                (1, [1, 0, 0, 0, 2, 2], [0, 0.5, 0.5, 0.5, 0.5, 0.5]),
                (2, [2, 0, 0, 3, 4, 5], [0, 0.5, 0.5, 0.5, 0.5, 0.5]),
            ],
        )
        cases = {
            "same_width_k3": fit_case(wide, n_neighbors=3),
            "same_width_k5": fit_case(wide, n_neighbors=5),
            "width_above_row_count": fit_case(wider_than_training, n_neighbors=9),
            "threshold_adjacent_values": fit_case(
                supplied(6, [(0, [0, 1, 2, 3, 4, 5], [0, *adjacent, 0.9375, 1.25])]),
                disconnection_distance=1.0,
            ),
            "fractional_connectivity_first_column": fit_case(
                supplied(6, [(0, [4, 0, 2, 3, 1, 5], [3, 0, 1, 2, 4, 5])]),
                local_connectivity=0.4,
            ),
            "unsorted_connectivity_saturates": fit_case(
                supplied(4, [(0, [1, 2, 0, 3], [3, 1, 0, 2])]),
                local_connectivity=4.0,
            ),
            "self_distance_nonzero": fit_case(
                supplied(4, [(0, [1, 0, 2, 3], [0.5, 0.125, 0.75, 1.0])])
            ),
            "leading_missing": fit_case(
                supplied(6, [(0, [-1, 1, 0, 2, 3, 4], [np.inf, 1, 0, 2, 3, 4])])
            ),
            "missing_row_mean_floor": fit_case(
                supplied(6, [(0, [0, 1, 2, 3, 4, -1], [0, 1, 1, 1, 1.001, np.inf])])
            ),
            "missing_batch_mean_floor": fit_case(
                supplied(6, [(0, [0, 1, 2, 3, 4, -1], [0, 0, 0, 0, 0.001, np.inf])]),
                local_connectivity=0.0,
            ),
            "duplicates_union": fit_case(duplicates),
            "duplicates_intersection": fit_case(duplicates, set_op_mix_ratio=0.0),
            "duplicates_mixed": fit_case(duplicates, set_op_mix_ratio=0.35),
            "duplicates_with_distinct_memberships": fit_case(
                supplied(
                    8,
                    [(0, [0, 4, 2, 4, 3, 4, 5, 4], [0, 2**-10, 0.75, 4, 0.75, 1.5, 3.5, 2.25])],
                ),
                n_neighbors=5,
            ),
        }
        for name, case in cases.items():
            assert case["transform_held_out"]["error"]["type"] == "NotImplementedError", name
        return {"training": training.tolist(), "held_out": held_out.tolist(), "cases": cases}

    cases = {
        "computed": fit_case(),
        "exact_width": fit_case((indices, distances)),
        "override_distances": fit_case((indices, overridden)),
        "one_fewer_column": fit_case((indices[:, :-1], distances[:, :-1])),
        "one_extra_column": fit_case(
            (ordered[:, : k + 1], np.column_stack((overridden, ordered_distances[:, k])))
        ),
        "one_fewer_row": fit_case((indices[:-1], distances[:-1])),
        "mismatched_shapes": fit_case((indices, distances[:, :-1])),
        "missing_neighbors": fit_case((missing_indices, missing_distances)),
        "only_self": fit_case((empty_indices, empty_distances)),
        "self_second": fit_case((unordered_indices, unordered_distances)),
        "no_self": fit_case((no_self_indices, no_self_distances)),
        "repeated_neighbor": fit_case((repeated_indices, repeated_distances)),
        "unsorted_distances": fit_case((indices, unsorted_distances)),
    }
    for name, threshold in (
        ("zero", np.float32(0.0)),
        ("before_one", np.nextafter(np.float32(1.0), np.float32(0.0))),
        ("one", np.float32(1.0)),
        ("after_one", np.nextafter(np.float32(1.0), np.float32(2.0))),
    ):
        cases[f"cutoff_{name}"] = fit_case((indices, overridden), disconnection_distance=float(threshold))

    baseline = cases["computed"]["fit_transform"]["graph"]
    baseline_transform = cases["computed"]["transform_held_out"]["graph"]
    for name in ("exact_width", "one_fewer_column", "one_fewer_row"):
        assert cases[name]["fit_transform"]["graph"] == baseline, name
    for name in ("one_fewer_column", "one_fewer_row"):
        assert cases[name]["transform_held_out"]["graph"] == baseline_transform, name
    assert cases["exact_width"]["transform_held_out"]["error"]["type"] == "NotImplementedError"
    assert cases["mismatched_shapes"]["fit_transform"]["error"]["type"] == "ValueError"
    assert len(cases["one_extra_column"]["fit_transform"]["graph"]["indices"]) == 38
    assert cases["cutoff_before_one"]["fit_transform"]["graph"] == cases["cutoff_one"]["fit_transform"]["graph"]
    assert cases["cutoff_after_one"]["fit_transform"]["graph"] != cases["cutoff_one"]["fit_transform"]["graph"]
    return {"training": training.tolist(), "held_out": held_out.tolist(), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rust-held-out", action="store_true")
    group.add_argument("--python-estimator", action="store_true")
    args = parser.parse_args()
    default_fixture = RUST_HELD_OUT_FIXTURE if args.rust_held_out else FIXTURE
    if args.python_estimator:
        default_fixture = PYTHON_LINUX_FIXTURE if platform.system() == "Linux" else PYTHON_FIXTURE
    output = args.output or default_fixture
    environment = pinned_environment()
    os.environ.update(THREADS)
    with tempfile.TemporaryDirectory(prefix="umap-precomputed-neighbors-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        fixture = {
            "environment": environment,
            "base_parameters": PARAMETERS,
            **capture(rust_held_out=args.rust_held_out, python_estimator=args.python_estimator),
        }
    content = json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n"
    if args.check:
        if output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"public fixture differs from {output}")
    else:
        output.write_bytes(content.encode("utf-8"))


if __name__ == "__main__":
    main()
