"""Capture public UMAP sparse graph behavior and replay public distance inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tomllib
import warnings
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory


BASE_PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 4,
    "init": "random",
    "random_state": 42,
    "n_jobs": 1,
    "n_epochs": 0,
    "compatibility_layout": True,
    "transform_mode": "graph",
    "force_approximation_algorithm": False,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
METRICS = ("euclidean", "manhattan", "chebyshev", "cosine")


def csr_record(matrix) -> dict:
    assert matrix.format == "csr" and matrix.dtype.name == "float32"
    return {
        "type": type(matrix).__name__,
        "format": matrix.format,
        "dtype": matrix.dtype.name,
        "shape": list(matrix.shape),
        "nnz": matrix.nnz,
        "indptr_dtype": matrix.indptr.dtype.name,
        "indptr": matrix.indptr.tolist(),
        "indices_dtype": matrix.indices.dtype.name,
        "indices": matrix.indices.tolist(),
        "data_bits_uint32": matrix.data.view("uint32").tolist(),
    }


def csr_input_summary(matrix) -> dict:
    rows = [matrix.indices[lo:hi] for lo, hi in zip(matrix.indptr, matrix.indptr[1:])]
    return {
        "nnz": matrix.nnz,
        "row_0_indices": rows[0].tolist(),
        "rows_sorted": all((row[:-1] <= row[1:]).all() for row in rows),
        "rows_strictly_increasing": all((row[:-1] < row[1:]).all() for row in rows),
    }


def feature_inputs():
    import numpy as np
    from scipy.sparse import csr_matrix

    # Exact dyadic float32 values from the initial public probe.
    state = 374
    values = []
    for _ in range(12 * 8):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append(((state >> 16) - 32768) / 8192.0 if state & 3 else 0.0)
    dense = np.array(values, dtype=np.float32).reshape(12, 8)
    dense[2] = 0
    dense[10] = dense[1]

    rng = np.random.RandomState(374)
    holdout = rng.uniform(-1, 1, (12, 8)).astype(np.float32)
    holdout[rng.random_sample(holdout.shape) < 0.6] = 0
    return csr_matrix(dense), csr_matrix(holdout)


def feature_variants(csr) -> dict:
    import numpy as np
    from scipy.sparse import coo_matrix, csr_matrix

    reversed_values = csr.data.copy()
    reversed_columns = csr.indices.copy()
    for lo, hi in zip(csr.indptr, csr.indptr[1:]):
        reversed_values[lo:hi] = csr.data[lo:hi][::-1]
        reversed_columns[lo:hi] = csr.indices[lo:hi][::-1]
    reverse_rows = csr_matrix(
        (reversed_values, reversed_columns, csr.indptr.copy()), shape=csr.shape
    )

    coo = csr.tocoo()
    halves = np.tile(coo.data * np.float32(0.5), 2)
    split_coo = coo_matrix(
        (halves, (np.tile(coo.row, 2), np.tile(coo.col, 2))), shape=csr.shape
    )
    split_csr = csr_matrix(
        (np.repeat(csr.data * np.float32(0.5), 2), np.repeat(csr.indices, 2), csr.indptr * 2),
        shape=csr.shape,
    )
    assert csr_record(split_coo.tocsr()) == csr_record(csr)
    assert not split_csr.has_canonical_format
    np.testing.assert_array_equal(split_csr.toarray(), csr.toarray())

    return {
        "csr": csr,
        "dense": csr.toarray(),
        "csc": csr.tocsc(),
        "split_coo": split_coo,
        "reverse_rows_csr": reverse_rows,
        "split_csr": split_csr,
    }


def sparse_precomputed_input():
    import numpy as np
    from scipy.sparse import csr_matrix

    # Six cycle edges and six two-step chords, with four stored distances per row.
    dense = np.zeros((6, 6), dtype=np.float32)
    for step in (1, 2):
        for row in range(6):
            column = (row + step) % 6
            dense[row, column] = dense[column, row] = (step - 1) * 6 + row + 1
    return csr_matrix(dense)


def precomputed_variants(csr) -> dict:
    dense = csr.toarray()
    boundary = csr.copy()
    boundary[0, 2] = boundary[2, 0] = 0
    boundary.eliminate_zeros()
    insufficient = boundary.copy()
    insufficient[0, 4] = insufficient[4, 0] = 0
    insufficient.eliminate_zeros()
    asymmetric = csr.copy()
    asymmetric[0, 2] = 0
    asymmetric.eliminate_zeros()
    unequal_values = csr.copy()
    unequal_values[0, 1] = 3
    explicit_diagonal = csr.copy()
    explicit_diagonal.setdiag(0)
    explicit_diagonal.sort_indices()
    return {
        "csr": csr,
        "dense": dense,
        "explicit_zero_diagonal": explicit_diagonal,
        "exactly_n_neighbors": boundary,
        "too_few_distances": insufficient,
        "asymmetric_support": asymmetric,
        "asymmetric_values": unequal_values,
    }


def fit_graph(values, metric: str, n_neighbors: int = BASE_PARAMETERS["n_neighbors"]) -> dict:
    from umap import UMAP

    parameters = BASE_PARAMETERS | {"metric": metric, "n_neighbors": n_neighbors}
    estimator = UMAP(**parameters)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            graph = estimator.fit_transform(values)
        except ValueError as error:
            result = {"error": {"type": type(error).__name__, "message": str(error)}}
        else:
            result = {"graph": csr_record(graph)}
    result["warnings"] = [
        {"type": item.category.__name__, "message": str(item.message)} for item in captured
    ]
    return result


def merged_differences(csr, row: int, column: int):
    """Yield float64 differences in increasing feature order from canonical CSR."""
    left, left_end = map(int, csr.indptr[row : row + 2])
    right, right_end = map(int, csr.indptr[column : column + 2])
    indices, values = csr.indices, csr.data
    while left < left_end or right < right_end:
        if right == right_end or (left < left_end and indices[left] < indices[right]):
            yield float(values[left])
            left += 1
        elif left == left_end or indices[right] < indices[left]:
            yield -float(values[right])
            right += 1
        else:
            yield float(values[left]) - float(values[right])
            left += 1
            right += 1


def independent_distances(csr) -> dict:
    import numpy as np

    assert csr.has_canonical_format
    n_rows = csr.shape[0]
    distances = {
        name: np.zeros((n_rows, n_rows), dtype=np.float32)
        for name in ("euclidean", "euclidean_late_cast", "manhattan", "chebyshev")
    }
    for row in range(n_rows):
        for column in range(row):
            squared = absolute_sum = absolute_max = 0.0
            for difference in merged_differences(csr, row, column):
                squared += difference * difference
                absolute_sum += abs(difference)
                absolute_max = max(absolute_max, abs(difference))
            for name, value in (
                ("euclidean", np.sqrt(np.float32(squared))),
                ("euclidean_late_cast", math.sqrt(squared)),
                ("manhattan", absolute_sum),
                ("chebyshev", absolute_max),
            ):
                distances[name][row, column] = distances[name][column, row] = value
    return distances


def compare_graphs(expected: dict, observed: dict) -> dict:
    same_structure = all(
        expected[key] == observed[key] for key in expected if key != "data_bits_uint32"
    )
    mismatches = None
    if same_structure:
        mismatches = sum(
            expected_bit != observed_bit
            for expected_bit, observed_bit in zip(
                expected["data_bits_uint32"], observed["data_bits_uint32"], strict=True
            )
        )
    return {"identical_csr_structure": same_structure, "different_weight_bits": mismatches}


def euclidean_replay(csr, distances: dict, expected: dict) -> dict:
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances

    sklearn = euclidean_distances(csr)
    n_neighbors = BASE_PARAMETERS["n_neighbors"]
    columns = np.argsort(sklearn, axis=1, kind="stable")[:, :n_neighbors]
    reference_bits = np.take_along_axis(sklearn, columns, axis=1).view(np.uint32)
    nearest_bits = np.take_along_axis(distances["euclidean"], columns, axis=1).view(np.uint32)
    late_bits = np.take_along_axis(distances["euclidean_late_cast"], columns, axis=1).view(
        np.uint32
    )
    np.testing.assert_array_equal(nearest_bits, reference_bits)
    row, rank = np.argwhere(nearest_bits != late_bits)[0]
    replays = {}
    for name, matrix in (
        ("cast_squared_sum_to_float32_before_sqrt", distances["euclidean"]),
        ("take_float64_sqrt_before_cast", distances["euclidean_late_cast"]),
        ("sklearn_euclidean_distances_csr", sklearn),
    ):
        replay = fit_graph(matrix, "precomputed")["graph"]
        replays[name] = compare_graphs(expected, replay)
    assert replays["cast_squared_sum_to_float32_before_sqrt"] == {
        "identical_csr_structure": True,
        "different_weight_bits": 0,
    }
    assert (
        replays["sklearn_euclidean_distances_csr"]
        == replays["cast_squared_sum_to_float32_before_sqrt"]
    )
    assert replays["take_float64_sqrt_before_cast"]["different_weight_bits"] > 0
    return {
        "nearest_entries_compared": int(reference_bits.size),
        "different_nearest_bits_for_float64_sqrt": int(np.count_nonzero(nearest_bits != late_bits)),
        "distance_witness": {
            "row": int(row),
            "column": int(columns[row, rank]),
            "sklearn_and_float32_sqrt_bits_uint32": int(nearest_bits[row, rank]),
            "float64_sqrt_bits_uint32": int(late_bits[row, rank]),
        },
        "precomputed_graph_replays": replays,
    }


def capture_cases(features, holdout, precomputed) -> tuple[dict, dict]:
    from sklearn.metrics.pairwise import cosine_distances

    cases = {}
    variants = feature_variants(features)
    comparisons = {}
    for metric in METRICS:
        baseline = f"features_csr_{metric}"
        for variant, matrix in variants.items():
            values = matrix.copy()
            inspect_input = metric == "cosine" and variant in ("reverse_rows_csr", "split_csr")
            before = csr_input_summary(values) if inspect_input else None
            result = fit_graph(values, metric)
            graph = result["graph"]
            if inspect_input:
                result["input_before"] = before
                result["input_after"] = csr_input_summary(values)
                expected_input = features if variant == "reverse_rows_csr" else matrix
                assert csr_record(values) == csr_record(expected_input)
            if variant != "csr":
                expected = cases[baseline]["graph"]
                if graph == expected:
                    del result["graph"]
                    result["same_graph_as"] = baseline
                if variant == "dense" or (variant == "split_csr" and metric == "cosine"):
                    comparisons[f"features_{variant}_{metric}"] = compare_graphs(expected, graph)
                else:
                    assert graph == expected, (variant, metric)
            cases[f"features_{variant}_{metric}"] = {
                "input": "features", "variant": variant, "metric": metric, **result
            }

    cases["holdout_csr_euclidean"] = {
        "input": "holdout",
        "variant": "csr",
        "metric": "euclidean",
        **fit_graph(holdout.copy(), "euclidean"),
    }
    for variant, matrix in precomputed_variants(precomputed).items():
        cases[f"precomputed_{variant}"] = {
            "input": "precomputed",
            "variant": variant,
            "metric": "precomputed",
            "n_neighbors": 3,
            **fit_graph(matrix, "precomputed", n_neighbors=3),
        }

    assert comparisons["features_dense_euclidean"]["different_weight_bits"] > 0
    assert comparisons["features_dense_cosine"]["different_weight_bits"] > 0
    assert comparisons["features_split_csr_cosine"]["different_weight_bits"] > 0
    assert (
        cases["features_csr_manhattan"]["graph"]["indices"]
        != cases["features_csr_chebyshev"]["graph"]["indices"]
    )
    for metric in ("manhattan", "chebyshev"):
        assert comparisons[f"features_dense_{metric}"]["different_weight_bits"] == 0

    feature_distances = independent_distances(features)
    replays = {
        "features_csr_euclidean": euclidean_replay(
            features, feature_distances, cases["features_csr_euclidean"]["graph"]
        ),
        "holdout_csr_euclidean": euclidean_replay(
            holdout,
            independent_distances(holdout),
            cases["holdout_csr_euclidean"]["graph"],
        ),
    }
    for case, matrix in (
        ("features_csr_manhattan", feature_distances["manhattan"]),
        ("features_csr_chebyshev", feature_distances["chebyshev"]),
        ("features_csr_cosine", cosine_distances(features)),
        ("features_split_csr_cosine", cosine_distances(variants["split_csr"])),
    ):
        expected = cases[case]["graph"]
        assert fit_graph(matrix, "precomputed")["graph"] == expected, case
        replays[case] = {"identical_csr_structure": True, "different_weight_bits": 0}
    return cases, {"feature_graph_comparisons_to_csr": comparisons, "distance_replays": replays}


def pinned_environment() -> dict:
    lock = tomllib.loads(Path(__file__).with_name("uv.lock").read_text(encoding="utf-8"))
    packages = {item.metadata["Name"].lower(): item.version for item in distributions()}
    required = {
        item["name"]: item["version"] for item in lock["package"] if "registry" in item["source"]
    }
    if platform.system() != "Windows":
        del required["colorama"]  # The lock supplies this dependency only on Windows.
    if packages != required or f"=={platform.python_version()}" != lock["requires-python"]:
        raise RuntimeError("Use the Python and packages pinned in reference/uv.lock")
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "packages": dict(sorted(packages.items())),
        "thread_environment": THREAD_ENVIRONMENT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = pinned_environment()
    os.environ.update(THREAD_ENVIRONMENT)
    with TemporaryDirectory(prefix="umap-sparse-graph-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        features, holdout = feature_inputs()
        precomputed = sparse_precomputed_input()
        cases, checks = capture_cases(features, holdout, precomputed)

    fixture = {
        "operation": "UMAP(transform_mode='graph').fit_transform(X)",
        "environment": environment,
        "base_parameters": BASE_PARAMETERS,
        "inputs": {
            "features": csr_record(features),
            "holdout": csr_record(holdout),
            "precomputed": csr_record(precomputed),
        },
        "cases": cases,
        "checks": checks,
    }
    args.output.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
