"""Record public UMAP 0.6.0 dense supervised fits and continuous-target probes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

from generate_supervised_graph_fixture import (
    BASE_PARAMETERS,
    CASES,
    CATEGORICAL_LABELS,
    DATA_ROWS,
    PARTIAL_LABELS,
    THREAD_ENVIRONMENT,
    csr_record,
)


SPECTRAL_REPEATS = {
    "categorical_spectral",
    "categorical_full_weight_spectral",
    "three_rows_partial_spectral",
}


def array_record(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "bits": array.ravel().view("uint32" if array.dtype.itemsize == 4 else "uint64").tolist(),
    }


def fit_case(data, labels, options, operation):
    import numpy as np
    from umap import UMAP

    rng = np.random.RandomState(options["random_state"])
    if isinstance(options["init"], list):
        options = options | {"init": np.asarray(options["init"], dtype=np.float32)}
    model = UMAP(**(options | {"random_state": rng}))
    result = None
    error = None
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            result = getattr(model, operation)(data.copy(), y=labels)
        except Exception as exc:
            error = type(exc).__name__

    _, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
    graph = getattr(model, "graph_", None)
    embedding = getattr(model, "embedding_", None)
    same_input = None
    update_error = None
    if error is None and len(data) > 1:
        output = graph if options["transform_mode"] == "graph" else embedding
        same_input = model.transform(data.copy()) is output
        if labels is not None and options["transform_mode"] == "embedding":
            try:
                model.update(data[:1].copy())
            except Exception as exc:
                update_error = type(exc).__name__
    density = None
    if error is None and options.get("output_dens"):
        density = [array_record(values) for values in (model.rad_orig_, model.rad_emb_)]
    checkpoints = getattr(model, "embedding_list_", None)
    if isinstance(result, tuple):
        result = result[0]
    return {
        "error": error,
        "warnings": [[item.category.__name__, str(item.message)] for item in observed],
        "graph": None if graph is None else csr_record(graph),
        "embedding": None if embedding is None else array_record(embedding),
        "fit_returns_self": result is model,
        "fit_transform_returns_embedding": result is not None and result is embedding,
        "fit_transform_dtype": getattr(getattr(result, "dtype", None), "name", None),
        "same_input_transform_returns_fitted_output": same_input,
        "supervised_update_error": update_error,
        **({"density": density} if density is not None else {}),
        **({"checkpoints": [array_record(values) for values in checkpoints]}
           if checkpoints is not None else {}),
        "rng": [cursor, has_gaussian, cached_gaussian, rng.randint(0, 2**32, 4, dtype=np.uint32).tolist()],
    }


def make_fixture(continuous_only):
    import numpy as np

    base = BASE_PARAMETERS | {"transform_mode": "embedding", "a": 1.5, "b": 0.9}
    features = np.asarray(DATA_ROWS, dtype=np.float32)
    definitions = {}

    def add(name, labels, *, data=features, **options):
        definitions[name] = (data, None if labels is None else np.asarray(labels), options)

    if continuous_only:
        scalar = np.asarray([0, 0.125, 0.5, 1.125, 0.75, 2.125, 2.375, 3], dtype=np.float32)
        vector = np.column_stack((scalar, scalar[::-1] ** 2))
        for metric in ("euclidean", "manhattan", "cosine", "ordinal"):
            for shape, labels in (("scalar", scalar), ("vector", vector)):
                for weight in (0.0, 0.5, 1.0):
                    add(f"{metric}_{shape}_{weight:g}", labels, target_metric=metric,
                        target_weight=weight, transform_mode="graph")
        for weight in (0.25, 0.75, float(np.nextafter(0.5, 0.0)), float(np.nextafter(0.5, 1.0))):
            add(f"euclidean_vector_{weight!r}", vector, target_metric="euclidean",
                target_weight=weight, transform_mode="graph")
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            add(f"chebyshev_vector_{weight:g}", vector, target_metric="chebyshev",
                target_weight=weight, transform_mode="graph")
        add("euclidean_embedding", scalar, target_metric="euclidean")
        add("euclidean_vector_spectral", vector, target_metric="euclidean", init="spectral",
            target_weight=0.75)
        add("manhattan_vector_embedding", vector, target_metric="manhattan", target_weight=0.25)
        add("chebyshev_vector_embedding", vector, target_metric="chebyshev", target_weight=0.75)
        add("euclidean_singleton", scalar[:1], data=features[:1], target_metric="euclidean")
        add("euclidean_two_rows", scalar[:2], data=features[:2], target_metric="euclidean")
        add("euclidean_two_rows_explicit_target_neighbors", scalar[:2], data=features[:2],
            target_metric="euclidean", target_n_neighbors=2, transform_mode="graph")
        add("euclidean_three_rows", scalar[:3], data=features[:3], target_metric="euclidean")
        add("euclidean_more_target_neighbors_than_rows", scalar, target_metric="euclidean",
            target_n_neighbors=9, transform_mode="graph")
        add("euclidean_all_rows_are_target_neighbors", scalar, target_metric="euclidean",
            target_n_neighbors=8, transform_mode="graph")
        add("euclidean_one_target_neighbor", scalar, target_metric="euclidean",
            target_n_neighbors=1, transform_mode="graph")
        add("euclidean_two_target_neighbors", scalar, target_metric="euclidean",
            target_n_neighbors=2, transform_mode="graph")
        add("euclidean_partly_negative", np.where(scalar < 0.6, -1, scalar),
            target_metric="euclidean", transform_mode="graph")
        add("euclidean_zero_feature_connectivity", scalar, target_metric="euclidean",
            local_connectivity=0.0, transform_mode="graph")
        separated = np.asarray([[0, 0], [1, 0], [0, 1], [30, 40], [30, 41], [31, 42],
                                [32, 41], [33, 40]], dtype=np.float32)
        add("euclidean_small_feature_floor", scalar, data=separated,
            target_metric="euclidean", target_weight=0.0, transform_mode="graph")
        holdout = np.random.RandomState(2718)
        x = holdout.normal(size=(11, 3)).astype(np.float32)
        y = holdout.uniform(-1, 2, (len(x), 3)).astype(np.float32)
        add("euclidean_holdout", y, data=x, target_metric="euclidean", n_neighbors=5,
            target_n_neighbors=6, target_weight=0.37, random_state=13, init="spectral", n_epochs=19)
        add("manhattan_holdout", y, data=x, target_metric="manhattan", n_neighbors=5,
            target_weight=0.87, random_state=13, n_epochs=19)
        add("chebyshev_holdout", y, data=x, target_metric="chebyshev", n_neighbors=5,
            target_weight=0.87, random_state=13, n_epochs=19)
    else:
        for name, labels, weight in CASES:
            add(f"{name}_random", labels, target_weight=weight)
            add(f"{name}_spectral", labels, target_weight=weight, init="spectral")

        for name, weight in (
            ("next_above_zero", np.nextafter(0.0, 1.0)),
            ("near_zero", 0.000001),
            ("retains_cross_edges", 0.95),
            ("underflows_cross_edges", 0.98),
            ("next_below_one", np.nextafter(1.0, 0.0)),
        ):
            add(name, CATEGORICAL_LABELS, target_weight=float(weight))
        add("partial_near_one", PARTIAL_LABELS, target_weight=float(np.nextafter(1.0, 0.0)))
        add("partial_zero_epochs", PARTIAL_LABELS, n_epochs=0)
        add("partial_active_spectral", PARTIAL_LABELS, init="spectral", n_epochs=11, n_jobs=-1)
        add("partial_manhattan", PARTIAL_LABELS, metric="manhattan", n_epochs=30)
        add("partial_chebyshev", PARTIAL_LABELS, metric="chebyshev", n_epochs=30)
        add("partial_explicit", PARTIAL_LABELS, init=[[0, 0], [1, 2], [2, 1], [3, 4],
            [4, 3], [5, 7], [6, 6], [7, 5]])
        add("partial_checkpoints", PARTIAL_LABELS, n_epochs=[0, 2, 5, 30])
        add("partial_density_diagnostics", PARTIAL_LABELS, output_dens=True)
        add("singleton_with_extra_labels", PARTIAL_LABELS, data=features[:1], init="spectral")
        add("two_rows", CATEGORICAL_LABELS[:2], data=features[:2])
        add("three_rows_same_class", [1, 1, 1], data=features[:3])
        add("three_rows_one_isolated", [1, 2, 1], data=features[:3], target_weight=1.0)
        add("three_rows_all_isolated", [1, 2, 3], data=features[:3], target_weight=1.0)
        add("three_rows_partial_spectral", [1, -1, 1], data=features[:3], init="spectral", n_components=1)
        add("isolated_rows_large_fraction", list(range(6)) + [10, 10], target_weight=1.0)
        add("all_isolated", list(range(8)), target_weight=1.0)

    cases = {}
    for name, (data, labels, options) in definitions.items():
        parameters = base | options
        record = fit_case(data, labels, parameters, "fit_transform")
        if not continuous_only:
            repeats = []
            for _ in range(5 if name in SPECTRAL_REPEATS else 1):
                fitted = fit_case(data, labels, parameters, "fit")
                for key in ("error", "warnings", "graph", "rng", "density", "checkpoints"):
                    if fitted.get(key) != record.get(key):
                        raise RuntimeError(f"{name}: fit and fit_transform differ in {key}")
                if record["error"] is None and not fitted["fit_returns_self"]:
                    raise RuntimeError(f"{name}: fit did not return its estimator")
                repeats.append(fitted["embedding"])
            if name in SPECTRAL_REPEATS:
                record["repeat_embeddings"] = repeats
            elif repeats[0] != record["embedding"]:
                raise RuntimeError(f"{name}: repeated fits differ in the embedding")
        cases[name] = {
            "data_rows": len(data),
            **({"data": data.tolist()} if not np.array_equal(data, features[:len(data)]) else {}),
            "labels": None if labels is None else labels.tolist(),
            "options": options,
            **record,
        }
    return {
        "platform": [platform.system(), platform.machine(), platform.python_version()],
        "packages": {package: version(package) for package in (
            "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "pynndescent"
        )},
        "numba_disable_jit": os.getenv("NUMBA_DISABLE_JIT", "0"),
        "base_parameters": base,
        "data": DATA_ROWS,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuous-only", action="store_true")
    args = parser.parse_args()
    if version("umap-learn") != "0.6.0":
        parser.error("The reference environment requires umap-learn==0.6.0")
    os.environ.update(THREAD_ENVIRONMENT)
    fixture = make_fixture(args.continuous_only)
    lines = ["{"]
    for key, value in fixture.items():
        if key != "cases":
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
    lines.append('  "cases": {')
    cases = list(fixture["cases"].items())
    for index, (name, record) in enumerate(cases):
        suffix = "," if index < len(cases) - 1 else ""
        lines.append(f"    {json.dumps(name)}: {json.dumps(record, allow_nan=False)}{suffix}")
    lines.extend(("  }", "}", ""))
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
