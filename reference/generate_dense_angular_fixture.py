"""Compare independent angular distances with public UMAP graph-mode results."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from math import sqrt
from pathlib import Path

import numpy as np


PARAMETERS = dict(
    random_state=42, n_jobs=1, init="random", n_epochs=30,
    compatibility_layout=True, transform_mode="graph",
)

GENERIC = [
    [0.12, 0.07, 1.2], [1, 4, 2], [0.78, 1.2, 3.1], [2.2, -3, 0.5],
    [-0.1, 0.7, 6], [0.3, -0.5, -0.9], [5, 6, 7],
]
QUERIES = [[0, 0, 0], [2, 2, 2], [1, 0.11, 0.75], [1, 0, -1]]
REPEATS = [
    [0, 0, 0], [0, 0, 0], [1, 1, 1], [-2, -2, -2], [1, 0, -1],
    [-1, 0, 1], [1, 0, -1], [2, 4, 1], [0.3, -0.5, 0.7],
]
SIGN_REVERSAL = [[1, 0], [-1, 0], [1, 0], [-1, 0]]


def inputs():
    yield "generic", dict(training=GENERIC, queries=QUERIES, k=4)
    yield "cutoff_one", dict(training=GENERIC, queries=QUERIES, k=4, cutoff=1.0)
    yield "graph_options", dict(
        training=GENERIC, queries=QUERIES, k=4,
        options={"local_connectivity": 1.5, "set_op_mix_ratio": 0.4},
    )
    yield "zero_constant_repeated", dict(training=REPEATS, queries=QUERIES, k=4)
    yield "one_feature_k2", dict(training=[[0], [0], [1], [1], [-1], [2], [-3]], k=2)
    yield "equal_cutoff_k2", dict(training=[
        [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0],
    ], k=2)
    yield "held_out_ties", dict(training=[
        [1, 0, -1], [-1, 0, 1], [0, 1, 0], [-1, 1, 0], [0, -1, 0], [0, 0, 0],
    ], queries=[[1, 0, -1], [-1, 0, 1]], k=4)
    yield "sign_reversal", dict(training=SIGN_REVERSAL, queries=SIGN_REVERSAL[:2], k=3)
    yield "sign_reversal_uncut", dict(
        training=SIGN_REVERSAL, queries=SIGN_REVERSAL[:2], k=3, cutoff="inf",
    )
    yield "cancellation", dict(training=[
        [1, 1, 1, 1], [1e8, -1e8, 1, -1], [1e8, 1, -1e8, 1],
        [1e8, 1, -1e8, -1], [1, 4, 5, 7], [1.0000001, 1, 1, 0.9999999],
        [-1, -1, -1, -1],
    ], k=4)
    yield "rounded_negative", dict(
        random_shape=[6, 5], seed=5, scaled_rows=[[1, 0, 3], [3, 2, 7]],
        query_prefix=2, k=2, cutoff=0.0,
    )
    yield "product_overflow", dict(training=[
        [1e20, 1e20], [2e20, 2e20], [3e20, 3e20], [-1e20, 1e20],
    ], query_prefix=2, k=3)
    for width in (3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        spec = dict(random_shape=[13, width], seed=width + 87, k=5)
        if width in (8, 32, 64, 128):
            spec["query_prefix"] = 3
        yield f"width_{width}", spec


def features(spec):
    if "training" in spec:
        data = np.asarray(spec["training"], dtype=np.float32)
    else:
        rng = np.random.default_rng(spec["seed"])
        if "scaled_rows" in spec:
            data = rng.uniform(0.1, 2, spec["random_shape"]).astype(np.float32)
            for destination, source, factor in spec["scaled_rows"]:
                data[destination] = factor * data[source]
        else:
            data = rng.standard_normal(spec["random_shape"]).astype(np.float32)
    queries = None
    if "queries" in spec:
        queries = np.asarray(spec["queries"], dtype=np.float32)
    elif "query_prefix" in spec:
        queries = data[:spec["query_prefix"]].copy()
    return data, queries


def independent_distances(data, *, center, narrow_products):
    rows = data.copy() if narrow_products else data.astype(np.float64)
    if center:
        for row in rows:
            total = 0.0
            for value in row:
                total += float(value)
            row -= total / len(row)
    result = np.zeros((len(rows), len(rows)), dtype=np.float32)
    for i, left in enumerate(rows):
        for j, right in enumerate(rows[i + 1:], i + 1):
            dot = left_squared = right_squared = 0.0
            for x, y in zip(left, right):
                dot += float(x * y)
                left_squared += float(x * x)
                right_squared += float(y * y)
            if left_squared == right_squared == 0:
                distance = 0.0
            elif left_squared == 0 or right_squared == 0:
                distance = 1.0
            else:
                distance = 1 - dot / sqrt(left_squared * right_squared)
            result[i, j] = result[j, i] = distance
    return result


def record(graph):
    assert graph.dtype == np.dtype(np.float32)
    output = dict(
        format=graph.format, dtype=graph.dtype.name, shape=list(graph.shape),
        data_bits=graph.data.view(np.uint32).tolist(),
    )
    if graph.format == "csr":
        return output | dict(indptr=graph.indptr.tolist(), indices=graph.indices.tolist())
    assert graph.format == "coo"
    return output | dict(row=graph.row.tolist(), col=graph.col.tolist())


def observe(estimator, data, metric, k, cutoff, options, queries=None):
    outputs = []
    for _ in range(2):
        try:
            model = estimator(
                **PARAMETERS, **options, metric=metric, n_neighbors=k, disconnection_distance=cutoff
            )
            fitted = model.fit_transform(data.copy())
        except Exception as error:
            outputs.append({"fit_error": [type(error).__name__, str(error)]})
            continue
        output = {"fit": record(fitted)}
        if queries is not None:
            try:
                transformed = model.transform(queries)
            except Exception as error:
                output["transform_error"] = [type(error).__name__, str(error)]
            else:
                output["transform"] = record(transformed)
        outputs.append(output)
    assert outputs[0] == outputs[1]
    return outputs[0]


def compare(observed, calculated):
    structure_equal = all(
        observed[key] == calculated[key] for key in ("shape", "indptr", "indices")
    )
    return {
        "structure_equal": structure_equal,
        "different_weight_bits": (
            sum(a != b for a, b in zip(observed["data_bits"], calculated["data_bits"]))
            if structure_equal else None
        ),
        "weight_count": len(observed["data_bits"]),
    }


def generate(estimator):
    cases = {}
    for name, spec in inputs():
        data, queries = features(spec)
        cutoff = spec.get("cutoff")
        if cutoff == "inf":
            cutoff = np.inf
        options = spec.get("options", {})
        results = {}
        for metric in ("cosine", "correlation"):
            observed = observe(estimator, data, metric, spec["k"], cutoff, options, queries)
            narrow_products = metric == "cosine"
            comparison = {}
            for label, narrow in (
                ("angular_formula", narrow_products), ("reversed_product_precision", not narrow_products),
            ):
                with np.errstate(over="ignore", invalid="ignore"):
                    independent = independent_distances(
                        data, center=metric == "correlation", narrow_products=narrow
                    )
                calculated = observe(
                    estimator, independent, "precomputed", spec["k"],
                    2.0 if cutoff is None else cutoff, options,
                )
                comparison[label] = (
                    compare(observed["fit"], calculated["fit"])
                    if "fit" in observed and "fit" in calculated else calculated.get("fit_error")
                )
            results[metric] = observed | {"independent_comparison": comparison}
        cases[name] = {"input": spec, "metrics": results}
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "llvmlite", "pynndescent",
        )},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        from umap import UMAP
    except Exception as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate(UMAP)
    cases = result.pop("cases")
    # Put each case on one line to keep repeated CSR/COO arrays compact.
    lines = json.dumps(result, indent=2).splitlines()
    lines[-1] = '  "cases": {'
    lines[-2] += ","
    lines.append(",\n".join(f"    {json.dumps(name)}: {json.dumps(case)}" for name, case in cases.items()))
    lines.extend(["  }", "}", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
