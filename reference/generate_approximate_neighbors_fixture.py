"""Capture seeded approximate-neighbor behavior through public UMAP calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import warnings
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


PARAMETERS = dict(
    compatibility_layout=True, transform_mode="graph", init="random", n_epochs=30,
    n_jobs=1, random_state=42, transform_seed=42,
)
FLOAT32_BELOW_ONE = np.nextafter(np.float32(1), np.float32(0)).item()
FLOAT32_ABOVE_ONE = np.nextafter(np.float32(1), np.float32(2)).item()
FIXTURES = Path(__file__).parent / "fixtures"
DUPLICATE_TIE_REPEATS = 40


def cases():
    gaussian = dict(training_seed=57, query_seed=62, features=7, query_rows=4)
    for metric in ("euclidean", "cosine", "correlation"):
        for k in (2, 5):
            yield f"gaussian_96_{metric}_k{k}", gaussian | dict(rows=96, metric=metric, k=k)
    for rows in (3999, 4000, 4095, 4096, 4097):
        yield f"gaussian_{rows}_euclidean", gaussian | dict(rows=rows, metric="euclidean", k=5)
    for metric in ("cosine", "correlation"):
        for rows in (4095, 4096):
            yield f"gaussian_{rows}_{metric}", gaussian | dict(rows=rows, metric=metric, k=5)
    for distinct_rows in (4095, 4096):
        yield f"unique_4097_from_{distinct_rows}", gaussian | dict(
            rows=4097, metric="euclidean", k=5, distinct_rows=distinct_rows, unique=True,
        )
    for k in (2, 5):
        yield f"repeated_k{k}", dict(
            rows=96, features=7, metric="euclidean", k=k,
            prefix=[0, 0, 0, 1, 1, 1, -1, -1], queries=[0, 0.5, 2, 3],
        )
    cutoff = dict(
        rows=96, features=7, metric="euclidean", k=7, queries=[0, 2],
        prefix=[0, 0.25, 0.5, FLOAT32_BELOW_ONE, 1, FLOAT32_ABOVE_ONE, 1.5, 2],
    )
    yield "adjacent_cutoff_uncut", cutoff
    yield "adjacent_cutoff_one", cutoff | {"cutoff": 1}
    clamp = dict(rows=5, features=7, metric="euclidean", prefix=[0, 1, 4, 10, 18], queries=[-1, 2, 8, 20])
    for k in (4, 5, 8):
        yield f"five_rows_k{k}", clamp | {"k": k, "fit_rank_replay": True}


def inputs(spec):
    rows, features = spec["rows"], spec["features"]
    if "prefix" in spec:
        training = np.zeros((rows, features), dtype=np.float32)
        prefix = spec["prefix"]
        training[:len(prefix), 0] = prefix
        training[len(prefix):, 0] = np.arange(4, 4 + rows - len(prefix))
        queries = np.zeros((len(spec["queries"]), features), dtype=np.float32)
        queries[:, 0] = spec["queries"]
    else:
        training = np.random.default_rng(spec["training_seed"]).standard_normal((rows, features)).astype(np.float32)
        queries = np.random.default_rng(spec["query_seed"]).standard_normal((spec["query_rows"], features)).astype(np.float32)
        if "distinct_rows" in spec:
            training[spec["distinct_rows"]:] = training[0]
    return training, queries


def public_call(call, *args, **kwargs):
    try:
        return call(*args, **kwargs), None
    except Exception as error:
        # JIT diagnostics can append implementation excerpts to multiline messages.
        return None, [type(error).__name__, str(error).partition("\n")[0]]


def capture(graph):
    assert graph.dtype == np.float32
    record = {"shape": list(graph.shape), "format": graph.format, "bits": graph.data.view(np.uint32).tolist()}
    if graph.format == "csr":
        return record | {"indptr": graph.indptr.tolist(), "columns": graph.indices.tolist()}
    assert graph.format == "coo"
    return record | {"rows": graph.row.tolist(), "columns": graph.col.tolist()}


def observe(umap, training, queries, spec, force):
    model, error = public_call(
        umap, **PARAMETERS, metric=spec["metric"], n_neighbors=spec["k"],
        force_approximation_algorithm=force, disconnection_distance=spec.get("cutoff"),
        unique=spec.get("unique", False),
    )
    if error:
        return {"fit_error": error}
    graph, error = public_call(model.fit_transform, training.copy())
    if error:
        return {"fit_error": error}
    result = {"fit": capture(graph)}
    transformed, error = public_call(model.transform, queries.copy())
    if error:
        return result | {"query_error": error}
    return result | {"query": capture(transformed)}


def handoff_cases():
    indices = [[6, 4, 2, 5, 7], [5, 3, 1, 2, 6], [7, 6, 5, 4, 3]]
    distances = [[0, FLOAT32_BELOW_ONE, 1, FLOAT32_ABOVE_ONE, 2], [0, 0, 0.125, 0.125, 0.75], [0.01, 0.02, 0.04, 0.4, 0.8]]
    finite = dict(indices=indices, distances=distances, metric="euclidean", connectivity=1.0, cutoff=None)
    yield "finite", finite
    yield "adjacent_cutoff", finite | {"cutoff": 1.0}
    yield "fractional_connectivity", finite | {"connectivity": 2.5, "cutoff": 1.0}
    yield "angular_index", finite | {"metric": "cosine"}
    yield "tied_k2", finite | {"indices": [[4, 2], [5, 3]], "distances": [[0.125, 0.125], [0, 0]]}
    missing = finite | {
        "indices": indices[:1] + [[5, 3, 1, 2, -1], indices[2], [-1] * 5],
        "distances": distances[:1] + [[0, 0.25, 0.5, 0.75, None], [0, 0, 0, 0, 0.001], [None] * 5],
        "cutoff": 1.0,
    }
    yield "missing_slots", missing
    yield "missing_slots_fractional_connectivity", missing | {"connectivity": 2.5}
    yield "rank_width_exceeds_training_rows", finite | {
        "indices": [row + [-1] * 4 for row in indices],
        "distances": [row + [None] * 4 for row in distances],
    }


def generate_handoff(umap, search_type):
    class RankedSearch(search_type):
        def __init__(self, training, metric, k, indices, distances):
            super().__init__(training, metric=metric, n_neighbors=min(k, len(training) - 1), n_jobs=1, random_state=42)
            self.supplied = indices, distances
            self.calls = []

        def query(self, query_data, k=10, epsilon=0.1):
            self.calls.append(dict(shape=list(query_data.shape), k=k, epsilon=epsilon))
            return tuple(array.copy() for array in self.supplied)

    train = np.arange(16, dtype=np.float32).reshape(8, 2) / np.float32(16)
    train_distances = np.linalg.norm(train[:, None] - train[None, :], axis=2)
    for name, spec in handoff_cases():
        indices = np.array(spec["indices"], dtype=np.int32)
        distances = np.array([[np.inf if d is None else d for d in row] for row in spec["distances"]], dtype=np.float32)
        query_rows, k = indices.shape
        queries = -train[1:query_rows + 1].copy()
        train_indices = np.argsort(train_distances, axis=1, kind="stable")[:, :k].astype(np.int32)
        selected = np.take_along_axis(train_distances, train_indices, axis=1)
        padding = ((0, 0), (0, max(0, k - len(train))))
        train_indices = np.pad(train_indices, padding, constant_values=-1)
        selected = np.pad(selected, padding, constant_values=np.inf)
        outputs = []
        for _ in range(2):
            index, error = public_call(RankedSearch, train.copy(), spec["metric"], k, indices, distances)
            if error:
                outputs.append({"index_error": error})
                continue
            model, error = public_call(
                umap, **PARAMETERS, metric=spec["metric"], n_neighbors=k,
                force_approximation_algorithm=True, disconnection_distance=spec["cutoff"],
                local_connectivity=spec["connectivity"],
                precomputed_knn=(train_indices.copy(), selected.copy(), index),
            )
            if not error:
                _, error = public_call(model.fit_transform, train.copy())
            if error:
                outputs.append({"fit_error": error})
                continue
            query, error = public_call(model.transform, queries.copy())
            result = {"query_error": error} if error else {"query": capture(query)}
            outputs.append(result | {"public_query_calls": index.calls})
        yield name, {
            "input": spec, "training_shape": list(train.shape), "query_shape": list(queries.shape),
            "query_rank_width": k, "scale_neighbors": min(k, len(train) - 1),
            **outputs[0], "fresh_fit_repeats_equal": outputs[0] == outputs[1],
        }


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.asarray(values, dtype="<u4").tobytes())
    return digest.hexdigest()


def compact_fit(graph, sample_count):
    pointers, columns, bits = graph["indptr"], graph["columns"], graph["bits"]
    return {
        "shape": graph["shape"], "nnz": len(bits),
        "support_sha256": fingerprint(graph["shape"], pointers, columns),
        "weight_sha256": fingerprint(bits),
        "isolated_rows": sum(a == b for a, b in zip(pointers, pointers[1:])),
        "first_rows": [list(zip(columns[a:b], bits[a:b])) for a, b in zip(pointers[:sample_count], pointers[1:])],
    }


def entries(graph):
    if graph["format"] == "csr":
        rows = (row for row, (a, b) in enumerate(zip(graph["indptr"], graph["indptr"][1:])) for _ in range(b - a))
    else:
        rows = graph["rows"]
    return list(zip(rows, graph["columns"], graph["bits"]))


def compare(left_graph, right_graph):
    def grouped(records):
        groups = defaultdict(list)
        for row, col, bits in records:
            if bits & 0x7fffffff:
                groups[row, col].append(bits)
        return {coordinate: sorted(bits) for coordinate, bits in groups.items()}

    left, right = entries(left_graph), entries(right_graph)
    lhs, rhs = grouped(left), grouped(right)
    shared = lhs.keys() & rhs.keys()
    differences = [(a, b) for ij in shared for a, b in zip(lhs[ij], rhs[ij]) if a != b]
    finite = [(a, b) for a, b in differences if finite_bits(a) and finite_bits(b)]
    floats = np.asarray(finite, dtype=np.uint32).view(np.float32).astype(np.float64)
    left_coordinates = [(i, j) for i, j, _ in left]
    right_coordinates = [(i, j) for i, j, _ in right]
    left_counts, right_counts = Counter(left_coordinates), Counter(right_coordinates)
    result = {
        "identical": left_graph == right_graph,
        "left_only_nonzero": len(lhs.keys() - rhs.keys()),
        "right_only_nonzero": len(rhs.keys() - lhs.keys()),
        "shared_nonzero": len(shared),
        "shared_weight_bit_differences": len(differences),
        "max_shared_abs_weight_error": float(np.max(np.abs(floats[:, 0] - floats[:, 1]))) if finite else 0.0,
        "max_shared_weight_ulp": max((abs(a - b) for a, b in finite), default=0),
    }
    if len(finite) < len(differences):
        result["shared_nonfinite_weight_bit_differences"] = len(differences) - len(finite)
    if left_graph["format"] == "coo":
        result |= {
            "left_only_stored_coordinates": sum((left_counts - right_counts).values()),
            "right_only_stored_coordinates": sum((right_counts - left_counts).values()),
            "ordered_coordinates_equal": left_coordinates == right_coordinates,
        }
    return result


def finite_bits(bits):
    return (bits & 0x7f800000) != 0x7f800000


def compact_observation(observation, sample_count):
    if "fit" in observation:
        return observation | {"fit": compact_fit(observation["fit"], sample_count)}
    return observation


def observe_mode(umap, training, queries, spec, force, duplicate_probe):
    count = DUPLICATE_TIE_REPEATS if duplicate_probe else 2
    observations = [observe(umap, training, queries, spec, force) for _ in range(count)]
    first = observations[0]
    sample_count = 8 if "prefix" in spec else 2
    repeats = {"fresh_fit_repeats_equal": all(observation == first for observation in observations)}
    if duplicate_probe:
        variants = []
        for observation in observations:
            if observation not in variants:
                variants.append(observation)
        repeats["repeat_probe"] = [
            compact_observation(observation, sample_count) | {"count": observations.count(observation)}
            for observation in variants
        ]
        first = next((
            observation for observation in variants
            if "query" in observation and all(finite_bits(bit) for bit in observation["query"]["bits"])
        ), first)
    return first, compact_observation(first, sample_count) | repeats


def normalize_duplicate_repeat(document):
    """Keep the finite exemplar and validate the changing duplicate-query slots."""
    forced = document["cases"]["repeated_k2"]["forced"]
    variants = forced.pop("repeat_probe")
    forced.pop("fresh_fit_repeats_equal")
    assert sum(variant["count"] for variant in variants) == DUPLICATE_TIE_REPEATS
    for variant in variants:
        assert variant["fit"] == forced["fit"]
        query = variant["query"]
        bits, expected = query["bits"], forced["query"]["bits"]
        assert (query | {"bits": expected}) == forced["query"]
        assert bits[2:] == expected[2:]
        assert bits[:2] == expected[:2] or all(
            (bit & 0x7fffffff) > 0x7f800000 for bit in bits[:2]
        )
    return document


def exact_query_rank(training, queries, metric, query, k, cutoff):
    left = queries[query]
    if metric == "euclidean":
        distances = np.linalg.norm(training.astype(np.float64) - left, axis=1).astype(np.float32)
    else:
        right = training
        if metric == "correlation":
            left, right = left.astype(np.float64), right.astype(np.float64)
            left -= left.mean()
            right -= right.mean(axis=1)[:, None]
        left_norm = (left * left).sum(dtype=np.float64)
        right_norm = (right * right).sum(axis=1, dtype=np.float64)
        dot = (right * left).sum(axis=1, dtype=np.float64)
        denominator = np.sqrt(left_norm * right_norm)
        ratio = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
        distances = np.where((left_norm == 0) & (right_norm == 0), 0, 1 - ratio).astype(np.float32)
    nearest = np.argsort(distances, kind="stable")[:k]
    radius = distances[nearest[-1]]
    return {
        "columns": nearest.tolist(), "distance_bits": distances[nearest].view(np.uint32).tolist(),
        "rank_boundary_ties": np.flatnonzero(distances == radius).tolist(),
        "uncut_columns": nearest[distances[nearest] < cutoff].tolist(),
    }, distances


def query_baseline(training, queries, spec, outputs):
    cutoff = spec.get("cutoff", 2.0 if spec["metric"] != "euclidean" else np.inf)
    baselines = {}
    for label, output in outputs.items():
        if "query" in output:
            observed = entries(output["query"])
            width = min(len(observed) // len(queries), len(training))
            if width not in baselines:
                ranked = [exact_query_rank(training, queries, spec["metric"], i, width, cutoff) for i in range(len(queries))]
                baselines[width] = {"rows": [rank for rank, _ in ranked]}, [distances for _, distances in ranked]
            baseline, distance_rows = baselines[width]
            residuals = []
            for i, (rank, distances) in enumerate(zip(baseline["rows"], distance_rows)):
                boundary = distances[rank["columns"][-1]]
                columns = [col for row, col, bits in observed if row == i and bits & 0x7fffffff]
                residuals.append({
                    "beyond_rank_boundary": [col for col in columns if distances[col] > boundary],
                    "missing_strictly_nearer_uncut": sorted(set(np.flatnonzero(
                        (distances < boundary) & (distances < cutoff)
                    ).tolist()) - set(columns)),
                })
            baseline[label] = residuals
    return {width: baseline for width, (baseline, _) in baselines.items()}


def compare_fit_replays(umap, training, k, forced_graph):
    distances = np.abs(training[:, :1] - training[:, 0])
    comparisons = {}
    for width in (4, 5):
        order = np.argsort(distances, axis=1, kind="stable")[:, :width].astype(np.int32)
        selected = np.take_along_axis(distances, order, axis=1)
        padding = ((0, 0), (0, max(0, k - width)))
        indices = np.ascontiguousarray(np.pad(order, padding, constant_values=-1))
        selected = np.ascontiguousarray(np.pad(selected, padding, constant_values=np.inf))
        model, error = public_call(
            umap, **PARAMETERS, n_neighbors=k, force_approximation_algorithm=True,
            precomputed_knn=(indices, selected, None),
        )
        if not error:
            graph, error = public_call(model.fit_transform, training.copy())
        comparisons[width] = {"fit_error": error} if error else compare(forced_graph, capture(graph))
    return comparisons


def generate(umap):
    for name, spec in cases():
        training, queries = inputs(spec)
        outputs = {}
        compact = {}
        for label, force in (("automatic", False), ("forced", True)):
            duplicate_probe = name == "repeated_k2" and force
            first, compact[label] = observe_mode(umap, training, queries, spec, force, duplicate_probe)
            outputs[label] = first
            if "query" in first:
                compact[label]["query_rank_width"] = len(first["query"]["bits"]) // len(queries)
        comparisons = {
            stage: compare(outputs["automatic"][stage], outputs["forced"][stage])
            for stage in ("fit", "query") if all(stage in output for output in outputs.values())
        }
        result = {
            "input": spec, "training_shape": list(training.shape), "query_shape": list(queries.shape),
            "dataset_clamped_k": min(spec["k"], spec.get("distinct_rows", spec["rows"]) - 1), **compact,
            "automatic_vs_forced": comparisons,
        }
        # The deduplicated search and the input matrix can have different column orderings.
        if not spec.get("unique"):
            result["query_exhaustive_baseline"] = query_baseline(training, queries, spec, outputs)
        if spec.get("fit_rank_replay") and "fit" in outputs["forced"]:
            result["forced_vs_supplied_fit_ranks"] = compare_fit_replays(umap, training, spec["k"], outputs["forced"]["fit"])
        yield name, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--handoff", action="store_true", help="Probe the public supplied search-index handoff.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stage = "approximate-query-handoff" if args.handoff else "approximate-neighbors"
    machine = platform.machine().lower()
    architecture = "x86_64" if machine == "amd64" else machine
    output = args.output or FIXTURES / (
        f"umap-learn-0.6.0-{stage}-{platform.system().lower()}-{architecture}.json"
    )
    umap, error = public_call(__import__, "umap", fromlist=["UMAP"])
    if error:
        raise SystemExit(f"{error[0]}: {error[1]}") from None
    metadata = dict(
        platform=platform.system(), machine=platform.machine(), python=platform.python_version(),
        packages={name: version(name) for name in (
            "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "llvmlite", "pynndescent",
        )},
        settings=PARAMETERS, numerical_threads=1, fresh_fit_repeats=2,
    )
    if not args.handoff:
        metadata["forced_repeated_k2_fresh_fits"] = DUPLICATE_TIE_REPEATS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.handoff:
            search, error = public_call(__import__, "pynndescent", fromlist=["NNDescent"])
            if error:
                raise SystemExit(f"{error[0]}: {error[1]}") from None
            results = dict(generate_handoff(umap.UMAP, search.NNDescent))
        else:
            results = dict(generate(umap.UMAP))
    lines = json.dumps(metadata, indent=2).splitlines()[:-1]
    lines[-1] += ","
    lines.append('  "cases": {')
    lines.append(",\n".join(f"    {json.dumps(name)}: {json.dumps(case, allow_nan=False)}" for name, case in results.items()))
    lines.extend(["  }", "}", ""])
    result = "\n".join(lines)
    if args.check:
        expected = output.read_text(encoding="utf-8")
        if args.handoff:
            matches = result == expected
        else:
            matches = normalize_duplicate_repeat(json.loads(result)) == normalize_duplicate_repeat(json.loads(expected))
        if not matches:
            raise SystemExit(f"Fixture differs: {output}")
    else:
        output.write_text(result, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
