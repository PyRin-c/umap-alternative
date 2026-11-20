"""Capture densMAP and output_dens using the published umap-learn 0.6.0 API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

THREADS = dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"),
    "1",
)
os.environ.update(THREADS)

import numpy as np
from umap import UMAP


REFERENCE = Path(__file__).resolve().parent
PATH = REFERENCE / "fixtures/umap-learn-0.6.0-densmap-windows-x86_64.json"
DATA = [[0, 0, 0], [.2, .07, .1], [.25, .89, .13], [.76, .1, .83],
        [.99, .95, .37], [1.9, 1.31, .33], [2.61, 1.7, .9], [3.3, 1.2, 1.1]]
INIT = [[0, 6], [1, 3], [4, 0], [2, 1], [5, 4], [6, 5], [3, 2], [7, 7]]
PAIRS = [[0, 0], [.00008, .00007], [10, 10], [9.9, 9.7], [.5, 8], [.4, 8.1]]
BASE = dict(n_neighbors=4, n_components=2, metric="euclidean", n_jobs=1,
            random_state=42, compatibility_layout=True, learning_rate=.1,
            negative_sample_rate=2, a=1.7, b=.9)
CHECKPOINTS = (0, 1, 2, 3, 10, 30)
DISTANCE_SCALES = (1e-3, 1e-2, .1, 1., 3., 10., 100.)
OUTPUT = {"output_dens": True}
DENS = {**OUTPUT, "densmap": True}
OPTIONS = {
    "standard": {},
    "density_output": OUTPUT,
    "densmap": DENS,
    "dens_lambda_0": {**DENS, "dens_lambda": 0.0},
    "dens_frac_0": {**DENS, "dens_frac": 0.0},
    "dens_frac_1": {**DENS, "dens_frac": 1.0},
    "dens_var_0": {**DENS, "dens_var_shift": 0.0},
}


def floats(array):
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise AssertionError(f"expected a finite float32 array, got {array.dtype}")
    return np.ascontiguousarray(array).view(np.uint32)


def array_record(array):
    return {"dtype": array.dtype.name, "shape": list(array.shape),
            "bits_uint32": floats(array).flatten().tolist()}


def graph_record(graph):
    return {"indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(),
            "weights": array_record(graph.data)}


def intern(records, record):
    if record not in records:
        records.append(record)
    return records.index(record)


def run(data, init, **options):
    model = UMAP(**(BASE | options), init=init.copy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_transform(data.copy())
    arrays = result if isinstance(result, tuple) else (result,)
    expected_shapes = [(len(data), 2)]
    if options.get("output_dens"):
        expected_shapes += [(len(data),), (len(data),)]
    if (isinstance(result, tuple) != bool(options.get("output_dens"))
            or [a.shape for a in arrays] != expected_shapes):
        raise AssertionError(f"unexpected fit_transform result for {options}")
    return model.graph_.tocsr(), arrays, [item.category.__name__ for item in caught]


def capture_error(action):
    try:
        action()
    except (ValueError, NotImplementedError) as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("expected the public call to raise")


def error_cases(data, init):
    cases = []
    for parameter, bound, toward in [("dens_frac", 0., -1.), ("dens_frac", 1., 2.),
                                     ("dens_lambda", 0., -1.), ("dens_var_shift", 0., -1.)]:
        value = float(np.nextafter(bound, toward))
        params = {**DENS, "n_epochs": 3, parameter: value}
        cases.append({"options": params, **capture_error(lambda: run(data, init, **params))})
    query = data[:2] + np.float32(.02)
    model = UMAP(**BASE, densmap=True, init=init.copy(), n_epochs=3).fit(data.copy())
    return {"invalid_fit_transform": cases, "transform": {
        "options": {"densmap": True, "n_epochs": 3}, "query_float32": query.tolist(),
        **capture_error(lambda: model.transform(query)),
    }}


def schedule(data, init, final_bits):
    summaries = {}
    for total in (2, 3, 10, 30):
        epochs = list(range(total + 1))
        histories = {}
        for mode in ("standard", "densmap", "dens_lambda_0", "dens_frac_0", "dens_frac_1"):
            params = {**OPTIONS[mode], "output_dens": False}
            model = UMAP(**BASE, **params, init=init.copy(), n_epochs=epochs).fit(data.copy())
            checkpoints = [floats(array) for array in model.embedding_list_]
            if (len(checkpoints) != len(epochs)
                    or not np.array_equal(checkpoints[0], final_bits[0, "standard"])
                    or not np.array_equal(checkpoints[-1], final_bits[total, mode])
                    or not np.array_equal(checkpoints[-2], checkpoints[-1])):
                raise AssertionError(f"unexpected list-form checkpoints for {total}, {mode}")
            histories[mode] = np.stack(checkpoints)
        control = histories.pop("standard")
        summaries[str(total)] = {
            "changed_elements_since_previous_standard":
                np.count_nonzero(control[1:] != control[:-1], axis=(1, 2)).tolist(),
            "elements_different_from_standard": {
                mode: np.count_nonzero(value != control, axis=(1, 2)).tolist()
                for mode, value in histories.items()
            },
        }
    return summaries


def euclidean(points, dtype):
    coords = points.astype(dtype)
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def embedding_distances(points):
    """Round x*x, then y*y + x*x in one operation, then the square root."""
    delta = (points[:, None, :] - points[None, :, :]).astype(np.float64)
    x_squared = (delta[..., 0] ** 2).astype(np.float32)
    squared = (x_squared.astype(np.float64) + delta[..., 1] ** 2).astype(np.float32)
    return np.sqrt(squared)


def refit_graph(points, distances, k):
    neighbors = np.argsort(distances, axis=1)[:, :k].astype(np.int32)
    neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
    refit = UMAP(**(BASE | {"n_neighbors": k}), init=points.copy(), n_epochs=0,
                 precomputed_knn=(neighbors, neighbor_distances))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"precomputed_knn\[2\]", category=UserWarning)
        return refit.fit(points.copy()).graph_.toarray()


def fuzzy_union(distances, k):
    """Independent binary solve for the published unpruned, Euclidean fuzzy graph."""
    directed = np.zeros_like(distances)
    target = np.log2(k)
    for i, row in enumerate(distances):
        neighbors = np.argsort(row)[1:k]  # These probes have distinct points.
        deltas = row[neighbors] - row[neighbors[0]]
        if k == 2:
            directed[i, neighbors] = 1.0
            continue
        lower, upper = 0., max(1., float(deltas.max()))
        while np.exp(-deltas / upper).sum() < target:
            upper *= 2
        for _ in range(60):
            midpoint = (lower + upper) / 2
            if np.exp(-deltas / midpoint).sum() < target:
                lower = midpoint
            else:
                upper = midpoint
        directed[i, neighbors] = np.exp(-deltas / upper)
    return directed + directed.T - directed * directed.T


def weighted_log(weights, terms, *, both=True, dtype=np.float32,
                 term_dtype=None, offset=1e-8):
    """Evaluate a weighted log mean in row-major order with explicit rounding."""
    numerator = np.zeros(len(weights), dtype=dtype)
    denominator = np.zeros_like(numerator)
    term_dtype = term_dtype or dtype
    rows, columns = np.nonzero(weights)
    for i, j in zip(rows, columns):
        weight = dtype(weights[i, j])
        term = term_dtype(weight) * term_dtype(terms[i, j])
        for point in ((i, j) if both else (i,)):
            numerator[point] += term
            denominator[point] += weight
    return np.log(numerator / denominator + dtype(offset)).astype(np.float32)


def score(actual, predicted):
    return {"equal_float32_bits": int(np.count_nonzero(floats(actual) == floats(predicted))),
            "total": actual.size,
            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - predicted)))}


def original_scores(cohort):
    trials = {}
    for weights, distance, radius in cohort:
        wide_distance = distance.astype(np.float64)
        squared = wide_distance * wide_distance
        candidates = {
            "log_mean_of_squared_distances_wide_product":
                weighted_log(weights, squared, term_dtype=np.float64),
            "log_mean_of_squared_distances_separate_roundings":
                weighted_log(weights, distance * distance),
            "log_mean_of_squared_distances_one_endpoint":
                weighted_log(weights, squared, both=False, term_dtype=np.float64),
            "log_mean_of_squared_distances_float64":
                weighted_log(weights, squared, dtype=np.float64),
            "log_mean_of_unsquared_distances": weighted_log(weights, distance),
        }
        for name, result in candidates.items():
            trials.setdefault(name, []).append(result)
    all_radii = np.concatenate([radius for _, _, radius in cohort])
    return {name: score(all_radii, np.concatenate(results)) for name, results in trials.items()}


def embedding_scores(cohort, k):
    trials = {}
    actual = []
    same_support = 0
    weight_error = 0.0
    for embedding, radius, original in cohort:
        precise = euclidean(embedding, np.float64)
        distance = embedding_distances(embedding)
        weights = refit_graph(embedding, distance, k)
        theory = fuzzy_union(distance.astype(np.float64), k)
        same_support += np.array_equal(weights > 0, theory > 0)
        weight_error = max(weight_error, float(np.max(np.abs(weights - theory))))
        squared = weighted_log(weights, distance * distance)
        candidates = {
            "log_mean_of_unsquared_distances_refit_graph":
                weighted_log(weights, distance, term_dtype=np.float64),
            "log_mean_of_unsquared_distances_with_separate_roundings":
                weighted_log(weights, distance),
            "log_mean_with_distance_squares_rounded_separately":
                weighted_log(weights, euclidean(embedding, np.float32), term_dtype=np.float64),
            "log_mean_of_unsquared_distances_refit_graph_one_endpoint":
                weighted_log(weights, distance, both=False, term_dtype=np.float64),
            "log_mean_of_unsquared_distances_refit_graph_float64":
                weighted_log(weights, distance, dtype=np.float64),
            "log_mean_of_unsquared_distances_theory_graph":
                weighted_log(theory, precise, dtype=np.float64),
            "log_mean_of_unsquared_distances_input_graph": weighted_log(original, distance),
            "log_unweighted_mean_of_unsquared_distances_refit_graph":
                weighted_log(weights > 0, distance),
            "log_mean_of_squared_distances_refit_graph": squared,
            "half_log_mean_of_squared_distances_refit_graph": np.float32(.5) * squared,
        }
        for name, result in candidates.items():
            trials.setdefault(name, []).append(result)
        actual.append(radius)
    flat = np.concatenate(actual)
    return {"embeddings": len(cohort), "theory_graphs_with_refit_support": same_support,
            "max_theory_vs_refit_weight_error": weight_error,
            "candidates": {name: score(flat, np.concatenate(output)) for name, output in trials.items()}}


def pair_scores(data, graph, outputs):
    weights = graph.toarray()
    embedding, original_radius, embedding_radius = outputs
    distance = euclidean(data, np.float64).astype(np.float32).astype(np.float64)
    embedding_distance = embedding_distances(embedding)
    embedding_weights = fuzzy_union(euclidean(embedding, np.float64), 2)
    return {
        space: {str(epsilon): score(radius, weighted_log(g, terms, term_dtype=term_dtype,
                                                       offset=epsilon))
                for epsilon in (0., 1e-9, 1e-8, 1e-7)}
        for space, radius, g, terms, term_dtype in (
            ("original", original_radius, weights, distance * distance, np.float64),
            ("embedding", embedding_radius, embedding_weights, embedding_distance, np.float64),
        )
    }


def build():
    data, init = np.array(DATA, np.float32), np.array(INIT, np.float32)
    pairs = np.array(PAIRS, np.float32)
    arrays, graphs, cases, histories, projections = [], [], {}, {}, {}
    original_distances = euclidean(data, np.float64)
    original_cohort = []
    for epoch in CHECKPOINTS:
        for mode, options in OPTIONS.items():
            graph, outputs, notes = run(data, init, n_epochs=epoch, **options)
            ids = [intern(arrays, array_record(output)) for output in outputs]
            cases[f"{epoch}-{mode}"] = {"graph": intern(graphs, graph_record(graph)),
                                        "outputs": ids, "warnings": notes}
            histories[epoch, mode] = floats(outputs[0])
            if len(outputs) == 3:
                projections[tuple(ids)] = (outputs[0], outputs[2], graph.toarray())
            if epoch == 0 and mode == "density_output":
                original_cohort.append((graph.toarray(), original_distances.astype(np.float32), outputs[1]))
    first = cases["0-density_output"]
    precomputed = {}
    for scale in DISTANCE_SCALES:
        distance = (original_distances * scale).astype(np.float32)
        graph, outputs, notes = run(distance, init, n_epochs=0, metric="precomputed", **OUTPUT)
        original_cohort.append((graph.toarray(), distance, outputs[1]))
        precomputed[str(scale)] = {"graph": intern(graphs, graph_record(graph)),
                                   "outputs": [intern(arrays, array_record(out)) for out in outputs],
                                   "warnings": notes}
    pair_graph, pair_outputs, pair_notes = run(pairs, pairs, n_epochs=0, n_neighbors=2, **OUTPUT)
    pair_record = {
        "data_and_init_float32": PAIRS, "options": {"n_neighbors": 2, "n_epochs": 0, **OUTPUT},
        "graph": intern(graphs, graph_record(pair_graph)),
        "outputs": [intern(arrays, array_record(output)) for output in pair_outputs],
        "warnings": pair_notes,
        "offset_candidates": pair_scores(pairs, pair_graph, pair_outputs),
    }
    neighbor_widths = []
    for width in (2, 3, 7, 8, 9, 15):
        options = {"n_neighbors": width, "n_epochs": 2, **OUTPUT}
        graph, outputs, notes = run(data, init, **options)
        neighbor_widths.append({
            "options": options,
            "graph": intern(graphs, graph_record(graph)),
            "outputs": [intern(arrays, array_record(output)) for output in outputs],
            "warnings": notes,
        })
    lock = (REFERENCE / "uv.lock").read_text(encoding="utf-8").encode("utf-8")
    return {
        "schema_version": 1,
        "environment": {"system": platform.system(), "machine": platform.machine(),
                        "python": platform.python_version(), "threads": THREADS,
                        "reference_uv_lock_lf_sha256": hashlib.sha256(lock).hexdigest(),
                        "packages": {name: version(name) for name in (
                            "umap-learn", "numpy", "scipy", "scikit-learn", "numba",
                            "llvmlite", "pynndescent",
                        )}},
        "data_float32": DATA, "init_float32": INIT, "common_options": BASE,
        "mode_options": OPTIONS, "cases": cases, "arrays": arrays, "graphs": graphs,
        "errors": error_cases(data, init), "schedule": schedule(data, init, histories),
        "arithmetic": {"original_radius_id": first["outputs"][1],
                       "original_base": original_scores(original_cohort[:1]),
                       "original_base_and_precomputed": original_scores(original_cohort),
                       "precomputed_at_n_epochs_0": precomputed,
                       "embedding": embedding_scores(list(projections.values()), BASE["n_neighbors"]),
                       "close_pairs": pair_record,
                       "neighbor_widths": neighbor_widths},
    }


def fixture_json(value, level=0):
    indent = "  " * level
    if isinstance(value, dict) and level < 2:
        items = [f"{json.dumps(key)}: {fixture_json(item, level + 1)}"
                 for key, item in value.items()]
        opening, closing = "{", "}"
    elif level == 1 and isinstance(value, list) and value and isinstance(value[0], dict):
        items = [fixture_json(item, level + 1) for item in value]
        opening, closing = "[", "]"
    else:
        return json.dumps(value, allow_nan=False)
    return opening + "\n" + ",\n".join(indent + "  " + item for item in items) + "\n" + indent + closing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PATH)
    parser.add_argument("--check", action="store_true", help="compare with the regenerated fixture")
    args = parser.parse_args()
    fixture = build()
    contents = fixture_json(fixture) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != contents:
            parser.exit(1, f"{args.output} differs from the regenerated fixture\n")
    else:
        args.output.write_text(contents, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
