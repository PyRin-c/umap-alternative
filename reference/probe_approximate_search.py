"""Probe NNDescent construction controls and export inputs for the Rust check."""

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

# Keep isolated Python mode while sharing the original fixture input generator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_approximate_neighbors_fixture import (
    FIXTURES, PARAMETERS, capture, cases, compact_fit, compare, fingerprint, inputs, np,
)


def ranks(index):
    indices, distances = index.neighbor_graph
    return rank_record(indices, distances)


def rank_record(indices, distances):
    return {"indices": indices.ravel().tolist(), "distance_dtype": str(distances.dtype),
            "bits": distances.astype(np.float32).view(np.uint32).ravel().tolist()}


def wheel_index(search, training, metric, k, **controls):
    return search(training.copy(), metric=metric, n_neighbors=k, n_jobs=1,
                  random_state=42, **controls)


def controlled(search):
    data = np.array([0, 3, 4, 7, 32, 15, 22, 52], dtype=np.float32).reshape(-1, 1)
    seeds = np.array([[0, -1], [1, 0], [2, 0], [3, 4], [4, 5], [5, 6], [6, 7], [7, 3]], dtype=np.int32)
    rounds = {}
    for n_iters in (0, 1, 2, 3):
        rounds[n_iters] = ranks(wheel_index(search, data, "euclidean", 2,
            init_graph=seeds.copy(), tree_init=False, n_iters=n_iters, max_candidates=60, delta=0))
    candidate_caps = {
        cap: ranks(wheel_index(search, data, "euclidean", 2, init_graph=seeds.copy(),
                               tree_init=False, n_iters=2, max_candidates=cap, delta=0))
        for cap in (1, 2)
    }

    # With a single tree and no refinement, shifting the maximum leaf size
    # distinguishes leaf-local pair scoring from a global random seed graph.
    line = (np.arange(32, dtype=np.float32) ** 2).reshape(-1, 1)
    leaf_sweep = {}
    for n_trees, leaf_size in ((0, 4), (1, 4), (1, 32)):
        output = ranks(wheel_index(search, line, "euclidean", 3, n_trees=n_trees,
                                   leaf_size=leaf_size, n_iters=0, max_candidates=60))
        nearest = np.argsort(abs(line - line.T), axis=1, kind="stable")[:, :3].ravel()
        leaf_sweep[f"trees={n_trees}, leaf={leaf_size}"] = {
            "aligned_rank_matches": sum(a == int(b) for a, b in zip(output["indices"], nearest)),
            "rank_fingerprint": fingerprint(output["indices"]),
            "first_four_rows": output["indices"][:12],
        }
    return {"data": data[:, 0].tolist(), "seeds": seeds.ravel().tolist(), "rounds": rounds,
            "candidate_caps": candidate_caps, "leaf_sweep": leaf_sweep}


def first_round_changes(before, after, n_rows, k):
    previous = [set(row) - {-1, i} for i, row in enumerate(np.reshape(before, (n_rows, k)).tolist())]
    incident = [row.copy() for row in previous]
    for source, row in enumerate(previous):
        for target in row:
            incident[target].add(source)
    added = [(source, target) for source, row in enumerate(np.reshape(after, (n_rows, k)).tolist())
             for target in set(row) - previous[source] - {-1, source}]
    return {"added_directed_ranks": len(added),
            "without_shared_seed_pivot": sum(not incident[a] & incident[b] for a, b in added),
            "maximum_incident_seed_degree": max(map(len, incident))}


def observe(umap, training, queries, spec, force, supplied=None):
    params = PARAMETERS | dict(metric=spec["metric"], n_neighbors=spec["k"],
                               force_approximation_algorithm=force,
                               disconnection_distance=spec.get("cutoff"))
    if supplied is not None:
        params["precomputed_knn"] = supplied
    model = umap(**params)
    fit = capture(model.fit_transform(training.copy()))
    return {"fit": fit, "query": capture(model.transform(queries.copy()))}


def sweep(umap, search, baseline, names):
    for name, spec in cases():
        n_rows, k, metric = spec["rows"], spec["k"], spec["metric"]
        if n_rows not in (5, 96, 4095, 4096) or (names and name not in names):
            continue
        training, queries = inputs(spec)
        expected = observe(umap, training, queries, spec, True)
        reference = baseline[name]["forced"]
        if name == "repeated_k2" and not all(np.isfinite(np.array(expected["query"]["bits"], dtype=np.uint32).view(np.float32))):
            # This recorded variant differs only in its first two weight bits.
            variant = expected["query"]["bits"][:2]
            assert np.isnan(np.array(variant, dtype=np.uint32).view(np.float32)).all()
            expected["query"]["bits"][:2] = reference["query"]["bits"][:2]
        fit = expected["fit"]
        summary = compact_fit(fit, len(reference["fit"]["first_rows"]))
        summary["first_rows"] = [[list(entry) for entry in row] for row in summary["first_rows"]]
        assert summary == reference["fit"], name
        assert expected["query"] == reference["query"], name

        trees = min(64, 5 + round(n_rows ** 0.5 / 20))
        iterations = max(5, round(np.log2(n_rows)))
        controls = dict(n_trees=trees, max_candidates=60)
        fitted_k = min(k, n_rows - 1)
        snapshots = {}
        for count in (0, 1, iterations):
            index = wheel_index(search, training, metric, fitted_k, n_iters=count, **controls)
            snapshots[str(count)] = ranks(index)
        initial, first = snapshots["0"], snapshots["1"]
        first_round = first_round_changes(initial["indices"], first["indices"], n_rows, fitted_k)
        restarts = {}
        if n_rows == 4096:
            seeds = np.asarray(initial["indices"], dtype=np.int32).reshape(n_rows, fitted_k)
            for count in (1, iterations):
                restarts[str(count)] = ranks(wheel_index(search, training, metric, fitted_k,
                    init_graph=seeds.copy(), tree_init=False, n_iters=count, max_candidates=60))
            restart = restarts["1"]
            first_round["restart_from_zero_round"] = first_round_changes(
                initial["indices"], restart["indices"], n_rows, fitted_k)
            first_round["restart_aligned_columns"] = sum(a == b for a, b in zip(first["indices"], restart["indices"]))

        # Replay the end-to-end handoff using the constructor candidate and its
        # public query method. Keep small-width padding checks in the original fixture.
        replay = None
        query_sweep = {}
        if fitted_k == k:
            columns, distances = index.neighbor_graph
            public = observe(umap, training, queries, spec, True, (columns.copy(), distances.copy(), index))
            replay = {stage: compare(expected[stage], public[stage]) for stage in expected}
            if n_rows >= 4095 or (n_rows == 96 and "gaussian" in name):
                for epsilon in (0.0, 0.12, 0.24, 0.5):
                    query_indices, query_distances = index.query(queries.copy(), k=k, epsilon=epsilon)
                    query_sweep[str(epsilon)] = rank_record(query_indices, query_distances)

        automatic = None
        if n_rows >= 4095:
            automatic = observe(umap, training, queries, spec, False)
            assert compare(automatic["fit"], fit) == baseline[name]["automatic_vs_forced"]["fit"], name
            assert automatic["query"] == baseline[name]["automatic"]["query"], name
        print(name, flush=True)
        yield name, {
            "metric": metric, "cutoff": spec.get("cutoff"), "n_rows": n_rows,
            "n_features": spec["features"], "width": k, "fitted_k": fitted_k,
            "n_query_rows": len(queries), "iterations": iterations,
            "training_bits": training.view(np.uint32).ravel().tolist(),
            "query_bits": queries.view(np.uint32).ravel().tolist(),
            "snapshots": snapshots, "restarts": restarts, "forced": expected, "automatic": automatic,
            "constructor_replay": replay, "query_epsilon_sweep": query_sweep, "first_round": first_round,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Restrict to named fixture cases.")
    args = parser.parse_args()
    architecture = platform.machine().lower().replace("amd64", "x86_64")
    fixture = FIXTURES / f"umap-learn-0.6.0-approximate-neighbors-{platform.system().lower()}-{architecture}.json"
    baseline = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent
            from umap import UMAP
            observations = {"controlled": controlled(NNDescent),
                            "cases": dict(sweep(UMAP, NNDescent, baseline, args.case))}
    except Exception as error:
        # Never emit a wheel exception message or traceback containing JIT source.
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(observations, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
