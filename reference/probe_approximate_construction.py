"""Observe hidden seed offers and projection controls on the public wheel."""

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_search import (
    FIXTURES, PARAMETERS, capture, cases, first_round_changes, np, rank_record, ranks, sweep,
)


class RecordedRandomState(np.random.RandomState):
    """Observe draws through the constructor's public random_state argument."""

    def __init__(self, seed, forest_seed=None):
        super().__init__(seed)
        self.draws = []
        self.forest_seed = forest_seed

    def randint(self, low, high=None, size=None, dtype=int):
        value = super().randint(low, high=high, size=size, dtype=dtype)
        if size is None and low == 0 and high == 1 << 62 and self.forest_seed is not None:
            value = self.forest_seed
        self.draws.append([low, high, size, np.asarray(value).tolist()])
        return value


def index(search, data, metric, seed, width=5, **options):
    return search(data.copy(), metric=metric, n_jobs=1, random_state=seed,
                  n_neighbors=width, max_candidates=60, **options)


def hidden_seed_control(search):
    data = np.array([0, 10, 11, -51, -29, 100, 138, 155], dtype=np.float32).reshape(-1, 1)
    seeds = np.array([[1, 2], [1, 3], [2, 4], [3, 5], [4, 5], [5, 6], [6, 7], [7, 6]], dtype=np.int32)
    out = {}
    for label, first in [("without_self", [1, 2]), ("with_self", [0, 1])]:
        seeds[0] = first
        out[label] = [ranks(index(search, data, "euclidean", 42, 2, n_iters=rounds,
                                   init_graph=seeds.copy(), tree_init=False, delta=0))
                      for rounds in (0, 1)]
    return out


def group_crossings(indices, data, a, b):
    """Compare candidate pairs to alternative single-root partitions."""
    centered = data - data.mean(axis=1, keepdims=True)
    normal = lambda points: points / np.linalg.norm(points, axis=1, keepdims=True)
    partitions = {
        "euclidean": np.sum((data - data[a]) ** 2, axis=1) < np.sum((data - data[b]) ** 2, axis=1),
        "cosine": data @ (normal(data)[a] - normal(data)[b]) > 0,
        "centered_cosine": centered @ (normal(centered)[a] - normal(centered)[b]) > 0,
    }
    return {name: int((mask[:, None] != mask[indices]).sum()) for name, mask in partitions.items()}


def projection_controls(search):
    observed = {}
    # These pivot witnesses were identified from the two connected groups
    # of the one-tree zero-round rank graph. Width 7 fills within both groups.
    for n, dimension, data_seed, seed, pivots in (
        (4096, 7, 57, 42, (612, 2584)), (4096, 11, 88, 73, (139, 1895)),
        (131, 7, 57, 42, (27, 117)), (131, 11, 57, 42, (27, 117)),
        (131, 11, 57, 73, (9, 15)),
    ):
        data = np.random.default_rng(data_seed).standard_normal((n, dimension)).astype(np.float32)
        for metric in ("euclidean", "cosine", "correlation"):
            state = RecordedRandomState(seed)
            root_indices, _ = index(search, data, metric, state, 7,
                                    n_trees=1, leaf_size=n - 1, n_iters=0).neighbor_graph
            assert (root_indices >= 0).all()
            entry = {"dimension": dimension, "seed": seed, "pivots": pivots,
                     "root_crossings": group_crossings(root_indices, data, *pivots),
                     "control_crossings": group_crossings(root_indices, data, pivots[0] + 1, pivots[1]),
                     "draws": state.draws}
            if n == 4096:
                default = index(search, data, metric, seed, n_trees=8, n_iters=0).neighbor_graph[0]
                entry["leaf_size_matches"] = {
                    cap: int((index(search, data, metric, seed, n_trees=8, n_iters=0,
                                    leaf_size=cap).neighbor_graph[0] == default).sum())
                    for cap in (10, 59, 60, 61)
                }
            name = f"{n}_{dimension}d_{seed}_{metric}"
            observed[name] = entry
            print("projection", name, flush=True)
    return observed


def seed_controls(search):
    n, width = 131, 7
    data = np.random.default_rng(57).standard_normal((n, 11)).astype(np.float32)
    # Keep the forest scalar constant while changing the two signed draws.
    seed42 = RecordedRandomState(42)
    base = index(search, data, "euclidean", seed42, width,
                 n_trees=1, leaf_size=n - 1, n_iters=0).neighbor_graph[0]
    forced = RecordedRandomState(73, forest_seed=seed42.draws[2][3])
    same_forest = index(search, data, "euclidean", forced, width,
                        n_trees=1, leaf_size=n - 1, n_iters=0).neighbor_graph[0]
    other_forest = index(search, data, "euclidean", 73, width,
                         n_trees=1, leaf_size=n - 1, n_iters=0).neighbor_graph[0]
    random42, random73 = [index(search, data, "euclidean", seed, width,
                               n_trees=0, n_iters=0).neighbor_graph[0] for seed in (42, 73)]
    return {"total": base.size, "same_forest_seed_equal": int((base == same_forest).sum()),
            "other_forest_seed_equal": int((base == other_forest).sum()),
            "no_tree_different_seeds_equal": int((random42 == random73).sum())}


def partial_fill_control(search):
    n, width, seed = 61, 50, 42
    data = np.random.default_rng(57).standard_normal((n, 11)).astype(np.float32)
    a, b = 24, 58
    left = np.sum((data - data[a]) ** 2, axis=1) < np.sum((data - data[b]) ** 2, axis=1)
    expected = ranks(index(search, data, "euclidean", seed, width,
                           n_trees=1, leaf_size=n - 1, n_iters=0))
    return {"n_rows": n, "n_features": 11, "width": width, "seed": seed,
            "training_bits": data.view(np.uint32).ravel().tolist(),
            "left": left.tolist(), "expected": expected}


def scheduling_controls(search, examples):
    observations = {}
    for name, case in examples.items():
        n, d = case["n_rows"], case["n_features"]
        if n != 4096:
            continue
        seed, metric = case.get("seed", 42), case["metric"]
        data = np.asarray(case["training_bits"], dtype=np.uint32).view(np.float32).reshape(n, d)
        zero = np.asarray(case["snapshots"]["0"]["indices"], dtype=np.int32).reshape(n, 5)
        first = np.asarray(case["snapshots"]["1"]["indices"], dtype=np.int32).reshape(n, 5)
        wider = index(search, data, metric, seed, 6, n_trees=8, leaf_size=60, n_iters=0).neighbor_graph[0]
        hidden = wider[:, 1:].copy()
        frozen = index(search, data, metric, seed, n_iters=1, init_graph=zero.copy(), tree_init=False)
        resumed = index(search, data, metric, seed, n_iters=1, init_graph=hidden.copy(), tree_init=False)
        finished = index(search, data, metric, seed, n_iters=12, init_graph=hidden.copy(), tree_init=False)
        full_rounds = index(search, data, metric, seed, n_iters=12, init_graph=hidden.copy(), tree_init=False, delta=0)
        random = [ranks(index(search, data, metric, seed, n_iters=rounds, n_trees=0)) for rounds in (0, 1)]
        observations[name] = {
            "hidden_indices": hidden.ravel().tolist(),
            "wider_prefix_matches": int((wider[:, :5] == zero).sum()),
            "normal_round": first_round_changes(zero, first, n, 5),
            "frozen_first": ranks(frozen), "hidden_first": ranks(resumed),
            "hidden_final": ranks(finished), "random_only": random,
            "delta_zero_final_column_changes": int((finished.neighbor_graph[0] != full_rounds.neighbor_graph[0]).sum()),
        }
        print("scheduling", name, flush=True)
    return observations


def holdout_cases(umap, search):
    for n in (4095, 4096):
        d, seed = 11, 73
        data = np.random.default_rng(88).standard_normal((n, d)).astype(np.float32)
        queries = np.random.default_rng(101).standard_normal((4, d)).astype(np.float32)
        for metric in ("euclidean", "cosine", "correlation"):
            params = PARAMETERS | dict(metric=metric, n_neighbors=5, random_state=seed)
            def graph(force):
                model = umap(**params, force_approximation_algorithm=force)
                return {"fit": capture(model.fit_transform(data.copy())),
                        "query": capture(model.transform(queries.copy()))}

            forced, automatic = graph(True), graph(False)
            assert (forced == automatic) == (n == 4096)
            snapshots = {}
            for count in (0, 1, 12):
                model = index(search, data, metric, seed, n_trees=8, n_iters=count)
                snapshots[str(count)] = ranks(model)
            epsilon = 0.12 if metric == "euclidean" else 0.24
            query_ranks = rank_record(*model.query(queries.copy(), k=5, epsilon=epsilon))
            name = f"gaussian_{n}_11d_73_{metric}"
            print(name, flush=True)
            yield name, {
                "metric": metric, "cutoff": None, "n_rows": n, "n_features": d, "width": 5,
                "fitted_k": 5, "seed": seed, "n_query_rows": 4, "iterations": 12,
                "training_bits": data.view(np.uint32).ravel().tolist(),
                "query_bits": queries.view(np.uint32).ravel().tolist(),
                "snapshots": snapshots, "forced": forced, "automatic": automatic,
                "restarts": {}, "query_epsilon_sweep": {str(epsilon): query_ranks},
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arch = platform.machine().lower().replace("amd64", "x86_64")
    fixture = FIXTURES / f"umap-learn-0.6.0-approximate-neighbors-{platform.system().lower()}-{arch}.json"
    reference = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    names = {name for name, spec in cases() if spec["rows"] in (4095, 4096) and not spec.get("unique")}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent
            from umap import UMAP
            projection = projection_controls(NNDescent)
            streams = seed_controls(NNDescent)
            partial = partial_fill_control(NNDescent)
            inputs = dict(sweep(UMAP, NNDescent, reference, names)) | dict(holdout_cases(UMAP, NNDescent))
            result = {"cases": inputs, "projection": projection, "seed_streams": streams,
                      "partial_fill": partial,
                      "hidden_seed_control": hidden_seed_control(NNDescent),
                      "scheduling": scheduling_controls(NNDescent, inputs)}
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
