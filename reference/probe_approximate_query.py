"""Compare held-out queries over a fixed topology using public wheel calls."""

import argparse
import json
import statistics
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_approximate_neighbors_fixture import PARAMETERS, capture, np
from probe_approximate_pivots import VacancyState


def index(search, data, graph, seed, weights=None):
    result = search(data.copy(), metric="euclidean", n_neighbors=graph.shape[1],
                    n_jobs=1, random_state=seed, n_iters=0, tree_init=False,
                    n_search_trees=0, init_graph=graph.copy(),
                    init_dist=None if weights is None else weights.copy(),
                    pruning_degree_multiplier=100, diversify_prob=0,
                    degree_prune_aggressiveness=0)
    observed, _ = result.neighbor_graph
    for row, (expected, actual) in enumerate(zip(graph.tolist(), observed.tolist())):
        if set(expected) - {-1, row} != set(actual) - {-1, row}:
            print("training topology", row, expected, actual, flush=True)
            raise AssertionError
    result.prepare()
    return result


def bits(values):
    return values.astype(np.float32, copy=False).view(np.uint32).ravel().tolist()


def ranks(result):
    indices, distances = result
    return {"indices": np.where(np.isfinite(distances), indices, -1).ravel().tolist(),
            "bits": bits(distances)}


def example(search, name, data, graph, queries, seed, widths, epsilons, weights=None):
    result = index(search, data, graph, seed, weights)
    return dict(name=name, n=len(data), d=data.shape[1], seed=seed, cap=graph.shape[1],
                train_bits=bits(data), query_bits=bits(queries), graph=graph.ravel().tolist(),
                edge_bits=None if weights is None else bits(weights),
                results=[dict(width=k, epsilon=epsilon, **ranks(result.query(queries.copy(), k=k, epsilon=epsilon)))
                         for k in widths for epsilon in epsilons])


def controls(search):
    empty = np.full((25, 3), -1, np.int32)
    data = (np.arange(25, dtype=np.float32) ** 2).reshape(-1, 1)
    queries = np.array([[82.5], [85], [82.5], [83], [82.5]], np.float32)
    for seed in (0, 42, 73, 2**32 - 1):
        yield example(search, f"entry_{seed}", data, empty, queries, seed, (1, 2, 3, 5), (0, 0.12))
    tied_data = np.ones_like(data)
    tied_data[::2] = -1
    yield example(search, "equal_keys", tied_data, empty, np.zeros_like(queries), 42,
                  (1, 2, 3, 4, 5), (0, 0.12))

    for reverse in (False, True):
        graph = empty.copy()
        if reverse:
            graph[8, 0], graph[9, 0] = 6, 8
        else:
            graph[6, 0], graph[8, 0] = 8, 9
        data = np.arange(25, dtype=np.float32).reshape(-1, 1) * 1000
        data[[6, 8, 9], 0] = [10, 10.5, 0]
        weights = np.where(graph >= 0, np.float32(1), np.inf).astype(np.float32)
        yield example(search, f"bridge_reverse_{reverse}", data, graph, np.zeros((1, 1), np.float32),
                      42, (1,), (0, 0.1, 0.2), weights)

    graph = empty.copy()
    graph[6, 0], graph[8, 0] = 8, 9
    data[[6, 8, 9], 0] = [10, 5, 0]
    for a, b in ((1, 1), (25, 25), (100, 1), (1, 64), (64, 25), (100, 100)):
        weights = np.full(graph.shape, np.inf, np.float32)
        weights[6, 0], weights[8, 0] = a, b
        for query in (0, -0.001):
            yield example(search, f"edge_{a}_{b}_q{query}", data, graph, np.array([[query]], np.float32),
                          42, (1,), (0.12, 10), weights)

    graph[12, 0] = 15  # Unreachable from the entry point or its component.
    data[8, 0] = 10.5
    for floor in (0, 25, 50, 80, 99, 100):
        weights = np.full(graph.shape, np.inf, np.float32)
        weights[6, 0] = weights[8, 0] = 100
        weights[12, 0] = floor
        yield example(search, f"disconnected_floor_{floor}", data, graph, np.zeros((1, 1), np.float32),
                      42, (1, 2), (0, 0.12, 0.15, 0.2, 0.21, 0.5, 0.6, 10, 11), weights)

    data[[2, 4, 6], 0] = [-2, 2, 10]
    for order in ((2, 4), (4, 2)):
        graph = empty.copy()
        graph[6, :2] = order
        weights = np.where(graph >= 0, np.float32(1), np.inf).astype(np.float32)
        yield example(search, f"neighbor_tie_{order[0]}", data, graph, np.zeros((1, 1), np.float32),
                      42, (1,), (0.12,), weights)

    data[[6, 7, 8, 2, 4, 9, 11], 0] = [10, 3, 4, 5, -5, 1, -1]
    for order in ((4, 2), (2, 4)):
        graph = empty.copy()
        graph[6, :2] = [7, 8]
        graph[7, 0], graph[8, 0] = order
        graph[4, 0], graph[2, 0] = 9, 11
        weights = np.where(graph >= 0, np.float32(0.001), np.inf).astype(np.float32)
        yield example(search, f"frontier_tie_{order[0]}", data, graph, np.zeros((1, 1), np.float32),
                      42, (1,), (3,), weights)


def caller_seed_control(search):
    query = np.zeros((1, 1), np.float32)
    data = np.arange(25, dtype=np.float32).reshape(-1, 1)
    graph = np.full((25, 3), -1, np.int32)
    def first_row(seed):
        return ranks(index(search, data, graph, seed).query(query, k=3, epsilon=0))

    seeded = np.random.RandomState(73)
    seeded.randint(-2147483647, 2147483646, 3, dtype=np.int64)
    replacement = seeded.randint(-2147483647, 2147483646, 3, dtype=np.int64)
    original, other = first_row(42), first_row(73)
    first = first_row(VacancyState(42, 1, replacement))
    second = first_row(VacancyState(42, 2, replacement))
    assert original == first and other == second and first != second
    return dict(seed_42=original, seed_73=other, first_replaced=first, second_replaced=second)


def fit_observations(umap, search, data, queries, graph, seed):
    class RecordedQuery(search):
        def query(self, query_data, k=10, epsilon=0.1):
            result = super().query(query_data, k=k, epsilon=epsilon)
            self.observed_query = dict(width=k, epsilon=epsilon, **ranks(result))
            return result

    shared = index(RecordedQuery, data, graph, seed)
    training_ids, training_distances = shared.neighbor_graph
    model = umap(**(PARAMETERS | dict(random_state=seed, n_neighbors=graph.shape[1])),
                 force_approximation_algorithm=True,
                 precomputed_knn=(training_ids, training_distances, shared))
    fitted = capture(model.fit_transform(data.copy()))
    exact = capture(model.transform(data.copy()))
    assert not hasattr(shared, "observed_query")
    held_out = capture(model.transform(queries.copy()))
    assert hasattr(shared, "observed_query")
    result = dict(exact_input_equals_fit=exact == fitted, fit=fitted,
                  training=ranks((training_ids, training_distances)),
                  query=held_out, ranks=shared.observed_query)
    reordered = model.transform(data[::-1].copy())
    return result | {"reordered_format": reordered.format, "reordered_nnz": int(reordered.nnz)}


def boundary(search, umap, nearest_tree):
    for n, d, seed, query_seed, flat in (
        (4095, 7, 42, 62, False), (4096, 7, 42, 62, False),
        (4095, 11, 0, 91, False), (4096, 11, 73, 91, False),
        (4096, 7, 42, 62, True),
    ):
        data = np.random.default_rng(57).standard_normal((n, d)).astype(np.float32)
        queries = np.random.default_rng(query_seed).standard_normal((7, d)).astype(np.float32)
        queries[4] = queries[0]
        queries[5:] = data[[6, n - 1]]
        nearest = nearest_tree(data)
        graph = nearest.query(data, k=5, workers=1)[1].astype(np.int32)
        weights = np.where(graph == np.arange(n)[:, None], 0, 1e-8).astype(np.float32) if flat else None
        query_index = index(search, data, graph, seed, weights)
        results = [dict(width=k, epsilon=epsilon,
                        **ranks(query_index.query(queries.copy(), k=k, epsilon=epsilon)))
                   for k in (1, 2, 3, 5, 8, 13) for epsilon in (0, 0.03, 0.12, 0.24, 0.5, 1)]
        full = ranks(query_index.query(queries.copy(), k=5, epsilon=0.12))
        assert full == ranks(query_index.query(queries.copy(), k=5, epsilon=0.12))
        singleton = ranks(query_index.query(queries[:1].copy(), k=5, epsilon=0.12))
        reversed_batch = ranks(query_index.query(queries[::-1].copy(), k=5, epsilon=0.12))
        wide_tiny = ranks(query_index.query(queries[:1].copy(), k=13, epsilon=1e-300))
        samples = []
        for _ in range(31):
            start = perf_counter_ns()
            query_index.query(queries, k=5, epsilon=0.12)
            samples.append(perf_counter_ns() - start)
        name = f"{n}_{d}d_seed{seed}" + ("_flat" if flat else "")
        print(name, flush=True)
        yield dict(name=name, n=n, d=d, seed=seed, cap=graph.shape[1], train_bits=bits(data),
                   query_bits=bits(queries), graph=graph.ravel().tolist(), results=results,
                   edge_bits=None if weights is None else bits(weights),
                   exact_nearest=nearest.query(queries, k=1, workers=1)[1].tolist(),
                   singleton=singleton, reversed_batch=reversed_batch, wide_tiny=wide_tiny,
                   wheel_median_ns=int(statistics.median(samples)),
                   estimator=None if flat else fit_observations(umap, search, data, queries, graph, seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.spatial import cKDTree
            from pynndescent import NNDescent
            from umap import UMAP

            observations = {"versions": {name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                            "caller_seed": caller_seed_control(NNDescent),
                            "controls": list(controls(NNDescent)),
                            "cases": list(boundary(NNDescent, UMAP, cKDTree))}
    except Exception as error:
        # Wheel JIT errors may contain implementation excerpts.
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(observations, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
