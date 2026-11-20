"""Measure search-tree entry offers over a fixed public neighbor graph."""

import argparse
import json
import statistics
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_construction import RecordedRandomState
from probe_approximate_pivots import VacancyState
from probe_approximate_query import bits, index, np, ranks


SUBNORMAL = float(np.nextafter(0.0, 1.0))
EPSILONS = (0, SUBNORMAL, 1e-300, 0.03, 0.12, 0.24, 0.5)
WIDTHS = (1, 5, 6, 8, 13)
SEARCH_OPTIONS = dict(pruning_degree_multiplier=100, diversify_prob=0,
                      degree_prune_aggressiveness=0, n_jobs=1)


def trained(search, x, seed, trees, leaf, rounds, search_trees):
    return search(x.copy(), metric="euclidean", n_neighbors=5, random_state=seed,
                  n_iters=rounds, delta=0, max_candidates=60, n_trees=trees,
                  leaf_size=leaf, n_search_trees=search_trees, **SEARCH_OPTIONS)


def sweep(index, q, widths=WIDTHS, epsilons=EPSILONS):
    return [dict(width=k, epsilon=e, **ranks(index.query(q, k=k, epsilon=e)))
            for k in widths for e in epsilons]


def timing(index, q):
    samples = []
    index.query(q, k=5, epsilon=0.12)
    for _ in range(41):
        start = perf_counter_ns()
        index.query(q, k=5, epsilon=0.12)
        samples.append(perf_counter_ns() - start)
    return int(statistics.median(samples))


def shared_case(search, name, x, q, seed, trees, leaf, rounds, native_seed=None,
                required_graph=None, widths=WIDTHS):
    state = RecordedRandomState(seed)
    enabled = trained(search, x, state, trees, leaf, rounds, 1)
    graph, _ = enabled.neighbor_graph
    if required_graph is not None:
        assert np.array_equal(graph, required_graph)
    disabled = index(search, x, graph, seed)
    assert np.array_equal(disabled.neighbor_graph[0], graph)
    enabled.prepare()
    assert np.array_equal(enabled.neighbor_graph[0], graph)
    test = sweep(enabled, q, widths)
    no_tree = sweep(disabled, q, widths)
    settings = {}
    if trees > 1:
        for search_count in (0, 2, trees):
            choice = trained(search, x, seed, trees, leaf, rounds, search_count)
            assert np.array_equal(choice.neighbor_graph[0], graph)
            choice.prepare()
            candidate = sweep(choice, q, widths)
            settings[search_count] = sum(left == right for left, right in zip(candidate, test))
    print(name, flush=True)
    return dict(name=name, n=len(x), d=x.shape[1], cap=graph.shape[1], seed=seed,
                trees=trees, leaf=leaf, native_seed=seed if native_seed is None else native_seed,
                train_bits=bits(x), query_bits=bits(q), graph=graph.ravel().tolist(),
                draws=state.draws, prepared=test, no_tree=no_tree, settings=settings,
                prepared_ns=timing(enabled, q), no_tree_ns=timing(disabled, q))


def matching_pivots_seed(n, a, b):
    for seed in range(1 << 20):
        raw = np.random.RandomState(seed).randint(0, 1 << 32, size=2, dtype=np.uint32)
        left = (int(raw[0]) * n) >> 32
        right = (int(raw[1]) * (n - 1)) >> 32
        right += right >= left
        if {left, right} == {a, b}:
            return seed
    raise AssertionError


def root_probe(search):
    n, d, seed = 131, 7, 42
    x = np.random.default_rng(57).standard_normal((n, d)).astype(np.float32)
    queries = np.random.default_rng(309).standard_normal((256, d)).astype(np.float32)
    repeated = np.concatenate((queries, queries))
    sq = ((queries[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    training_sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(sq, axis=1)
    anchors = {}
    for trees in (1, 8):
        prepared = trained(search, x, seed, trees, n - 1, 0, 1)
        prepared.prepare()
        found = prepared.query(repeated, k=1, epsilon=0)[0].ravel()
        assert np.array_equal(found[:len(queries)], found[len(queries):])
        if trees == 1:
            initial = found
        counts = []
        for a in range(n):
            for b in range(a):
                in_leaf = training_sq[:, a] < training_sq[:, b]
                source_side = sq[:, a] < sq[:, b]
                rank = (in_leaf[order] == source_side[:, None]).argmax(axis=1)
                matched = (order[np.arange(len(queries)), rank] == found[:len(queries)]).sum()
                counts.append((int(matched), b, a))
        counts.sort(reverse=True)
        best, a, b = counts[0]
        assert best == len(queries) and counts[1][0] < best
        anchors[trees] = dict(pivots=[a, b], matches=best, runner_up=counts[1][0])

    other = np.random.RandomState(73)
    for row in (1, 2):
        words = other.randint(-2147483647, 2147483646, 3, dtype=np.int64)
        forced = trained(search, x, VacancyState(seed, row, words), 1, n - 1, 0, 1)
        forced.prepare()
        assert np.array_equal(forced.query(repeated, k=1, epsilon=0)[0].ravel(), initial)
    return x, anchors


def root_control(search, x, anchors):
    n, d = x.shape
    # Re-create the recovered query split using the independent native RNG.
    # The initial training split on this input uses rows 27 and 117.
    a, b = anchors[1]["pivots"]
    midpoint = np.float32(0.5) * (x[a] + x[b])
    direction = x[a] - x[b]
    q = np.asarray([midpoint + np.float32(scale) * direction for scale in (-1, -0.01, 0.01, 1)], np.float32)
    holdout = np.random.default_rng(1701).standard_normal((5, d)).astype(np.float32)
    q = np.vstack((q, holdout))
    result = shared_case(search, "one_split", x, q, 42, 1, n - 1, 0,
                         matching_pivots_seed(n, a, b), widths=WIDTHS + (n,))
    tiny32 = float(np.nextafter(np.float32(0), np.float32(1)))
    tree = trained(search, x, 42, 1, n - 1, 0, 1)
    tree.prepare()
    result["underflow"] = sweep(tree, q, (n,), (0, tiny32 * 0.25, tiny32 * 0.75, tiny32, 1e-37))
    return result


def cutoff_control(search):
    data = np.arange(25, dtype=np.float32).reshape(-1, 1) * 1000
    data[[6, 8, 9], 0] = [10, 11, 0]
    graph = np.full((25, 5), -1, np.int32)
    graph[:, 0] = np.arange(25)
    graph[6, 1], graph[8, 1], graph[12, 1] = 8, 9, 15
    weights = np.where(graph >= 0, np.float32(100), np.inf)
    weights[:, 0] = 0
    weights[12, 1] = 16
    ref = index(search, data, graph, 42, weights)
    q = np.zeros((1, 1), np.float32)
    # Squared distances are 100 and 121. The 16 edge floor puts the
    # bridge exactly at the cutoff when epsilon is 21 / (100 - 16).
    crossing = 0.25
    below = float(np.nextafter(np.float32(121), np.float32(0)))
    above = float(np.nextafter(np.float32(121), np.float32(np.inf)))
    close = ((np.nextafter(121.0, 0.0) - 100) / 84, crossing,
             (np.nextafter(121.0, np.inf) - 100) / 84,
             ((121 + below) / 2 - 100) / 84, ((121 + above) / 2 - 100) / 84,
             (above - 100) / 84)
    return dict(name="queue_cutoff", n=25, d=1, cap=5, seed=42,
                train_bits=bits(data), query_bits=bits(q), graph=graph.ravel().tolist(),
                edge_bits=bits(weights), no_tree=sweep(ref, q, (1,), (*close, 0.24, 0.26)))


def underfilled_controls(search):
    q = np.zeros((1, 1), np.float32)
    data = (np.arange(25, dtype=np.float32) * 3 + 100).reshape(-1, 1)
    data[[6, 18, 24, 23], 0] = [10, 34, 36, 32]
    base = [i for i in range(25) if i not in (6, 18, 24, 23)]
    cases = [(f"degree_{n}", base[:n]) for n in range(1, 15)]
    cases += [("seen_inside", sorted(base[10:20] + [18])), ("seen_tail", base[:11] + [24])]
    for name, targets in cases:
        graph = np.full((25, 5), -1, np.int32)
        graph[:, 0] = np.arange(25)
        graph[targets, 1] = 6
        weights = np.where(graph >= 0, np.float32(1), np.inf)
        weights[:, 0] = 0
        ref = index(search, data, graph, 42, weights)
        yield dict(name=name, n=25, d=1, cap=5, seed=42,
                   train_bits=bits(data), query_bits=bits(q), graph=graph.ravel().tolist(),
                   edge_bits=bits(weights), no_tree=sweep(ref, q, (5, 6, 7, 8, 13, 22),
                                                          (0, SUBNORMAL, 1e-300)))


def boundary(search, nearest_tree):
    for n, d, seed, data_seed, query_seed in (
        (4095, 7, 42, 57, 62), (4096, 7, 42, 57, 62),
        (4095, 11, 19, 117, 103), (4096, 11, 73, 88, 309),
    ):
        x = np.random.default_rng(data_seed).standard_normal((n, d)).astype(np.float32)
        q = np.random.default_rng(query_seed).standard_normal((7, d)).astype(np.float32)
        q[4] = q[0]
        q[5:] = x[[6, n - 1]]
        exact = nearest_tree(x).query(x, k=5, workers=1)[1].astype(np.int32)
        name = f"{n}_{d}d_seed{seed}"
        yield shared_case(search, name + "_one_leaf", x, q, seed, 1, n, 0, required_graph=exact)
        yield shared_case(search, name + "_forest", x, q, seed, 8, 60, 12)
        if d == 7:
            yield shared_case(search, name + "_exact_graph_forest", x, q, seed, 16, 512, 12,
                              required_graph=exact)


def supplied_graph_control(search):
    x = (np.arange(25, dtype=np.float32) ** 2).reshape(-1, 1)
    q = np.array([[82.5], [85], [-50], [460]], np.float32)
    graph = np.full((len(x), 5), -1, np.int32)
    graph[:, 0] = np.arange(len(x))
    plain = index(search, x, graph, 42)
    state = RecordedRandomState(42)
    flagged = search(x.copy(), n_neighbors=5, init_graph=graph.copy(), tree_init=True,
                     n_trees=8, n_search_trees=8, n_iters=0, random_state=state, **SEARCH_OPTIONS)
    flagged.prepare()
    assert len(state.draws) == 2
    assert sweep(plain, q) == sweep(flagged, q)
    return state.draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.spatial import cKDTree
            from pynndescent import NNDescent

            root_data, root_anchors = root_probe(NNDescent)
            observations = dict(
                versions={name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                supplied_graph_draws=supplied_graph_control(NNDescent),
                root_anchors=root_anchors,
                cases=[cutoff_control(NNDescent), *underfilled_controls(NNDescent),
                       root_control(NNDescent, root_data, root_anchors), *boundary(NNDescent, cKDTree)],
            )
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(observations, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
