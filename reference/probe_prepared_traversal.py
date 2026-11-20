"""Probe prepared traversal using zero-round public NNDescent neighbor graphs."""

import argparse
import itertools
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_construction import RecordedRandomState, index, np
from probe_approximate_query import bits, index as graph_only
from probe_prepared_approximate_query import sweep, timing
from probe_projection_words import SCALARS, observed_index, partition, recover_prepared


EPSILONS = (0.0, float(np.nextafter(0.0, 1.0)),
            float(np.nextafter(np.float32(0), np.float32(1))), 0.12, 0.24)


def build(search, x, caller, scalar, width, training_cap, search_cap, depth):
    return index(search, x, "euclidean", RecordedRandomState(caller, forest_seed=scalar), width,
                 n_trees=1, n_iters=0, max_rptree_depth=2, leaf_size=training_cap,
                 n_search_trees=1, search_tree_leaf_size=search_cap, max_search_tree_depth=depth,
                 pruning_degree_multiplier=100, diversify_prob=0, degree_prune_aggressiveness=0)


def encode(x, geometry, capacity, depth):
    words = [None] * 3
    leaves = []

    def visit(rows, remaining, slot, pair):
        if not remaining or len(rows) <= capacity:
            assert pair is None
            leaves.append(sorted(rows))
            return
        a, b = (rows.index(pivot) for pivot in pair)
        words[slot] = [a, b - (b > a)]
        left, right = partition(x, rows, *pair)
        if slot == 0:
            visit(left, remaining - 1, 1, geometry[1])
            visit(right, remaining - 1, 2, geometry[2])
        else:
            assert remaining == 1
            leaves.extend((sorted(left), sorted(right)))

    visit(list(range(len(x))), depth, 0, geometry[0])
    return [word or [0, 0] for word in words], leaves


def encode_order(x, geometry, flips, capacity, depth):
    directed = [pair[::-1] if pair is not None and flip else pair
                for pair, flip in zip(geometry, flips)]
    if flips[0]:
        directed[1], directed[2] = directed[2], directed[1]
    return encode(x, directed, capacity, depth)


def orientation(x, index, live, geometry, capacity, depth, query_seed):
    n, d = x.shape
    q = np.random.default_rng(query_seed).standard_normal((96, d)).astype(np.float32)
    roots = index.query(q, k=1, epsilon=0.0)[0].ravel()
    wide, distances = index.query(q, k=n, epsilon=0.0)
    wide = np.where(np.isfinite(distances), wide, -1)
    edges = adjacency(live)
    options = []
    for flips in itertools.product(*[(False, True) if p else (False,) for p in geometry]):
        words, leaves = encode_order(x, geometry, flips, capacity, depth)
        leaf_by_id = {}
        for leaf in leaves:
            leaf_by_id.update(dict.fromkeys(leaf, set(leaf)))
        position = {row: i for i, row in enumerate(itertools.chain.from_iterable(leaves))}
        exact = 0
        for root, actual in zip(roots, wide):
            leaf = leaf_by_id[root]
            offers = sorted(edges[root] - leaf, key=position.__getitem__)
            prefix = len(offers) // 4 * 4
            retained = {j for i, j in enumerate(offers) if i >= prefix or i % 4 < 2}
            exact += retained == set(actual) - leaf - {-1}
        options.append((exact, words))
    options.sort(key=lambda item: -item[0])
    return options[0][1], [exact for exact, _ in options]


def adjacency(graph):
    edges = [set() for _ in graph]
    for row, neighbors in enumerate(graph):
        for neighbor in set(neighbors) - {row, -1}:
            edges[row].add(int(neighbor))
            edges[neighbor].add(row)
    return edges


def equal_frontiers(x, live, geometry, capacity, depth):
    _, leaves = encode(x, geometry, capacity, depth)
    edges = adjacency(live)
    queries = []
    choices = []
    pairs = list(itertools.combinations(range(len(x)), 2))
    order = np.random.default_rng(78169).permutation(len(pairs))
    for a, b in itertools.chain([(0, 8)], (pairs[i] for i in order)):
        q = (x[a] + x[b]) * np.float32(0.5)
        square = ((x - q) ** 2).sum(axis=1)
        root, left, right = geometry
        if square[root[0]] == square[root[1]]:
            continue
        side = left if square[root[0]] < square[root[1]] else right
        if side is not None and square[side[0]] == square[side[1]]:
            continue
        branch = next(leaf for leaf in leaves if a in leaf)
        if b not in branch or square[a] != square[b] or square[a] != square[branch].min():
            continue
        if np.count_nonzero(square[branch] == square[a]) != 2:
            continue
        root_side = partition(x, range(len(x)), *root)[int(square[root[0]] > square[root[1]])]
        if a not in root_side:
            continue
        if side is not None and a not in partition(x, root_side, *side)[int(square[side[0]] > square[side[1]])]:
            continue
        if edges[a] - set(branch) == edges[b] - set(branch):
            continue
        choices.append([a, b, float(square[a])])
        queries.append(q)
        if len(queries) == 12:
            break
    assert len(queries) >= 6
    return np.asarray(queries), choices


def record(search, x, q, name, caller, scalar, training_cap, search_cap, depth, geometry=None,
           archived=False, frontier=False):
    n, d = x.shape

    def fit(width):
        if archived:
            return observed_index(search, x, scalar, depth, training_cap, width)
        return build(search, x, caller, scalar, width, training_cap, search_cap, depth)

    prepared = fit(5)
    graph, distances = prepared.neighbor_graph
    wide_fit = fit(6)
    live, live_distances = wide_fit.neighbor_graph
    assert np.array_equal(live[:, :5], graph)
    assert np.array_equal(live_distances[:, :5].view(np.uint32), distances.view(np.uint32))
    prepared.prepare()
    assert np.array_equal(prepared.neighbor_graph[0], graph)
    plain = graph_only(search, x, graph, caller)
    if geometry is None:
        witnesses = np.random.default_rng(872363).standard_normal((768, d)).astype(np.float32)
        winners = prepared.query(witnesses, k=1, epsilon=0.0)[0].ravel()
        if depth == 0:
            geometry = [None] * 3
        else:
            geometry, = recover_prepared(x, witnesses, winners)
    words, scores = orientation(x, prepared, live, geometry, search_cap, depth, 123434)
    choices = []
    if frontier:
        q, choices = equal_frontiers(x, live, geometry, search_cap, depth)
    ordinary = sweep(prepared, q, (1, 3, 5, 8, 13), EPSILONS)
    broad = sweep(prepared, q, (n, n + 17), EPSILONS)
    checks = sweep(plain, q, (1, 3, 5, 8, 13), (0.0, EPSILONS[1], 0.12, 0.24))
    print(name, "orientation", scores, flush=True)
    return dict(name=name, n=n, d=d, caller=caller, cap=search_cap, depth=depth,
                geometry=geometry, words=words, orientation_scores=scores, choices=choices,
                train_bits=bits(x), query_bits=bits(q), graph=graph.ravel().tolist(),
                live=live.ravel().tolist(), prepared=ordinary, wide=broad, plain=checks,
                wheel_ns=timing(prepared, q), wheel_plain_ns=timing(plain, q))


def archives(search, paths):
    original_scalars = (SCALARS[3], SCALARS[3], SCALARS[3], SCALARS[4], SCALARS[4], SCALARS[4], 1)
    for path in paths:
        archive = json.loads(path.read_bytes())
        for position, prior in enumerate(archive["queries"]):
            d = prior["d"]
            x = np.array(prior["train_bits"], np.uint32).view(np.float32).reshape(-1, d)
            q = np.array(prior["query_bits"], np.uint32).view(np.float32).reshape(-1, d)
            cap, depth = prior["cap"], prior["depth"]
            scalar = (int(prior["name"].split("_")[-2]) if "generators" in archive
                      else original_scalars[position])
            result = record(search, x, q, prior["name"], 42, scalar, prior["training_cap"],
                            cap, depth, prior["geometry"], archived=True)
            assert result["graph"] == prior["graph"]
            observed = {(run["width"], run["epsilon"]): run for run in result["prepared"]}
            assert all(run == observed[run["width"], run["epsilon"]] for run in prior["prepared"])
            yield result


def holdouts(search):
    for n, d, data_seed, scalar, caller, capacity, depth in (
        (113, 5, 1671, 0, 73, 113, 0),
        (137, 7, 2617, 1, 0, 60, 1),
        (179, 9, 735, 4294967296, 73, 41, 2),
        (191, 11, 2618, 4611686018427387903, 42, 45, 2),
    ):
        rng = np.random.default_rng(data_seed)
        x = rng.standard_normal((n, d)).astype(np.float32)
        q = rng.standard_normal((19, d)).astype(np.float32)
        q[-3] = q[0]
        q[-2:] = x[[0, -1]]
        name = f"holdout_{n}_{d}d_{data_seed}_{scalar}_{depth}"
        yield record(search, x, q, name, caller, scalar, n, capacity, depth)
    for n, d, caller, scalar, data_seed, capacity, depth in (
        (137, 7, 0, 1, 2617, 60, 1),
        (137, 9, 73, 4294967296, 735, 60, 1),
        (179, 9, 73, 4294967296, 735, 41, 2),
    ):
        data = np.random.default_rng(data_seed).standard_normal((n, d))
        x = (np.rint(data * 64) / 64).astype(np.float32)
        yield record(search, x, None, f"equal_frontier_{caller}_{depth}", caller, scalar,
                     n, capacity, depth, frontier=True)


def heap_controls(search):
    for n, caller, scalar in ((25, 0, 0), (31, 73, 4294967296)):
        x = np.ones((n, 1), np.float32)
        x[::2] = -1
        q = np.array([[0.0], [-1.0], [1.0], [0.0]], np.float32)
        fitted = build(search, x, caller, scalar, 5, n, n, 0)
        fitted.prepare()
        yield dict(name=f"equal_heap_{n}_{caller}", n=n, d=1, caller=caller, cap=n, depth=0,
                   words=[], train_bits=bits(x), query_bits=bits(q),
                   graph=[-1] * (5 * n), live=[-1] * (6 * n), orientation_scores=[],
                   prepared=sweep(fitted, q, (1, 2, 3, 5, 8, n, n + 2), EPSILONS),
                   wide=[], plain=[], wheel_ns=timing(fitted, q), wheel_plain_ns=0)


def projection_ties(search):
    x = np.ones((25, 1), np.float32)
    x[::2] = -1
    q = np.array([[0.0], [-1.0], [1.0], [0.0]], np.float32)
    for caller in (0, 42, 73):
        prepared = build(search, x, caller, None, 5, 25, 12, 1)
        prepared.prepare()
        yield dict(caller=caller, ranks=prepared.query(q, k=1, epsilon=0.0)[0].ravel().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent

            report = dict(versions={name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                          cases=[*holdouts(NNDescent), *heap_controls(NNDescent),
                                 *archives(NNDescent, args.archive)],
                          projection_ties=list(projection_ties(NNDescent)))
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
