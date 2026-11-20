"""Recover unsigned projection words through public graph and query witnesses."""

import argparse
import json
import math
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_construction import RecordedRandomState, index, np
from probe_approximate_pivots import components, pivot_witness
from probe_approximate_query import bits, index as graph_only, ranks


SCALARS = (0, 1, 42, 1819927849474927636, 2033886794655244609)
SIZES = (31, 32, 33, 47, 48, 49, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         95, 96, 97, 127, 128, 129, 130, 131, 132, 133, 134, 135, 161, 162,
         163, 164, 165, 191, 192, 193, 255, 256, 257, 258, 259, 260, 511, 512, 513)
LIMIT = 1 << 64


def observed_index(search, x, scalar, depth, capacity, width=7):
    return index(search, x, "euclidean", RecordedRandomState(42, forest_seed=scalar), width,
                 n_trees=1, n_iters=0, max_rptree_depth=depth, leaf_size=capacity,
                 n_search_trees=1, pruning_degree_multiplier=100, diversify_prob=0,
                 degree_prune_aggressiveness=0)


def constrain(observations):
    values = {0}
    modulus = 1
    for size, digits in observations:
        divisor = math.gcd(size, modulus)
        step = size // divisor
        inverse = pow(modulus // divisor, -1, step)
        values = {a + modulus * ((b - a) // divisor * inverse % step)
                  for a in values for b in digits if (b - a) % divisor == 0}
        values = {value for value in values if value < LIMIT}
        modulus *= step
        if not values:
            break
    return values, modulus


def recover_words(observations):
    first, span = constrain(observations)
    pairs = []
    for word in first:
        second, second_span = constrain([
            (size - 1, [index - (index > word % size) for index in candidates if index != word % size])
            for size, candidates in observations
        ])
        if span >= LIMIT and second_span >= LIMIT:
            pairs.extend([word, other] for other in second)
    return pairs


def pivot_positions(words, n):
    a = words[0] % n
    compact = words[1] % (n - 1)
    return a, compact + (compact >= a)


def partition(x, rows, a, b):
    rows = list(rows)
    values = x[rows]
    first = ((values - x[a]) ** 2).sum(axis=1) <= ((values - x[b]) ** 2).sum(axis=1)
    cursor, end = 0, len(rows)
    while cursor < end:
        if first[cursor]:
            cursor += 1
        else:
            end -= 1
            rows[cursor], rows[end] = rows[end], rows[cursor]
            first[cursor], first[end] = first[end], first[cursor]
    return rows[:end], rows[end:]


def calibrate(search):
    data = np.random.default_rng(122).standard_normal((max(SIZES), 11)).astype(np.float32)
    streams = {}
    for scalar in SCALARS:
        roots = []
        for n in SIZES:
            x = data[:n]
            graph = observed_index(search, x, scalar, 1, 5).neighbor_graph[0]
            candidate = pivot_witness(x, graph)["compatible_pairs"]
            if len(candidate) == 1:
                roots.append((n, candidate[0]))
        root, = recover_words(roots)
        children = [[], []]
        stable = [[], []]
        for n in range(109, 251, 4):
            for dimension, data_seed in ((7, 57), (11, 113)):
                x = np.random.default_rng(data_seed).standard_normal((n, dimension)).astype(np.float32)
                graph = observed_index(search, x, scalar, 2, 4).neighbor_graph[0]
                a, b = pivot_positions(root, n)
                groups = partition(x, range(n), a, b)
                labels = np.zeros(n, bool)
                labels[groups[0]] = True
                if (graph < 0).any() or (labels[:, None] != labels[graph]).any():
                    continue
                for side, order in enumerate(groups):
                    rows = sorted(order)
                    if not np.isin(graph[rows], rows).all():
                        continue
                    child = pivot_witness(x[rows], np.searchsorted(rows, graph[rows]))["compatible_pairs"]
                    if len(child) == 1:
                        children[side].append((len(rows), [order.index(rows[k]) for k in child[0]]))
                        stable[side].append((len(rows), child[0]))
        assert not any(recover_words(group) for group in stable)
        left, = recover_words(children[0])
        right, = recover_words(children[1])
        streams[scalar] = dict(words=[root, left, right], root_witnesses=len(roots),
                               child_witnesses=list(map(len, children)))
        print("calibrated scalar", scalar, flush=True)
    return streams


def model(x, rows, words, capacity, depth, slot=0):
    if depth == 0 or len(rows) <= capacity:
        return [], [rows]
    a, b = (rows[index] for index in pivot_positions(words[slot], len(rows)))
    left, right = partition(x, rows, a, b)
    lp, ll = model(x, left, words, capacity, depth - 1, 2 * slot + 1)
    rp, rl = model(x, right, words, capacity, depth - 1, 2 * slot + 2)
    return [[int(a), int(b)], *lp, *rp], ll + rl


def component_labels(graph):
    _, labels = components(graph)
    return [int(np.flatnonzero(labels == label)[0]) for label in labels]


def training_holdouts(search, streams):
    for n, dimension, scalar, data_seed in (
        (33, 11, 1, 122), (135, 7, 0, 1611), (133, 11, SCALARS[3], 911),
        (131, 7, SCALARS[3], 57), (131, 7, SCALARS[3], 32), (163, 9, SCALARS[3], 1921),
        (197, 13, SCALARS[4], 532), (257, 5, 42, 1978),
        (4095, 7, SCALARS[3], 57), (4096, 7, SCALARS[3], 57),
    ):
        x = np.random.default_rng(data_seed).standard_normal((n, dimension)).astype(np.float32)
        words = streams[scalar]["words"]
        a, b = pivot_positions(words[0], n)
        left, right = map(len, partition(x, range(n), a, b))
        smaller = min(left, right)
        caps = (smaller - 1, smaller, smaller + 1) if n >= 50 else ()
        caps += (n - 1, n, n + 1) if n < 4095 else (n - 1,)
        print("training holdout", n, dimension, data_seed, [left, right], flush=True)
        for capacity in sorted(set(caps)):
            depth = 2
            pivots, leaves = model(x, list(range(n)), words, capacity, depth)
            expected = np.zeros(n, np.int32)
            for rows in leaves:
                expected[rows] = min(rows)
            for width in ((7, 4) if min(map(len, leaves)) < 7 else (7,)):
                graph = observed_index(search, x, scalar, depth, capacity, width).neighbor_graph[0]
                labels = component_labels(graph)
                shifted = None
                if left <= capacity < right:
                    _, alternate = model(x, list(range(n)), [words[0], words[1], words[1]], capacity, depth)
                    altered = np.zeros(n, np.int32)
                    for rows in alternate:
                        altered[rows] = min(rows)
                    shifted = int((altered == labels).sum())
                name = f"{n}_{dimension}d_{data_seed}_{scalar}_{capacity}_k{width}"
                cross_leaf = ((expected[:, None] != expected[graph]) & (graph >= 0)).sum()
                yield dict(name=name, n=n, d=dimension, cap=capacity, depth=depth, pivots=pivots,
                           rows_by_leaf=leaves, labels=labels, words=words, train_bits=bits(x),
                           matching_component_labels=int((expected == labels).sum()),
                           shifted_right_matches=shifted, cross_leaf_offers=int(cross_leaf))


def query_pairs(x, q, winners):
    training = ((x[:, None] - x[None, :]) ** 2).sum(axis=2)
    distances = ((q[:, None] - x[None, :]) ** 2).sum(axis=2)
    order = distances.argsort(axis=1)
    before = (order == winners[:, None]).cumsum(axis=1) == 0
    earlier = order[before]
    query_rows = np.repeat(np.arange(len(q)), before.sum(axis=1))
    pairs = []
    for a in range(len(x)):
        leaves = training[:, a, None] < training[:, a + 1:]
        chosen = distances[:, a, None] < distances[:, a + 1:]
        candidates = np.flatnonzero((leaves[winners] == chosen).all(axis=0))
        excluded = (leaves[earlier[:, None], candidates] == chosen[query_rows[:, None], candidates]).any(axis=0)
        pairs.extend([a, int(b) + a + 1] for b in candidates[~excluded])
    return pairs


def recover_prepared(x, q, winners):
    square = ((x[:, None] - x[None, :]) ** 2).sum(axis=2)
    distances = ((q[:, None] - x[None, :]) ** 2).sum(axis=2)
    for a in range(len(x)):
        it = (square[winners, a, None] < square[winners, a + 1:]) == (distances[:, a, None] < distances[:, a + 1:])
        for b in np.flatnonzero(it.all(axis=0)) + a + 1:
            branches = []
            for first in (True, False):
                rows = np.flatnonzero((square[:, a] < square[:, b]) == first)
                active = (distances[:, a] < distances[:, b]) == first
                target = np.searchsorted(rows, winners[active])
                if np.array_equal(distances[active][:, rows].argmin(axis=1), target):
                    branches.append([None])
                else:
                    branches.append([[int(rows[i]), int(rows[j])]
                                     for i, j in query_pairs(x[rows], q[active], target)])
            for left in branches[0]:
                for right in branches[1]:
                    yield [int(a), int(b)], left, right


def encode_prepared(x, geometry):
    root, left, right = geometry
    a, b = root
    rows = partition(x, range(len(x)), a, b)
    words = [[a, b - (b > a)], [0, 0], [0, 0]]
    stopped = [len(group) for group, pivots in zip(rows, (left, right)) if pivots is None]
    cap = max(stopped, default=1)
    for slot, (group, pivots) in enumerate(zip(rows, (left, right)), start=1):
        if pivots is not None:
            assert len(group) > cap
            first, second = (group.index(pivot) for pivot in pivots)
            words[slot] = [first, second - (second > first)]
    return words, cap


def query_sweep(index, q):
    tiny64 = float(np.nextafter(0.0, 1.0))
    tiny32 = float(np.nextafter(np.float32(0), np.float32(1)))
    return [dict(width=k, epsilon=e, **ranks(index.query(q, k=k, epsilon=e)))
            for k in (1, 5, 8, 13) for e in (0, tiny64, tiny32, 0.12, 0.24)]


def timing(index, q):
    index.query(q, k=5, epsilon=0.12)
    samples = []
    for _ in range(41):
        start = perf_counter_ns()
        index.query(q, k=5, epsilon=0.12)
        samples.append(perf_counter_ns() - start)
    return int(median(samples))


def prepared_holdouts(search, streams):
    for n, dimension, scalar, data_seed, capacity in (
        (131, 7, SCALARS[3], 57, 130), (131, 7, SCALARS[3], 57, 43),
        (163, 7, SCALARS[3], 57, 54), (131, 11, SCALARS[4], 57, 43),
        (131, 7, SCALARS[4], 57, 130), (131, 11, SCALARS[4], 57, 130),
        (197, 11, 1, 111, 65),
    ):
        x = np.random.default_rng(data_seed).standard_normal((n, dimension)).astype(np.float32)
        q = np.random.default_rng(637).standard_normal((640, dimension)).astype(np.float32)
        enabled = observed_index(search, x, scalar, 2, capacity, 5)
        ids = enabled.neighbor_graph[0]
        plain = graph_only(search, x, ids, 42)
        assert np.array_equal(np.sort(plain.neighbor_graph[0], axis=1), np.sort(ids, axis=1))
        enabled.prepare()
        observed = enabled.query(q, k=1, epsilon=0)[0].ravel()
        geometry, = recover_prepared(x, q, observed)
        if scalar == SCALARS[3] and capacity == n - 1:
            assert geometry == ([30, 127], None, None)
        words, query_cap = encode_prepared(x, geometry)
        independent = np.random.default_rng(2003).standard_normal((9, dimension)).astype(np.float32)
        independent[-2:] = x[[0, -1]]
        name = f"{n}_{dimension}d_{data_seed}_{capacity}"
        print("prepared", name, geometry, flush=True)
        yield dict(name=name, n=n, d=dimension, cap=query_cap, training_cap=capacity, depth=2,
                   graph=ids.ravel().tolist(), train_bits=bits(x), query_bits=bits(independent),
                   words=words, training_words=streams[scalar]["words"], geometry=geometry,
                   prepared=query_sweep(enabled, independent), plain=query_sweep(plain, independent),
                   wheel_ns=timing(enabled, independent), wheel_plain_ns=timing(plain, independent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent

            streams = calibrate(NNDescent)
            output = dict(versions={name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                          streams=streams, training=list(training_holdouts(NNDescent, streams)),
                          queries=list(prepared_holdouts(NNDescent, streams)))
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
