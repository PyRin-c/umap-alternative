"""Verify training projection words and tied leaf membership through public zero-round fits."""

import argparse
import itertools
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_pivots import components
from probe_approximate_query import bits, np
from probe_prepared_projection_ties import ControlledState, index
from probe_projection_generator import SIZES
from probe_projection_words import pivot_positions, pivot_witness, recover_words


MASK = (1 << 64) - 1
GOLDEN_RATIO = 0x9E3779B97F4A7C15
PIVOT_SALT = 0x5851F42D4C957F2D
MAX_SCALAR = (1 << 62) - 1


def splitmix64(key):
    key = (key + GOLDEN_RATIO) & MASK
    key = ((key ^ (key >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    key = ((key ^ (key >> 27)) * 0x94D049BB133111EB) & MASK
    return key ^ (key >> 31)


def tree_key(scalar, ordinal=0):
    return splitmix64(scalar ^ splitmix64(ordinal + 1))


def child(key, side):
    return splitmix64(((key << 1) | side) & MASK)


def pivot_words(key):
    return [splitmix64(key), splitmix64(key ^ PIVOT_SALT)]


def split_order(rows, sides):
    rows = list(rows)
    cursor, end = 0, len(rows)
    while cursor < end:
        if sides[rows[cursor]] == 0:
            cursor += 1
        else:
            end -= 1
            rows[cursor], rows[end] = rows[end], rows[cursor]
    return rows[:end], rows[end:]


def fit(search, x, caller, scalar, depth, cap, width, replacement=None, trees=1):
    state = ControlledState(caller, scalar, replacement)
    trained = index(search, x, "euclidean", state, width, n_trees=trees, n_iters=0,
                    leaf_size=cap, max_rptree_depth=depth, n_search_trees=1,
                    pruning_degree_multiplier=100, diversify_prob=0, degree_prune_aggressiveness=0)
    return trained.neighbor_graph, state.draws


def levels(scalar, amount):
    keys = [tree_key(scalar)]
    for i in range((amount - 1) // 2):
        keys.extend(child(keys[i], side) for side in (0, 1))
    return [pivot_words(key) for key in keys]


def word_controls(search, archive):
    expected = {int(key): words for key, words in archive["streams"].items()}
    for scalar, words in expected.items():
        assert levels(scalar, len(words)) == words
        yield dict(scalar=scalar, words=words, witnesses=None)
    features = np.random.default_rng(145926).standard_normal((max(SIZES), 9)).astype(np.float32)
    for scalar in (7, 8191, 0x123456789ABCDEF, MAX_SCALAR - 2):
        witnesses = []
        for n in SIZES:
            x = features[:n]
            (neighbors, _), _ = fit(search, x, 19, scalar, 1, 5, 8)
            pairs = pivot_witness(x, neighbors)["compatible_pairs"]
            if len(pairs) == 1:
                witnesses.append((n, pairs[0]))
        words, = recover_words(witnesses)
        assert words == pivot_words(tree_key(scalar))
        print("fresh scalar", scalar, "root witnesses", len(witnesses), flush=True)
        yield dict(scalar=scalar, words=[words], witnesses=len(witnesses))


def component_order(labels, leaves):
    sides = np.full(len(labels), -1, np.int32)
    for i, leaf in enumerate(leaves):
        ids = np.asarray(leaf)
        expected = labels[ids]
        representative = np.bincount(expected).argmax()
        sides[labels == representative] = i
    return sides


def model_rows(x, scalar, depth, cap, ordinal=0):
    leaves = []

    def split(rows, key, remaining):
        if len(rows) <= cap or remaining == 0:
            leaves.append(rows)
            return
        a, b = (rows[i] for i in pivot_positions(pivot_words(key), len(rows)))
        margins = signed_margins(x, a, b)
        sides = np.where(margins > 0, 0, 1)
        for row in rows:
            if abs(margins[row]) <= np.float32(1e-8):
                sides[row] = splitmix64(key ^ row) & 1
        groups = split_order(rows, sides)
        if not all(groups):
            leaves.append(list(rows))
            return
        for side, group in enumerate(groups):
            split(group, child(key, side), remaining - 1)

    split(list(range(len(x))), tree_key(scalar, ordinal), depth)
    return leaves


def signed_margins(x, a, b):
    normal = x[a] - x[b]
    midpoint = (x[a] + x[b]) * np.float32(0.5)
    offset = projection_dot(-midpoint[None, :], normal, np.float32(0))[0]
    return projection_dot(x, normal, offset)


def projection_dot(x, normal, offset):
    def add(column, seed):
        return (x[:, column].astype(np.float64) * float(normal[column]) + seed).astype(np.float32)

    def accumulate(width, start, end, seed):
        lanes = np.zeros((width, len(x)), np.float32)
        lanes[0] = seed
        for offset in range(start, end, width):
            for lane in range(width):
                lanes[lane] = add(offset + lane, lanes[lane])
        return lanes

    wide_end = x.shape[1] // 32 * 32
    if wide_end:
        wide = accumulate(32, 0, wide_end, offset)
        four = (wide[:8] + wide[8:16]) + (wide[16:24] + wide[24:32])
        four = four[:4] + four[4:]
        offset = (four[0] + four[2]) + (four[1] + four[3])
    block_end = x.shape[1] // 4 * 4
    four = accumulate(4, wide_end, block_end, offset)
    margin = (four[0] + four[2]) + (four[1] + four[3])
    for column in range(block_end, x.shape[1]):
        margin = add(column, margin)
    return margin


def record(search, name, x, caller, scalar, depth, cap, width, replacement=None):
    leaves = model_rows(x, scalar, depth, cap)
    (neighbors, _), draws = fit(search, x, caller, scalar, depth, cap, width, replacement)
    count, connected = components(neighbors)
    sides = component_order(connected, leaves)
    if depth == 1 and count == 2:
        a, b = pivot_positions(pivot_words(tree_key(scalar)), len(x))
        if connected[a] != connected[b]:
            sides = (connected != connected[a]).astype(np.int32)
    expected = np.zeros(len(x), np.int32)
    for i, leaf in enumerate(leaves):
        expected[leaf] = i
    matched = int(np.count_nonzero(expected == sides))
    print(name, "components", count, "rows", matched, "/", len(x), flush=True)
    assert count == len(leaves) and matched == len(x)
    return dict(name=name, n=len(x), d=x.shape[1], caller=caller, scalar=scalar, depth=depth,
                cap=cap, fitted_components=count, rows=matched, sides=sides.tolist(),
                train_bits=bits(x), leaves=leaves, draws=draws, replacement=replacement)


def center_data(n, d, scalar, data_seed, scale):
    x = np.random.default_rng(data_seed).standard_normal((n, d)).astype(np.float32) * np.float32(.2)
    x[1::3] = x[::3][:len(x[1::3])]
    a, b = pivot_positions(pivot_words(tree_key(scalar)), n)
    x[:, 0] = 0
    x[a] = 0
    x[b] = 0
    x[a, 0], x[b, 0] = -scale, scale
    return x, a, b


def root_controls(search):
    for scalar, data_seed in ((0, 19843), (1, 1765), (MAX_SCALAR, 26917)):
        for n in (131, 197, 4095, 4096):
            x, a, b = center_data(n, 3 if n >= 4095 else 9, scalar, data_seed, 1.)
            width = 15 if n >= 4095 else 11
            sizes = (42, 0, 1, 2**32 - 2, 2**32 - 1) if n < 200 else (42,)
            for caller in sizes:
                label = f"root_{scalar}_{n}_{caller}"
                yield record(search, label, x, caller, scalar, 1, 7, width)
            if n == 131:
                for draw, words in itertools.product((1, 2), ((0, 0, 0), (12345, -23456, 34567))):
                    label = f"root_{scalar}_{n}_signed_{draw}_{words[0]}"
                    yield record(search, label, x, 42, scalar, 1, 7, width, (draw, words))
                move = [i for i in range(n) if i not in (a, b)]
                permuted = x.copy()
                permuted[move] = x[move[::-1]]
                yield record(search, f"root_{scalar}_{n}_permuted", permuted, 42, scalar, 1, 7, width)
            if n == 197:
                cutoff = np.float32(0.5e-8)
                subnormal = np.nextafter(np.float32(0), np.float32(1))
                choices = (subnormal, np.nextafter(cutoff, np.float32(0)), cutoff,
                           np.nextafter(cutoff, np.float32(np.inf)))
                use = [i for i in range(n) if i % 2 == 0 and i not in (a, b)]
                for sign, value in itertools.product((-1, 1), choices):
                    variant = x.copy()
                    shift = np.float32(sign) * value
                    variant[use, 0] = shift
                    label = f"margin_{scalar}_{int(shift.view(np.uint32)):08x}"
                    yield record(search, label, variant, 42, scalar, 1, 7, width)


def nested_data(n, d, scalar, data_seed, depth):
    for attempt in range(64):
        rng = np.random.default_rng(data_seed + attempt)
        features = np.zeros((n, d), np.float32)
        features[:, depth:] = rng.standard_normal((n, d - depth)).astype(np.float32) * .2
        features[1::3] = features[::3][:len(features[1::3])]
        used = set()

        def configure(rows, key, level):
            if level == depth:
                return len(rows) > 21
            a, b = (rows[i] for i in pivot_positions(pivot_words(key), len(rows)))
            if {a, b} & used or not np.array_equal(features[a, :level], features[b, :level]):
                return False
            used.update((a, b))
            features[a, depth:] = 0
            features[b, depth:] = 0
            features[a, level], features[b, level] = -1., 1.
            sides = np.zeros(n, np.intp)
            for row in rows:
                if row in (a, b):
                    sides[row] = int(row == b)
                elif depth < 4 and row % 11 == level and row not in used:
                    sides[row] = int(rng.integers(2))
                    features[row, level] = np.nextafter(np.float32(-.5e-8 if sides[row] == 0 else .5e-8),
                                                        np.float32(-np.inf if sides[row] == 0 else np.inf))
                else:
                    sides[row] = splitmix64(key ^ row) & 1
            groups = split_order(rows, sides)
            return all(len(group) > 21 and configure(group, child(key, side), level + 1)
                       for side, group in enumerate(groups))

        if configure(list(range(n)), tree_key(scalar), 0):
            return features
    raise AssertionError


def nested_controls(search):
    for n, dimension, scalar, depth, caller, data_seed in (
        (295, 5, 7, 2, 0, 555), (373, 7, 8191, 2, 73, 15),
        (557, 9, 0x123456789ABCDEF, 3, 2**32 - 1, 447),
        (811, 13, MAX_SCALAR - 2, 3, 1, 4657),
        (1273, 7, 21, 4, 2**32 - 2, 492),
    ):
        x = nested_data(n, dimension, scalar, data_seed, depth)
        for capacity in (10, n // 4):
            tag = f"nested_{n}_{dimension}_{scalar}_{capacity}"
            yield record(search, tag, x, caller, scalar, depth, capacity, 13)


def reparented_controls(search):
    n, scalar = 353, 50429
    base, a, b = center_data(n, 7, scalar, 5581, 16.)
    key = tree_key(scalar)
    initial = np.random.default_rng(2193).permutation(n) % 2
    initial[a], initial[b] = 0, 1
    compare = []
    for variant in (0, 1):
        root_sides = initial.copy()
        if variant:
            swap = [np.flatnonzero((initial == side) & (np.arange(n) % 3 == 1)) for side in (0, 1)]
            swap = [[i for i in branch if i not in (a, b)] for branch in swap]
            amount = min(map(len, swap))
            for branch in swap:
                root_sides[branch[:amount]] ^= 1
        x = base.copy()
        x[:, 0] = np.where(root_sides == 0, -4., 4.)
        x[:, 1] = 0
        x[a, 0], x[b, 0] = -16., 16.
        groups = split_order(range(n), root_sides)
        pivots = []
        for side, group in enumerate(groups):
            pair = [group[i] for i in pivot_positions(pivot_words(child(key, side)), len(group))]
            assert not set(pair) & {a, b}
            pivots.append(pair)
            x[pair, 1:] = 0
            x[pair, 1] = (-1., 1.)
        control = record(search, f"reparented_{variant}", x, 7, scalar, 2, 7, 13)
        observed = control["sides"]
        choices = [{row: observed[row] % 2 for row in group if row not in pair}
                   for group, pair in zip(groups, pivots)]
        compare.append((groups, choices))
        yield control
    (groups_a, choices_a), (groups_b, choices_b) = compare
    absolute = positional = absolute_total = positional_total = 0
    for first, second, a, b in zip(groups_a, groups_b, choices_a, choices_b):
        common = a.keys() & b.keys()
        absolute += sum(a[i] == b[i] for i in common)
        absolute_total += len(common)
        pairs = [(a[i], b[j]) for i, j in zip(first, second) if i in a and j in b]
        positional += sum(left == right for left, right in pairs)
        positional_total += len(pairs)
    assert absolute == absolute_total and positional < positional_total
    print("child tie choice by original row", absolute, "/", absolute_total,
          "by incoming position", positional, "/", positional_total, flush=True)


def angled_controls(search):
    for dimension, n, scalar, data_seed in (
        (7, 197, 8231, 542), (31, 229, MAX_SCALAR - 3, 8214),
        (32, 263, 841, 14726), (33, 249, (1 << 32) + 6, 935), (65, 241, 6403, 9217),
        (4, 181, 3447, 832), (17, 231, 67, 173), (64, 271, 131233, 288),
        (97, 297, 987654, 1571),
    ):
        rng = np.random.default_rng(data_seed)
        x = rng.standard_normal((n, dimension)).astype(np.float32)
        a, b = pivot_positions(pivot_words(tree_key(scalar)), n)
        left, right = x[[a, b]]
        normal = (left - right).astype(np.float64)
        midpoint = (left.astype(np.float64) + right.astype(np.float64)) * .5
        x -= x.mean(axis=0, keepdims=True)
        aligned = x.astype(np.float64) * .2
        projection = aligned - (aligned @ normal / (normal @ normal))[:, None] * normal + midpoint
        projection[::7] += normal * 0.015625
        projection[1::7] -= normal * 0.015625
        x = projection.astype(np.float32)
        axis = np.argmax(np.abs(normal))
        for i in range(1, n, 3):
            x[i, axis] = np.nextafter(x[i, axis], np.float32(np.inf if i % 2 else -np.inf))
        x[a], x[b] = left, right
        yield record(search, f"angled_{dimension}_{scalar}", x, 42, scalar, 1, 7, 13)


def coincident_pivots(search):
    for n, d, scalar in ((173, 7, 4), (241, 11, MAX_SCALAR)):
        x = np.random.default_rng(193 + n).standard_normal((n, d)).astype(np.float32)
        a, b = pivot_positions(pivot_words(tree_key(scalar)), n)
        x[b] = x[a]
        yield record(search, f"coincident_pivots_{scalar}", x, 1, scalar, 1, 7, 13)


def forests(search):
    for scalar, n, d, seed, depth, width in ((6, 163, 7, 1151, 1, 7),
                                              (917, 257, 11, 892, 2, 9),
                                              (MAX_SCALAR, 779, 9, 773, 3, 11)):
        x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
        cap = 8
        for trees in (1, 2, 3, 5):
            allowed = np.zeros((n, n), bool)
            for tree in range(trees):
                for group in model_rows(x, scalar, depth, cap, tree):
                    allowed[np.ix_(group, group)] = True
            (indices, distances), _ = fit(search, x, 73, scalar, depth, cap, width, trees=trees)
            square = np.sum((x[:, None] - x[None, :]) ** 2, axis=2)
            closest = np.argsort(np.where(allowed, square, np.inf), axis=1)[:, :width]
            enough = allowed.sum(axis=1) >= width + 1
            matches = int(np.count_nonzero(closest[enough] == indices[enough]))
            slots = int(enough.sum()) * width
            print("forest", scalar, trees, matches, "/", slots, flush=True)
            assert matches == slots
            yield dict(scalar=scalar, n=n, d=d, cap=cap, width=width, trees=trees, depth=depth,
                       matches=matches, slots=slots, train_bits=bits(x), indices=indices.ravel().tolist(),
                       distance_bits=bits(distances), enough=enough.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent

            archive = json.loads(args.archive.read_bytes())
            output = dict(versions={name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                          words=list(word_controls(NNDescent, archive)),
                          cases=[*root_controls(NNDescent), *nested_controls(NNDescent),
                                 *reparented_controls(NNDescent),
                                 *angled_controls(NNDescent), *coincident_pivots(NNDescent)],
                          forests=list(forests(NNDescent)))
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
