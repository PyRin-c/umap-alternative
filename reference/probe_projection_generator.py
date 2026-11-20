"""Compare scalar-to-word hypotheses with held-out public projection witnesses."""

import argparse
import json
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_projection_words import (
    RecordedRandomState, bits, component_labels, encode_prepared, graph_only,
    index, model, np, observed_index, partition, pivot_positions, pivot_witness,
    query_sweep, recover_prepared, recover_words, timing,
)


SCALAR_MAX = (1 << 62) - 1
DEFAULT = 1819927849474927636
OTHER = 2033886794655244609
SCALARS = (0, 1, 2, 3, 4, 41, 42, 43, (1 << 31) - 1, 1 << 31,
           (1 << 32) - 1, 1 << 32, (1 << 32) + 1, SCALAR_MAX - 1, SCALAR_MAX,
           DEFAULT - 1, DEFAULT, DEFAULT + 1, OTHER)
SIZES = (65, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 128,
         129, 130, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
         197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 255, 256, 257, 263,
         271, 283, 293, 307, 311, 317, 331, 509, 511, 512, 513, 521)
DEEP_SCALARS = (0, 1, 2, (1 << 32) - 1, 1 << 32, SCALAR_MAX, DEFAULT, OTHER)
MASK = (1 << 64) - 1


def temper(value):
    value ^= value >> 11
    value ^= (value << 7) & 0x9D2C5680
    value ^= (value << 15) & 0xEFC60000
    return value ^ (value >> 18)


def untemper(value):
    original = value
    for mask, shift in ((0, -18), (0xEFC60000, 15), (0x9D2C5680, 7), (0, -11)):
        source = value
        for _ in range(32 // abs(shift)):
            value = source ^ (value >> -shift if shift < 0 else (value << shift) & mask)
    assert temper(value) == original
    return value


def caller_draws(search):
    x = np.random.default_rng(501).standard_normal((37, 7)).astype(np.float32)
    for seed in (0, 1, 2, 42, (1 << 32) - 2, (1 << 32) - 1):
        state = RecordedRandomState(seed)
        index(search, x, "euclidean", state, 4, n_trees=1, n_iters=0)
        yield dict(seed=seed, draws=state.draws)
    raw = [0xFFFFFFFF, 0xFFFFFFFD, 13, 0xFFFFFFFC, 0xFFFFFFFE, 1, 2,
           0xFFFFFFFF, 3, 4, 0xFFFFFFFF, 0xFFFFFFFF, 15]
    words = [untemper(v) for v in raw] + [0] * (624 - len(raw))
    state = RecordedRandomState(0)
    state.set_state(("MT19937", np.array(words, dtype=np.uint32), 0, 0, 0.0))
    index(search, x, "euclidean", state, 4, n_trees=1, n_iters=0)
    yield dict(state=words, draws=state.draws)


def candidate_words(seed):
    for bitgen in (np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64):
        yield bitgen.__name__, bitgen(seed).random_raw(2).tolist()
    for order in (False, True):
        raw = np.random.MT19937(seed).random_raw(4).tolist()
        result = [(b << 32) | a if order else (a << 32) | b for a, b in zip(raw[::2], raw[1::2])]
        yield f"MT19937_high_first_{not order}", result
    generator = random.Random(seed)
    yield "python_random", [generator.getrandbits(64) for _ in range(2)]
    state = seed
    result = []
    for _ in range(2):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        result.append(z ^ (z >> 31))
    yield "SplitMix64", result


def root_streams(search, archive):
    x = np.random.default_rng(2771).standard_normal((max(SIZES), 9)).astype(np.float32)
    streams, scores = {}, {}
    for scalar in SCALARS:
        observations = []
        for n in SIZES:
            fit = observed_index(search, x[:n], scalar, 1, 7, 8)
            compatible = pivot_witness(x[:n], fit.neighbor_graph[0])["compatible_pairs"]
            if len(compatible) == 1:
                observations.append((n, compatible[0]))
        root, = recover_words(observations)
        control = archive.get(str(scalar))
        if control:
            assert root == control["words"][0]
        streams[scalar] = [root]
        # Use the public observations both to compare raw words and to score
        # actual bounded positions across all available widths.
        cohort = "archive" if control else "heldout"
        for name, generated in candidate_words(scalar):
            counts = scores.setdefault(cohort, {}).setdefault(name, [0, 0, 0, 0])
            counts[0] += sum(a == b for a, b in zip(root, generated))
            counts[1] += 2
            counts[2] += sum(sorted(pivot_positions(generated, n)) == pair for n, pair in observations)
            counts[3] += len(observations)
        print("root", scalar, len(observations), flush=True)
    return streams, scores


def extend_words(search, scalar, prefix, depth):
    width = 1 << (depth - 1)
    witnesses = [[] for _ in range(width)]
    sizes = range(109, 257, 4) if depth == 2 else range(193, 721, 6)
    for n in sizes:
        for dimension, seed in ((7, 577), (11, 113), (13, 971)):
            x = np.random.default_rng(seed).standard_normal((n, dimension)).astype(np.float32)
            _, groups = model(x, list(range(n)), prefix, 1, depth - 1)
            graph = observed_index(search, x, scalar, depth, 5).neighbor_graph[0]
            order = np.zeros(n, np.int32)
            for k, group in enumerate(groups):
                order[group] = k
            if (graph < 0).any() or (order[:, None] != order[graph]).any():
                continue
            for k, group in enumerate(groups):
                rows = sorted(group)
                witness = pivot_witness(x[rows], np.searchsorted(rows, graph[rows]))["compatible_pairs"]
                if len(witness) == 1:
                    witnesses[k].append((len(rows), [group.index(rows[i]) for i in witness[0]]))
    result = [recover_words(observations) for observations in witnesses]
    assert all(len(word) == 1 for word in result)
    return [options[0] for options in result], [len(group) for group in witnesses]


def depth_streams(search, streams, archive):
    counts = {}
    for scalar in DEEP_SCALARS:
        words = streams[scalar]
        children, two = extend_words(search, scalar, words, 2)
        words.extend(children)
        control = archive.get(str(scalar))
        if control:
            assert words == control["words"]
        grandchildren, three = extend_words(search, scalar, words, 3)
        words.extend(grandchildren)
        counts[scalar] = {"depth2": two, "depth3": three}
        print("depth", scalar, two, three, flush=True)
    return counts


def active_slots(x, rows, words, cap, depth, slot=0):
    if len(rows) <= cap or depth == 0:
        return []
    a, b = (rows[j] for j in pivot_positions(words[slot], len(rows)))
    left, right = partition(x, rows, a, b)
    return [slot, *active_slots(x, left, words, cap, depth - 1, 2 * slot + 1),
            *active_slots(x, right, words, cap, depth - 1, 2 * slot + 2)]


def leaf_labels(n, groups):
    result = np.zeros(n, np.int32)
    for rows in groups:
        result[rows] = min(rows)
    return result


def packed_tree(x, rows, words, cap, depth):
    if len(rows) <= cap or depth == 0:
        return [rows]
    a, b = (rows[j] for j in pivot_positions(next(words), len(rows)))
    left, right = partition(x, rows, a, b)
    return packed_tree(x, left, words, cap, depth - 1) + packed_tree(x, right, words, cap, depth - 1)


def heldout_leaves(search, streams):
    for n, dimension, seed, scalar in (
        (515, 7, 263, 0), (607, 13, 839, 1), (767, 9, 313, 2), (768, 7, 421, SCALAR_MAX),
        (769, 11, 411, (1 << 32) - 1), (1023, 5, 1441, 1 << 32),
        (1024, 11, 263, DEFAULT), (1025, 9, 127, OTHER),
        (4095, 7, 57, DEFAULT), (4096, 7, 57, DEFAULT),
    ):
        words = streams[scalar]
        x = np.random.default_rng(seed).standard_normal((n, dimension)).astype(np.float32)
        _, inner = model(x, list(range(n)), words, 1, 2)
        cap = min(map(len, inner))
        for capacity in sorted({max(1, cap - 1), cap, cap + 1}):
            pivots, leaves = model(x, list(range(n)), words, capacity, 3)
            graph = observed_index(search, x, scalar, 3, capacity, 4).neighbor_graph[0]
            expected = leaf_labels(n, leaves)
            wheel = component_labels(graph)
            packed = packed_tree(x, list(range(n)), iter(words), capacity, 3)
            different = ((graph >= 0) & (expected[:, None] != expected[graph])).sum()
            tag = f"{n}_{dimension}d_{seed}_{scalar}_{capacity}"
            yield dict(name=tag, n=n, d=dimension, cap=capacity, depth=3,
                       pivots=pivots, rows_by_leaf=leaves, labels=wheel, words=words,
                       train_bits=bits(x), matching_component_labels=int((expected == wheel).sum()),
                       cross_leaf_offers=int(different), shifted_right_matches=None,
                       active_slots=active_slots(x, list(range(n)), words, capacity, 3),
                       packed_labels=int((leaf_labels(n, packed) == wheel).sum()))


def fixed_graph_controls(search, streams):
    x = np.random.default_rng(57).standard_normal((131, 7)).astype(np.float32)
    q = np.random.default_rng(637).standard_normal((640, 7)).astype(np.float32)
    base = observed_index(search, x, DEFAULT, 2, 130, 5)
    original = base.neighbor_graph[0]
    base.prepare()
    nearest = base.query(q, k=1, epsilon=0)[0].ravel()
    a, b = pivot_positions(streams[DEFAULT][0], len(x))
    keep = [i for i in range(len(x)) if i not in (a, b)]
    p = np.arange(len(x))
    p[keep] = np.random.default_rng(566).permutation(keep)
    changed = []
    for name, data, cap, width in (("permutation", x[p], 130, 5), ("capacity", x, 80, 5),
                                   ("rank width", x, 130, 7)):
        other = observed_index(search, data, DEFAULT, 2, cap, width)
        g = other.neighbor_graph[0]
        other.prepare()
        ids = other.query(q, k=1, epsilon=0)[0].ravel()
        geometry = list(recover_prepared(data, q, ids))
        if name == "permutation":
            g = p[g][np.argsort(p)]
            ids = p[ids]
            geometry = [[[int(p[i]) for i in node] if node is not None else None
                         for node in tree] for tree in geometry]
        same = (int((np.sort(g, axis=1) == np.sort(original, axis=1)).sum())
                if g.shape == original.shape else None)
        if name in ("permutation", "capacity"):
            assert same == original.size
        changed.append(dict(name=name, changed_ranks=int((ids != nearest).sum()),
                            equal_graph_slots=same, geometry=geometry))

    collisions = []
    for n in range(37, 132):
        seen = {}
        for scalar, words in streams.items():
            pair = pivot_positions(words[0], n)
            if pair not in seen:
                seen[pair] = scalar
                continue
            other = seen[pair]
            left = observed_index(search, x[:n], other, 1, n - 1, 4)
            right = observed_index(search, x[:n], scalar, 1, n - 1, 4)
            g1, d1 = left.neighbor_graph
            g2, d2 = right.neighbor_graph
            if not np.array_equal(g1, g2) or not np.array_equal(d1.view(np.uint32), d2.view(np.uint32)):
                continue
            left.prepare()
            right.prepare()
            before, bd = left.query(q, k=1, epsilon=0)
            after, ad = right.query(q, k=1, epsilon=0)
            collisions.append(dict(n=n, pair=pair, scalars=[other, scalar], graph_slots=g1.size,
                                   rank_matches=int((before == after).sum()),
                                   bit_matches=int((bd.view(np.uint32) == ad.view(np.uint32)).sum())))
            if len(collisions) == 3:
                return dict(fixed_graph=changed, collisions=collisions)
    assert collisions
    return dict(fixed_graph=changed, collisions=collisions)


def prepared_holdouts(search, streams):
    for n, dimension, scalar, data_seed, capacity in (
        (131, 7, DEFAULT, 57, 80), (131, 7, SCALAR_MAX, 57, 130),
        (163, 9, 2, 207, 162), (197, 11, 1 << 32, 111, 65),
    ):
        x = np.random.default_rng(data_seed).standard_normal((n, dimension)).astype(np.float32)
        q = np.random.default_rng(637).standard_normal((640, dimension)).astype(np.float32)
        fitted = observed_index(search, x, scalar, 2, capacity, 5)
        links = fitted.neighbor_graph[0]
        plain = graph_only(search, x, links, 42)
        fitted.prepare()
        rank = fitted.query(q, k=1, epsilon=0)[0].ravel()
        geometry, = recover_prepared(x, q, rank)
        query_words, cap = encode_prepared(x, geometry)
        probes = np.random.default_rng(1477).standard_normal((9, dimension)).astype(np.float32)
        probes[-2:] = x[[0, -1]]
        name = f"{n}_{dimension}d_{scalar}_{capacity}"
        print("prepared holdout", name, geometry, flush=True)
        yield dict(name=name, n=n, d=dimension, cap=cap, depth=2, training_cap=capacity, graph=links.ravel().tolist(),
                   train_bits=bits(x), query_bits=bits(probes), words=query_words, training_words=streams[scalar],
                   geometry=geometry, prepared=query_sweep(fitted, probes), plain=query_sweep(plain, probes),
                   wheel_ns=timing(fitted, probes), wheel_plain_ns=timing(plain, probes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent

            archive = json.loads(args.archive.read_text(encoding="utf8"))["streams"]
            callers = list(caller_draws(NNDescent))
            streams, scores = root_streams(NNDescent, archive)
            counts = depth_streams(NNDescent, streams, archive)
            controls = fixed_graph_controls(NNDescent, streams)
            output = dict(versions={name: version(name) for name in ("umap-learn", "pynndescent", "numba")},
                          caller_draws=callers, streams=streams, generators=scores, depth_witnesses=counts,
                          prepared_controls=controls, training=list(heldout_leaves(NNDescent, streams)),
                          queries=list(prepared_holdouts(NNDescent, streams)))
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
