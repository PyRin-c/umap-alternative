"""Observe prepared projection ties using public NNDescent queries and fixed geometry."""

import argparse
import itertools
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_construction import RecordedRandomState, index
from probe_approximate_query import bits, np, ranks
from probe_prepared_traversal import encode
from probe_projection_words import partition, pivot_positions, query_pairs


EPSILONS = (0.0, float(np.nextafter(0.0, 1.0)),
            float(np.nextafter(np.float32(0), np.float32(1))), 0.12)
FIRST = "equal_frontier_0_1"
SECOND = "equal_frontier_73_1"
DEEP = "equal_frontier_73_2"
FLOAT_CASES = (
    ("holdout_179_9d_735_4294967296_2", 73, 1 << 32),
    ("holdout_191_11d_2618_4611686018427387903_2", 42, (1 << 62) - 1),
)


class ControlledState(RecordedRandomState):
    def __init__(self, caller, scalar, replaced_draw=None):
        super().__init__(caller, forest_seed=scalar)
        self.replaced_draw = replaced_draw

    def randint(self, low, high=None, size=None, dtype=int):
        value = super().randint(low, high, size, dtype)
        if self.replaced_draw is not None and (len(self.draws), size) == (self.replaced_draw[0], 3):
            value = np.asarray(self.replaced_draw[1], dtype=dtype)
            self.draws[-1][-1] = value.tolist()
        return value


def fit(search, x, state, capacity, depth, width):
    return index(search, x, "euclidean", state, width, n_trees=1, n_iters=0,
                 max_rptree_depth=0, leaf_size=len(x), n_search_trees=1,
                 search_tree_leaf_size=capacity, max_search_tree_depth=depth,
                 search_tree_leaf_budget=1, pruning_degree_multiplier=100,
                 diversify_prob=0, degree_prune_aggressiveness=0)


def oriented_splits(x, words, capacity, depth):
    splits = []

    def visit(rows, slot, steps):
        if steps == 0 or len(rows) <= capacity:
            return
        a, b = (rows[i] for i in pivot_positions(words[slot], len(rows)))
        splits.append([a, b])
        for side, group in enumerate(partition(x, rows, a, b)):
            visit(group, 2 * slot + 1 + side, steps - 1)

    visit(list(range(len(x))), 0, depth)
    return splits


def exact_intersection(x, pairs):
    # The archived quantized data is integer-valued at scale 64. Solve
    # the bisector equations at twice that scale using an integer adjugate.
    x = (x * 64).astype(np.int64)
    a, b = (x[np.asarray(pairs)[:, side]] for side in (0, 1))
    normal = b - a
    target = (b * b - a * a).sum(axis=1)
    size, d = normal.shape
    rng = np.random.default_rng(556847)
    matrices = []
    for columns in itertools.combinations(range(d), size):
        matrix = normal[:, columns]
        determinant = round(np.linalg.det(matrix))
        if determinant:
            matrices.append((abs(determinant), determinant, columns, matrix))
    for _, determinant, columns, matrix in sorted(matrices)[:4]:
        adjugate = np.rint(np.linalg.inv(matrix) * determinant).astype(np.int64)
        assert np.array_equal(matrix @ adjugate, determinant * np.eye(size, dtype=np.int64))
        other = [i for i in range(d) if i not in columns]
        for scale in (1, 2, 4, 8):
            free = rng.integers(-140 * scale, 141 * scale, size=(180000, len(other)),
                                dtype=np.int64)
            numerator = (scale * target - free @ normal[:, other].T) @ adjugate.T
            good = np.flatnonzero(np.all(numerator % determinant == 0, axis=1))
            if len(good):
                points = np.zeros((len(good), d), dtype=np.int64)
                points[:, other] = free[good]
                points[:, columns] = numerator[good] // determinant
                assert np.all(points @ normal.T == scale * target)
                order = np.argsort(np.max(np.abs(points), axis=1))
                if np.max(np.abs(points[order[0]])) <= 300 * scale:
                    return (points[order[:4]] / (128 * scale)).astype(np.float32)
    raise AssertionError


def centered_queries(x, split):
    a, b = split
    normal = x[a] - x[b]
    axis = np.argmax(np.abs(normal))
    direction = np.eye(x.shape[1], dtype=np.float32)[axis]
    threshold = np.float32(1e-8)
    delta = threshold / np.abs(normal[axis])
    while np.float32(delta * abs(normal[axis])) > threshold:
        delta = np.nextafter(delta, np.float32(0))
    while np.nextafter(delta, np.float32(np.inf)) * abs(normal[axis]) <= threshold:
        delta = np.nextafter(delta, np.float32(np.inf))
    adjacent = [np.nextafter(delta, np.float32(0)), delta, np.nextafter(delta, np.float32(np.inf))]
    tiny = np.nextafter(np.float32(0), np.float32(1))
    q = np.array([direction * amount for amount in
                  [0, -tiny, tiny, *adjacent, *[-v for v in adjacent], 0.1, -0.1]],
                 dtype=np.float32)
    return q, [float(value * abs(normal[axis])) for value in adjacent]


def schedules(q, query_seed):
    rng = np.random.default_rng(query_seed)
    paired = np.repeat(q, 2, axis=0)
    interleaved = np.vstack((q, q[::-1]))[rng.permutation(2 * len(q))]
    return [("repeat", np.repeat(q[:1], 32, axis=0)), ("pairs", paired),
            ("reversed", paired[::-1].copy()), ("mixed", interleaved),
            ("single", q[:1])]


def adjacent_queries(x, splits):
    q = []
    for a, b in splits:
        mid = (x[a] + x[b]) * np.float32(0.5)
        q.append(mid)
        for axis, end in itertools.product(range(len(mid)), (-np.inf, np.inf)):
            neighbor = mid.copy()
            neighbor[axis] = np.nextafter(neighbor[axis], np.float32(end))
            q.append(neighbor)
    return np.array(q)


def observe(search, name, x, q, archived, words, caller, scalar, boundary=None, entry_only=False):
    n, d = x.shape
    cap, depth = archived["cap"], archived["depth"]
    state = ControlledState(caller, scalar)
    prepared = fit(search, x, state, cap, depth, 5)
    graph, distance = prepared.neighbor_graph
    live, live_distance = fit(search, x, ControlledState(caller, scalar), cap, depth, 6).neighbor_graph
    assert np.array_equal(graph, live[:, :5])
    assert np.array_equal(distance.view(np.uint32), live_distance[:, :5].view(np.uint32))
    assert np.array_equal(graph.ravel(), archived["graph"])
    prepared.prepare()
    runs = []
    for label, rows in schedules(q, 57947):
        widths = (1, 3, 5, 13, n + 17) if label in ("pairs", "mixed") else (1,)
        eps = EPSILONS if label in ("pairs", "mixed") else (0.0,)
        if entry_only:
            widths, eps = (1,), (0.0,)
        for width, epsilon in itertools.product(widths, eps):
            result = ranks(prepared.query(rows, k=width, epsilon=epsilon))
            assert result == ranks(prepared.query(rows, k=width, epsilon=epsilon))
            runs.append(dict(label=label, query_bits=bits(rows), width=width, epsilon=epsilon,
                             **result))
    print(name, "batches", len(runs), flush=True)
    return dict(name=name, n=n, d=d, caller=caller, cap=cap, depth=depth, words=words,
                train_bits=bits(x), graph=graph.ravel().tolist(), live=live.ravel().tolist(),
                splits=oriented_splits(x, words, cap, depth), draws=state.draws,
                boundary=boundary, runs=runs)


def cases(search, archive):
    by_name = {c["name"]: c for c in archive["cases"]}
    for name, caller, scalar, scales in (
        (FIRST, 0, 1, (1.0, 1 / 16)),
        (SECOND, 73, 1 << 32, (1.0, 64.0)),
    ):
        archive_case = by_name[name]
        d, cap, depth = (archive_case[key] for key in ("d", "cap", "depth"))
        x = np.array(archive_case["train_bits"], np.uint32).view(np.float32).reshape(-1, d)
        # The unordered depth-one archives leave the root direction
        # undetermined. Calibrate bit zero as the child of the larger pivot.
        a, b = archive_case["geometry"][0][::-1]
        words, _ = encode(x, [[a, b], None, None], cap, depth)
        midpoint = (x[a] + x[b]) * np.float32(0.5)
        for multiplier in scales:
            center = (x - midpoint) * np.float32(multiplier)
            q, boundary = centered_queries(center, [a, b])
            holdout = 2**32 - 1 if caller == 0 else 65537
            result = observe(search, f"{name}_center_{multiplier}", center, q, archive_case,
                             words, caller if multiplier == 1 else holdout, scalar, boundary)
            yield result
    c = by_name[DEEP]
    x = np.array(c["train_bits"], np.uint32).view(np.float32).reshape(-1, c["d"])
    pairs = oriented_splits(x, c["words"], c["cap"], c["depth"])
    midpoints = [(x[a] + x[b]) * np.float32(0.5) for a, b in pairs]
    q = np.vstack((exact_intersection(x, pairs), exact_intersection(x, pairs[:2]), midpoints,
                   x[[0, -1]]))
    for caller in (1, 73, 2**32 - 2, 2**32 - 1):
        yield observe(search, f"{DEEP}_caller_{caller}", x, q, c, c["words"], caller, 1 << 32)
    for name, caller, scalar in FLOAT_CASES:
        c = by_name[name]
        x = np.array(c["train_bits"], np.uint32).view(np.float32).reshape(-1, c["d"])
        q = adjacent_queries(x, oriented_splits(x, c["words"], c["cap"], c["depth"]))
        yield observe(search, f"{name}_float_nextafter", x, q, c, c["words"], caller, scalar,
                      entry_only=True)


def feature_holdouts(search):
    for dimension, caller, scalar in (
        (4, 181, 450), (5, 27, 19), (8, 76, 9217), (12, 773, 17), (17, 192, 1234),
        (19, 43, 65548), (31, 84, 4462), (32, 177, 71), (33, 449, 995),
        (34, 578, 164), (48, 1, 187), (64, 395, 157), (65, 4294967295, 467), (97, 12345, 33),
    ):
        rng = np.random.default_rng(12451 + dimension)
        n, cap = 107, 106
        x = rng.standard_normal((n, dimension)).astype(np.float32)
        calibrated = fit(search, x, ControlledState(caller, scalar), cap, 1, 5)
        public_graph = calibrated.neighbor_graph[0].ravel().tolist()
        calibrated.prepare()
        witnesses = rng.standard_normal((1024, dimension)).astype(np.float32)
        winners = calibrated.query(witnesses, k=1, epsilon=0)[0].ravel()
        pair, = query_pairs(x, witnesses, winners)
        a, b = map(int, pair)
        scale = np.float32(1 / 128)
        narrow = x * scale
        mid = (narrow[a] + narrow[b]) * np.float32(0.5)
        sides = []
        for lanes in ((0, 0, 0), (7, 15, 31)):
            forced = fit(search, narrow, ControlledState(caller, scalar, (2, lanes)), cap, 1, 5)
            forced.prepare()
            row = forced.query(mid[None, :], k=1, epsilon=0)[0][0, 0]
            sides.append(((x[row] - x[b]) ** 2).sum() < ((x[row] - x[a]) ** 2).sum())
        assert sides[0] != sides[1]
        if sides[0]:
            a, b = b, a
        words, _ = encode(x, [[a, b], None, None], cap, 1)
        q = adjacent_queries(x, [(a, b)])
        setup = dict(cap=cap, depth=1, graph=public_graph)
        yield observe(search, f"float_width_{dimension}_caller_{caller}", x, q, setup, words,
                      caller, scalar, entry_only=True)


def draw_controls(search):
    x = np.ones((25, 1), np.float32)
    x[::2] = -1
    q = np.zeros((64, 1), np.float32)
    known = (1819927849474927636, 0, 1, 42)
    controls = []
    for caller, scalars, replace in itertools.chain(
        ((seed, known, None) for seed in (0, 1, 2, 42, 73, 65537, 2**32 - 2, 2**32 - 1)),
        ((42, known[:1], (i, words)) for i, words in
         itertools.product((1, 2), ((0, 0, 0), (12345, -23456, 34567), (7, 15, 31)))),
    ):
        for scalar in scalars:
            state = ControlledState(caller, scalar, replace)
            ref = fit(search, x, state, 12, 1, 5)
            ref.prepare()
            seq = ref.query(q, k=1, epsilon=0)[0].ravel().tolist()
            assert seq == ref.query(q, k=1, epsilon=0)[0].ravel().tolist()
            controls.append(dict(caller=caller, scalar=scalar, replacement=replace,
                                 draws=state.draws, ranks=seq))
    return controls


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
                          controls=draw_controls(NNDescent),
                          cases=[*cases(NNDescent, archive), *feature_holdouts(NNDescent)])
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
