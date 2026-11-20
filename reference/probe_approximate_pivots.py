"""Acquire public one-tree pivot, tie, and seeded approximate-rank controls."""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_approximate_construction import RecordedRandomState, holdout_cases, index, np, ranks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(indices):
    n, width = indices.shape
    row = np.repeat(np.arange(n), width)
    column = indices.ravel()
    keep = column >= 0
    graph = coo_matrix((np.ones(keep.sum(), dtype=np.uint8), (row[keep], column[keep])), shape=(n, n))
    return connected_components(graph)


def pivot_witness(data, indices):
    count, side = components(indices)
    if count != 2:
        return {"components": count, "compatible_pairs": []}
    square = ((data[:, None] - data[None, :]) ** 2).sum(axis=2)
    pairs = []
    for a in range(len(data)):
        left = square[a] < square[a + 1:]
        exact = (left == side).all(axis=1) | (left != side).all(axis=1)
        pairs.extend([a, int(b)] for b in np.flatnonzero(exact) + a + 1)
    return {"components": count, "compatible_pairs": pairs}


def root_controls(search):
    observed = {}
    scalar_seeds = (0, 1, 42, (1 << 32) + 1, 1819927849474927636, 2033886794655244609)
    sizes = (29, 37, 53, 61, 79, 101, 131, 163, 197, 251, 271)
    for scalar in scalar_seeds:
        results = {}
        for n in sizes:
            data = np.random.default_rng(122).standard_normal((n, 11)).astype(np.float32)
            state = RecordedRandomState(42, forest_seed=scalar)
            columns = index(search, data, "euclidean", state, 7, n_trees=1,
                            max_rptree_depth=1, leaf_size=4, n_iters=0).neighbor_graph[0]
            results[n] = pivot_witness(data, columns) | {
                "direct_pcg64": np.random.default_rng(scalar).integers(n, size=2).tolist(),
            }
        observed[str(scalar)] = results
    return observed


def caller_seed_controls(search):
    n = 131
    data = np.random.default_rng(122).standard_normal((n, 11)).astype(np.float32)
    observations = {}
    for seed in (0, 1, 19, 42, 73, 2 ** 32 - 1):
        rng = RecordedRandomState(seed)
        columns = index(search, data, "euclidean", rng, 7, n_trees=1,
                        max_rptree_depth=1, leaf_size=4, n_iters=0).neighbor_graph[0]
        observations[seed] = pivot_witness(data, columns) | {
            "first_state": rng.draws[0][3], "second_state": rng.draws[1][3], "scalar": rng.draws[2][3],
        }
    return observations


def pivot_order_controls(search, roots):
    """Swap the pivot rows and relabel output before comparing two levels."""
    n = 131
    data = np.random.default_rng(122).standard_normal((n, 11)).astype(np.float32)
    observations = {}
    for scalar in ("0", "1819927849474927636", "2033886794655244609"):
        a, b = roots[scalar][n]["compatible_pairs"][0]
        permutation = np.arange(n)
        permutation[[a, b]] = permutation[[b, a]]
        checked = {}
        for depth in (1, 2):
            def columns(sample):
                state = RecordedRandomState(42, forest_seed=int(scalar))
                return index(search, sample, "euclidean", state, 7, n_trees=1,
                             max_rptree_depth=depth, leaf_size=4, n_iters=0).neighbor_graph[0]
            original = columns(data)
            swapped = columns(data[permutation])[permutation]
            relabeled = np.where(swapped >= 0, permutation[swapped], -1)
            checked[depth] = {"changed_rank_slots": int((original != relabeled).sum()),
                              "components": components(original)[0],
                              "swapped_components": components(relabeled)[0]}
        observations[scalar] = checked
    return observations


def tie_control(search, scalar, pair, n=131):
    data = np.random.default_rng(191).standard_normal((n, 5)).astype(np.float32)
    a, b = pair
    data[:, 0] = 0.0
    data[a] = [-1, 0, 0, 0, 0]
    data[b] = [1, 0, 0, 0, 0]
    others = [i for i in range(n) if i not in pair]
    data[others[:9], 0] = -2
    data[others[9:18], 0] = 2
    tied = others[18:]
    results = {}
    for label, sample in (("exact_ties", data), ("swap_pivots", data.copy()),
                          ("reverse_tie_values", data.copy()), ("one_positive_margin", data.copy()),
                          ("positive_margin", data.copy())):
        if label == "swap_pivots":
            sample[[a, b]] = sample[[b, a]]
        if label == "reverse_tie_values":
            sample[tied] = sample[tied[::-1]]
        if label == "positive_margin":
            sample[tied, 0] = 2 ** -12
        if label == "one_positive_margin":
            sample[tied[0], 0] = 2 ** -12
        state = RecordedRandomState(42, forest_seed=scalar)
        columns = index(search, sample, "euclidean", state, 11, n_trees=1,
                        max_rptree_depth=1, leaf_size=4, n_iters=0).neighbor_graph[0]
        count, groups = components(columns)
        results[label] = {"components": count, "with_pivot_a": (groups[tied] == groups[a]).tolist()}
    return {"pivots": pair, "tied_rows": tied, "partitions": results}


class VacancyState(RecordedRandomState):
    def __init__(self, seed, draw, words):
        super().__init__(seed)
        self.draw = draw
        self.words = words

    def randint(self, low, high=None, size=None, dtype=int):
        values = super().randint(low, high, size, dtype)
        if len(self.draws) == self.draw and size == 3:
            return np.asarray(self.words, dtype=dtype)
        return values


def vacancy_control(search):
    data = np.random.default_rng(57).standard_normal((131, 11)).astype(np.float32)
    replacement = np.random.RandomState(73).randint(-2147483647, 2147483646, 3, dtype=np.int64)
    outcomes = {}
    for trees, width, leaf in ((0, 7, 130), (1, 7, 130), (1, 95, 130)):
        def build(seed):
            return index(search, data, "euclidean", seed, width, n_trees=trees,
                         leaf_size=leaf, n_iters=0).neighbor_graph[0]
        expected = build(42)
        outcomes[f"{trees}/{width}"] = {
            "rank_slots": expected.size,
            "edit_first_draw": int((expected != build(VacancyState(42, 1, replacement))).sum()),
            "edit_second_draw": int((expected != build(VacancyState(42, 2, replacement))).sum()),
        }
    return outcomes


def sample_controls(search, cases):
    observations = {}
    for name, case in cases.items():
        n, d, metric, seed = (case[k] for k in ("n_rows", "n_features", "metric", "seed"))
        data = np.asarray(case["training_bits"], dtype=np.uint32).view(np.float32).reshape(n, d)
        note = {}
        previous = None
        for trees in (1, 2, 4, 8):
            model = index(search, data, metric, seed, n_trees=trees, n_iters=0, leaf_size=60)
            cols, distances = model.neighbor_graph
            worsened = lost = 0
            if previous is not None:
                prev_cols, prev_distances = previous
                retained = (prev_cols[:, :, None] == cols[:, None, :]).any(axis=2)
                worsened = int((distances[:, -1] > prev_distances[:, -1]).sum())
                lost = int(((~retained) & (prev_distances < distances[:, -1:])).sum())
            note[trees] = {
                "empty_slots": int((cols < 0).sum()),
                "worsened_last_rank": worsened, "lost_strict_candidates": lost,
            }
            previous = cols, distances
        observations[name] = note
    return observations


def later_round_controls(search, cases):
    observations = {}
    for name, case in cases.items():
        if case["n_rows"] != 4096 or case["metric"] != "correlation":
            continue
        n, d, seed = (case[k] for k in ("n_rows", "n_features", "seed"))
        data = np.asarray(case["training_bits"], dtype=np.uint32).view(np.float32).reshape(n, d)
        width_six = index(search, data, "correlation", seed, 6, n_trees=8,
                          n_iters=0, leaf_size=60).neighbor_graph[0]
        live = width_six[:, 1:].copy()
        original = {}
        frozen = {}
        continued = {}
        for step in (1, 2, 3, 12):
            original[step] = index(search, data, "correlation", seed, n_trees=8, leaf_size=60,
                                   n_iters=step, delta=0).neighbor_graph[0]
            frozen[step] = index(search, data, "correlation", seed, n_iters=step, delta=0,
                                 init_graph=width_six[:, :5].copy(), tree_init=False).neighbor_graph[0]
            continued[step] = index(search, data, "correlation", seed, n_iters=step, delta=0,
                                    init_graph=live.copy(), tree_init=False).neighbor_graph[0]
        observations[name] = {
            step: {"normal_vs_frozen": int((original[step] != frozen[step]).sum()),
                   "normal_vs_live": int((original[step] != continued[step]).sum())}
            for step in original
        }
    return observations


def distance_controls(search):
    observations = {}
    for width in (1, 2, 3, 4, 5, 7, 8, 11, 16, 23, 31, 32, 33, 36, 47, 63, 64, 65, 96, 127):
        n = 29
        data = np.random.default_rng(321 + width % 3).standard_normal((n, width)).astype(np.float32)
        fixture = ranks(index(search, data, "euclidean", 19, n, n_trees=1, leaf_size=n, n_iters=0))
        observations[str(width)] = {"data": data.view(np.uint32).ravel().tolist(), **fixture}
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pynndescent import NNDescent
            from umap import UMAP
            roots = root_controls(NNDescent)
            ordering = pivot_order_controls(NNDescent, roots)
            ties = {f"{n}/{key}": tie_control(NNDescent, int(key), roots[key][n]["compatible_pairs"][0], n)
                    for key in ("0", "1819927849474927636", "2033886794655244609")
                    for n in (131, 197, 251)}
            caller_seeds = caller_seed_controls(NNDescent)
            vacancies = vacancy_control(NNDescent)
            distances = distance_controls(NNDescent)
            inputs = dict(holdout_cases(UMAP, NNDescent, 7, 0, 57, 62)) | dict(
                holdout_cases(UMAP, NNDescent, 11, 19, 88, 101))
            observations = {"cases": inputs, "roots": roots, "pivot_order": ordering,
                            "ties": ties, "vacancies": vacancies,
                            "caller_seeds": caller_seeds,
                            "tree_sampling": sample_controls(NNDescent, inputs),
                            "later_rounds": later_round_controls(NNDescent, inputs),
                            "distance_controls": distances}
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    args.output.write_text(json.dumps(observations, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
