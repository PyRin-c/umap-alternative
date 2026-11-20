"""Record recursive initialization through public UMAP 0.6.0 fits."""

import argparse
import json
import os
import platform
import warnings
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
from umap import UMAP

from generate_modern_spectral_fit_fixture import NINE, bits, rng_checkpoint


class TracedRandomState(np.random.RandomState):
    def __init__(self, seed):
        super().__init__(seed)
        self.calls = []

    def normal(self, *args, **kwds):
        start = rng_checkpoint(self)
        values = super().normal(*args, **kwds)
        self.calls.append(["normal", list(values.shape), start, rng_checkpoint(self)])
        return values

    def randint(self, *args, **kwds):
        start = rng_checkpoint(self)
        values = super().randint(*args, **kwds)
        self.calls.append(["randint", list(values.shape), start, rng_checkpoint(self)])
        return values

    def shuffle(self, values):
        start = rng_checkpoint(self)
        super().shuffle(values)
        self.calls.append(["shuffle", list(values.shape), start, rng_checkpoint(self)])


def make_data(spec):
    if isinstance(spec, list):
        return np.asarray(spec, dtype=np.float32)
    rng = np.random.RandomState(spec["seed"])
    values = rng.uniform(-5, 7, spec["shape"]).astype(np.float32)
    return pairwise_distances(values) if spec.get("precomputed") else values


def observe(case, method):
    opts = case["options"]
    rng = TracedRandomState(case["seed"])
    if case.get("warm_gaussian"):
        rng.standard_normal()
    rng_start = rng_checkpoint(rng)
    model = UMAP(**opts, random_state=rng, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = getattr(model, method)(make_data(case["data"]))
        except Exception as error:
            raise SystemExit(f"{case['name']}, {method}: {type(error).__name__}: {error}") from None
    if method == "fit":
        assert result is model
    else:
        np.testing.assert_array_equal(result, model.embedding_)
    graph = model.graph_.tocsr()
    _, labels = connected_components(graph, directed=False)
    return {
        "graph": {"indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(), "weights_bits": bits(graph.data)},
        "component_sizes": np.bincount(labels).tolist(),
        "embedding_bits": bits(model.embedding_),
        "history_bits": [bits(layout) for layout in model.embedding_list_],
        "rng_start": rng_start,
        "rng_calls": rng.calls.copy(),
        "rng": rng_checkpoint(rng) | {"next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()},
    }


def star_distances(sizes, seed, *, chain=False, hierarchy=False):
    size = sum(sizes)
    distances = np.random.RandomState(seed).uniform(6, 10, (size, size)).astype(np.float32)
    cuts = np.r_[0, np.cumsum(sizes)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        distances[start:stop, start] = 1
        distances[start, start + 1] = 2
    if hierarchy:
        for row in (0, 6, 9, 15, 18, 24):
            distances[row, row // 9 * 9 + 3] = 1
        if chain:
            for row in (3, 12):
                distances[row, row + 9] = 1
    elif chain:
        distances[cuts[:-2], cuts[1:-1]] = 1
    np.fill_diagonal(distances, 0)
    return distances.tolist()


def make_fixture():
    cases, probes = [], []

    def add(name, data=NINE, *, seed=42, warm=False, levels=0, probe=False, **opts):
        case = {
            "name": name, "data": data, "seed": seed, "warm_gaussian": warm, "levels": levels,
            "options": {
                "init": "recursive", "compatibility_layout": False, "optimizer": "adam",
                "n_epochs": [0, 1, 2, 4, 6], "n_neighbors": 4, "a": 1.0, "b": 1.0,
            } | opts,
        }
        outcome = observe(case, "fit")
        assert sum(call[0] == "randint" for call in outcome["rng_calls"]) == levels + 1, name
        assert observe(case, "fit_transform") == outcome, name
        if probe:
            if len(outcome["graph"]["indices"]) > 5000:
                graph = outcome.pop("graph")
                payload = b"".join(np.asarray(graph[key], dtype="<u4").tobytes() for key in (
                    "indptr", "indices", "weights_bits",
                ))
                outcome["graph_sha256"] = sha256(payload).hexdigest()
            probes.append(case | outcome)
        else:
            cases.append(case | outcome)

    separated = [[0., 0.], [.1, .08], [.3, .809], [4., .117], [4.2, -.07], [5.9, 0.], [6.2, .2], [8., 4.]]
    for optimizer in ("adam", "momentum"):
        add(f"{optimizer}-default-base", optimizer=optimizer)
        add(f"{optimizer}-explicit-base-boundary", optimizer=optimizer, warm=True, seed=77,
            recursive_coarsening_ratio=2, coarse_layout_weights="fuzzy_union", coarsen_on_layout_weights=False,
            recursive_init_kwds={"base_init_threshold": 9, "n_embedding_epochs": 0},
            n_epochs=[0, 1, 2, 3], negative_sample_rate=0, init_rescale="graph_edge", init_scale_factor=.8)
        add(f"{optimizer}-cutoff-base", separated, optimizer=optimizer, warm=True, seed=97,
            n_neighbors=3, disconnection_distance=.5, init_rescale="per_axis", negative_selection_range=3,
            exclude_graph_neighbors=True)
    add("auto-selects-recursive", init="auto", seed=93)
    add("tall-randomized-sketch", {"seed": 831, "shape": [57, 31]}, seed=883,
        n_components=3, n_neighbors=2, recursive_init_kwds={"base_init_threshold": 57, "anchor_method": "sampled_pca"})
    add("wide-randomized-sketch", {"seed": 713, "shape": [12, 27]}, seed=771,
        n_components=3, n_neighbors=2, n_epochs=[0, 1, 3], optimizer="momentum")
    add("precomputed-base", {"seed": 741, "shape": [12, 3], "precomputed": True},
        metric="precomputed", n_neighbors=2, n_components=1, n_epochs=[0, 1, 3])
    add("default-base-limit", {"seed": 141, "shape": [1024, 3]}, n_neighbors=2, n_epochs=[0])
    small_pairs = [[0], [.039], [.71], [.91], [2.74], [2.88]]
    add("adam-refinement-phase-boundary", small_pairs, n_components=1, n_neighbors=2,
        n_epochs=[0, 1, 2, 3, 7, 15, 16], learning_rate=2.25, negative_sample_rate=0,
        init_rescale="graph_edge", init_scale_factor=.75)
    add("momentum-refinement-with-zero-rate", small_pairs, optimizer="momentum", n_components=1,
        n_neighbors=2, n_epochs=[0, 1, 2, 4, 6], learning_rate=0., negative_sample_rate=0,
        init_rescale="graph_edge", init_scale_factor=.75)

    repeated = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for seed in (41, 42):
        add(f"repeated-axis-{seed}", repeated, seed=seed, n_neighbors=2, n_epochs=[0])

    separate_stars = [row for c, y in ((0., .1), (4., -.3), (6., .4))
                      for row in ((c, y), (c - .12, y), (c + .13, y + .02))]
    add("first-level-natural-components", separate_stars, optimizer="momentum", levels=1, n_neighbors=2,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    linked_stars = star_distances([3, 4, 5], 121, chain=True)
    add("first-level-connected-stars", linked_stars, levels=1, metric="precomputed", n_neighbors=2,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    with_tails = np.asarray(star_distances([5, 5, 5], 184, chain=True), dtype=np.float32)
    for start in (0, 5, 10):
        with_tails[start + 3:start + 5, start] = 8
        with_tails[start + 3, start + 1] = with_tails[start + 4, start + 3] = 1
    add("first-level-paths-to-hubs", with_tails.tolist(), levels=1, metric="precomputed", n_neighbors=2,
        seed=21, optimizer="momentum", set_op_mix_ratio=.0625, n_epochs=[0, 1, 5],
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    deep = star_distances([3] * 9, 620, chain=True, hierarchy=True)
    for weights in ("transition", "fuzzy_union"):
        add(f"two-level-connected-{weights}", deep, levels=2, metric="precomputed", n_neighbors=2,
            recursive_coarsening_ratio=3, coarse_layout_weights=weights,
            set_op_mix_ratio=.5, warm=True, seed=81, negative_sample_rate=0,
            recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0, "remove_coarse_diagonal": True})
    add("two-level-fuzzy-partition-with-transition-weights", deep, levels=2, metric="precomputed", n_neighbors=2,
        recursive_coarsening_ratio=3, coarsen_on_layout_weights=False, seed=81, warm=True,
        set_op_mix_ratio=.5, negative_sample_rate=0, n_epochs=[0],
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0, "remove_coarse_diagonal": True})
    add("two-level-component-forest", star_distances([3] * 9, 619, hierarchy=True), optimizer="momentum",
        levels=2, metric="precomputed", n_neighbors=2, recursive_coarsening_ratio=3,
        coarsen_on_layout_weights=False, negative_sample_rate=0, seed=83,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0, "remove_coarse_diagonal": False})
    paired = np.asarray(star_distances([2] * 6, 373), dtype=np.float32)
    for start in (0, 4, 8):
        paired[start, start + 2] = 1
    paired[2, 6] = paired[6, 10] = 1
    add("two-level-ratio-two-boundary", paired.tolist(), levels=2, metric="precomputed", n_neighbors=2,
        recursive_coarsening_ratio=2, negative_sample_rate=0, n_epochs=[0, 1, 3],
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})

    connect = {"seed": 711, "shape": [12, 3]}
    add("first-level-ratio-four", connect, probe=True, levels=1, n_neighbors=8,
        recursive_init_kwds={"base_init_threshold": 6, "n_embedding_epochs": 0})
    add("first-level-fuzzy-ratio-three", connect, probe=True, levels=1, n_neighbors=8,
        recursive_init_kwds={"base_init_threshold": 4, "n_embedding_epochs": 0}, recursive_coarsening_ratio=3,
        coarse_layout_weights="fuzzy_union", coarsen_on_layout_weights=False)
    add("deep-ratio-two", connect, probe=True, levels=3, n_neighbors=8,
        recursive_init_kwds={"base_init_threshold": 2, "n_embedding_epochs": 0}, recursive_coarsening_ratio=2)
    add("first-level-isolated", separated, probe=True, levels=1, n_neighbors=3, disconnection_distance=.5,
        recursive_init_kwds={"base_init_threshold": 2, "n_embedding_epochs": 0})
    add("first-level-optimized", linked_stars, probe=True, levels=1, n_neighbors=2, n_epochs=[0],
        metric="precomputed", recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 3})
    add("above-default-base-limit", {"seed": 141, "shape": [1025, 3]}, probe=True, levels=1, n_neighbors=8,
        recursive_init_kwds={"n_embedding_epochs": 0}, n_epochs=[0])
    return {
        "environment": {
            "platform": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "packages": {name: version(name) for name in (
                "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "llvmlite"
            )},
            "threads": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"
            )},
        },
        "cases": cases,
        "probes": probes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_fixture()
    header = json.dumps({"environment": fixture["environment"]}, indent=2)[:-2]
    sections = []
    for key in ("cases", "probes"):
        encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in fixture[key])
        sections.append(f'  "{key}": [\n{encoded}\n  ]')
    args.output.write_text(f"{header},\n" + ",\n".join(sections) + "\n}\n", encoding="utf-8", newline="\n")
