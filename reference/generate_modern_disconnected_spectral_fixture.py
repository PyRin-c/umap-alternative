"""Record disconnected modern spectral fits through the public 0.6.0 wheel."""

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
from umap import UMAP

from generate_modern_spectral_fit_fixture import bits, cycle, rng_checkpoint, tree_with_weak_edges


class TracedRandomState(np.random.RandomState):
    def __init__(self, seed, jitter):
        super().__init__(seed)
        self.jitter = jitter
        self.draws = []

    def uniform(self, low=0.0, high=1.0, size=None):
        before = rng_checkpoint(self)
        result = super().uniform(low, high, size)
        self.draws.append(["uniform", list(result.shape), [low, high], before, rng_checkpoint(self)])
        return result

    def normal(self, loc=0.0, scale=1.0, size=None):
        before = rng_checkpoint(self)
        result = super().normal(loc, scale, size)
        self.draws.append(["normal", list(result.shape), [loc, scale], before, rng_checkpoint(self)])
        return np.zeros_like(result) if self.jitter == "zero" else result


def observe(spec, method):
    rng = TracedRandomState(spec["seed"], spec.get("jitter"))
    if spec.get("warm_gaussian"):
        rng.standard_normal()
    start = rng_checkpoint(rng)
    options = spec["options"]
    model = UMAP(**options, init="spectral", random_state=rng, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = getattr(model, method)(np.array(spec["data"], dtype=np.float32))
        except Exception as error:
            raise SystemExit(f"{spec['name']}, {method}: {type(error).__name__}: {error}") from None
    if method == "fit":
        assert result is model
    else:
        np.testing.assert_array_equal(result, model.embedding_)
    graph = model.graph_.tocsr()
    sampled = graph.copy()
    budget = options["n_epochs"]
    maximum_epoch = max(budget) if isinstance(budget, list) else budget
    sampled.data[sampled.data < np.float32(sampled.data.max() / (maximum_epoch if maximum_epoch > 10 else 500))] = 0
    sampled.eliminate_zeros()
    assert (graph.nnz > sampled.nnz) == spec.get("pruned", False)
    if spec.get("pruned"):
        assert connected_components(graph, directed=False, return_labels=False) == 1
    count, labels = connected_components(sampled, directed=False)
    assert count > 1
    width = options.get("n_components", 2)
    sizes = np.bincount(labels)
    calls = [["uniform", [count]]] if count > 2 * width else []
    calls += [["uniform", [int(size), width]] for size in sizes if size < max(2 * width, width + 2)]
    calls += [["normal", [graph.shape[0], width]]]
    assert [entry[:2] for entry in rng.draws] == calls, spec["name"]
    assert not any("Spectral initialisation needs more samples" in str(w.message) for w in caught)
    end = rng_checkpoint(rng)
    end["next_words"] = rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()
    return {
        "graph": {"indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(), "weights_bits": bits(graph.data)},
        "embedding_bits": bits(model.embedding_),
        "history_bits": [bits(x) for x in model.embedding_list_] if hasattr(model, "embedding_list_") else None,
        "sampled_labels": labels.tolist(),
        "rng_start": start,
        "initialization_draws": rng.draws,
        "rng": end,
        "affinity_warning": any("Graph is not fully connected" in str(w.message) for w in caught),
    }


def two_trees_with_weak_bridge():
    tree = np.asarray(tree_with_weak_edges(), dtype=np.float32)
    length = len(tree)
    distances = np.full((2 * length, 2 * length), 20, dtype=np.float32)
    distances[:length, :length] = distances[length:, length:] = tree
    source, target = np.argwhere((tree < 20) & (tree.T == 20))[0]
    distances[source, length] = distances[source, target]
    distances[source, target] = 20
    return distances


def observe_all_isolated():
    data = np.arange(6, dtype=np.float32)[:, None]
    options = {"optimizer": "adam", "n_neighbors": 2, "n_epochs": 2, "disconnection_distance": 0.5}
    rng = TracedRandomState(11, None)
    rng.standard_normal()
    before = rng_checkpoint(rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            UMAP(**options, init="spectral", random_state=rng, n_jobs=1).fit(data)
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("all-isolated public fit unexpectedly succeeded")
    assert any("fully disconnected" in str(w.message) for w in caught)
    assert not rng.draws and rng_checkpoint(rng) == before
    return {
        "data": data.tolist(), "options": options, "seed": 11, "error": message,
        "rng": before | {"next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()},
    }


def make_fixture():
    cases = []
    residuals = []

    def add(name, data, optimizer, *, seed=42, warm=False, pruned=False, repeated=False, **options):
        spec = {
            "name": name,
            "seed": seed,
            "data": np.asarray(data, dtype=np.float32).tolist(),
            "options": {
                "compatibility_layout": False, "optimizer": optimizer, "n_neighbors": 2,
                "n_epochs": [0, 1, 2, 4, 6],
            } | options,
        }
        if warm:
            spec["warm_gaussian"] = True
        if pruned:
            spec["pruned"] = True
        if repeated:
            spec["jitter"] = "zero"
        result = observe(spec, "fit")
        second = observe(spec, "fit_transform")
        if not repeated:
            assert result == second, name
            cases.append(spec | result)
        else:
            repeats = [result, second] + [observe(spec, "fit") for _ in range(6)]
            residuals.append(spec | result | {"observations": [r["embedding_bits"] for r in repeats]})

    pairs = np.array([0, .01, .12, .13, .288, .299, .44, .449, .83, .839, 1.1, 1.107], np.float32)[:, None]
    interleaved = pairs[[0, 2, 4, 1, 5, 3]]
    isolated = np.array([0, .11, .247, 4.1, 4.17, 8], np.float32)[:, None]
    clusters = np.concatenate([
        np.random.RandomState(531 + index).normal(center, .018, (size, 2))
        for index, (center, size) in enumerate(zip(
            [(-2., 0.), (.8, 1.), (1.2, -.6), (.1, .4), (1.52, .03)], [8, 7, 6, 5, 3],
        ))
    ]).astype(np.float32)
    precomputed = pairwise_distances(pairs)
    asymmetric = precomputed + np.random.RandomState(130).uniform(0, .04, precomputed.shape).astype(np.float32)
    np.fill_diagonal(asymmetric, 0)

    for optimizer in ("adam", "momentum"):
        add(f"{optimizer}-interleaved-axes", interleaved, optimizer, seed=109)
        add(f"{optimizer}-cutoff-isolated", isolated, optimizer, seed=113, warm=True,
            n_neighbors=3, disconnection_distance=.6, init_rescale="graph_edge",
            negative_selection_range=2, exclude_graph_neighbors=True)
        add(f"{optimizer}-centroid-placement", pairs, optimizer, seed=42, warm=True,
            n_components=1, negative_sample_rate=0)
        add(f"{optimizer}-cutoff-local-solve", clusters, optimizer,
            n_neighbors=5, disconnection_distance=.15, a=1.7, b=.9)
        add(f"{optimizer}-disconnected-rank-limit", interleaved, optimizer, seed=821,
            n_components=6 if optimizer == "adam" else 7, init_rescale="per_axis")
    add("component-count-equals-axis-limit", pairs[:8], "adam", seed=803, n_epochs=7)
    add("local-size-threshold-width-three", clusters, "momentum", seed=43, warm=True,
        n_neighbors=5, disconnection_distance=.15, n_components=3, n_epochs=[0, 1, 2, 5],
        a=1.7, b=.9)
    add("precomputed-directed-block-means", asymmetric, "momentum", metric="precomputed")
    add("precomputed-symmetric-block-means", precomputed, "adam", metric="precomputed", n_components=1)
    add("pruned-weak-bridge", two_trees_with_weak_bridge(), "adam", n_neighbors=4,
        set_op_mix_ratio=.0625, local_connectivity=3.0, metric="precomputed", pruned=True,
        n_epochs=[12, 2, 0, 2, 5])

    repeated = np.full((8, 8), 8, dtype=np.float32)
    repeated[:6, :6] = cycle(6)
    repeated[6:, 6:] = [[0, 1], [1, 0]]
    add("cycle-and-pair", repeated, "momentum", repeated=True, metric="precomputed",
        n_neighbors=3, disconnection_distance=2.0, local_connectivity=2.0, n_epochs=[0])
    add("precomputed-cutoff-meta-affinity", precomputed, "adam", metric="precomputed",
        disconnection_distance=.05, n_epochs=[0])

    return {
        "operation": ["UMAP.fit", "UMAP.fit_transform"],
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
        "residuals": residuals,
        "all_isolated": observe_all_isolated(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_fixture()
    header = json.dumps({key: fixture[key] for key in ("operation", "environment", "all_isolated")}, indent=2)[:-2]
    sections = []
    for key in ("cases", "residuals"):
        encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in fixture[key])
        sections.append(f'  "{key}": [\n{encoded}\n  ]')
    args.output.write_text(f'{header},\n' + ",\n".join(sections) + "\n}\n", encoding="utf-8", newline="\n")
