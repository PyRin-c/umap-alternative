"""Record connected spectral initialization through the public modern estimator."""

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from umap import UMAP


NINE = [
    [0.2, 2, 1], [0.7, 0, 2], [1, 3, 1], [3, 1, 0], [4, 0, 3],
    [4, 3, 2], [6, 3, 3], [7, 1, 8], [8, 0, 2],
]


def cycle(n_rows):
    distances = np.full((n_rows, n_rows), 8, dtype=np.float32)
    np.fill_diagonal(distances, 0)
    for i in range(n_rows):
        j = (i + 1) % n_rows
        distances[i, j] = distances[j, i] = 1
    return distances.tolist()


def tree_with_weak_edges():
    """Build a connected asymmetric tree with additional unilateral edges."""
    n_rows = 9
    neighbors = [set() for _ in range(n_rows)]
    for source, target in ((0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7), (7, 8)):
        neighbors[source].add(target)
        neighbors[target].add(source)
    for source, row in enumerate(neighbors):
        for target in range(n_rows):
            if len(row) == 3:
                break
            if source != target and source not in neighbors[target]:
                row.add(target)
        assert len(row) == 3
    distances = np.full((n_rows, n_rows), 20, dtype=np.float32)
    np.fill_diagonal(distances, 0)
    for source, row in enumerate(neighbors):
        for rank, target in enumerate(sorted(row), 1):
            distances[source, target] = rank
    return distances.tolist()


def bits(matrix):
    assert matrix.dtype == np.float32
    return np.ascontiguousarray(matrix).view(np.uint32).ravel().tolist()


def rng_checkpoint(rng):
    _, _, cursor, has_gaussian, gaussian = rng.get_state()
    return {
        "cursor": cursor,
        "gaussian_bits": int(np.float64(gaussian).view(np.uint64)) if has_gaussian else None,
    }


class SpectralRandomState(np.random.RandomState):
    def __init__(self, seed, jitter):
        super().__init__(seed)
        self.events = []
        self.jitter = jitter

    def normal(self, *args, **kwargs):
        before = rng_checkpoint(self)
        values = super().normal(*args, **kwargs)
        self.events.append({
            "method": "normal", "scale": kwargs["scale"], "shape": list(values.shape),
            "before": before, "after": rng_checkpoint(self),
        })
        return np.zeros_like(values) if self.jitter == "zero" else values

    def uniform(self, *args, **kwargs):
        before = rng_checkpoint(self)
        values = super().uniform(*args, **kwargs)
        self.events.append({
            "method": "uniform", "bounds": [kwargs["low"], kwargs["high"]],
            "shape": list(values.shape), "before": before, "after": rng_checkpoint(self),
        })
        return values

    def standard_normal(self, size=None):
        before = rng_checkpoint(self)
        values = super().standard_normal(size)
        self.events.append({
            "method": "standard_normal", "shape": list(np.shape(values)),
            "before": before, "after": rng_checkpoint(self),
        })
        return values


def observe(spec, fit_transform=False):
    state = SpectralRandomState(spec["seed"], spec.get("jitter"))
    if spec.get("warm_gaussian"):
        state.standard_normal()
        state.events.clear()
    start = rng_checkpoint(state)
    data = np.asarray(spec["data"], dtype=np.float32)
    options = spec["options"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = UMAP(**options, random_state=state, init="spectral", n_jobs=1)
        if fit_transform:
            np.testing.assert_array_equal(estimator.fit_transform(data), estimator.embedding_)
        else:
            assert estimator.fit(data) is estimator
    graph = estimator.graph_
    budget = max(options["n_epochs"]) if isinstance(options["n_epochs"], list) else options["n_epochs"]
    sampled = graph.copy()
    sampled.data[sampled.data < np.float32(sampled.data.max() / (budget if budget > 10 else 500))] = 0
    sampled.eliminate_zeros()
    assert connected_components(sampled, directed=False, return_labels=False) == 1
    if spec.get("pruned"):
        assert sampled.nnz < graph.nnz
    rng = rng_checkpoint(state)
    rng["next_words"] = state.randint(0, 2**32, size=4, dtype=np.uint32).tolist()
    jitter_index = next(i for i, event in enumerate(state.events) if event["method"] == "normal")
    return {
        "graph": {
            "indptr": graph.indptr.tolist(),
            "indices": graph.indices.tolist(),
            "weights_bits": bits(graph.data),
        },
        "embedding_bits": bits(estimator.embedding_),
        "history_bits": [bits(x) for x in estimator.embedding_list_]
        if hasattr(estimator, "embedding_list_") else None,
        "rng_start": start,
        "initialization_draws": state.events[:jitter_index + 1],
        "rng": rng,
        "rank_warning": any("Spectral initialisation needs more samples" in str(w.message)
                            for w in caught),
    }


def make_fixture():
    cases = []

    def add(name, optimizer, *, data=NINE, seed=42, warm=False, jitter=None,
            pruned=False, **options):
        spec = {
            "name": name,
            "seed": seed,
            "data": data,
            "options": {
                "compatibility_layout": False, "optimizer": optimizer,
                "n_neighbors": 4, "n_epochs": [0, 1, 2, 4, 5],
            } | options,
        }
        if warm:
            spec["warm_gaussian"] = True
        if jitter is not None:
            spec["jitter"] = jitter
        if pruned:
            spec["pruned"] = True
        result = observe(spec)
        assert observe(spec, fit_transform=True) == result, name
        methods = [event["method"] for event in result["initialization_draws"]]
        fallback = spec["options"].get("n_components", 2) >= len(data)
        assert methods == (["uniform", "normal"] if fallback else ["normal"]), (name, methods)
        assert result["rank_warning"] == fallback
        cases.append(spec | result)

    for optimizer in ("momentum", "adam"):
        add(f"{optimizer}-default", optimizer)
        add(f"{optimizer}-different-seed", optimizer, seed=43, n_epochs=[0])
        add(f"{optimizer}-zero-jitter", optimizer, seed=15, jitter="zero", n_epochs=[0])
        add(f"{optimizer}-one-component", optimizer, seed=113, warm=True, n_components=1,
            init_rescale="per_axis", init_scale_factor=0.7, negative_sample_rate=0)
        add(f"{optimizer}-full-rank", optimizer, seed=112, n_components=8,
            n_epochs=[0, 1, 2, 3], init_rescale="graph_edge", a=1.7, b=0.9)
        add(f"{optimizer}-rank-limit", optimizer, seed=113, warm=True, n_components=9,
            n_epochs=[0, 2, 5, 7], init_rescale="per_axis")
        add(f"{optimizer}-pruned-before-solve", optimizer, seed=109, data=tree_with_weak_edges(),
            n_neighbors=4, local_connectivity=3.0, set_op_mix_ratio=0.0625,
            metric="precomputed", pruned=True,
            n_epochs=[12, 2, 0, 2, 5], negative_selection_range=3, exclude_graph_neighbors=True)
    add("near-rank-limit", "momentum", n_components=7, seed=117, n_epochs=[0, 1, 2])
    add("above-rank-limit", "adam", n_components=5, data=NINE[:3], seed=819, warm=True,
        n_neighbors=2, n_epochs=[0, 1, 3], init_rescale="graph_edge")
    add("small-full-rank", "momentum", n_components=2, data=NINE[:3],
        n_neighbors=2, n_epochs=[0, 1, 2])
    add("default-modern-adam", None, seed=711, n_epochs=9)
    for n, width, optimizer in ((40, 7, "adam"), (83, 2, "momentum")):
        data = np.random.RandomState(703).uniform(-3, 2, (n, 3)).astype(np.float32).tolist()
        add(f"heldout-{n}x{width}", optimizer, data=data, seed=813,
            n_neighbors=5, n_components=width, n_epochs=[0, 1, 4, 9], a=1.0, b=1.0)

    # Keep rank-degenerate results separate so repeated bases can be reported.
    degenerate = {
        "name": "symmetric-cycle",
        "data": cycle(6),
        "seed": 42,
        "jitter": "zero",
        "options": {
            "compatibility_layout": False, "optimizer": "momentum", "n_neighbors": 3,
            "metric": "precomputed", "local_connectivity": 2.0, "n_epochs": [0],
        },
    }
    outputs = [observe(degenerate) for _ in range(8)]
    residual = degenerate | outputs[0] | {
        "observations": [r["embedding_bits"] for r in outputs]
    }
    return {
        "operation": ["UMAP.fit", "UMAP.fit_transform"],
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in (
                "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "llvmlite"
            )},
            "threads": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"
            )},
        },
        "cases": cases,
        "rank_degenerate": residual,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_fixture()
    cases = fixture.pop("cases")
    header = json.dumps(fixture, indent=2)[:-2]
    encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in cases)
    args.output.write_text(f'{header},\n  "cases": [\n{encoded}\n  ]\n}}\n', encoding="utf-8", newline="\n")
