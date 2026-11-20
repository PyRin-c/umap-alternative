"""Record seeded modern training through public UMAP.fit and fit_transform."""

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


FEATURES = [
    [-3.0, -0.2], [-1.5, 0.88], [-0.28, -3.0], [0.5, -0.72],
    [1.2, 3.0], [2.7, 0.3], [4.1, 1.0], [5.0, -4.0],
]
EXPLICIT = [
    [-9.0, 0.0], [-4.0, 3.0], [-0.3, -5.5], [1.0, 7.0],
    [6.0, 2.0], [-7.0, 8.0], [0.0, 10.0], [0.25, -15.0],
]


def ring(n_rows, *, weak_leaf=False):
    data = np.full((n_rows, n_rows), 20, dtype=np.float32)
    np.fill_diagonal(data, 0)
    n_ring = n_rows - int(weak_leaf)
    for i in range(n_ring):
        j = (i + 1) % n_ring
        data[i, j] = data[j, i] = np.float32(1 + i / 10)
    if weak_leaf:
        data[-1, 0] = data[0, -1] = 4
        data[-1, 1] = data[1, -1] = 5
    return data.tolist()


class TiedProjectionRandomState(np.random.RandomState):
    """Provide a deterministic first direction with several equal keys."""

    def __init__(self, seed):
        super().__init__(seed)
        self.first = True

    def randn(self, width):
        direction = super().randn(width)
        if self.first:
            direction[:] = np.arange(width) % 4
            self.first = False
        return direction


def random_state(spec):
    rng_type = TiedProjectionRandomState if spec.get("tie") else np.random.RandomState
    rng = rng_type(spec["seed"])
    if spec.get("warm_gaussian"):
        rng.standard_normal()
    return rng


def bits(matrix):
    assert matrix.dtype == np.float32
    return np.ascontiguousarray(matrix).view(np.uint32).ravel().tolist()


def observe(spec, *, fit_transform):
    rng = random_state(spec)
    options = spec["options"].copy()
    if not isinstance(options["init"], str):
        options["init"] = np.asarray(options["init"], dtype=np.float32)
    data = np.asarray(spec["data"], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = UMAP(**options, random_state=rng, n_jobs=1)
        if fit_transform:
            result = estimator.fit_transform(data)
            np.testing.assert_array_equal(result, estimator.embedding_)
        else:
            assert estimator.fit(data) is estimator
    graph = estimator.graph_
    _, _, cursor, has_gaussian, gaussian = rng.get_state()
    return {
        "indptr": graph.indptr.tolist(),
        "indices": graph.indices.tolist(),
        "weights_bits": bits(graph.data),
        "embedding_bits": bits(estimator.embedding_),
        "history_bits": [bits(x) for x in estimator.embedding_list_]
        if hasattr(estimator, "embedding_list_") else None,
        "rng": {
            "cursor": cursor,
            "gaussian_bits": int(np.float64(gaussian).view(np.uint64)) if has_gaussian else None,
            "next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
        },
    }


def make_fixture():
    cases = []
    heldout = np.random.RandomState(9501)
    heldout_data = heldout.normal(size=(29, 5)).astype(np.float32).tolist()
    heldout_init = heldout.uniform(-3, 3, (29, 9)).astype(np.float32).tolist()

    def add(name, optimizer, *, data=FEATURES, seed=42, warm=False, tie=False, **options):
        spec = {
            "name": name,
            "seed": seed,
            "data": data,
            "options": {
                "optimizer": optimizer, "init": "random", "compatibility_layout": False,
                "n_neighbors": 4, "n_epochs": [0, 1, 2, 3, 4, 5],
            } | options,
        }
        if warm:
            spec["warm_gaussian"] = True
        if tie:
            spec["tie"] = True
        observed = observe(spec, fit_transform=False)
        assert observe(spec, fit_transform=True) == observed, name
        cases.append(spec | observed)

    for optimizer in ("momentum", "adam"):
        add(f"{optimizer}-random-default", optimizer, seed=217)
        add(f"{optimizer}-attraction", optimizer, seed=42, negative_sample_rate=0)
        add(f"{optimizer}-explicit-signed", optimizer, seed=19, init=EXPLICIT,
            n_epochs=[7, 2, 0, 2, 5], n_neighbors=3, learning_rate=0.32)
        add(f"{optimizer}-late-negatives", optimizer, seed=7, init=EXPLICIT,
            negative_sample_rate=1, exclude_graph_neighbors=True, init_rescale="per_axis",
            n_epochs=[0, 1, 2, 3, 4, 6, 8, 9], a=1.7, b=0.9)
        add(f"{optimizer}-weak-leaf", optimizer, seed=77, data=ring(7, weak_leaf=True),
            init=EXPLICIT[:7], init_rescale="graph_edge", n_epochs=[0, 1, 5, 10, 11],
            n_neighbors=3, metric="precomputed", set_op_mix_ratio=0.01,
            local_connectivity=2.0, exclude_graph_neighbors=True)
        add(f"{optimizer}-isolated", optimizer, seed=77, data=ring(7, weak_leaf=True),
            init=EXPLICIT[:7], init_rescale="graph_edge", n_epochs=[0, 1, 5, 10, 11],
            n_neighbors=3, metric="precomputed", disconnection_distance=3.0,
            local_connectivity=2.0, exclude_graph_neighbors=True)
        add(f"{optimizer}-weak-median", optimizer, seed=61, data=ring(7, weak_leaf=True),
            init=[[-3.0], [-2.0], [-1.0], [0.0], [5.0], [10.0], [96.0]],
            n_components=1, init_rescale="graph_edge", n_epochs=[0, 1, 10, 11],
            n_neighbors=3, metric="precomputed", set_op_mix_ratio=0.01, local_connectivity=2.0,
            negative_sample_rate=0)
        add(f"{optimizer}-projection", optimizer, seed=507, warm=True,
            negative_selection_range=3, n_components=3, n_epochs=[0, 1, 3, 4, 5, 6])
        add(f"{optimizer}-projection-ties", optimizer, data=ring(24), seed=42, tie=True,
            init=np.eye(24, dtype=np.float32).tolist(), negative_selection_range=1,
            n_components=24, n_neighbors=3, n_epochs=[2, 1, 0], metric="precomputed",
            local_connectivity=2.0, init_rescale="graph_edge", a=1.0, b=1.0,
            learning_rate=0.35)
        add(f"{optimizer}-signed-zero", optimizer, data=ring(4), seed=102,
            init=[[-2.0, -0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]],
            metric="precomputed", n_neighbors=3, local_connectivity=2.0, a=1.0, b=1.0,
            negative_sample_rate=1, repulsion_strength=0.6, init_scale_factor=0.71)
        add(f"{optimizer}-default-budget", optimizer, seed=89, n_epochs=None)
        add(f"{optimizer}-wide", optimizer, seed=112, n_components=14,
            n_epochs=[0, 1, 2, 5, 13], learning_rate=0.41, repulsion_strength=0.4)
        add(f"{optimizer}-wide-prefix-remainder", optimizer, seed=127, n_components=37,
            n_epochs=[0, 1, 3, 5, 8], a=1.7, b=0.9, repulsion_strength=0.7)
        add(f"{optimizer}-zero-repulsion-scale", optimizer, seed=207,
            n_epochs=[0, 1, 3, 5], repulsion_strength=0.0, init_rescale="per_axis")
        add(f"{optimizer}-heldout-graph-scale", optimizer, seed=271, data=heldout_data,
            init=heldout_init, n_components=9, n_epochs=[0, 1, 9, 17], init_rescale="graph_edge",
            n_neighbors=5, repulsion_strength=0.55, exclude_graph_neighbors=True,
            negative_selection_range=23)
    add("default-modern-adam", None, seed=91, negative_sample_rate=1.5, n_epochs=9)
    add("momentum-one-epoch", "momentum", seed=17, n_epochs=1)
    add("adam-duplicates-cached", "adam", seed=41, init=[EXPLICIT[0], *EXPLICIT[0:7]],
        warm=True, init_rescale="per_axis", negative_selection_range=1, n_epochs=[0, 1, 2, 5])
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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = make_fixture()
    cases = fixture.pop("cases")
    header = json.dumps(fixture, indent=2)[:-2]
    encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in cases)
    args.output.write_text(f'{header},\n  "cases": [\n{encoded}\n  ]\n}}\n', encoding="utf-8", newline="\n")
