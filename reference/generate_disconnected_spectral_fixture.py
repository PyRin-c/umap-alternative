"""Capture disconnected layouts from the installed public UMAP estimator."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import warnings

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.special import ndtri
from umap import UMAP

from generate_spectral_handoff_fixture import graph_bits


def pairs(count, *, center_seed=38, spread=2):
    rng = np.random.RandomState(center_seed)
    centers = rng.uniform(-spread, spread, (count, 2))
    return (centers[:, None] + rng.normal(0, 0.001, (count, 2, 2))).reshape(-1, 2).astype("f")


def quantile_groups(sizes, centers, seed):
    rng = np.random.RandomState(seed)
    groups = []
    for size, center in zip(sizes, centers):
        quantiles = (np.arange(size) + rng.uniform(0.2, 0.8, size)) / size
        offsets = ndtri(quantiles) * 0.025
        groups.append(center + np.column_stack([offsets, rng.permutation(offsets)]))
    order = rng.permutation(sum(sizes))
    labels = np.repeat(np.arange(len(sizes)), sizes)[order]
    return np.concatenate(groups).astype("f")[order], labels


def fit_parameters(points, labels, neighbors):
    if labels is None:
        return {}
    distances = cdist(points, points).astype(np.float32)
    distances[labels[:, None] != labels] = np.inf
    indices = np.argsort(distances, axis=1)[:, :neighbors].astype(np.int32)
    distances = np.take_along_axis(distances, indices, axis=1)
    indices[~np.isfinite(distances)] = -1
    return {"precomputed_knn": (indices, distances, None), "force_approximation_algorithm": True}


def snapshot(model, points, labels, params, seed, method):
    rng = np.random.RandomState(seed)
    model.set_params(**params, **fit_parameters(points, labels, params["n_neighbors"]), random_state=rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = getattr(model, method)(points.copy())
    assert fitted is (model if method == "fit" else model.embedding_)
    assert model.embedding_.dtype == np.float32
    positive = model.graph_.copy()
    positive.eliminate_zeros()
    count, component = connected_components(positive, directed=False)
    _, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
    return {
        "graph": graph_bits(model.graph_),
        "component_sizes": np.bincount(component, minlength=count).tolist(),
        "bits": model.embedding_.view(np.uint32).ravel().tolist(),
        "warnings": [str(w.message).splitlines()[0] for w in caught],
        "post_fit_rng": {
            "cursor": cursor,
            "has_gaussian": has_gaussian,
            "cached_gaussian": cached_gaussian,
            "next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
        },
    }


def generate():
    mixed = quantile_groups([1, 2, 3, 4, 5], [(-1, 0), (0, 1), (2, 0), (1, -1), (-1, -2)], 7)
    holdout = quantile_groups(
        [1, 6, 7, 8], [(0.3, 0.8), (0.1, -0.6), (-0.8, -0.3), (1.2, 0.6)], 87
    )
    meta_holdout = quantile_groups(
        [1, 3, 4, 5, 7, 8],
        [(-0.8, -0.4), (0.1, 1.2), (1.5, -0.6), (0.2, 0.6), (-1.3, 0.1), (1.1, 0.5)],
        73,
    )
    inputs = {
        "pairs": (pairs(5), None),
        "manhattan": (pairs(6, center_seed=18, spread=1.2), None),
        "chebyshev": (pairs(7, center_seed=72, spread=1.3), None),
        "wide": (pairs(5, spread=80), None),
        "mixed": mixed,
        "holdout": holdout,
        "meta-holdout": meta_holdout,
        "bridge": (np.random.RandomState(72).normal(size=(20, 2)).astype("f"), None),
    }
    base = dict(compatibility_layout=True, init="spectral", n_jobs=1, a=1.7, b=0.9)
    specs = [
        ("pairs", 2, 2, 0, 0, {}),
        ("pairs", 2, 2, 1, 0, {}),
        ("pairs", 2, 2, 2, 0, {}),
        ("pairs", 2, 2, 0, 7, {}),
        ("pairs", 2, 3, 0, 7, {}),
        ("manhattan", 2, 2, 0, 123, {"metric": "manhattan"}),
        ("chebyshev", 2, 2, 0, 24, {"metric": "chebyshev"}),
        ("chebyshev", 2, 2, 0, 24, {}),
        ("wide", 2, 2, 0, 0, {}),
        ("mixed", 3, 2, 0, 7, {}),
        ("mixed", 3, 3, 0, 2**32 - 1, {}),
        ("mixed", 3, 3, 1, 2**32 - 1, {}),
        ("mixed", 3, 3, 2, 2**32 - 1, {}),
        ("holdout", 3, 4, 0, 42, {}),
        ("meta-holdout", 3, 2, 0, 41, {}),
        ("meta-holdout", 3, 3, 0, 41, {}),
        ("bridge", 4, 2, 10, 0, {"learning_rate": 0}),
        ("bridge", 4, 2, 11, 0, {"learning_rate": 0}),
        ("bridge", 4, 2, 11, 0, {}),
    ]
    cases = {}
    graphs = {}
    for name, neighbors, dims, epochs, seed, options in specs:
        points, labels = inputs[name]
        params = dict(n_neighbors=neighbors, n_components=dims, n_epochs=epochs) | options
        key = f"{name}-d{dims}-e{epochs}-s{seed}"
        if options:
            key += "-" + "-".join(str(value) for value in options.values())
        model = UMAP(**base)
        first = snapshot(model, points, labels, params, seed, "fit")
        second = snapshot(model, points, labels, params, seed, "fit_transform")
        graph = first.pop("graph")
        assert graph == second.pop("graph")
        assert first["post_fit_rng"] == second["post_fit_rng"]
        assert first["component_sizes"] == second["component_sizes"]
        graph_key = next((key for key, value in graphs.items() if value == graph), key)
        graphs[graph_key] = graph
        cases[key] = {
            "input": name,
            "seed": seed,
            "params": params,
            "graph": graph_key,
            **first,
            "repeated_bits": second["bits"] if first["bits"] != second["bits"] else None,
        }
    return {
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "umap-learn")},
        },
        "settings": base,
        "inputs": {name: points.tolist() for name, (points, _) in inputs.items()},
        "explicit_labels": {name: labels.tolist() for name, (_, labels) in inputs.items() if labels is not None},
        "graphs": graphs,
        "cases": cases,
    }


if __name__ == "__main__":
    sys.tracebacklimit = 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sections = generate()
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("{\n")
        for index, (name, content) in enumerate(sections.items()):
            if index:
                stream.write(",\n")
            stream.write(f"  {json.dumps(name)}: {{\n")
            stream.write(",\n".join(f"    {json.dumps(key)}: {json.dumps(value)}" for key, value in content.items()))
            stream.write("\n  }")
        stream.write("\n}\n")
