"""Observe UMAP 0.6.0 initialization through public layout APIs."""

import argparse
import json
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from umap import UMAP
from umap.label_prop import label_propagation_init, recursive_init
from umap.spectral import spectral_layout
from umap.umap_ import simplicial_set_embedding


METHODS = ("graph_edge", "per_axis", "isotropic")
PAIRS = np.array([[0, 0], [1, 0], [0, 6], [3, 10]], dtype=np.float32)


def bits(values):
    assert values.dtype == np.float32
    return np.ascontiguousarray(values).view(np.uint32).ravel().tolist()


def graph(n, edges):
    edges = [(i, j, w) for i, j, w in edges] + [(j, i, w) for i, j, w in edges if i != j]
    source, target, weight = zip(*edges) if edges else ([], [], [])
    return csr_matrix((np.asarray(weight, dtype=np.float32), (source, target)), shape=(n, n))


def graph_record(value):
    return {
        "n_rows": value.shape[0],
        "indptr": value.indptr.tolist(),
        "indices": value.indices.tolist(),
        "weights_bits": bits(value.data),
    }


def rng_record(rng):
    _, _, position, has_gaussian, gaussian = rng.get_state()
    replica = np.random.RandomState()
    replica.set_state(rng.get_state())
    return {
        "position": position,
        "gaussian": gaussian if has_gaussian else None,
        "next_words": replica.randint(2**32, size=4, dtype=np.uint32).tolist(),
    }


def direct(data, graph, initial, rng, *, n_components, method, factor, epochs=0, **kwds):
    return simplicial_set_embedding(
        data, graph.copy(), n_components, 0.0, 1.0, 1.0, 0.0, 0, epochs, initial,
        rng, "euclidean", {}, False, {}, False,
        compatibility_layout=False, optimizer="momentum", init_rescale=method,
        init_scale_factor=factor, **kwds,
    )


def estimator(data, initial, rng, *, n_components=2, method="graph_edge", factor=1.0, **kwds):
    options = dict(
        n_epochs=0, n_neighbors=2, n_jobs=1, negative_sample_rate=0,
        compatibility_layout=False, optimizer="momentum", a=1.0, b=1.0,
    )
    return UMAP(
        init=initial, n_components=n_components, random_state=rng, init_rescale=method,
        init_scale_factor=factor, **(options | kwds),
    ).fit(data)


def make_fixture():
    graphs = {}
    cases = []

    def record(name, data, initial, method, factor, *, seed=42, n_components=2, graph_name=None,
               warmup_normal=False):
        rng = np.random.RandomState(seed)
        if warmup_normal:
            rng.standard_normal()
        rng_start = rng_record(rng)
        if graph_name is None:
            fitted = estimator(data, initial, rng, n_components=n_components, method=method, factor=factor)
            value = fitted.embedding_
            graphs[f"fit_{len(data)}"] = fitted.graph_
            graph_name = f"fit_{len(data)}"
        else:
            value, _ = direct(data, graphs[graph_name], initial, rng,
                              n_components=n_components, method=method, factor=factor)
        return {
            "name": name, "graph": graph_name, "method": method, "factor": factor,
            "n_components": n_components, "seed": seed,
            "init": initial if isinstance(initial, str) else "explicit",
            "initial_bits": [] if isinstance(initial, str) else bits(initial),
            "embedding_bits": bits(value), "rng": rng_record(rng),
            **({"rng_start": rng_start} if warmup_normal else {}),
        }

    for method in METHODS:
        for factor in (1.0, 0.73):
            cases.append(record(f"pair-{method}-{factor}", PAIRS, PAIRS, method, factor))
        cases.append(record(f"flat-{method}", PAIRS, np.ones_like(PAIRS), method, 2.0))
    cases.append(record("pair-translation", PAIRS, PAIRS + np.float32([128, -17]), "graph_edge", 2.0))
    duplicates = np.array([[0, 0], [0, 0], [0, 6], [3, 10]], "f")
    cases.append(record("pair-duplicates", PAIRS, duplicates, "graph_edge", 1.0))
    cases.append(record("duplicates-gaussian-cache", PAIRS, duplicates, "graph_edge", 1.0,
                        seed=1178, warmup_normal=True))
    for factor in (1e-40, 1e39):
        cases.append(record(f"positive-{factor}", PAIRS, PAIRS, "graph_edge", factor))

    for n, d in ((9, 1), (13, 2), (17, 4), (23, 5)):
        draws = np.random.RandomState(7234 + n)
        data = draws.uniform(-100, 100, (n, 3)).astype("f")
        initial = (draws.normal(size=(n, d)) * np.power(2.0, np.arange(d))).astype("f")
        for method in METHODS:
            cases.append(record(f"holdout-{n}x{d}-{method}", data, initial, method, 0.741253,
                                n_components=d))
            cases.append(record(f"random-{n}x{d}-{method}", data, "random", method, 1.2,
                                n_components=d, seed=95 + n))

    data = np.arange(22, dtype="f")[:, None]
    initial = np.column_stack((data[:, 0], np.zeros(22, "f"), np.arange(22, dtype="f") * 1e-12))
    initial[-1, 1] = 1.0
    for method in METHODS:
        cases.append(record(f"zero-robust-span-{method}", data, initial, method, 1.2, n_components=3))

    graphs.update({
        "short-heavy": graph(4, [(0, 1, 1), (2, 3, 0.99)]),
        "long-heavy": graph(4, [(0, 1, 0.99), (2, 3, 1)]),
        "long-one-step-heavier": graph(4, [(0, 1, np.float32(0.99999994)), (2, 3, 1)]),
        "loops": graph(4, [(0, 0, 1), (2, 3, 0.1)]),
        "zero-weights": graph(4, [(0, 1, 0), (2, 3, 0)]),
        "spectral": graph(9, [(i, i + 1, 0.5 + i / 20) for i in range(8)]
                          + [(0, 3, 0.8), (2, 4, 0.9), (4, 7, 0.7), (5, 8, 0.9)]),
        "coarse": graph(6, [(0, 1, 1), (2, 3, 1), (4, 5, 1), (1, 2, 0.3), (3, 4, 0.4)]),
    })
    for name in ("short-heavy", "long-heavy", "long-one-step-heavier", "loops", "zero-weights"):
        cases.append(record(name, PAIRS, PAIRS, "graph_edge", 2.0, graph_name=name))
    for gap in (1e-30, 1e-22, 1e-8):
        initial = np.array([[0, 0], [gap, 0], [0, 100], [gap, 100]], "f")
        cases.append(record(f"tiny-edge-{gap}", PAIRS, initial, "graph_edge", 2.0,
                            graph_name="short-heavy"))
    for method in ("per_axis", "isotropic"):
        cases.append(record(f"direct-{method}", PAIRS, PAIRS, method, 2.0, graph_name="short-heavy"))

    # The zero-epoch graph is unchanged even when short weights would be below
    # the positive sampler cutoff for the requested epoch budget.
    sparse_weights = graph(14, [(0, 13, 1), (1, 12, 1)] + [(i, i + 1, 0.4) for i in range(2, 11)])
    line = np.arange(14, dtype="f")[:, None]
    cutoffs = {}
    for epochs in (0, 2, 100):
        y, checkpoints = direct(line, sparse_weights, line, np.random.RandomState(11),
                                n_components=1, method="graph_edge", factor=1.0, epochs=[0, epochs])
        cutoffs[str(epochs)] = {"epoch_zero_bits": bits(checkpoints["embedding_list"][0]), "final_bits": bits(y)}

    spectral = []
    data = np.array([[0.2, 2, 1], [0.7, 0, 2], [1, 3, 1], [3, 1, 0], [4, 0, 3],
                     [4, 3, 2], [6, 3, 3], [7, 1, 8], [8, 0, 2]], "f")
    for seed, d in ((42, 2), (43, 2), (113, 1), (6, 3)):
        new = record(f"spectral-{seed}x{d}", data, "spectral", "graph_edge", 1.0,
                     seed=seed, n_components=d, graph_name="spectral")
        rng = np.random.RandomState(seed)
        vectors = spectral_layout(data, graphs["spectral"], d, rng, compatibility_layout=False)
        new["initial_bits"] = bits(vectors)
        new["spectral_rng"] = rng_record(rng)
        again = record(new["name"], data, "spectral", "graph_edge", 1.0,
                       seed=seed, n_components=d, graph_name="spectral")
        new["repeat_bits_equal"] = new["embedding_bits"] == again["embedding_bits"]
        spectral.append(new)

    heldout = np.random.RandomState(710).uniform(-8, 11, (25, 4)).astype("f")
    graphs["recursive_holdout"] = graph(
        len(heldout),
        [(i, i + 1, 0.37 + (i % 9) / 20) for i in range(24)]
        + [(i, i + 3, 0.25 + (i % 5) / 16) for i in range(22)],
    )
    recursive = []
    for key, seed, threshold, ratio, weights, epochs, coarsen_on_layout in (
        ("spectral", 42, 9, 4, "transition", 0, True),
        ("spectral", 43, 9, 4, "transition", 0, True),
        ("spectral", 42, 4, 4, "transition", 0, True),
        ("spectral", 43, 4, 4, "transition", 0, True),
        ("spectral", 42, 3, 3, "fuzzy_union", 0, True),
        ("spectral", 42, 3, 2, "transition", 4, True),
        ("recursive_holdout", 1, 3, 2, "transition", 0, True),
        ("recursive_holdout", 1, 3, 2, "transition", 0, False),
        ("recursive_holdout", 4, 4, 4, "fuzzy_union", 0, True),
    ):
        features = data if key == "spectral" else heldout
        graph_ = graphs[key]
        coarse_opts = dict(coarse_layout_weights=weights, coarsen_on_layout_weights=coarsen_on_layout)
        kwargs = {"base_init_threshold": threshold, "n_embedding_epochs": epochs}
        rng = np.random.RandomState(seed)
        raw = recursive_init(
            graph_, features, 1.0, 1.0, random_state=rng, recursive_parallel=False,
            coarsening_ratio=ratio, **coarse_opts, **kwargs,
        )
        raw_rng = rng_record(rng)
        rng = np.random.RandomState(seed)
        coords, _ = direct(features, graph_, "recursive", rng, n_components=2,
                           method="graph_edge", factor=1.0, recursive_coarsening_ratio=ratio,
                           recursive_init_kwds=kwargs, **coarse_opts)
        rng_auto = np.random.RandomState(seed)
        auto, _ = direct(features, graph_, "auto", rng_auto, n_components=2,
                         method="graph_edge", factor=1.0, recursive_coarsening_ratio=ratio,
                         recursive_init_kwds=kwargs, **coarse_opts)
        recursive.append({
            "graph": key, "seed": seed, "threshold": threshold, "ratio": ratio, "weights": weights,
            "epochs": epochs, "coarsen_on_layout": coarsen_on_layout,
            "raw_bits": bits(raw), "raw_rng": raw_rng,
            "embedding_bits": bits(coords), "rng": rng_record(rng),
            "auto_matches": bits(coords) == bits(auto) and rng_record(rng) == rng_record(rng_auto),
        })

    anchor = np.array([[0, 5], [1, 7], [4, 0], [5, 1], [8, 7], [11, 8]], "f")
    coarse = []
    for threshold, depth in ((6, 1), (6, 2), (3, 1)):
        rng = np.random.RandomState(42)
        value = label_propagation_init(
            graphs["coarse"], anchor, np.ones(len(anchor), dtype=bool), 1.0, 1.0,
            n_components=2, approx_n_parts=3, n_embedding_epochs=0, depth=depth,
            base_init_threshold=threshold, random_state=rng, recursive_parallel=False,
        )
        coarse.append({"threshold": threshold, "depth": depth, "embedding_bits": bits(value),
                       "rng": rng_record(rng)})

    invalid = []
    for name, init, kwds in (
        ("invalid-mode", PAIRS, {"method": "GRAPH_EDGE"}),
        ("null-mode", PAIRS, {"method": None}),
        ("zero-scale", PAIRS, {"factor": 0.0}),
        ("negative-scale", PAIRS, {"factor": -1.0}),
        ("nan-scale", PAIRS, {"factor": np.nan}),
        ("infinite-scale", PAIRS, {"factor": np.inf}),
        ("boolean-scale", PAIRS, {"factor": True}),
        ("string-scale", PAIRS, {"factor": "2"}),
        ("one-dimensional", PAIRS[:, 0], {}),
        ("few-columns", PAIRS[:, :1], {}),
        ("few-rows", PAIRS[:2], {}),
        ("nan-coordinate", np.full_like(PAIRS, np.nan), {}),
        ("infinite-coordinate", np.full_like(PAIRS, np.inf), {}),
        ("invalid-ratio", "recursive", {"recursive_coarsening_ratio": 5}),
        ("invalid-weights", "recursive", {"coarse_layout_weights": "none"}),
        ("invalid-recursive-key", "recursive", {"recursive_init_kwds": {"unknown": 0}}),
    ):
        rng = np.random.RandomState(42)
        try:
            estimator(PAIRS, init, rng, **kwds)
        except (TypeError, ValueError, IndexError) as exc:
            invalid.append({"name": name, "exception": type(exc).__name__, "rng": rng_record(rng)})
        else:
            raise AssertionError(f"{name} unexpectedly accepted")
    try:
        direct(PAIRS, graph(4, []), PAIRS, np.random.RandomState(42), n_components=2,
               method="graph_edge", factor=1.0)
    except ValueError as exc:
        invalid.append({"name": "empty-graph", "exception": type(exc).__name__})
    else:
        raise AssertionError("empty graph unexpectedly accepted")

    return {
        "environment": {
            "platform": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba", "scikit-learn")},
            "threads": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"
            )},
        },
        "graphs": {name: graph_record(value) for name, value in graphs.items()},
        "cases": cases, "epoch_cutoffs": cutoffs, "spectral": spectral,
        "recursive_data": {"spectral": data.tolist(), "recursive_holdout": heldout.tolist()},
        "recursive": recursive,
        "coarse_anchor": anchor.tolist(), "coarse": coarse, "invalid": invalid,
    }


if __name__ == "__main__":
    sys.tracebacklimit = 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixture = make_fixture()
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
