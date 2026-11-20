"""Observe recursive contraction through public UMAP 0.6.0 calls."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from umap.label_prop import label_propagation_init

from generate_modern_recursive_fit_fixture import bits, make_fixture, observe, rng_checkpoint


def _csr_record(graph):
    return {
        "indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(),
        "weights_bits": bits(graph.data),
    }


def _graph(n_rows, edges):
    source, target, weights = zip(*edges)
    return csr_matrix((np.asarray(weights, dtype=np.float32), (source, target)),
                      shape=(n_rows, n_rows)).maximum(
        csr_matrix((np.asarray(weights, dtype=np.float32), (target, source)), shape=(n_rows, n_rows)),
    )


def _placements():
    observations = []

    def add(name, graph, seed=42, order=None):
        rng = np.random.RandomState(seed)
        anchor = np.random.RandomState(993).uniform(-4, 8, (graph.shape[0], 2)).astype(np.float32)
        if order is not None:
            graph = graph[order][:, order]
            graph.sort_indices()
            anchor = anchor[order]
        try:
            layout = label_propagation_init(
                graph, anchor.copy(), np.ones(len(anchor), dtype=bool), 1.0, 1.0,
                n_components=2, approx_n_parts=3, n_embedding_epochs=0, depth=1,
                base_init_threshold=len(anchor) - 1, random_state=rng, recursive_parallel=False,
            )
        except Exception as error:
            raise SystemExit(f"{name}: {type(error).__name__}: {error}") from None
        observations.append({
            "name": name, "graph": _csr_record(graph), "anchor": anchor.tolist(), "seed": seed,
            "layout_bits": bits(layout), "rng": rng_checkpoint(rng) | {
                "next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
            },
        })

    base = [(0, 3, .9), (3, 5, .94), (0, 5, .92), (0, 4, .8), (5, 6, .1), (0, 1, .1)]
    half = np.float32(.5)
    for name, weight in (("below", np.nextafter(half, np.float32(0))), ("equal", half),
                         ("above", np.nextafter(half, np.float32(1)))):
        graph = _graph(7, base + [(1, 2, weight), (1, 3, weight), (2, 3, 1.)])
        add(f"support-{name}-one", graph)
    ties = _graph(7, [(0, 3, 1.), (3, 5, .9), (0, 5, .9), (0, 2, .25), (3, 4, .15),
                      (5, 6, .4), (0, 1, 1.), (1, 3, 1.)])
    add("equal-support-distinct-degrees", ties)
    add("equal-support-permuted", ties, order=[3, 1, 2, 0, 4, 5, 6])
    cycle = _graph(7, [(row, (row + 1) % 7, 1.) for row in range(7)])
    add("equal-degrees", cycle)
    add("equal-degrees-permuted", cycle, order=[4, 1, 3, 6, 5, 0, 2])
    missing = _graph(8, [(0, 1, .6), (1, 2, .75), (2, 3, .8), (3, 4, .7), (4, 5, .35)])
    for seed in (1, 7, 44):
        add(f"two-isolated-seed-{seed}", missing, seed=seed)
    add("two-isolated-permuted", missing, seed=14, order=[7, 2, 5, 0, 3, 4, 6, 1])
    return observations


def generate():
    previous = make_fixture()
    archived = []
    fits = []
    for probe in previous["probes"]:
        spec = {key: probe[key] for key in ("name", "options", "warm_gaussian", "seed", "data", "levels")}
        record = observe(spec, "fit")
        assert record == observe(spec, "fit_transform"), spec["name"]
        archived.append(spec | record)

    def add(name, data, *, seed=113, warm=False, **options):
        defaults = dict(init="recursive", compatibility_layout=False, optimizer="adam",
                        n_epochs=[0, 1, 3, 6], n_neighbors=4, a=1., b=1.)
        spec = {"name": name, "data": data, "seed": seed, "warm_gaussian": warm, "options": defaults | options}
        record = observe(spec, "fit")
        assert record == observe(spec, "fit_transform"), name
        fits.append(spec | record)

    for n_rows in (5, 6, 7):
        add(f"size-{n_rows}-threshold-six", {"seed": 601, "shape": [n_rows, 4]}, seed=773, warm=True,
            recursive_init_kwds={"base_init_threshold": 6, "n_embedding_epochs": 0})
    add("holdout-competing-hubs", {"seed": 2429, "shape": [8, 3]}, seed=947,
        n_neighbors=3, optimizer="momentum", negative_sample_rate=0,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    add("holdout-two-level-fuzzy", {"seed": 202, "shape": [18, 3]}, seed=407, warm=True,
        n_neighbors=5, recursive_coarsening_ratio=3, coarse_layout_weights="fuzzy_union",
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    add("holdout-three-level-transition", {"seed": 555, "shape": [21, 3]}, seed=257, warm=True,
        n_neighbors=5, recursive_coarsening_ratio=2, init_rescale="graph_edge", optimizer="momentum",
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 0})
    add("holdout-one-coarse-epoch", {"seed": 673, "shape": [14, 3]}, seed=755, warm=True,
        n_neighbors=5, recursive_coarsening_ratio=2, negative_selection_range=3,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 1})
    add("holdout-two-coarse-epochs", {"seed": 137, "shape": [11, 4]}, seed=631, warm=True,
        n_neighbors=4, a=1.7, b=.9, n_components=3, optimizer="momentum",
        recursive_init_kwds={"base_init_threshold": 5, "n_embedding_epochs": 2})
    add("holdout-three-coarse-epochs", {"seed": 673, "shape": [14, 3]}, seed=98,
        n_neighbors=5, recursive_coarsening_ratio=2, negative_selection_range=3,
        recursive_init_kwds={"base_init_threshold": 3, "n_embedding_epochs": 3})
    disconnected = [[0., 0.], [.1, .08], [.3, .809], [4., .117], [4.2, -.07], [5.9, 0.], [6.2, .2], [8., 4.]]
    for i, order in enumerate(([0, 1, 2, 3, 4, 5, 6, 7], [5, 7, 1, 3, 2, 6, 0, 4])):
        add(f"cutoff-isolated-order-{i}", [disconnected[j] for j in order], seed=61 + i, warm=True,
            n_neighbors=3, disconnection_distance=.5, optimizer="momentum", init_rescale="graph_edge",
            recursive_init_kwds={"base_init_threshold": 2, "n_embedding_epochs": 0})
    return {"environment": previous["environment"], "placements": _placements(), "archived": archived, "fits": fits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixture = generate()
    header = json.dumps({"environment": fixture["environment"]}, indent=2)[:-2]
    sections = []
    for key in ("placements", "archived", "fits"):
        encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in fixture[key])
        sections.append(f'  "{key}": [\n{encoded}\n  ]')
    args.output.write_text(f"{header},\n" + ",\n".join(sections) + "\n}\n", encoding="utf-8", newline="\n")
