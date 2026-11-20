"""Observe inactive coarse edges and adjacent recursive budgets in UMAP 0.6.0."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from umap.label_prop import label_propagation_init

from generate_modern_recursive_fit_fixture import TracedRandomState, bits, observe, rng_checkpoint, star_distances
from generate_modern_recursive_hub_fixture import graph_record, symmetric_graph


def stage(name, graph, *, threshold, budget, rate, selection):
    seed = 481
    rng = TracedRandomState(seed)
    anchor = np.random.RandomState(1107).uniform(-7, 5, (graph.shape[0], 2)).astype(np.float32)
    original_graph = graph_record(graph)
    try:
        layout = label_propagation_init(
            graph, anchor.copy(), np.ones(len(anchor), dtype=bool), 1., 1.,
            n_components=2, approx_n_parts=9, n_embedding_epochs=budget, depth=1,
            base_init_threshold=threshold, random_state=rng, recursive_parallel=False, coarsening_ratio=3,
            recursive_negative_sample_rate=rate, recursive_negative_selection_range_mode=selection,
        )
    except Exception as error:
        raise SystemExit(f"{name}: {type(error).__name__}: {error}") from None
    assert graph_record(graph) == original_graph
    return {
        "name": name, "graph": original_graph, "anchor": anchor.tolist(), "threshold": threshold,
        "budget": budget, "rate": rate, "selection": selection, "seed": seed, "layout_bits": bits(layout),
        "rng_calls": rng.calls.copy(), "rng": rng_checkpoint(rng) | {
            "next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
        },
    }


def generate():
    stages, fits = [], []

    def add_fit(name, data, *, seed, threshold, budget, rate, bound, **options):
        spec = {
            "name": name, "data": data, "seed": seed, "warm_gaussian": False, "atol": bound,
            "options": {
                "init": "recursive", "compatibility_layout": False, "optimizer": "adam", "n_epochs": [0],
                "n_neighbors": 10, "a": 1., "b": 1., "recursive_coarsening_ratio": 2,
                "recursive_init_kwds": {"base_init_threshold": threshold, "n_embedding_epochs": budget,
                                        "recursive_negative_sample_rate": rate},
            } | options,
        }
        result = observe(spec, "fit")
        assert observe(spec, "fit_transform") == result, name
        fits.append(spec | result)

    for tied in (True, False):
        bridge = [(0, 3), (6, 3), (9, 12), (15, 12), (18, 21), (24, 21), (3, 12), (12, 21)]
        spokes = [(hub, leaf, 1. if tied or leaf == hub + 1 else 1. - .0004 * (hub // 3 + 1))
                  for hub in range(0, 27, 3) for leaf in (hub + 1, hub + 2)]
        edges = spokes + [(i, j, 1.) for i, j in bridge]
        graph = symmetric_graph(27, edges + [(1, 10, .0001)])
        kind = "tied" if tied else "untied"
        for threshold, budget, rate, selection in (
            (9, 3, 1, "scaled_coarse"), (9, 4, 0, "scaled_coarse"), (9, 4, 1, "scaled_coarse"),
            (9, 4, 1, "coarse_n"), (3, 4, 1, "scaled_coarse"), (3, 6, 5, "coarse_n"),
        ):
            stages.append(stage(f"{kind}-{threshold}-{budget}-{rate}-{selection}", graph.copy(),
                                threshold=threshold, budget=budget, rate=rate, selection=selection))

    dense = {"seed": 231, "shape": [28, 4]}
    for threshold, budget, rate, bound in (
        (14, 3, 1, 5e-6), (14, 4, 0, 5e-6), (14, 4, 1, 5e-6), (14, 5, 1, 5e-6),
        (7, 2, 1, 5e-6), (7, 3, 0, 5e-6), (7, 3, 1, 5e-6), (7, 4, 1, 5e-6),
        (2, 2, 1, 5e-6), (2, 4, 1, 5e-6),
    ):
        add_fit(f"dense-{threshold}-{budget}-{rate}", dense, seed=71, threshold=threshold, budget=budget,
                rate=rate, bound=bound)

    stars = star_distances([3] * 9, 620, chain=True, hierarchy=True)
    for threshold, budget, bound in ((9, 10, 5e-6), (9, 11, 5e-6), (9, 12, 5e-6),
                                      (3, 10, 1e-5), (3, 12, 5e-6)):
        add_fit(f"star-attraction-{threshold}-{budget}", stars, seed=42, threshold=threshold, budget=budget,
                rate=0, bound=bound, n_neighbors=2, metric="precomputed", recursive_coarsening_ratio=3)
    return {"stages": stages, "fits": fits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = generate()
    entries = [f'  "{key}": [\n' + ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in values)
               + "\n  ]" for key, values in cases.items()]
    args.output.write_text("{\n" + ",\n".join(entries) + "\n}\n", encoding="utf-8", newline="\n")
