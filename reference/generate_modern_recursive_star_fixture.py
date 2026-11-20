"""Freeze public UMAP 0.6.0 star placement and the first Adam updates."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from umap.label_prop import label_propagation_init
from umap.umap_ import simplicial_set_embedding

from generate_modern_recursive_fit_fixture import bits, observe, rng_checkpoint, star_distances
from generate_modern_recursive_hub_fixture import graph_record, symmetric_graph


NINE_ROW_ANCHOR = np.array([
    [5.3962493, 3.4563084], [3.114322, 1.2392639], [3.1508932, 3.905185],
    [2.3621445, .71130395], [3.3326633, 5.87044], [4.8249154, 3.304372],
    [2.541829, 9.415429], [7.0257, 3.6945069], [7.7208004, 5.942602],
], dtype=np.float32)


def placement(graph, anchor, *, parts, depth, budget=0, mode="transition"):
    try:
        return label_propagation_init(
            graph.copy(), anchor.copy(), np.ones(len(anchor), dtype=bool), 1., 1.,
            n_components=anchor.shape[1], approx_n_parts=parts, n_embedding_epochs=budget,
            depth=depth, base_init_threshold=parts, random_state=np.random.RandomState(73),
            recursive_parallel=False, coarsening_ratio=3, coarse_layout_weights=mode,
            recursive_negative_sample_rate=0,
        )
    except Exception as error:
        raise SystemExit(type(error).__name__) from None


class FrozenJitter(np.random.RandomState):
    def normal(self, loc=0., scale=1., size=None):
        # Explicit initialization requests noise for coincident rows. Supply
        # zero noise so the optimizer sees the forced coordinate boundaries.
        return np.zeros(size, dtype=np.float64)


def optimizer_probe(name, graph, initial, *, budget, rate):
    seed = 844
    rng = FrozenJitter(seed)
    lower, upper = np.percentile(initial, (5, 95), axis=0)
    scale = np.sqrt(np.prod(upper - lower)) / (3. * np.log10(len(initial)) * np.log2(5.))
    recorded_graph = graph_record(graph)
    try:
        _, auxiliary = simplicial_set_embedding(
            initial, graph, 2, .5, 1., 1., 4., rate, list(range(budget + 1)), initial.copy(),
            rng, "euclidean", {}, False, {}, False, optimizer="adam", compatibility_layout=False,
            init_rescale="isotropic", init_scale_factor=float(scale), negative_selection_range=3,
            good_initialization=False,
        )
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    assert graph_record(graph) == recorded_graph
    return {
        "name": name, "graph": recorded_graph, "budget": budget, "rate": rate, "seed": seed,
        "history_bits": [bits(value) for value in auxiliary["embedding_list"]],
        "rng": rng_checkpoint(rng) | {"next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()},
    }


def graph_at_half(weight):
    spokes = [(0, 7, 1.), (1, 7, 1.), (2, 6, 1.), (3, 8, 1.), (4, 6, 1.), (5, 8, 1.)]
    return symmetric_graph(9, spokes + [(6, 8, weight), (7, 8, weight)])


def scalar_probes():
    cases = []
    for parts in (2, 3, 4, 5):
        for seed in (170, 171, 173):
            rng = np.random.RandomState(seed)
            hubs = np.arange(0, parts * 3, 3)
            anchor = np.repeat(rng.uniform(-2.5, 3.25, (parts, 1)).astype(np.float32), 3, axis=0)
            edges = [(hub, leaf, 1.) for hub in hubs for leaf in (hub + 1, hub + 2)]
            edges += [(i, j, rng.uniform(.1, .9)) for i, j in zip(hubs[:-1], hubs[1:], strict=True)]
            graph = symmetric_graph(len(anchor), edges)
            degree = np.asarray(graph.sum(axis=1)).ravel()
            ranked = hubs[np.argsort(degree[hubs], kind="stable")]
            labels = np.repeat(np.argsort(ranked // 3), 3)
            child = placement(csr_matrix(np.eye(parts, dtype=np.float32)), anchor[ranked], parts=parts, depth=2)
            layout = placement(graph, anchor, parts=parts, depth=1)
            cases.append({"name": f"scalar-{parts}-{seed}", "graph": graph_record(graph), "anchor": anchor.tolist(),
                          "labels": labels.tolist(), "child_bits": bits(child), "layout_bits": bits(layout)})
    return cases


def generate():
    replay, fits = [], []
    half = np.float32(.5)
    for name, weight, mode, budget, rate in (
        ("long-attraction", half, "transition", 11, 0),
        ("short-attraction", half, "transition", 10, 0),
        ("later-attraction", half, "transition", 12, 0),
        ("half-below", np.nextafter(half, np.float32(0)), "transition", 11, 0),
        ("half-above", np.nextafter(half, np.float32(1)), "transition", 11, 0),
        ("repulsion", half, "transition", 11, 1),
        ("fuzzy-repulsion", half, "fuzzy_union", 11, 1),
        ("anchor-nextafter", half, "transition", 11, 0),
    ):
        graph = graph_at_half(weight)
        anchor = NINE_ROW_ANCHOR.copy()
        if name == "anchor-nextafter":
            anchor[0, 0] = np.nextafter(anchor[0, 0], np.float32(np.inf))
        before = placement(graph, anchor, parts=3, depth=2, budget=int(budget * np.sqrt(2)), mode=mode)
        replay.append(optimizer_probe(name, graph.copy(), before, budget=budget, rate=rate))

    for seed, source, threshold, budget, rate, mode in (
        (44, 868, 9, 11, 0, "transition"), (44, 868, 3, 10, 0, "transition"),
        (44, 868, 3, 11, 0, "transition"), (44, 868, 3, 12, 0, "transition"),
        (61, 621, 3, 11, 1, "transition"), (61, 621, 3, 11, 0, "fuzzy_union"),
        (61, 621, 3, 11, 1, "fuzzy_union"),
    ):
        spec = {
            "name": f"star-{seed}-{source}-{threshold}-{budget}-{rate}-{mode}", "seed": seed,
            "warm_gaussian": False, "data": star_distances([3] * 9, source, chain=True, hierarchy=True),
            "options": {
                "init": "recursive", "compatibility_layout": False, "optimizer": "adam", "n_epochs": [0],
                "n_neighbors": 2, "metric": "precomputed", "a": 1., "b": 1., "recursive_coarsening_ratio": 3,
                "coarse_layout_weights": mode,
                "recursive_init_kwds": {"base_init_threshold": threshold, "n_embedding_epochs": budget,
                                        "recursive_negative_sample_rate": rate},
            },
        }
        fit = observe(spec, "fit", exception_class_only=True)
        assert fit == observe(spec, "fit_transform", exception_class_only=True), spec["name"]
        fits.append(spec | fit)
    return {"scalar": scalar_probes(), "replay": replay, "fits": fits}


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
