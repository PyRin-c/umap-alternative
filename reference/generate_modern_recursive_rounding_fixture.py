"""Record public UMAP 0.6.0 placement before outer normalization."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from umap import UMAP
from umap.label_prop import label_propagation_init, recursive_init

from generate_modern_recursive_fit_fixture import TracedRandomState, bits, rng_checkpoint, star_distances
from generate_modern_recursive_hub_fixture import graph_record, symmetric_graph


def checkpoint(rng):
    return rng_checkpoint(rng) | {"next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()}


def observe_placement(name, graph, anchor, *, parts, threshold, budget, depth=1, seed=44):
    rng = TracedRandomState(seed)
    graph_bits = graph_record(graph)
    try:
        layout = label_propagation_init(
            graph, anchor.copy(), np.ones(len(anchor), dtype=bool), 1.0, 1.0,
            n_components=anchor.shape[1], approx_n_parts=parts, base_init_threshold=threshold,
            n_embedding_epochs=budget, depth=depth, random_state=rng, recursive_parallel=False,
            coarsening_ratio=3, recursive_negative_sample_rate=0,
        )
    except Exception as error:
        raise SystemExit(f"{name} {type(error).__name__}") from None
    assert graph_record(graph) == graph_bits, name
    return {
        "name": name, "graph": graph_bits, "anchor_bits": bits(anchor), "n_components": anchor.shape[1],
        "threshold": threshold, "budget": budget, "depth": depth, "seed": seed,
        "layout_bits": bits(layout), "rng_calls": rng.calls.copy(), "rng": checkpoint(rng),
    }


def base_cases():
    anchors = []
    for rows, width in ((2, 1), (3, 2), (4, 3), (7, 4), (10, 7), (17, 3), (29, 2), (65, 3), (131, 3)):
        values = np.random.RandomState(99).normal(size=(rows, width)).astype(np.float32)
        values *= np.exp2(np.linspace(-12, 16, width, dtype=np.float32))
        values[::3] *= np.float32(-2**-8)
        anchors.append((f"mixed-{rows}-{width}", values))
    anchors.extend((
        ("signed-offset", np.random.RandomState(570).normal(size=(25, 6)).astype(np.float32) + np.float32(250)),
        ("subnormal-width", np.array([[0., 1e-40], [1e-40, 0.], [-1e-40, -2e-40], [0., 2e-40]], np.float32)),
        ("one-ulp-width", np.array([[0., 1.], [1., 1.], [2., 1.], [4., 1.],
                                   [8., np.nextafter(np.float32(1), np.float32(2))]], np.float32)),
        ("constant-width", np.array([[-1., 3.], [0., 3.], [1., 3.]], np.float32)),
    ))
    cases = []
    for name, values in anchors:
        rows = len(values)
        graph = symmetric_graph(rows, [(i, i + 1, 1.) for i in range(rows - 1)])
        for depth in (1, 4):
            cases.append(observe_placement(f"{name}-depth-{depth}", graph, values,
                                           parts=rows, threshold=rows, budget=0, depth=depth))
    return cases


def contraction_cases():
    cases = []
    edge = np.float32(2**24)
    anchors = np.repeat(np.array([[-3.25], [2.33], [9.125]], dtype=np.float32), 3, axis=0)
    spokes = [(hub, i, 1.) for hub in (0, 3, 6) for i in (hub + 1, hub + 2)]
    for name, weight, extra in (
        ("exact", edge, .53125),
        ("below", np.nextafter(edge, np.float32(0)), .875),
        ("above", np.nextafter(edge, np.float32(np.inf)), .875),
    ):
        small = [(i, j, extra) for i in (1, 2) for j in (4, 5)]
        graph = symmetric_graph(9, spokes + [(0, 3, weight), (0, 6, edge), (3, 6, edge)] + small)
        for budget in (0, 2, 3):
            cases.append(observe_placement(f"large-{name}-{budget}", graph, anchors,
                                           parts=3, threshold=3, budget=budget))

    spokes = [(hub, i, 1.) for hub in range(0, 27, 3) for i in (hub + 1, hub + 2)]
    branches = [(v * 3, u * 3, 1.) for v, u in ((0, 7), (1, 7), (2, 6), (3, 8), (4, 6), (5, 8))]
    for bridge, extra, scale, budget in ((1., .875, 1., 2), (32., .50000006, 2**17, 0),
                                         (32., .875, 1., 4), (2**16, .50000006, 2**17, 2)):
        weight = np.float32(bridge)
        edges = spokes + branches + [(18, 24, np.nextafter(weight, np.float32(np.inf))), (21, 24, weight),
                                     (19, 25, extra), (20, 26, extra)]
        axis = np.random.RandomState(782 if scale == 1 else 783).uniform(-7, 12, (9, 1)).astype(np.float32)
        anchors = np.repeat(axis * np.float32(scale), 3, axis=0)
        cases.append(observe_placement(f"nested-{bridge}-{budget}", symmetric_graph(27, edges), anchors,
                                       parts=9, threshold=3, budget=budget, seed=831))
    return cases


def archive_cases():
    values = np.asarray(star_distances([3] * 9, 620, chain=True, hierarchy=True), dtype=np.float32)
    try:
        graph = UMAP(metric="precomputed", n_neighbors=2, n_epochs=[0], init="random", optimizer="adam",
                     compatibility_layout=False, random_state=42, n_jobs=1).fit(values.copy()).graph_.tocsr()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    graph_bits = graph_record(graph)
    cases = []
    for threshold, budget in ((27, 0), (9, 0), (9, 11), (3, 0), (3, 11)):
        rng = TracedRandomState(42)
        try:
            layout = recursive_init(
                graph.copy(), values.copy(), 1., 1., n_components=2, random_state=rng, coarsening_ratio=3,
                base_init_threshold=threshold, n_embedding_epochs=budget, recursive_parallel=False,
                recursive_negative_sample_rate=0,
            )
        except Exception as error:
            raise SystemExit(type(error).__name__) from None
        cases.append({"name": f"star-raw-{threshold}-{budget}", "threshold": threshold, "budget": budget,
                      "layout_bits": bits(layout), "rng_calls": rng.calls.copy(), "rng": checkpoint(rng)})
    return {"input_bits": bits(values), "graph": graph_bits, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = {"base": base_cases(), "contractions": contraction_cases()}
        archive = archive_cases()
    lines = [f'  "{key}": [\n' + ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in cases)
             + "\n  ]" for key, cases in parts.items()]
    lines.append('  "archive": ' + json.dumps(archive, separators=(",", ":")))
    args.output.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8", newline="\n")
