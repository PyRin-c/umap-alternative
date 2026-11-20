"""Observe tied hub ordering and coarse layouts through public UMAP 0.6.0 calls."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from umap.label_prop import label_propagation_init

from generate_modern_recursive_fit_fixture import TracedRandomState, bits, observe, rng_checkpoint


def graph_record(graph):
    return {"indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(), "weights_bits": bits(graph.data)}


def symmetric_graph(size, edges):
    pairs = [(source, target, weight) for i, j, weight in edges for source, target in ((i, j), (j, i))]
    head, tail, weights = zip(*pairs, strict=True)
    return csr_matrix((np.asarray(weights, dtype=np.float32), (head, tail)), shape=(size, size))


def generate():
    stages, fits = [], []

    def stage(name, graph, n_parts, budget, rate=1, mode="transition", selection="scaled_coarse"):
        anchor = np.random.RandomState(210).uniform(-4., 5., (graph.shape[0], 2)).astype(np.float32)
        seed = 421
        rng = TracedRandomState(seed)
        initial_graph = graph_record(graph)
        try:
            layout = label_propagation_init(
                graph, anchor.copy(), np.ones(len(anchor), dtype=bool), 1., 1.,
                n_components=2, approx_n_parts=n_parts, n_embedding_epochs=budget, depth=1,
                base_init_threshold=n_parts, random_state=rng, recursive_parallel=False, recursive_init=False,
                coarse_layout_weights=mode, recursive_negative_sample_rate=rate,
                good_initialization_policy="never", recursive_negative_selection_range_mode=selection,
            )
        except Exception as error:
            raise SystemExit(f"{name}: {type(error).__name__}: {error}") from None
        assert graph_record(graph) == initial_graph, name
        stages.append({
            "name": name, "anchor": anchor.tolist(), "graph": initial_graph, "parts": n_parts,
            "budget": budget, "rate": rate, "weights": mode, "selection": selection,
            "seed": seed, "layout_bits": bits(layout), "rng_calls": rng.calls.copy(),
            "rng": rng_checkpoint(rng) | {"next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist()},
        })

    for size in (15, 16, 17):
        edges = [(row, (row + 1) % size, 1.) for row in range(size)]
        stage(f"tied-cycle-{size}", symmetric_graph(size, edges), 5, 4)
    below_one = np.nextafter(np.float32(1), np.float32(0))
    base = [(start, leaf, 1.) for start in range(0, 27, 3) for leaf in (start + 1, start + 2)]
    base += [(0, 3, 1.), (6, 3, 1.), (9, 12, 1.), (15, 12, 1.), (18, 21, 1.), (24, 21, 1.),
             (3, 12, 1.), (12, 21, 1.)]
    stars = symmetric_graph(27, base)
    changed = symmetric_graph(27, [(0, 1, below_one), *base[1:]])
    order = np.random.RandomState(37).permutation(27)
    permuted = stars[order][:, order]
    permuted.sort_indices()
    for mode in ("transition", "fuzzy_union"):
        stage(f"{mode}-tied-zero", stars, 9, 0, mode=mode)
        stage(f"{mode}-tied-projected", stars, 9, 4, mode=mode)
        stage(f"{mode}-tied-full-range", stars, 9, 4, mode=mode, selection="coarse_n")
        stage(f"{mode}-single-edge-below-tie", changed, 9, 4, mode=mode)
        stage(f"{mode}-permuted-attraction", permuted, 9, 11, rate=0, mode=mode)

    for threshold, budget, mode in ((14, 3, "transition"), (7, 0, "transition"),
                                    (3, 0, "transition"), (2, 3, "fuzzy_union")):
        name = f"four-level-fit-threshold-{threshold}-{mode}"
        spec = {
            "name": name, "data": {"seed": 231, "shape": [28, 4]}, "seed": 71, "warm_gaussian": False,
            "options": {
                "init": "recursive", "compatibility_layout": False, "optimizer": "adam", "n_epochs": [0],
                "n_neighbors": 10, "a": 1., "b": 1., "recursive_coarsening_ratio": 2,
                "coarse_layout_weights": mode,
                "recursive_init_kwds": {"base_init_threshold": threshold, "n_embedding_epochs": budget},
            },
        }
        result = observe(spec, "fit")
        assert observe(spec, "fit_transform") == result, name
        fits.append(spec | result)
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
