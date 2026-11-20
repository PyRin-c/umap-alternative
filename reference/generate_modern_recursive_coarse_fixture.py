"""Record positive recursive optimizer checkpoints through public UMAP 0.6.0 fits."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

from generate_modern_recursive_fit_fixture import observe, star_distances


def generate():
    cases = []

    def add(name, data, *, budget, threshold, ratio, seed=98, warm=False, coarse=None, **options):
        recursive = {"base_init_threshold": threshold} | (coarse or {})
        if budget is not None:
            recursive["n_embedding_epochs"] = budget
        spec = {
            "name": name, "data": data, "seed": seed, "warm_gaussian": warm,
            "options": {
                "init": "recursive", "compatibility_layout": False, "optimizer": "adam",
                "n_epochs": [0], "n_neighbors": 5, "a": 1., "b": 1.,
                "recursive_coarsening_ratio": ratio, "recursive_init_kwds": recursive,
            } | options,
        }
        observed = observe(spec, "fit")
        assert observe(spec, "fit_transform") == observed, name
        cases.append(spec | observed)

    dense = {"seed": 673, "shape": [14, 3]}
    for budget in (0, 1, 2, 3, 4, 11, 24, 48, None):
        add(f"two-level-budget-{budget}", dense, budget=budget, threshold=3, ratio=2)
    add("three-coarse-epochs-outer-projection", dense, budget=3, threshold=3, ratio=2,
        n_epochs=[0, 1, 3, 6], negative_selection_range=3)
    add("two-level-warm-negative-two", dense, budget=11, threshold=3, ratio=2, seed=307, warm=True,
        optimizer="momentum", n_epochs=[0, 1, 4, 9], negative_sample_rate=0., learning_rate=2.25,
        coarse={"recursive_negative_sample_rate": 2, "recursive_repulsion_strength": 2.5})
    add("two-level-full-range", dense, budget=8, threshold=3, ratio=2, n_epochs=[0, 1, 3, 6],
        coarse={"recursive_negative_selection_range_mode": "coarse_n"})
    add("two-level-range-scale-one", dense, budget=8, threshold=3, ratio=2,
        coarse={"recursive_negative_selection_range_scale": 1.})
    add("two-level-fixed-range", dense, budget=8, threshold=3, ratio=2,
        coarse={"recursive_negative_selection_range_mode": "fixed_200k"})
    for schedule in ("strong_to_one", "original"):
        add(f"two-level-{schedule}", dense, budget=9, threshold=3, ratio=2, a=1.7, b=.9,
            coarse={"curve_schedule": schedule, "recursive_negative_sample_rate": 0,
                    "good_initialization_policy": "never"})

    three_levels = {"seed": 711, "shape": [12, 3]}
    for budget in (2, 3, 4, 6, 24, None):
        add(f"three-level-budget-{budget}", three_levels, budget=budget, threshold=2, ratio=2,
            seed=42, n_neighbors=8)
    for policy in ("never", "available", "always"):
        add(f"three-level-policy-{policy}", three_levels, budget=5, threshold=2, ratio=2, seed=42,
            n_neighbors=8, coarse={"good_initialization_policy": policy})
    add("three-level-changed-neighbors", {"seed": 1051, "shape": [21, 4]}, budget=2,
        threshold=2, ratio=2, seed=29, n_neighbors=5)
    add("three-level-coarse-eleven", {"seed": 1067, "shape": [37, 4]}, budget=11,
        threshold=4, ratio=2, seed=42, n_neighbors=5)
    add("ratio-three-local-half", {"seed": 436, "shape": [19, 3]}, budget=8,
        threshold=3, ratio=3, seed=78, warm=True, local_connectivity=.5)
    add("ratio-two-local-one-and-half", {"seed": 436, "shape": [19, 3]}, budget=8,
        threshold=3, ratio=2, seed=78, warm=True, local_connectivity=1.5)
    add("single-level-ratio-four", three_levels, budget=48, threshold=6, ratio=4,
        seed=42, n_neighbors=8, optimizer="momentum", n_epochs=[0, 1, 3, 6], learning_rate=.8)

    depth_four = {"seed": 231, "shape": [28, 4]}
    for budget in (0, 2, 3):
        add(f"four-level-budget-{budget}", depth_four, budget=budget, threshold=2, ratio=2,
            seed=71, n_neighbors=10)
    add("four-level-positive-only", depth_four, budget=3, threshold=2, ratio=2, seed=71, n_neighbors=10,
        coarse={"recursive_negative_sample_rate": 0})

    stars = star_distances([3] * 9, 620, chain=True, hierarchy=True)
    for budget in (0, 3, 4, 11, None):
        add(f"ratio-three-stars-budget-{budget}", stars, budget=budget, threshold=3, ratio=3,
            seed=42, n_neighbors=2, metric="precomputed")
    add("ratio-three-stars-positive-only", stars, budget=5, threshold=3, ratio=3, seed=42,
        n_neighbors=2, metric="precomputed", coarse={"recursive_negative_sample_rate": 0})
    add("ratio-three-stars-positive-only-eleven", stars, budget=11, threshold=3, ratio=3, seed=42,
        n_neighbors=2, metric="precomputed", coarse={"recursive_negative_sample_rate": 0})
    add("ratio-three-stars-full-range", stars, budget=4, threshold=3, ratio=3, seed=42,
        n_neighbors=2, metric="precomputed", coarse={"recursive_negative_selection_range_mode": "coarse_n"})

    isolated = [[0., 0.], [.1, .08], [.3, .809], [4., .117], [4.2, -.07], [5.9, 0.], [6.2, .2], [8., 4.]]
    add("cutoff-isolates-coarse-five", isolated, budget=5, threshold=2, ratio=4, seed=61, warm=True,
        n_neighbors=3, disconnection_distance=.5, optimizer="momentum", init_rescale="graph_edge",
        n_epochs=[0, 1, 3, 6])

    environment = {
        "system": platform.system(), "python": platform.python_version(),
        "packages": {package: version(package) for package in (
            "umap-learn", "numpy", "scipy", "scikit-learn", "numba", "llvmlite",
        )},
    }
    return environment, cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment, cases = generate()
    header = json.dumps({"environment": environment}, indent=2)[:-2]
    encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in cases)
    args.output.write_text(f'{header},\n  "cases": [\n{encoded}\n  ]\n}}\n', encoding="utf-8", newline="\n")
