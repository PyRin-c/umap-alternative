"""Record serial densMAP checkpoints through the published UMAP 0.6.0 API."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from umap import UMAP


REFERENCE = Path(__file__).resolve().parent
BASE = dict(n_jobs=1, random_state=42, compatibility_layout=True, metric="precomputed",
            learning_rate=.1, negative_sample_rate=1, repulsion_strength=0., a=1.7, b=.9)
DATASETS = {
    "paired_holdout": dict(data=[[4 * row + offset] for row in range(9)
                                  for offset in (0, .15 + (row * 11 % 19) / 16)],
                           init=[[row * 3 + offset] for row in range(9)
                                 for offset in (0, .1 + (row * 7 % 13) / 5)],
                           n_neighbors=2),
    "pairs": dict(data=[[0], [.2], [4], [4.8], [10], [11.4]],
                  init=[[0], [2], [3], [4], [7], [10]], n_neighbors=2),
    "balanced_pairs": dict(data=[[0], [4], [10], [11]],
                           init=[[0], [1], [9], [10]], n_neighbors=2),
    "equal_input": dict(data=[[0], [1], [4], [5]],
                        init=[[0], [2], [3], [4]], n_neighbors=2),
    "equal_embedding": dict(data=[[0], [.2], [4], [4.8]],
                            init=[[0], [1], [3], [4]], n_neighbors=2),
    "weighted": dict(data=[[0, 0, 0], [.2, .07, .1], [.25, .89, .13], [.76, .1, .83],
                           [.99, .95, .37], [1.9, 1.31, .33], [2.61, 1.7, .9], [3.3, 1.2, 1.1]],
                     init=[[0, 6], [1, 3], [4, 0], [2, 1], [5, 4], [6, 5], [3, 2], [7, 7]],
                     n_neighbors=4),
    "weak_edges": dict(data=[[0], [.1], [.205], [5], [5.25], [5.515], [12], [12.9], [13.81]],
                       init=[[7], [2], [8], [10], [1], [4], [0], [9], [6]], n_neighbors=4),
    "held_out_3d": dict(data=[[.2, 1.1], [.7, -.8], [-.9, -.3], [2.4, 1.2], [3.1, -.4],
                              [4.2, 2.2], [1.1, 3.4]],
                        init=[[3, 1, 5], [6, 7, 1], [2, 6, 9], [8, 3, 2], [4, 9, 7],
                              [9, 2, 4], [1, 4, 8]], n_neighbors=5),
    "held_out_1d": dict(data=[[.01, .2], [.8, .3], [-.34, .88], [2.1, 1.3], [3.1, 0.]],
                        init=[[4], [8], [2], [6], [0]], n_neighbors=3),
}


def bits(values):
    if values.dtype != np.float32:
        raise AssertionError(f"expected float32, got {values.dtype}")
    return np.ascontiguousarray(values).view(np.uint32).ravel().tolist()


def matrix(data):
    values = np.asarray(data, dtype=np.float32).astype(np.float64)
    delta = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)


def fit(dataset, *, epochs, density, options, scalar):
    config = DATASETS[dataset]
    distances = matrix(config["data"])
    init = np.asarray(config["init"], np.float32)
    params = BASE | dict(n_neighbors=config["n_neighbors"], n_components=init.shape[1],
                         densmap=density, n_epochs=epochs if scalar else list(range(epochs + 1)))
    model = UMAP(**(params | options), init=init.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = model.fit_transform(distances.copy())
    graph = model.graph_.tocsr()
    checkpoints = ({} if scalar else
                   {str(i): bits(value) for i, value in enumerate(model.embedding_list_)})
    if not scalar and len(checkpoints) != epochs + 1:
        raise AssertionError(f"unexpected history size for {dataset} at {epochs}")
    result = {
        "dataset": dataset, "epochs": epochs, "density": density, "options": options,
        "graph": {
            "indptr": graph.indptr.tolist(),
            "indices": graph.indices.tolist(),
            "weights": bits(graph.data),
            "distances": bits(distances[graph.tocoo().row, graph.indices]),
        },
        "checkpoints": checkpoints, "final": bits(model.embedding_),
    }
    if options.get("output_dens"):
        result["original_radii"] = bits(outputs[1])
    return result


def build():
    cases = {}
    errors = {}

    def add(name, dataset, epochs, *, density=True, scalar=False, expected_error=None, **options):
        try:
            case = fit(dataset, epochs=epochs, density=density, options=options, scalar=scalar)
        except Exception as error:
            # Report only the public exception type. A traceback can expose source.
            if type(error).__name__ == expected_error:
                errors[name] = dict(dataset=dataset, epochs=epochs, options=options, error=expected_error)
                return
            raise RuntimeError(f"{name}: {type(error).__name__}: {error}") from None
        if expected_error:
            raise AssertionError(f"{name}: expected {expected_error}")
        cases[name] = case

    add("zero", "weighted", 0, scalar=True, output_dens=True)
    add("one", "weighted", 1, dens_frac=1.)
    add("pairs_plain", "pairs", 2, density=False)
    add("pairs_first_active_scalar", "pairs", 2, dens_frac=1., scalar=True)
    for shift in (0., .1, 1., 4.):
        add(f"pairs_shift_{shift}", "pairs", 2, dens_frac=1., dens_var_shift=shift)
    for shift in (0., 1e-12, .27):
        add(f"paired_holdout_shift_{shift}", "paired_holdout", 2, dens_frac=1.,
            dens_var_shift=shift, dens_lambda=1.2, learning_rate=.13, a=.87, b=1.42,
            output_dens=True)
    add("equal_embedding_tiny_shift", "equal_embedding", 2, dens_frac=1., dens_var_shift=1e-12)
    add("held_out_3d_small_shift", "held_out_3d", 2, dens_frac=1., dens_var_shift=1e-12,
        dens_lambda=.4, a=.73, b=1.43, learning_rate=.19)
    # The first pair starts at squared distance one with a standardized radius
    # of +1. Balance the two forces so the head isolates their last-bit errors.
    a, b = BASE["a"], BASE["b"]
    affinity = 1 / (1 + a)
    derivative = a * b * affinity
    shift = .17
    strength = derivative * math.sqrt(shift) / (1 - b * (1 - affinity) + derivative)
    add("balanced_pair_forces", "balanced_pairs", 2, dens_frac=1., dens_var_shift=shift,
        dens_lambda=strength, learning_rate=.0001)
    add("equal_embedding_subnormal_shift", "equal_embedding", 2, dens_frac=1., dens_var_shift=1e-40)
    add("equal_embedding_underflow_shift", "equal_embedding", 2, dens_frac=1., dens_var_shift=1e-50,
        expected_error="ZeroDivisionError")
    add("pairs_clipped", "pairs", 2, dens_frac=1., dens_lambda=80.)
    add("weighted_plain", "weighted", 4, density=False)
    add("weighted_active", "weighted", 4, dens_frac=1.)
    add("weighted_weak_density", "weighted", 4, dens_frac=1., dens_lambda=.2)
    add("weighted_repel_plain", "weighted", 4, density=False,
        repulsion_strength=1., negative_sample_rate=2)
    add("weighted_repel", "weighted", 4, dens_frac=1., repulsion_strength=1., negative_sample_rate=2)
    add("lambda_zero", "weighted", 4, dens_frac=1., dens_lambda=0.)
    add("frac_zero", "weighted", 4, dens_frac=0.)
    add("frac_above_zero", "weighted", 4, dens_frac=float(np.nextafter(0., 1.)))
    add("frac_below_one", "weighted", 4, dens_frac=float(np.nextafter(1., 0.)))
    for name, frac in (("below", np.nextafter(.5, 0.)), ("exact", .5),
                       ("above", np.nextafter(.5, 1.))):
        add(f"frac_{name}_half", "weighted", 4, dens_frac=float(frac))
    add("frac_default", "weighted", 10)
    add("plain_ten", "weighted", 10, density=False)
    add("weak_edges_short", "weak_edges", 3, dens_frac=1.)
    add("weak_edges_long", "weak_edges", 16, dens_frac=1.)
    add("equal_input_one", "equal_input", 1, dens_frac=1.)
    add("equal_input_disabled", "equal_input", 2, dens_frac=1., dens_lambda=0.)
    add("equal_input", "equal_input", 2, dens_frac=1.)
    add("equal_embedding_shifted", "equal_embedding", 2, dens_frac=1.)
    add("equal_embedding_unshifted", "equal_embedding", 2, dens_frac=1., dens_var_shift=0.,
        expected_error="ZeroDivisionError")
    add("frac_below_zero", "weighted", 2, dens_frac=float(np.nextafter(0., -1.)),
        expected_error="ValueError")
    add("frac_above_one", "weighted", 2, dens_frac=float(np.nextafter(1., 2.)),
        expected_error="ValueError")
    add("held_out_3d", "held_out_3d", 8, dens_frac=.625, dens_var_shift=.03, dens_lambda=1.3,
        a=.65, b=1.2, random_state=73, learning_rate=.31, repulsion_strength=.65, negative_sample_rate=3)
    add("held_out_3d_plain", "held_out_3d", 8, density=False,
        a=.65, b=1.2, random_state=73, learning_rate=.31, repulsion_strength=.65, negative_sample_rate=3)
    add("held_out_3d_first_active", "held_out_3d", 2, dens_frac=1., dens_var_shift=.03, dens_lambda=1.3,
        a=.65, b=1.2, random_state=73, learning_rate=.31, output_dens=True)
    add("held_out_1d", "held_out_1d", 7, dens_frac=.43, dens_var_shift=.8, dens_lambda=.25,
        a=2.1, b=.74, random_state=9, learning_rate=.07)
    lock = (REFERENCE / "uv.lock").read_text(encoding="utf-8").encode("utf-8")
    return {"schema_version": 1, "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                            "llvmlite", "scikit-learn", "pynndescent")},
            "common_options": BASE, "datasets": DATASETS, "errors": errors, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = build()
    text = "{\n" + ",\n".join(f"  {json.dumps(key)}: {json.dumps(value, allow_nan=False)}"
                                for key, value in fixture.items() if key != "cases") + ",\n"
    text += '  "cases": {\n' + ",\n".join(
        f"    {json.dumps(key)}: {json.dumps(value, allow_nan=False)}"
        for key, value in fixture["cases"].items()) + "\n  }\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != text:
            parser.exit(1, f"{args.output} differs from the regenerated fixture\n")
    else:
        args.output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit(str(error))
