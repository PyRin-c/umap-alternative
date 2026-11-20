"""Capture packed-feature supervision from the locked public UMAP wheel."""

import argparse
import os
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import csr_matrix
from umap import UMAP

from generate_boolean_update_fixture import observe
from generate_packed_metric_fixture import write_fixture
from generate_sparse_lp_transform_fixture import encode
from generate_sparse_update_fixture import state


def parameters(options):
    options = options.copy()
    for key in ("target_weight", "disconnection_distance"):
        if options.get(key) in ("inf", "nan"):
            options[key] = float(options[key])
    return options


def build():
    train = np.array([[0, 3, 128], [1, 1, 129], [2, 2, 130], [4, 3, 128], [15, 0, 192],
                      [3, 1, 224], [7, 2, 248], [128, 128, 128], [255, 16, 240], [5, 0, 128]], np.uint8)
    query = np.array([[0, 0, 0], [1, 3, 128], [2, 7, 145], [1, 0, 240]], np.uint8)
    axis = np.array([[0], [0], [1], [2], [3], [7], [4], [8], [128], [255]], np.uint8)
    labels = np.array([0, 1, 1, -1, 1, 2, -1, 0, 2, 0], np.int32)
    target = np.array([0., 1.5, 2.3, 3.2, 4.25, 5.1, 5.25, 3.75, 4.5, 11.2], np.float64)
    vectors = np.column_stack((target - .1, target[::-1])).astype(np.float32)
    inputs = {
        "bytes": train, "signed": train.view(np.int8), "f64": train.astype(np.float64) + .75,
        "replacement": train[::-1].copy(), "replacement_short": train[1:].copy(),
        "axis": axis, "query": query, "query_f64": query.astype(np.float64) + .75,
        "prefix": query[:, :1], "axis_query": np.array([[0], [1], [3], [255]], np.uint8),
        "partial": labels, "known": np.where(labels == -1, 2, labels),
        "unknown": np.full(labels.shape, -1, np.int32),
        "negative": np.where(labels == -1, -2, labels),
        "fractional": np.where(labels == -1, -1., labels + .25),
        "uint64": np.where(labels == -1, 3, labels).astype(np.uint64) + np.uint64(2**53),
        "column": labels[:, None], "short": labels[:-1], "short_twice": labels[:-2],
        "categorical_nan": np.where(labels == -1, np.nan, labels),
        "categorical_string": np.array(list("ababbabaab")),
        "categorical_sparse": csr_matrix(labels[:, None]),
        "target": target, "target_vectors": vectors, "target_square": np.eye(10, dtype=np.float32),
        "target_sparse": csr_matrix(vectors),
    }
    options = {
        "n_neighbors": 4, "n_components": 2, "n_epochs": 0, "init": "random",
        "random_state": 41, "transform_seed": 0, "n_jobs": 1, "compatibility_layout": True,
        "a": 1.576943460, "b": .895060879, "transform_mode": "graph",
    }
    cases = {}

    def capture(metric, name, y, *, training="bytes", query="query", method="fit_transform",
                warm=False, query_failed=False, update=False, **overrides):
        model = UMAP(**parameters(options | {"metric": metric} | (
            {"transform_mode": overrides.get("transform_mode", "graph"), "disconnection_distance": "inf"}
            if warm else overrides
        )))
        record = {"training": training, "target": y, "query": query, "method": method,
                  "options": {"metric": metric} | overrides, "warm": warm}
        if warm:
            record["warm_fit"] = observe(lambda: model.fit(inputs["bytes"].copy(), inputs["partial"].copy()))
            record["before"] = state(model)
            model.set_params(**parameters(overrides))
        previous = {name: getattr(model, name, None) for name in ("graph_", "embedding_")}
        x = inputs[training].copy()
        labels = inputs[y].copy()
        if method == "fit_transform":
            record["fit"] = observe(lambda: model.fit_transform(x, labels), model)
        else:
            record["fit"] = observe(lambda: model.fit(x, labels))
        record["after"] = state(model)
        record["replaced"] = {key: getattr(model, key, None) is not value for key, value in previous.items()}
        if "error" not in record["fit"] or query_failed:
            queries = {"fitted": x, "held_out": inputs[query].copy()}
            if warm:
                queries["original"] = inputs["bytes"].copy()
            if "error" not in record["fit"] and training != "axis":
                queries["prefix"] = inputs["prefix"].copy()
            record["transforms"] = {
                key: observe(lambda: model.transform(value), model) for key, value in queries.items()
            }
        if update:
            record["update"] = observe(lambda: model.update(inputs[query].copy()), expect_none=True)
        record["inputs_unchanged"] = encode(x) == encode(inputs[training]) and encode(labels) == encode(inputs[y])
        cases[f"{metric}_{name}"] = record

    for metric in ("bit_hamming", "bit_jaccard"):
        for label, weight in (
            ("zero", 0.), ("next_below_zero", float(np.nextafter(0., -1.))),
            ("next_above_zero", float(np.nextafter(0., 1.))),
            ("half", .5), ("next_below_half", float(np.nextafter(.5, 0.))),
            ("next_above_half", float(np.nextafter(.5, 1.))),
            ("below_one", float(np.nextafter(1., 0.))), ("one", 1.),
            ("above_one", float(np.nextafter(1., 2.))), ("negative", -.01), ("nan", "nan"),
        ):
            capture(metric, f"weight_{label}", "partial", target_weight=weight)
        for label in ("known", "unknown", "negative", "fractional", "uint64", "column"):
            capture(metric, f"labels_{label}", label, target_weight=1.)
        for name, method, training, epochs in (("cat_embedding_fit", "fit", "signed", 9),
                                              ("cat_embedding_transform", "fit_transform", "f64", 9),
                                              ("cat_embedding_zero", "fit_transform", "bytes", 0)):
            capture(metric, name, "partial", training=training, method=method, n_epochs=epochs,
                    query="query_f64" if training == "f64" else "query",
                    transform_mode="embedding", update=epochs == 9)
        for label in ("short", "categorical_nan", "categorical_string", "categorical_sparse", "target_vectors"):
            capture(metric, f"invalid_labels_{label}", label, query_failed=label == "short")
        capture(metric, "invalid_cat_weight", "partial", target_weight="bad")
        for target_metric in ("euclidean", "l2", "manhattan", "cityblock", "chebyshev", "hamming", "jaccard", "cosine"):
            for data in ("target", "target_vectors"):
                capture(metric, f"{target_metric}_{data}", data, target_metric=target_metric)
        capture(metric, "correlation_vectors", "target_vectors", target_metric="correlation")
        capture(metric, "precomputed_square", "target_square", target_metric="precomputed")
        for weight in (0., .25, float(np.nextafter(.5, 0.)), float(np.nextafter(.5, 1.)),
                       1., float(np.nextafter(1., 2.)), -.01):
            capture(metric, f"continuous_weight_{weight}", "target_vectors",
                    target_metric="euclidean", target_weight=weight)
        for k in (1, 2, 10):
            capture(metric, f"target_neighbors_{k}", "target", target_metric="euclidean", target_n_neighbors=k)
        capture(metric, "continuous_graph_options", "target_vectors", target_metric="manhattan",
                local_connectivity=.5, set_op_mix_ratio=.4, target_n_neighbors=3)
        for target_metric, data in (("euclidean", "target"), ("manhattan", "target_vectors"),
                                    ("chebyshev", "target_vectors")):
            capture(metric, f"continuous_embedding_{target_metric}", data, target_metric=target_metric,
                    transform_mode="embedding", n_epochs=9, update=target_metric == "euclidean")
        for target_metric, data in (("bit_hamming", "target"), ("bit_jaccard", "target"),
                                    ("precomputed", "target_vectors"), ("missing", "target"),
                                    ("euclidean", "target_sparse")):
            capture(metric, f"invalid_target_{target_metric}_{data}", data, target_metric=target_metric)
        threshold = np.float32(1.0 if metric == "bit_hamming" else .6931471824645996)
        for cut_name, cutoff in (
            ("zero", 0.), ("equal", float(threshold)),
            ("next64", float(np.nextafter(np.float64(threshold), np.inf))),
            ("next32", float(np.nextafter(threshold, np.float32(np.inf)))),
        ):
            for target_metric, data in (("categorical", "partial"), ("euclidean", "target")):
                capture(metric, f"cutoff_{target_metric}_{cut_name}", data, target_metric=target_metric,
                        training="axis", query="axis_query", disconnection_distance=cutoff,
                        query_failed=cut_name == "zero")
        for target_metric, data in (("categorical", "partial"), ("euclidean", "target")):
            capture(metric, f"empty_embedding_{target_metric}", data, target_metric=target_metric,
                    transform_mode="embedding", n_epochs=9, disconnection_distance=0., query_failed=True)
        capture(metric, "warm_bad_labels", "short_twice", training="replacement_short", method="fit",
                warm=True, query_failed=True, update=True)
        capture(metric, "warm_bad_target", "short", training="replacement_short", method="fit",
                target_metric="bit_jaccard", warm=True, query_failed=True)
        capture(metric, "warm_empty_graph", "short", training="replacement_short", method="fit",
                target_metric="euclidean", disconnection_distance=0., warm=True, query_failed=True)
        capture(metric, "warm_bad_neighbor_count", "short", training="replacement_short", method="fit",
                target_metric="euclidean", target_n_neighbors=1, warm=True, query_failed=True)

    encoded_inputs = {name: encode(value) for name, value in inputs.items()}
    encoded_inputs["categorical_string"]["dtype"] = inputs["categorical_string"].dtype.str
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": options, "inputs": encoded_inputs, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_fixture(build(), args.output, check=args.check)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
