"""Capture packed-feature distances and estimators from the locked public wheels."""

import argparse
import json
import os
import warnings
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from pynndescent.distances import bit_hamming, bit_jaccard
from scipy.sparse import csr_matrix
from umap import UMAP

from generate_sparse_lp_transform_fixture import encode


def observe(action, model=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = action()
        except Exception as error:
            result = {"error": type(error).__name__}
        else:
            if model is not None and output is getattr(model, "embedding_", None):
                result = {"identity": "embedding_"}
            elif model is not None and output is getattr(model, "graph_", None):
                result = {"identity": "graph_"}
            else:
                result = {"value": encode(output)}
    return result | {"warnings": [item.category.__name__ for item in caught]}


def build():
    train = np.array([[0, 3, 128], [1, 1, 129], [2, 2, 130], [4, 3, 128], [15, 0, 192],
                      [3, 1, 224], [7, 2, 248], [128, 128, 128], [255, 16, 240], [5, 0, 128]], np.uint8)
    query = np.array([[0, 0, 0], [1, 3, 128], [2, 7, 145], [1, 0, 240]], np.uint8)
    axis_train = np.array([[0], [0], [1], [2], [3], [7], [4], [8], [128], [255]], np.uint8)
    axis_query = np.array([[0], [1], [2], [3], [255]], np.uint8)
    rng = np.random.RandomState(294)
    alphabet = np.array([0, 1, 2, 4, 8, 128], np.uint8)
    wrapped = train.astype(np.int32)
    wrapped[0] = [-1, 256, 300]
    wrapped_query = query.astype(np.int16)
    wrapped_query[0] = [-255, 512, -129]
    nonfinite = train.astype(np.float64)
    nonfinite[0] = [np.nan, np.inf, -np.inf]
    inputs = {
        "bytes": train, "query": query, "prefix": query[:, :2], "axis": axis_train,
        "axis_query": axis_query, "ties": rng.choice(alphabet, (28, 4)),
        "ties_query": rng.choice(alphabet, (5, 4)), "signed": train.view(np.int8),
        "fractions": train.astype(np.float64) + .75, "wrapped": wrapped,
        "wrapped_u8": wrapped.astype(np.uint8),
        "boolean": train.astype(bool),
        "query_f64": query.astype(np.float64) + .75, "query_f16": query.astype(np.float16) + .75,
        "f16": train.astype(np.float16) + .5, "query_signed": query.view(np.int8),
        "query_wrapped": wrapped_query, "nonfinite": nonfinite, "sparse": csr_matrix(train),
        "empty": train[:0], "vector": train[0], "empty_width": train[:, :0],
        "all_zero": np.zeros((5, 3), np.uint8), "one_zero": np.zeros((1, 3), np.uint8),
        "connected": np.array([[128 | i] for i in (0, 1, 3, 2, 6, 7, 5, 4)], np.uint8),
        "connected_query": np.array([[129], [135]], np.uint8),
    }
    options = {"n_neighbors": 4, "n_components": 2, "n_epochs": 0, "init": "random",
               "random_state": 41, "transform_seed": 0, "n_jobs": 1,
               "compatibility_layout": True, "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(metric, name, training, queries=(), *, inverse=False, **settings):
        spec = {"metric": metric} | settings
        kwargs = options | spec
        if isinstance(kwargs.get("disconnection_distance"), str):
            kwargs["disconnection_distance"] = float(kwargs["disconnection_distance"])
        model = UMAP(**kwargs)
        fitted = observe(lambda: model.fit_transform(inputs[training].copy()))
        record = {"training": training, "options": spec, "fit": fitted}
        if hasattr(model, "graph_"):
            record["graph"] = encode(model.graph_)
        if "error" not in fitted:
            record["queries"] = {name: observe(lambda: model.transform(inputs[name].copy()), model)
                                 for name in queries}
            if inverse:
                record["inverse"] = observe(lambda: model.inverse_transform(np.zeros((1, 2), np.float32)))
        cases[f"{metric}_{name}"] = record

    for metric in ("bit_hamming", "bit_jaccard"):
        queries = ("query", "query_f64", "query_signed", "query_wrapped", "prefix", "bytes",
                   "signed", "fractions", "nonfinite", "sparse", "empty", "vector", "empty_width")
        capture(metric, "graph", "bytes", queries, inverse=True, transform_mode="graph")
        for form in ("fractions", "wrapped", "boolean", "signed"):
            capture(metric, form, form, (form, "bytes", "wrapped_u8", "query"), transform_mode="graph")
        capture(metric, "ties", "ties", ("ties_query",), n_neighbors=5, transform_mode="graph")
        capture(metric, "local_connectivity", "axis", ("axis_query",), local_connectivity=2.0,
                set_op_mix_ratio=.4, transform_mode="graph")
        capture(metric, "all_zero", "all_zero", ("one_zero", "axis_query"), transform_mode="graph")
        capture(metric, "uncut", "axis", ("axis_query",), disconnection_distance="inf", transform_mode="graph")
        capture(metric, "nan_cutoff", "axis", ("axis_query",), disconnection_distance="nan", transform_mode="graph")
        capture(metric, "invalid_seed", "bytes", ("query", "bytes"), transform_seed=-1, transform_mode="graph")
        for epochs in (0, 1, 9):
            capture(metric, f"embedding_{epochs}", "bytes",
                    ("query", "query_f64", "prefix", "bytes", "fractions"),
                    n_epochs=epochs, inverse=epochs == 9)
        capture(metric, "embedding_coerced", "fractions", ("query", "query_f64", "bytes", "fractions"), n_epochs=9)
        capture(metric, "embedding_f16", "f16", ("query_f16", "bytes", "f16"), n_epochs=9)
        capture(metric, "embedding_unique", "axis", ("axis", "axis_query"), n_epochs=9, unique=True)
        capture(metric, "embedding_ties", "ties", ("ties_query",), n_neighbors=5, n_epochs=9,
                transform_seed=2**32 - 1)
        capture(metric, "zero_cutoff_empty", "bytes", disconnection_distance=0.0)
        for invalid in ("sparse", "nonfinite", "empty", "vector", "empty_width"):
            capture(metric, f"invalid_{invalid}", invalid)
        threshold = np.float32(1.0 if metric == "bit_hamming" else 0.6931471824645996)
        for label, cutoff in (("equal", float(threshold)),
                              ("next64", float(np.nextafter(np.float64(threshold), np.inf))),
                              ("next32", float(np.nextafter(threshold, np.float32(np.inf))))):
            capture(metric, label, "axis", ("axis_query",), disconnection_distance=cutoff, transform_mode="graph")
        capture(metric, "spectral", "connected", ("connected_query",), init="spectral", n_epochs=9)
    for invalid in ("BIT_HAMMING", "Bit_Jaccard", "bit_dice", "bitjaccard"):
        capture(invalid, "invalid", "bytes")

    pairs = [
        ([0], [0]), ([0], [255]), ([1], [2]), ([3], [1]), ([3], [255]),
        ([7], [1]), ([127], [128]), ([255], [255]), ([128], [129]), ([15], [4]),
        ([3, 9], [2, 8]), ([128, 128], [129, 127]), ([3, 0], [7, 0]),
        ([63], [127]),
    ]
    def named_pair(left, right):
        a, b = np.array(left, np.uint8), np.array(right, np.uint8)
        return {"left": left, "right": right,
                "hamming": encode(np.float32(bit_hamming(a, b))),
                "jaccard": encode(np.float32(bit_jaccard(a, b)))}

    def packed(count):
        return np.packbits(np.arange(48) < count, bitorder="little").tolist()

    grid_rows = [np.packbits(np.arange(128) < count, bitorder="little") for count in range(100)]
    grid = np.asarray([
        bit_jaccard(grid_rows[shared], grid_rows[union])
        for union in range(1, 100)
        for shared in range(1, union + 1)
    ], dtype="<f4")
    platform = {
        "named": [named_pair(packed(i), packed(u)) for i, u in ((37, 38), (30, 47), (44, 47))],
        "jaccard_grid_sha256": sha256(grid.tobytes()).hexdigest(),
        "cases": {key: cases.pop(key) for key in ("bit_hamming_spectral", "bit_jaccard_spectral")},
    }
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": options, "inputs": {name: encode(values) for name, values in inputs.items()},
        "named": [named_pair(left, right) for left, right in pairs], "cases": cases,
    }, platform


def write_fixture(record, path, *, check):
    header = json.dumps({key: value for key, value in record.items() if key != "cases"},
                        allow_nan=False, separators=(",", ":"))
    cases = ",\n".join(
        f"    {json.dumps(key)}: {json.dumps(value, allow_nan=False, separators=(',', ':'))}"
        for key, value in record["cases"].items()
    )
    result = f'{header[:-1]},\n  "cases": {{\n{cases}\n  }}\n}}\n'
    if check:
        if path.read_text(encoding="utf-8") != result:
            raise SystemExit(f"Regenerated fixture differs: {path.name}")
    else:
        path.write_text(result, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record, platform = build()
    write_fixture(record, args.output, check=args.check)
    if args.host_output is not None:
        write_fixture(platform, args.host_output, check=args.check)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Public JIT errors can carry source excerpts.
        raise SystemExit(type(error).__name__) from None
