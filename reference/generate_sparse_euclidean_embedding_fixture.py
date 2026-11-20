"""Capture seeded sparse Euclidean fits through the public UMAP 0.6.0 API."""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from umap import UMAP


OPTIONS = dict(
    metric="euclidean", n_neighbors=4, n_components=2, n_jobs=1, random_state=42,
    compatibility_layout=True, init="random", a=1.7, b=.9, learning_rate=.13,
    negative_sample_rate=2, n_epochs=6,
)


def encode_array(values):
    return {
        "shape": list(values.shape), "dtype": values.dtype.name,
        "bits": values.view(f"uint{8 * values.itemsize}").ravel().tolist(),
    }


def encode_sparse(values):
    record = {
        "format": values.format, "shape": list(values.shape),
        "data": encode_array(values.data),
    }
    if values.format == "coo":
        return record | {"row": values.row.tolist(), "col": values.col.tolist()}
    return record | {"indices": values.indices.tolist(), "indptr": values.indptr.tolist()}


def observe(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except ValueError as error:
            record = {"error": type(error).__name__, "message": str(error)}
        except Exception as error:
            # JIT diagnostics can contain source details.
            record = {"error": type(error).__name__}
        else:
            record = {"result": encode_array(result)}
    return record | {"warnings": [[item.category.__name__, str(item.message)] for item in caught]}


def make_inputs():
    prior_fixture = Path(__file__).with_name("fixtures") / (
        "umap-learn-0.6.0-sparse-graph-windows-x86_64.json"
    )
    source = json.loads(prior_fixture.read_text(encoding="utf-8"))["inputs"]["features"]
    base = csr_matrix((
        np.asarray(source["data_bits_uint32"], dtype=np.uint32).view(np.float32),
        source["indices"], source["indptr"],
    ), shape=source["shape"])
    reverse = base.copy()
    for start, end in zip(reverse.indptr[:-1], reverse.indptr[1:]):
        reverse.data[start:end] = reverse.data[start:end][::-1]
        reverse.indices[start:end] = reverse.indices[start:end][::-1]
    split = csr_matrix((
        np.repeat(base.data / 2, 2), np.repeat(base.indices, 2), base.indptr * 2,
    ), shape=base.shape)
    coo = base.tocoo()
    split_coo = coo_matrix((
        np.tile(coo.data / 2, 2), (np.tile(coo.row, 2), np.tile(coo.col, 2)),
    ), shape=base.shape)
    cancel = csr_matrix((
        np.array([1e8, 1, -1e8, 2, 4, 7, 10], dtype=np.float64),
        [0] * 7, [0, 3, 4, 5, 6, 7],
    ), shape=(5, 1))
    axis = np.array([0, .125, .5, .8125, 1.21875, 1.7, 2.125, 9.25], dtype=np.float32)
    points = [0, 1.1, 2.7, 3.8, 5.1, 6.2]
    separated = np.sort(np.add.outer(points, [0, .03, .07]).ravel()).astype(np.float32)
    return {
        "csr": base, "csc": base.tocsc(), "coo": split_coo, "split": split, "reversed": reverse,
        "float64": base.astype(np.float64),
        "integers": csr_matrix(np.array([[0], [1], [3], [5], [9], [12]], dtype=np.int32)),
        "cancel_csr": cancel, "cancel_csc": cancel.tocsc(), "cancel_coo": cancel.tocoo(),
        "wide": csr_matrix((axis[1:], (np.arange(1, len(axis)), [4096] * (len(axis) - 1))),
                           shape=(len(axis), 4097)),
        "singleton": csr_matrix((1, 5), dtype=np.float32),
        "all_zero_two": csr_matrix((2, 5), dtype=np.float32),
        "all_zero_eight": csr_matrix((8, 5), dtype=np.float32),
        "stored_zeros": csr_matrix((np.array([0, -0., 0], np.float32), ([0, 2, 3], [4, 2, 1])),
                                   shape=(5, 5)),
        "cutoff": csr_matrix((axis, (np.arange(len(axis)), [4] * len(axis))),
                             shape=(len(axis), 5)),
        "separated": csr_matrix(separated.reshape(-1, 1)),
    }


def build():
    training = make_inputs()
    cases = {}

    def record(name, source, init="random", **options):
        values = training[source].copy()
        init_record = init
        if init == "explicit":
            t = np.arange(values.shape[0], dtype=np.float32)
            init = np.column_stack((t / 7, ((t * 7) % 11) / 4 + t / 64))
            init_record = encode_array(init)
        model = UMAP(**(OPTIONS | options | {"init": init}))
        fit = observe(lambda: model.fit_transform(values))
        record = {"training": source, "options": options, "init": init_record, "fit": fit}
        if "result" in fit:
            if hasattr(model, "graph_"):
                record["graph"] = encode_sparse(model.graph_)
            if hasattr(model, "embedding_list_"):
                record["history"] = [encode_array(values) for values in model.embedding_list_]
            record["embedding"] = encode_array(model.embedding_)
        cases[name] = record

    for init in ("random", "explicit"):
        for epochs in (0, 1, 2, 3, 6, 11, 30, [0, 2, 6]):
            record(f"{init}_{epochs}", "csr", init, n_epochs=epochs)
    for source in ("csc", "coo", "split", "reversed", "float64", "integers", "wide", "singleton"):
        record(source, source)
    for source in ("cancel_csr", "cancel_csc", "cancel_coo"):
        record(source, source, n_neighbors=3)
    for source in ("all_zero_two", "all_zero_eight", "stored_zeros"):
        record(source, source, n_neighbors=2)
    record("truncated", "all_zero_eight", n_neighbors=15)
    record("graph_options", "csr", local_connectivity=2.5, set_op_mix_ratio=.4)
    for cutoff in (0., .5, .65, 1., 20.):
        record(f"cutoff_{cutoff}", "cutoff", disconnection_distance=cutoff)
    record("cutoff_explicit", "cutoff", "explicit", disconnection_distance=.5)
    for epochs in (0, 6):
        record(f"spectral_{epochs}", "csr", "spectral", n_epochs=epochs)
    record("spectral_cutoff", "cutoff", "spectral", disconnection_distance=.5)
    record("spectral_components", "separated", "spectral", disconnection_distance=.1,
           n_neighbors=3)

    identity = {}
    identity_queries = {
        name: training[name] for name in ("csr", "csc", "coo", "split", "reversed", "float64")
    }
    identity_queries["dense"] = training["csr"].toarray()
    identity_queries["dense_f64"] = identity_queries["dense"].astype(np.float64)
    identity_queries["heldout"] = training["csr"][1:3]
    identity_queries["explicit_zeros"] = training["cutoff"]
    identity_queries["implicit_zeros"] = csr_matrix(training["cutoff"].toarray())
    for name in ("csr", "csc", "coo", "split", "reversed", "float64", "cutoff"):
        model = UMAP(**OPTIONS)
        model.fit(training[name].copy())
        queried = ("explicit_zeros", "implicit_zeros") if name == "cutoff" else (
            "csr", "csc", "coo", "split", "reversed", "float64", "dense", "dense_f64", "heldout",
        )
        identity[name] = {}
        for query in queried:
            result = model.transform(identity_queries[query].copy())
            identity[name][query] = {
                "same_object": result is model.embedding_,
                "same_coordinates": np.array_equal(result, model.embedding_, equal_nan=True),
                "dtype": result.dtype.name,
            }
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                      "scikit-learn", "pynndescent")},
        "options": OPTIONS,
        "training": {name: encode_sparse(values) for name, values in training.items()},
        "identity_queries": {
            name: encode_array(values) if isinstance(values, np.ndarray) else encode_sparse(values)
            for name, values in identity_queries.items()
        },
        "cases": cases,
        "identity": identity,
    }


def write_fixture(path, results, check):
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in results.items()
    ) + "\n}\n"
    if check:
        if path.read_text(encoding="utf-8") != output:
            raise SystemExit(f"{path} differs from the regenerated fixture")
    else:
        path.write_text(output, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spectral-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = build()
    spectral = results["cases"].pop("spectral_components")
    write_fixture(args.output, results, args.check)
    if args.spectral_output is not None:
        write_fixture(args.spectral_output, {"spectral_components": spectral}, args.check)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
