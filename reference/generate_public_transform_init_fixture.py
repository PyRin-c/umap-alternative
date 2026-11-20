"""Record the public transform initialization functions of the pinned wheel."""

import argparse
from base64 import b64encode
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from umap.umap_ import init_graph_transform, init_transform


def encode(array):
    array = np.asarray(array)
    return {"shape": array.shape, "dtype": array.dtype.str,
            "bytes": b64encode(array.tobytes()).decode("ascii")}


def encode_graph(graph):
    axes = (graph.row, graph.col) if graph.format == "coo" else (graph.indptr, graph.indices)
    return {"type": type(graph).__name__, "shape": graph.shape,
            "indices": [encode(axis.astype(np.int64)) for axis in axes], "data": encode(graph.data)}


def observe(function, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            result = {"coordinates": encode(function(*args))}
        except Exception as error:
            result = {"error": type(error).__name__}
    result["warnings"] = [str(item.message) for item in seen]
    return result


def generate():
    assert version("umap-learn") == "0.6.0"
    cases = []

    def graph(name, source, embedding):
        cases.append({"name": name, "mode": "graph", "graph": encode_graph(source),
                      "embedding": encode(embedding), "observed": observe(init_graph_transform, source, embedding)})

    def ranked(name, indices, weights, embedding):
        cases.append({"name": name, "mode": "ranked", "indices": encode(indices), "weights": encode(weights),
                      "embedding": encode(embedding), "observed": observe(init_transform, indices, weights, embedding)})

    rng = np.random.RandomState(8241)
    for n in (2, 7, 8, 9, 127, 128, 129, 130, 258):
        indices = np.arange(n, dtype=np.int32)
        weights = rng.uniform(.0001, .99, n).astype(np.float32)
        embedding = (rng.normal(size=(n, 3)) * 8192).astype(np.float32)
        directed = sparse.csr_matrix((weights, indices, [0, n, n]), shape=(2, n))
        graph(f"directed-width-{n}", directed, embedding)
        ranked(f"ranked-width-{n}", indices[None, ::-1], weights[None, ::-1], embedding)
    for weights_type in (np.float32, np.float64, np.int16, np.uint64):
        for embedding_type in (np.float16, np.float32, np.float64, np.int16, np.int32, np.uint64):
            offset = 2**61 if embedding_type == np.uint64 else 0
            embedding = (offset + rng.randint(3, 4096, (9, 3)).astype(np.int64)).astype(embedding_type)
            if weights_type in (np.float32, np.float64):
                weights = rng.uniform(0.05, .95, 8).astype(weights_type)
            else:
                weights = rng.randint(1, 50, 8).astype(weights_type)
            indices = np.arange(8, dtype=np.int32)
            name = f"{weights_type.__name__}-{embedding_type.__name__}"
            matrix = sparse.csr_array((weights, indices, [0, 4, 8]), shape=(2, 9))
            graph(f"graph-{name}", matrix, embedding)
            ranked(f"ranked-{name}", indices.reshape(2, 4), weights.reshape(2, 4), embedding)

    finite = np.array([[7, 1], [13, 2], [23, 3], [31, 4]], dtype=np.float32)
    nonfinite = finite.copy()
    nonfinite[0, 0], nonfinite[1, 0] = np.nan, np.inf
    for name, data, columns, offsets in (
        ("zeros-and-empty", [0, 0, 0], [0, 1, 2], [0, 1, 1, 2, 3]),
        ("unit-order", [1, 1, .5], [3, 0, 1], [0, 3]),
        ("unit-with-negative", [1, .25, -1.25], [2, 1, 3], [0, 3]),
        ("duplicates", [.15, .45, .07, .09], [3, 2, 3, 0], [0, 4]),
        ("cancel", [.25, -.25, 0], [3, 2, 1], [0, 3]),
        ("signed-weight", [1.5, -.25], [2, 3], [0, 2]),
        ("infinite", [np.inf, .5], [0, 1], [0, 2]),
        ("opposite-infinite", [np.inf, -np.inf], [3, 1], [0, 2]),
        ("stored-nan", [np.nan, .3], [3, 1], [0, 2]),
        ("zero-before-nan", [0, .3], [0, 2], [0, 2]),
        ("subnormal", [np.nextafter(np.float32(0), np.float32(1))], [2], [0, 1]),
        ("positive-with-infinite-coordinate", [.4, .7], [1, 2], [0, 2]),
    ):
        value_type = np.float32
        source = sparse.csr_matrix((np.asarray(data, value_type), columns, offsets), shape=(len(offsets) - 1, 4))
        graph(name, source, nonfinite if name in ("zero-before-nan", "zeros-and-empty", "positive-with-infinite-coordinate") else finite)

    matrix = np.array([[0, .5, 1, 0], [0, 0, .25, 0], [0, .7, .6, 0], [.9, 0, 1, 0]], np.float32)
    for fmt in (sparse.csr_matrix, sparse.csr_array, sparse.csc_matrix, sparse.csc_array, sparse.coo_matrix):
        graph(f"format-{fmt.__name__}", fmt(matrix), finite)
    for shape in ((0, 0), (0, 4), (3, 0), (3, 4)):
        for embedding in (finite, finite[:0], finite[:, :0]):
            graph(f"empty-{shape}-{embedding.shape}", sparse.csr_array(shape, dtype=np.float32), embedding)
    for name, columns, weights in (("unit-with-short-fit", [2], [1]), ("absent-unit", [4], [1]),
                                   ("weighted-with-short-fit", [2], [.8])):
        graph(name, sparse.csr_matrix((np.asarray(weights, np.float32), columns, [0, 1]), shape=(1, 5)), finite)
    graph("rectangular-csc-wide", sparse.csc_array(matrix[:2]), finite)
    graph("rectangular-csc-narrow", sparse.csc_matrix(matrix[:, :3]), finite[:3])

    indices = np.array([[0, -1, 2], [3, 3, 1], [-4, 2, -2]], dtype=np.int64)
    weights = np.array([[0, 0, 1], [1, 1, .5], [0, -2, 3]], dtype=np.float32)
    ranked("ranked-negative-indices", indices, weights, finite)
    ranked("ranked-nonfinite-product", indices, weights, nonfinite)
    ranked("ranked-no-components", indices, weights, finite[:, :0])
    ranked("ranked-no-queries", indices[:0], weights[:0], finite)
    ranked("ranked-no-neighbors", indices[:, :0], weights[:, :0], finite)

    ints = np.array([[2**61 + 9, 2**62 + 34], [3, 4], [6, 7]], dtype=np.uint64)
    r = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int32)
    for kind in (np.int64, np.uint64):
        weight_int = np.array([[2**40 + 3, 2**32 + 5, 1], [9, 7, 3]], dtype=kind)
        ranked(f"ranked-64-bit-{kind.__name__}", r, weight_int, ints)
    return {"umap-learn": "0.6.0", "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = json.dumps(generate(), sort_keys=True, separators=(",", ":")) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != fixture:
            raise SystemExit("public transform initialization fixture differs")
    else:
        args.output.write_bytes(fixture.encode("utf-8"))


if __name__ == "__main__":
    main()
