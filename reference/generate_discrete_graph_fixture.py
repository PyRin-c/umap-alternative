"""Capture discrete target distances and graph operations on the public wheel."""

import argparse
from base64 import b64encode
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from numba import njit, types
from numba.typed import Dict, List
from scipy import sparse
from umap import distances, umap_


HOST_CASES = {
    "count-reversed", "count-default", "count-invalid-result",
    "count-0-2", "count-0-5", "count-1-2", "count-1-5",
    "count-negative-f64-lambda",
}


@njit
def scalar_metric(x, y, offset):
    return np.float32(abs(x - y) + offset)


@njit
def failing_metric(x, y):
    if x == 2:
        raise ValueError("callback failure")
    return abs(x - y)


def encode(value):
    value = np.asarray(value)
    return {"dtype": value.dtype.str, "shape": value.shape,
            "bytes": b64encode(value.tobytes()).decode("ascii")}


def encode_graph(graph):
    axes = (graph.row, graph.col) if graph.format == "coo" else (graph.indptr, graph.indices)
    return {"type": type(graph).__name__, "shape": graph.shape,
            "axes": [encode(axis.astype(np.int64)) for axis in axes], "data": encode(graph.data)}


def observe(function, args, convert=encode):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            returned = function(*args)
            observed = {"value": None if returned is None else convert(returned)}
        except Exception as error:
            observed = {"error": type(error).__name__}
    observed["warnings"] = [type(item.message).__name__ for item in seen]
    return observed


def generate():
    assert version("umap-learn") == "0.6.0"
    tests = {"graphs": [], "edges": [], "distances": [], "parameters": []}

    def distance(metric, name, *args):
        result = observe(distances.named_distances[metric], args)
        tests["distances"].append({"name": name, "metric": metric, "args": [encode(x) for x in args],
                                   "observed": result})

    def source(fmt="coo_matrix", dtype=np.float32):
        rows = [0, 0, 0, 1, 2, 2, 2, 3, 4, 4]
        cols = [2, 1, 3, 0, 1, 3, 3, 2, 0, 4]
        values = np.array([.25, .5, 0, .2, .1, .3, .1, .9, .337, .85], dtype=dtype)
        return getattr(sparse, fmt)(sparse.coo_matrix((values, (rows, cols)), shape=(5, 5)))

    def graph(name, matrix, target, **options):
        before = encode_graph(matrix)
        result = observe(lambda a, y: umap_.discrete_metric_simplicial_set_intersection(a, y, **options),
                         (matrix, target), encode_graph)
        tests["graphs"].append({"name": name, "graph": before, "target": encode(target),
                                "options": options, "observed": result, "after": encode_graph(matrix)})

    labels = np.array([0, 0, -1, 1, -1], dtype=np.int64)
    for fmt in ("coo_matrix", "csr_matrix", "csc_matrix", "coo_array", "csr_array", "csc_array"):
        for dtype in (np.float32, np.float64, np.int32):
            name = f"{fmt}-{dtype.__name__}"
            graph(name, source(fmt, dtype), labels, unknown_dist=.25, far_dist=1000.0)
    limit = float(np.nextafter(np.float32(0), np.float32(1)))
    for name, y, unknown, far in (
        ("unknown-zero", labels, 0.0, 5.0), ("unknown-infinite", labels, float("inf"), 5.0),
        ("subnormal-decay", labels, -np.log(limit), -np.log(limit)),
        ("negative-decay", labels, -.25, -1.0),
        ("large-unsigned", np.array([2**64-2, 2**64-1, 2**64-2, 2, 2**64-1], np.uint64), .125, 1.5),
        ("mixed-fractional", np.array([.12, -1., 1., 2., -1.], np.float32), .125, 1.5),
        ("NaN-label", np.array([np.nan, np.nan, -1., 0., 1.]), .125, 1.5),
        ("unicode-label", np.array(["a", "a", "a\U0001f642", "a\U0001f600", "a"]), .125, 1.5),
        ("boolean-label", np.array([True, False, True, False, True]), .125, 1.5),
    ):
        graph(name, source(), y, unknown_dist=unknown, far_dist=far)
    graph("numeric-default", source(), labels, metric="categorical", unknown_dist=.25, metric_scale=1.25)
    graph("ordinal-positional", source("csr_matrix", np.float64), labels, metric="ordinal",
          metric_kws={"not_a_parameter_name": 1.75}, metric_scale=.75)
    graph("ordinal-zero", source(), labels, metric="ordinal", metric_kws={"support_size": 0.})
    counts = np.array([11, 12, 14, 9, 10], dtype=np.int64)
    for name, kwargs in (("natural", {"poisson_lambda": 3.5, "normalisation": -7.5}),
                         ("reversed", {"normalisation": -7.5, "poisson_lambda": 3.5}), ("default", {})):
        graph(f"count-{name}", source("csr_matrix", np.float64), counts, metric="count", metric_kws=kwargs)
    graph("count-invalid-result", source(), labels, metric="count", metric_scale=1.25)
    values = np.array(["abcdefghij", "abcfedghit", "a\U0001f642", "a\U0001f600", ""], dtype="U12")
    for name in ("string", "myers"):
        graph(name, source(), values, metric=name, metric_scale=.9,
              metric_kws={"normalisation": 2.5, "max_distance": 2})
    for kind in (np.float32, np.float64):
        y = np.array([[0, 0], [3.2, 1.9], [2, .5], [-1.33, .25], [1.9, -1.33]], dtype=kind)
        for metric in ("euclidean", "l1", "linfinity", "hamming"):
            graph(f"vector-{metric}-{kind.__name__}", source("csr_array", np.float64), y, metric=metric, metric_scale=.225)
    graph("unknown-metric", source(), labels, metric="unknown")
    graph("empty-graph", sparse.coo_matrix((5, 5), dtype=np.float32), labels)
    graph("rectangle", sparse.coo_matrix(([.2, .6], ([0, 1], [1, 2])), shape=(2, 3)), labels)

    rng = np.random.RandomState(347)
    for case, lo in enumerate((-3, -1, 0, 1, 2, 7, 8, 9, 10, 11, 12, 63, 64, 65, 127, 128, 1024, 2**30+1, 2**53)):
        for gap, scale in ((1, 1.0), (2, .5), (5, -2.5)):
            distance("count", f"count-{case}-{gap}", lo, lo + gap, 4.0, scale)
    for case in range(32):
        lo = int(rng.randint(2, 999))
        distance("count", f"held-out-count-{case}", lo, lo + int(rng.randint(5, 20)),
                 float(rng.uniform(1., 32.)), float(rng.uniform(.1, 10.)))
    distance("count", "zero-lambda", 0, 2, 0.0, 1.0)
    distance("count", "nan-count", np.nan, 2, 1.0, 1.0)
    distance("count", "zero-normalisation", 0, 1, 2.0, 0.0)
    distance("count", "count-negative-f64-lambda", 1, 2, -7.5, 1.0)
    distance("count", "count-negative-f32-lambda", 1, 2, np.float32(-7.5), 1.0)
    distance("count", "fractional-count", 1.3, 7.5, np.float32(2.13), 1.0)
    for bound in (0, 1, 2, 4, 20):
        for nx, (a, b) in enumerate((('', ''), ('', 'abc'), ('decadea', 'ddbc'), ('abcd', 'abdc'),
                                    ('a' * 63 + 'b', 'a' * 63 + 'c'), ('x' * 65, 'b' * 65),
                                    ('a' * 64 + 'bc', 'a' * 64 + 'cd'), ('a\U0001f642', 'a\U0001f600'),
                                    ('e\u0301', '\u00e9'), ('xyz', 'abc'))):
            for metric in ("string", "myers"):
                distance(metric, f"{metric}-{nx}-{bound}", a, b, 2.5, bound)
    distance("ordinal", "ordinal-negative-support", 1.3, 1.5, -2.)
    distance("ordinal", "ordinal-f32", np.float32(.132), np.float32(.42), np.float32(.1349))
    distance("ordinal", "ordinal-unsigned-decreasing", np.uint64(1), np.uint64(2), 1.)
    distance("ordinal", "ordinal-zero-support", 0, 0, 0.)
    distance("categorical", "categorical-nan", np.nan, np.nan)
    for width in (61, 62, 63):
        left, right = "a" * width + "bc", "a" * width + "cd"
        distance("myers", f"myers-boundary-{width + 2}", left, right, 1.0, 1)
        distance("myers", f"myers-boundary-reversed-{width + 2}", right, left, 1.0, 1)
    distance("myers", "myers-mixed-length", "a" * 61 + "bc", "a" * 61 + "cde", 1.0, 1)

    layers = [[(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 0), (2, 1)], [(0, 0), (1, 0), (2, 0)]]
    typed = List()
    for layer in layers:
        values = Dict.empty(types.int64, types.int64)
        values.update(layer)
        typed.append(values)
    for x, y in ((0, 0), (0, 1), (1, 2), (1, 4)):
        result = observe(distances.hierarchical_categorical_distance, (x, y, typed))
        tests["distances"].append({"name": f"hierarchy-{x}-{y}", "metric": "hierarchical_categorical",
                                   "args": [encode(x), encode(y)], "layers": layers, "observed": result})

    rows = np.array([0, 1, 0, 2, 2], dtype=np.int32)
    cols = np.array([0, 0, 1, 0, 2], dtype=np.int32)
    for width in (np.float32, np.float64, np.int32):
        for option_width in (np.float32, np.float64):
            for operation, target, options in (
                ("categorical", np.array([0, -1, 1], np.int32), (option_width(.21314), option_width(.56781))),
                ("scalar", np.array([.1, .42019, -1], np.float32), (scalar_metric, (np.float32(.21314),), option_width(.56781))),
                ("failure", np.array([1, 3, 2], np.float64), (failing_metric, (), option_width(.75))),
            ):
                edge = np.array([1., .3, .8, .6, 1.], width)
                before = encode(edge)
                function = umap_.fast_intersection if operation == "categorical" else umap_.fast_metric_intersection
                result = observe(function, (rows, cols, edge, target, *options))
                arguments = [encode(x) for x in options] if operation == "categorical" else [encode(options[-1])]
                tests["edges"].append({"name": f"{operation}-{width.__name__}-{option_width.__name__}",
                  "operation": operation, "rows": encode(rows), "cols": encode(cols), "before": before,
                  "after": encode(edge), "target": encode(target), "args": arguments, "observed": result})

    for name, data in (("ordinal", [0., 1., 2., 4.5]), ("count", [3, 4, 9]),
                       ("count", [1, 1]), ("string", ['a', 'bbbbbb', 'cc']), ("myers", ['a']),
                       ("count", np.array([], np.float64)), ("string", []), ("ordinal", [])):
        data = np.asarray(data)
        result = observe(distances.get_discrete_params, (data, name), lambda item: {k: encode(v) for k, v in item.items()})
        tests["parameters"].append({"name": f"{name}-{len(tests['parameters'])}", "metric": name,
                                    "data": encode(data), "observed": result})
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = generate()
    host = {}
    for cases in fixture.values():
        for case in cases:
            if case["name"] in HOST_CASES:
                host[case["name"]] = {key: case.pop(key) for key in ("observed", "after") if key in case}
    for path, value in ((args.output, fixture), (args.host_output, host)):
        contents = json.dumps(value, separators=(",", ":")) + "\n"
        if args.check:
            if path.read_text(encoding="utf-8") != contents:
                raise SystemExit(f"discrete graph fixture differs: {path}")
        else:
            path.write_bytes(contents.encode("utf-8"))


if __name__ == "__main__":
    main()
