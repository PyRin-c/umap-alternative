from base64 import b64decode
import builtins
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from scipy import sparse

from umap_alternative import distances, umap_


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "umap-learn-0.6.0-discrete-graph-x86_64.json"
CASES = json.loads(FIXTURE.read_bytes())
HOST = "windows" if sys.platform == "win32" else "ubuntu"
VARIANTS = json.loads(FIXTURE.with_name(f"umap-learn-0.6.0-discrete-graph-{HOST}-x86_64.json").read_bytes())
for group in CASES.values():
    for case in group:
        case.update(VARIANTS.get(case["name"], {}))


def decode(record):
    return np.frombuffer(b64decode(record["bytes"]), dtype=record["dtype"]).reshape(record["shape"]).copy()


def argument(record):
    value = decode(record)
    return value[()] if value.ndim == 0 else value


def graph(record):
    first, second = (decode(axis) for axis in record["axes"])
    data = decode(record["data"])
    buffer = (data, (first, second)) if record["type"].startswith("coo") else (data, second, first)
    return getattr(sparse, record["type"])(buffer, shape=record["shape"])


def assert_array(actual, record):
    expected = decode(record)
    value = np.asarray(actual, dtype=expected.dtype)
    assert value.shape == expected.shape
    np.testing.assert_array_equal(value.ravel().view(np.uint8), expected.ravel().view(np.uint8))


def assert_graph(actual, record):
    assert type(actual).__name__ == record["type"]
    assert actual.shape == tuple(record["shape"])
    axes = (actual.row, actual.col) if actual.format == "coo" else (actual.indptr, actual.indices)
    for axis, expected in zip(axes, record["axes"]):
        assert_array(axis, expected)
    assert actual.data.dtype == np.dtype(record["data"]["dtype"])
    assert_array(actual.data, record["data"])


def assert_observation(function, args, observed, assert_value=assert_array):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in observed:
            name = "TypeError" if observed["error"] == "TypingError" else observed["error"]
            with pytest.raises(getattr(builtins, name)):
                function(*args)
        else:
            actual = function(*args)
            if observed["value"] is None:
                assert actual is None
            else:
                assert_value(actual, observed["value"])
    assert [type(item.message).__name__ for item in seen] == observed["warnings"]


@pytest.mark.parametrize("case", CASES["graphs"], ids=lambda case: case["name"])
def test_public_discrete_graph_and_mutation(case):
    source = graph(case["graph"])

    def intersect(matrix, labels):
        return umap_.discrete_metric_simplicial_set_intersection(matrix, labels, **case["options"])

    assert_observation(intersect, (source, decode(case["target"])), case["observed"], assert_graph)
    assert_graph(source, case["after"])


def scalar_metric(x, y, offset):
    return np.float32(abs(x - y) + offset)


def failing_metric(x, y):
    if x == 2:
        raise ValueError("callback failure")
    return abs(x - y)


@pytest.mark.parametrize("case", CASES["edges"], ids=lambda case: case["name"])
def test_public_edge_attenuation(case):
    weights = decode(case["before"])
    base = (decode(case["rows"]), decode(case["cols"]), weights, decode(case["target"]))
    arguments = tuple(argument(record) for record in case["args"])
    if case["operation"] == "categorical":
        function = umap_.fast_intersection
    else:
        function = umap_.fast_metric_intersection
        metric = (scalar_metric, (np.float32(.21314),)) if case["operation"] == "scalar" else (failing_metric, ())
        arguments = metric + arguments
    assert_observation(function, base + arguments, case["observed"])
    assert_array(weights, case["after"])


@pytest.mark.parametrize("case", CASES["distances"], ids=lambda case: case["name"])
def test_public_named_distance(case):
    args = [argument(record) for record in case["args"]]
    if "layers" in case:
        args.append([dict(layer) for layer in case["layers"]])
    assert_observation(distances.named_distances[case["metric"]], args, case["observed"])


@pytest.mark.parametrize("case", CASES["parameters"], ids=lambda case: case["name"])
def test_public_discrete_parameters(case):
    def assert_params(actual, expected):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_array(actual[key], expected[key])

    args = (decode(case["data"]), case["metric"])
    assert_observation(distances.get_discrete_params, args, case["observed"], assert_params)
