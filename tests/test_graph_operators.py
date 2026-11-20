from base64 import b64decode
import builtins
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy import sparse as scipy_sparse

from umap_alternative import sparse, umap_


PATH = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-graph-operators-x86_64.json"
)
REFERENCE = json.loads(PATH.read_bytes())
OPERATIONS = {
    "union": umap_.general_simplicial_set_union,
    "intersection": umap_.general_simplicial_set_intersection,
    "reset": umap_.reset_local_connectivity,
}


def array(record):
    return np.frombuffer(b64decode(record["bytes"]), dtype=record["dtype"]).copy()


def graph(record):
    if record is None:
        return None
    constructor = getattr(scipy_sparse, record["type"])
    first, second = map(array, record["indices"])
    data = array(record["data"])
    values = (data, (first, second)) if record["type"].startswith("coo") else (data, second, first)
    return constructor(values, shape=record["shape"])


def assert_graph(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert type(actual).__name__ == expected["type"]
    assert actual.shape == tuple(expected["shape"])
    names = ("row", "col") if actual.format == "coo" else ("indptr", "indices")
    for name, values in zip(names, expected["indices"]):
        np.testing.assert_array_equal(getattr(actual, name), array(values))
    assert actual.data.dtype.str == expected["data"]["dtype"]
    np.testing.assert_array_equal(actual.data.view(np.uint8), array(expected["data"]).view(np.uint8))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_graph_operators_match_locked_public_wheel(case):
    original_left, original_right = REFERENCE["inputs"][case["input"]]
    left, right = graph(original_left), graph(original_right)
    if case["same_operand"]:
        right = left
    expected = case["observed"]
    op = OPERATIONS[case["operation"]]
    args = (left,) if case["operation"] == "reset" else (left, right)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(getattr(builtins, expected["error"])):
                op(*args, **case["options"])
        else:
            result = op(*args, **case["options"])
            assert_graph(result, expected["result"])
            assert (result is left) == expected["returns_left"]
            assert np.shares_memory(result.data, left.data) == expected["shares_left"]
    assert [type(item.message).__name__ for item in seen] == expected["warnings"]
    assert_graph(left, expected.get("left", original_left))
    assert_graph(right, expected.get("right", original_right))


@pytest.mark.parametrize("case", REFERENCE["raw"], ids=lambda case: case["name"])
def test_sparse_buffer_operators_preserve_storage_and_match_public_wheel(case):
    left, right = map(graph, REFERENCE["inputs"][case["input"]])
    op = getattr(sparse, f"general_sset_{case['operation']}")
    result = array(case["before"])
    out = op(left.indptr, left.indices, left.data, right.indptr, right.indices, right.data,
             array(case["rows"]), array(case["columns"]), result, **case["options"])
    assert out is None
    expected = array(case["after"])
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
