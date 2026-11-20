from base64 import b64decode
import builtins
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy import sparse

from umap_alternative.umap_ import init_graph_transform, init_transform


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-public-transform-init-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())


def decode(record):
    return np.frombuffer(b64decode(record["bytes"]), dtype=record["dtype"]).reshape(record["shape"]).copy()


def graph(record):
    constructor = getattr(sparse, record["type"])
    first, second = map(decode, record["indices"])
    data = decode(record["data"])
    components = (data, (first, second)) if record["type"].startswith("coo") else (data, second, first)
    return constructor(components, shape=record["shape"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_public_transform_initialization_matches_wheel(case):
    embedding = decode(case["embedding"])
    if case["mode"] == "graph":
        function = init_graph_transform
        other = (graph(case["graph"]),)
    else:
        function = init_transform
        other = (decode(case["indices"]), decode(case["weights"]))
    expected = case["observed"]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(getattr(builtins, expected["error"])):
                function(*other, embedding)
        else:
            result = function(*other, embedding)
            np.testing.assert_array_equal(result.view(np.uint32), decode(expected["coordinates"]).view(np.uint32))
            assert result.dtype == np.float32
    assert [str(item.message) for item in seen] == expected["warnings"]


def test_ranked_initialization_validates_checked_shape_and_index_boundaries():
    coordinates = np.ones((2, 2), dtype=np.float32)
    weights = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(IndexError, match="outside the fitted embedding"):
        init_transform(np.array([[2]]), weights, coordinates)
    with pytest.raises(IndexError, match="outside the fitted embedding"):
        init_transform(np.array([[-3]]), weights, coordinates)
    with pytest.raises(ValueError, match="incompatible transform initialization shapes"):
        init_transform(np.array([[0, 1]]), weights, coordinates)
