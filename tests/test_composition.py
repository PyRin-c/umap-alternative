import json
import operator
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from umap_alternative import UMAP


_FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
_FIXTURE = json.loads((_FIXTURES / "umap-learn-0.6.0-composition-x86_64.json").read_text(
    encoding="utf-8"
))
_PLATFORM = {"win32": "windows", "linux": "ubuntu"}.get(sys.platform)
_BRIDGE_LAYOUTS = {} if _PLATFORM is None else json.loads(
    (_FIXTURES / f"umap-learn-0.6.0-composition-bridge-{_PLATFORM}-x86_64.json")
    .read_text(encoding="utf-8")
)
_OPERATIONS = {"union": operator.add, "intersection": operator.mul, "difference": operator.sub}


def _graph_bits(graph):
    return [graph.indptr.tolist(), graph.indices.tolist(), graph.data.view(np.uint32).tolist()]


def _operands(case):
    features = [np.asarray(values, dtype=np.float32) for values in _FIXTURE["features"]]
    operands = [UMAP(**options).fit(data) for options, data in zip(case["options"], features)]
    if case["override_graphs"]:
        for operand, (indptr, indices, bits) in zip(operands, case["operands"]):
            operand.graph_ = csr_matrix(
                (np.asarray(bits, dtype=np.uint32).view(np.float32), indices, indptr), shape=(8, 8)
            )
    return operands


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("name", _OPERATIONS)
def test_public_composition_checkpoints(case, name):
    left, right = _operands(case)
    assert [_graph_bits(left.graph_), _graph_bits(right.graph_)] == case["operands"]
    expected = case["outputs"][name]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        combined = _OPERATIONS[name](left, right)
    assert [notice.category.__name__ for notice in seen] == expected["warnings"]
    assert [_graph_bits(left.graph_), _graph_bits(right.graph_)] == expected["operands_after"]
    assert _graph_bits(combined.graph_) == expected["graph"]
    assert combined.embedding_.shape == (8, 2)
    coordinates = (_BRIDGE_LAYOUTS.get(name) if case["name"] == "pruned_bridge"
                   else expected["embedding"])
    if coordinates is not None:
        np.testing.assert_array_equal(combined.embedding_.view(np.uint32), coordinates)
    assert combined.n_components == (2, 3)
    assert combined.random_state == (34, 88)
    assert combined.compatibility_layout is True


def test_nested_composition_and_epoch_lists():
    left, right = _operands(_FIXTURE["cases"][0])
    left.n_epochs = [0, 2]
    right.n_epochs = None
    nested = (left + right) * right
    assert nested.n_components == (2, 3, 3)
    assert nested.n_neighbors == (4, 4, 4)
    assert nested.n_epochs == (0, 2, -1, -1)
    assert nested.embedding_.shape == (8, 2)


@pytest.mark.parametrize("name", _OPERATIONS)
def test_composition_validates_estimators_and_graph_size(name):
    operation = _OPERATIONS[name]
    left, right = _operands(_FIXTURE["cases"][0])
    with pytest.raises(NotFittedError):
        operation(left, UMAP())
    with pytest.raises(TypeError):
        operation(left, None)
    right.graph_ = right.graph_[:-1, :-1].tocsr()
    with pytest.raises(ValueError, match="same number of vertices"):
        operation(left, right)


@pytest.mark.parametrize("name", _OPERATIONS)
def test_composition_rejects_empty_operands(name):
    left, right = _operands(_FIXTURE["cases"][0])
    right.graph_ = csr_matrix(right.graph_.shape, dtype=np.float32)
    with pytest.raises(ValueError):
        _OPERATIONS[name](left, right)
    with pytest.raises(ValueError):
        _OPERATIONS[name](right, left)


def test_subtraction_can_mutate_left_before_empty_result_error():
    left, _ = _operands(_FIXTURE["cases"][0])
    left.graph_.data[:] = 1
    with pytest.raises(ValueError):
        left - left
    assert not np.any(left.graph_.data)
