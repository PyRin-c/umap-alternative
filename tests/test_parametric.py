import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from umap_alternative import UMAP
from umap_alternative.parametric_umap import _prepare_edges


def test_parametric_namespace_and_singleton_without_backend():
    code = """
import sys


class BlockOptionalBackend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'keras'}:
            raise ModuleNotFoundError(fullname)


sys.meta_path.insert(0, BlockOptionalBackend())
from umap_alternative import ParametricUMAP, UMAP
from umap_alternative.parametric_umap import ParametricUMAP as ModuleEstimator

assert ModuleEstimator is ParametricUMAP
assert issubclass(ParametricUMAP, UMAP)
estimator = ParametricUMAP(n_neighbors=2, n_epochs=10)
embedding = estimator.fit_transform([[0, 1]])
assert embedding.shape == (1, 2)
assert embedding.dtype.name == 'float64'
assert not embedding.any()
try:
    estimator.fit([[0, 1], [1, 1], [2, 2], [3, 4]])
except ImportError as exc:
    assert 'TensorFlow and Keras' in str(exc)
else:
    raise AssertionError('multi-row training should require the optional backend')
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


_STAGE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "reference/fixtures/umap-learn-0.6.0-parametric-stages-windows-x86_64.json"
)
_EDGE_CASES = json.loads(_STAGE_FIXTURE.read_text(encoding="utf-8"))["edge_periods"]


@pytest.mark.parametrize("case", _EDGE_CASES)
def test_python_native_positive_edges(case):
    expected = _EDGE_CASES[case]
    n_rows = expected["n_rows"]
    features = np.array(expected["input_bits"], dtype=np.uint32).view(np.float32).reshape(n_rows, -1)
    model = UMAP(n_neighbors=4, transform_mode="graph", **expected["graph_options"]).fit(features)
    graph = model.graph_
    np.testing.assert_array_equal(graph.indptr, expected["indptr"])
    np.testing.assert_array_equal(graph.indices, expected["indices"])
    expected_weights = np.asarray(expected["weight_bits"], dtype=np.uint32).view(np.float32)
    np.testing.assert_allclose(graph.data, expected_weights, rtol=1e-6, atol=0)
    heads = np.repeat(np.arange(n_rows), np.diff(graph.indptr))
    expected_edges = [
        (head, tail, count)
        for head, tail, count in zip(heads, graph.indices, expected["observed_counts"], strict=True)
        if count
    ]
    edges, total = _prepare_edges(graph, expected["resolved_epochs"])
    assert edges == expected_edges
    assert total == expected["total_repeats"]
