import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from umap_alternative import UMAP


_FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
_FIXTURE = json.loads((_FIXTURES / "umap-learn-0.6.0-ratio-metrics-x86_64.json").read_text(encoding="utf-8"))
_PLATFORM = {"win32": "windows", "linux": "ubuntu"}.get(sys.platform)
_PLATFORM_WEIGHTS = {} if _PLATFORM is None else json.loads(
    (_FIXTURES / f"umap-learn-0.6.0-ratio-metrics-query-{_PLATFORM}-x86_64.json").read_text(encoding="utf-8")
)


def _case_name(case):
    options = case["options"]
    kind = "csr" if case["sparse"] else "dense"
    mode = options.get("transform_mode", "embedding")
    return f"{options['metric']}-{kind}-{mode}-{case['dataset']}-{options.get('disconnection_distance')}"


def _assert_graph(graph, expected):
    assert graph.format == expected["format"]
    assert graph.shape == tuple(expected["shape"])
    buffers = (graph.indptr, graph.indices) if graph.format == "csr" else (graph.row, graph.col)
    for actual, target in zip(buffers, expected["coordinates"]):
        np.testing.assert_array_equal(actual, target)
    weights = expected.get("bits", _PLATFORM_WEIGHTS.get(expected.get("platform_weights")))
    if weights is not None:
        np.testing.assert_array_equal(graph.data.view(np.uint32), weights)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=_case_name)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_public_ratio_metrics(case, method):
    x, queries = _FIXTURE["data"][case["dataset"]]
    convert = csr_matrix if case["sparse"] else np.asarray
    x = convert(np.asarray(x, dtype=np.float32))
    model = UMAP(**case["options"])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = getattr(model, method)(x)
    assert [notice.category.__name__ for notice in seen] == case["fit_warnings"]
    _assert_graph(model.graph_, case["graph"])
    if "embedding" in case:
        np.testing.assert_array_equal(model.embedding_.view(np.uint32), case["embedding"])
        if method == "fit_transform":
            assert result is model.embedding_
    elif method == "fit_transform":
        assert result is model.graph_
    if "transform" in case:
        q = convert(np.asarray(queries, dtype=np.float32))
        result = model.transform(q)
        if model.transform_mode == "graph":
            _assert_graph(result, case["transform"])
        else:
            np.testing.assert_array_equal(result.view(np.uint32), case["transform"])
    if "inverse" in case:
        points = np.asarray(case["inverse_points"], dtype=np.uint32).view(np.float32)
        model.set_params(n_epochs=48)
        np.testing.assert_array_equal(model.inverse_transform(points).view(np.uint32), case["inverse"])
    if "updated_graph" in case:
        model.set_params(n_epochs=2)
        assert model.update(q) is None
        _assert_graph(model.graph_, case["updated_graph"])
        np.testing.assert_array_equal(model.embedding_.view(np.uint32), case["updated_embedding"])
