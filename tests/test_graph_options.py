import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from umap_alternative import UMAP


FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "reference/fixtures/umap-learn-0.6.0-estimator-graph-options-x86_64.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[
    f"{case['kind']}-{case['settings'].get('init', 'graph')}-{index}"
    for index, case in enumerate(FIXTURE["cases"])
])
def test_graph_options_fit_matches_public_bits(case, method):
    settings = FIXTURE["options"] | case["settings"]
    if settings["init"] == "explicit":
        settings["init"] = np.asarray(FIXTURE["initial"], dtype=np.float32)
    kind = case["kind"]
    X = np.asarray(FIXTURE["X"], dtype=np.float32)
    y = None
    if kind == "categorical":
        y = np.asarray(FIXTURE["labels"])
    elif kind == "sparse":
        X = csr_matrix(X)
    elif kind == "distances":
        X = np.asarray(FIXTURE["distances"], dtype=np.float32)
        settings["metric"] = "precomputed"
    model = UMAP(**settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        returned = getattr(model, method)(X, y=y)
    assert model.graph_.shape == (len(FIXTURE["X"]),) * 2
    assert model.graph_.dtype == np.float32
    np.testing.assert_array_equal(model.graph_.indptr, case["indptr"])
    np.testing.assert_array_equal(model.graph_.indices, case["indices"])
    np.testing.assert_array_equal(model.graph_.data.view(np.uint32), case["weight_bits"])
    if "coordinates_bits" in case:
        np.testing.assert_array_equal(model.embedding_.view(np.uint32), case["coordinates_bits"])
        assert returned is (model if method == "fit" else model.embedding_)
    else:
        assert returned is (model if method == "fit" else model.graph_)


@pytest.mark.parametrize("case", [case for case in FIXTURE["cases"] if "query_graph" in case])
def test_graph_transform_uses_current_local_connectivity(case):
    X = np.asarray(FIXTURE["X"], dtype=np.float32)
    queries = np.asarray(FIXTURE["queries"], dtype=np.float32)
    model = UMAP(**FIXTURE["options"]).fit(X)
    model.set_params(**case["settings"])
    graph = model.transform(queries)
    expected = case["query_graph"]
    assert graph.shape == (len(queries), len(X))
    np.testing.assert_array_equal(graph.row, expected["row"])
    np.testing.assert_array_equal(graph.col, expected["col"])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["weight_bits"])
