import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from umap_alternative import UMAP, _native


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-sparse-euclidean-transform-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
LP_REFERENCE = json.loads(FIXTURE.with_name(
    "umap-learn-0.6.0-sparse-lp-transform-x86_64.json"
).read_bytes())


def decode(record):
    if "format" not in record:
        dtype = np.dtype(record["dtype"])
        bits = np.asarray(record["bits"], dtype=f"uint{dtype.itemsize * 8}")
        return bits.view(dtype).reshape(record["shape"])
    data = decode(record["data"])
    if record["format"] == "coo":
        return coo_matrix((data, (record["row"], record["col"])), shape=record["shape"])
    constructor = csr_matrix if record["format"] == "csr" else csc_matrix
    return constructor((data, record["indices"], record["indptr"]), shape=record["shape"])


def assert_record(actual, expected):
    assert actual.shape == tuple(expected["shape"])
    if "format" not in expected:
        assert actual.dtype == np.dtype(expected["dtype"])
        np.testing.assert_array_equal(
            actual.view(f"uint{actual.itemsize * 8}").ravel(), expected["bits"]
        )
        return
    assert actual.format == expected["format"]
    if actual.format == "coo":
        assert isinstance(actual, coo_matrix)
        np.testing.assert_array_equal(actual.row, expected["row"])
        np.testing.assert_array_equal(actual.col, expected["col"])
    else:
        np.testing.assert_array_equal(actual.indptr, expected["indptr"])
        np.testing.assert_array_equal(actual.indices, expected["indices"])
    assert_record(actual.data, expected["data"])


def warning_records(caught):
    return [[item.category.__name__, str(item.message)] for item in caught]


def check_transform_case(reference, name):
    case = reference["cases"][name]
    training = decode(reference["training"][case["training"]])
    model = UMAP(**(reference["options"] | case["options"]), transform_mode=case["fit_mode"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(training)
    assert warning_records(caught) == case["fit_warnings"]
    model.set_params(transform_mode=case["mode"])
    fitted = model.graph_ if case["mode"] == "graph" else model.embedding_
    assert_record(fitted, case["fitted"])
    if "fitted_graph" in case:
        assert_record(model.graph_, case["fitted_graph"])
    for expected in case["transforms"].values():
        original = {key: getattr(model, key) for key in expected["parameters"]}
        model.set_params(**expected["parameters"])
        query = reference["queries"][expected["query"]]
        X = decode(query)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in expected:
                assert expected["error"] == "ValueError"
                with pytest.raises(ValueError) as error:
                    model.transform(X)
                assert str(error.value) == expected["message"]
            else:
                transformed = model.transform(X)
                assert (transformed is fitted) == expected["identity"]
                if not expected["identity"]:
                    assert_record(transformed, expected["result"])
        assert warning_records(caught) == expected["warnings"]
        assert_record(X, query)
        model.set_params(**original)
    assert_record(fitted, case["fitted"])
    if "fitted_graph" in case:
        assert_record(model.graph_, case["fitted_graph"])


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_sparse_euclidean_transform_matches_public_060(name):
    check_transform_case(REFERENCE, name)


@pytest.mark.parametrize("name", LP_REFERENCE["cases"])
def test_sparse_manhattan_and_chebyshev_match_public_060(name):
    check_transform_case(LP_REFERENCE, name)


@pytest.mark.parametrize("mode", ["graph", "embedding"])
def test_identical_sparse_features_bypass_neighbor_search(monkeypatch, mode):
    def fail(*args, **kwargs):
        pytest.fail("fitted input must bypass the query search")

    monkeypatch.setattr(_native, "_rank_sparse_feature_queries", fail)
    X = decode(REFERENCE["training"]["csr"])
    model = UMAP(**REFERENCE["options"], transform_mode=mode).fit(X)
    fitted = model.graph_ if mode == "graph" else model.embedding_
    assert model.transform(X.tocoo()) is fitted


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_wide_sparse_transform_uses_sparse_feature_buffers(monkeypatch, metric):
    def fail(*args, **kwargs):
        pytest.fail("CSR feature buffers must stay sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    training = decode(REFERENCE["training"]["wide_features"])
    queries = decode(REFERENCE["queries"]["wide_features"])
    model = UMAP(**(REFERENCE["options"] | {"metric": metric})).fit(training)
    assert model.transform(queries).shape == (3, 2)


def test_wider_sparse_query_is_rejected():
    case = REFERENCE["residuals"]["wider_query"]
    training = decode(REFERENCE["training"][case["training"]])
    query = decode(REFERENCE["queries"]["width_residual"])
    model = UMAP(**(REFERENCE["options"] | case["options"]), transform_mode="graph")
    with pytest.raises(ValueError, match="X has 4 features, but UMAP was fit with 3"):
        model.fit(training).transform(query)


def test_query_ties_match_after_the_graph_only_fit():
    case = REFERENCE["residuals"]["graph_only_ties"]
    training = decode(REFERENCE["training"][case["training"]])
    expected = case["transforms"]["wide_ties"]["result"]
    model = UMAP(**REFERENCE["options"], transform_mode="graph").fit(training)
    fitted = model.graph_.data.copy()
    assert_record(model.transform(decode(REFERENCE["queries"]["wide_ties"])), expected)
    np.testing.assert_array_equal(model.graph_.data, fitted)
