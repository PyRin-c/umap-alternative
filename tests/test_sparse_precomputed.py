import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-sparse-precomputed-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
ERRORS = {"AssertionError": AssertionError, "ValueError": ValueError}


def matrix(record):
    data = np.asarray(record["bits"], dtype=np.uint32).view(np.float32)
    if record["kind"] == "dense":
        return data.reshape(record["shape"])
    return csr_matrix((data, record["indices"], record["indptr"]), shape=record["shape"])


def assert_matrix(actual, expected):
    assert actual.dtype == np.float32
    assert actual.shape == tuple(expected["shape"])
    if expected["kind"] == "dense":
        assert isinstance(actual, np.ndarray)
        data = actual
    elif expected["kind"] == "csr":
        assert isinstance(actual, csr_matrix)
        np.testing.assert_array_equal(actual.indptr, expected["indptr"])
        np.testing.assert_array_equal(actual.indices, expected["indices"])
        data = actual.data
    else:
        assert isinstance(actual, coo_matrix)
        np.testing.assert_array_equal(actual.row, expected["row"])
        np.testing.assert_array_equal(actual.col, expected["col"])
        data = actual.data
    np.testing.assert_array_equal(data.view(np.uint32).ravel(), expected["bits"])


def check_call(call, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(ERRORS[expected["error"]]) as error:
                call()
            assert str(error.value) == expected["message"]
        else:
            assert_matrix(call(), expected["result"])
    assert [[item.category.__name__, str(item.message)] for item in caught] == expected["warnings"]


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_sparse_precomputed_matches_public_060(name):
    case = REFERENCE["cases"][name]
    training = REFERENCE["training"][case["training"]]
    model = UMAP(**(REFERENCE["options"] | case["options"]), transform_mode=case["mode"])
    fit_input = matrix(training)
    check_call(lambda: model.fit_transform(fit_input), case["fit"])
    assert_matrix(fit_input, case.get("training_after", training))
    if "result" not in case["fit"]:
        return
    assert_matrix(model.graph_, case.get("graph", case["fit"]["result"]))
    fitted = model.graph_ if case["mode"] == "graph" else model.embedding_
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (model.transform(fit_input.copy()) is fitted) == case["identity"]
    assert not caught
    for query_name, expected in case["queries"].items():
        query = REFERENCE["queries"][query_name]
        values = matrix(query)
        check_call(lambda: model.transform(values), expected)
        assert_matrix(values, query)


@pytest.mark.parametrize("mode", ["graph", "embedding"])
def test_sparse_precomputed_handoff_does_not_densify(monkeypatch, mode):
    def fail(*args, **kwargs):
        pytest.fail("CSR distances must stay sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    training = matrix(REFERENCE["training"]["implicit_diagonal"])
    query = matrix(REFERENCE["queries"]["duplicates"])
    model = UMAP(**REFERENCE["options"], transform_mode=mode)
    with pytest.warns(UserWarning, match="using precomputed metric"):
        model.fit(training)
    with pytest.warns(UserWarning, match="Transforming new data with precomputed metric"):
        model.transform(query)


def test_sparse_precomputed_identity_uses_normalized_csr_storage():
    model = UMAP(**REFERENCE["options"], transform_mode="graph")
    training = matrix(REFERENCE["training"]["full"])
    with pytest.warns(UserWarning, match="using precomputed metric"):
        model.fit(training.tocsc())
    assert model.transform(training.copy()) is model.graph_
    assert model.transform(training.tocoo()) is model.graph_
    for values in (training.toarray(), matrix(REFERENCE["training"]["implicit_diagonal"])):
        with pytest.warns(UserWarning, match="Transforming new data with precomputed metric"):
            assert isinstance(model.transform(values), coo_matrix)
