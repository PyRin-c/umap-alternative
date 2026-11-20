import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from test_sparse_euclidean_transform import assert_record, decode
from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
REFERENCE = json.loads((FIXTURES / "umap-learn-0.6.0-boolean-metrics-x86_64.json").read_bytes())
RESIDUALS = json.loads((FIXTURES / "umap-learn-0.6.0-boolean-model-residuals-x86_64.json").read_bytes())


def read_inputs(spec):
    training, queries = map(decode, REFERENCE["inputs"][spec["input"]])
    if spec["form"] == "csr":
        training, queries = csr_matrix(training), csr_matrix(queries)
    return training, queries


def estimator(spec):
    cutoff = float("inf") if spec["cutoff"] == "inf" else spec["cutoff"]
    return UMAP(**(REFERENCE["options"] | {
        "metric": spec["metric"], "n_neighbors": spec["k"], "n_epochs": spec["epochs"],
        "transform_mode": spec["mode"], "disconnection_distance": cutoff,
    }))


def assert_call(action, record):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in record:
            assert record["error"] == "ValueError"
            with pytest.raises(ValueError):
                action()
        else:
            assert_record(action(), record["value"])
    assert [item.category.__name__ for item in seen] == record["warnings"]
    return seen


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_boolean_features_match_public_060(name):
    case = REFERENCE["cases"][name]
    spec = case["spec"]
    observed = REFERENCE["observations"][case["public"]]
    training, queries = read_inputs(spec)
    model = estimator(spec)
    fit_warnings = assert_call(lambda: model.fit_transform(training), observed["fit"])
    assert any(f"{spec['metric']} distance metric; inverse_transform" in str(item.message)
               for item in fit_warnings)
    if "fit_graph" in observed:
        assert_record(model.graph_, observed["fit_graph"])
    assert_call(lambda: model.transform(queries), observed["query"])
    if "mixed_query" in observed:
        alternate = queries.toarray() if issparse(queries) else csr_matrix(queries)
        assert_call(lambda: model.transform(alternate), observed["mixed_query"])
    if "fit_input_identity" in observed:
        assert (model.transform(training.copy()) is model.graph_) == observed["fit_input_identity"]
    if "without_zeros_identity" in observed:
        without_zeros = training.copy()
        without_zeros.eliminate_zeros()
        assert (model.transform(without_zeros) is model.graph_) == observed["without_zeros_identity"]
        assert observed["fit_storage_unchanged"]
        assert_record(training, REFERENCE["inputs"][spec["input"]][0])


@pytest.mark.parametrize("name", RESIDUALS["cases"])
def test_public_invalid_boolean_cases(name):
    case = RESIDUALS["cases"][name]
    observed = RESIDUALS["observations"][case["public"]]
    training, _ = read_inputs(case["spec"])
    model = estimator(case["spec"])
    assert_call(lambda: model.fit_transform(training), observed["fit"])
    if "fit_graph" in observed:
        assert_record(model.graph_, observed["fit_graph"])
    else:
        assert not hasattr(model, "graph_")


@pytest.mark.parametrize("form", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["embedding", "graph"])
def test_boolean_inverse_is_unavailable(form, mode):
    name = f"hamming_{form}_{'embedding_0' if mode == 'embedding' else 'default'}"
    spec = REFERENCE["cases"][name]["spec"]
    training, _ = read_inputs(spec)
    with pytest.warns(UserWarning, match="gradient function is not yet implemented"):
        model = estimator(spec).fit(training)
    reason = "sparse input" if form == "csr" else "given metric"
    with pytest.raises(ValueError, match=rf"^Inverse transform not available for {reason}\.$"):
        model.inverse_transform(np.zeros((1, 2), np.float32))


@pytest.mark.parametrize("form", ["dense", "csr"])
def test_boolean_embeddings_return_the_float64_input_dtype(form):
    case = REFERENCE["cases"][f"hamming_{form}_embedding_0"]
    training, queries = read_inputs(case["spec"])
    model = estimator(case["spec"])
    with pytest.warns(UserWarning, match="gradient function is not yet implemented"):
        fitted = model.fit_transform(training.astype(np.float64))
    observed = REFERENCE["observations"][case["public"]]
    assert fitted.dtype == model.transform(training.astype(np.float64)).dtype == np.float64
    np.testing.assert_array_equal(fitted, decode(observed["fit"]["value"]).astype(np.float64))
    with pytest.warns(RuntimeWarning):
        query = model.transform(queries.astype(np.float64))
    assert query.dtype == np.float64
    np.testing.assert_array_equal(query, decode(observed["query"]["value"]).astype(np.float64))


@pytest.mark.parametrize("form", ["dense", "csr"])
def test_boolean_sparse_queries_use_csr_buffers(monkeypatch, form):
    def fail(*args, **kwargs):
        pytest.fail("boolean CSR distances must keep features sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    case = REFERENCE["cases"][f"hamming_{form}_default"]
    training, queries = read_inputs(case["spec"])
    model = estimator(case["spec"])
    with pytest.warns(UserWarning, match="gradient function is not yet implemented"):
        model.fit(training)
    observed = REFERENCE["observations"][case["public"]]
    assert_record(model.transform(csr_matrix(queries)), observed["query"]["value"])
