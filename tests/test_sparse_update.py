import json
import warnings
from pathlib import Path

import pytest
from scipy.sparse import csr_array, csr_matrix, vstack

from test_sparse_euclidean_transform import assert_record, decode, warning_records
from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-sparse-update-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
ERRORS = {"AttributeError": AttributeError, "TypeError": TypeError, "ValueError": ValueError}


def input_matrix(name):
    record = REFERENCE["inputs"][name]
    matrix = decode(record)
    return csr_array(matrix) if record["type"] == "csr_array" else matrix


def assert_state(model, expected):
    for attr, record in expected.items():
        value = getattr(model, attr, None)
        if record is None:
            assert value is None
        else:
            assert_record(value, record)


def transform_outcome(model, matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.transform(matrix.copy())
        except ValueError:
            return "ValueError"
    if result is getattr(model, "embedding_", None):
        return "embedding"
    if result is getattr(model, "graph_", None):
        return "graph"
    return list(result.shape)


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_sparse_euclidean_update_matches_public_060(name):
    case = REFERENCE["cases"][name]
    model = UMAP(**(REFERENCE["options"] | case["options"]))
    training = input_matrix(case["training"])
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        if case["fit_error"] is None:
            model.fit(training)
        else:
            with pytest.raises(ERRORS[case["fit_error"]]):
                model.fit(training)
    assert warning_records(observed) == case["fit_warnings"]
    assert_state(model, case["fit"])
    fitted_neighbors = min(model.n_neighbors, training.shape[0] - 1)
    if hasattr(model, "graph_"):
        assert model.n_neighbors_ == fitted_neighbors
    exact_input = training
    before = case["fit"]

    for step in case["steps"]:
        old_params = {name: getattr(model, name) for name in step["options"]}
        model.set_params(**step["options"])
        old_graph = getattr(model, "graph_", None)
        old_embedding = getattr(model, "embedding_", None)
        batch = input_matrix(step["batch"])
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            if step["status"] == "ok":
                assert model.update(batch, **step["kwargs"]) is None
            else:
                with pytest.raises(ERRORS[step["status"]]):
                    model.update(batch, **step["kwargs"])
        assert warning_records(observed) == step["warnings"]
        assert (getattr(model, "graph_", None) is not old_graph) == step["graph_replaced"]
        assert (getattr(model, "embedding_", None) is not old_embedding) == step["embedding_replaced"]
        assert_state(model, step["after"])
        if old_graph is not None:
            assert step["previous_graph_unchanged"]
            assert_record(old_graph, before["graph_"])
            assert model.n_neighbors_ == fitted_neighbors
        if old_embedding is not None:
            assert step["previous_embedding_unchanged"]
            assert_record(old_embedding, before["embedding_"])
        assert step["batch_unchanged"]
        assert_record(batch, REFERENCE["inputs"][step["batch"]])

        if "transform" in step:
            inputs = {"old_input": exact_input}
            if batch.shape[1] == exact_input.shape[1] and batch.shape[0] > 0:
                inputs["combined_input"] = vstack((exact_input, csr_matrix(batch)), format="csr")
            for label, outcome in step["transform"].items():
                assert transform_outcome(model, inputs[label]) == outcome
            if "stale_layout_transform" in step:
                # Held-out queries cannot use a graph larger than the retained embedding.
                assert transform_outcome(model, inputs["combined_input"]) == "ValueError"
            if step["status"] == "ok":
                exact_input = inputs["combined_input"]
        model.set_params(**old_params)
        before = step["after"]

    assert_record(training, REFERENCE["inputs"][case["training"]])


def test_sparse_update_rejects_nonfinite_rank_inputs_with_validation_disabled():
    case = REFERENCE["residuals"]["nan_allowed"]
    model = UMAP(**REFERENCE["options"]).fit(input_matrix(case["training"]))
    previous_graph = model.graph_
    previous_embedding = model.embedding_
    step = case["steps"][0]
    with pytest.raises(ValueError):
        model.update(input_matrix(step["batch"]), **step["kwargs"])
    assert model.graph_ is previous_graph
    assert model.embedding_ is previous_embedding
    assert_state(model, case["fit"])


def test_wide_sparse_update_uses_csr_buffers(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("sparse Euclidean update must keep features sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    model = UMAP(**REFERENCE["options"]).fit(input_matrix("wide"))
    model.update(input_matrix("wide_append"))
    assert model.embedding_.shape == (9, 2)
