import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def load_reference(name):
    return json.loads(
        (FIXTURES / f"umap-learn-0.6.0-update-{name}.json").read_text(encoding="utf-8")
    )


REFERENCE = load_reference("windows-x86_64")
HOLDOUTS = load_reference("initialization-holdouts-x86_64")
LAYOUT = load_reference("layout-windows-x86_64")
INPUT_FIXTURES = (REFERENCE, HOLDOUTS)


def array_bits(record):
    return np.asarray(record["bits_uint32"], dtype=np.uint32).reshape(record["shape"])


def feature_input(fixture, name):
    return array_bits(fixture["inputs"][name]).view(np.float32)


def assert_embedding(actual, record):
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), array_bits(record))


def assert_graph(actual, record):
    assert isinstance(actual, csr_matrix)
    assert actual.dtype == np.float32
    assert actual.shape == tuple(record["shape"])
    np.testing.assert_array_equal(actual.indptr, record["indptr"])
    np.testing.assert_array_equal(actual.indices, record["indices"])
    np.testing.assert_array_equal(actual.data.view(np.uint32), record["data_bits_uint32"])


@pytest.mark.parametrize(
    ("fixture", "case"),
    [
        pytest.param(fixture, case, id=name)
        for fixture in INPUT_FIXTURES
        for name, case in fixture["cases"].items()
        if case["input_format"] == "dense"
        and case["updates"]
        and all(step["status"] == "ok" for step in case["updates"])
    ],
)
def test_update_matches_public_graph_and_embedding_after_each_batch(fixture, case):
    settings = fixture["base_parameters"] | case["parameter_overrides"]
    training = feature_input(fixture, case["fit_input"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = UMAP(**settings).fit(training)
    fit_neighbors = model.n_neighbors_
    for step in case["updates"]:
        batch = feature_input(fixture, step["input"])
        batch_before = batch.copy()
        previous_embedding = model.embedding_
        previous_bits = previous_embedding.view(np.uint32).copy()
        assert model.update(batch) is None
        np.testing.assert_array_equal(batch, batch_before)
        np.testing.assert_array_equal(previous_embedding.view(np.uint32), previous_bits)
        assert model.n_neighbors_ == fit_neighbors
        assert_embedding(model.embedding_, step["state"]["embedding"])
        assert_graph(model.graph_, step["state"]["graph"])
        training = np.concatenate((training, batch), axis=0)
        assert model.transform(training.copy()) is model.embedding_


@pytest.mark.parametrize(
    ("case", "trial"),
    [
        pytest.param(case, trial, id=f"{name}-{schedule}")
        for name, case in LAYOUT["cases"].items()
        for schedule, trial in case["trials"].items()
    ],
)
def test_update_replays_public_optimizer_with_fitted_curve_and_live_schedule(case, trial):
    model = UMAP(**(LAYOUT["fit_parameters"] | case["fit_overrides"])).fit(
        feature_input(REFERENCE, "training")
    )
    assert_embedding(model.embedding_, case["fit_embedding"])
    model.set_params(**trial["update_overrides"])
    assert model.update(feature_input(REFERENCE, case["update_input"]), True) is None
    assert_embedding(model.embedding_, trial["embedding"])
    assert_graph(model.graph_, case["updated_graph"])


@pytest.mark.parametrize(
    "name",
    [
        "before_fit", "one_training_row", "empty_update", "zero_features",
        "one_dimensional", "nan_update", "fewer_features", "more_features",
        "precomputed", "supervised", "graph_only",
    ],
)
def test_update_error_and_state_match_public_boundary(name):
    case = REFERENCE["cases"][name]
    step = case["updates"][0]
    settings = REFERENCE["base_parameters"] | case["parameter_overrides"]
    if name in ("precomputed", "supervised"):
        settings["transform_mode"] = "graph"
    model = UMAP(**settings)
    if case["fit_input"] is not None:
        training = feature_input(REFERENCE, case["fit_input"])
        labels = REFERENCE["labels"] if name == "supervised" else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(training, y=labels)
    original_graph = getattr(model, "graph_", None)
    original_embedding = getattr(model, "embedding_", None)
    batch = feature_input(REFERENCE, step["input"])
    errors = {"AttributeError": AttributeError, "ValueError": ValueError}
    with pytest.raises(errors[step["exception_class"]]) as failure:
        model.update(batch)
    assert str(failure.value) == step["exception_message"]
    if name == "graph_only":
        assert_graph(model.graph_, step["state"]["graph"])
        assert model.transform(training) is model.graph_
        assert model.transform(batch).shape == (1, len(training) + 1)
    else:
        assert getattr(model, "graph_", None) is original_graph
    assert getattr(model, "embedding_", None) is original_embedding
