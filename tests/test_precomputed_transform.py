import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-precomputed-transform-windows-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())


def matrix(record):
    return np.asarray(record["bits"], dtype=np.uint32).view(np.float32).reshape(record["shape"])


def warning_records(caught):
    return [[item.category.__name__, str(item.message)] for item in caught]


def assert_bits(actual, expected):
    assert actual.dtype == np.float32
    assert actual.shape == tuple(expected["shape"])
    np.testing.assert_array_equal(actual.view(np.uint32).ravel(), expected["bits"])


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_rectangular_precomputed_transform_matches_public_bits(name):
    record = REFERENCE["cases"][name]
    sample = REFERENCE["inputs"][record["dataset"]]
    training = matrix(sample["training"])
    queries = matrix(sample["queries"])
    model = UMAP(**(REFERENCE["options"] | record["options"]),
                 transform_mode=record["mode"], n_epochs=record["epochs"])
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        model.fit(training)
    assert warning_records(messages) == record["fit_warnings"]
    assert_bits(training, record.get("training_after", sample["training"]))
    np.testing.assert_array_equal(model.graph_.indptr, record["graph"]["indptr"])
    np.testing.assert_array_equal(model.graph_.indices, record["graph"]["indices"])
    assert_bits(model.graph_.data, record["graph"]["data"])

    fitted = model.graph_ if record["mode"] == "graph" else model.embedding_
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        assert (model.transform(matrix(sample["training"])) is fitted) == record["identity"]
    assert warning_records(messages) == record["identity_warnings"]
    if "training_after" in record:
        with pytest.raises(ValueError, match="Input contains infinity"):
            model.transform(training.copy())
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        actual = model.transform(queries)
    assert warning_records(messages) == record["transform_warnings"]
    assert record["queries_preserved"]
    assert_bits(queries, sample["queries"])
    if record["mode"] == "graph":
        assert isinstance(actual, coo_matrix)
        expected = record["result"]
        assert actual.shape == tuple(expected["shape"])
        np.testing.assert_array_equal(actual.row, expected["row"])
        np.testing.assert_array_equal(actual.col, expected["column"])
        assert_bits(actual.data, expected["data"])
    else:
        assert_bits(fitted, record["fitted"])
        assert_bits(actual, record["result"])
        assert_bits(model.embedding_, record["fitted"])


@pytest.mark.parametrize("name", REFERENCE["errors"])
def test_rectangular_precomputed_transform_replays_input_failure(name):
    training = matrix(REFERENCE["inputs"]["base"]["training"])
    queries = matrix(REFERENCE["inputs"]["base"]["queries"])
    model = UMAP(**REFERENCE["options"], transform_mode="graph", n_epochs=6)
    with pytest.warns(UserWarning, match="using precomputed metric"):
        model.fit(training)
    if name == "negative":
        queries = queries.copy()
        queries[0, 0] = -np.nextafter(np.float32(0), np.float32(1))
    elif name == "short":
        queries = queries[:, :-1]
    elif name == "wide":
        queries = np.pad(queries, ((0, 0), (0, 1)))
    elif name == "empty":
        queries = queries[:0]
    elif name == "nan":
        queries = np.full_like(queries, np.nan)
    elif name == "infinite":
        queries = np.full_like(queries, np.inf)
    record = REFERENCE["errors"][name]
    error_type = {"ValueError": ValueError, "AssertionError": AssertionError}[record["type"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error_type) as error:
            model.transform(queries)
    assert str(error.value) == record["message"]
    assert warning_records(caught) == record["warnings"]


def test_finite_cutoff_fit_rejects_read_only_precomputed_input():
    training_record = REFERENCE["inputs"]["cutoff"]["training"]
    training = matrix(training_record)
    training.flags.writeable = False
    model = UMAP(**REFERENCE["options"], transform_mode="graph", disconnection_distance=.7)
    with pytest.warns(UserWarning, match="using precomputed metric"):
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            model.fit(training)
    assert not hasattr(model, "graph_")
    assert_bits(training, training_record)
