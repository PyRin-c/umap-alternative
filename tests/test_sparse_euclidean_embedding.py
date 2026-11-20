import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-sparse-euclidean-embedding-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
SPECTRAL_FIXTURE = FIXTURE.with_name(
    f"umap-learn-0.6.0-sparse-euclidean-spectral-{platform.system().lower()}-x86_64.json"
)
if platform.machine().lower() in ("amd64", "x86_64") and SPECTRAL_FIXTURE.is_file():
    REFERENCE["cases"].update(json.loads(SPECTRAL_FIXTURE.read_bytes()))


def from_record(record):
    if "format" not in record:
        dtype = np.dtype(record["dtype"])
        unsigned = f"uint{8 * dtype.itemsize}"
        return np.asarray(record["bits"], dtype=unsigned).view(dtype).reshape(record["shape"])
    data = from_record(record["data"])
    if record["format"] == "coo":
        return coo_matrix((data, (record["row"], record["col"])), shape=record["shape"])
    constructor = csr_matrix if record["format"] == "csr" else csc_matrix
    return constructor((data, record["indices"], record["indptr"]), shape=record["shape"])


def assert_bits(actual, record):
    assert actual.shape == tuple(record["shape"])
    assert actual.dtype == np.dtype(record["dtype"])
    unsigned = f"uint{8 * actual.itemsize}"
    np.testing.assert_array_equal(actual.view(unsigned).ravel(), record["bits"])


def assert_sparse(actual, record):
    assert actual.format == record["format"]
    assert actual.shape == tuple(record["shape"])
    assert_bits(actual.data, record["data"])
    if record["format"] == "coo":
        np.testing.assert_array_equal(actual.row, record["row"])
        np.testing.assert_array_equal(actual.col, record["col"])
    else:
        np.testing.assert_array_equal(actual.indptr, record["indptr"])
        np.testing.assert_array_equal(actual.indices, record["indices"])


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_sparse_euclidean_embedding_matches_public_060(name):
    case = REFERENCE["cases"][name]
    training = REFERENCE["training"][case["training"]]
    X = from_record(training)
    init = from_record(case["init"]) if isinstance(case["init"], dict) else case["init"]
    model = UMAP(**(REFERENCE["options"] | case["options"] | {"init": init}))
    expected = case["fit"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            assert expected["error"] == "ValueError"
            with pytest.raises(ValueError) as error:
                model.fit_transform(X)
            assert str(error.value) == expected["message"]
        else:
            assert_bits(model.fit_transform(X), expected["result"])
            assert_bits(model.embedding_, case["embedding"])
            assert model.n_features_in_ == X.shape[1]
            if "graph" in case:
                assert isinstance(model.graph_, csr_matrix)
                assert_sparse(model.graph_, case["graph"])
            else:
                assert not hasattr(model, "graph_")
            history = getattr(model, "embedding_list_", [])
            assert len(history) == len(case.get("history", []))
            for actual, record in zip(history, case.get("history", [])):
                assert_bits(actual, record)
    assert [[item.category.__name__, str(item.message)] for item in caught] == expected["warnings"]
    assert_sparse(X, training)


@pytest.mark.parametrize("name", REFERENCE["identity"])
def test_sparse_feature_fitted_input_compares_csr_storage(name):
    training = REFERENCE["training"][name]
    model = UMAP(**REFERENCE["options"]).fit(from_record(training))
    for query, expected in REFERENCE["identity"][name].items():
        X = from_record(REFERENCE["identity_queries"][query])
        if expected["same_coordinates"]:
            result = model.transform(X)
            assert (result is model.embedding_) == expected["same_object"]
            assert result.dtype.name == expected["dtype"]
            np.testing.assert_array_equal(result, model.embedding_)
        else:
            with pytest.raises(NotImplementedError, match="after fitting sparse features"):
                model.transform(X)


def test_wide_csr_embedding_and_fitted_input_transform_stay_sparse(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("sparse feature matrices must stay sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    training = from_record(REFERENCE["training"]["wide"])
    model = UMAP(**REFERENCE["options"]).fit(training)
    assert model.transform(training.copy()) is model.embedding_
    model.set_params(transform_mode="graph")
    assert model.transform(training.copy()) is model.graph_


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"compatibility_layout": False}, "require the compatibility layout"),
        ({"densmap": True}, "density fitting with sparse features"),
        ({"output_dens": True}, "density fitting with sparse features"),
    ],
)
def test_remaining_sparse_feature_fit_paths_fail_explicitly(overrides, message):
    model = UMAP(**(REFERENCE["options"] | overrides))
    with pytest.raises(NotImplementedError, match=message):
        model.fit(from_record(REFERENCE["training"]["csr"]))


def test_sparse_euclidean_embedding_inverse_replays_public_error():
    model = UMAP(**REFERENCE["options"]).fit(from_record(REFERENCE["training"]["csr"]))
    with pytest.raises(ValueError, match=r"^Inverse transform not available for sparse input\.$"):
        model.inverse_transform(model.embedding_[:2])
