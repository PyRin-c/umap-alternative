import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from umap_alternative import UMAP
from umap_alternative.utils import csr_unique


_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "reference" / "fixtures" /
     "umap-learn-0.6.0-unique-x86_64.json").read_text(encoding="utf-8")
)


def _assert_matches(actual, expected):
    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            _assert_matches(value, target)
    elif isinstance(expected, dict):
        assert actual.format == expected["format"]
        assert actual.shape == tuple(expected["shape"])
        coordinates = (
            [actual.indptr, actual.indices] if actual.format == "csr" else [actual.row, actual.col]
        )
        for indices, target in zip(coordinates, expected["coordinates"]):
            np.testing.assert_array_equal(indices, target)
        np.testing.assert_array_equal(actual.data.view(np.uint32), expected["bits"])
    else:
        np.testing.assert_array_equal(actual.view(np.uint32), expected)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_public_unique_checkpoints(case, method):
    name = case["name"]
    convert = (
        csr_matrix if name.startswith("csr") else
        csc_matrix if name == "csc" else coo_matrix if name == "coo" else np.asarray
    )
    x = convert(np.asarray(_FIXTURE["features"], dtype=np.float32))
    queries = convert(np.asarray(_FIXTURE["queries"], dtype=np.float32))
    options = case["options"].copy()
    if name == "explicit":
        options["init"] = np.asarray(options["init"], dtype=np.float32)
    model = UMAP(**options)
    labels = None if case["labels"] is None else np.asarray(case["labels"])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        output = getattr(model, method)(x, labels)
    assert [w.category.__name__ for w in seen] == case["warnings"]
    _assert_matches(model.graph_, case["graph"])
    if method == "fit":
        if model.transform_mode == "graph":
            output = model.graph_
        elif model.output_dens:
            output = (model.embedding_, model.rad_orig_, model.rad_emb_)
        else:
            output = model.embedding_
    _assert_matches(output, case["result"])
    if "coordinates_by_epoch" in case:
        assert len(model.embedding_list_) == len(case["coordinates_by_epoch"])
        for coordinates, expected in zip(model.embedding_list_, case["coordinates_by_epoch"]):
            _assert_matches(coordinates, expected)
    if "training_transform" in case:
        _assert_matches(model.transform(x), case["training_transform"])
        _assert_matches(model.transform(queries), case["query_transform"])
    if "updated_graph" in case:
        model.set_params(n_epochs=0)
        _assert_matches(model.inverse_transform(model.embedding_[[1, 4]]), case["inverse"])
        assert model.update(queries) is None
        _assert_matches(model.graph_, case["updated_graph"])
        _assert_matches(model.embedding_, case["updated_embedding"])


@pytest.mark.parametrize("case", _FIXTURE["utility"])
def test_public_csr_unique_keys(case):
    matrix = csr_matrix(
        (np.asarray(case["values"], dtype=np.float32), case["indices"], case["indptr"]),
        shape=(len(case["indptr"]) - 1, 3),
    )
    stored = matrix.data.view(np.uint32).copy()
    assert [part.tolist() for part in csr_unique(matrix)] == case["groups"]
    assert [part.tolist() for part in csr_unique(matrix, False, True, True)] == case["groups"][1:]
    np.testing.assert_array_equal(matrix.data.view(np.uint32), stored)


@pytest.mark.parametrize("case", _FIXTURE["inverse"], ids=["permuted", "duplicate"])
def test_inverse_uses_unique_fitted_scale_order(case):
    model = UMAP(**case["options"]).fit(np.asarray(case["features"], dtype=np.float32))
    model.embedding_ = np.asarray(case["layout"], dtype=np.float32)
    model.n_epochs = 48
    _assert_matches(model.inverse_transform(np.asarray(case["queries"], dtype=np.float32)), case["result"])


def test_sparse_row_keys_do_not_round_wide_integer_values():
    matrix = csr_matrix(np.asarray([[2**63 + 1], [2**63], [2**63 + 1]], dtype=np.uint64))
    first, inverse, counts = csr_unique(matrix)
    assert (first.tolist(), inverse.tolist(), counts.tolist()) == ([1, 0], [1, 0, 1], [1, 2])


@pytest.mark.parametrize("convert", [np.asarray, csr_matrix])
def test_unique_singleton_uses_one_zero_coordinate(convert):
    data = convert(np.ones((5, 3), dtype=np.float32))
    model = UMAP(unique=True, init="random", random_state=7, n_jobs=1, compatibility_layout=True)
    np.testing.assert_array_equal(model.fit_transform(data), np.zeros((1, 2), np.float32))
    assert not hasattr(model, "graph_")
    with pytest.raises(AttributeError):
        model.transform(data)
    with pytest.raises(AttributeError):
        model.update(data)
    with pytest.raises(AttributeError):
        model.set_params(transform_mode="graph").fit_transform(data)
    one_row = convert(np.ones((1, 3), dtype=np.float32))
    model.set_params(transform_mode="embedding").fit(one_row)
    with pytest.raises(ValueError, match="single data sample"):
        model.transform(one_row)


def test_unique_rejects_precomputed_distances_and_wrong_init_count():
    data = np.asarray(_FIXTURE["features"], np.float32)
    model = UMAP(unique=True, init="random", random_state=7, n_jobs=1, compatibility_layout=True)
    with pytest.raises(ValueError, match="precomputed"):
        model.set_params(metric="precomputed").fit(np.zeros((len(data), len(data)), np.float32))
    indices = np.zeros((len(data), 3), dtype=np.int32)
    distances = np.zeros_like(indices, dtype=np.float32)
    with pytest.raises(ValueError, match="precomputed_knn"):
        model.set_params(metric="euclidean", precomputed_knn=(indices, distances, None)).fit(data)
    with pytest.raises(IndexError):
        model.set_params(precomputed_knn=(None, None, None), init=data.copy()).fit(data)
