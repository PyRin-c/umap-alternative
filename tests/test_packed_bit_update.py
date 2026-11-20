import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse

from test_sparse_euclidean_transform import assert_record, decode
from test_sparse_update import ERRORS, assert_state
from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-packed-update-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
EXCEPTION_TYPES = ERRORS | {"IndexError": IndexError}


def assert_call(action, expected, *, model=None, update=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(EXCEPTION_TYPES[expected["error"]]):
                action()
        else:
            result = action()
            if update:
                assert result is None
            if model is not None:
                if "identity" in expected:
                    assert result is getattr(model, expected["identity"])
                else:
                    assert_record(result, expected["result"])
    assert [item.category.__name__ for item in caught] == expected["warnings"]


def fixture_input(name, fill):
    rows = decode(REFERENCE["inputs"][name])
    if fill is None:
        return rows
    storage = np.full(rows.size + 16, fill, dtype=np.uint8)
    view = storage[:rows.size].reshape(rows.shape)
    view[:] = rows
    return view


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_packed_update_matches_public_060(name):
    case = REFERENCE["cases"][name]
    fill = case["storage"]
    original = fixture_input(case["training"], fill)
    query = fixture_input(case["query"], fill)
    options = REFERENCE["options"] | case["options"]
    if isinstance(options.get("disconnection_distance"), str):
        options["disconnection_distance"] = float(options["disconnection_distance"])
    model = UMAP(**options)
    assert_call(lambda: model.fit(original), case["fit"])
    before = case["initial"]
    assert_state(model, before)
    fitted_neighbors = min(model.n_neighbors, original.shape[0] - 1)
    training = original.astype(np.uint8)
    for step in case["steps"]:
        batch = fixture_input(step["batch"], fill)
        previous = {attr: getattr(model, attr, None) for attr in before}
        saved = {key: getattr(model, key) for key in step["options"]}
        model.set_params(**step["options"])
        assert_call(
            lambda: model.update(batch, ensure_all_finite=step["finite"]),
            step["update"], update=True,
        )
        assert_state(model, step["after"])
        if step["after"]["graph_"] is not None:
            assert model.n_neighbors_ == fitted_neighbors
        assert {attr: getattr(model, attr, None) is not old for attr, old in previous.items()} == step["replaced"]
        assert step["prior_unchanged"]
        for attr, old in previous.items():
            if old is not None:
                assert_record(old, before[attr])
        assert step["input_unchanged"]
        assert_record(batch, REFERENCE["inputs"][step["batch"]])
        if fill is not None:
            assert np.all(batch.base[batch.size:] == fill)

        queries = {"original": original, "held_out": query, "batch": batch,
                   "prefix": query[:, :1], "retained": original[:1]}
        candidate = None
        if (not issparse(batch) and batch.ndim == 2 and batch.shape[1] == training.shape[1]
                and np.isfinite(batch).all()):
            candidate = np.concatenate((training, batch.astype(np.uint8)), axis=0)
            queries["combined"] = candidate
            if batch.dtype.kind == "f":
                queries["mixed_combined"] = np.concatenate((training, batch), axis=0)
        for label, expected in step["transforms"].items():
            assert_call(lambda: model.transform(queries[label]), expected, model=model)
        mode = model.transform_mode
        model.set_params(transform_mode="graph")
        assert_call(lambda: model.transform(query), step["held_out_graph"], model=model)
        model.set_params(transform_mode=mode)
        if "after_epoch_reset" in step:
            model.set_params(n_epochs=0)
            assert_call(lambda: model.transform(original[:1]), step["after_epoch_reset"], model=model)
        model.set_params(**saved)
        if candidate is not None and step["after"]["graph_"] is not None and (
            model.graph_.shape[0] == candidate.shape[0]
        ):
            training = candidate
        before = step["after"]
    assert case["training_unchanged"]
    assert_record(original, REFERENCE["inputs"][case["training"]])
    if fill is not None:
        assert np.all(original.base[original.size:] == fill)


def test_packed_jaccard_single_zero_query_matches_appended_zero_row():
    model = UMAP(**REFERENCE["options"], metric="bit_jaccard", disconnection_distance=float("inf"))
    with pytest.warns(UserWarning, match="gradient function is not yet implemented"):
        model.fit(fixture_input("base", None))
    zero = fixture_input("first", None)
    model.update(zero)
    with pytest.warns(RuntimeWarning):
        coordinates = model.transform(zero)
    np.testing.assert_array_equal(coordinates.view(np.uint32), model.embedding_[-1:].view(np.uint32))
