import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse, vstack

from test_sparse_euclidean_transform import assert_record, decode
from test_sparse_update import ERRORS, assert_state
from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-boolean-update-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
EXCEPTION_TYPES = ERRORS | {"IndexError": IndexError}


def assert_call(call, expected, *, model=None, update=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(EXCEPTION_TYPES[expected["error"]]):
                call()
        else:
            actual = call()
            if update:
                assert actual is None
            if model is not None:
                if "identity" in expected:
                    assert actual is getattr(model, expected["identity"])
                else:
                    assert actual is not getattr(model, "graph_", None)
                    assert actual is not getattr(model, "embedding_", None)
                    assert_record(actual, expected["result"])
    assert [item.category.__name__ for item in caught] in expected.get(
        "warning_variants", [expected.get("warnings")],
    )


def candidate_rows(training, batch):
    if training.shape[1] != batch.shape[1]:
        return None
    if issparse(training):
        return vstack((training, batch), format="csr")
    if not issparse(batch):
        return np.concatenate((training, batch), axis=0)
    return None


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_boolean_feature_update_matches_public_060(name):
    case = REFERENCE["cases"][name]
    original = decode(REFERENCE["inputs"][case["training"]])
    training = original.copy()
    model = UMAP(**(REFERENCE["options"] | case["options"]))
    assert_call(lambda: model.fit(original), case["fit"])
    assert_state(model, case["initial"])
    fitted_neighbors = min(model.n_neighbors, original.shape[0] - 1)
    before = case["initial"]
    for step in case["steps"]:
        batch_record = REFERENCE["inputs"][step["batch"]]
        batch = decode(batch_record)
        previous = {attr: getattr(model, attr, None) for attr in before}
        saved = {key: getattr(model, key) for key in step["options"]}
        model.set_params(**step["options"])
        assert_call(lambda: model.update(batch), step["outcome"], update=True)
        assert model.n_neighbors_ == fitted_neighbors
        assert_state(model, step["after"])
        assert [getattr(model, attr, None) is not old for attr, old in previous.items()] == step["replaced"]
        for (attr, old), unchanged in zip(previous.items(), step["unchanged"], strict=True):
            assert unchanged
            if old is not None:
                assert_record(old, before[attr])
        assert step["input_unchanged"]
        assert_record(batch, batch_record)

        candidate = candidate_rows(training, batch)
        queries = {"original": original, "held_out": original[:1], "concatenated": candidate}
        for key, expected in step["transforms"].items():
            assert_call(lambda: model.transform(queries[key].copy()), expected, model=model)
        mode = model.transform_mode
        model.set_params(transform_mode="graph")
        assert_call(lambda: model.transform(original[:1].copy()), step["held_out_graph"], model=model)
        model.set_params(transform_mode=mode)
        if "after_epoch_reset" in step:
            model.set_params(n_epochs=6)
            assert_call(
                lambda: model.transform(original[:1].copy()),
                step["after_epoch_reset"]["held_out"], model=model,
            )
            assert_call(lambda: model.transform(batch.copy()), step["after_epoch_reset"]["batch"], model=model)
        model.set_params(**saved)
        if candidate is not None and model.graph_.shape[0] == candidate.shape[0]:
            training = candidate
        before = step["after"]
    assert_record(original, REFERENCE["inputs"][case["training"]])
