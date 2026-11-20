import json
import warnings
from pathlib import Path

import pytest
from scipy.sparse import csr_matrix, vstack

from test_sparse_euclidean_transform import assert_record, decode, warning_records
from test_sparse_update import ERRORS, assert_state, input_matrix
from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-sparse-metric-update-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())


def inputs(name):
    return decode(REFERENCE["inputs"][name])


def assert_transform(model, matrix, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "error" in expected:
            with pytest.raises(ERRORS[expected["error"]]):
                model.transform(matrix.copy())
            return
        actual = model.transform(matrix.copy())
    if "identity" in expected:
        assert actual is getattr(model, expected["identity"])
    else:
        assert actual is not getattr(model, "embedding_", None)
        assert actual is not getattr(model, "graph_", None)
        assert_record(actual, expected["result"])


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_sparse_metric_updates_match_public_060(name):
    case = REFERENCE["cases"][name]
    original = inputs(case["training"])
    stacked = csr_matrix(original)
    model = UMAP(**(REFERENCE["options"] | case["options"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(original)
    assert warning_records(caught) == case["fit_warnings"]
    assert_state(model, case["fit"])
    fitted_neighbors = min(model.n_neighbors, original.shape[0] - 1)
    before = case["fit"]

    for step in case["steps"]:
        saved = {key: getattr(model, key) for key in step["options"]}
        model.set_params(**step["options"])
        previous = {attr: getattr(model, attr, None) for attr in before}
        batch = inputs(step["batch"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if step["status"] == "ok":
                assert model.update(batch) is None
            else:
                with pytest.raises(ERRORS[step["status"]]):
                    model.update(batch)
        assert warning_records(caught) == step["warnings"]
        assert model.n_neighbors_ == fitted_neighbors
        assert_state(model, step["after"])
        for attr, old_value in previous.items():
            assert (getattr(model, attr, None) is not old_value) == step[f"{attr}replaced"]
            if old_value is not None:
                assert_record(old_value, before[attr])
        assert_record(batch, REFERENCE["inputs"][step["batch"]])
        assert_transform(model, original, step["transform"]["original"])

        if step["status"] == "ok":
            stacked = vstack((stacked, csr_matrix(batch)), format="csr")
            assert_transform(model, stacked, step["transform"]["combined"])
            assert_transform(model, original[:1], step["transform"]["held_out"])
            if "canonical" in step["transform"]:
                canonical = stacked.copy()
                canonical.sum_duplicates()
                assert_transform(model, canonical, step["transform"]["canonical"])
            if "held_out_graph" in step:
                model.set_params(transform_mode="graph")
                assert_transform(model, original[:1], step["held_out_graph"])
                model.set_params(transform_mode="embedding")
        model.set_params(**saved)
        before = step["after"]

    assert_record(original, REFERENCE["inputs"][case["training"]])


@pytest.mark.parametrize("name", ["fit", "append"])
def test_manhattan_update_uses_live_entries_in_duplicate_csr(name):
    reference = REFERENCE["cases"][f"manhattan_canonical_{name}"]
    if name == "fit":
        source = REFERENCE["residuals"]["manhattan_split_fit"]
    else:
        first = REFERENCE["cases"]["manhattan_split"]
        source = first | {"steps": [*first["steps"], REFERENCE["residuals"]["manhattan_split_repeat"]]}
    training = inputs(source["training"])
    model = UMAP(**(REFERENCE["options"] | source["options"])).fit(training)
    assert_state(model, reference["fit"])
    for step, canonical in zip(source["steps"], reference["steps"], strict=True):
        saved = {key: getattr(model, key) for key in step["options"]}
        model.set_params(**step["options"])
        batch = inputs(step["batch"])
        assert model.update(batch) is None
        assert_state(model, canonical["after"])
        assert_record(batch, REFERENCE["inputs"][step["batch"]])
        model.set_params(**saved)
    assert_record(training, REFERENCE["inputs"][source["training"]])


@pytest.mark.parametrize("metric", ["manhattan", "chebyshev", "cosine"])
def test_sparse_metric_update_keeps_csr_storage(monkeypatch, metric):
    def fail(*args, **kwargs):
        pytest.fail("sparse feature update must keep features sparse")

    monkeypatch.setattr(csr_matrix, "toarray", fail)
    monkeypatch.setattr(csr_matrix, "todense", fail)
    training = input_matrix("wide")
    model = UMAP(**(REFERENCE["options"] | {"metric": metric})).fit(training)
    model.update(input_matrix("wide_append"))
    assert model.embedding_.shape == (9, 2)
