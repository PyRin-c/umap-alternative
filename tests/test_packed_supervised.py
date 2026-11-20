import json
import warnings
from pathlib import Path

import pytest

from test_sparse_euclidean_transform import assert_record, decode
from test_sparse_update import ERRORS, assert_state
from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-packed-supervised-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
EXCEPTION_TYPES = ERRORS | {"KeyError": KeyError, "IndexError": IndexError}


def parameters(options):
    options = options.copy()
    for key in ("target_weight", "disconnection_distance"):
        if options.get(key) in ("inf", "nan"):
            options[key] = float(options[key])
    return options


def input_array(name):
    return decode(REFERENCE["inputs"][name])


def assert_call(action, expected, model, *, returns_self=False, update=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(EXCEPTION_TYPES[expected["error"]]):
                action()
        else:
            result = action()
            if "identity" in expected:
                assert result is getattr(model, expected["identity"])
            elif "result" in expected:
                assert_record(result, expected["result"])
            elif returns_self:
                assert result is model
            elif update:
                assert result is None
    assert [item.category.__name__ for item in caught] == expected["warnings"]


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_packed_supervision_matches_public_060(name):
    case = REFERENCE["cases"][name]
    model = UMAP(**parameters(REFERENCE["options"] | (
        {"metric": case["options"]["metric"], "disconnection_distance": "inf",
         "transform_mode": case["options"].get("transform_mode", "graph")}
        if case["warm"] else case["options"]
    )))
    if case["warm"]:
        assert_call(
            lambda: model.fit(input_array("bytes"), input_array("partial")),
            case["warm_fit"], model, returns_self=True,
        )
        assert_state(model, case["before"])
        model.set_params(**parameters(case["options"]))
    before = {key: getattr(model, key, None) for key in case["after"]}
    data, target = input_array(case["training"]), input_array(case["target"])
    assert_call(lambda: getattr(model, case["method"])(data, target), case["fit"], model,
                returns_self=case["method"] == "fit")
    assert_state(model, case["after"])
    assert case["replaced"] == {key: getattr(model, key, None) is not old for key, old in before.items()}
    for key, old in before.items():
        if old is not None:
            assert_record(old, case["before"][key])
    queries = {"fitted": data, "held_out": input_array(case["query"]),
               "original": input_array("bytes"), "prefix": input_array("prefix")}
    for key, expected in case.get("transforms", {}).items():
        assert_call(lambda: model.transform(queries[key]), expected, model)
    if "update" in case:
        assert_call(lambda: model.update(queries["held_out"]), case["update"], model, update=True)
        assert_state(model, case["after"])
    assert case["inputs_unchanged"]
    assert_record(data, REFERENCE["inputs"][case["training"]])
    assert_record(target, REFERENCE["inputs"][case["target"]])
