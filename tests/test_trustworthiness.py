from base64 import b64decode
import builtins
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from umap_alternative.validation import (
    make_trustworthiness_calculator,
    trustworthiness_vector,
    trustworthiness_vector_bulk,
)


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
REFERENCE = json.loads((FIXTURES / "umap-learn-0.6.0-trustworthiness-x86_64.json").read_bytes())
PLATFORM = "windows" if sys.platform == "win32" else "ubuntu"
PLATFORM_CASES = json.loads(
    (FIXTURES / f"umap-learn-0.6.0-trustworthiness-{PLATFORM}-x86_64.json").read_bytes()
)


def decode(record):
    return np.frombuffer(b64decode(record["bytes"]), dtype=record["dtype"]).reshape(record["shape"])


@pytest.mark.parametrize("spec", REFERENCE["cases"], ids=lambda case: case["name"])
def test_trustworthiness_matches_public_060(spec):
    source, embedded = map(decode, REFERENCE["inputs"][spec["input"]])
    mode, k, metric = spec["mode"], spec["k"], spec["metric"]
    if mode == "vector":
        def run():
            return trustworthiness_vector(source, embedded, k, metric)
    elif mode == "bulk":
        def run():
            return trustworthiness_vector_bulk(source, embedded, k)
    else:
        calculate = make_trustworthiness_calculator(lambda left, right: right[0])

        def run():
            return calculate(source, embedded, k)

    expected = spec["observed"] if "observed" in spec else PLATFORM_CASES[spec["name"]]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(getattr(builtins, expected["error"])):
                run()
        else:
            actual = run()
            assert actual.dtype == np.float64
            reference = decode(expected["output"])
            np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
    assert [type(item.message).__name__ for item in seen] == expected["warnings"]


def test_trustworthiness_accepts_strided_coordinates_and_rankings():
    source, embedded = map(decode, REFERENCE["inputs"]["ordinary"])
    expected = trustworthiness_vector(source, embedded, 3)
    expanded = np.zeros((source.shape[0], source.shape[1] * 2), dtype=source.dtype)
    expanded[:, ::2] = source
    actual = trustworthiness_vector(expanded[:, ::2], embedded.tolist(), np.int32(3))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    source, embedded = map(decode, REFERENCE["inputs"]["ranked-full"])
    expected = trustworthiness_vector_bulk(source[:, ::-1].copy(), embedded[::-1].copy(), 4)
    actual = trustworthiness_vector_bulk(source[:, ::-1], embedded[::-1], 4)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_trustworthiness_checks_rank_lengths_and_missing_ids():
    source = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="not ranked"):
        trustworthiness_vector_bulk(source, np.array([[2], [0]]), 1)
    with pytest.raises(ValueError, match="incompatible shapes"):
        trustworthiness_vector_bulk(source, np.array([[0], [1]]), 2)
    with pytest.raises(ValueError, match="same number of samples"):
        trustworthiness_vector(source, source[:1], 0)
    calculate = make_trustworthiness_calculator(lambda left, right: left[0] - right[0])
    with pytest.raises(ValueError, match="not ranked"):
        calculate(source, np.array([[2], [0]]), 1)
