import builtins
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from umap_alternative import distances


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "umap-learn-0.6.0-row-distance-x86_64.json"
CASES = json.loads(FIXTURE.read_bytes())


@pytest.mark.parametrize(
    "case, metric",
    [(case, metric) for case in CASES for metric in case["observed"]],
    ids=[f"{case['name']}-{metric}" for case in CASES for metric in case["observed"]],
)
def test_public_named_feature_distance(case, metric):
    left, right = (np.frombuffer(bytes.fromhex(raw), dtype=dtype) for dtype, raw in (case["left"], case["right"]))
    observed = case["observed"][metric]
    distance = distances.named_distances[metric]
    assert getattr(distances, distance.__name__) is distance
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in observed:
            name = "TypeError" if observed["error"] == "TypingError" else observed["error"]
            with pytest.raises(getattr(builtins, name)):
                distance(left, right)
        else:
            result = np.float64(distance(left, right)).view(np.uint64).item()
            assert f"{result:016x}" == observed["bits"]
    assert [type(item.message).__name__ for item in seen] == observed["warnings"]


@pytest.mark.parametrize("metric", ("hamming", "dice", "canberra", "cosine"))
def test_named_distance_rejects_a_short_second_buffer(metric):
    with pytest.raises(ValueError, match="shorter"):
        distances.named_distances[metric](np.ones(2), np.ones(1))


@pytest.mark.parametrize("metric", ("hamming", "dice", "canberra", "cosine"))
def test_named_distance_preserves_values_from_strided_vectors(metric):
    case = next(case for case in CASES if case["name"] == "float32-width-33")
    buffers = []
    for dtype, raw in (case["left"], case["right"]):
        row = np.frombuffer(bytes.fromhex(raw), dtype=dtype)
        view = np.empty(row.size * 3, dtype=dtype)[::3]
        view[:] = row
        buffers.append(view)
    bits = np.float64(distances.named_distances[metric](*buffers)).view(np.uint64).item()
    assert f"{bits:016x}" == case["observed"][metric]["bits"]
