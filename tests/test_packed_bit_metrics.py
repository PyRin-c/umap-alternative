import json
import sys
import warnings
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from test_sparse_euclidean_transform import assert_record, decode
from umap_alternative import UMAP, _native


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-packed-metrics-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
HOST = {"win32": "windows", "linux": "ubuntu"}.get(sys.platform)
HOST_REFERENCE = None
if HOST is not None:
    HOST_REFERENCE = json.loads(FIXTURE.with_name(
        f"umap-learn-0.6.0-packed-metrics-{HOST}-x86_64.json"
    ).read_bytes())
    REFERENCE["cases"].update(HOST_REFERENCE["cases"])
    REFERENCE["named"].extend(HOST_REFERENCE["named"])
ERRORS = {"ValueError": ValueError, "TypeError": TypeError, "AttributeError": AttributeError}


def assert_call(action, expected, model=None):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(ERRORS[expected["error"]]):
                action()
        else:
            result = action()
            if "identity" in expected:
                assert result is getattr(model, expected["identity"])
            else:
                assert_record(result, expected["value"])
    assert [item.category.__name__ for item in seen] == expected["warnings"]


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_packed_bits_match_public_060(name):
    case = REFERENCE["cases"][name]
    input_record = REFERENCE["inputs"][case["training"]]
    training = decode(input_record)
    options = REFERENCE["options"] | case["options"]
    if isinstance(options.get("disconnection_distance"), str):
        options["disconnection_distance"] = float(options["disconnection_distance"])
    model = UMAP(**options)
    assert_call(lambda: model.fit_transform(training), case["fit"])
    assert_record(training, input_record)
    if "graph" in case:
        assert_record(model.graph_, case["graph"])
    else:
        assert not hasattr(model, "graph_")
    for query, expected in case.get("queries", {}).items():
        query_record = REFERENCE["inputs"][query]
        data = decode(query_record)
        assert_call(lambda: model.transform(data), expected, model)
        assert_record(data, query_record)
    if "inverse" in case:
        assert_call(lambda: model.inverse_transform(np.zeros((1, 2), np.float32)), case["inverse"])


@pytest.mark.parametrize("pair", REFERENCE["named"])
@pytest.mark.parametrize("metric", ["hamming", "jaccard"])
def test_packed_native_distances_match_named_calls(metric, pair):
    left = bytes(pair["left"])
    right = bytes(pair["right"])
    columns, distances = _native._rank_packed_queries(
        (right, (1, len(right))), (left, (1, len(left))), 1, f"bit_{metric}",
        lambda values, k: np.argsort(values)[:k].tolist(),
    )
    assert_record(np.float32(distances[0]), pair[metric])
    assert columns == [0]


@pytest.mark.skipif(HOST is None, reason="The grid records the two pinned libm implementations")
def test_packed_jaccard_matches_host_log_grid():
    rows = [(np.packbits(np.arange(128) < count, bitorder="little").tobytes(), (1, 16))
            for count in range(100)]
    distances = []
    for union in range(1, 100):
        for shared in range(1, union + 1):
            _, values = _native._rank_packed_queries(
                rows[union], rows[shared], 1, "bit_jaccard", lambda values, k: [0],
            )
            distances.append(values[0])
    assert sha256(np.asarray(distances, dtype="<f4").tobytes()).hexdigest() == HOST_REFERENCE["jaccard_grid_sha256"]


@pytest.mark.parametrize("metric", ["bit_hamming", "bit_jaccard"])
def test_packed_queries_reject_reads_past_training_width(metric):
    training = decode(REFERENCE["inputs"]["bytes"])
    options = REFERENCE["options"] | {
        "metric": metric, "transform_mode": "graph", "disconnection_distance": float("inf"),
    }
    with pytest.warns(UserWarning, match="gradient function is not yet implemented"):
        model = UMAP(**options).fit(training)
    with pytest.raises(ValueError):
        model.transform(np.zeros((1, training.shape[1] + 1), dtype=np.uint8))
