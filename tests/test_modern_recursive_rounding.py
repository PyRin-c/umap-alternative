"""Raw public recursive placement across percentile and edge-weight boundaries."""

from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_coarse import CoarseRandomState
from test_modern_recursive_fit import PLATFORM, check_graph, recorded_layout, rng_checkpoint
from umap_alternative import _recursive
from umap_alternative._umap import _refine_recursive_level


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-rounding-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def replay(graph_record, values, case, seed):
    graph = csr_matrix((
        np.asarray(graph_record["weights_bits"], dtype=np.uint32).view(np.float32),
        graph_record["indices"], graph_record["indptr"],
    ), shape=(len(values), len(values)))
    rng = CoarseRandomState(seed)
    options = {"base_init_threshold": case["threshold"], "n_embedding_epochs": case["budget"],
               "recursive_negative_sample_rate": 0}
    layout = _recursive._modern_recursive_initialization(
        graph, values, case.get("n_components", 2), rng, options, 3, "transition",
        partial(_refine_recursive_level, a=1., b=1., options=options),
    )
    check_graph(graph, graph_record)
    assert rng.calls == case["rng_calls"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    return layout


@pytest.mark.parametrize("case", REFERENCE["base"], ids=lambda case: case["name"])
def test_recursive_percentile_boundaries(case):
    values = recorded_layout(case["anchor_bits"], (-1, case["n_components"]))
    actual = _recursive._recursive_base(values, case["depth"])
    assert case["rng_calls"] == []
    assert actual.view(np.uint32).ravel().tolist() == case["layout_bits"]


@pytest.mark.parametrize("case", REFERENCE["contractions"], ids=lambda case: case["name"])
def test_recursive_transition_weight_boundaries(case, monkeypatch):
    anchor = recorded_layout(case["anchor_bits"], (-1, case["n_components"]))
    monkeypatch.setattr(_recursive, "_sampled_pca_anchor", lambda *_: anchor.copy())
    actual = replay(case["graph"], anchor, case, case["seed"])
    assert actual.view(np.uint32).ravel().tolist() == case["layout_bits"]


@pytest.mark.parametrize("case", REFERENCE["archive"]["cases"], ids=lambda case: case["name"])
def test_recursive_star_before_outer_normalization(case):
    archive = REFERENCE["archive"]
    values = recorded_layout(archive["input_bits"], (27, 27))
    actual = replay(archive["graph"], values, case, 42)
    expected = recorded_layout(case["layout_bits"], actual.shape)
    if case["threshold"] == len(values):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        return
    bound = 2e-7 if case["name"] == "star-raw-3-11" else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)


@pytest.mark.parametrize("case", REFERENCE["mixed_scale_pca"], ids=lambda case: str(case["seed"]))
def test_recursive_full_pca_keeps_the_weak_axis(case):
    seed = case["seed"]
    values = np.random.RandomState(seed).normal(size=(129, 3)).astype(np.float32)
    values *= np.exp2(np.linspace(-12, 16, 3, dtype=np.float32))
    values[::3] *= np.float32(-2**-8)
    rng = np.random.RandomState(seed)
    anchors = _recursive._sampled_pca_anchor(values, 3, rng, sample_size=len(values))
    layout = _recursive._recursive_base(anchors, depth=1)
    assert layout.view(np.uint32).ravel().tolist() == case["layout_bits"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
