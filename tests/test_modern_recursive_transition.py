"""Coarse layout dispatch around inactive memberships and contraction boundaries."""

from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_coarse import CoarseRandomState
from test_modern_recursive_fit import PLATFORM, check_graph, fit_case, recorded_layout, rng_checkpoint
from umap_alternative import _recursive
from umap_alternative._umap import _refine_recursive_level


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-transition-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", REFERENCE["stages"], ids=lambda case: case["name"])
def test_recursive_transition_stage(case, monkeypatch):
    anchor = np.array(case["anchor"], dtype=np.float32)
    record = case["graph"]
    graph = csr_matrix((
        np.asarray(record["weights_bits"], dtype=np.uint32).view(np.float32),
        record["indices"], record["indptr"],
    ), shape=(len(anchor), len(anchor)))
    monkeypatch.setattr(_recursive, "_sampled_pca_anchor", lambda *_: anchor.copy())
    rng = CoarseRandomState(case["seed"])
    options = {"base_init_threshold": case["threshold"], "n_embedding_epochs": case["budget"],
               "recursive_negative_sample_rate": case["rate"],
               "recursive_negative_selection_range_mode": case["selection"]}
    layout = _recursive._modern_recursive_initialization(
        graph, anchor, 2, rng, options, 3, "transition",
        partial(_refine_recursive_level, a=1., b=1., options=options),
    )
    check_graph(graph, record)
    assert rng.calls == case["rng_calls"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    bound = 3e-5 if case["name"] == "untied-3-6-5-coarse_n" else 5e-6
    np.testing.assert_allclose(layout, recorded_layout(case["layout_bits"], layout.shape), rtol=0, atol=bound)


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_transition_fit(case, method):
    model, _ = fit_case(case, method)
    actual = model.embedding_list_[0]
    np.testing.assert_allclose(
        actual, recorded_layout(case["history_bits"][0], actual.shape), rtol=0, atol=case["atol"],
    )
