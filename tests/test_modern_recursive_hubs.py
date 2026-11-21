"""Direct public contraction observations with frozen input graph and RNG."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_coarse import CoarseRandomState
from test_modern_recursive_fit import PLATFORM, check_graph, check_layout_bits, fit_case, rng_checkpoint
from umap_alternative._recursive import (
    _aggregate_anchors, _coarse_graphs, _expand_layout, _recursive_base, _recursive_partition,
)
from umap_alternative._umap import _refine_recursive_level


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-hubs-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", REFERENCE["stages"], ids=lambda case: case["name"])
def test_recursive_hub_layout(case):
    anchor = np.array(case["anchor"], dtype=np.float32)
    record = case["graph"]
    graph = csr_matrix((
        np.asarray(record["weights_bits"], dtype=np.uint32).view(np.float32),
        record["indices"], record["indptr"],
    ), shape=(len(anchor), len(anchor)))
    rng = CoarseRandomState(case["seed"])
    seed = rng.randint(-2**31 + 1, 2**31 - 2, 3)
    labels = _recursive_partition(graph, case["parts"], seed)
    means = _aggregate_anchors(anchor, labels)
    layout = _recursive_base(means, 2)
    if case["budget"]:
        _, coarse = _coarse_graphs(graph, labels, case["parts"], case["weights"], False)
        layout = _refine_recursive_level(
            coarse, layout, case["budget"], rng, a=1., b=1., depth=1, use_refinement=False,
            options={"recursive_negative_sample_rate": case["rate"],
                     "recursive_negative_selection_range_mode": case["selection"]},
        )
    result = _expand_layout(graph, labels, layout, anchor)
    check_graph(graph, record)
    assert rng.calls == case["rng_calls"]
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    check_layout_bits(result, case["layout_bits"])


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_hub_fit_controls(case, method):
    model, _ = fit_case(case, method)
    check_layout_bits(model.embedding_list_[0], case["history_bits"][0])
