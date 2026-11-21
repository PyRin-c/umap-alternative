"""Public-wheel recursive contraction and caller RNG checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_fit import PLATFORM, check_layout_bits, fit_case, make_data, rng_checkpoint
from umap_alternative import UMAP
from umap_alternative._recursive import _aggregate_anchors, _expand_layout, _recursive_base, _recursive_partition


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-contraction-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASES = REFERENCE["archived"] + REFERENCE["fits"]


@pytest.mark.parametrize("case", REFERENCE["placements"], ids=lambda value: value["name"])
def test_recursive_contraction_placement(case):
    anchor = np.asarray(case["anchor"], dtype=np.float32)
    expected = case["graph"]
    graph = csr_matrix((
        np.asarray(expected["weights_bits"], dtype=np.uint32).view(np.float32),
        expected["indices"], expected["indptr"],
    ), shape=(len(anchor), len(anchor)))
    rng = np.random.RandomState(case["seed"])
    seed = rng.randint(-2**31 + 1, 2**31 - 2, 3)
    labels = _recursive_partition(graph, 3, seed)
    means = _aggregate_anchors(anchor, labels)
    actual = _expand_layout(graph, labels, _recursive_base(means, 2), anchor)
    check_layout_bits(actual, case["layout_bits"])
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]


@pytest.mark.parametrize("case", CASES, ids=lambda value: value["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_contraction_fit(case, method):
    model, _ = fit_case(case, method)
    check_layout_bits(model.embedding_, case["embedding_bits"])
    for actual, bits in zip(model.embedding_list_, case["history_bits"], strict=True):
        check_layout_bits(actual, bits)


@pytest.mark.parametrize("budget", [-1, 3.5])
def test_invalid_coarse_epoch_budget_preserves_rng(budget):
    case = next(case for case in REFERENCE["archived"] if case["name"] == "first-level-optimized")
    rng = np.random.RandomState(case["seed"])
    options = case["options"] | {"recursive_init_kwds": {"base_init_threshold": 3, "n_embedding_epochs": budget}}
    model = UMAP(**options, random_state=rng, n_jobs=1)
    with pytest.warns(UserWarning, match="using precomputed metric"), pytest.raises(
        ValueError, match="n_embedding_epochs must be a nonnegative integer",
    ):
        model.fit(make_data(case["data"]))
    assert rng_checkpoint(rng) == case["rng_start"]
