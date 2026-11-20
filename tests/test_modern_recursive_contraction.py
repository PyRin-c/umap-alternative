"""Public-wheel recursive contraction and caller RNG checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_fit import PLATFORM, fit_case, make_data, recorded_layout, rng_checkpoint
from umap_alternative import UMAP
from umap_alternative._recursive import _expand_layout, _recursive_base, _recursive_partition


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-contraction-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
NESTED_OPTIMIZATION = "holdout-three-coarse-epochs"
CASES = [case for case in REFERENCE["archived"] + REFERENCE["fits"] if case["name"] != NESTED_OPTIMIZATION]


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
    means = np.asarray([anchor[labels == i].mean(axis=0) for i in range(3)], dtype=np.float32)
    actual = _expand_layout(graph, labels, _recursive_base(means, 2), anchor)
    reference = recorded_layout(case["layout_bits"], anchor.shape)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]


@pytest.mark.parametrize("case", CASES, ids=lambda value: value["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_contraction_fit(case, method):
    model, _ = fit_case(case, method)
    # Near-coincident pairs in the two rank-one holdouts amplify PCA rounding
    # during the outer optimizer's first nonzero updates.
    refined_bound = {
        "holdout-two-level-fuzzy": 1.2e-4,
        "holdout-three-level-transition": 2.5e-4,
        "holdout-two-coarse-epochs": 2e-5,
    }.get(case["name"], 1.2e-5)
    initial_bound = 2e-5 if case["name"] == "holdout-two-coarse-epochs" else 3e-6
    expected = recorded_layout(case["embedding_bits"], model.embedding_.shape)
    np.testing.assert_allclose(model.embedding_, expected, rtol=0, atol=refined_bound)
    for epoch, actual, bits in zip(model.n_epochs, model.embedding_list_, case["history_bits"], strict=True):
        expected = recorded_layout(bits, actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=initial_bound if epoch == 0 else refined_bound)


def test_unverified_coarse_epoch_budget_preserves_rng():
    case = next(case for case in REFERENCE["archived"] if case["name"] == "first-level-optimized")
    rng = np.random.RandomState(case["seed"])
    options = case["options"] | {"recursive_init_kwds": {"base_init_threshold": 3, "n_embedding_epochs": 4}}
    model = UMAP(**options, random_state=rng, n_jobs=1)
    with pytest.warns(UserWarning, match="using precomputed metric"), pytest.raises(
        NotImplementedError, match="zero to three explicit coarse epochs",
    ):
        model.fit(make_data(case["data"]))
    assert rng_checkpoint(rng) == case["rng_start"]


def test_unverified_nested_optimization_preserves_rng():
    case = next(case for case in REFERENCE["fits"] if case["name"] == NESTED_OPTIMIZATION)
    rng = np.random.RandomState(case["seed"])
    model = UMAP(**case["options"], random_state=rng, n_jobs=1)
    with pytest.raises(NotImplementedError, match="multiple recursive levels support zero or one coarse epoch"):
        model.fit(make_data(case["data"]))
    assert rng_checkpoint(rng) == case["rng_start"]
