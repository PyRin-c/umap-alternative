"""Scalar placement and frozen nine-row Adam captures from UMAP 0.6.0."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from test_modern_recursive_fit import PLATFORM, check_layout_bits, fit_case, recorded_layout, rng_checkpoint
from umap_alternative import _native
from umap_alternative._recursive import _expand_layout
from umap_alternative._umap import _modern_projector, _modern_rng_state


FIXTURE = (
    Path(__file__).parents[1]
    / f"reference/fixtures/umap-learn-0.6.0-modern-recursive-star-{PLATFORM}-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", REFERENCE["scalar"], ids=lambda case: case["name"])
def test_recursive_star_scalar_placement(case):
    anchor = np.array(case["anchor"], dtype=np.float32)
    record = case["graph"]
    graph = csr_matrix((
        np.asarray(record["weights_bits"], dtype=np.uint32).view(np.float32),
        record["indices"], record["indptr"],
    ), shape=(len(anchor), len(anchor)))
    child = recorded_layout(case["child_bits"], (-1, 1))
    actual = _expand_layout(graph, np.asarray(case["labels"], dtype=np.intp), child, anchor)
    assert actual.view(np.uint32).ravel().tolist() == case["layout_bits"]


@pytest.mark.parametrize("case", REFERENCE["replay"], ids=lambda case: case["name"])
def test_recursive_star_frozen_adam(case):
    expected = case["history_bits"]
    initial = recorded_layout(expected[0], (9, 2))
    graph = case["graph"]
    components = (9, 9), graph["indptr"], graph["indices"], np.asarray(
        graph["weights_bits"], dtype=np.uint32,
    ).view(np.float32).tolist()
    rng = np.random.RandomState(case["seed"])
    rng.randint(-2**31 + 1, 2**31 - 2, 3)
    _, _, history = _native._fit_modern_layout(
        components, initial.ravel().tolist(), _modern_rng_state(rng), optimizer="adam",
        n_components=2, n_epochs=case["budget"], a=1., b=1., learning_rate=.5,
        repulsion_strength=4., negative_sample_rate=case["rate"], exclude_graph_neighbors=False,
        project=_modern_projector(rng, initial.shape, draw_seed=False), coarse=True,
        checkpoint_epochs=list(range(case["budget"] + 1)),
    )
    assert rng_checkpoint(rng) == {key: case["rng"][key] for key in ("cursor", "gaussian_bits")}
    assert rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist() == case["rng"]["next_words"]
    assert np.asarray(history, dtype=np.float32).view(np.uint32).tolist() == expected


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda case: case["name"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_recursive_star_fit(case, method):
    model, _ = fit_case(case, method)
    check_layout_bits(model.embedding_, case["embedding_bits"])
