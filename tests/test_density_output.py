import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from umap_alternative import UMAP


FIXTURE = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / (
    "umap-learn-0.6.0-densmap-windows-x86_64.json"
)
REFERENCE = json.loads(FIXTURE.read_bytes())
FEATURES = np.asarray(REFERENCE["data_float32"], dtype=np.float32)
INITIAL = np.asarray(REFERENCE["init_float32"], dtype=np.float32)


def assert_diagnostics(output, outputs):
    assert len(output) == len(outputs) == 3
    for values, array_id in zip(output, outputs):
        expected = REFERENCE["arrays"][array_id]
        assert values.dtype == np.float32
        assert values.shape == tuple(expected["shape"])
        np.testing.assert_array_equal(values.view(np.uint32).ravel(), expected["bits_uint32"])


@pytest.mark.parametrize("epochs", [0, 1, 2, 3, 10, 30])
def test_public_output_dens_tuple_and_fitted_radii(epochs):
    model = UMAP(
        **REFERENCE["common_options"], init=INITIAL, n_epochs=epochs, output_dens=True
    )
    result = model.fit_transform(FEATURES)
    expected = REFERENCE["cases"][f"{epochs}-density_output"]["outputs"]
    assert_diagnostics(result, expected)
    assert all(
        returned is stored
        for returned, stored in zip(result, (model.embedding_, model.rad_orig_, model.rad_emb_))
    )
    assert model.transform(FEATURES.copy()) is model.embedding_


def test_public_radius_offset_for_close_pairs():
    case = REFERENCE["arithmetic"]["close_pairs"]
    data = np.asarray(case["data_and_init_float32"], dtype=np.float32)
    model = UMAP(**(REFERENCE["common_options"] | case["options"]), init=data)
    assert_diagnostics(model.fit_transform(data), case["outputs"])


@pytest.mark.parametrize(
    "case", REFERENCE["arithmetic"]["neighbor_widths"],
    ids=lambda case: f"k={case['options']['n_neighbors']}",
)
def test_radius_refit_uses_requested_width(case):
    model = UMAP(**(REFERENCE["common_options"] | case["options"]), init=INITIAL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_transform(FEATURES)
    assert [item.category.__name__ for item in caught] == case["warnings"]
    assert_diagnostics(result, case["outputs"])


def test_output_dens_graph_mode_and_singleton():
    settings = dict(
        n_neighbors=4, init="random", random_state=42, n_jobs=1,
        n_epochs=0, compatibility_layout=True, output_dens=True,
    )
    graph_model = UMAP(**settings, transform_mode="graph")
    assert isinstance(graph_model.fit_transform(FEATURES), csr_matrix)
    assert not hasattr(graph_model, "rad_orig_")
    assert not hasattr(graph_model, "rad_emb_")

    singleton = UMAP(**settings).fit(FEATURES[:1])
    assert singleton.embedding_.shape == (1, 2)
    assert not hasattr(singleton, "rad_orig_")
    with pytest.raises(AttributeError, match="rad_orig_"):
        singleton.fit_transform(FEATURES[:1])


def test_density_diagnostics_expand_on_update_and_survive_an_ordinary_refit():
    model = UMAP(
        **REFERENCE["common_options"], init="random", n_epochs=2, output_dens=True
    ).fit(FEATURES)
    original = model.rad_orig_
    assert model.update(FEATURES[:2] + np.float32(0.021)) is None
    assert model.rad_orig_.shape == model.rad_emb_.shape == (10,)
    assert model.rad_orig_.dtype == model.rad_emb_.dtype == np.float32
    assert original.shape == (8,)
    radii_after_update = model.rad_orig_, model.rad_emb_
    model.set_params(output_dens=False).fit(FEATURES)
    assert model.rad_orig_ is radii_after_update[0]
    assert model.rad_emb_ is radii_after_update[1]
