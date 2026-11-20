"""Platform checkpoints for AlignedUMAP's public fit and append paths."""

import json
from pathlib import Path
import platform

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from umap_alternative import AlignedUMAP


pytestmark = pytest.mark.filterwarnings("ignore:n_jobs value -1 overridden:UserWarning")
FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def load_fixture(stage):
    system = "windows" if platform.system() == "Windows" else "ubuntu"
    return json.loads(
        (FIXTURES / f"umap-learn-0.6.0-{stage}-{system}-x86_64.json").read_text(encoding="utf-8")
    )


PUBLIC = load_fixture("aligned")
OPTIMIZER = load_fixture("aligned-optimizer")


@pytest.fixture(autouse=True)
def single_blas_thread():
    with threadpool_limits(limits=1):
        yield


def inputs(family="base", relations="complete"):
    matrices = [np.asarray(x, dtype=np.float32) for x in PUBLIC["data"][family]]
    return matrices, [dict(items) for items in PUBLIC["relations"][relations]]


def assert_layout(layout, bits):
    assert layout.dtype == np.float32
    assert layout.ndim == 2
    np.testing.assert_array_equal(layout.view(np.uint32), bits)


def assert_graphs(model, expected):
    assert len(model.mappers_) == model.n_models_ == len(expected)
    for mapper, graph_id in zip(model.mappers_, expected, strict=True):
        graph = mapper.graph_
        assert isinstance(graph, csr_matrix)
        assert graph.dtype == np.float32
        other = PUBLIC["graphs"][graph_id]
        np.testing.assert_array_equal(graph.indptr, other["indptr"])
        np.testing.assert_array_equal(graph.indices, other["indices"])
        np.testing.assert_array_equal(graph.data.view(np.uint32), other["bits"])


@pytest.mark.parametrize("name", [
    "epoch-0-weight-0.01", "epoch-1-weight-0.01", "identity", "partial",
    "duplicate_target", "spectral-mappers", "holdout",
])
def test_initial_alignment(name):
    expected = PUBLIC["cases"][name]
    data, relations = inputs(expected["family"], expected["relation_group"])
    model = AlignedUMAP(**(PUBLIC["settings"] | expected["overrides"]))
    assert model.fit(data, relations=relations) is model
    assert_graphs(model, expected["graphs"])
    for actual, bits in zip(model.embeddings_, expected["embeddings"], strict=True):
        assert_layout(actual, bits)


def test_joint_optimisation_uses_the_window_and_regularisation():
    data, links = inputs()

    def fit(epochs, weight=0.01, window=2):
        model = AlignedUMAP(**(PUBLIC["settings"] | {
            "n_epochs": epochs, "alignment_regularisation": weight, "alignment_window_size": window,
        }))
        result = model.fit_transform(data, relations=links)
        assert result is model.embeddings_
        assert_graphs(model, PUBLIC["cases"]["fit-transform"]["graphs"])
        return np.vstack(result)

    initial = fit(0)
    second_epoch = fit(2)
    eighth_epoch = fit(8)
    assert np.isfinite(eighth_epoch).all()
    assert np.count_nonzero(initial != second_epoch) == 48
    assert np.count_nonzero(second_epoch != fit(2, weight=0.3)) == 48
    assert np.count_nonzero(eighth_epoch != fit(8, weight=0.3)) == 54
    assert np.count_nonzero(eighth_epoch != fit(8, window=1)) == 54
    np.testing.assert_array_equal(eighth_epoch, fit(8))


@pytest.mark.parametrize("name", [
    "epoch-2-weight-0.0", "epoch-2-weight-0.01",
    "epoch-8-weight-0.0", "epoch-8-weight-0.01", "epoch-8-weight-0.3",
    "zero-window-unweighted", "zero-window-weighted", "single-window-first-edge", "single-window",
    "batch-partial-2", "batch-partial-8", "batch-duplicate_target-2", "batch-duplicate_target-8",
    "batch-all_but_last-window-1-epochs-8", "batch-include_last-window-2-epochs-8",
])
def test_batch_trajectory(name):
    expected = PUBLIC["cases"][name]
    data, links = inputs(expected["family"], expected["relation_group"])
    model = AlignedUMAP(**(PUBLIC["settings"] | expected["overrides"]))
    result = model.fit_transform(data, relations=links)
    assert_graphs(model, expected["graphs"])
    for actual, bits in zip(result, expected["embeddings"], strict=True):
        assert_layout(actual, bits)


@pytest.mark.parametrize("name", ["update-0", "update-2", "update-8", "update-partial-0"])
def test_append_from_public_frozen_layouts(name):
    data, relations = inputs()
    expected = PUBLIC["cases"][name]
    epochs = expected["overrides"]["n_epochs"]
    model = AlignedUMAP(**PUBLIC["settings"]).fit(data[:2], relations=relations[:1])
    # Isolate the append from the prior fit by supplying the public checkpoints.
    for layout, bits in zip(model.embeddings_, expected["before"]["embeddings"], strict=True):
        layout[:] = np.asarray(bits, dtype=np.uint32).view(np.float32)
    old_layouts = tuple(model.embeddings_)
    assert model.update(data[2], relations=dict(expected["relations"]), n_epochs=epochs) is None
    assert_graphs(model, expected["after"]["graphs"])
    assert all(before is after for before, after in zip(old_layouts, model.embeddings_[:2], strict=True))
    for actual, bits in zip(model.embeddings_, expected["after"]["embeddings"], strict=True):
        assert_layout(actual, bits)


@pytest.mark.parametrize("name", OPTIMIZER["cases"])
def test_isolated_append_kernel(name):
    probe = OPTIMIZER["cases"][name]
    data = np.asarray(OPTIMIZER["data"], dtype=np.float32)
    fixed = np.asarray(OPTIMIZER["fixed"], dtype=np.float32)
    identity = dict(enumerate(range(len(data))))
    model = AlignedUMAP(**probe["config"]).fit([data, data], relations=[identity])
    if "scale" in probe:
        model.embeddings_[-1][:] = fixed * probe["scale"]
    else:
        model.embeddings_[-1][:] = fixed
        model.embeddings_[0][:] = fixed
        model.embeddings_[0][2, 0] += 4.0
        graph = model.mappers_[-1].graph_.toarray()
        graph[2] = 0.0
        graph[2, probe["neighbors"]] = probe["membership"]
        model.mappers_[-1].graph_ = csr_matrix(graph)
    model.update(data, relations=identity, n_epochs=probe["epochs"])
    expected = np.asarray(probe["layout"], dtype=np.uint32).view(np.float32)
    assert model.embeddings_[-1].dtype == np.float32
    np.testing.assert_array_max_ulp(model.embeddings_[-1], expected, maxulp=1)


def test_partial_append_uses_linked_neighbors_and_last_source():
    data = np.asarray(OPTIMIZER["data"], dtype=np.float32)
    probe = OPTIMIZER["partial"]
    identity = dict(enumerate(range(len(data))))
    model = AlignedUMAP(**probe["config"]).fit([data, data], relations=[identity])
    model.embeddings_[-1][:] = np.asarray(probe["previous"], dtype=np.uint32).view(np.float32)
    model.update(probe["data"], relations=dict(probe["relations"]))
    assert_layout(model.embeddings_[-1], probe["layout"])


@pytest.mark.parametrize("seed", [37, 42, 95])
def test_unlinked_append_uses_the_seeded_epoch_stream(seed):
    data = np.asarray(OPTIMIZER["data"], dtype=np.float32)
    probe = OPTIMIZER["unlinked"][str(seed)]
    identity = dict(enumerate(range(len(data))))
    model = AlignedUMAP(**probe["config"]).fit([data, data], relations=[identity])
    model.update(data, relations={0: 60}, n_epochs=0)
    assert_layout(model.embeddings_[-1], probe["layout"])


@pytest.mark.parametrize("relation_group,error", [
    ("short", AssertionError), ("long", AssertionError), ("empty", ValueError),
])
def test_batch_relation_errors(relation_group, error):
    data, links = inputs(relations=relation_group)
    model = AlignedUMAP(**PUBLIC["settings"])
    with pytest.raises(error):
        model.fit(data, relations=links)
    assert not hasattr(model, "embeddings_")
    if relation_group == "empty":
        assert_graphs(model, PUBLIC["cases"]["empty"]["graphs"])
    else:
        assert not hasattr(model, "mappers_")


def test_missing_and_invalid_update_relations_preserve_the_last_successful_fit():
    data, relations = inputs()
    model = AlignedUMAP(**PUBLIC["settings"])
    with pytest.raises(ValueError, match="requires relations"):
        model.fit(data)
    assert not hasattr(model, "n_models_")
    with pytest.raises(AttributeError, match="n_models_"):
        model.update(data[2], relations=relations[1])
    model.fit(data[:2], relations=relations[:1])
    fitted = model.embeddings_
    with pytest.raises(ValueError, match="requires relations"):
        model.update(data[2])
    with pytest.raises(AssertionError):
        model.update(data[2], relations=relations[1:])
    with pytest.raises(ValueError, match="max"):
        model.update(data[2], relations={})
    assert model.embeddings_ is fitted
    assert len(fitted) == 2
    assert_graphs(model, PUBLIC["cases"]["update-0"]["after"]["graphs"])
    model.update(data[2], relations=relations[1])
    assert len(fitted) == model.n_models_ == 3
