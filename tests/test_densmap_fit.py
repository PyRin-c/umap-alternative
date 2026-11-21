import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from umap_alternative import UMAP


FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"
LAYOUT = json.loads(
    (FIXTURES / "umap-learn-0.6.0-densmap-layout-windows-x86_64.json").read_bytes()
)
DENSE = json.loads((FIXTURES / "umap-learn-0.6.0-densmap-windows-x86_64.json").read_bytes())
FEATURES = np.asarray(DENSE["data_float32"], dtype=np.float32)
INITIAL = np.asarray(DENSE["init_float32"], dtype=np.float32)


def assert_values(actual, expected_bits):
    assert actual.dtype == np.float32
    actual = actual.ravel()
    expected = np.asarray(expected_bits, dtype=np.uint32)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected)


def precomputed_case(name, **overrides):
    case = LAYOUT["cases"][name]
    dataset = LAYOUT["datasets"][case["dataset"]]
    wide = np.asarray(dataset["data"], dtype=np.float32).astype(np.float64)
    delta = wide[:, None, :] - wide[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)
    initial = np.array(dataset["init"], dtype=np.float32)
    options = {
        "n_components": initial.shape[1],
        "n_neighbors": dataset["n_neighbors"],
        "n_epochs": [int(key) for key in case["checkpoints"]] or case["epochs"],
        "densmap": case["density"],
        "init": initial,
    }
    model = UMAP(**(LAYOUT["common_options"] | options | case["options"] | overrides))
    return case, model, distances


@pytest.mark.parametrize("name", LAYOUT["cases"])
def test_precomputed_density_and_plain_checkpoints_match_public_graph(name):
    case, model, distances = precomputed_case(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = model.fit_transform(distances)
    assert any("using precomputed metric" in str(w.message) for w in caught)
    runtime_warnings = [str(w.message) for w in caught if w.category is RuntimeWarning]
    expected_warnings = ["invalid value encountered in divide"] if case["dataset"] == "equal_input" else []
    assert runtime_warnings == expected_warnings
    expected_graph = case["graph"]
    np.testing.assert_array_equal(model.graph_.indptr, expected_graph["indptr"])
    np.testing.assert_array_equal(model.graph_.indices, expected_graph["indices"])
    assert_values(model.graph_.data, expected_graph["weights"])
    assert_values(model.embedding_, case["final"])
    if "original_radii" in case:
        assert fitted[0] is model.embedding_
        assert_values(fitted[1], case["original_radii"])
    else:
        assert fitted is model.embedding_
    if case["checkpoints"]:
        for snapshot, expected in zip(
            model.embedding_list_, case["checkpoints"].values(), strict=True
        ):
            assert snapshot.shape == model.embedding_.shape
            assert_values(snapshot, expected)
    else:
        assert not hasattr(model, "embedding_list_")
    assert model.transform(distances.copy()) is model.embedding_


@pytest.mark.parametrize("name", ["equal_embedding_unshifted", "equal_embedding_underflow_shift"])
def test_sampled_zero_embedding_variance_raises_public_exception(name):
    error = LAYOUT["errors"][name]
    _, model, distances = precomputed_case("equal_embedding_shifted", **error["options"])
    with pytest.warns(UserWarning, match="using precomputed metric"):
        with pytest.raises(ZeroDivisionError, match="^division by zero$"):
            model.fit_transform(distances)
    assert hasattr(model, "graph_")
    assert not hasattr(model, "embedding_")


def test_output_dens_rejects_nan_after_publishing_embedding():
    case, model, distances = precomputed_case("equal_input", output_dens=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="using precomputed metric")
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            with pytest.raises(ValueError, match=r"^Input contains NaN\.$"):
                model.fit_transform(distances)
    assert hasattr(model, "graph_")
    assert not hasattr(model, "rad_orig_")
    assert_values(model.embedding_, case["final"])
    for snapshot, expected in zip(model.embedding_list_, case["checkpoints"].values(), strict=True):
        assert_values(snapshot, expected)


def test_zero_input_variance_obeys_numpy_invalid_mode_without_updates():
    _, model, distances = precomputed_case("equal_input_zero")
    with pytest.warns(UserWarning, match="using precomputed metric"):
        with np.errstate(invalid="raise"):
            with pytest.raises(FloatingPointError, match="invalid value encountered in divide"):
                model.fit(distances)


@pytest.mark.parametrize("epochs", [0, 1, 2, 3, 10, 30])
@pytest.mark.parametrize("mode", ["densmap", "dens_lambda_0", "dens_frac_0", "dens_frac_1", "dens_var_0"])
def test_dense_densmap_fit_returns_embedding_and_radii_at_public_epochs(epochs, mode):
    config = DENSE["common_options"] | DENSE["mode_options"][mode]
    model = UMAP(**config, init=INITIAL, n_epochs=epochs)
    outputs = model.fit_transform(FEATURES)
    record = DENSE["cases"][f"{epochs}-{mode}"]
    for actual, index in zip(outputs, record["outputs"], strict=True):
        expected = DENSE["arrays"][index]
        assert actual.shape == tuple(expected["shape"])
        assert_values(actual, expected["bits_uint32"])
    assert model.transform(FEATURES.copy()) is model.embedding_
    with pytest.raises(NotImplementedError, match="not supported for densMAP"):
        model.transform(FEATURES[:2])
    with pytest.raises(ValueError, match="Inverse transform not available for densMAP"):
        model.inverse_transform(model.embedding_[:1])


@pytest.mark.parametrize("epochs", [[4, 0, 2, 2], (4, 0, 2), np.array([4, 0, 2])])
def test_requested_checkpoints_are_deduplicated_and_ordered_by_epoch(epochs):
    case, model, distances = precomputed_case("weighted_plain", n_epochs=epochs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(distances)
    assert len(model.embedding_list_) == 3
    for key, actual in zip((0, 2, 4), model.embedding_list_, strict=True):
        assert_values(actual, case["checkpoints"][str(key)])
    assert_values(model.embedding_, case["checkpoints"]["4"])
    snapshots = model.embedding_list_
    assert not np.shares_memory(snapshots[-1], model.embedding_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.set_params(n_epochs=4).fit(distances)
    assert model.embedding_list_ is snapshots


@pytest.mark.parametrize("epochs", [[], [0, -1], [0, 1.5], np.array([[0, 2]])])
def test_invalid_checkpoint_requests_are_rejected(epochs):
    model = UMAP(**DENSE["common_options"], n_epochs=epochs)
    with pytest.raises(ValueError, match="list of nonnegative integers"):
        model.fit(FEATURES)


@pytest.mark.parametrize("name", ["dens_frac", "dens_lambda", "dens_var_shift"])
def test_density_parameters_reject_the_first_value_below_zero(name):
    with np.errstate(under="ignore"):
        invalid = float(np.nextafter(0.0, -1.0))
    model = UMAP(**DENSE["common_options"], **{name: invalid})
    with pytest.raises(ValueError, match=name):
        model.fit(FEATURES)


def test_density_fraction_rejects_the_first_value_above_one():
    model = UMAP(**DENSE["common_options"], densmap=True, dens_frac=np.nextafter(1.0, 2.0))
    with pytest.raises(ValueError, match="dens_frac must be between"):
        model.fit(FEATURES)


def test_densmap_default_adds_two_hundred_epochs():
    options = DENSE["common_options"] | dict(init=INITIAL, densmap=True)
    default = UMAP(**options).fit_transform(FEATURES)
    explicit = UMAP(**options, n_epochs=700).fit_transform(FEATURES)
    assert_values(default, explicit.view(np.uint32).ravel())


def test_densmap_early_epoch_cutoff_preserves_public_weak_edges():
    # Public eight-row seed-298 distance probe. The directed edge weights
    # are between the reciprocals of the two short-run sampling budgets.
    features = np.array([
        [-0.9711973071098328], [-0.47282737493515015], [-0.38705480098724365],
        [-0.2833501398563385], [0.6346836686134338], [0.6785641312599182],
        [0.7226971387863159], [1.7857415676116943],
    ], dtype=np.float32)
    options = DENSE["common_options"] | dict(n_components=1, init="random", n_epochs=[0, 2])
    plain = UMAP(**options).fit(features)
    density = UMAP(**options, densmap=True, dens_frac=0.0).fit(features)
    np.testing.assert_array_equal(plain.graph_.indptr, density.graph_.indptr)
    np.testing.assert_array_equal(plain.graph_.indices, density.graph_.indices)
    edge_changed = plain.graph_.data != density.graph_.data
    assert np.count_nonzero(edge_changed) == 2
    np.testing.assert_array_equal(plain.graph_.data[edge_changed], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(
        density.graph_.data[edge_changed], np.full(2, 0.001743602566421032, dtype=np.float32)
    )


def test_zero_input_variance_scalar_fit_publishes_nan_embedding():
    case, model, distances = precomputed_case("equal_input", n_epochs=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="using precomputed metric")
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            assert model.fit(distances) is model
    assert_values(model.embedding_, case["final"])
    assert not hasattr(model, "embedding_list_")
