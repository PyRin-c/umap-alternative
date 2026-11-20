import json
import platform
from pathlib import Path

import numpy as np
import pytest
import scipy
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import ArpackNoConvergence

import umap_alternative._spectral as spectral


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "reference/fixtures"
    / "umap-learn-0.6.0-spectral-x86_64.json"
)
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
RUNTIME = {
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
}
REFERENCE_RUNTIME = (
    platform.system() in {"Windows", "Linux"}
    and platform.machine().lower() in {"amd64", "x86_64"}
    and all(REFERENCE["environment"][key] == value for key, value in RUNTIME.items())
)


@pytest.mark.parametrize(
    "case",
    REFERENCE["cases"],
    ids=[f"{case['graph']}-{case['n_components']}d-{case['seed']}" for case in REFERENCE["cases"]],
)
def test_spectral_matches_public_compatibility_layout(case):
    graph_data = REFERENCE["graphs"][case["graph"]]
    n_rows = len(graph_data["indptr"]) - 1
    weights = np.array(graph_data["bits"], dtype=np.uint32).view(np.float32)
    graph = csr_matrix(
        (weights, graph_data["indices"], graph_data["indptr"]), shape=(n_rows, n_rows)
    )
    rng = np.random.RandomState(case["seed"])
    coords = spectral._spectral_initialization(graph, case["n_components"], rng)
    assert coords.dtype == np.float32
    assert coords.flags.c_contiguous
    assert coords.shape == (n_rows, case["n_components"])
    assert np.all(np.isfinite(coords))
    assert 9.999 <= np.abs(coords).max() <= 10.001

    # The public fit uses three optimizer seed words after initialization.
    words = rng.randint(0, 2**32, size=7, dtype=np.uint32)
    np.testing.assert_array_equal(words[3:], case["next_rng_words"])

    # The optimizer normalizes the raw coordinates using this float32 order.
    low = coords.min(axis=0)
    normalized = np.float32(10) * (coords - low) / (coords.max(axis=0) - low)
    if REFERENCE_RUNTIME:
        np.testing.assert_array_equal(normalized.view(np.uint32).ravel(), case["bits"])


def test_spectral_two_connected_vertices_preserve_an_existing_rng_stream():
    graph = csr_matrix([[0, 1], [1, 0]], dtype=np.float32)
    rng = np.random.RandomState(42)
    rng.standard_normal()
    expected_rng = np.random.RandomState()
    expected_rng.set_state(rng.get_state())
    expected_rng.standard_normal((2, 2))
    noise = (1e-4 * expected_rng.standard_normal((2, 1))).astype(np.float32)

    coords = spectral._spectral_initialization(graph, 1, rng)
    # The two-vertex Laplacian has one nontrivial eigenvector proportional to [1, -1].
    scaled = coords - noise
    np.testing.assert_allclose(abs(scaled), 10.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(scaled[0], -scaled[1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(rng.standard_normal(9), expected_rng.standard_normal(9))


@pytest.mark.parametrize(
    "shape, dimensions", [((0, 0), 1), ((1, 1), 1), ((3, 3), 0), ((3, 3), 3), ((3, 3), 4)]
)
def test_spectral_rejects_out_of_range_dimensions(shape, dimensions):
    with pytest.raises(ValueError, match="1 <= n_components < n_rows"):
        spectral._spectral_initialization(csr_matrix(shape), dimensions, np.random.RandomState(0))


def test_spectral_requires_a_square_graph():
    with pytest.raises(ValueError, match="requires a square graph"):
        spectral._spectral_initialization(csr_matrix((3, 2)), 1, np.random.RandomState(0))


def test_spectral_rejects_a_disconnected_graph_before_consuming_rng():
    graph = diags([[1, 0, 1], [1, 0, 1]], [-1, 1], format="csr", dtype=np.float32)
    rng = np.random.RandomState(0)
    with pytest.raises(NotImplementedError, match="requires a connected graph"):
        spectral._spectral_initialization(graph, 1, rng)
    np.testing.assert_array_equal(rng.standard_normal(9), np.random.RandomState(0).standard_normal(9))


def test_spectral_propagates_solver_failures_before_consuming_rng(monkeypatch):
    graph = diags([1, 1], [-1, 1], shape=(3, 3), format="csr", dtype=np.float32)
    rng = np.random.RandomState(0)

    def fail(*args, **kwargs):
        raise ArpackNoConvergence("test failure", np.empty(0), np.empty((3, 0)))

    monkeypatch.setattr(spectral, "eigsh", fail)
    with pytest.raises(ArpackNoConvergence, match="test failure"):
        spectral._spectral_initialization(graph, 1, rng)
    np.testing.assert_array_equal(rng.standard_normal(9), np.random.RandomState(0).standard_normal(9))
