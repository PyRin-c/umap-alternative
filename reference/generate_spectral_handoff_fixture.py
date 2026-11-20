"""Capture a compact spectral fit and MT19937 snapshots using public APIs."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import warnings

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from umap import UMAP


def word_draws(rng, count):
    return rng.randint(0, 2**32, size=count, dtype=np.uint32)


def rng_snapshot(rng):
    algorithm, words, cursor, has_gaussian, cached_gaussian = rng.get_state()
    assert algorithm == "MT19937"
    assert words.shape == (624,) and words.dtype == np.uint32
    return words.tolist(), cursor, has_gaussian, cached_gaussian


def state_case(seed):
    rng = np.random.RandomState(seed)
    seeded_words, cursor, _, _ = rng_snapshot(rng)
    assert cursor == 624
    first_block = word_draws(rng, 624)
    block_words, cursor, _, _ = rng_snapshot(rng)
    assert cursor == 624

    continuations = {}
    for cursor in (0, 623, 624):
        rng.set_state(("MT19937", np.array(block_words, dtype=np.uint32), cursor, 0, 0.0))
        uniform_bits = rng.random_sample(3).view(np.uint64).tolist()
        next_words = word_draws(rng, 8).tolist()
        continuations[str(cursor)] = {"uniform_bits": uniform_bits, "next_words": next_words}
    rng.set_state(("MT19937", np.array(block_words, dtype=np.uint32), 0, 0, 0.0))
    np.testing.assert_array_equal(word_draws(rng, 624), first_block)
    return {"seeded_words": seeded_words, "block_words": block_words, "continuations": continuations}


def graph_bits(graph):
    assert graph.has_sorted_indices and graph.data.dtype == np.float32
    return {
        "indptr": graph.indptr.tolist(),
        "indices": graph.indices.tolist(),
        "bits": graph.data.view(np.uint32).tolist(),
    }


def coordinate_bits(values):
    assert values.dtype == np.float32 and np.isfinite(values).all()
    return values.view(np.uint32).ravel().tolist()


def spectral_layout(graph, n_components, rng):
    """Reconstruct the raw coordinates and draws from public SciPy and NumPy."""
    n_rows = graph.shape[0]
    degree = np.asarray(graph.sum(axis=0)).ravel()
    inverse_roots = diags(1 / np.sqrt(degree))
    laplacian = eye(n_rows, dtype=np.float64, format="csr") - (inverse_roots @ graph @ inverse_roots)
    k = n_components + 1
    values, vectors = eigsh(
        laplacian, k=k, which="SM", v0=np.ones(n_rows), ncv=2 * k + 1, tol=1e-4,
    )
    basis = vectors[:, np.argsort(values)[1:k]]
    scaled = np.ascontiguousarray((10 / np.abs(basis).max()) * basis, dtype=np.float32)
    rng.standard_normal((n_rows, k))
    return scaled + (1e-4 * rng.standard_normal(scaled.shape)).astype(np.float32)


def spectral_case(points, settings, seed, expected_words):
    def fit(epochs):
        rng = np.random.RandomState(seed)
        estimator = UMAP(**settings, random_state=rng, n_epochs=epochs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = estimator.fit_transform(points.copy())
        return estimator.graph_, values, rng_snapshot(rng)

    graph, zero_epoch, post_fit_state = fit(0)
    rng = np.random.RandomState(seed)
    raw = spectral_layout(graph, settings["n_components"], rng)
    words, cursor, _, _ = rng_snapshot(rng)
    assert words == expected_words
    low = raw.min(axis=0)
    normalized = np.float32(10) * (raw - low) / (raw.max(axis=0) - low)
    assert coordinate_bits(normalized) == coordinate_bits(zero_epoch)
    rng.randint(-2147483647, 2147483646, size=3)
    assert rng_snapshot(rng) == post_fit_state
    assert post_fit_state[0] == words

    graph_record = graph_bits(graph)
    checkpoints = {"0": coordinate_bits(zero_epoch)}
    for epochs in (2, 3, 11, 30, 100):
        observed_graph, coordinates, observed_state = fit(epochs)
        assert graph_bits(observed_graph) == graph_record
        assert observed_state == post_fit_state
        checkpoints[str(epochs)] = coordinate_bits(coordinates)
    return graph_record, {
        "seed": seed,
        "cursor": cursor,
        "raw_bits": coordinate_bits(raw),
        "post_fit_cursor": post_fit_state[1],
        "post_fit_next_words": word_draws(rng, 8).tolist(),
        "checkpoints": checkpoints,
    }


def generate():
    points = np.array(
        [[0, 0, 0], [0.2, 0.07, 0.1], [0.25, 0.89, 0.13], [0.76, 0.1, 0.83],
         [0.99, 0.95, 0.37], [1.9, 1.31, 0.33], [2.61, 1.7, 0.9], [3.3, 1.2, 1.1]],
        dtype=np.float32,
    )
    settings = dict(
        n_neighbors=4, n_components=2, metric="euclidean", n_jobs=1, init="spectral",
        compatibility_layout=True, learning_rate=0.13, repulsion_strength=0.37,
        negative_sample_rate=5, a=1.7, b=0.9,
    )
    states = {seed: state_case(seed) for seed in (0, 2**32 - 1)}
    graphs = []
    cases = []
    for seed, state in states.items():
        graph, case = spectral_case(points, settings, seed, state["block_words"])
        graphs.append(graph)
        cases.append(case)
    assert graphs[0] == graphs[1]
    return {
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "umap-learn")},
        "input": points.tolist(),
        "settings": settings,
        "graph": graphs[0],
        "states": states,
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = generate()
    entries = ",\n".join(f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in fixture.items())
    args.output.write_text(f"{{\n{entries}\n}}\n", encoding="utf-8", newline="\n")
