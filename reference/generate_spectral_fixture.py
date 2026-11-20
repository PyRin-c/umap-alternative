"""Capture zero-epoch spectra and caller RNG checkpoints through public UMAP."""

import argparse
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import warnings

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from scipy.sparse.csgraph import connected_components
from umap import UMAP


def snapshot(points, neighbors, dimensions, seed):
    rng = np.random.RandomState(seed)
    model = UMAP(
        init="spectral", n_epochs=0, n_neighbors=neighbors, n_components=dimensions,
        random_state=rng, n_jobs=1, compatibility_layout=True, a=1.7, b=0.9,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        embedding = model.fit_transform(points.copy())

    graph = model.graph_
    assert graph.shape == (len(points), len(points))
    assert graph.data.dtype == np.float32
    assert connected_components(graph, directed=False, return_labels=False) == 1
    assert embedding.shape == (len(points), dimensions)
    assert embedding.dtype == np.float32
    return (
        {
            "indptr": graph.indptr.tolist(),
            "indices": graph.indices.tolist(),
            "bits": graph.data.view(np.uint32).tolist(),
        },
        embedding.view(np.uint32).ravel().tolist(),
        rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
    )


def generate():
    seeded_fixture = (
        Path(__file__).resolve().parent / "fixtures"
        / "umap-learn-0.6.0-seeded-fit-windows-x86_64.json"
    )
    weighted = json.loads(seeded_fixture.read_text(encoding="utf-8"))["inputs"]["weighted"]
    inputs = {"weighted": np.array(weighted, dtype=np.float32)}
    cases = [
        ("weighted", 4, dimensions, seed)
        for dimensions, seed in [(2, 42), (2, 0), (2, 2**32 - 1), (1, 7), (6, 42), (7, 42)]
    ]
    for rows, features, neighbors, dimensions, seed in [
        (3, 2, 2, 2, 9), (4, 2, 3, 3, 64), (9, 3, 5, 8, 88),
        (13, 3, 5, 2, 77), (19, 4, 7, 3, 92), (32, 5, 8, 2, 438),
        (59, 5, 9, 2, 21),
        (36, 3, 7, 1, 54), (48, 3, 7, 1, 54),
        (53, 3, 7, 1, 12), (83, 4, 10, 2, 18),
    ]:
        name = f"cloud_{rows}"
        inputs[name] = np.random.RandomState(seed).uniform(-1, 1, (rows, features)).astype(np.float32)
        cases.append((name, neighbors, dimensions, seed))

    graphs = {}
    results = []
    for name, neighbors, dimensions, seed in cases:
        graph, bits, next_words = snapshot(inputs[name], neighbors, dimensions, seed)
        assert (graph, bits, next_words) == snapshot(inputs[name], neighbors, dimensions, seed)
        assert graph == graphs.setdefault(name, graph)
        results.append({
            "graph": name,
            "n_components": dimensions,
            "seed": seed,
            "bits": bits,
            "next_rng_words": next_words,
        })
    return {
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "umap-learn")},
        },
        "graphs": graphs,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = generate()
    # Keep each CSR graph and each result together on one line.
    environment = json.dumps(fixture["environment"])
    graphs = ",\n".join(
        f"    {json.dumps(name)}: {json.dumps(graph)}" for name, graph in fixture["graphs"].items()
    )
    cases = ",\n".join("    " + json.dumps(case) for case in fixture["cases"])
    output = (
        f'{{\n  "environment": {environment},\n'
        f'  "graphs": {{\n{graphs}\n  }},\n'
        f'  "cases": [\n{cases}\n  ]\n}}\n'
    )
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
