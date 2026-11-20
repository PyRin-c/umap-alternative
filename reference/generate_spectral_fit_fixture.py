"""Record public spectral fit graph, embedding, and RNG checkpoints."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

from generate_spectral_handoff_fixture import coordinate_bits, graph_bits
import numpy as np
from umap import UMAP


def snapshot(points, params, seed, method):
    rng = np.random.RandomState(seed)
    model = UMAP(**params, random_state=rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = getattr(model, method)(points.copy())
    if "compatibility_layout" in params:
        assert not caught
    else:
        assert len(caught) == 1 and caught[0].category is FutureWarning
        assert "compatibility_layout was not set" in str(caught[0].message)
    assert fitted is (model if method == "fit" else model.embedding_)
    algorithm, _, cursor, has_gaussian, cached_gaussian = rng.get_state()
    assert algorithm == "MT19937"
    return (
        graph_bits(model.graph_),
        coordinate_bits(model.embedding_),
        {
            "cursor": cursor,
            "has_gaussian": has_gaussian,
            "cached_gaussian": cached_gaussian,
            "next_words": rng.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
        },
    )


def generate():
    fixtures = Path(__file__).resolve().parent / "fixtures"
    handoff = json.loads(
        (fixtures / "umap-learn-0.6.0-spectral-handoff-windows-x86_64.json").read_text(
            encoding="utf-8"
        )
    )
    metric = json.loads(
        (fixtures / "umap-learn-0.6.0-metric-graph-windows-x86_64.json").read_text(
            encoding="utf-8"
        )
    )
    inputs = {
        "weighted": np.asarray(handoff["input"], dtype=np.float32),
        "metric": np.asarray(metric["input"]["values"], dtype=np.float32),
        "weak": np.random.RandomState(4).normal(size=(13, 2)).astype(np.float32),
    }
    explicit = handoff["settings"]
    weak = explicit | {"n_neighbors": 5}
    specs = [
        ("weighted-one-epoch", "weighted", explicit | {"n_epochs": 1}, seed)
        for seed in (0, 2**32 - 1)
    ] + [
        ("default", "weighted", {"n_neighbors": 4, "n_jobs": 1}, 0),
        ("full-spectrum", "weighted", explicit | {"n_components": 7, "n_epochs": 0}, 42),
        ("one-component", "weighted", explicit | {"n_components": 1, "n_epochs": 11}, 7),
        ("manhattan", "metric", explicit | {
            "metric": "manhattan", "n_neighbors": 3, "n_epochs": 11,
        }, 0),
        ("chebyshev", "weighted", explicit | {"metric": "chebyshev", "n_epochs": 11}, 0),
    ] + [
        (f"weak-{epochs}-static", "weak", weak | {"n_epochs": epochs, "learning_rate": 0}, 0)
        for epochs in (0, 1, 10, 11)
    ] + [
        ("weak-11-optimized", "weak", weak | {"n_epochs": 11}, 2**32 - 1),
    ]

    baseline = UMAP(
        **weak, random_state=0, n_epochs=11, transform_mode="graph"
    ).fit(inputs["weak"])
    graphs = {"weak-unpruned": graph_bits(baseline.graph_)}
    cases = []
    for name, input_name, params, seed in specs:
        points = inputs[input_name]
        graph, bits, rng = snapshot(points, params, seed, "fit")
        assert (graph, bits, rng) == snapshot(points, params, seed, "fit_transform")
        graph_name = next((key for key, value in graphs.items() if value == graph), name)
        graphs[graph_name] = graph
        cases.append({
            "name": name,
            "input": input_name,
            "params": params,
            "seed": seed,
            "graph": graph_name,
            "bits": bits,
            "post_fit_rng": rng,
        })
    return {
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "umap-learn")},
        },
        "inputs": {name: points.tolist() for name, points in inputs.items()},
        "graphs": graphs,
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = generate()
    graphs = ",\n".join(
        f"    {json.dumps(name)}: {json.dumps(graph)}" for name, graph in fixture["graphs"].items()
    )
    cases = ",\n".join("    " + json.dumps(case) for case in fixture["cases"])
    args.output.write_text(
        f'{{\n  "environment": {json.dumps(fixture["environment"])},\n'
        f'  "inputs": {json.dumps(fixture["inputs"])},\n'
        f'  "graphs": {{\n{graphs}\n  }},\n'
        f'  "cases": [\n{cases}\n  ]\n}}\n',
        encoding="utf-8",
        newline="\n",
    )
