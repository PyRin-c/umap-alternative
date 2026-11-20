"""Record inverse-transform checkpoints from the public umap-learn estimator."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from umap import UMAP


OPTIONS = dict(
    n_neighbors=4,
    init="random",
    compatibility_layout=True,
    n_epochs=0,
    learning_rate=1.0,
    random_state=42,
    transform_seed=24,
    n_jobs=1,
    a=1.7,
    b=0.9,
)


def checkpoint(model, queries):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.inverse_transform(queries)
    assert result.dtype == np.float32
    return dict(
        bits=result.ravel().view(np.uint32).tolist(),
        warnings=[str(item.message) for item in caught],
    )


def observe(name, training, embedding, queries, settings, **overrides):
    options = OPTIONS | dict(n_components=embedding.shape[1]) | overrides
    model = UMAP(**options).fit(training)
    # Hold the public embedding fixed as we vary inverse settings after fitting.
    model.embedding_ = embedding.copy()
    checks = []
    for epochs, rate, strength in settings:
        model.set_params(
            n_epochs=epochs, negative_sample_rate=rate, repulsion_strength=strength
        )
        checks.append(
            dict(
                fit_epochs=epochs,
                negative_sample_rate=rate,
                repulsion_strength=strength,
                **checkpoint(model, queries),
            )
        )
    assert np.array_equal(model.embedding_, embedding)
    # SciPy's public geometry API supplies the Rust test's fixed geometry.
    geometry = Delaunay(embedding, qhull_options="QJ")
    return dict(
        name=name,
        n_features=training.shape[1],
        n_components=embedding.shape[1],
        n_neighbors=options["n_neighbors"],
        a=options["a"],
        b=options["b"],
        initial_alpha=options["learning_rate"] / 4,
        transform_seed=options["transform_seed"],
        training=training.ravel().tolist(),
        embedding=embedding.ravel().tolist(),
        queries=queries.ravel().tolist(),
        simplices=geometry.simplices.ravel().tolist(),
        locations=[None if i < 0 else int(i) for i in geometry.find_simplex(queries)],
        checkpoints=checks,
    )


def boundaries(training):
    rows = len(training)
    test_inputs = [
        ("empty", {}, training, None, np.empty((0, 2), np.float32)),
        ("wrong_width", {}, training, None, [[0.0]]),
        ("nonfinite_query", {}, training, None, [[0.0, np.nan]]),
        ("sparse_query", {}, training, None, csr_matrix([[0.0, 1.0]])),
        ("sparse_training", {}, csr_matrix(training), None, [[0.0, 1.0]]),
        ("graph", {"transform_mode": "graph"}, training, None, [[0.0, 1.0]]),
        ("densmap", {"densmap": True}, training, None, [[0.0, 1.0]]),
        ("metric_without_gradient", {"metric": "hamming"}, training, None, [[0.0, 1.0]]),
        (
            "precomputed",
            {"metric": "precomputed"},
            np.linalg.norm(training[:, None] - training, axis=-1),
            None,
            [[0.0, 1.0]],
        ),
        ("single_row", {}, training[:1], None, [[0.0, 1.0]]),
        (
            "too_few_for_simplex",
            {"n_components": rows},
            training,
            np.eye(rows, dtype=np.float32),
            np.zeros((1, rows), np.float32),
        ),
        ("one_component", {"n_components": 1}, training, None, [[0.0]]),
        ("unfitted", {}, None, None, [[0.0, 1.0]]),
    ]
    output = {}
    for name, options, features, embedding, query in test_inputs:
        model = UMAP(**(OPTIONS | options))
        if features is not None:
            model.fit(features)
        if embedding is not None:
            model.embedding_ = embedding
        try:
            result = checkpoint(model, query)
        except Exception as error:
            # Record only the public exception and first message line.
            result = dict(error=type(error).__name__, message=str(error).splitlines()[0])
        output[name] = result
    return output


def diagnostic_ties():
    cases = []
    for rows, dim in [(8, 2), (25, 2), (36, 3)]:
        if dim == 2:
            angles = np.linspace(0, 2 * np.pi, rows, endpoint=False)
            points = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
            queries = np.array([[0, 0], [0, 9], [9, 0]], np.float32)
        else:
            points = np.array(
                list(product((-1, 0, 1), repeat=3)) + [(0, -1, 0)] * 9,
                np.float32,
            )
            queries = np.array(
                [[0, 0, 0], [0, -9, 0], [0.5, 0.5, 0.5], [0, 0, 7]],
                np.float32,
            )
        for width in (1, 2, 3, 5, 8):
            raw = np.random.default_rng(44).normal(size=(rows, width)).astype(np.float32)
            cases.append(
                observe(f"ties_{rows}_{width}", raw, points, queries, [(0, 5, 1.0)])
            )
    return cases


def default_schedule():
    training = np.array([[0, 0], [1.2, 1.5], [-0.1, 0.6], [1.6, -0.8], [-0.2, 0.1]], np.float32)
    points = np.array([[0, 0], [2, 0.1], [0.2, 1], [1, 1.1], [-0.3, -0.1]], np.float32)
    model = UMAP(**(OPTIONS | dict(n_neighbors=3, repulsion_strength=0))).fit(training)
    model.embedding_ = points
    result = []
    for rows in (1, 9_999, 10_000, 10_001):
        query = np.repeat(np.array([[0.2, 0.2]], np.float32), rows, axis=0)
        outputs = {}
        for epochs in (None, 90, 300):
            model.set_params(n_epochs=epochs)
            output = model.inverse_transform(query)
            assert np.array_equal(output, np.broadcast_to(output[0], output.shape))
            outputs[str(epochs)] = output[0].view(np.uint32).tolist()
        result.append(dict(query_rows=rows, first_row_bits=outputs))
    return result


def make_fixture():
    rng = np.random.default_rng(40)
    embedding = (rng.normal(size=(12, 2)) * 2).astype(np.float32)
    queries = np.vstack(
        [
            embedding[:2],
            embedding.mean(axis=0),
            np.float32(0.3) * embedding[3] + np.float32(0.7) * embedding[1],
            [[9, 8], [-9, -8], [-8, 9], [0, 7]],
            embedding[:1],
            [[9, 8]],
        ]
    ).astype(np.float32)
    training = np.random.default_rng(41).uniform(-2, 2, (len(embedding), 3)).astype(np.float32)
    cases = [
        observe(
            "fixed_schedule",
            training,
            embedding,
            queries,
            [
                (0, 5, 1.0), (5, 5, 1.0), (6, 1, 0.0), (9, 1, 0.0),
                (12, 1, 0.0), (30, 1, 0.0), (45, 1, 0.0),
                (6, 2, 0.37), (6, 5, 1.0), (9, 2, 0.37), (30, 5, 1.0),
                (45, 5, 1.0), (90, 5, 1.0), ([6, 12], 5, 1.0),
                (300, 5, 1.0), (None, 5, 1.0),
            ],
        ),
    ]
    schedule_training = np.array(
        [[0, 0], [1.2, 1.5], [-0.1, 0.6], [1.6, -0.8], [-0.2, 0.1]], np.float32
    )
    schedule_points = np.array(
        [[0, 0], [1, 0], [0.2, 3], [1, 3.1], [-3, -0.1]], np.float32
    )
    for label, a, epochs in [
        ("interval_below_two", 1.0 - 2.0**-30, 9),
        ("interval_above_two", 1.0 + 2.0**-30, 9),
        ("interval_rounded_above_three", 2.0, 15),
    ]:
        cases.append(
            observe(
                label, schedule_training, schedule_points, schedule_points[:1],
                [(0, 1, 0.0), (epochs, 1, 0.0)],
                a=a, b=1.0, n_neighbors=3, learning_rate=0.4, transform_seed=2**32 - 1,
            )
        )
    for n, dim in [(12, 2), (40, 3), (25, 5)]:
        rng = np.random.default_rng(40 + n + dim)
        points = rng.normal(size=(n, dim)).astype(np.float32)
        query = np.vstack(
            [points[:3], rng.normal(size=(15, dim)) * 0.7, rng.normal(size=(8, dim)) * 7]
        ).astype(np.float32)
        for width in (1, 3, 8, 17):
            raw = rng.normal(size=(n, width)).astype(np.float32)
            cases.append(observe(f"geometry_{n}_{dim}_{width}", raw, points, query, [(0, 5, 1.0)]))

    points = np.column_stack([np.arange(len(training))] * 2).astype(np.float32)
    cases.append(
        observe(
            "rank_deficient", training, points, np.array([[0, 1], [-1, -1]], np.float32),
            [(0, 5, 1.0)],
        )
    )
    rng = np.random.default_rng(112)
    rng.uniform(-1, 1, 12)
    angles = np.sort(rng.uniform(-1, 1, 12))
    points = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    cases.append(
        observe(
            "float32_exterior_seed", np.arange(12, dtype=np.float32).reshape(-1, 1),
            points, np.zeros((1, 2), np.float32), [(0, 5, 1.0)],
        )
    )
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    raw = np.random.default_rng(44).normal(size=(len(points), 3)).astype(np.float32)
    cases.append(
        observe(
            "distance_ties", raw, points, np.array([[0, 0], [0, 9], [9, 0]], np.float32),
            [(0, 5, 1.0)],
        )
    )

    # Reuse the recorded original fit and forward query values independently
    # of either implementation's fit and transform rounding.
    original = json.loads(
        Path(__file__).with_name("fixtures")
        .joinpath("umap-learn-0.6.0-transform-windows-x86_64.json").read_text()
    )
    fit = original["cases"]["embedding"]

    def as_float(record):
        return np.array(record["bits_uint32"], np.uint32).view(np.float32).reshape(record["shape"])

    points = as_float(fit["training"]["embedding"])
    original_training = as_float(original["inputs"]["training"])
    direct_queries = as_float(original["inputs"]["inverse_queries"])
    roundtrip_queries = as_float(fit["transform"]["output"])
    curve = json.loads(
        Path(__file__).with_name("fixtures")
        .joinpath("umap-learn-0.6.0-curve-windows-x86_64.json").read_text()
    )
    bits = next(row["bits"] for row in curve["samples"] if row["spread"] == 1 and row["min_dist"] == 0.1)
    a, b = np.array(bits, np.uint64).view(np.float64).tolist()
    for name, query, original_key in (
        ("original_interior", direct_queries, "inverse_transform"),
        ("original_exterior", roundtrip_queries, "inverse_transform_roundtrip"),
    ):
        case = observe(
            name, original_training, points, query,
            [(0, 5, 1.0), (45, 5, 1.0)], a=a, b=b,
        )
        previous_bits = as_float(fit[original_key]["output"]).ravel().view(np.uint32).tolist()
        assert case["checkpoints"][-1]["bits"] == previous_bits
        cases.append(case)
    cases.append(
        observe(
            "exterior_distance_overflow",
            training,
            embedding,
            np.array([[1e20, 2e20]], np.float32),
            [(0, 5, 1.0)],
        )
    )
    return dict(
        versions={
            key: version(key)
            for key in ("umap-learn", "numpy", "scipy", "numba", "scikit-learn")
        },
        cases=cases,
        tie_diagnostics=diagnostic_ties(),
        boundaries=boundaries(training),
        default_schedule=default_schedule(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = make_fixture()
    # Keep the matrices compact and each checkpoint easy to locate in a diff.
    cases = "cases", "tie_diagnostics"
    sections = []
    for key, value in result.items():
        if key in cases:
            records = ",\n".join("    " + json.dumps(row, separators=(",", ":")) for row in value)
            sections.append(f'  "{key}": [\n{records}\n  ]')
        else:
            section = json.dumps(value, indent=2).replace("\n", "\n  ")
            sections.append(f'  "{key}": {section}')
    args.output.write_text(
        "{\n" + ",\n".join(sections) + "\n}\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"{type(error).__name__}: {str(error).splitlines()[0]}") from None
