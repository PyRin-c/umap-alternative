"""Capture serial held-out layout checkpoints from the locked public wheel."""

import argparse
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from umap import UMAP


def bits(values):
    assert values.dtype == np.float32
    return values.view(np.uint32).tolist()


def from_bits(record):
    return np.asarray(record["bits_uint32"], dtype=np.uint32).view(np.float32)


def capture(training, queries, settings, *, fit_epochs=0, checked_query_rows=None):
    params = dict(
        n_neighbors=4,
        n_components=2,
        metric="euclidean",
        init="random",
        random_state=42,
        transform_seed=24,
        n_jobs=1,
        compatibility_layout=True,
        a=1.7,
        b=0.9,
        learning_rate=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        disconnection_distance=None,
        n_epochs=0,
    )
    params.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UMAP(**params).fit(training.copy())
        fitted = bits(model.embedding_)
        model.set_params(n_epochs=fit_epochs)
        if checked_query_rows is None:
            output = model.transform(queries.copy())
            assert bits(model.transform(queries.copy())) == bits(output)
            model.set_params(transform_mode="graph")
            graph = model.transform(queries.copy()).tocoo()
            stored_zeros = int(np.count_nonzero(graph.data == 0))
        else:
            repeated = np.resize(queries, (checked_query_rows, queries.shape[1]))
            result = model.transform(repeated)
            output = result[: len(queries)]
            assert np.array_equal(result.view(np.uint32), np.resize(
                output.view(np.uint32), result.shape
            ))
            stored_zeros = None
        assert bits(model.embedding_) == fitted

    return dict(
        training=bits(training),
        queries=bits(queries),
        checked_query_rows=checked_query_rows,
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        metric=params["metric"],
        fit_epochs=fit_epochs,
        training_epochs=params["n_epochs"],
        init=params["init"] if isinstance(params["init"], str) else None,
        fit_seed=params["random_state"],
        transform_seed=params["transform_seed"],
        a=params["a"],
        b=params["b"],
        learning_rate=params["learning_rate"],
        repulsion_strength=params["repulsion_strength"],
        negative_sample_rate=params["negative_sample_rate"],
        disconnection_distance=params["disconnection_distance"],
        stored_zeros=stored_zeros,
        fitted=fitted,
        output=bits(output),
    )


def column(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def make_fixture():
    base = json.loads(Path(__file__).with_name("fixtures").joinpath(
        "umap-learn-0.6.0-transform-windows-x86_64.json"
    ).read_text(encoding="utf-8"))
    x = from_bits(base["inputs"]["training"])
    q = from_bits(base["inputs"]["transform_queries"])
    repeated = np.concatenate([q, x[[0]], q[[0]]])
    cases = {}
    for epoch in (0, 1, 3, 5, 6, 8, 9, 18):
        cases[f"fit_{epoch}"] = capture(
            x, repeated, dict(n_epochs=epoch), fit_epochs=epoch
        )
    cases["different_transform_seed"] = capture(
        x, repeated, dict(n_epochs=18, transform_seed=4294967295), fit_epochs=18
    )
    cases["different_fit_seed"] = capture(
        x, repeated, dict(n_epochs=18, random_state=7), fit_epochs=18
    )
    assert cases["fit_18"]["fitted"] == cases["different_transform_seed"]["fitted"]
    assert cases["fit_18"]["output"] != cases["different_transform_seed"]["output"]
    assert cases["fit_18"]["fitted"] != cases["different_fit_seed"]["fitted"]

    for query_rows, explicit_epochs in ((10000, 300), (10001, 90)):
        case = capture(x, q, {}, fit_epochs=None, checked_query_rows=query_rows)
        explicit = capture(
            x, q, {}, fit_epochs=explicit_epochs, checked_query_rows=query_rows
        )
        assert case["output"] == explicit["output"]
        cases[f"default_{query_rows}"] = case

    zero_training = column([0, 0, 0, 0, 0.1, 10, 20, 21, 25, 30])
    cases["stored_zeros"] = capture(
        zero_training, column([0, 0]), dict(n_neighbors=5), fit_epochs=45
    )
    assert cases["stored_zeros"]["stored_zeros"] == 2
    coincident = np.column_stack(([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0]))
    cases["coincident_tails"] = capture(
        column(np.arange(8)), column([0.01, 4.01, 5.01, 7]),
        dict(init=coincident.astype(np.float32)), fit_epochs=45,
    )

    disconnected = column([0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7, 1.8, 5])
    for suffix, cutoff in (("at", 1.0), ("above", 1.0000001)):
        cases[f"cutoff_{suffix}"] = capture(
            disconnected, column([-1, 5, 15, 1.51, 0.43]),
            dict(disconnection_distance=cutoff), fit_epochs=45,
        )
    cases["all_disconnected"] = capture(
        disconnected, column([15, 23]), dict(disconnection_distance=1.0), fit_epochs=45
    )
    assert cases["all_disconnected"]["stored_zeros"] == 8

    # The float32 value is just below 7/11. Rounding the expected count
    # to float32 makes the positive cursor cross an integer epoch boundary.
    weight = np.float32(7 / 11)
    d1 = np.float32(-2.5 * np.log(float(weight)))
    d2 = np.float32(-2.5 * np.log((2 - weight) / 2))
    cases["schedule_boundary"] = capture(
        column([0, d1, d2, -d2, 8, 9, 10, 11]), column([0]),
        dict(a=1.31, b=1.063, learning_rate=0.71, negative_sample_rate=1, repulsion_strength=0),
        fit_epochs=42,
    )
    pruning_training = column(np.arange(12))
    pruning_queries = column([0, 500])
    pruning_init = capture(pruning_training, pruning_queries, dict(n_neighbors=10))
    cases["weak_query"] = capture(
        pruning_training, pruning_queries, dict(n_neighbors=10), fit_epochs=6
    )
    assert cases["weak_query"]["output"][1] == pruning_init["output"][1]
    assert cases["weak_query"]["output"][0] != pruning_init["output"][0]

    # Hold out different dimensions, RNG seeds, rates and weighting curves.
    for dim in (1, 4, 7, 35, 65):
        rng = np.random.RandomState(3917 + dim)
        settings = dict(
            init=rng.uniform(-7, 7, (len(x), dim)).astype(np.float32),
            n_components=dim,
            a=1.31,
            b=1.063,
            learning_rate=0.71,
            negative_sample_rate=7,
            repulsion_strength=0.41,
            transform_seed=4294967295 - dim * 15,
        )
        cases[f"wide_{dim}"] = capture(
            x, np.concatenate([q, x[[0, 3, 6]], q[[0]]]), settings, fit_epochs=81
        )

    # A separate training graph, query batch and metric exercise the replay
    # beyond the eight-row feature fixture and its original hyperparameters.
    rng = np.random.RandomState(8093)
    other_training = rng.uniform(-5, 5, (34, 5)).astype(np.float32)
    other_queries = rng.uniform(-5, 5, (25, 5)).astype(np.float32)
    other_queries = np.concatenate([other_queries, other_training[[3, 30]], other_queries[[8]]])
    cases["chebyshev_holdout"] = capture(
        other_training, other_queries,
        dict(
            metric="chebyshev", n_neighbors=10, n_components=4,
            init=rng.uniform(-7, 7, (34, 4)).astype(np.float32),
            a=0.95, b=1.18, repulsion_strength=0.47, learning_rate=0.66,
            negative_sample_rate=8, random_state=4, transform_seed=2808,
        ),
        fit_epochs=1200,
    )
    return {"umap-learn": version("umap-learn"), "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    assert version("umap-learn") == "0.6.0"
    fixture = make_fixture()
    arguments.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
