"""Observe dense metric updates using the public umap-learn wheel."""

import argparse
import json
import runpy
import warnings
from importlib.metadata import version
from pathlib import Path


# Reuse pinned update inputs and record formats without importing the port.
shared = runpy.run_path(str(Path(__file__).with_name("generate_update_fixture.py")))
np = shared["np"]
UMAP = shared["UMAP"]
array_record = shared["array_record"]
snapshot = shared["snapshot"]

TRAINING = [
    [0.12, 0.07, 1.2], [1, 4, 2], [0.78, 1.2, 3.1], [2.2, -3, 0.5],
    [-0.1, 0.7, 6], [0.3, -0.5, -0.9], [5, 6, 7],
]
BATCH = [[1.2, 2.8, 2.34], [-0.65, -0.70, -0.90], [1.1, -1.0, 0.42]]
SIGN_REVERSAL = [[1, 0], [-1, 0], [1, 0], [-1, 0]]
BASE = shared["BASE_PARAMETERS"] | {"a": 1.3, "b": 0.8}
NAN_ERROR = "Input contains NaN."


def cases():
    for metric in ("euclidean", "manhattan", "chebyshev", "cosine", "correlation"):
        yield metric + "_successive", TRAINING, {"metric": metric, "n_epochs": 9}, [
            (BATCH[:2], {"n_epochs": 0}),
            (BATCH[2:], {"n_epochs": 11, "n_neighbors": 6, "learning_rate": 0.4}),
        ]
        yield metric + "_positive", TRAINING, {"metric": metric, "n_epochs": 0}, [
            (BATCH, {"n_epochs": 9, "transform_seed": 2**32 - 1}),
        ]
    for metric in ("cosine", "correlation"):
        yield metric + "_antipodal", SIGN_REVERSAL, {
            "metric": metric, "n_neighbors": 3, "n_epochs": 0,
        }, [(SIGN_REVERSAL[:1], {"n_epochs": 6, "transform_seed": 0})]
    for metric, cutoff in (
        ("euclidean", 8.0), ("euclidean", 4.0), ("manhattan", 4.0),
        ("cosine", 0.31), ("correlation", 0.3),
    ):
        yield f"{metric}_cutoff_{cutoff}", TRAINING, {
            "metric": metric, "disconnection_distance": cutoff, "n_epochs": 0,
        }, [(BATCH, {"n_epochs": 6})]
    yield "manhattan_connectivity", TRAINING, {
        "metric": "manhattan", "local_connectivity": 1.5, "set_op_mix_ratio": 0.75, "n_epochs": 9,
    }, [(BATCH, {"n_epochs": None})]


def capture(training, fit_options, updates):
    training = np.asarray(training, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = UMAP(**(BASE | fit_options)).fit(training.copy())
    fitted = snapshot(model)
    fit_warnings = [warning.category.__name__ for warning in caught]
    records = []
    for batch, options in updates:
        model.set_params(**options)
        old_layout = model.embedding_
        status = "ok"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                assert model.update(np.asarray(batch, dtype=np.float32)) is None
            except ValueError as error:
                if str(error) != NAN_ERROR:
                    raise
                status = "coincident_nan"
        record = {
            "batch": batch, "options": options, "status": status,
            "warnings": [warning.category.__name__ for warning in caught],
            "state": snapshot(model),
        }
        if status == "ok":
            training = np.concatenate((training, np.asarray(batch, dtype=np.float32)))
            assert model.transform(training.copy()) is model.embedding_
            if fit_options.get("disconnection_distance") == 8.0:
                query = np.asarray([[40, 40, 40], [0.17, 0.24, 1.1]], dtype=np.float32)
                record["query"] = query.tolist()
                record["transform"] = array_record(model.transform(query))
        else:
            assert model.embedding_ is old_layout
            assert model.transform(training.copy()) is old_layout
        records.append(record)
    return {"fit": fitted, "fit_warnings": fit_warnings, "updates": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert version("umap-learn") == "0.6.0"
    result = {}
    for name, training, settings, updates in cases():
        try:
            first = capture(training, settings, updates)
            assert first == capture(training, settings, updates)
        except Exception as error:
            raise SystemExit(f"{name}: {type(error).__name__}") from None
        result[name] = {"training": training, "settings": BASE | settings} | first
    output = "{\n" + ",\n".join(
        f"  {json.dumps(name)}: {json.dumps(record, allow_nan=False)}" for name, record in result.items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            raise SystemExit("dense update metrics fixture differs")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
