"""Capture boolean-feature UMAP updates from the pinned public wheel."""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack
from umap import UMAP

from generate_sparse_lp_transform_fixture import encode
from generate_sparse_update_fixture import state


def observe(call, model=None, *, expect_none=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except Exception as error:
            outcome = {"error": type(error).__name__}
        else:
            outcome = {}
            if expect_none:
                assert result is None
            if model is not None:
                for attribute in ("graph_", "embedding_"):
                    if result is getattr(model, attribute, None):
                        outcome["identity"] = attribute
                        break
                else:
                    outcome["result"] = encode(result)
    return outcome | {"warnings": [item.category.__name__ for item in caught]}


def append_candidate(training, batch):
    if training.shape[1] != batch.shape[1]:
        return None
    if issparse(training):
        return vstack((training, batch), format="csr")
    if not issparse(batch):
        return np.concatenate((training, batch), axis=0)
    return None


def build():
    rng = np.random.RandomState(417)
    values = rng.choice(np.array([-3, -1, 0, 0, 0, 2], np.float32), (28, 6))
    values[[0, 13]] = 0
    values[2] = -values[1]
    values[3] = values[1]
    values[:12, -1] = 1
    values[0] = 0
    base = values[:12]
    axes = np.array([[1, 0], [-2, 0], [3, 0], [0, 4], [0, -5], [0, 6],
                     [1, 1], [-1, -1], [1, -1], [0, 0]], np.float32)
    stored = csr_matrix((np.array([-0., 0.], np.float32), [0, 5], [0, 2]), shape=(1, 6))
    inputs = {}
    for stem, matrix in {"base": base, "tails": values[12:14], "last": values[14:15],
                         "repeat": values[1:2], "larger": values[:24], "larger_tail": values[24:],
                         "axes": axes[:9], "axis_tail": axes[9:], "wide": np.zeros((1, 7), np.float32),
                         "zero": np.zeros((1, 6), np.float32), "empty": base[:0],
                         "nan": np.full((1, 6), np.nan, np.float32)}.items():
        inputs[stem + "_dense"] = matrix
        inputs[stem + "_csr"] = csr_matrix(matrix)
    inputs["stored_csr"] = stored
    inputs["stored_base_csr"] = vstack((stored, inputs["base_csr"][1:]), format="csr")
    inputs["tails_csc"] = inputs["tails_csr"].tocsc()
    inputs["last_coo"] = inputs["last_csr"].tocoo()
    options = {"n_neighbors": 4, "n_components": 2, "random_state": 41, "transform_seed": 0,
               "n_epochs": 0, "init": "random", "n_jobs": 1, "compatibility_layout": True,
               "a": 1.576943460, "b": .895060879}
    cases = {}

    def capture(metric, form, name, calls, *, source="base", **overrides):
        settings = {"metric": metric} | overrides
        original_name = f"{source}_{form}"
        original = inputs[original_name].copy()
        model = UMAP(**(options | settings))
        fit = observe(lambda: model.fit(original))
        record = {"training": original_name, "options": settings, "fit": fit,
                  "initial": state(model), "steps": []}
        cases[f"{metric}_{form}_{name}"] = record
        if "error" in fit:
            return
        training = original
        for batch_name, parameters in calls:
            batch = inputs[batch_name].copy()
            before = state(model)
            previous = [getattr(model, attribute, None) for attribute in before]
            saved = {key: getattr(model, key) for key in parameters}
            model.set_params(**parameters)
            outcome = observe(lambda: model.update(batch), expect_none=True)
            after = state(model)
            candidate = append_candidate(training, batch)
            queries = {"original": original, "held_out": original[:1]}
            if candidate is not None:
                queries["concatenated"] = candidate
            transforms = {key: observe(lambda: model.transform(query.copy()), model)
                          for key, query in queries.items()}
            if (metric, form, name, batch_name) == ("dice", "dense", "epochs", "tails_dense"):
                query = transforms["original"]
                # The same returned bits have been observed with either warning outcome.
                variants = [[], ["RuntimeWarning"]]
                assert query.pop("warnings") in variants
                query["warning_variants"] = variants
            query_mode = model.transform_mode
            model.set_params(transform_mode="graph")
            graph_query = observe(lambda: model.transform(original[:1].copy()), model)
            model.set_params(transform_mode=query_mode)
            step = {
                "batch": batch_name, "options": parameters, "outcome": outcome, "after": after,
                "replaced": [getattr(model, attr, None) is not old for attr, old in zip(before, previous)],
                "unchanged": [old is None or encode(old) == before[attr] for attr, old in zip(before, previous)],
                "input_unchanged": encode(batch) == encode(inputs[batch_name]),
                "transforms": transforms, "held_out_graph": graph_query,
            }
            if parameters.get("n_epochs") == -1 and after["graph_"]["shape"] != before["graph_"]["shape"]:
                model.set_params(n_epochs=6)
                step["after_epoch_reset"] = {
                    "held_out": observe(lambda: model.transform(original[:1].copy()), model),
                    "batch": observe(lambda: model.transform(batch.copy()), model),
                }
            model.set_params(**saved)
            record["steps"].append(step)
            if candidate is not None and after["graph_"] is not None and (
                len(candidate.shape) == 2 and after["graph_"]["shape"][0] == candidate.shape[0]
            ):
                training = candidate

    def call(batch, **parameters):
        return batch, parameters

    metrics = ("hamming", "matching", "jaccard", "dice", "russellrao", "rogerstanimoto",
               "sokalmichener", "sokalsneath", "kulsinski", "yule")
    for metric in metrics:
        for form in ("dense", "csr"):
            if metric == "yule" and form == "csr":
                capture(metric, form, "invalid", [])
                continue
            capture(metric, form, "epochs", [call(f"tails_{form}"), call(f"last_{form}", n_epochs=1),
                                             call(f"repeat_{form}", n_epochs=9, transform_seed=2**32-1)])
            capture(metric, form, "cap", [call(f"tails_{form}", n_epochs=6, n_neighbors=3),
                                         call(f"last_{form}", n_epochs=6)], n_neighbors=15, n_epochs=6)
            capture(metric, form, "ties", [call(f"larger_tail_{form}", n_epochs=9)],
                    source="larger", n_neighbors=5)
    upper = float(np.nextafter(np.float32(1), np.float32(2)))
    for form in ("dense", "csr"):
        for metric in ("hamming", "jaccard", "dice"):
            for label, cutoff in (("equal", 1.0), ("next32", upper)):
                capture(metric, form, f"cutoff_{label}", [call(f"axis_tail_{form}", n_epochs=6)],
                        source="axes", n_neighbors=7, disconnection_distance=cutoff)
        opposite = "csr" if form == "dense" else "dense"
        capture("hamming", form, "formats", [call("tails_csc"), call("last_coo", n_epochs=6),
                                             call(f"tails_{opposite}"), call(f"repeat_{form}")])
        capture("hamming", form, "errors", [call(f"empty_{form}"), call(f"wide_{form}"), call(f"nan_{form}"),
                                            call(f"last_{form}", transform_seed=-1),
                                            call(f"last_{form}", n_epochs=-1), call(f"repeat_{form}")])
        capture("matching", form, "graph", [call(f"last_{form}"), call(f"repeat_{form}")], transform_mode="graph")
    for metric in ("hamming", "jaccard", "kulsinski"):
        capture(metric, "csr", "zeros", [call("stored_csr"), call("zero_csr", n_epochs=6)],
                source="stored_base")
    return {"packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                        "scikit-learn", "pynndescent")},
            "options": options, "inputs": {key: encode(value) for key, value in inputs.items()}, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = build()
    header = json.dumps({key: value for key, value in fixture.items() if key != "cases"},
                        allow_nan=False, separators=(",", ":"))
    cases = ",\n".join(
        f"    {json.dumps(key)}: {json.dumps(value, allow_nan=False, separators=(',', ':'))}"
        for key, value in fixture["cases"].items()
    )
    output = f'{header[:-1]},\n  "cases": {{\n{cases}\n  }}\n}}\n'
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, "Regenerated fixture differs\n")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
