"""Verify the installed wheel against public Windows CPU ParametricUMAP probes."""

import json
from math import prod
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
INFERENCE = runpy.run_path(str(ROOT / "generate_parametric_fixture.py"))
STAGES = runpy.run_path(str(ROOT / "generate_parametric_stages_fixture.py"))


def values_only(record):
    return {key: value for key, value in record.items() if key not in ("message", "warnings")}


def report_bits(label, actual, expected):
    import numpy as np

    actual = np.asarray(actual, dtype=np.uint32)
    expected = np.asarray(expected, dtype=np.uint32)
    assert actual.shape == expected.shape, label
    exact = np.count_nonzero(actual == expected)
    error = np.max(np.abs(actual.view(np.float32) - expected.view(np.float32)), initial=0)
    print(f"{label}  {exact}/{actual.size} bits, max abs {error:.9g}", flush=True)


def check_stages():
    expected = STAGES["capture"](include_training_bits=True)
    actual = STAGES["capture"](use_wheel=True, include_training_bits=True)
    assert actual["edge_periods"] == expected["edge_periods"]
    for name, observed in actual["training_boundaries"].items():
        reference = expected["training_boundaries"][name]
        assert values_only(observed["fit_transform"]) == values_only(reference["fit_transform"]), name
        assert observed.get("observed_batch_size") == reference.get("observed_batch_size"), name
        assert observed.get("keras") == reference.get("keras"), name

    inference_checks = 0
    for name, observed in actual["inference"].items():
        reference = expected["inference"][name]
        assert observed["fit_embedding_vs_encoder"] == reference["fit_embedding_vs_encoder"], name
        assert observed["training_callback_ran"] == reference["training_callback_ran"], name
        for field in ("transform", "inverse"):
            for case, operation in observed.get(field, {}).items():
                assert values_only(operation) == values_only(reference[field][case]), (name, field, case)
                inference_checks += 1
        for case, observed_save in observed["persistence"].items():
            reference_save = reference["persistence"][case]
            if reference_save["save"]["status"] == "error":
                assert observed_save["save"]["status"] == "ok", (name, case)
                assert set(observed_save["files"]) == set(reference_save["files"]) | {"model.pkl"}, (name, case)
                loaded = observed_save["load"]
                assert loaded["status"] == "ok", (name, case)
                for prediction in loaded["result"].values():
                    assert prediction["status"] == "ok"
                    result = prediction["result"]
                    assert result["exact_bits"] == prod(result["shape"])
                for artifact in ("keras_encoder", "keras_decoder"):
                    if artifact in reference_save:
                        assert values_only(observed_save[artifact]) == values_only(reference_save[artifact])
                continue
            assert observed_save["files"] == reference_save["files"], (name, case)
            for artifact in observed_save.keys() - {"files"}:
                assert values_only(observed_save[artifact]) == values_only(reference_save[artifact]), (name, case)
    print(f"Frozen inference and persistence verified across {inference_checks} prediction cases.", flush=True)

    for name, observed in actual["losses"].items():
        if "batch" not in observed:
            continue
        reference = expected["losses"][name]
        assert observed["batch"] == reference["batch"], name
        residual = abs(observed["loss"] - reference["loss"])
        assert residual <= 3e-7, name
        if name != "negative_step":
            assert observed["coordinate_bits"] == reference["coordinate_bits"], name
        if name in ("positive_step", "reconstruction_step"):
            assert observed["encoder_weight_bits"] == reference["encoder_weight_bits"], name
        if "decoder_weight_bits" in observed:
            assert observed["decoder_weight_bits"] == reference["decoder_weight_bits"], name
        print(f"{name}  loss residual {residual:.9g}", flush=True)
        report_bits("  fitted coordinates", observed["coordinate_bits"], reference["coordinate_bits"])
        for field in ("encoder_weight_bits", "decoder_weight_bits"):
            if field in observed:
                report_bits(f"  {field.removesuffix('_bits').replace('_', ' ')}",
                            sum(observed[field], []), sum(reference[field], []))

    observed = actual["five_positive_batches"]
    reference = expected["five_positive_batches"]
    assert observed["batches"] == reference["batches"]
    residuals = [abs(a - b) for a, b in zip(observed["losses"], reference["losses"], strict=True)]
    print(f"five positive batches  max loss residual {max(residuals):.9g}", flush=True)
    report_bits("  fitted coordinates", observed["coordinate_bits"], reference["coordinate_bits"])
    report_bits("  encoder weights", sum(observed["encoder_weight_bits"], []),
                sum(reference["encoder_weight_bits"], []))


def check_default_networks():
    import warnings

    import numpy as np
    from tensorflow import keras
    from umap import ParametricUMAP as PublicEstimator
    from umap_alternative import ParametricUMAP

    class Stop(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            self.model.stop_training = True

    def architecture(model):
        if model is None:
            return None
        layers = []
        for layer in model.layers:
            config = layer.get_config()
            layers.append((type(layer).__name__, config.get("units"), config.get("activation"),
                           config.get("target_shape")))
        return model.input_shape, model.output_shape, layers

    features = np.random.RandomState(11).uniform(0, 1, size=(32, 3)).astype(np.float32)
    for reconstruct, dims in ((False, None), (True, None), (True, (1, 3))):
        networks = []
        for estimator in (PublicEstimator, ParametricUMAP):
            keras.utils.set_random_seed(42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fitted = estimator(
                    n_neighbors=4, n_epochs=10, n_jobs=1, random_state=42, dims=dims,
                    parametric_reconstruction=reconstruct, compatibility_layout=True,
                    keras_fit_kwargs={"verbose": 0, "callbacks": [Stop()]},
                ).fit(features)
            networks.append(tuple(architecture(getattr(fitted, f"{name}_", getattr(fitted, name)))
                                  for name in ("encoder", "decoder")))
        assert networks[0] == networks[1], (reconstruct, dims)
    print("Default encoder and decoder layer configurations matched.", flush=True)


def report_full_training():
    expected = json.loads(INFERENCE["FIXTURE"].read_text(encoding="utf-8"))
    actual = INFERENCE["capture"](use_wheel=True)
    for name, outputs in actual["cases"].items():
        for field, observed in outputs.items():
            if field == "reload" and field not in expected["cases"][name]:
                assert values_only(observed) == values_only(outputs["held_out"]), name
                continue
            reference = expected["cases"][name][field]
            if field == "save" and reference["status"] == "error":
                assert observed["status"] == "ok", name
                continue
            assert observed["status"] == reference["status"], (name, field)
            if observed["status"] == "error":
                assert observed["type"] == reference["type"], (name, field)
            else:
                report_bits(f"{name} {field}", observed["result"]["bits"], reference["result"]["bits"])


if __name__ == "__main__":
    try:
        check_stages()
        check_default_networks()
        report_full_training()
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None
