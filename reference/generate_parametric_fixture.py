"""Record public ParametricUMAP outputs on the pinned Windows CPU stack."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path


FIXTURE = Path(__file__).with_name("fixtures") / "umap-learn-0.6.0-parametric-windows-x86_64.json"
REQUIREMENTS = Path(__file__).with_name("parametric-windows-x86_64.txt")


def configure_reference_environment():
    if (
        platform.system() != "Windows"
        or platform.machine().lower() not in ("amd64", "x86_64")
        or sys.version_info[:3] != (3, 13, 12)
    ):
        raise RuntimeError("this fixture requires Windows x86-64 and Python 3.13.12")
    for requirement in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        name, expected = requirement.split("==")
        if version(name) != expected:
            raise RuntimeError(f"{name} must be {expected} to generate this fixture")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_CPP_MIN_LOG_LEVEL="3",
        TF_ENABLE_ONEDNN_OPTS="0",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )


def dense_model(tf, name, n_features, n_outputs, seed, input_layer=None, activation=None):
    inputs = tf.keras.Input(shape=(n_features,))
    features = inputs if input_layer is None else input_layer(inputs)
    hidden = tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    )(features)
    values = tf.keras.layers.Dense(
        n_outputs,
        activation=activation,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
    )(hidden)
    return tf.keras.Model(inputs, values, name=name)


def capture(*, use_wheel=False) -> dict:
    configure_reference_environment()

    import numpy as np
    import tensorflow as tf
    if use_wheel:
        from umap_alternative import ParametricUMAP, load_ParametricUMAP
    else:
        from umap import ParametricUMAP
        from umap.parametric_umap import load_ParametricUMAP

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    tf.get_logger().setLevel("ERROR")
    points = np.random.RandomState(11).uniform(-1, 1, size=(32, 3)).astype(np.float32)
    queries = np.ascontiguousarray(points[:3] + np.float32(0.03))

    def array_record(values):
        values = np.ascontiguousarray(values)
        if values.dtype != np.float32:
            raise AssertionError(f"expected float32, got {values.dtype}")
        return {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "bits": values.ravel().view(np.uint32).tolist(),
        }

    packages = ("umap-learn", "tensorflow", "keras", "numpy", "scipy", "scikit-learn")
    report = {"versions": {name: version(name) for name in packages}, "cases": {}}
    for reconstruct in (False, True):
        tf.keras.utils.set_random_seed(42)
        model = ParametricUMAP(
            n_neighbors=4,
            n_components=2,
            n_epochs=10,
            random_state=42,
            init="random",
            compatibility_layout=True,
            batch_size=16,
            n_jobs=1,
            encoder=dense_model(tf, "encoder", 3, 2, 41),
            decoder=dense_model(tf, "decoder", 2, 3, 51) if reconstruct else None,
            parametric_reconstruction=reconstruct,
            keras_fit_kwargs={"verbose": 0, "shuffle": False},
        )
        case = {
            "fit_transform": observe(lambda: model.fit_transform(points), array_record),
            "held_out": observe(lambda: model.transform(queries, batch_size=3), array_record),
            "one_row": observe(lambda: model.transform(queries[:1], batch_size=1), array_record),
            "zero_rows": observe(lambda: model.transform(queries[:0], batch_size=1), array_record),
            "inverse": observe(lambda: model.inverse_transform(model.embedding_[:3]), array_record),
        }
        with tempfile.TemporaryDirectory(prefix="umap-parametric-save-") as saved:
            case["save"] = observe(lambda: model.save(saved, verbose=False, exclude_raw_data=True))
            if case["save"]["status"] == "ok":
                case["reload"] = observe(
                    lambda: load_ParametricUMAP(saved, verbose=False).transform(queries, batch_size=3),
                    array_record,
                )
        report["cases"][f"reconstruction={reconstruct}"] = case
    return report


def observe(operation, summarize=lambda value: value, *, include_exception_message=True):
    with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        try:
            value = operation()
        except Exception as exc:
            result = {"status": "error", "type": type(exc).__name__}
            if include_exception_message:
                message = str(exc)
                if not any(token in message.lower() for token in ("jit", "xla", "graph execution error")):
                    result["message"] = message.splitlines()[0] if message else ""
        else:
            result = {"status": "ok", "result": summarize(value)}
    result["warnings"] = sorted({item.category.__name__ for item in captured})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    content = json.dumps(capture(), indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if arguments.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"public fixture differs from {arguments.output}")
    else:
        arguments.output.write_bytes(content.encode("utf-8"))


if __name__ == "__main__":
    sys.tracebacklimit = 0
    main()
