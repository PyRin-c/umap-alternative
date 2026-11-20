"""Probe the public ParametricUMAP methods and the Keras training boundary."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch


FIXTURE = Path(__file__).with_name("fixtures") / "umap-learn-0.6.0-parametric-stages-windows-x86_64.json"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_parametric_fixture.py")))
observe = REFERENCE["observe"]
dense_model = REFERENCE["dense_model"]


def capture(*, use_wheel=False, include_training_bits=False):
    REFERENCE["configure_reference_environment"]()
    import numpy as np
    import tensorflow as tf
    if use_wheel:
        from umap_alternative import ParametricUMAP, UMAP, load_ParametricUMAP
    else:
        from umap import ParametricUMAP, UMAP
        from umap.parametric_umap import load_ParametricUMAP

    tf.config.experimental.enable_op_determinism()
    tf.get_logger().setLevel("ERROR")
    points = np.random.RandomState(11).uniform(-1, 1, size=(32, 3)).astype(np.float32)
    unit_points = (points + np.float32(1)) / np.float32(2)
    queries = np.ascontiguousarray(points + np.float32(0.03))
    graph_options = {"n_neighbors": 4, "random_state": 42, "n_jobs": 1}

    def compare(actual, expected):
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        exact_bits = None
        if actual.dtype == expected.dtype == np.float32:
            exact_bits = int(np.count_nonzero(actual.view(np.uint32) == expected.view(np.uint32)))
        return {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "exact_bits": exact_bits,
            "values": actual.size,
            "max_abs": float(np.max(np.abs(actual - expected), initial=0)),
        }

    class RecordInputs(tf.keras.layers.Layer):
        def __init__(self, record):
            super().__init__()
            self.record = record

        def call(self, inputs):
            def append(values):
                self.record(values.numpy())
                return np.int32(0)

            token = tf.py_function(append, [inputs], Tout=tf.int32)
            with tf.control_dependencies([token]):
                return tf.identity(inputs)

    class FirstBatch(tf.keras.callbacks.Callback):
        def __init__(self, encoder_calls, decoder_calls, zero_rate, stop_after):
            super().__init__()
            self.encoder_calls = encoder_calls
            self.decoder_calls = decoder_calls
            self.zero_rate = zero_rate
            self.training_params = None
            self.loss = None
            self.losses = []
            self.stop_after = stop_after

        def on_train_begin(self, logs=None):
            self.training_params = dict(self.params)
            self.first_encoder_call = len(self.encoder_calls)
            self.first_decoder_call = len(self.decoder_calls)
            self.learning_rate = float(self.model.optimizer.learning_rate.numpy())
            if self.zero_rate:
                self.model.optimizer.learning_rate.assign(0)

        def on_train_batch_end(self, batch, logs=None):
            self.loss = float(logs["loss"])
            self.losses.append(self.loss)
            if len(self.losses) < self.stop_after:
                return
            self.encoder_batch = self.encoder_calls[self.first_encoder_call :]
            self.decoder_batch = self.decoder_calls[self.first_decoder_call :]
            self.model.stop_training = True

    def make_estimator(
        *, n_epochs=10, batch_size=4, negative_sample_rate=0, reconstruct=False,
        learning=True, trace=True, freeze=False, sigmoid=False, stop_after=1, **options
    ):
        tf.keras.utils.set_random_seed(42)
        encoder_calls, decoder_calls = [], []
        encoder_trace = RecordInputs(encoder_calls.append) if trace else None
        decoder_trace = RecordInputs(decoder_calls.append) if trace and reconstruct else None
        encoder = dense_model(tf, "encoder", 3, 2, 41, encoder_trace)
        decoder = (
            dense_model(tf, "decoder", 2, 3, 51, decoder_trace, "sigmoid" if sigmoid else None)
            if reconstruct else None
        )
        if freeze:
            encoder.trainable = False
            if decoder is not None:
                decoder.trainable = False
        first_batch = FirstBatch(encoder_calls, decoder_calls, not learning, stop_after)
        estimator = ParametricUMAP(
            **graph_options,
            n_components=2,
            n_epochs=n_epochs,
            init="random",
            compatibility_layout=True,
            batch_size=batch_size,
            negative_sample_rate=negative_sample_rate,
            a=1.5,
            b=0.75,
            encoder=encoder,
            decoder=decoder,
            parametric_reconstruction=reconstruct,
            keras_fit_kwargs={"verbose": 0, "shuffle": False, "callbacks": [first_batch]},
            **options,
        )
        return estimator, encoder, decoder, first_batch

    def training_pairs(encoder_batch, inputs):
        assert len(encoder_batch) == 2, len(encoder_batch)
        rows = {tuple(row): index for index, row in enumerate(inputs)}
        heads, tails = ([rows[tuple(row)] for row in batch] for batch in encoder_batch)
        return list(zip(heads, tails, strict=True))

    def logits(head, tail):
        distance = np.linalg.norm(head - tail, axis=1)
        return -np.log1p(np.float32(1.5) * distance ** np.float32(1.5))

    def weight_bits(model):
        return [value.ravel().view(np.uint32).tolist() for value in model.get_weights()]

    def layout_and_reconstruction():
        report, embeddings = {}, {}
        specifications = {
            "positive_no_op": {"learning": False},
            "positive_step": {},
            "negative_no_op": {"learning": False, "negative_sample_rate": 2},
            "negative_step": {"negative_sample_rate": 2},
            "negative_repulsion_2.5": {
                "learning": False, "negative_sample_rate": 2, "repulsion_strength": 2.5,
            },
            "reconstruction_no_op": {"learning": False, "reconstruct": True},
            "reconstruction_quarter_weight": {
                "learning": False, "reconstruct": True, "parametric_reconstruction_loss_weight": 0.25,
            },
            "reconstruction_step": {"reconstruct": True},
            "autoencoder_step": {"reconstruct": True, "autoencoder_loss": True},
        }
        for name, settings in specifications.items():
            estimator, encoder, decoder, callback = make_estimator(sigmoid=True, **settings)
            baseline = encoder.predict(unit_points, batch_size=4, verbose=0)
            decoder_baseline = (
                decoder.predict(baseline[:3], batch_size=3, verbose=0) if decoder is not None else None
            )
            fit = observe(lambda: estimator.fit_transform(unit_points), lambda values: compare(values, baseline))
            assert fit["status"] == "ok", (name, fit)
            embeddings[name] = estimator.embedding_
            pairs = training_pairs(callback.encoder_batch, unit_points)
            heads, tails = np.asarray(pairs).T
            head, tail = baseline[heads], baseline[tails]
            positives = np.logaddexp(0, -logits(head, tail))
            record = {
                "fit_transform_vs_before": fit,
                "learning_rate": 0.0 if callback.zero_rate else callback.learning_rate,
                "batch": pairs,
                "loss": callback.loss,
                "positive_layout_candidate": float(positives.mean()),
            }
            if include_training_bits:
                record["coordinate_bits"] = estimator.embedding_.ravel().view(np.uint32).tolist()
                record["encoder_weight_bits"] = weight_bits(encoder)
                if decoder is not None:
                    record["decoder_weight_bits"] = weight_bits(decoder)
            rate = settings.get("negative_sample_rate", 0)
            if rate:
                repeated_head = np.repeat(head, rate, axis=0)
                candidates = []
                for order in set(permutations(index for index in range(len(tail)) for _ in range(rate))):
                    negative_terms = np.logaddexp(0, logits(repeated_head, tail[list(order)]))
                    candidate = float(np.concatenate((positives, negative_terms)).mean())
                    candidates.append((abs(candidate - callback.loss), order, candidate))
                residual, order, candidate = min(candidates)
                record["negative_layout_candidate"] = {
                    "loss": candidate,
                    "residual": residual,
                    "tail_indices": order,
                    "permutations_checked": len(candidates),
                }
                assert residual <= 3e-7, (name, residual)
            else:
                candidate = float(positives.mean())
                if decoder is not None:
                    assert len(callback.decoder_batch) == 1, name
                    # The BCE comparison uses the decoder values before any active update.
                    decoder_before = dense_model(tf, "decoder_before", 2, 3, 51, activation="sigmoid")
                    prediction = decoder_before.predict(head, batch_size=len(head), verbose=0)
                    inputs = unit_points[heads]
                    bce = float(np.mean(-inputs * np.log(prediction) - (1 - inputs) * np.log1p(-prediction)))
                    candidate += settings.get("parametric_reconstruction_loss_weight", 1.0) * bce
                    record["decoder_vs_head"] = compare(callback.decoder_batch[0], head)
                    record["decoder_vs_tail"] = compare(callback.decoder_batch[0], tail)
                    record["binary_cross_entropy_candidate"] = bce
                    record["mean_squared_error_candidate"] = float(np.mean((inputs - prediction) ** 2))
                    decoded = observe(
                        lambda: estimator.inverse_transform(baseline[:3]),
                        lambda values: compare(values, decoder_baseline),
                    )
                    record["inverse_vs_before"] = decoded
                record["layout_plus_reconstruction_candidate"] = candidate
                assert abs(candidate - callback.loss) <= 3e-7, (name, candidate, callback.loss)
            report[name] = record
        for left, right in (
            ("positive_no_op", "positive_step"),
            ("negative_no_op", "negative_step"),
            ("negative_no_op", "negative_repulsion_2.5"),
            ("positive_step", "reconstruction_step"),
            ("reconstruction_step", "autoencoder_step"),
        ):
            report[f"{left}_vs_{right}"] = {
                "same_batch": report[left]["batch"] == report[right]["batch"],
                "loss_difference": abs(report[left]["loss"] - report[right]["loss"]),
                "embedding": compare(embeddings[left], embeddings[right]),
            }
        return report

    def edge_period(n_rows, n_epochs, total_expected, **options):
        inputs = points[:n_rows]
        estimator, _, _, callback = make_estimator(
            n_epochs=n_epochs, batch_size=2 * total_expected, learning=False, **options,
        )
        original = UMAP(**graph_options, transform_mode="graph", **options)
        fitted = observe(lambda: original.fit(inputs), lambda value: value is original)
        assert fitted["status"] == "ok", fitted
        csr = original.graph_.tocsr()
        csr.sort_indices()

        shuffle_calls = []
        shuffle = tf.data.Dataset.shuffle

        def bypass_edge_shuffle(dataset, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
            spec = dataset.element_spec
            if isinstance(spec, tuple) and len(spec) == 2 and all(
                isinstance(value, tf.TensorSpec) and value.shape.rank == 0 and value.dtype.is_integer
                for value in spec
            ):
                shuffle_calls.append({"cardinality": int(dataset.cardinality().numpy()), "buffer": buffer_size})
                return dataset
            return shuffle(dataset, buffer_size, seed, reshuffle_each_iteration, name)

        # One intervention at TensorFlow's public dataset API exposes two consecutive periods.
        with patch.object(tf.data.Dataset, "shuffle", bypass_edge_shuffle):
            fit = observe(lambda: estimator.fit(inputs), lambda value: value is estimator)
        assert fit["status"] == "ok", (n_rows, n_epochs, fit)
        assert len(shuffle_calls) == 1, (n_rows, n_epochs, shuffle_calls)
        pairs = training_pairs(callback.encoder_batch, inputs)
        assert len(pairs) == 2 * total_expected, (n_rows, n_epochs, len(pairs))
        cycle = pairs[:total_expected]
        assert cycle == pairs[total_expected:], (n_rows, n_epochs, total_expected)
        counts = Counter(cycle)
        graph = csr.tocoo()
        observed = [counts[int(head), int(tail)] for head, tail in zip(graph.row, graph.col, strict=True)]
        epochs = 500 if n_epochs is None else n_epochs
        expected = (graph.data * np.float32(epochs)).astype(int).tolist()
        assert observed == expected, (n_rows, n_epochs, observed, expected)
        assert sum(observed) == total_expected, (n_rows, n_epochs, sum(observed))
        double_counts = (graph.data.astype(np.float64) * epochs).astype(int)
        return {
            "n_rows": n_rows,
            "input_bits": inputs.ravel().view(np.uint32).tolist(),
            "graph_options": options,
            "n_epochs": n_epochs,
            "resolved_epochs": epochs,
            "total_repeats": total_expected,
            "indptr": csr.indptr.tolist(),
            "indices": csr.indices.tolist(),
            "weight_bits": csr.data.view(np.uint32).tolist(),
            "observed_counts": observed,
            "float64_count_differences": int(np.count_nonzero(double_counts != observed)),
            "parametric_nonzero_weights": int(np.count_nonzero(estimator.graph_.data)),
            "shuffle": shuffle_calls[0],
        }

    def training_boundaries():
        report = {}
        for name, rows, batch, epochs in (
            ("empty", 0, 4, 10),
            ("singleton", 1, 4, 10),
            ("two_rows", 2, 4, 10),
            ("zero_epochs", 8, 4, 0),
            ("batch_negative", 8, -1, 2),
            ("batch_zero", 8, 0, 2),
            ("batch_one", 32, 1, 10),
            ("batch_default", 32, None, 10),
            ("epoch_default", 8, None, None),
            ("steps_boundary_below", 32, 84, 10),
            ("steps_boundary_above", 32, 85, 10),
            ("period_size", 32, 848, 10),
            ("above_period_size", 32, 849, 10),
        ):
            estimator, _, _, callback = make_estimator(n_epochs=epochs, batch_size=batch, learning=False)
            inputs = points[:rows]
            fit = observe(
                lambda: estimator.fit_transform(inputs),
                lambda value: {"shape": list(value.shape), "all_zero": not bool(np.any(value))},
            )
            entry = {"rows": rows, "n_epochs": epochs, "batch_size": batch, "fit_transform": fit}
            if callback.loss is not None:
                pairs = training_pairs(callback.encoder_batch, inputs)
                graph = estimator.graph_.tocsr()
                assert all(graph[head, tail] > 0 for head, tail in pairs), name
                entry["observed_batch_size"] = len(pairs)
                entry["keras"] = callback.training_params
            report[name] = entry
        return report

    def save_load(estimator, encoder, decoder, latents):
        report = {}
        for exclude in (False, True):
            with tempfile.TemporaryDirectory(prefix="umap-parametric-stages-") as folder:
                path = Path(folder)
                transform = estimator.transform(queries[:3], batch_size=3)
                save = observe(lambda: estimator.save(folder, verbose=False, exclude_raw_data=exclude))
                entry = {"save": save, "files": sorted(file.name for file in path.iterdir())}
                loaded = observe(lambda: load_ParametricUMAP(folder, verbose=False))
                if loaded["status"] == "ok":
                    model = loaded["result"]
                    loaded["result"] = {
                        "transform": observe(
                            lambda: model.transform(queries[:3], batch_size=3),
                            lambda values: compare(values, transform),
                        ),
                    }
                    if decoder is not None:
                        inverse = estimator.inverse_transform(latents)
                        loaded["result"]["inverse"] = observe(
                            lambda: model.inverse_transform(latents),
                            lambda values: compare(values, inverse),
                        )
                elif "message" in loaded:
                    loaded["message"] = loaded["message"].replace(repr(folder)[1:-1], "<save-dir>")
                entry["load"] = loaded
                if (path / "encoder.keras").is_file():
                    direct = encoder.predict(queries[:3], batch_size=3, verbose=0)
                    entry["keras_encoder"] = observe(
                        lambda: tf.keras.models.load_model(path / "encoder.keras")
                        .predict(queries[:3], batch_size=3, verbose=0),
                        lambda values: compare(values, direct),
                    )
                if decoder is not None and (path / "decoder.keras").is_file():
                    direct = decoder.predict(latents, verbose=0)
                    entry["keras_decoder"] = observe(
                        lambda: tf.keras.models.load_model(path / "decoder.keras").predict(latents, verbose=0),
                        lambda values: compare(values, direct),
                    )
                report[f"exclude_raw_data={exclude}"] = entry
        return report

    def inference_and_persistence():
        report = {}
        for reconstruct in (False, True):
            for n_rows in (1, 32):
                estimator, encoder, decoder, callback = make_estimator(
                    batch_size=16, negative_sample_rate=5, reconstruct=reconstruct,
                    trace=False, freeze=True,
                )
                inputs = points[:n_rows]
                baseline = encoder.predict(inputs, batch_size=16, verbose=0)
                fit = observe(lambda: estimator.fit(inputs), lambda value: value is estimator)
                assert fit["status"] == "ok", fit
                entry = {
                    "fit": fit,
                    "fit_embedding_vs_encoder": compare(estimator.embedding_, baseline),
                    "training_callback_ran": callback.loss is not None,
                }
                if n_rows == 32:
                    transform_results = {}
                    for rows, batch in ((0, 16), (1, 1), (15, 16), (16, 16), (17, 16), (17, 1), (17, 0), (17, None), (3, -1)):
                        query = queries[:rows]
                        expected = None
                        if rows and batch != -1:
                            expected = encoder.predict(query, batch_size=batch or 16, verbose=0)
                        result = observe(lambda: estimator.transform(query, batch_size=batch))
                        if result["status"] == "ok":
                            transformed = result["result"]
                            result["result"] = compare(transformed, expected)
                            if not batch:
                                keras_default = encoder.predict(query, verbose=0)
                                result["vs_keras_default_batch"] = compare(transformed, keras_default)
                        transform_results[f"rows={rows},batch={batch}"] = result
                    entry["transform"] = transform_results
                inverse_results = {}
                for rows in (0, 1, 3):
                    latents = baseline[:rows]
                    if n_rows == 1 and rows:
                        latents = encoder.predict(queries[:rows], verbose=0)
                    expected = decoder.predict(latents, verbose=0) if decoder is not None and rows else None
                    inverse_results[str(rows)] = observe(
                        lambda: estimator.inverse_transform(latents),
                        lambda values: compare(values, expected) if expected is not None else {"shape": list(values.shape)},
                    )
                entry["inverse"] = inverse_results
                entry["persistence"] = save_load(
                    estimator, encoder, decoder, encoder.predict(queries[:3], verbose=0),
                )
                report[f"reconstruction={reconstruct},rows={n_rows}"] = entry
        return report

    report = {
        "edge_periods": {
            "unit_only": edge_period(8, 1, 10),
            "fractional": edge_period(8, 2, 32),
            "weighted": edge_period(8, 5, 98),
            "wide": edge_period(32, 10, 848),
            "below_unit_maximum": edge_period(8, 3, 22, local_connectivity=0, set_op_mix_ratio=0.25),
            "rounding_boundary": edge_period(8, 4176, 51502, local_connectivity=0, set_op_mix_ratio=0.25),
            "default_epochs": edge_period(8, None, 10876),
        },
        "training_boundaries": training_boundaries(),
        "losses": layout_and_reconstruction(),
        "inference": inference_and_persistence(),
    }
    if include_training_bits:
        estimator, encoder, _, callback = make_estimator(stop_after=5)
        fit = observe(lambda: estimator.fit(unit_points), lambda model: model is estimator,
                      include_exception_message=False)
        assert fit["status"] == "ok", fit.get("type")
        report["five_positive_batches"] = {
            "losses": callback.losses,
            "batches": [
                training_pairs(callback.encoder_batch[i : i + 2], unit_points)
                for i in range(0, len(callback.encoder_batch), 2)
            ],
            "coordinate_bits": estimator.embedding_.ravel().view(np.uint32).tolist(),
            "encoder_weight_bits": weight_bits(encoder),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = json.dumps(capture(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            raise SystemExit(f"public fixture differs from {args.output}")
    else:
        args.output.write_bytes(output.encode("utf-8"))


if __name__ == "__main__":
    sys.tracebacklimit = 0
    main()
