"""TensorFlow-backed Keras training for ParametricUMAP."""

from math import prod
from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model
except ImportError as error:
    raise ImportError("ParametricUMAP training requires TensorFlow and Keras") from error


@keras.utils.register_keras_serializable(package="umap_alternative")
class _EdgeModel(Model):
    def __init__(
        self, encoder, decoder, a, b, negative_sample_rate, autoencoder_loss,
        reconstruction_loss, reconstruction_weight, **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.a = a
        self.b = b
        self.negative_sample_rate = negative_sample_rate
        self.autoencoder_loss = autoencoder_loss
        self.reconstruction_loss = keras.losses.get(reconstruction_loss)
        self.reconstruction_weight = reconstruction_weight
        self.shuffle_seed = keras.random.SeedGenerator()

    def build(self, input_shape):
        if not self.encoder.built:
            self.encoder.build(input_shape[0])
        if self.decoder is not None and not self.decoder.built:
            self.decoder.build(self.encoder.compute_output_shape(input_shape[0]))
        super().build(input_shape)

    def call(self, inputs, training=False):
        head = self.encoder(inputs[0], training=training)
        tail = self.encoder(inputs[1], training=training)
        if self.decoder is None:
            return head, tail
        code = head if self.autoencoder_loss else tf.stop_gradient(head)
        return head, tail, self.decoder(code, training=training)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        head, tail = y_pred[:2]
        repeated_head = tf.repeat(head, self.negative_sample_rate, axis=0)
        repeated_tail = tf.repeat(tail, self.negative_sample_rate, axis=0)
        order = keras.random.shuffle(tf.range(tf.shape(repeated_tail)[0]), seed=self.shuffle_seed)
        positives = self._logits(head, tail)
        negatives = self._logits(repeated_head, tf.gather(repeated_tail, order))
        logits = tf.concat((positives, negatives), axis=0)
        labels = tf.concat((tf.ones_like(positives), tf.zeros_like(negatives)), axis=0)
        layout = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        if self.decoder is None:
            return layout
        reconstruction = tf.reduce_mean(self.reconstruction_loss(y, y_pred[2]))
        return layout + self.reconstruction_weight * reconstruction

    def _logits(self, head, tail):
        squared = tf.reduce_sum(tf.square(head - tail), axis=-1)
        nonzero = squared > 0
        # Use a finite derivative for coincident embeddings, including self-paired negatives.
        distance = tf.sqrt(tf.where(nonzero, squared, tf.ones_like(squared)))
        power = tf.where(nonzero, distance ** (2 * self.b), tf.zeros_like(distance))
        return -tf.math.log1p(self.a * power)

    def get_config(self):
        return {
            **super().get_config(),
            "encoder": keras.utils.serialize_keras_object(self.encoder),
            "decoder": keras.utils.serialize_keras_object(self.decoder),
            "a": self.a,
            "b": self.b,
            "negative_sample_rate": self.negative_sample_rate,
            "autoencoder_loss": self.autoencoder_loss,
            "reconstruction_loss": keras.losses.serialize(self.reconstruction_loss),
            "reconstruction_weight": self.reconstruction_weight,
        }

    @classmethod
    def from_config(cls, config):
        for name in ("encoder", "decoder", "reconstruction_loss"):
            config[name] = keras.utils.deserialize_keras_object(config[name])
        return cls(**config)


def _hidden_layers():
    return [keras.layers.Dense(100, activation="relu") for _ in range(3)]


def _default_encoder(dims, n_components):
    return keras.Sequential(
        [keras.Input(shape=dims), keras.layers.Flatten(), *_hidden_layers(),
         keras.layers.Dense(n_components, name="z")],
        name="encoder",
    )


def _default_decoder(dims, n_components):
    return keras.Sequential(
        [keras.Input(shape=(n_components,)), *_hidden_layers(),
         keras.layers.Dense(prod(dims), name="recon"), keras.layers.Reshape(dims)],
        name="decoder",
    )


def train(estimator, features, head, tail, counts, total, a, b):
    if isinstance(estimator.random_state, Integral):
        keras.utils.set_random_seed(int(estimator.random_state))

    encoder = estimator.encoder
    if encoder is None:
        encoder = _default_encoder(estimator.dims_, estimator.n_components)
    decoder = estimator.decoder
    if decoder is None and estimator.parametric_reconstruction:
        decoder = _default_decoder(estimator.dims_, estimator.n_components)
    reconstruction = estimator.parametric_reconstruction_loss_fcn
    if reconstruction is None:
        reconstruction = keras.losses.BinaryCrossentropy(from_logits=True)

    model = _EdgeModel(
        encoder, decoder if estimator.parametric_reconstruction else None, a, b,
        int(estimator.negative_sample_rate), estimator.autoencoder_loss,
        reconstruction, estimator.parametric_reconstruction_loss_weight,
        name="parametric_model",
    )
    feature_shape = (None, *estimator.dims_)
    model.build((feature_shape, feature_shape))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=4.0), jit_compile=False)

    order = check_random_state(estimator.random_state).permutation(total)
    edges = tf.data.Dataset.from_tensor_slices(
        (np.repeat(head, counts)[order], np.repeat(tail, counts)[order])
    )
    edges = edges.repeat().shuffle(10_000)
    batch_size = estimator.batch_size_
    if batch_size < 0:
        raise ValueError(f"Dimension {batch_size} must be >= 0")
    options = {"verbose": 0, "epochs": 10, **(estimator.keras_fit_kwargs or {})}
    edges = edges.batch(batch_size, drop_remainder=True)
    options.setdefault("steps_per_epoch", total // batch_size // 10 or None)
    if options["steps_per_epoch"] is None:
        # Keras receives unknown steps with a full batch even when the pool is smaller than it.
        edges = edges.enumerate().take_while(lambda i, _: i < 1).map(lambda _, pair: pair)

    inputs = tf.convert_to_tensor(features)

    def gather(head_indices, tail_indices):
        head_features = tf.gather(inputs, head_indices)
        tail_features = tf.gather(inputs, tail_indices)
        return (head_features, tail_features), head_features

    dataset = edges.map(gather).prefetch(1)
    history = model.fit(dataset, **options)
    return encoder, decoder, model, history.history
