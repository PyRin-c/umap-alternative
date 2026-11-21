"""Scikit-learn estimator for an optional Keras parametric layout."""

from __future__ import annotations

import pickle
from math import prod
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_array, check_is_fitted

from . import _native
from ._curve import _resolve_curve_params
from ._umap import UMAP, _DEFAULT_PRECOMPUTED_KNN, _default_epochs, _validate_epochs


def _prepare_edges(graph, n_epochs):
    return _native._prepare_parametric_edges(
        (graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()), n_epochs
    )


def _predict(model, values, batch_size=None):
    if len(values) == 0:
        raise UnboundLocalError("cannot predict on an empty batch")
    return model.predict(values, batch_size=batch_size, verbose=0)


class ParametricUMAP(UMAP):
    """Learn a UMAP embedding with a TensorFlow-backed Keras encoder.

    The encoder maps feature rows to ``n_components`` coordinates. Enabling
    ``parametric_reconstruction`` trains a decoder and uses it for inverse
    transforms. If omitted, the networks are built during multi-row fitting.
    ``n_epochs`` controls positive-edge multiplicity and ``keras_fit_kwargs``
    supplies additional arguments to Keras training.
    """

    def __init__(
        self,
        batch_size=None,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_fcn=None,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        keras_fit_kwargs=None,
        *,
        n_components=2,
        n_neighbors=15,
        metric="euclidean",
        init="spectral",
        random_state=None,
        n_jobs=-1,
        n_epochs=None,
        compatibility_layout=None,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        target_metric="categorical",
        target_weight=0.5,
        learning_rate=1.0,
        min_dist=0.1,
        spread=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        transform_seed=42,
        precomputed_knn=_DEFAULT_PRECOMPUTED_KNN,
        disconnection_distance=None,
        metric_kwds=None,
        force_approximation_algorithm=False,
        n_training_epochs=1,
        loss_report_frequency=10,
        optimizer=None,
    ):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            init=init,
            random_state=random_state,
            n_jobs=n_jobs,
            n_epochs=n_epochs,
            compatibility_layout=compatibility_layout,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            target_metric=target_metric,
            target_weight=target_weight,
            learning_rate=learning_rate,
            min_dist=min_dist,
            spread=spread,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            transform_seed=transform_seed,
            precomputed_knn=precomputed_knn,
            disconnection_distance=disconnection_distance,
            metric_kwds=metric_kwds,
            force_approximation_algorithm=force_approximation_algorithm,
        )
        self.batch_size = batch_size
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.parametric_reconstruction_loss_weight = parametric_reconstruction_loss_weight
        self.autoencoder_loss = autoencoder_loss
        self.keras_fit_kwargs = keras_fit_kwargs
        self.n_training_epochs = n_training_epochs
        self.loss_report_frequency = loss_report_frequency
        self.optimizer = optimizer

    def fit(self, X, y=None):
        """Build the native fuzzy graph, train on directed edges, and embed X."""

        features = check_array(X, dtype=np.float32, order="C")
        if self.metric == "precomputed":
            raise NotImplementedError("parametric fitting requires input features")
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, Integral)
            or self.n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer")
        if not isinstance(self.negative_sample_rate, Integral) or self.negative_sample_rate < 0:
            raise ValueError("negative_sample_rate must be a nonnegative integer")
        for name in ("n_training_epochs", "loss_report_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        n_rows, n_features = features.shape
        dims = (n_features,) if self.dims is None else tuple(self.dims)
        if any(not isinstance(dim, Integral) or dim <= 0 for dim in dims) or prod(dims) != n_features:
            raise ValueError("dims must describe the feature dimensions of one row")
        a, b = _resolve_curve_params(self.spread, self.min_dist, self.a, self.b)
        epochs = _default_epochs(n_rows) if self.n_epochs is None else _validate_epochs(self.n_epochs)

        self._fit(features, y, embedding_mode=False)
        self._embedding_transform_options = {"a": a, "b": b, "learning_rate": self.learning_rate}
        self.dims_ = dims
        self.batch_size_ = self.batch_size
        self.parametric_model_ = None
        self.__dict__.pop("history_", None)
        self.encoder_ = self.encoder
        self.decoder_ = self.decoder
        if n_rows == 1:
            self.embedding_ = np.zeros((1, self.n_components), dtype=np.float64)
            return self
        if self.batch_size_ is None:
            self.batch_size_ = n_rows

        graph = self.graph_
        if graph.nnz == 0:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        edges, total = _prepare_edges(graph, epochs)
        if not edges:
            raise IndexError("arrays used as indices must be of integer (or boolean) type")
        head, tail, counts = np.asarray(edges, dtype=np.int64).T
        weights = np.asarray(graph[head, tail]).ravel()
        self.graph_ = csr_matrix((weights, (head, tail)), shape=graph.shape)

        from . import _parametric_keras

        network_input = features.reshape(n_rows, *dims)
        self.encoder_, self.decoder_, self.parametric_model_, self.history_ = _parametric_keras.train(
            self, network_input, head, tail, counts, total, a, b
        )
        self.embedding_ = _predict(self.encoder_, network_input, self.batch_size_)
        return self

    def fit_transform(self, X, y=None):
        """Fit and return the encoder coordinates or the singleton zero layout."""

        return self.fit(X, y).embedding_

    def transform(self, X, batch_size=None):
        """Apply the fitted encoder with the selected prediction batch size."""

        check_is_fitted(self, "embedding_")
        features = np.asarray(X, dtype=np.float32)
        features = features.reshape(len(features), *self.dims_)
        if self.encoder_ is None:
            raise AttributeError("the singleton fit has no encoder")
        return _predict(self.encoder_, features, batch_size or self.batch_size_)

    def inverse_transform(self, X):
        """Apply the decoder or the dense UMAP inverse in the learned layout."""

        check_is_fitted(self, "embedding_")
        if self.parametric_reconstruction:
            if self.decoder_ is None:
                raise AttributeError("the singleton fit has no decoder")
            return _predict(self.decoder_, np.asarray(X, dtype=np.float32))
        if self._graph_transform_fit[0].shape[0] == 0:
            raise ValueError("Inverse transform requires the original training features or a decoder")
        return super().inverse_transform(X)

    def update(self, X, ensure_all_finite=True):
        raise NotImplementedError("incremental fitting is not supported for ParametricUMAP")

    def save(self, save_location, verbose=True, exclude_raw_data=False):
        """Save the estimator and its fitted Keras networks."""

        check_is_fitted(self, "embedding_")
        path = Path(save_location)
        path.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        networks = {
            "encoder": self.encoder_, "decoder": self.decoder_,
            "parametric_model": self.parametric_model_,
        }
        networks = {name: model for name, model in networks.items() if model is not None}
        state["_saved_networks"] = tuple(networks)
        for name in ("encoder", "decoder", "encoder_", "decoder_", "parametric_model_"):
            state.pop(name, None)
        if state["keras_fit_kwargs"] is not None:
            state["keras_fit_kwargs"] = state["keras_fit_kwargs"].copy()
            state["keras_fit_kwargs"].pop("callbacks", None)
        if state["optimizer"] is not None and not isinstance(state["optimizer"], (str, dict)):
            from ._parametric_keras import keras

            state["optimizer"] = keras.optimizers.serialize(state["optimizer"])
        if exclude_raw_data:
            state["_exact_fit_input"] = None
            state["_feature_search_index"] = None
            state["_inverse_distances"] = None
            state["_inverse_unique_fit"] = None
            state["precomputed_knn"] = _DEFAULT_PRECOMPUTED_KNN
            if state["_graph_transform_fit"] is not None:
                _, metric = state["_graph_transform_fit"]
                state["_graph_transform_fit"] = (np.empty((0, self.n_features_in_), np.float32), metric)
        with TemporaryDirectory(dir=path) as directory:
            staging = Path(directory)
            for name, model in networks.items():
                model.save(staging / f"{name}.keras")
            with (staging / "model.pkl").open("wb") as file:
                pickle.dump(state, file)
            for name in networks:
                artifact = f"{name}.keras"
                (staging / artifact).replace(path / artifact)
            (staging / "model.pkl").replace(path / "model.pkl")
        if verbose:
            print(f"Saved ParametricUMAP to {path}")


def load_ParametricUMAP(save_location, verbose=True):
    """Load a saved estimator from a trusted directory."""

    path = Path(save_location)
    with (path / "model.pkl").open("rb") as file:
        state = pickle.load(file)
    names = state.pop("_saved_networks", None)
    if names is None:
        names = [name for name in ("encoder", "decoder") if (path / f"{name}.keras").is_file()]
    estimator = ParametricUMAP()
    estimator.__dict__.update(state)
    networks = {}
    if names:
        from ._parametric_keras import keras

        networks = {name: keras.models.load_model(path / f"{name}.keras", compile=False) for name in names}
    estimator.parametric_model_ = networks.get("parametric_model")
    for name in ("encoder", "decoder"):
        model = getattr(estimator.parametric_model_, name, None)
        if model is None:
            model = networks.get(name)
        setattr(estimator, name, model)
        setattr(estimator, f"{name}_", model)
    if verbose:
        print(f"Loaded ParametricUMAP from {path}")
    return estimator
