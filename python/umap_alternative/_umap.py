"""Public estimator for dense graph and serial seeded embedding fits."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from . import _native
from ._curve import _finite_float, _resolve_curve_params


_METRIC_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "cityblock": "manhattan",
    "taxicab": "manhattan",
    "chebyshev": "chebyshev",
    "linfty": "chebyshev",
    "linfinity": "chebyshev",
    "max": "chebyshev",
}

_TRUNCATION_WARNING = (
    "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
)
_PRECOMPUTED_WARNING = "using precomputed metric; inverse_transform will be unavailable"
_DISCONNECTION_WARNING = (
    "A large number of your vertices were disconnected from the manifold.\n"
    "Disconnection_distance = inf has removed 0 edges.\n"
    "It has fully disconnected {count} vertices.\n"
    "You might consider using find_disconnected_points() to find and remove "
    "these points from your data.\n"
    "Use umap.utils.disconnected_vertices() to identify them."
)
_COMPATIBILITY_WARNING = (
    "compatibility_layout was not set, so UMAP is using the legacy (0.5.x) "
    "nearest-neighbor, initialization and layout pipeline. In a future release "
    "the default will change to compatibility_layout=False, which uses recursive "
    "initialization and the Adam optimizer and produces different embeddings. "
    "Set compatibility_layout=True to keep the current behavior and silence this "
    "warning, or compatibility_layout=False (or optimizer='adam') to adopt the new behavior now."
)


class UMAP(BaseEstimator, TransformerMixin):
    """Fit a fuzzy graph or a serial seeded dense UMAP embedding.

    Dense ``transform_mode="graph"`` fitting is implemented and returns the
    fuzzy simplicial-set graph as a SciPy CSR matrix.
    ``transform`` on held-out dense features returns a float32 SciPy COO
    matrix with query rows and fitted training columns. Transforming the
    fitted data returns ``graph_`` itself.

    For categorical supervision with dense features, supply one real numeric
    label per row in ``y``. A label of ``-1`` indicates an unlabeled row.
    ``target_weight`` controls the reduction of edges between known classes
    and must lie in the interval [0, 1].

    With ``metric="precomputed"``, provide a square distance matrix. After
    conversion to float32, its entries must be finite and non-negative and its
    diagonal must be zero. The rows are ranked independently, so asymmetric
    distances are accepted. Each row's selected neighbors must include its
    diagonal entry, which can be excluded when there are ties at zero distance.

    For an embedding, use dense features, an integer ``random_state`` in
    ``[0, 2**32 - 1]``, and ``compatibility_layout=True``. ``init`` can be
    ``"random"`` or a finite NumPy array of shape ``(n_samples, n_components)``
    with distinct rows and nonconstant columns. Fitting exposes ``embedding_``
    and ``graph_``. A singleton has a zero embedding and no graph. Spectral
    initialization, supervised embeddings, modern optimizers, out-of-sample
    embedding transforms, and inverse transforms raise explicit errors.
    """

    def __init__(
        self,
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
        transform_mode="embedding",
        target_metric="categorical",
        target_weight=0.5,
        learning_rate=1.0,
        min_dist=0.1,
        spread=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        optimizer=None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_epochs = n_epochs
        self.compatibility_layout = compatibility_layout
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.transform_mode = transform_mode
        self.target_metric = target_metric
        self.target_weight = target_weight
        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.spread = spread
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.a = a
        self.b = b
        self.optimizer = optimizer

    def fit(self, X, y=None):
        """Fit the fuzzy graph and, in embedding mode, the training layout."""

        self._validate_mode()
        self._validate_metric()
        if self.set_op_mix_ratio != 1.0:
            raise NotImplementedError(
                "set_op_mix_ratio values other than 1.0 are not implemented"
            )
        if self.local_connectivity != 1.0:
            raise NotImplementedError(
                "local_connectivity values other than 1.0 are not implemented"
            )
        if not isinstance(self.n_neighbors, numbers.Integral):
            raise ValueError("n_neighbors must be an integer")
        n_neighbors = int(self.n_neighbors)
        if n_neighbors <= 1:
            raise ValueError("n_neighbors must be greater than 1")

        if issparse(X):
            raise TypeError("sparse input is not supported; pass a dense array")
        precomputed = self.metric == "precomputed"
        embedding_mode = self.transform_mode == "embedding"
        if embedding_mode and precomputed:
            raise NotImplementedError("metric='precomputed' requires transform_mode='graph'")
        if embedding_mode and y is not None:
            raise NotImplementedError("supervised embedding fit is not implemented")
        if precomputed and y is not None:
            raise NotImplementedError(
                "supervised fitting with metric='precomputed' is not implemented"
            )
        # check_array is public sklearn validation.  Its default finite-value
        # check supplies the captured ``Input contains NaN.`` error while the
        # dtype argument gives float64 inputs the native f32 behavior.
        X_checked = check_array(
            X,
            dtype=np.float32,
            order="C",
            copy=not precomputed,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=1,
        )
        n_rows, n_features = X_checked.shape
        metric = self.metric if precomputed else _METRIC_ALIASES[self.metric.lower()]
        transform_fit = None if precomputed else (X_checked, metric)
        if precomputed:
            if n_rows != n_features:
                raise ValueError("precomputed distances must be a square matrix")
            warnings.warn(_PRECOMPUTED_WARNING, UserWarning, stacklevel=2)
        embedding_options = self._embedding_options(n_rows) if embedding_mode else None

        # Singletons have no graph. Embedding mode exposes the public zero layout.
        if n_rows == 1:
            if precomputed:
                _native._fit_precomputed_graph([X_checked.item()], 1, 1)
            embedding = (
                np.zeros((1, embedding_options["n_components"]), dtype=np.float32)
                if embedding_mode
                else None
            )
            return self._store_fit(n_features, transform_fit, None, None, embedding)

        if n_neighbors > n_rows - 1:
            warnings.warn(_TRUNCATION_WARNING, UserWarning, stacklevel=2)
            n_neighbors = n_rows - 1

        values = X_checked.ravel(order="C").tolist()
        embedding = None
        if precomputed:
            components = _native._fit_precomputed_graph(values, n_rows, n_neighbors)
        else:
            dense_args = (values, n_rows, n_features, n_neighbors, metric)
            if embedding_mode:
                components, embedding_shape, coordinates = _native._fit_dense_embedding(
                    *dense_args, **embedding_options
                )
                embedding = np.asarray(coordinates, dtype=np.float32).reshape(embedding_shape)
            elif y is None:
                components = _native._fit_dense_graph(*dense_args)
            else:
                labels = self._categorical_labels(y, n_rows)
                components = _native._fit_dense_categorical_graph(
                    *dense_args, labels, self.target_weight
                )
        shape, indptr, indices, data = components
        graph = csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=tuple(shape),
            dtype=np.float32,
        )
        # Two rows have no edges after self-neighbors are removed. The public
        # graph fit returns the empty graph and the layout fit rejects it.
        if graph.nnz == 0:
            warnings.warn(
                _DISCONNECTION_WARNING.format(count=n_rows),
                UserWarning,
                stacklevel=2,
            )
            if embedding_mode:
                raise ValueError(
                    "zero-size array to reduction operation maximum which has no identity"
                )
        if embedding is not None:
            embedding[np.diff(graph.indptr) == 0] = np.nan
        return self._store_fit(n_features, transform_fit, graph, n_neighbors, embedding)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the fitted output."""

        if fit_params:
            if self.transform_mode == "graph":
                raise NotImplementedError(
                    "fit parameters are not implemented for the graph-only path"
                )
            raise NotImplementedError("additional fit parameters are not implemented")
        self.fit(X, y=y)
        if self.transform_mode == "embedding":
            return self.embedding_
        return self.graph_

    def transform(self, X):
        """Return the fitted output or a held-out query-to-training COO graph."""

        self._validate_mode()
        if not hasattr(self, "_graph_transform_fit"):
            # Preserve the error text observed through the public transform call.
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_raw_data'")
        if self._graph_transform_fit is None:
            raise NotImplementedError("transform with metric='precomputed' is not implemented")
        training, metric = self._graph_transform_fit
        n_training_rows, n_features = training.shape
        if n_training_rows == 1:
            raise ValueError("Transform unavailable when model was fit with only a single data sample.")
        if issparse(X):
            raise TypeError("sparse input is not supported; pass a dense array")
        queries = check_array(X, dtype=np.float32, order="C")
        n_query_rows, n_query_features = queries.shape
        if n_query_features != n_features:
            raise ValueError(
                f"X has {n_query_features} features, but UMAP is expecting "
                f"{n_features} features as input."
            )
        # The observed fast path compares normalized float32 data by bits.
        if np.array_equal(queries.view(np.uint32), training.view(np.uint32)):
            return self.embedding_ if self.transform_mode == "embedding" else self.graph_
        if self.transform_mode == "embedding":
            raise NotImplementedError("out-of-sample embedding transform is not implemented")

        shape, row, col, data = _native._transform_dense_graph(
            training.ravel().tolist(),
            n_training_rows,
            queries.ravel().tolist(),
            n_query_rows,
            n_features,
            self.n_neighbors_,
            metric,
        )
        return coo_matrix(
            (
                np.asarray(data, dtype=np.float32),
                (np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)),
            ),
            shape=shape,
        )

    def inverse_transform(self, X):
        """Reject inverse transforms."""

        if self.transform_mode == "graph":
            raise NotImplementedError(
                "inverse_transform is not implemented for the graph-only estimator"
            )
        raise NotImplementedError("inverse_transform is not implemented")

    def _validate_mode(self):
        if self.transform_mode not in ("graph", "embedding"):
            raise NotImplementedError(
                "supported transform_mode values are 'graph' and 'embedding'"
            )

    def _store_fit(self, n_features, transform_fit, graph, n_neighbors, embedding):
        if graph is not None:
            self.graph_ = graph
            self.n_neighbors_ = n_neighbors
        elif hasattr(self, "graph_"):
            del self.graph_
            del self.n_neighbors_
        if embedding is not None:
            self.embedding_ = embedding
        elif hasattr(self, "embedding_"):
            del self.embedding_
        self.n_features_in_ = n_features
        self._graph_transform_fit = transform_fit
        return self

    def _validate_embedding_optimizer(self):
        optimizer = self.optimizer
        compatibility = self.compatibility_layout
        if compatibility is not None and not isinstance(compatibility, (bool, np.bool_)):
            raise ValueError("compatibility_layout must be a boolean or None")
        if optimizer is not None and not isinstance(optimizer, str):
            raise ValueError("optimizer must be a string or None")
        if optimizer in ("adam", "momentum"):
            if compatibility:
                raise ValueError(
                    f"optimizer={optimizer!r} requires compatibility_layout=False; "
                    "compatibility_layout=True always uses the legacy optimizer."
                )
            raise NotImplementedError(f"optimizer={optimizer!r} is not implemented")
        if compatibility is None:
            warnings.warn(_COMPATIBILITY_WARNING, FutureWarning, stacklevel=4)
        if optimizer not in (None, "compatibility"):
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Must be one of "
                "'adam', 'momentum', 'compatibility'."
            )
        if compatibility is not None and not compatibility:
            raise NotImplementedError("embedding with compatibility_layout=False is not implemented")

    def _embedding_options(self, n_rows):
        self._validate_embedding_optimizer()
        n_components = self.n_components
        if isinstance(n_components, (bool, np.bool_)):
            raise TypeError("an integer is required for n_components")
        if not isinstance(n_components, numbers.Real):
            raise ValueError("n_components must be an int")
        if n_components <= 0:
            raise ValueError("n_components must be greater than 0")
        if n_components != int(n_components):
            raise ValueError("n_components must be a whole number")
        n_components = int(n_components)

        n_epochs = self.n_epochs
        if n_epochs is None:
            n_epochs = 500 if n_rows <= 10_000 else 200
        elif isinstance(n_epochs, (list, tuple)):
            raise NotImplementedError("embedding supports a scalar n_epochs")
        elif (
            isinstance(n_epochs, bool)
            or not isinstance(n_epochs, numbers.Integral)
            or n_epochs < 0
        ):
            raise ValueError("n_epochs must be a nonnegative integer")

        seed = self.random_state
        if seed is None and n_rows == 1:
            seed = 0  # The singleton layout uses no random draws.
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("embedding requires random_state to be an integer seed")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2**32 - 1")

        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be a positive integer or -1")
        if self.n_jobs != 1 and self.random_state is not None:
            warnings.warn(
                f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. "
                "Use no seed for parallelism.",
                UserWarning,
                stacklevel=3,
            )

        a, b = _resolve_curve_params(self.spread, self.min_dist, self.a, self.b)
        learning_rate = _finite_float(self.learning_rate, "learning_rate")
        repulsion_strength = _finite_float(self.repulsion_strength, "repulsion_strength")
        negative_sample_rate = _finite_float(self.negative_sample_rate, "negative_sample_rate")
        if learning_rate < 0:
            raise ValueError("learning_rate must be nonnegative")
        if repulsion_strength < 0:
            raise ValueError("repulsion_strength must be nonnegative")
        if negative_sample_rate <= 0:
            raise ValueError("negative_sample_rate must be positive")

        return dict(
            n_components=n_components,
            n_epochs=int(n_epochs),
            seed=int(seed),
            a=a,
            b=b,
            learning_rate=learning_rate,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            initial_coordinates=self._embedding_init(n_rows, n_components),
        )

    def _embedding_init(self, n_rows, n_components):
        if isinstance(self.init, str):
            if self.init == "random" or (
                n_rows == 1 and self.init in ("auto", "spectral")
            ):
                return None
            if self.init in ("auto", "pca", "tswspectral", "spectral", "recursive"):
                raise NotImplementedError(f"init={self.init!r} is not implemented for embedding")
            raise ValueError(f"Unknown embedding init {self.init!r}")
        if not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        initial = check_array(self.init, dtype=np.float32, order="C")
        if initial.shape[1] != n_components:
            raise ValueError("init ndarray must match n_components value")
        if initial.shape[0] != n_rows:
            raise ValueError("init ndarray must have the same number of rows as X")
        if np.unique(initial, axis=0).shape[0] != n_rows:
            raise NotImplementedError("explicit init with repeated rows is not implemented")
        return initial.ravel().tolist()

    def _validate_metric(self):
        if callable(self.metric):
            raise NotImplementedError(
                "callable metrics are not implemented by the dense graph path"
            )
        if not isinstance(self.metric, str) or (
            self.metric != "precomputed" and self.metric.lower() not in _METRIC_ALIASES
        ):
            raise ValueError("metric is neither callable nor a recognised string")

    def _categorical_labels(self, y, n_rows):
        if not isinstance(self.target_metric, str) or self.target_metric != "categorical":
            raise NotImplementedError("supervised fitting requires target_metric='categorical'")
        label_count = len(y)
        if label_count != n_rows:
            raise ValueError(
                f"Length of x = {n_rows}, length of y = {label_count}, while it must be equal."
            )
        labels = check_array(y, dtype="numeric", ensure_2d=False)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1:
            raise ValueError("categorical y must contain one label per row")
        if labels.dtype.kind not in "biuf":
            raise ValueError("categorical labels must be real numeric values")

        # Encode by equality to preserve fractional classes and the full uint64
        # range while reserving -1 as the native unlabeled sentinel.
        _, encoded = np.unique(labels, return_inverse=True)
        encoded[labels == -1] = -1
        return encoded.tolist()


__all__ = ["UMAP"]
