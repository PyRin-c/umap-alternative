"""Public estimator for feature graphs and serial seeded dense embeddings."""

from __future__ import annotations

import numbers
import sys
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array

from . import _native
from ._curve import _finite_float, _resolve_curve_params
from ._spectral import _disconnected_spectral_initialization, _spectral_initialization


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
_DEFAULT_PRECOMPUTED_KNN = (None, None, None)
_KNN_SEARCH_WARNING = (
    "precomputed_knn[2] (knn_search_index) is not an NNDescent object: "
    "transforming new data with transform will be unavailable."
)
_KNN_WIDTH_WARNING = (
    "precomputed_knn has a lower number of neighbors than n_neighbors parameter. "
    "precomputed_knn will be ignored and the k-nn will be computed normally."
)
_KNN_ROWS_WARNING = (
    "precomputed_knn has a different number of samples than the data you are fitting. "
    "precomputed_knn will be ignored andthe k-nn will be computed normally."
)
_NO_SEARCH_INDEX = (
    "No search index available: transforming data into an existing embedding is not supported"
)
_EXTERNAL_SEARCH_UNAVAILABLE = "transform with a supplied search index is not implemented"
_INVERSE_EXTERIOR_WARNING = (
    "{count} inverse-transform point(s) lie outside the embedding convex hull; "
    "using nearest-vertex extrapolation."
)
_DISCONNECTION_WARNING = (
    "A large number of your vertices were disconnected from the manifold.\n"
    "Disconnection_distance = {cutoff} has removed {removed} edges.\n"
    "It has fully disconnected {count} vertices.\n"
    "You might consider using find_disconnected_points() to find and remove "
    "these points from your data.\n"
    "Use umap.utils.disconnected_vertices() to identify them."
)
_FEW_DISCONNECTIONS_WARNING = (
    "A few of your vertices were disconnected from the manifold.  This shouldn't cause problems.\n"
    "Disconnection_distance = {cutoff} has removed {removed} edges.\n"
    "It has only fully disconnected {count} vertices.\n"
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


def _default_epochs(n_rows):
    return 500 if n_rows <= 10_000 else 200


def _validate_epochs(n_epochs):
    if isinstance(n_epochs, (list, tuple)):
        raise NotImplementedError("embedding supports a scalar n_epochs")
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, numbers.Integral) or n_epochs < 0:
        raise ValueError("n_epochs must be a nonnegative integer")
    return int(n_epochs)


def _fit_graph_embedding(
    graph, features, metric, rng, initial, *, spectral, n_epochs, n_components, **layout_options
):
    # Public graph_ retains the CSR entries whose memberships were zeroed.
    # Fits of at most ten epochs use the default budget for this cutoff.
    sampling_epochs = n_epochs if n_epochs > 10 else _default_epochs(graph.shape[0])
    unsampled = graph.data < np.float32(graph.data.max() / sampling_epochs)
    sampling_graph = graph
    if np.any(unsampled):
        graph.data[unsampled] = 0
        # SciPy's connectivity check treats stored zeros as edges.
        sampling_graph = graph.copy()
        sampling_graph.eliminate_zeros()

    if spectral:
        count, labels = connected_components(sampling_graph, directed=False)
        if count == 1:
            initial = _spectral_initialization(sampling_graph, n_components, rng)
        else:
            initial = _disconnected_spectral_initialization(
                sampling_graph, features, metric, n_components, rng, labels
            )
        initial = initial.ravel().tolist()
    elif initial is None:
        initial = rng.uniform(-10, 10, (graph.shape[0], n_components)).astype(np.float32)
        initial = initial.ravel().tolist()
    algorithm, words, cursor, has_gaussian, cached_gaussian = rng.get_state()
    shape, coordinates, (words, cursor) = _native._fit_graph_embedding_with_rng(
        (graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()),
        initial,
        (words.tolist(), cursor),
        n_epochs=n_epochs,
        n_components=n_components,
        **layout_options,
    )
    rng.set_state(
        (algorithm, np.asarray(words, dtype=np.uint32), cursor, has_gaussian, cached_gaussian)
    )
    embedding = np.asarray(coordinates, dtype=np.float32).reshape(shape)
    # Isolated rows participate in normalization and sampling before masking.
    embedding[np.diff(sampling_graph.indptr) == 0] = np.nan
    return embedding


def _fit_spectral_embedding(graph, features, metric, rng, **layout_options):
    return _fit_graph_embedding(graph, features, metric, rng, None, spectral=True, **layout_options)


def _fit_neighbor_arrays(indices, distances, k, graph_options):
    if indices.dtype.kind == "f":
        if not np.all(np.isfinite(indices) & (indices == np.trunc(indices))):
            raise ValueError("precomputed_knn indices must contain integer values")
        indices = indices.astype(np.int64)
    if indices.dtype.kind not in "iu" or distances.dtype.kind not in "fiu":
        raise ValueError("precomputed_knn requires integer indices and real distances")
    return _native._fit_supplied_neighbors_graph(
        indices.ravel().tolist(),
        distances.astype(np.float32, copy=False).ravel().tolist(),
        indices.shape,
        k,
        graph_options=graph_options,
    )


def _fit_dense_graph_with_cutoff(features, metric, k, cutoff, graph_options):
    distances = pairwise_distances(features, metric=metric).astype(np.float32, copy=False)
    # Rank the finite distances before masking because all excluded entries
    # become the same missing marker.
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k].copy()
    removed = np.count_nonzero(distances >= cutoff)
    distances = np.take_along_axis(distances, indices, axis=1)
    cut = distances >= cutoff
    indices[cut] = -1
    distances[cut] = np.inf
    return _fit_neighbor_arrays(indices, distances, k, graph_options), removed


def _check_sparse_indices(indices, n_entries, axis_size):
    if indices.shape != (n_entries,) or indices.dtype.kind not in "iu":
        raise ValueError("sparse indices must be one integer per stored value")
    if np.any((indices < 0) | (indices >= axis_size)):
        raise ValueError("sparse indices are out of bounds")


def _sparse_features(X, *, preserve_duplicates):
    if X.format not in ("csr", "csc", "coo"):
        raise TypeError("sparse graph fitting requires SciPy CSR, CSC, or COO input")
    if X.ndim != 2:
        raise ValueError("sparse features must be two-dimensional")
    data = np.asarray(X.data)
    if data.ndim != 1:
        raise ValueError("sparse data must be one-dimensional")
    n_entries = data.size
    n_rows, n_features = X.shape

    # Check raw indices before any SciPy conversion. Rebuilding the matrix
    # also avoids trusting cached sorted/canonical flags after direct edits.
    if X.format == "coo":
        rows, columns = np.asarray(X.row), np.asarray(X.col)
        _check_sparse_indices(rows, n_entries, n_rows)
        _check_sparse_indices(columns, n_entries, n_features)
        csr = coo_matrix((data, (rows, columns)), shape=X.shape, copy=True).tocsr()
    else:
        compressed_rows = X.format == "csr"
        major_size, minor_size = (n_rows, n_features) if compressed_rows else (n_features, n_rows)
        offsets, indices = np.asarray(X.indptr), np.asarray(X.indices)
        if (
            offsets.shape != (major_size + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != n_entries
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("sparse indptr must be nondecreasing from zero to the stored value count")
        _check_sparse_indices(indices, n_entries, minor_size)
        # Compressed inputs are cast before coalescing. COO conversion
        # coalesces in the input dtype, as observed through the public API.
        data = check_array(data, dtype=np.float32, ensure_2d=False, ensure_min_samples=0)
        constructor = csr_matrix if compressed_rows else csc_matrix
        csr = constructor((data, indices, offsets), shape=X.shape, copy=True).tocsr()
    if preserve_duplicates:
        # Cosine normalizes compressed rows with their stored values intact.
        csr.sort_indices()
    else:
        csr.sum_duplicates()
    return check_array(csr, dtype=np.float32, accept_sparse="csr")


class UMAP(BaseEstimator, TransformerMixin):
    """Fit a fuzzy graph or a serial seeded dense UMAP embedding.

    With ``transform_mode="graph"``, ``fit_transform`` returns the fuzzy
    simplicial-set graph as a float32 SciPy CSR matrix. Unsupervised feature
    fits accept dense input and SciPy CSR, CSC, or COO input for Euclidean,
    Manhattan, and Chebyshev. Sparse features also accept cosine. Sparse
    storage is normalized on a copy, preserving repeated CSR entries for cosine.
    Zero-distance ties that exclude a row's self neighbor raise ``ValueError``.
    A singleton graph fit has no ``graph_``.
    Graph ``transform`` requires dense features for both training and queries.
    It returns a float32 SciPy COO matrix with query rows and training columns
    for held-out data, and returns ``graph_`` itself for the fitted data.

    For categorical supervision with dense features, supply one real numeric
    label per row in ``y``. A label of ``-1`` indicates an unlabeled row.
    ``target_weight`` controls the reduction of edges between known classes
    and must lie in the interval [0, 1].

    With ``metric="precomputed"``, provide a dense square distance matrix. After
    conversion to float32, its entries must be finite and non-negative and its
    diagonal must be zero. The rows are ranked independently, so asymmetric
    distances are accepted. Each row's selected neighbors must include its
    diagonal entry, which can be excluded when there are ties at zero distance.

    ``precomputed_knn=(indices, distances, search_index)`` supplies neighbor
    arrays for an unsupervised fit. All columns of matching two-dimensional
    arrays are used when the width reaches ``n_neighbors``. A short width or
    mismatched row count falls back to computed neighbors. The inclusive
    ``disconnection_distance`` cutoff writes ``(-1, +inf)`` into the caller's
    arrays. The fitted input still returns the stored graph or coordinates.
    Held-out transforms after a supplied fit require a search backend, which
    this estimator does not implement.

    For an embedding, use dense features, an integer ``random_state`` in
    ``[0, 2**32 - 1]``, and ``compatibility_layout=True``. The default
    ``init="spectral"`` uses the sampled graph to place its connected components.
    For a connected graph the embedding dimension must be smaller than the
    number of samples. ``init`` can also be ``"random"`` or a finite
    NumPy array of shape ``(n_samples, n_components)`` with distinct rows
    and nonconstant columns. Fitting exposes ``embedding_`` and ``graph_``.
    Dense held-out features transform to ``float32`` coordinates with the
    training layout fixed. ``transform_seed`` (default 42) is an integer in
    ``[0, 2**32 - 1]``, independent of ``random_state``. The normalized fitted
    input returns ``embedding_`` itself. A singleton has a zero embedding and
    no graph. Supervised embeddings and modern optimizers raise explicit errors.
    For a fitted dense Euclidean embedding, ``inverse_transform`` reconstructs
    the original features as a new float32 array. It warns when any query
    requires extrapolation outside the convex hull of the fitted layout.
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
        transform_seed=42,
        precomputed_knn=_DEFAULT_PRECOMPUTED_KNN,
        disconnection_distance=None,
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
        self.transform_seed = transform_seed
        self.precomputed_knn = precomputed_knn
        self.disconnection_distance = disconnection_distance

    def fit(self, X, y=None):
        """Fit the fuzzy graph and, in embedding mode, the training layout."""

        self._validate_mode()
        sparse = issparse(X)
        metric = self._validate_metric(sparse=sparse)
        graph_options = self._graph_options()
        if not isinstance(self.n_neighbors, numbers.Integral):
            raise ValueError("n_neighbors must be an integer")
        n_neighbors = int(self.n_neighbors)
        if n_neighbors <= 1:
            raise ValueError("n_neighbors must be greater than 1")

        precomputed = metric == "precomputed"
        embedding_mode = self.transform_mode == "embedding"
        if sparse:
            if precomputed:
                raise TypeError("sparse input is not supported for metric='precomputed'")
            if embedding_mode:
                raise TypeError("sparse input is not supported in embedding mode")
            if y is not None:
                raise TypeError("sparse input is not supported for supervised fitting")
        if embedding_mode and precomputed:
            raise NotImplementedError("metric='precomputed' requires transform_mode='graph'")
        if embedding_mode and y is not None:
            raise NotImplementedError("supervised embedding fit is not implemented")
        if precomputed and y is not None:
            raise NotImplementedError(
                "supervised fitting with metric='precomputed' is not implemented"
            )
        if sparse:
            X_checked = _sparse_features(X, preserve_duplicates=metric == "cosine")
        else:
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
        transform_fit = None if precomputed else (X_checked, metric)
        if precomputed:
            if n_rows != n_features:
                raise ValueError("precomputed distances must be a square matrix")
            warnings.warn(_PRECOMPUTED_WARNING, UserWarning, stacklevel=2)
        spectral = False
        if embedding_mode:
            embedding_options = self._embedding_options(n_rows)
            initial_coordinates = self._embedding_init(n_rows, embedding_options["n_components"])
            spectral = isinstance(self.init, str) and self.init == "spectral"
        supplied_requested = self.precomputed_knn is not _DEFAULT_PRECOMPUTED_KNN
        if supplied_requested and hasattr(self, "_graph_transform_fit"):
            # Held-out search sees the newly validated input during a refit.
            # The exact-input shortcut still identifies the last completed fit.
            self._graph_transform_fit = transform_fit
        cutoff = self.disconnection_distance
        if cutoff is None:
            cutoff = np.inf
        elif not isinstance(cutoff, (int, float)):
            raise ValueError("disconnection_distance must either be None or a numeric.")
        supplied = self._supplied_neighbors(n_rows, n_neighbors)
        requested_neighbors = n_neighbors

        # Singletons have no graph. Embedding mode exposes the public zero layout.
        if n_rows == 1:
            if precomputed:
                _native._fit_precomputed_graph([X_checked.item()], 1, 1)
            embedding = (
                np.zeros((1, embedding_options["n_components"]), dtype=np.float64)
                if embedding_mode
                else None
            )
            return self._store_fit(n_features, transform_fit, None, None, embedding)

        if n_neighbors > n_rows - 1:
            warnings.warn(_TRUNCATION_WARNING, UserWarning, stacklevel=2)
            n_neighbors = n_rows - 1

        embedding = None
        transform_error = None
        transform_cutoff = None
        removed = 0
        graph_from_neighbors = supplied is not None or cutoff < np.inf
        if supplied is not None:
            if y is not None:
                raise NotImplementedError("supervised fitting with precomputed_knn is not implemented")
            indices, distances, search_index = supplied
            transform_error = _NO_SEARCH_INDEX if search_index is None else _EXTERNAL_SEARCH_UNAVAILABLE
            self._supplied_transform_error = transform_error
            cut = distances >= cutoff
            # Assignment order matters if the second array is read-only.
            indices[cut] = -1
            distances[cut] = np.inf
            removed = np.count_nonzero(cut)
            components = _fit_neighbor_arrays(indices, distances, requested_neighbors, graph_options)
            transform_fit = (X_checked.copy() if precomputed else X_checked, metric)
        elif cutoff < np.inf:
            if sparse or precomputed or y is not None:
                raise NotImplementedError(
                    "disconnection_distance with computed neighbors requires unsupervised dense features"
                )
            transform_cutoff = cutoff
            components, removed = _fit_dense_graph_with_cutoff(
                X_checked, metric, n_neighbors, cutoff, graph_options
            )
        elif precomputed:
            components = _native._fit_precomputed_graph(
                X_checked.ravel(order="C").tolist(), n_rows, n_neighbors, graph_options=graph_options
            )
        elif sparse:
            components = _native._fit_sparse_graph(
                X_checked.indptr.tolist(),
                X_checked.indices.tolist(),
                X_checked.data.tolist(),
                X_checked.shape,
                n_neighbors,
                metric,
                graph_options=graph_options,
            )
        else:
            values = X_checked.ravel(order="C").tolist()
            dense_args = (values, n_rows, n_features, n_neighbors, metric)
            if embedding_mode and not spectral:
                components, embedding_shape, coordinates = _native._fit_dense_embedding(
                    *dense_args,
                    initial_coordinates=initial_coordinates,
                    graph_options=graph_options,
                    **embedding_options,
                )
                embedding = np.asarray(coordinates, dtype=np.float32).reshape(embedding_shape)
            elif y is None:
                components = _native._fit_dense_graph(*dense_args, graph_options=graph_options)
            else:
                labels = self._categorical_labels(y, n_rows)
                components = _native._fit_dense_categorical_graph(
                    *dense_args, labels, self.target_weight, graph_options=graph_options
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
        if supplied is not None and embedding_mode:
            # This stage has succeeded even if the subsequent layout fails.
            self.graph_ = graph
            self.n_neighbors_ = n_neighbors
        isolated = np.count_nonzero(np.diff(graph.indptr) == 0) if graph_from_neighbors else 0
        if isolated or graph.nnz == 0:
            isolated = isolated or n_rows
            warning = (
                _DISCONNECTION_WARNING if isolated * 10 > n_rows else _FEW_DISCONNECTIONS_WARNING
            )
            warnings.warn(
                warning.format(count=isolated, cutoff=cutoff, removed=removed),
                UserWarning,
                stacklevel=2,
            )
        if embedding_mode and graph.nnz == 0:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        if spectral:
            rng = np.random.RandomState(embedding_options.pop("seed"))
            embedding = _fit_spectral_embedding(graph, X_checked, metric, rng, **embedding_options)
        elif embedding_mode and graph_from_neighbors:
            rng = np.random.RandomState(embedding_options.pop("seed"))
            embedding = _fit_graph_embedding(
                graph, X_checked, metric, rng, initial_coordinates, spectral=False, **embedding_options
            )
        elif embedding is not None:
            embedding[np.diff(graph.indptr) == 0] = np.nan
        return self._store_fit(
            n_features, transform_fit, graph, n_neighbors, embedding,
            embedding_options=embedding_options if embedding_mode else None,
            transform_error=transform_error,
            transform_cutoff=transform_cutoff,
        )

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
            return self.embedding_.astype(np.float32, copy=False)
        return self.graph_

    def transform(self, X):
        """Return fitted output or transform held-out features into a layout or COO graph."""

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
        if issparse(training):
            raise NotImplementedError("transform after fitting sparse features is not implemented")
        if issparse(X):
            raise TypeError("sparse input is not supported; pass a dense array")
        queries = check_array(X, dtype=np.float32, order="C")
        n_query_rows, n_query_features = queries.shape
        # The observed fast path compares normalized float32 data by bits.
        fitted_input = getattr(self, "_exact_fit_input", training)
        if fitted_input is not None and np.array_equal(
            queries.view(np.uint32), fitted_input.view(np.uint32)
        ):
            return self.embedding_ if self.transform_mode == "embedding" else self.graph_
        if metric == "precomputed":
            raise NotImplementedError("transform with metric='precomputed' is not implemented")
        transform_error = getattr(self, "_supplied_transform_error", None)
        if transform_error is not None:
            raise NotImplementedError(transform_error)
        if n_query_features != n_features:
            raise ValueError(
                f"X has {n_query_features} features, but UMAP is expecting "
                f"{n_features} features as input."
            )
        if self.transform_mode == "embedding":
            return self._transform_embedding(training, queries, metric)

        shape, row, col, data = _native._transform_dense_graph(
            training.ravel().tolist(),
            n_training_rows,
            queries.ravel().tolist(),
            n_query_rows,
            n_features,
            self.n_neighbors_,
            metric,
            local_connectivity=self.local_connectivity,
            disconnection_distance=getattr(self, "_transform_cutoff", None),
        )
        return coo_matrix(
            (
                np.asarray(data, dtype=np.float32),
                (np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)),
            ),
            shape=shape,
        )

    def _transform_embedding(self, training, queries, metric):
        fitted = self.embedding_
        seed = self._embedding_transform_seed()
        fit_epochs = None if self.n_epochs is None else _validate_epochs(self.n_epochs)
        repulsion, negative_rate = self._sampling_options()
        n_components = fitted.shape[1]
        coordinates = _native._transform_dense_embedding(
            training.ravel().tolist(),
            training.shape[0],
            queries.ravel().tolist(),
            queries.shape[0],
            training.shape[1],
            fitted.ravel().tolist(),
            k=self.n_neighbors_,
            metric=metric,
            n_components=n_components,
            fit_epochs=fit_epochs,
            repulsion_strength=repulsion,
            negative_sample_rate=negative_rate,
            transform_seed=seed,
            local_connectivity=self.local_connectivity,
            disconnection_distance=getattr(self, "_transform_cutoff", None),
            **self._embedding_transform_options,
        )
        return np.asarray(coordinates, dtype=np.float32).reshape(queries.shape[0], n_components)

    def inverse_transform(self, X):
        """Reconstruct dense Euclidean features from points in the fitted layout."""

        if not hasattr(self, "_graph_transform_fit"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_sparse_data'")
        transform_fit = self._graph_transform_fit
        if transform_fit is None:
            raise ValueError("Inverse transform not available for given metric.")
        training, metric = transform_fit
        if issparse(training):
            raise ValueError("Inverse transform not available for sparse input.")
        if self.transform_mode == "graph":
            raise ValueError("Inverse transform not available for transform_mode = 'graph'")
        if metric != "euclidean":
            raise ValueError("Inverse transform not available for given metric.")

        fitted = self.embedding_
        queries = check_array(X, dtype=np.float32, order="C")
        n_components = fitted.shape[1]
        if queries.shape[1] != n_components:
            raise ValueError(
                "Inverse transform input must have the same number of dimensions "
                "as the fitted embedding."
            )
        seed = self._embedding_transform_seed()
        triangulation = Delaunay(fitted, qhull_options="QJ")
        locations = triangulation.find_simplex(queries)
        exterior = np.count_nonzero(locations < 0)
        if exterior:
            warnings.warn(_INVERSE_EXTERIOR_WARNING.format(count=exterior), UserWarning, stacklevel=2)

        fit_epochs = self.n_epochs
        if isinstance(fit_epochs, (list, tuple)):
            fit_epochs = np.max(fit_epochs)
        if fit_epochs is not None:
            fit_epochs = _validate_epochs(fit_epochs)
        repulsion, negative_rate = self._sampling_options()
        n_features = training.shape[1]
        use_computed_scales = (
            getattr(self, "_supplied_transform_error", None) is None
            and getattr(self, "_transform_cutoff", None) is None
        )
        coordinates = _native._inverse_transform_dense_embedding(
            training.ravel().tolist(),
            fitted.ravel().tolist(),
            queries.ravel().tolist(),
            triangulation.simplices.ravel().tolist(),
            [None if index < 0 else index for index in locations.tolist()],
            n_features=n_features,
            n_components=n_components,
            k=self.n_neighbors_,
            fit_epochs=fit_epochs,
            repulsion_strength=repulsion,
            negative_sample_rate=negative_rate,
            transform_seed=seed,
            local_connectivity=getattr(self, "_inverse_local_connectivity", self.local_connectivity),
            use_computed_scales=use_computed_scales,
            **self._embedding_transform_options,
        )
        return np.asarray(coordinates, dtype=np.float32).reshape(queries.shape[0], n_features)

    def _embedding_transform_seed(self):
        seed = self.transform_seed
        if not isinstance(seed, numbers.Integral):
            raise ValueError("embedding transform requires transform_seed to be an integer seed")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        return int(seed)

    def _validate_mode(self):
        if self.transform_mode not in ("graph", "embedding"):
            raise NotImplementedError(
                "supported transform_mode values are 'graph' and 'embedding'"
            )

    def _graph_options(self):
        connectivity = _finite_float(self.local_connectivity, "local_connectivity")
        mix = _finite_float(self.set_op_mix_ratio, "set_op_mix_ratio")
        if connectivity < 0.0:
            raise ValueError("local_connectivity must be nonnegative")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("set_op_mix_ratio must be between 0 and 1, inclusive")
        return None if (connectivity, mix) == (1.0, 1.0) else (connectivity, mix)

    def _supplied_neighbors(self, n_rows, n_neighbors):
        neighbors = self.precomputed_knn
        indices, distances = neighbors[0], neighbors[1]
        search_index = None if len(neighbors) == 2 else neighbors[2]
        if distances is None:
            return None
        if not isinstance(indices, np.ndarray):
            raise ValueError("precomputed_knn[0] must be ndarray object.")
        if not isinstance(distances, np.ndarray):
            raise ValueError("precomputed_knn[1] must be ndarray object.")
        if indices.shape != distances.shape:
            raise ValueError("precomputed_knn[0] and precomputed_knn[1] must be numpy arrays of the same size.")
        # A supplied NNDescent instance already has its public module loaded.
        search_type = getattr(sys.modules.get("pynndescent"), "NNDescent", ())
        if not isinstance(search_index, search_type):
            warnings.warn(_KNN_SEARCH_WARNING, UserWarning, stacklevel=3)
        width = indices.shape[1]
        if width < n_neighbors:
            warnings.warn(_KNN_WIDTH_WARNING, UserWarning, stacklevel=3)
            return None
        if indices.shape[0] != n_rows:
            warnings.warn(_KNN_ROWS_WARNING, UserWarning, stacklevel=3)
            return None
        if indices.ndim != 2:
            raise ValueError("precomputed_knn arrays must be two-dimensional")
        return indices, distances, search_index

    def _store_fit(
        self, n_features, transform_fit, graph, n_neighbors, embedding, *,
        embedding_options=None, transform_error=None, transform_cutoff=None,
    ):
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
        self._exact_fit_input = (
            transform_fit[0] if transform_fit is not None and not issparse(transform_fit[0]) else None
        )
        self._supplied_transform_error = transform_error
        self._transform_cutoff = transform_cutoff
        self._inverse_local_connectivity = float(self.local_connectivity)
        self._embedding_transform_options = None
        if embedding_options is not None:
            self._embedding_transform_options = {
                name: embedding_options[name] for name in ("a", "b", "learning_rate")
            }
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

        n_epochs = _default_epochs(n_rows) if self.n_epochs is None else _validate_epochs(self.n_epochs)

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
        if learning_rate < 0:
            raise ValueError("learning_rate must be nonnegative")
        repulsion_strength, negative_sample_rate = self._sampling_options()

        return dict(
            n_components=n_components,
            n_epochs=n_epochs,
            seed=int(seed),
            a=a,
            b=b,
            learning_rate=learning_rate,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
        )

    def _sampling_options(self):
        repulsion_strength = _finite_float(self.repulsion_strength, "repulsion_strength")
        negative_sample_rate = _finite_float(self.negative_sample_rate, "negative_sample_rate")
        if repulsion_strength < 0:
            raise ValueError("repulsion_strength must be nonnegative")
        if negative_sample_rate <= 0:
            raise ValueError("negative_sample_rate must be positive")
        return repulsion_strength, negative_sample_rate

    def _embedding_init(self, n_rows, n_components):
        if isinstance(self.init, str):
            if self.init in ("random", "spectral") or (n_rows == 1 and self.init == "auto"):
                return None
            if self.init in ("auto", "pca", "tswspectral", "recursive"):
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

    def _validate_metric(self, *, sparse):
        if callable(self.metric):
            raise NotImplementedError(
                "callable metrics are not implemented by the dense graph path"
            )
        if not isinstance(self.metric, str):
            raise ValueError("metric is neither callable nor a recognised string")
        if self.metric == "precomputed":
            return self.metric
        name = self.metric.lower()
        if sparse and name == "cosine":
            return name
        if name not in _METRIC_ALIASES:
            raise ValueError("metric is neither callable nor a recognised string")
        return _METRIC_ALIASES[name]

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
