"""Public estimator for feature graphs and serial seeded dense embeddings."""

from __future__ import annotations

import numbers
import sys
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse, tril, triu
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from sklearn.base import BaseEstimator, TransformerMixin
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
_ANGULAR_METRICS = ("cosine", "correlation")

_TRUNCATION_WARNING = (
    "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
)
_PRECOMPUTED_WARNING = "using precomputed metric; inverse_transform will be unavailable"
_PRECOMPUTED_TRANSFORM_WARNING = (
    "Transforming new data with precomputed metric. We are assuming the input data is a matrix "
    "of distances from the new points to the points in the training set. If the input matrix is "
    "sparse, it should contain distances from the new points to their nearest neighbours or "
    "approximate nearest neighbours in the training set."
)
_DENSITY_NEIGHBOR_WARNING = (
    "Failed to correctly find n_neighbors for some samples. Results may be less than "
    "ideal. Try re-running with different parameters."
)
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


def _default_epochs(n_rows, densmap=False):
    return (500 if n_rows <= 10_000 else 200) + (200 if densmap else 0)


def _csr_graph(components):
    shape, indptr, indices, data = components
    return csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=tuple(shape),
        dtype=np.float32,
    )


_EPOCHS_ERROR = "n_epochs must be a nonnegative integer or a list of nonnegative integers"


def _epoch_checkpoints(n_epochs):
    if not isinstance(n_epochs, (list, tuple, np.ndarray)):
        return None
    if isinstance(n_epochs, np.ndarray) and n_epochs.ndim != 1:
        raise ValueError(_EPOCHS_ERROR)
    if len(n_epochs) == 0 or any(
        not isinstance(epoch, numbers.Integral) or epoch < 0 for epoch in n_epochs
    ):
        raise ValueError(_EPOCHS_ERROR)
    return sorted({int(epoch) for epoch in n_epochs})


def _validate_epochs(n_epochs):
    checkpoints = _epoch_checkpoints(n_epochs)
    if checkpoints is not None:
        return checkpoints[-1]
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, numbers.Integral) or n_epochs < 0:
        raise ValueError(_EPOCHS_ERROR)
    return int(n_epochs)


def _fit_graph_embedding(
    graph, features, metric, rng, initial, *, spectral, n_epochs, n_components,
    checkpoint_epochs=None, densmap=None, **layout_options
):
    # Public graph_ retains the CSR entries whose memberships were zeroed.
    # Fits of at most ten epochs use the default budget for this cutoff.
    sampling_epochs = n_epochs if n_epochs > 10 else _default_epochs(
        graph.shape[0], densmap is not None
    )
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
    density = None if densmap is None else (
        densmap, (features.ravel().tolist(), features.shape[1], metric),
    )
    shape, coordinates, (words, cursor), snapshots = _native._fit_graph_embedding_with_rng(
        (graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()),
        initial,
        (words.tolist(), cursor),
        n_epochs=n_epochs,
        n_components=n_components,
        checkpoint_epochs=checkpoint_epochs,
        density=density,
        **layout_options,
    )
    rng.set_state(
        (algorithm, np.asarray(words, dtype=np.uint32), cursor, has_gaussian, cached_gaussian)
    )
    embedding = np.asarray(coordinates, dtype=np.float32).reshape(shape)
    # Isolated rows participate in normalization and sampling before masking.
    embedding[np.diff(sampling_graph.indptr) == 0] = np.nan
    history = None if checkpoint_epochs is None else [
        np.asarray(values, dtype=np.float32).reshape(shape) for values in snapshots
    ]
    return embedding, history


def _fit_spectral_embedding(graph, features, metric, rng, **layout_options):
    return _fit_graph_embedding(graph, features, metric, rng, None, spectral=True, **layout_options)


def _modern_rng_state(rng):
    _, words, cursor, has_gaussian, gaussian = rng.get_state()
    return words.tolist(), cursor, gaussian if has_gaussian else None


def _restore_modern_rng(rng, state):
    words, cursor, gaussian = state
    rng.set_state((
        "MT19937", np.asarray(words, dtype=np.uint32), cursor,
        gaussian is not None, 0.0 if gaussian is None else gaussian,
    ))


def _fit_modern_initialization(
    components, rng, initial, n_components, rescale, scale_factor, n_epochs, repulsion_strength,
):
    n_rows = components[0][0]
    if rescale == "graph_edge":
        shape, indptr, indices, weights = components
        weights = np.asarray(weights, dtype=np.float32)
        budget = n_epochs if n_epochs > 10 else _default_epochs(n_rows)
        weak = weights < np.float32(weights.max() / budget)
        if weak.any():
            weights[weak] = 0.0
            components = (shape, indptr, indices, weights.tolist())
    coordinates, state = _native._initialize_modern_layout(
        components,
        _modern_rng_state(rng),
        n_components=n_components,
        init="random" if initial is None else "explicit",
        init_rescale=rescale if rescale == "graph_edge" else "unscaled",
        init_scale_factor=scale_factor,
        coordinates=initial,
    )
    _restore_modern_rng(rng, state)
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(n_rows, n_components)
    if rescale != "graph_edge":
        # NumPy's float32 log and exp can round differently from scalar libm.
        lower, upper = np.percentile(coordinates, (5, 95), axis=0).astype(np.float32)
        spans = upper - lower
        spans[spans == 0] = 1.0
        span = np.exp(np.log(spans).mean()) if rescale == "isotropic" else spans.astype(np.float64)
        target_span = 3.0 * np.log10(n_rows) * np.log2(1.0 + repulsion_strength) * scale_factor
        coordinates *= np.asarray(target_span / span, dtype=np.float32)
    return coordinates


def _rank_tied_dense_distances(distances, k):
    return np.argsort(np.asarray(distances, dtype=np.float32))[:k].tolist()


def _fit_modern_embedding(
    components, rng, initial, *, optimizer, n_components, n_epochs,
    rescale, scale_factor, checkpoint_epochs, negative_selection_range,
    exclude_graph_neighbors, repulsion_strength, **layout_options,
):
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    embedding = _fit_modern_initialization(
        components, rng, initial, n_components, rescale, scale_factor, n_epochs, repulsion_strength,
    )
    if n_epochs == 0:
        rng.randint(-2**31 + 1, 2**31 - 2, 3)
        return embedding, None if checkpoint_epochs is None else [embedding.copy()]

    projector = None
    if len(embedding) > negative_selection_range:
        shape = embedding.shape
        order = np.arange(len(embedding), dtype=np.int32)

        def projector(epoch, values):
            if epoch == 0:
                rng.randint(-2**31 + 1, 2**31 - 2, 3)
            rng.shuffle(order)
            direction = rng.randn(n_components)
            direction /= np.linalg.norm(direction)
            coords = np.asarray(values, dtype=np.float32).reshape(shape)
            return np.argsort(coords @ direction).tolist()

    shape = embedding.shape
    values, state, snapshots = _native._fit_modern_layout(
        components, embedding.ravel().tolist(), _modern_rng_state(rng),
        optimizer=optimizer, n_components=n_components, n_epochs=n_epochs,
        checkpoint_epochs=checkpoint_epochs, project=projector,
        exclude_graph_neighbors=exclude_graph_neighbors,
        repulsion_strength=repulsion_strength, **layout_options,
    )
    if projector is None:
        _restore_modern_rng(rng, state)
    history = None if checkpoint_epochs is None else [
        np.asarray(values, dtype=np.float32).reshape(shape) for values in snapshots
    ]
    return np.asarray(values, dtype=np.float32).reshape(shape), history


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


def _check_sparse_indices(indices, n_entries, axis_size):
    if indices.shape != (n_entries,) or indices.dtype.kind not in "iu":
        raise ValueError("sparse indices must be one integer per stored value")
    if np.any((indices < 0) | (indices >= axis_size)):
        raise ValueError("sparse indices are out of bounds")


def _check_compressed_storage(X):
    data = np.asarray(X.data)
    if data.ndim != 1:
        raise ValueError("sparse data must be one-dimensional")
    major_size, minor_size = X.shape if X.format == "csr" else X.shape[::-1]
    offsets, indices = np.asarray(X.indptr), np.asarray(X.indices)
    if (
        offsets.shape != (major_size + 1,)
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or offsets[-1] != data.size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("sparse indptr must be nondecreasing from zero to the stored value count")
    _check_sparse_indices(indices, data.size, minor_size)


def _check_sparse_input(X, *, input_name):
    if X.ndim != 2:
        raise ValueError(f"{input_name} must be two-dimensional")
    if X.format in ("csr", "csc"):
        _check_compressed_storage(X)
    elif X.format == "coo":
        _check_sparse_indices(np.asarray(X.row), X.data.size, X.shape[0])
        _check_sparse_indices(np.asarray(X.col), X.data.size, X.shape[1])
    return check_array(X, dtype=np.float32, accept_sparse="csr")


def _rank_sparse_distances(X, k, *, requested_neighbors=None):
    counts = np.diff(X.indptr)
    if requested_neighbors not in (None, k) and k != 1 and counts[0] > k:
        # Selection uses the effective count but fit buffers use the requested width.
        raise ValueError(
            f"could not broadcast input array from shape ({k},) into shape ({requested_neighbors},)"
        )
    insufficient = np.flatnonzero(counts <= k)
    if insufficient.size:
        row = insufficient[0]
        if counts[row] == k:
            # Keep NumPy's kth bounds error for a row with exactly k stored values.
            np.argpartition(X.data[X.indptr[row]:X.indptr[row + 1]], k)
        if requested_neighbors is not None:
            raise ValueError("Some rows contain fewer than n_neighbors distances!")
        raise ValueError(f"Need at least n_neighbors ({k}) distances for each row!")

    # Validate row widths before allocating rank buffers, whose sizes are now
    # bounded by the stored value count.
    indices = np.empty((X.shape[0], k), dtype=X.indices.dtype)
    distances = np.empty((X.shape[0], k), dtype=np.float32)
    for row, (start, end) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        values = X.data[start:end]
        # kth=k requires an extra stored entry, even though only k ranks are used.
        ranks = np.argpartition(values, k)[:k]
        ranks = ranks[np.argsort(values[ranks])]
        indices[row] = X.indices[start:end][ranks]
        distances[row] = values[ranks]
    return indices, distances


def _fit_sparse_precomputed_graph(X, k, requested_neighbors, cutoff, graph_options):
    # The reference's symmetry check counts stored entries in each triangle.
    if tril(X, k=-1).nnz != triu(X, k=1).nnz:
        raise ValueError("Sparse precomputed distance matrices should be symmetrical!")
    if np.any(X.diagonal() != 0):
        raise ValueError("Non-zero distances from samples to themselves!")
    indices, distances = _rank_sparse_distances(X, k, requested_neighbors=requested_neighbors)
    if requested_neighbors != k:
        # Truncation to a single rank broadcasts it across the requested width.
        indices = np.broadcast_to(indices, (X.shape[0], requested_neighbors)).copy()
        distances = np.broadcast_to(distances, indices.shape).copy()
    cut = distances >= cutoff
    indices[cut] = -1
    distances[cut] = np.inf
    components = _native._fit_supplied_neighbors_graph(
        indices.ravel().tolist(),
        distances.ravel().tolist(),
        indices.shape,
        k,
        graph_options=graph_options,
        sparse_precomputed=True,
    )
    return components, np.count_nonzero(cut)


def _same_transform_input(queries, fitted):
    if fitted is None:
        return False
    if issparse(queries) and issparse(fitted):
        return (
            type(queries) is type(fitted)
            and queries.shape == fitted.shape
            and np.array_equal(queries.indptr, fitted.indptr)
            and np.array_equal(queries.indices, fitted.indices)
            and np.array_equal(queries.data.view(np.uint32), fitted.data.view(np.uint32))
        )
    if not issparse(queries) and not issparse(fitted):
        return np.array_equal(queries.view(np.uint32), fitted.view(np.uint32))
    return False


def _sparse_feature_embedding_output(embedding, X):
    dtype = np.float64 if getattr(X, "dtype", None) == np.dtype(np.float64) else np.float32
    return embedding.astype(dtype, copy=False)


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
        _check_compressed_storage(X)
        offsets, indices = np.asarray(X.indptr), np.asarray(X.indices)
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
    Dense graph and embedding fits also accept cosine and correlation. For these two metrics
    the ranked neighbors may omit self, and the default distance cutoff is 2.
    A singleton graph fit has no ``graph_``.
    Graph ``transform`` accepts dense features or precomputed distances.
    It returns a float32 SciPy COO matrix with query rows and training columns
    for held-out data, and returns ``graph_`` itself for the fitted data.

    For categorical supervision with dense features, supply one real numeric
    label per row in ``y``. A label of ``-1`` indicates an unlabeled row.
    ``target_weight`` controls the reduction of edges between known classes
    and must lie in the interval [0, 1]. In embedding mode the adjusted graph
    determines initialization and sampling.
    For continuous supervision, supply a scalar or vector per row in ``y``
    and select ``target_metric="euclidean"``, ``"manhattan"``, or
    ``"chebyshev"``. ``target_n_neighbors=-1`` uses the effective feature
    neighbor count for the target graph.

    With ``metric="precomputed"``, provide a dense or CSR distance matrix. A
    dense matrix is square. After conversion to float32, its entries must be
    finite and non-negative and its diagonal must be zero. The rows are
    ranked independently, so asymmetric distances are accepted. Each dense
    row's selected neighbors must include its diagonal entry, which can be
    excluded when there are ties at zero distance.
    CSR training requires equal stored counts above and below the diagonal and
    zero diagonal distances. Training sorts CSR columns, and stored zeros and
    duplicate distances count as candidates. Each CSR training or held-out row
    needs more than the effective number of neighbors in stored entries.
    Held-out distances have one column per fitted sample. A finite distance
    cutoff writes infinity into cut elements of a normalized dense fitting
    matrix. CSR fitting applies the cutoff to the selected neighbors.

    ``precomputed_knn=(indices, distances, search_index)`` supplies neighbor
    arrays for an unsupervised fit. All columns of matching two-dimensional
    arrays are used when the width reaches ``n_neighbors``. A short width or
    mismatched row count falls back to computed neighbors. The inclusive
    ``disconnection_distance`` cutoff writes ``(-1, +inf)`` into the caller's
    arrays. The fitted input still returns the stored graph or coordinates.
    Held-out transforms after a supplied fit require a search backend, which
    this estimator does not implement.

    For an embedding, use dense features, sparse Euclidean features, or
    precomputed distances, with an integer ``random_state`` in
    ``[0, 2**32 - 1]`` and ``compatibility_layout=True``.
    The default ``init="spectral"`` uses the sampled graph to place its connected components.
    For a connected graph the embedding dimension must be smaller than the
    number of samples. ``init`` can also be ``"random"`` or a finite
    NumPy array of shape ``(n_samples, n_components)`` with distinct rows
    and nonconstant columns. Fitting exposes ``embedding_`` and ``graph_``.
    Dense held-out features transform to ``float32`` coordinates with the
    training layout fixed. ``transform_seed`` (default 42) is an integer in
    ``[0, 2**32 - 1]``, independent of ``random_state``. The normalized fitted
    input returns ``embedding_`` itself. Sparse Euclidean features also
    support this shortcut, with a float64 copy for float64 queries.
    A singleton has a zero embedding and no graph. Seeded modern Adam and
    momentum layouts accept random or explicit initialization.
    For a fitted dense Euclidean embedding, ``inverse_transform`` reconstructs
    the original features as a new float32 array. It warns when any query
    requires extrapolation outside the convex hull of the fitted layout.
    With ``output_dens=True``, ``fit_transform`` returns the layout and two
    diagnostic log radii for dense Euclidean features.
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
        output_dens=False,
        densmap=False,
        dens_lambda=2.0,
        dens_frac=0.3,
        dens_var_shift=0.1,
        target_n_neighbors=-1,
        init_rescale="isotropic",
        init_scale_factor=1.0,
        exclude_graph_neighbors=False,
        negative_selection_range=200_000,
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
        self.output_dens = output_dens
        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.target_n_neighbors = target_n_neighbors
        self.init_rescale = init_rescale
        self.init_scale_factor = init_scale_factor
        self.exclude_graph_neighbors = exclude_graph_neighbors
        self.negative_selection_range = negative_selection_range

    def fit(self, X, y=None):
        """Fit the fuzzy graph and, in embedding mode, the training layout."""

        self._validate_mode()
        return self._fit(X, y, embedding_mode=self.transform_mode == "embedding")

    def _fit(self, X, y, *, embedding_mode):
        density_options = self._density_options()
        densmap = density_options if getattr(self, "densmap", False) else None
        sparse = issparse(X)
        metric = self._validate_metric(sparse=sparse)
        graph_options = self._graph_options()
        if not isinstance(self.n_neighbors, numbers.Integral):
            raise ValueError("n_neighbors must be an integer")
        n_neighbors = int(self.n_neighbors)
        if n_neighbors <= 1:
            raise ValueError("n_neighbors must be greater than 1")

        precomputed = metric == "precomputed"
        sparse_embedding = sparse and not precomputed and embedding_mode
        angular = not sparse and metric in _ANGULAR_METRICS
        if sparse and not precomputed:
            if sparse_embedding and metric != "euclidean":
                raise TypeError("sparse input in embedding mode requires the Euclidean metric")
            if y is not None:
                raise TypeError("sparse input is not supported for supervised fitting")
        if precomputed and y is not None:
            raise NotImplementedError(
                "supervised fitting with metric='precomputed' is not implemented"
            )
        if sparse and precomputed:
            X_checked = _check_sparse_input(X, input_name="precomputed distances")
            X_checked.sort_indices()
        elif sparse:
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
        transform_fit = (X_checked, metric)
        if sparse_embedding:
            # The identity check compares sorted, validated CSR storage,
            # including any duplicates left by format and dtype conversion.
            checked = _check_sparse_input(X, input_name="sparse features")
            fitted_input = csr_matrix(checked, copy=True)
            fitted_input.sort_indices()
            transform_fit = (fitted_input, metric)
        if precomputed:
            if not sparse and n_rows != n_features:
                raise ValueError("precomputed distances must be a square matrix")
            warnings.warn(_PRECOMPUTED_WARNING, UserWarning, stacklevel=2)
            transform_fit = (X_checked.copy(), metric)
        spectral = False
        modern = False
        checkpoint_epochs = None
        if embedding_mode:
            modern = self._uses_modern_layout()
            if modern and sparse_embedding:
                raise NotImplementedError("sparse features require the compatibility layout")
            checkpoint_epochs = _epoch_checkpoints(self.n_epochs)
            embedding_options = self._embedding_options(
                n_rows, densmap is not None, checkpoint_epochs, modern=modern
            )
            if modern:
                if densmap is not None or self.output_dens:
                    raise NotImplementedError("modern density fitting is not implemented")
                if self.init_rescale not in ("isotropic", "per_axis", "graph_edge"):
                    raise ValueError("invalid init_rescale")
                scale_factor = _finite_float(self.init_scale_factor, "init_scale_factor")
                if scale_factor <= 0:
                    raise ValueError("init_scale_factor must be positive")
                selection_range = _finite_float(
                    self.negative_selection_range, "negative_selection_range"
                )
            initial_coordinates = self._embedding_init(
                n_rows, embedding_options["n_components"], modern=modern
            )
            spectral = isinstance(self.init, str) and self.init == "spectral"
        supplied_requested = self.precomputed_knn is not _DEFAULT_PRECOMPUTED_KNN
        if supplied_requested and hasattr(self, "_graph_transform_fit"):
            # Held-out search sees the newly validated input during a refit.
            # The exact-input shortcut still identifies the last completed fit.
            self._graph_transform_fit = transform_fit
        cutoff = self.disconnection_distance
        if cutoff is None:
            cutoff = 2.0 if angular else np.inf
        elif not isinstance(cutoff, (int, float)):
            raise ValueError("disconnection_distance must either be None or a numeric.")
        if angular and y is not None and self.disconnection_distance is not None:
            raise NotImplementedError(
                "disconnection_distance with computed neighbors requires unsupervised dense features"
            )
        supplied = self._supplied_neighbors(n_rows, n_neighbors)
        if embedding_mode and angular and (densmap is not None or self.output_dens):
            raise NotImplementedError("density fitting with angular distances is not implemented")
        if sparse_embedding and (densmap is not None or self.output_dens):
            raise NotImplementedError("density fitting with sparse features is not implemented")
        if embedding_mode and sparse and precomputed and (
            densmap is not None or getattr(self, "output_dens", False)
        ):
            raise NotImplementedError(
                "density fitting with sparse precomputed distances is not implemented"
            )
        if embedding_mode and (densmap is not None or getattr(self, "output_dens", False)) and (
            supplied is not None or np.isfinite(cutoff)
        ):
            raise NotImplementedError(
                "density fitting requires computed neighbors without a finite cutoff"
            )
        requested_neighbors = n_neighbors

        # Singletons have no graph. Embedding mode exposes the public zero layout.
        if n_rows == 1:
            if precomputed and not sparse:
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
        history = None
        transform_error = None
        transform_cutoff = cutoff if angular else None
        removed = 0
        feature_isolated = None
        graph_from_neighbors = (
            supplied is not None or cutoff < np.inf or (sparse and precomputed) or sparse_embedding
        )
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
            if precomputed:
                transform_fit = (X_checked.copy(), metric)
        elif (angular or cutoff < np.inf) and not (precomputed or sparse) and y is None:
            transform_cutoff = cutoff
            components, removed = _native._fit_dense_graph_with_cutoff(
                X_checked.ravel().tolist(), n_rows, n_features, n_neighbors, metric, cutoff,
                graph_options=graph_options,
            )
        elif precomputed and sparse:
            components, removed = _fit_sparse_precomputed_graph(
                X_checked, n_neighbors, requested_neighbors, cutoff, graph_options
            )
            if cutoff < np.inf:
                transform_cutoff = cutoff
        elif precomputed:
            if cutoff < np.inf:
                components, removed = _native._fit_precomputed_graph_with_cutoff(
                    X_checked.ravel().tolist(), n_rows, n_neighbors, cutoff,
                    graph_options=graph_options,
                )
                X_checked[X_checked >= cutoff] = np.inf
                transform_fit = (X_checked.copy(), metric)
                transform_cutoff = cutoff
            else:
                components = _native._fit_precomputed_graph(
                    X_checked.ravel().tolist(), n_rows, n_neighbors,
                    graph_options=(graph_options or (1.0, 1.0)) if embedding_mode else graph_options,
                )
        elif sparse_embedding:
            components, removed = _native._fit_sparse_euclidean_graph_with_cutoff(
                X_checked.indptr.tolist(),
                X_checked.indices.tolist(),
                X_checked.data.tolist(),
                X_checked.shape,
                n_neighbors,
                cutoff,
                graph_options=graph_options,
            )
        elif cutoff < np.inf:
            raise NotImplementedError(
                "disconnection_distance with computed neighbors requires unsupervised dense features"
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
            if y is not None:
                components, feature_isolated = self._fit_supervised_graph(dense_args, y, graph_options)
            elif embedding_mode and not (spectral or modern) and densmap is None and checkpoint_epochs is None:
                components, embedding_shape, coordinates = _native._fit_dense_embedding(
                    *dense_args,
                    initial_coordinates=initial_coordinates,
                    graph_options=graph_options,
                    **embedding_options,
                )
                embedding = np.asarray(coordinates, dtype=np.float32).reshape(embedding_shape)
            else:
                components = _native._fit_dense_graph(*dense_args, graph_options=graph_options)
        graph = _csr_graph(components)
        empty_continuous_graph = (
            graph.nnz == 0 and y is not None and self.target_metric != "categorical"
        )
        if (supplied is not None and embedding_mode) or (
            y is not None and (embedding_mode or empty_continuous_graph)
            and not hasattr(self, "_graph_transform_fit")
        ):
            # This stage has succeeded even if the subsequent layout fails.
            self.graph_ = graph
            self.n_neighbors_ = n_neighbors
        if feature_isolated is None:
            isolated = np.count_nonzero(np.diff(graph.indptr) == 0) if graph_from_neighbors else 0
            isolated = isolated or (n_rows if graph.nnz == 0 else 0)
        else:
            # Report disconnections in the feature graph before target attenuation.
            isolated = feature_isolated
        if isolated:
            warning = (
                _DISCONNECTION_WARNING if isolated * 10 > n_rows else _FEW_DISCONNECTIONS_WARNING
            )
            warnings.warn(
                warning.format(count=isolated, cutoff=cutoff, removed=removed),
                UserWarning,
                stacklevel=2,
            )
        if graph.nnz == 0 and (embedding_mode or empty_continuous_graph):
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        if modern:
            if embedding_options["negative_sample_rate"] == 0:
                warnings.warn("divide by zero encountered in divide", RuntimeWarning, stacklevel=2)
            layout_options = embedding_options.copy()
            rng = layout_options.pop("seed")
            embedding, history = _fit_modern_embedding(
                components, rng, initial_coordinates,
                optimizer="momentum" if self.optimizer == "momentum" else "adam",
                rescale=self.init_rescale, scale_factor=scale_factor,
                checkpoint_epochs=checkpoint_epochs, negative_selection_range=selection_range,
                exclude_graph_neighbors=bool(self.exclude_graph_neighbors), **layout_options,
            )
            embedding[np.diff(graph.indptr) == 0] = np.nan
        elif spectral:
            rng = np.random.RandomState(embedding_options.pop("seed"))
            embedding, history = _fit_spectral_embedding(
                graph, X_checked, metric, rng, checkpoint_epochs=checkpoint_epochs,
                densmap=densmap, **embedding_options,
            )
        elif embedding_mode and embedding is None:
            rng = np.random.RandomState(embedding_options.pop("seed"))
            embedding, history = _fit_graph_embedding(
                graph, X_checked, metric, rng, initial_coordinates, spectral=False,
                checkpoint_epochs=checkpoint_epochs, densmap=densmap, **embedding_options,
            )
        elif embedding is not None:
            embedding[np.diff(graph.indptr) == 0] = np.nan
        if embedding_mode and getattr(self, "output_dens", False):
            self._store_density_diagnostics(X_checked, graph, embedding, metric)
        if history is not None:
            self.embedding_list_ = history
        return self._store_fit(
            n_features, transform_fit, graph, n_neighbors, embedding,
            embedding_options=embedding_options if embedding_mode else None,
            transform_error=transform_error,
            transform_cutoff=transform_cutoff,
            supervised=y is not None,
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
            if getattr(self, "output_dens", False):
                return self.embedding_, self.rad_orig_, self.rad_emb_
            if issparse(X) and self._graph_transform_fit[1] == "euclidean":
                return _sparse_feature_embedding_output(self.embedding_, X)
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
        if metric != "precomputed" and issparse(training):
            fitted_input = getattr(self, "_exact_fit_input", None)
            if fitted_input is not None:
                if issparse(X):
                    queries = csr_matrix(_check_sparse_input(X, input_name="sparse features"))
                else:
                    queries = csr_matrix(check_array(X, dtype=np.float32, order="C"))
                if _same_transform_input(queries, fitted_input):
                    if self.transform_mode == "graph":
                        return self.graph_
                    return _sparse_feature_embedding_output(self.embedding_, X)
            raise NotImplementedError("transform after fitting sparse features is not implemented")
        sparse = issparse(X)
        if sparse:
            if metric != "precomputed":
                raise TypeError("sparse input is not supported; pass a dense array")
            queries = _check_sparse_input(X, input_name="precomputed distances")
        else:
            queries = check_array(X, dtype=np.float32, order="C")
        n_query_rows, n_query_features = queries.shape
        fitted_input = getattr(self, "_exact_fit_input", training)
        if _same_transform_input(queries, fitted_input):
            return self.embedding_ if self.transform_mode == "embedding" else self.graph_
        if getattr(self, "densmap", False):
            raise NotImplementedError("Transforming data into an existing embedding not supported for densMAP.")
        transform_error = getattr(self, "_supplied_transform_error", None)
        if transform_error is not None:
            raise NotImplementedError(transform_error)
        ranked_columns = None
        if metric == "precomputed":
            warnings.warn(_PRECOMPUTED_TRANSFORM_WARNING, UserWarning, stacklevel=2)
            if n_query_features != n_training_rows:
                raise AssertionError
            if sparse:
                order, queries = _rank_sparse_distances(queries, self.n_neighbors_)
            else:
                # Dense queries rank all columns before selecting neighbors.
                order = np.argsort(queries, axis=1)[:, :self.n_neighbors_]
                queries = np.take_along_axis(queries, order, axis=1)
            if queries.min() < 0.0:
                raise AssertionError
            ranked_columns = order.ravel().tolist()
        elif n_query_features != n_features:
            raise ValueError(
                f"X has {n_query_features} features, but UMAP is expecting "
                f"{n_features} features as input."
            )
        if self.transform_mode == "embedding":
            return self._transform_embedding(training, queries, metric, ranked_columns)

        shape, row, col, data = _native._transform_dense_graph(
            [] if metric == "precomputed" else training.ravel().tolist(),
            n_training_rows,
            queries.ravel().tolist(),
            n_query_rows,
            n_features,
            self.n_neighbors_,
            metric,
            local_connectivity=self.local_connectivity,
            disconnection_distance=getattr(self, "_transform_cutoff", None),
            ranked_columns=ranked_columns,
            rank_ties=_rank_tied_dense_distances if metric in _ANGULAR_METRICS else None,
        )
        return coo_matrix(
            (
                np.asarray(data, dtype=np.float32),
                (np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)),
            ),
            shape=shape,
        )

    def _transform_embedding(self, training, queries, metric, ranked_columns):
        fitted = self.embedding_
        seed = self._embedding_transform_seed()
        fit_epochs = None if self.n_epochs is None else _validate_epochs(self.n_epochs)
        repulsion, negative_rate = self._sampling_options()
        n_components = fitted.shape[1]
        cutoff = getattr(self, "_transform_cutoff", None)
        if metric == "precomputed" and fit_epochs is not None and fit_epochs < 3:
            # The reference computes its pruning threshold even at zero epochs.
            has_weight = cutoff is None or np.any(queries < cutoff)
            np.float32(has_weight) / np.float32(0)
        coordinates = _native._transform_dense_embedding(
            [] if metric == "precomputed" else training.ravel().tolist(),
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
            disconnection_distance=cutoff,
            ranked_columns=ranked_columns,
            rank_ties=_rank_tied_dense_distances if metric in _ANGULAR_METRICS else None,
            **self._embedding_transform_options,
        )
        return np.asarray(coordinates, dtype=np.float32).reshape(queries.shape[0], n_components)

    def update(self, X, ensure_all_finite=True):
        """Append dense features and update the fitted embedding in place."""
        if not hasattr(self, "_graph_transform_fit"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_raw_data'")
        if self._graph_transform_fit is None or self._graph_transform_fit[1] == "precomputed":
            raise ValueError("Update does not currently support precomputed metrics")
        if not hasattr(self, "n_neighbors_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_supervised'")
        if getattr(self, "_supervised_fit", False):
            raise ValueError("Updating supervised models is not currently supported")
        training, metric = self._graph_transform_fit
        if issparse(training) or self._supplied_transform_error is not None:
            raise NotImplementedError("update requires dense features with computed neighbors")
        batch = check_array(
            X, dtype=np.float32, order="C", ensure_all_finite=ensure_all_finite
        )
        combined = np.concatenate((training, batch), axis=0)
        n_rows, n_features = combined.shape
        data = combined.ravel().tolist()
        graph_options = self._graph_options()
        n_neighbors = self.n_neighbors_
        if self._embedding_transform_options is None:
            if metric in _ANGULAR_METRICS:
                components, _ = _native._fit_dense_graph_with_cutoff(
                    data, n_rows, n_features, n_neighbors, metric, np.inf,
                    graph_options=graph_options,
                )
            else:
                components = _native._fit_dense_graph(
                    data, n_rows, n_features, n_neighbors, metric, graph_options=graph_options
                )
            self.graph_ = _csr_graph(components)
            self._graph_transform_fit = (combined, metric)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'embedding_'")
        fitted = self.embedding_
        n_epochs = 0 if self.n_epochs is None else _validate_epochs(self.n_epochs)
        repulsion, negative_rate = self._sampling_options()
        try:
            components, shape, coordinates = _native._update_dense_embedding(
                data,
                combined.shape,
                training.shape[0],
                fitted.ravel().tolist(),
                k=n_neighbors,
                metric=metric,
                n_components=fitted.shape[1],
                n_epochs=n_epochs,
                repulsion_strength=repulsion,
                negative_sample_rate=negative_rate,
                transform_seed=self._embedding_transform_seed(),
                graph_options=graph_options,
                **self._embedding_transform_options,
            )
        except ValueError as error:
            failed_graph = getattr(error, "_update_graph", None)
            if failed_graph is not None:
                # The graph rebuild precedes duplicate-layout jitter.
                self.graph_ = _csr_graph(failed_graph)
                self._graph_transform_fit = (combined, metric)
            raise
        graph = _csr_graph(components)
        embedding = np.asarray(coordinates, dtype=np.float32).reshape(shape)
        if getattr(self, "output_dens", False):
            self._store_density_diagnostics(combined, graph, embedding, metric)
        self._store_fit(
            n_features,
            (combined, metric),
            graph,
            n_neighbors,
            embedding,
            embedding_options=self._embedding_transform_options,
            transform_cutoff=self._transform_cutoff,
        )

    def inverse_transform(self, X):
        """Reconstruct dense Euclidean features from points in the fitted layout."""

        if not hasattr(self, "_graph_transform_fit"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_sparse_data'")
        if getattr(self, "densmap", False):
            raise ValueError("Inverse transform not available for densMAP.")
        transform_fit = self._graph_transform_fit
        if transform_fit is None or transform_fit[1] == "precomputed":
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
        embedding_options=None, transform_error=None, transform_cutoff=None, supervised=False,
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
        self._exact_fit_input = None
        if transform_fit is not None:
            training, metric = transform_fit
            if metric == "precomputed" or not issparse(training) or (
                metric == "euclidean" and embedding is not None
            ):
                self._exact_fit_input = training
        self._supplied_transform_error = transform_error
        self._supervised_fit = supervised
        self._transform_cutoff = transform_cutoff
        self._inverse_local_connectivity = float(self.local_connectivity)
        self._embedding_transform_options = None
        if embedding_options is not None:
            self._embedding_transform_options = {
                name: embedding_options[name] for name in ("a", "b", "learning_rate")
            }
        return self

    def _uses_modern_layout(self):
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
            return True
        if compatibility is None:
            warnings.warn(_COMPATIBILITY_WARNING, FutureWarning, stacklevel=4)
        if optimizer not in (None, "compatibility"):
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Must be one of "
                "'adam', 'momentum', 'compatibility'."
            )
        return compatibility is not None and not compatibility

    def _embedding_options(self, n_rows, densmap, checkpoint_epochs, *, modern):
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

        if self.n_epochs is None:
            n_epochs = _default_epochs(n_rows, densmap)
        else:
            n_epochs = checkpoint_epochs[-1] if checkpoint_epochs else _validate_epochs(self.n_epochs)
        if modern and n_epochs > 0 and self.optimizer == "compatibility":
            raise NotImplementedError(
                "optimizer='compatibility' with compatibility_layout=False supports zero epochs"
            )

        seed = self.random_state
        if seed is None and n_rows == 1:
            seed = 0  # The singleton layout uses no random draws.
        if not (modern and isinstance(seed, np.random.RandomState)):
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                raise ValueError("embedding requires random_state to be an integer seed")
            if not 0 <= seed <= 2**32 - 1:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            seed = int(seed)

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
        repulsion_strength, negative_sample_rate = self._sampling_options(allow_zero=modern)

        return dict(
            n_components=n_components,
            n_epochs=n_epochs,
            seed=seed,
            a=a,
            b=b,
            learning_rate=learning_rate,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
        )

    def _sampling_options(self, *, allow_zero=False):
        repulsion_strength = _finite_float(self.repulsion_strength, "repulsion_strength")
        negative_sample_rate = _finite_float(self.negative_sample_rate, "negative_sample_rate")
        if repulsion_strength < 0:
            raise ValueError("repulsion_strength must be nonnegative")
        if negative_sample_rate < 0 or (negative_sample_rate == 0 and not allow_zero):
            raise ValueError("negative_sample_rate must be positive")
        return repulsion_strength, negative_sample_rate

    def _density_options(self):
        strength = _finite_float(getattr(self, "dens_lambda", 2.0), "dens_lambda")
        fraction = _finite_float(getattr(self, "dens_frac", 0.3), "dens_frac")
        shift = _finite_float(getattr(self, "dens_var_shift", 0.1), "dens_var_shift")
        if strength < 0:
            raise ValueError("dens_lambda cannot be negative")
        if not 0 <= fraction <= 1:
            raise ValueError("dens_frac must be between 0.0 and 1.0")
        if shift < 0:
            raise ValueError("dens_var_shift cannot be negative")
        return strength, fraction, shift

    def _store_density_diagnostics(self, features, graph, embedding, metric):
        if self.n_neighbors >= len(embedding):
            warnings.warn(_DENSITY_NEIGHBOR_WARNING, UserWarning, stacklevel=3)
        original, embedded = _native._density_radius_log_inputs(
            (graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()),
            (features.ravel().tolist(), features.shape[1], metric),
            embedding.ravel().tolist(),
            embedding.shape[1],
            self.n_neighbors,
        )
        self.rad_orig_ = np.log(np.asarray(original, dtype=np.float32))
        self.rad_emb_ = np.log(np.asarray(embedded, dtype=np.float32))

    def _embedding_init(self, n_rows, n_components, *, modern):
        if isinstance(self.init, str):
            if self.init == "random" or (n_rows == 1 and self.init == "auto"):
                return None
            if self.init == "spectral" and not modern:
                return None
            if self.init in ("auto", "pca", "tswspectral", "recursive", "spectral"):
                raise NotImplementedError(f"init={self.init!r} is not implemented for embedding")
            raise ValueError(f"Unknown embedding init {self.init!r}")
        if not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        initial = check_array(self.init, dtype=np.float32, order="C")
        if initial.shape[1] != n_components:
            raise ValueError("init ndarray must match n_components value")
        if initial.shape[0] != n_rows:
            raise ValueError("init ndarray must have the same number of rows as X")
        if not modern and np.unique(initial, axis=0).shape[0] != n_rows:
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
        if name == "cosine" or (name == "correlation" and not sparse):
            return name
        if name not in _METRIC_ALIASES:
            raise ValueError("metric is neither callable nor a recognised string")
        return _METRIC_ALIASES[name]

    def _fit_supervised_graph(self, dense_args, y, graph_options):
        _, n_rows, _, n_neighbors, _ = dense_args
        metric = self.target_metric
        if not isinstance(metric, str) or (metric != "categorical" and metric not in _METRIC_ALIASES):
            raise NotImplementedError(f"target_metric={metric!r} is not implemented")
        label_count = len(y)
        if label_count != n_rows:
            raise ValueError(
                f"Length of x = {n_rows}, length of y = {label_count}, while it must be equal."
            )
        if metric == "categorical":
            return _native._fit_dense_categorical_graph(
                *dense_args, self._categorical_labels(y), self.target_weight,
                graph_options=graph_options,
            )

        if self.transform_mode == "embedding" and (
            getattr(self, "densmap", False) or getattr(self, "output_dens", False)
        ):
            raise NotImplementedError("density fitting with continuous targets is not implemented")
        labels = check_array(y, dtype=np.float32, ensure_2d=False)
        if labels.ndim == 1:
            labels = labels[:, None]
        target_neighbors = getattr(self, "target_n_neighbors", -1)
        if not isinstance(target_neighbors, numbers.Integral) or (
            target_neighbors != -1 and target_neighbors < 2
        ):
            raise ValueError("target_n_neighbors must be -1 or an integer greater than 1")
        if target_neighbors == -1:
            target_neighbors = n_neighbors
        target = (
            labels.ravel().tolist(), labels.shape[1], target_neighbors,
            _METRIC_ALIASES[metric], self.target_weight,
        )
        return _native._fit_dense_continuous_graph(*dense_args, target, graph_options=graph_options)

    def _categorical_labels(self, y):
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
