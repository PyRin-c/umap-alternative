"""Public estimator for feature graphs and serial seeded dense embeddings."""

from __future__ import annotations

import numbers
import sys
import warnings
from functools import partial

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse, tril, triu, vstack
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from . import _native
from ._curve import _finite_float, _resolve_curve_params
from ._feature_search import (
    ApproximateFeatureIndex, ExactFeatureIndex, FeatureSearch,
    rank_tied_dense_distances as _rank_tied_dense_distances,
)
from ._recursive import _modern_recursive_initialization, _robust_spans, _validate_recursive_initialization
from ._spectral import (
    _disconnected_spectral_initialization,
    _modern_spectral_initialization,
    _spectral_initialization,
)
from .distances import _CONFIGURABLE_METRICS


_METRIC_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "cityblock": "manhattan",
    "taxicab": "manhattan",
    "chebyshev": "chebyshev",
    "linfty": "chebyshev",
    "linf": "chebyshev",
    "linfinity": "chebyshev",
    "max": "chebyshev",
    "canberra": "canberra",
    "braycurtis": "braycurtis",
}
_ANGULAR_METRICS = ("cosine", "correlation")
_RATIO_METRICS = ("canberra", "braycurtis")
_SPARSE_BOOLEAN_METRICS = (
    "hamming", "matching", "jaccard", "dice", "russellrao", "rogerstanimoto",
    "sokalmichener", "sokalsneath", "kulsinski",
)
_BOOLEAN_METRICS = _SPARSE_BOOLEAN_METRICS + ("yule",)
_PACKED_METRICS = ("bit_hamming", "bit_jaccard")
_TIE_RANKED_METRICS = _ANGULAR_METRICS + _RATIO_METRICS + _BOOLEAN_METRICS
_SPARSE_LAYOUT_METRICS = (
    ("euclidean", "manhattan", "chebyshev", "cosine") + _RATIO_METRICS + _SPARSE_BOOLEAN_METRICS
)

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


def _is_supplied_search_index(search_index):
    # A supplied NNDescent instance already has its public module loaded.
    pynndescent = getattr(sys.modules.get("pynndescent"), "NNDescent", ())
    return isinstance(search_index, (ExactFeatureIndex, ApproximateFeatureIndex)) or isinstance(search_index, pynndescent)


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


def _warn_disconnected(count, n_rows, cutoff, removed):
    if count:
        warning = _DISCONNECTION_WARNING if count * 10 > n_rows else _FEW_DISCONNECTIONS_WARNING
        warnings.warn(warning.format(count=count, cutoff=cutoff, removed=removed), UserWarning, stacklevel=3)


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
    checkpoint_epochs=None, densmap=None, mask_isolated=True, **layout_options
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
        densmap, _density_input(features, graph, metric),
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
    if mask_isolated:
        embedding[np.diff(sampling_graph.indptr) == 0] = np.nan
    history = None if checkpoint_epochs is None else [
        np.asarray(values, dtype=np.float32).reshape(shape) for values in snapshots
    ]
    return embedding, history


def _fit_spectral_embedding(graph, features, metric, rng, **layout_options):
    return _fit_graph_embedding(graph, features, metric, rng, None, spectral=True, **layout_options)


def _density_input(features, graph, metric):
    if isinstance(metric, FeatureSearch):
        distance = metric
    elif metric in _PACKED_METRICS:
        def distance(x, y):
            mismatch = np.count_nonzero(np.unpackbits(x ^ y))
            if metric == "bit_hamming":
                return mismatch
            union = np.count_nonzero(np.unpackbits(x | y))
            return mismatch / union if union else 0.0
    else:
        return features.ravel().tolist(), features.shape[1], metric
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    values = [distance(features[int(row)], features[int(col)]) for row, col in zip(rows, graph.indices)]
    return values, 0, "edge_distances"


_COMPOSITION_PARAMETERS = (
    "a", "b", "n_components", "n_neighbors", "init", "metric", "random_state",
    "set_op_mix_ratio", "local_connectivity", "target_metric", "target_weight",
    "target_n_neighbors", "learning_rate", "min_dist", "spread", "repulsion_strength",
    "negative_sample_rate", "transform_seed", "dens_lambda", "dens_frac", "dens_var_shift",
    "unique",
)


def _composition_epochs(value):
    if value is None:
        return (-1,)
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(value)
    return (value,)


def _composition_curves(estimator):
    if isinstance(estimator.n_components, tuple):
        return estimator._combined_curves
    layout = getattr(estimator, "_embedding_transform_options", None)
    if layout is not None:
        return ((layout["a"], layout["b"]),)
    return (_resolve_curve_params(
        spread=estimator.spread, min_dist=estimator.min_dist, a=estimator.a, b=estimator.b,
    ),)


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
    *, graph_initializer,
):
    n_rows = components[0][0]
    if graph_initializer is not None or rescale == "graph_edge":
        shape, indptr, indices, weights = components
        weights = np.asarray(weights, dtype=np.float32)
        budget = n_epochs if n_epochs > 10 else _default_epochs(n_rows)
        weak = weights < np.float32(weights.max() / budget)
        if weak.any():
            weights[weak] = 0.0
            components = (shape, indptr, indices, weights.tolist())
    init = "random" if initial is None else "explicit"
    if graph_initializer is not None:
        sampling_graph = _csr_graph(components)
        sampling_graph.eliminate_zeros()
        initial = graph_initializer(sampling_graph, n_components=n_components, rng=rng).ravel().tolist()
        init = "prepared"
    coordinates, state = _native._initialize_modern_layout(
        components,
        _modern_rng_state(rng),
        n_components=n_components,
        init=init,
        init_rescale=rescale if rescale == "graph_edge" else "unscaled",
        init_scale_factor=scale_factor,
        coordinates=initial,
    )
    _restore_modern_rng(rng, state)
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(n_rows, n_components)
    if rescale != "graph_edge":
        # NumPy's float32 log and exp can round differently from scalar libm.
        spans = _robust_spans(coordinates)
        span = np.exp(np.log(spans).mean()) if rescale == "isotropic" else spans.astype(np.float64)
        target_span = 3.0 * np.log10(n_rows) * np.log2(1.0 + repulsion_strength) * scale_factor
        coordinates *= np.asarray(target_span / span, dtype=np.float32)
    return coordinates


def _rank_tied_float64_distances(distances, k):
    return np.argsort(np.asarray(distances, dtype=np.float64))[:k].tolist()


def _rank_tied_sparse_training_distances(distances, k):
    return np.argsort(distances, kind="stable")[:k].tolist()


def _sparse_tie_ranker(metric):
    return _rank_tied_float64_distances if metric == "manhattan" else _rank_tied_dense_distances


def _modern_projector(rng, shape, *, draw_seed):
    order = np.arange(shape[0], dtype=np.int32)

    def project(epoch, values):
        if draw_seed and epoch == 0:
            rng.randint(-2**31 + 1, 2**31 - 2, 3)
        rng.shuffle(order)
        direction = rng.randn(shape[1])
        direction /= np.linalg.norm(direction)
        coords = np.asarray(values, dtype=np.float32).reshape(shape)
        return np.argsort(coords @ direction).tolist()

    return project


def _recursive_shuffle(rng, n_rows):
    adam_order = np.arange(n_rows, dtype=np.int32)
    candidates = adam_order.copy()

    def order(_epoch, _values):
        # Full-range Adam advances both orders. The second dispatches samples.
        rng.shuffle(adam_order)
        rng.shuffle(candidates)
        return candidates.tolist()

    return order


def _refine_recursive_level(graph, coordinates, n_epochs, rng, *, a, b, options, depth, use_refinement):
    shape = coordinates.shape
    mode = options.get("recursive_negative_selection_range_mode", "scaled_coarse")
    projected = mode == "scaled_coarse" and options.get("recursive_negative_selection_range_scale", 0.5) < 1.0
    order = _modern_projector(rng, shape, draw_seed=False) if projected else _recursive_shuffle(rng, shape[0])
    if options.get("curve_schedule", "strong_to_one") == "strong_to_one":
        exponent = 0.25 ** depth
        a, b = a ** exponent, b ** exponent
    components = graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()
    values, _, _ = _native._fit_modern_layout(
        components, coordinates.ravel().tolist(), _modern_rng_state(rng),
        optimizer="adam", n_components=shape[1], n_epochs=n_epochs, a=a, b=b,
        learning_rate=0.5, repulsion_strength=options.get("recursive_repulsion_strength", 4.0),
        negative_sample_rate=options.get("recursive_negative_sample_rate", 1.0),
        exclude_graph_neighbors=False, project=order, refinement_schedule=use_refinement, coarse=True,
    )
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _fit_modern_embedding(
    components, rng, initial, *, optimizer, n_components, n_epochs,
    rescale, scale_factor, checkpoint_epochs, negative_selection_range,
    exclude_graph_neighbors, repulsion_strength, graph_initializer, refinement_schedule, **layout_options,
):
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    embedding = _fit_modern_initialization(
        components, rng, initial, n_components, rescale, scale_factor, n_epochs, repulsion_strength,
        graph_initializer=graph_initializer,
    )
    if n_epochs == 0:
        rng.randint(-2**31 + 1, 2**31 - 2, 3)
        return embedding, None if checkpoint_epochs is None else [embedding.copy()]

    shape = embedding.shape
    projector = (
        _modern_projector(rng, shape, draw_seed=True) if len(embedding) > negative_selection_range else None
    )
    values, state, snapshots = _native._fit_modern_layout(
        components, embedding.ravel().tolist(), _modern_rng_state(rng),
        optimizer=optimizer, n_components=n_components, n_epochs=n_epochs,
        checkpoint_epochs=checkpoint_epochs, project=projector,
        refinement_schedule=refinement_schedule,
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


def _check_sparse_input(X, *, input_name, ensure_all_finite=True):
    if X.ndim != 2:
        raise ValueError(f"{input_name} must be two-dimensional")
    if X.format in ("csr", "csc"):
        _check_compressed_storage(X)
    elif X.format == "coo":
        _check_sparse_indices(np.asarray(X.row), X.data.size, X.shape[0])
        _check_sparse_indices(np.asarray(X.col), X.data.size, X.shape[1])
    return check_array(X, dtype=np.float32, accept_sparse="csr", ensure_all_finite=ensure_all_finite)


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
        if queries.dtype == fitted.dtype == np.dtype(np.uint8):
            return np.array_equal(queries, fitted)
        return np.array_equal(queries.view(np.uint32), fitted.view(np.uint32))
    return False


def _feature_embedding_output(embedding, X, *, preserve_float_dtype=False):
    dtype = getattr(X, "dtype", None)
    if dtype != np.dtype(np.float64) and not (
        preserve_float_dtype and getattr(dtype, "kind", None) == "f"
    ):
        dtype = np.float32
    return embedding.astype(dtype, copy=False)


def _csr_buffers(X):
    return X.shape, X.indptr.tolist(), X.indices.tolist(), X.data.tolist()


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
        # Some distance paths need the individual stored values.
        csr.sort_indices()
    else:
        csr.sum_duplicates()
    return check_array(csr, dtype=np.float32, accept_sparse="csr")


class UMAP(BaseEstimator, TransformerMixin):
    """Fit a fuzzy graph or a serial seeded UMAP embedding.

    With ``transform_mode="graph"``, ``fit_transform`` returns the fuzzy
    simplicial-set graph as a float32 SciPy CSR matrix. Unsupervised feature
    fits accept dense input and SciPy CSR, CSC, or COO input for Euclidean,
    Manhattan, Chebyshev, and cosine. Dense fits also accept correlation.
    Sparse storage is normalized on a copy, preserving repeated CSR entries for cosine.
    Uncut Euclidean graph fits require a self neighbor when there are zero-distance ties.
    Cosine and correlation fits use a default distance cutoff of 2 and permit ties that omit self.
    A singleton graph fit has no ``graph_``.
    Graph ``transform`` accepts dense features or precomputed distances.
    After a sparse feature fit, it also accepts sparse features.
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

    For an embedding, use dense features, sparse Euclidean, Manhattan,
    Chebyshev, or cosine features, or precomputed distances. An integer
    ``random_state`` in ``[0, 2**32 - 1]`` makes the serial layout repeatable.
    The default ``init="spectral"`` uses the sampled graph to place its connected components.
    For a connected graph the embedding dimension must be smaller than the
    number of samples. ``init`` can also be ``"random"`` or a finite
    NumPy array of shape ``(n_samples, n_components)`` with distinct rows
    and nonconstant columns. Fitting exposes ``embedding_`` and ``graph_``.
    Dense held-out features transform to ``float32`` coordinates with the
    training layout fixed. ``transform_seed`` (default 42) is an integer in
    ``[0, 2**32 - 1]``, independent of ``random_state``. The normalized fitted
    input returns ``embedding_`` itself. Sparse feature fits accept dense, CSR,
    CSC, and COO queries. Float64 queries return float64 coordinates for this path.
    A singleton has a zero embedding and no graph. Seeded modern Adam and
    momentum layouts accept spectral, random, and explicit initialization.
    Dense recursive initialization supports the PCA base case and coarsening.
    Coarse layouts use an independent Adam schedule at each level.
    For a fitted dense embedding, ``inverse_transform`` reconstructs
    the original features as a new float32 array. It warns when any query
    requires extrapolation outside the convex hull of the fitted layout.
    With ``output_dens=True``, ``fit_transform`` returns the layout and two
    diagnostic log radii for dense Euclidean features.
    """

    # Defaults for estimators serialized before these options were introduced.
    unique = False
    recursive_coarsening_ratio = 4
    recursive_init_kwds = None
    coarse_layout_weights = "transition"
    coarsen_on_layout_weights = True
    metric_kwds = None
    force_approximation_algorithm = False

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
        unique=False,
        recursive_coarsening_ratio=4,
        recursive_init_kwds=None,
        coarse_layout_weights="transition",
        coarsen_on_layout_weights=True,
        metric_kwds=None,
        force_approximation_algorithm=False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.force_approximation_algorithm = force_approximation_algorithm
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
        self.unique = unique
        self.recursive_coarsening_ratio = recursive_coarsening_ratio
        self.recursive_init_kwds = recursive_init_kwds
        self.coarse_layout_weights = coarse_layout_weights
        self.coarsen_on_layout_weights = coarsen_on_layout_weights

    def fit(self, X, y=None):
        """Fit the fuzzy graph and, in embedding mode, the training layout."""

        self._validate_mode()
        return self._fit(X, y, embedding_mode=self.transform_mode == "embedding")

    def _fit(self, X, y, *, embedding_mode):
        density_options = self._density_options()
        densmap = density_options if getattr(self, "densmap", False) else None
        sparse = issparse(X)
        metric = self._validate_metric(sparse=sparse)
        configured_metric = isinstance(metric, FeatureSearch)
        boolean_metric = metric in _BOOLEAN_METRICS
        packed_metric = metric in _PACKED_METRICS
        packed_supervised = packed_metric and y is not None
        if packed_metric:
            X_checked = check_array(X, dtype=np.uint8, order="C", copy=True)
        if packed_supervised:
            if hasattr(self, "_graph_transform_fit"):
                self._graph_transform_fit = (X_checked, metric)
            if self.target_n_neighbors != -1 and self.target_n_neighbors < 2:
                raise ValueError("target_n_neighbors must be -1 or greater than 1")
        if boolean_metric or packed_metric:
            warnings.warn(
                f"gradient function is not yet implemented for {metric} distance metric; "
                "inverse_transform will be unavailable",
                UserWarning, stacklevel=3,
            )
        graph_options = self._graph_options()
        if not isinstance(self.n_neighbors, numbers.Integral):
            raise ValueError("n_neighbors must be an integer")
        n_neighbors = int(self.n_neighbors)
        if n_neighbors <= 1:
            raise ValueError("n_neighbors must be greater than 1")

        precomputed = metric == "precomputed"
        if precomputed and self.unique:
            raise ValueError("unique=True is not supported with precomputed distances")
        sparse_feature_layout = sparse and (metric in _SPARSE_LAYOUT_METRICS or configured_metric)
        sparse_embedding = sparse and not precomputed and embedding_mode
        angular = not sparse and metric in _ANGULAR_METRICS
        ranked_non_angular = not sparse and (boolean_metric or packed_metric or metric in _RATIO_METRICS)
        if sparse and not precomputed:
            if sparse_embedding and not sparse_feature_layout:
                raise TypeError(
                    "sparse input in embedding mode requires Euclidean, Manhattan, Chebyshev, "
                    "cosine, Canberra, or Bray–Curtis"
                )
            if y is not None and not configured_metric:
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
        elif not packed_metric:
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
        fitted_input = None
        if precomputed:
            if not sparse and n_rows != n_features:
                raise ValueError("precomputed distances must be a square matrix")
            warnings.warn(_PRECOMPUTED_WARNING, UserWarning, stacklevel=2)
            transform_fit = (X_checked.copy(), metric)
        inverse_rows = None
        if self.unique:
            if sparse:
                distinct_input = X_checked
                if not distinct_input.has_canonical_format:
                    distinct_input = X_checked.copy()
                    distinct_input.sum_duplicates()
                first, inverse_rows, _ = _native._unique_csr_rows(_csr_buffers(distinct_input))
                X_checked = distinct_input[first]
            else:
                X_checked, first, inverse_rows = np.unique(
                    X_checked, axis=0, return_index=True, return_inverse=True
                )
            if y is not None:
                if len(y) != n_rows:
                    raise ValueError(
                        f"Length of x = {n_rows}, length of y = {len(y)}, while it must be equal."
                    )
                y = np.asarray(y)[first]
            n_rows = X_checked.shape[0]
        spectral = False
        modern = False
        recursive_options = None
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
                if isinstance(self.init, str) and self.init in ("recursive", "auto"):
                    recursive_options = _validate_recursive_initialization(
                        self.recursive_coarsening_ratio, self.coarse_layout_weights, self.recursive_init_kwds,
                    )
            if self.unique and not isinstance(self.init, str):
                raw_init = np.asarray(self.init)
                if raw_init.ndim == 2 and raw_init.shape[0] != n_rows:
                    raise IndexError("explicit initialization must index the distinct rows")
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
            if angular or metric == "cosine":
                cutoff = 2.0
            elif metric == "bit_jaccard":
                cutoff = 1
            else:
                cutoff = 1.0 if metric in ("jaccard", "dice") else np.inf
        elif not isinstance(cutoff, (int, float)):
            raise ValueError("disconnection_distance must either be None or a numeric.")
        if angular and y is not None and self.disconnection_distance is not None:
            raise NotImplementedError(
                "disconnection_distance with computed neighbors requires unsupervised dense features"
            )
        supplied = self._supplied_neighbors(n_rows, n_neighbors)
        if sparse_feature_layout:
            checked = _check_sparse_input(X, input_name="sparse features")
            if metric == "manhattan" and supplied is None:
                # When fitting coalesces stored entries, ordinary CSR copies
                # take the held-out path in the public estimator.
                if checked.nnz == transform_fit[0].nnz:
                    fitted_input = transform_fit[0]
            else:
                # Other paths retain duplicates left by format and dtype conversion.
                fitted_input = csr_matrix(checked, copy=True)
                fitted_input.sort_indices()
        if embedding_mode and angular and (densmap is not None or self.output_dens):
            raise NotImplementedError("density fitting with angular distances is not implemented")
        if sparse_embedding and not configured_metric and (densmap is not None or self.output_dens):
            raise NotImplementedError("density fitting with sparse features is not implemented")
        if embedding_mode and sparse and precomputed and (
            densmap is not None or getattr(self, "output_dens", False)
        ):
            raise NotImplementedError(
                "density fitting with sparse precomputed distances is not implemented"
            )
        if embedding_mode and (densmap is not None or getattr(self, "output_dens", False)) and (
            supplied is not None or (np.isfinite(cutoff) and not (configured_metric or packed_metric or sparse_embedding))
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
            return self._store_fit(
                n_features, transform_fit, None, None, embedding, exact_fit_input=fitted_input,
            )

        if n_neighbors > n_rows - 1:
            warnings.warn(_TRUNCATION_WARNING, UserWarning, stacklevel=2)
            n_neighbors = n_rows - 1

        embedding = None
        history = None
        transform_error = None
        feature_search_index = None
        inverse_distances = None
        transform_cutoff = cutoff if angular else None
        removed = 0
        feature_isolated = None
        ranked_sparse_graph = sparse_feature_layout and (
            embedding_mode or cutoff < np.inf or metric != "euclidean"
        )
        approximate_supported = (
            not sparse and metric in ("euclidean", "cosine", "correlation")
            and y is None and densmap is None and not self.output_dens and n_neighbors >= 2
        )
        approximate = supplied is None and (
            self.force_approximation_algorithm
            or (approximate_supported and _native._uses_approximate_search(n_rows, False))
        )
        graph_from_neighbors = (
            supplied is not None or cutoff < np.inf or (sparse and precomputed) or ranked_sparse_graph
            or ranked_non_angular or configured_metric or approximate
        )
        if supplied is not None:
            if y is not None:
                raise NotImplementedError("supervised fitting with precomputed_knn is not implemented")
            indices, distances, search_index = supplied
            if _is_supplied_search_index(search_index):
                feature_search_index = search_index
                transform_cutoff = cutoff
            else:
                transform_error = _NO_SEARCH_INDEX if search_index is None else _EXTERNAL_SEARCH_UNAVAILABLE
            self._supplied_transform_error = transform_error
            self._feature_search_index = feature_search_index
            cut = distances >= cutoff
            # Assignment order matters if the second array is read-only.
            indices[cut] = -1
            distances[cut] = np.inf
            removed = np.count_nonzero(cut)
            components = _fit_neighbor_arrays(indices, distances, requested_neighbors, graph_options)
            if embedding_mode and not (precomputed or sparse):
                inverse_distances = (
                    np.array(distances, dtype=np.float32, order="C", copy=True),
                    requested_neighbors,
                )
            if precomputed:
                transform_fit = (X_checked.copy(), metric)
        elif approximate:
            if not approximate_supported:
                raise NotImplementedError(
                    "native approximate fitting requires dense unsupervised Euclidean, cosine, "
                    "or correlation features with at least two effective neighbors"
                )
            rng = check_random_state(self.random_state)
            feature_search_index = ApproximateFeatureIndex(
                X_checked, n_neighbors, metric, 0,
                query_rows=first if inverse_rows is not None else None, rng=rng,
            )
            if embedding_mode:
                embedding_options["seed"] = rng
            indices, distances = feature_search_index.training_ranks()
            cut = (indices >= 0) & (distances >= cutoff)
            removed = np.count_nonzero(cut)
            indices[cut] = -1
            distances[cut] = np.inf
            components = _fit_neighbor_arrays(indices, distances, n_neighbors, graph_options)
            transform_cutoff = cutoff
        elif packed_metric:
            transform_cutoff = cutoff
            components, removed = _native._fit_packed_graph(
                X_checked.tobytes(), X_checked.shape, n_neighbors, metric, cutoff,
                graph_options=graph_options,
            )
            if packed_supervised:
                # Target validation follows publication of the feature graph.
                self.graph_ = _csr_graph(components)
                self.n_neighbors_ = n_neighbors
                self._transform_cutoff = cutoff
                _warn_disconnected(np.count_nonzero(np.diff(self.graph_.indptr) == 0), n_rows, cutoff, removed)
                components = self._apply_supervision_to_graph(components, y, n_rows, n_neighbors)
        elif configured_metric:
            transform_cutoff = cutoff
            indices, distances, removed = metric.rank(X_checked, n_neighbors, cutoff=cutoff)
            components = _fit_neighbor_arrays(indices, distances, n_neighbors, graph_options)
            if y is not None:
                feature_isolated = np.count_nonzero(np.diff(components[1]) == 0)
                components = self._apply_supervision_to_graph(
                    components, y, n_rows, n_neighbors, feature_kind="configured features",
                )
        elif (angular or ranked_non_angular or cutoff < np.inf) and not (precomputed or sparse) and y is None:
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
        elif ranked_sparse_graph:
            # Euclidean pairwise distances coalesce these validated entries
            # after widening their values, while the identity keeps them stored.
            ranking_input = (
                fitted_input if metric == "euclidean" and inverse_rows is None else X_checked
            )
            components, removed = _native._fit_sparse_feature_graph_with_cutoff(
                _csr_buffers(ranking_input),
                n_neighbors,
                metric,
                cutoff,
                _rank_tied_sparse_training_distances,
                graph_options=graph_options,
            )
            if cutoff < np.inf:
                transform_cutoff = cutoff
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
            elif (embedding_mode and not (spectral or modern) and densmap is None and checkpoint_epochs is None
                  and isinstance(embedding_options["seed"], int)):
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
        if packed_supervised or (supplied is not None and embedding_mode) or (
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
        if not packed_supervised:
            _warn_disconnected(isolated, n_rows, cutoff, removed)
        if graph.nnz == 0 and (embedding_mode or empty_continuous_graph):
            if (boolean_metric or (packed_metric and not packed_supervised)
                    or (sparse_feature_layout and metric == "euclidean")) and not hasattr(
                self, "_graph_transform_fit"
            ):
                # A cold fit can publish the empty graph before layout fails.
                self._store_fit(
                    n_features, transform_fit, graph, n_neighbors, None,
                    transform_cutoff=transform_cutoff, exact_fit_input=fitted_input,
                )
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        if modern:
            if embedding_options["negative_sample_rate"] == 0:
                warnings.warn("divide by zero encountered in divide", RuntimeWarning, stacklevel=2)
            layout_options = embedding_options.copy()
            rng = layout_options.pop("seed")
            graph_initializer = None
            if spectral:
                graph_initializer = partial(_modern_spectral_initialization, features=X_checked, metric=metric)
            elif recursive_options is not None:
                graph_initializer = partial(
                    _modern_recursive_initialization, features=X_checked, options=recursive_options,
                    ratio=self.recursive_coarsening_ratio,
                    weights=self.coarse_layout_weights if self.coarsen_on_layout_weights else "fuzzy_union",
                    refine=partial(
                        _refine_recursive_level, a=layout_options["a"], b=layout_options["b"],
                        options=recursive_options,
                    ),
                )
            embedding, history = _fit_modern_embedding(
                components, rng, initial_coordinates,
                optimizer="momentum" if self.optimizer == "momentum" else "adam",
                graph_initializer=graph_initializer,
                refinement_schedule=recursive_options is not None,
                rescale=self.init_rescale, scale_factor=scale_factor,
                checkpoint_epochs=checkpoint_epochs, negative_selection_range=selection_range,
                exclude_graph_neighbors=bool(self.exclude_graph_neighbors), **layout_options,
            )
            embedding[np.diff(graph.indptr) == 0] = np.nan
        elif spectral or (embedding_mode and embedding is None):
            rng = check_random_state(embedding_options.pop("seed"))
            layout_options = dict(
                checkpoint_epochs=checkpoint_epochs, densmap=densmap, **embedding_options,
            )
            try:
                if spectral:
                    embedding, history = _fit_spectral_embedding(
                        graph, X_checked, metric, rng, **layout_options,
                    )
                else:
                    embedding, history = _fit_graph_embedding(
                        graph, X_checked, metric, rng, initial_coordinates, spectral=False,
                        **layout_options,
                    )
            except (ZeroDivisionError, FloatingPointError, RuntimeWarning):
                if densmap is not None:
                    self.graph_ = graph
                raise
        elif embedding is not None:
            embedding[np.diff(graph.indptr) == 0] = np.nan
        if embedding_mode and getattr(self, "output_dens", False):
            if densmap is not None and np.isnan(embedding).any():
                self.graph_ = graph
                self.embedding_ = embedding
                if history is not None:
                    self.embedding_list_ = history
                raise ValueError("Input contains NaN.")
            self._store_density_diagnostics(X_checked, graph, embedding, metric)
            if inverse_rows is not None:
                self.rad_orig_ = self.rad_orig_[inverse_rows]
                self.rad_emb_ = self.rad_emb_[inverse_rows]
        if embedding is not None and inverse_rows is not None:
            embedding = embedding[inverse_rows]
        if history is not None:
            self.embedding_list_ = (
                history if inverse_rows is None else [layout[inverse_rows] for layout in history]
            )
        return self._store_fit(
            n_features, transform_fit, graph, n_neighbors, embedding,
            embedding_options=embedding_options if embedding_mode else None,
            transform_error=transform_error,
            feature_search_index=feature_search_index,
            transform_cutoff=transform_cutoff,
            supervised=y is not None,
            exact_fit_input=fitted_input,
            inverse_cutoff=transform_cutoff,
            inverse_distances=inverse_distances,
            fit_inverse_rows=inverse_rows,
            inverse_unique_fit=(
                (X_checked, inverse_rows) if embedding_mode and inverse_rows is not None and not sparse
                else None
            ),
        )

    def __add__(self, other):
        return self._combine(other, "union")

    def __mul__(self, other):
        return self._combine(other, "intersection")

    def __sub__(self, other):
        return self._combine(other, "difference")

    def _combine(self, other, operation):
        if not isinstance(other, UMAP):
            return NotImplemented
        check_is_fitted(self, attributes=["graph_"])
        check_is_fitted(other, attributes=["graph_"])
        if self.graph_.shape != other.graph_.shape:
            raise ValueError("UMAP models must have the same number of vertices to combine")
        left = self.graph_.tocsr()
        right = other.graph_.tocsr()
        if not left.nnz or not right.nnz:
            raise ValueError("zero-size array to reduction operation minimum which has no identity")
        components, left_weights = _native._combine_fuzzy_graphs(
            _csr_buffers(left), _csr_buffers(right), operation
        )
        if left_weights is not None:
            left.data[:] = left_weights
        graph = _csr_graph(components)
        if not graph.nnz:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")

        parameters = {}
        for name in _COMPOSITION_PARAMETERS:
            values = []
            for estimator in (self, other):
                value = getattr(estimator, name)
                if isinstance(estimator.n_components, tuple):
                    values.extend(value if isinstance(value, tuple) else (value,) * len(estimator.n_components))
                else:
                    values.append(value)
            parameters[name] = tuple(values)
        epochs = _composition_epochs(self.n_epochs) + _composition_epochs(other.n_epochs)
        parameters["n_epochs"] = epochs if max(epochs) >= 0 else None
        n_epochs = _default_epochs(graph.shape[0]) if parameters["n_epochs"] is None else max(epochs)
        curves = _composition_curves(self) + _composition_curves(other)
        a, b = np.mean(curves, axis=0)
        spectral = connected_components(graph, directed=False, return_labels=False) == 1
        if not spectral:
            warnings.warn(
                "Combined graph is not connected but mult-component layout is unsupported. "
                "Falling back to random initialization.",
                UserWarning,
                stacklevel=3,
            )
        embedding, _ = _fit_graph_embedding(
            graph, None, "euclidean", np.random.RandomState(42), None,
            spectral=spectral, mask_isolated=False, n_epochs=int(n_epochs),
            n_components=min(parameters["n_components"]), a=float(a), b=float(b),
            learning_rate=min(parameters["learning_rate"]),
            repulsion_strength=float(np.mean(parameters["repulsion_strength"])),
            negative_sample_rate=float(np.mean(parameters["negative_sample_rate"])),
        )
        result = UMAP(**parameters, compatibility_layout=True, optimizer="compatibility")
        result._combined_curves = curves
        result.graph_ = graph
        result.embedding_ = embedding
        return result

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
            metric = self._graph_transform_fit[1]
            packed_metric = metric in _PACKED_METRICS
            if packed_metric or metric in _BOOLEAN_METRICS or (issparse(X) and metric in _SPARSE_LAYOUT_METRICS):
                return _feature_embedding_output(self.embedding_, X, preserve_float_dtype=packed_metric)
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
        configured_metric = isinstance(metric, FeatureSearch)
        n_training_rows, n_features = training.shape
        if n_training_rows == 1:
            raise ValueError("Transform unavailable when model was fit with only a single data sample.")
        if self.unique and not hasattr(self, "n_neighbors_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_input_hash'")
        sparse_fitted = metric != "precomputed" and issparse(training)
        if sparse_fitted and not (configured_metric or metric in _SPARSE_LAYOUT_METRICS):
            raise NotImplementedError("transform after fitting sparse features is not implemented")
        sparse = issparse(X)
        boolean_metric = metric in _BOOLEAN_METRICS
        packed_metric = metric in _PACKED_METRICS
        sparse_features = sparse_fitted or (sparse and (boolean_metric or configured_metric))
        if packed_metric:
            queries = check_array(X, dtype=np.uint8, order="C")
        elif sparse:
            if sparse_features:
                queries = csr_matrix(_check_sparse_input(X, input_name="sparse features"))
            elif metric == "precomputed":
                queries = _check_sparse_input(X, input_name="precomputed distances")
            else:
                raise TypeError("sparse input is not supported; pass a dense array")
        else:
            queries = check_array(X, dtype=np.float32, order="C")
            if sparse_features:
                queries = csr_matrix(queries)
        n_query_rows, n_query_features = queries.shape
        fitted_input = getattr(self, "_exact_fit_input", training)
        if fitted_input is None and sparse_features and metric == "euclidean":
            fitted_input = training
        if _same_transform_input(queries, fitted_input):
            if self.transform_mode == "graph":
                return self.graph_
            if sparse_features or boolean_metric or packed_metric:
                return _feature_embedding_output(self.embedding_, X, preserve_float_dtype=packed_metric)
            return self.embedding_
        if getattr(self, "densmap", False):
            raise NotImplementedError("Transforming data into an existing embedding not supported for densMAP.")
        transform_error = getattr(self, "_supplied_transform_error", None)
        if transform_error is not None:
            raise NotImplementedError(transform_error)
        if (boolean_metric or packed_metric) and self.transform_mode == "graph":
            check_random_state(self.transform_seed)
        ranked_columns = None
        external_index = getattr(self, "_feature_search_index", None)
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
        elif external_index is not None:
            if n_query_features != n_features:
                raise ValueError(f"expected {n_features} query features, got {n_query_features}")
            order, distances = external_index.query(queries, k=self.n_neighbors_)
            order = np.asarray(order)
            distances = np.asarray(distances, dtype=np.float32)
            if (order.shape != (n_query_rows, self.n_neighbors_) or distances.shape != order.shape
                    or order.dtype.kind not in "iu"
                    or np.any((order < -1) | (order >= min(n_training_rows, 2**31)))
                    or np.isnan(distances).any() or np.any(distances < 0)):
                raise ValueError("the supplied search index returned invalid neighbor ranks")
            missing = (order == -1) | np.isposinf(distances)
            ranked_columns = np.where(missing, -1, order.astype(np.int32, copy=False)).ravel().tolist()
            queries = np.where(missing, np.float32(np.inf), distances)
        elif packed_metric:
            ranked_columns, distances = _native._rank_packed_queries(
                (training.tobytes(), training.shape), (queries.tobytes(), queries.shape),
                self.n_neighbors_, metric, _rank_tied_dense_distances,
            )
            queries = np.asarray(distances, dtype=np.float32).reshape(n_query_rows, self.n_neighbors_)
        elif configured_metric:
            order, queries, _ = metric.rank(training, self.n_neighbors_, queries=queries)
            ranked_columns = order.ravel().tolist()
        elif sparse_features:
            if n_query_features > n_features or (boolean_metric and n_query_features != n_features):
                raise ValueError(
                    f"X has {n_query_features} features, but UMAP was fit with {n_features}."
                )
            preserve_duplicates = metric == "cosine" and n_query_features == n_features
            if not sparse_fitted:
                training = csr_matrix(training)
            # Narrow cosine uses coalesced values in the shared feature prefix.
            # Earlier fits may have kept the raw CSR identity snapshot here.
            if not preserve_duplicates and not training.has_canonical_format:
                training = _sparse_features(training, preserve_duplicates=False)
            rank_ties = (
                _rank_tied_dense_distances
                if n_query_features < n_features
                else _sparse_tie_ranker(metric)
            )
            ranked_columns, distances = _native._rank_sparse_feature_queries(
                _csr_buffers(training),
                _csr_buffers(_sparse_features(queries, preserve_duplicates=preserve_duplicates)),
                self.n_neighbors_,
                metric,
                rank_ties,
            )
            queries = np.asarray(distances, dtype=np.float32).reshape(n_query_rows, self.n_neighbors_)
        elif n_query_features != n_features:
            raise ValueError(
                f"X has {n_query_features} features, but UMAP is expecting "
                f"{n_features} features as input."
            )
        if self.transform_mode == "embedding":
            coordinates = self._transform_embedding(training, queries, metric, ranked_columns)
            if sparse_features or boolean_metric or packed_metric:
                return _feature_embedding_output(coordinates, X, preserve_float_dtype=packed_metric)
            return coordinates

        shape, row, col, data = _native._transform_dense_graph(
            [] if ranked_columns is not None else training.ravel().tolist(),
            n_training_rows,
            queries.ravel().tolist(),
            n_query_rows,
            n_features,
            self.n_neighbors_,
            "euclidean" if configured_metric else metric,
            local_connectivity=self.local_connectivity,
            disconnection_distance=getattr(self, "_transform_cutoff", None),
            ranked_columns=ranked_columns,
            rank_ties=_rank_tied_dense_distances if self.unique or metric in _TIE_RANKED_METRICS else None,
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
        packed_metric = metric in _PACKED_METRICS
        partial_fit = (packed_metric or metric in _BOOLEAN_METRICS) and fitted.shape[0] < training.shape[0]
        epochs = self.n_epochs
        if (partial_fit and packed_metric and isinstance(epochs, numbers.Real)
                and not isinstance(epochs, numbers.Integral) and np.isfinite(epochs) and epochs >= 0):
            # A failed update can leave a floating budget that transform floors.
            epochs = int(epochs)
        invalid_epochs = False
        try:
            fit_epochs = None if epochs is None else _validate_epochs(epochs)
        except ValueError:
            if not partial_fit:
                raise
            # Membership initialization can fail before the invalid schedule.
            fit_epochs, invalid_epochs = 0, True
        repulsion, negative_rate = self._sampling_options()
        n_components = fitted.shape[1]
        cutoff = getattr(self, "_transform_cutoff", None)
        if metric == "precomputed" and fit_epochs is not None and fit_epochs < 3:
            # The reference computes its pruning threshold even at zero epochs.
            has_weight = cutoff is None or np.any(queries < cutoff)
            np.float32(has_weight) / np.float32(0)
        coordinates = _native._transform_dense_embedding(
            [] if ranked_columns is not None else training.ravel().tolist(),
            training.shape[0],
            queries.ravel().tolist(),
            queries.shape[0],
            training.shape[1],
            fitted.ravel().tolist(),
            k=self.n_neighbors_,
            metric="euclidean" if isinstance(metric, FeatureSearch) else metric,
            n_components=n_components,
            fit_epochs=fit_epochs,
            repulsion_strength=repulsion,
            negative_sample_rate=negative_rate,
            transform_seed=seed,
            local_connectivity=self.local_connectivity,
            disconnection_distance=cutoff,
            ranked_columns=ranked_columns,
            rank_ties=_rank_tied_dense_distances if self.unique or metric in _TIE_RANKED_METRICS else None,
            partial_fit=partial_fit,
            invalid_epochs=invalid_epochs,
            **self._embedding_transform_options,
        )
        return np.asarray(coordinates, dtype=np.float32).reshape(queries.shape[0], n_components)

    def update(self, X, ensure_all_finite=True):
        """Append feature rows and update the fitted embedding."""
        if not hasattr(self, "_graph_transform_fit"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_raw_data'")
        if self._graph_transform_fit is None or self._graph_transform_fit[1] == "precomputed":
            raise ValueError("Update does not currently support precomputed metrics")
        if not hasattr(self, "n_neighbors_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_supervised'")
        if getattr(self, "_supervised_fit", False):
            raise ValueError("Updating supervised models is not currently supported")
        training, metric = self._graph_transform_fit
        sparse = issparse(training)
        if ((sparse and metric not in _SPARSE_LAYOUT_METRICS and not isinstance(metric, FeatureSearch))
                or self._supplied_transform_error is not None
                or getattr(self, "_feature_search_index", None) is not None):
            raise NotImplementedError("update requires dense features with computed neighbors")
        if sparse and not isinstance(metric, FeatureSearch):
            return self._update_sparse_features(X, training, metric, ensure_all_finite)
        if metric in _PACKED_METRICS:
            batch = check_array(X, dtype=np.uint8, order="C", ensure_all_finite=ensure_all_finite)
            combined = np.concatenate((training, batch), axis=0)
            seed = self._embedding_transform_seed()
            n_existing = training.shape[0]
            components, ranks = _native._fit_packed_update_graph(
                (combined.tobytes(), combined.shape), n_existing, self.n_neighbors_, metric,
                graph_options=self._graph_options(),
            )
            return self._update_ranked_features(combined, metric, n_existing, components, ranks, seed)
        boolean_metric = metric in _BOOLEAN_METRICS
        if boolean_metric and issparse(X):
            raise ValueError("Sparse batches cannot be appended to a dense fit.")
        if isinstance(metric, FeatureSearch) and (sparse or issparse(X)):
            batch = check_array(X, dtype=np.float32, accept_sparse="csr", ensure_all_finite=ensure_all_finite)
            combined = vstack((csr_matrix(training), csr_matrix(batch)), format="csr")
        else:
            batch = check_array(X, dtype=np.float32, order="C", ensure_all_finite=ensure_all_finite)
            combined = np.concatenate((training, batch), axis=0)
        if isinstance(metric, FeatureSearch):
            if self.densmap:
                raise NotImplementedError("updating a densMAP layout is unavailable for configured metrics")
            n_existing, k = training.shape[0], self.n_neighbors_
            indices, distances, _ = metric.rank(combined, k)
            components = _fit_neighbor_arrays(indices, distances, k, self._graph_options())
            if self.transform_mode == "graph":
                return self._store_fit(
                    combined.shape[1], (combined, metric), _csr_graph(components), k, None,
                    transform_cutoff=self._transform_cutoff, exact_fit_input=combined,
                    fit_inverse_rows=getattr(self, "_fit_inverse_rows", None),
                )
            appended_ranks = indices[n_existing:]
            # The layout initializer ignores ranks into appended rows.
            appended_rows = np.arange(n_existing, combined.shape[0])[:, None]
            appended_ranks = np.where(appended_ranks < 0, appended_rows, appended_ranks)
            return self._update_ranked_features(
                combined, metric, n_existing, components, appended_ranks.ravel().tolist(),
                self._embedding_transform_seed(), exact_input=combined,
            )
        n_rows, n_features = combined.shape
        data = combined.ravel().tolist()
        graph_options = self._graph_options()
        n_neighbors = self.n_neighbors_
        if boolean_metric and not self.output_dens:
            seed = self._embedding_transform_seed()
            n_existing = training.shape[0]
            components, ranks = _native._fit_dense_update_graph(
                data, combined.shape, n_existing, n_neighbors, metric, graph_options=graph_options,
            )
            return self._update_ranked_features(combined, metric, n_existing, components, ranks, seed)
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
            fit_inverse_rows=getattr(self, "_fit_inverse_rows", None),
        )

    def _update_sparse_features(self, X, training, metric, ensure_all_finite):
        if issparse(X):
            batch = _check_sparse_input(X, input_name="sparse features", ensure_all_finite=ensure_all_finite)
        else:
            batch = check_array(X, dtype=np.float32, order="C", ensure_all_finite=ensure_all_finite)
        snapshot = self._exact_fit_input
        retained = snapshot if snapshot is not None and snapshot.shape == training.shape else training
        stacked = vstack((retained, batch))
        combined = _sparse_features(stacked, preserve_duplicates=True)
        ranking_input = combined
        exact_input = stacked
        if (metric in ("manhattan", "chebyshev") + _RATIO_METRICS + _SPARSE_BOOLEAN_METRICS
                and not combined.has_canonical_format):
            ranking_input = _sparse_features(combined, preserve_duplicates=False)
            if metric == "manhattan":
                # Copies of either representation take the held-out path.
                exact_input = None
        seed = self._embedding_transform_seed()
        k = self.n_neighbors_
        n_existing = training.shape[0]
        components, appended_ranks = _native._fit_sparse_update_graph(
            _csr_buffers(ranking_input), n_existing, k, metric, graph_options=self._graph_options(),
        )
        return self._update_ranked_features(
            combined, metric, n_existing, components, appended_ranks, seed, exact_input=exact_input,
        )

    def _update_ranked_features(
        self, combined, metric, n_existing, components, appended_ranks, seed, *, exact_input=None,
    ):
        # The combined graph and training rows are visible before layout starts.
        self.graph_ = _csr_graph(components)
        self._graph_transform_fit = (combined, metric)
        fitted = self.embedding_
        epochs = self.n_epochs
        negative_epochs = metric != "euclidean" and isinstance(epochs, numbers.Integral) and epochs < 0
        floating_packed_epochs = (
            metric in _PACKED_METRICS and isinstance(epochs, numbers.Real)
            and not isinstance(epochs, numbers.Integral)
        )
        if fitted.shape[0] == n_existing and (negative_epochs or floating_packed_epochs):
            # The public update zeros weak memberships before the schedule fails.
            weights = self.graph_.data
            if weights.size:
                sampling_epochs = epochs if epochs > 10 else _default_epochs(combined.shape[0])
                cutoff = np.float32(weights.max() / sampling_epochs)
                weights[weights < cutoff] = 0
        if epochs is not None and not isinstance(epochs, (numbers.Integral, list, tuple, np.ndarray)):
            raise TypeError(_EPOCHS_ERROR)
        n_epochs = 0 if epochs is None else _validate_epochs(epochs)
        repulsion, negative_rate = self._sampling_options()
        components, shape, coordinates = _native._update_ranked_embedding(
            components, n_existing, appended_ranks, fitted.ravel().tolist(),
            n_components=fitted.shape[1], n_epochs=n_epochs, repulsion_strength=repulsion,
            negative_sample_rate=negative_rate, transform_seed=seed,
            **self._embedding_transform_options,
        )
        embedding = np.asarray(coordinates, dtype=np.float32).reshape(shape)
        if isinstance(metric, FeatureSearch) and self.output_dens:
            self._store_density_diagnostics(combined, _csr_graph(components), embedding, metric)
        self._store_fit(
            combined.shape[1], (combined, metric), _csr_graph(components), self.n_neighbors_, embedding,
            embedding_options=self._embedding_transform_options,
            transform_cutoff=self._transform_cutoff, exact_fit_input=exact_input,
            fit_inverse_rows=getattr(self, "_fit_inverse_rows", None),
        )

    def inverse_transform(self, X):
        """Reconstruct dense features from points in the fitted layout."""

        if not hasattr(self, "_graph_transform_fit"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_sparse_data'")
        if getattr(self, "densmap", False):
            raise ValueError("Inverse transform not available for densMAP.")
        transform_fit = self._graph_transform_fit
        if transform_fit is None or transform_fit[1] == "precomputed":
            raise ValueError("Inverse transform not available for given metric.")
        training, metric = transform_fit
        configured = isinstance(metric, FeatureSearch)
        if issparse(training):
            raise ValueError("Inverse transform not available for sparse input.")
        if metric in _BOOLEAN_METRICS + _PACKED_METRICS:
            raise ValueError("Inverse transform not available for given metric.")
        if self.transform_mode == "graph":
            raise ValueError("Inverse transform not available for transform_mode = 'graph'")
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
        uses_supplied = getattr(self, "_supplied_transform_error", None) is not None
        saved_distances = getattr(self, "_inverse_distances", None)
        distances = None
        if saved_distances is not None:
            values, target = saved_distances
            distances = (values.ravel().tolist(), values.shape[1], target)
        elif configured:
            _, values, _ = metric.rank(training, self.n_neighbors_)
            distances = (values.ravel().tolist(), values.shape[1], self.n_neighbors_)
        unique_fit = getattr(self, "_inverse_unique_fit", None)
        if unique_fit is not None:
            values, inverse = unique_fit
            unique_fit = (values.ravel().tolist(), inverse.tolist())
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
            use_computed_scales=not uses_supplied,
            metric="euclidean" if configured else metric,
            cutoff=getattr(self, "_inverse_cutoff", self._transform_cutoff),
            fitted_distances=distances,
            unique_fit=unique_fit,
            metric_gradient=metric.distance_gradient if configured else None,
            rank_ties=_rank_tied_float64_distances,
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
        if self.unique:
            raise ValueError("unique is not currently available for precomputed_knn.")
        if not isinstance(indices, np.ndarray):
            raise ValueError("precomputed_knn[0] must be ndarray object.")
        if not isinstance(distances, np.ndarray):
            raise ValueError("precomputed_knn[1] must be ndarray object.")
        if indices.shape != distances.shape:
            raise ValueError("precomputed_knn[0] and precomputed_knn[1] must be numpy arrays of the same size.")
        if not _is_supplied_search_index(search_index):
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
        exact_fit_input=None, inverse_cutoff=None, inverse_distances=None, inverse_unique_fit=None,
        fit_inverse_rows=None, feature_search_index=None,
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
        self._exact_fit_input = exact_fit_input
        if exact_fit_input is None and transform_fit is not None:
            training, metric = transform_fit
            if metric == "precomputed" or not issparse(training):
                self._exact_fit_input = training
        self._supplied_transform_error = transform_error
        self._feature_search_index = feature_search_index
        self._supervised_fit = supervised
        self._transform_cutoff = transform_cutoff
        self._inverse_local_connectivity = float(self.local_connectivity)
        self._inverse_cutoff = inverse_cutoff
        self._inverse_distances = inverse_distances
        self._inverse_unique_fit = inverse_unique_fit
        self._fit_inverse_rows = fit_inverse_rows
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
        if seed is None:
            seed = 0 if n_rows == 1 else check_random_state(None)
        if not isinstance(seed, np.random.RandomState):
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                raise ValueError("random_state must be None, an integer seed, or a NumPy RandomState")
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
            _density_input(features, graph, metric),
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
            if modern and self.init in ("recursive", "auto"):
                return None
            if self.init == "spectral":
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
        if not modern and np.unique(initial, axis=0).shape[0] != n_rows:
            raise NotImplementedError("explicit init with repeated rows is not implemented")
        return initial.ravel().tolist()

    def _validate_metric(self, *, sparse):
        if callable(self.metric):
            return FeatureSearch(self.metric, self.metric_kwds)
        if not isinstance(self.metric, str):
            raise ValueError("metric is neither callable nor a recognised string")
        name = self.metric.lower()
        if name in _CONFIGURABLE_METRICS:
            return FeatureSearch(name, self.metric_kwds)
        if self.metric_kwds:
            raise TypeError(f"metric {self.metric!r} does not accept metric_kwds")
        if self.metric == "precomputed":
            return self.metric
        if self.metric in _PACKED_METRICS:
            return self.metric
        if name == "cosine" or (name == "correlation" and not sparse):
            return name
        if name in _BOOLEAN_METRICS:
            if sparse and name == "yule":
                raise ValueError("Metric yule is not supported for sparse data")
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
        target = self._continuous_target(y, n_neighbors, _METRIC_ALIASES[metric])
        return _native._fit_dense_continuous_graph(*dense_args, target, graph_options=graph_options)

    def _apply_supervision_to_graph(self, graph, y, n_rows, n_neighbors, *, feature_kind="packed features"):
        if len(y) != n_rows:
            raise ValueError(f"Length of x = {n_rows}, length of y = {len(y)}, while it must be equal.")
        metric = self.target_metric
        if metric == "categorical":
            return _native._apply_categorical_target(graph, self._categorical_labels(y), self.target_weight)
        if callable(metric):
            raise NotImplementedError(f"Callable target metrics are unavailable with {feature_kind}.")
        native_metric = _METRIC_ALIASES.get(metric, metric)
        if native_metric not in (
            "euclidean", "manhattan", "chebyshev", "hamming", "jaccard",
            "cosine", "correlation", "precomputed",
        ):
            if metric in _BOOLEAN_METRICS or metric in _METRIC_ALIASES:
                raise NotImplementedError(f"target_metric={metric!r} is unavailable with {feature_kind}")
            raise KeyError(metric)

        labels = check_array(y, dtype="numeric", ensure_2d=False)
        if labels.ndim == 1:
            labels = labels[:, None]
        if native_metric == "precomputed" and labels.shape[0] != labels.shape[1]:
            raise KeyError(metric)
        k = self._target_neighbor_count(n_neighbors)
        if k > n_rows:
            raise ValueError("target_n_neighbors exceeds the number of fitted rows")
        indices, distances = [], []
        # Target distances retain the label dtype until after ranking.
        for block in pairwise_distances_chunked(labels, metric=native_metric):
            order = np.argsort(block, axis=1, kind="stable")[:, :k]
            indices.extend(order.ravel().tolist())
            distances.extend(np.take_along_axis(block, order, axis=1).astype(np.float32).ravel().tolist())
        return _native._apply_continuous_target(graph, indices, distances, k, self.target_weight)

    def _continuous_target(self, y, n_neighbors, metric):
        labels = check_array(y, dtype=np.float32, ensure_2d=False)
        if labels.ndim == 1:
            labels = labels[:, None]
        return (
            labels.ravel().tolist(), labels.shape[1], self._target_neighbor_count(n_neighbors),
            metric, self.target_weight,
        )

    def _target_neighbor_count(self, n_neighbors):
        target_neighbors = getattr(self, "target_n_neighbors", -1)
        if not isinstance(target_neighbors, numbers.Integral) or (
            target_neighbors != -1 and target_neighbors < 2
        ):
            raise ValueError("target_n_neighbors must be -1 or an integer greater than 1")
        return n_neighbors if target_neighbors == -1 else target_neighbors

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
