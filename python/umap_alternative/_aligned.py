"""Joint embeddings for a sequence of related fuzzy graphs."""

from math import isfinite
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.linalg import svd
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from . import _native
from ._curve import _finite_float, _resolve_curve_params
from ._spectral import _disconnected_spectral_coordinates, _spectral_eigenvectors
from ._umap import UMAP, _COMPATIBILITY_WARNING, _default_epochs, _validate_epochs


_MISSING = object()
_MISSING_RELATIONS = "Aligned UMAP requires relations between data to be specified"
_PER_SLICE = (
    "n_neighbors", "n_epochs", "min_dist", "spread", "learning_rate", "repulsion_strength",
    "negative_sample_rate", "local_connectivity", "set_op_mix_ratio",
)


def _parameter_at(value, index):
    return value[index] if isinstance(value, (list, tuple)) else value


def _relations(links, source_size, target_size):
    if not isinstance(links, dict):
        raise TypeError("each relation must be a dictionary of row indices")
    if not links:
        raise ValueError("max() iterable argument is empty")
    result = {}
    for source, target in links.items():
        if not isinstance(source, Integral) or not isinstance(target, Integral):
            raise TypeError("relations must map integer row indices")
        if 0 <= source < source_size and 0 <= target < target_size:
            result[int(source)] = int(target)
    return result


def _backward(links):
    return {target: source for source, target in links.items()}


def _graph_components(graph):
    return graph.shape, graph.indptr.tolist(), graph.indices.tolist(), graph.data.tolist()


def _initial_layout(mapper, n_components, rng):
    graph = mapper.graph_.copy()
    graph.eliminate_zeros()
    count, labels = connected_components(graph, directed=False)
    if count == 1:
        coordinates = _spectral_eigenvectors(graph, n_components, rng)
    else:
        features, metric = mapper._graph_transform_fit
        coordinates = _disconnected_spectral_coordinates(
            graph, features, metric, n_components, rng, labels,
        )
    return (coordinates * (10.0 / np.abs(coordinates).max())).astype(np.float32)


class AlignedUMAP(BaseEstimator):
    """Jointly fit float32 embeddings for related dense datasets.

    Pass one dictionary of row indices per adjacent pair to ``fit`` or
    ``fit_transform``. For each dictionary, the keys index the earlier
    dataset and the values index the later one. ``update`` appends a new
    dataset using one dictionary and preserves the fitted coordinates of
    earlier slices. ``mappers_`` exposes each slice's fitted fuzzy graph.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        alignment_regularisation=0.01,
        alignment_window_size=3,
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        recursive_coarsening_ratio=4,
        verbose=False,
        unique=False,
        nearest_neighbor_backend="auto",
        compatibility_layout=None,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.alignment_regularisation = alignment_regularisation
        self.alignment_window_size = alignment_window_size
        self.min_dist = min_dist
        self.spread = spread
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.a = a
        self.b = b
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.recursive_coarsening_ratio = recursive_coarsening_ratio
        self.verbose = verbose
        self.unique = unique
        self.nearest_neighbor_backend = nearest_neighbor_backend
        self.compatibility_layout = compatibility_layout

    def _validate_settings(self):
        if not isinstance(self.n_components, Real) or isinstance(self.n_components, (bool, np.bool_)):
            raise ValueError("n_components must be a single integer, and cannot vary")
        if not isfinite(self.n_components) or self.n_components <= 0 or int(self.n_components) != self.n_components:
            raise ValueError("n_components must be a positive integer")
        if not isinstance(self.alignment_window_size, Integral):
            raise TypeError("alignment_window_size must be an integer")
        if self.alignment_window_size < 0:
            raise ValueError("negative dimensions are not allowed")
        _finite_float(self.alignment_regularisation, "alignment_regularisation")
        check_random_state(self.random_state)
        np.random.RandomState(self.transform_seed)
        if self.compatibility_layout is not None and not isinstance(
            self.compatibility_layout, (bool, np.bool_),
        ):
            raise ValueError("compatibility_layout must be a boolean or None")
        if self.compatibility_layout is not None and not self.compatibility_layout:
            raise NotImplementedError("AlignedUMAP requires the compatibility layout")
        if self.compatibility_layout is None:
            warnings.warn(_COMPATIBILITY_WARNING, FutureWarning, stacklevel=3)
        if (
            self.target_metric_kwds or self.unique or self.angular_rp_forest
            or self.nearest_neighbor_backend != "auto" or self.target_n_neighbors != -1
            or self.metric == "precomputed"
        ):
            raise NotImplementedError("graph options outside the dense compatibility path")
        if not isinstance(self.init, str) or self.init not in (
            "random", "spectral", "pca", "tswspectral", "auto",
        ):
            raise ValueError("unsupported AlignedUMAP mapper initialization")

    def _fit_mapper(self, X, y, index, overrides):
        X = check_array(X, dtype=np.float32, order="C")
        n_components = int(self.n_components)
        if n_components >= len(X):
            raise ValueError("n_components must be smaller than the number of samples in each slice")
        params = {
            name: overrides[name] if name in overrides else _parameter_at(getattr(self, name), index)
            for name in _PER_SLICE
        }
        if params["n_epochs"] is not None:
            params["n_epochs"] = _validate_epochs(params["n_epochs"])
        _resolve_curve_params(params["spread"], params["min_dist"], self.a, self.b)
        if _finite_float(params["learning_rate"], "learning_rate") < 0:
            raise ValueError("learning_rate must be nonnegative")
        if _finite_float(params["repulsion_strength"], "repulsion_strength") < 0:
            raise ValueError("repulsion_strength must be nonnegative")
        if _finite_float(params["negative_sample_rate"], "negative_sample_rate") <= 0:
            raise ValueError("negative_sample_rate must be positive")
        mapper = UMAP(
            **params, n_components=n_components, metric=self.metric, init=self.init,
            metric_kwds=self.metric_kwds, force_approximation_algorithm=self.force_approximation_algorithm,
            random_state=self.random_state, a=self.a, b=self.b,
            compatibility_layout=self.compatibility_layout, target_metric=self.target_metric,
            target_weight=self.target_weight, transform_mode="graph", transform_seed=self.transform_seed,
        )
        if self.random_state is not None:
            warnings.warn(
                "n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.",
                UserWarning, stacklevel=3,
            )
        mapper.fit(X, y)
        graph = mapper.graph_
        if graph.nnz == 0:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        epochs = params["n_epochs"]
        sampling_epochs = epochs if epochs is not None and epochs > 10 else _default_epochs(len(X))
        cutoff = graph.data.max() / sampling_epochs
        graph.data[graph.data < cutoff] = 0
        return mapper

    def _optimize(self, mappers, embeddings, links, n_epochs, first_moving, rng):
        fallback = None
        if n_epochs > 0:
            seed = rng.randint(-2147483647, 2147483646, size=3, dtype=np.int64)
            arrays, (words, cursor) = _native._optimize_aligned_layout(
                [_graph_components(mapper.graph_) for mapper in mappers],
                [layout.ravel(order="C").tolist() for layout in embeddings],
                [list(relation.items()) for relation in links],
                int(self.n_components), n_epochs, first_moving,
                int(self.alignment_window_size), float(self.alignment_regularisation), seed.tolist(),
            )
            embeddings[first_moving:] = [
                np.asarray(array, dtype=np.float32).reshape(-1, int(self.n_components))
                for array in arrays
            ]
            fallback = np.random.RandomState(0)
            fallback.set_state(("MT19937", np.asarray(words, dtype=np.uint32), cursor, 0, 0.0))
        for mapper, layout in zip(mappers[first_moving:], embeddings[first_moving:], strict=True):
            layout[np.asarray(mapper.graph_.sum(axis=1)).ravel() == 0] = np.nan
        return fallback

    def fit(self, X, y=None, *, relations=_MISSING):
        """Fit each graph and jointly optimize the aligned layouts."""
        if relations is _MISSING:
            raise ValueError(_MISSING_RELATIONS)
        if not isinstance(relations, (list, tuple)) or len(relations) != len(X) - 1:
            raise AssertionError
        if y is not None and len(y) != len(X):
            raise AssertionError
        self._validate_settings()
        mappers = [
            self._fit_mapper(matrix, None if y is None else y[index], index, {})
            for index, matrix in enumerate(X)
        ]
        self.mappers_ = mappers
        self.n_models_ = len(mappers)
        if not relations:
            raise ValueError("max() iterable argument is empty")
        links = [
            _relations(relation, left.graph_.shape[0], right.graph_.shape[0])
            for relation, left, right in zip(relations, mappers, mappers[1:])
        ]
        n_components = int(self.n_components)
        rng = np.random.RandomState(self.transform_seed)
        embeddings = [_initial_layout(mappers[0], n_components, rng)]
        for mapper, relation in zip(mappers[1:], links, strict=True):
            current = _initial_layout(mapper, n_components, rng)
            if relation and self.alignment_window_size:
                backward = _backward(relation)
                destinations = sorted(backward)
                sources = [backward[destination] for destination in destinations]
                covariance = current[destinations].T @ embeddings[-1][sources]
                u, _, vt = svd(covariance, lapack_driver="gesdd")
                current = current @ (u @ vt)
            embeddings.append(current)
        n_epochs = max(200 if mapper.n_epochs is None else mapper.n_epochs for mapper in mappers)
        fallback = self._optimize(mappers, embeddings, links, n_epochs, 0, rng)
        self.embeddings_ = embeddings
        self._links = links
        self._fallback_rng = fallback if fallback is not None else np.random.RandomState()
        self._fitted_mappers = mappers
        self._fitted_n_components = n_components
        return self

    def fit_transform(self, X, y=None, *, relations=_MISSING):
        """Fit and return the same list of arrays exposed as ``embeddings_``."""
        return self.fit(X, y, relations=relations).embeddings_

    def update(self, X, y=None, *, relations=_MISSING, **params):
        """Append a dataset while keeping all fitted slices fixed."""
        if relations is _MISSING:
            raise ValueError(_MISSING_RELATIONS)
        if not isinstance(relations, dict):
            raise AssertionError
        # Public state shows the graph of an attempted append even when its
        # relations fail. Use the last successful fit as the base of a retry.
        if not hasattr(self, "n_models_"):
            raise AttributeError("'AlignedUMAP' object has no attribute 'n_models_'")
        if not hasattr(self, "_fitted_mappers"):
            raise ValueError("AlignedUMAP requires a successful fit before update")
        mappers = self._fitted_mappers
        index = len(mappers)
        if self._fitted_n_components != self.n_components:
            raise ValueError("n_components must stay constant when updating AlignedUMAP")
        for name in params:
            if name not in _PER_SLICE:
                raise TypeError(f"unsupported AlignedUMAP update parameter {name!r}")
        mapper = self._fit_mapper(X, y, index, params)
        self.mappers_ = mappers + [mapper]
        self.n_models_ = index + 1
        graph = mapper.graph_
        relation = _relations(relations, mappers[-1].graph_.shape[0], graph.shape[0])
        previous = self.embeddings_[-1]
        backward = {
            row: source for row, source in _backward(relation).items()
            if np.isfinite(previous[source]).all()
        }
        current = np.empty((graph.shape[0], int(self.n_components)), dtype=np.float32)
        for destination, source in backward.items():
            current[destination] = previous[source]
        n_epochs = 200 if mapper.n_epochs is None else mapper.n_epochs
        rng = np.random.RandomState(0)
        rng.set_state(self._fallback_rng.get_state())
        for row in range(len(current)):
            if row in backward:
                continue
            start, end = graph.indptr[row:row + 2]
            weights = 0.0
            total = np.zeros(current.shape[1], dtype=np.float32)
            for neighbor, weight in zip(graph.indices[start:end], graph.data[start:end]):
                if neighbor in backward and weight > 0:
                    total += weight * previous[backward[neighbor]]
                    weights += float(weight)
            current[row] = (
                total.astype(np.float64) / weights if weights
                else rng.uniform(-10.0, 10.0, current.shape[1])
            )
        layouts = [*self.embeddings_, current]
        links = [*self._links, relation]
        fallback = self._optimize(
            self.mappers_, layouts, links, n_epochs, index, np.random.RandomState(self.transform_seed),
        )
        self.embeddings_.append(layouts[-1])
        self._fallback_rng = fallback if fallback is not None else rng
        self._fitted_mappers = self.mappers_
        self._links = links
        for name in params:
            setattr(self, name, tuple(getattr(model, name) for model in self.mappers_))
