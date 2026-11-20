"""Public estimator for the dense graph primitives in the native extension."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from . import _native


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


class UMAP(BaseEstimator, TransformerMixin):
    """A public, graph-only UMAP estimator.

    Dense ``transform_mode="graph"`` fitting is implemented and returns the
    fuzzy simplicial-set graph as a SciPy CSR matrix.  Layout/embedding,
    supervised fitting, sparse input, callable metrics, and out-of-sample
    transforms are intentionally not implemented yet and fail explicitly.

    With ``metric="precomputed"``, provide a square distance matrix. After
    conversion to float32, its entries must be finite and non-negative and its
    diagonal must be zero. The rows are ranked independently, so asymmetric
    distances are accepted. Each row's selected neighbors must include its
    diagonal entry, which can be excluded when there are ties at zero distance.

    The constructor keeps the commonly used public UMAP parameters needed by
    the graph contract.  Parameters related only to a future embedding stage
    are retained as configuration values where they are part of that public
    contract; they do not alter graph construction.
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
        compatibility_layout=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        transform_mode="embedding",
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

    def fit(self, X, y=None):
        """Fit the dense fuzzy graph and return this estimator.

        For one input row, the captured public behavior is an estimator with
        no ``graph_`` attribute.  Consequently ``fit_transform`` on that input
        raises the same attribute error as the public contract.
        """

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
        if y is not None:
            raise NotImplementedError(
                "supervised fitting is not implemented; omit y for graph mode"
            )
        if not isinstance(self.n_neighbors, numbers.Integral):
            raise ValueError("n_neighbors must be an integer")
        n_neighbors = int(self.n_neighbors)
        if n_neighbors <= 1:
            raise ValueError("n_neighbors must be greater than 1")

        if issparse(X):
            raise TypeError("sparse input is not supported; pass a dense array")
        # check_array is public sklearn validation.  Its default finite-value
        # check supplies the captured ``Input contains NaN.`` error while the
        # dtype argument gives float64 inputs the native f32 behavior.
        X_checked = check_array(
            X,
            dtype=np.float32,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=1,
        )
        X_checked = np.ascontiguousarray(X_checked, dtype=np.float32)
        n_rows, n_features = X_checked.shape
        precomputed = self.metric == "precomputed"
        if precomputed:
            if n_rows != n_features:
                raise ValueError("precomputed distances must be a square matrix")
            warnings.warn(_PRECOMPUTED_WARNING, UserWarning, stacklevel=2)

        # A singleton is accepted by fit but has no graph in the public
        # behavior captured for this project.  Remove a stale graph when a
        # fitted estimator is reused so this branch remains truthful.
        if n_rows == 1:
            if precomputed:
                _native._fit_precomputed_graph([X_checked.item()], 1, 1)
            self.n_features_in_ = n_features
            if hasattr(self, "graph_"):
                del self.graph_
            if hasattr(self, "n_neighbors_"):
                del self.n_neighbors_
            return self

        if n_neighbors > n_rows - 1:
            warnings.warn(_TRUNCATION_WARNING, UserWarning, stacklevel=2)
            n_neighbors = n_rows - 1

        values = X_checked.ravel(order="C").tolist()
        if precomputed:
            shape, indptr, indices, data = _native._fit_precomputed_graph(
                values, n_rows, n_neighbors
            )
        else:
            shape, indptr, indices, data = _native._fit_dense_graph(
                values, n_rows, n_features, n_neighbors, _METRIC_ALIASES[self.metric.lower()]
            )
        graph = csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=tuple(shape),
            dtype=np.float32,
        )
        self.n_features_in_ = n_features
        self.graph_ = graph
        self.n_neighbors_ = n_neighbors

        # The tiny two-row case has no edges after self-neighbors are removed;
        # preserve the public warning text for this disconnected graph.
        if graph.nnz == 0:
            warnings.warn(
                _DISCONNECTION_WARNING.format(count=n_rows),
                UserWarning,
                stacklevel=2,
            )
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the exact ``graph_`` CSR object."""

        if fit_params:
            raise NotImplementedError(
                "fit parameters are not implemented for the graph-only path"
            )
        self.fit(X, y=y)
        # Deliberately access the public fitted attribute directly: on a
        # singleton this reproduces the captured AttributeError behavior.
        return self.graph_

    def transform(self, X):
        """Reject out-of-sample transforms until embedding support exists."""

        raise NotImplementedError(
            "transform is not implemented; only dense graph fit_transform is supported"
        )

    def inverse_transform(self, X):
        """Reject inverse transforms until embedding support exists."""

        raise NotImplementedError(
            "inverse_transform is not implemented for the graph-only estimator"
        )

    def _validate_mode(self):
        if self.transform_mode != "graph":
            raise NotImplementedError(
                "only transform_mode='graph' is implemented; embedding mode is not supported"
            )

    def _validate_metric(self):
        if callable(self.metric):
            raise NotImplementedError(
                "callable metrics are not implemented by the dense graph path"
            )
        if not isinstance(self.metric, str) or (
            self.metric != "precomputed" and self.metric.lower() not in _METRIC_ALIASES
        ):
            raise ValueError("metric is neither callable nor a recognised string")


__all__ = ["UMAP"]
