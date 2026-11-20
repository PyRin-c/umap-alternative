//! Dense compatibility-path graph fitting.
//!
//! This module is intentionally limited to the graph stage: exhaustive
//! neighbor search followed by fuzzy simplicial-set construction. It does not
//! expose an estimator, layout, optimization, or any other sklearn behavior.

use std::fmt;

use crate::fuzzy_graph::{CsrGraph, GraphError, fuzzy_simplicial_set};
use crate::neighbors::{DistanceMetric, KnnError, exhaustive_knn_with_metric};

/// Errors returned while fitting a dense fuzzy graph.
///
/// The underlying primitive errors are retained so shape, `k`, and value
/// validation have the same semantics as [`exhaustive_knn_with_metric`] and
/// [`fuzzy_simplicial_set`].
#[derive(Clone, Debug, PartialEq)]
pub enum DenseGraphFitError {
    /// Exhaustive nearest-neighbor search rejected the matrix or `k`.
    Knn(KnnError),
    /// Fuzzy simplicial-set construction rejected the neighbor rows.
    Graph(GraphError),
}

impl fmt::Display for DenseGraphFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Knn(error) => write!(formatter, "dense graph neighbor search failed: {error}"),
            Self::Graph(error) => {
                write!(formatter, "dense fuzzy graph construction failed: {error}")
            }
        }
    }
}

impl std::error::Error for DenseGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Knn(error) => Some(error),
            Self::Graph(error) => Some(error),
        }
    }
}

impl From<KnnError> for DenseGraphFitError {
    fn from(error: KnnError) -> Self {
        Self::Knn(error)
    }
}

impl From<GraphError> for DenseGraphFitError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

/// Fit a fuzzy graph from a dense row-major matrix.
///
/// `data` must contain exactly `n_rows * n_features` finite `f32` values. The
/// requested `k` is passed unchanged to exhaustive search and therefore must
/// be in `1..=n_rows`; this graph-only path does not add or remove neighbors.
/// The selected `metric` determines the distances used for neighbor ranking.
/// The returned graph is canonical CSR with strictly ascending columns in
/// every row and no diagonal entries.
///
/// This function covers only dense neighbor search and fuzzy graph fitting. It
/// does not implement a complete sklearn-compatible UMAP estimator or any
/// downstream layout/optimization stage.
pub fn fit_dense_graph(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<CsrGraph, DenseGraphFitError> {
    let neighbors = exhaustive_knn_with_metric(data, n_rows, n_features, k, metric)?;
    Ok(fuzzy_simplicial_set(&neighbors)?)
}

#[cfg(test)]
mod tests {
    use super::{DenseGraphFitError, fit_dense_graph};
    use crate::fuzzy_graph::{CsrGraph, fuzzy_simplicial_set};
    use crate::neighbors::{DistanceMetric, KnnError, exhaustive_knn_with_metric};

    const TIES_DATA: [f32; 16] = [
        0.0, 0.0, // 0
        0.0, 0.0, // 1
        1.0, 0.0, // 2
        1.0, 0.0, // 3
        0.0, 1.0, // 4
        0.0, 1.0, // 5
        1.0, 1.0, // 6
        1.0, 1.0, // 7
    ];

    #[test]
    fn fits_ties_fixture_as_canonical_unit_weight_csr() {
        let graph = fit_dense_graph(&TIES_DATA, 8, 2, 4, DistanceMetric::Euclidean)
            .expect("ties fixture must fit");

        assert_eq!(
            graph,
            CsrGraph {
                n_rows: 8,
                n_cols: 8,
                indptr: vec![0, 5, 10, 15, 20, 23, 26, 29, 32],
                indices: vec![
                    1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 6, 7, 0, 1, 2, 6, 7, 0, 1, 5, 0, 1, 4,
                    2, 3, 7, 2, 3, 6,
                ],
                data: vec![1.0; 32],
            }
        );
        for row in 0..graph.n_rows {
            let columns = &graph.indices[graph.indptr[row]..graph.indptr[row + 1]];
            assert!(columns.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(!columns.contains(&row));
        }
    }

    #[test]
    fn composes_the_selected_metric_without_changing_neighbor_count() {
        let data = [0.0, 0.0, 4.0, 1.0, 3.0, 3.0];
        let expected = fuzzy_simplicial_set(
            &exhaustive_knn_with_metric(&data, 3, 2, 3, DistanceMetric::Manhattan)
                .expect("valid Manhattan neighbors"),
        )
        .expect("valid Manhattan graph");

        assert_eq!(
            fit_dense_graph(&data, 3, 2, 3, DistanceMetric::Manhattan).expect("valid dense graph"),
            expected
        );
    }

    #[test]
    fn preserves_shape_k_and_non_finite_validation_errors() {
        assert_eq!(
            fit_dense_graph(&[0.0, 1.0], 2, 2, 1, DistanceMetric::Euclidean),
            Err(DenseGraphFitError::Knn(KnnError::InvalidShape {
                n_rows: 2,
                n_features: 2,
                values: 2,
            }))
        );
        assert_eq!(
            fit_dense_graph(&[0.0, 1.0], 2, 1, 3, DistanceMetric::Euclidean),
            Err(DenseGraphFitError::Knn(KnnError::InvalidK {
                k: 3,
                n_rows: 2
            }))
        );
        assert!(matches!(
            fit_dense_graph(&[0.0, f32::NAN], 1, 2, 1, DistanceMetric::Euclidean),
            Err(DenseGraphFitError::Knn(KnnError::NonFiniteValue { index: 1, value }))
                if value.is_nan()
        ));
    }
}
