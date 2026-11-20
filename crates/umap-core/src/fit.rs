//! Dense compatibility-path graph fitting.
//!
//! This module is intentionally limited to the graph stage: exhaustive or
//! distance-matrix neighbor search, supplied neighbor arrays, fuzzy
//! simplicial-set construction, and optional categorical supervision.
//! It does not expose an estimator, layout, optimization, or any other
//! sklearn behavior.
//!
//! The precomputed path is the first raw core layer only: it accepts a flat
//! `f32` square distance matrix and leaves Python-facing input conversion and
//! boundary behavior to a later layer.

use std::fmt;

use crate::fuzzy_graph::supplied_neighbors::build_ranked_graph;
use crate::fuzzy_graph::{
    CsrGraph, GraphError, GraphOptions, categorical_intersection, fuzzy_simplicial_set,
    fuzzy_simplicial_set_with_options,
};
use crate::neighbors::{
    DistanceMetric, KnnError, Neighbor, PrecomputedKnnError, dense_knn_counting_cutoff,
    exhaustive_knn_with_metric, precomputed_knn,
};

pub use crate::fuzzy_graph::supplied_neighbors::{
    SuppliedNeighborsGraphError, fit_supplied_neighbors_graph,
};

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

impl From<PrecomputedKnnError> for DenseGraphFitError {
    fn from(error: PrecomputedKnnError) -> Self {
        match error {
            PrecomputedKnnError::Knn(error) => Self::Knn(error),
            PrecomputedKnnError::Graph(error) => Self::Graph(error),
        }
    }
}

/// Fit a fuzzy graph from a dense row-major matrix.
///
/// `data` must contain exactly `n_rows * n_features` finite `f32` values. The
/// requested `k` is passed unchanged to exhaustive search and therefore must
/// be in `1..=n_rows`. For cosine and correlation, the graph omits distances
/// of two or more and accepts ranked rows that omit self at a tie.
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
    fit_dense_graph_with_options(data, n_rows, n_features, k, metric, None)
}

/// Fit a dense fuzzy graph with explicit graph settings when provided.
///
/// The neighbors follow [`fit_dense_graph`]. Cosine and correlation use the
/// given options or [`GraphOptions::default`]. For the other metrics, passing
/// `Some(options)` uses [`fuzzy_simplicial_set_with_options`] and passing
/// `None` uses [`fuzzy_simplicial_set`].
pub fn fit_dense_graph_with_options(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    options: Option<GraphOptions>,
) -> Result<CsrGraph, DenseGraphFitError> {
    if let Some(cutoff) = metric.default_disconnection_distance() {
        return fit_dense_graph_with_cutoff(
            data,
            n_rows,
            n_features,
            k,
            metric,
            cutoff,
            options.unwrap_or_default(),
        )
        .map(|(graph, _)| graph);
    }
    let neighbors = exhaustive_knn_with_metric(data, n_rows, n_features, k, metric)?;
    Ok(build_fitted_graph(&neighbors, options)?)
}

/// Fit from ranked dense neighbors with an inclusive distance cutoff.
///
/// Returns canonical CSR and the number of excluded entries across all ordered
/// row pairs. Cut neighbors remain empty slots for the scale target. Rounded
/// angular distances can precede the diagonal, and an equal-distance cutoff
/// can exclude the diagonal from the ranked list. The diagonal never emits
/// a membership.
pub fn fit_dense_graph_with_cutoff(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
    options: GraphOptions,
) -> Result<(CsrGraph, usize), DenseGraphFitError> {
    let (neighbors, removed) =
        dense_knn_counting_cutoff(data, n_rows, n_features, k, metric, cutoff)?;
    let mut distances = Vec::with_capacity(n_rows * k);
    for (row, neighbors) in neighbors.iter().enumerate() {
        for (position, &Neighbor { index, distance }) in neighbors.iter().enumerate() {
            if distance >= cutoff {
                distances.push(f32::INFINITY);
            } else if distance.is_finite() {
                distances.push(distance);
            } else {
                return Err(GraphError::InvalidDistance {
                    row,
                    position,
                    index,
                    distance,
                }
                .into());
            }
        }
    }
    let indices = neighbors.iter().flatten().map(|neighbor| neighbor.index);
    Ok((
        build_ranked_graph(indices, &distances, n_rows, k, k, options),
        removed,
    ))
}

/// Errors returned while fitting a dense categorical fuzzy graph.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseCategoricalGraphFitError {
    /// The target slice must contain one label per input row.
    LabelCount { n_rows: usize, labels: usize },
    /// The target weight must be finite and in `0..=1`.
    InvalidTargetWeight { target_weight: f64 },
    /// Dense neighbor search or fuzzy graph construction failed.
    DenseGraph(DenseGraphFitError),
}

impl fmt::Display for DenseCategoricalGraphFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LabelCount { n_rows, labels } => {
                write!(
                    formatter,
                    "expected {n_rows} categorical labels, got {labels}"
                )
            }
            Self::InvalidTargetWeight { target_weight } => {
                write!(formatter, "target weight {target_weight} must be in 0..=1")
            }
            Self::DenseGraph(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DenseCategoricalGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DenseGraph(error) => Some(error),
            _ => None,
        }
    }
}

/// Fit a dense fuzzy graph with categorical or partially labeled targets.
///
/// The dense input, neighbor count, and feature metric follow
/// [`fit_dense_graph`]. Supply one `i64` label per input row. A label of `-1`
/// indicates an unlabeled row. All other values identify classes by equality.
/// `target_weight` must be finite and in `0..=1`.
///
/// Edges incident to any unlabeled row are scaled by `exp(-1)`. Edges between
/// different known classes are scaled by `exp(-2.5 / (1 - target_weight))`,
/// or removed at weight one. Edges between equal known classes keep their
/// membership at this stage. The adjusted memberships are rounded to `f32`,
/// divided by their row maxima, and symmetrized by fuzzy union. The result is
/// canonical CSR. It omits the diagonal and zero entries, and represents
/// isolated rows as empty CSR rows.
///
/// This adjustment also runs at target weight zero and when all rows are
/// unlabeled. Use [`fit_dense_graph`] when no categorical targets are supplied.
///
/// # Errors
///
/// Returns [`DenseCategoricalGraphFitError::LabelCount`] for a label count
/// mismatch or [`DenseCategoricalGraphFitError::InvalidTargetWeight`] for an
/// invalid weight. Dense input errors are wrapped in
/// [`DenseCategoricalGraphFitError::DenseGraph`].
pub fn fit_dense_categorical_graph(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    labels: &[i64],
    target_weight: f64,
) -> Result<CsrGraph, DenseCategoricalGraphFitError> {
    fit_dense_categorical_graph_with_options(
        data,
        n_rows,
        n_features,
        k,
        metric,
        labels,
        target_weight,
        None,
    )
}

/// Fit a dense categorical graph, applying the settings before target weighting.
///
/// The features and targets follow [`fit_dense_categorical_graph`]. The optional
/// graph settings follow [`fit_dense_graph_with_options`].
#[allow(clippy::too_many_arguments)]
pub fn fit_dense_categorical_graph_with_options(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    labels: &[i64],
    target_weight: f64,
    options: Option<GraphOptions>,
) -> Result<CsrGraph, DenseCategoricalGraphFitError> {
    if labels.len() != n_rows {
        return Err(DenseCategoricalGraphFitError::LabelCount {
            n_rows,
            labels: labels.len(),
        });
    }
    if !(0.0..=1.0).contains(&target_weight) {
        return Err(DenseCategoricalGraphFitError::InvalidTargetWeight { target_weight });
    }

    let graph = fit_dense_graph_with_options(data, n_rows, n_features, k, metric, options)
        .map_err(DenseCategoricalGraphFitError::DenseGraph)?;
    Ok(categorical_intersection(graph, labels, target_weight))
}

/// Fit a fuzzy graph from a precomputed row-major square distance matrix.
///
/// `distances` must contain exactly `n_rows * n_rows` finite, non-negative
/// `f32` values, with an exactly zero diagonal. Each row is ranked as supplied;
/// the matrix may be asymmetric, and this function does not infer or enforce
/// symmetry. The requested `k` is passed unchanged to the neighbor primitive
/// and therefore must be in `1..=n_rows`.
///
/// This is the first raw core precomputed-distance scope. It returns the same
/// canonical CSR graph as the dense metric path, while Python-facing boundary
/// conversion and estimator behavior remain outside this API.
pub fn fit_precomputed_graph(
    distances: &[f32],
    n_rows: usize,
    k: usize,
) -> Result<CsrGraph, DenseGraphFitError> {
    fit_precomputed_graph_with_options(distances, n_rows, k, None)
}

/// Fit a precomputed-distance graph with optional fuzzy graph settings.
///
/// The distances follow [`fit_precomputed_graph`]. The optional graph settings
/// follow [`fit_dense_graph_with_options`].
pub fn fit_precomputed_graph_with_options(
    distances: &[f32],
    n_rows: usize,
    k: usize,
    options: Option<GraphOptions>,
) -> Result<CsrGraph, DenseGraphFitError> {
    let neighbors = precomputed_knn(distances, n_rows, k).map_err(DenseGraphFitError::from)?;
    Ok(build_fitted_graph(&neighbors, options)?)
}

/// Fit from precomputed distances with an inclusive disconnection cutoff.
///
/// Ranking uses the finite square input before masking. The graph retains the
/// requested neighbor count as the scale target even when candidates are cut.
/// Returns the graph and the number of cut positions in the input matrix.
pub fn fit_precomputed_graph_with_cutoff(
    distances: &[f32],
    n_rows: usize,
    k: usize,
    cutoff: f32,
    options: GraphOptions,
) -> Result<(CsrGraph, usize), DenseGraphFitError> {
    let neighbors = precomputed_knn(distances, n_rows, k)?;
    let selected = neighbors
        .iter()
        .flatten()
        .map(|neighbor| {
            if neighbor.distance >= cutoff {
                f32::INFINITY
            } else {
                neighbor.distance
            }
        })
        .collect::<Vec<_>>();
    let indices = neighbors.iter().flatten().map(|neighbor| neighbor.index);
    let graph = build_ranked_graph(indices, &selected, n_rows, k, k, options);
    let removed = distances
        .iter()
        .filter(|&&distance| distance >= cutoff)
        .count();
    Ok((graph, removed))
}

pub(crate) fn build_fitted_graph(
    neighbors: &[Vec<Neighbor>],
    options: Option<GraphOptions>,
) -> Result<CsrGraph, GraphError> {
    match options {
        Some(options) => fuzzy_simplicial_set_with_options(neighbors, options),
        None => fuzzy_simplicial_set(neighbors),
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseGraphFitError, fit_dense_graph, fit_precomputed_graph};
    use crate::fuzzy_graph::{CsrGraph, fuzzy_simplicial_set};
    use crate::neighbors::{DistanceMetric, KnnError, exhaustive_knn_with_metric, precomputed_knn};

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

    #[test]
    fn fits_precomputed_asymmetric_rows_without_symmetrizing_support() {
        let distances = [
            0.0, 5.0, 1.0, // row 0
            2.0, 0.0, 4.0, // row 1
            3.0, 1.0, 0.0, // row 2
        ];
        let expected = fuzzy_simplicial_set(
            &precomputed_knn(&distances, 3, 2).expect("valid asymmetric distances"),
        )
        .expect("valid precomputed neighbors");

        assert_eq!(
            fit_precomputed_graph(&distances, 3, 2).expect("valid precomputed graph"),
            expected
        );
        assert_eq!(expected.indptr, vec![0, 2, 4, 6]);
        assert_eq!(expected.indices, vec![1, 2, 0, 2, 0, 1]);
    }

    #[test]
    fn fit_precomputed_preserves_distance_validation_errors() {
        assert_eq!(
            fit_precomputed_graph(&[0.0, 1.0, 2.0], 2, 1),
            Err(DenseGraphFitError::Knn(KnnError::InvalidShape {
                n_rows: 2,
                n_features: 2,
                values: 3,
            }))
        );
        assert_eq!(
            fit_precomputed_graph(&[0.0, -1.0, 1.0, 0.0], 2, 2),
            Err(DenseGraphFitError::Graph(
                crate::fuzzy_graph::GraphError::InvalidDistance {
                    row: 0,
                    position: 1,
                    index: 1,
                    distance: -1.0,
                },
            ))
        );
        assert_eq!(
            fit_precomputed_graph(&[0.0, 1.0, 2.0, 0.5], 2, 2),
            Err(DenseGraphFitError::Graph(
                crate::fuzzy_graph::GraphError::InvalidSelfDistance {
                    row: 1,
                    position: 1,
                    distance: 0.5,
                },
            ))
        );
    }
}
