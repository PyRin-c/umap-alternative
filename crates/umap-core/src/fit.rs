//! Dense compatibility-path graph fitting.
//!
//! This module is intentionally limited to the graph stage: exhaustive or
//! distance-matrix neighbor search, supplied neighbor arrays, fuzzy
//! simplicial-set construction, and optional supervision.
//! It does not expose an estimator, layout, optimization, or any other
//! sklearn behavior.
//!
//! Precomputed fits accept a flat `f32` square distance matrix or ranked
//! neighbor buffers. The estimator handles CSR storage and rank selection.

use std::fmt;

use crate::fuzzy_graph::supplied_neighbors::build_ranked_graph;
use crate::fuzzy_graph::{
    CsrGraph, GraphError, GraphOptions, categorical_intersection, continuous_intersection,
    fuzzy_simplicial_set, fuzzy_simplicial_set_with_options,
};
use crate::layout::{LayoutError, validate_csr_rows, validate_csr_shape};
use crate::neighbors::{
    DistanceMetric, KnnError, Neighbor, PrecomputedKnnError, dense_knn_counting_cutoff,
    exhaustive_knn_with_metric, exhaustive_manhattan_target_knn, precomputed_knn,
};

pub use crate::fuzzy_graph::supplied_neighbors::{
    SuppliedNeighborsGraphError, fit_sparse_precomputed_ranked_graph, fit_supplied_neighbors_graph,
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
    if metric.uses_ranked_graph() {
        return fit_dense_graph_with_cutoff(
            data,
            n_rows,
            n_features,
            k,
            metric,
            metric
                .default_disconnection_distance()
                .unwrap_or(f32::INFINITY),
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
    let graph = build_graph_with_cutoff(&neighbors, k, cutoff, options)?;
    Ok((graph, removed))
}

pub(crate) fn build_graph_with_cutoff(
    neighbors: &[Vec<Neighbor>],
    k: usize,
    cutoff: f32,
    options: GraphOptions,
) -> Result<CsrGraph, GraphError> {
    let n_rows = neighbors.len();
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
                });
            }
        }
    }
    let indices = neighbors.iter().flatten().map(|neighbor| neighbor.index);
    Ok(build_ranked_graph(
        indices, &distances, n_rows, k, k, options,
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
    /// A supplied feature graph is invalid.
    InvalidFeatureGraph(LayoutError),
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
            Self::InvalidFeatureGraph(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DenseCategoricalGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DenseGraph(error) => Some(error),
            Self::InvalidFeatureGraph(error) => Some(error),
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
    fit_dense_categorical_graph_with_diagnostics(
        data,
        n_rows,
        n_features,
        k,
        metric,
        labels,
        target_weight,
        options,
    )
    .map(|(graph, _)| graph)
}

/// Fit a dense categorical graph and count isolated vertices before supervision.
#[allow(clippy::too_many_arguments)]
pub fn fit_dense_categorical_graph_with_diagnostics(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    labels: &[i64],
    target_weight: f64,
    options: Option<GraphOptions>,
) -> Result<(CsrGraph, usize), DenseCategoricalGraphFitError> {
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
    let isolated = count_isolated_rows(&graph);
    Ok((
        categorical_intersection(graph, labels, target_weight),
        isolated,
    ))
}

/// Apply categorical supervision to an already built feature graph.
/// A weight at or beyond one, or NaN, uses the full cross-class attenuation.
pub fn apply_categorical_target(
    graph: CsrGraph,
    labels: &[i64],
    target_weight: f64,
) -> Result<CsrGraph, DenseCategoricalGraphFitError> {
    validate_csr_shape(&graph)
        .and_then(|()| validate_csr_rows(&graph))
        .map_err(DenseCategoricalGraphFitError::InvalidFeatureGraph)?;
    if labels.len() != graph.n_rows {
        return Err(DenseCategoricalGraphFitError::LabelCount {
            n_rows: graph.n_rows,
            labels: labels.len(),
        });
    }
    let weight = if target_weight < 1.0 {
        target_weight
    } else {
        1.0
    };
    Ok(categorical_intersection(graph, labels, weight))
}

fn count_isolated_rows(graph: &CsrGraph) -> usize {
    graph
        .indptr
        .windows(2)
        .filter(|row| {
            graph.data[row[0]..row[1]]
                .iter()
                .all(|&weight| weight == 0.0)
        })
        .count()
}

/// Dense scalar or vector targets and their graph settings.
pub struct DenseContinuousTarget<'a> {
    /// Row-major target values, one row for each row in the feature matrix.
    pub values: &'a [f32],
    /// Number of target values per row.
    pub n_features: usize,
    /// Number of neighbors to retain from each target row.
    pub n_neighbors: usize,
    /// Metric for target-space neighbors.
    pub metric: DistanceMetric,
    /// Relative weight of the directed target graph, in `0..=1`.
    pub weight: f64,
}

/// Errors returned while building the feature and continuous target graphs.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseContinuousGraphFitError {
    /// The target weight must be finite and in `0..=1`.
    InvalidTargetWeight { target_weight: f64 },
    /// The target metric must be Euclidean, Manhattan, or Chebyshev.
    UnsupportedTargetMetric { metric: DistanceMetric },
    /// The feature graph could not be built.
    FeatureGraph(DenseGraphFitError),
    /// Target neighbor search or graph construction failed.
    TargetGraph(DenseGraphFitError),
    /// A supplied feature graph is invalid.
    InvalidFeatureGraph(LayoutError),
}

impl fmt::Display for DenseContinuousGraphFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTargetWeight { target_weight } => {
                write!(formatter, "target weight {target_weight} must be in 0..=1")
            }
            Self::UnsupportedTargetMetric { metric } => {
                write!(formatter, "unsupported continuous target metric {metric:?}")
            }
            Self::FeatureGraph(error) => write!(formatter, "feature graph: {error}"),
            Self::TargetGraph(error) => write!(formatter, "target graph: {error}"),
            Self::InvalidFeatureGraph(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DenseContinuousGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidTargetWeight { .. } | Self::UnsupportedTargetMetric { .. } => None,
            Self::FeatureGraph(error) | Self::TargetGraph(error) => Some(error),
            Self::InvalidFeatureGraph(error) => Some(error),
        }
    }
}

/// Fit dense features with a second fuzzy graph built from continuous targets.
///
/// Feature options apply only to the feature graph. The target graph uses
/// directed memberships with local connectivity one. The two graphs contribute
/// on the union of directed supports, including at the weight endpoints.
/// Returns the blended CSR and the number of isolated rows in the feature graph.
pub fn fit_dense_continuous_graph(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    options: Option<GraphOptions>,
    target: &DenseContinuousTarget<'_>,
) -> Result<(CsrGraph, usize), DenseContinuousGraphFitError> {
    if !matches!(
        target.metric,
        DistanceMetric::Euclidean | DistanceMetric::Manhattan | DistanceMetric::Chebyshev
    ) {
        return Err(DenseContinuousGraphFitError::UnsupportedTargetMetric {
            metric: target.metric,
        });
    }
    if !(0.0..=1.0).contains(&target.weight) {
        return Err(DenseContinuousGraphFitError::InvalidTargetWeight {
            target_weight: target.weight,
        });
    }
    let graph = fit_dense_graph_with_options(data, n_rows, n_features, k, metric, options)
        .map_err(DenseContinuousGraphFitError::FeatureGraph)?;
    let isolated = count_isolated_rows(&graph);
    Ok((
        compose_continuous_target(graph, target, isolated)?,
        isolated,
    ))
}

/// Apply row-major ranked target distances to an existing feature graph.
/// The target distance calculation can use a different input precision.
pub fn apply_continuous_target(
    graph: CsrGraph,
    columns: &[usize],
    distances: &[f32],
    k: usize,
    weight: f64,
) -> Result<CsrGraph, DenseContinuousGraphFitError> {
    if !weight.is_finite() {
        return Err(DenseContinuousGraphFitError::InvalidTargetWeight {
            target_weight: weight,
        });
    }
    validate_csr_shape(&graph)
        .and_then(|()| validate_csr_rows(&graph))
        .map_err(DenseContinuousGraphFitError::InvalidFeatureGraph)?;
    let n_rows = graph.n_rows;
    let target_graph = |error: GraphError| DenseContinuousGraphFitError::TargetGraph(error.into());
    if !(1..=n_rows).contains(&k) {
        return Err(target_graph(GraphError::InvalidK { k, n_rows }));
    }
    if n_rows.checked_mul(k) != Some(columns.len()) || columns.len() != distances.len() {
        return Err(DenseContinuousGraphFitError::TargetGraph(
            KnnError::InvalidShape {
                n_rows,
                n_features: k,
                values: distances.len(),
            }
            .into(),
        ));
    }
    let mut neighbors = Vec::with_capacity(n_rows);
    for (row, (columns, distances)) in columns
        .chunks_exact(k)
        .zip(distances.chunks_exact(k))
        .enumerate()
    {
        let mut ranks = Vec::with_capacity(k);
        for (position, (&index, &distance)) in columns.iter().zip(distances).enumerate() {
            if index >= n_rows {
                return Err(target_graph(GraphError::InvalidNeighborIndex {
                    row,
                    position,
                    index,
                    n_rows,
                }));
            }
            if !distance.is_finite() || distance < 0.0 {
                return Err(target_graph(GraphError::InvalidDistance {
                    row,
                    position,
                    index,
                    distance,
                }));
            }
            ranks.push(Neighbor { index, distance });
        }
        neighbors.push(ranks);
    }
    if count_isolated_rows(&graph) == n_rows {
        return Ok(graph);
    }
    Ok(continuous_intersection(&graph, &neighbors, weight))
}

fn compose_continuous_target(
    graph: CsrGraph,
    target: &DenseContinuousTarget<'_>,
    isolated: usize,
) -> Result<CsrGraph, DenseContinuousGraphFitError> {
    let n_rows = graph.n_rows;
    let neighbors = if target.metric == DistanceMetric::Manhattan {
        exhaustive_manhattan_target_knn(
            target.values,
            n_rows,
            target.n_features,
            target.n_neighbors,
        )
    } else {
        exhaustive_knn_with_metric(
            target.values,
            n_rows,
            target.n_features,
            target.n_neighbors,
            target.metric,
        )
    }
    .map_err(|error| DenseContinuousGraphFitError::TargetGraph(error.into()))?;
    for (row, candidates) in neighbors.iter().enumerate() {
        for (position, neighbor) in candidates.iter().enumerate() {
            if !neighbor.distance.is_finite() || neighbor.distance < 0.0 {
                let error = GraphError::InvalidDistance {
                    row,
                    position,
                    index: neighbor.index,
                    distance: neighbor.distance,
                };
                return Err(DenseContinuousGraphFitError::TargetGraph(error.into()));
            }
        }
    }
    if isolated == n_rows {
        return Ok(graph);
    }
    Ok(continuous_intersection(&graph, &neighbors, target.weight))
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
