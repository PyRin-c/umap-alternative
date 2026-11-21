//! Density diagnostics over the input graph and a graph refit on the embedding.
//!
//! The reduction uses Rust's `f32` logarithm. Its last bit can differ from
//! the logarithm used by the public reference even for identical inputs.

use std::fmt;

use crate::fuzzy_graph::{
    CsrGraph, GraphError, fuzzy_simplicial_set, fuzzy_simplicial_set_with_neighbor_target,
};
use crate::neighbors::{
    DistanceMetric, KnnError, Neighbor, collect_neighbors, euclidean_distance_wide,
};

pub(crate) const LOG_OFFSET: f32 = 1e-8;

/// Selects the distance term in the weighted radius calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadiusSpace {
    /// Use squared distances on the fitted input graph.
    Original,
    /// Use unsquared distances on a graph refit on the embedding.
    Embedding,
}

/// Row-major input used to recover distances along a fitted graph.
#[derive(Clone, Copy, Debug)]
pub enum OriginalRadiusInput<'a> {
    /// Feature rows to compare using the fitted metric.
    Dense {
        /// Feature values in row-major order.
        values: &'a [f32],
        /// Number of features per observation.
        n_features: usize,
        /// Distance used to fit the graph.
        metric: DistanceMetric,
    },
    /// Square row-major array of unsquared, possibly asymmetric distances.
    Precomputed(&'a [f32]),
    /// Unsquared distances in the training graph's CSR storage order.
    EdgeDistances(&'a [f32]),
}

/// Input or graph refit failures in the density diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub enum DensityError {
    /// The graph must be square CSR with strictly increasing columns per row.
    InvalidCsr,
    /// There must be one distance per stored CSR entry.
    InvalidDistanceCount { expected: usize, actual: usize },
    /// An edge distance is negative or non-finite.
    InvalidDistance {
        row: usize,
        column: usize,
        distance: f32,
    },
    /// A fuzzy membership is outside the finite interval `0..=1`.
    InvalidWeight { entry: usize, weight: f32 },
    /// The coordinates or requested neighbor count are invalid.
    Matrix(KnnError),
    /// The embedding neighbors could not be turned into a fuzzy graph.
    Refit(GraphError),
}

impl fmt::Display for DensityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCsr => {
                write!(formatter, "expected a square CSR graph with sorted columns")
            }
            Self::InvalidDistanceCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} CSR edge distances, got {actual}"
                )
            }
            Self::InvalidDistance {
                row,
                column,
                distance,
            } => {
                write!(
                    formatter,
                    "edge ({row}, {column}) has invalid distance {distance}"
                )
            }
            Self::InvalidWeight { entry, weight } => {
                write!(
                    formatter,
                    "CSR entry {entry} has invalid fuzzy weight {weight}"
                )
            }
            Self::Matrix(error) => write!(formatter, "invalid density coordinate matrix: {error}"),
            Self::Refit(error) => write!(formatter, "embedding graph refit failed: {error}"),
        }
    }
}

impl std::error::Error for DensityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            Self::Refit(error) => Some(error),
            _ => None,
        }
    }
}

/// Reduce unsquared distances aligned with the stored CSR entries to log radii.
///
/// Each directed entry contributes its weight and weighted distance term to
/// both endpoints, in CSR order. Reciprocal entries contribute separately.
/// Distances may be asymmetric and must be finite and non-negative, including
/// those at explicitly stored zero weights. The numerator update is evaluated
/// in `f64` and rounded to `f32` after each contribution. The weight sums and
/// diagnostic logarithm use `f32`. A row with no
/// positive incident weight has an undefined radius (`NaN`). An overflowing
/// numerator produces an infinite radius.
pub fn radii_from_edge_distances(
    graph: &CsrGraph,
    distances: &[f32],
    space: RadiusSpace,
) -> Result<Vec<f32>, DensityError> {
    radii_and_weights_from_edge_distances(graph, distances, space).map(|(radii, _)| radii)
}

pub(crate) fn radii_and_weights_from_edge_distances(
    graph: &CsrGraph,
    distances: &[f32],
    space: RadiusSpace,
) -> Result<(Vec<f32>, Vec<f32>), DensityError> {
    if distances.len() != graph.data.len() {
        return Err(DensityError::InvalidDistanceCount {
            expected: graph.data.len(),
            actual: distances.len(),
        });
    }
    let (values, weights) = reduce_radius_means(graph, space, |_, _, entry| distances[entry])?;
    Ok((
        values
            .into_iter()
            .map(|mean| (mean + LOG_OFFSET).ln())
            .collect(),
        weights,
    ))
}

/// Compute the input-space diagnostic logarithm arguments using the fitted graph.
pub fn original_radius_log_inputs(
    graph: &CsrGraph,
    input: OriginalRadiusInput<'_>,
) -> Result<Vec<f32>, DensityError> {
    original_log_inputs_and_weights(graph, input).map(|(values, _)| values)
}

pub(crate) fn original_log_inputs_and_weights(
    graph: &CsrGraph,
    input: OriginalRadiusInput<'_>,
) -> Result<(Vec<f32>, Vec<f32>), DensityError> {
    let (values, weights) = original_means_and_weights(graph, input)?;
    Ok((
        values.into_iter().map(|mean| mean + LOG_OFFSET).collect(),
        weights,
    ))
}

pub(crate) fn original_radii_and_weights(
    graph: &CsrGraph,
    input: OriginalRadiusInput<'_>,
) -> Result<(Vec<f32>, Vec<f32>), DensityError> {
    let (values, weights) = original_means_and_weights(graph, input)?;
    Ok((
        values
            .into_iter()
            .map(|mean| (f64::from(mean) + 1e-8).ln() as f32)
            .collect(),
        weights,
    ))
}

fn original_means_and_weights(
    graph: &CsrGraph,
    input: OriginalRadiusInput<'_>,
) -> Result<(Vec<f32>, Vec<f32>), DensityError> {
    let space = RadiusSpace::Original;
    match input {
        OriginalRadiusInput::Dense {
            values,
            n_features,
            metric,
        } => {
            validate_coordinates(values, graph.n_rows, n_features, space)?;
            reduce_radius_means(graph, space, |row, column, _| {
                metric.distance(
                    coordinate_row(values, n_features, row),
                    coordinate_row(values, n_features, column),
                )
            })
        }
        OriginalRadiusInput::Precomputed(distances) => {
            validate_coordinates(distances, graph.n_rows, graph.n_rows, space)?;
            reduce_radius_means(graph, space, |row, column, _| {
                distances[row * graph.n_rows + column]
            })
        }
        OriginalRadiusInput::EdgeDistances(distances) => {
            if distances.len() != graph.data.len() {
                return Err(DensityError::InvalidDistanceCount {
                    expected: graph.data.len(),
                    actual: distances.len(),
                });
            }
            reduce_radius_means(graph, space, |_, _, entry| distances[entry])
        }
    }
}

/// Compute original radii using a fitted graph and row-major Euclidean input.
///
/// The input shape is `graph.n_rows` by `n_features`. Coordinate differences
/// and their squares use `f32`. The sum and square root use `f64`, and each
/// edge distance is rounded to `f32` before the reduction. For a fitted
/// [`crate::embedding::DenseEmbedding`],
/// pass its graph and the original training data. For other metrics or
/// precomputed distances, use [`radii_from_edge_distances`].
pub fn original_radii_euclidean(
    graph: &CsrGraph,
    input: &[f32],
    n_features: usize,
) -> Result<Vec<f32>, DensityError> {
    euclidean_radii(graph, input, n_features, RadiusSpace::Original)
}

/// Compute embedding radii using a supplied refit graph and row-major layout.
///
/// The layout shape is `graph.n_rows` by `n_components`. Each coordinate
/// difference is rounded to `f32`, then its square is accumulated with a fused
/// multiply-add in feature order before an `f32` square root. This distance
/// arithmetic was verified on two-dimensional embeddings.
pub fn embedding_radii_from_refit_euclidean(
    graph: &CsrGraph,
    coordinates: &[f32],
    n_components: usize,
) -> Result<Vec<f32>, DensityError> {
    euclidean_radii(graph, coordinates, n_components, RadiusSpace::Embedding)
}

/// Refit a native fuzzy graph and compute its embedding radii.
///
/// Pass the stored coordinates and shape from a fitted
/// [`crate::embedding::DenseEmbedding`] and the neighbor count used to fit it.
/// See [`refit_embedding_graph_euclidean`] for the graph and tie policy.
pub fn embedding_radii_euclidean(
    coordinates: &[f32],
    n_rows: usize,
    n_components: usize,
    n_neighbors: usize,
) -> Result<Vec<f32>, DensityError> {
    let graph = refit_embedding_graph_euclidean(coordinates, n_rows, n_components, n_neighbors)?;
    embedding_radii_from_refit_euclidean(&graph, coordinates, n_components)
}

/// Refit a fuzzy graph on row-major embedding coordinates for density diagnostics.
///
/// This native refit uses exhaustive fused Euclidean distances and the core
/// fuzzy graph solver. `n_neighbors` includes the row itself. Among the other
/// points, ties are resolved by ascending row index. The public diagnostic
/// implementation's graph weights and tie behavior may differ from this refit.
/// Pass the returned graph to [`embedding_radii_from_refit_euclidean`].
pub fn refit_embedding_graph_euclidean(
    coordinates: &[f32],
    n_rows: usize,
    n_components: usize,
    n_neighbors: usize,
) -> Result<CsrGraph, DensityError> {
    refit_embedding_graph(coordinates, n_rows, n_components, n_neighbors, n_neighbors)
}

/// Compute log inputs after an embedding refit using the requested neighbor count.
///
/// A request wider than the layout ranks all rows and keeps the requested
/// count as the fuzzy membership target.
pub fn embedding_radius_log_inputs_euclidean(
    coordinates: &[f32],
    n_rows: usize,
    n_components: usize,
    requested_neighbors: usize,
) -> Result<Vec<f32>, DensityError> {
    let graph = refit_embedding_graph(
        coordinates,
        n_rows,
        n_components,
        requested_neighbors.min(n_rows),
        requested_neighbors,
    )?;
    euclidean_radius_log_inputs(&graph, coordinates, n_components, RadiusSpace::Embedding)
}

fn refit_embedding_graph(
    coordinates: &[f32],
    n_rows: usize,
    n_components: usize,
    n_neighbors: usize,
    requested_neighbors: usize,
) -> Result<CsrGraph, DensityError> {
    validate_coordinates(coordinates, n_rows, n_components, RadiusSpace::Embedding)?;
    if !(1..=n_rows).contains(&n_neighbors) {
        return Err(DensityError::Matrix(KnnError::InvalidK {
            k: n_neighbors,
            n_rows,
        }));
    }

    let mut invalid_distance = None;
    let neighbors = collect_neighbors(n_rows, n_neighbors, |row, candidates| {
        let point = coordinate_row(coordinates, n_components, row);
        for column in 0..n_rows {
            let distance = if row == column {
                // total_cmp puts -0.0 first, so coincident points cannot displace self.
                -0.0
            } else {
                fused_euclidean_distance(point, coordinate_row(coordinates, n_components, column))
            };
            if !distance.is_finite() && invalid_distance.is_none() {
                invalid_distance = Some(DensityError::InvalidDistance {
                    row,
                    column,
                    distance,
                });
            }
            candidates.push(Neighbor {
                index: column,
                distance,
            });
        }
    });
    if let Some(error) = invalid_distance {
        return Err(error);
    }
    if requested_neighbors > n_neighbors {
        fuzzy_simplicial_set_with_neighbor_target(&neighbors, requested_neighbors)
    } else {
        fuzzy_simplicial_set(&neighbors)
    }
    .map_err(DensityError::Refit)
}

fn reduce_radius_means(
    graph: &CsrGraph,
    space: RadiusSpace,
    distance: impl Fn(usize, usize, usize) -> f32,
) -> Result<(Vec<f32>, Vec<f32>), DensityError> {
    if graph.n_rows != graph.n_cols
        || graph.n_rows.checked_add(1) != Some(graph.indptr.len())
        || graph.indptr.first() != Some(&0)
        || graph.indptr.last() != Some(&graph.indices.len())
        || graph.indices.len() != graph.data.len()
    {
        return Err(DensityError::InvalidCsr);
    }

    let mut numerators = vec![0.0_f32; graph.n_rows];
    let mut weights = vec![0.0_f32; graph.n_rows];
    for (row, offsets) in graph.indptr.windows(2).enumerate() {
        let (start, end) = (offsets[0], offsets[1]);
        if start > end || end > graph.indices.len() {
            return Err(DensityError::InvalidCsr);
        }
        let mut previous = None;
        for entry in start..end {
            let column = graph.indices[entry];
            if column >= graph.n_cols || previous.is_some_and(|last| column <= last) {
                return Err(DensityError::InvalidCsr);
            }
            previous = Some(column);

            let weight = graph.data[entry];
            if !(0.0..=1.0).contains(&weight) {
                return Err(DensityError::InvalidWeight { entry, weight });
            }
            let distance = distance(row, column, entry);
            if !distance.is_finite() || distance < 0.0 {
                return Err(DensityError::InvalidDistance {
                    row,
                    column,
                    distance,
                });
            }

            let distance = f64::from(distance);
            let term = match space {
                RadiusSpace::Original => distance * distance,
                RadiusSpace::Embedding => distance,
            };
            let weighted = f64::from(weight) * term;
            for endpoint in [row, column] {
                numerators[endpoint] = (f64::from(numerators[endpoint]) + weighted) as f32;
                weights[endpoint] += weight;
            }
        }
    }

    for (numerator, &weight) in numerators.iter_mut().zip(&weights) {
        *numerator /= weight;
    }
    Ok((numerators, weights))
}

fn euclidean_radii(
    graph: &CsrGraph,
    coordinates: &[f32],
    n_features: usize,
    space: RadiusSpace,
) -> Result<Vec<f32>, DensityError> {
    Ok(
        euclidean_radius_log_inputs(graph, coordinates, n_features, space)?
            .into_iter()
            .map(f32::ln)
            .collect(),
    )
}

/// Compute the float32 arguments to the diagnostic natural logarithm.
///
/// Use the fitted feature graph with [`RadiusSpace::Original`], or a graph
/// refit on the layout with [`RadiusSpace::Embedding`]. This preserves the
/// reduction and offset of the native diagnostics for clients that apply the
/// logarithm with their own numerical library.
pub fn euclidean_radius_log_inputs(
    graph: &CsrGraph,
    coordinates: &[f32],
    n_features: usize,
    space: RadiusSpace,
) -> Result<Vec<f32>, DensityError> {
    validate_coordinates(coordinates, graph.n_rows, n_features, space)?;
    reduce_radius_means(graph, space, |row, column, _| {
        let left = coordinate_row(coordinates, n_features, row);
        let right = coordinate_row(coordinates, n_features, column);
        match space {
            RadiusSpace::Original => euclidean_distance_wide(left, right) as f32,
            RadiusSpace::Embedding => fused_euclidean_distance(left, right),
        }
    })
    .map(|(values, _)| values.into_iter().map(|mean| mean + LOG_OFFSET).collect())
}

fn validate_coordinates(
    coordinates: &[f32],
    n_rows: usize,
    n_features: usize,
    space: RadiusSpace,
) -> Result<(), DensityError> {
    if n_rows.checked_mul(n_features) != Some(coordinates.len())
        || (space == RadiusSpace::Embedding && n_features == 0)
    {
        return Err(DensityError::Matrix(KnnError::InvalidShape {
            n_rows,
            n_features,
            values: coordinates.len(),
        }));
    }
    if let Some((index, &value)) = coordinates.iter().enumerate().find(|(_, x)| !x.is_finite()) {
        return Err(DensityError::Matrix(KnnError::NonFiniteValue {
            index,
            value,
        }));
    }
    Ok(())
}

fn coordinate_row(coordinates: &[f32], n_features: usize, row: usize) -> &[f32] {
    let start = row * n_features;
    &coordinates[start..start + n_features]
}

fn fused_euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    let squared = left.iter().zip(right).fold(0.0, |sum, (&left, &right)| {
        let delta = left - right;
        delta.mul_add(delta, sum)
    });
    squared.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_optimizer_radii_keep_offset_at_one() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![0.5, 0.5],
        };
        let (radii, weights) =
            original_radii_and_weights(&graph, OriginalRadiusInput::EdgeDistances(&[1.0; 2]))
                .unwrap();
        assert_eq!(weights, [1.0; 2]);
        assert_eq!(radii, [1e-8_f32; 2]);
    }
}
