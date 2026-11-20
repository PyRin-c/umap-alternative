//! Directed bipartite graphs and embedding initialization for held-out UMAP inputs.

use std::fmt;

use crate::fuzzy_graph::membership;
use crate::neighbors::{DistanceMetric, KnnError, Neighbor};

/// A rectangular coordinate (COO) graph of directed fuzzy memberships.
///
/// Each stored entry consists of the query row, training column, and membership
/// at the same position in the three vectors. Entries are grouped by ascending
/// query row and ordered within a row by ascending distance, breaking ties by
/// ascending training index. The shape is `n_rows x n_cols`.
#[derive(Clone, Debug, PartialEq)]
pub struct CooGraph {
    /// Number of query rows.
    pub n_rows: usize,
    /// Number of training rows.
    pub n_cols: usize,
    /// Zero-based query index of each entry.
    pub row: Vec<usize>,
    /// Zero-based training index of each entry.
    pub col: Vec<usize>,
    /// Directed membership of each entry, including any rounded zeros.
    pub data: Vec<f32>,
}

/// Validation failures for [`transform_dense_graph`].
#[derive(Clone, Debug, PartialEq)]
pub enum DenseGraphTransformError {
    /// The training matrix or requested neighbor count is invalid.
    Training(KnnError),
    /// The query matrix is invalid.
    Queries(KnnError),
    /// At least one feature is required by both matrices.
    NoFeatures,
    /// The selected distance overflowed the distance metric's precision.
    NonFiniteDistance {
        query: usize,
        training: usize,
        distance: f32,
    },
}

impl fmt::Display for DenseGraphTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Training(error) => write!(formatter, "invalid training input: {error}"),
            Self::Queries(error) => write!(formatter, "invalid query input: {error}"),
            Self::NoFeatures => write!(formatter, "dense transform requires at least one feature"),
            Self::NonFiniteDistance {
                query,
                training,
                distance,
            } => write!(
                formatter,
                "distance from query row {query} to training row {training} is not finite: {distance}"
            ),
        }
    }
}

impl std::error::Error for DenseGraphTransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Training(error) | Self::Queries(error) => Some(error),
            _ => None,
        }
    }
}

/// Validation failures for [`initialize_transform_embedding`].
#[derive(Clone, Debug, PartialEq)]
pub enum TransformInitializationError {
    /// The fitted coordinates do not have the shape implied by the graph.
    InvalidFittedShape {
        n_rows: usize,
        n_components: usize,
        values: usize,
    },
    /// The requested output length exceeds the addressable range.
    OutputShapeOverflow { n_rows: usize, n_components: usize },
    /// COO rows, columns, and weights must have equal lengths.
    InconsistentGraphLengths,
    /// An entry is outside the graph shape or its row is out of order.
    InvalidGraphIndex {
        position: usize,
        row: usize,
        column: usize,
    },
    /// A membership is outside `[0, 1]`.
    InvalidMembership { position: usize, weight: f32 },
    /// A query and training row share multiple nonzero entries.
    RepeatedEdge { row: usize, column: usize },
}

impl fmt::Display for TransformInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFittedShape {
                n_rows,
                n_components,
                values,
            } => write!(
                formatter,
                "fitted embedding requires a positive component count and shape {n_rows}x{n_components}, got {values} values"
            ),
            Self::OutputShapeOverflow {
                n_rows,
                n_components,
            } => write!(
                formatter,
                "transform embedding shape {n_rows}x{n_components} exceeds the addressable range"
            ),
            Self::InconsistentGraphLengths => {
                write!(formatter, "COO row, column, and weight lengths differ")
            }
            Self::InvalidGraphIndex {
                position,
                row,
                column,
            } => write!(
                formatter,
                "COO entry {position} at ({row}, {column}) is out of bounds or out of row order"
            ),
            Self::InvalidMembership { position, weight } => write!(
                formatter,
                "COO membership at entry {position} is outside [0, 1]: {weight}"
            ),
            Self::RepeatedEdge { row, column } => {
                write!(
                    formatter,
                    "COO graph repeats the nonzero edge ({row}, {column})"
                )
            }
        }
    }
}

impl std::error::Error for TransformInitializationError {}

/// Initialize a row-major query embedding from the held-out graph and fitted coordinates.
///
/// The fitted matrix has shape `graph.n_cols x n_components` with at least
/// one component. The output shape is `graph.n_rows x n_components`.
/// The nonzero COO entries must be grouped by ascending query row, with
/// unique training columns within each query row. Stored zeros are skipped.
/// A query with no positive memberships starts with NaN coordinates.
/// When a membership is exactly one, the initializer
/// copies the fitted row at the smallest training column with that membership.
/// Otherwise it L1-normalizes the weights and accumulates weighted fitted
/// coordinates in descending training-column order, with separate float32
/// rounding at each step. Fitted NaNs propagate from contributing rows.
///
/// # Errors
///
/// Returns an error for an invalid embedding shape or malformed COO entries.
pub fn initialize_transform_embedding(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    n_components: usize,
) -> Result<Vec<f32>, TransformInitializationError> {
    use TransformInitializationError as Error;

    if n_components == 0 || graph.n_cols.checked_mul(n_components) != Some(fitted_embedding.len()) {
        return Err(Error::InvalidFittedShape {
            n_rows: graph.n_cols,
            n_components,
            values: fitted_embedding.len(),
        });
    }
    if graph.row.len() != graph.col.len() || graph.row.len() != graph.data.len() {
        return Err(Error::InconsistentGraphLengths);
    }
    let output_len = graph
        .n_rows
        .checked_mul(n_components)
        .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or(Error::OutputShapeOverflow {
            n_rows: graph.n_rows,
            n_components,
        })?;
    let mut embedding = vec![f32::NAN; output_len];
    let mut entries = Vec::new();
    let mut previous_row = None;
    let mut positions = (0..graph.data.len())
        .filter(|&position| graph.data[position] != 0.0)
        .peekable();

    while let Some(&start) = positions.peek() {
        let row = graph.row[start];
        if row >= graph.n_rows || previous_row.is_some_and(|previous| row <= previous) {
            return Err(Error::InvalidGraphIndex {
                position: start,
                row,
                column: graph.col[start],
            });
        }
        previous_row = Some(row);
        entries.clear();
        while let Some(position) = positions.next_if(|&position| graph.row[position] == row) {
            let column = graph.col[position];
            let weight = graph.data[position];
            if column >= graph.n_cols {
                return Err(Error::InvalidGraphIndex {
                    position,
                    row,
                    column,
                });
            }
            if !(0.0..=1.0).contains(&weight) {
                return Err(Error::InvalidMembership { position, weight });
            }
            entries.push((column, weight));
        }
        entries.sort_unstable_by_key(|&(column, _)| column);
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::RepeatedEdge {
                row,
                column: pair[0].0,
            });
        }
        let row_start = row * n_components;
        let coordinates = &mut embedding[row_start..row_start + n_components];
        if let Some(&(column, _)) = entries.iter().find(|&&(_, weight)| weight == 1.0) {
            let start = column * n_components;
            coordinates.copy_from_slice(&fitted_embedding[start..start + n_components]);
            continue;
        }

        // The sparse row reduction adds the first membership after summing the tail.
        let scale = (sum_memberships(&entries[1..]) + entries[0].1).recip();
        coordinates.fill(0.0);
        for &(column, weight) in entries.iter().rev() {
            let start = column * n_components;
            let normalized = weight * scale;
            for (coordinate, &fitted) in coordinates
                .iter_mut()
                .zip(&fitted_embedding[start..start + n_components])
            {
                *coordinate += normalized * fitted;
            }
        }
    }
    Ok(embedding)
}

// The float32 tail reduction starts with eight lanes and splits large
// blocks at a lane boundary. Keep the grouping to preserve the rounding.
fn sum_memberships(entries: &[(usize, f32)]) -> f32 {
    if entries.len() < 8 {
        return entries.iter().map(|&(_, weight)| weight).sum();
    }
    if entries.len() > 128 {
        let middle = (entries.len() / 2) & !7;
        let (left, right) = entries.split_at(middle);
        return sum_memberships(left) + sum_memberships(right);
    }

    let mut blocks = entries.chunks_exact(8);
    let mut lanes = [0.0; 8];
    for block in &mut blocks {
        for (lane, &(_, weight)) in lanes.iter_mut().zip(block) {
            *lane += weight;
        }
    }
    let [a, b, c, d, e, f, g, h] = lanes;
    let sum = ((a + b) + (c + d)) + ((e + f) + (g + h));
    blocks
        .remainder()
        .iter()
        .fold(sum, |total, &(_, weight)| total + weight)
}

/// Construct the fuzzy graph from held-out query rows to fitted training rows.
///
/// Supply two finite row-major matrices with the same positive feature count.
/// `k` is the effective neighbor count in `1..=n_training_rows`. An empty
/// query matrix produces an empty graph with `n_training_rows` columns.
///
/// The transform for training local connectivity one uses zero distance offset.
/// It calibrates each local scale over neighbor ranks `1..k` (excluding the
/// closest candidate) toward a membership sum of `log2(k)`. The minimum
/// scale is 0.001 times the mean selected distance across all query rows.
/// Every selected candidate contributes a membership. A query index and a
/// training index with the same numeric value identify independent vertices.
/// The result has exactly `k` directed entries per query row.
///
/// # Errors
///
/// Returns an error for invalid matrix shapes, a zero feature count, invalid
/// `k`, non-finite input values, or non-finite selected distances.
pub fn transform_dense_graph(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<CooGraph, DenseGraphTransformError> {
    if n_features == 0 {
        return Err(DenseGraphTransformError::NoFeatures);
    }
    validate_matrix(training, n_training_rows, n_features)
        .map_err(DenseGraphTransformError::Training)?;
    validate_matrix(queries, n_query_rows, n_features)
        .map_err(DenseGraphTransformError::Queries)?;
    if !(1..=n_training_rows).contains(&k) {
        return Err(DenseGraphTransformError::Training(KnnError::InvalidK {
            k,
            n_rows: n_training_rows,
        }));
    }

    let mut graph = CooGraph {
        n_rows: n_query_rows,
        n_cols: n_training_rows,
        row: Vec::new(),
        col: Vec::new(),
        data: Vec::new(),
    };
    if n_query_rows == 0 {
        return Ok(graph);
    }

    let mut candidates = Vec::with_capacity(n_training_rows);
    for (query_index, query) in queries.chunks_exact(n_features).enumerate() {
        candidates.clear();
        candidates.extend(training.chunks_exact(n_features).enumerate().map(
            |(index, candidate)| Neighbor {
                index,
                distance: metric.distance(query, candidate),
            },
        ));
        candidates.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.index.cmp(&right.index))
        });
        for neighbor in &candidates[..k] {
            if !neighbor.distance.is_finite() {
                return Err(DenseGraphTransformError::NonFiniteDistance {
                    query: query_index,
                    training: neighbor.index,
                    distance: neighbor.distance,
                });
            }
            graph.row.push(query_index);
            graph.col.push(neighbor.index);
            graph.data.push(neighbor.distance);
        }
    }

    let target = (k as f64).log2();
    let minimum_scale = minimum_transform_scale(&graph.data);
    for distances in graph.data.chunks_exact_mut(k) {
        let sigma = transform_scale(distances, target).max(minimum_scale);
        for distance in distances {
            *distance = membership(*distance, 0.0, sigma);
        }
    }
    Ok(graph)
}

fn validate_matrix(data: &[f32], n_rows: usize, n_features: usize) -> Result<(), KnnError> {
    if n_rows.checked_mul(n_features) != Some(data.len()) {
        return Err(KnnError::InvalidShape {
            n_rows,
            n_features,
            values: data.len(),
        });
    }
    if let Some((index, &value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KnnError::NonFiniteValue { index, value });
    }
    Ok(())
}

fn minimum_transform_scale(distances: &[f32]) -> f32 {
    let sum = distances.iter().sum::<f32>();
    let mean = if sum.is_finite() {
        sum / distances.len() as f32
    } else {
        // A sum of individually finite distances can overflow even when their mean fits.
        (distances
            .iter()
            .map(|&distance| f64::from(distance))
            .sum::<f64>()
            / distances.len() as f64) as f32
    };
    (0.001 * f64::from(mean)) as f32
}

fn transform_scale(distances: &[f32], target: f64) -> f32 {
    let mut lower = 0.0_f64;
    let mut upper = f64::INFINITY;
    let mut sigma = 1.0_f64;
    for _ in 0..64 {
        let sum: f64 = distances[1..]
            .iter()
            .map(|&distance| (-f64::from(distance) / sigma).exp())
            .sum();
        if (sum - target).abs() < 1e-5 {
            break;
        }
        if sum > target {
            upper = sigma;
        } else {
            lower = sigma;
        }
        sigma = if upper.is_infinite() {
            sigma * 2.0
        } else {
            (lower + upper) * 0.5
        };
    }
    sigma as f32
}
