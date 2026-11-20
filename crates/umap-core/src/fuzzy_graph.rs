//! First-pass fuzzy graph construction for dense UMAP neighbors.
//!
//! The implementation here follows the fuzzy-simplicial-set construction from
//! the UMAP paper. The result is a deterministic CSR graph. Supervised paths
//! also adjust and renormalize memberships before returning the graph.

use std::fmt;

use crate::neighbors::Neighbor;

mod composition;
mod primitives;
pub(crate) mod supplied_neighbors;

pub use composition::{FuzzySetOperation, combine_fuzzy_graphs};
pub use primitives::{
    KnnDistances, KnnMemberships, KnnStageError, SmoothKnnConfig, compute_knn_memberships,
    smooth_knn_distances,
};

const SMOOTH_TOLERANCE: f64 = 1e-5;
const MIN_SCALE_MULTIPLIER: f64 = 0.001;

/// A square compressed-sparse-row fuzzy graph.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrGraph {
    /// Number of rows in the graph.
    pub n_rows: usize,
    /// Number of columns in the graph (equal to [`CsrGraph::n_rows`]).
    pub n_cols: usize,
    /// Row offsets. This always has length `n_rows + 1`.
    pub indptr: Vec<usize>,
    /// Sorted column indices for each row.
    pub indices: Vec<usize>,
    /// Fuzzy membership for each entry in [`CsrGraph::indices`].
    pub data: Vec<f32>,
}

/// Validated local connectivity and set-operation mix for fuzzy graphs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphOptions {
    local_connectivity: f64,
    set_op_mix_ratio: f64,
}

impl GraphOptions {
    /// Set the local connectivity and the weight of the fuzzy union.
    ///
    /// Local connectivity must be finite and non-negative. The mix ratio must
    /// be in `0..=1`, with zero selecting intersection and one selecting union.
    pub fn new(local_connectivity: f64, set_op_mix_ratio: f64) -> Result<Self, GraphOptionsError> {
        if !local_connectivity.is_finite() || local_connectivity < 0.0 {
            return Err(GraphOptionsError::InvalidLocalConnectivity { local_connectivity });
        }
        if !(0.0..=1.0).contains(&set_op_mix_ratio) {
            return Err(GraphOptionsError::InvalidSetOpMixRatio { set_op_mix_ratio });
        }
        Ok(Self {
            local_connectivity,
            set_op_mix_ratio,
        })
    }

    /// Return the local connectivity used when fitting the graph.
    #[must_use]
    pub fn local_connectivity(self) -> f64 {
        self.local_connectivity
    }
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
        }
    }
}

/// Validation failures for [`GraphOptions::new`].
#[derive(Clone, Debug, PartialEq)]
pub enum GraphOptionsError {
    /// Local connectivity must be finite and non-negative.
    InvalidLocalConnectivity { local_connectivity: f64 },
    /// The set-operation mix ratio must be finite and in `0..=1`.
    InvalidSetOpMixRatio { set_op_mix_ratio: f64 },
}

impl fmt::Display for GraphOptionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLocalConnectivity { local_connectivity } => write!(
                formatter,
                "local connectivity {local_connectivity} must be finite and non-negative"
            ),
            Self::InvalidSetOpMixRatio { set_op_mix_ratio } => {
                write!(
                    formatter,
                    "set operation mix ratio {set_op_mix_ratio} must be in 0..=1"
                )
            }
        }
    }
}

impl std::error::Error for GraphOptionsError {}

/// Validation failures for [`fuzzy_simplicial_set`].
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// A query row has no candidates, so its local scale is undefined.
    EmptyRow { row: usize },
    /// Every query row must contain the same number of candidates.
    InconsistentRowLength {
        row: usize,
        expected: usize,
        actual: usize,
    },
    /// The neighbor count cannot exceed the number of graph rows.
    InvalidK { k: usize, n_rows: usize },
    /// A candidate index is outside the square graph.
    InvalidNeighborIndex {
        row: usize,
        position: usize,
        index: usize,
        n_rows: usize,
    },
    /// A query row contains the same candidate more than once.
    DuplicateNeighbor {
        row: usize,
        position: usize,
        index: usize,
    },
    /// Every query row must include itself as a candidate.
    MissingSelf { row: usize },
    /// A self candidate must have exactly zero distance.
    InvalidSelfDistance {
        row: usize,
        position: usize,
        distance: f32,
    },
    /// Distances must be finite and non-negative.
    InvalidDistance {
        row: usize,
        position: usize,
        index: usize,
        distance: f32,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRow { row } => write!(formatter, "neighbor row {row} is empty"),
            Self::InconsistentRowLength {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "neighbor row {row} has length {actual}, expected {expected}"
            ),
            Self::InvalidK { k, n_rows } => {
                write!(formatter, "k={k} is outside the valid range 1..={n_rows}")
            }
            Self::InvalidNeighborIndex {
                row,
                position,
                index,
                n_rows,
            } => write!(
                formatter,
                "neighbor row {row}, position {position} has index {index}, outside 0..{n_rows}"
            ),
            Self::DuplicateNeighbor {
                row,
                position,
                index,
            } => write!(
                formatter,
                "neighbor row {row}, position {position} repeats index {index}"
            ),
            Self::MissingSelf { row } => {
                write!(formatter, "neighbor row {row} does not include itself")
            }
            Self::InvalidSelfDistance {
                row,
                position,
                distance,
            } => write!(
                formatter,
                "neighbor row {row}, position {position} has self distance {distance}, expected zero"
            ),
            Self::InvalidDistance {
                row,
                position,
                index,
                distance,
            } => write!(
                formatter,
                "neighbor row {row}, position {position}, index {index} has invalid distance {distance}"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// Build a first-pass UMAP fuzzy graph from per-row neighbor results.
///
/// The input rows are expected to come from
/// [`crate::neighbors::exhaustive_knn`]: all rows have one common `k`, every
/// row includes itself exactly once at distance zero, and distances are
/// Euclidean. Rows do not need to be sorted because this function validates and
/// canonicalizes their support.
///
/// For query row `i`, local connectivity one sets `rho_i` to the smallest
/// strictly positive non-self distance (or zero if no such distance exists).
/// A positive scale `sigma_i` is then found by the UMAP smooth-scale search
/// (starting at one, expanding an unbounded upper bracket, then bisecting) so
/// the sum of directed memberships over the non-self candidates approaches
/// `log2(k)`:
///
/// `sum(exp(-max(0, d_ij - rho_i) / sigma_i)) = log2(k)`.
///
/// The scale search computes the target and sum in `f64` with an absolute
/// tolerance of `1e-5`. The offsets, trial scales, and emitted memberships
/// use `f32`. If the zero-scale limit exceeds the target, the scale is reduced to
/// `0.001 * max_offset`. Rows with no positive offset use a unit scale, and
/// all their memberships are one.
///
/// Directed memberships are combined with the fuzzy union
/// `a + b - a * b`. The diagonal is omitted from the emitted graph, while all
/// non-self directed supports are retained, including supports whose computed
/// membership underflows to zero. CSR columns are strictly ascending within
/// every row, and the output is square with shape `n_rows x n_rows`.
pub fn fuzzy_simplicial_set(neighbors: &[Vec<Neighbor>]) -> Result<CsrGraph, GraphError> {
    build_fuzzy_graph(neighbors, GraphOptions::default(), true, None)
}

/// Build a UMAP fuzzy graph with explicit local connectivity and mix options.
///
/// Neighbor rows follow the validation rules of [`fuzzy_simplicial_set`].
/// Local connectivity indexes the sorted positive distances, interpolating
/// fractional ranks and saturating at the largest available distance. Rows
/// without a positive distance use a zero offset. For ranks of one or higher,
/// fractional parts at most `1e-5` leave the offset at the lower distance.
///
/// The minimum scale is `0.001` times the mean distance of the row when its
/// offset is positive, or the mean across all neighbor rows otherwise. The
/// means include the zero self distances. These scale floors and the
/// omission of zero weights also apply with [`GraphOptions::default`].
/// [`fuzzy_simplicial_set`] keeps its original scale and explicit-zero behavior.
///
/// For reciprocal directed memberships `a` and `b`, the weight is
/// `mix * (a + b - a * b) + (1 - mix) * (a * b)`. An absent directed
/// membership is zero. The result is square CSR with ascending columns and
/// no diagonal. At mix zero, only positive reciprocal products are stored.
pub fn fuzzy_simplicial_set_with_options(
    neighbors: &[Vec<Neighbor>],
    options: GraphOptions,
) -> Result<CsrGraph, GraphError> {
    build_fuzzy_graph(neighbors, options, false, None)
}

/// Build a graph whose requested neighbor count includes missing candidates.
pub(crate) fn fuzzy_simplicial_set_with_neighbor_target(
    neighbors: &[Vec<Neighbor>],
    requested_neighbors: usize,
) -> Result<CsrGraph, GraphError> {
    build_fuzzy_graph(
        neighbors,
        GraphOptions::default(),
        false,
        Some(requested_neighbors),
    )
}

fn build_fuzzy_graph(
    neighbors: &[Vec<Neighbor>],
    options: GraphOptions,
    preserve_default_behavior: bool,
    requested_neighbors: Option<usize>,
) -> Result<CsrGraph, GraphError> {
    let n_rows = neighbors.len();
    if n_rows == 0 {
        return Ok(CsrGraph {
            n_rows: 0,
            n_cols: 0,
            indptr: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        });
    }

    let k = neighbors[0].len();
    if k == 0 {
        return Err(GraphError::EmptyRow { row: 0 });
    }
    if k > n_rows {
        return Err(GraphError::InvalidK { k, n_rows });
    }

    for (row_index, row) in neighbors.iter().enumerate() {
        if row.len() != k {
            return Err(GraphError::InconsistentRowLength {
                row: row_index,
                expected: k,
                actual: row.len(),
            });
        }

        let mut seen = vec![false; n_rows];
        let mut has_self = false;
        for (position, neighbor) in row.iter().enumerate() {
            if neighbor.index >= n_rows {
                return Err(GraphError::InvalidNeighborIndex {
                    row: row_index,
                    position,
                    index: neighbor.index,
                    n_rows,
                });
            }
            if seen[neighbor.index] {
                return Err(GraphError::DuplicateNeighbor {
                    row: row_index,
                    position,
                    index: neighbor.index,
                });
            }
            seen[neighbor.index] = true;

            if !neighbor.distance.is_finite() || neighbor.distance < 0.0 {
                return Err(GraphError::InvalidDistance {
                    row: row_index,
                    position,
                    index: neighbor.index,
                    distance: neighbor.distance,
                });
            }
            if neighbor.index == row_index {
                has_self = true;
                if neighbor.distance != 0.0 {
                    return Err(GraphError::InvalidSelfDistance {
                        row: row_index,
                        position,
                        distance: neighbor.distance,
                    });
                }
            }
        }
        if !has_self {
            return Err(GraphError::MissingSelf { row: row_index });
        }
    }

    let sorted_distances =
        (!preserve_default_behavior).then(|| sorted_neighbor_distances(neighbors));
    let batch_floor = sorted_distances.as_deref().map_or(0.0, batch_scale_floor);
    let mut directed = Vec::with_capacity(n_rows);
    for (row_index, row) in neighbors.iter().enumerate() {
        let (rho, sigma) = if let Some(distances) = &sorted_distances {
            let distances = &distances[row_index * k..(row_index + 1) * k];
            row_parameters_with_target(
                distances,
                options.local_connectivity,
                batch_floor,
                requested_neighbors.unwrap_or(k),
            )
        } else {
            let rho = default_local_offset(row);
            (rho, default_distance_scale(row_index, row, rho))
        };
        let mut memberships = row
            .iter()
            .filter(|neighbor| neighbor.index != row_index)
            .map(|neighbor| (neighbor.index, membership(neighbor.distance, rho, sigma)))
            .collect::<Vec<_>>();
        memberships.sort_unstable_by_key(|(index, _)| *index);
        directed.push(memberships);
    }

    let mix = options.set_op_mix_ratio as f32;
    let intersection_mix = (1.0 - options.set_op_mix_ratio) as f32;
    Ok(symmetrize(
        &directed,
        preserve_default_behavior,
        |forward, reverse| {
            let union = fuzzy_union(forward, reverse);
            let intersection = forward * reverse;
            (mix * union + intersection_mix * intersection).clamp(0.0, 1.0)
        },
    ))
}

fn symmetrize(
    directed: &[Vec<(usize, f32)>],
    keep_zero_weights: bool,
    combine: impl Fn(f32, f32) -> f32,
) -> CsrGraph {
    let n_rows = directed.len();
    let mut support = vec![Vec::new(); n_rows];
    for (row_index, memberships) in directed.iter().enumerate() {
        for &(index, _) in memberships {
            support[row_index].push(index);
            support[index].push(row_index);
        }
    }

    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for (row_index, columns) in support.iter_mut().enumerate() {
        columns.sort_unstable();
        columns.dedup();
        for &column in columns.iter() {
            let forward = find_membership(&directed[row_index], column);
            let reverse = find_membership(&directed[column], row_index);
            let weight = combine(forward, reverse);
            if weight == 0.0 && !keep_zero_weights {
                continue;
            }
            indices.push(column);
            data.push(weight);
        }
        indptr.push(indices.len());
    }

    CsrGraph {
        n_rows,
        n_cols: n_rows,
        indptr,
        indices,
        data,
    }
}

// `graph` is the symmetric output of `fuzzy_simplicial_set`, and `labels`
// and `target_weight` have been validated by the categorical fit entry point.
pub(crate) fn categorical_intersection(
    mut graph: CsrGraph,
    labels: &[i64],
    target_weight: f64,
) -> CsrGraph {
    const UNLABELED: i64 = -1;
    let unlabeled_factor = (-1.0_f64).exp();
    let cross_factor = if target_weight == 1.0 {
        0.0
    } else {
        (-2.5 / (1.0 - target_weight)).exp()
    };

    let mut row_maxima = vec![0.0_f32; graph.n_rows];
    for (row, row_max) in row_maxima.iter_mut().enumerate() {
        let row_label = labels[row];
        for entry in graph.indptr[row]..graph.indptr[row + 1] {
            let column_label = labels[graph.indices[entry]];
            let weight = &mut graph.data[entry];
            // Round the product once, after computing the attenuation in f64.
            if row_label == UNLABELED || column_label == UNLABELED {
                *weight = (f64::from(*weight) * unlabeled_factor) as f32;
            } else if row_label != column_label {
                *weight = (f64::from(*weight) * cross_factor) as f32;
            }
            *row_max = row_max.max(*weight);
        }
    }

    let mut result = CsrGraph {
        n_rows: graph.n_rows,
        n_cols: graph.n_cols,
        indptr: Vec::with_capacity(graph.indptr.len()),
        indices: Vec::with_capacity(graph.indices.len()),
        data: Vec::with_capacity(graph.data.len()),
    };
    result.indptr.push(0);
    for row in 0..graph.n_rows {
        for entry in graph.indptr[row]..graph.indptr[row + 1] {
            let weight = graph.data[entry];
            if weight == 0.0 {
                continue;
            }

            // Symmetric attenuation preserves the same weight in both rows,
            // so a stored positive edge guarantees both maxima are positive.
            let column = graph.indices[entry];
            let forward = weight / row_maxima[row];
            let reverse = weight / row_maxima[column];
            result.indices.push(column);
            result.data.push(fuzzy_union(forward, reverse));
        }
        result.indptr.push(result.indices.len());
    }
    result
}

// Continuous targets contribute directed memberships. Merge their supports before
// scaling each row and taking a fuzzy union with the transpose.
pub(crate) fn continuous_intersection(
    features: &CsrGraph,
    neighbors: &[Vec<Neighbor>],
    target_weight: f64,
) -> CsrGraph {
    let distances = sorted_neighbor_distances(neighbors);
    let batch_floor = batch_scale_floor(&distances);
    let feature_floor = features
        .data
        .iter()
        .copied()
        .filter(|&weight| weight > 0.0)
        .reduce(f32::min)
        .map_or(1.0e-8, |minimum| (f64::from(minimum) / 2.0).max(1.0e-8));
    let target_floor = 1.0e-4_f64;
    let k = neighbors[0].len();
    let exponent = if target_weight < 0.5 {
        target_weight / (1.0 - target_weight)
    } else {
        (1.0 - target_weight) / target_weight
    };

    let mut directed = Vec::with_capacity(features.n_rows);
    for (row, neighbors) in neighbors.iter().enumerate() {
        let (rho, sigma) = row_parameters(&distances[row * k..(row + 1) * k], 1.0, batch_floor);
        let mut target = neighbors
            .iter()
            .filter(|neighbor| neighbor.index != row)
            .map(|neighbor| (neighbor.index, membership(neighbor.distance, rho, sigma)))
            .filter(|&(_, weight)| weight > 0.0)
            .collect::<Vec<_>>();
        let entries = features.indptr[row]..features.indptr[row + 1];
        let columns = &features.indices[entries.clone()];
        let feature_weights = &features.data[entries];
        let mut support = columns.to_vec();
        support.extend(target.iter().map(|&(column, _)| column));
        support.sort_unstable();
        support.dedup();

        target.sort_unstable_by_key(|&(column, _)| column);
        let mut memberships = Vec::with_capacity(support.len());
        for column in support {
            let left = columns
                .binary_search(&column)
                .map_or(0.0, |i| feature_weights[i]);
            let right = find_membership(&target, column);
            if left == 0.0 && right == 0.0 {
                continue;
            }
            let left = if left == 0.0 {
                feature_floor
            } else {
                f64::from(left)
            };
            let right = if right == 0.0 {
                target_floor
            } else {
                f64::from(right)
            };
            let weight = if target_weight < 0.5 {
                left * right.powf(exponent)
            } else {
                left.powf(exponent) * right
            };
            memberships.push((column, weight as f32));
        }
        let maximum = memberships
            .iter()
            .map(|&(_, weight)| weight)
            .fold(0.0_f32, f32::max);
        if maximum > 0.0 {
            for (_, weight) in &mut memberships {
                *weight /= maximum;
            }
        }
        directed.push(memberships);
    }
    symmetrize(&directed, false, fuzzy_union)
}

fn fuzzy_union(forward: f32, reverse: f32) -> f32 {
    forward + reverse - forward * reverse
}

fn default_local_offset(row: &[Neighbor]) -> f32 {
    row.iter()
        .map(|neighbor| neighbor.distance)
        .filter(|&distance| distance > 0.0)
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0)
}

fn default_distance_scale(row_index: usize, row: &[Neighbor], rho: f32) -> f32 {
    let target = (row.len() as f64).log2();
    let offsets = row
        .iter()
        .filter(|neighbor| neighbor.index != row_index)
        .map(|neighbor| {
            if neighbor.distance > rho {
                neighbor.distance - rho
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    if target <= 0.0 {
        return 1.0;
    }

    let Some(max_offset) = offsets
        .iter()
        .copied()
        .filter(|offset| *offset > 0.0)
        .max_by(|left, right| left.total_cmp(right))
    else {
        return 1.0;
    };

    let zero_scale_sum = offsets.iter().filter(|offset| **offset == 0.0).count() as f64;
    if zero_scale_sum > target {
        return (max_offset * 0.001).max(f32::MIN_POSITIVE);
    }

    search_scale(&offsets, target, true).max(f32::MIN_POSITIVE)
}

fn sorted_neighbor_distances(neighbors: &[Vec<Neighbor>]) -> Vec<f32> {
    let mut distances = Vec::with_capacity(neighbors.len() * neighbors[0].len());
    for row in neighbors {
        let start = distances.len();
        distances.extend(row.iter().map(|neighbor| neighbor.distance));
        distances[start..].sort_unstable_by(f32::total_cmp);
    }
    distances
}

fn local_offset<T>(positive: &[T], local_connectivity: f64) -> f32
where
    T: Copy + Into<f64> + std::ops::Sub<Output = T>,
{
    if positive.is_empty() {
        return 0.0;
    }
    if local_connectivity > positive.len() as f64 {
        return positive.iter().copied().map(Into::into).fold(0.0, f64::max) as f32;
    }
    let whole = local_connectivity as usize;
    let fraction = local_connectivity - whole as f64;
    if whole == 0 {
        return (fraction * positive[0].into()) as f32;
    }
    let base = positive[whole - 1];
    if fraction > SMOOTH_TOLERANCE {
        // Round the adjacent distance difference before interpolating the offset.
        (base.into() + fraction * (positive[whole] - base).into()) as f32
    } else {
        base.into() as f32
    }
}

pub(crate) fn row_parameters(
    distances: &[f32],
    local_connectivity: f64,
    batch_floor: f32,
) -> (f32, f32) {
    row_parameters_with_target(distances, local_connectivity, batch_floor, distances.len())
}

pub(crate) fn row_parameters_with_target(
    distances: &[f32],
    local_connectivity: f64,
    batch_floor: f32,
    requested_neighbors: usize,
) -> (f32, f32) {
    let positive = &distances[distances.partition_point(|&distance| distance == 0.0)..];
    let rho = local_offset(positive, local_connectivity);
    let target = (requested_neighbors as f64).log2();
    // Omit one zero for self. The other zero-distance candidates still
    // contribute unit memberships, irrespective of the order of the ties.
    let offsets = distances[1..]
        .iter()
        .map(|&distance| (distance - rho).max(0.0))
        .collect::<Vec<_>>();
    let floor = if rho > 0.0 {
        (MIN_SCALE_MULTIPLIER * mean_distance(distances)) as f32
    } else {
        batch_floor
    };
    (rho, search_scale(&offsets, target, false).max(floor))
}

pub(crate) fn batch_scale_floor(distances: &[f32]) -> f32 {
    // The batch mean rounds before multiplication. The row mean stays wide.
    let mean = mean_distance(distances) as f32;
    (MIN_SCALE_MULTIPLIER * f64::from(mean)) as f32
}

fn mean_distance(distances: &[f32]) -> f64 {
    // The public Windows probes distinguish these two f32 reduction groupings.
    let (blocks, tail) = distances.as_chunks::<32>();
    let mut lanes = [0.0; 32];
    for block in blocks {
        for (sum, &distance) in lanes.iter_mut().zip(block) {
            *sum += distance;
        }
    }
    let [a, b, c, d, e, f, g, h] =
        std::array::from_fn(|i| (lanes[i] + lanes[i + 8]) + (lanes[i + 16] + lanes[i + 24]));
    let wide_sum = ((a + e) + (c + g)) + ((b + f) + (d + h));

    let (blocks, tail) = tail.as_chunks::<4>();
    let [a, b, c, d] = blocks
        .iter()
        .fold([wide_sum, 0.0, 0.0, 0.0], |[a, b, c, d], block| {
            [a + block[0], b + block[1], c + block[2], d + block[3]]
        });
    let sum = tail
        .iter()
        .fold((a + c) + (b + d), |sum, value| sum + value);
    let sum = if sum.is_finite() {
        f64::from(sum)
    } else {
        // Finite distances can overflow an f32 sum even when their mean fits.
        distances.iter().map(|&d| f64::from(d)).sum::<f64>()
    };
    sum / distances.len() as f64
}

fn search_scale(offsets: &[f32], target: f64, inclusive_tolerance: bool) -> f32 {
    bisect_scale(target, inclusive_tolerance, 64, |sigma| {
        membership_sum(offsets, sigma)
    })
}

fn bisect_scale(
    target: f64,
    inclusive_tolerance: bool,
    iterations: usize,
    membership_sum: impl Fn(f32) -> f64,
) -> f32 {
    let mut lower = 0.0_f32;
    let mut upper = f32::INFINITY;
    let mut middle = 1.0_f32;

    for _ in 0..iterations {
        // A rounded f32 sum can cross the convergence threshold early.
        let sum = membership_sum(middle);
        let residual = (sum - target).abs();
        if residual < SMOOTH_TOLERANCE || (inclusive_tolerance && residual == SMOOTH_TOLERANCE) {
            break;
        }
        if sum > target {
            upper = middle;
        } else {
            lower = middle;
        }
        middle = if upper.is_infinite() {
            middle * 2.0
        } else {
            (lower + upper) * 0.5
        };
    }

    middle
}

fn membership_sum(offsets: &[f32], sigma: f32) -> f64 {
    let sigma = f64::from(sigma);
    offsets
        .iter()
        .map(|&offset| (-f64::from(offset) / sigma).exp())
        .sum()
}

pub(crate) fn membership(distance: f32, rho: f32, sigma: f32) -> f32 {
    if distance <= rho {
        1.0
    } else {
        (-((distance - rho) / sigma)).exp()
    }
}

fn find_membership(row: &[(usize, f32)], index: usize) -> f32 {
    row.binary_search_by_key(&index, |(candidate, _)| *candidate)
        .map_or(0.0, |position| row[position].1)
}

#[cfg(test)]
#[path = "fuzzy_graph/options_tests.rs"]
mod options_tests;

#[cfg(test)]
mod tests {
    use super::{CsrGraph, GraphError, fuzzy_simplicial_set};
    use crate::neighbors::{Neighbor, exhaustive_knn};

    #[test]
    fn emits_sorted_csr_with_fuzzy_union_weights() {
        let neighbors = vec![
            vec![
                Neighbor {
                    index: 0,
                    distance: 0.0,
                },
                Neighbor {
                    index: 1,
                    distance: 1.0,
                },
                Neighbor {
                    index: 2,
                    distance: 2.0,
                },
                Neighbor {
                    index: 3,
                    distance: 3.0,
                },
            ],
            vec![
                Neighbor {
                    index: 1,
                    distance: 0.0,
                },
                Neighbor {
                    index: 0,
                    distance: 1.0,
                },
                Neighbor {
                    index: 2,
                    distance: 1.0,
                },
                Neighbor {
                    index: 3,
                    distance: 2.0,
                },
            ],
            vec![
                Neighbor {
                    index: 2,
                    distance: 0.0,
                },
                Neighbor {
                    index: 1,
                    distance: 1.0,
                },
                Neighbor {
                    index: 3,
                    distance: 1.0,
                },
                Neighbor {
                    index: 0,
                    distance: 2.0,
                },
            ],
            vec![
                Neighbor {
                    index: 3,
                    distance: 0.0,
                },
                Neighbor {
                    index: 2,
                    distance: 1.0,
                },
                Neighbor {
                    index: 1,
                    distance: 2.0,
                },
                Neighbor {
                    index: 0,
                    distance: 3.0,
                },
            ],
        ];
        let graph = fuzzy_simplicial_set(&neighbors).expect("valid neighbors");

        assert_eq!(graph.n_rows, 4);
        assert_eq!(graph.n_cols, 4);
        assert_eq!(graph.indptr, vec![0, 3, 6, 9, 12]);
        assert_eq!(graph.indices, vec![1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]);
        assert!(
            graph
                .data
                .iter()
                .all(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
        );
        assert!(
            graph
                .indices
                .chunks_exact(3)
                .all(|columns| columns.windows(2).all(|pair| pair[0] < pair[1]))
        );
        assert!(graph.data.iter().any(|weight| *weight < 1.0));
    }

    #[test]
    fn duplicate_points_produce_unit_memberships() {
        let neighbors = exhaustive_knn(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 4, 2, 4)
            .expect("valid neighbors");
        let graph = fuzzy_simplicial_set(&neighbors).expect("valid graph");

        assert_eq!(graph.indptr, vec![0, 3, 6, 9, 12]);
        assert_eq!(graph.indices, vec![1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]);
        assert!(graph.data.iter().all(|weight| *weight == 1.0));
    }

    #[test]
    fn fuzzy_union_keeps_reciprocal_only_support() {
        let neighbors = vec![
            vec![
                Neighbor {
                    index: 0,
                    distance: 0.0,
                },
                Neighbor {
                    index: 1,
                    distance: 1.0,
                },
            ],
            vec![
                Neighbor {
                    index: 1,
                    distance: 0.0,
                },
                Neighbor {
                    index: 0,
                    distance: 1.0,
                },
            ],
            vec![
                Neighbor {
                    index: 2,
                    distance: 0.0,
                },
                Neighbor {
                    index: 1,
                    distance: 1.0,
                },
            ],
        ];
        let graph = fuzzy_simplicial_set(&neighbors).expect("valid neighbors");

        assert_eq!(graph.indptr, vec![0, 1, 3, 4]);
        assert_eq!(graph.indices, vec![1, 0, 2, 1]);
        assert_eq!(graph.data, vec![1.0; 4]);
    }

    #[test]
    fn singleton_has_an_empty_off_diagonal_graph() {
        let neighbors = exhaustive_knn(&[3.0, -4.0], 1, 2, 1).expect("valid neighbors");
        assert_eq!(
            fuzzy_simplicial_set(&neighbors).expect("valid graph"),
            CsrGraph {
                n_rows: 1,
                n_cols: 1,
                indptr: vec![0, 0],
                indices: Vec::new(),
                data: Vec::new(),
            }
        );
    }

    #[test]
    fn empty_input_has_an_empty_graph() {
        assert_eq!(
            fuzzy_simplicial_set(&[]).expect("valid empty input"),
            CsrGraph {
                n_rows: 0,
                n_cols: 0,
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
            }
        );
    }

    #[test]
    fn rejects_malformed_neighbor_rows() {
        let self_neighbor = Neighbor {
            index: 0,
            distance: 0.0,
        };
        assert_eq!(
            fuzzy_simplicial_set(&[vec![self_neighbor], vec![]]),
            Err(GraphError::InconsistentRowLength {
                row: 1,
                expected: 1,
                actual: 0,
            })
        );
        assert!(matches!(
            fuzzy_simplicial_set(&[
                vec![Neighbor {
                    index: 1,
                    distance: 0.0,
                }],
                vec![Neighbor {
                    index: 0,
                    distance: 1.0,
                }],
            ]),
            Err(GraphError::MissingSelf { row: 0 })
        ));
        assert!(matches!(
            fuzzy_simplicial_set(&[
                vec![
                    Neighbor {
                        index: 0,
                        distance: 0.0,
                    },
                    Neighbor {
                        index: 0,
                        distance: 1.0,
                    },
                ],
                vec![
                    Neighbor {
                        index: 1,
                        distance: 0.0,
                    },
                    Neighbor {
                        index: 0,
                        distance: 1.0,
                    },
                ],
            ]),
            Err(GraphError::DuplicateNeighbor {
                row: 0,
                index: 0,
                ..
            })
        ));
    }

    #[test]
    fn rejects_invalid_distances_and_indices() {
        assert!(matches!(
            fuzzy_simplicial_set(&[vec![Neighbor {
                index: 0,
                distance: f32::NAN,
            }]]),
            Err(GraphError::InvalidDistance { row: 0, .. })
        ));
        assert!(matches!(
            fuzzy_simplicial_set(&[vec![Neighbor {
                index: 2,
                distance: 0.0,
            }]]),
            Err(GraphError::InvalidNeighborIndex {
                row: 0,
                index: 2,
                ..
            })
        ));
    }
}
