//! Exhaustive nearest-neighbor search over a dense row-major matrix.
//!
//! This module intentionally implements the simple, exhaustive primitive used
//! by later UMAP stages.  It does not build a graph or perform any layout
//! optimization.

use std::cmp::Ordering;
use std::fmt;

use crate::fuzzy_graph::GraphError;

/// One member of a query row's nearest-neighbor result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Row index of the candidate in the input matrix.
    pub index: usize,
    /// Distance from the query row to [`Neighbor::index`].
    pub distance: f32,
}

/// Distance functions supported by exhaustive nearest-neighbor search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Manhattan (city-block or L1) distance.
    Manhattan,
    /// Chebyshev (L-infinity) distance.
    Chebyshev,
    /// One minus cosine similarity.
    Cosine,
    /// One minus the cosine similarity of mean-centered rows.
    Correlation,
    /// Sum of coordinate differences divided by the corresponding absolute sums.
    Canberra,
    /// Sum of absolute differences divided by the sum of absolute pair sums.
    BrayCurtis,
}

/// Input validation failures for [`exhaustive_knn`] and
/// [`exhaustive_knn_with_metric`].
#[derive(Clone, Debug, PartialEq)]
pub enum KnnError {
    /// `n_rows * n_features` overflowed `usize`, or did not equal `data.len()`.
    InvalidShape {
        /// Number of rows requested by the caller.
        n_rows: usize,
        /// Number of features requested by the caller.
        n_features: usize,
        /// Number of values supplied in `data`.
        values: usize,
    },
    /// `k` must be at least one and no greater than the number of rows.
    InvalidK { k: usize, n_rows: usize },
    /// The selected distance is undefined for a non-finite matrix value.
    NonFiniteValue { index: usize, value: f32 },
}

impl fmt::Display for KnnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_rows,
                n_features,
                values,
            } => write!(
                formatter,
                "matrix shape {n_rows}x{n_features} does not match {values} values"
            ),
            Self::InvalidK { k, n_rows } => {
                write!(formatter, "k={k} is outside the valid range 1..={n_rows}")
            }
            Self::NonFiniteValue { index, value } => {
                write!(
                    formatter,
                    "matrix value at flat index {index} is not finite: {value}"
                )
            }
        }
    }
}

impl std::error::Error for KnnError {}

/// Validation failures returned by [`precomputed_knn`].
///
/// Shape, `k`, and non-finite-value failures retain the existing
/// [`KnnError`] contract. Negative distances and non-zero diagonal entries
/// use the existing [`GraphError`] distance validation variants so callers can
/// preserve the dense graph fit error surface without adding a new branch to
/// unrelated consumers.
#[derive(Clone, Debug, PartialEq)]
pub enum PrecomputedKnnError {
    /// The matrix shape, `k`, or finite-value validation failed.
    Knn(KnnError),
    /// A distance or diagonal entry violates the fuzzy graph input contract.
    Graph(GraphError),
}

impl fmt::Display for PrecomputedKnnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Knn(error) => write!(formatter, "precomputed neighbor search failed: {error}"),
            Self::Graph(error) => {
                write!(formatter, "precomputed distance validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for PrecomputedKnnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Knn(error) => Some(error),
            Self::Graph(error) => Some(error),
        }
    }
}

impl From<KnnError> for PrecomputedKnnError {
    fn from(error: KnnError) -> Self {
        Self::Knn(error)
    }
}

impl From<GraphError> for PrecomputedKnnError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

/// Find the `k` nearest rows for every row of a dense matrix.
///
/// `data` is interpreted as a contiguous row-major matrix with shape
/// `n_rows x n_features`.  The returned outer vector has one entry per query
/// row, and each inner vector is sorted from nearest to farthest.  Every query
/// row, including the query itself, is considered as a candidate, so a valid
/// `k` is in `1..=n_rows`.
///
/// Distances are computed from the `f32` inputs with `f32` subtraction and
/// squaring, followed by `f64` accumulation (in feature order) and square
/// root, then rounded to `f32` for the returned neighbor record. Ties are ordered
/// deterministically by ascending candidate row index after [`f32::total_cmp`]
/// distance ordering. Matrix values must be finite; reject non-finite values
/// rather than returning an undefined distance.
pub fn exhaustive_knn(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
) -> Result<Vec<Vec<Neighbor>>, KnnError> {
    exhaustive_knn_with_metric(data, n_rows, n_features, k, DistanceMetric::Euclidean)
}

/// Find the `k` nearest rows for every row of a dense matrix using `metric`.
///
/// `data` is interpreted as a contiguous row-major matrix with shape
/// `n_rows x n_features`.  The returned outer vector has one entry per query
/// row, and each inner vector is sorted from nearest to farthest.  Every query
/// row, including the query itself, is considered as a candidate, so a valid
/// `k` is in `1..=n_rows`.
///
/// Euclidean distances use `f32` subtraction and squaring, followed by `f64`
/// accumulation (in feature order) and square root before rounding to `f32`.
/// Manhattan distances sum the absolute `f32` coordinate differences in
/// `f64`, and Chebyshev distances select the largest absolute coordinate
/// difference. Both are
/// rounded to `f32` for the returned neighbor record. Cosine accumulates
/// float32 coordinate products in float64. Correlation centers the rows and
/// computes products and sums in float64. Both use distance zero when both
/// norms are zero and distance one when only one norm is zero. Rounding can
/// give a small negative distance between nearly parallel rows. Ties are ordered
/// deterministically by ascending candidate row index after [`f32::total_cmp`]
/// distance ordering. Matrix values must be finite. Cosine products can
/// overflow at large magnitudes, yielding NaN for some candidate distances.
pub fn exhaustive_knn_with_metric(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<Vec<Vec<Neighbor>>, KnnError> {
    dense_knn_counting_cutoff(data, n_rows, n_features, k, metric, f32::INFINITY)
        .map(|(neighbors, _)| neighbors)
}

pub(crate) fn dense_knn_counting_cutoff(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
) -> Result<(Vec<Vec<Neighbor>>, usize), KnnError> {
    dense_knn_by(
        data,
        n_rows,
        n_features,
        k,
        cutoff,
        metric.is_angular(),
        |left, right| metric.distance(left, right),
    )
}

pub(crate) fn exhaustive_manhattan_target_knn(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
) -> Result<Vec<Vec<Neighbor>>, KnnError> {
    dense_knn_by(
        data,
        n_rows,
        n_features,
        k,
        f32::INFINITY,
        false,
        |left, right| {
            // Continuous Manhattan targets use wide subtraction.
            left.iter()
                .zip(right)
                .map(|(&left, &right)| (f64::from(left) - f64::from(right)).abs())
                .sum::<f64>() as f32
        },
    )
    .map(|(neighbors, _)| neighbors)
}

fn dense_knn_by(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    cutoff: f32,
    zero_diagonal: bool,
    distance: impl Fn(&[f32], &[f32]) -> f32,
) -> Result<(Vec<Vec<Neighbor>>, usize), KnnError> {
    let expected_values = n_rows.checked_mul(n_features);
    if expected_values != Some(data.len()) {
        return Err(KnnError::InvalidShape {
            n_rows,
            n_features,
            values: data.len(),
        });
    }

    if !(1..=n_rows).contains(&k) {
        return Err(KnnError::InvalidK { k, n_rows });
    }

    if let Some((index, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KnnError::NonFiniteValue { index, value });
    }

    let mut removed = 0;
    let neighbors = collect_neighbors(n_rows, k, |query_index, candidates| {
        let query_start = query_index * n_features;
        let query = &data[query_start..query_start + n_features];

        for candidate_index in 0..n_rows {
            let candidate_start = candidate_index * n_features;
            let candidate = &data[candidate_start..candidate_start + n_features];
            let distance = if zero_diagonal && candidate_index == query_index {
                0.0
            } else {
                distance(query, candidate)
            };
            removed += usize::from(distance >= cutoff);
            candidates.push(Neighbor {
                index: candidate_index,
                distance,
            });
        }
    });
    Ok((neighbors, removed))
}

/// Find the `k` nearest rows for every row of a precomputed distance matrix.
///
/// `distances` is interpreted as a contiguous row-major square matrix with
/// shape `n_rows x n_rows`. Every entry must be finite and non-negative, and
/// every diagonal entry must be exactly zero. Distances are consumed as
/// provided: rows may be asymmetric, and no symmetry check or averaging is
/// performed. The returned rows use the same deterministic ordering as the
/// metric-based search (`f32::total_cmp`, followed by ascending candidate
/// index).
///
/// This is a graph-stage primitive. It does not infer distances from feature
/// vectors or alter the requested neighbor count.
pub fn precomputed_knn(
    distances: &[f32],
    n_rows: usize,
    k: usize,
) -> Result<Vec<Vec<Neighbor>>, PrecomputedKnnError> {
    let expected_values = n_rows.checked_mul(n_rows);
    if expected_values != Some(distances.len()) {
        return Err(KnnError::InvalidShape {
            n_rows,
            n_features: n_rows,
            values: distances.len(),
        }
        .into());
    }

    if !(1..=n_rows).contains(&k) {
        return Err(KnnError::InvalidK { k, n_rows }.into());
    }

    if let Some((index, distance)) = distances
        .iter()
        .copied()
        .enumerate()
        .find(|(_, distance)| !distance.is_finite())
    {
        return Err(KnnError::NonFiniteValue {
            index,
            value: distance,
        }
        .into());
    }

    if let Some((index, distance)) = distances
        .iter()
        .copied()
        .enumerate()
        .find(|(_, distance)| *distance < 0.0)
    {
        return Err(GraphError::InvalidDistance {
            row: index / n_rows,
            position: index % n_rows,
            index: index % n_rows,
            distance,
        }
        .into());
    }

    for row in 0..n_rows {
        let diagonal = distances[row * n_rows + row];
        if diagonal != 0.0 {
            return Err(GraphError::InvalidSelfDistance {
                row,
                position: row,
                distance: diagonal,
            }
            .into());
        }
    }

    Ok(collect_neighbors(n_rows, k, |query_index, candidates| {
        let row_start = query_index * n_rows;
        let row = &distances[row_start..row_start + n_rows];
        candidates.extend(
            row.iter()
                .copied()
                .enumerate()
                .map(|(index, distance)| Neighbor { index, distance }),
        );
    }))
}

pub(crate) fn collect_neighbors(
    n_rows: usize,
    k: usize,
    mut fill_candidates: impl FnMut(usize, &mut Vec<Neighbor>),
) -> Vec<Vec<Neighbor>> {
    let mut result = Vec::with_capacity(n_rows);
    if k == n_rows {
        for query_index in 0..n_rows {
            let mut candidates = Vec::with_capacity(n_rows);
            fill_candidates(query_index, &mut candidates);
            sort_closest_neighbors(&mut candidates, k);
            result.push(candidates);
        }
    } else {
        let mut candidates = Vec::with_capacity(n_rows);
        for query_index in 0..n_rows {
            candidates.clear();
            fill_candidates(query_index, &mut candidates);
            result.push(sort_closest_neighbors(&mut candidates, k).to_vec());
        }
    }
    result
}

pub(crate) fn sort_closest_neighbors(candidates: &mut [Neighbor], k: usize) -> &[Neighbor] {
    let closest = if k < candidates.len() {
        let (prefix, _, _) = candidates.select_nth_unstable_by(k, compare_neighbors);
        prefix
    } else {
        candidates
    };
    closest.sort_unstable_by(compare_neighbors);
    closest
}

fn compare_neighbors(left: &Neighbor, right: &Neighbor) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.index.cmp(&right.index))
}

impl DistanceMetric {
    pub(crate) fn is_angular(self) -> bool {
        matches!(self, Self::Cosine | Self::Correlation)
    }

    /// Return the finite default cutoff for cosine and correlation graphs.
    pub fn default_disconnection_distance(self) -> Option<f32> {
        self.is_angular().then_some(2.0)
    }

    pub(crate) fn uses_ranked_graph(self) -> bool {
        self.is_angular() || matches!(self, Self::Canberra | Self::BrayCurtis)
    }

    pub(crate) fn distance(self, left: &[f32], right: &[f32]) -> f32 {
        match self {
            Self::Euclidean => euclidean_distance(left, right),
            Self::Manhattan => manhattan_distance(left, right),
            Self::Chebyshev => chebyshev_distance(left, right),
            Self::Cosine => cosine_distance(left, right),
            Self::Correlation => correlation_distance(left, right),
            Self::Canberra => {
                canberra_distance_pairs(left.iter().copied().zip(right.iter().copied())) as f32
            }
            Self::BrayCurtis => {
                bray_curtis_distance_pairs(left.iter().copied().zip(right.iter().copied())) as f32
            }
        }
    }
}

pub(crate) fn canberra_distance_pairs(values: impl Iterator<Item = (f32, f32)>) -> f64 {
    values.fold(0.0, |distance, (left, right)| {
        let divisor = left.abs() + right.abs();
        if divisor == 0.0 {
            distance
        } else {
            distance + f64::from((left - right).abs() / divisor)
        }
    })
}

pub(crate) fn bray_curtis_sums(values: impl Iterator<Item = (f32, f32)>) -> (f64, f64) {
    values.fold((0.0, 0.0), |(numerator, denominator), (left, right)| {
        (
            numerator + f64::from((left - right).abs()),
            denominator + f64::from((left + right).abs()),
        )
    })
}

pub(crate) fn bray_curtis_distance_pairs(values: impl Iterator<Item = (f32, f32)>) -> f64 {
    let (numerator, denominator) = bray_curtis_sums(values);
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let (dot, left_squared, right_squared) = cosine_products(left, right);
    angular_distance(dot, left_squared, right_squared)
}

pub(crate) fn cosine_products(left: &[f32], right: &[f32]) -> (f64, f64, f64) {
    cosine_products_pairs(left.iter().copied().zip(right.iter().copied()))
}

pub(crate) fn cosine_distance_pairs(values: impl Iterator<Item = (f32, f32)>) -> f32 {
    let (dot, left_squared, right_squared) = cosine_products_pairs(values);
    angular_distance(dot, left_squared, right_squared)
}

fn cosine_products_pairs(values: impl Iterator<Item = (f32, f32)>) -> (f64, f64, f64) {
    let mut dot = 0.0_f64;
    let mut left_squared = 0.0_f64;
    let mut right_squared = 0.0_f64;
    for (x, y) in values {
        dot += f64::from(x * y);
        left_squared += f64::from(x * x);
        right_squared += f64::from(y * y);
    }
    (dot, left_squared, right_squared)
}

fn correlation_distance(left: &[f32], right: &[f32]) -> f32 {
    let ((dot, left_squared, right_squared), _) = correlation_products(left, right);
    angular_distance(dot, left_squared, right_squared)
}

pub(crate) fn correlation_products(left: &[f32], right: &[f32]) -> ((f64, f64, f64), (f64, f64)) {
    if left.is_empty() {
        return ((0.0, 0.0, 0.0), (0.0, 0.0));
    }
    let left_mean = left.iter().map(|&x| f64::from(x)).sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().map(|&y| f64::from(y)).sum::<f64>() / right.len() as f64;
    let mut dot = 0.0_f64;
    let mut left_squared = 0.0_f64;
    let mut right_squared = 0.0_f64;
    for (&x, &y) in left.iter().zip(right) {
        let x = f64::from(x) - left_mean;
        let y = f64::from(y) - right_mean;
        dot += x * y;
        left_squared += x * x;
        right_squared += y * y;
    }
    ((dot, left_squared, right_squared), (left_mean, right_mean))
}

fn angular_distance(dot: f64, left_squared: f64, right_squared: f64) -> f32 {
    match (left_squared == 0.0, right_squared == 0.0) {
        (true, true) => 0.0,
        (true, false) | (false, true) => 1.0,
        (false, false) => {
            let distance = (1.0 - dot / (left_squared * right_squared).sqrt()) as f32;
            if distance.is_nan() {
                f32::NAN
            } else {
                distance
            }
        }
    }
}

fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    euclidean_distance_wide(left, right) as f32
}

pub(crate) fn euclidean_distance_wide(left: &[f32], right: &[f32]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    euclidean_distance_pairs(left.iter().copied().zip(right.iter().copied()))
}

pub(crate) fn euclidean_distance_pairs(values: impl Iterator<Item = (f32, f32)>) -> f64 {
    let squared_sum = values.fold(0.0_f64, |sum, (left_value, right_value)| {
        let difference = left_value - right_value;
        sum + f64::from(difference * difference)
    });
    squared_sum.sqrt()
}

fn manhattan_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    manhattan_distance_pairs(left.iter().copied().zip(right.iter().copied()))
}

pub(crate) fn manhattan_distance_pairs(values: impl Iterator<Item = (f32, f32)>) -> f32 {
    manhattan_distance_pairs_wide(values) as f32
}

pub(crate) fn manhattan_distance_pairs_wide(values: impl Iterator<Item = (f32, f32)>) -> f64 {
    values.fold(0.0_f64, |sum, (left_value, right_value)| {
        sum + f64::from((left_value - right_value).abs())
    })
}

fn chebyshev_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(0.0_f64, |maximum, (&left_value, &right_value)| {
            let difference = (f64::from(left_value) - f64::from(right_value)).abs();
            maximum.max(difference)
        }) as f32
}

#[cfg(test)]
mod tests {
    use super::{
        DistanceMetric, KnnError, Neighbor, PrecomputedKnnError, exhaustive_knn,
        exhaustive_knn_with_metric, precomputed_knn,
    };

    #[test]
    fn computes_analytic_euclidean_distances() {
        let data = [0.0, 0.0, 3.0, 4.0, 0.0, 4.0];
        let neighbors = exhaustive_knn(&data, 3, 2, 3).expect("valid matrix");

        assert_eq!(
            neighbors[0],
            vec![
                Neighbor {
                    index: 0,
                    distance: 0.0,
                },
                Neighbor {
                    index: 2,
                    distance: 4.0,
                },
                Neighbor {
                    index: 1,
                    distance: 5.0,
                },
            ]
        );
        assert_eq!(
            neighbors[1],
            vec![
                Neighbor {
                    index: 1,
                    distance: 0.0,
                },
                Neighbor {
                    index: 2,
                    distance: 3.0,
                },
                Neighbor {
                    index: 0,
                    distance: 5.0,
                },
            ]
        );
    }

    #[test]
    fn rounds_coordinate_products_before_f64_accumulation() {
        let data = [0.12_f32, 0.03, 0.29, 0.18];
        let neighbors = exhaustive_knn(&data, 2, 2, 2).expect("valid matrix");
        assert_eq!(neighbors[0][1].distance.to_bits(), 0x3e68_2827);
        assert_eq!(neighbors[1][1].distance.to_bits(), 0x3e68_2827);
    }

    #[test]
    fn cosine_uses_f32_products_and_norm_degeneracy() {
        let cosine = DistanceMetric::Cosine;
        assert_eq!(cosine.distance(&[0.1; 3], &[1.0; 3]).to_bits(), 0x32af_ffff);
        assert_eq!(cosine.distance(&[0.0; 3], &[-0.0; 3]), 0.0);
        assert_eq!(cosine.distance(&[0.0; 3], &[1.0; 3]), 1.0);
        assert_eq!(cosine.distance(&[1.0, 0.0, -1.0], &[-1.0, 0.0, 1.0]), 2.0);
    }

    #[test]
    fn correlation_centers_each_row_and_handles_constant_rows() {
        let correlation = DistanceMetric::Correlation;
        assert_eq!(correlation.distance(&[1.0; 3], &[-2.0; 3]), 0.0);
        assert_eq!(correlation.distance(&[1.0; 3], &[1.0, 0.0, -1.0]), 1.0);
        assert_eq!(
            correlation.distance(&[1.0, 0.0, -1.0], &[3.0, 2.0, 1.0]),
            0.0
        );
        assert_eq!(
            correlation.distance(&[1.0, 0.0, -1.0], &[1.0, 2.0, 3.0]),
            2.0
        );
    }

    #[test]
    fn metric_selection_changes_order_and_reports_analytic_distances() {
        let data = [0.0, 0.0, 4.0, 1.0, 3.0, 3.0];
        let euclidean = exhaustive_knn(&data, 3, 2, 3).expect("valid matrix");
        let euclidean_from_selection =
            exhaustive_knn_with_metric(&data, 3, 2, 3, DistanceMetric::Euclidean)
                .expect("valid matrix");
        let manhattan = exhaustive_knn_with_metric(&data, 3, 2, 3, DistanceMetric::Manhattan)
            .expect("valid matrix");
        let chebyshev = exhaustive_knn_with_metric(&data, 3, 2, 3, DistanceMetric::Chebyshev)
            .expect("valid matrix");

        assert_eq!(euclidean, euclidean_from_selection);
        assert_eq!(
            euclidean[0]
                .iter()
                .map(|neighbor| neighbor.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            manhattan[0],
            vec![
                Neighbor {
                    index: 0,
                    distance: 0.0,
                },
                Neighbor {
                    index: 1,
                    distance: 5.0,
                },
                Neighbor {
                    index: 2,
                    distance: 6.0,
                },
            ]
        );
        assert_eq!(
            chebyshev[0],
            vec![
                Neighbor {
                    index: 0,
                    distance: 0.0,
                },
                Neighbor {
                    index: 2,
                    distance: 3.0,
                },
                Neighbor {
                    index: 1,
                    distance: 4.0,
                },
            ]
        );
    }

    #[test]
    fn singleton_includes_itself() {
        let neighbors = exhaustive_knn(&[3.0, -4.0], 1, 2, 1).expect("valid matrix");

        assert_eq!(
            neighbors,
            vec![vec![Neighbor {
                index: 0,
                distance: 0.0,
            }]]
        );
    }

    #[test]
    fn ties_and_duplicates_are_ordered_by_index() {
        let data = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let neighbors = exhaustive_knn(&data, 4, 2, 4).expect("valid matrix");

        assert_eq!(
            neighbors[0]
                .iter()
                .map(|neighbor| neighbor.index)
                .collect::<Vec<_>>(),
            vec![0, 3, 1, 2]
        );
        assert_eq!(
            neighbors[1]
                .iter()
                .map(|neighbor| neighbor.index)
                .collect::<Vec<_>>(),
            vec![1, 0, 3, 2]
        );
    }

    #[test]
    fn non_euclidean_ties_are_ordered_by_index() {
        let data = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        for metric in [DistanceMetric::Manhattan, DistanceMetric::Chebyshev] {
            let neighbors =
                exhaustive_knn_with_metric(&data, 4, 2, 4, metric).expect("valid matrix");

            assert_eq!(
                neighbors[0]
                    .iter()
                    .map(|neighbor| neighbor.index)
                    .collect::<Vec<_>>(),
                vec![0, 3, 1, 2]
            );
        }
    }

    #[test]
    fn accepts_a_zero_feature_matrix() {
        let neighbors = exhaustive_knn(&[], 3, 0, 3).expect("valid zero-feature matrix");

        assert_eq!(
            neighbors
                .iter()
                .map(|row| row
                    .iter()
                    .map(|neighbor| neighbor.index)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]]
        );
        assert!(
            neighbors
                .iter()
                .flatten()
                .all(|neighbor| neighbor.distance == 0.0)
        );
    }

    #[test]
    fn rejects_invalid_shape_and_k() {
        assert_eq!(
            exhaustive_knn(&[0.0, 1.0], 2, 2, 1),
            Err(KnnError::InvalidShape {
                n_rows: 2,
                n_features: 2,
                values: 2,
            })
        );
        assert_eq!(
            exhaustive_knn(&[0.0, 1.0], 2, 1, 0),
            Err(KnnError::InvalidK { k: 0, n_rows: 2 })
        );
        assert_eq!(
            exhaustive_knn(&[0.0, 1.0], 2, 1, 3),
            Err(KnnError::InvalidK { k: 3, n_rows: 2 })
        );
    }

    #[test]
    fn rejects_non_finite_values() {
        assert!(matches!(
            exhaustive_knn(&[0.0, f32::NAN], 1, 2, 1),
            Err(KnnError::NonFiniteValue { index: 1, value }) if value.is_nan()
        ));
        assert_eq!(
            exhaustive_knn(&[0.0, f32::INFINITY], 1, 2, 1),
            Err(KnnError::NonFiniteValue {
                index: 1,
                value: f32::INFINITY,
            })
        );
    }

    #[test]
    fn every_metric_rejects_non_finite_values() {
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Chebyshev,
        ] {
            assert!(matches!(
                exhaustive_knn_with_metric(&[0.0, f32::NAN], 1, 2, 1, metric),
                Err(KnnError::NonFiniteValue { index: 1, value }) if value.is_nan()
            ));
        }
    }

    #[test]
    fn precomputed_rows_keep_ordered_asymmetric_distances() {
        let distances = [
            0.0, 5.0, 1.0, // row 0
            2.0, 0.0, 4.0, // row 1
            3.0, 1.0, 0.0, // row 2
        ];
        let neighbors = precomputed_knn(&distances, 3, 3).expect("valid distance matrix");

        assert_eq!(
            neighbors
                .iter()
                .map(|row| row
                    .iter()
                    .map(|neighbor| neighbor.index)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![0, 2, 1], vec![1, 0, 2], vec![2, 1, 0]]
        );
        assert_eq!(neighbors[0][1].distance, 1.0);
        assert_eq!(neighbors[1][1].distance, 2.0);
    }

    #[test]
    fn precomputed_validation_rejects_negative_nonfinite_and_nonzero_diagonal_values() {
        assert_eq!(
            precomputed_knn(&[0.0, -1.0, 1.0, 0.0], 2, 2),
            Err(PrecomputedKnnError::Graph(
                crate::fuzzy_graph::GraphError::InvalidDistance {
                    row: 0,
                    position: 1,
                    index: 1,
                    distance: -1.0,
                },
            ))
        );
        assert!(matches!(
            precomputed_knn(&[0.0, f32::NAN, 1.0, 0.0], 2, 2),
            Err(PrecomputedKnnError::Knn(KnnError::NonFiniteValue { index: 1, value }))
                if value.is_nan()
        ));
        assert_eq!(
            precomputed_knn(&[0.0, 1.0, 2.0, 0.5], 2, 2),
            Err(PrecomputedKnnError::Graph(
                crate::fuzzy_graph::GraphError::InvalidSelfDistance {
                    row: 1,
                    position: 1,
                    distance: 0.5,
                },
            ))
        );
    }
}
