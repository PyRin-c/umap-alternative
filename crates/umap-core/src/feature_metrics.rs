//! Configurable real-valued feature metrics and their exhaustive search.

use crate::neighbors::{Neighbor, sort_closest_neighbors};
use crate::transform::QueryTieRanking;

/// A feature distance with its parameter buffers borrowed from the caller.
#[derive(Clone, Copy, Debug)]
pub enum FeatureMetric<'a> {
    /// Lp distance with exponent p.
    Minkowski(f64),
    /// Weighted Lp distance with exponent p and one weight per feature.
    WeightedMinkowski(f64, &'a [f64]),
    /// Euclidean distance scaled by per-feature variances.
    StandardizedEuclidean(&'a [f64]),
    /// Mahalanobis distance with a row-major inverse covariance matrix.
    Mahalanobis(&'a [f64]),
    /// Great-circle angular distance for latitude and longitude in radians.
    Haversine,
    /// Hyperbolic distance in the Poincaré ball.
    Poincare,
    /// Hellinger distance between normalized nonnegative feature rows.
    Hellinger,
    /// Hellinger distance between the softmax distributions of the rows.
    SoftmaxHellinger,
    /// Symmetric KL divergence between smoothed feature distributions.
    SymmetricKl(f64),
}

impl FeatureMetric<'_> {
    /// Check that the metric parameters apply to vectors of the given width.
    pub fn validate(self, width: usize) -> Result<(), &'static str> {
        match self {
            Self::Minkowski(p) | Self::WeightedMinkowski(p, _) if !(p.is_finite() && p > 0.0) => {
                Err("Minkowski p must be finite and positive")
            }
            Self::WeightedMinkowski(_, weights)
                if weights.len() != width || weights.iter().any(|w| !w.is_finite() || *w < 0.0) =>
            {
                Err("w must contain one finite nonnegative weight per feature")
            }
            Self::StandardizedEuclidean(variance)
                if variance.len() != width
                    || variance.iter().any(|v| !v.is_finite() || *v <= 0.0) =>
            {
                Err("sigma must contain one finite positive variance per feature")
            }
            Self::Mahalanobis(vinv)
                if width.checked_mul(width) != Some(vinv.len())
                    || vinv.iter().any(|value| !value.is_finite()) =>
            {
                Err("vinv must be a finite square matrix matching the feature width")
            }
            Self::Haversine if width != 2 => Err("Haversine requires latitude and longitude"),
            Self::SymmetricKl(z) if !(z.is_finite() && z > 0.0) => {
                Err("symmetric KL smoothing z must be finite and positive")
            }
            _ => Ok(()),
        }
    }

    /// Measure the distance between two finite, equally sized feature rows.
    pub fn distance(self, left: &[f64], right: &[f64]) -> Result<f64, &'static str> {
        if left.len() != right.len() || left.is_empty() {
            return Err("distance requires equal positive feature widths");
        }
        self.validate(left.len())?;
        if !left.iter().chain(right).all(|value| value.is_finite()) {
            return Err("metric inputs must be finite");
        }
        let distance = self.distance_unchecked(left, right);
        if distance.is_nan() {
            return Err("distance is undefined for the given features");
        }
        Ok(distance)
    }

    fn distance_unchecked(self, left: &[f64], right: &[f64]) -> f64 {
        let differences = left.iter().zip(right).map(|(a, b)| a - b);
        match self {
            Self::Minkowski(p) => differences
                .map(|d| d.abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p),
            Self::WeightedMinkowski(p, weights) => differences
                .zip(weights)
                .map(|(d, w)| w * d.abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p),
            Self::StandardizedEuclidean(variance) => differences
                .zip(variance)
                .map(|(d, v)| d * d / v)
                .sum::<f64>()
                .sqrt(),
            Self::Mahalanobis(vinv) => {
                let diff: Vec<_> = differences.collect();
                vinv.chunks_exact(diff.len())
                    .zip(&diff)
                    .map(|(row, d)| {
                        d * row
                            .iter()
                            .zip(&diff)
                            .map(|(v, other)| v * other)
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    .sqrt()
            }
            Self::Haversine => {
                let latitude = ((left[0] - right[0]) * 0.5).sin();
                let longitude = ((left[1] - right[1]) * 0.5).sin();
                let chord =
                    latitude * latitude + left[0].cos() * right[0].cos() * longitude * longitude;
                2.0 * chord.clamp(0.0, 1.0).sqrt().asin()
            }
            Self::Poincare => {
                let a = 1.0 - left.iter().map(|value| value * value).sum::<f64>();
                let b = 1.0 - right.iter().map(|value| value * value).sum::<f64>();
                if a <= 0.0 || b <= 0.0 {
                    return f64::NAN;
                }
                let squared = differences.map(|d| d * d).sum::<f64>();
                (1.0 + 2.0 * squared / (a * b)).acosh()
            }
            Self::Hellinger => hellinger(left.iter().copied(), right.iter().copied()),
            Self::SoftmaxHellinger => {
                let left_max = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let right_max = right.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                hellinger(
                    left.iter().map(|value| (value - left_max).exp()),
                    right.iter().map(|value| (value - right_max).exp()),
                )
            }
            Self::SymmetricKl(z) => {
                if left.iter().chain(right).any(|value| *value < 0.0) {
                    return f64::NAN;
                }
                let a_total = left.iter().map(|a| a + z).sum::<f64>();
                let b_total = right.iter().map(|b| b + z).sum::<f64>();
                let divergence = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| {
                        let p = (a + z) / a_total;
                        let q = (b + z) / b_total;
                        (p - q) * (p.ln() - q.ln())
                    })
                    .sum::<f64>();
                0.5 * divergence
            }
        }
    }
}

fn hellinger(left: impl Iterator<Item = f64>, right: impl Iterator<Item = f64>) -> f64 {
    let (mut left_total, mut right_total, mut overlap) = (0.0, 0.0, 0.0);
    for (a, b) in left.zip(right) {
        if a < 0.0 || b < 0.0 {
            return f64::NAN;
        }
        left_total += a;
        right_total += b;
        let product = a * b;
        overlap += if product.is_normal() {
            product.sqrt()
        } else {
            a.sqrt() * b.sqrt()
        };
    }
    match (left_total == 0.0, right_total == 0.0) {
        (true, true) => 0.0,
        (true, false) | (false, true) => 1.0,
        (false, false) => {
            let product = left_total * right_total;
            let scale = if product.is_normal() {
                product.sqrt()
            } else {
                left_total.sqrt() * right_total.sqrt()
            };
            (1.0 - overlap / scale).max(0.0).sqrt()
        }
    }
}

/// Compact, row-major neighbor search result.
#[derive(Debug)]
pub struct RankedFeatureRows {
    /// Candidate column indices, with -1 for cut entries.
    pub columns: Vec<i32>,
    /// Candidate distances, with infinity for cut entries.
    pub distances: Vec<f32>,
    /// Count of distances discarded by the cutoff, including unselected distances.
    pub removed: usize,
}

/// Rank features using a reusable candidate buffer and retaining only k entries per row.
///
/// With no query matrix, each row ranks the training set and places itself first
/// before applying the cutoff. Otherwise every query ranks the training set.
pub fn rank_feature_rows(
    metric: FeatureMetric<'_>,
    training: &[f64],
    queries: Option<&[f64]>,
    width: usize,
    k: usize,
    cutoff: f64,
) -> Result<RankedFeatureRows, &'static str> {
    rank_feature_rows_with_tie_ranking(
        metric,
        training,
        queries,
        width,
        k,
        cutoff,
        None::<fn(&[f32], usize) -> Vec<usize>>,
    )
}

/// Rank feature rows, optionally resolving held-out distance ties with a caller's ordering.
///
/// The callback receives distances in training-column order and returns the closest
/// k columns in the desired order. Training ranks retain deterministic index ordering.
pub fn rank_feature_rows_with_tie_ranking(
    metric: FeatureMetric<'_>,
    training: &[f64],
    queries: Option<&[f64]>,
    width: usize,
    k: usize,
    cutoff: f64,
    mut rank_ties: Option<impl FnMut(&[f32], usize) -> Vec<usize>>,
) -> Result<RankedFeatureRows, &'static str> {
    if width == 0 || !training.len().is_multiple_of(width) {
        return Err("invalid training feature shape");
    }
    let n_training = training.len() / width;
    if k == 0 || k > n_training || n_training > i32::MAX as usize {
        return Err("neighbor count or training row count is out of range");
    }
    if cutoff.is_nan() || cutoff < 0.0 {
        return Err("disconnection distance must be nonnegative");
    }
    let training_rows = queries.is_none();
    let queries = queries.unwrap_or(training);
    if !queries.len().is_multiple_of(width) {
        return Err("invalid query feature shape");
    }
    metric.validate(width)?;
    if !training
        .iter()
        .chain(queries)
        .all(|value| value.is_finite())
    {
        return Err("metric inputs must be finite");
    }
    let capacity = (queries.len() / width)
        .checked_mul(k)
        .ok_or("neighbor result is too large")?;
    let mut result = RankedFeatureRows {
        columns: Vec::with_capacity(capacity),
        distances: Vec::with_capacity(capacity),
        removed: 0,
    };
    let mut candidates = Vec::with_capacity(n_training);
    let mut tie_ranking = QueryTieRanking::default();
    for (row, query) in queries.chunks_exact(width).enumerate() {
        candidates.clear();
        for (index, other) in training.chunks_exact(width).enumerate() {
            if training_rows && index == row {
                continue;
            }
            let distance = metric.distance_unchecked(query, other);
            if distance.is_nan() {
                return Err("distance is undefined for the given features");
            }
            let cut = distance >= cutoff || (distance as f32).is_infinite();
            result.removed += usize::from(cut);
            candidates.push(Neighbor {
                index,
                distance: if cut { f32::INFINITY } else { distance as f32 },
            });
        }
        let k = if training_rows {
            let cut = 0.0 >= cutoff;
            result.removed += usize::from(cut);
            result.columns.push(if cut { -1 } else { row as i32 });
            result.distances.push(if cut { f32::INFINITY } else { 0.0 });
            k - 1
        } else {
            k
        };
        let neighbors = if !training_rows && let Some(rank_ties) = rank_ties.as_mut() {
            tie_ranking
                .select(row, &mut candidates, k, true, rank_ties)
                .map_err(|_| "invalid configured feature tie ranking")?;
            &candidates[..k]
        } else {
            sort_closest_neighbors(&mut candidates, k)
        };
        for neighbor in neighbors {
            result.columns.push(if neighbor.distance.is_infinite() {
                -1
            } else {
                neighbor.index as i32
            });
            result.distances.push(neighbor.distance);
        }
    }
    Ok(result)
}
