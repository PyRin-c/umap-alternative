//! Standalone smoothing and membership stages for ranked neighbor arrays.

use std::{fmt, ops::Sub};

use super::{MIN_SCALE_MULTIPLIER, bisect_scale, local_offset, mean_distance};

/// Borrowed floating-point neighbor or scale values in their input precision.
#[derive(Clone, Copy, Debug)]
pub enum KnnDistances<'a> {
    /// Values stored in float32.
    Float32(&'a [f32]),
    /// Values stored in float64.
    Float64(&'a [f64]),
}

impl KnnDistances<'_> {
    fn len(self) -> usize {
        match self {
            Self::Float32(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }
}

/// Parameters for smoothing a row-major matrix of ranked distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothKnnConfig {
    /// The neighbor count used in the base-two logarithmic target.
    pub k: f64,
    /// Maximum steps of scale bisection.
    pub iterations: usize,
    /// Number, possibly fractional, of locally connected neighbors.
    pub local_connectivity: f64,
    /// Multiplier for the target membership sum.
    pub bandwidth: f64,
}

/// Validation failure in a standalone neighbor stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnStageError {
    /// Declared dimensions and argument lengths disagree.
    Shape,
    /// Local connectivity must be nonnegative.
    LocalConnectivity,
    /// Row numbers in a membership coordinate list must fit in int32.
    RowIndex,
}

impl fmt::Display for KnnStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Shape => "neighbor buffers do not match the declared shape",
            Self::LocalConnectivity => "local_connectivity must be nonnegative",
            Self::RowIndex => "membership row indices exceed int32",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for KnnStageError {}

/// Compute a vector of scales and a vector of offsets, both in float32.
///
/// Neighbor order is retained. The scale search skips the first column.
/// Finite positive values determine local connectivity, and the floor uses
/// finite distances. Positive infinity adds no mass to the search.
pub fn smooth_knn_distances(
    distances: KnnDistances<'_>,
    shape: (usize, usize),
    config: SmoothKnnConfig,
) -> Result<(Vec<f32>, Vec<f32>), KnnStageError> {
    if shape.0.checked_mul(shape.1) != Some(distances.len()) {
        return Err(KnnStageError::Shape);
    }
    if config.local_connectivity.is_nan() || config.local_connectivity < 0.0 {
        return Err(KnnStageError::LocalConnectivity);
    }
    Ok(match distances {
        KnnDistances::Float32(distances) => smooth_rows(distances, shape, config),
        KnnDistances::Float64(distances) => smooth_rows(distances, shape, config),
    })
}

trait SmoothValue: Copy + Into<f64> + Sub<Output = Self> {
    fn from_offset(value: f32) -> Self;
    fn mean(values: &[Self]) -> f64;
    fn batch_mean(values: &[Self]) -> f64 {
        Self::mean(values)
    }
}

impl SmoothValue for f32 {
    fn from_offset(value: f32) -> Self {
        value
    }

    fn mean(values: &[Self]) -> f64 {
        mean_distance(values)
    }

    fn batch_mean(values: &[Self]) -> f64 {
        f64::from(Self::mean(values) as f32)
    }
}

impl SmoothValue for f64 {
    fn from_offset(value: f32) -> Self {
        f64::from(value)
    }

    fn mean(values: &[Self]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn smooth_rows<T: SmoothValue>(
    distances: &[T],
    (n_rows, width): (usize, usize),
    config: SmoothKnnConfig,
) -> (Vec<f32>, Vec<f32>) {
    let filtered;
    let finite = if distances.iter().all(|&value| value.into().is_finite()) {
        distances
    } else {
        filtered = distances
            .iter()
            .copied()
            .filter(|&value| value.into().is_finite())
            .collect::<Vec<_>>();
        &filtered
    };
    let batch_floor = MIN_SCALE_MULTIPLIER * T::batch_mean(finite);
    let target = config.k.log2() * config.bandwidth;
    let mut sigmas = Vec::with_capacity(n_rows);
    let mut rhos = Vec::with_capacity(n_rows);
    let mut positive = Vec::with_capacity(width);
    let mut finite_row = Vec::with_capacity(width);
    let mut shifts = Vec::with_capacity(width.saturating_sub(1));
    for row in 0..n_rows {
        let values = &distances[row * width..][..width];
        positive.clear();
        finite_row.clear();
        for &value in values {
            let distance = value.into();
            if distance.is_finite() {
                finite_row.push(value);
                if distance > 0.0 {
                    positive.push(value);
                }
            }
        }
        let rho = local_offset(&positive, config.local_connectivity);
        let offset = T::from_offset(rho);
        shifts.clear();
        shifts.extend(
            values
                .iter()
                .skip(1)
                .map(|&value| (value - offset).into().max(0.0)),
        );
        let sigma = bisect_scale(target, false, config.iterations, |sigma| {
            let sigma = f64::from(sigma);
            shifts.iter().map(|&shift| (-shift / sigma).exp()).sum()
        });
        let floor = if rho > 0.0 {
            MIN_SCALE_MULTIPLIER * T::mean(&finite_row)
        } else {
            batch_floor
        };
        sigmas.push(sigma.max(floor as f32));
        rhos.push(rho);
    }
    (sigmas, rhos)
}

/// Flat coordinate arrays for the unsymmetrized memberships.
#[derive(Debug, PartialEq)]
pub struct KnnMemberships {
    /// One row index per input neighbor slot.
    pub rows: Vec<i32>,
    /// One column index per input neighbor slot.
    pub columns: Vec<i32>,
    /// Corresponding float32 membership strengths.
    pub weights: Vec<f32>,
    /// Optional float32 distance for each slot.
    pub distances: Option<Vec<f32>>,
}

/// Convert ranked neighbors into flat directed memberships.
///
/// Index -1 leaves a zero placeholder. A square graph assigns zero strength
/// to self indices; a bipartite graph can use those columns. This function
/// preserves duplicates and slot order and performs no fuzzy set operations.
pub fn compute_knn_memberships(
    indices: &[i32],
    distances: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    (n_rows, width): (usize, usize),
    return_distances: bool,
    bipartite: bool,
) -> Result<KnnMemberships, KnnStageError> {
    if n_rows.checked_mul(width) != Some(indices.len())
        || distances.len() != indices.len()
        || sigmas.len() != n_rows
        || rhos.len() != n_rows
    {
        return Err(KnnStageError::Shape);
    }
    if n_rows.saturating_sub(1) > i32::MAX as usize {
        return Err(KnnStageError::RowIndex);
    }
    let count = indices.len();
    let mut result = KnnMemberships {
        rows: vec![0; count],
        columns: vec![0; count],
        weights: vec![0.0; count],
        distances: return_distances.then(|| vec![0.0; count]),
    };
    for (slot, &column) in indices.iter().enumerate() {
        if column == -1 {
            continue;
        }
        let row = slot / width;
        let distance = distances[slot];
        let sigma = sigmas[row];
        let shift = distance - rhos[row];
        result.rows[slot] = row as i32;
        result.columns[slot] = column;
        result.weights[slot] = if !bipartite && row as i32 == column {
            0.0
        } else if shift <= 0.0 || sigma == 0.0 {
            1.0
        } else {
            (-shift / sigma).exp()
        };
        if let Some(values) = &mut result.distances {
            values[slot] = distance;
        }
    }
    Ok(result)
}
