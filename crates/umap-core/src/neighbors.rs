//! Exhaustive nearest-neighbor search over a dense row-major matrix.
//!
//! This module intentionally implements the simple, exhaustive primitive used
//! by later UMAP stages.  It does not build a graph or perform any layout
//! optimization.

use std::fmt;

/// One member of a query row's nearest-neighbor result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Row index of the candidate in the input matrix.
    pub index: usize,
    /// Euclidean distance from the query row to [`Neighbor::index`].
    pub distance: f32,
}

/// Input validation failures for [`exhaustive_knn`].
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
    /// Euclidean distance is undefined for a non-finite matrix value.
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

/// Find the `k` nearest rows for every row of a dense matrix.
///
/// `data` is interpreted as a contiguous row-major matrix with shape
/// `n_rows x n_features`.  The returned outer vector has one entry per query
/// row, and each inner vector is sorted from nearest to farthest.  Every query
/// row, including the query itself, is considered as a candidate, so a valid
/// `k` is in `1..=n_rows`.
///
/// Distances are computed as Euclidean distances using `f32` subtraction,
/// multiplication, accumulation (in feature order), and square root.  This
/// makes rounding follow ordinary `f32` arithmetic but does not promise
/// bit-for-bit equivalence with another implementation or hardware.  Ties are
/// ordered deterministically by ascending candidate row index after
/// [`f32::total_cmp`] distance ordering.  Matrix values must be finite; reject
/// non-finite values rather than returning an undefined distance.
pub fn exhaustive_knn(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
) -> Result<Vec<Vec<Neighbor>>, KnnError> {
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

    let mut result = Vec::with_capacity(n_rows);
    for query_index in 0..n_rows {
        let query_start = query_index * n_features;
        let query = &data[query_start..query_start + n_features];

        let mut candidates = Vec::with_capacity(n_rows);
        for candidate_index in 0..n_rows {
            let candidate_start = candidate_index * n_features;
            let candidate = &data[candidate_start..candidate_start + n_features];
            candidates.push(Neighbor {
                index: candidate_index,
                distance: euclidean_distance(query, candidate),
            });
        }

        candidates.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.index.cmp(&right.index))
        });
        candidates.truncate(k);
        result.push(candidates);
    }

    Ok(result)
}

fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let squared_sum = left
        .iter()
        .zip(right)
        .fold(0.0_f32, |sum, (&left_value, &right_value)| {
            let difference = left_value - right_value;
            sum + difference * difference
        });
    squared_sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{KnnError, Neighbor, exhaustive_knn};

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
}
