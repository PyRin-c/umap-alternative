//! Fuzzy set operators on already fitted graphs.

use super::{CsrGraph, SMOOTH_TOLERANCE, fuzzy_union, symmetrize};
use crate::layout::{LayoutError, validate_csr_rows, validate_csr_shape};

/// Operations on the fuzzy graphs of two estimators over the same vertices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuzzySetOperation {
    /// Fuzzy union.
    Union,
    /// Fuzzy intersection.
    Intersection,
    /// Intersect the left graph with the complement of the right graph.
    Difference,
}

/// Combine two canonical square CSR memberships with the same vertex count.
///
/// The result has symmetric memberships and sorted column indices. The row
/// adjustment used by union and intersection targets a membership sum of
/// `log2(15)`. A difference uses row maximum normalization alone. Weights
/// below the eventual layout cutoff are retained for the caller to handle.
/// A difference also updates the stored left memberships to the raw product
/// with the right complement, before row normalization.
pub fn combine_fuzzy_graphs(
    left: &mut CsrGraph,
    right: &CsrGraph,
    operation: FuzzySetOperation,
) -> Result<CsrGraph, LayoutError> {
    for graph in [&*left, right] {
        validate_csr_shape(graph)?;
        if graph.n_rows != left.n_rows
            || graph.n_rows != graph.n_cols
            || graph.indptr.last().copied() != Some(graph.indices.len())
        {
            return Err(LayoutError::InvalidGraphShape {
                n_rows: graph.n_rows,
                n_cols: graph.n_cols,
                indptr: graph.indptr.len(),
                indices: graph.indices.len(),
                data: graph.data.len(),
            });
        }
        validate_csr_rows(graph)?;
    }

    let left_floor = membership_floor(left.data.iter().copied());
    let right_floor = match operation {
        FuzzySetOperation::Union => membership_floor(right.data.iter().copied()),
        FuzzySetOperation::Intersection => membership_floor(right.data.iter().copied()).min(1.0e-4),
        FuzzySetOperation::Difference => {
            membership_floor(right.data.iter().map(|&weight| 1.0 - weight)).min(1.0e-4)
        }
    };
    let directed = (0..left.n_rows)
        .map(|row| {
            let left_row = left.indptr[row]..left.indptr[row + 1];
            let right_row = right.indptr[row]..right.indptr[row + 1];
            let left_columns = &left.indices[left_row.clone()];
            let right_columns = &right.indices[right_row.clone()];
            let mut columns = left_columns.to_vec();
            if operation != FuzzySetOperation::Difference {
                columns.extend_from_slice(right_columns);
                columns.sort_unstable();
                columns.dedup();
            }
            let mut memberships: Vec<_> = columns
                .into_iter()
                .filter_map(|column| {
                    let left_position = left_columns
                        .binary_search(&column)
                        .ok()
                        .map(|index| left_row.start + index);
                    let right_entry = right_columns
                        .binary_search(&column)
                        .ok()
                        .map(|index| f64::from(right.data[right_row.start + index]));
                    let left_weight =
                        left_position.map_or(left_floor, |position| f64::from(left.data[position]));
                    let right_weight = right_entry.unwrap_or(right_floor);
                    let weight = match operation {
                        FuzzySetOperation::Union => {
                            left_weight + right_weight - left_weight * right_weight
                        }
                        FuzzySetOperation::Intersection => left_weight * right_weight,
                        FuzzySetOperation::Difference => {
                            let complement = right_entry.map_or(right_floor, |value| 1.0 - value);
                            if left_weight > left_floor || complement > right_floor {
                                left_weight * complement
                            } else {
                                left_weight
                            }
                        }
                    } as f32;
                    if operation == FuzzySetOperation::Difference {
                        left.data[left_position.expect("difference keeps the left support")] =
                            weight;
                    }
                    (weight > 0.0).then_some((column, weight))
                })
                .collect();
            let maximum = memberships
                .iter()
                .map(|&(_, weight)| weight)
                .fold(0.0, f32::max);
            for (_, weight) in &mut memberships {
                *weight /= maximum;
            }
            if operation != FuzzySetOperation::Difference && !memberships.is_empty() {
                reset_row_memberships(&mut memberships);
            }
            memberships
        })
        .collect::<Vec<_>>();
    Ok(symmetrize(&directed, false, fuzzy_union))
}

fn membership_floor(weights: impl Iterator<Item = f32>) -> f64 {
    let minimum = weights.fold(1.0, f32::min);
    (f64::from(minimum) / 2.0).max(1.0e-8)
}

fn reset_row_memberships(row: &mut [(usize, f32)]) {
    let target = 15.0_f64.log2();
    let mut low = 0.0;
    let mut high = f64::INFINITY;
    let mut power = 1.0;
    for _ in 0..64 {
        let total: f64 = row
            .iter()
            .map(|&(_, weight)| f64::from(weight).powf(power))
            .sum();
        if (total - target).abs() < SMOOTH_TOLERANCE {
            break;
        }
        if total > target {
            low = power;
        } else {
            high = power;
        }
        power = if high.is_finite() {
            (low + high) / 2.0
        } else {
            power * 2.0
        };
    }
    for (_, weight) in row {
        *weight = f64::from(*weight).powf(power) as f32;
    }
}
