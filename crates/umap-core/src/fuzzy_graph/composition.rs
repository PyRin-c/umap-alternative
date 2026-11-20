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

/// Directed fuzzy operations evaluated at an existing set of coordinates.
#[derive(Clone, Copy, Debug)]
pub enum MembershipOperation {
    Union,
    Intersection { weight: f64, right_complement: bool },
}

/// Borrowed CSR storage for membership calculations, including explicit zeros.
pub struct MembershipRows<'a> {
    indptr: &'a [usize],
    columns: &'a [usize],
    values: &'a [f64],
    sorted: bool,
}

impl<'a> MembershipRows<'a> {
    /// Validate the offsets while retaining the original order of repeated entries.
    pub fn new(
        indptr: &'a [usize],
        columns: &'a [usize],
        values: &'a [f64],
    ) -> Result<Self, &'static str> {
        if columns.len() != values.len() || !valid_membership_offsets(indptr, values.len()) {
            return Err("invalid CSR membership buffers");
        }
        let sorted = indptr
            .windows(2)
            .all(|range| columns[range[0]..range[1]].is_sorted());
        Ok(Self {
            indptr,
            columns,
            values,
            sorted,
        })
    }

    fn position(&self, row: usize, column: usize) -> Option<usize> {
        let start = self.indptr[row];
        let columns = &self.columns[start..self.indptr[row + 1]];
        let index = if self.sorted {
            let index = columns
                .partition_point(|&entry| entry <= column)
                .checked_sub(1)?;
            (columns[index] == column).then_some(index)
        } else {
            columns.iter().rposition(|&entry| entry == column)
        }?;
        Some(start + index)
    }
}

/// Write directed membership weights into an existing COO value buffer.
///
/// Missing entries use the operand's fallback. Explicit zeros are present
/// entries. For repeated CSR columns the last stored weight is used.
/// `aliases` indicates which operand value buffers share the full output
/// storage. `store` rounds a new weight to the output dtype before reuse.
pub fn combine_memberships(
    operands: [&MembershipRows<'_>; 2],
    coordinates: (&[usize], &[usize]),
    values: &mut [f64],
    aliases: [bool; 2],
    operation: MembershipOperation,
    store: impl Fn(f64) -> f64,
) -> Result<(), &'static str> {
    let [left, right] = operands;
    let (rows, columns) = coordinates;
    if values.len() != rows.len()
        || values.len() != columns.len()
        || rows
            .iter()
            .any(|&row| row >= left.indptr.len().min(right.indptr.len()) - 1)
        || operands
            .iter()
            .zip(aliases)
            .any(|(graph, aliases)| aliases && graph.values.len() != values.len())
    {
        return Err("incompatible fuzzy membership shapes");
    }
    let ([alias_left, alias_right], right_complement) = (
        aliases,
        matches!(
            operation,
            MembershipOperation::Intersection {
                right_complement: true,
                ..
            }
        ),
    );
    let right_value = |value| if right_complement { 1.0 - value } else { value };
    let get_floor = |iter_min: Option<f64>| {
        iter_min
            .map(|min| (min / 2.0).max(1.0e-8))
            .ok_or("membership graph has no stored weights")
    };
    let left_floor = get_floor(left.values.iter().copied().reduce(f64::min))?;
    let mut right_floor = get_floor(
        right
            .values
            .iter()
            .copied()
            .map(right_value)
            .reduce(f64::min),
    )?;
    if matches!(operation, MembershipOperation::Intersection { .. }) {
        right_floor = right_floor.min(1.0e-4);
    }
    for (index, (&row, &column)) in rows.iter().zip(columns).enumerate() {
        let a = left.position(row, column).map_or(left_floor, |position| {
            if alias_left {
                values[position]
            } else {
                left.values[position]
            }
        });
        let b = right.position(row, column).map_or(right_floor, |position| {
            right_value(if alias_right {
                values[position]
            } else {
                right.values[position]
            })
        });
        let weight = match operation {
            MembershipOperation::Union => union_weight(a, b),
            MembershipOperation::Intersection { weight, .. } => {
                if a > left_floor || b > right_floor {
                    intersection_weight(a, b, weight)
                } else {
                    continue;
                }
            }
        };
        values[index] = store(weight);
    }
    Ok(())
}

fn union_weight(left: f64, right: f64) -> f64 {
    left + right - left * right
}

fn intersection_weight(left: f64, right: f64, weight: f64) -> f64 {
    if weight < 0.5 {
        left * right.powf(weight / (1.0 - weight))
    } else {
        left.powf((1.0 - weight) / weight) * right
    }
}

fn valid_membership_offsets(indptr: &[usize], entries: usize) -> bool {
    indptr.first() == Some(&0) && indptr.last() == Some(&entries) && indptr.is_sorted()
}

/// Recalibrate a CSR buffer of max-normalized rows toward `log2(15)` membership mass.
pub fn reset_membership_scales(indptr: &[usize], values: &mut [f64]) -> Result<(), &'static str> {
    if !valid_membership_offsets(indptr, values.len()) {
        return Err("invalid CSR membership offsets");
    }
    for bounds in indptr.windows(2) {
        let row = &mut values[bounds[0]..bounds[1]];
        if !row.is_empty() {
            let power = membership_power(row.iter().copied(), 32);
            for weight in row {
                *weight = weight.powf(power);
            }
        }
    }
    Ok(())
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
                        FuzzySetOperation::Union => union_weight(left_weight, right_weight),
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
    let power = membership_power(row.iter().map(|&(_, weight)| f64::from(weight)), 64);
    for (_, weight) in row {
        *weight = f64::from(*weight).powf(power) as f32;
    }
}

fn membership_power(weights: impl Iterator<Item = f64> + Clone, iterations: usize) -> f64 {
    let target = 15.0_f64.log2();
    let mut low = 0.0;
    let mut high = f64::INFINITY;
    let mut power = 1.0;
    for _ in 0..iterations {
        let total: f64 = weights.clone().map(|weight| weight.powf(power)).sum();
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
    power
}
