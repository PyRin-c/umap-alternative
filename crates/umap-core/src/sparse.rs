//! Exhaustive neighbor search and fuzzy graph fitting for CSR feature matrices.

use std::cmp::Ordering;
use std::fmt;

use crate::fuzzy_graph::{CsrGraph, GraphError, fuzzy_simplicial_set};
use crate::neighbors::{DistanceMetric, Neighbor, collect_neighbors};

/// Input validation failures for [`exhaustive_csr_knn`].
#[derive(Clone, Debug, PartialEq)]
pub enum SparseKnnError {
    /// Row pointer, column index, and stored value lengths must agree with the shape.
    InvalidShape {
        n_rows: usize,
        n_features: usize,
        offsets: usize,
        indices: usize,
        values: usize,
    },
    /// The row pointers must begin at zero and end at the stored entry count.
    InvalidEndpoints {
        first: usize,
        last: usize,
        entries: usize,
    },
    /// Row pointers must be nondecreasing and within the stored entry count.
    InvalidRowOffsets {
        row: usize,
        start: usize,
        end: usize,
        entries: usize,
    },
    /// Stored column indices must be in `0..n_features`.
    InvalidColumnIndex {
        row: usize,
        position: usize,
        column: usize,
        n_features: usize,
    },
    /// Stored column indices must increase strictly within each row.
    UnorderedColumnIndices {
        row: usize,
        position: usize,
        previous: usize,
        column: usize,
    },
    /// `k` must be at least one and no greater than the number of rows.
    InvalidK { k: usize, n_rows: usize },
    /// A stored feature value must be finite.
    NonFiniteValue { position: usize, value: f32 },
}

impl fmt::Display for SparseKnnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_rows,
                n_features,
                offsets,
                indices,
                values,
            } => write!(
                formatter,
                "CSR shape {n_rows}x{n_features} does not match {offsets} row pointers, \
                 {indices} column indices, and {values} stored values"
            ),
            Self::InvalidEndpoints {
                first,
                last,
                entries,
            } => write!(
                formatter,
                "CSR row pointers must start at 0 and end at {entries}, got {first} and {last}"
            ),
            Self::InvalidRowOffsets {
                row,
                start,
                end,
                entries,
            } => write!(
                formatter,
                "CSR row {row} has invalid offsets {start}..{end} for {entries} stored entries"
            ),
            Self::InvalidColumnIndex {
                row,
                position,
                column,
                n_features,
            } => write!(
                formatter,
                "CSR row {row}, entry {position} has column {column}, outside 0..{n_features}"
            ),
            Self::UnorderedColumnIndices {
                row,
                position,
                previous,
                column,
            } => write!(
                formatter,
                "CSR row {row}, entry {position} has column {column}, expected greater than {previous}"
            ),
            Self::InvalidK { k, n_rows } => {
                write!(formatter, "k={k} is outside the valid range 1..={n_rows}")
            }
            Self::NonFiniteValue { position, value } => {
                write!(
                    formatter,
                    "CSR value at entry {position} is not finite: {value}"
                )
            }
        }
    }
}

impl std::error::Error for SparseKnnError {}

/// Errors returned while fitting a sparse fuzzy graph.
#[derive(Clone, Debug, PartialEq)]
pub enum SparseGraphFitError {
    /// CSR validation or neighbor count validation failed.
    Knn(SparseKnnError),
    /// Fuzzy simplicial-set construction rejected the neighbor rows.
    Graph(GraphError),
}

impl fmt::Display for SparseGraphFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Knn(error) => write!(formatter, "sparse graph neighbor search failed: {error}"),
            Self::Graph(error) => {
                write!(formatter, "sparse fuzzy graph construction failed: {error}")
            }
        }
    }
}

impl std::error::Error for SparseGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Knn(error) => Some(error),
            Self::Graph(error) => Some(error),
        }
    }
}

/// Find the `k` nearest rows for every row of a canonical CSR feature matrix.
///
/// `indptr` contains `n_rows + 1` nondecreasing offsets, starting at zero and
/// ending at `indices.len() == data.len()`. Each row has strictly ascending
/// column indices less than `n_features`. Stored values must be finite.
/// Missing entries represent zero, and empty rows and explicit signed zeros
/// are allowed. Callers must sort and combine duplicate columns before using
/// this API. Compatibility with public noncanonical input behavior and sparse
/// precomputed-distance semantics is outside this feature-matrix contract.
///
/// Distances use a two-pointer merge in ascending column order, visiting only
/// stored coordinates. Euclidean distances sum squared `f64` differences,
/// round the sum to `f32`, and take its square root in `f32`. Manhattan distances
/// sum absolute `f64` differences, and Chebyshev distances take their maximum.
/// Both round the final result to `f32`. Computed distances can overflow to
/// positive infinity.
///
/// Every row is considered as a candidate, including the query itself, and
/// `k` must be in `1..=n_rows`. As in the dense search, rows are ranked by
/// [`f32::total_cmp`] distance and then by ascending candidate index.
pub fn exhaustive_csr_knn(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<Vec<Vec<Neighbor>>, SparseKnnError> {
    if n_rows.checked_add(1) != Some(indptr.len()) || indices.len() != data.len() {
        return Err(SparseKnnError::InvalidShape {
            n_rows,
            n_features,
            offsets: indptr.len(),
            indices: indices.len(),
            values: data.len(),
        });
    }

    let first = indptr[0];
    let last = indptr[n_rows];
    if first != 0 || last != data.len() {
        return Err(SparseKnnError::InvalidEndpoints {
            first,
            last,
            entries: data.len(),
        });
    }

    if !(1..=n_rows).contains(&k) {
        return Err(SparseKnnError::InvalidK { k, n_rows });
    }

    for (row, offsets) in indptr.windows(2).enumerate() {
        let (start, end) = (offsets[0], offsets[1]);
        if start > end || end > data.len() {
            return Err(SparseKnnError::InvalidRowOffsets {
                row,
                start,
                end,
                entries: data.len(),
            });
        }

        for position in start..end {
            let column = indices[position];
            if column >= n_features {
                return Err(SparseKnnError::InvalidColumnIndex {
                    row,
                    position,
                    column,
                    n_features,
                });
            }
            if position > start && indices[position - 1] >= column {
                return Err(SparseKnnError::UnorderedColumnIndices {
                    row,
                    position,
                    previous: indices[position - 1],
                    column,
                });
            }
            let value = data[position];
            if !value.is_finite() {
                return Err(SparseKnnError::NonFiniteValue { position, value });
            }
        }
    }

    let row_at = |row| {
        let entries = indptr[row]..indptr[row + 1];
        CsrRow {
            columns: &indices[entries.clone()],
            values: &data[entries],
        }
    };
    Ok(collect_neighbors(n_rows, k, |query_index, candidates| {
        let query = row_at(query_index);
        candidates.extend((0..n_rows).map(|index| Neighbor {
            index,
            distance: sparse_distance(query, row_at(index), metric),
        }));
    }))
}

/// Fit a square fuzzy graph from a canonical CSR feature matrix.
///
/// The input and `k` follow [`exhaustive_csr_knn`]. The graph is canonical CSR
/// with strictly ascending columns per row and no diagonal entries. Graph
/// fitting requires finite selected distances and a self neighbor in every
/// selected row. A tie at zero distance can exclude the self neighbor and
/// produce [`GraphError::MissingSelf`].
pub fn fit_sparse_graph(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<CsrGraph, SparseGraphFitError> {
    let neighbors = exhaustive_csr_knn(indptr, indices, data, n_rows, n_features, k, metric)
        .map_err(SparseGraphFitError::Knn)?;
    fuzzy_simplicial_set(&neighbors).map_err(SparseGraphFitError::Graph)
}

#[derive(Clone, Copy)]
struct CsrRow<'a> {
    columns: &'a [usize],
    values: &'a [f32],
}

fn sparse_distance(left: CsrRow<'_>, right: CsrRow<'_>, metric: DistanceMetric) -> f32 {
    let (mut left_position, mut right_position) = (0, 0);
    let differences = std::iter::from_fn(move || {
        // Validated feature indices are always less than usize::MAX.
        let left_column = left
            .columns
            .get(left_position)
            .copied()
            .unwrap_or(usize::MAX);
        let right_column = right
            .columns
            .get(right_position)
            .copied()
            .unwrap_or(usize::MAX);
        Some(match left_column.cmp(&right_column) {
            Ordering::Less => {
                let difference = f64::from(left.values[left_position]);
                left_position += 1;
                difference
            }
            Ordering::Greater => {
                let difference = -f64::from(right.values[right_position]);
                right_position += 1;
                difference
            }
            Ordering::Equal if left_column == usize::MAX => return None,
            Ordering::Equal => {
                let difference =
                    f64::from(left.values[left_position]) - f64::from(right.values[right_position]);
                left_position += 1;
                right_position += 1;
                difference
            }
        })
    });

    match metric {
        DistanceMetric::Euclidean => {
            let squared_sum =
                differences.fold(0.0, |sum, difference| sum + difference * difference);
            (squared_sum as f32).sqrt()
        }
        DistanceMetric::Manhattan => {
            differences.fold(0.0, |sum, difference| sum + difference.abs()) as f32
        }
        DistanceMetric::Chebyshev => {
            differences.fold(0.0_f64, |maximum, difference| maximum.max(difference.abs())) as f32
        }
    }
}
