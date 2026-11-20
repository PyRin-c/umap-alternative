//! First-pass fuzzy graph construction for dense UMAP neighbors.
//!
//! The implementation here follows the fuzzy-simplicial-set construction from
//! the UMAP paper, with local connectivity fixed to one. It deliberately stops
//! at a deterministic CSR graph: it does not perform graph pruning, layout, or
//! optimization.

use std::fmt;

use crate::neighbors::Neighbor;

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
    /// Fuzzy-union membership for each entry in [`CsrGraph::indices`].
    pub data: Vec<f32>,
}

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
/// The scale search and membership arithmetic use `f32`, with an absolute sum
/// tolerance of `1e-5`. If the zero-scale limit exceeds the target, the scale
/// is reduced to `0.001 * max_offset`; rows with no positive offset use a unit
/// scale, though all their memberships are one.
///
/// Directed memberships are combined with the fuzzy union
/// `a + b - a * b`. The diagonal is omitted from the emitted graph, while all
/// non-self directed supports are retained, including supports whose computed
/// membership underflows to zero. CSR columns are strictly ascending within
/// every row, and the output is square with shape `n_rows x n_rows`.
pub fn fuzzy_simplicial_set(neighbors: &[Vec<Neighbor>]) -> Result<CsrGraph, GraphError> {
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

    let mut directed = Vec::with_capacity(n_rows);
    for (row_index, row) in neighbors.iter().enumerate() {
        let rho = local_offset(row_index, row);
        let sigma = distance_scale(row_index, row, rho);
        let mut memberships = row
            .iter()
            .filter(|neighbor| neighbor.index != row_index)
            .map(|neighbor| (neighbor.index, membership(neighbor.distance, rho, sigma)))
            .collect::<Vec<_>>();
        memberships.sort_unstable_by_key(|(index, _)| *index);
        directed.push(memberships);
    }

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
            let weight = (forward + reverse - forward * reverse).clamp(0.0, 1.0);
            indices.push(column);
            data.push(weight);
        }
        indptr.push(indices.len());
    }

    Ok(CsrGraph {
        n_rows,
        n_cols: n_rows,
        indptr,
        indices,
        data,
    })
}

fn local_offset(row_index: usize, row: &[Neighbor]) -> f32 {
    row.iter()
        .filter(|neighbor| neighbor.index != row_index && neighbor.distance > 0.0)
        .map(|neighbor| neighbor.distance)
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0)
}

fn distance_scale(row_index: usize, row: &[Neighbor], rho: f32) -> f32 {
    let target = (row.len() as f32).log2();
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

    let zero_scale_sum = offsets.iter().filter(|offset| **offset == 0.0).count() as f32;
    if zero_scale_sum > target {
        return (max_offset * 0.001).max(f32::MIN_POSITIVE);
    }

    let mut lower = 0.0_f32;
    let mut upper = f32::INFINITY;
    let mut middle = 1.0_f32;

    for _ in 0..64 {
        let sum = membership_sum(&offsets, middle);
        if (sum - target).abs() <= 1e-5 {
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

    middle.max(f32::MIN_POSITIVE)
}

fn membership_sum(offsets: &[f32], sigma: f32) -> f32 {
    offsets.iter().fold(0.0_f32, |sum, offset| {
        if *offset == 0.0 {
            sum + 1.0
        } else {
            sum + (-(*offset / sigma)).exp()
        }
    })
}

fn membership(distance: f32, rho: f32, sigma: f32) -> f32 {
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
