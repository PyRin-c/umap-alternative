//! Positive-edge preparation for a downstream parametric UMAP trainer.

use std::fmt;

use crate::fuzzy_graph::CsrGraph;

/// One directed edge and its multiplicity in the training pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParametricEdge {
    pub head: usize,
    pub tail: usize,
    pub repeats: usize,
}

/// The training pool in CSR order, with consecutive copies kept as counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParametricEdges {
    edges: Vec<ParametricEdge>,
    total_repeats: usize,
}

impl ParametricEdges {
    /// The directed edges with at least one copy in the pool.
    #[must_use]
    pub fn edges(&self) -> &[ParametricEdge] {
        &self.edges
    }

    /// The length of the expanded pool.
    #[must_use]
    pub const fn total_repeats(&self) -> usize {
        self.total_repeats
    }
}

/// Errors preparing a positive-edge pool.
#[derive(Clone, Debug, PartialEq)]
pub enum ParametricEdgeError {
    InvalidCsrShape,
    InvalidCsrRow { row: usize },
    InvalidWeight { position: usize, weight: f32 },
    RepeatCountOverflow,
}

impl fmt::Display for ParametricEdgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCsrShape => write!(formatter, "expected a square CSR graph"),
            Self::InvalidCsrRow { row } => {
                write!(
                    formatter,
                    "invalid offsets or column indices in CSR row {row}"
                )
            }
            Self::InvalidWeight { position, weight } => {
                write!(
                    formatter,
                    "membership at entry {position} must be in 0..=1, got {weight}"
                )
            }
            Self::RepeatCountOverflow => {
                write!(formatter, "positive-edge repeat count exceeds usize")
            }
        }
    }
}

impl std::error::Error for ParametricEdgeError {}

/// Prepare the directed training pool for an explicit graph epoch count.
///
/// Each membership is multiplied by `n_epochs` in float32 and truncated toward
/// zero. A zero count omits the edge. The input must be a square CSR matrix
/// with strictly increasing column indices within each row and finite
/// memberships in `0..=1`. An empty graph or zero epochs yields an empty pool.
///
/// Callers choose the epoch count and use the pool as input to their batching
/// and training procedures.
pub fn prepare_parametric_edges(
    graph: &CsrGraph,
    n_epochs: usize,
) -> Result<ParametricEdges, ParametricEdgeError> {
    if graph.n_rows != graph.n_cols
        || graph.n_rows.checked_add(1) != Some(graph.indptr.len())
        || graph.indptr.first() != Some(&0)
        || graph.indptr.last() != Some(&graph.data.len())
        || graph.indices.len() != graph.data.len()
    {
        return Err(ParametricEdgeError::InvalidCsrShape);
    }

    let mut edges = Vec::new();
    let mut total_repeats = 0_usize;
    let epochs = n_epochs as f32;
    for (head, offsets) in graph.indptr.windows(2).enumerate() {
        let start = offsets[0];
        let end = offsets[1];
        let columns = graph
            .indices
            .get(start..end)
            .ok_or(ParametricEdgeError::InvalidCsrRow { row: head })?;
        if columns.last().is_some_and(|&column| column >= graph.n_cols)
            || columns.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(ParametricEdgeError::InvalidCsrRow { row: head });
        }
        for (offset, &tail) in columns.iter().enumerate() {
            let position = start + offset;
            let weight = graph.data[position];
            if !(0.0..=1.0).contains(&weight) {
                return Err(ParametricEdgeError::InvalidWeight { position, weight });
            }
            let count = (epochs * weight) as u128;
            let repeats =
                usize::try_from(count).map_err(|_| ParametricEdgeError::RepeatCountOverflow)?;
            if repeats > 0 {
                edges.push(ParametricEdge {
                    head,
                    tail,
                    repeats,
                });
                total_repeats = total_repeats
                    .checked_add(repeats)
                    .ok_or(ParametricEdgeError::RepeatCountOverflow)?;
            }
        }
    }
    Ok(ParametricEdges {
        edges,
        total_repeats,
    })
}
