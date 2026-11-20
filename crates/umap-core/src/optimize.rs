//! Sequential CSR optimization with an injected negative-vertex sampler.
//!
//! The optimizer composes the existing [`crate::layout::attractive_update`]
//! and [`crate::layout::negative_example_update`] kernels.  It deliberately
//! keeps vertex selection outside the core: callers supply a deterministic or
//! random `FnMut(usize) -> usize`, receiving the current source row and
//! returning a candidate vertex.

use std::fmt;

use crate::fuzzy_graph::CsrGraph;
use crate::layout::{
    LayoutError, attractive_update, negative_example_update, validate_csr_inputs,
    validate_repulsion_strength,
};
use crate::sampling::{EdgeSamplingCursor, SamplingError};

/// Failures while running [`optimize_csr`].
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizeError {
    /// An existing layout kernel or CSR validator rejected the input or an
    /// arithmetic result.
    Layout(LayoutError),
    /// A positive/negative timing cursor rejected schedule arithmetic.
    Sampling(SamplingError),
    /// The injected sampler returned a vertex outside the graph.
    InvalidNegativeVertex {
        /// Source row for which the sampler was called.
        source: usize,
        /// Vertex returned by the sampler.
        vertex: usize,
        /// Number of graph rows and valid vertex indices.
        n_rows: usize,
    },
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(formatter, "layout optimizer update failed: {error}"),
            Self::Sampling(error) => write!(formatter, "sampling schedule failed: {error}"),
            Self::InvalidNegativeVertex {
                source,
                vertex,
                n_rows,
            } => write!(
                formatter,
                "negative sampler returned vertex {vertex} for source {source}, outside 0..{n_rows}"
            ),
        }
    }
}

impl std::error::Error for OptimizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout(error) => Some(error),
            Self::Sampling(error) => Some(error),
            Self::InvalidNegativeVertex { .. } => None,
        }
    }
}

impl From<LayoutError> for OptimizeError {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<SamplingError> for OptimizeError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

/// Run a sequential, in-place CSR UMAP update with an injected sampler.
///
/// Positive graph memberships are normalized by the largest nonzero graph
/// membership, so an edge with weight `w` receives a positive interval of
/// `maximum_nonzero_weight / w`.  For every epoch, rows and then each row's
/// stored CSR columns are visited in order.  If an edge is due, its symmetric
/// attractive update is applied before its negative samples.  The sampler is
/// called once for every scheduled negative sample, including samples that
/// return the source row; self samples are validated and counted but skipped
/// as updates.  A non-self candidate updates only the source endpoint.
///
/// `learning_rate` is shared by all edges in an epoch and decays after the
/// epoch using the same ordering as [`crate::layout::attractive_optimize_csr`]:
/// `initial_learning_rate * (1 - epoch / n_epochs)`.  `n_epochs` is the total
/// number of epochs and may be zero.  `negative_sample_rate` is the number of
/// negative samples per positive sample and is used to derive each edge's
/// negative interval.
///
/// The attraction and repulsion forces and the learning-rate decay use
/// `f64`, including when the caller supplies `f32` values.
///
/// The published unit-edge checkpoints establish the timing at the boundary
/// needed here: rate one has no negative draw at `n_epochs = 2` and its first
/// draw at `n_epochs = 3`; rate two has one draw at `n_epochs = 2`.  Those
/// counts do not distinguish whether a negative backlog is drained on every
/// epoch or only when its positive edge is due.  This optimizer follows the
/// edge-driven UMAP convention and drains an edge's negative cursor only on
/// epochs where that edge is positively due.
///
/// Graph shape, CSR order, embedding values, layout parameters, repulsion
/// strength, and negative sample rate are validated before the first embedding
/// update.  A sampler index is checked before its corresponding negative
/// kernel call; if a later callback returns an invalid index, earlier valid
/// updates remain applied just as they do for a later arithmetic failure.
#[allow(clippy::too_many_arguments)]
pub fn optimize_csr<F>(
    embedding: &mut [f32],
    n_components: usize,
    graph: &CsrGraph,
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
    repulsion_strength: impl Into<f64>,
    negative_sample_rate: f64,
    n_epochs: usize,
    negative_sampler: F,
) -> Result<(), OptimizeError>
where
    F: FnMut(usize) -> usize,
{
    let (a, b, learning_rate, repulsion_strength) = (
        a.into(),
        b.into(),
        learning_rate.into(),
        repulsion_strength.into(),
    );
    validate_csr_inputs(embedding, n_components, graph, a, b, learning_rate)?;
    validate_repulsion_strength(repulsion_strength)?;
    // Constructing a harmless unit-interval cursor validates the rate even
    // when the graph has no positive entries and therefore needs no cursors.
    EdgeSamplingCursor::from_positive_interval(1.0, negative_sample_rate)?;

    let maximum_positive_weight = graph
        .data
        .iter()
        .copied()
        .filter(|weight| *weight > 0.0)
        .max_by(f32::total_cmp);

    let Some(maximum_positive_weight) = maximum_positive_weight else {
        return Ok(());
    };

    let mut cursors = graph
        .data
        .iter()
        .copied()
        .map(|weight| {
            if weight == 0.0 {
                Ok(None)
            } else {
                EdgeSamplingCursor::from_graph_weights(
                    maximum_positive_weight,
                    weight,
                    negative_sample_rate,
                )
                .map(Some)
            }
        })
        .collect::<Result<Vec<_>, SamplingError>>()?;

    let mut negative_sampler = negative_sampler;
    let mut alpha = learning_rate;
    for epoch in 0..n_epochs {
        let epoch_as_f64 = epoch as f64;
        for row in 0..graph.n_rows {
            let start = graph.indptr[row];
            let end = graph.indptr[row + 1];
            for (cursor, &index) in cursors[start..end]
                .iter_mut()
                .zip(graph.indices[start..end].iter())
            {
                let Some(cursor) = cursor.as_mut() else {
                    continue;
                };

                let positive_due = cursor.positive_due(epoch_as_f64)?;
                let negative_count = if positive_due {
                    cursor.take_negative_samples(epoch_as_f64)?
                } else {
                    0
                };

                if positive_due {
                    attractive_update(embedding, n_components, row, index, a, b, alpha)?;
                }

                for _ in 0..negative_count {
                    let vertex = negative_sampler(row);
                    if vertex >= graph.n_rows {
                        return Err(OptimizeError::InvalidNegativeVertex {
                            source: row,
                            vertex,
                            n_rows: graph.n_rows,
                        });
                    }
                    if vertex == row {
                        continue;
                    }
                    negative_update_rows(
                        embedding,
                        n_components,
                        row,
                        vertex,
                        a,
                        b,
                        repulsion_strength,
                        alpha,
                    )?;
                }
            }
        }

        // Keep alpha fixed for every edge in this epoch, then decay it for
        // the next epoch.  n_epochs == 0 never enters this loop.
        alpha = learning_rate * (1.0 - epoch_as_f64 / n_epochs as f64);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn negative_update_rows(
    embedding: &mut [f32],
    n_components: usize,
    source: usize,
    vertex: usize,
    a: f64,
    b: f64,
    repulsion_strength: f64,
    learning_rate: f64,
) -> Result<(), LayoutError> {
    let source_start = source * n_components;
    let vertex_start = vertex * n_components;
    if source_start < vertex_start {
        let (before_vertex, after_vertex) = embedding.split_at_mut(vertex_start);
        let current = &mut before_vertex[source_start..source_start + n_components];
        let other = &after_vertex[..n_components];
        negative_example_update(current, other, a, b, repulsion_strength, learning_rate)
    } else {
        let (before_source, after_source) = embedding.split_at_mut(source_start);
        let other = &before_source[vertex_start..vertex_start + n_components];
        let current = &mut after_source[..n_components];
        negative_example_update(current, other, a, b, repulsion_strength, learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::{OptimizeError, optimize_csr};
    use crate::fuzzy_graph::CsrGraph;
    use crate::layout::LayoutError;

    fn unit_edge_graph() -> CsrGraph {
        CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 1],
            indices: vec![1],
            data: vec![1.0],
        }
    }

    #[test]
    fn unit_rate_has_no_draw_at_two_epochs_and_first_draw_at_three() {
        let graph = unit_edge_graph();
        let mut draws = 0;
        let mut embedding = [0.0_f32, 2.0];
        optimize_csr(
            &mut embedding,
            1,
            &graph,
            1.0_f32,
            1.0_f32,
            0.1_f32,
            1.0,
            1.0,
            2,
            |_| {
                draws += 1;
                1
            },
        )
        .expect("two epochs with rate one must be valid");
        assert_eq!(draws, 0);

        let mut draws = 0;
        let mut embedding = [0.0_f32, 2.0];
        optimize_csr(
            &mut embedding,
            1,
            &graph,
            1.0,
            1.0,
            0.1,
            1.0,
            1.0,
            3,
            |_| {
                draws += 1;
                1
            },
        )
        .expect("three epochs with rate one must be valid");
        assert_eq!(draws, 1);
    }

    #[test]
    fn rate_two_draws_at_two_epochs_and_updates_only_the_source() {
        let graph = CsrGraph {
            n_rows: 3,
            n_cols: 3,
            indptr: vec![0, 1, 1, 1],
            indices: vec![1],
            data: vec![1.0],
        };
        let mut draws = 0;
        let mut embedding = [0.0_f32, 2.0, 5.0];
        optimize_csr(
            &mut embedding,
            1,
            &graph,
            1.0,
            1.0,
            0.1,
            0.5,
            2.0,
            2,
            |_| {
                draws += 1;
                2
            },
        )
        .expect("two epochs with rate two must be valid");
        assert_eq!(draws, 1);
        assert_eq!(embedding[2], 5.0);
        assert!((embedding[1] - 1.92).abs() < 1.0e-6);
        assert!(embedding[0] < 0.08);
    }

    #[test]
    fn self_samples_are_counted_but_skipped() {
        let graph = unit_edge_graph();
        let mut draws = 0;
        let mut embedding = [0.0_f32, 2.0];
        optimize_csr(
            &mut embedding,
            1,
            &graph,
            1.0,
            1.0,
            0.1,
            1.0,
            2.0,
            2,
            |_| {
                draws += 1;
                0
            },
        )
        .expect("self samples are valid no-ops");
        assert_eq!(draws, 1);
        assert!((embedding[0] - 0.08).abs() < 1.0e-6);
        assert!((embedding[1] - 1.92).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_sampler_vertex_is_reported_before_its_negative_update() {
        let graph = unit_edge_graph();
        let mut embedding = [0.0_f32, 2.0];
        let error =
            optimize_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1, 1.0, 2.0, 2, |_| 2).unwrap_err();
        assert_eq!(
            error,
            OptimizeError::InvalidNegativeVertex {
                source: 0,
                vertex: 2,
                n_rows: 2,
            }
        );
        // The positive update precedes the invalid negative callback; no
        // negative kernel ran, so the endpoint remains finite and unchanged
        // by an out-of-range slice.
        assert!(embedding.iter().all(|value| value.is_finite()));
        assert!((embedding[0] - 0.08).abs() < 1.0e-6);
        assert!((embedding[1] - 1.92).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_shape_and_settings_are_rejected_before_updates() {
        let graph = unit_edge_graph();
        let mut embedding = [0.0_f32, 2.0];
        let original = embedding;
        let error =
            optimize_csr(&mut embedding, 0, &graph, 1.0, 1.0, 0.1, 1.0, 1.0, 2, |_| 1).unwrap_err();
        assert!(matches!(
            error,
            OptimizeError::Layout(LayoutError::InvalidShape { .. })
        ));
        assert_eq!(embedding, original);

        let error = optimize_csr(
            &mut embedding,
            1,
            &graph,
            1.0,
            1.0,
            0.1,
            -1.0,
            1.0,
            2,
            |_| 1,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OptimizeError::Layout(LayoutError::InvalidParameter {
                parameter: "repulsion_strength",
                ..
            })
        ));
        assert_eq!(embedding, original);
    }
}
