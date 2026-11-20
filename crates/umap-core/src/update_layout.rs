//! Serial Euclidean optimization for a dense update.

use crate::embedding::EmbeddingLayoutConfig;
use crate::fuzzy_graph::CsrGraph;
use crate::layout::LayoutError;
use crate::optimize::{OptimizeError, OptimizeMode, optimize_csr_with_mode};
use crate::rng::{Mt19937, NegativeSamplerError, SerialNegativeSampler, draw_negative_seed};

/// Start the transform RNG for a dense update and consume its initial seed draw.
///
/// Use the returned RNG to draw any Gaussian initialization noise, then pass
/// the advanced RNG and normalized coordinates to [`optimize_update_layout`].
#[must_use]
pub fn begin_update_layout_rng(transform_seed: u32) -> Mt19937 {
    let mut rng = Mt19937::new(transform_seed);
    draw_negative_seed(|| rng.next_u32());
    rng
}

/// Optimize an already normalized row-major layout over a combined dense update graph.
///
/// The graph contains both existing and appended rows, and both endpoints of
/// each positive edge move. Each head has one negative stream that can sample
/// any row. The stream starts from the head's first normalized coordinate
/// and three accepted words drawn from the transform RNG after initialization.
///
/// Set `config.n_epochs` to zero for an omitted update epoch count. Use the
/// learning rate captured at fit and the negative sample rate and repulsion
/// strength configured at update. The input graph is kept intact.
///
/// The layout is modified in place, and failures during optimization may leave
/// earlier steps applied. The RNG advances by the optimizer seed draw on success.
pub fn optimize_update_layout(
    coordinates: &mut [f32],
    graph: &CsrGraph,
    config: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    rng: &mut Mt19937,
) -> Result<(), OptimizeError> {
    let mut next_rng = rng.clone();
    let mut sampler = SerialNegativeSampler::new(
        &mut next_rng,
        coordinates,
        graph.n_rows,
        config.n_components,
    )
    .map_err(|error| match error {
        NegativeSamplerError::InvalidShape {
            n_components,
            values,
            ..
        } => LayoutError::InvalidShape {
            n_components,
            values,
        },
        NegativeSamplerError::NonFiniteHeadCoordinate { head, value } => {
            LayoutError::NonFiniteValue {
                endpoint: "embedding",
                component: head * config.n_components,
                value,
            }
        }
    })?;

    optimize_csr_with_mode(
        coordinates,
        config.n_components,
        graph,
        config.a,
        config.b,
        config.learning_rate,
        config.repulsion_strength,
        config.negative_sample_rate,
        config.n_epochs,
        |head| {
            sampler
                .sample(head)
                .expect("validated CSR head is in bounds")
        },
        OptimizeMode::SeededCompatibility,
    )?;
    *rng = next_rng;
    Ok(())
}
