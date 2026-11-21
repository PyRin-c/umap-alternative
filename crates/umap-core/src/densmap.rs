//! Density-preserving updates for the serial Euclidean compatibility layout.

use std::fmt;

use crate::density::{DensityError, OriginalRadiusInput, original_radii_and_weights};
use crate::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingFitError, EmbeddingLayoutConfig,
    prepare_dense_embedding, serial_negative_sampler, zero_weak_memberships,
};
use crate::fuzzy_graph::CsrGraph;
use crate::initialization::normalize_columns;
use crate::layout::{DistanceReduction, LayoutError};
use crate::numeric::pairwise_sum;
use crate::optimize::{OptimizeError, OptimizeMode, optimize_csr_with_density};
use crate::rng::Mt19937;

/// Settings for the density correlation objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensmapOptions {
    /// Relative strength of the density objective.
    pub dens_lambda: f64,
    /// Fraction of epochs at the end of training eligible for density updates.
    pub dens_frac: f64,
    /// Stabilizer added to the embedding log-radius variance.
    pub dens_var_shift: f64,
}

impl Default for DensmapOptions {
    fn default() -> Self {
        Self {
            dens_lambda: 2.0,
            dens_frac: 0.3,
            dens_var_shift: 0.1,
        }
    }
}

impl DensmapOptions {
    fn validate(self) -> Result<(), DensmapError> {
        for (parameter, value, maximum) in [
            ("dens_lambda", self.dens_lambda, f64::MAX),
            ("dens_frac", self.dens_frac, 1.0),
            ("dens_var_shift", self.dens_var_shift, f64::MAX),
        ] {
            if !(0.0..=maximum).contains(&value) {
                return Err(DensmapError::InvalidOption { parameter, value });
            }
        }
        Ok(())
    }

    fn active_at(self, epoch: usize, n_epochs: usize) -> bool {
        self.dens_lambda > 0.0 && (epoch + 1) as f64 / n_epochs as f64 > 1.0 - self.dens_frac
    }
}

/// Failures in density-preserving fitting and optimization.
#[derive(Clone, Debug, PartialEq)]
pub enum DensmapError {
    /// A density option is outside its supported finite range.
    InvalidOption { parameter: &'static str, value: f64 },
    /// The supplied graph or its input-space distances cannot be reduced.
    OriginalRadii(DensityError),
    /// Graph fitting, layout initialization, or sampler initialization failed.
    Preparation(DenseEmbeddingFitError),
    /// A compatibility schedule, sample, or layout update failed.
    Optimize(OptimizeError),
}

impl fmt::Display for DensmapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOption { parameter, value } => {
                write!(formatter, "invalid densMAP option {parameter}={value}")
            }
            Self::OriginalRadii(error) => {
                write!(formatter, "original log-radius reduction failed: {error}")
            }
            Self::Preparation(error) => error.fmt(formatter),
            Self::Optimize(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DensmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidOption { .. } => None,
            Self::OriginalRadii(error) => Some(error),
            Self::Preparation(error) => Some(error),
            Self::Optimize(error) => Some(error),
        }
    }
}

/// Optimize initialized row-major coordinates using the serial compatibility schedule.
///
/// Pass the CSR graph with its training cutoff already applied and an aligned
/// unsquared input-space distance for each stored entry. Density preparation
/// is bypassed when the density term cannot run. Supply an initialized negative
/// sampler, such as [`crate::rng::SerialNegativeSampler`], to reproduce the
/// seeded compatibility stream.
///
/// The callback receives the coordinates after epoch `e` with key `e`, then
/// receives the final coordinates with key `layout.n_epochs`. At zero epochs
/// only key zero is reported. A failed update leaves previously completed
/// updates applied. Undefined density statistics produce an arithmetic error
/// when a sampled noncoincident edge needs its density gradient.
pub fn optimize_densmap_csr(
    embedding: &mut [f32],
    graph: &CsrGraph,
    edge_distances: &[f32],
    layout: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    options: DensmapOptions,
    negative_sampler: impl FnMut(usize) -> usize,
    checkpoint: impl FnMut(usize, &[f32]),
) -> Result<(), DensmapError> {
    let density = prepare_density(layout.n_epochs, options, || {
        original_radii_and_weights(graph, OriginalRadiusInput::EdgeDistances(edge_distances))
    })?;
    optimize_layout(
        embedding,
        graph,
        layout,
        negative_sampler,
        density,
        Some(checkpoint),
    )
}

/// Normalize a layout and fit a supplied graph with optional density preservation.
///
/// Supply a graph with its sampling cutoff already applied and a caller RNG at
/// the end of initialization. When enabled, density radii use the input-space
/// distances on this graph. The callback receives each completed epoch and the
/// final coordinates, using the same keys as [`optimize_densmap_csr`].
/// On error the caller RNG keeps its original state.
pub fn fit_graph_with_checkpoints(
    graph: CsrGraph,
    initial: &[f32],
    layout: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    density: Option<(OriginalRadiusInput<'_>, DensmapOptions)>,
    caller_rng: &mut Mt19937,
    checkpoint: Option<impl FnMut(usize, &[f32])>,
) -> Result<DenseEmbedding, DensmapError> {
    let mut coordinates = initial.to_vec();
    normalize_columns(&mut coordinates, graph.n_rows, layout.n_components)
        .map_err(|error| DensmapError::Preparation(error.into()))?;
    fit_initialized_graph(graph, coordinates, layout, density, caller_rng, checkpoint)
}

/// Fit a dense fuzzy graph and a seeded density-preserving embedding.
///
/// Dense feature distances follow `config.metric`. The optimizer uses the
/// fitted graph with weak memberships set to zero as prescribed by the density
/// sampling budget. Explicit (rather than seeded random) initialization uses
/// `explicit_seed` for negative sampling.
pub fn fit_dense_densmap_seeded(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    explicit_seed: u32,
    options: DensmapOptions,
) -> Result<DenseEmbedding, DensmapError> {
    let (mut graph, coordinates, caller_rng) =
        prepare_dense_embedding(data, n_rows, n_features, config, None)
            .map_err(DensmapError::Preparation)?;
    zero_weak_memberships(&mut graph, config.n_epochs, true);
    let input = OriginalRadiusInput::Dense {
        values: data,
        n_features,
        metric: config.metric,
    };
    let mut caller_rng = caller_rng.unwrap_or_else(|| Mt19937::new(explicit_seed));
    fit_initialized_graph(
        graph,
        coordinates,
        &config.layout_config(),
        Some((input, options)),
        &mut caller_rng,
        None::<fn(usize, &[f32])>,
    )
}

fn fit_initialized_graph(
    graph: CsrGraph,
    mut coordinates: Vec<f32>,
    layout: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    density: Option<(OriginalRadiusInput<'_>, DensmapOptions)>,
    caller_rng: &mut Mt19937,
    checkpoint: Option<impl FnMut(usize, &[f32])>,
) -> Result<DenseEmbedding, DensmapError> {
    let shape = (graph.n_rows, layout.n_components);
    let mut next_rng = caller_rng.clone();
    let mut sampler = serial_negative_sampler(&mut next_rng, &coordinates, shape.0, shape.1)
        .map_err(|error| DensmapError::Preparation(error.into()))?;
    let density = if let Some((input, options)) = density {
        prepare_density(layout.n_epochs, options, || {
            original_radii_and_weights(&graph, input)
        })?
    } else {
        None
    };
    optimize_layout(
        &mut coordinates,
        &graph,
        layout,
        |row| {
            sampler
                .sample(row)
                .expect("optimizer source row is in bounds")
        },
        density,
        checkpoint,
    )?;
    *caller_rng = next_rng;
    Ok(DenseEmbedding {
        graph,
        coordinates,
        shape,
    })
}

fn prepare_density(
    n_epochs: usize,
    options: DensmapOptions,
    input_radii: impl FnOnce() -> Result<(Vec<f32>, Vec<f32>), DensityError>,
) -> Result<Option<DensityOptimizer>, DensmapError> {
    options.validate()?;
    if n_epochs <= 1 || !options.active_at(n_epochs - 1, n_epochs) {
        return Ok(None);
    }
    let (radii, incident_weights) = input_radii().map_err(DensmapError::OriginalRadii)?;
    Ok(Some(DensityOptimizer::new(
        radii,
        &incident_weights,
        options,
    )))
}

fn optimize_layout(
    embedding: &mut [f32],
    graph: &CsrGraph,
    layout: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    negative_sampler: impl FnMut(usize) -> usize,
    density: Option<DensityOptimizer>,
    checkpoint: Option<impl FnMut(usize, &[f32])>,
) -> Result<(), DensmapError> {
    optimize_csr_with_density(
        embedding,
        layout.n_components,
        graph,
        layout.a,
        layout.b,
        layout.learning_rate,
        layout.repulsion_strength,
        layout.negative_sample_rate,
        layout.n_epochs,
        negative_sampler,
        OptimizeMode::SeededCompatibility,
        density,
        checkpoint,
    )
    .map_err(DensmapError::Optimize)
}

pub(crate) struct DensityOptimizer {
    options: DensmapOptions,
    original: Vec<f32>,
    membership_sum: f64,
    epoch: DensityEpoch,
    squared_deviations: Vec<f32>,
}

impl DensityOptimizer {
    fn new(mut original: Vec<f32>, incident_weights: &[f32], options: DensmapOptions) -> Self {
        let n_rows = original.len();
        let mut squared_deviations = vec![0.0; n_rows];
        let mean = mean(&original);
        let standard_deviation = variance(&original, mean, &mut squared_deviations).sqrt();
        for radius in &mut original {
            *radius = (*radius - mean) / standard_deviation;
        }
        Self {
            options,
            original,
            membership_sum: f64::from(pairwise_sum(incident_weights)) * 0.5,
            epoch: DensityEpoch {
                affinity_sums: vec![0.0; n_rows],
                log_radii: vec![0.0; n_rows],
                correlation_weights: vec![0.0; n_rows],
                scale_per_vertex: 0.0,
                standard_deviation: 0.0,
            },
            squared_deviations,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_epoch(
        &mut self,
        epoch: usize,
        n_epochs: usize,
        coordinates: &[f32],
        n_components: usize,
        graph: &CsrGraph,
        a: f64,
        b: f64,
    ) -> Option<&DensityEpoch> {
        if !self.options.active_at(epoch, n_epochs) {
            return None;
        }

        let DensityEpoch {
            affinity_sums,
            log_radii,
            correlation_weights,
            scale_per_vertex,
            standard_deviation,
        } = &mut self.epoch;
        affinity_sums.fill(0.0);
        log_radii.fill(0.0);
        let point = |row| &coordinates[row * n_components..(row + 1) * n_components];
        for (head, offsets) in graph.indptr.windows(2).enumerate() {
            for entry in offsets[0]..offsets[1] {
                if graph.data[entry] == 0.0 {
                    continue;
                }
                let tail = graph.indices[entry];
                let q = DistanceReduction::SeededCompatibility.squared(point(head), point(tail));
                let affinity = (1.0 + a * f64::from(q).powf(b)).recip();
                let weighted_distance = affinity * f64::from(q);
                for endpoint in [head, tail] {
                    affinity_sums[endpoint] =
                        (f64::from(affinity_sums[endpoint]) + affinity) as f32;
                    log_radii[endpoint] =
                        (f64::from(log_radii[endpoint]) + weighted_distance) as f32;
                }
            }
        }

        for (radius, &affinity_sum) in log_radii.iter_mut().zip(affinity_sums.iter()) {
            // Round the weighted mean before adding the offset and taking the log.
            *radius = (f64::from(*radius / affinity_sum) + 1e-8).ln() as f32;
        }
        let radius_mean = mean(log_radii);
        let shifted_var = variance(log_radii, radius_mean, &mut self.squared_deviations)
            + self.options.dens_var_shift as f32;
        *standard_deviation = shifted_var.sqrt();
        let correlation_variance = *standard_deviation * *standard_deviation;
        // The input radii are standardized already. The public calculation
        // uses their dot product with the uncentered embedding log radii.
        let covariance = radius_dot(log_radii, &self.original) / (graph.n_rows - 1) as f32;
        for ((weight, &radius), &original) in correlation_weights
            .iter_mut()
            .zip(log_radii.iter())
            .zip(&self.original)
        {
            *weight =
                f64::from(original - covariance * (radius - radius_mean) / correlation_variance);
        }
        *scale_per_vertex =
            2.0 * self.options.dens_lambda * self.membership_sum / graph.n_rows as f64;
        Some(&self.epoch)
    }
}

pub(crate) struct DensityEpoch {
    affinity_sums: Vec<f32>,
    log_radii: Vec<f32>,
    correlation_weights: Vec<f64>,
    scale_per_vertex: f64,
    standard_deviation: f32,
}

impl DensityEpoch {
    pub(crate) fn edge_scale(
        &self,
        head: usize,
        tail: usize,
        membership: f32,
        b: f64,
        affinity: f64,
        negative_log_affinity_derivative: f64,
    ) -> Result<f64, LayoutError> {
        let contribution = |row: usize| {
            let radius = f64::from(self.log_radii[row].exp());
            let dr_dq = affinity / f64::from(self.affinity_sums[row])
                * ((1.0 - b * (1.0 - affinity)) / radius + negative_log_affinity_derivative);
            self.correlation_weights[row] * dr_dq
        };
        let scale = self.scale_per_vertex * (contribution(head) + contribution(tail))
            / f64::from(membership * self.standard_deviation);
        if !scale.is_finite() {
            return Err(LayoutError::NonFiniteArithmetic {
                quantity: "density correlation gradient",
                value: scale as f32,
            });
        }
        Ok(scale)
    }
}

fn mean(values: &[f32]) -> f32 {
    pairwise_sum(values) / values.len() as f32
}

fn radius_dot(left: &[f32], right: &[f32]) -> f32 {
    // Match the float32 dot product used by the pinned NumPy x86-64 build.
    let (blocks_left, tail_left) = left.as_chunks::<32>();
    let (blocks_right, tail_right) = right.as_chunks::<32>();
    let mut lanes = [0.0; 32];
    for (left, right) in blocks_left.iter().zip(blocks_right) {
        for lane in 0..32 {
            lanes[lane] = left[lane].mul_add(right[lane], lanes[lane]);
        }
    }
    let [a, b, c, d] = std::array::from_fn(|i| {
        ((lanes[i] + lanes[i + 4]) + (lanes[i + 8] + lanes[i + 12]))
            + ((lanes[i + 16] + lanes[i + 20]) + (lanes[i + 24] + lanes[i + 28]))
    });
    let tail: f64 = tail_left
        .iter()
        .zip(tail_right)
        .map(|(&left, &right)| f64::from(left * right))
        .sum();
    (tail + f64::from((a + b) + (c + d))) as f32
}

fn variance(values: &[f32], mean: f32, squared_deviations: &mut [f32]) -> f32 {
    for (&value, squared) in values.iter().zip(squared_deviations.iter_mut()) {
        let centered = value - mean;
        *squared = centered * centered;
    }
    pairwise_sum(squared_deviations) / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::radius_dot;

    #[test]
    fn radius_dot_matches_numpy_at_block_boundaries() {
        let left: [f32; 65] = std::array::from_fn(|i| {
            (((i * i * 127 + 13 * i + 558) % 65521) as f32 - 32760.0) / 673.0
        });
        let right: [f32; 65] = std::array::from_fn(|i| {
            (((i * i * 913 + 173 * i + 337) % 65519) as f32 - 32759.0) / 797.0
        });
        // np.dot(left[:length], right[:length]) with the pinned NumPy wheel.
        for (length, expected) in [
            (31, 0x460b_729c),
            (32, 0x460a_6e0e),
            (33, 0x4601_9dff),
            (63, 0x45cd_b5c3),
            (64, 0x45cc_96ce),
            (65, 0x45af_f5b0),
        ] {
            assert_eq!(
                radius_dot(&left[..length], &right[..length]).to_bits(),
                expected,
                "length {length}"
            );
        }
    }
}
