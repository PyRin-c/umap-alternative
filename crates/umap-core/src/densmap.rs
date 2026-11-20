//! Density-preserving updates for the serial Euclidean compatibility layout.

use std::fmt;

use crate::density::{
    DensityError, LOG_OFFSET, RadiusSpace, radii_and_weights_from_edge_distances,
};
use crate::embedding::EmbeddingLayoutConfig;
use crate::fuzzy_graph::CsrGraph;
use crate::layout::{DistanceReduction, LayoutError};
use crate::numeric::pairwise_sum;
use crate::optimize::{OptimizeError, OptimizeMode, optimize_csr_with_density};

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

/// Failures in the supplied-graph densMAP optimizer.
#[derive(Clone, Debug, PartialEq)]
pub enum DensmapError {
    /// A density option is outside its supported finite range.
    InvalidOption { parameter: &'static str, value: f64 },
    /// The supplied graph or its input-space distances cannot be reduced.
    OriginalRadii(DensityError),
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
            Self::Optimize(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DensmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidOption { .. } => None,
            Self::OriginalRadii(error) => Some(error),
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
    options.validate()?;
    let density = if layout.n_epochs > 1 && options.active_at(layout.n_epochs - 1, layout.n_epochs)
    {
        Some(DensityOptimizer::new(graph, edge_distances, options)?)
    } else {
        None
    };
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
        Some(checkpoint),
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
    fn new(
        graph: &CsrGraph,
        distances: &[f32],
        options: DensmapOptions,
    ) -> Result<Self, DensmapError> {
        let (mut original, incident_weights) =
            radii_and_weights_from_edge_distances(graph, distances, RadiusSpace::Original)
                .map_err(DensmapError::OriginalRadii)?;
        let mut squared_deviations = vec![0.0; graph.n_rows];
        let mean = mean(&original);
        let standard_deviation = variance(&original, mean, &mut squared_deviations).sqrt();
        for radius in &mut original {
            *radius = (*radius - mean) / standard_deviation;
        }

        Ok(Self {
            options,
            original,
            membership_sum: f64::from(pairwise_sum(&incident_weights)) * 0.5,
            epoch: DensityEpoch {
                affinity_sums: vec![0.0; graph.n_rows],
                log_radii: vec![0.0; graph.n_rows],
                correlation_weights: vec![0.0; graph.n_rows],
                scale: 0.0,
            },
            squared_deviations,
        })
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
            scale,
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
                let weighted_distance = (affinity * f64::from(q)) as f32;
                for endpoint in [head, tail] {
                    affinity_sums[endpoint] += affinity as f32;
                    log_radii[endpoint] += weighted_distance;
                }
            }
        }

        for (radius, &affinity_sum) in log_radii.iter_mut().zip(affinity_sums.iter()) {
            *radius = (*radius / affinity_sum + LOG_OFFSET).ln();
        }
        let radius_mean = mean(log_radii);
        let shifted_var = f64::from(variance(
            log_radii,
            radius_mean,
            &mut self.squared_deviations,
        )) + self.options.dens_var_shift;
        let cross_product = log_radii
            .iter()
            .zip(&self.original)
            .fold(0.0_f32, |sum, (&radius, &original)| {
                (radius - radius_mean).mul_add(original, sum)
            });
        let covariance = f64::from(cross_product) / (graph.n_rows as f64 - 1.0);
        for ((weight, &radius), &original) in correlation_weights
            .iter_mut()
            .zip(log_radii.iter())
            .zip(&self.original)
        {
            *weight = f64::from(original)
                - covariance * (f64::from(radius) - f64::from(radius_mean)) / shifted_var;
        }
        *scale = 2.0 * self.options.dens_lambda * self.membership_sum
            / (graph.n_rows as f64 * shifted_var.sqrt());
        Some(&self.epoch)
    }
}

pub(crate) struct DensityEpoch {
    affinity_sums: Vec<f32>,
    log_radii: Vec<f32>,
    correlation_weights: Vec<f64>,
    scale: f64,
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
        let scale = self.scale * (contribution(head) + contribution(tail)) / f64::from(membership);
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

fn variance(values: &[f32], mean: f32, squared_deviations: &mut [f32]) -> f32 {
    for (&value, squared) in values.iter().zip(squared_deviations.iter_mut()) {
        let centered = value - mean;
        *squared = centered * centered;
    }
    pairwise_sum(squared_deviations) / values.len() as f32
}
