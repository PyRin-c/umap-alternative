//! Graph rebuilding and coordinate initialization for dense incremental fits.

use crate::embedding::{DenseEmbedding, EmbeddingLayoutConfig, zero_weak_memberships};
use crate::fit::{DenseGraphFitError, build_fitted_graph, build_graph_with_cutoff};
use crate::fuzzy_graph::{CsrGraph, GraphOptions};
use crate::initialization::{
    InitializationError, coincident_layout_noise_scale, has_coincident_rows, normalize_columns,
    validate_finite, write_mean,
};
use crate::layout::{validate_parameters, validate_repulsion_strength};
use crate::neighbors::{DistanceMetric, Neighbor, exhaustive_knn_with_metric};
use crate::optimize::OptimizeError;
use crate::rng::Mt19937;
use crate::sampling::EdgeSamplingCursor;
use crate::update_layout::{begin_update_layout_rng, optimize_update_layout};
use std::fmt;

/// Fitted settings and transform seed used to prepare a dense update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseUpdateConfig {
    /// Effective neighbor count retained from the original fit.
    pub n_neighbors: usize,
    /// Feature metric used by the graph.
    pub metric: DistanceMetric,
    /// Number of fitted coordinates per row.
    pub n_components: usize,
    /// Optional local connectivity and fuzzy-set mixing settings.
    pub graph_options: Option<GraphOptions>,
    /// Seed for resolving repeated provisional coordinates.
    pub transform_seed: u32,
}

/// Failures while preparing the graph and initial coordinates for an update.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseUpdateError {
    /// An update requires existing rows, new rows, and a positive feature width.
    InvalidDataShape {
        n_existing: usize,
        n_rows: usize,
        n_features: usize,
    },
    /// Feature validation, neighbor search or fuzzy graph construction failed.
    Graph(DenseGraphFitError),
    /// The fitted coordinates or provisional layout cannot be normalized.
    Initialization(InitializationError),
    /// Duplicate provisional coordinates require a finite layout for jitter.
    /// The combined graph was rebuilt before this error.
    CoincidentNonFiniteLayout { graph: CsrGraph },
    /// The optimizer uses a different coordinate width.
    LayoutWidth { fitted: usize, optimizer: usize },
    /// The optimizer rejected the graph, parameters, or a coordinate update.
    Optimization(OptimizeError),
}

impl fmt::Display for DenseUpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDataShape {
                n_existing,
                n_rows,
                n_features,
            } => write!(
                formatter,
                "update shape {n_rows}x{n_features} cannot extend {n_existing} existing rows"
            ),
            Self::Graph(error) => error.fmt(formatter),
            Self::Initialization(error) => error.fmt(formatter),
            Self::CoincidentNonFiniteLayout { .. } => formatter.write_str("Input contains NaN."),
            Self::LayoutWidth { fitted, optimizer } => {
                write!(
                    formatter,
                    "optimizer width {optimizer} differs from fitted width {fitted}"
                )
            }
            Self::Optimization(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DenseUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDataShape { .. }
            | Self::LayoutWidth { .. }
            | Self::CoincidentNonFiniteLayout { .. } => None,
            Self::Graph(error) => Some(error),
            Self::Initialization(error) => Some(error),
            Self::Optimization(error) => Some(error),
        }
    }
}

impl From<InitializationError> for DenseUpdateError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

/// Prepare a graph and normalized initial coordinates for a dense update.
///
/// `data` is the combined row-major feature matrix, with all `n_existing`
/// fitted rows preceding appended rows. For each appended row, average its
/// fitted neighbors in distance order, or use the mean of the fitted layout
/// if none of its selected neighbors were fitted before this update.
/// Rebuilding the graph uses the fitted metric without a distance cutoff.
///
/// If provisional rows coincide, the noise scale is 0.001 times the mean
/// closest-other-row distance. The transform seed consumes the layout's
/// three negative-sampler draws before drawing the Gaussian noise. Finally,
/// each coordinate column is normalized over the combined layout.
///
/// The returned graph and coordinates own their data. Both inputs remain
/// read-only throughout. This stage finishes before epoch optimization.
/// Rows masked entirely with NaNs in a multi-row fit propagate through column normalization.
/// If the provisional layout also has duplicate rows, preparation reports
/// the rebuilt graph and a nonfinite-layout error.
pub fn prepare_dense_update(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_existing: usize,
    fitted: &[f32],
    config: &DenseUpdateConfig,
) -> Result<DenseEmbedding, DenseUpdateError> {
    let mut rng = begin_update_layout_rng(config.transform_seed);
    prepare_with_rng(
        data, n_rows, n_features, n_existing, fitted, config, &mut rng,
    )
}

/// Rebuild and optimize a serial dense embedding after appending feature rows.
///
/// Pass the same inputs as [`prepare_dense_update`] and the layout settings
/// for this update. Use the fitted curve and learning rate, and pass zero
/// epochs for an omitted update epoch count. The optimizer starts with the
/// RNG stream left by preparation and can move both old and new points.
/// Memberships below the sampling cutoff are retained as stored CSR zeros.
pub fn update_dense_embedding(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_existing: usize,
    fitted: &[f32],
    config: &DenseUpdateConfig,
    layout: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
) -> Result<DenseEmbedding, DenseUpdateError> {
    if layout.n_components != config.n_components {
        return Err(DenseUpdateError::LayoutWidth {
            fitted: config.n_components,
            optimizer: layout.n_components,
        });
    }
    let mut rng = begin_update_layout_rng(config.transform_seed);
    let mut embedding = prepare_with_rng(
        data, n_rows, n_features, n_existing, fitted, config, &mut rng,
    )?;
    zero_weak_memberships(&mut embedding.graph, layout.n_epochs, false);
    if embedding.coordinates[0].is_nan() {
        // Rows masked by the original fit cause every normalized coordinate
        // to be NaN. No attraction or repulsion can restore a finite coordinate.
        validate_parameters(
            layout.a.into(),
            layout.b.into(),
            layout.learning_rate.into(),
        )
        .map_err(|error| DenseUpdateError::Optimization(error.into()))?;
        validate_repulsion_strength(layout.repulsion_strength.into())
            .map_err(|error| DenseUpdateError::Optimization(error.into()))?;
        EdgeSamplingCursor::from_positive_interval(1.0, layout.negative_sample_rate)
            .map_err(|error| DenseUpdateError::Optimization(error.into()))?;
    } else {
        optimize_update_layout(
            &mut embedding.coordinates,
            &embedding.graph,
            layout,
            &mut rng,
        )
        .map_err(DenseUpdateError::Optimization)?;
    }
    Ok(embedding)
}

fn prepare_with_rng(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_existing: usize,
    fitted: &[f32],
    config: &DenseUpdateConfig,
    rng: &mut Mt19937,
) -> Result<DenseEmbedding, DenseUpdateError> {
    if n_existing == 0 || n_existing >= n_rows || n_features == 0 {
        return Err(DenseUpdateError::InvalidDataShape {
            n_existing,
            n_rows,
            n_features,
        });
    }
    let width = config.n_components;
    let shape = (n_rows, width);
    let length = n_rows
        .checked_mul(width)
        .ok_or(InitializationError::ShapeOverflow {
            n_rows,
            n_components: width,
        })?;
    if width == 0 || n_existing * width != fitted.len() {
        return Err(InitializationError::InvalidShape {
            n_rows: n_existing,
            n_components: width,
            values: fitted.len(),
        }
        .into());
    }
    let mut has_masked_rows = false;
    for (row, coordinates) in fitted.chunks_exact(width).enumerate() {
        if n_existing > 1 && coordinates.iter().all(|value| value.is_nan()) {
            has_masked_rows = true;
        } else {
            validate_finite(coordinates).map_err(|error| match error {
                InitializationError::NonFiniteValue { index, value } => {
                    InitializationError::NonFiniteValue {
                        index: row * width + index,
                        value,
                    }
                }
                other => other,
            })?;
        }
    }
    let neighbors =
        exhaustive_knn_with_metric(data, n_rows, n_features, config.n_neighbors, config.metric)
            .map_err(|error| DenseUpdateError::Graph(error.into()))?;
    let graph = if config.metric.is_angular() {
        build_graph_with_cutoff(
            &neighbors,
            config.n_neighbors,
            f32::INFINITY,
            config.graph_options.unwrap_or_default(),
        )
    } else {
        build_fitted_graph(&neighbors, config.graph_options)
    }
    .map_err(|error| DenseUpdateError::Graph(error.into()))?;
    let mut coordinates = Vec::with_capacity(length);
    coordinates.extend_from_slice(fitted);
    coordinates.resize(length, 0.0);
    append_centroids(
        &mut coordinates[fitted.len()..],
        fitted,
        width,
        &neighbors[n_existing..],
    );
    if has_masked_rows {
        if has_coincident_rows(&coordinates, width) {
            return Err(DenseUpdateError::CoincidentNonFiniteLayout { graph });
        }
        coordinates.fill(f32::NAN);
    } else {
        validate_finite(&coordinates)?;
        if has_coincident_rows(&coordinates, width) {
            perturb_coincident_layout(&mut coordinates, width, rng);
        }
        normalize_columns(&mut coordinates, n_rows, width)?;
    }
    Ok(DenseEmbedding {
        graph,
        coordinates,
        shape,
    })
}

fn append_centroids(
    appended: &mut [f32],
    fitted: &[f32],
    width: usize,
    neighbors: &[Vec<Neighbor>],
) {
    let n_existing = fitted.len() / width;
    let mut fallback = None;
    for (coordinate, neighbors) in appended.chunks_exact_mut(width).zip(neighbors) {
        let selected = neighbors
            .iter()
            .map(|neighbor| neighbor.index)
            .filter(|&row| row < n_existing)
            .map(|row| &fitted[row * width..(row + 1) * width]);
        let mut selected = selected.peekable();
        if selected.peek().is_none() {
            let mean = fallback.get_or_insert_with(|| {
                let mut totals = vec![0.0_f64; width];
                for row in fitted.chunks_exact(width) {
                    for (total, &value) in totals.iter_mut().zip(row) {
                        *total += f64::from(value);
                    }
                }
                totals
                    .into_iter()
                    .map(|sum| (sum / n_existing as f64) as f32)
                    .collect::<Vec<_>>()
            });
            coordinate.copy_from_slice(mean);
        } else {
            write_mean(coordinate, selected);
        }
    }
}

fn perturb_coincident_layout(layout: &mut [f32], width: usize, rng: &mut Mt19937) {
    let scale = coincident_layout_noise_scale(layout, width);
    let mut spare = None;
    for coordinate in layout {
        let noise = (scale * rng.next_standard_normal(&mut spare)) as f32;
        *coordinate += noise;
    }
}
