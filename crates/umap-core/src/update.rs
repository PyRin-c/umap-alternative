//! Graph rebuilding and coordinate initialization for dense incremental fits.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign};

use crate::embedding::DenseEmbedding;
use crate::fit::{DenseGraphFitError, build_fitted_graph};
use crate::fuzzy_graph::GraphOptions;
use crate::initialization::{InitializationError, normalize_columns};
use crate::neighbors::{DistanceMetric, Neighbor, exhaustive_knn_with_metric};
use crate::rng::{Mt19937, draw_negative_seed};

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
        }
    }
}

impl std::error::Error for DenseUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDataShape { .. } => None,
            Self::Graph(error) => Some(error),
            Self::Initialization(error) => Some(error),
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
///
/// If provisional rows coincide, the noise scale is 0.001 times the mean
/// closest-other-row distance. The transform seed consumes the layout's
/// three negative-sampler draws before drawing the Gaussian noise. Finally,
/// each coordinate column is normalized over the combined layout.
///
/// The returned graph and coordinates own their data. Both inputs remain
/// read-only throughout. This stage finishes before epoch optimization.
pub fn prepare_dense_update(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    n_existing: usize,
    fitted: &[f32],
    config: &DenseUpdateConfig,
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
    validate_finite(fitted)?;
    let neighbors =
        exhaustive_knn_with_metric(data, n_rows, n_features, config.n_neighbors, config.metric)
            .map_err(|error| DenseUpdateError::Graph(error.into()))?;
    let graph = build_fitted_graph(&neighbors, config.graph_options)
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
    validate_finite(&coordinates)?;
    if has_coincident_rows(&coordinates, width) {
        perturb_coincident_layout(&mut coordinates, width, config.transform_seed);
    }
    normalize_columns(&mut coordinates, n_rows, width)?;
    Ok(DenseEmbedding {
        graph,
        coordinates,
        shape,
    })
}

fn validate_finite(layout: &[f32]) -> Result<(), InitializationError> {
    if let Some((index, &value)) = layout
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Err(InitializationError::NonFiniteValue { index, value })
    } else {
        Ok(())
    }
}

fn append_centroids(
    appended: &mut [f32],
    fitted: &[f32],
    width: usize,
    neighbors: &[Vec<Neighbor>],
) {
    let n_existing = fitted.len() / width;
    let mut fallback = vec![0.0; width];
    write_mean(&mut fallback, fitted.chunks_exact(width));
    for (coordinate, neighbors) in appended.chunks_exact_mut(width).zip(neighbors) {
        let selected = neighbors
            .iter()
            .map(|neighbor| neighbor.index)
            .filter(|&row| row < n_existing)
            .map(|row| &fitted[row * width..(row + 1) * width]);
        let mut selected = selected.peekable();
        if selected.peek().is_none() {
            coordinate.copy_from_slice(&fallback);
        } else {
            write_mean(coordinate, selected);
        }
    }
}

fn write_mean<'a>(output: &mut [f32], rows: impl Iterator<Item = &'a [f32]>) {
    if output.len() == 1 {
        // The reduced axis is contiguous in a one-coordinate layout.
        let column: Vec<_> = rows.map(|row| row[0]).collect();
        output[0] = pairwise_sum(&column) / column.len() as f32;
    } else {
        let mut count = 0;
        for row in rows {
            for (coordinate, &value) in output.iter_mut().zip(row) {
                *coordinate += value;
            }
            count += 1;
        }
        for coordinate in output {
            *coordinate /= count as f32;
        }
    }
}

fn has_coincident_rows(layout: &[f32], width: usize) -> bool {
    let mut rows: Vec<_> = layout.chunks_exact(width).collect();
    rows.sort_unstable_by(|left, right| {
        left.iter()
            .zip(*right)
            .map(|(left, right)| {
                if left == right {
                    Ordering::Equal
                } else {
                    left.total_cmp(right)
                }
            })
            .find(|&order| order != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    });
    rows.windows(2).any(|pair| pair[0] == pair[1])
}

fn perturb_coincident_layout(layout: &mut [f32], width: usize, seed: u32) {
    let count = layout.len() / width;
    let mut nearest = vec![f64::INFINITY; count];
    let mut squared_components = Vec::with_capacity(width);
    for (left_index, left) in layout.chunks_exact(width).enumerate() {
        for (right_index, right) in layout.chunks_exact(width).enumerate().skip(left_index + 1) {
            squared_components.clear();
            squared_components.extend(left.iter().zip(right).map(|(&left, &right)| {
                let difference = f64::from(left) - f64::from(right);
                difference * difference
            }));
            let squared_distance = pairwise_sum(&squared_components);
            nearest[left_index] = nearest[left_index].min(squared_distance);
            nearest[right_index] = nearest[right_index].min(squared_distance);
        }
    }
    for value in &mut nearest {
        *value = value.sqrt();
    }
    let scale = 0.001 * (pairwise_sum(&nearest) / count as f64);
    let mut generator = Mt19937::new(seed);
    draw_negative_seed(|| generator.next_u32());
    let mut spare = None;
    for coordinate in layout {
        let noise = scale * generator.next_standard_normal(&mut spare);
        *coordinate = (f64::from(*coordinate) + noise) as f32;
    }
}

// Contiguous reductions combine complete groups of eight before adding any tail.
fn pairwise_sum<T>(values: &[T]) -> T
where
    T: Copy + From<i8> + Add<Output = T> + AddAssign,
{
    if values.len() < 8 {
        return values.iter().copied().fold(T::from(0), Add::add);
    }
    if values.len() > 128 {
        let (left, right) = values.split_at((values.len() / 2) & !7);
        return pairwise_sum(left) + pairwise_sum(right);
    }
    let (first, rest) = values
        .split_first_chunk::<8>()
        .expect("at least eight values");
    let mut lanes = *first;
    let (blocks, tail) = rest.as_chunks::<8>();
    for block in blocks {
        for (lane, &value) in lanes.iter_mut().zip(block) {
            *lane += value;
        }
    }
    let [a, b, c, d, e, f, g, h] = lanes;
    tail.iter()
        .copied()
        .fold(((a + b) + (c + d)) + ((e + f) + (g + h)), Add::add)
}
