//! Initialization and rescaling for modern UMAP layouts.

use std::fmt;

use crate::fuzzy_graph::CsrGraph;
use crate::initialization::{
    InitializationError, checked_length, coincident_layout_noise_scale, has_coincident_rows,
    uniform_layout, validate_finite, write_mean,
};
use crate::layout::{LayoutError, validate_csr_rows, validate_csr_shape};
use crate::numeric::pairwise_sum;
use crate::rng::Mt19937;

/// Rank recursive hubs by degree and preserve the observed order of equal degrees.
pub fn recursive_hub_order(degrees: &[f32]) -> Vec<usize> {
    let mut vertices: Vec<_> = (0..degrees.len()).collect();
    sort_recursive_degrees(degrees, &mut vertices);
    vertices
}

fn sort_recursive_degrees(degrees: &[f32], mut vertices: &mut [usize]) {
    // The observed index quicksort switches to insertion sort below 16 rows.
    while vertices.len() >= 16 {
        let last = vertices.len() - 1;
        let middle = last / 2;
        for (left, right) in [(0, middle), (middle, last), (0, middle)] {
            if degrees[vertices[right]] < degrees[vertices[left]] {
                vertices.swap(left, right);
            }
        }
        let pivot = degrees[vertices[middle]];
        vertices.swap(middle, last);
        let (mut left, mut right) = (0, last - 1);
        loop {
            while degrees[vertices[left]] < pivot {
                left += 1;
            }
            while pivot < degrees[vertices[right]] {
                right -= 1;
            }
            if left >= right {
                break;
            }
            vertices.swap(left, right);
            left += 1;
            right -= 1;
        }
        vertices.swap(left, last);
        let (lower, pivot_and_upper) = vertices.split_at_mut(left);
        let upper = &mut pivot_and_upper[1..];
        if lower.len() < upper.len() {
            sort_recursive_degrees(degrees, lower);
            vertices = upper;
        } else {
            sort_recursive_degrees(degrees, upper);
            vertices = lower;
        }
    }
    for i in 1..vertices.len() {
        let value = vertices[i];
        let mut j = i;
        while j > 0 && degrees[value] < degrees[vertices[j - 1]] {
            vertices[j] = vertices[j - 1];
            j -= 1;
        }
        vertices[j] = value;
    }
}

/// How to scale the centered initialization before optimization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModernInitRescale {
    /// Return centered coordinates for a caller that handles rescaling.
    Unscaled,
    /// Use the geometric mean of the robust spans to preserve aspect ratio.
    #[default]
    Isotropic,
    /// Set the robust span of each coordinate independently.
    PerAxis,
    /// Set the membership-weighted median edge length.
    GraphEdge,
}

/// Coordinates or a seeded coordinate generator for the modern layout.
pub enum ModernInit<'a> {
    /// User-supplied row-major coordinates. Coincident rows trigger jitter.
    Explicit(&'a [f32]),
    /// Row-major uniform draws from the caller's RNG.
    Random,
    /// Row-major graph eigenvectors. The caller chooses the eigensolver.
    Spectral(&'a [f32]),
    /// Coordinates from a separate initializer, ready for rescaling.
    Prepared(&'a [f32]),
}

/// Shape and scaling settings for the modern initialization stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModernInitConfig {
    /// Number of coordinates per graph vertex.
    pub n_components: usize,
    /// Rule for selecting the multiplier or per-axis multipliers.
    pub rescale: ModernInitRescale,
    /// Multiplier for the robust target span or target median edge length.
    pub scale_factor: f64,
}

/// Validation failures before constructing a modern layout.
#[derive(Debug, Clone, PartialEq)]
pub enum ModernInitError {
    /// The row-major coordinate matrix is invalid.
    Coordinates(InitializationError),
    /// The input fuzzy graph is invalid.
    Graph(LayoutError),
    /// The graph contains no stored entries.
    EmptyGraph,
    /// The scale factor must be positive and finite.
    InvalidScale(f64),
    /// The supplied spectral coordinates have no nonzero amplitude.
    ZeroSpectralSpan,
}

impl fmt::Display for ModernInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinates(error) => error.fmt(f),
            Self::Graph(error) => error.fmt(f),
            Self::EmptyGraph => write!(
                f,
                "modern initialization requires a graph with stored edges"
            ),
            Self::InvalidScale(value) => write!(f, "invalid init_scale_factor: {value}"),
            Self::ZeroSpectralSpan => write!(f, "spectral coordinates have zero amplitude"),
        }
    }
}

impl std::error::Error for ModernInitError {}

impl From<InitializationError> for ModernInitError {
    fn from(error: InitializationError) -> Self {
        Self::Coordinates(error)
    }
}

impl From<LayoutError> for ModernInitError {
    fn from(error: LayoutError) -> Self {
        Self::Graph(error)
    }
}

/// Generate and rescale estimator coordinates on a square fuzzy graph.
///
/// The supplied graph is the graph at the initialization boundary. Every
/// stored positive membership participates in graph-edge scaling. This stage
/// advances the caller RNG through coordinate draws only. The caller owns the
/// Gaussian cache and any later sampling seed draws, including for zero
/// optimizer epochs. Robust-span rescaling uses the estimator target
/// `3 * log10(n_rows)` multiplied by the scale factor.
pub fn initialize_modern_layout(
    graph: &CsrGraph,
    init: ModernInit<'_>,
    config: ModernInitConfig,
    rng: &mut Mt19937,
    gaussian_cache: &mut Option<f64>,
) -> Result<Vec<f32>, ModernInitError> {
    let n_rows = graph.n_rows;
    let width = config.n_components;
    let length = checked_length(n_rows, width)?;
    if !config.scale_factor.is_finite() || config.scale_factor <= 0.0 {
        return Err(ModernInitError::InvalidScale(config.scale_factor));
    }
    if n_rows == 0 || width == 0 || graph.n_cols != n_rows {
        return Err(InitializationError::InvalidShape {
            n_rows,
            n_components: width,
            values: length,
        }
        .into());
    }
    validate_csr_shape(graph)?;
    validate_csr_rows(graph)?;
    if graph.indptr.last().copied() != Some(graph.data.len()) {
        return Err(LayoutError::InvalidGraphShape {
            n_rows,
            n_cols: graph.n_cols,
            indptr: graph.indptr.len(),
            indices: graph.indices.len(),
            data: graph.data.len(),
        }
        .into());
    }
    if graph.data.is_empty() {
        return Err(ModernInitError::EmptyGraph);
    }

    let mut coordinates = match init {
        ModernInit::Random => uniform_layout(rng, length),
        ModernInit::Explicit(values)
        | ModernInit::Spectral(values)
        | ModernInit::Prepared(values) => {
            if values.len() != length {
                return Err(InitializationError::InvalidShape {
                    n_rows,
                    n_components: width,
                    values: values.len(),
                }
                .into());
            }
            validate_finite(values)?;
            values.to_vec()
        }
    };

    match init {
        ModernInit::Explicit(_) if has_coincident_rows(&coordinates, width) => {
            let scale = coincident_layout_noise_scale(&coordinates, width);
            for coordinate in &mut coordinates {
                let noise = scale * rng.next_standard_normal(gaussian_cache);
                *coordinate = (f64::from(*coordinate) + noise) as f32;
            }
        }
        ModernInit::Spectral(_) => {
            let max_abs = coordinates.iter().fold(0.0_f32, |max, &x| max.max(x.abs()));
            if max_abs == 0.0 {
                return Err(ModernInitError::ZeroSpectralSpan);
            }
            let expansion = 4.0 / max_abs;
            for coordinate in &mut coordinates {
                let noise = (0.001 * rng.next_standard_normal(gaussian_cache)) as f32;
                *coordinate = expansion * *coordinate + noise;
            }
        }
        _ => {}
    }

    center_columns(&mut coordinates, width);
    match config.rescale {
        ModernInitRescale::Unscaled => {}
        ModernInitRescale::GraphEdge => {
            let multiplier = graph_edge_multiplier(graph, &coordinates, width, config.scale_factor);
            for coordinate in &mut coordinates {
                *coordinate *= multiplier;
            }
        }
        method @ (ModernInitRescale::Isotropic | ModernInitRescale::PerAxis) => {
            let target_span = 3.0 * (n_rows as f64).log10() * config.scale_factor;
            scale_robust_spans(&mut coordinates, width, method, target_span);
        }
    }

    Ok(coordinates)
}

fn center_columns(values: &mut [f32], width: usize) {
    let mut center = vec![0.0; width];
    if width == 1 {
        // The reduced axis is contiguous for one-coordinate layouts.
        center[0] = pairwise_sum(values) / values.len() as f32;
    } else {
        write_mean(&mut center, values.chunks_exact(width));
    }
    for row in values.chunks_exact_mut(width) {
        for (value, &mean) in row.iter_mut().zip(&center) {
            *value -= mean;
        }
    }
}

fn scale_robust_spans(
    values: &mut [f32],
    width: usize,
    method: ModernInitRescale,
    target_span: f64,
) {
    let mut column = Vec::with_capacity(values.len() / width);
    let mut spans = Vec::with_capacity(width);
    for i in 0..width {
        column.clear();
        column.extend(values.chunks_exact(width).map(|row| row[i]));
        column.sort_unstable_by(f32::total_cmp);
        let span = percentile(&column, 0.95) - percentile(&column, 0.05);
        spans.push(if span == 0.0 { 1.0 } else { span });
    }

    if method == ModernInitRescale::Isotropic {
        for span in &mut spans {
            *span = span.ln();
        }
        let geometric_mean = (pairwise_sum(&spans) / width as f32).exp();
        spans.fill(geometric_mean);
    }
    for span in &mut spans {
        *span = (target_span / f64::from(*span)) as f32;
    }
    for row in values.chunks_exact_mut(width) {
        for (coordinate, &multiplier) in row.iter_mut().zip(&spans) {
            *coordinate *= multiplier;
        }
    }
}

fn percentile(sorted: &[f32], probability: f64) -> f32 {
    let rank = (sorted.len() - 1) as f64 * probability;
    let index = rank as usize;
    let fraction = rank - index as f64;
    if fraction == 0.0 {
        return sorted[index];
    }
    let lower = f64::from(sorted[index]);
    let upper = f64::from(sorted[index + 1]);
    let interval = f64::from(sorted[index + 1] - sorted[index]);
    if fraction < 0.5 {
        (lower + fraction * interval) as f32
    } else {
        (upper - (1.0 - fraction) * interval) as f32
    }
}

fn graph_edge_multiplier(graph: &CsrGraph, values: &[f32], width: usize, target: f64) -> f32 {
    let mut edges = Vec::with_capacity(graph.data.len());
    let mut squares = Vec::with_capacity(width);
    for (source, row) in graph.indptr.windows(2).enumerate() {
        let from = &values[source * width..(source + 1) * width];
        for (&target, &weight) in graph.indices[row[0]..row[1]]
            .iter()
            .zip(&graph.data[row[0]..row[1]])
        {
            if weight == 0.0 {
                continue;
            }
            let to = &values[target * width..(target + 1) * width];
            squares.clear();
            squares.extend(from.iter().zip(to).map(|(&a, &b)| {
                let delta = a - b;
                delta * delta
            }));
            edges.push((pairwise_sum(&squares).sqrt(), f64::from(weight)));
        }
    }
    if edges.is_empty() {
        return 1.0;
    }
    edges.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    let half_weight = 0.5 * edges.iter().map(|edge| edge.1).sum::<f64>();
    let mut cumulative = 0.0;
    for (length, weight) in edges {
        cumulative += weight;
        if cumulative >= half_weight {
            return target as f32 / if length > 0.0 { length } else { 1.0 };
        }
    }
    unreachable!("the cumulative edge weight reaches half the total")
}
