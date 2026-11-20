//! Dense inverse UMAP for a fixed embedding and its Delaunay tessellation.

use std::fmt;

use crate::fuzzy_graph::{batch_scale_floor, row_parameters};
use crate::layout::{LayoutError, validate_parameters, validate_repulsion_strength};
use crate::neighbors::{KnnError, euclidean_distance_wide, exhaustive_knn};
use crate::rng::{Mt19937, draw_negative_seed, sample_vertex};
use crate::sampling::{EdgeSamplingCursor, SamplingError};

/// Output of locating the queries in a Delaunay tessellation of the fitted embedding.
#[derive(Clone, Copy)]
pub struct InverseTriangulation<'a> {
    /// Row-major simplex vertex indices, with `n_components + 1` vertices per simplex.
    pub simplices: &'a [usize],
    /// One simplex index per query, or `None` for exterior queries.
    pub query_simplices: &'a [Option<usize>],
}

/// Fixed inputs to dense inverse transformation.
pub struct DenseInverseInput<'a> {
    /// Original dense training rows in row-major order.
    pub training: &'a [f32],
    /// Corresponding fitted embedding rows in row-major order.
    pub embedding: &'a [f32],
    /// Query coordinates in row-major order, in the fitted embedding space.
    pub queries: &'a [f32],
    /// Width of each original training row.
    pub n_features: usize,
    /// Width of each embedding and query row.
    pub n_components: usize,
    /// Tessellation and query locations in the embedding space.
    pub triangulation: InverseTriangulation<'a>,
}

/// Original Euclidean metric scale at a training row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseLocalScale {
    /// Local offset in feature space.
    pub rho: f32,
    /// Feature-space exponential scale.
    pub sigma: f32,
}

/// Parameters for the inverse optimizer following weighted initialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseInverseLayout {
    /// Actual inverse epochs, after converting the estimator's fit epoch setting.
    pub n_epochs: usize,
    /// Initial learning rate for the inverse optimizer.
    pub initial_alpha: f64,
    /// Magnitude of the repulsive term.
    pub repulsion_strength: f64,
    /// Negative samples per positive edge.
    pub negative_sample_rate: f64,
    /// Seed for the serial negative sample stream.
    pub seed: u32,
}

/// Feature-space reconstruction and the number of extrapolated queries.
#[derive(Debug, PartialEq)]
pub struct DenseInverseOutput {
    /// Row-major query reconstructions with the original feature count.
    pub coordinates: Vec<f32>,
    /// Number of queries outside the fitted embedding's convex hull.
    pub exterior_queries: usize,
}

/// Errors validating or reconstructing a dense inverse query batch.
#[derive(Debug, PartialEq)]
pub enum DenseInverseError {
    /// The training matrix shape or values are invalid.
    Training,
    /// The embedding matrix shape or values are invalid.
    Embedding,
    /// The queries are empty, nonfinite or have the wrong width.
    Queries,
    /// The tessellation or query locations are inconsistent with the supplied rows.
    Triangulation,
    /// At least one local metric scale is nonfinite or nonpositive, or the row count differs.
    LocalScales,
    /// The exhaustive Euclidean neighborhood cannot be constructed.
    Neighbors(KnnError),
    /// An active layout has a query whose memberships all underflowed to zero.
    NoMembership { row: usize },
    /// A curve or learning parameter is invalid.
    Layout(LayoutError),
    /// The sampling schedule cannot be represented.
    Sampling(SamplingError),
}

impl fmt::Display for DenseInverseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Training => f.write_str("invalid dense inverse training matrix"),
            Self::Embedding => f.write_str("invalid fitted embedding for dense inverse transform"),
            Self::Queries => f.write_str("invalid dense inverse query matrix"),
            Self::Triangulation => f.write_str("invalid dense inverse Delaunay tessellation"),
            Self::LocalScales => f.write_str("invalid dense inverse metric scales"),
            Self::Neighbors(error) => error.fmt(f),
            Self::NoMembership { row } => {
                write!(f, "inverse query row {row} has no positive membership")
            }
            Self::Layout(error) => error.fmt(f),
            Self::Sampling(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DenseInverseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout(error) => Some(error),
            Self::Sampling(error) => Some(error),
            Self::Neighbors(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SamplingError> for DenseInverseError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

/// Resolve the inverse epoch count from the configured maximum fit epoch.
///
/// For an epoch list, pass its maximum. `None` represents default fit epochs.
#[must_use]
pub const fn dense_inverse_epochs(fit_epochs: Option<usize>, n_query_rows: usize) -> usize {
    match fit_epochs {
        Some(epochs) => epochs / 3,
        None if n_query_rows <= 10_000 => 100,
        None => 30,
    }
}

/// Fit inverse metric scales for exhaustive Euclidean training with local connectivity one.
///
/// Supply the effective neighbor count used for fitting, including self.
pub fn fit_dense_inverse_scales(
    training: &[f32],
    n_features: usize,
    n_neighbors: usize,
) -> Result<Vec<InverseLocalScale>, DenseInverseError> {
    let n_rows = matrix_rows(training, n_features).ok_or(DenseInverseError::Training)?;
    let neighbors = exhaustive_knn(training, n_rows, n_features, n_neighbors)
        .map_err(DenseInverseError::Neighbors)?;
    let distances = neighbors
        .iter()
        .flatten()
        .map(|neighbor| neighbor.distance)
        .collect::<Vec<_>>();
    if distances.iter().any(|distance| !distance.is_finite()) {
        return Err(DenseInverseError::LocalScales);
    }
    let batch_floor = batch_scale_floor(&distances);
    Ok(distances
        .chunks_exact(n_neighbors)
        .map(|row| {
            let (rho, sigma) = row_parameters(row, 1.0, batch_floor);
            InverseLocalScale { rho, sigma }
        })
        .collect())
}

/// Reconstruct dense features by weighting neighbors in the embedding space.
///
/// The supplied tessellation provides geometry only. Rust chooses the starting
/// vertex, traverses its Delaunay neighbors and computes the inverse weights.
/// Tied distances preserve candidate discovery order.
pub fn initialize_dense_inverse(
    input: &DenseInverseInput<'_>,
    a: f64,
    b: f64,
) -> Result<DenseInverseOutput, DenseInverseError> {
    initialize(input, a, b).map(|(output, _)| output)
}

/// Reconstruct dense Euclidean features and run serial inverse layout updates.
///
/// Supply the offsets and scales fitted in original feature space in training
/// row order. This function uses the fitted data as fixed tails and moves only
/// the reconstructed queries.
pub fn inverse_transform_dense(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    a: f64,
    b: f64,
    layout: DenseInverseLayout,
) -> Result<DenseInverseOutput, DenseInverseError> {
    validate_parameters(a, b, layout.initial_alpha).map_err(DenseInverseError::Layout)?;
    validate_repulsion_strength(layout.repulsion_strength).map_err(DenseInverseError::Layout)?;
    let (mut output, edges) = initialize(input, a, b)?;
    if layout.n_epochs <= 1 || layout.initial_alpha == 0.0 {
        return Ok(output);
    }
    let training = input.training.chunks_exact(input.n_features);
    if scales.len() != training.len()
        || scales.iter().any(|scale| {
            !scale.rho.is_finite()
                || scale.rho < 0.0
                || !scale.sigma.is_finite()
                || scale.sigma <= 0.0
        })
    {
        return Err(DenseInverseError::LocalScales);
    }
    if let Some(row) = edges
        .chunk_by(|a, b| a.row == b.row)
        .find(|row| row.iter().all(|edge| edge.weight == 0.0))
    {
        return Err(DenseInverseError::NoMembership { row: row[0].row });
    }
    optimize(input, scales, &edges, &mut output.coordinates, layout)?;
    Ok(output)
}

#[derive(Clone, Copy)]
struct InverseEdge {
    row: usize,
    vertex: usize,
    weight: f64,
}

fn initialize(
    input: &DenseInverseInput<'_>,
    a: f64,
    b: f64,
) -> Result<(DenseInverseOutput, Vec<InverseEdge>), DenseInverseError> {
    validate_parameters(a, b, 0.0).map_err(DenseInverseError::Layout)?;
    let DenseInverseInput {
        training,
        embedding,
        queries,
        n_features,
        n_components,
        triangulation,
    } = *input;
    let n_training = matrix_rows(training, n_features).ok_or(DenseInverseError::Training)?;
    if matrix_rows(embedding, n_components) != Some(n_training) {
        return Err(DenseInverseError::Embedding);
    }
    let n_query = matrix_rows(queries, n_components).ok_or(DenseInverseError::Queries)?;
    let size = n_query
        .checked_mul(n_features)
        .ok_or(DenseInverseError::Queries)?;
    let adjacency = build_adjacency(&triangulation, n_training, n_query, n_components)?;
    let vertex_coordinates = |i: usize| &embedding[i * n_components..][..n_components];
    let simplex_width = n_components + 1;
    let k = n_features.min(n_training);
    let mut output = DenseInverseOutput {
        coordinates: vec![0.0; size],
        exterior_queries: 0,
    };
    let mut edges = Vec::new();
    let mut seen = vec![usize::MAX; n_training];
    let mut candidates = Vec::new();
    let mut ranked = Vec::new();
    let mut weights = Vec::new();
    for (row, (query, simplex)) in queries
        .chunks_exact(n_components)
        .zip(triangulation.query_simplices)
        .enumerate()
    {
        let distance = |vertex| euclidean_distance_wide(query, vertex_coordinates(vertex));
        let seed = if let Some(simplex) = *simplex {
            triangulation.simplices[simplex * simplex_width]
        } else {
            output.exterior_queries += 1;
            (0..n_training)
                .map(|vertex| (vertex, exterior_distance(query, vertex_coordinates(vertex))))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("training rows are nonempty")
                .0
        };
        candidates.clear();
        candidates.push(seed);
        seen[seed] = row;
        // The queue is also the discovered candidate list. Each expansion can
        // discover several vertices, even on the last permitted expansion.
        for i in 0..k {
            let Some(&vertex) = candidates.get(i) else {
                break;
            };
            for &neighbor in &adjacency[vertex] {
                if seen[neighbor] != row {
                    seen[neighbor] = row;
                    candidates.push(neighbor);
                }
            }
        }
        if candidates.len() < k {
            return Err(DenseInverseError::Triangulation);
        }
        ranked.clear();
        ranked.extend(candidates.iter().map(|&vertex| (vertex, distance(vertex))));
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        ranked.truncate(k);
        weights.clear();
        weights.extend(ranked.iter().map(|&(vertex, distance)| InverseEdge {
            row,
            vertex,
            weight: 1.0 / (1.0 + a * distance.powf(2.0 * b)),
        }));
        // The optimizer visits distance-ranked edges. Initialization traverses
        // the same edges in original row order, rounding after each addition.
        edges.extend_from_slice(&weights);
        weights.sort_unstable_by_key(|edge| edge.vertex);
        let total = weights.iter().map(|edge| edge.weight).sum::<f64>();
        if !total.is_finite() {
            return Err(DenseInverseError::NoMembership { row });
        }
        if total == 0.0 {
            continue;
        }
        let coordinates = &mut output.coordinates[row * n_features..][..n_features];
        for &edge in &weights {
            let features = &training[edge.vertex * n_features..][..n_features];
            let weight = edge.weight / total;
            for (out, &value) in coordinates.iter_mut().zip(features) {
                *out = (f64::from(*out) + f64::from(value) * weight) as f32;
            }
        }
    }
    Ok((output, edges))
}

fn matrix_rows(matrix: &[f32], width: usize) -> Option<usize> {
    (width > 0
        && !matrix.is_empty()
        && matrix.len().is_multiple_of(width)
        && matrix.iter().all(|value| value.is_finite()))
    .then(|| matrix.len() / width)
}

fn exterior_distance(query: &[f32], vertex: &[f32]) -> f32 {
    // Exterior seeding ranks the squared sums at float32 precision.
    query
        .iter()
        .zip(vertex)
        .map(|(&x, &y)| {
            let delta = x - y;
            delta * delta
        })
        .sum()
}

fn build_adjacency(
    triangulation: &InverseTriangulation<'_>,
    n_training: usize,
    n_query: usize,
    n_components: usize,
) -> Result<Vec<Vec<usize>>, DenseInverseError> {
    let Some(width) = n_components.checked_add(1) else {
        return Err(DenseInverseError::Triangulation);
    };
    let simplices = triangulation.simplices.chunks_exact(width);
    if n_components < 2
        || n_training <= width
        || simplices.len() == 0
        || !simplices.remainder().is_empty()
        || triangulation.query_simplices.len() != n_query
        || triangulation
            .query_simplices
            .iter()
            .flatten()
            .any(|&i| i >= simplices.len())
    {
        return Err(DenseInverseError::Triangulation);
    }
    let mut adjacency = vec![Vec::new(); n_training];
    for simplex in simplices {
        for (i, &vertex) in simplex.iter().enumerate() {
            if vertex >= n_training {
                return Err(DenseInverseError::Triangulation);
            }
            for &other in &simplex[..i] {
                if vertex == other {
                    return Err(DenseInverseError::Triangulation);
                }
                adjacency[vertex].push(other);
                adjacency[other].push(vertex);
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    Ok(adjacency)
}

fn optimize(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    edges: &[InverseEdge],
    coordinates: &mut [f32],
    layout: DenseInverseLayout,
) -> Result<(), DenseInverseError> {
    const EPSILON: f64 = 1e-6;
    EdgeSamplingCursor::from_positive_interval(1.0, layout.negative_sample_rate)?;
    let maximum = edges.iter().map(|e| e.weight).fold(0.0_f64, f64::max);
    let n_epochs = layout.n_epochs as f64;
    let mut scheduled = Vec::with_capacity(edges.len());
    for edge in edges {
        let expected_samples = n_epochs * (edge.weight / maximum);
        if expected_samples >= 1.0 {
            let cursor = EdgeSamplingCursor::from_positive_interval(
                n_epochs / expected_samples,
                layout.negative_sample_rate,
            )?;
            scheduled.push((edge, cursor));
        }
    }
    let mut rng = Mt19937::new(layout.seed);
    let mut random_state = draw_negative_seed(|| rng.next_u32());
    let features = |vertex: usize| &input.training[vertex * input.n_features..][..input.n_features];
    let mut alpha = layout.initial_alpha;
    for epoch in 0..layout.n_epochs {
        let time = epoch as f64;
        for (edge, cursor) in &mut scheduled {
            if !cursor.positive_due(time)? {
                continue;
            }
            let current = &mut coordinates[edge.row * input.n_features..][..input.n_features];
            let tail = features(edge.vertex);
            let dist = euclidean_distance_wide(current, tail);
            let coefficient = -1.0 / (edge.weight * f64::from(scales[edge.vertex].sigma) + EPSILON);
            update_features(current, tail, dist + EPSILON, coefficient, alpha);
            for _ in 0..cursor.take_negative_samples(time)? {
                let vertex = sample_vertex(&mut random_state, scales.len());
                if layout.repulsion_strength == 0.0 {
                    continue;
                }
                let tail = features(vertex);
                let dist = euclidean_distance_wide(current, tail);
                let scale = scales[vertex];
                let sigma = f64::from(scale.sigma);
                let offset = (dist - f64::from(scale.rho)).max(0.0);
                let membership = (-offset / (sigma + EPSILON)).exp();
                let coefficient =
                    layout.repulsion_strength * membership / ((1.0 - membership) * sigma + EPSILON);
                update_features(current, tail, dist + EPSILON, coefficient, alpha);
            }
        }
        alpha = layout.initial_alpha * (1.0 - time / n_epochs);
    }
    Ok(())
}

fn update_features(current: &mut [f32], tail: &[f32], distance: f64, coefficient: f64, alpha: f64) {
    for (value, &reference) in current.iter_mut().zip(tail) {
        let gradient = (f64::from(*value - reference) / distance) * coefficient;
        *value = (f64::from(*value) + alpha * gradient.clamp(-4.0, 4.0)) as f32;
    }
}
