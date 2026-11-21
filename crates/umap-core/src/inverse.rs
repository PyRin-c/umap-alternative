//! Dense inverse UMAP for a fixed embedding and its Delaunay tessellation.

use std::fmt;

use crate::fuzzy_graph::supplied_neighbors::ranked_scales;
use crate::fuzzy_graph::{GraphOptions, batch_scale_floor, row_parameters};
use crate::layout::{LayoutError, validate_parameters, validate_repulsion_strength};
use crate::neighbors::{
    DistanceMetric, KnnError, bray_curtis_sums, correlation_products, cosine_products,
    euclidean_distance_wide, exhaustive_knn_with_metric,
};
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

/// Original metric scale at a training row.
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

/// Select the closest candidate positions in distance order when distances tie.
pub type InverseCandidateRanker<'a> = dyn FnMut(&[f64], usize) -> Vec<usize> + 'a;

/// Errors validating or reconstructing a dense inverse query batch.
#[derive(Debug, PartialEq)]
pub enum DenseInverseError {
    /// A custom feature metric returned an invalid distance or gradient.
    InvalidGradient,
    /// A tie ranker returned invalid positions or distances out of order.
    CandidateOrder,
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
    /// The feature metric has no inverse gradient.
    UnsupportedMetric(DistanceMetric),
}

impl fmt::Display for DenseInverseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGradient => f.write_str("invalid inverse metric distance or gradient"),
            Self::CandidateOrder => f.write_str("invalid inverse candidate order"),
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
            Self::UnsupportedMetric(metric) => {
                write!(f, "inverse transform is unavailable for {metric:?}")
            }
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
    fit_dense_inverse_scales_with_options(
        training,
        n_features,
        n_neighbors,
        GraphOptions::default(),
    )
}

/// Fit inverse metric scales using the fitted local connectivity.
pub fn fit_dense_inverse_scales_with_options(
    training: &[f32],
    n_features: usize,
    n_neighbors: usize,
    options: GraphOptions,
) -> Result<Vec<InverseLocalScale>, DenseInverseError> {
    fit_dense_inverse_scales_with_metric(
        training,
        n_features,
        n_neighbors,
        DistanceMetric::Euclidean,
        None,
        options,
    )
}

/// Fit inverse scales with the original feature metric and optional inclusive cutoff.
pub fn fit_dense_inverse_scales_with_metric(
    training: &[f32],
    n_features: usize,
    n_neighbors: usize,
    metric: DistanceMetric,
    cutoff: Option<f32>,
    options: GraphOptions,
) -> Result<Vec<InverseLocalScale>, DenseInverseError> {
    let n_rows = matrix_rows(training, n_features).ok_or(DenseInverseError::Training)?;
    let neighbors = exhaustive_knn_with_metric(training, n_rows, n_features, n_neighbors, metric)
        .map_err(DenseInverseError::Neighbors)?;
    let distances = neighbors
        .iter()
        .flatten()
        .map(|neighbor| {
            if cutoff.is_some_and(|cutoff| neighbor.distance >= cutoff) {
                f32::INFINITY
            } else {
                neighbor.distance
            }
        })
        .collect::<Vec<_>>();
    if cutoff.is_some() || metric.uses_ranked_graph() {
        return fit_inverse_scales_from_ranked_distances(
            &distances,
            n_neighbors,
            n_neighbors,
            options,
        );
    }
    if distances.iter().any(|distance| !distance.is_finite()) {
        return Err(DenseInverseError::LocalScales);
    }
    let batch_floor = batch_scale_floor(&distances);
    Ok(distances
        .chunks_exact(n_neighbors)
        .map(|row| {
            let (rho, sigma) = row_parameters(row, options.local_connectivity(), batch_floor);
            InverseLocalScale { rho, sigma }
        })
        .collect())
}

/// Recover inverse scales from the same ranked distances used to construct the fit graph.
///
/// Distances are row-major with the supplied width, including positive
/// infinities at missing positions. The target count can be smaller than
/// the supplied width.
pub fn fit_inverse_scales_from_ranked_distances(
    distances: &[f32],
    width: usize,
    target_neighbors: usize,
    options: GraphOptions,
) -> Result<Vec<InverseLocalScale>, DenseInverseError> {
    if !(1..=width).contains(&target_neighbors)
        || distances.is_empty()
        || !distances.len().is_multiple_of(width)
        || distances
            .iter()
            .any(|&d| d.is_nan() || d == f32::NEG_INFINITY)
    {
        return Err(DenseInverseError::LocalScales);
    }
    Ok(ranked_scales(distances, width, target_neighbors, options)
        .map(|(rho, sigma)| InverseLocalScale { rho, sigma })
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
    initialize(input, a, b, None).map(|(output, _)| output)
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
    inverse_transform_dense_with_metric(input, scales, a, b, layout, DistanceMetric::Euclidean)
}

/// Reconstruct dense features with an original-space metric gradient.
pub fn inverse_transform_dense_with_metric(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    a: f64,
    b: f64,
    layout: DenseInverseLayout,
    metric: DistanceMetric,
) -> Result<DenseInverseOutput, DenseInverseError> {
    inverse_transform_dense_with_metric_and_ranker(input, scales, a, b, layout, metric, None)
}

/// Reconstruct features with a metric and optional caller-defined distance tie order.
pub fn inverse_transform_dense_with_metric_and_ranker(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    a: f64,
    b: f64,
    layout: DenseInverseLayout,
    metric: DistanceMetric,
    ranker: Option<&mut InverseCandidateRanker<'_>>,
) -> Result<DenseInverseOutput, DenseInverseError> {
    if metric.is_boolean() {
        return Err(DenseInverseError::UnsupportedMetric(metric));
    }
    inverse_transform_dense_with_gradient_and_ranker(
        input,
        scales,
        a,
        b,
        layout,
        |left, right, gradient| Ok(metric_gradient(metric, left, right, gradient)),
        ranker,
    )
}

/// Reconstruct features using a caller-provided original-space distance and gradient.
pub fn inverse_transform_dense_with_gradient(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    a: f64,
    b: f64,
    layout: DenseInverseLayout,
    gradient: impl FnMut(&[f32], &[f32], &mut [f64]) -> Result<f64, DenseInverseError>,
) -> Result<DenseInverseOutput, DenseInverseError> {
    inverse_transform_dense_with_gradient_and_ranker(input, scales, a, b, layout, gradient, None)
}

/// Reconstruct features with a caller-defined metric gradient and optional distance tie order.
///
/// When a tie affects the selected prefix, the ranker receives the full list
/// of candidate distances in discovery order and returns the closest positions
/// in distance order. Without a ranker, equal distances keep discovery order.
pub fn inverse_transform_dense_with_gradient_and_ranker(
    input: &DenseInverseInput<'_>,
    scales: &[InverseLocalScale],
    a: f64,
    b: f64,
    layout: DenseInverseLayout,
    gradient: impl FnMut(&[f32], &[f32], &mut [f64]) -> Result<f64, DenseInverseError>,
    ranker: Option<&mut InverseCandidateRanker<'_>>,
) -> Result<DenseInverseOutput, DenseInverseError> {
    validate_parameters(a, b, layout.initial_alpha).map_err(DenseInverseError::Layout)?;
    validate_repulsion_strength(layout.repulsion_strength).map_err(DenseInverseError::Layout)?;
    let (mut output, edges) = initialize(input, a, b, ranker)?;
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
    optimize(
        input,
        scales,
        &edges,
        &mut output.coordinates,
        layout,
        gradient,
    )?;
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
    mut ranker: Option<&mut InverseCandidateRanker<'_>>,
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
    let mut distances = Vec::new();
    let mut order = Vec::new();
    let mut selected = Vec::new();
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
        distances.clear();
        distances.extend(candidates.iter().map(|&vertex| distance(vertex)));
        order.clear();
        order.extend(0..candidates.len());
        order.sort_by(|&left, &right| distances[left].total_cmp(&distances[right]));
        let ranked_ties;
        let order = match ranker.as_deref_mut() {
            Some(rank)
                if order
                    .windows(2)
                    .take(k)
                    .any(|pair| distances[pair[0]] == distances[pair[1]]) =>
            {
                ranked_ties = rank(&distances, k);
                selected.clear();
                selected.resize(distances.len(), false);
                if ranked_ties.len() != k
                    || ranked_ties
                        .iter()
                        .zip(&order)
                        .any(|(&position, &expected)| {
                            position >= distances.len()
                                || distances[position] != distances[expected]
                                || std::mem::replace(&mut selected[position], true)
                        })
                {
                    return Err(DenseInverseError::CandidateOrder);
                }
                &ranked_ties
            }
            _ => &order[..k],
        };
        weights.clear();
        weights.extend(order.iter().map(|&index| InverseEdge {
            row,
            vertex: candidates[index],
            weight: 1.0 / (1.0 + a * distances[index].powf(2.0 * b)),
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
    mut metric: impl FnMut(&[f32], &[f32], &mut [f64]) -> Result<f64, DenseInverseError>,
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
    let mut gradient = vec![0.0; input.n_features];
    let mut alpha = layout.initial_alpha;
    for epoch in 0..layout.n_epochs {
        let time = epoch as f64;
        for (edge, cursor) in &mut scheduled {
            if !cursor.positive_due(time)? {
                continue;
            }
            let current = &mut coordinates[edge.row * input.n_features..][..input.n_features];
            let tail = features(edge.vertex);
            metric(current, tail, &mut gradient)?;
            let coefficient = -1.0 / (edge.weight * f64::from(scales[edge.vertex].sigma) + EPSILON);
            update_features(current, &gradient, coefficient, alpha);
            for _ in 0..cursor.take_negative_samples(time)? {
                let vertex = sample_vertex(&mut random_state, scales.len());
                if layout.repulsion_strength == 0.0 {
                    continue;
                }
                let tail = features(vertex);
                let dist = metric(current, tail, &mut gradient)?;
                let scale = scales[vertex];
                let sigma = f64::from(scale.sigma);
                let offset = (dist - f64::from(scale.rho)).max(0.0);
                let membership = (-offset / (sigma + EPSILON)).exp();
                let coefficient =
                    layout.repulsion_strength * membership / ((1.0 - membership) * sigma + EPSILON);
                update_features(current, &gradient, coefficient, alpha);
            }
        }
        alpha = layout.initial_alpha * (1.0 - time / n_epochs);
    }
    Ok(())
}

fn update_features(current: &mut [f32], gradient: &[f64], coefficient: f64, alpha: f64) {
    for (value, &component) in current.iter_mut().zip(gradient) {
        *value = (f64::from(*value) + alpha * (component * coefficient).clamp(-4.0, 4.0)) as f32;
    }
}

fn metric_gradient(
    metric: DistanceMetric,
    left: &[f32],
    right: &[f32],
    gradient: &mut [f64],
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => {
            let distance = euclidean_distance_wide(left, right);
            for (value, (&left, &right)) in gradient.iter_mut().zip(left.iter().zip(right)) {
                *value = f64::from(left - right) / (distance + 1e-6);
            }
            distance
        }
        DistanceMetric::Manhattan => {
            let mut distance = 0.0;
            for (value, (&left, &right)) in gradient.iter_mut().zip(left.iter().zip(right)) {
                let diff = left - right;
                distance += f64::from(diff.abs());
                *value = signed_step(diff);
            }
            distance
        }
        DistanceMetric::Chebyshev => {
            let mut axis = 0;
            let mut largest = 0.0_f32;
            for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
                let diff = left - right;
                if diff.abs() > largest.abs() {
                    axis = index;
                    largest = diff;
                }
            }
            gradient.fill(0.0);
            gradient[axis] = signed_step(largest);
            f64::from(largest.abs())
        }
        DistanceMetric::Cosine | DistanceMetric::Correlation => {
            angular_gradient(left, right, gradient, metric == DistanceMetric::Correlation)
        }
        DistanceMetric::Canberra => {
            let mut distance = 0.0;
            for ((value, &left), &right) in gradient.iter_mut().zip(left).zip(right) {
                let denominator = left.abs() + right.abs();
                let difference = left - right;
                *value = if denominator == 0.0 {
                    0.0
                } else {
                    distance += f64::from(difference.abs() / denominator);
                    f64::from(
                        signed_step(difference) as f32 / denominator
                            - signed_step(left) as f32 * difference.abs()
                                / (denominator * denominator),
                    )
                };
            }
            distance
        }
        DistanceMetric::BrayCurtis => {
            let (numerator, denominator) =
                bray_curtis_sums(left.iter().copied().zip(right.iter().copied()));
            if denominator == 0.0 {
                gradient.fill(0.0);
                return 0.0;
            }
            let distance = numerator / denominator;
            for ((value, &left), &right) in gradient.iter_mut().zip(left).zip(right) {
                *value = (signed_step(left - right) - distance) / denominator;
            }
            distance
        }
        _ => unreachable!("boolean metrics have no inverse gradient"),
    }
}

fn signed_step(diff: f32) -> f64 {
    if diff != 0.0 {
        f64::from(diff.signum())
    } else {
        0.0
    }
}

fn angular_gradient(left: &[f32], right: &[f32], gradient: &mut [f64], centered: bool) -> f64 {
    let ((dot, norm_x, norm_y), (left_mean, right_mean)) = if centered {
        correlation_products(left, right)
    } else {
        (cosine_products(left, right), (0.0, 0.0))
    };
    if norm_x == 0.0 || norm_y == 0.0 {
        gradient.fill(0.0);
        return if norm_x == norm_y { 0.0 } else { 1.0 };
    }
    let norm = (norm_x * norm_y).sqrt();
    for (value, (&x, &y)) in gradient.iter_mut().zip(left.iter().zip(right)) {
        let x = f64::from(x) - left_mean;
        let y = f64::from(y) - right_mean;
        *value = f64::from(((dot / norm_x * x - y) / norm) as f32);
    }
    1.0 - dot / norm
}
