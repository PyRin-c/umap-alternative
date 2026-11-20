//! Directed bipartite graphs and serial embeddings for held-out UMAP inputs.

use std::fmt;

use crate::fuzzy_graph::{GraphOptions, batch_scale_floor, membership, row_parameters_with_target};
use crate::layout::{
    attractive_update_fixed_tail, negative_update_fixed_tail, validate_parameters,
    validate_repulsion_strength,
};
use crate::neighbors::{DistanceMetric, KnnError, Neighbor, sort_closest_neighbors};
use crate::numeric::pairwise_sum_by;
use crate::optimize::OptimizeError;
use crate::rng::{Mt19937, SerialNegativeSampler};
use crate::sampling::EdgeSamplingCursor;

mod initialization;
pub use initialization::{
    NeighborhoodInitializationError, initialize_csr_transform, initialize_ranked_transform,
};

/// A rectangular coordinate (COO) graph of directed fuzzy memberships.
///
/// Each stored entry consists of the query row, training column, and membership
/// at the same position in the three vectors. Uncut entries are grouped by
/// ascending query row and preserve the search order within each row. Exact
/// dense search breaks distance ties by ascending training index. Cut or missing
/// entries retain stored zeros at placeholder coordinates. The shape is
/// `n_rows x n_cols`.
#[derive(Clone, Debug, PartialEq)]
pub struct CooGraph {
    /// Number of query rows.
    pub n_rows: usize,
    /// Number of training rows.
    pub n_cols: usize,
    /// Zero-based query index of each entry.
    pub row: Vec<usize>,
    /// Zero-based training index of each entry.
    pub col: Vec<usize>,
    /// Directed membership of each entry, including any rounded zeros.
    pub data: Vec<f32>,
}

/// Borrowed row-major distances from held-out rows to fitted rows.
#[derive(Clone, Copy, Debug)]
pub struct PrecomputedTransformInput<'a> {
    /// Rectangular matrix of finite, nonnegative distances.
    pub distances: &'a [f32],
    /// Number of columns in the distance matrix.
    pub n_training_rows: usize,
    /// Number of query rows in the distance matrix.
    pub n_query_rows: usize,
    /// Optional `n_query_rows * k` columns already ranked by increasing distance.
    ///
    /// Supplying a ranking preserves the caller's tie order and selection.
    /// Without one, equal distances are ordered by increasing column index.
    pub ranked_columns: Option<&'a [usize]>,
}

/// Validation failures for [`transform_dense_graph`].
#[derive(Clone, Debug, PartialEq)]
pub enum DenseGraphTransformError {
    /// The training matrix or requested neighbor count is invalid.
    Training(KnnError),
    /// The query matrix is invalid.
    Queries(KnnError),
    /// At least one feature is required by both matrices.
    NoFeatures,
    /// The selected distance overflowed the distance metric's precision.
    NonFiniteDistance {
        query: usize,
        training: usize,
        distance: f32,
    },
    /// A precomputed distance is negative.
    NegativeDistance { index: usize, distance: f32 },
    /// A supplied rank is out of bounds, duplicated, or out of distance order.
    InvalidRank { query: usize, column: usize },
}

impl fmt::Display for DenseGraphTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Training(error) => write!(formatter, "invalid training input: {error}"),
            Self::Queries(error) => write!(formatter, "invalid query input: {error}"),
            Self::NoFeatures => write!(formatter, "dense transform requires at least one feature"),
            Self::NonFiniteDistance {
                query,
                training,
                distance,
            } => write!(
                formatter,
                "distance from query row {query} to training row {training} is not finite: {distance}"
            ),
            Self::NegativeDistance { index, distance } => {
                write!(
                    formatter,
                    "precomputed distance at offset {index} is negative: {distance}"
                )
            }
            Self::InvalidRank { query, column } => {
                write!(
                    formatter,
                    "invalid ranked column {column} for query {query}"
                )
            }
        }
    }
}

impl std::error::Error for DenseGraphTransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Training(error) | Self::Queries(error) => Some(error),
            _ => None,
        }
    }
}

/// Validation failures for [`transform_ranked_neighbors_graph`].
#[derive(Clone, Debug, PartialEq)]
pub enum RankedQueryGraphError {
    /// The training row count must be positive and the scale count must fit the rank width.
    InvalidDimensions {
        rank_width: usize,
        scale_neighbors: usize,
        n_training_rows: usize,
    },
    /// Both flat arrays must have exactly `n_query_rows * rank_width` entries.
    InvalidShape {
        n_query_rows: usize,
        rank_width: usize,
        indices: usize,
        distances: usize,
    },
    /// A disconnection distance cannot be NaN.
    InvalidCutoff,
    /// A rank requires an in-range training column and nonnegative finite distance, or a missing pair.
    InvalidNeighbor {
        position: usize,
        index: i32,
        distance: f32,
    },
    /// Distances must be nondecreasing within a query row.
    UnsortedDistances { position: usize },
}

impl fmt::Display for RankedQueryGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidDimensions {
                rank_width,
                scale_neighbors,
                n_training_rows,
            } => {
                write!(
                    formatter,
                    "invalid rank width {rank_width}, scale count {scale_neighbors}, or training row count {n_training_rows}"
                )
            }
            Self::InvalidShape {
                n_query_rows,
                rank_width,
                indices,
                distances,
            } => write!(
                formatter,
                "ranked queries require arrays of shape {n_query_rows}x{rank_width}, got {indices} indices and {distances} distances"
            ),
            Self::InvalidCutoff => write!(formatter, "disconnection distance cannot be NaN"),
            Self::InvalidNeighbor {
                position,
                index,
                distance,
            } => write!(
                formatter,
                "invalid ranked query neighbor ({index}, {distance}) at position {position}"
            ),
            Self::UnsortedDistances { position } => {
                write!(
                    formatter,
                    "ranked query distances decrease at position {position}"
                )
            }
        }
    }
}

impl std::error::Error for RankedQueryGraphError {}

/// Validation failures for [`initialize_transform_embedding`].
#[derive(Clone, Debug, PartialEq)]
pub enum TransformInitializationError {
    /// The fitted coordinates do not have the shape implied by the graph.
    InvalidFittedShape {
        n_rows: usize,
        n_components: usize,
        values: usize,
    },
    /// A membership needed a fitted row that has no coordinates yet.
    MissingFittedRow { row: usize },
    /// The requested output length exceeds the addressable range.
    OutputShapeOverflow { n_rows: usize, n_components: usize },
    /// COO rows, columns, and weights must have equal lengths.
    InconsistentGraphLengths,
    /// An entry is outside the graph shape or its row is out of order.
    InvalidGraphIndex {
        position: usize,
        row: usize,
        column: usize,
    },
    /// A membership is outside `[0, 1]`.
    InvalidMembership { position: usize, weight: f32 },
    /// A query and training row share multiple nonzero entries.
    RepeatedEdge { row: usize, column: usize },
}

impl fmt::Display for TransformInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFittedShape {
                n_rows,
                n_components,
                values,
            } => write!(
                formatter,
                "fitted embedding requires a positive component count and shape {n_rows}x{n_components}, got {values} values"
            ),
            Self::MissingFittedRow { row } => {
                write!(
                    formatter,
                    "fitted embedding has no coordinates for row {row}"
                )
            }
            Self::OutputShapeOverflow {
                n_rows,
                n_components,
            } => write!(
                formatter,
                "transform embedding shape {n_rows}x{n_components} exceeds the addressable range"
            ),
            Self::InconsistentGraphLengths => {
                write!(formatter, "COO row, column, and weight lengths differ")
            }
            Self::InvalidGraphIndex {
                position,
                row,
                column,
            } => write!(
                formatter,
                "COO entry {position} at ({row}, {column}) is out of bounds or out of row order"
            ),
            Self::InvalidMembership { position, weight } => write!(
                formatter,
                "COO membership at entry {position} is outside [0, 1]: {weight}"
            ),
            Self::RepeatedEdge { row, column } => {
                write!(
                    formatter,
                    "COO graph repeats the nonzero edge ({row}, {column})"
                )
            }
        }
    }
}

impl std::error::Error for TransformInitializationError {}

/// Settings for a serial compatibility transform against a fitted dense embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEmbeddingTransformConfig {
    /// Effective number of neighbors used to build query-to-training memberships.
    pub n_neighbors: usize,
    /// Distance metric over feature rows. Layout distances are Euclidean.
    pub metric: DistanceMetric,
    /// Number of coordinates in each fitted and transformed row.
    pub n_components: usize,
    /// Curve coefficient used for the fit.
    pub a: f64,
    /// Curve exponent used for the fit.
    pub b: f64,
    /// Fit learning rate. The transform starts at one quarter of this value.
    pub learning_rate: f64,
    /// Fit epoch setting. An explicit count is divided by three, rounding down.
    /// The default uses 100 transform epochs for up to 10,000 queries and 30 above that.
    pub fit_epochs: Option<usize>,
    /// Non-negative scale for negative-example updates.
    pub repulsion_strength: f64,
    /// Negative samples per positive sample.
    pub negative_sample_rate: f64,
    /// Seed for each transform call, independent of the fitting RNG.
    pub transform_seed: u32,
    /// Exclude neighbors at or above this positive feature distance.
    /// Using `None` keeps every selected neighbor.
    pub disconnection_distance: Option<f32>,
}

/// Failures returned by [`transform_dense_embedding`].
#[derive(Clone, Debug, PartialEq)]
pub enum DenseEmbeddingTransformError {
    /// Query-to-training graph construction failed.
    Graph(DenseGraphTransformError),
    /// Layout initialization failed.
    Initialization(TransformInitializationError),
    /// Layout parameters, timing, or arithmetic were invalid.
    Optimize(OptimizeError),
    /// The disconnection distance was NaN or nonpositive.
    InvalidDisconnectionDistance(f32),
}

impl fmt::Display for DenseEmbeddingTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "dense embedding transform graph: {error}"),
            Self::Initialization(error) => {
                write!(formatter, "dense transform initialization: {error}")
            }
            Self::Optimize(error) => write!(formatter, "dense transform optimization: {error}"),
            Self::InvalidDisconnectionDistance(value) => {
                write!(
                    formatter,
                    "disconnection distance must be positive, got {value}"
                )
            }
        }
    }
}

impl std::error::Error for DenseEmbeddingTransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Initialization(error) => Some(error),
            Self::Optimize(error) => Some(error),
            Self::InvalidDisconnectionDistance(_) => None,
        }
    }
}

/// Compute row-major held-out coordinates with the serial compatibility layout.
///
/// Supply finite training and query feature matrices with the same feature
/// count and the corresponding fitted coordinates. The exact neighbor search
/// uses training local connectivity one. Fitted coordinates serve as fixed
/// endpoints throughout optimization. A query without positive memberships
/// returns NaN coordinates. Fitted NaNs may propagate during initialization
/// and are ignored by layout updates involving those rows.
///
/// This function treats all supplied rows as held-out queries, including exact
/// matches to individual training rows. Estimators that return their existing
/// embedding for the full fitted input handle that special case before calling
/// this function.
///
/// # Errors
///
/// Returns an error for invalid input shapes, graph or layout settings, or
/// overflowing arithmetic in a scheduled update.
pub fn transform_dense_embedding(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    transform_dense_embedding_with_options(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        fitted_embedding,
        config,
        GraphOptions::default(),
    )
}

/// Transform held-out points with the specified training local connectivity.
///
/// The directed query graph uses the local connectivity from `options`.
/// Other arguments follow [`transform_dense_embedding`].
#[allow(clippy::too_many_arguments)]
pub fn transform_dense_embedding_with_options(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
    options: GraphOptions,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    use DenseEmbeddingTransformError as Error;

    let cutoff = embedding_cutoff(
        config,
        config
            .metric
            .default_disconnection_distance()
            .unwrap_or(f32::INFINITY),
    )?;
    let graph = transform_dense_graph_with_cutoff(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        config.n_neighbors,
        config.metric,
        cutoff,
        options,
    )
    .map_err(Error::Graph)?;
    transform_embedding_from_graph(&graph, fitted_embedding, config)
}

/// Transform held-out rows from precomputed query-to-training distances.
///
/// The matrix and optional rank order follow [`transform_precomputed_graph`].
/// The optimizer reuses the supplied fitted positions and transform settings.
/// The feature metric field is unused because the input contains distances.
pub fn transform_precomputed_embedding(
    input: PrecomputedTransformInput<'_>,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
    options: GraphOptions,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    let graph = transform_precomputed_graph(
        input,
        config.n_neighbors,
        embedding_cutoff(config, f32::INFINITY)?,
        options,
    )
    .map_err(DenseEmbeddingTransformError::Graph)?;
    transform_embedding_from_graph(&graph, fitted_embedding, config)
}

fn embedding_cutoff(
    config: &DenseEmbeddingTransformConfig,
    default: f32,
) -> Result<f32, DenseEmbeddingTransformError> {
    let cutoff = config.disconnection_distance.unwrap_or(default);
    if cutoff.is_nan() || cutoff <= 0.0 {
        return Err(DenseEmbeddingTransformError::InvalidDisconnectionDistance(
            cutoff,
        ));
    }
    Ok(cutoff)
}

/// Initialize and optimize held-out coordinates from a directed query graph.
///
/// The graph already contains its selected memberships and disconnection
/// cutoff. Repeated edges are summed for initialization, while the optimizer
/// visits the original COO entries in order.
pub fn transform_embedding_from_graph(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    transform_embedding_from_graph_impl(graph, fitted_embedding, config, false)
}

/// Transform against a graph rebuilt by an update that could not publish new
/// coordinates. Initialization can only read the retained fitted rows.
pub fn transform_embedding_from_partial_graph(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    transform_embedding_from_graph_impl(graph, fitted_embedding, config, true)
}

fn transform_embedding_from_graph_impl(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    config: &DenseEmbeddingTransformConfig,
    allow_partial: bool,
) -> Result<Vec<f32>, DenseEmbeddingTransformError> {
    use DenseEmbeddingTransformError as Error;
    let mut embedding = initialize_transform_embedding_impl(
        graph,
        fitted_embedding,
        config.n_components,
        true,
        allow_partial,
    )
    .map_err(Error::Initialization)?;
    optimize_transform(&mut embedding, fitted_embedding, graph, config).map_err(Error::Optimize)?;
    Ok(embedding)
}

fn optimize_transform(
    embedding: &mut [f32],
    fitted_embedding: &[f32],
    graph: &CooGraph,
    config: &DenseEmbeddingTransformConfig,
) -> Result<(), OptimizeError> {
    let n_components = config.n_components;
    let (a, b, repulsion) = (config.a, config.b, config.repulsion_strength);
    validate_parameters(a, b, config.learning_rate)?;
    validate_repulsion_strength(repulsion)?;
    EdgeSamplingCursor::from_positive_interval(1.0, config.negative_sample_rate)?;
    let n_epochs = config.fit_epochs.map_or_else(
        || if graph.n_rows <= 10_000 { 100 } else { 30 },
        |fit_epochs| fit_epochs / 3,
    );
    if n_epochs < 2 {
        return Ok(());
    }

    let maximum_weight = graph.data.iter().copied().fold(0.0_f32, f32::max);
    if maximum_weight == 0.0 {
        return Ok(());
    }
    let minimum_weight = maximum_weight / n_epochs as f32;
    let mut edges = Vec::new();
    for ((&head, &tail), &weight) in graph.row.iter().zip(&graph.col).zip(&graph.data) {
        if weight > 0.0 && weight >= minimum_weight {
            // Round the expected positive count in float32 before deriving
            // the float64 interval. Division by weight alone can miss a due epoch.
            let positive_count = n_epochs as f32 * (weight / maximum_weight);
            let cursor = EdgeSamplingCursor::from_positive_interval(
                n_epochs as f64 / f64::from(positive_count),
                config.negative_sample_rate,
            )?;
            edges.push((head, tail, cursor));
        }
    }

    let fitted_rows: Vec<_> = fitted_embedding
        .chunks_exact(n_components)
        .map(|row| (!row.iter().any(|value| value.is_nan())).then_some(row))
        .collect();
    let mut rng = Mt19937::new(config.transform_seed);
    let mut sampler = SerialNegativeSampler::from_head_coordinates(
        embedding.chunks_exact(n_components).map(|row| row[0]),
        || rng.next_u32(),
    );

    let initial_alpha = config.learning_rate / 4.0;
    let mut alpha = initial_alpha;
    for epoch in 0..n_epochs {
        let epoch = epoch as f64;
        for (head, tail, cursor) in &mut edges {
            if !cursor.positive_due(epoch)? {
                continue;
            }
            let start = *head * n_components;
            let current = &mut embedding[start..start + n_components];
            if current.iter().any(|value| value.is_nan()) {
                continue;
            }
            if let Some(Some(other)) = fitted_rows.get(*tail) {
                attractive_update_fixed_tail(current, other, a, b, alpha)?;
            }
            // An incomplete fit still schedules samples over the full graph.
            for _ in 0..cursor.take_negative_samples(epoch)? {
                let vertex = sampler
                    .sample_from(*head, graph.n_cols)
                    .expect("graph head has a negative sampling stream");
                if let Some(Some(other)) = fitted_rows.get(vertex) {
                    negative_update_fixed_tail(current, other, a, b, repulsion, alpha)?;
                }
            }
        }
        alpha = initial_alpha * (1.0 - epoch / n_epochs as f64);
    }
    Ok(())
}

/// Initialize a row-major query embedding from the held-out graph and fitted coordinates.
///
/// The fitted matrix has shape `graph.n_cols x n_components` with at least
/// one component. The output shape is `graph.n_rows x n_components`.
/// The nonzero COO entries must be grouped by ascending query row, with
/// unique training columns within each query row. Stored zeros are skipped.
/// A query with no positive memberships starts with NaN coordinates.
/// When a membership is exactly one, the initializer
/// copies the fitted row at the smallest training column with that membership.
/// Otherwise it L1-normalizes the weights and accumulates weighted fitted
/// coordinates in descending training-column order, with separate float32
/// rounding at each step. Fitted NaNs propagate from contributing rows.
///
/// # Errors
///
/// Returns an error for an invalid embedding shape or malformed COO entries.
pub fn initialize_transform_embedding(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    n_components: usize,
) -> Result<Vec<f32>, TransformInitializationError> {
    initialize_transform_embedding_impl(graph, fitted_embedding, n_components, false, false)
}

fn initialize_transform_embedding_impl(
    graph: &CooGraph,
    fitted_embedding: &[f32],
    n_components: usize,
    coalesce_duplicates: bool,
    allow_partial: bool,
) -> Result<Vec<f32>, TransformInitializationError> {
    use TransformInitializationError as Error;

    let fits_graph = graph.n_cols.checked_mul(n_components) == Some(fitted_embedding.len());
    let retained_prefix = allow_partial
        && fitted_embedding
            .len()
            .checked_div(n_components)
            .is_some_and(|rows| {
                rows > 0
                    && rows < graph.n_cols
                    && fitted_embedding.len().is_multiple_of(n_components)
            });
    if n_components == 0 || (!fits_graph && !retained_prefix) {
        return Err(Error::InvalidFittedShape {
            n_rows: graph.n_cols,
            n_components,
            values: fitted_embedding.len(),
        });
    }
    if graph.row.len() != graph.col.len() || graph.row.len() != graph.data.len() {
        return Err(Error::InconsistentGraphLengths);
    }
    let output_len = graph
        .n_rows
        .checked_mul(n_components)
        .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or(Error::OutputShapeOverflow {
            n_rows: graph.n_rows,
            n_components,
        })?;
    let mut embedding = vec![f32::NAN; output_len];
    let mut entries = Vec::new();
    let mut previous_row = None;
    let mut positions = (0..graph.data.len())
        .filter(|&position| graph.data[position] != 0.0)
        .peekable();
    let fitted_row = |row| {
        fitted_embedding
            .chunks_exact(n_components)
            .nth(row)
            .ok_or(Error::MissingFittedRow { row })
    };

    while let Some(&start) = positions.peek() {
        let row = graph.row[start];
        if row >= graph.n_rows || previous_row.is_some_and(|previous| row <= previous) {
            return Err(Error::InvalidGraphIndex {
                position: start,
                row,
                column: graph.col[start],
            });
        }
        previous_row = Some(row);
        entries.clear();
        while let Some(position) = positions.next_if(|&position| graph.row[position] == row) {
            let column = graph.col[position];
            let weight = graph.data[position];
            if column >= graph.n_cols {
                return Err(Error::InvalidGraphIndex {
                    position,
                    row,
                    column,
                });
            }
            if !(0.0..=1.0).contains(&weight) {
                return Err(Error::InvalidMembership { position, weight });
            }
            entries.push((column, weight));
        }
        if coalesce_duplicates {
            // Match COO-to-CSR summation order for repeated columns.
            entries.sort_by_key(|&(column, _)| column);
            entries.dedup_by(|(column, weight), (previous, sum)| {
                if column == previous {
                    *sum += *weight;
                    true
                } else {
                    false
                }
            });
        } else {
            entries.sort_unstable_by_key(|&(column, _)| column);
            if let Some(pair) = entries.windows(2).find(|pair| pair[0].0 == pair[1].0) {
                return Err(Error::RepeatedEdge {
                    row,
                    column: pair[0].0,
                });
            }
        }
        let row_start = row * n_components;
        let coordinates = &mut embedding[row_start..row_start + n_components];
        if let Some(&(column, _)) = entries.iter().find(|&&(_, weight)| weight == 1.0) {
            coordinates.copy_from_slice(fitted_row(column)?);
            continue;
        }

        // The sparse row reduction adds the first membership after summing the tail.
        let scale = (pairwise_sum_by(&entries[1..], |&(_, weight)| weight) + entries[0].1).recip();
        coordinates.fill(0.0);
        for &(column, weight) in entries.iter().rev() {
            let fitted = fitted_row(column)?;
            let normalized = weight * scale;
            for (coordinate, &fitted) in coordinates.iter_mut().zip(fitted) {
                *coordinate += normalized * fitted;
            }
        }
    }
    Ok(embedding)
}

/// Construct the fuzzy graph from held-out query rows to fitted training rows.
///
/// Supply two finite row-major matrices with the same positive feature count.
/// `k` is the effective neighbor count in `1..=n_training_rows`. An empty
/// query matrix produces an empty graph with `n_training_rows` columns.
///
/// The transform for training local connectivity one uses zero distance offset.
/// It calibrates each local scale over neighbor ranks `1..k` (excluding the
/// closest candidate) toward a membership sum of `log2(k)`. The minimum
/// scale is 0.001 times the mean selected distance across all query rows.
/// Every selected candidate contributes a membership. A query index and a
/// training index with the same numeric value identify independent vertices.
/// The result has exactly `k` directed entries per query row.
///
/// # Errors
///
/// Returns an error for invalid matrix shapes, a zero feature count, invalid
/// `k`, non-finite input values, or non-finite selected distances.
pub fn transform_dense_graph(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
) -> Result<CooGraph, DenseGraphTransformError> {
    transform_dense_graph_with_options(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        k,
        metric,
        GraphOptions::default(),
    )
}

/// Build a held-out graph using `max(0, local_connectivity - 1)` as query connectivity.
///
/// The query graph is directed, so fuzzy-set mixing does not affect it.
#[allow(clippy::too_many_arguments)]
pub fn transform_dense_graph_with_options(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    options: GraphOptions,
) -> Result<CooGraph, DenseGraphTransformError> {
    transform_dense_graph_with_cutoff(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        k,
        metric,
        metric
            .default_disconnection_distance()
            .unwrap_or(f32::INFINITY),
        options,
    )
}

/// Build a held-out graph with an inclusive distance cutoff.
/// Cut entries retain a zero weight at coordinate `(0, 0)` in their original
/// storage positions. Query scales still use every selected distance.
#[allow(clippy::too_many_arguments)]
pub fn transform_dense_graph_with_cutoff(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
    options: GraphOptions,
) -> Result<CooGraph, DenseGraphTransformError> {
    transform_dense_graph_with_selector(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        k,
        metric,
        cutoff,
        options,
        |_, candidates| {
            sort_closest_neighbors(candidates, k);
            Ok(())
        },
    )
}

/// Build a dense query graph, resolving selected distance ties with a caller's ranking.
///
/// The callback sees the full distance row in training-column order and the
/// requested count. It returns the first `k` distinct columns in nondecreasing
/// distance order. Rows without a tie within the first `k` ranks or at the
/// cutoff rank use native partial selection alone.
#[allow(clippy::too_many_arguments)]
pub fn transform_dense_graph_with_tie_ranking(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
    options: GraphOptions,
    mut rank_ties: impl FnMut(&[f32], usize) -> Vec<usize>,
) -> Result<CooGraph, DenseGraphTransformError> {
    let mut tie_ranking = QueryTieRanking::default();
    transform_dense_graph_with_selector(
        training,
        n_training_rows,
        queries,
        n_query_rows,
        n_features,
        k,
        metric,
        cutoff,
        options,
        |query, candidates| tie_ranking.select(query, candidates, k, false, &mut rank_ties),
    )
}

#[derive(Default)]
pub(crate) struct QueryTieRanking {
    distances: Vec<f32>,
    seen: Vec<bool>,
}

impl QueryTieRanking {
    pub(crate) fn select(
        &mut self,
        query: usize,
        candidates: &mut [Neighbor],
        k: usize,
        allow_missing: bool,
        rank_ties: impl FnOnce(&[f32], usize) -> Vec<usize>,
    ) -> Result<(), DenseGraphTransformError> {
        let selected = sort_closest_neighbors(candidates, k);
        let boundary = selected[k - 1].distance;
        let tied = selected
            .windows(2)
            .any(|pair| pair[0].distance == pair[1].distance)
            || (k < candidates.len() && boundary == candidates[k].distance);
        if !tied {
            return Ok(());
        }
        let strictly_closer =
            candidates[..k].partition_point(|neighbor| neighbor.distance < boundary);
        let n_training_rows = candidates.len();
        self.distances.resize(n_training_rows, 0.0);
        for candidate in candidates.iter() {
            self.distances[candidate.index] = candidate.distance;
        }
        let columns = rank_ties(&self.distances, k);
        if columns.len() != k {
            return Err(DenseGraphTransformError::InvalidRank {
                query,
                column: n_training_rows,
            });
        }
        self.seen.resize(n_training_rows, false);
        self.seen.fill(false);
        let mut previous = f32::NEG_INFINITY;
        for (rank, (candidate, column)) in candidates.iter_mut().zip(columns).enumerate() {
            if column >= n_training_rows
                || self.seen[column]
                || self.distances[column] < previous
                || self.distances[column] > boundary
                || (rank < strictly_closer && self.distances[column] == boundary)
            {
                return Err(DenseGraphTransformError::InvalidRank { query, column });
            }
            let distance = self.distances[column];
            if !(distance.is_finite() || (allow_missing && distance == f32::INFINITY)) {
                return Err(DenseGraphTransformError::NonFiniteDistance {
                    query,
                    training: column,
                    distance,
                });
            }
            self.seen[column] = true;
            previous = distance;
            *candidate = Neighbor {
                index: column,
                distance,
            };
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn transform_dense_graph_with_selector(
    training: &[f32],
    n_training_rows: usize,
    queries: &[f32],
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
    options: GraphOptions,
    mut select: impl FnMut(usize, &mut [Neighbor]) -> Result<(), DenseGraphTransformError>,
) -> Result<CooGraph, DenseGraphTransformError> {
    if n_features == 0 {
        return Err(DenseGraphTransformError::NoFeatures);
    }
    validate_matrix(training, n_training_rows, n_features)
        .map_err(DenseGraphTransformError::Training)?;
    validate_matrix(queries, n_query_rows, n_features)
        .map_err(DenseGraphTransformError::Queries)?;
    build_transform_graph(
        n_training_rows,
        n_query_rows,
        k,
        cutoff,
        options,
        |row, candidates| {
            let query = &queries[row * n_features..(row + 1) * n_features];
            candidates.extend(training.chunks_exact(n_features).enumerate().map(
                |(index, candidate)| Neighbor {
                    index,
                    distance: metric.distance(query, candidate),
                },
            ));
            select(row, candidates)
        },
    )
}

/// Build the directed held-out graph from rectangular precomputed distances.
///
/// A supplied ranking must contain exactly `k` distinct in-range columns per
/// query, ordered by nondecreasing distance. The same local-connectivity and
/// inclusive cutoff rules as [`transform_dense_graph_with_cutoff`] apply.
pub fn transform_precomputed_graph(
    input: PrecomputedTransformInput<'_>,
    k: usize,
    cutoff: f32,
    options: GraphOptions,
) -> Result<CooGraph, DenseGraphTransformError> {
    let PrecomputedTransformInput {
        distances,
        n_training_rows,
        n_query_rows,
        ranked_columns,
    } = input;
    validate_matrix(distances, n_query_rows, n_training_rows)
        .map_err(DenseGraphTransformError::Queries)?;
    if let Some((index, &distance)) = distances
        .iter()
        .enumerate()
        .find(|(_, value)| **value < 0.0)
    {
        return Err(DenseGraphTransformError::NegativeDistance { index, distance });
    }
    if let Some(columns) = ranked_columns {
        validate_precomputed_ranks(input, columns, k)?;
    }
    build_transform_graph(
        n_training_rows,
        n_query_rows,
        k,
        cutoff,
        options,
        |row, candidates| {
            let distances = &distances[row * n_training_rows..(row + 1) * n_training_rows];
            if let Some(columns) = ranked_columns {
                candidates.extend(
                    columns[row * k..(row + 1) * k]
                        .iter()
                        .map(|&index| Neighbor {
                            index,
                            distance: distances[index],
                        }),
                );
            } else {
                candidates.extend(distances.iter().enumerate().map(|(index, &distance)| {
                    Neighbor {
                        index,
                        distance: if distance == 0.0 { 0.0 } else { distance },
                    }
                }));
                sort_closest_neighbors(candidates, k);
            }
            Ok(())
        },
    )
}

fn validate_precomputed_ranks(
    input: PrecomputedTransformInput<'_>,
    columns: &[usize],
    k: usize,
) -> Result<(), DenseGraphTransformError> {
    let PrecomputedTransformInput {
        distances,
        n_training_rows,
        n_query_rows,
        ..
    } = input;
    if n_query_rows.checked_mul(k) != Some(columns.len()) {
        return Err(DenseGraphTransformError::Queries(KnnError::InvalidShape {
            n_rows: n_query_rows,
            n_features: k,
            values: columns.len(),
        }));
    }
    if k == 0 || k > n_training_rows || n_query_rows == 0 {
        // Neighbor-count validation also covers the no-ranking path.
        return Ok(());
    }
    let mut seen = vec![usize::MAX; n_training_rows];
    for (query, ranked_row) in columns.chunks_exact(k).enumerate() {
        let mut previous = 0.0;
        for &column in ranked_row {
            let distance = if column < n_training_rows {
                distances[query * n_training_rows + column]
            } else {
                return Err(DenseGraphTransformError::InvalidRank { query, column });
            };
            if seen[column] == query || distance < previous {
                return Err(DenseGraphTransformError::InvalidRank { query, column });
            }
            seen[column] = query;
            previous = distance;
        }
    }
    Ok(())
}

fn build_transform_graph(
    n_training_rows: usize,
    n_query_rows: usize,
    k: usize,
    cutoff: f32,
    options: GraphOptions,
    mut fill_candidates: impl FnMut(usize, &mut Vec<Neighbor>) -> Result<(), DenseGraphTransformError>,
) -> Result<CooGraph, DenseGraphTransformError> {
    if !(1..=n_training_rows).contains(&k) {
        return Err(DenseGraphTransformError::Training(KnnError::InvalidK {
            k,
            n_rows: n_training_rows,
        }));
    }

    let mut graph = CooGraph {
        n_rows: n_query_rows,
        n_cols: n_training_rows,
        row: Vec::new(),
        col: Vec::new(),
        data: Vec::new(),
    };
    if n_query_rows == 0 {
        return Ok(graph);
    }

    let mut candidates = Vec::new();
    for query_index in 0..n_query_rows {
        candidates.clear();
        fill_candidates(query_index, &mut candidates)?;
        for neighbor in &candidates[..k] {
            if !neighbor.distance.is_finite() {
                return Err(DenseGraphTransformError::NonFiniteDistance {
                    query: query_index,
                    training: neighbor.index,
                    distance: neighbor.distance,
                });
            }
            graph.row.push(query_index);
            graph.col.push(neighbor.index);
            graph.data.push(neighbor.distance);
        }
    }
    apply_query_memberships(&mut graph, k, k, cutoff, options, false);
    Ok(graph)
}

/// Construct a held-out COO graph from an external search's ranked results.
///
/// Supply flat row-major `n_query_rows x rank_width` arrays. Each rank contains an
/// in-range training index and a finite nonnegative distance, or `(-1, +inf)`
/// for a missing neighbor. Distances are nondecreasing, and equal distances
/// retain the supplied order. The search owns candidate selection and tie
/// breaking. The query width includes missing ranks and can exceed
/// the number of training rows.
///
/// The scale solve skips rank zero and targets `log2(scale_neighbors)`. Pass
/// the effective fitted count in `1..=rank_width`, which can be smaller than
/// the width requested from the search. Finite distances at or above the
/// inclusive `cutoff` still contribute to scaling. Those ranks
/// and missing ranks are stored as zero at `(0, 0)`. Missing distances do
/// not enter scale floors. The query local connectivity is
/// `max(0, options.local_connectivity() - 1)`.
///
/// # Errors
///
/// Returns an error for invalid counts, shapes, cutoff, index-distance pairs,
/// or decreasing distances within a row.
#[allow(clippy::too_many_arguments)]
pub fn transform_ranked_neighbors_graph(
    indices: &[i32],
    distances: &[f32],
    n_query_rows: usize,
    n_training_rows: usize,
    rank_width: usize,
    scale_neighbors: usize,
    cutoff: f32,
    options: GraphOptions,
) -> Result<CooGraph, RankedQueryGraphError> {
    use RankedQueryGraphError as Error;

    if n_training_rows == 0 || !(1..=rank_width).contains(&scale_neighbors) {
        return Err(Error::InvalidDimensions {
            rank_width,
            scale_neighbors,
            n_training_rows,
        });
    }
    if n_query_rows.checked_mul(rank_width) != Some(indices.len())
        || indices.len() != distances.len()
    {
        return Err(Error::InvalidShape {
            n_query_rows,
            rank_width,
            indices: indices.len(),
            distances: distances.len(),
        });
    }
    if cutoff.is_nan() {
        return Err(Error::InvalidCutoff);
    }
    let mut graph = CooGraph {
        n_rows: n_query_rows,
        n_cols: n_training_rows,
        row: Vec::with_capacity(indices.len()),
        col: Vec::with_capacity(indices.len()),
        data: distances.to_vec(),
    };
    let mut has_missing = false;
    for (position, (&index, &distance)) in indices.iter().zip(distances).enumerate() {
        let column = if index == -1 && distance == f32::INFINITY {
            has_missing = true;
            0
        } else if index >= 0
            && (index as usize) < n_training_rows
            && distance.is_finite()
            && distance >= 0.0
        {
            index as usize
        } else {
            return Err(Error::InvalidNeighbor {
                position,
                index,
                distance,
            });
        };
        if position % rank_width != 0 && distance < distances[position - 1] {
            return Err(Error::UnsortedDistances { position });
        }
        graph.row.push(position / rank_width);
        graph.col.push(column);
    }
    if !distances.is_empty() {
        apply_query_memberships(
            &mut graph,
            rank_width,
            scale_neighbors,
            cutoff,
            options,
            has_missing,
        );
    }
    Ok(graph)
}

fn apply_query_memberships(
    graph: &mut CooGraph,
    rank_width: usize,
    scale_neighbors: usize,
    cutoff: f32,
    options: GraphOptions,
    has_missing: bool,
) {
    let target = (scale_neighbors as f64).log2();
    let connectivity = (options.local_connectivity() - 1.0).max(0.0);
    let finite_distances;
    let scale_distances = if has_missing {
        finite_distances = graph
            .data
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .collect::<Vec<_>>();
        finite_distances.as_slice()
    } else {
        &graph.data
    };
    let min_scale = if scale_distances.is_empty() {
        0.0
    } else if connectivity == 0.0 {
        minimum_transform_scale(scale_distances)
    } else {
        batch_scale_floor(scale_distances)
    };
    for (row, distances) in graph.data.chunks_exact_mut(rank_width).enumerate() {
        let ordered: &[f32] = if has_missing {
            &distances[..distances.partition_point(|d| d.is_finite())]
        } else {
            distances
        };
        let (rho, sigma) = if ordered.is_empty() {
            (0.0, 1.0)
        } else if connectivity == 0.0 {
            (0.0, transform_scale(ordered, target).max(min_scale))
        } else {
            row_parameters_with_target(ordered, connectivity, min_scale, scale_neighbors)
        };
        for (column, distance) in distances.iter_mut().enumerate() {
            *distance = if *distance >= cutoff {
                let position = row * rank_width + column;
                graph.row[position] = 0;
                graph.col[position] = 0;
                0.0
            } else {
                membership(*distance, rho, sigma)
            };
        }
    }
}

fn validate_matrix(data: &[f32], n_rows: usize, n_features: usize) -> Result<(), KnnError> {
    if n_rows.checked_mul(n_features) != Some(data.len()) {
        return Err(KnnError::InvalidShape {
            n_rows,
            n_features,
            values: data.len(),
        });
    }
    if let Some((index, &value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KnnError::NonFiniteValue { index, value });
    }
    Ok(())
}

fn minimum_transform_scale(distances: &[f32]) -> f32 {
    let sum = distances.iter().sum::<f32>();
    let mean = if sum.is_finite() {
        sum / distances.len() as f32
    } else {
        // A sum of individually finite distances can overflow even when their mean fits.
        (distances
            .iter()
            .map(|&distance| f64::from(distance))
            .sum::<f64>()
            / distances.len() as f64) as f32
    };
    (0.001 * f64::from(mean)) as f32
}

fn transform_scale(distances: &[f32], target: f64) -> f32 {
    let mut lower = 0.0_f64;
    let mut upper = f64::INFINITY;
    let mut sigma = 1.0_f64;
    for _ in 0..64 {
        let sum: f64 = distances[1..]
            .iter()
            .map(|&distance| (-f64::from(distance) / sigma).exp())
            .sum();
        if (sum - target).abs() < 1e-5 {
            break;
        }
        if sum > target {
            upper = sigma;
        } else {
            lower = sigma;
        }
        sigma = if upper.is_infinite() {
            sigma * 2.0
        } else {
            (lower + upper) * 0.5
        };
    }
    sigma as f32
}
