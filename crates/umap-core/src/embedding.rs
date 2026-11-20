//! Dense end-to-end graph and embedding fitting.
//!
//! Both seeded and injected negative sampling use the same dense graph,
//! initialization, and optimization pipeline.

use std::fmt;

use crate::fit::{DenseGraphFitError, fit_dense_graph_with_options};
use crate::fuzzy_graph::{CsrGraph, GraphOptions};
use crate::initialization::{
    InitializationError, normalize_columns, random_initialization_with_rng,
};
use crate::neighbors::DistanceMetric;
use crate::optimize::{OptimizeError, OptimizeMode, optimize_csr_with_mode};
use crate::rng::{Mt19937, NegativeSamplerError, SerialNegativeSampler};

/// Initialization choices for a dense embedding fit.
///
/// Explicit coordinates are interpreted as a row-major `n_rows` by
/// `n_components` matrix.  They are cloned and column-normalized before any
/// optimizer call, so the caller's values are never modified.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseEmbeddingInit {
    /// Generate a seeded row-major layout and normalize each component.
    Random {
        /// Seed used by the existing deterministic initializer.
        seed: u32,
    },
    /// Use these row-major coordinates, after column normalization.
    Explicit {
        /// Caller-supplied row-major coordinates.
        values: Vec<f32>,
    },
}

impl DenseEmbeddingInit {
    /// Construct a seeded random initialization.
    #[must_use]
    pub const fn random(seed: u32) -> Self {
        Self::Random { seed }
    }

    /// Construct an explicit initialization by copying row-major values.
    #[must_use]
    pub fn explicit(values: &[f32]) -> Self {
        Self::Explicit {
            values: values.to_vec(),
        }
    }
}

/// Narrow configuration for dense graph and embedding fitting.
///
/// The curve parameters `a` and `b` are intentionally explicit.  This API
/// does not infer them from a UMAP spread/minimum-distance policy.  The dense
/// input is interpreted using `n_neighbors`, `metric`, and the caller's
/// `n_rows`/`n_features` arguments to [`fit_dense_embedding`].
/// The first type parameter controls the precision of the curve parameters
/// and the initial learning rate. The second controls repulsion strength.
/// Use `DenseEmbeddingConfig<f64, f64>` to supply those parameters in double precision.
/// Both type parameters default to `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEmbeddingConfig<T = f32, R = f32> {
    /// Number of neighbors retained for each input row.
    pub n_neighbors: usize,
    /// Distance metric used by exhaustive dense neighbor search.
    pub metric: DistanceMetric,
    /// Number of output coordinates per row.
    pub n_components: usize,
    /// Positive curve coefficient used by the layout objective.
    pub a: T,
    /// Positive curve exponent used by the layout objective.
    pub b: T,
    /// Initial optimizer learning rate.
    pub learning_rate: T,
    /// Number of optimizer epochs.
    pub n_epochs: usize,
    /// Non-negative scale for negative-example updates.
    pub repulsion_strength: R,
    /// Number of negative samples per positive sample.
    pub negative_sample_rate: f64,
    /// Seeded or explicit initial coordinates.
    pub init: DenseEmbeddingInit,
}

impl<T: Copy, R: Copy> DenseEmbeddingConfig<T, R> {
    fn layout_config(&self) -> EmbeddingLayoutConfig<T, R> {
        EmbeddingLayoutConfig {
            n_components: self.n_components,
            a: self.a,
            b: self.b,
            learning_rate: self.learning_rate,
            n_epochs: self.n_epochs,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
        }
    }
}

/// Layout settings for a fit with an existing fuzzy graph.
///
/// `T` controls the precision of the curve parameters and learning rate. `R`
/// controls repulsion strength, as in [`DenseEmbeddingConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingLayoutConfig<T = f32, R = f32> {
    /// Number of output coordinates per row.
    pub n_components: usize,
    /// Positive curve coefficient used by the layout objective.
    pub a: T,
    /// Positive curve exponent used by the layout objective.
    pub b: T,
    /// Initial optimizer learning rate.
    pub learning_rate: T,
    /// Number of optimizer epochs.
    pub n_epochs: usize,
    /// Non-negative scale for negative-example updates.
    pub repulsion_strength: R,
    /// Number of negative samples per positive sample.
    pub negative_sample_rate: f64,
}

/// Output of a dense embedding fit.
///
/// `coordinates` is a contiguous row-major `shape.0` by `shape.1` matrix.
/// The fitted graph and coordinates are owned independently of the caller's
/// input buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEmbedding {
    /// Fuzzy graph used for optimization.
    pub graph: CsrGraph,
    /// Row-major output coordinates.
    pub coordinates: Vec<f32>,
    /// `(n_rows, n_components)` shape of [`Self::coordinates`].
    pub shape: (usize, usize),
}

/// Failures returned by the dense embedding fit functions.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseEmbeddingFitError {
    /// Dense graph construction rejected the input matrix or neighbor count.
    Graph(DenseGraphFitError),
    /// Initialization or explicit-coordinate normalization failed.
    Initialization(InitializationError),
    /// The layout optimizer rejected the graph, settings, sampler result, or
    /// an arithmetic update.
    Optimize(OptimizeError),
}

impl fmt::Display for DenseEmbeddingFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "dense embedding graph fit failed: {error}"),
            Self::Initialization(error) => {
                write!(formatter, "dense embedding initialization failed: {error}")
            }
            Self::Optimize(error) => {
                write!(formatter, "dense embedding optimization failed: {error}")
            }
        }
    }
}

impl std::error::Error for DenseEmbeddingFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Initialization(error) => Some(error),
            Self::Optimize(error) => Some(error),
        }
    }
}

impl From<DenseGraphFitError> for DenseEmbeddingFitError {
    fn from(error: DenseGraphFitError) -> Self {
        Self::Graph(error)
    }
}

impl From<InitializationError> for DenseEmbeddingFitError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

impl From<OptimizeError> for DenseEmbeddingFitError {
    fn from(error: OptimizeError) -> Self {
        Self::Optimize(error)
    }
}

/// Fit a fuzzy graph and row-major dense embedding.
///
/// `data` is a finite row-major matrix with shape `n_rows` by `n_features`.
/// The graph is built by [`fit_dense_graph`]. The fit generates or copies the
/// initial coordinates into a private buffer, normalizes the columns, and
/// passes the result to [`crate::optimize::optimize_csr`]. At zero epochs, the
/// fit returns the normalized coordinates after validation. Each scheduled
/// negative sample passes its source row to the sampler, which returns a
/// candidate vertex.
///
/// If graph construction, initialization, or optimizer validation fails before
/// an update, no caller-owned buffer is changed.  The returned coordinates are
/// always row-major with `shape == (n_rows, config.n_components)`.
pub fn fit_dense_embedding<F>(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    negative_sampler: F,
) -> Result<DenseEmbedding, DenseEmbeddingFitError>
where
    F: FnMut(usize) -> usize,
{
    let (graph, coordinates, _) = prepare_dense_embedding(data, n_rows, n_features, config, None)?;
    finish_embedding(
        graph,
        coordinates,
        &config.layout_config(),
        negative_sampler,
        OptimizeMode::Injected,
    )
}

/// Fit a dense embedding using the serial seeded negative sampler.
///
/// For random initialization, the seed in [`DenseEmbeddingInit::Random`]
/// initializes an RNG that draws the layout and then seeds the sampler.
/// For explicit initialization, `explicit_seed` initializes the RNG and the
/// sampler is seeded after the supplied coordinates have been normalized.
/// Each row's sampling stream persists across all edges and epochs. The
/// compatibility schedule rounds normalized weights and expected sample
/// counts to `f32` before computing the intervals in `f64`. The compatibility
/// distance reduction uses the same component grouping for attraction and
/// repulsion.
pub fn fit_dense_embedding_seeded(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    explicit_seed: u32,
) -> Result<DenseEmbedding, DenseEmbeddingFitError> {
    fit_dense_embedding_seeded_with_graph_options(
        data,
        n_rows,
        n_features,
        config,
        explicit_seed,
        None,
    )
}

/// Fit a serial seeded embedding with optional fuzzy graph settings.
///
/// With explicit settings, retain weak memberships as stored zeros using
/// the public compatibility fit's epoch-dependent threshold.
pub fn fit_dense_embedding_seeded_with_graph_options(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    explicit_seed: u32,
    graph_options: Option<GraphOptions>,
) -> Result<DenseEmbedding, DenseEmbeddingFitError> {
    let (mut graph, coordinates, caller_rng) =
        prepare_dense_embedding(data, n_rows, n_features, config, graph_options)?;
    if graph_options.is_some() {
        zero_weak_memberships(&mut graph, config.n_epochs);
    }
    let mut caller_rng = caller_rng.unwrap_or_else(|| Mt19937::new(explicit_seed));
    finish_seeded_embedding(graph, coordinates, &config.layout_config(), &mut caller_rng)
}

pub(crate) fn zero_weak_memberships(graph: &mut CsrGraph, n_epochs: usize) {
    let sampling_epochs = match n_epochs {
        epochs if epochs > 10 => epochs,
        _ if graph.n_rows <= 10_000 => 500,
        _ => 200,
    };
    let maximum = graph.data.iter().copied().fold(0.0_f32, f32::max);
    let cutoff = maximum / sampling_epochs as f32;
    for weight in &mut graph.data {
        if *weight < cutoff {
            *weight = 0.0;
        }
    }
}

/// Fit an existing fuzzy graph from raw, row-major explicit coordinates.
///
/// Supply `graph.n_rows * config.n_components` coordinates and the caller RNG
/// positioned immediately after any draws used to produce them. This function
/// copies and normalizes the layout, seeds the serial negative sampler from
/// the current stream position (including at zero epochs), and optimizes the
/// layout with the same compatibility schedule as [`fit_dense_embedding_seeded`].
/// The returned embedding owns the supplied graph and final coordinates.
/// On success the caller RNG advances past sampler seeding. On error it keeps
/// its original state.
pub fn fit_graph_embedding_with_rng(
    graph: CsrGraph,
    initial_coordinates: &[f32],
    config: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    caller_rng: &mut Mt19937,
) -> Result<DenseEmbedding, DenseEmbeddingFitError> {
    let mut coordinates = initial_coordinates.to_vec();
    normalize_columns(&mut coordinates, graph.n_rows, config.n_components)?;

    let mut next_rng = caller_rng.clone();
    let fit = finish_seeded_embedding(graph, coordinates, config, &mut next_rng)?;
    *caller_rng = next_rng;
    Ok(fit)
}

fn finish_seeded_embedding(
    graph: CsrGraph,
    coordinates: Vec<f32>,
    config: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    caller_rng: &mut Mt19937,
) -> Result<DenseEmbedding, DenseEmbeddingFitError> {
    let mut sampler =
        SerialNegativeSampler::new(caller_rng, &coordinates, graph.n_rows, config.n_components)
            .map_err(|error| match error {
                NegativeSamplerError::InvalidShape {
                    n_rows,
                    n_components,
                    values,
                } => InitializationError::InvalidShape {
                    n_rows,
                    n_components,
                    values,
                },
                NegativeSamplerError::NonFiniteHeadCoordinate { head, value } => {
                    InitializationError::NonFiniteValue {
                        index: head * config.n_components,
                        value,
                    }
                }
            })?;
    finish_embedding(
        graph,
        coordinates,
        config,
        |row| {
            sampler
                .sample(row)
                .expect("optimizer source row is in bounds")
        },
        OptimizeMode::SeededCompatibility,
    )
}

fn prepare_dense_embedding(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    graph_options: Option<GraphOptions>,
) -> Result<(CsrGraph, Vec<f32>, Option<Mt19937>), DenseEmbeddingFitError> {
    let graph = fit_dense_graph_with_options(
        data,
        n_rows,
        n_features,
        config.n_neighbors,
        config.metric,
        graph_options,
    )?;

    let (coordinates, caller_rng) = match &config.init {
        DenseEmbeddingInit::Random { seed } => {
            let mut caller_rng = Mt19937::new(*seed);
            let coordinates =
                random_initialization_with_rng(&mut caller_rng, n_rows, config.n_components)?;
            (coordinates, Some(caller_rng))
        }
        DenseEmbeddingInit::Explicit { values } => {
            let mut coordinates = values.clone();
            normalize_columns(&mut coordinates, n_rows, config.n_components)?;
            (coordinates, None)
        }
    };

    Ok((graph, coordinates, caller_rng))
}

fn finish_embedding(
    graph: CsrGraph,
    mut coordinates: Vec<f32>,
    config: &EmbeddingLayoutConfig<impl Copy + Into<f64>, impl Copy + Into<f64>>,
    negative_sampler: impl FnMut(usize) -> usize,
    mode: OptimizeMode,
) -> Result<DenseEmbedding, DenseEmbeddingFitError> {
    optimize_csr_with_mode(
        &mut coordinates,
        config.n_components,
        &graph,
        config.a,
        config.b,
        config.learning_rate,
        config.repulsion_strength,
        config.negative_sample_rate,
        config.n_epochs,
        negative_sampler,
        mode,
    )?;

    Ok(DenseEmbedding {
        shape: (graph.n_rows, config.n_components),
        graph,
        coordinates,
    })
}
