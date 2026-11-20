//! Dense end-to-end graph and embedding fitting.
//!
//! This module deliberately composes the existing dense graph, initialization,
//! and injected-sampler optimization primitives.  It does not select random
//! negative vertices: callers supply that policy through the sampler passed to
//! [`fit_dense_embedding`].

use std::fmt;

use crate::fit::{DenseGraphFitError, fit_dense_graph};
use crate::fuzzy_graph::CsrGraph;
use crate::initialization::{InitializationError, normalize_columns, random_initialization};
use crate::neighbors::DistanceMetric;
use crate::optimize::{OptimizeError, optimize_csr};

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
/// Use `DenseEmbeddingConfig<f64>` to supply the curve parameters and initial
/// learning rate at full attraction precision. The default `f32` type remains
/// available for callers with single-precision settings.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEmbeddingConfig<T = f32> {
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
    pub repulsion_strength: f32,
    /// Number of negative samples per positive sample.
    pub negative_sample_rate: f64,
    /// Seeded or explicit initial coordinates.
    pub init: DenseEmbeddingInit,
}

/// Output of [`fit_dense_embedding`].
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

/// Failures returned by [`fit_dense_embedding`].
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
/// passes the result to [`optimize_csr`]. At zero epochs, the fit returns the
/// normalized coordinates after validation. Each scheduled negative sample
/// passes its source row to the sampler, which returns a candidate vertex.
///
/// If graph construction, initialization, or optimizer validation fails before
/// an update, no caller-owned buffer is changed.  The returned coordinates are
/// always row-major with `shape == (n_rows, config.n_components)`.
pub fn fit_dense_embedding<F>(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingConfig<impl Copy + Into<f64>>,
    negative_sampler: F,
) -> Result<DenseEmbedding, DenseEmbeddingFitError>
where
    F: FnMut(usize) -> usize,
{
    let graph = fit_dense_graph(data, n_rows, n_features, config.n_neighbors, config.metric)?;

    let mut coordinates = match &config.init {
        DenseEmbeddingInit::Random { seed } => {
            random_initialization(*seed, n_rows, config.n_components)?
        }
        DenseEmbeddingInit::Explicit { values } => {
            let mut coordinates = values.clone();
            normalize_columns(&mut coordinates, n_rows, config.n_components)?;
            coordinates
        }
    };

    optimize_csr(
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
    )?;

    Ok(DenseEmbedding {
        graph,
        coordinates,
        shape: (n_rows, config.n_components),
    })
}
