use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyRuntimeWarning, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

use umap_core::aligned::{AlignedLayoutConfig, optimize_aligned_layout};
use umap_core::approximate_neighbors::DenseApproximateIndex;
use umap_core::density::{
    DensityError, OriginalRadiusInput, embedding_radius_log_inputs_euclidean,
    original_radius_log_inputs,
};
use umap_core::densmap::{DensmapOptions, fit_graph_with_checkpoints};
use umap_core::discrete::{count_distance, levenshtein, myers};
use umap_core::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingInit, EmbeddingLayoutConfig,
    fit_dense_embedding_seeded_with_graph_options, fit_graph_embedding_with_rng,
};
use umap_core::feature_metrics::{FeatureMetric, rank_feature_rows_with_tie_ranking};
use umap_core::fit::{
    DenseContinuousTarget, apply_categorical_target, apply_continuous_target,
    fit_dense_categorical_graph_with_diagnostics, fit_dense_continuous_graph,
    fit_dense_graph_with_cutoff, fit_dense_graph_with_options, fit_precomputed_graph_with_cutoff,
    fit_precomputed_graph_with_options, fit_sparse_precomputed_ranked_graph,
    fit_supplied_neighbors_graph,
};
use umap_core::fuzzy_graph::{
    CsrGraph, FuzzySetOperation, GraphOptions, KnnDistances, MembershipOperation, MembershipRows,
    SmoothKnnConfig, attenuate_categorical_edges, attenuate_metric_edges, combine_fuzzy_graphs,
    combine_memberships, compute_knn_memberships, reset_membership_scales, smooth_knn_distances,
};
use umap_core::inverse::{
    DenseInverseError, DenseInverseInput, DenseInverseLayout, InverseTriangulation,
    dense_inverse_epochs, fit_dense_inverse_scales_with_metric,
    fit_inverse_scales_from_ranked_distances, inverse_transform_dense_with_metric,
};
use umap_core::modern_initialization::{
    ModernInit, ModernInitConfig, ModernInitRescale, initialize_modern_layout, recursive_hub_order,
};
use umap_core::modern_layout::{ModernLayoutConfig, ModernOptimizer};
use umap_core::neighbors::DistanceMetric;
use umap_core::packed::{
    PackedBitMetric, PackedRows, fit_packed_graph, fit_packed_update_graph, rank_packed_queries,
};
use umap_core::parametric::prepare_parametric_edges;
use umap_core::rng::{Mt19937, recursive_part_labels};
use umap_core::row_distances::{MetricRow, row_distance};
use umap_core::sampling::{ModernSampledLayout, ModernSamplingConfig};
use umap_core::sparse::{
    CsrFeatures, SparseTieDistances, fit_sparse_euclidean_graph_with_cutoff,
    fit_sparse_feature_graph_with_cutoff, fit_sparse_graph_with_options, fit_sparse_update_graph,
    rank_sparse_feature_queries, unique_csr_rows,
};
use umap_core::transform::{
    CooGraph, DenseEmbeddingTransformConfig, DenseEmbeddingTransformError,
    NeighborhoodInitializationError, TransformInitializationError, initialize_csr_transform,
    initialize_ranked_transform, transform_dense_embedding_with_options,
    transform_dense_graph_with_cutoff, transform_dense_graph_with_tie_ranking,
    transform_embedding_from_graph, transform_embedding_from_partial_graph,
    transform_ranked_neighbors_graph,
};
use umap_core::update::{
    DenseUpdateConfig, DenseUpdateError, fit_dense_update_graph, update_dense_embedding,
    update_embedding_from_ranked_graph,
};
use umap_core::validation::{
    Trustworthiness, TrustworthinessError, TrustworthinessMetric, trustworthiness_from_dense,
};

type CsrComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type PackedComponents<'py> = (Bound<'py, PyBytes>, (usize, usize));
type FloatBuffer = (Vec<f64>, bool);
type MembershipBuffers = (Vec<usize>, Vec<usize>, Vec<f64>);
type NeighborMembershipComponents = (Vec<i32>, Vec<i32>, Vec<f32>, Option<Vec<f32>>);
type RowGroups = (Vec<usize>, Vec<usize>, Vec<usize>);
type CooComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type EmbeddingComponents = (CsrComponents, (usize, usize), Vec<f32>);
type RngState = (Vec<u32>, usize);
type GaussianRngState = (Vec<u32>, usize, Option<f64>);
type GraphEmbeddingComponents = ((usize, usize), Vec<f32>, RngState, Vec<Vec<f32>>);
type ModernEmbeddingComponents = (Vec<f32>, GaussianRngState, Vec<Vec<f32>>);
type RadiusInput = (Vec<f32>, usize, String);
type DensityInput = ((f64, f64, f64), RadiusInput);
type AlignedEmbeddingComponents = (Vec<Vec<f32>>, Option<RngState>);
type ParametricEdgeComponents = (Vec<(usize, usize, usize)>, usize);

fn neighborhood_init_error(error: NeighborhoodInitializationError) -> PyErr {
    match error {
        NeighborhoodInitializationError::Shape => PyValueError::new_err(error.to_string()),
        NeighborhoodInitializationError::Index => PyIndexError::new_err(error.to_string()),
    }
}

#[pyfunction]
fn _initialize_csr_transform(
    py: Python<'_>,
    graph: MembershipBuffers,
    graph_columns: usize,
    embedding: (Vec<f64>, (usize, usize)),
    precision: (bool, bool),
) -> PyResult<Vec<f32>> {
    let (offsets, columns, weights) = graph;
    let (embedding, shape) = embedding;
    py.detach(|| {
        initialize_csr_transform(
            (&offsets, &columns, &weights, graph_columns),
            (&embedding, shape),
            precision,
        )
    })
    .map_err(neighborhood_init_error)
}

#[pyfunction]
fn _initialize_ranked_transform(
    py: Python<'_>,
    ranks: (Vec<i64>, (usize, usize)),
    weights: &Bound<'_, PyAny>,
    embedding: &Bound<'_, PyAny>,
    embedding_shape: (usize, usize),
    dtype: &str,
) -> PyResult<Vec<f32>> {
    let (indices, rank_shape) = ranks;
    macro_rules! interpolate {
        ($dtype:ty, $add_product:expr) => {{
            let weights = weights.extract::<Vec<$dtype>>()?;
            let embedding = embedding.extract::<Vec<$dtype>>()?;
            py.detach(|| {
                initialize_ranked_transform(
                    &indices,
                    rank_shape,
                    &weights,
                    (&embedding, embedding_shape),
                    $add_product,
                )
            })
        }};
    }
    match dtype {
        "f32" => interpolate!(f32, |sum, weight, coordinate| sum + weight * coordinate),
        "f64" => interpolate!(
            f64,
            |sum, weight, coordinate| (f64::from(sum) + weight * coordinate) as f32
        ),
        "i64" => interpolate!(i64, |sum, weight: i64, coordinate| (f64::from(sum)
            + weight.wrapping_mul(coordinate) as f64)
            as f32),
        "u64" => interpolate!(u64, |sum, weight: u64, coordinate| (f64::from(sum)
            + weight.wrapping_mul(coordinate) as f64)
            as f32),
        _ => {
            return Err(PyValueError::new_err(
                "unsupported ranked initialization dtype",
            ));
        }
    }
    .map_err(neighborhood_init_error)
}

fn trustworthiness_error(error: TrustworthinessError) -> PyErr {
    match error {
        TrustworthinessError::ZeroDenominator => PyZeroDivisionError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pyfunction]
fn _trustworthiness_ranked(
    py: Python<'_>,
    source: Vec<Vec<i64>>,
    embedded: Vec<Vec<i64>>,
    max_k: usize,
) -> PyResult<Vec<f64>> {
    py.detach(|| {
        if source.len() != embedded.len() {
            return Err(TrustworthinessError::Shape);
        }
        let mut scores = Trustworthiness::new(source.len(), max_k)?;
        for (source_row, embedded_row) in source.iter().zip(&embedded) {
            scores.add_ranked_row(source_row, embedded_row)?;
        }
        scores.finish(0.0)
    })
    .map_err(trustworthiness_error)
}

#[pyfunction]
fn _trustworthiness_distances(
    source_rows: &Bound<'_, PyAny>,
    embedded: Vec<Vec<usize>>,
    max_k: usize,
) -> PyResult<Vec<f64>> {
    let mut scores = Trustworthiness::new(embedded.len(), max_k).map_err(trustworthiness_error)?;
    for (source, neighbors) in source_rows.try_iter()?.zip(&embedded) {
        let source: Vec<f64> = source?.extract()?;
        scores
            .add_distance_row(&source, neighbors)
            .map_err(trustworthiness_error)?;
    }
    scores.finish(1.0).map_err(trustworthiness_error)
}

#[pyfunction]
fn _trustworthiness_dense(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    n_features: usize,
    embedded: Vec<Vec<usize>>,
    max_k: usize,
    metric: &str,
    dtype: &str,
) -> PyResult<Vec<f64>> {
    let metric = match metric {
        "euclidean" => TrustworthinessMetric::Euclidean,
        "manhattan" => TrustworthinessMetric::Manhattan,
        "chebyshev" => TrustworthinessMetric::Chebyshev,
        _ => {
            return Err(PyValueError::new_err(
                "unknown trustworthiness source metric",
            ));
        }
    };
    macro_rules! evaluate {
        ($dtype:ty, $distance:ident) => {{
            let source = source.extract::<Vec<$dtype>>()?;
            py.detach(|| {
                trustworthiness_from_dense(&source, n_features, &embedded, max_k, |a, b| {
                    metric.$distance(a, b)
                })
            })
        }};
    }
    match dtype {
        "f32" => evaluate!(f32, distance_f32),
        "f64" => evaluate!(f64, distance_f64),
        "i64" => evaluate!(i64, distance_i64),
        "u64" => evaluate!(u64, distance_u64),
        _ => {
            return Err(PyValueError::new_err(
                "unknown trustworthiness source dtype",
            ));
        }
    }
    .map_err(trustworthiness_error)
}

#[pyfunction]
fn _prepare_parametric_edges(
    py: Python<'_>,
    graph: CsrComponents,
    n_epochs: usize,
) -> PyResult<ParametricEdgeComponents> {
    let ((n_rows, n_cols), indptr, indices, data) = graph;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    let pool = py
        .detach(|| prepare_parametric_edges(&graph, n_epochs))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let edges = pool
        .edges()
        .iter()
        .map(|edge| (edge.head, edge.tail, edge.repeats))
        .collect();
    Ok((edges, pool.total_repeats()))
}

/// Compute the log inputs for original and embedding radius diagnostics.
#[pyfunction]
fn _density_radius_log_inputs(
    py: Python<'_>,
    graph: CsrComponents,
    input: RadiusInput,
    coordinates: Vec<f32>,
    n_components: usize,
    n_neighbors: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let ((n_rows, n_cols), indptr, indices, data_weights) = graph;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data: data_weights,
    };
    let (data, n_features, metric) = input;
    let input = parse_radius_input(&data, n_features, &metric)?;
    py.detach(|| {
        let original = original_radius_log_inputs(&graph, input)?;
        let embedded =
            embedding_radius_log_inputs_euclidean(&coordinates, n_rows, n_components, n_neighbors)?;
        Ok((original, embedded))
    })
    .map_err(|error: DensityError| PyValueError::new_err(error.to_string()))
}

fn parse_radius_input<'a>(
    data: &'a [f32],
    n_features: usize,
    metric: &str,
) -> PyResult<OriginalRadiusInput<'a>> {
    if metric == "precomputed" {
        Ok(OriginalRadiusInput::Precomputed(data))
    } else {
        Ok(OriginalRadiusInput::Dense {
            values: data,
            n_features,
            metric: parse_metric(metric)?,
        })
    }
}

fn csr_components(graph: CsrGraph) -> CsrComponents {
    (
        (graph.n_rows, graph.n_cols),
        graph.indptr,
        graph.indices,
        graph.data,
    )
}

enum KnnFloatBuffer {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl KnnFloatBuffer {
    fn new((values, narrow): FloatBuffer) -> Self {
        if narrow {
            Self::Float32(values.into_iter().map(|value| value as f32).collect())
        } else {
            Self::Float64(values)
        }
    }

    fn borrowed(&self) -> KnnDistances<'_> {
        match self {
            Self::Float32(values) => KnnDistances::Float32(values),
            Self::Float64(values) => KnnDistances::Float64(values),
        }
    }
}

#[pyfunction]
fn _smooth_knn_distances(
    py: Python<'_>,
    distances: FloatBuffer,
    shape: (usize, usize),
    parameters: (f64, usize, f64, f64),
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let distances = KnnFloatBuffer::new(distances);
    let (k, iterations, local_connectivity, bandwidth) = parameters;
    let config = SmoothKnnConfig {
        k,
        iterations,
        local_connectivity,
        bandwidth,
    };
    py.detach(|| smooth_knn_distances(distances.borrowed(), shape, config))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _compute_knn_memberships(
    py: Python<'_>,
    indices: Vec<i32>,
    distances: Vec<f32>,
    sigmas: Vec<f32>,
    rhos: Vec<f32>,
    shape: (usize, usize),
    return_distances: bool,
    bipartite: bool,
) -> PyResult<NeighborMembershipComponents> {
    let result = py
        .detach(|| {
            compute_knn_memberships(
                &indices,
                &distances,
                &sigmas,
                &rhos,
                shape,
                return_distances,
                bipartite,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        result.rows,
        result.columns,
        result.weights,
        result.distances,
    ))
}

#[pyfunction]
fn _combine_fuzzy_graphs(
    py: Python<'_>,
    left: CsrComponents,
    right: CsrComponents,
    operation: &str,
) -> PyResult<(CsrComponents, Option<Vec<f32>>)> {
    let operation = match operation {
        "union" => FuzzySetOperation::Union,
        "intersection" => FuzzySetOperation::Intersection,
        "difference" => FuzzySetOperation::Difference,
        _ => return Err(PyValueError::new_err("unknown fuzzy set operation")),
    };
    let [mut left, right] =
        [left, right].map(|((n_rows, n_cols), indptr, indices, data)| CsrGraph {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
        });
    let graph = py
        .detach(|| combine_fuzzy_graphs(&mut left, &right, operation))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let left_weights = (operation == FuzzySetOperation::Difference).then_some(left.data);
    Ok((csr_components(graph), left_weights))
}

#[pyfunction]
fn _combine_memberships(
    py: Python<'_>,
    left: MembershipBuffers,
    right: MembershipBuffers,
    coordinates: (Vec<usize>, Vec<usize>),
    output: (Vec<f64>, String),
    aliases: [bool; 2],
    intersection: Option<(bool, f64)>,
) -> PyResult<Vec<f64>> {
    let (rows, columns) = coordinates;
    let (mut values, storage) = output;
    let operation = match intersection {
        Some((right_complement, weight)) => MembershipOperation::Intersection {
            weight,
            right_complement,
        },
        None => MembershipOperation::Union,
    };
    let store: fn(f64) -> f64 = match storage.as_str() {
        "f32" => |value| f64::from(value as f32),
        "int" => f64::trunc,
        "bool" => |value| if value == 0.0 { 0.0 } else { 1.0 },
        _ => |value| value,
    };
    py.detach(|| -> Result<_, &'static str> {
        let left = MembershipRows::new(&left.0, &left.1, &left.2)?;
        let right = MembershipRows::new(&right.0, &right.1, &right.2)?;
        combine_memberships(
            [&left, &right],
            (&rows, &columns),
            &mut values,
            aliases,
            operation,
            store,
        )?;
        Ok(values)
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction]
fn _reset_membership_scales(
    py: Python<'_>,
    indptr: Vec<usize>,
    mut values: Vec<f64>,
) -> PyResult<Vec<f64>> {
    py.detach(|| reset_membership_scales(&indptr, &mut values))
        .map_err(PyValueError::new_err)?;
    Ok(values)
}

#[pyfunction]
fn _attenuate_categorical_edges(
    py: Python<'_>,
    coordinates: (Vec<usize>, Vec<usize>),
    weights: FloatBuffer,
    labels: Vec<i64>,
    unknown: (f64, bool),
    far: (f64, bool),
) -> PyResult<Vec<f64>> {
    let (rows, columns) = coordinates;
    let (mut values, single) = weights;
    py.detach(|| {
        attenuate_categorical_edges(
            (&rows, &columns),
            &mut values,
            &labels,
            unknown,
            far,
            single,
        )
    })
    .map_err(PyValueError::new_err)?;
    Ok(values)
}

#[pyfunction]
fn _attenuate_metric_edges(
    py: Python<'_>,
    weights: FloatBuffer,
    exponents: Vec<(f64, bool)>,
) -> PyResult<Vec<f64>> {
    let (mut values, single) = weights;
    py.detach(|| attenuate_metric_edges(&mut values, &exponents, single))
        .map_err(PyValueError::new_err)?;
    Ok(values)
}

#[pyfunction]
fn _discrete_count_distance(py: Python<'_>, x: i64, y: i64, poisson_lambda: (f64, bool)) -> f64 {
    py.detach(|| count_distance(x, y, poisson_lambda))
}

#[pyfunction]
fn _discrete_edit_distance(py: Python<'_>, x: Vec<u32>, y: Vec<u32>, bound: f64, cap: bool) -> f64 {
    py.detach(|| {
        if cap {
            myers(&x, &y, bound)
        } else {
            levenshtein(&x, &y, bound)
        }
    })
}

#[pyfunction]
fn _discrete_magnitude_distance(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    name: &str,
    dtype: &str,
) -> PyResult<f64> {
    let metric = match name {
        "euclidean" => TrustworthinessMetric::Euclidean,
        "manhattan" => TrustworthinessMetric::Manhattan,
        "chebyshev" => TrustworthinessMetric::Chebyshev,
        _ => return Err(PyValueError::new_err("unknown magnitude distance")),
    };
    macro_rules! evaluate {
        ($type:ty, $distance:ident) => {{
            let x: Vec<$type> = x.extract()?;
            let y: Vec<$type> = y.extract()?;
            if x.len() != y.len() {
                return Err(PyValueError::new_err(
                    "distance vectors must have the same length",
                ));
            }
            Ok(metric.$distance(&x, &y))
        }};
    }
    match dtype {
        "f32" => evaluate!(f32, distance_f32),
        "f64" => evaluate!(f64, distance_f64),
        "i64" => evaluate!(i64, distance_i64),
        "u64" => evaluate!(u64, distance_u64),
        _ => Err(PyValueError::new_err(
            "unsupported magnitude distance dtype",
        )),
    }
}

#[pyfunction]
fn _support_distance(
    py: Python<'_>,
    x: &Bound<'_, PyBytes>,
    y: &Bound<'_, PyBytes>,
    name: &str,
) -> PyResult<f64> {
    let metric = parse_metric(name)?;
    let (x, y) = (x.as_bytes(), y.as_bytes());
    py.detach(|| {
        let x: Vec<_> = x.iter().map(|&value| value != 0).collect();
        let y: Vec<_> = y.iter().map(|&value| value != 0).collect();
        metric.support_distance(&x, &y)
    })
    .ok_or_else(|| {
        PyValueError::new_err("support distance requires a boolean metric and equal widths")
    })
}

fn metric_row_words<T, const N: usize>(
    values: &[u8],
    decode: fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    let (words, trailing) = values.as_chunks::<N>();
    if !trailing.is_empty() {
        return Err(PyValueError::new_err("invalid metric row byte width"));
    }
    Ok(words.iter().copied().map(decode).collect())
}

fn with_metric_row(
    values: &[u8],
    dtype: &str,
    evaluate: impl FnOnce(MetricRow<'_>) -> PyResult<f64>,
) -> PyResult<f64> {
    match dtype {
        "f32" => evaluate(MetricRow::F32(&metric_row_words(
            values,
            f32::from_ne_bytes,
        )?)),
        "f64" => evaluate(MetricRow::F64(&metric_row_words(
            values,
            f64::from_ne_bytes,
        )?)),
        "i64" => evaluate(MetricRow::Signed(&metric_row_words(
            values,
            i64::from_ne_bytes,
        )?)),
        "u64" => evaluate(MetricRow::Unsigned(&metric_row_words(
            values,
            u64::from_ne_bytes,
        )?)),
        _ => Err(PyValueError::new_err("unsupported metric row dtype")),
    }
}

#[pyfunction]
fn _numeric_row_distance(
    py: Python<'_>,
    x: &Bound<'_, PyBytes>,
    y: &Bound<'_, PyBytes>,
    metric: &str,
    x_dtype: &str,
    y_dtype: &str,
) -> PyResult<f64> {
    let metric = parse_metric(metric)?;
    let (x, y) = (x.as_bytes(), y.as_bytes());
    py.detach(|| {
        with_metric_row(x, x_dtype, |left| {
            with_metric_row(y, y_dtype, |right| {
                row_distance(metric, left, right)
                    .ok_or_else(|| PyValueError::new_err("unsupported metric or invalid row width"))
            })
        })
    })
}

fn parse_feature_metric<'a>(
    name: &str,
    parameter: f64,
    values: &'a [f64],
) -> PyResult<FeatureMetric<'a>> {
    Ok(match name {
        "minkowski" => FeatureMetric::Minkowski(parameter),
        "weighted_minkowski" => FeatureMetric::WeightedMinkowski(parameter, values),
        "standardised_euclidean" => FeatureMetric::StandardizedEuclidean(values),
        "mahalanobis" => FeatureMetric::Mahalanobis(values),
        "haversine" => FeatureMetric::Haversine,
        "poincare" => FeatureMetric::Poincare,
        "hellinger" => FeatureMetric::Hellinger,
        "softmax_hellinger" => FeatureMetric::SoftmaxHellinger,
        "symmetric_kl" => FeatureMetric::SymmetricKl(parameter),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported feature metric {name:?}"
            )));
        }
    })
}

#[pyfunction]
fn _feature_metric_distance(
    py: Python<'_>,
    name: &str,
    parameter: f64,
    values: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
) -> PyResult<f64> {
    let metric = parse_feature_metric(name, parameter, &values)?;
    py.detach(|| metric.distance(&x, &y))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (name, parameter, values, training, queries, width, k, cutoff, rank_ties=None))]
#[allow(clippy::too_many_arguments)]
fn _rank_configured_features(
    py: Python<'_>,
    name: &str,
    parameter: f64,
    values: Vec<f64>,
    training: &Bound<'_, PyBytes>,
    queries: Option<&Bound<'_, PyBytes>>,
    width: usize,
    k: usize,
    cutoff: f64,
    rank_ties: Option<Py<PyAny>>,
) -> PyResult<(Vec<i32>, Vec<f32>, usize)> {
    let metric = parse_feature_metric(name, parameter, &values)?;
    let training = training.as_bytes();
    let queries = queries.map(|rows| rows.as_bytes());
    let ranked = py.detach(|| {
        let training = metric_row_words(training, f64::from_ne_bytes)?;
        let queries = queries
            .map(|data| metric_row_words(data, f64::from_ne_bytes))
            .transpose()?;
        let mut callback_error = None;
        let rank_ties = rank_ties.as_ref().map(|callback| {
            |distances: &[f32], width| {
                python_rank_ties(callback, distances, width, &mut callback_error)
            }
        });
        let ranked = rank_feature_rows_with_tie_ranking(
            metric,
            &training,
            queries.as_deref(),
            width,
            k,
            cutoff,
            rank_ties,
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        ranked.map_err(PyValueError::new_err)
    })?;
    Ok((ranked.columns, ranked.distances, ranked.removed))
}

#[pyclass(module = "umap_alternative._native")]
struct NativeApproximateIndex {
    index: DenseApproximateIndex<'static>,
}

#[pymethods]
impl NativeApproximateIndex {
    #[new]
    fn new(
        py: Python<'_>,
        training: &Bound<'_, PyBytes>,
        shape: (usize, usize),
        k: usize,
        metric: &str,
        seed: u32,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let data = training.as_bytes();
        let index = py.detach(|| {
            let data = metric_row_words(data, f32::from_ne_bytes)?;
            DenseApproximateIndex::build(&data, shape.0, shape.1, k, metric, seed)
                .map(DenseApproximateIndex::into_owned)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })?;
        Ok(Self { index })
    }

    fn training_ranks(&self) -> (Vec<i32>, Vec<f32>) {
        let ranks = self.index.training_ranks();
        (ranks.indices.clone(), ranks.distances.clone())
    }

    fn query(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyBytes>,
        n_rows: usize,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        let bytes = queries.as_bytes();
        let ranks = py.detach(|| {
            let queries = metric_row_words(bytes, f32::from_ne_bytes)?;
            self.index
                .query(&queries, n_rows)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })?;
        Ok((ranks.indices, ranks.distances))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _optimize_aligned_layout(
    py: Python<'_>,
    graphs: Vec<CsrComponents>,
    mut layouts: Vec<Vec<f32>>,
    relations: Vec<Vec<(usize, usize)>>,
    n_components: usize,
    n_epochs: usize,
    first_moving: usize,
    window: usize,
    regularisation: f64,
    seed: [i64; 3],
) -> PyResult<AlignedEmbeddingComponents> {
    let graphs: Vec<_> = graphs
        .into_iter()
        .map(|((n_rows, n_cols), indptr, indices, data)| CsrGraph {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
        })
        .collect();
    let config = AlignedLayoutConfig {
        n_components,
        n_epochs,
        first_moving,
        window,
        regularisation,
        seed,
    };
    let rng = py
        .detach(|| optimize_aligned_layout(&graphs, &mut layouts, &relations, &config))
        .map_err(PyValueError::new_err)?;
    let rng_state = rng.map(|rng| {
        let (words, cursor) = rng.state();
        (words.to_vec(), cursor)
    });
    Ok((layouts.split_off(first_moving), rng_state))
}

/// Parse the metric names supported by the feature graph compatibility paths.
///
/// Metric names and aliases are case-insensitive at this boundary.
fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" | "cityblock" | "taxicab" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linfty" | "linfinity" | "max" => Ok(DistanceMetric::Chebyshev),
        "cosine" => Ok(DistanceMetric::Cosine),
        "correlation" => Ok(DistanceMetric::Correlation),
        "canberra" => Ok(DistanceMetric::Canberra),
        "braycurtis" => Ok(DistanceMetric::BrayCurtis),
        "hamming" => Ok(DistanceMetric::Hamming),
        "matching" => Ok(DistanceMetric::Matching),
        "jaccard" => Ok(DistanceMetric::Jaccard),
        "dice" => Ok(DistanceMetric::Dice),
        "russellrao" => Ok(DistanceMetric::RussellRao),
        "rogerstanimoto" => Ok(DistanceMetric::RogersTanimoto),
        "sokalmichener" => Ok(DistanceMetric::SokalMichener),
        "sokalsneath" => Ok(DistanceMetric::SokalSneath),
        "yule" => Ok(DistanceMetric::Yule),
        "kulsinski" => Ok(DistanceMetric::Kulsinski),
        _ => Err(PyValueError::new_err(format!(
            "unsupported metric {metric:?}"
        ))),
    }
}

fn parse_graph_options(settings: Option<(f64, f64)>) -> PyResult<Option<GraphOptions>> {
    settings
        .map(|(local_connectivity, set_op_mix_ratio)| {
            GraphOptions::new(local_connectivity, set_op_mix_ratio)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .transpose()
}

/// Fit a fuzzy graph from a flat row-major f32 matrix and return its CSR data.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, *, graph_options=None))]
fn _fit_dense_graph(
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    graph_options: Option<(f64, f64)>,
) -> PyResult<CsrComponents> {
    let metric = parse_metric(metric)?;
    fit_dense_graph_with_options(
        &data,
        n_rows,
        n_features,
        k,
        metric,
        parse_graph_options(graph_options)?,
    )
    .map(csr_components)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a dense fuzzy graph and count entries excluded by the cutoff.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, cutoff, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_dense_graph_with_cutoff(
    py: Python<'_>,
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    cutoff: f32,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| fit_dense_graph_with_cutoff(&data, n_rows, n_features, k, metric, cutoff, options))
        .map(|(graph, removed)| (csr_components(graph), removed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn parse_packed_metric(metric: &str) -> PyResult<PackedBitMetric> {
    match metric {
        "bit_hamming" => Ok(PackedBitMetric::Hamming),
        "bit_jaccard" => Ok(PackedBitMetric::Jaccard),
        _ => Err(PyValueError::new_err(format!(
            "unsupported packed metric {metric:?}"
        ))),
    }
}

fn is_packed_metric(metric: &str) -> bool {
    matches!(metric, "bit_hamming" | "bit_jaccard")
}

#[pyfunction]
#[pyo3(signature = (data, shape, k, metric, cutoff, *, graph_options=None))]
fn _fit_packed_graph(
    py: Python<'_>,
    data: Bound<'_, PyBytes>,
    shape: (usize, usize),
    k: usize,
    metric: &str,
    cutoff: f64,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let training = PackedRows {
        data: data.as_bytes(),
        shape,
    };
    let metric = parse_packed_metric(metric)?;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| fit_packed_graph(training, k, metric, cutoff, options))
        .map(|(graph, removed)| (csr_components(graph), removed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (data, n_existing, k, metric, *, graph_options=None))]
fn _fit_packed_update_graph(
    py: Python<'_>,
    data: PackedComponents<'_>,
    n_existing: usize,
    k: usize,
    metric: &str,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, Vec<usize>)> {
    let training = PackedRows {
        data: data.0.as_bytes(),
        shape: data.1,
    };
    let metric = parse_packed_metric(metric)?;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| fit_packed_update_graph(training, n_existing, k, metric, options))
        .map(|(graph, ranks)| (csr_components(graph), ranks))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn _rank_packed_queries(
    py: Python<'_>,
    training: PackedComponents<'_>,
    queries: PackedComponents<'_>,
    k: usize,
    metric: &str,
    rank_ties: Py<PyAny>,
) -> PyResult<(Vec<i32>, Vec<f32>)> {
    let training = PackedRows {
        data: training.0.as_bytes(),
        shape: training.1,
    };
    let queries = PackedRows {
        data: queries.0.as_bytes(),
        shape: queries.1,
    };
    let metric = parse_packed_metric(metric)?;
    py.detach(|| {
        let mut callback_error = None;
        let result = rank_packed_queries(training, queries, k, metric, |distances, width| {
            python_rank_ties(&rank_ties, distances, width, &mut callback_error)
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        let ranks = result.map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut columns = Vec::with_capacity(ranks.len());
        let mut distances = Vec::with_capacity(ranks.len());
        for rank in ranks {
            columns.push(
                i32::try_from(rank.index)
                    .map_err(|_| PyValueError::new_err("packed query index exceeds i32"))?,
            );
            distances.push(rank.distance);
        }
        Ok((columns, distances))
    })
}

/// Fit a fuzzy graph from sorted CSR features and return its CSR data.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, shape, k, metric, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_sparse_graph(
    py: Python<'_>,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    shape: (usize, usize),
    k: usize,
    metric: &str,
    graph_options: Option<(f64, f64)>,
) -> PyResult<CsrComponents> {
    let metric = parse_metric(metric)?;
    let (n_rows, n_features) = shape;
    let options = parse_graph_options(graph_options)?;
    py.detach(|| {
        fit_sparse_graph_with_options(
            &indptr, &indices, &data, n_rows, n_features, k, metric, options,
        )
    })
    .map(csr_components)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (training, k, metric, cutoff, rank_ties, *, graph_options=None))]
fn _fit_sparse_feature_graph_with_cutoff(
    py: Python<'_>,
    training: CsrComponents,
    k: usize,
    metric: &str,
    cutoff: f64,
    rank_ties: Py<PyAny>,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| {
        let input = csr_features(&training);
        let mut callback_error = None;
        let result = if metric == DistanceMetric::Euclidean {
            fit_sparse_euclidean_graph_with_cutoff(
                input.indptr,
                input.indices,
                input.data,
                input.shape,
                k,
                cutoff as f32,
                options,
            )
        } else {
            fit_sparse_feature_graph_with_cutoff(input, k, metric, cutoff, options, |row, width| {
                python_sparse_rank_ties(&rank_ties, row, width, &mut callback_error)
            })
        };
        if let Some(error) = callback_error {
            return Err(error);
        }
        result
            .map(|(graph, removed)| (csr_components(graph), removed))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    })
}

/// Fit the serial seeded dense layout and return its CSR graph and coordinates.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, n_components, a, b, learning_rate, n_epochs, repulsion_strength, negative_sample_rate, seed, initial_coordinates=None, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_dense_embedding(
    py: Python<'_>,
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    n_epochs: usize,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    seed: u32,
    initial_coordinates: Option<Vec<f32>>,
    graph_options: Option<(f64, f64)>,
) -> PyResult<EmbeddingComponents> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?;
    let init = match initial_coordinates {
        Some(values) => DenseEmbeddingInit::Explicit { values },
        None => DenseEmbeddingInit::random(seed),
    };
    let config = DenseEmbeddingConfig::<f64, f64> {
        n_neighbors: k,
        metric,
        n_components,
        a,
        b,
        learning_rate,
        n_epochs,
        repulsion_strength,
        negative_sample_rate,
        init,
    };
    let DenseEmbedding {
        graph,
        coordinates,
        shape,
    } = py
        .detach(|| {
            fit_dense_embedding_seeded_with_graph_options(
                &data, n_rows, n_features, &config, seed, options,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((csr_components(graph), shape, coordinates))
}

/// Append rows and optimize the combined dense layout using the fitted settings.
#[pyfunction]
#[pyo3(signature = (
    data, shape, n_existing, fitted, *, k, metric, n_components, a, b, learning_rate,
    n_epochs, repulsion_strength, negative_sample_rate, transform_seed, graph_options=None
))]
#[allow(clippy::too_many_arguments)]
fn _update_dense_embedding(
    py: Python<'_>,
    data: Vec<f32>,
    shape: (usize, usize),
    n_existing: usize,
    fitted: Vec<f32>,
    k: usize,
    metric: &str,
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    n_epochs: usize,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    transform_seed: u32,
    graph_options: Option<(f64, f64)>,
) -> PyResult<EmbeddingComponents> {
    let config = DenseUpdateConfig {
        n_neighbors: k,
        metric: parse_metric(metric)?,
        n_components,
        graph_options: parse_graph_options(graph_options)?,
        transform_seed,
    };
    let layout = EmbeddingLayoutConfig {
        n_components,
        a,
        b,
        learning_rate,
        n_epochs,
        repulsion_strength,
        negative_sample_rate,
    };
    let result = py.detach(|| {
        update_dense_embedding(
            &data, shape.0, shape.1, n_existing, &fitted, &config, &layout,
        )
    });
    let DenseEmbedding {
        graph,
        coordinates,
        shape,
    } = match result {
        Ok(embedding) => embedding,
        Err(DenseUpdateError::CoincidentNonFiniteLayout { graph }) => {
            let error = PyValueError::new_err("Input contains NaN.");
            error
                .value(py)
                .setattr("_update_graph", csr_components(graph))?;
            return Err(error);
        }
        Err(error) => return Err(PyValueError::new_err(error.to_string())),
    };
    Ok((csr_components(graph), shape, coordinates))
}

/// Rebuild the dense feature graph before the update layout is attempted.
#[pyfunction]
#[pyo3(signature = (data, shape, n_existing, k, metric, *, graph_options=None))]
fn _fit_dense_update_graph(
    py: Python<'_>,
    data: Vec<f32>,
    shape: (usize, usize),
    n_existing: usize,
    k: usize,
    metric: &str,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, Vec<usize>)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?;
    py.detach(|| fit_dense_update_graph(&data, shape, n_existing, k, metric, options))
        .map(|(graph, ranks)| (csr_components(graph), ranks))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Rebuild the sparse feature graph and return the ranks used to initialize new rows.
#[pyfunction]
#[pyo3(signature = (data, n_existing, k, metric, *, graph_options=None))]
fn _fit_sparse_update_graph(
    py: Python<'_>,
    data: CsrComponents,
    n_existing: usize,
    k: usize,
    metric: &str,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, Vec<usize>)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| fit_sparse_update_graph(csr_features(&data), n_existing, k, metric, options))
        .map(|(graph, ranks)| (csr_components(graph), ranks))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Initialize and optimize appended rows from the rebuilt graph and their ranks.
#[pyfunction]
#[pyo3(signature = (
    graph, n_existing, appended_ranks, fitted, *, n_components, a, b, learning_rate,
    n_epochs, repulsion_strength, negative_sample_rate, transform_seed
))]
#[allow(clippy::too_many_arguments)]
fn _update_ranked_embedding(
    py: Python<'_>,
    graph: CsrComponents,
    n_existing: usize,
    appended_ranks: Vec<usize>,
    fitted: Vec<f32>,
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    n_epochs: usize,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    transform_seed: u32,
) -> PyResult<EmbeddingComponents> {
    let ((n_rows, n_cols), indptr, indices, data) = graph;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    let layout = EmbeddingLayoutConfig {
        n_components,
        a,
        b,
        learning_rate,
        n_epochs,
        repulsion_strength,
        negative_sample_rate,
    };
    let DenseEmbedding {
        graph,
        coordinates,
        shape,
    } = py
        .detach(|| {
            update_embedding_from_ranked_graph(
                graph,
                n_existing,
                &appended_ranks,
                &fitted,
                transform_seed,
                &layout,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((csr_components(graph), shape, coordinates))
}

/// Optimize raw coordinates with the same graph and MT19937 stream used for initialization.
#[pyfunction]
#[pyo3(signature = (
    graph, initial_coordinates, rng_state, n_components, a, b, learning_rate, n_epochs,
    repulsion_strength, negative_sample_rate, *, checkpoint_epochs=None, density=None
))]
#[allow(clippy::too_many_arguments)]
fn _fit_graph_embedding_with_rng(
    py: Python<'_>,
    graph: CsrComponents,
    initial_coordinates: Vec<f32>,
    rng_state: RngState,
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    n_epochs: usize,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    checkpoint_epochs: Option<Vec<usize>>,
    density: Option<DensityInput>,
) -> PyResult<GraphEmbeddingComponents> {
    let ((n_rows, n_cols), indptr, indices, data) = graph;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    let (words, cursor) = rng_state;
    let mut rng = Mt19937::from_state(&words, cursor)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let config = EmbeddingLayoutConfig {
        n_components,
        a,
        b,
        learning_rate,
        n_epochs,
        repulsion_strength,
        negative_sample_rate,
    };
    let density = if let Some(((dens_lambda, dens_frac, dens_var_shift), (data, width, metric))) =
        density.as_ref()
    {
        Some((
            parse_radius_input(data, *width, metric)?,
            DensmapOptions {
                dens_lambda: *dens_lambda,
                dens_frac: *dens_frac,
                dens_var_shift: *dens_var_shift,
            },
        ))
    } else {
        None
    };
    let mut history = Vec::new();
    let DenseEmbedding {
        coordinates, shape, ..
    } = if density.is_none() && checkpoint_epochs.is_none() {
        py.detach(|| fit_graph_embedding_with_rng(graph, &initial_coordinates, &config, &mut rng))
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        let checkpoint = checkpoint_epochs.as_ref().map(|epochs| {
            let mut keys = epochs.iter().peekable();
            let history = &mut history;
            move |key: usize, values: &[f32]| {
                if keys.peek().is_some_and(|&&next| next == key) {
                    history.push(values.to_vec());
                    keys.next();
                }
            }
        });
        py.detach(|| {
            fit_graph_with_checkpoints(
                graph,
                &initial_coordinates,
                &config,
                density,
                &mut rng,
                checkpoint,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?
    };
    let (words, cursor) = rng.state();
    Ok((shape, coordinates, (words.to_vec(), cursor), history))
}

/// Prepare modern coordinates on a fitted CSR graph and return the advanced RNG.
#[pyfunction]
#[pyo3(signature = (components, rng_state, *, n_components, init, init_rescale, init_scale_factor, coordinates=None))]
#[allow(clippy::too_many_arguments)]
fn _initialize_modern_layout(
    py: Python<'_>,
    components: CsrComponents,
    rng_state: GaussianRngState,
    n_components: usize,
    init: &str,
    init_rescale: &str,
    init_scale_factor: f64,
    coordinates: Option<Vec<f32>>,
) -> PyResult<(Vec<f32>, GaussianRngState)> {
    let ((n_rows, n_cols), indptr, indices, data) = components;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    let init = match (init, coordinates.as_deref()) {
        ("explicit", Some(values)) => ModernInit::Explicit(values),
        ("random", None) => ModernInit::Random,
        ("spectral", Some(values)) => ModernInit::Spectral(values),
        ("prepared", Some(values)) => ModernInit::Prepared(values),
        _ => {
            return Err(PyValueError::new_err(
                "invalid layout initializer or coordinate input",
            ));
        }
    };
    let rescale = match init_rescale {
        "unscaled" => ModernInitRescale::Unscaled,
        "graph_edge" => ModernInitRescale::GraphEdge,
        "per_axis" => ModernInitRescale::PerAxis,
        "isotropic" => ModernInitRescale::Isotropic,
        _ => return Err(PyValueError::new_err("invalid init_rescale")),
    };
    let config = ModernInitConfig {
        n_components,
        rescale,
        scale_factor: init_scale_factor,
    };
    let (words, cursor, mut gaussian) = rng_state;
    let mut rng = Mt19937::from_state(&words, cursor)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let coordinates = py
        .detach(|| initialize_modern_layout(&graph, init, config, &mut rng, &mut gaussian))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (words, cursor) = rng.state();
    Ok((coordinates, (words.to_vec(), cursor, gaussian)))
}

#[pyfunction]
fn _recursive_part_labels(
    seed: [i64; 3],
    rows: Vec<usize>,
    n_parts: std::num::NonZeroUsize,
) -> Vec<usize> {
    recursive_part_labels(seed, &rows, n_parts)
}

#[pyfunction]
fn _recursive_hub_order(degrees: Vec<f32>) -> Vec<usize> {
    recursive_hub_order(&degrees)
}

/// Apply modern epochs to already initialized coordinates.
#[pyfunction]
#[pyo3(signature = (
    components, coordinates, rng_state, *, optimizer, n_components, n_epochs, a, b,
    learning_rate, repulsion_strength, negative_sample_rate, exclude_graph_neighbors,
    checkpoint_epochs=None, project=None, refinement_schedule=false, coarse=false
))]
#[allow(clippy::too_many_arguments)]
fn _fit_modern_layout(
    py: Python<'_>,
    components: CsrComponents,
    coordinates: Vec<f32>,
    rng_state: GaussianRngState,
    optimizer: &str,
    n_components: usize,
    n_epochs: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    exclude_graph_neighbors: bool,
    checkpoint_epochs: Option<Vec<usize>>,
    project: Option<Py<PyAny>>,
    refinement_schedule: bool,
    coarse: bool,
) -> PyResult<ModernEmbeddingComponents> {
    let ((n_rows, n_cols), indptr, indices, data) = components;
    let graph = CsrGraph {
        n_rows,
        n_cols,
        indptr,
        indices,
        data,
    };
    let optimizer = match optimizer {
        "adam" => ModernOptimizer::Adam,
        "momentum" => ModernOptimizer::Momentum,
        _ => return Err(PyValueError::new_err("invalid modern optimizer")),
    };
    let config = ModernLayoutConfig {
        optimizer,
        n_components,
        n_epochs,
        learning_rate,
        a,
        b,
        repulsion_strength,
    };
    // The Python projector owns the rank order and caller RNG on that path.
    let sampling = ModernSamplingConfig {
        negative_sample_rate,
        exclude_graph_neighbors,
        negative_selection_range: if project.is_some() { 0 } else { n_rows },
    };
    let (words, cursor, mut gaussian) = rng_state;
    let mut rng = Mt19937::from_state(&words, cursor)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let use_projection = project.is_some();
    if coarse && !use_projection {
        return Err(PyValueError::new_err(
            "coarse layout requires a supplied candidate order",
        ));
    }
    let mut sampled = py
        .detach(|| {
            if coarse {
                ModernSampledLayout::with_coarse_candidates(&graph, coordinates, config, sampling)
            } else if use_projection {
                ModernSampledLayout::with_supplied_candidates(&graph, coordinates, config, sampling)
            } else {
                ModernSampledLayout::new(&graph, coordinates, config, sampling, &mut rng)
            }
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if refinement_schedule {
        sampled = sampled.with_refinement_schedule();
    }
    let mut order = if use_projection {
        (0..n_rows).collect()
    } else {
        Vec::new()
    };
    let mut history = Vec::new();
    let mut checkpoints = checkpoint_epochs.into_iter().flatten().peekable();
    for epoch in 0..n_epochs {
        py.detach(|| {
            if use_projection {
                sampled.step_with_candidate_order(&order)
            } else {
                sampled.step(&mut rng, &mut gaussian)
            }
            .map(|_| ())
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if checkpoints.next_if_eq(&epoch).is_some() {
            history.push(sampled.coordinates().to_vec());
        }
        if let Some(project) = &project {
            order = project
                .call1(py, (epoch, sampled.coordinates().to_vec()))?
                .extract(py)?;
        }
    }
    if checkpoints.next_if_eq(&n_epochs).is_some() {
        history.push(sampled.coordinates().to_vec());
    }
    let (words, cursor) = rng.state();
    Ok((
        sampled.into_coordinates(),
        (words.to_vec(), cursor, gaussian),
        history,
    ))
}

/// Fit a fuzzy graph from dense features and one categorical label per row.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, labels, target_weight, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_dense_categorical_graph(
    py: Python<'_>,
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    labels: Vec<i64>,
    target_weight: f64,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?;
    py.detach(|| {
        fit_dense_categorical_graph_with_diagnostics(
            &data,
            n_rows,
            n_features,
            k,
            metric,
            &labels,
            target_weight,
            options,
        )
    })
    .map(|(graph, isolated)| (csr_components(graph), isolated))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

type ContinuousTargetInput = (Vec<f32>, usize, usize, String, f64);

fn continuous_target(input: &ContinuousTargetInput) -> PyResult<DenseContinuousTarget<'_>> {
    let (values, width, count, metric, weight) = input;
    Ok(DenseContinuousTarget {
        values,
        n_features: *width,
        n_neighbors: *count,
        metric: parse_metric(metric)?,
        weight: *weight,
    })
}

fn csr_graph((shape, indptr, indices, data): CsrComponents) -> CsrGraph {
    CsrGraph {
        n_rows: shape.0,
        n_cols: shape.1,
        indptr,
        indices,
        data,
    }
}

#[pyfunction]
fn _apply_categorical_target(
    py: Python<'_>,
    graph: CsrComponents,
    labels: Vec<i64>,
    target_weight: f64,
) -> PyResult<CsrComponents> {
    py.detach(|| apply_categorical_target(csr_graph(graph), &labels, target_weight))
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn _apply_continuous_target(
    py: Python<'_>,
    graph: CsrComponents,
    columns: Vec<usize>,
    distances: Vec<f32>,
    k: usize,
    weight: f64,
) -> PyResult<CsrComponents> {
    py.detach(|| apply_continuous_target(csr_graph(graph), &columns, &distances, k, weight))
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a fuzzy graph from dense features and scalar or vector targets.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, target, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_dense_continuous_graph(
    py: Python<'_>,
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    target: ContinuousTargetInput,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let metric = parse_metric(metric)?;
    let options = parse_graph_options(graph_options)?;
    let target = continuous_target(&target)?;
    py.detach(|| fit_dense_continuous_graph(&data, n_rows, n_features, k, metric, options, &target))
        .map(|(graph, isolated)| (csr_components(graph), isolated))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a fuzzy graph from a flat row-major square f32 distance matrix.
#[pyfunction]
#[pyo3(signature = (distances, n_rows, k, *, graph_options=None))]
fn _fit_precomputed_graph(
    distances: Vec<f32>,
    n_rows: usize,
    k: usize,
    graph_options: Option<(f64, f64)>,
) -> PyResult<CsrComponents> {
    fit_precomputed_graph_with_options(&distances, n_rows, k, parse_graph_options(graph_options)?)
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (distances, n_rows, k, cutoff, *, graph_options=None))]
fn _fit_precomputed_graph_with_cutoff(
    py: Python<'_>,
    distances: Vec<f32>,
    n_rows: usize,
    k: usize,
    cutoff: f32,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| fit_precomputed_graph_with_cutoff(&distances, n_rows, k, cutoff, options))
        .map(|(graph, removed)| (csr_components(graph), removed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a fuzzy graph from row-major indices and distances after estimator masking.
#[pyfunction]
#[pyo3(signature = (indices, distances, shape, k, *, graph_options=None, sparse_precomputed=false))]
fn _fit_supplied_neighbors_graph(
    py: Python<'_>,
    indices: Vec<i32>,
    distances: Vec<f32>,
    shape: (usize, usize),
    k: usize,
    graph_options: Option<(f64, f64)>,
    sparse_precomputed: bool,
) -> PyResult<CsrComponents> {
    let (n_rows, width) = shape;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| {
        if sparse_precomputed {
            return fit_sparse_precomputed_ranked_graph(&indices, &distances, shape, k, options);
        }
        fit_supplied_neighbors_graph(
            &indices,
            &distances,
            n_rows,
            width,
            k,
            options,
            f32::INFINITY,
        )
    })
    .map(csr_components)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return the shape, row indices, column indices, and memberships in COO order.
#[pyfunction]
#[pyo3(signature = (
    training, n_training_rows, queries, n_query_rows, n_features, k, metric, *,
    local_connectivity=1.0, disconnection_distance=None, ranked_columns=None, rank_ties=None
))]
#[allow(clippy::too_many_arguments)]
fn _transform_dense_graph(
    py: Python<'_>,
    training: Vec<f32>,
    n_training_rows: usize,
    queries: Vec<f32>,
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    local_connectivity: f64,
    disconnection_distance: Option<f32>,
    ranked_columns: Option<Vec<i32>>,
    rank_ties: Option<Py<PyAny>>,
) -> PyResult<CooComponents> {
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let CooGraph {
        n_rows,
        n_cols,
        row,
        col,
        data,
    } = if metric == "precomputed" || ranked_columns.is_some() {
        ranked_query_graph(
            py,
            ranked_columns.as_deref(),
            &queries,
            (n_query_rows, n_training_rows, k),
            disconnection_distance.unwrap_or(f32::INFINITY),
            options,
            is_packed_metric(metric),
        )?
    } else {
        feature_query_graph(
            py,
            &training,
            &queries,
            (n_training_rows, n_query_rows, n_features),
            k,
            parse_metric(metric)?,
            disconnection_distance,
            options,
            rank_ties.as_ref(),
        )?
    };
    Ok(((n_rows, n_cols), row, col, data))
}

#[allow(clippy::too_many_arguments)]
fn feature_query_graph(
    py: Python<'_>,
    training: &[f32],
    queries: &[f32],
    shape: (usize, usize, usize),
    k: usize,
    metric: DistanceMetric,
    cutoff: Option<f32>,
    options: GraphOptions,
    rank_ties: Option<&Py<PyAny>>,
) -> PyResult<CooGraph> {
    let (n_training_rows, n_query_rows, n_features) = shape;
    let cutoff = cutoff
        .or_else(|| metric.default_disconnection_distance())
        .unwrap_or(f32::INFINITY);
    py.detach(|| {
        let mut callback_error = None;
        let graph = if let Some(rank_ties) = rank_ties {
            transform_dense_graph_with_tie_ranking(
                training,
                n_training_rows,
                queries,
                n_query_rows,
                n_features,
                k,
                metric,
                cutoff,
                options,
                |distances, width| {
                    python_rank_ties(rank_ties, distances, width, &mut callback_error)
                },
            )
        } else {
            transform_dense_graph_with_cutoff(
                training,
                n_training_rows,
                queries,
                n_query_rows,
                n_features,
                k,
                metric,
                cutoff,
                options,
            )
        };
        if let Some(error) = callback_error {
            return Err(error);
        }
        graph.map_err(|error| PyValueError::new_err(error.to_string()))
    })
}

fn python_rank_ties<T>(
    rank_ties: &Py<PyAny>,
    distances: &[T],
    k: usize,
    callback_error: &mut Option<PyErr>,
) -> Vec<usize>
where
    T: Copy + for<'py> IntoPyObject<'py>,
{
    let ranked = Python::attach(|py| {
        rank_ties
            .call1(py, (distances.to_vec(), k))?
            .extract::<Vec<usize>>(py)
    });
    match ranked {
        Ok(columns) => columns,
        Err(error) => {
            *callback_error = Some(error);
            Vec::new()
        }
    }
}

fn python_sparse_rank_ties(
    rank_ties: &Py<PyAny>,
    distances: SparseTieDistances<'_>,
    k: usize,
    callback_error: &mut Option<PyErr>,
) -> Vec<usize> {
    match distances {
        SparseTieDistances::Float32(values) => {
            python_rank_ties(rank_ties, values, k, callback_error)
        }
        SparseTieDistances::Float64(values) => {
            python_rank_ties(rank_ties, values, k, callback_error)
        }
    }
}

fn csr_features((shape, indptr, indices, data): &CsrComponents) -> CsrFeatures<'_> {
    CsrFeatures {
        shape: *shape,
        indptr,
        indices,
        data,
    }
}

/// Group canonical feature rows in the public sparse unique order.
#[pyfunction]
fn _unique_csr_rows(py: Python<'_>, matrix: CsrComponents) -> PyResult<RowGroups> {
    let groups = py
        .detach(|| unique_csr_rows(csr_features(&matrix)))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((groups.first, groups.inverse, groups.counts))
}

/// Return only the selected query columns and distances from canonical CSR.
#[pyfunction]
fn _rank_sparse_feature_queries(
    py: Python<'_>,
    training: CsrComponents,
    queries: CsrComponents,
    k: usize,
    metric: &str,
    rank_ties: Py<PyAny>,
) -> PyResult<(Vec<i32>, Vec<f32>)> {
    let metric = parse_metric(metric)?;
    py.detach(|| {
        let mut callback_error = None;
        let result = rank_sparse_feature_queries(
            csr_features(&training),
            csr_features(&queries),
            k,
            metric,
            |distances, width| {
                python_sparse_rank_ties(&rank_ties, distances, width, &mut callback_error)
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        let ranks = result.map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut columns = Vec::with_capacity(ranks.len());
        let mut distances = Vec::with_capacity(ranks.len());
        for rank in ranks {
            columns.push(i32::try_from(rank.index).map_err(|_| {
                PyValueError::new_err("sparse query column does not fit in a signed 32-bit index")
            })?);
            distances.push(rank.distance);
        }
        Ok((columns, distances))
    })
}

fn ranked_query_graph(
    py: Python<'_>,
    columns: Option<&[i32]>,
    distances: &[f32],
    (n_query_rows, n_training_rows, k): (usize, usize, usize),
    cutoff: f32,
    options: GraphOptions,
    packed: bool,
) -> PyResult<CooGraph> {
    let columns = columns.ok_or_else(|| PyValueError::new_err("expected ranked query columns"))?;
    py.detach(|| {
        let packed_columns = (packed && distances.contains(&f32::INFINITY)).then(|| {
            columns
                .iter()
                .enumerate()
                .map(|(i, &column)| {
                    if distances.get(i) == Some(&f32::INFINITY) {
                        -1
                    } else {
                        column
                    }
                })
                .collect::<Vec<_>>()
        });
        let preserve_infinite = packed && cutoff.is_nan();
        let mut graph = transform_ranked_neighbors_graph(
            packed_columns.as_deref().unwrap_or(columns),
            distances,
            n_query_rows,
            n_training_rows,
            k,
            k,
            if preserve_infinite {
                f32::INFINITY
            } else {
                cutoff
            },
            options,
        )?;
        if preserve_infinite {
            // A NaN threshold leaves disjoint ranks stored at their original
            // coordinates with zero membership.
            for (rank, (&column, &distance)) in columns.iter().zip(distances).enumerate() {
                if distance == f32::INFINITY
                    && let Ok(column) = usize::try_from(column)
                    && column < n_training_rows
                {
                    graph.row[rank] = rank / k;
                    graph.col[rank] = column;
                }
            }
        }
        Ok::<_, umap_core::transform::RankedQueryGraphError>(graph)
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Transform held-out features while keeping the fitted coordinates fixed.
#[pyfunction]
#[pyo3(signature = (
    training, n_training_rows, queries, n_query_rows, n_features, fitted_embedding,
    *, k, metric, n_components, a, b, learning_rate, fit_epochs, repulsion_strength,
    negative_sample_rate, transform_seed, disconnection_distance=None, local_connectivity=1.0,
    ranked_columns=None, rank_ties=None, partial_fit=false, invalid_epochs=false
))]
#[allow(clippy::too_many_arguments)]
fn _transform_dense_embedding(
    py: Python<'_>,
    training: Vec<f32>,
    n_training_rows: usize,
    queries: Vec<f32>,
    n_query_rows: usize,
    n_features: usize,
    fitted_embedding: Vec<f32>,
    k: usize,
    metric: &str,
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    fit_epochs: Option<usize>,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    transform_seed: u32,
    disconnection_distance: Option<f32>,
    local_connectivity: f64,
    ranked_columns: Option<Vec<i32>>,
    rank_ties: Option<Py<PyAny>>,
    partial_fit: bool,
    invalid_epochs: bool,
) -> PyResult<Vec<f32>> {
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let precomputed = metric == "precomputed";
    let config = DenseEmbeddingTransformConfig {
        n_neighbors: k,
        metric: if precomputed || (ranked_columns.is_some() && is_packed_metric(metric)) {
            DistanceMetric::Euclidean
        } else {
            parse_metric(metric)?
        },
        n_components,
        a,
        b,
        learning_rate,
        fit_epochs,
        repulsion_strength,
        negative_sample_rate,
        transform_seed,
        disconnection_distance,
    };
    let ranked_graph = if precomputed || ranked_columns.is_some() {
        let graph = ranked_query_graph(
            py,
            ranked_columns.as_deref(),
            &queries,
            (n_query_rows, n_training_rows, k),
            disconnection_distance.unwrap_or(f32::INFINITY),
            options,
            is_packed_metric(metric),
        )?;
        Some(graph)
    } else if let Some(rank_ties) = rank_ties.as_ref() {
        let graph = feature_query_graph(
            py,
            &training,
            &queries,
            (n_training_rows, n_query_rows, n_features),
            k,
            config.metric,
            disconnection_distance,
            options,
            Some(rank_ties),
        )?;
        Some(graph)
    } else {
        None
    };
    if let Some(graph) = ranked_graph {
        let retained_coordinates = if partial_fit {
            let coordinates = py
                .detach(|| {
                    transform_embedding_from_partial_graph(&graph, &fitted_embedding, &config)
                })
                .map_err(|error| match error {
                    DenseEmbeddingTransformError::Initialization(
                        TransformInitializationError::MissingFittedRow { .. },
                    ) => PyIndexError::new_err(error.to_string()),
                    _ => PyValueError::new_err(error.to_string()),
                })?;
            if invalid_epochs {
                return Err(PyValueError::new_err("invalid transform epoch count"));
            }
            Some(coordinates)
        } else {
            None
        };
        if !precomputed && fit_epochs.is_some_and(|epochs| epochs < 3) {
            let message = if graph.data.iter().any(|&weight| weight > 0.0) {
                c"divide by zero encountered in scalar divide"
            } else {
                c"invalid value encountered in scalar divide"
            };
            PyErr::warn(py, py.get_type::<PyRuntimeWarning>().as_any(), message, 2)?;
        }
        if let Some(coordinates) = retained_coordinates {
            return Ok(coordinates);
        }
        return py
            .detach(|| transform_embedding_from_graph(&graph, &fitted_embedding, &config))
            .map_err(|error| PyValueError::new_err(error.to_string()));
    }
    py.detach(|| {
        transform_dense_embedding_with_options(
            &training,
            n_training_rows,
            &queries,
            n_query_rows,
            n_features,
            &fitted_embedding,
            &config,
            options,
        )
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Reconstruct features using the fitted data and the query locations in its layout.
#[pyfunction]
#[pyo3(signature = (
    training, fitted_embedding, queries, simplices, query_simplices, *,
    n_features, n_components, k, a, b, learning_rate, fit_epochs,
    repulsion_strength, negative_sample_rate, transform_seed, local_connectivity,
    use_computed_scales=true, metric="euclidean", cutoff=None, fitted_distances=None, unique_fit=None
))]
#[allow(clippy::too_many_arguments)]
fn _inverse_transform_dense_embedding(
    py: Python<'_>,
    training: Vec<f32>,
    fitted_embedding: Vec<f32>,
    queries: Vec<f32>,
    simplices: Vec<usize>,
    query_simplices: Vec<Option<usize>>,
    n_features: usize,
    n_components: usize,
    k: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    fit_epochs: Option<usize>,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    transform_seed: u32,
    local_connectivity: f64,
    use_computed_scales: bool,
    metric: &str,
    cutoff: Option<f32>,
    fitted_distances: Option<(Vec<f32>, usize, usize)>,
    unique_fit: Option<(Vec<f32>, Vec<usize>)>,
) -> PyResult<Vec<f32>> {
    let metric = parse_metric(metric)?;
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let layout = DenseInverseLayout {
        n_epochs: dense_inverse_epochs(fit_epochs, query_simplices.len()),
        initial_alpha: learning_rate / 4.0,
        repulsion_strength,
        negative_sample_rate,
        seed: transform_seed,
    };
    let needs_scales = layout.n_epochs > 1 && layout.initial_alpha != 0.0;
    if needs_scales && !use_computed_scales && fitted_distances.is_none() {
        return Err(PyNotImplementedError::new_err(
            "Inverse optimization with supplied neighbors or a finite cutoff requires fitted neighbor scales.",
        ));
    }
    let input = DenseInverseInput {
        training: &training,
        embedding: &fitted_embedding,
        queries: &queries,
        n_features,
        n_components,
        triangulation: InverseTriangulation {
            simplices: &simplices,
            query_simplices: &query_simplices,
        },
    };
    py.detach(|| {
        let scales = if needs_scales {
            if let Some((distances, width, target)) = &fitted_distances {
                fit_inverse_scales_from_ranked_distances(distances, *width, *target, options)?
            } else if let Some((distinct, inverse)) = &unique_fit {
                let mut fitted = fit_dense_inverse_scales_with_metric(
                    distinct, n_features, k, metric, cutoff, options,
                )?;
                if training.len().checked_div(n_features) != Some(inverse.len())
                    || inverse.len() < fitted.len()
                    || inverse.iter().any(|&row| row >= fitted.len())
                {
                    return Err(DenseInverseError::LocalScales);
                }
                // The reference indexes the fitted scales in distinct-row order.
                // Original rows past that prefix use the scale of their representative.
                for &row in &inverse[fitted.len()..] {
                    fitted.push(fitted[row]);
                }
                fitted
            } else {
                fit_dense_inverse_scales_with_metric(
                    &training, n_features, k, metric, cutoff, options,
                )?
            }
        } else {
            Vec::new()
        };
        inverse_transform_dense_with_metric(&input, &scales, a, b, layout, metric)
            .map(|result| result.coordinates)
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_initialize_csr_transform, module)?)?;
    module.add_function(wrap_pyfunction!(_initialize_ranked_transform, module)?)?;
    module.add_function(wrap_pyfunction!(_trustworthiness_ranked, module)?)?;
    module.add_function(wrap_pyfunction!(_trustworthiness_distances, module)?)?;
    module.add_function(wrap_pyfunction!(_trustworthiness_dense, module)?)?;
    module.add("__version__", umap_core::VERSION)?;
    module.add_function(wrap_pyfunction!(_smooth_knn_distances, module)?)?;
    module.add_function(wrap_pyfunction!(_compute_knn_memberships, module)?)?;
    module.add_function(wrap_pyfunction!(_unique_csr_rows, module)?)?;
    module.add_function(wrap_pyfunction!(_combine_fuzzy_graphs, module)?)?;
    module.add_function(wrap_pyfunction!(_combine_memberships, module)?)?;
    module.add_function(wrap_pyfunction!(_reset_membership_scales, module)?)?;
    module.add_function(wrap_pyfunction!(_attenuate_categorical_edges, module)?)?;
    module.add_function(wrap_pyfunction!(_attenuate_metric_edges, module)?)?;
    module.add_function(wrap_pyfunction!(_discrete_count_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_discrete_edit_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_discrete_magnitude_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_support_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_numeric_row_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_feature_metric_distance, module)?)?;
    module.add_function(wrap_pyfunction!(_rank_configured_features, module)?)?;
    module.add_class::<NativeApproximateIndex>()?;
    module.add_function(wrap_pyfunction!(_optimize_aligned_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_prepare_parametric_edges, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph_with_cutoff, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_packed_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_packed_update_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_rank_packed_queries, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_sparse_graph, module)?)?;
    module.add_function(wrap_pyfunction!(
        _fit_sparse_feature_graph_with_cutoff,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_update_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_sparse_update_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_update_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_update_ranked_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_density_radius_log_inputs, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_graph_embedding_with_rng, module)?)?;
    module.add_function(wrap_pyfunction!(_initialize_modern_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_modern_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_recursive_part_labels, module)?)?;
    module.add_function(wrap_pyfunction!(_recursive_hub_order, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_categorical_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_continuous_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_apply_categorical_target, module)?)?;
    module.add_function(wrap_pyfunction!(_apply_continuous_target, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_precomputed_graph, module)?)?;
    module.add_function(wrap_pyfunction!(
        _fit_precomputed_graph_with_cutoff,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_fit_supplied_neighbors_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_rank_sparse_feature_queries, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(
        _inverse_transform_dense_embedding,
        module
    )?)
}
