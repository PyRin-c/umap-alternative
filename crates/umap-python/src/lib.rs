use pyo3::exceptions::{PyNotImplementedError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use umap_core::aligned::{AlignedLayoutConfig, optimize_aligned_layout};
use umap_core::density::{
    DensityError, OriginalRadiusInput, embedding_radius_log_inputs_euclidean,
    original_radius_log_inputs,
};
use umap_core::densmap::{DensmapOptions, fit_graph_with_checkpoints};
use umap_core::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingInit, EmbeddingLayoutConfig,
    fit_dense_embedding_seeded_with_graph_options, fit_graph_embedding_with_rng,
};
use umap_core::fit::{
    DenseContinuousTarget, fit_dense_categorical_graph_with_diagnostics,
    fit_dense_continuous_graph, fit_dense_graph_with_cutoff, fit_dense_graph_with_options,
    fit_precomputed_graph_with_cutoff, fit_precomputed_graph_with_options,
    fit_sparse_precomputed_ranked_graph, fit_supplied_neighbors_graph,
};
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};
use umap_core::inverse::{
    DenseInverseInput, DenseInverseLayout, InverseTriangulation, dense_inverse_epochs,
    fit_dense_inverse_scales_with_options, inverse_transform_dense,
};
use umap_core::modern_initialization::{
    ModernInit, ModernInitConfig, ModernInitRescale, initialize_modern_layout,
};
use umap_core::neighbors::DistanceMetric;
use umap_core::parametric::prepare_parametric_edges;
use umap_core::rng::Mt19937;
use umap_core::sparse::{fit_sparse_euclidean_graph_with_cutoff, fit_sparse_graph_with_options};
use umap_core::transform::{
    CooGraph, DenseEmbeddingTransformConfig, transform_dense_embedding_with_options,
    transform_dense_graph_with_cutoff, transform_dense_graph_with_tie_ranking,
    transform_embedding_from_graph, transform_ranked_neighbors_graph,
};
use umap_core::update::{DenseUpdateConfig, DenseUpdateError, update_dense_embedding};

type CsrComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type CooComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type EmbeddingComponents = (CsrComponents, (usize, usize), Vec<f32>);
type RngState = (Vec<u32>, usize);
type GaussianRngState = (Vec<u32>, usize, Option<f64>);
type GraphEmbeddingComponents = ((usize, usize), Vec<f32>, RngState, Vec<Vec<f32>>);
type RadiusInput = (Vec<f32>, usize, String);
type DensityInput = ((f64, f64, f64), RadiusInput);
type AlignedEmbeddingComponents = (Vec<Vec<f32>>, Option<RngState>);
type ParametricEdgeComponents = (Vec<(usize, usize, usize)>, usize);

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
#[pyo3(signature = (indptr, indices, data, shape, k, cutoff, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_sparse_euclidean_graph_with_cutoff(
    py: Python<'_>,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    shape: (usize, usize),
    k: usize,
    cutoff: f32,
    graph_options: Option<(f64, f64)>,
) -> PyResult<(CsrComponents, usize)> {
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| {
        fit_sparse_euclidean_graph_with_cutoff(&indptr, &indices, &data, shape, k, cutoff, options)
    })
    .map(|(graph, removed)| (csr_components(graph), removed))
    .map_err(|error| PyValueError::new_err(error.to_string()))
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
    let (values, target_width, target_k, target_metric, target_weight) = target;
    let target = DenseContinuousTarget {
        values: &values,
        n_features: target_width,
        n_neighbors: target_k,
        metric: parse_metric(&target_metric)?,
        weight: target_weight,
    };
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
    } = if metric == "precomputed" {
        precomputed_query_graph(
            py,
            ranked_columns.as_deref(),
            &queries,
            (n_query_rows, n_training_rows),
            k,
            disconnection_distance.unwrap_or(f32::INFINITY),
            options,
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
                    let ranked = Python::attach(|py| {
                        rank_ties
                            .call1(py, (distances.to_vec(), width))?
                            .extract::<Vec<usize>>(py)
                    });
                    match ranked {
                        Ok(columns) => columns,
                        Err(error) => {
                            callback_error = Some(error);
                            Vec::new()
                        }
                    }
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

fn precomputed_query_graph(
    py: Python<'_>,
    columns: Option<&[i32]>,
    distances: &[f32],
    shape: (usize, usize),
    k: usize,
    cutoff: f32,
    options: GraphOptions,
) -> PyResult<CooGraph> {
    let columns =
        columns.ok_or_else(|| PyValueError::new_err("expected ranked precomputed columns"))?;
    py.detach(|| {
        transform_ranked_neighbors_graph(
            columns, distances, shape.0, shape.1, k, k, cutoff, options,
        )
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Transform held-out features while keeping the fitted coordinates fixed.
#[pyfunction]
#[pyo3(signature = (
    training, n_training_rows, queries, n_query_rows, n_features, fitted_embedding,
    *, k, metric, n_components, a, b, learning_rate, fit_epochs, repulsion_strength,
    negative_sample_rate, transform_seed, disconnection_distance=None, local_connectivity=1.0,
    ranked_columns=None, rank_ties=None
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
) -> PyResult<Vec<f32>> {
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let precomputed = metric == "precomputed";
    let config = DenseEmbeddingTransformConfig {
        n_neighbors: k,
        metric: if precomputed {
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
    let ranked_graph = if precomputed {
        let graph = precomputed_query_graph(
            py,
            ranked_columns.as_deref(),
            &queries,
            (n_query_rows, n_training_rows),
            k,
            disconnection_distance.unwrap_or(f32::INFINITY),
            options,
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
        if fit_epochs.is_some_and(|epochs| epochs < 3) {
            let message = if graph.data.iter().any(|&weight| weight > 0.0) {
                c"divide by zero encountered in scalar divide"
            } else {
                c"invalid value encountered in scalar divide"
            };
            PyErr::warn(py, py.get_type::<PyRuntimeWarning>().as_any(), message, 2)?;
        }
        Some(graph)
    } else {
        None
    };
    if let Some(graph) = ranked_graph {
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
    use_computed_scales=true
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
) -> PyResult<Vec<f32>> {
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
    if needs_scales && !use_computed_scales {
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
            fit_dense_inverse_scales_with_options(&training, n_features, k, options)?
        } else {
            Vec::new()
        };
        inverse_transform_dense(&input, &scales, a, b, layout).map(|result| result.coordinates)
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", umap_core::VERSION)?;
    module.add_function(wrap_pyfunction!(_optimize_aligned_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_prepare_parametric_edges, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph_with_cutoff, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_sparse_graph, module)?)?;
    module.add_function(wrap_pyfunction!(
        _fit_sparse_euclidean_graph_with_cutoff,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_update_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_density_radius_log_inputs, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_graph_embedding_with_rng, module)?)?;
    module.add_function(wrap_pyfunction!(_initialize_modern_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_categorical_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_continuous_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_precomputed_graph, module)?)?;
    module.add_function(wrap_pyfunction!(
        _fit_precomputed_graph_with_cutoff,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_fit_supplied_neighbors_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(
        _inverse_transform_dense_embedding,
        module
    )?)
}
