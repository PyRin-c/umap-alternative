use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use umap_core::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingInit, EmbeddingLayoutConfig,
    fit_dense_embedding_seeded_with_graph_options, fit_graph_embedding_with_rng,
};
use umap_core::fit::{
    fit_dense_categorical_graph_with_options, fit_dense_graph_with_options,
    fit_precomputed_graph_with_options, fit_supplied_neighbors_graph,
};
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};
use umap_core::inverse::{
    DenseInverseInput, DenseInverseLayout, InverseTriangulation, dense_inverse_epochs,
    fit_dense_inverse_scales_with_options, inverse_transform_dense,
};
use umap_core::neighbors::DistanceMetric;
use umap_core::rng::Mt19937;
use umap_core::sparse::{SparseDistanceMetric, fit_sparse_graph_with_options};
use umap_core::transform::{
    CooGraph, DenseEmbeddingTransformConfig, transform_dense_embedding_with_options,
    transform_dense_graph_with_cutoff,
};
use umap_core::update::{DenseUpdateConfig, update_dense_embedding};

type CsrComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type CooComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type EmbeddingComponents = (CsrComponents, (usize, usize), Vec<f32>);
type RngState = (Vec<u32>, usize);
type GraphEmbeddingComponents = ((usize, usize), Vec<f32>, RngState);

fn csr_components(graph: CsrGraph) -> CsrComponents {
    (
        (graph.n_rows, graph.n_cols),
        graph.indptr,
        graph.indices,
        graph.data,
    )
}

/// Parse the metric names supported by the feature graph compatibility paths.
///
/// The aliases match the names commonly used for these three distance
/// functions by the Python-facing UMAP API: L1/city-block, L2, and the
/// infinity norm. Metric names are case-insensitive at this boundary.
fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" | "cityblock" | "taxicab" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linfty" | "linfinity" | "max" => Ok(DistanceMetric::Chebyshev),
        _ => Err(PyValueError::new_err(format!(
            "unsupported metric {metric:?}; expected Euclidean, Manhattan, or Chebyshev"
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
    let metric = if metric.eq_ignore_ascii_case("cosine") {
        SparseDistanceMetric::Cosine
    } else {
        parse_metric(metric)?.into()
    };
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
    let DenseEmbedding {
        graph,
        coordinates,
        shape,
    } = py
        .detach(|| {
            update_dense_embedding(
                &data, shape.0, shape.1, n_existing, &fitted, &config, &layout,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((csr_components(graph), shape, coordinates))
}

/// Optimize raw coordinates with the same graph and MT19937 stream used for initialization.
#[pyfunction]
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
    let DenseEmbedding {
        coordinates, shape, ..
    } = py
        .detach(|| fit_graph_embedding_with_rng(graph, &initial_coordinates, &config, &mut rng))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (words, cursor) = rng.state();
    Ok((shape, coordinates, (words.to_vec(), cursor)))
}

/// Fit a fuzzy graph from dense features and one categorical label per row.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric, labels, target_weight, *, graph_options=None))]
#[allow(clippy::too_many_arguments)]
fn _fit_dense_categorical_graph(
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    labels: Vec<i64>,
    target_weight: f64,
    graph_options: Option<(f64, f64)>,
) -> PyResult<CsrComponents> {
    let metric = parse_metric(metric)?;
    fit_dense_categorical_graph_with_options(
        &data,
        n_rows,
        n_features,
        k,
        metric,
        &labels,
        target_weight,
        parse_graph_options(graph_options)?,
    )
    .map(csr_components)
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

/// Fit a fuzzy graph from row-major indices and distances after estimator masking.
#[pyfunction]
#[pyo3(signature = (indices, distances, shape, k, *, graph_options=None))]
fn _fit_supplied_neighbors_graph(
    py: Python<'_>,
    indices: Vec<i32>,
    distances: Vec<f32>,
    shape: (usize, usize),
    k: usize,
    graph_options: Option<(f64, f64)>,
) -> PyResult<CsrComponents> {
    let (n_rows, width) = shape;
    let options = parse_graph_options(graph_options)?.unwrap_or_default();
    py.detach(|| {
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
#[pyo3(signature = (training, n_training_rows, queries, n_query_rows, n_features, k, metric, *, local_connectivity=1.0, disconnection_distance=None))]
#[allow(clippy::too_many_arguments)]
fn _transform_dense_graph(
    training: Vec<f32>,
    n_training_rows: usize,
    queries: Vec<f32>,
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    local_connectivity: f64,
    disconnection_distance: Option<f32>,
) -> PyResult<CooComponents> {
    let metric = parse_metric(metric)?;
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let CooGraph {
        n_rows,
        n_cols,
        row,
        col,
        data,
    } = transform_dense_graph_with_cutoff(
        &training,
        n_training_rows,
        &queries,
        n_query_rows,
        n_features,
        k,
        metric,
        disconnection_distance.unwrap_or(f32::INFINITY),
        options,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(((n_rows, n_cols), row, col, data))
}

/// Transform held-out features while keeping the fitted coordinates fixed.
#[pyfunction]
#[pyo3(signature = (
    training, n_training_rows, queries, n_query_rows, n_features, fitted_embedding,
    *, k, metric, n_components, a, b, learning_rate, fit_epochs, repulsion_strength,
    negative_sample_rate, transform_seed, disconnection_distance=None, local_connectivity=1.0
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
) -> PyResult<Vec<f32>> {
    let options = GraphOptions::new(local_connectivity, 1.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let config = DenseEmbeddingTransformConfig {
        n_neighbors: k,
        metric: parse_metric(metric)?,
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
    module.add_function(wrap_pyfunction!(_fit_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_sparse_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_update_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_graph_embedding_with_rng, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_categorical_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_precomputed_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_supplied_neighbors_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(
        _inverse_transform_dense_embedding,
        module
    )?)
}
