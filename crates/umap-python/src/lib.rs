use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use umap_core::fit::{fit_dense_categorical_graph, fit_dense_graph, fit_precomputed_graph};
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::DistanceMetric;
use umap_core::transform::{CooGraph, transform_dense_graph};

type CsrComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);
type CooComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);

fn csr_components(graph: CsrGraph) -> CsrComponents {
    (
        (graph.n_rows, graph.n_cols),
        graph.indptr,
        graph.indices,
        graph.data,
    )
}

/// Parse the metric names supported by the dense graph compatibility path.
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

/// Fit a fuzzy graph from a flat row-major f32 matrix and return its CSR data.
#[pyfunction]
#[pyo3(signature = (data, n_rows, n_features, k, metric))]
fn _fit_dense_graph(
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
) -> PyResult<CsrComponents> {
    let metric = parse_metric(metric)?;
    fit_dense_graph(&data, n_rows, n_features, k, metric)
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a fuzzy graph from dense features and one categorical label per row.
#[pyfunction]
fn _fit_dense_categorical_graph(
    data: Vec<f32>,
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
    labels: Vec<i64>,
    target_weight: f64,
) -> PyResult<CsrComponents> {
    let metric = parse_metric(metric)?;
    fit_dense_categorical_graph(&data, n_rows, n_features, k, metric, &labels, target_weight)
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Fit a fuzzy graph from a flat row-major square f32 distance matrix.
#[pyfunction]
fn _fit_precomputed_graph(distances: Vec<f32>, n_rows: usize, k: usize) -> PyResult<CsrComponents> {
    fit_precomputed_graph(&distances, n_rows, k)
        .map(csr_components)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return the shape, row indices, column indices, and memberships in COO order.
#[pyfunction]
fn _transform_dense_graph(
    training: Vec<f32>,
    n_training_rows: usize,
    queries: Vec<f32>,
    n_query_rows: usize,
    n_features: usize,
    k: usize,
    metric: &str,
) -> PyResult<CooComponents> {
    let metric = parse_metric(metric)?;
    let CooGraph {
        n_rows,
        n_cols,
        row,
        col,
        data,
    } = transform_dense_graph(
        &training,
        n_training_rows,
        &queries,
        n_query_rows,
        n_features,
        k,
        metric,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(((n_rows, n_cols), row, col, data))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", umap_core::VERSION)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_dense_categorical_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_precomputed_graph, module)?)?;
    module.add_function(wrap_pyfunction!(_transform_dense_graph, module)?)
}
