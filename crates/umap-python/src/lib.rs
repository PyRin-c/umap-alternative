use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use umap_core::neighbors::DistanceMetric;

type CsrComponents = ((usize, usize), Vec<usize>, Vec<usize>, Vec<f32>);

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
///
/// This is an internal bridge for the eventual Python UMAP wrapper. It
/// intentionally exposes only the graph-stage primitive, not a Python
/// estimator or any downstream layout behavior.
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
    let graph = umap_core::fit::fit_dense_graph(&data, n_rows, n_features, k, metric)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok((
        (graph.n_rows, graph.n_cols),
        graph.indptr,
        graph.indices,
        graph.data,
    ))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", umap_core::VERSION)?;
    module.add_function(wrap_pyfunction!(_fit_dense_graph, module)?)
}
