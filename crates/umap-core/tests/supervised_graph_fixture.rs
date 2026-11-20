//! CSR and exact-bit checks through the public dense graph fit APIs.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fit::{
    DenseCategoricalGraphFitError, DenseGraphFitError, fit_dense_categorical_graph, fit_dense_graph,
};
use umap_core::neighbors::{DistanceMetric, KnnError};

const FIXTURE_JSON: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-supervised-dense-graph-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    input: Input,
    base_parameters: Parameters,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Input {
    dtype: String,
    shape: [usize; 2],
    values: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Parameters {
    metric: String,
    n_neighbors: usize,
    target_metric: String,
}

#[derive(Deserialize)]
struct Case {
    target_weight: f64,
    y: Option<Labels>,
    expected: ExpectedGraph,
}

#[derive(Deserialize)]
struct Labels {
    dtype: String,
    shape: [usize; 1],
    values: Vec<i64>,
}

#[derive(Deserialize)]
struct ExpectedGraph {
    format: String,
    dtype: String,
    shape: [usize; 2],
    nnz: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

#[test]
fn dense_categorical_cases_match_public_csr_and_bits() {
    let fixture: Fixture = serde_json::from_str(FIXTURE_JSON).expect("valid public fixture");
    let [n_rows, n_features] = fixture.input.shape;
    assert_eq!(fixture.input.dtype, "float32");
    assert_eq!(fixture.input.values.len(), n_rows);
    assert!(
        fixture
            .input
            .values
            .iter()
            .all(|row| row.len() == n_features)
    );
    assert_eq!(fixture.base_parameters.metric, "euclidean");
    assert_eq!(fixture.base_parameters.target_metric, "categorical");

    let data: Vec<_> = fixture.input.values.into_iter().flatten().collect();
    let k = fixture.base_parameters.n_neighbors;
    let metric = DistanceMetric::Euclidean;
    let mut mismatches = Vec::new();
    for (name, case) in fixture.cases {
        let actual = match case.y {
            Some(labels) => {
                assert_eq!(labels.dtype, "int64");
                assert_eq!(labels.shape, [n_rows]);
                fit_dense_categorical_graph(
                    &data,
                    n_rows,
                    n_features,
                    k,
                    metric,
                    &labels.values,
                    case.target_weight,
                )
                .expect("valid categorical fit")
            }
            None => fit_dense_graph(&data, n_rows, n_features, k, metric)
                .expect("valid unsupervised fit"),
        };
        let expected = case.expected;
        assert_eq!(expected.format, "csr", "{name}: format");
        assert_eq!(expected.dtype, "float32", "{name}: dtype");
        assert_eq!(
            [actual.n_rows, actual.n_cols],
            expected.shape,
            "{name}: shape"
        );
        assert_eq!(actual.indptr, expected.indptr, "{name}: indptr");
        assert_eq!(actual.indices, expected.indices, "{name}: indices");
        assert_eq!(actual.data.len(), expected.nnz, "{name}: nnz");
        assert_eq!(
            actual.data.len(),
            expected.data_bits_uint32.len(),
            "{name}: bits length"
        );

        for (row, bounds) in actual.indptr.windows(2).enumerate() {
            for position in bounds[0]..bounds[1] {
                let expected_bits = expected.data_bits_uint32[position];
                let actual_bits = actual.data[position].to_bits();
                if actual_bits != expected_bits {
                    let column = actual.indices[position];
                    mismatches.push(format!(
                        "{name} ({row}, {column}) CSR[{position}]: expected 0x{expected_bits:08x}, got 0x{actual_bits:08x}"
                    ));
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn full_weight_can_leave_an_isolated_row() {
    let graph = fit_dense_categorical_graph(
        &[0.0, 1.0, 4.0],
        3,
        1,
        2,
        DistanceMetric::Euclidean,
        &[-7, -7, 8],
        1.0,
    )
    .expect("valid categorical fit");

    assert_eq!([graph.n_rows, graph.n_cols], [3, 3]);
    assert_eq!(graph.indptr, [0, 1, 2, 2]);
    assert_eq!(graph.indices, [1, 0]);
    assert_eq!(graph.data, [1.0, 1.0]);
}

#[test]
fn categorical_fit_validates_targets_and_preserves_dense_errors() {
    let metric = DistanceMetric::Euclidean;
    assert_eq!(
        fit_dense_categorical_graph(&[0.0, 1.0], 2, 1, 2, metric, &[0], 0.5),
        Err(DenseCategoricalGraphFitError::LabelCount {
            n_rows: 2,
            labels: 1
        })
    );
    for invalid in [-0.1, 1.1, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(matches!(
            fit_dense_categorical_graph(&[0.0], 1, 1, 1, metric, &[-1], invalid),
            Err(DenseCategoricalGraphFitError::InvalidTargetWeight { target_weight })
                if target_weight.to_bits() == invalid.to_bits()
        ));
    }
    assert_eq!(
        fit_dense_categorical_graph(&[0.0, 1.0], 2, 2, 2, metric, &[0, 1], 0.5),
        Err(DenseCategoricalGraphFitError::DenseGraph(
            DenseGraphFitError::Knn(KnnError::InvalidShape {
                n_rows: 2,
                n_features: 2,
                values: 2
            })
        ))
    );
}
