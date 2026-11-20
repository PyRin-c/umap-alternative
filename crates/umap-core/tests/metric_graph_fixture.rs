//! Exact-bit metric graph checks against the public Windows fixtures.
//!
//! Each case is exercised through the same public core path as callers use:
//! dense input -> metric nearest neighbors -> fuzzy graph -> CSR comparison.

use serde::Deserialize;
use umap_core::fit::fit_dense_graph;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::DistanceMetric;

const METRIC_GRAPH_WINDOWS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-metric-graph-windows-x86_64.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    input: Input,
    ordering_witness: OrderingWitness,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Input {
    dtype: String,
    shape: [usize; 2],
    values: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct OrderingWitness {
    row: usize,
    orders: std::collections::BTreeMap<String, OrderingCase>,
}

#[derive(Debug, Deserialize)]
struct OrderingCase {
    columns_by_including_self: Vec<usize>,
    distances_float32: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Case {
    metric: String,
    parameters: Parameters,
    expected: ExpectedGraph,
}

#[derive(Debug, Deserialize)]
struct Parameters {
    n_neighbors: usize,
}

#[derive(Debug, Deserialize)]
struct ExpectedGraph {
    format: String,
    dtype: String,
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

fn parse_fixture() -> Fixture {
    serde_json::from_str(METRIC_GRAPH_WINDOWS).expect("metric graph fixture must be valid JSON")
}

fn metric(name: &str) -> DistanceMetric {
    match name {
        "manhattan" => DistanceMetric::Manhattan,
        "chebyshev" => DistanceMetric::Chebyshev,
        other => panic!("fixture has unsupported metric {other:?}"),
    }
}

fn graph_for_case(fixture: &Fixture, case: &Case) -> CsrGraph {
    let [n_rows, n_features] = fixture.input.shape;
    let data = fixture
        .input
        .values
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    fit_dense_graph(
        &data,
        n_rows,
        n_features,
        case.parameters.n_neighbors,
        metric(&case.metric),
    )
    .expect("fixture input must produce a valid fitted metric graph")
}

fn assert_graph_exact(actual: &CsrGraph, expected: &ExpectedGraph, case: &str) {
    assert_eq!(expected.format, "csr", "{case}: format");
    assert_eq!(expected.dtype, "float32", "{case}: dtype");
    assert_eq!(actual.n_rows, expected.shape[0], "{case}: row shape");
    assert_eq!(actual.n_cols, expected.shape[1], "{case}: column shape");
    assert_eq!(actual.indptr, expected.indptr, "{case}: indptr");
    assert_eq!(actual.indices, expected.indices, "{case}: indices");
    assert_eq!(
        actual.data.len(),
        expected.data_bits_uint32.len(),
        "{case}: data length"
    );

    let mismatches = actual
        .data
        .iter()
        .zip(&expected.data_bits_uint32)
        .enumerate()
        .filter_map(|(position, (actual, expected))| {
            let actual_bits = actual.to_bits();
            (actual_bits != *expected).then_some((position, *expected, actual_bits))
        })
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{case}: {} data bit mismatches: {mismatches:?}",
        mismatches.len()
    );
}

#[test]
fn metric_graph_cases_match_public_windows_fixture() {
    let fixture = parse_fixture();
    assert_eq!(fixture.input.dtype, "float32");
    assert_eq!(fixture.input.values.len(), fixture.input.shape[0]);
    assert!(
        fixture
            .input
            .values
            .iter()
            .all(|row| row.len() == fixture.input.shape[1])
    );

    let witness = fixture
        .ordering_witness
        .orders
        .get("manhattan")
        .expect("fixture must include Manhattan ordering witness");
    assert_eq!(fixture.ordering_witness.row, 0);
    assert_eq!(witness.columns_by_including_self[0], 0);
    assert_eq!(witness.distances_float32[0].to_bits(), 0.0_f32.to_bits());

    for case in &fixture.cases {
        let actual = graph_for_case(&fixture, case);
        assert_graph_exact(&actual, &case.expected, &case.metric);
    }
}
