//! Exact-bit comparison of the pinned fuzzy-graph fixtures.
//!
//! The graph fixtures are intentionally checked through the public core API:
//! fixture input -> exhaustive neighbors -> fuzzy simplicial set.  Linux and
//! Windows reference files carry the same graph bits; keeping both files in
//! this test also catches accidental platform-specific fixture drift.

use serde::Deserialize;
use umap_core::fuzzy_graph::{CsrGraph, fuzzy_simplicial_set};
use umap_core::neighbors::exhaustive_knn;

const CONNECTED_LINUX: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-connected-dense-float32-linux-x86_64.json"
);
const CONNECTED_WINDOWS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-connected-dense-float32.json");
const TIES_LINUX: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-ties-duplicates-float32-linux-x86_64.json"
);
const TIES_WINDOWS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-ties-duplicates-float32.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    case: String,
    input: Input,
    parameters: Parameters,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Input {
    dtype: String,
    values: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct Parameters {
    n_neighbors: usize,
}

#[derive(Debug, Deserialize)]
struct Expected {
    graph: ExpectedGraph,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ExpectedGraph {
    format: String,
    dtype: String,
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

fn parse_fixture(json: &str) -> Fixture {
    serde_json::from_str(json).expect("pinned graph fixture must be valid JSON")
}

fn expected_graph(fixture: &Fixture) -> &ExpectedGraph {
    assert_eq!(fixture.input.dtype, "float32");
    assert_eq!(fixture.expected.graph.format, "csr");
    assert_eq!(fixture.expected.graph.dtype, "float32");
    &fixture.expected.graph
}

fn graph_from_fixture(fixture: &Fixture) -> CsrGraph {
    let n_rows = fixture.input.values.len();
    let n_features = fixture.input.values.first().map_or(0, Vec::len);
    assert!(
        fixture
            .input
            .values
            .iter()
            .all(|row| row.len() == n_features)
    );
    let data = fixture
        .input
        .values
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let neighbors = exhaustive_knn(&data, n_rows, n_features, fixture.parameters.n_neighbors)
        .expect("fixture input must produce valid neighbors");
    fuzzy_simplicial_set(&neighbors).expect("fixture neighbors must produce a valid graph")
}

fn assert_graph_bits(
    actual: &CsrGraph,
    expected: &ExpectedGraph,
    case: &str,
    known_residuals: &[(usize, u32, u32)],
) {
    assert_eq!(actual.n_rows, expected.shape[0], "{case}: row shape");
    assert_eq!(actual.n_cols, expected.shape[1], "{case}: column shape");
    assert_eq!(actual.indptr, expected.indptr, "{case}: indptr");
    assert_eq!(actual.indices, expected.indices, "{case}: indices");
    assert_eq!(
        actual.data.len(),
        expected.data_bits_uint32.len(),
        "{case}: data length"
    );

    for (position, (&actual, &expected_bits)) in actual
        .data
        .iter()
        .zip(&expected.data_bits_uint32)
        .enumerate()
    {
        let actual_bits = actual.to_bits();
        if actual_bits == expected_bits {
            continue;
        }

        let expected_residual = known_residuals
            .iter()
            .find(|(residual_position, _, _)| *residual_position == position);
        assert_eq!(
            expected_residual,
            Some(&(position, expected_bits, actual_bits)),
            "{case}: unexpected data[{position}] bits"
        );
    }
}

#[test]
fn connected_dense_graph_regression_matches_known_bits() {
    let linux = parse_fixture(CONNECTED_LINUX);
    let windows = parse_fixture(CONNECTED_WINDOWS);
    assert_eq!(linux.case, "connected-dense");
    assert_eq!(windows.case, linux.case);
    assert_eq!(expected_graph(&linux), expected_graph(&windows));

    let actual = graph_from_fixture(&linux);
    // The coordinate-level f32 subtraction and squaring sequence now matches
    // every public connected-dense graph bit exactly.
    assert_graph_bits(&actual, expected_graph(&linux), &linux.case, &[]);
}

#[test]
fn ties_duplicates_graph_matches_linux_and_windows_bits() {
    let linux = parse_fixture(TIES_LINUX);
    let windows = parse_fixture(TIES_WINDOWS);
    assert_eq!(linux.case, "ties-duplicates");
    assert_eq!(windows.case, linux.case);
    assert_eq!(expected_graph(&linux), expected_graph(&windows));

    let actual = graph_from_fixture(&linux);
    assert_graph_bits(&actual, expected_graph(&linux), &linux.case, &[]);
}

#[test]
fn scale_search_matches_reference_bits_at_convergence_boundary() {
    // The end rows start just inside the tolerance for log2(3).
    // Rounding the target to f32 moves their sums outside the tolerance.
    let gap = 0.0625;
    let separation = f32::from_bits(0x3f19_43c7);
    let points = [0.0, gap, separation, separation + gap];
    let neighbors = exhaustive_knn(&points, points.len(), 1, 3).expect("valid neighbors");
    let graph = fuzzy_simplicial_set(&neighbors).expect("valid graph");

    // Captured from umap-learn 0.6.0's public graph-mode fit_transform
    // with Euclidean distance, k=3, random initialization, and seed 42.
    assert_eq!((graph.n_rows, graph.n_cols), (4, 4));
    assert_eq!(graph.indptr, [0, 2, 5, 8, 10]);
    assert_eq!(graph.indices, [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]);
    let unit = 1.0_f32.to_bits();
    let one_way = 0x3f15_c0c2;
    let reciprocal = 0x3f53_e74c;
    let bits: Vec<_> = graph.data.iter().map(|weight| weight.to_bits()).collect();
    assert_eq!(
        bits,
        [
            unit, one_way, unit, reciprocal, one_way, one_way, reciprocal, unit, one_way, unit,
        ]
    );
}
