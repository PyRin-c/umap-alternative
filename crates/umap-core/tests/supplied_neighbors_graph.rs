//! Exact CSR checks for the supplied-neighbor graph stage.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fit::{SuppliedNeighborsGraphError, fit_supplied_neighbors_graph};
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};

const PUBLIC_FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-precomputed-neighbors-windows-x86_64.json"
);
const HELD_OUT_FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-supplied-neighbors-held-out-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    training: Vec<Vec<f32>>,
    base_parameters: Parameters,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Parameters {
    n_neighbors: usize,
}

#[derive(Deserialize)]
struct Case {
    overrides: Overrides,
    indices: Option<Vec<Vec<i32>>>,
    distance_bits: Option<Vec<Vec<u32>>>,
    indices_after_fit: Option<Vec<Vec<i32>>>,
    distance_bits_after_fit: Option<Vec<Vec<u32>>>,
    fit_transform: FitResult,
}

#[derive(Deserialize)]
struct Overrides {
    n_neighbors: Option<usize>,
    local_connectivity: Option<f64>,
    set_op_mix_ratio: Option<f64>,
    disconnection_distance: Option<f32>,
}

#[derive(Deserialize)]
struct FitResult {
    graph: Option<ExpectedGraph>,
}

#[derive(Deserialize)]
struct ExpectedGraph {
    format: String,
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weight_bits: Vec<u32>,
}

fn assert_exact(actual: CsrGraph, expected: &ExpectedGraph, case: &str) {
    assert_eq!(expected.format, "csr", "{case}: format");
    assert_eq!(
        [actual.n_rows, actual.n_cols],
        expected.shape,
        "{case}: shape"
    );
    assert_eq!(actual.indptr, expected.indptr, "{case}: indptr");
    assert_eq!(actual.indices, expected.indices, "{case}: indices");
    assert_eq!(
        actual.data.len(),
        expected.weight_bits.len(),
        "{case}: data length"
    );
    let mismatches = actual
        .data
        .iter()
        .map(|weight| weight.to_bits())
        .zip(&expected.weight_bits)
        .enumerate()
        .filter_map(|(position, (actual, &expected))| {
            (actual != expected).then_some((position, actual, expected))
        })
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{case}: weight bit mismatches {mismatches:?}"
    );
}

fn compare_supplied_graph(fixture: &Fixture, name: &str, case: &Case) {
    let expected = case.fit_transform.graph.as_ref().expect(name);
    let indices = case.indices.as_ref().expect(name);
    let distance_bits = case.distance_bits.as_ref().expect(name);
    let n_rows = fixture.training.len();
    let width = indices[0].len();
    let k = case
        .overrides
        .n_neighbors
        .unwrap_or(fixture.base_parameters.n_neighbors);
    let options = GraphOptions::new(
        case.overrides.local_connectivity.unwrap_or(1.0),
        case.overrides.set_op_mix_ratio.unwrap_or(1.0),
    )
    .expect(name);
    let compare = |indices: &[Vec<i32>], distance_bits: &[Vec<u32>], cutoff, label| {
        assert_eq!(indices.len(), n_rows, "{label}: indices row count");
        assert_eq!(distance_bits.len(), n_rows, "{label}: distances row count");
        assert!(
            indices.iter().all(|row| row.len() == width),
            "{label}: indices width"
        );
        assert!(
            distance_bits.iter().all(|row| row.len() == width),
            "{label}: distances width"
        );
        let indices = indices.iter().flatten().copied().collect::<Vec<_>>();
        let distances = distance_bits
            .iter()
            .flatten()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        let actual =
            fit_supplied_neighbors_graph(&indices, &distances, n_rows, width, k, options, cutoff)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_exact(actual, expected, label);
    };
    compare(
        indices,
        distance_bits,
        case.overrides
            .disconnection_distance
            .unwrap_or(f32::INFINITY),
        name,
    );
    if let (Some(indices), Some(distance_bits)) =
        (&case.indices_after_fit, &case.distance_bits_after_fit)
    {
        compare(
            indices,
            distance_bits,
            f32::INFINITY,
            &format!("{name} after cutoff"),
        );
    }
}

#[test]
fn supplied_arrays_match_public_fit() {
    let fixture: Fixture =
        serde_json::from_str(PUBLIC_FIXTURE).expect("public fixture is valid JSON");
    for (name, case) in &fixture.cases {
        // The estimator performs search for these cases or rejects the inputs.
        if matches!(
            name.as_str(),
            "computed" | "one_fewer_column" | "one_fewer_row" | "mismatched_shapes"
        ) {
            continue;
        }
        compare_supplied_graph(&fixture, name, case);
    }
}

#[test]
fn supplied_arrays_match_additional_public_cases() {
    let fixture: Fixture =
        serde_json::from_str(HELD_OUT_FIXTURE).expect("held-out fixture is valid JSON");
    for (name, case) in &fixture.cases {
        compare_supplied_graph(&fixture, name, case);
    }
}

#[test]
fn rejects_malformed_inputs_before_using_neighbors() {
    use SuppliedNeighborsGraphError as Error;

    let fit = |indices: &[i32], distances: &[f32], rows, width, k, cutoff| {
        fit_supplied_neighbors_graph(
            indices,
            distances,
            rows,
            width,
            k,
            GraphOptions::default(),
            cutoff,
        )
    };
    for (indices, distances, rows, width) in [
        (&[][..], &[][..], 0, usize::MAX),
        (&[][..], &[][..], usize::MAX, 2),
        (&[0][..], &[][..], 1, 1),
        (&[][..], &[0.0][..], 1, 1),
        (&[0][..], &[0.0][..], 2, 1),
    ] {
        assert!(matches!(
            fit(indices, distances, rows, width, 1, f32::INFINITY),
            Err(Error::InvalidShape { .. })
        ));
    }
    assert_eq!(
        fit(&[], &[], usize::MAX, 0, 1, f32::INFINITY),
        Err(Error::InvalidNeighborCount { width: 0, k: 1 })
    );
    for k in [0, 2] {
        assert_eq!(
            fit(&[0, 1], &[0.0, 0.0], 2, 1, k, f32::INFINITY),
            Err(Error::InvalidNeighborCount { width: 1, k })
        );
    }
    for (index, distance) in [
        (1, 0.0),
        (i32::MAX, 1.0),
        (-2, f32::INFINITY),
        (-1, 0.0),
        (0, f32::INFINITY),
        (0, f32::NEG_INFINITY),
        (0, f32::NAN),
        (0, -1.0),
    ] {
        assert!(matches!(
            fit(&[0, index], &[0.0, distance], 1, 2, 1, 0.0),
            Err(Error::InvalidNeighbor { row: 0, position: 1, index: actual_index, distance: actual_distance })
                if actual_index == index && actual_distance.to_bits() == distance.to_bits()
        ));
    }
    for distance in [-1.0, f32::NEG_INFINITY, f32::NAN] {
        assert!(matches!(
            fit(&[0], &[0.0], 1, 1, 1, distance),
            Err(Error::InvalidDisconnectionDistance { distance: actual }) if actual.to_bits() == distance.to_bits()
        ));
    }
}
