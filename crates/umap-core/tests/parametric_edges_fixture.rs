//! Public Windows CPU parametric edge-pool probes.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fit::fit_dense_graph_with_options;
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};
use umap_core::neighbors::DistanceMetric;
use umap_core::parametric::{ParametricEdge, ParametricEdgeError, prepare_parametric_edges};

const FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-parametric-stages-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    edge_periods: BTreeMap<String, ReferencePeriod>,
}

#[derive(Deserialize)]
struct ReferencePeriod {
    n_rows: usize,
    resolved_epochs: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weight_bits: Vec<u32>,
    input_bits: Vec<u32>,
    graph_options: BTreeMap<String, f64>,
    observed_counts: Vec<usize>,
    total_repeats: usize,
}

fn assert_observed_repeats(graph: &CsrGraph, period: &ReferencePeriod, label: &str) {
    let expected: Vec<_> = period
        .indptr
        .windows(2)
        .enumerate()
        .flat_map(|(head, offsets)| {
            (offsets[0]..offsets[1]).filter_map(move |position| {
                let repeats = period.observed_counts[position];
                (repeats > 0).then(|| ParametricEdge {
                    head,
                    tail: period.indices[position],
                    repeats,
                })
            })
        })
        .collect();
    let actual = prepare_parametric_edges(graph, period.resolved_epochs).unwrap();
    assert_eq!(actual.edges(), expected, "{label}");
    assert_eq!(actual.total_repeats(), period.total_repeats, "{label}");
}

#[test]
fn prepares_exact_repeat_counts_from_public_csr_memberships() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    for (label, period) in fixture.edge_periods {
        let graph = CsrGraph {
            n_rows: period.n_rows,
            n_cols: period.n_rows,
            indptr: period.indptr.clone(),
            indices: period.indices.clone(),
            data: period
                .weight_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect(),
        };
        assert_observed_repeats(&graph, &period, &label);
    }
}

#[test]
fn native_graphs_produce_the_same_directed_pools() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    for (label, period) in fixture.edge_periods {
        let inputs: Vec<_> = period
            .input_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        let options = if period.graph_options.is_empty() {
            None
        } else {
            Some(
                GraphOptions::new(
                    period.graph_options["local_connectivity"],
                    period.graph_options["set_op_mix_ratio"],
                )
                .unwrap(),
            )
        };
        let graph = fit_dense_graph_with_options(
            &inputs,
            period.n_rows,
            3,
            4,
            DistanceMetric::Euclidean,
            options,
        )
        .unwrap();
        assert_eq!(graph.indptr, period.indptr, "{label}");
        assert_eq!(graph.indices, period.indices, "{label}");
        let exact_bits = graph
            .data
            .iter()
            .zip(&period.weight_bits)
            .filter(|(actual, expected)| actual.to_bits() == **expected)
            .count();
        let max_abs = graph
            .data
            .iter()
            .zip(&period.weight_bits)
            .map(|(actual, expected)| (actual - f32::from_bits(*expected)).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "{label} {exact_bits}/{} graph weight bits, max abs {max_abs:e}",
            graph.data.len()
        );
        assert_observed_repeats(&graph, &period, &label);
    }
}

#[test]
fn validates_csr_and_checks_repeat_count_overflows() {
    let mut graph = CsrGraph {
        n_rows: 2,
        n_cols: 2,
        indptr: vec![0, 1, 2],
        indices: vec![1, 0],
        data: vec![0.7, 0.0],
    };
    assert_eq!(
        prepare_parametric_edges(&graph, 10)
            .unwrap()
            .total_repeats(),
        7
    );
    assert_eq!(
        prepare_parametric_edges(&graph, 0).unwrap().total_repeats(),
        0
    );

    graph.data[0] = f32::NAN;
    assert!(matches!(
        prepare_parametric_edges(&graph, 10),
        Err(ParametricEdgeError::InvalidWeight { position: 0, .. })
    ));
    graph.data = vec![1.0, 1.0];
    for epochs in [usize::MAX, usize::MAX / 2 + 1] {
        assert_eq!(
            prepare_parametric_edges(&graph, epochs),
            Err(ParametricEdgeError::RepeatCountOverflow)
        );
    }
    graph.indptr[1] = 3;
    assert_eq!(
        prepare_parametric_edges(&graph, 10),
        Err(ParametricEdgeError::InvalidCsrRow { row: 0 })
    );
    graph.indptr[1] = 1;
    graph.indptr[2] = 1;
    assert_eq!(
        prepare_parametric_edges(&graph, 10),
        Err(ParametricEdgeError::InvalidCsrShape)
    );
}
