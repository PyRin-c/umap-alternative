use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::transform::{RankedQueryGraphError, transform_ranked_neighbors_graph};

const FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-approximate-query-handoff-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    input: Input,
    training_shape: [usize; 2],
    query_shape: [usize; 2],
    query_rank_width: usize,
    scale_neighbors: usize,
    query: Query,
    fresh_fit_repeats_equal: bool,
}

#[derive(Deserialize)]
struct Input {
    metric: String,
    cutoff: Option<f32>,
    connectivity: f64,
    indices: Vec<Vec<i32>>,
    distances: Vec<Vec<Option<f32>>>,
}

#[derive(Deserialize)]
struct Query {
    shape: [usize; 2],
    rows: Vec<usize>,
    columns: Vec<usize>,
    bits: Vec<u32>,
}

#[test]
fn replays_public_query_handoff_in_supplied_rank_order() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    for (name, case) in fixture.cases {
        assert!(case.fresh_fit_repeats_equal, "{name}");
        let input = case.input;
        let indices: Vec<i32> = input.indices.into_iter().flatten().collect();
        let distances: Vec<f32> = input
            .distances
            .into_iter()
            .flatten()
            .map(|distance| distance.unwrap_or(f32::INFINITY))
            .collect();
        let default_cutoff = match input.metric.as_str() {
            "euclidean" => f32::INFINITY,
            "cosine" | "correlation" => 2.0,
            metric => panic!("unknown fixture metric {metric}"),
        };
        let graph = transform_ranked_neighbors_graph(
            &indices,
            &distances,
            case.query_shape[0],
            case.training_shape[0],
            case.query_rank_width,
            case.scale_neighbors,
            input.cutoff.unwrap_or(default_cutoff),
            GraphOptions::new(input.connectivity, 1.0).unwrap(),
        )
        .unwrap();
        assert_eq!([graph.n_rows, graph.n_cols], case.query.shape, "{name}");
        assert_eq!(graph.row, case.query.rows, "{name}");
        assert_eq!(graph.col, case.query.columns, "{name}");
        let bits: Vec<u32> = graph.data.into_iter().map(f32::to_bits).collect();
        assert_eq!(bits, case.query.bits, "{name}");
    }
}

#[test]
fn validates_ranked_query_shapes_and_missing_pairs() {
    let options = GraphOptions::default();
    let build = |indices, distances| {
        transform_ranked_neighbors_graph(indices, distances, 1, 4, 2, 2, f32::INFINITY, options)
    };
    assert!(matches!(
        build(&[0, 1], &[0.0]),
        Err(RankedQueryGraphError::InvalidShape { .. })
    ));
    assert!(matches!(
        build(&[0, 4], &[0.0, 1.0]),
        Err(RankedQueryGraphError::InvalidNeighbor { position: 1, .. })
    ));
    assert!(matches!(
        build(&[0, -1], &[0.0, 1.0]),
        Err(RankedQueryGraphError::InvalidNeighbor { position: 1, .. })
    ));
    assert!(matches!(
        build(&[-1, 0], &[f32::INFINITY, 0.0]),
        Err(RankedQueryGraphError::UnsortedDistances { position: 1 })
    ));
    assert!(matches!(
        transform_ranked_neighbors_graph(&[0, 1], &[0.0, 1.0], 1, 4, 2, 3, 1.0, options),
        Err(RankedQueryGraphError::InvalidDimensions { .. })
    ));
    let empty = transform_ranked_neighbors_graph(&[], &[], 0, 4, 2, 2, 1.0, options).unwrap();
    assert_eq!([empty.n_rows, empty.n_cols], [0, 4]);
    assert!(empty.data.is_empty());
}
