//! Held-out dense transform checks against the public Windows graph fixture.

use serde::Deserialize;
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::neighbors::{DistanceMetric, KnnError};
use umap_core::transform::{
    DenseGraphTransformError, transform_dense_graph, transform_dense_graph_with_tie_ranking,
};

const FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json");

#[test]
fn external_tie_ranking_is_used_only_when_selected_ranks_can_change() {
    let training = [0.0, 1.0, -1.0, 2.0, -3.0];
    let queries = [0.0, 0.25, 1.25];
    let graph = transform_dense_graph_with_tie_ranking(
        &training,
        5,
        &queries,
        3,
        1,
        2,
        DistanceMetric::Euclidean,
        f32::INFINITY,
        GraphOptions::default(),
        |distances, k| {
            assert_eq!(distances, [0.0, 1.0, 1.0, 2.0, 3.0]);
            assert_eq!(k, 2);
            vec![0, 2]
        },
    )
    .unwrap();
    assert_eq!(graph.col, [0, 2, 0, 1, 1, 3]);

    for invalid in [
        vec![],
        vec![0, 6],
        vec![0, 0],
        vec![2, 0],
        vec![1, 2],
        vec![0, 3],
    ] {
        assert!(matches!(
            transform_dense_graph_with_tie_ranking(
                &training,
                5,
                &queries[..1],
                1,
                1,
                2,
                DistanceMetric::Euclidean,
                f32::INFINITY,
                GraphOptions::default(),
                |_, _| invalid.clone(),
            ),
            Err(DenseGraphTransformError::InvalidRank { query: 0, .. })
        ));
    }
}

#[derive(Deserialize)]
struct Fixture {
    parameters: Parameters,
    inputs: Inputs,
    cases: Cases,
}

#[derive(Deserialize)]
struct Parameters {
    n_neighbors: usize,
    local_connectivity: f32,
    metric: String,
}

#[derive(Deserialize)]
struct Inputs {
    training: Matrix,
    transform_queries: Matrix,
}

#[derive(Deserialize)]
struct Matrix {
    shape: [usize; 2],
    bits_uint32: Vec<Vec<u32>>,
}

impl Matrix {
    fn values(&self) -> Vec<f32> {
        self.bits_uint32
            .iter()
            .flatten()
            .copied()
            .map(f32::from_bits)
            .collect()
    }
}

#[derive(Deserialize)]
struct Cases {
    graph: GraphCase,
}

#[derive(Deserialize)]
struct GraphCase {
    transform: TransformCase,
}

#[derive(Deserialize)]
struct TransformCase {
    output: ExpectedGraph,
}

#[derive(Deserialize)]
struct ExpectedGraph {
    format: String,
    dtype: String,
    shape: [usize; 2],
    nnz: usize,
    row: Vec<usize>,
    col: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

#[test]
fn held_out_graph_matches_public_coo_order_and_bits() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid transform fixture");
    let training = fixture.inputs.training;
    let queries = fixture.inputs.transform_queries;
    assert_eq!(training.shape[1], queries.shape[1]);
    assert_eq!(fixture.parameters.metric, "euclidean");
    assert_eq!(fixture.parameters.local_connectivity, 1.0);

    let graph = transform_dense_graph(
        &training.values(),
        training.shape[0],
        &queries.values(),
        queries.shape[0],
        training.shape[1],
        fixture.parameters.n_neighbors,
        DistanceMetric::Euclidean,
    )
    .expect("fixture input must produce a valid transform graph");
    let expected = fixture.cases.graph.transform.output;
    assert_eq!(expected.format, "coo");
    assert_eq!(expected.dtype, "float32");
    assert_eq!([graph.n_rows, graph.n_cols], expected.shape);
    assert_eq!(graph.row, expected.row);
    assert_eq!(graph.col, expected.col);
    assert_eq!(graph.data.len(), expected.nnz);
    assert_eq!(
        graph
            .data
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>(),
        expected.data_bits_uint32,
    );
}

#[test]
fn identical_coordinates_and_overlapping_indices_keep_their_edges() {
    let graph = transform_dense_graph(
        &[0.0, 2.0],
        2,
        &[0.0, 2.0, 1.0],
        3,
        1,
        2,
        DistanceMetric::Euclidean,
    )
    .expect("valid rectangular inputs");

    assert_eq!([graph.n_rows, graph.n_cols], [3, 2]);
    assert_eq!(graph.row, [0, 0, 1, 1, 2, 2]);
    assert_eq!(graph.col, [0, 1, 1, 0, 0, 1]);
    assert_eq!(graph.data[0], 1.0);
    assert_eq!(graph.data[2], 1.0);
    assert_eq!(graph.data[4], graph.data[5]);
    assert!(
        graph
            .data
            .iter()
            .all(|&weight| 0.0 < weight && weight <= 1.0)
    );
}

#[test]
fn duplicate_point_probe_matches_public_batch_scale_and_stored_zeros() {
    // Public UMAP graph mode with Euclidean distance and five neighbors.
    let training = [0.0, 0.0, 0.0, 0.0, 0.1, 10.0, 20.0, 21.0, 25.0, 30.0];
    let metric = DistanceMetric::Euclidean;
    let single = transform_dense_graph(&training, training.len(), &[0.0], 1, 1, 5, metric)
        .expect("valid single-query input");
    assert_eq!(single.col, [0, 1, 2, 3, 4]);
    assert_eq!(single.data, [1.0, 1.0, 1.0, 1.0, 0.0]);

    let batch = transform_dense_graph(&training, training.len(), &[0.0, 500.0], 2, 1, 5, metric)
        .expect("valid two-query input");
    assert_eq!(batch.row, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    assert_eq!(batch.col, [0, 1, 2, 3, 4, 9, 8, 7, 6, 5]);
    assert_eq!(
        batch
            .data
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>(),
        [
            1065353216, 1065353216, 1065353216, 1065353216, 1059624871, 1058436579, 1058380984,
            1058336734, 1058325703, 1058216074,
        ],
    );
}

#[test]
fn empty_queries_produce_an_empty_rectangular_graph() {
    let graph = transform_dense_graph(&[0.0, 1.0], 2, &[], 0, 1, 2, DistanceMetric::Euclidean)
        .expect("valid empty query matrix");
    assert_eq!([graph.n_rows, graph.n_cols], [0, 2]);
    assert!(graph.row.is_empty() && graph.col.is_empty() && graph.data.is_empty());
}

#[test]
fn held_out_partial_selection_orders_cutoff_ties_and_ignores_unselected_overflow() {
    let training = [f32::MAX, -1.0, 1.0, -0.0, 2.0, -2.0, -f32::MAX, 0.0, 1.0];
    let order = [3, 7, 1, 2, 8, 4, 5];
    let transform = |k| {
        transform_dense_graph(
            &training,
            training.len(),
            &[0.0],
            1,
            1,
            k,
            DistanceMetric::Euclidean,
        )
    };
    for k in 1..=order.len() {
        let graph = transform(k).expect("selected distances are finite");
        assert_eq!(graph.col, order[..k]);
        assert_eq!(graph.row, vec![0; k]);
    }
    assert_eq!(
        transform(order.len() + 1),
        Err(DenseGraphTransformError::NonFiniteDistance {
            query: 0,
            training: 0,
            distance: f32::INFINITY,
        })
    );
}

#[test]
fn validation_identifies_invalid_input_and_distance_overflow() {
    let metric = DistanceMetric::Euclidean;
    assert_eq!(
        transform_dense_graph(&[], usize::MAX, &[0.0, 0.0], 1, 2, 1, metric),
        Err(DenseGraphTransformError::Training(KnnError::InvalidShape {
            n_rows: usize::MAX,
            n_features: 2,
            values: 0,
        }))
    );
    assert_eq!(
        transform_dense_graph(&[0.0], 1, &[0.0, 1.0], 1, 1, 1, metric),
        Err(DenseGraphTransformError::Queries(KnnError::InvalidShape {
            n_rows: 1,
            n_features: 1,
            values: 2,
        }))
    );
    for k in [0, 2] {
        assert_eq!(
            transform_dense_graph(&[0.0], 1, &[0.0], 1, 1, k, metric),
            Err(DenseGraphTransformError::Training(KnnError::InvalidK {
                k,
                n_rows: 1,
            }))
        );
    }
    assert_eq!(
        transform_dense_graph(&[], 1, &[], 1, 0, 1, metric),
        Err(DenseGraphTransformError::NoFeatures)
    );
    assert_eq!(
        transform_dense_graph(&[f32::INFINITY], 1, &[0.0], 1, 1, 1, metric),
        Err(DenseGraphTransformError::Training(
            KnnError::NonFiniteValue {
                index: 0,
                value: f32::INFINITY,
            }
        ))
    );
    assert!(matches!(
        transform_dense_graph(&[0.0], 1, &[f32::NAN], 1, 1, 1, metric),
        Err(DenseGraphTransformError::Queries(KnnError::NonFiniteValue { index: 0, value }))
            if value.is_nan()
    ));
    assert_eq!(
        transform_dense_graph(&[f32::MAX], 1, &[0.0], 1, 1, 1, metric),
        Err(DenseGraphTransformError::NonFiniteDistance {
            query: 0,
            training: 0,
            distance: f32::INFINITY,
        })
    );
}
