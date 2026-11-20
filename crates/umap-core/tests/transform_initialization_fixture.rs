//! Held-out initialization from public umap-learn 0.6.0 checkpoints.

use serde_json::Value;
use umap_core::initialization::normalize_columns;
use umap_core::neighbors::DistanceMetric;
use umap_core::transform::{
    CooGraph, TransformInitializationError, initialize_transform_embedding, transform_dense_graph,
};

const TRANSFORM_FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json");

fn from_bits(bits: impl IntoIterator<Item = u32>) -> Vec<f32> {
    bits.into_iter().map(f32::from_bits).collect()
}

fn to_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn fixture_matrix(matrix: &Value) -> Vec<f32> {
    from_bits(
        matrix["bits_uint32"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|bit| u32::try_from(bit.as_u64().unwrap()).unwrap())
            }),
    )
}

#[test]
fn fixed_fitted_checkpoints_match_public_transform_before_the_first_update() {
    let fixture: Value = serde_json::from_str(TRANSFORM_FIXTURE).unwrap();
    let training = fixture_matrix(&fixture["inputs"]["training"]);
    let mut queries = fixture_matrix(&fixture["inputs"]["transform_queries"]);
    queries.extend_from_slice(&training[..3]);
    queries.extend_from_within(..3);
    let graph = transform_dense_graph(&training, 8, &queries, 5, 3, 4, DistanceMetric::Euclidean)
        .expect("transform graph for the fixed checkpoints");

    // Public fitted arrays and transform outputs, with 0 and 5 fit epochs.
    let checkpoints = [
        (
            [
                1082885031, 1092368893, 1091318821, 1085147600, 1070958481, 0, 1055685941,
                1091279779, 1088737803, 1087966787, 0, 1092616192, 1092616192, 1060190299,
                1073624946, 1051487427,
            ],
            [
                1085326490, 1085369786, 1082927083, 1087338961, 1076855753, 1079624818, 1082885031,
                1092368893, 1085326490, 1085369786,
            ],
        ),
        (
            [
                1089778821, 1086923179, 1086241337, 1088685798, 1072902951, 1066226403, 1071877395,
                1090635989, 1083452532, 1089385309, 1084934524, 1091356538, 1091275402, 1066801789,
                3209908165, 3223379271,
            ],
            [
                1084385724, 1085338957, 1084675581, 1086941435, 1071322425, 1077655906, 1089778821,
                1086923179, 1084385724, 1085338957,
            ],
        ),
    ];
    for (fitted, expected) in checkpoints {
        let actual = initialize_transform_embedding(&graph, &from_bits(fitted), 2).unwrap();
        assert_eq!(to_bits(&actual), expected);
    }
}

fn modular_checkpoint(n_rows: usize) -> Vec<f32> {
    let mut coordinates: Vec<f32> = (0..n_rows)
        .flat_map(|i| {
            [
                ((i * 17) % 31) as f32,
                ((i * 23) % 37) as f32,
                ((i * 41) % 47) as f32,
            ]
        })
        .collect();
    // These explicit fitted checkpoints match the public zero-epoch fits.
    normalize_columns(&mut coordinates, n_rows, 3).unwrap();
    coordinates
}

#[test]
fn wider_graphs_match_public_initialization_across_reduction_boundaries() {
    let cases = [
        (
            11,
            9,
            [
                1084013995, 1081031878, 1083984043, 1082656691, 1084107089, 1085066549, 1085584681,
                1086275322, 1086620236,
            ],
        ),
        (
            30,
            18,
            [
                1082379880, 1080894237, 1083983504, 1083505969, 1086403325, 1084591686, 1085873910,
                1084208520, 1085714298,
            ],
        ),
        (
            140,
            130,
            [
                1083863800, 1083553458, 1082212286, 1080683287, 1085675866, 1084398866, 1086449908,
                1082958711, 1084764494,
            ],
        ),
        (
            271,
            258,
            [
                1083507798, 1083508374, 1082506763, 1084640538, 1085519332, 1084533533, 1084725269,
                1082980802, 1083729781,
            ],
        ),
    ];
    for (n_rows, k, expected) in cases {
        let mut training: Vec<usize> = (0..n_rows).collect();
        if n_rows == 11 {
            for x in &mut training {
                *x = (15 * *x + 6) % n_rows;
            }
        } else {
            training.sort_unstable_by_key(|x| (x * 19) % (n_rows + 3));
        }
        let training: Vec<f32> = training.into_iter().map(|x| x as f32).collect();
        let queries = [1.25, n_rows as f32 / 2.0 + 0.125, n_rows as f32 - 2.1875];
        let graph = transform_dense_graph(
            &training,
            n_rows,
            &queries,
            queries.len(),
            1,
            k,
            DistanceMetric::Euclidean,
        )
        .unwrap();
        let actual =
            initialize_transform_embedding(&graph, &modular_checkpoint(n_rows), 3).unwrap();
        assert_eq!(to_bits(&actual), expected, "k={k}");
    }
}

#[test]
fn smallest_unit_membership_column_wins_even_for_a_different_exact_match() {
    let training = [2e-8, 1.0, 0.0, 2.0, 4.0, 5.0, 6.0, 8.0];
    let queries = [0.0, 2e-8, 1e-8, 3e-8, 2e-7];
    let graph =
        transform_dense_graph(&training, 8, &queries, 5, 1, 4, DistanceMetric::Euclidean).unwrap();
    assert_eq!(&graph.col[..2], &[2, 0]);
    assert_eq!(&graph.data[..2], &[1.0, 1.0]);
    assert!(graph.data[16..].iter().all(|&weight| weight < 1.0));
    let actual = initialize_transform_embedding(&graph, &modular_checkpoint(8), 3).unwrap();
    assert_eq!(
        to_bits(&actual),
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1076723982, 1080836130, 1085918320
        ],
    );
}

#[test]
fn skips_underflowed_memberships_before_reducing_the_row() {
    let training = [
        0.00011, 0.3, 0.00015, 0.00017, 0.4, 0.00019, 0.00023, 0.00029, 0.00031, 0.5, 0.00041,
        0.00043, 1.0, 2.0, 3.0, 4.0,
    ];
    let queries = [
        -0.00018, -0.000234, -0.000345, -0.00019, -0.0002, -0.00025, -0.00026, -0.00029,
    ];
    let graph = transform_dense_graph(&training, 16, &queries, 8, 1, 12, DistanceMetric::Euclidean)
        .unwrap();
    for weights in graph.data.chunks_exact(12) {
        assert_eq!(&weights[9..], &[0.0, 0.0, 0.0]);
    }
    let actual = initialize_transform_embedding(&graph, &modular_checkpoint(16), 3).unwrap();
    assert_eq!(
        to_bits(&actual),
        [
            1082378261, 1083544056, 1083640489, 1082397719, 1083606398, 1083692304, 1082423978,
            1083698556, 1083770133, 1082382342, 1083556780, 1083651011, 1082386183, 1083568924,
            1083661079, 1082402448, 1083622241, 1083705578, 1082405210, 1083631630, 1083713465,
            1082412708, 1083657684, 1083735432,
        ],
    );
}

#[test]
fn empty_rows_and_disconnected_fitted_coordinates_are_supported() {
    // Stored zero placeholders can appear between positive entries in
    // graph-mode output when the reference disconnection threshold is applied.
    let graph = CooGraph {
        n_rows: 5,
        n_cols: 3,
        row: vec![0, 1, 0, 1, 2, 3, 3],
        col: vec![0, 2, 0, 1, 0, 1, 0],
        data: vec![0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.5],
    };
    let fitted = [f32::NAN, f32::NAN, 3.0, 8.0, 5.0, 2.0];
    let actual = initialize_transform_embedding(&graph, &fitted, 2).unwrap();
    assert_eq!(actual.len(), 10);
    assert_eq!(actual[2..4], [4.0, 5.0]);
    assert!(actual[..2].iter().chain(&actual[4..]).all(|v| v.is_nan()));
    let empty = CooGraph {
        n_rows: 0,
        row: vec![],
        col: vec![],
        data: vec![],
        ..graph
    };
    assert!(
        initialize_transform_embedding(&empty, &fitted, 2)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn malformed_graph_and_fitted_shape_return_errors() {
    let mut graph = CooGraph {
        n_rows: 1,
        n_cols: 2,
        row: vec![0, 0],
        col: vec![1, 1],
        data: vec![0.5, 0.5],
    };
    assert!(matches!(
        initialize_transform_embedding(&graph, &[0.0], 1),
        Err(TransformInitializationError::InvalidFittedShape { .. })
    ));
    assert_eq!(
        initialize_transform_embedding(&graph, &[0.0, 1.0], 1),
        Err(TransformInitializationError::RepeatedEdge { row: 0, column: 1 })
    );
    graph.col[1] = 2;
    assert!(matches!(
        initialize_transform_embedding(&graph, &[0.0, 1.0], 1),
        Err(TransformInitializationError::InvalidGraphIndex { position: 1, .. })
    ));
    graph.col[1] = 0;
    graph.data[1] = f32::NAN;
    assert!(matches!(
        initialize_transform_embedding(&graph, &[0.0, 1.0], 1),
        Err(TransformInitializationError::InvalidMembership { position: 1, .. })
    ));
}
