//! Dense update checkpoints through graph rebuilding and initialization.

use serde_json::Value;
use umap_core::initialization::InitializationError;
use umap_core::neighbors::DistanceMetric;
use umap_core::update::{DenseUpdateConfig, DenseUpdateError, prepare_dense_update};

const FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-update-windows-x86_64.json");
const HOLDOUTS: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-update-initialization-holdouts-x86_64.json"
);

fn dimensions(record: &Value) -> (usize, usize) {
    (
        record["shape"][0].as_u64().unwrap() as usize,
        record["shape"][1].as_u64().unwrap() as usize,
    )
}

fn float32_matrix(record: &Value) -> Vec<f32> {
    let (rows, columns) = dimensions(record);
    let words: Vec<Vec<u32>> = serde_json::from_value(record["bits_uint32"].clone()).unwrap();
    assert_eq!(words.len(), rows);
    assert!(words.iter().all(|row| row.len() == columns));
    words.into_iter().flatten().map(f32::from_bits).collect()
}

#[test]
fn dense_update_initialization_and_graph_match_public_checkpoints() {
    assert_eq!(check_initialization_fixture(FIXTURE), 16);
    assert_eq!(check_initialization_fixture(HOLDOUTS), 4);
}

fn check_initialization_fixture(source: &str) -> usize {
    let fixture: Value = serde_json::from_str(source).unwrap();
    let mut matched = 0;
    for (name, case) in fixture["cases"].as_object().unwrap() {
        let parameters = |key: &str| {
            case["parameter_overrides"]
                .get(key)
                .unwrap_or(&fixture["base_parameters"][key])
        };
        if case["input_format"] != "dense"
            || parameters("n_epochs")
                .as_u64()
                .is_none_or(|epochs| epochs > 1)
            || case["fit"]["status"] != "ok"
        {
            continue;
        }
        let updates = case["updates"].as_array().unwrap();
        if updates.first().is_none_or(|step| step["status"] != "ok") {
            continue;
        }

        let input = &fixture["inputs"][case["fit_input"].as_str().unwrap()];
        let (mut n_existing, n_features) = dimensions(input);
        let mut combined = float32_matrix(input);
        let fit_embedding = &case["fit"]["state"]["embedding"];
        let mut coordinates = float32_matrix(fit_embedding);
        let config = DenseUpdateConfig {
            n_neighbors: (parameters("n_neighbors").as_u64().unwrap() as usize).min(n_existing - 1),
            n_components: dimensions(fit_embedding).1,
            metric: DistanceMetric::Euclidean,
            graph_options: None,
            transform_seed: parameters("transform_seed").as_u64().unwrap() as u32,
        };
        for (stage, update) in updates.iter().enumerate() {
            let data = &fixture["inputs"][update["input"].as_str().unwrap()];
            combined.extend(float32_matrix(data));
            let expected = float32_matrix(&update["state"]["embedding"]);
            let n_rows = dimensions(&update["state"]["embedding"]).0;
            let prepared = prepare_dense_update(
                &combined,
                n_rows,
                n_features,
                n_existing,
                &coordinates,
                &config,
            )
            .unwrap_or_else(|error| panic!("{name}, update {}: {error}", stage + 1));
            assert_eq!(prepared.shape, (n_rows, config.n_components), "{name}");
            for (entry, (actual, expected)) in prepared.coordinates.iter().zip(expected).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{name}, update {}, coordinate {entry}",
                    stage + 1,
                );
            }
            let expected_graph = &update["state"]["graph"];
            let indptr: Vec<usize> =
                serde_json::from_value(expected_graph["indptr"].clone()).unwrap();
            let indices: Vec<usize> =
                serde_json::from_value(expected_graph["indices"].clone()).unwrap();
            let weights: Vec<u32> =
                serde_json::from_value(expected_graph["data_bits_uint32"].clone()).unwrap();
            assert_eq!(prepared.graph.indptr, indptr, "{name}");
            assert_eq!(prepared.graph.indices, indices, "{name}");
            assert_eq!(
                prepared
                    .graph
                    .data
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                weights,
                "{name}",
            );
            coordinates = prepared.coordinates;
            n_existing = n_rows;
            matched += 1;
        }
    }
    matched
}

#[test]
fn invalid_update_shapes_and_fitted_coordinates_fail_before_initialization() {
    let config = DenseUpdateConfig {
        n_neighbors: 2,
        n_components: 1,
        metric: DistanceMetric::Euclidean,
        graph_options: None,
        transform_seed: u32::MAX,
    };
    for (n_rows, n_features, n_existing) in [(2, 1, 0), (2, 1, 2), (2, 0, 1)] {
        assert_eq!(
            prepare_dense_update(&[], n_rows, n_features, n_existing, &[], &config),
            Err(DenseUpdateError::InvalidDataShape {
                n_existing,
                n_rows,
                n_features,
            })
        );
    }
    assert!(matches!(
        prepare_dense_update(
            &[],
            usize::MAX,
            2,
            1,
            &[],
            &DenseUpdateConfig {
                n_components: 2,
                ..config
            }
        ),
        Err(DenseUpdateError::Initialization(
            InitializationError::ShapeOverflow { .. }
        ))
    ));
    assert!(matches!(
        prepare_dense_update(&[0.0, 1.0], 2, 1, 1, &[], &config),
        Err(DenseUpdateError::Initialization(
            InitializationError::InvalidShape { .. }
        ))
    ));
    assert!(matches!(
        prepare_dense_update(&[0.0, 1.0], 2, 1, 1, &[f32::NAN], &config),
        Err(DenseUpdateError::Initialization(
            InitializationError::NonFiniteValue { index: 0, .. }
        ))
    ));
}
