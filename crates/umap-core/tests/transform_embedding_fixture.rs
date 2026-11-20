//! End-to-end Rust checks against public umap-learn 0.6.0 transforms.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use umap_core::embedding::{DenseEmbeddingConfig, DenseEmbeddingInit, fit_dense_embedding_seeded};
use umap_core::neighbors::DistanceMetric;
use umap_core::transform::{DenseEmbeddingTransformConfig, transform_dense_embedding};

const CHECKPOINTS: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-transform-embedding-windows-x86_64.json"
);
const ORIGINAL: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json");
const CURVE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-curve-windows-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    cases: BTreeMap<String, Checkpoint>,
}

#[derive(Deserialize)]
struct Checkpoint {
    training: Vec<Vec<u32>>,
    queries: Vec<Vec<u32>>,
    checked_query_rows: Option<usize>,
    n_neighbors: usize,
    n_components: usize,
    metric: String,
    fit_epochs: Option<usize>,
    training_epochs: usize,
    fit_seed: u32,
    transform_seed: u32,
    a: f64,
    b: f64,
    learning_rate: f64,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    disconnection_distance: Option<f32>,
    fitted: Vec<Vec<u32>>,
    output: Vec<Vec<u32>>,
}

impl Checkpoint {
    fn config(&self) -> DenseEmbeddingTransformConfig {
        DenseEmbeddingTransformConfig {
            n_neighbors: self.n_neighbors,
            metric: match self.metric.as_str() {
                "euclidean" => DistanceMetric::Euclidean,
                "chebyshev" => DistanceMetric::Chebyshev,
                other => panic!("unexpected fixture metric {other}"),
            },
            n_components: self.n_components,
            a: self.a,
            b: self.b,
            learning_rate: self.learning_rate,
            fit_epochs: self.fit_epochs,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            transform_seed: self.transform_seed,
            disconnection_distance: self.disconnection_distance,
        }
    }
}

fn from_bits(matrix: &[Vec<u32>]) -> Vec<f32> {
    matrix
        .iter()
        .flatten()
        .copied()
        .map(f32::from_bits)
        .collect()
}

fn assert_bits(name: &str, actual: &[f32], expected: impl Iterator<Item = u32>) {
    let mut expected = expected;
    for (index, value) in actual.iter().enumerate() {
        assert_eq!(
            Some(value.to_bits()),
            expected.next(),
            "{name}, coordinate {index}"
        );
    }
    assert_eq!(expected.next(), None, "truncated output for {name}");
}

fn fit(
    training: &[f32],
    n_rows: usize,
    n_features: usize,
    config: &DenseEmbeddingTransformConfig,
    epochs: usize,
    seed: u32,
) -> Vec<f32> {
    let fit_config = DenseEmbeddingConfig {
        n_neighbors: config.n_neighbors,
        metric: config.metric,
        n_components: config.n_components,
        a: config.a,
        b: config.b,
        learning_rate: config.learning_rate,
        n_epochs: epochs,
        repulsion_strength: config.repulsion_strength,
        negative_sample_rate: config.negative_sample_rate,
        init: DenseEmbeddingInit::random(seed),
    };
    fit_dense_embedding_seeded(training, n_rows, n_features, &fit_config, seed)
        .unwrap()
        .coordinates
}

#[test]
fn held_out_public_checkpoints_match_bit_for_bit() {
    let fixture: Fixture = serde_json::from_str(CHECKPOINTS).unwrap();
    for (name, case) in fixture.cases {
        let config = case.config();
        let n_features = case.training[0].len();
        let training = from_bits(&case.training);
        let n_query_rows = case.checked_query_rows.unwrap_or(case.queries.len());
        let queries = from_bits(&case.queries)
            .into_iter()
            .cycle()
            .take(n_query_rows * n_features)
            .collect::<Vec<_>>();
        let fitted = if name.starts_with("fit_") || name.starts_with("different_") {
            fit(
                &training,
                case.training.len(),
                n_features,
                &config,
                case.training_epochs,
                case.fit_seed,
            )
        } else {
            from_bits(&case.fitted)
        };
        let transform = || {
            transform_dense_embedding(
                &training,
                case.training.len(),
                &queries,
                n_query_rows,
                n_features,
                &fitted,
                &config,
            )
            .unwrap()
        };
        let expected = || {
            case.output
                .iter()
                .flatten()
                .copied()
                .cycle()
                .take(n_query_rows * config.n_components)
        };
        assert_bits(&name, &transform(), expected());
        assert_bits(&name, &transform(), expected());
        assert_bits(&name, &fitted, case.fitted.iter().flatten().copied());
    }
}

#[test]
fn original_45_epoch_fit_and_transform_match_bit_for_bit() {
    let original: Value = serde_json::from_str(ORIGINAL).unwrap();
    let curve: Value = serde_json::from_str(CURVE).unwrap();
    let default_curve = &curve["samples"][2];
    assert_eq!(default_curve["spread"], 1.0);
    assert_eq!(default_curve["min_dist"], 0.1);
    let config = DenseEmbeddingTransformConfig {
        n_neighbors: 4,
        metric: DistanceMetric::Euclidean,
        n_components: 2,
        a: f64::from_bits(default_curve["bits"][0].as_u64().unwrap()),
        b: f64::from_bits(default_curve["bits"][1].as_u64().unwrap()),
        learning_rate: 1.0,
        fit_epochs: Some(45),
        repulsion_strength: 1.0,
        negative_sample_rate: 5.0,
        transform_seed: 24,
        disconnection_distance: None,
    };
    let matrix = |record: &Value| {
        serde_json::from_value::<Vec<Vec<u32>>>(record["bits_uint32"].clone()).unwrap()
    };
    let training = matrix(&original["inputs"]["training"]);
    let queries = matrix(&original["inputs"]["transform_queries"]);
    let n_features = training[0].len();
    let n_training = training.len();
    let n_query = queries.len();
    let training = from_bits(&training);
    let queries = from_bits(&queries);
    let fitted = fit(&training, n_training, n_features, &config, 45, 42);
    let embedding = &original["cases"]["embedding"];
    let fitted_bits = matrix(&embedding["training"]["embedding"]);
    let transformed_bits = matrix(&embedding["transform"]["output"]);
    assert_bits("fitted", &fitted, fitted_bits.into_iter().flatten());
    let transformed = transform_dense_embedding(
        &training, n_training, &queries, n_query, n_features, &fitted, &config,
    )
    .unwrap();
    assert_bits(
        "transformed",
        &transformed,
        transformed_bits.into_iter().flatten(),
    );
}
