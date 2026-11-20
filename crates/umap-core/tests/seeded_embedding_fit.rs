use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::embedding::{DenseEmbeddingConfig, DenseEmbeddingInit, fit_dense_embedding_seeded};
use umap_core::neighbors::DistanceMetric;

const PUBLIC_FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-seeded-fit-windows-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    inputs: BTreeMap<String, Vec<Vec<f32>>>,
    graphs: BTreeMap<String, Graph>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    input: String,
    graph: String,
    params: Parameters,
    init: Initialization,
    checkpoints: BTreeMap<usize, Vec<Vec<u32>>>,
}

#[derive(Deserialize)]
struct Parameters {
    n_neighbors: usize,
    n_components: usize,
    metric: String,
    random_state: u32,
    learning_rate: f64,
    repulsion_strength: f64,
    negative_sample_rate: f64,
    a: f64,
    b: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Initialization {
    Random(String),
    Explicit(Vec<Vec<f32>>),
}

#[test]
fn seeded_dense_fit_matches_public_checkpoints() {
    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).expect("valid public fixture");
    let mut failures = Vec::new();
    for case in fixture.cases {
        let data = &fixture.inputs[&case.input];
        let n_rows = data.len();
        let n_features = data[0].len();
        let data = data.iter().flatten().copied().collect::<Vec<_>>();
        let expected_graph = &fixture.graphs[&case.graph];
        let params = case.params;
        assert_eq!(params.metric, "euclidean");
        let (init, explicit_seed) = match case.init {
            Initialization::Random(kind) => {
                assert_eq!(kind, "random");
                // The random layout must use its configured seed for both draws.
                (
                    DenseEmbeddingInit::random(params.random_state),
                    !params.random_state,
                )
            }
            Initialization::Explicit(values) => {
                assert_eq!(values.len(), n_rows);
                assert!(values.iter().all(|row| row.len() == params.n_components));
                let values = values.into_iter().flatten().collect();
                (DenseEmbeddingInit::Explicit { values }, params.random_state)
            }
        };
        let mut config = DenseEmbeddingConfig::<f64, f64> {
            n_neighbors: params.n_neighbors,
            metric: DistanceMetric::Euclidean,
            n_components: params.n_components,
            a: params.a,
            b: params.b,
            learning_rate: params.learning_rate,
            n_epochs: 0,
            repulsion_strength: params.repulsion_strength,
            negative_sample_rate: params.negative_sample_rate,
            init,
        };

        for (epoch, expected) in case.checkpoints {
            config.n_epochs = epoch;
            let label = format!("{} epoch {epoch}", case.name);
            let actual =
                fit_dense_embedding_seeded(&data, n_rows, n_features, &config, explicit_seed)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));

            assert_eq!(
                (actual.graph.n_rows, actual.graph.n_cols),
                (n_rows, n_rows),
                "{label}"
            );
            assert_eq!(
                actual.graph.indptr, expected_graph.indptr,
                "{label}: graph indptr"
            );
            assert_eq!(
                actual.graph.indices, expected_graph.indices,
                "{label}: graph indices"
            );
            let graph_bits: Vec<_> = actual
                .graph
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(
                graph_bits.len(),
                expected_graph.bits.len(),
                "{label}: graph weights"
            );
            for (edge, (actual, expected)) in
                graph_bits.iter().zip(&expected_graph.bits).enumerate()
            {
                if actual != expected {
                    failures.push(format!(
                        "{label}, edge {edge}: {actual:08x} != {expected:08x}"
                    ));
                }
            }
            assert_eq!(actual.shape, (n_rows, params.n_components), "{label}");
            assert_eq!(
                actual.coordinates.len(),
                n_rows * params.n_components,
                "{label}"
            );
            assert_eq!(expected.len(), n_rows, "{label}");
            assert!(
                expected.iter().all(|row| row.len() == params.n_components),
                "{label}"
            );
            let differences: Vec<_> = actual
                .coordinates
                .iter()
                .zip(expected.iter().flatten())
                .enumerate()
                .filter_map(|(index, (actual, expected))| {
                    let actual = actual.to_bits();
                    (actual != *expected).then(|| {
                        format!(
                            "({},{}) {actual:08x}/{expected:08x}",
                            index / params.n_components,
                            index % params.n_components
                        )
                    })
                })
                .collect();
            if !differences.is_empty() {
                failures.push(format!(
                    "{label}, {}/{} differing coordinates: {}",
                    differences.len(),
                    n_rows * params.n_components,
                    differences.join(", ")
                ));
            }

            if params.random_state == u32::MAX && epoch == 500 {
                let rounded_config = DenseEmbeddingConfig::<f64> {
                    n_neighbors: config.n_neighbors,
                    metric: config.metric,
                    n_components: config.n_components,
                    a: config.a,
                    b: config.b,
                    learning_rate: config.learning_rate,
                    n_epochs: config.n_epochs,
                    repulsion_strength: config.repulsion_strength as f32,
                    negative_sample_rate: config.negative_sample_rate,
                    init: config.init.clone(),
                };
                let rounded = fit_dense_embedding_seeded(
                    &data,
                    n_rows,
                    n_features,
                    &rounded_config,
                    explicit_seed,
                )
                .expect("valid single-precision repulsion strength");
                if rounded.coordinates == actual.coordinates {
                    failures.push(format!("{label}: f32 rounding must change the result"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
