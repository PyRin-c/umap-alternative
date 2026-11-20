//! Replay 76 positive-epoch coordinates from recorded initial layouts and
//! public CSR graphs. The dense-input fixtures also exercise the full fit
//! from raw data and explicit initialization for 28 of those coordinates.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::embedding::{
    DenseEmbedding, DenseEmbeddingConfig, DenseEmbeddingInit, fit_dense_embedding,
};
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::layout::attractive_optimize_csr;
use umap_core::neighbors::DistanceMetric;
use umap_core::optimize::optimize_csr;

const ISOLATED_PAIRS: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-isolated-edge-arithmetic-windows-x86_64.json"
);
const WEIGHTED_UNIT_CURVE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-arithmetic-4x1-base-windows-x86_64.json"
);
const WEIGHTED_NON_UNIT_CURVE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-arithmetic-4x1-ab17-09-windows-x86_64.json"
);
const DENSE_TWO_COMPONENTS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-optimizer-windows-x86_64.json");

#[derive(Deserialize)]
struct IsolatedPairsFixture {
    parameters: Sampling,
    scenarios: BTreeMap<String, IsolatedPairsScenario>,
}

#[derive(Deserialize)]
struct IsolatedPairsScenario {
    settings: Settings,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Deserialize)]
struct DenseFixture {
    input: DenseInput,
    parameters: DenseParameters,
    expected: Checkpoints,
}

impl DenseFixture {
    fn fit(&self, n_epochs: usize) -> DenseEmbedding {
        let parameters = &self.parameters;
        let config: DenseEmbeddingConfig<f64> = DenseEmbeddingConfig {
            n_neighbors: parameters.n_neighbors,
            metric: DistanceMetric::Euclidean,
            n_components: parameters.n_components,
            a: parameters.attraction.a,
            b: parameters.attraction.b,
            learning_rate: parameters.attraction.learning_rate(),
            n_epochs,
            repulsion_strength: parameters.sampling.repulsion_strength,
            negative_sample_rate: parameters.sampling.negative_sample_rate(),
            init: DenseEmbeddingInit::Explicit {
                values: self.input.init_values.iter().flatten().copied().collect(),
            },
        };
        assert_eq!(parameters.metric, "euclidean");
        let data = self
            .input
            .values
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        fit_dense_embedding(
            &data,
            self.input.values.len(),
            self.input.values[0].len(),
            &config,
            std::convert::identity,
        )
        .expect("valid public dense attraction input")
    }
}

#[derive(Deserialize)]
struct DenseInput {
    values: Vec<Vec<f32>>,
    init_values: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct DenseParameters {
    n_neighbors: usize,
    n_components: usize,
    metric: String,
    #[serde(flatten)]
    attraction: Settings,
    #[serde(flatten)]
    sampling: Sampling,
}

#[derive(Deserialize)]
struct Checkpoints {
    checkpoints: Vec<Checkpoint>,
}

#[derive(Deserialize)]
struct Settings {
    a: f64,
    b: f64,
    learning_rate: Option<f64>,
}

impl Settings {
    fn learning_rate(&self) -> f64 {
        // The original five-point fixture used the public default.
        self.learning_rate.unwrap_or(1.0)
    }
}

#[derive(Deserialize)]
struct Sampling {
    repulsion_strength: f32,
    negative_sample_rate: Option<f64>,
}

impl Sampling {
    fn negative_sample_rate(&self) -> f64 {
        self.negative_sample_rate.unwrap_or(5.0)
    }
}

#[derive(Deserialize)]
struct Checkpoint {
    n_epochs: usize,
    graph: Graph,
    embedding: Embedding,
}

#[derive(Deserialize)]
struct Graph {
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

impl Graph {
    fn to_csr(&self) -> CsrGraph {
        CsrGraph {
            n_rows: self.shape[0],
            n_cols: self.shape[1],
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self
                .data_bits_uint32
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct Embedding {
    shape: [usize; 2],
    bits_uint32: Vec<Vec<u32>>,
}

fn initial_checkpoint(checkpoints: &[Checkpoint]) -> &Checkpoint {
    checkpoints
        .iter()
        .find(|checkpoint| checkpoint.n_epochs == 0)
        .expect("fixture records the initialization")
}

impl Embedding {
    fn coordinates(&self) -> Vec<f32> {
        self.bits_uint32
            .iter()
            .flatten()
            .copied()
            .map(f32::from_bits)
            .collect()
    }

    fn assert_bits(&self, actual: &[f32], case: &str, route: &str, n_epochs: usize) {
        let [n_rows, n_components] = self.shape;
        assert_eq!(actual.len(), n_rows * n_components);
        assert_eq!(self.bits_uint32.len(), n_rows);
        for (row, (coordinates, expected)) in actual
            .chunks_exact(n_components)
            .zip(&self.bits_uint32)
            .enumerate()
        {
            let actual_bits = coordinates
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                &actual_bits, expected,
                "{route}: {case}, n_epochs={n_epochs}, row {row}"
            );
        }
    }
}

fn replay_checkpoints(
    case: &str,
    settings: &Settings,
    sampling: &Sampling,
    checkpoints: &[Checkpoint],
) {
    let initial = initial_checkpoint(checkpoints).embedding.coordinates();
    for checkpoint in checkpoints {
        let [n_rows, n_components] = checkpoint.embedding.shape;
        let n_epochs = checkpoint.n_epochs;
        let expected = &checkpoint.embedding;
        let graph = checkpoint.graph.to_csr();
        assert_eq!(checkpoint.graph.shape, [n_rows, n_rows]);

        let mut actual = initial.clone();
        attractive_optimize_csr(
            &mut actual,
            n_components,
            &graph,
            settings.a,
            settings.b,
            settings.learning_rate(),
            n_epochs,
        )
        .expect("valid public attraction checkpoint");
        expected.assert_bits(&actual, case, "attractive_optimize_csr", n_epochs);

        actual.copy_from_slice(&initial);
        optimize_csr(
            &mut actual,
            n_components,
            &graph,
            settings.a,
            settings.b,
            settings.learning_rate(),
            sampling.repulsion_strength,
            sampling.negative_sample_rate(),
            n_epochs,
            std::convert::identity,
        )
        .expect("valid public attraction checkpoint with self samples");
        expected.assert_bits(&actual, case, "optimize_csr", n_epochs);
    }
}

#[test]
fn isolated_pairs_match_public_attraction_checkpoints() {
    let fixture: IsolatedPairsFixture = serde_json::from_str(ISOLATED_PAIRS).unwrap();
    for (case, scenario) in fixture.scenarios {
        replay_checkpoints(
            &case,
            &scenario.settings,
            &fixture.parameters,
            &scenario.checkpoints,
        );
    }
}

#[test]
fn dense_graphs_match_public_attraction_checkpoints() {
    for (case, json) in [
        ("weighted unit curve", WEIGHTED_UNIT_CURVE),
        ("weighted non-unit curve", WEIGHTED_NON_UNIT_CURVE),
        ("five points, two components", DENSE_TWO_COMPONENTS),
    ] {
        let fixture: DenseFixture = serde_json::from_str(json).unwrap();
        replay_checkpoints(
            case,
            &fixture.parameters.attraction,
            &fixture.parameters.sampling,
            &fixture.expected.checkpoints,
        );
        for checkpoint in &fixture.expected.checkpoints {
            let n_epochs = checkpoint.n_epochs;
            let actual = fixture.fit(n_epochs);
            let [n_rows, n_components] = checkpoint.embedding.shape;
            assert_eq!(actual.shape, (n_rows, n_components));
            assert_eq!(
                actual.graph,
                checkpoint.graph.to_csr(),
                "{case}, n_epochs={n_epochs}: graph"
            );
            checkpoint.embedding.assert_bits(
                &actual.coordinates,
                case,
                "fit_dense_embedding",
                n_epochs,
            );
        }
    }
}

#[test]
fn dense_fit_retains_full_attraction_precision_at_five_epochs() {
    // Both precisions happen to give the same bits at this fixture's public
    // two-epoch checkpoint. Compare a longer run to the standalone Rust
    // kernel using the recorded graph and normalized coordinates.
    let fixture: DenseFixture = serde_json::from_str(WEIGHTED_NON_UNIT_CURVE).unwrap();
    let n_epochs = 5;
    let initial = initial_checkpoint(&fixture.expected.checkpoints);
    let graph = initial.graph.to_csr();
    let mut expected = initial.embedding.coordinates();
    let settings = &fixture.parameters.attraction;
    attractive_optimize_csr(
        &mut expected,
        initial.embedding.shape[1],
        &graph,
        settings.a,
        settings.b,
        settings.learning_rate(),
        n_epochs,
    )
    .unwrap();

    let actual = fixture.fit(n_epochs);
    assert_eq!(actual.graph, graph);
    assert_eq!(
        actual
            .coordinates
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
}
