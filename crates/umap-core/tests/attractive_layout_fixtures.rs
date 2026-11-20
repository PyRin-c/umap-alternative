//! Replay attraction from the recorded initial coordinates and public CSR
//! graphs so graph construction and initialization cannot affect the comparison.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::layout::attractive_optimize_csr;

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
    scenarios: BTreeMap<String, IsolatedPairsScenario>,
}

#[derive(Deserialize)]
struct IsolatedPairsScenario {
    settings: Settings,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Deserialize)]
struct DenseFixture {
    parameters: Settings,
    expected: Checkpoints,
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

fn replay_checkpoints(case: &str, settings: &Settings, checkpoints: &[Checkpoint]) {
    let initial = checkpoints
        .iter()
        .find(|checkpoint| checkpoint.n_epochs == 0)
        .expect("fixture records the initialization")
        .embedding
        .bits_uint32
        .iter()
        .flatten()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    // The original five-point fixture used the public default learning rate.
    let learning_rate = settings.learning_rate.unwrap_or(1.0);
    for checkpoint in checkpoints {
        let [n_rows, n_components] = checkpoint.embedding.shape;
        assert_eq!(checkpoint.graph.shape, [n_rows, n_rows]);
        let graph = checkpoint.graph.to_csr();
        let mut actual = initial.clone();
        attractive_optimize_csr(
            &mut actual,
            n_components,
            &graph,
            settings.a,
            settings.b,
            learning_rate,
            checkpoint.n_epochs,
        )
        .expect("valid public attraction checkpoint");
        assert_eq!(actual.len(), n_rows * n_components);
        assert_eq!(checkpoint.embedding.bits_uint32.len(), n_rows);
        for (row, (coordinates, expected)) in actual
            .chunks_exact(n_components)
            .zip(&checkpoint.embedding.bits_uint32)
            .enumerate()
        {
            let actual_bits = coordinates
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                &actual_bits, expected,
                "{case}, n_epochs={}, row {row}",
                checkpoint.n_epochs
            );
        }
    }
}

#[test]
fn isolated_pairs_match_public_attraction_checkpoints() {
    let fixture: IsolatedPairsFixture = serde_json::from_str(ISOLATED_PAIRS).unwrap();
    for (case, scenario) in fixture.scenarios {
        replay_checkpoints(&case, &scenario.settings, &scenario.checkpoints);
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
        replay_checkpoints(case, &fixture.parameters, &fixture.expected.checkpoints);
    }
}
