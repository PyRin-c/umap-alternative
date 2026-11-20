use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::embedding::{
    DenseEmbeddingFitError, EmbeddingLayoutConfig, fit_graph_embedding_with_rng,
};
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::initialization::InitializationError;
use umap_core::layout::LayoutError;
use umap_core::optimize::OptimizeError;
use umap_core::rng::{Mt19937, Mt19937StateError};

const PUBLIC_FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-spectral-handoff-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    graph: Graph,
    settings: Settings,
    states: BTreeMap<u32, StateCase>,
    cases: Vec<SpectralCase>,
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    bits: Vec<u32>,
}

impl Graph {
    fn csr(self) -> CsrGraph {
        let n_rows = self.indptr.len() - 1;
        CsrGraph {
            n_rows,
            n_cols: n_rows,
            indptr: self.indptr,
            indices: self.indices,
            data: self.bits.into_iter().map(f32::from_bits).collect(),
        }
    }
}

#[derive(Deserialize)]
struct Settings {
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
    repulsion_strength: f64,
    negative_sample_rate: f64,
}

impl Settings {
    fn layout(&self, n_epochs: usize) -> EmbeddingLayoutConfig<f64, f64> {
        EmbeddingLayoutConfig {
            n_components: self.n_components,
            a: self.a,
            b: self.b,
            learning_rate: self.learning_rate,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            n_epochs,
        }
    }
}

#[derive(Deserialize)]
struct StateCase {
    seeded_words: Vec<u32>,
    block_words: Vec<u32>,
    continuations: BTreeMap<usize, Continuation>,
}

#[derive(Deserialize)]
struct Continuation {
    uniform_bits: Vec<u64>,
    next_words: Vec<u32>,
}

#[derive(Deserialize)]
struct SpectralCase {
    seed: u32,
    cursor: usize,
    raw_bits: Vec<u32>,
    post_fit_cursor: usize,
    post_fit_next_words: Vec<u32>,
    checkpoints: BTreeMap<usize, Vec<u32>>,
}

#[test]
fn state_snapshots_match_numpy_randomstate_at_seed_endpoints_and_twist_boundaries() {
    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).unwrap();

    for (seed, expected) in fixture.states {
        let mut seeded = Mt19937::new(seed);
        let (words, cursor) = seeded.state();
        assert_eq!(cursor, 624, "seed {seed}");
        assert_eq!(words.as_slice(), expected.seeded_words, "seed {seed}");

        let mut restored = Mt19937::from_state(&expected.seeded_words, cursor).unwrap();
        for index in 0..624 {
            assert_eq!(
                restored.next_u32(),
                seeded.next_u32(),
                "seed {seed} draw {index}"
            );
        }
        let (words, cursor) = restored.state();
        assert_eq!(cursor, 624, "seed {seed} after its first block");
        assert_eq!(words.as_slice(), expected.block_words, "seed {seed}");

        for (cursor, continuation) in expected.continuations {
            let mut rng = Mt19937::from_state(&expected.block_words, cursor).unwrap();
            assert_eq!(rng.state(), (words, cursor));
            for bits in continuation.uniform_bits {
                assert_eq!(
                    rng.next_f64().to_bits(),
                    bits,
                    "seed {seed}, cursor {cursor}"
                );
            }
            let (words, position) = rng.state();
            let mut continued = Mt19937::from_state(&words, position).unwrap();
            for word in continuation.next_words {
                assert_eq!(rng.next_u32(), word, "seed {seed}, cursor {cursor}");
                assert_eq!(continued.next_u32(), word, "seed {seed}, cursor {cursor}");
            }
            assert_eq!(continued.state(), rng.state());
        }
    }
}

#[test]
fn rejects_invalid_state_shape_or_cursor() {
    for actual in [0, 623, 625] {
        assert_eq!(
            Mt19937::from_state(&vec![0; actual], 0).unwrap_err(),
            Mt19937StateError::InvalidLength { actual }
        );
    }
    let (words, _) = Mt19937::new(0).state();
    for cursor in [625, usize::MAX] {
        assert_eq!(
            Mt19937::from_state(&words, cursor).unwrap_err(),
            Mt19937StateError::InvalidCursor { cursor }
        );
    }
}

#[test]
fn advanced_spectral_stream_matches_public_optimizer_checkpoints() {
    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).unwrap();
    let graph = fixture.graph.csr();

    for case in fixture.cases {
        let raw: Vec<_> = case.raw_bits.iter().copied().map(f32::from_bits).collect();
        let words = &fixture.states[&case.seed].block_words;
        for (epoch, expected) in &case.checkpoints {
            let config = fixture.settings.layout(*epoch);
            let mut rng = Mt19937::from_state(words, case.cursor).unwrap();
            let fit = fit_graph_embedding_with_rng(graph.clone(), &raw, &config, &mut rng).unwrap();
            let label = format!("seed {}, epoch {epoch}", case.seed);
            assert_eq!(fit.graph, graph, "{label}");
            assert_eq!(fit.shape, (graph.n_rows, config.n_components), "{label}");
            let bits: Vec<_> = fit
                .coordinates
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(bits, *expected, "{label}");

            let (post_fit_words, post_fit_cursor) = rng.state();
            assert_eq!(post_fit_words.as_slice(), words, "{label}");
            assert_eq!(post_fit_cursor, case.post_fit_cursor, "{label}");
            for &word in &case.post_fit_next_words {
                assert_eq!(rng.next_u32(), word, "{label}");
            }
        }

        let mut restarted = Mt19937::new(case.seed);
        let config = fixture.settings.layout(30);
        let fit =
            fit_graph_embedding_with_rng(graph.clone(), &raw, &config, &mut restarted).unwrap();
        let bits: Vec<_> = fit
            .coordinates
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_ne!(bits, case.checkpoints[&30], "restarting seed {}", case.seed);
    }
}

#[test]
fn graph_and_coordinate_errors_preserve_the_caller_rng() {
    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).unwrap();
    let case = &fixture.cases[0];
    let words = &fixture.states[&case.seed].block_words;
    let mut rng = Mt19937::from_state(words, case.cursor).unwrap();
    let start = rng.state();
    let mut graph = fixture.graph.csr();
    let config = fixture.settings.layout(2);
    let raw: Vec<_> = case.raw_bits.iter().copied().map(f32::from_bits).collect();

    let error = fit_graph_embedding_with_rng(graph.clone(), &[], &config, &mut rng).unwrap_err();
    assert!(matches!(
        error,
        DenseEmbeddingFitError::Initialization(InitializationError::InvalidShape { .. })
    ));
    assert_eq!(rng.state(), start);

    graph.indices[0] = graph.n_rows;
    let error = fit_graph_embedding_with_rng(graph, &raw, &config, &mut rng).unwrap_err();
    assert!(matches!(
        error,
        DenseEmbeddingFitError::Optimize(OptimizeError::Layout(
            LayoutError::InvalidGraphIndex { .. }
        ))
    ));
    assert_eq!(rng.state(), start);
}
