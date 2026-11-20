use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use umap_core::embedding::EmbeddingLayoutConfig;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::layout::{attractive_update, negative_example_update};
use umap_core::rng::Mt19937;
use umap_core::update_layout::{begin_update_layout_rng, optimize_update_layout};

const CHECKPOINTS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-update-layout-windows-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    fit_parameters: Value,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    fit_overrides: Value,
    updated_graph: Graph,
    numpy_rng_model: SeedBoundary,
    trials: BTreeMap<String, Trial>,
}

#[derive(Deserialize)]
struct Graph {
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

impl From<Graph> for CsrGraph {
    fn from(record: Graph) -> Self {
        Self {
            n_rows: record.shape[0],
            n_cols: record.shape[1],
            indptr: record.indptr,
            indices: record.indices,
            data: record
                .data_bits_uint32
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct SeedBoundary {
    initial_raw_words: [u32; 3],
    jitter_raw_word_count: usize,
    optimizer_raw_words: [u32; 3],
}

impl SeedBoundary {
    fn prepared_rng(&self, seed: u32) -> (Mt19937, Mt19937) {
        let mut expected_rng = Mt19937::new(seed);
        assert_eq!(
            std::array::from_fn(|_| expected_rng.next_u32()),
            self.initial_raw_words
        );
        let mut rng = begin_update_layout_rng(seed);
        assert_eq!(rng.state(), expected_rng.state());
        // The zero-epoch public layout already includes jitter. Reproduce only
        // the independent NumPy model's raw word consumption here.
        for _ in 0..self.jitter_raw_word_count {
            rng.next_u32();
        }
        expected_rng = rng.clone();
        assert_eq!(
            std::array::from_fn(|_| expected_rng.next_u32()),
            self.optimizer_raw_words
        );
        (rng, expected_rng)
    }
}

#[derive(Deserialize)]
struct Trial {
    update_overrides: Value,
    embedding: Embedding,
}

#[derive(Deserialize)]
struct Embedding {
    shape: [usize; 2],
    bits_uint32: Vec<Vec<u32>>,
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
}

#[test]
fn serial_dense_update_matches_public_checkpoint_bits() {
    let fixture: Fixture = serde_json::from_str(CHECKPOINTS).unwrap();
    for (case_name, case) in fixture.cases {
        let graph = CsrGraph::from(case.updated_graph);
        let initial = &case.trials["0"].embedding;
        let fit = |key: &str| {
            case.fit_overrides
                .get(key)
                .unwrap_or(&fixture.fit_parameters[key])
        };
        for (trial_name, trial) in &case.trials {
            let update = |key: &str| trial.update_overrides.get(key).unwrap_or(fit(key));
            let config = EmbeddingLayoutConfig {
                n_components: initial.shape[1],
                a: fit("a").as_f64().unwrap(),
                b: fit("b").as_f64().unwrap(),
                learning_rate: fit("learning_rate").as_f64().unwrap(),
                n_epochs: update("n_epochs").as_u64().unwrap_or(0) as usize,
                repulsion_strength: update("repulsion_strength").as_f64().unwrap_or(1.0),
                negative_sample_rate: update("negative_sample_rate").as_f64().unwrap(),
            };
            let seed = u32::try_from(fit("transform_seed").as_u64().unwrap()).unwrap();
            let (mut rng, expected_rng) = case.numpy_rng_model.prepared_rng(seed);
            let mut actual = initial.coordinates();
            optimize_update_layout(&mut actual, &graph, &config, &mut rng).unwrap();
            let label = format!("{case_name} / {trial_name}");
            assert_eq!(trial.embedding.shape, initial.shape, "{label}");
            let expected = trial.embedding.bits_uint32.iter().flatten().copied();
            assert!(
                actual.iter().map(|value| value.to_bits()).eq(expected),
                "{label}"
            );
            assert_eq!(rng.state(), expected_rng.state(), "{label}");
        }
    }
}

#[test]
fn two_epoch_public_checkpoint_identifies_negative_targets() {
    let mut fixture: Fixture = serde_json::from_str(CHECKPOINTS).unwrap();
    let case = fixture.cases.remove("pair").unwrap();
    let graph = CsrGraph::from(case.updated_graph);
    let mut edges = Vec::new();
    for head in 0..graph.n_rows {
        for entry in graph.indptr[head]..graph.indptr[head + 1] {
            if graph.data[entry] == 1.0 {
                edges.push((head, graph.indices[entry]));
            }
        }
    }

    // At rate two, the unit edges are the only edges due in this run, and
    // each has one negative draw. Look for exact replays independently of RNG.
    let decoder = Decoder {
        edges: &edges,
        target: &case.trials["2"].embedding.coordinates(),
        a: fixture.fit_parameters["a"].as_f64().unwrap(),
        b: fixture.fit_parameters["b"].as_f64().unwrap(),
        alpha: fixture.fit_parameters["learning_rate"].as_f64().unwrap(),
    };
    let mut matches = Vec::new();
    decoder.try_draws(
        &case.trials["0"].embedding.coordinates(),
        &mut Vec::new(),
        &mut matches,
    );
    assert_eq!(matches, [[6, 6, 4, 0, 2, 2, 2, 6, 6, 8, 5, 2]]);
}

struct Decoder<'a> {
    edges: &'a [(usize, usize)],
    target: &'a [f32],
    a: f64,
    b: f64,
    alpha: f64,
}

impl Decoder<'_> {
    fn try_draws(
        &self,
        coordinates: &[f32],
        draws: &mut Vec<usize>,
        matches: &mut Vec<Vec<usize>>,
    ) {
        let Some((&(head, tail), later_edges)) = self.edges[draws.len()..].split_first() else {
            if coordinates
                .iter()
                .map(|value| value.to_bits())
                .eq(self.target.iter().map(|value| value.to_bits()))
            {
                matches.push(draws.clone());
            }
            return;
        };
        let mut attracted = coordinates.to_vec();
        attractive_update(&mut attracted, 2, head, tail, self.a, self.b, self.alpha).unwrap();
        for candidate in 0..coordinates.len() / 2 {
            let mut next = attracted.clone();
            if candidate != head {
                negative_example_update(
                    &mut next[head * 2..head * 2 + 2],
                    &attracted[candidate * 2..candidate * 2 + 2],
                    self.a,
                    self.b,
                    1.0,
                    self.alpha,
                )
                .unwrap();
            }
            // Once a row has no remaining incident edge, its final bits are known.
            let matches_final_rows = [head, tail].into_iter().all(|row| {
                later_edges
                    .iter()
                    .any(|&(later_head, later_tail)| row == later_head || row == later_tail)
                    || (row * 2..row * 2 + 2).all(|i| next[i].to_bits() == self.target[i].to_bits())
            });
            if matches_final_rows {
                draws.push(candidate);
                self.try_draws(&next, draws, matches);
                draws.pop();
            }
        }
    }
}
