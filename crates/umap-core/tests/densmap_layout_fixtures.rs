use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::density::{RadiusSpace, radii_from_edge_distances};
use umap_core::densmap::{DensmapError, DensmapOptions, optimize_densmap_csr};
use umap_core::embedding::EmbeddingLayoutConfig;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::initialization::normalize_columns;
use umap_core::layout::LayoutError;
use umap_core::optimize::OptimizeError;
use umap_core::rng::{Mt19937, SerialNegativeSampler};

const WINDOWS: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-densmap-layout-windows-x86_64.json");
const UBUNTU: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-densmap-layout-ubuntu-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    common_options: Parameters,
    datasets: BTreeMap<String, Dataset>,
    cases: BTreeMap<String, Case>,
    errors: BTreeMap<String, ErrorCase>,
}

#[derive(Deserialize)]
struct Dataset {
    init: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Case {
    dataset: String,
    epochs: usize,
    density: bool,
    options: Parameters,
    graph: Graph,
    checkpoints: BTreeMap<usize, Vec<u32>>,
    #[serde(rename = "final")]
    final_bits: Vec<u32>,
    original_radii: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrorCase {
    epochs: usize,
    options: Parameters,
    error: String,
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<u32>,
    distances: Vec<u32>,
}

#[derive(Deserialize)]
struct Parameters {
    a: Option<f64>,
    b: Option<f64>,
    learning_rate: Option<f64>,
    repulsion_strength: Option<f64>,
    negative_sample_rate: Option<f64>,
    random_state: Option<u32>,
    dens_frac: Option<f64>,
    dens_lambda: Option<f64>,
    dens_var_shift: Option<f64>,
}

struct Replay {
    graph: CsrGraph,
    distances: Vec<f32>,
    initial: Vec<f32>,
    seed: u32,
    layout: EmbeddingLayoutConfig<f64, f64>,
    density: DensmapOptions,
}

impl Replay {
    fn new(fixture: &Fixture, case: &Case) -> Self {
        let input = &fixture.datasets[&case.dataset].init;
        let (n_rows, n_components) = (input.len(), input[0].len());
        let mut initial = input.iter().flatten().copied().collect::<Vec<_>>();
        normalize_columns(&mut initial, n_rows, n_components).unwrap();
        let graph = CsrGraph {
            n_rows,
            n_cols: n_rows,
            indptr: case.graph.indptr.clone(),
            indices: case.graph.indices.clone(),
            data: case
                .graph
                .weights
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect(),
        };
        let options = &case.options;
        let base = &fixture.common_options;
        let layout = EmbeddingLayoutConfig {
            n_components,
            a: options.a.or(base.a).unwrap(),
            b: options.b.or(base.b).unwrap(),
            learning_rate: options.learning_rate.or(base.learning_rate).unwrap(),
            repulsion_strength: options
                .repulsion_strength
                .or(base.repulsion_strength)
                .unwrap(),
            negative_sample_rate: options
                .negative_sample_rate
                .or(base.negative_sample_rate)
                .unwrap(),
            n_epochs: case.epochs,
        };
        let defaults = DensmapOptions::default();
        let density = DensmapOptions {
            dens_frac: options.dens_frac.unwrap_or(defaults.dens_frac),
            dens_lambda: if case.density {
                options.dens_lambda.unwrap_or(defaults.dens_lambda)
            } else {
                0.0
            },
            dens_var_shift: options.dens_var_shift.unwrap_or(defaults.dens_var_shift),
        };
        Self {
            graph,
            distances: case
                .graph
                .distances
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect(),
            initial,
            seed: options.random_state.or(base.random_state).unwrap(),
            layout,
            density,
        }
    }

    fn run(&self, checkpoint: impl FnMut(usize, &[f32])) -> Result<Vec<u32>, DensmapError> {
        let mut embedding = self.initial.clone();
        let mut sampler = SerialNegativeSampler::new(
            &mut Mt19937::new(self.seed),
            &embedding,
            self.graph.n_rows,
            self.layout.n_components,
        )
        .unwrap();
        optimize_densmap_csr(
            &mut embedding,
            &self.graph,
            &self.distances,
            &self.layout,
            self.density,
            |row| sampler.sample(row).unwrap(),
            checkpoint,
        )?;
        Ok(bits(&embedding))
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_checkpoint(platform: &str, name: &str, key: usize, actual: &[u32], expected: &[u32]) {
    assert_eq!(actual, expected, "{platform} {name} epoch {key}");
}

#[test]
fn public_densmap_checkpoints() {
    for (platform, source) in [("Windows", WINDOWS), ("Ubuntu", UBUNTU)] {
        let fixture: Fixture = serde_json::from_str(source).unwrap();
        for (name, case) in &fixture.cases {
            let replay = Replay::new(&fixture, case);
            if let Some(expected) = &case.original_radii {
                let actual = radii_from_edge_distances(
                    &replay.graph,
                    &replay.distances,
                    RadiusSpace::Original,
                )
                .unwrap();
                assert_eq!(&bits(&actual), expected, "{platform} {name} original radii");
            }
            let mut seen = Vec::new();
            let actual = replay
                .run(|key, values| {
                    seen.push(key);
                    if let Some(expected) = case.checkpoints.get(&key) {
                        assert_checkpoint(platform, name, key, &bits(values), expected);
                    }
                })
                .unwrap_or_else(|error| panic!("{platform} {name}: {error}"));
            assert_eq!(
                seen,
                (0..=case.epochs).collect::<Vec<_>>(),
                "{platform} {name}"
            );
            assert_checkpoint(platform, name, case.epochs, &actual, &case.final_bits);
        }
    }
}

#[test]
fn variance_and_fraction_errors_preserve_completed_checkpoints() {
    let fixture: Fixture = serde_json::from_str(WINDOWS).unwrap();
    let check_error = |replay: &Replay, initial: &[u32]| {
        let mut seen = Vec::new();
        let error = replay.run(|key, coordinates| seen.push((key, bits(coordinates))));
        assert_eq!(seen, [(0, initial.to_vec())]);
        assert!(matches!(
            error,
            Err(DensmapError::Optimize(OptimizeError::Layout(
                LayoutError::DensityDivisionByZero
            )))
        ));
    };

    let shifted = &fixture.cases["equal_embedding_shifted"];
    let mut replay = Replay::new(&fixture, shifted);
    for name in [
        "equal_embedding_unshifted",
        "equal_embedding_underflow_shift",
    ] {
        let invalid = &fixture.errors[name];
        assert_eq!(invalid.error, "ZeroDivisionError");
        replay.density.dens_var_shift = invalid.options.dens_var_shift.unwrap();
        check_error(&replay, &shifted.checkpoints[&0]);
    }
    replay.initial = vec![0.0, 0.0, 4.0, 4.0];
    check_error(&replay, &bits(&replay.initial));

    for name in ["frac_below_zero", "frac_above_one"] {
        let invalid = &fixture.errors[name];
        assert_eq!(invalid.error, "ValueError");
        replay.layout.n_epochs = invalid.epochs;
        replay.density.dens_frac = invalid.options.dens_frac.unwrap();
        assert!(matches!(
            replay.run(|_, _| panic!("invalid options emit no checkpoints")),
            Err(DensmapError::InvalidOption {
                parameter: "dens_frac",
                ..
            })
        ));
    }
}

#[test]
fn undefined_input_poisoning_follows_the_positive_edge_schedule() {
    let fixture: Fixture = serde_json::from_str(WINDOWS).unwrap();
    let mut replay = Replay::new(&fixture, &fixture.cases["equal_input"]);
    replay.initial = vec![0.0, 1.0, 4.0, 4.0];
    replay.graph.data = vec![1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0];
    replay.layout.n_epochs = 5;
    replay.layout.negative_sample_rate = 3.0;
    replay.layout.repulsion_strength = 0.8;
    // Observed by calling the public optimizer with intervals [1, 1, 3, 3].
    let nan = 0xffc0_0000;
    let partial = [nan, nan, 4.0_f32.to_bits(), 4.0_f32.to_bits()];
    let expected = [
        bits(&replay.initial).try_into().unwrap(),
        partial,
        partial,
        [nan; 4],
        [nan; 4],
        [nan; 4],
    ];
    let mut keys = Vec::new();
    let final_bits = replay
        .run(|key, coordinates| {
            keys.push(key);
            assert_eq!(bits(coordinates), expected[key], "checkpoint {key}");
        })
        .unwrap();
    assert_eq!(keys, [0, 1, 2, 3, 4, 5]);
    assert_eq!(final_bits, [nan; 4]);
}
