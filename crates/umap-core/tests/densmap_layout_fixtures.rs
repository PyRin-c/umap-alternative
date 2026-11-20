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

fn distance(got: u32, want: u32) -> (u32, f32) {
    let ulp = if got >> 31 == want >> 31 {
        got.abs_diff(want)
    } else {
        (got & 0x7fff_ffff).saturating_add(want & 0x7fff_ffff)
    };
    (ulp, (f32::from_bits(got) - f32::from_bits(want)).abs())
}

fn assert_checkpoint(platform: &str, name: &str, key: usize, actual: &[u32], expected: &[u32]) {
    // Only the observed active residuals receive a tolerance. Inactive
    // checkpoints and isolated cases with exact replay require identical bits.
    let residual = match name {
        "pairs_shift_0.0" | "pairs_shift_4.0" if key >= 1 => Some((1, 2.4e-7)),
        "weighted_active" | "frac_above_half" | "frac_below_one" if key >= 1 => Some((3, 1e-6)),
        "frac_below_half" | "frac_exact_half" if key >= 2 => Some((2, 2.4e-7)),
        "frac_default" if key >= 7 => Some((1, 4.8e-7)),
        "weighted_repel" if key >= 1 => Some((1, 1e-6)),
        "weighted_weak_density" if key >= 1 => Some((1, 4e-9)),
        "weak_edges_short" if key >= 1 => Some((10, 1.2e-7)),
        "weak_edges_long" if key >= 2 => Some((12, 3.4e-6)),
        "held_out_1d" if key >= 4 => Some((1, 7.5e-9)),
        "held_out_3d_first_active" if key >= 1 => Some((16, 2.2e-6)),
        "held_out_3d" if key >= 3 => Some((211, 6.7e-5)),
        _ => None,
    };
    let Some((max_steps, max_error)) = residual else {
        assert_eq!(actual, expected, "{platform} {name} epoch {key}");
        return;
    };
    assert_eq!(
        actual.len(),
        expected.len(),
        "{platform} {name} epoch {key}"
    );
    for (coordinate, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let (steps, error) = distance(got, want);
        assert!(
            steps <= max_steps && error <= max_error,
            "{platform} {name} epoch {key} coordinate {coordinate}: {steps} steps, {error:e}"
        );
    }
}

#[test]
fn public_densmap_checkpoints() {
    for (platform, source) in [("Windows", WINDOWS), ("Ubuntu", UBUNTU)] {
        let fixture: Fixture = serde_json::from_str(source).unwrap();
        for (name, case) in &fixture.cases {
            if name == "equal_input" {
                continue;
            }
            let replay = Replay::new(&fixture, case);
            if let Some(expected) = &case.original_radii {
                let actual = radii_from_edge_distances(
                    &replay.graph,
                    &replay.distances,
                    RadiusSpace::Original,
                )
                .unwrap();
                assert_eq!(actual.len(), expected.len(), "{platform} {name}");
                for (got, want) in bits(&actual).into_iter().zip(expected) {
                    assert!(distance(got, *want).0 <= 2, "{platform} {name}");
                }
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
    let original = &fixture.cases["equal_input"];
    assert!(
        original
            .final_bits
            .iter()
            .all(|&bits| f32::from_bits(bits).is_nan())
    );
    let mut replay = Replay::new(&fixture, original);
    let check_error = |replay: &Replay, initial: &[u32]| {
        let mut seen = Vec::new();
        let error = replay.run(|key, coordinates| seen.push((key, bits(coordinates))));
        assert_eq!(seen, [(0, initial.to_vec())]);
        assert!(matches!(
            error,
            Err(DensmapError::Optimize(OptimizeError::Layout(
                LayoutError::NonFiniteArithmetic {
                    quantity: "density correlation gradient",
                    ..
                }
            )))
        ));
    };
    check_error(&replay, &original.checkpoints[&0]);

    let shifted = &fixture.cases["equal_embedding_shifted"];
    let unshifted = &fixture.errors["equal_embedding_unshifted"];
    assert_eq!(unshifted.error, "ZeroDivisionError");
    replay = Replay::new(&fixture, shifted);
    replay.density.dens_var_shift = unshifted.options.dens_var_shift.unwrap();
    check_error(&replay, &shifted.checkpoints[&0]);

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
