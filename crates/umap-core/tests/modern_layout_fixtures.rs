use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use umap_core::fit::{fit_dense_graph_with_options, fit_precomputed_graph_with_options};
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::modern_initialization::{
    ModernInit, ModernInitConfig, ModernInitRescale, initialize_modern_layout,
};
use umap_core::modern_layout::{
    ModernLayout, ModernLayoutConfig, ModernLayoutError, ModernOptimizer,
};
use umap_core::neighbors::DistanceMetric;
use umap_core::rng::Mt19937;
use umap_core::sampling::EdgeSamplingCursor;

const WHEEL: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-modern-layout-ubuntu-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    inputs: BTreeMap<String, Input>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Input {
    data: Vec<Vec<f32>>,
    init: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    graph_input: String,
    parameters: Parameters,
    graph: Graph,
    final_bits: Vec<u32>,
    #[serde(default)]
    checkpoint_bits: Vec<Vec<u32>>,
    repeat_bits_equal: bool,
}

#[derive(Deserialize)]
struct Parameters {
    optimizer: Option<String>,
    init_scale_factor: Option<f64>,
    metric: Option<String>,
    n_epochs: Epochs,
    n_neighbors: Option<usize>,
    local_connectivity: Option<f64>,
    learning_rate: Option<f64>,
    a: Option<f64>,
    b: Option<f64>,
    repulsion_strength: Option<f64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Epochs {
    Scalar(usize),
    Checkpoints(Vec<usize>),
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights_bits: Vec<u32>,
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn public_modern_dense_epoch_replay() {
    let fixture: Fixture = serde_json::from_str(WHEEL).unwrap();
    let initial_pair = &fixture.cases[0].checkpoint_bits[0];
    assert_eq!(
        *initial_pair,
        bits(&[-2.0, -8.0, 0.0, -8.0, -2.0, 4.0, 4.0, 12.0])
    );

    for case in &fixture.cases {
        let p = &case.parameters;
        let input = &fixture.inputs[&case.graph_input];
        let optimizer = match p.optimizer.as_deref() {
            Some("momentum") => ModernOptimizer::Momentum,
            Some("adam") | None => ModernOptimizer::Adam,
            _ => unreachable!(),
        };
        let (n_epochs, checkpoints) = match &p.n_epochs {
            Epochs::Scalar(n) => (*n, BTreeSet::new()),
            Epochs::Checkpoints(epochs) => (
                *epochs.iter().max().unwrap(),
                epochs.iter().copied().collect(),
            ),
        };
        assert!(case.repeat_bits_equal, "{optimizer:?} {}", case.name);
        let data: Vec<_> = input.data.iter().flatten().copied().collect();
        let neighbors = p.n_neighbors.unwrap_or(2);
        let options = Some(GraphOptions::new(p.local_connectivity.unwrap_or(1.0), 1.0).unwrap());
        let graph = if p.metric.as_deref() == Some("euclidean") {
            fit_dense_graph_with_options(
                &data,
                input.data.len(),
                input.data[0].len(),
                neighbors,
                DistanceMetric::Euclidean,
                options,
            )
        } else {
            fit_precomputed_graph_with_options(&data, input.data.len(), neighbors, options)
        }
        .unwrap();
        assert_eq!(graph.indptr, case.graph.indptr);
        assert_eq!(graph.indices, case.graph.indices);
        assert_eq!(bits(&graph.data), case.graph.weights_bits);
        let positive: Vec<_> = graph
            .indptr
            .windows(2)
            .enumerate()
            .flat_map(|(row, p)| graph.indices[p[0]..p[1]].iter().map(move |&col| (row, col)))
            .collect();
        let mut cursors: Vec<_> = graph
            .data
            .iter()
            .map(|&weight| EdgeSamplingCursor::from_graph_weights(1.0, weight, 1.0).unwrap())
            .collect();
        let explicit: Vec<_> = input.init.iter().flatten().copied().collect();
        let initial = initialize_modern_layout(
            &graph,
            ModernInit::Explicit(&explicit),
            ModernInitConfig {
                n_components: input.init[0].len(),
                rescale: ModernInitRescale::GraphEdge,
                scale_factor: p.init_scale_factor.unwrap_or(2.0),
            },
            &mut Mt19937::new(42),
            &mut None,
        )
        .unwrap();
        assert_eq!(
            &bits(&initial),
            case.checkpoint_bits.first().unwrap_or(initial_pair),
            "{}: initial layout",
            case.name,
        );
        let mut layout = ModernLayout::new(
            initial,
            ModernLayoutConfig {
                optimizer,
                n_components: input.init[0].len(),
                n_epochs,
                learning_rate: p.learning_rate.unwrap_or(1.0),
                a: p.a.unwrap_or(1.0),
                b: p.b.unwrap_or(1.0),
                repulsion_strength: p.repulsion_strength.unwrap_or(0.0),
            },
        )
        .unwrap();

        let mut reported = Vec::new();
        for epoch in 0..n_epochs {
            let positive: Vec<_> = positive
                .iter()
                .zip(&mut cursors)
                .filter_map(|(&edge, cursor)| {
                    cursor.positive_due(epoch as f64).unwrap().then_some(edge)
                })
                .collect();
            // The cycle graph has one eligible target per row. Its observed
            // negative counts follow from the separately isolated pair draw.
            let repeats = match (case.name.as_str(), epoch) {
                ("square-repulsion-single" | "cycle-cancellation", 1) => 2,
                ("cycle-repulsion" | "cycle-repulsion-2d" | "square-repulsion", 1..=4) => {
                    [4, 6, 6, 8][epoch - 1]
                }
                _ => 0,
            };
            let negative = if case.name == "pairs-single-negative" && epoch == 3 {
                let target = if optimizer == ModernOptimizer::Adam {
                    3
                } else {
                    2
                };
                vec![(0, target), (1, target), (2, 0), (3, 0)]
            } else {
                [(0, 2), (1, 3), (2, 0), (3, 1)].repeat(repeats)
            };
            assert_eq!(layout.next_epoch(), epoch);
            let snapshot = layout.step(&positive, &negative).unwrap();
            if checkpoints.contains(&epoch) {
                reported.push(bits(snapshot));
            }
        }
        let final_bits = bits(&layout.into_coordinates());
        if checkpoints.contains(&n_epochs) {
            reported.push(final_bits.clone());
        }
        let max_float32_steps = match (optimizer, case.name.as_str()) {
            (ModernOptimizer::Momentum, "cycle-attraction" | "cycle-repulsion") => 1,
            (ModernOptimizer::Momentum, "cycle-cancellation") => 64,
            (ModernOptimizer::Momentum, "cycle-repulsion-2d") => 3,
            (ModernOptimizer::Adam, "square-repulsion-single" | "cycle-repulsion-2d") => 1,
            (ModernOptimizer::Adam, "cycle-cancellation") => 2,
            _ => 0,
        };
        let context = format!("{optimizer:?} {} ({n_epochs} epochs)", case.name);
        assert_eq!(reported.len(), case.checkpoint_bits.len(), "{context}");
        for (actual, expected) in reported
            .iter()
            .zip(&case.checkpoint_bits)
            .chain(std::iter::once((&final_bits, &case.final_bits)))
        {
            if max_float32_steps == 0 {
                assert_eq!(actual, expected, "{context}");
            } else {
                assert_eq!(actual.len(), expected.len(), "{context}");
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        actual.abs_diff(*expected) <= max_float32_steps,
                        "{context}: {actual:#010x} differs from {expected:#010x}"
                    );
                }
            }
        }
    }
}

#[test]
fn rejected_epoch_can_be_retried_with_unchanged_state() {
    let config = ModernLayoutConfig {
        optimizer: ModernOptimizer::Adam,
        n_components: 1,
        n_epochs: 3,
        learning_rate: 1.0,
        a: 1.0,
        b: 1.0,
        repulsion_strength: 1.0,
    };
    let coordinates = vec![0.0, 1.0, f32::MAX];
    let mut retry = ModernLayout::new(coordinates.clone(), config).unwrap();
    let mut clean = ModernLayout::new(coordinates, config).unwrap();
    retry.step(&[], &[]).unwrap();
    clean.step(&[], &[]).unwrap();
    retry.step(&[(0, 1)], &[]).unwrap();
    clean.step(&[(0, 1)], &[]).unwrap();
    assert_eq!(
        retry.step(&[(0, 1), (1, 4)], &[]),
        Err(ModernLayoutError::InvalidRow { row: 4, n_rows: 3 })
    );
    assert_eq!(
        retry.step(&[(0, 1), (1, 2)], &[]),
        Err(ModernLayoutError::NonFiniteArithmetic)
    );
    assert_eq!(retry.next_epoch(), 2);
    assert_eq!(retry.coordinates(), clean.coordinates());
    assert_eq!(
        retry.step(&[(0, 1)], &[(1, 0)]).unwrap(),
        clean.step(&[(0, 1)], &[(1, 0)]).unwrap()
    );
    assert_eq!(retry.step(&[], &[]), Err(ModernLayoutError::EpochLimit));
}
