use serde::Deserialize;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::modern_layout::{ModernLayoutConfig, ModernLayoutError, ModernOptimizer};
use umap_core::rng::Mt19937;
use umap_core::sampling::{ModernSampledLayout, ModernSamplingConfig, ModernSamplingError};

const WHEEL: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-modern-sampling-ubuntu-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    learning_rate: f64,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    seed: u32,
    n_rows: usize,
    n_components: usize,
    n_epochs: usize,
    negative_sample_rate: f64,
    exclude_graph_neighbors: bool,
    negative_selection_range: usize,
    curve: [f64; 2],
    graph: Graph,
    momentum_bits: Vec<Vec<u32>>,
    adam_bits: Option<Vec<Vec<u32>>>,
    positive: Vec<Vec<usize>>,
    negative: Vec<Vec<usize>>,
    seed_draw: [i64; 4],
    random_epochs: Vec<RandomEpoch>,
    adam_random_epochs: Option<Vec<RandomEpoch>>,
    continuation: RandomContinuation,
    adam_continuation: Option<RandomContinuation>,
    decode_error: f64,
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct RandomEpoch {
    shuffle_cursor: usize,
    normal_cursor: Option<usize>,
}

#[derive(Deserialize)]
struct RandomContinuation {
    cursor: usize,
    gaussian_bits: Option<u64>,
    next_words: [u32; 4],
}

fn check_coordinates(actual: &[f32], expected: &[u32], context: &str) {
    let actual: Vec<_> = actual
        .iter()
        .map(|coordinate| coordinate.to_bits())
        .collect();
    assert_eq!(actual, expected, "{context}");
}

fn run_fixture(optimizer: ModernOptimizer) {
    let fixture: Fixture = serde_json::from_str(WHEEL).unwrap();
    let mut cases = 0;
    let mut epochs = 0;
    let mut total_positive = 0;
    let mut total_negative = 0;
    let mut coordinate_count = 0;
    for case in fixture.cases {
        let (snapshots, events, continuation) = match optimizer {
            ModernOptimizer::Momentum => {
                (case.momentum_bits, case.random_epochs, case.continuation)
            }
            ModernOptimizer::Adam => {
                let Some(snapshots) = case.adam_bits else {
                    continue;
                };
                (
                    snapshots,
                    case.adam_random_epochs.unwrap(),
                    case.adam_continuation.unwrap(),
                )
            }
        };
        assert!(case.decode_error < 0.02, "{}", case.name);
        let initial = snapshots[0].iter().copied().map(f32::from_bits).collect();
        let graph = CsrGraph {
            n_rows: case.n_rows,
            n_cols: case.n_rows,
            indptr: case.graph.indptr,
            indices: case.graph.indices,
            data: case
                .graph
                .weights_bits
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        };
        let layout_config = ModernLayoutConfig {
            optimizer,
            n_components: case.n_components,
            n_epochs: case.n_epochs,
            learning_rate: fixture.learning_rate,
            a: case.curve[0],
            b: case.curve[1],
            repulsion_strength: 1.0,
        };
        let sampling_config = ModernSamplingConfig {
            negative_sample_rate: case.negative_sample_rate,
            exclude_graph_neighbors: case.exclude_graph_neighbors,
            negative_selection_range: case.negative_selection_range,
        };
        let mut rng = Mt19937::new(case.seed);
        let mut cache = None;
        let mut sampled =
            ModernSampledLayout::new(&graph, initial, layout_config, sampling_config, &mut rng)
                .unwrap();
        assert_eq!(rng.state().1, case.seed_draw[3] as usize);
        let two_shuffles =
            optimizer == ModernOptimizer::Adam && case.n_rows <= case.negative_selection_range;
        let stride = if two_shuffles { 2 } else { 1 };
        assert_eq!(events.len(), stride * case.n_epochs);
        let encode = |&(head, tail): &(usize, usize)| head * case.n_rows + tail;
        for (epoch, expected) in snapshots.iter().take(case.n_epochs).enumerate() {
            let context = format!("{optimizer:?} {} epoch {epoch}", case.name);
            assert_eq!(sampled.next_epoch(), epoch, "{context}");
            let step = sampled
                .step(&mut rng, &mut cache)
                .unwrap_or_else(|error| panic!("{context}: {error}"));
            let positions: Vec<_> = step.positive.iter().map(encode).collect();
            assert_eq!(positions, case.positive[epoch], "{context}");
            if optimizer == ModernOptimizer::Momentum {
                let mut accepted: Vec<_> = step.negative.iter().map(encode).collect();
                accepted.sort_unstable();
                assert_eq!(accepted, case.negative[epoch], "{context}");
            }
            check_coordinates(step.coordinates, expected, &context);
            coordinate_count += expected.len();
            epochs += 1;
            total_positive += step.positive.len();
            total_negative += step.negative.len();
            let event = &events[(epoch + 1) * stride - 1];
            assert_eq!(
                rng.state().1,
                event.normal_cursor.unwrap_or(event.shuffle_cursor),
                "{context}"
            );
        }
        check_coordinates(
            sampled.coordinates(),
            snapshots.last().unwrap(),
            &format!("{optimizer:?} {} final", case.name),
        );
        assert_eq!(
            sampled.step(&mut rng, &mut cache).unwrap_err(),
            ModernSamplingError::Layout(ModernLayoutError::EpochLimit)
        );
        assert_eq!(rng.state().1, continuation.cursor, "{}", case.name);
        match (cache, continuation.gaussian_bits) {
            (Some(actual), Some(expected)) => assert!(actual.to_bits().abs_diff(expected) <= 2),
            (None, None) => (),
            other => panic!("{} Gaussian cache {other:?}", case.name),
        }
        assert_eq!(
            std::array::from_fn::<_, 4, _>(|_| rng.next_u32()),
            continuation.next_words,
            "{}",
            case.name
        );
        cases += 1;
    }
    eprintln!(
        "{optimizer:?}: {cases} cases, {epochs} epochs, {total_positive} positive, {total_negative} negative, {coordinate_count} exact coordinates"
    );
}

#[test]
fn public_momentum_epochs_match_decoded_dispatch() {
    run_fixture(ModernOptimizer::Momentum);
}

#[test]
fn public_adam_epochs_match_modern_sampling_replay() {
    run_fixture(ModernOptimizer::Adam);
}

#[test]
fn ambiguous_projection_leaves_the_next_epoch_uncommitted() {
    let graph = CsrGraph {
        n_rows: 3,
        n_cols: 3,
        indptr: vec![0, 1, 3, 4],
        indices: vec![1, 0, 2, 1],
        data: vec![1.0; 4],
    };
    let mut rng = Mt19937::new(11);
    let mut cache = None;
    let mut sampled = ModernSampledLayout::new(
        &graph,
        vec![0.0, 0.0, 0.0],
        ModernLayoutConfig {
            optimizer: ModernOptimizer::Momentum,
            n_components: 1,
            n_epochs: 3,
            learning_rate: 1.0,
            a: 1.0,
            b: 1.0,
            repulsion_strength: 1.0,
        },
        ModernSamplingConfig {
            negative_selection_range: 1,
            ..ModernSamplingConfig::default()
        },
        &mut rng,
    )
    .unwrap();
    assert!(
        sampled
            .step(&mut rng, &mut cache)
            .unwrap()
            .negative
            .is_empty()
    );
    let before = (rng.state(), cache);
    for _ in 0..2 {
        assert_eq!(
            sampled.step(&mut rng, &mut cache).unwrap_err(),
            ModernSamplingError::AmbiguousProjection { epoch: 1 }
        );
        assert_eq!(sampled.next_epoch(), 1);
        assert_eq!(sampled.coordinates(), &[0.0; 3]);
        assert_eq!((rng.state(), cache), before);
    }
}
