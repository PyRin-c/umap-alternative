//! Replay public layout checkpoints with the destinations already decoded in
//! the reference fixture. Some destinations are unique only with a prefix
//! from another rate. The replay uses the recorded choices as-is.

use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::optimize::optimize_csr;

const NEGATIVE_SAMPLING: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-negative-sampling-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Fixture {
    parameters: Parameters,
    isolated_pair_scenarios: BTreeMap<String, Scenario>,
    two_edges_per_head: MultiEdgeScenarios,
    index_reduction: IndexReduction,
}

#[derive(Deserialize)]
struct Parameters {
    n_components: usize,
    a: f64,
    b: f64,
    learning_rate: f64,
}

#[derive(Deserialize)]
struct MultiEdgeScenarios {
    both_edges: Scenario,
    later_edge_only: Scenario,
}

#[derive(Deserialize)]
struct Scenario {
    graph: Graph,
    zero_epoch: Checkpoint,
    all_rows_normalized_init_bits_uint32: Option<Vec<[u32; 2]>>,
    epoch_two_by_rate: BTreeMap<u8, EpochTwo>,
    #[serde(default)]
    epoch_three_by_rate: BTreeMap<u8, EpochThree>,
}

#[derive(Clone, Deserialize)]
struct Graph {
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

impl Graph {
    fn into_csr(self) -> CsrGraph {
        CsrGraph {
            n_rows: self.shape[0],
            n_cols: self.shape[1],
            indptr: self.indptr,
            indices: self.indices,
            data: self
                .data_bits_uint32
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct Checkpoint {
    embedding_bits_uint32: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
struct EpochTwo {
    #[serde(flatten)]
    checkpoint: Checkpoint,
    interleaved_decode: Decode,
}

#[derive(Deserialize)]
struct EpochThree {
    #[serde(flatten)]
    checkpoint: Checkpoint,
    last_epoch_decode: Decode,
}

#[derive(Deserialize)]
struct Decode {
    targets_by_edge: BTreeMap<String, Vec<usize>>,
}

impl Decode {
    fn append_schedule(&self, graph: &CsrGraph, draws: &mut Vec<(usize, usize)>) {
        for (head, offsets) in graph.indptr.windows(2).enumerate() {
            for tail in &graph.indices[offsets[0]..offsets[1]] {
                let targets = &self.targets_by_edge[&format!("{head}->{tail}")];
                draws.extend(targets.iter().map(|&target| (head, target)));
            }
        }
    }
}

#[derive(Deserialize)]
struct IndexReduction {
    init_prefix_atlas_float32: Vec<[f32; 2]>,
    seeds: BTreeMap<String, SeedObservations>,
}

#[derive(Deserialize)]
struct SeedObservations {
    observations_by_vertex_count: BTreeMap<usize, IndexReductionCheckpoint>,
}

#[derive(Deserialize)]
struct IndexReductionCheckpoint {
    negative_indices_in_head_order: [usize; 2],
    active_embedding_bits_uint32: Vec<[u32; 2]>,
}

struct Replay<'a> {
    parameters: &'a Parameters,
    graph: &'a CsrGraph,
    initial: &'a [[u32; 2]],
}

impl Replay<'_> {
    fn run(&self, n_epochs: usize, rate: u8, gamma: f64, draws: &[(usize, usize)]) -> Vec<f32> {
        let mut embedding = self
            .initial
            .iter()
            .flatten()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        let mut draws = draws.iter();
        optimize_csr(
            &mut embedding,
            self.parameters.n_components,
            self.graph,
            self.parameters.a,
            self.parameters.b,
            self.parameters.learning_rate,
            gamma,
            f64::from(rate),
            n_epochs,
            |source| {
                let &(expected_source, target) = draws.next().expect("unexpected negative draw");
                assert_eq!(source, expected_source, "negative draw head");
                target
            },
        )
        .unwrap();
        assert!(draws.next().is_none(), "missing negative draw");
        embedding
    }
}

#[derive(Default)]
struct Comparison {
    matched: usize,
    total: usize,
    first_mismatch: Option<String>,
}

impl Comparison {
    fn record(&mut self, case: &str, initial: &[[u32; 2]], actual: &[f32], expected: &[[u32; 2]]) {
        assert_eq!(actual.len(), initial.len() * 2);
        assert!(expected.len() <= initial.len());
        for (position, (&value, &initial_bits)) in
            actual.iter().zip(initial.iter().flatten()).enumerate()
        {
            let expected_bits = expected.get(position / 2).map(|row| row[position % 2]);
            if let Some(bits) = expected_bits.filter(|&bits| f32::from_bits(bits).is_finite()) {
                self.total += 1;
                if value.to_bits() == bits {
                    self.matched += 1;
                } else if self.first_mismatch.is_none() {
                    self.first_mismatch = Some(format!(
                        "{case}, row {}, component {}: {:#x} != {bits:#x}",
                        position / 2,
                        position % 2,
                        value.to_bits(),
                    ));
                }
            } else {
                // Public output masks disconnected rows with NaNs. They can
                // still be sampled, so the optimizer uses their finite init.
                assert_eq!(
                    value.to_bits(),
                    initial_bits,
                    "{case}: inactive row changed"
                );
            }
        }
    }

    fn assert_exact(&self, expected_total: usize) {
        assert_eq!(self.total, expected_total);
        assert_eq!(
            self.matched, self.total,
            "first differing coordinate {:?}",
            self.first_mismatch
        );
    }
}

fn replay_scenario(
    comparison: &mut Comparison,
    parameters: &Parameters,
    name: &str,
    case: Scenario,
) {
    let graph = case.graph.into_csr();
    let initial = case
        .all_rows_normalized_init_bits_uint32
        .as_deref()
        .unwrap_or(&case.zero_epoch.embedding_bits_uint32);
    let replay = Replay {
        parameters,
        graph: &graph,
        initial,
    };
    for (&rate, epoch_two) in &case.epoch_two_by_rate {
        let mut draws = Vec::new();
        epoch_two
            .interleaved_decode
            .append_schedule(&graph, &mut draws);
        let actual = replay.run(2, rate, 1.0, &draws);
        comparison.record(
            &format!("{name}, 2 epochs, rate {rate}"),
            initial,
            &actual,
            &epoch_two.checkpoint.embedding_bits_uint32,
        );
    }
    for (&rate, epoch_three) in &case.epoch_three_by_rate {
        let mut draws = Vec::new();
        // Rate 1 has no draw in the first active epoch. Rate 2 carries over
        // the same first pass as the corresponding two-epoch checkpoint.
        if let Some(epoch_two) = case.epoch_two_by_rate.get(&rate) {
            epoch_two
                .interleaved_decode
                .append_schedule(&graph, &mut draws);
        }
        epoch_three
            .last_epoch_decode
            .append_schedule(&graph, &mut draws);
        let actual = replay.run(3, rate, 1.0, &draws);
        comparison.record(
            &format!("{name}, 3 epochs, rate {rate}"),
            initial,
            &actual,
            &epoch_three.checkpoint.embedding_bits_uint32,
        );
    }
}

#[test]
fn decoded_public_schedules_replay_bit_for_bit() {
    let fixture: Fixture = serde_json::from_str(NEGATIVE_SAMPLING).unwrap();
    let mut comparison = Comparison::default();
    for (name, case) in fixture.isolated_pair_scenarios {
        replay_scenario(&mut comparison, &fixture.parameters, &name, case);
    }
    for (name, case) in [
        ("both edges per head", fixture.two_edges_per_head.both_edges),
        (
            "later edge only",
            fixture.two_edges_per_head.later_edge_only,
        ),
    ] {
        replay_scenario(&mut comparison, &fixture.parameters, name, case);
    }

    // These probes also exercise targets outside the positive graph, at
    // thirteen row counts and two seeds. Only the first two rows are active.
    let atlas = fixture
        .index_reduction
        .init_prefix_atlas_float32
        .into_iter()
        .map(|row| row.map(f32::to_bits))
        .collect::<Vec<_>>();
    for (seed, observations) in fixture.index_reduction.seeds {
        for (n_rows, checkpoint) in observations.observations_by_vertex_count {
            let mut indptr = vec![2; n_rows + 1];
            indptr[..2].copy_from_slice(&[0, 1]);
            let graph = CsrGraph {
                n_rows,
                n_cols: n_rows,
                indptr,
                indices: vec![1, 0],
                data: vec![1.0, 1.0],
            };
            let initial = &atlas[..n_rows];
            let replay = Replay {
                parameters: &fixture.parameters,
                graph: &graph,
                initial,
            };
            let [first, second] = checkpoint.negative_indices_in_head_order;
            let actual = replay.run(2, 2, 1.0, &[(0, first), (1, second)]);
            comparison.record(
                &format!("index reduction, seed {seed}, {n_rows} rows"),
                initial,
                &actual,
                &checkpoint.active_embedding_bits_uint32,
            );
        }
    }
    comparison.assert_exact(342);
}

#[test]
fn public_negative_updates_cover_non_unit_curves_and_nearby_targets() {
    // Additional public fit_transform calls used the reference's precomputed
    // six-row matrix and settings, with the overrides below. The two rate-2
    // one-draw controls identified first destinations by enumerating all
    // choices. The rate-3 fit decodes uniquely with the rate-2 draws as a
    // prefix. The last pass of the three-epoch fit decodes uniquely with the
    // second rate-3 draws as a prefix.
    let fixture: Fixture = serde_json::from_str(NEGATIVE_SAMPLING).unwrap();
    let base = &fixture.isolated_pair_scenarios["base"];
    let graph = base.graph.clone().into_csr();
    let initial = &base.zero_epoch.embedding_bits_uint32;
    let parameters_a = Parameters {
        n_components: 2,
        a: 1.7,
        b: 0.9,
        learning_rate: 0.1,
    };
    let parameters_b = Parameters {
        n_components: 2,
        a: 0.82,
        b: 1.3,
        learning_rate: 0.2,
    };
    let replay_a = Replay {
        parameters: &parameters_a,
        graph: &graph,
        initial,
    };
    let replay_b = Replay {
        parameters: &parameters_b,
        graph: &graph,
        initial,
    };
    let mut comparison = Comparison::default();
    let actual = replay_a.run(
        2,
        3,
        0.75,
        &[
            (0, 4),
            (0, 4),
            (1, 0),
            (1, 0),
            (2, 0),
            (2, 5),
            (3, 4),
            (3, 1),
            (4, 0),
            (4, 2),
            (5, 2),
            (5, 1),
        ],
    );
    comparison.record(
        "non-unit curve, seed 42, 2 epochs, rate 3",
        initial,
        &actual,
        &[
            [1017240787, 1034336855],
            [1065023336, 1081800650],
            [1075924467, 1065744428],
            [1084194403, 1090427141],
            [1086374243, 1078100895],
            [1092593357, 1092576101],
        ],
    );
    let actual = replay_b.run(
        3,
        2,
        1.3,
        &[
            (0, 4),
            (1, 2),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 4),
            (0, 5),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 4),
            (2, 5),
            (3, 0),
            (3, 1),
            (4, 5),
            (4, 5),
            (5, 0),
            (5, 5),
        ],
    );
    comparison.record(
        "non-unit curve, seed 7, 3 epochs, rate 2",
        initial,
        &actual,
        &[
            [1037474884, 1054467570],
            [1063446477, 1080387792],
            [1076159968, 1067211247],
            [1084060871, 1090057686],
            [1086550992, 1078728022],
            [1092502999, 1092418074],
        ],
    );

    // Moving row 4 to [0.03, 0.11] makes both components of head 0's first
    // negative force reach the clipping limit after its attraction.
    let mut near_initial = initial.clone();
    near_initial[4] = [1022739086, 1038174126];
    let near_replay = Replay {
        initial: &near_initial,
        ..replay_a
    };
    let first_draws = [(0, 4), (1, 0), (2, 0), (3, 4), (4, 0), (5, 2)];
    let actual = near_replay.run(2, 2, 0.75, &first_draws);
    comparison.record(
        "near target, seed 42, 2 epochs, rate 2",
        &near_initial,
        &actual,
        &[
            [3200362873, 3198512471],
            [1064994182, 1081812534],
            [1075917405, 1065741327],
            [1084194032, 1090424844],
            [1041056531, 1046772687],
            [1092597150, 1092597320],
        ],
    );
    comparison.assert_exact(36);
}

#[test]
fn coincident_targets_are_consistent_with_the_public_nearby_control() {
    let fixture: Fixture = serde_json::from_str(NEGATIVE_SAMPLING).unwrap();
    let base = &fixture.isolated_pair_scenarios["base"];
    let graph = base.graph.clone().into_csr();
    let mut initial = base.zero_epoch.embedding_bits_uint32.clone();
    let parameters = Parameters {
        n_components: 2,
        a: 1.0,
        b: 1.0,
        learning_rate: 1.25,
    };
    initial[1] = [2.0_f32.to_bits(), 0];
    // These two public seed-42 fits use two epochs and rate 2. The nearby
    // case determines each destination. Setting the target's y to zero
    // makes rows 0, 1, and 4 coincide after the first positive update.
    // The public output then leaves the exact destinations underdetermined.
    let draws = [(0, 4), (1, 4), (2, 0), (3, 4), (4, 0), (5, 2)];
    let mut comparison = Comparison::default();
    initial[4] = [1.0_f32.to_bits(), 1008981770];
    let replay = Replay {
        parameters: &parameters,
        graph: &graph,
        initial: &initial,
    };
    let actual = replay.run(2, 2, 1.0, &draws);
    comparison.record(
        "near coincidence control",
        &initial,
        &actual,
        &[
            [1065353216, 3230702986],
            [1065353216, 3230807499],
            [1076827901, 1070910235],
            [1083743467, 1089160201],
            [1067466680, 1050660448],
            [1092354061, 1092325631],
        ],
    );
    initial[4][1] = 0;
    let replay = Replay {
        parameters: &parameters,
        graph: &graph,
        initial: &initial,
    };
    let actual = replay.run(2, 2, 1.0, &draws);
    comparison.record(
        "distinct coincident target",
        &initial,
        &actual,
        &[
            [1065353216, 0],
            [1065353216, 0],
            [1077539214, 1072113846],
            [1083749942, 1089136766],
            [1087109915, 1085177861],
            [1092225182, 1092142873],
        ],
    );
    comparison.assert_exact(24);
}
