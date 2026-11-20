use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use super::{Mt19937, NegativeSamplerError, SerialNegativeSampler, draw_negative_seed};
use crate::initialization::{normalize_columns, random_initialization_with_rng};

const STREAM_FIXTURE: &str = include_str!(
    "../../../../reference/fixtures/umap-learn-0.6.0-negative-stream-windows-x86_64.json"
);
const EARLIER_FIXTURE: &str = include_str!(
    "../../../../reference/fixtures/umap-learn-0.6.0-negative-sampling-windows-x86_64.json"
);

#[derive(Deserialize)]
struct StreamFixture {
    held_out_counts: [usize; 6],
    input: StreamInput,
    cases: BTreeMap<String, StreamCase>,
}

#[derive(Deserialize)]
struct StreamInput {
    explicit_init_prefix_atlas_float32: Vec<[f32; 2]>,
}

#[derive(Deserialize)]
struct StreamCase {
    caller_seed: u32,
    head_x_normalized_float32: [f32; 2],
    injected_mt_keys_uint32: Vec<u32>,
    caller_random_state: CallerObservation,
}

#[derive(Deserialize)]
struct CallerObservation {
    matched_raw_uint32_words: Vec<u32>,
}

struct ExpectedStream {
    name: &'static str,
    words: [[i32; 3]; 2],
    held_out_indices: [[[usize; 3]; 2]; 6],
}

// Decoded from the committed coordinate sweeps by uniquely replaying the
// next pair of targets at each rate. The nine coprime counts reconstruct the
// signed words and the six separate counts supply the held-out indices.
const EXPECTED_STREAMS: [ExpectedStream; 8] = [
    ExpectedStream {
        name: "seed42_reference",
        words: [
            [1_512_524_284, -1_869_814_856, -86_469_464],
            [-1_574_483_460, 797_748_276, 517_968_520],
        ],
        held_out_indices: [
            [[4, 4, 4], [0, 0, 4]],
            [[12, 8, 8], [12, 4, 8]],
            [[19, 7, 10], [3, 3, 13]],
            [[28, 24, 8], [28, 20, 8]],
            [[19, 14, 31], [10, 31, 20]],
            [[26, 13, 17], [36, 8, 7]],
        ],
    },
    ExpectedStream {
        name: "seed7_reference",
        words: [
            [-1_475_731_520, 968_143_481, 922_552_744],
            [-1_609_949_248, 2_034_958_957, -755_365_496],
        ],
        held_out_indices: [
            [[4, 5, 4], [8, 1, 4]],
            [[0, 9, 8], [0, 13, 8]],
            [[10, 11, 1], [2, 10, 10]],
            [[0, 25, 8], [0, 13, 8]],
            [[10, 11, 29], [2, 17, 24]],
            [[12, 1, 35], [6, 27, 12]],
        ],
    },
    ExpectedStream {
        name: "seed42_adjacent_x",
        words: [
            [1_512_524_284, -1_869_814_856, -86_469_464],
            [-1_573_171_796, 801_945_252, 568_246_940],
        ],
        held_out_indices: [
            [[4, 4, 4], [4, 0, 8]],
            [[12, 8, 8], [12, 4, 12]],
            [[19, 7, 10], [7, 3, 2]],
            [[28, 24, 8], [12, 4, 28]],
            [[19, 14, 31], [14, 17, 30]],
            [[26, 13, 17], [13, 0, 15]],
        ],
    },
    ExpectedStream {
        name: "seed12345_mixed_x",
        words: [
            [-54_125_364, 1_756_923_232, -339_830_247],
            [1_861_588_316, 1_550_869_793, -1_591_114_605],
        ],
        held_out_indices: [
            [[0, 4, 9], [8, 5, 3]],
            [[12, 0, 9], [12, 1, 3]],
            [[15, 1, 6], [14, 11, 18]],
            [[12, 0, 25], [28, 1, 19]],
            [[1, 22, 13], [21, 18, 25]],
            [[12, 25, 27], [27, 30, 21]],
        ],
    },
    ExpectedStream {
        name: "seed20261121_mixed_x",
        words: [
            [2_026_972_205, 283_458_012, -256_441_712],
            [22_094_893, 1_155_921_379, 2_086_830_684],
        ],
        held_out_indices: [
            [[5, 0, 4], [1, 7, 0]],
            [[13, 12, 0], [13, 3, 12]],
            [[20, 12, 19], [16, 4, 15]],
            [[13, 28, 16], [13, 3, 28]],
            [[20, 12, 33], [23, 4, 29]],
            [[21, 13, 34], [10, 13, 11]],
        ],
    },
    ExpectedStream {
        name: "seed42_high_accepted_word",
        words: [
            [1_982_123_768, 642_991_783, 1_111_776_564],
            [270_847_736, 1_470_916_888, -1_900_561_861],
        ],
        held_out_indices: [
            [[8, 7, 0], [8, 4, 11]],
            [[8, 7, 4], [8, 8, 11]],
            [[2, 7, 3], [5, 7, 5]],
            [[24, 7, 20], [24, 24, 27]],
            [[23, 28, 24], [26, 28, 19]],
            [[24, 11, 9], [3, 18, 8]],
        ],
    },
    ExpectedStream {
        name: "seed42_repeated_midpoint_words",
        words: [[0, 0, 0], [-2_113_929_216, 822_083_589, 554_893_360]],
        held_out_indices: [
            [[0, 0, 0], [0, 9, 4]],
            [[0, 0, 0], [0, 5, 0]],
            [[0, 0, 0], [0, 12, 7]],
            [[0, 0, 0], [0, 5, 16]],
            [[0, 0, 0], [14, 19, 0]],
            [[0, 0, 0], [35, 14, 31]],
        ],
    },
    ExpectedStream {
        name: "seed42_mixed_rejection_boundary",
        words: [
            [1_048_512, 3_456, -469_776_497],
            [-812_646_464, -2_097_191_384, 1_269_448_883],
        ],
        held_out_indices: [
            [[0, 0, 7], [4, 4, 11]],
            [[0, 0, 15], [0, 8, 3]],
            [[3, 12, 13], [7, 13, 17]],
            [[0, 0, 15], [0, 8, 19]],
            [[17, 26, 13], [21, 6, 3]],
            [[6, 15, 34], [2, 29, 10]],
        ],
    },
];

fn stream_case_sampler(
    fixture: &StreamFixture,
    case: &StreamCase,
    n_rows: usize,
) -> SerialNegativeSampler {
    let mut layout = fixture.input.explicit_init_prefix_atlas_float32[..n_rows].to_vec();
    layout[0][0] = case.head_x_normalized_float32[0];
    layout[1][0] = case.head_x_normalized_float32[1];
    let mut caller = Mt19937::new(case.caller_seed);
    let mut injected = case.injected_mt_keys_uint32.iter().map(|key| match key {
        0 => 0,
        0x02ff_df78 => 0xffff_fffc,
        0x12dd_9bb1 => 0xffff_fffd,
        0x92cd_b9b7 => 0x7fff_ffff,
        _ => panic!("unrecognized calibrated MT key {key:#x}"),
    });
    // The public generator warmed the MT state by one draw before injecting
    // the calibrated keys. Here inject their corresponding tempered words.
    if !case.injected_mt_keys_uint32.is_empty() {
        caller.next_u32();
    }
    let mut actual_raw = Vec::new();
    let sampler = SerialNegativeSampler::from_raw_words(layout.as_flattened(), n_rows, 2, || {
        let natural = caller.next_u32();
        let word = injected.next().unwrap_or(natural);
        actual_raw.push(word);
        word
    })
    .unwrap();
    assert_eq!(
        actual_raw,
        case.caller_random_state.matched_raw_uint32_words
    );
    sampler
}

#[test]
fn matches_public_signed_words_and_held_out_indices() {
    let fixture: StreamFixture = serde_json::from_str(STREAM_FIXTURE).unwrap();
    assert_eq!(fixture.cases.len(), EXPECTED_STREAMS.len());
    for expected in EXPECTED_STREAMS {
        let case = &fixture.cases[expected.name];
        let mut sampler = stream_case_sampler(&fixture, case, fixture.held_out_counts[0]);
        let actual = [0, 1].map(|head| [(); 3].map(|()| sampler.next_word(head).unwrap()));
        assert_eq!(actual, expected.words, "{}", expected.name);

        for (n_rows, expected_indices) in fixture
            .held_out_counts
            .into_iter()
            .zip(expected.held_out_indices)
        {
            let mut sampler = stream_case_sampler(&fixture, case, n_rows);
            let actual = [0, 1].map(|head| [(); 3].map(|()| sampler.sample(head).unwrap()));
            assert_eq!(
                actual, expected_indices,
                "{} with {n_rows} rows",
                expected.name
            );
        }
    }
}

#[test]
fn accepts_only_raw_words_below_the_numpy_limit() {
    let mut words = [
        0xffff_fffd,
        0,
        0xffff_fffe,
        0x7fff_ffff,
        0xffff_ffff,
        0xffff_fffc,
        42,
    ]
    .into_iter();
    let seed = draw_negative_seed(|| words.next().unwrap());
    assert_eq!(seed, [-2_147_483_647, 0, 2_147_483_645]);
    assert_eq!(words.next(), Some(42));
}

#[test]
fn uses_signed_remainders_and_supports_one_vertex() {
    let mut sampler = SerialNegativeSampler::new(
        &mut Mt19937::new(42),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        7,
        1,
    )
    .unwrap();
    assert_eq!(sampler.clone().next_word(1), Some(-1_574_483_460));
    assert_eq!(sampler.sample(1), Some(3));

    let mut singleton = SerialNegativeSampler::new(&mut Mt19937::new(42), &[0.0], 1, 1).unwrap();
    assert_eq!(singleton.sample(1), None);
    assert_eq!(singleton.clone().next_word(0), Some(1_512_524_284));
    assert_eq!([(); 3].map(|()| singleton.sample(0)), [Some(0); 3]);
    assert_eq!(singleton.next_word(0), Some(604_100_009));
}

#[test]
fn keeps_the_coordinate_sign_bit_and_wraps_seed_addition() {
    let mut midpoint =
        SerialNegativeSampler::from_raw_words(&[0.0, -0.0], 2, 1, || 0x7fff_ffff).unwrap();
    assert_eq!(
        [0, 1].map(|head| [(); 3].map(|()| midpoint.next_word(head).unwrap())),
        [[0, 0, 0], [0, 33_546_240, 1_073_610_816]]
    );

    let mut raw = [0, 0x7fff_ffff, 0xffff_fffc].into_iter();
    let mut mixed =
        SerialNegativeSampler::from_raw_words(&[-0.0], 1, 1, || raw.next().unwrap()).unwrap();
    assert_eq!(
        [(); 3].map(|()| mixed.next_word(0).unwrap()),
        [1_044_480, 16_780_863, 570_564_560]
    );
}

#[test]
fn rejects_invalid_layout_before_consuming_caller_words() {
    let mut caller = Mt19937::new(42);
    for (layout, n_rows, n_components) in [
        ([].as_slice(), 0, 1),
        ([].as_slice(), 1, 0),
        ([].as_slice(), usize::MAX, 2),
        ([0.0].as_slice(), 1, 2),
    ] {
        assert_eq!(
            SerialNegativeSampler::new(&mut caller, layout, n_rows, n_components).unwrap_err(),
            NegativeSamplerError::InvalidShape {
                n_rows,
                n_components,
                values: layout.len(),
            }
        );
    }
    for coordinate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            SerialNegativeSampler::new(&mut caller, &[0.0, 1.0, coordinate, 2.0], 2, 2),
            Err(NegativeSamplerError::NonFiniteHeadCoordinate { head: 1, value })
                if value.to_bits() == coordinate.to_bits()
        ));
    }
    assert_eq!(caller.next_u32(), 1_608_637_542);
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
}

fn assert_targets(sampler: &mut SerialNegativeSampler, graph: &Value, decode: &Value) {
    let graph = Graph::deserialize(graph).unwrap();
    let targets = BTreeMap::<String, Vec<usize>>::deserialize(&decode["targets_by_edge"]).unwrap();
    assert_eq!(graph.indices.len(), targets.len());
    for (head, bounds) in graph.indptr.windows(2).enumerate() {
        for tail in &graph.indices[bounds[0]..bounds[1]] {
            for &expected in &targets[&format!("{head}->{tail}")] {
                assert_eq!(sampler.sample(head), Some(expected), "{head}->{tail}");
            }
        }
    }
}

#[test]
fn carries_independent_head_streams_across_edges_and_epochs() {
    let earlier: Value = serde_json::from_str(EARLIER_FIXTURE).unwrap();
    let scenario = &earlier["isolated_pair_scenarios"]["base"];
    let layout = <[[f32; 2]; 6]>::deserialize(&scenario["init"]).unwrap();
    let mut sampler =
        SerialNegativeSampler::new(&mut Mt19937::new(42), layout.as_flattened(), 6, 2).unwrap();
    assert_targets(
        &mut sampler,
        &scenario["graph"],
        &scenario["epoch_two_by_rate"]["2"]["interleaved_decode"],
    );
    assert_targets(
        &mut sampler,
        &scenario["graph"],
        &scenario["epoch_three_by_rate"]["2"]["last_epoch_decode"],
    );

    for name in ["both_edges", "later_edge_only"] {
        let scenario = &earlier["two_edges_per_head"][name];
        let mut sampler =
            SerialNegativeSampler::new(&mut Mt19937::new(42), layout.as_flattened(), 6, 2).unwrap();
        assert_targets(
            &mut sampler,
            &scenario["graph"],
            &scenario["epoch_two_by_rate"]["2"]["interleaved_decode"],
        );
    }
}

#[test]
fn caller_rng_starts_after_the_random_initialization_words() {
    let earlier: Value = serde_json::from_str(EARLIER_FIXTURE).unwrap();
    let scenarios = &earlier["isolated_pair_scenarios"];
    let mut random_caller = Mt19937::new(42);
    let random_layout = random_initialization_with_rng(&mut random_caller, 6, 2).unwrap();
    let mut explicit_layout =
        <[[f32; 2]; 6]>::deserialize(&scenarios["explicit_random_values_unadvanced"]["init"])
            .unwrap();
    normalize_columns(explicit_layout.as_flattened_mut(), 6, 2).unwrap();

    for (name, mut caller, layout, init_words) in [
        (
            "random_init",
            random_caller.clone(),
            random_layout.as_slice(),
            24,
        ),
        (
            "explicit_random_values_advanced",
            random_caller,
            explicit_layout.as_flattened(),
            0,
        ),
        (
            "explicit_random_values_unadvanced",
            Mt19937::new(42),
            explicit_layout.as_flattened(),
            0,
        ),
    ] {
        let scenario = &scenarios[name];
        let observed_layout =
            <[[u32; 2]; 6]>::deserialize(&scenario["zero_epoch"]["embedding_bits_uint32"]).unwrap();
        assert!(
            layout
                .iter()
                .map(|value| value.to_bits())
                .eq(observed_layout.into_iter().flatten())
        );

        let observed_raw = CallerObservation::deserialize(&scenario["zero_epoch"]["random_state"])
            .unwrap()
            .matched_raw_uint32_words;
        let seed_words = &observed_raw[init_words..];
        assert_eq!(seed_words.len(), 3);
        let mut caller_probe = caller.clone();
        assert_eq!(
            [(); 3].map(|()| caller_probe.next_u32()),
            seed_words,
            "{name}"
        );
        let mut sampler = SerialNegativeSampler::new(&mut caller, layout, 6, 2).unwrap();
        assert_eq!(caller.next_u32(), caller_probe.next_u32(), "{name}");
        assert_targets(
            &mut sampler,
            &scenario["graph"],
            &scenario["epoch_two_by_rate"]["2"]["interleaved_decode"],
        );
    }
}
