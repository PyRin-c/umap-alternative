use serde::Deserialize;

use super::*;

#[test]
fn strict_margins_leave_the_prepared_tie_stream_in_place() {
    let data = [-1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let prepared =
        DenseApproximateQueryIndex::from_training_graph(&data, 4, 2, &[-1; 4], 1, 0, None)
            .unwrap()
            .with_projection_words(&[[0, 1]], NonZeroUsize::new(2).unwrap(), 1)
            .unwrap();
    let edge = 0.5e-8_f32;
    let beyond = edge.next_up();
    let queries = [
        0.0, 0.0, beyond, 0.0, -edge, 0.0, -beyond, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let indices = [0, 2, 0, 0, 2, 2];
    for epsilon in [0.0, 0.12] {
        let ranked = prepared.query(&queries, 6, 1, epsilon).unwrap();
        assert_eq!(ranked.indices, indices);
        assert_eq!(prepared.query(&queries, 6, 1, epsilon).unwrap(), ranked);
        assert_eq!(
            prepared
                .query(&queries[..2], 1, 1, epsilon)
                .unwrap()
                .indices,
            [0]
        );
    }
}

#[derive(Deserialize)]
struct Observations {
    controls: Vec<DrawControl>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct DrawControl {
    caller: u32,
    draws: Vec<(i64, i64, Option<usize>, serde_json::Value)>,
    ranks: Vec<usize>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    d: usize,
    caller: u32,
    cap: NonZeroUsize,
    depth: u8,
    words: Vec<[u64; 2]>,
    train_bits: Vec<u32>,
    graph: Vec<i32>,
    live: Vec<i32>,
    runs: Vec<QueryBatch>,
}

#[derive(Deserialize)]
struct QueryBatch {
    label: String,
    query_bits: Vec<u32>,
    width: usize,
    epsilon: f64,
    indices: Vec<i32>,
    bits: Vec<u32>,
}

#[test]
#[ignore = "requires the capture from reference/probe_prepared_projection_ties.py"]
fn compare_prepared_projection_ties_with_wheel() {
    let path = std::env::var_os("PREPARED_PROJECTION_TIES").unwrap();
    let observations: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut draw_matches = 0;
    for control in observations.controls {
        let [_, second, _] = control.draws.as_slice() else {
            panic!("unexpected draws for caller {}", control.caller);
        };
        let lanes = second.3.as_array().unwrap();
        let mut rng = std::array::from_fn(|i| lanes[i].as_i64().unwrap());
        for _ in 0..10 {
            sample_absolute_vertex(&mut rng, 2);
        }
        for expected in control.ranks {
            assert_eq!(sample_absolute_vertex(&mut rng, 2), expected);
            draw_matches += 1;
        }
    }
    println!("signed draw control bits {draw_matches}/{draw_matches}");
    let decode = |raw: &[u32]| raw.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
    let mut diff = Vec::new();
    for case in observations.cases {
        let data = decode(&case.train_bits);
        let prepared = DenseApproximateQueryIndex::from_training_graph(
            &data,
            case.n,
            case.d,
            &case.graph,
            5,
            case.caller,
            None,
        )
        .unwrap()
        .with_live_training_offers(&case.live, 6, None)
        .unwrap()
        .with_projection_words(&case.words, case.cap, case.depth)
        .unwrap();
        let mut slots = 0;
        let mut equal_ids = 0;
        let mut equal_distances = 0;
        let mut wheel_only = 0;
        let mut rust_only = 0;
        let mut first = None;
        for batch in case.runs {
            let q = decode(&batch.query_bits);
            let actual = prepared
                .query(&q, q.len() / case.d, batch.width, batch.epsilon)
                .unwrap();
            slots += batch.indices.len();
            equal_ids += actual
                .indices
                .iter()
                .zip(&batch.indices)
                .filter(|(a, b)| a == b)
                .count();
            equal_distances += actual
                .distances
                .iter()
                .zip(batch.bits)
                .filter(|(a, b)| a.to_bits() == *b)
                .count();
            for (row, (native, wheel)) in actual
                .indices
                .chunks_exact(batch.width)
                .zip(batch.indices.chunks_exact(batch.width))
                .enumerate()
            {
                if native != wheel {
                    let left = wheel.iter().filter(|&&id| id >= 0 && !native.contains(&id));
                    let right = native.iter().filter(|&&id| id >= 0 && !wheel.contains(&id));
                    wheel_only += left.clone().count();
                    rust_only += right.clone().count();
                    if first.is_none() {
                        first = Some((
                            batch.label.clone(),
                            batch.width,
                            batch.epsilon,
                            row,
                            left.copied().collect::<Vec<_>>(),
                            right.copied().collect::<Vec<_>>(),
                        ));
                    }
                }
            }
        }
        println!(
            "{} ordered {equal_ids}/{slots}, bits {equal_distances}/{slots}, wheel-only {wheel_only}, Rust-only {rust_only}, first {first:?}",
            case.name
        );
        if equal_ids != slots || equal_distances != slots {
            diff.push(case.name);
        }
    }
    assert!(diff.is_empty(), "{diff:?}");
}
