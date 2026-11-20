use std::hint::black_box;
use std::time::Instant;

use serde::Deserialize;

use super::*;

#[test]
fn pivot_words_skip_the_first_position_and_keep_the_partition_order() {
    let values = [9.0, 0.0, 8.0, 1.0, 7.0, 2.0];
    let data = DenseRows::training(&values, 6, 1).unwrap();
    let tree = build_word_tree(data, (0..6).collect(), 3, 2, &[[1, 1]], 0).unwrap();
    let Tree::Split { a, b, left, right } = tree else {
        panic!("expected a split");
    };
    assert_eq!((a, b), (1, 2));
    assert!(matches!(*left, Tree::Leaf(rows) if rows == [5, 1, 3]));
    assert!(matches!(*right, Tree::Leaf(rows) if rows == [4, 2, 0]));
    let tree = build_word_tree(data, (0..6).collect(), 1, 1, &[[5, 0]], 0).unwrap();
    assert!(matches!(tree, Tree::Split { a: 5, b: 0, .. }));
    assert!(matches!(
        build_word_tree(data, (0..6).collect(), 6, 1, &[], 0).unwrap(),
        Tree::Leaf(rows) if rows == [0, 1, 2, 3, 4, 5]
    ));
    assert!(matches!(
        build_word_tree(data, (0..6).collect(), 5, 1, &[], 0),
        Err(ApproximateSearchError::ProjectionWordsExhausted)
    ));
    let values = [0.0, 7.0, 9.0, 8.0, 10.0];
    let data = DenseRows::training(&values, 5, 1).unwrap();
    let tree = build_word_tree(data, (0..5).collect(), 1, 2, &[[0, 2], [1, 1], [0, 0]], 0).unwrap();
    let Tree::Split { left, right, .. } = tree else {
        panic!("expected a split");
    };
    assert!(matches!(*left, Tree::Leaf(rows) if rows == [0]));
    assert!(matches!(*right, Tree::Split { a: 2, b: 3, .. }));
}

#[derive(Deserialize)]
struct Observations {
    training: Vec<TrainingCase>,
    queries: Vec<QueryCase>,
}

#[derive(Deserialize)]
struct TrainingCase {
    name: String,
    n: usize,
    d: usize,
    cap: usize,
    depth: u8,
    train_bits: Vec<u32>,
    words: Vec<[u64; 2]>,
    pivots: Vec<[usize; 2]>,
    rows_by_leaf: Vec<Vec<usize>>,
    labels: Vec<usize>,
    matching_component_labels: usize,
    shifted_right_matches: Option<usize>,
    cross_leaf_offers: usize,
}

#[derive(Deserialize)]
struct QueryCase {
    name: String,
    n: usize,
    d: usize,
    depth: u8,
    cap: NonZeroUsize,
    train_bits: Vec<u32>,
    query_bits: Vec<u32>,
    graph: Vec<i32>,
    words: Vec<[u64; 2]>,
    training_words: Vec<[u64; 2]>,
    training_cap: NonZeroUsize,
    prepared: Vec<QuerySweep>,
    plain: Vec<QuerySweep>,
    wheel_ns: u64,
    wheel_plain_ns: u64,
}

#[derive(Deserialize)]
struct QuerySweep {
    width: usize,
    epsilon: f64,
    indices: Vec<i32>,
    bits: Vec<u32>,
}

fn record_splits(tree: Tree, pivots: &mut Vec<[usize; 2]>, leaves: &mut Vec<Vec<usize>>) {
    match tree {
        Tree::Leaf(rows) => leaves.push(rows),
        Tree::Split { a, b, left, right } => {
            pivots.push([a, b]);
            record_splits(*left, pivots, leaves);
            record_splits(*right, pivots, leaves);
        }
        Tree::Balanced { .. } => panic!("unexpected tree split"),
    }
}

#[test]
#[ignore = "requires reference/probe_projection_words.py output"]
fn compare_projection_words_with_wheel() {
    let path = std::env::var_os("PROJECTION_WORD_OBSERVATIONS").unwrap();
    let observations: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let decode = |values: Vec<u32>| values.into_iter().map(f32::from_bits).collect::<Vec<_>>();
    let mut total_pivots = 0;
    let mut total_rows = 0;
    let mut matching_rows = 0;
    for case in observations.training {
        let data = decode(case.train_bits);
        let rows = DenseRows::training(&data, case.n, case.d).unwrap();
        let tree = build_word_tree(
            rows,
            (0..case.n).collect(),
            case.cap,
            case.depth,
            &case.words,
            0,
        )
        .unwrap();
        let mut pivots = Vec::new();
        let mut leaves = Vec::new();
        record_splits(tree, &mut pivots, &mut leaves);
        assert_eq!(pivots, case.pivots, "{}", case.name);
        assert_eq!(leaves, case.rows_by_leaf, "{}", case.name);
        let aligned: usize = leaves
            .iter()
            .map(|rows| {
                let minimum = *rows.iter().min().unwrap();
                rows.iter()
                    .filter(|&&row| case.labels[row] == minimum)
                    .count()
            })
            .sum();
        assert_eq!(aligned, case.matching_component_labels, "{}", case.name);
        total_pivots += pivots.len();
        total_rows += case.n;
        matching_rows += aligned;
        println!(
            "{} pivot pairs {}, components {aligned}/{}, cross-leaf offers {}, shifted right {:?}",
            case.name,
            pivots.len(),
            case.n,
            case.cross_leaf_offers,
            case.shifted_right_matches
        );
    }
    println!("training pivot pairs {total_pivots}, components {matching_rows}/{total_rows}");
    for case in observations.queries {
        let data = decode(case.train_bits);
        let queries = decode(case.query_bits);
        let count = queries.len() / case.d;
        let build = || {
            DenseApproximateQueryIndex::from_training_graph(
                &data,
                case.n,
                case.d,
                &case.graph,
                5,
                42,
                None,
            )
            .unwrap()
        };
        let plain = build();
        let baseline = build().with_projection_tree(42, case.training_cap);
        let training = build()
            .with_projection_words(&case.training_words, case.training_cap, case.depth)
            .unwrap();
        let prepared = build()
            .with_projection_words(&case.words, case.cap, case.depth)
            .unwrap();
        let mut ordered = 0;
        let mut original = 0;
        let mut wrong_tree = 0;
        let mut slots = 0;
        let mut shared = 0;
        let mut shared_bits = 0;
        let mut wheel_only = 0;
        let mut rust_only = 0;
        let mut example = None;
        for (expected, graph_only) in case.prepared.into_iter().zip(case.plain) {
            let width = expected.width;
            let e = expected.epsilon;
            let actual = prepared.query(&queries, count, width, e).unwrap();
            let alternate = training.query(&queries, count, width, e).unwrap();
            let old = baseline.query(&queries, count, width, e).unwrap();
            let control = plain.query(&queries, count, width, e).unwrap();
            assert_eq!(
                control.indices, graph_only.indices,
                "{} width {width} epsilon {e}",
                case.name
            );
            assert!(
                control
                    .distances
                    .iter()
                    .map(|d| d.to_bits())
                    .eq(graph_only.bits)
            );
            if width == 1 && e == 0.0 {
                assert_eq!(actual.indices, expected.indices, "{}", case.name);
            }
            let aligned = |values: &[i32]| {
                values
                    .iter()
                    .zip(&expected.indices)
                    .filter(|(a, b)| a == b)
                    .count()
            };
            ordered += aligned(&actual.indices);
            original += aligned(&old.indices);
            wrong_tree += aligned(&alternate.indices);
            slots += expected.indices.len();
            for row in 0..count {
                let start = row * width;
                let native = &actual.indices[start..start + width];
                let wheel = &expected.indices[start..start + width];
                let mut missing = Vec::new();
                for (rank, &id) in wheel.iter().enumerate().filter(|(_, id)| **id >= 0) {
                    if let Some(offset) = native.iter().position(|&other| other == id) {
                        shared += 1;
                        shared_bits += usize::from(
                            actual.distances[start + offset].to_bits()
                                == expected.bits[start + rank],
                        );
                    } else {
                        missing.push(id);
                    }
                }
                let extra = native
                    .iter()
                    .copied()
                    .filter(|&id| id >= 0 && !wheel.contains(&id))
                    .collect::<Vec<_>>();
                wheel_only += missing.len();
                rust_only += extra.len();
                if example.is_none() && !extra.is_empty() && !missing.is_empty() {
                    example = Some((width, e, row, missing, extra));
                }
            }
        }
        assert_eq!(shared_bits, shared, "{}", case.name);
        let timing = |index: &DenseApproximateQueryIndex<'_>| {
            let run = || black_box(index.query(&queries, count, 5, 0.12).unwrap());
            run();
            let mut times = [0; 41];
            for time in &mut times {
                let start = Instant::now();
                run();
                *time = start.elapsed().as_nanos();
            }
            times.sort_unstable();
            times[20]
        };
        println!(
            "{} prepared {ordered}/{slots}, original {original}/{slots}, training as search {wrong_tree}/{slots}, shared bits {shared_bits}/{shared}, wheel-only {wheel_only}, Rust-only {rust_only}, graph-only {slots}/{slots}",
            case.name
        );
        println!(
            "  residual (width, eps, row, wheel-only, Rust-only) {example:?}, warm ns prepared Rust {} wheel {}, graph-only Rust {} wheel {}",
            timing(&prepared),
            case.wheel_ns,
            timing(&plain),
            case.wheel_plain_ns
        );
    }
}
