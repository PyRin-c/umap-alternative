use serde::Deserialize;

use super::*;
use crate::approximate_neighbors::approximate_squared_euclidean;

#[test]
fn prepared_planes_match_recomputed_margins_across_vector_boundaries() {
    let values = |width: usize, seed: usize| -> Vec<f32> {
        (0..width)
            .map(|i| ((i * 97 + seed) % 43) as f32 * 0.11 - 1.5)
            .collect()
    };
    for width in [1, 3, 4, 7, 31, 32, 33, 64, 65] {
        let left = values(width, 9);
        let right = values(width, 13);
        for angular in [false, true] {
            let plane = PreparedProjection::new(&left, &right, angular);
            for query in [values(width, 7), values(width, 51), vec![-0.0; width]] {
                let expected = if angular {
                    angular_margin(&left, &right, &query)
                } else {
                    prepared_margin(&left, &right, &query)
                };
                assert_eq!(plane.margin(&query).to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn prepared_planes_keep_nonfinite_sorting_margins() {
    let pivots = [f32::MAX; 32];
    let query = [0.0; 32];
    let expected = prepared_margin(&pivots, &pivots, &query);
    assert!(expected.is_nan());
    let actual = PreparedProjection::new(&pivots, &pivots, false).margin(&query);
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn training_ties_stay_with_original_rows() {
    let mut data = [0.0; 17];
    let rows: Vec<_> = (0..data.len()).collect();
    let (a, b) = projection_pivots(&rows, pivot_words(tree_key(0, 0)));
    data[a] = -1.0;
    data[b] = 1.0;
    let leaves = |data: &[f32]| {
        training_projection_leaves(data, data.len(), 1, 0, 0, NonZeroUsize::MIN, 1).unwrap()
    };
    let original = leaves(&data);
    let moved = *original[0].iter().find(|&&row| row != a).unwrap();
    let edge = 0.5e-8_f32;
    data[moved] = edge;
    assert_eq!(leaves(&data), original);
    data[moved] = edge.next_up();
    let result = leaves(&data);
    assert!(result[1].contains(&moved));
    assert!(
        original[0]
            .iter()
            .filter(|&&row| row != moved)
            .all(|row| result[0].contains(row))
    );
    assert!(original[1].iter().all(|row| result[1].contains(row)));
}

#[derive(Deserialize)]
struct Observations {
    words: Vec<WordControl>,
    cases: Vec<Case>,
    forests: Vec<Forest>,
}

#[derive(Deserialize)]
struct WordControl {
    scalar: u64,
    words: Vec<[u64; 2]>,
    witnesses: Option<usize>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    d: usize,
    scalar: u64,
    depth: u8,
    cap: NonZeroUsize,
    fitted_components: usize,
    sides: Vec<usize>,
    train_bits: Vec<u32>,
    leaves: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct Forest {
    n: usize,
    d: usize,
    scalar: u64,
    trees: usize,
    depth: u8,
    cap: NonZeroUsize,
    width: usize,
    train_bits: Vec<u32>,
    indices: Vec<usize>,
    distance_bits: Vec<u32>,
    enough: Vec<bool>,
}

#[test]
#[ignore = "requires the capture from reference/probe_training_projection_ties.py"]
fn compare_training_projection_with_wheel() {
    let path = std::env::var_os("TRAINING_PROJECTION_OBSERVATIONS").unwrap();
    let obs: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut words = 0;
    let mut witnesses = 0;
    for control in obs.words {
        let amount = control.words.len();
        let mut keys = vec![tree_key(control.scalar, 0)];
        for (slot, expected) in control.words.into_iter().enumerate() {
            let key = keys[slot];
            assert_eq!(
                pivot_words(key),
                expected,
                "scalar {} slot {slot}",
                control.scalar
            );
            if slot * 2 + 2 < amount {
                keys.extend([child_key(key, 0), child_key(key, 1)]);
            }
        }
        words += amount * 2;
        witnesses += control.witnesses.unwrap_or_default();
    }
    println!("exact pivot words {words}, fresh root witnesses {witnesses}");
    let decode = |values: Vec<u32>| values.into_iter().map(f32::from_bits).collect::<Vec<_>>();
    let mut membership = 0;
    for case in obs.cases {
        let data = decode(case.train_bits);
        let leaves =
            training_projection_leaves(&data, case.n, case.d, case.scalar, 0, case.cap, case.depth)
                .unwrap();
        assert_eq!(leaves, case.leaves, "{}", case.name);
        assert_eq!(leaves.len(), case.fitted_components, "{}", case.name);
        for (leaf, rows) in leaves.iter().enumerate() {
            for &row in rows {
                assert_eq!(case.sides[row], leaf, "{} row {row}", case.name);
                membership += 1;
            }
        }
    }
    println!("ordered leaf replay matches {membership} public component labels");
    let mut slots = 0;
    for forest in obs.forests {
        let data = decode(forest.train_bits);
        let rows = DenseRows::training(&data, forest.n, forest.d).unwrap();
        let mut allowed = vec![vec![false; forest.n]; forest.n];
        for ordinal in 0..forest.trees {
            let leaves = training_projection_leaves(
                &data,
                forest.n,
                forest.d,
                forest.scalar,
                ordinal,
                forest.cap,
                forest.depth,
            )
            .unwrap();
            for leaf in leaves {
                for &row in &leaf {
                    for &column in &leaf {
                        allowed[row][column] = true;
                    }
                }
            }
        }
        for (row, links) in allowed.iter().enumerate() {
            if !forest.enough[row] {
                continue;
            }
            let mut candidates = links
                .iter()
                .enumerate()
                .filter(|&(_, &linked)| linked)
                .map(|(column, _)| {
                    (
                        approximate_squared_euclidean(rows.row(row), rows.row(column)).sqrt(),
                        column,
                    )
                })
                .collect::<Vec<_>>();
            candidates.sort_unstable_by(|(a, i), (b, j)| a.total_cmp(b).then(i.cmp(j)));
            for (rank, &(distance, column)) in candidates.iter().take(forest.width).enumerate() {
                let slot = row * forest.width + rank;
                assert_eq!(
                    column, forest.indices[slot],
                    "n {} row {row} rank {rank}",
                    forest.n
                );
                assert_eq!(distance.to_bits(), forest.distance_bits[slot]);
                slots += 1;
            }
        }
    }
    println!("forest ranks and distance bits {slots}/{slots}");
}
