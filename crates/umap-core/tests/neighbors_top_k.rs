use umap_core::neighbors::{DistanceMetric, Neighbor, exhaustive_knn_with_metric, precomputed_knn};

const N_ROWS: usize = 31;

#[test]
fn metric_top_k_matches_full_sort() {
    let pattern = [
        [0.0, -0.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.12, 0.03],
        [0.29, 0.18],
        [f32::MAX, -f32::MAX],
        [-f32::MAX, f32::MAX],
    ];
    let rows: [[f32; 2]; N_ROWS] = std::array::from_fn(|i| pattern[i % pattern.len()]);

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Chebyshev,
        DistanceMetric::Cosine,
        DistanceMetric::Correlation,
        DistanceMetric::Canberra,
        DistanceMetric::BrayCurtis,
    ] {
        assert_top_k(
            |query, candidate| {
                let [x, y] = rows[query];
                let [other_x, other_y] = rows[candidate];
                match metric {
                    DistanceMetric::Euclidean => {
                        let dx = x - other_x;
                        let dy = y - other_y;
                        (f64::from(dx * dx) + f64::from(dy * dy)).sqrt() as f32
                    }
                    DistanceMetric::Manhattan => {
                        (f64::from((x - other_x).abs()) + f64::from((y - other_y).abs())) as f32
                    }
                    DistanceMetric::Chebyshev => {
                        let dx = f64::from(x) - f64::from(other_x);
                        let dy = f64::from(y) - f64::from(other_y);
                        dx.abs().max(dy.abs()) as f32
                    }
                    DistanceMetric::Cosine if query == candidate => 0.0,
                    DistanceMetric::Cosine => {
                        let left = f64::from(x * x) + f64::from(y * y);
                        let right = f64::from(other_x * other_x) + f64::from(other_y * other_y);
                        if left == 0.0 || right == 0.0 {
                            return f32::from(u8::from(left != right));
                        }
                        let dot = f64::from(x * other_x) + f64::from(y * other_y);
                        if dot.is_finite() {
                            (1.0 - dot / (left * right).sqrt()) as f32
                        } else {
                            f32::NAN
                        }
                    }
                    DistanceMetric::Correlation => {
                        if x == y || other_x == other_y {
                            f32::from(u8::from((x == y) != (other_x == other_y)))
                        } else if (x > y) == (other_x > other_y) {
                            0.0
                        } else {
                            2.0
                        }
                    }
                    DistanceMetric::Canberra => {
                        let term = |left: f32, right: f32| {
                            if left == 0.0 && right == 0.0 {
                                0.0
                            } else {
                                f64::from((left - right).abs() / (left.abs() + right.abs()))
                            }
                        };
                        (term(x, other_x) + term(y, other_y)) as f32
                    }
                    DistanceMetric::BrayCurtis => {
                        let divisor =
                            f64::from((x + other_x).abs()) + f64::from((y + other_y).abs());
                        if divisor == 0.0 {
                            0.0
                        } else {
                            ((f64::from((x - other_x).abs()) + f64::from((y - other_y).abs()))
                                / divisor) as f32
                        }
                    }
                    _ => unreachable!("the reference loop selects the original metrics"),
                }
            },
            |k| {
                exhaustive_knn_with_metric(rows.as_flattened(), N_ROWS, 2, k, metric)
                    .expect("valid feature matrix")
            },
        );
    }
}

#[test]
fn precomputed_top_k_matches_full_sort_with_signed_zeros() {
    let distances: [[f32; N_ROWS]; N_ROWS] = std::array::from_fn(|row| {
        std::array::from_fn(|column| match (row == column, (row + 3 * column) % 7) {
            (true, _) | (false, 0) => -0.0,
            (false, 1) => 0.0,
            (false, distance) => distance as f32,
        })
    });

    assert_top_k(
        |query, candidate| distances[query][candidate],
        |k| precomputed_knn(distances.as_flattened(), N_ROWS, k).expect("valid distance matrix"),
    );
}

fn assert_top_k(
    distance: impl Fn(usize, usize) -> f32,
    neighbors: impl Fn(usize) -> Vec<Vec<Neighbor>>,
) {
    let expected: [_; N_ROWS] = std::array::from_fn(|query| {
        let mut row: Vec<_> = (0..N_ROWS)
            .map(|candidate| (candidate, distance(query, candidate)))
            .collect();
        // This stable full sort keeps the original ascending index for ties.
        row.sort_by(|left, right| left.1.total_cmp(&right.1));
        row
    });

    for k in 1..=N_ROWS {
        let result = neighbors(k);
        assert_eq!(result.len(), N_ROWS);
        for (query, (actual, expected)) in result.iter().zip(&expected).enumerate() {
            assert_eq!(actual.len(), k);
            if k == 1 {
                assert!(actual.capacity() < N_ROWS);
            }
            for (neighbor, &(index, distance)) in actual.iter().zip(&expected[..k]) {
                assert_eq!(
                    (neighbor.index, neighbor.distance.to_bits()),
                    (index, distance.to_bits()),
                    "query {query}, k={k}"
                );
            }
        }
    }
}
