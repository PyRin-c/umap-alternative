use umap_core::approximate_neighbors::DenseApproximateIndex;
use umap_core::feature_metrics::{
    FeatureMetric, rank_feature_rows, rank_feature_rows_with_tie_ranking,
};
use umap_core::neighbors::DistanceMetric;

#[test]
fn configured_ranks_preserve_self_ties_and_the_inclusive_cutoff() {
    let training = [0.0, 2.0, 4.0, 6.0];
    let metric = FeatureMetric::Minkowski(1.0);

    let cut = rank_feature_rows(metric, &training, None, 1, 3, 2.0).unwrap();
    let [missing, zero] = [f32::INFINITY, 0.0];
    assert_eq!(cut.removed, 12);
    assert_eq!(cut.columns, [0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1]);
    assert_eq!(cut.distances, [zero, missing, missing].repeat(4));

    let kept = rank_feature_rows(metric, &training, None, 1, 3, 2.0_f64.next_up()).unwrap();
    assert_eq!(kept.removed, 6);
    assert_eq!(kept.columns, [0, 1, -1, 1, 0, 2, 2, 1, 3, 3, 2, -1]);
    assert_eq!(
        kept.distances,
        [
            0.0, 2.0, missing, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, missing
        ]
    );

    let self_only = rank_feature_rows(metric, &training, None, 1, 1, f64::INFINITY).unwrap();
    assert_eq!(self_only.columns, [0, 1, 2, 3]);
    assert_eq!(self_only.distances, [0.0; 4]);

    let queries =
        rank_feature_rows(metric, &training, Some(&[1.0, 5.0]), 1, 2, f64::INFINITY).unwrap();
    assert_eq!(queries.columns, [0, 1, 2, 3]);
    assert_eq!(queries.distances, [1.0; 4]);
}

#[test]
fn configured_rank_validation_precedes_slicing() {
    let metric = FeatureMetric::Minkowski(2.0);
    let training = [0.0, 1.0, 2.0, 3.0];
    for (width, k, cutoff) in [
        (0, 1, 1.0),
        (3, 1, 1.0),
        (2, 0, 1.0),
        (2, 3, 1.0),
        (2, 1, f64::NAN),
        (2, 1, -1.0),
    ] {
        assert!(rank_feature_rows(metric, &training, None, width, k, cutoff).is_err());
    }
    assert!(rank_feature_rows(metric, &training, Some(&[1.0]), 2, 1, 1.0).is_err());
    assert!(rank_feature_rows(metric, &[f64::INFINITY], None, 1, 1, 1.0).is_err());
    assert!(rank_feature_rows(FeatureMetric::Haversine, &training, None, 1, 1, 1.0).is_err());
    assert!(
        rank_feature_rows(
            FeatureMetric::WeightedMinkowski(1.0, &[1.0]),
            &training,
            None,
            2,
            1,
            1.0
        )
        .is_err()
    );
}

#[test]
fn configured_query_accepts_only_valid_tie_permutations() {
    let training = [0.0, 2.0, 5.0];
    let metric = FeatureMetric::Minkowski(1.0);
    let query = [1.0];
    let ranked = rank_feature_rows_with_tie_ranking(
        metric,
        &training,
        Some(&query),
        1,
        2,
        f64::INFINITY,
        Some(|distances: &[f32], k| {
            assert_eq!(k, 2);
            assert_eq!(distances, [1.0, 1.0, 4.0]);
            vec![1, 0]
        }),
    )
    .unwrap();
    assert_eq!(ranked.columns, [1, 0]);
    assert_eq!(ranked.distances, [1.0, 1.0]);
    let duplicate = rank_feature_rows_with_tie_ranking(
        metric,
        &training,
        Some(&query),
        1,
        2,
        f64::INFINITY,
        Some(|_: &[f32], _| vec![1, 1]),
    );
    assert!(duplicate.is_err());
}

#[test]
fn owned_approximate_index_keeps_the_borrowed_index_results() {
    let query = [15.25, 7.5, 30.5, 3.0];
    let (owned, expected_training, expected_query) = {
        let training: Vec<f32> = (0..80)
            .flat_map(|row| [row as f32, ((row * 17) % 19) as f32])
            .collect();
        let borrowed =
            DenseApproximateIndex::build(&training, 80, 2, 4, DistanceMetric::Euclidean, 31)
                .unwrap();
        let expected_training = borrowed.training_ranks().clone();
        let expected_query = borrowed.query(&query, 2).unwrap();
        (borrowed.into_owned(), expected_training, expected_query)
    };
    assert_eq!(owned.training_ranks(), &expected_training);
    assert_eq!(owned.query(&query, 2).unwrap(), expected_query);
    assert!(owned.query(&query[..3], 2).is_err());
}
