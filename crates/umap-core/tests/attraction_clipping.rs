use umap_core::embedding::{DenseEmbeddingConfig, DenseEmbeddingInit, fit_dense_embedding};
use umap_core::fit::fit_precomputed_graph;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::initialization::normalize_columns;
use umap_core::layout::{LayoutError, attractive_update_pair};
use umap_core::neighbors::DistanceMetric;
use umap_core::optimize::optimize_csr;

#[test]
fn attractively_updates_at_and_around_both_component_limits() {
    let limit = 4.0_f32;
    let alpha = 0.125;
    for sign in [1.0_f32, -1.0] {
        for (magnitude, expected) in [
            (limit.next_down(), limit.next_down()),
            (limit, limit),
            (limit.next_up(), limit),
        ] {
            // With a=1 and squared distance=1, the signed gradient is b * sign.
            let mut head = [0.0];
            let mut tail = [sign];
            attractive_update_pair(&mut head, &mut tail, 1.0, magnitude, alpha).unwrap();

            let step = alpha * f64::from(sign * expected);
            assert_eq!(head, [step as f32], "sign={sign}, gradient={magnitude:?}");
            assert_eq!(
                tail,
                [(f64::from(sign) - step) as f32],
                "sign={sign}, gradient={magnitude:?}"
            );
        }
    }
}

#[test]
fn clips_each_component_before_scaling_by_learning_rate() {
    let mut head = [0.0, 0.0];
    let mut tail = [0.125, -0.0625];
    attractive_update_pair(&mut head, &mut tail, 32.0, 1.0, 0.125).unwrap();

    // The raw gradients are 64/13 and -32/13. Only the first saturates.
    assert_eq!(head, [0.5, (-4.0_f64 / 13.0) as f32]);
    assert_eq!(tail, [-0.375, (-0.0625 + 4.0_f64 / 13.0) as f32]);
}

#[test]
fn clips_large_finite_gradient_before_it_can_overflow_during_scaling() {
    let mut head = [0.0];
    let mut tail = [1.0];
    attractive_update_pair(&mut head, &mut tail, 0.5, 1.0e308, 4.0).unwrap();
    assert_eq!(head, [16.0]);
    assert_eq!(tail, [-15.0]);
}

#[test]
fn coincident_endpoints_are_a_no_op_with_a_singular_curve() {
    let original = [-0.0_f32, 0.125];
    let mut head = original;
    let mut tail = original;
    attractive_update_pair(&mut head, &mut tail, 10_000.0, 0.25, 0.1).unwrap();
    assert_eq!(head.map(f32::to_bits), original.map(f32::to_bits));
    assert_eq!(tail.map(f32::to_bits), original.map(f32::to_bits));
}

#[test]
fn rejects_invalid_parameters_and_coordinates_before_updating() {
    let mut head = [0.0, 0.0];
    let mut tail = [0.125, -0.0625];
    for (a, b, alpha, name) in [
        (f64::INFINITY, 1.0, 0.125, "a"),
        (32.0, 0.0, 0.125, "b"),
        (32.0, 1.0, -0.125, "learning_rate"),
    ] {
        let error = attractive_update_pair(&mut head, &mut tail, a, b, alpha).unwrap_err();
        assert!(matches!(
            error,
            LayoutError::InvalidParameter { parameter, .. } if parameter == name
        ));
        assert_eq!(head, [0.0, 0.0]);
        assert_eq!(tail, [0.125, -0.0625]);
    }

    tail[1] = f32::NAN;
    let original_bits = tail.map(f32::to_bits);
    let error = attractive_update_pair(&mut head, &mut tail, 32.0, 1.0, 0.125).unwrap_err();
    assert!(matches!(
        error,
        LayoutError::NonFiniteValue {
            endpoint: "tail",
            component: 1,
            ..
        }
    ));
    assert_eq!(head, [0.0, 0.0]);
    assert_eq!(tail.map(f32::to_bits), original_bits);

    head[1] = f32::INFINITY;
    let error = attractive_update_pair(&mut head, &mut tail, 32.0, 1.0, 0.0).unwrap_err();
    assert!(matches!(
        error,
        LayoutError::NonFiniteValue {
            endpoint: "head",
            component: 1,
            ..
        }
    ));
    assert_eq!(head, [0.0, f32::INFINITY]);
    assert_eq!(tail.map(f32::to_bits), original_bits);
}

#[test]
fn rejects_nonfinite_arithmetic_before_updating() {
    for (mut head, mut tail, a, intermediate) in [
        ([f32::MAX], [-f32::MAX], 32.0, "squared distance"),
        ([0.0], [0.125], f64::MAX, "gradient scale"),
    ] {
        let original = (head, tail);
        let error = attractive_update_pair(&mut head, &mut tail, a, 1.0, 0.125).unwrap_err();
        assert!(matches!(
            error,
            LayoutError::NonFiniteArithmetic { quantity, .. } if quantity == intermediate
        ));
        assert_eq!((head, tail), original);
    }

    // The first component can be stored, but the second displaces by 4e38.
    let mut head = [0.0, 0.0];
    let mut tail = [0.000_001, 0.125];
    let error = attractive_update_pair(&mut head, &mut tail, 64.0, 1.0, 1.0e38).unwrap_err();
    assert!(matches!(
        error,
        LayoutError::NonFiniteUpdate {
            endpoint: "head",
            component: 1,
            ..
        }
    ));
    assert_eq!(head, [0.0, 0.0]);
    assert_eq!(tail, [0.000_001, 0.125]);
}

#[test]
fn clipped_layout_matches_public_fit_transform_bits() {
    // Public umap-learn 0.6.0 fit_transform with two isolated mutual pairs,
    // explicit initialization, one component, and compatibility layout.
    let distances = [
        [0.0, 0.1, 20.0, 20.0],
        [0.1, 0.0, 20.0, 20.0],
        [20.0, 20.0, 0.0, 0.1],
        [20.0, 20.0, 0.1, 0.0],
    ];
    let initial = [0.0, 0.01, 9.9, 10.0];
    let initial_bits = [0, 1_008_981_770, 1_092_511_334, 1_092_616_192];
    let fitted_bits = [1_041_655_464, 3_188_468_022, 1_092_631_231, 1_092_496_295];
    let config = DenseEmbeddingConfig {
        n_neighbors: 2,
        metric: DistanceMetric::Euclidean,
        n_components: 1,
        a: 10_000.0_f64,
        b: 1.0,
        learning_rate: 0.1,
        n_epochs: 2,
        repulsion_strength: 0.0,
        negative_sample_rate: 1.0,
        init: DenseEmbeddingInit::explicit(&initial),
    };

    let graph = fit_precomputed_graph(distances.as_flattened(), 4, config.n_neighbors).unwrap();
    assert_eq!(
        graph,
        CsrGraph {
            n_rows: 4,
            n_cols: 4,
            indptr: vec![0, 1, 2, 3, 4],
            indices: vec![1, 0, 3, 2],
            data: vec![1.0; 4],
        }
    );
    let mut coordinates = initial;
    normalize_columns(&mut coordinates, 4, config.n_components).unwrap();
    assert_eq!(coordinates.map(f32::to_bits), initial_bits);
    optimize_csr(
        &mut coordinates,
        config.n_components,
        &graph,
        config.a,
        config.b,
        config.learning_rate,
        config.repulsion_strength,
        config.negative_sample_rate,
        config.n_epochs,
        std::convert::identity,
    )
    .unwrap();
    assert_eq!(coordinates.map(f32::to_bits), fitted_bits);

    // These dense points produce the same graph as the public distance matrix.
    let data = [0.0, 0.1, 20.0, 20.1];
    let fitted = fit_dense_embedding(&data, 4, 1, &config, std::convert::identity).unwrap();
    assert_eq!(fitted.graph, graph);
    assert_eq!(fitted.shape, (4, 1));
    let coordinates: [f32; 4] = fitted.coordinates.try_into().unwrap();
    assert_eq!(coordinates.map(f32::to_bits), fitted_bits);
}
