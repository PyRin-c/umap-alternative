use serde::Deserialize;
use umap_core::embedding::{
    DenseEmbeddingConfig, DenseEmbeddingInit, EmbeddingLayoutConfig, fit_dense_embedding_seeded,
    fit_graph_embedding_with_rng,
};
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::DistanceMetric;
use umap_core::optimize::optimize_csr;
use umap_core::rng::Mt19937;

const PUBLIC_FIXTURE: &str = include_str!(
    "../../../reference/fixtures/umap-learn-0.6.0-seeded-layout-precision-windows-x86_64.json"
);

#[derive(Deserialize)]
struct Case {
    name: String,
    n_components: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<u32>,
    raw: Vec<u32>,
    seed: u32,
    block: usize,
    cursor: usize,
    post_cursor: usize,
    post_next_words: Vec<u32>,
    checkpoints: Vec<Checkpoint>,
    dense: Option<DenseInput>,
}

#[derive(Deserialize)]
struct DenseInput {
    rows: Vec<Vec<f32>>,
    neighbors: usize,
}

#[derive(Deserialize)]
struct Checkpoint {
    epochs: usize,
    alpha: f64,
    a: f64,
    b: f64,
    gamma: f64,
    rate: f64,
    expected: Vec<u32>,
}

#[test]
fn seeded_layout_matches_public_schedule_and_distance_bits() {
    let cases: Vec<Case> = serde_json::from_str(PUBLIC_FIXTURE).unwrap();
    let mut mismatches = Vec::new();
    for case in cases {
        let n_rows = case.indptr.len() - 1;
        let graph = CsrGraph {
            n_rows,
            n_cols: n_rows,
            indptr: case.indptr,
            indices: case.indices,
            data: case.weights.into_iter().map(f32::from_bits).collect(),
        };
        let raw: Vec<_> = case.raw.into_iter().map(f32::from_bits).collect();

        // Recover the public word array from its seed and twisted block index.
        let mut seeded = Mt19937::new(case.seed);
        for _ in 0..=624 * case.block {
            seeded.next_u32();
        }
        let (words, _) = seeded.state();
        for checkpoint in case.checkpoints {
            let config = EmbeddingLayoutConfig {
                n_components: case.n_components,
                a: checkpoint.a,
                b: checkpoint.b,
                learning_rate: checkpoint.alpha,
                n_epochs: checkpoint.epochs,
                repulsion_strength: checkpoint.gamma,
                negative_sample_rate: checkpoint.rate,
            };
            let label = format!(
                "{} at {} epochs with a={} and gamma={}",
                case.name, checkpoint.epochs, checkpoint.a, checkpoint.gamma,
            );
            let mut rng = Mt19937::from_state(&words, case.cursor).unwrap();
            let fit = fit_graph_embedding_with_rng(graph.clone(), &raw, &config, &mut rng).unwrap();
            assert_eq!(fit.shape, (n_rows, case.n_components), "{label}");
            assert_eq!(fit.graph, graph, "{label}");
            assert_eq!(fit.coordinates.len(), checkpoint.expected.len(), "{label}");
            assert_eq!(rng.state(), (words, case.post_cursor), "{label}");
            for &word in &case.post_next_words {
                assert_eq!(rng.next_u32(), word, "{label}");
            }
            if let Some(dense) = &case.dense {
                let data: Vec<_> = dense.rows.iter().flatten().copied().collect();
                let dense_config = DenseEmbeddingConfig {
                    n_neighbors: dense.neighbors,
                    metric: DistanceMetric::Euclidean,
                    n_components: config.n_components,
                    a: config.a as f32,
                    b: config.b as f32,
                    learning_rate: config.learning_rate as f32,
                    n_epochs: config.n_epochs,
                    repulsion_strength: config.repulsion_strength as f32,
                    negative_sample_rate: config.negative_sample_rate,
                    init: DenseEmbeddingInit::explicit(&raw),
                };
                let dense_fit = fit_dense_embedding_seeded(
                    &data,
                    n_rows,
                    dense.rows[0].len(),
                    &dense_config,
                    case.seed,
                )
                .unwrap();
                assert_eq!(
                    dense_fit, fit,
                    "{label}: dense fit must match the graph handoff"
                );
            }

            let mut changed = 0;
            let mut max_ulp = 0;
            let mut max_error = 0.0_f32;
            for (&actual, &expected) in fit.coordinates.iter().zip(&checkpoint.expected) {
                if actual.to_bits() != expected {
                    changed += 1;
                    max_ulp = max_ulp.max(actual.to_bits().abs_diff(expected));
                    max_error = max_error.max((actual - f32::from_bits(expected)).abs());
                }
            }
            if changed != 0 {
                mismatches.push(format!(
                    "{label}: {changed} of {} coordinates, max {max_ulp} ULP, abs {max_error}",
                    fit.coordinates.len(),
                ));
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn injected_f32_optimizer_retains_direct_weight_ratio_and_draws() {
    let graph = CsrGraph {
        n_rows: 4,
        n_cols: 4,
        indptr: vec![0, 1, 1, 2, 2],
        indices: vec![1, 3],
        data: vec![f32::from_bits(0x3f35_c28f), f32::from_bits(0x3c45_29d3)],
    };
    let mut embedding = [0.0_f32, 10.0, 0.0, 10.0];
    let mut draws = Vec::new();
    optimize_csr(
        &mut embedding,
        1,
        &graph,
        1.0_f32,
        1.0_f32,
        0.13_f32,
        0.0_f32,
        2.0,
        60,
        |row| {
            draws.push(row);
            row
        },
    )
    .unwrap();
    // The smaller edge is first due at epoch 60 with the direct ratio.
    assert_eq!(embedding[2].to_bits(), 0);
    assert_eq!(embedding[3].to_bits(), 10.0_f32.to_bits());
    assert_eq!(draws, vec![0; 117]);
}
