//! Public layout fixture parity is checked at zero epochs for random and
//! explicit initialization. The two-epoch tests check Rust sampler wiring
//! with a controlled graph and injected samples.

use serde::Deserialize;
use umap_core::embedding::{
    DenseEmbeddingConfig, DenseEmbeddingFitError, DenseEmbeddingInit, fit_dense_embedding,
};
use umap_core::fit::fit_dense_graph;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::DistanceMetric;
use umap_core::optimize::OptimizeError;

const WINDOWS_LAYOUT: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-layout-windows-x86_64.json");
const UBUNTU_LAYOUT: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-layout-ubuntu-x86_64.json");
const TWO_POINT_INPUT: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

#[derive(Deserialize)]
struct LayoutFixture {
    input: Float32Input,
    parameters: Parameters,
    cases: Cases,
}

#[derive(Deserialize)]
struct Float32Input {
    dtype: String,
    shape: [usize; 2],
    values: Vec<Vec<f32>>,
}

impl Float32Input {
    fn flatten(&self) -> Vec<f32> {
        let [n_rows, n_cols] = self.shape;
        assert_eq!(self.dtype, "float32");
        assert_eq!(self.values.len(), n_rows);
        assert!(self.values.iter().all(|row| row.len() == n_cols));
        self.values.iter().flatten().copied().collect()
    }
}

#[derive(Deserialize)]
struct Parameters {
    n_components: usize,
    n_neighbors: usize,
    metric: String,
}

#[derive(Deserialize)]
struct Cases {
    random_init: RandomCase,
    explicit_init: ExplicitCase,
}

#[derive(Deserialize)]
struct RandomCase {
    seed: u32,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Deserialize)]
struct ExplicitCase {
    init: Float32Input,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Deserialize)]
struct Checkpoint {
    n_epochs: usize,
    output: Float32Output,
}

#[derive(Deserialize)]
struct Float32Output {
    dtype: String,
    shape: [usize; 2],
    bits_uint32: Vec<Vec<u32>>,
}

#[test]
fn zero_epoch_dense_fit_matches_public_layout_bits() {
    for (platform, json) in [("Windows", WINDOWS_LAYOUT), ("Ubuntu", UBUNTU_LAYOUT)] {
        let fixture: LayoutFixture = serde_json::from_str(json).expect("valid layout fixture");
        let [n_rows, n_features] = fixture.input.shape;
        let data = fixture.input.flatten();
        let Parameters {
            n_components,
            n_neighbors,
            metric,
        } = fixture.parameters;
        assert_eq!(metric, "euclidean", "{platform}: fixture metric");
        let metric = DistanceMetric::Euclidean;
        let expected_graph = fit_dense_graph(&data, n_rows, n_features, n_neighbors, metric)
            .expect("valid fixture graph");

        let Cases {
            random_init,
            explicit_init,
        } = fixture.cases;
        assert_eq!(explicit_init.init.shape, [n_rows, n_components]);
        for (mode, init, checkpoints) in [
            (
                "random",
                DenseEmbeddingInit::random(random_init.seed),
                random_init.checkpoints,
            ),
            (
                "explicit",
                DenseEmbeddingInit::explicit(&explicit_init.init.flatten()),
                explicit_init.checkpoints,
            ),
        ] {
            let expected = &checkpoints
                .iter()
                .find(|checkpoint| checkpoint.n_epochs == 0)
                .expect("public fixture contains a zero-epoch checkpoint")
                .output;
            let config = DenseEmbeddingConfig {
                n_neighbors,
                metric,
                n_components,
                // Valid optimizer settings for a checkpoint before updates.
                a: 1.0,
                b: 1.0,
                learning_rate: 1.0,
                n_epochs: 0,
                repulsion_strength: 1.0,
                negative_sample_rate: 5.0,
                init,
            };
            let actual = fit_dense_embedding(&data, n_rows, n_features, &config, |source| {
                panic!("{platform} {mode}: unexpected sampler call from row {source}")
            })
            .unwrap_or_else(|error| panic!("{platform} {mode}: {error}"));

            assert_eq!(actual.graph, expected_graph, "{platform} {mode}: graph");
            assert_eq!(expected.dtype, "float32");
            assert_eq!(expected.shape, [n_rows, n_components]);
            assert_eq!(actual.shape, (n_rows, n_components));
            assert_eq!(actual.coordinates.len(), n_rows * n_components);
            assert_eq!(expected.bits_uint32.len(), n_rows);
            for (row, (values, bits)) in actual
                .coordinates
                .chunks_exact(n_components)
                .zip(&expected.bits_uint32)
                .enumerate()
            {
                let actual_bits = values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(&actual_bits, bits, "{platform} {mode}: row {row}");
            }
        }
    }
}

fn two_epoch_config() -> DenseEmbeddingConfig {
    DenseEmbeddingConfig {
        n_neighbors: 2,
        metric: DistanceMetric::Euclidean,
        n_components: 1,
        a: 1.0,
        b: 1.0,
        learning_rate: 1.0,
        n_epochs: 2,
        repulsion_strength: 1.0,
        negative_sample_rate: 2.0,
        init: DenseEmbeddingInit::explicit(&[2.0, 6.0]),
    }
}

#[test]
fn dense_fit_uses_the_injected_negative_sampler_in_row_order() {
    let config = two_epoch_config();
    let mut self_samples = Vec::new();
    let attraction = fit_dense_embedding(&TWO_POINT_INPUT, 2, 2, &config, |source| {
        self_samples.push(source);
        source
    })
    .expect("self samples are valid");
    let mut other_samples = Vec::new();
    let repulsion = fit_dense_embedding(&TWO_POINT_INPUT, 2, 2, &config, |source| {
        other_samples.push(source);
        1 - source
    })
    .expect("the other row is a valid negative sample");

    // For each row the other row is its only non-self neighbor, at the
    // local-connectivity distance. Both directions have unit membership.
    let expected_graph = CsrGraph {
        n_rows: 2,
        n_cols: 2,
        indptr: vec![0, 1, 2],
        indices: vec![1, 0],
        data: vec![1.0, 1.0],
    };
    assert_eq!(attraction.graph, expected_graph);
    assert_eq!(repulsion.graph, expected_graph);
    assert_eq!(attraction.shape, (2, 1));
    assert_eq!(repulsion.shape, (2, 1));

    // A unit positive interval and a negative interval of one half schedule
    // one negative sample per edge at epoch one.
    assert_eq!(self_samples, [0, 1]);
    assert_eq!(other_samples, [0, 1]);
    let attraction_gap = attraction.coordinates[1] - attraction.coordinates[0];
    let repulsion_gap = repulsion.coordinates[1] - repulsion.coordinates[0];
    assert!(0.0 < attraction_gap && attraction_gap < 10.0);
    assert!(repulsion_gap > attraction_gap);
}

#[test]
fn dense_fit_reports_an_out_of_range_negative_sample() {
    let config = two_epoch_config();
    let mut sources = Vec::new();
    let error = fit_dense_embedding(&TWO_POINT_INPUT, 2, 2, &config, |source| {
        sources.push(source);
        2
    })
    .unwrap_err();
    assert_eq!(sources, [0]);
    assert_eq!(
        error,
        DenseEmbeddingFitError::Optimize(OptimizeError::InvalidNegativeVertex {
            source: 0,
            vertex: 2,
            n_rows: 2,
        })
    );
}
