use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use umap_core::density::{
    DensityError, RadiusSpace, embedding_radii_euclidean, embedding_radii_from_refit_euclidean,
    original_radii_euclidean, radii_from_edge_distances, refit_embedding_graph_euclidean,
};
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::neighbors::KnnError;

const PUBLIC_FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-densmap-windows-x86_64.json");
// The public precomputed_knn refits in generate_densmap_fixture.py supply the
// CSR weights via graph_ and the aligned distances via embedding_distances.
// The log inputs record the argument to np.log in the fixture's weighted_log.
const DIAGNOSTIC_INPUTS: &str = include_str!("fixtures/density_diagnostic_inputs.json");

#[derive(Deserialize)]
struct Fixture {
    data_float32: Vec<Vec<f32>>,
    common_options: Options,
    cases: BTreeMap<String, Case>,
    arrays: Vec<Array>,
    graphs: Vec<Graph>,
    arithmetic: Arithmetic,
}

#[derive(Deserialize)]
struct Options {
    n_neighbors: usize,
}

#[derive(Deserialize)]
struct Case {
    graph: usize,
    outputs: Vec<usize>,
}

#[derive(Deserialize)]
struct Array {
    shape: Vec<usize>,
    bits_uint32: Vec<u32>,
}

#[derive(Deserialize)]
struct Graph {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights: Array,
}

#[derive(Deserialize)]
struct Arithmetic {
    precomputed_at_n_epochs_0: BTreeMap<String, Case>,
    close_pairs: ClosePairs,
}

#[derive(Deserialize)]
struct ClosePairs {
    data_and_init_float32: Vec<Vec<f32>>,
    options: Options,
    #[serde(flatten)]
    case: Case,
}

#[derive(Deserialize)]
struct DiagnosticInputs {
    original_log_inputs: BTreeMap<String, Vec<u32>>,
    close_pair_log_inputs: [Vec<u32>; 2],
    embedding_refits: Vec<Refit>,
}

#[derive(Deserialize)]
struct Refit {
    embedding_id: usize,
    radius_id: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights_bits: Vec<u32>,
    distances_bits: Vec<u32>,
    log_inputs_bits: Vec<u32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(PUBLIC_FIXTURE).unwrap()
}

fn diagnostic_inputs() -> DiagnosticInputs {
    serde_json::from_str(DIAGNOSTIC_INPUTS).unwrap()
}

fn floats(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn graph(indptr: Vec<usize>, indices: Vec<usize>, weights: Vec<f32>) -> CsrGraph {
    let n_rows = indptr.len() - 1;
    CsrGraph {
        n_rows,
        n_cols: n_rows,
        indptr,
        indices,
        data: weights,
    }
}

fn fixture_graph(record: &Graph) -> CsrGraph {
    graph(
        record.indptr.clone(),
        record.indices.clone(),
        floats(&record.weights.bits_uint32),
    )
}

fn scaled_edge_distances(graph: &CsrGraph, data: &[Vec<f32>], scale: f64) -> Vec<f32> {
    let mut distances = Vec::new();
    for (row, offsets) in graph.indptr.windows(2).enumerate() {
        for &column in &graph.indices[offsets[0]..offsets[1]] {
            let squared: f64 = data[row]
                .iter()
                .zip(&data[column])
                .map(|(&left, &right)| (f64::from(left) - f64::from(right)).powi(2))
                .sum();
            distances.push((squared.sqrt() * scale) as f32);
        }
    }
    distances
}

fn assert_radius_bits(actual: &[f32], log_inputs: &[u32], expected: &[u32], label: &str) -> usize {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), log_inputs.len());
    let mut matches = 0;
    for (index, ((value, &input), &reference)) in
        actual.iter().zip(log_inputs).zip(expected).enumerate()
    {
        let value = value.to_bits();
        let native_log = f32::from_bits(input).ln().to_bits();
        assert_eq!(value, native_log, "{label}, row {index}: reduction differs");
        assert!(
            value.abs_diff(reference) <= 1,
            "{label}, row {index}: public log differs by more than one ULP"
        );
        matches += usize::from(value == reference);
    }
    matches
}

fn positive_edges(graph: &CsrGraph) -> BTreeMap<(usize, usize), f32> {
    let mut result = BTreeMap::new();
    for (row, offsets) in graph.indptr.windows(2).enumerate() {
        for entry in offsets[0]..offsets[1] {
            if graph.data[entry] > 0.0 {
                result.insert((row, graph.indices[entry]), graph.data[entry]);
            }
        }
    }
    result
}

#[test]
fn original_radii_replay_dense_and_precomputed_fits() {
    let fixture = fixture();
    let inputs = diagnostic_inputs().original_log_inputs;
    let dense_case = &fixture.cases["0-density_output"];
    let input_graph = fixture_graph(&fixture.graphs[dense_case.graph]);
    let input = fixture.data_float32.concat();
    let radii = original_radii_euclidean(&input_graph, &input, 3).unwrap();
    let mut exact = assert_radius_bits(
        &radii,
        &inputs["dense"],
        &fixture.arrays[dense_case.outputs[1]].bits_uint32,
        "dense",
    );

    assert_eq!(fixture.arithmetic.precomputed_at_n_epochs_0.len(), 7);
    for (scale, case) in &fixture.arithmetic.precomputed_at_n_epochs_0 {
        let graph = fixture_graph(&fixture.graphs[case.graph]);
        let distances =
            scaled_edge_distances(&graph, &fixture.data_float32, scale.parse().unwrap());
        let radii = radii_from_edge_distances(&graph, &distances, RadiusSpace::Original).unwrap();
        exact += assert_radius_bits(
            &radii,
            &inputs[scale],
            &fixture.arrays[case.outputs[1]].bits_uint32,
            scale,
        );
    }
    eprintln!("original exact: {exact}/64");
}

#[test]
fn embedding_radii_replay_verified_refits_and_compare_native_refits() {
    let fixture = fixture();
    let refits = diagnostic_inputs().embedding_refits;
    let expected_pairs: BTreeSet<_> = fixture
        .cases
        .values()
        .filter(|case| case.outputs.len() == 3)
        .map(|case| (case.outputs[0], case.outputs[2]))
        .collect();
    let actual_pairs: BTreeSet<_> = refits
        .iter()
        .map(|r| (r.embedding_id, r.radius_id))
        .collect();
    assert_eq!(refits.len(), 16);
    assert_eq!(actual_pairs, expected_pairs);

    let mut exact = 0;
    let mut native_exact = 0;
    let mut native_matches_replay = 0;
    let mut native_support_matches = 0;
    let mut max_native_weight_error = 0.0_f32;
    let mut max_native_radius_error = 0.0_f32;
    let k = fixture.common_options.n_neighbors;
    for refit in refits {
        let embedding = &fixture.arrays[refit.embedding_id];
        let (n_rows, n_components) = (embedding.shape[0], embedding.shape[1]);
        let coordinates = floats(&embedding.bits_uint32);
        let verified = graph(refit.indptr, refit.indices, floats(&refit.weights_bits));
        let expected = &fixture.arrays[refit.radius_id].bits_uint32;
        assert_eq!(expected.len(), 8);

        let replay = radii_from_edge_distances(
            &verified,
            &floats(&refit.distances_bits),
            RadiusSpace::Embedding,
        )
        .unwrap();
        let derived =
            embedding_radii_from_refit_euclidean(&verified, &coordinates, n_components).unwrap();
        assert_eq!(
            bits(&derived),
            bits(&replay),
            "layout {}",
            refit.embedding_id
        );
        exact += assert_radius_bits(
            &replay,
            &refit.log_inputs_bits,
            expected,
            &format!("layout {}", refit.embedding_id),
        );

        let native =
            refit_embedding_graph_euclidean(&coordinates, n_rows, n_components, k).unwrap();
        let native_edges = positive_edges(&native);
        let verified_edges = positive_edges(&verified);
        for (edge, verified_weight) in &verified_edges {
            let native_weight = native_edges[edge];
            max_native_weight_error =
                max_native_weight_error.max((native_weight - verified_weight).abs());
        }
        native_support_matches += usize::from(native_edges.len() == verified_edges.len());
        for (&edge, &weight) in &native_edges {
            if !verified_edges.contains_key(&edge) {
                // In layout 0, row 2 has two non-self neighbors at the same
                // nearest distance and the native scale gives (2, 4) weight.
                assert_eq!(refit.embedding_id, 0);
                assert!(matches!(edge, (2, 4) | (4, 2)) && weight < 1e-8);
            }
        }
        let native_radii =
            embedding_radii_from_refit_euclidean(&native, &coordinates, n_components).unwrap();
        for ((&native, &replay), &reference) in native_radii.iter().zip(&replay).zip(expected) {
            native_exact += usize::from(native.to_bits() == reference);
            native_matches_replay += usize::from(native.to_bits() == replay.to_bits());
            max_native_radius_error =
                max_native_radius_error.max((native - f32::from_bits(reference)).abs());
        }
    }
    eprintln!(
        "verified reduction {exact}/128 public bits; native refit: {native_exact}/128 public bits, \
         {native_matches_replay}/128 verified-reduction bits, \
         {native_support_matches}/16 supports, max shared-edge weight error {max_native_weight_error}, \
         max radius error {max_native_radius_error}"
    );
    assert!(max_native_weight_error <= 2.0 * f32::EPSILON);
    assert!(max_native_radius_error <= 5e-7);
}

#[test]
fn close_pairs_reproduce_the_offset_in_both_spaces() {
    let fixture = fixture();
    let [original_inputs, embedding_inputs] = diagnostic_inputs().close_pair_log_inputs;
    let pairs = &fixture.arithmetic.close_pairs;
    let graph = fixture_graph(&fixture.graphs[pairs.case.graph]);
    let original =
        original_radii_euclidean(&graph, &pairs.data_and_init_float32.concat(), 2).unwrap();
    let embedding = &fixture.arrays[pairs.case.outputs[0]];
    let radii = embedding_radii_euclidean(
        &floats(&embedding.bits_uint32),
        embedding.shape[0],
        embedding.shape[1],
        pairs.options.n_neighbors,
    )
    .unwrap();

    assert_eq!(
        assert_radius_bits(
            &original,
            &original_inputs,
            &fixture.arrays[pairs.case.outputs[1]].bits_uint32,
            "close original"
        ),
        6
    );
    assert_eq!(
        assert_radius_bits(
            &radii,
            &embedding_inputs,
            &fixture.arrays[pairs.case.outputs[2]].bits_uint32,
            "close embedding"
        ),
        6
    );
}

#[test]
fn directed_edges_contribute_to_both_endpoints_without_symmetrizing_distances() {
    let graph = graph(vec![0, 1, 3, 3, 3], vec![1, 0, 2], vec![0.25, 0.75, 0.5]);
    let distances = [2.0, 4.0, 1.0];
    let original = radii_from_edge_distances(&graph, &distances, RadiusSpace::Original).unwrap();
    assert_eq!(&original[..3], &[13.0_f32.ln(), 9.0_f32.ln(), 0.0]);
    assert!(original[3].is_nan());

    let embedding = radii_from_edge_distances(&graph, &distances, RadiusSpace::Embedding).unwrap();
    assert_eq!(&embedding[..3], &[3.5_f32.ln(), (4.0_f32 / 1.5).ln(), 0.0]);
    assert!(embedding[3].is_nan());
}

#[test]
fn zero_weight_is_undefined_and_zero_distance_uses_the_offset() {
    let graph = graph(vec![0, 1, 1, 2, 2], vec![1, 3], vec![1.0, 0.0]);
    for space in [RadiusSpace::Original, RadiusSpace::Embedding] {
        let radius = radii_from_edge_distances(&graph, &[0.0, 1.0], space).unwrap();
        assert_eq!(&radius[..2], &[1e-8_f32.ln(); 2]);
        assert!(radius[2..].iter().all(|x| x.is_nan()));
    }
}

#[test]
fn refit_keeps_self_when_coordinates_coincide() {
    let coordinates = [3.0, -1.0, 3.0, -1.0, 3.0, -1.0];
    let graph = refit_embedding_graph_euclidean(&coordinates, 3, 2, 2).unwrap();
    assert_eq!(graph.indptr, [0, 2, 3, 4]);
    assert_eq!(graph.indices, [1, 2, 0, 0]);
    assert_eq!(graph.data, [1.0; 4]);
    let radius = embedding_radii_euclidean(&coordinates, 3, 2, 2).unwrap();
    assert_eq!(radius, [1e-8_f32.ln(); 3]);
}

#[test]
fn invalid_edge_distances_and_coordinates_are_rejected() {
    let graph = graph(vec![0, 1, 1], vec![1], vec![0.0]);
    for distance in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            radii_from_edge_distances(&graph, &[distance], RadiusSpace::Original),
            Err(DensityError::InvalidDistance { row: 0, column: 1, distance: invalid })
                if invalid.to_bits() == distance.to_bits()
        ));
    }
    assert!(matches!(
        original_radii_euclidean(&graph, &[0.0, f32::INFINITY], 1),
        Err(DensityError::Matrix(KnnError::NonFiniteValue {
            index: 1,
            ..
        }))
    ));
    let extreme = [f32::MAX, -f32::MAX];
    assert!(matches!(
        original_radii_euclidean(&graph, &extreme, 1),
        Err(DensityError::InvalidDistance {
            row: 0,
            column: 1,
            ..
        })
    ));
    assert!(matches!(
        embedding_radii_euclidean(&extreme, 2, 1, 2),
        Err(DensityError::InvalidDistance {
            row: 0,
            column: 1,
            ..
        })
    ));
}

#[test]
fn malformed_graphs_and_shape_mismatches_are_rejected() {
    let valid = graph(vec![0, 1, 1], vec![1], vec![1.0]);
    assert_eq!(
        radii_from_edge_distances(&valid, &[], RadiusSpace::Original),
        Err(DensityError::InvalidDistanceCount {
            expected: 1,
            actual: 0
        })
    );
    assert!(matches!(
        original_radii_euclidean(&valid, &[0.0], 1),
        Err(DensityError::Matrix(KnnError::InvalidShape { .. }))
    ));
    assert!(matches!(
        embedding_radii_from_refit_euclidean(&valid, &[], 0),
        Err(DensityError::Matrix(KnnError::InvalidShape { .. }))
    ));
    assert!(matches!(
        embedding_radii_euclidean(&[], usize::MAX, 2, 2),
        Err(DensityError::Matrix(KnnError::InvalidShape { .. }))
    ));
    assert!(matches!(
        embedding_radii_euclidean(&[0.0, 1.0], 2, 1, 3),
        Err(DensityError::Matrix(KnnError::InvalidK { k: 3, n_rows: 2 }))
    ));
    for malformed in [
        CsrGraph {
            n_cols: 3,
            ..valid.clone()
        },
        graph(vec![0, 1, 1], vec![2], vec![1.0]),
        graph(vec![0, 0, 0], vec![1], vec![1.0]),
        graph(vec![0, 2, 1], vec![1], vec![1.0]),
        graph(vec![0, 2, 2], vec![1, 0], vec![1.0; 2]),
    ] {
        let distances = vec![1.0; malformed.data.len()];
        assert_eq!(
            radii_from_edge_distances(&malformed, &distances, RadiusSpace::Embedding),
            Err(DensityError::InvalidCsr)
        );
    }
    let bad_weight = graph(vec![0, 1, 1], vec![1], vec![-1.0]);
    assert_eq!(
        radii_from_edge_distances(&bad_weight, &[1.0], RadiusSpace::Embedding),
        Err(DensityError::InvalidWeight {
            entry: 0,
            weight: -1.0
        })
    );
}
