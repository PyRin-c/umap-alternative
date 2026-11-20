use std::hint::black_box;
use std::time::Instant;

use serde::Deserialize;
use umap_core::approximate_neighbors::{ApproximateSearchError, DenseApproximateQueryIndex};
use umap_core::fit::fit_supplied_neighbors_graph;
use umap_core::fuzzy_graph::GraphOptions;
use umap_core::transform::transform_ranked_neighbors_graph;

#[test]
fn random_entries_restart_per_batch_and_consume_duplicate_attempts() {
    let data: Vec<f32> = (0..25).map(|row| (row * row) as f32).collect();
    let queries = [82.5, 85.0, 82.5, 83.0, 82.5];
    let search =
        DenseApproximateQueryIndex::from_training_graph(&data, 25, 1, &[-1; 75], 3, 42, None)
            .unwrap();
    let batch = search.query(&queries, 5, 2, 0.12).unwrap();
    assert_eq!(batch.indices, [6, 18, 6, 24, 16, 23, 3, 1, 11, -1]);
    assert_eq!(batch.distances[9], f32::INFINITY);
    assert_eq!(search.query(&queries, 5, 2, 0.12).unwrap(), batch);
    assert_eq!(
        search.query(&queries[..1], 1, 2, 0.12).unwrap().indices,
        [6, 18]
    );
    assert!(matches!(
        search.query(&queries, 5, 2, f64::NAN),
        Err(ApproximateSearchError::InvalidEpsilon)
    ));
}

#[test]
fn squared_epsilon_crossing_can_reach_a_second_hop_in_either_direction() {
    let mut data = [1000.0; 25];
    data[6] = 10.0;
    data[8] = 10.5;
    data[9] = 0.0;
    for (a, b, c, d) in [(6, 8, 8, 9), (8, 6, 9, 8)] {
        let mut graph = [-1; 25];
        graph[a] = b;
        graph[c] = d;
        graph[12] = 15;
        let mut weights = [f32::INFINITY; 25];
        weights[a] = 100.0;
        weights[c] = 100.0;
        weights[12] = 50.0;
        let search = DenseApproximateQueryIndex::from_training_graph(
            &data,
            25,
            1,
            &graph,
            1,
            42,
            Some(&weights),
        )
        .unwrap();
        assert_eq!(search.query(&[0.0], 1, 1, 0.1).unwrap().indices, [6]);
        assert_eq!(search.query(&[0.0], 1, 1, 0.2).unwrap().indices, [6]);
        let expanded = search.query(&[0.0], 1, 1, 0.21).unwrap();
        assert_eq!(expanded.indices, [9]);
        assert_eq!(expanded.distances, [0.0]);
    }
}

#[derive(Deserialize)]
struct Observations {
    controls: Vec<Case>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    d: usize,
    seed: u32,
    cap: usize,
    train_bits: Vec<u32>,
    query_bits: Vec<u32>,
    graph: Vec<i32>,
    edge_bits: Option<Vec<u32>>,
    results: Vec<Sweep>,
    wheel_median_ns: Option<u64>,
    estimator: Option<Estimator>,
    exact_nearest: Option<Vec<i32>>,
    singleton: Option<Ranks>,
    reversed_batch: Option<Ranks>,
    wide_tiny: Option<Ranks>,
}

#[derive(Deserialize)]
struct Sweep {
    width: usize,
    epsilon: f64,
    #[serde(flatten)]
    ranks: Ranks,
}

#[derive(Deserialize)]
struct Ranks {
    indices: Vec<i32>,
    bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Estimator {
    exact_input_equals_fit: bool,
    reordered_format: String,
    reordered_nnz: usize,
    training: Ranks,
    fit: Csr,
    query: Coo,
    ranks: Sweep,
}

#[derive(Deserialize)]
struct Csr {
    indptr: Vec<usize>,
    columns: Vec<usize>,
    bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Coo {
    rows: Vec<usize>,
    columns: Vec<usize>,
    bits: Vec<u32>,
}

#[test]
#[ignore = "requires the capture from reference/probe_approximate_query.py"]
fn compare_fixed_training_graph_with_public_wheel() {
    let path = std::env::var_os("APPROXIMATE_QUERY_OBSERVATIONS").unwrap();
    let observations: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in observations.controls.into_iter().chain(observations.cases) {
        let decode = |bits: Vec<u32>| bits.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let data = decode(case.train_bits);
        let queries = decode(case.query_bits);
        let n_queries = queries.len() / case.d;
        let build = |distances: Option<&[f32]>| {
            DenseApproximateQueryIndex::from_training_graph(
                &data,
                case.n,
                case.d,
                &case.graph,
                case.cap,
                case.seed,
                distances,
            )
            .unwrap()
        };
        let edge_distances = case.edge_bits.map(decode);
        let search = build(edge_distances.as_deref());
        let unweighted = build(Some(&vec![0.0; case.graph.len()]));
        let mut slots = 0;
        let mut ordered = 0;
        let mut unweighted_ordered = 0;
        let mut zero_wide_matches = 0;
        let mut zero_wide_slots = 0;
        let mut shared = 0;
        let mut shared_bits = 0;
        let mut finite = 0;
        let mut nn_hits = 0;
        let mut nn_total = 0;
        for sweep in case.results {
            let actual = search
                .query(&queries, n_queries, sweep.width, sweep.epsilon)
                .unwrap();
            slots += actual.indices.len();
            let matches = actual
                .indices
                .iter()
                .zip(&sweep.ranks.indices)
                .filter(|(a, b)| a == b)
                .count();
            ordered += matches;
            if sweep.epsilon == 0.0 && sweep.width > case.cap {
                zero_wide_matches += matches;
                zero_wide_slots += actual.indices.len();
            }
            let baseline = unweighted
                .query(&queries, n_queries, sweep.width, sweep.epsilon)
                .unwrap();
            unweighted_ordered += baseline
                .indices
                .iter()
                .zip(&sweep.ranks.indices)
                .filter(|(a, b)| a == b)
                .count();
            for row in 0..n_queries {
                let start = row * sweep.width;
                let range = start..start + sweep.width;
                let native = &actual.indices[range.clone()];
                let expected = &sweep.ranks.indices[range.clone()];
                for (rank, &id) in expected.iter().enumerate().filter(|(_, id)| **id >= 0) {
                    finite += 1;
                    if let Some(offset) = native.iter().position(|&candidate| candidate == id) {
                        shared += 1;
                        shared_bits += usize::from(
                            actual.distances[start + offset].to_bits()
                                == sweep.ranks.bits[start + rank],
                        );
                    }
                }
                if expected[0] >= 0 {
                    nn_total += 1;
                    nn_hits += usize::from(native[0] == expected[0]);
                }
            }
        }
        println!(
            "{} first {nn_hits}/{nn_total} ordered {ordered}/{slots} zero floor {unweighted_ordered}/{slots} support {shared}/{finite} bits {shared_bits}/{shared}",
            case.name
        );
        if case.wheel_median_ns.is_none() {
            assert_eq!(ordered, slots, "{}", case.name);
            assert_eq!(shared_bits, shared, "{}", case.name);
        }
        if let Some(wheel_ns) = case.wheel_median_ns {
            assert_eq!(
                ordered - zero_wide_matches,
                slots - zero_wide_slots,
                "{}",
                case.name
            );
            assert_eq!(shared_bits, shared, "{}", case.name);
            let actual = search.query(&queries, n_queries, case.cap, 0.12).unwrap();
            let nearest = case.exact_nearest.unwrap();
            let hits = actual
                .indices
                .chunks_exact(case.cap)
                .zip(nearest)
                .filter(|(a, b)| a[0] == *b)
                .count();
            let singleton = case.singleton.unwrap();
            let one = search.query(&queries[..case.d], 1, case.cap, 0.12).unwrap();
            let reversed: Vec<_> = queries
                .chunks_exact(case.d)
                .rev()
                .flatten()
                .copied()
                .collect();
            let reversed = search.query(&reversed, n_queries, case.cap, 0.12).unwrap();
            let reverse_columns = reversed
                .indices
                .iter()
                .zip(case.reversed_batch.unwrap().indices)
                .filter(|(a, b)| **a == *b)
                .count();
            assert_eq!(one.indices, singleton.indices, "{}", case.name);
            assert_eq!(reverse_columns, reversed.indices.len(), "{}", case.name);
            println!(
                "  exact nearest {hits}/{n_queries}, singleton columns {}, reversed {reverse_columns}/{}",
                one.indices == singleton.indices,
                reversed.indices.len()
            );
            println!("  wide zero-epsilon ordered {zero_wide_matches}/{zero_wide_slots}");
            let tiny_expected = case.wide_tiny.unwrap();
            let tiny = search.query(&queries[..case.d], 1, 13, 1e-300).unwrap();
            assert_eq!(tiny.indices, tiny_expected.indices, "{}", case.name);
            let filled = |indices: &[i32]| indices.iter().filter(|&&index| index >= 0).count();
            let zero = search.query(&queries[..case.d], 1, 13, 0.0).unwrap();
            println!(
                "  tiny epsilon first row ordered {}/13, native epsilon-zero fill {}/13",
                tiny.indices
                    .iter()
                    .zip(&tiny_expected.indices)
                    .filter(|(a, b)| a == b)
                    .count(),
                filled(&zero.indices)
            );
            let mut samples = [0; 31];
            for sample in &mut samples {
                let start = Instant::now();
                black_box(
                    search
                        .query(black_box(&queries), n_queries, case.cap, 0.12)
                        .unwrap(),
                );
                *sample = start.elapsed().as_nanos();
            }
            samples.sort_unstable();
            println!(
                "  median batch query native {} ns, wheel {wheel_ns} ns",
                samples[15]
            );
        }
        if let Some(reference) = case.estimator {
            let options = GraphOptions::default();
            let actual = fit_supplied_neighbors_graph(
                &reference.training.indices,
                &decode(reference.training.bits),
                case.n,
                case.cap,
                case.cap,
                options,
                f32::INFINITY,
            )
            .unwrap();
            let baseline = reference.fit;
            let csr_bits = actual
                .data
                .iter()
                .zip(&baseline.bits)
                .filter(|(a, b)| a.to_bits() == **b)
                .count();
            assert_eq!(actual.indptr, baseline.indptr, "{}", case.name);
            assert_eq!(actual.indices, baseline.columns, "{}", case.name);
            assert_eq!(csr_bits, baseline.bits.len(), "{}", case.name);
            assert!(reference.exact_input_equals_fit, "{}", case.name);
            assert_eq!(reference.reordered_format, "coo", "{}", case.name);
            println!(
                "  csr bits {csr_bits}/{}, exact input {}, reverse {} {} entries",
                actual.data.len(),
                reference.exact_input_equals_fit,
                reference.reordered_format,
                reference.reordered_nnz
            );

            let sweep = reference.ranks;
            let query = |indices, distances| {
                transform_ranked_neighbors_graph(
                    indices,
                    distances,
                    n_queries,
                    case.n,
                    sweep.width,
                    case.cap,
                    f32::INFINITY,
                    options,
                )
                .unwrap()
            };
            let expected_distances = decode(sweep.ranks.bits);
            let wheel_coo = query(&sweep.ranks.indices, &expected_distances);
            assert_eq!(wheel_coo.row, reference.query.rows, "{}", case.name);
            assert_eq!(wheel_coo.col, reference.query.columns, "{}", case.name);
            let replay_bits = wheel_coo
                .data
                .iter()
                .zip(&reference.query.bits)
                .filter(|(a, b)| a.to_bits() == **b)
                .count();
            let ranks = search
                .query(&queries, n_queries, sweep.width, sweep.epsilon)
                .unwrap();
            let native_coo = query(&ranks.indices, &ranks.distances);
            let mut coords = 0;
            let mut exact = 0;
            for (i, (&row, &col)) in native_coo.row.iter().zip(&native_coo.col).enumerate() {
                if reference.query.rows[i] == row && reference.query.columns[i] == col {
                    coords += 1;
                    exact += usize::from(native_coo.data[i].to_bits() == reference.query.bits[i]);
                }
            }
            assert_eq!(
                native_coo.data.len(),
                reference.query.bits.len(),
                "{}",
                case.name
            );
            assert_eq!(exact, reference.query.bits.len(), "{}", case.name);
            println!(
                "  coo k={} eps={} coordinates {coords}/{} exact triples {exact}/{} replay bits {replay_bits}/{}",
                sweep.width,
                sweep.epsilon,
                reference.query.bits.len(),
                reference.query.bits.len(),
                reference.query.bits.len()
            );
        }
    }
}
