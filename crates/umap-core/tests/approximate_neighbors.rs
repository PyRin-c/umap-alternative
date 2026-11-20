use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::approximate_neighbors::{
    ApproximateSearchError, DenseApproximateIndex, RankedNeighbors, refine_training_neighbors,
    use_approximate_search,
};
use umap_core::fit::{fit_dense_graph_with_cutoff, fit_supplied_neighbors_graph};
use umap_core::fuzzy_graph::{CsrGraph, GraphOptions};
use umap_core::neighbors::DistanceMetric;
use umap_core::transform::{
    CooGraph, transform_dense_graph_with_cutoff, transform_ranked_neighbors_graph,
};

#[test]
fn reverse_co_candidates_can_change_a_later_round() {
    let data = [0.0, 3.0, 4.0, 7.0, 32.0, 15.0, 22.0, 52.0];
    let seeds = [0, -1, 1, 0, 2, 0, 3, 4, 4, 5, 5, 6, 6, 7, 7, 3];
    let run = |rounds| {
        refine_training_neighbors(&data, 8, 1, 2, DistanceMetric::Euclidean, &seeds, rounds)
            .unwrap()
    };
    let zero = run(0);
    assert_eq!(zero.indices, seeds);
    assert_eq!(zero.distances[1], f32::INFINITY);
    let first = run(1);
    assert_eq!(
        first.indices,
        [0, -1, 1, 2, 2, 1, 3, 5, 4, 6, 5, 6, 6, 4, 7, 4]
    );
    assert_eq!(
        first.distances,
        [
            0.0,
            f32::INFINITY,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            8.0,
            0.0,
            10.0,
            0.0,
            7.0,
            0.0,
            10.0,
            0.0,
            20.0
        ]
    );
    let second = run(2);
    assert_eq!(
        second.indices,
        [0, 1, 1, 2, 2, 1, 3, 5, 4, 6, 5, 6, 6, 5, 7, 4]
    );
    assert_eq!(
        second.distances,
        [
            0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 8.0, 0.0, 10.0, 0.0, 7.0, 0.0, 7.0, 0.0, 20.0
        ]
    );
    assert_eq!(second, run(3));
    let invalid = [0, 9, 1, 0, 2, 0, 3, 4, 4, 5, 5, 6, 6, 7, 7, 3];
    assert_eq!(
        refine_training_neighbors(&data, 8, 1, 2, DistanceMetric::Euclidean, &invalid, 0),
        Err(ApproximateSearchError::InvalidSeed {
            position: 1,
            index: 9
        }),
    );
}

#[test]
fn keeps_small_data_routing_and_wide_rank_scales_separate() {
    assert!(!use_approximate_search(1, true));
    assert!(!use_approximate_search(4095, false));
    assert!(use_approximate_search(4095, true));
    assert!(use_approximate_search(4096, false));

    let data = [0.0, 1.0, 4.0, 10.0, 18.0];
    let metric = DistanceMetric::Euclidean;
    let index = DenseApproximateIndex::build(&data, 5, 1, 8, metric, 42).unwrap();
    let ranks = index.training_ranks();
    assert_eq!(ranks.indices.len(), 40);
    assert_eq!(&ranks.indices[..8], &[0, 1, 2, 3, -1, -1, -1, -1]);
    let options = GraphOptions::default();
    let graph = fit_supplied_neighbors_graph(
        &ranks.indices,
        &ranks.distances,
        5,
        8,
        8,
        options,
        f32::INFINITY,
    )
    .unwrap();
    assert_eq!(graph.data.len(), 18);

    let queries = index.query(&[-1.0, 2.0, 8.0, 20.0], 4).unwrap();
    assert_eq!(&queries.indices[..8], &[0, 1, 2, 3, 4, -1, -1, -1]);
    assert_eq!(index.query_scale_neighbors(), 4);
    let graph = transform_ranked_neighbors_graph(
        &queries.indices,
        &queries.distances,
        4,
        5,
        8,
        4,
        f32::INFINITY,
        options,
    )
    .unwrap();
    for row in graph.data.chunks_exact(8) {
        assert!((row[1..5].iter().sum::<f32>() - 2.0).abs() < 8e-6);
        assert_eq!(&row[5..], &[0.0; 3]);
    }
    assert!(index.query(&[], 0).unwrap().indices.is_empty());
    assert!(matches!(
        index.query(&[f32::NAN], 1),
        Err(ApproximateSearchError::Matrix(_))
    ));
}

#[derive(Deserialize)]
struct Observations {
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    metric: String,
    cutoff: Option<f32>,
    n_rows: usize,
    n_features: usize,
    width: usize,
    fitted_k: usize,
    n_query_rows: usize,
    iterations: usize,
    training_bits: Vec<u32>,
    query_bits: Vec<u32>,
    snapshots: BTreeMap<usize, Snapshot>,
    restarts: BTreeMap<usize, Snapshot>,
    forced: Graphs,
    automatic: Option<Graphs>,
    query_epsilon_sweep: BTreeMap<String, Snapshot>,
}

#[derive(Deserialize)]
struct Snapshot {
    indices: Vec<i32>,
    bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Graphs {
    fit: Graph,
    query: Graph,
}

#[derive(Deserialize)]
struct Graph {
    shape: [usize; 2],
    columns: Vec<usize>,
    bits: Vec<u32>,
    #[serde(default)]
    rows: Vec<usize>,
    #[serde(default)]
    indptr: Vec<usize>,
}

impl Graph {
    fn csr(graph: CsrGraph) -> Self {
        Self {
            shape: [graph.n_rows, graph.n_cols],
            columns: graph.indices,
            bits: graph.data.into_iter().map(f32::to_bits).collect(),
            rows: Vec::new(),
            indptr: graph.indptr,
        }
    }

    fn coo(graph: CooGraph) -> Self {
        Self {
            shape: [graph.n_rows, graph.n_cols],
            columns: graph.col,
            bits: graph.data.into_iter().map(f32::to_bits).collect(),
            rows: graph.row,
            indptr: Vec::new(),
        }
    }

    fn entries(&self) -> BTreeMap<(usize, usize), u32> {
        let expanded;
        let rows = if self.indptr.is_empty() {
            &self.rows
        } else {
            expanded = self
                .indptr
                .windows(2)
                .enumerate()
                .flat_map(|(row, pair)| std::iter::repeat_n(row, pair[1] - pair[0]))
                .collect();
            &expanded
        };
        rows.iter()
            .copied()
            .zip(self.columns.iter().copied())
            .zip(self.bits.iter().copied())
            .filter(|&(_, bits)| bits & 0x7fff_ffff != 0)
            .collect()
    }
}

fn compare(label: &str, native: &Graph, wheel: &Graph) {
    assert_eq!(native.shape, wheel.shape, "{label}");
    let left = native.entries();
    let right = wheel.entries();
    let shared = left.keys().filter(|key| right.contains_key(key)).count();
    let changed = left
        .iter()
        .filter(|(key, bits)| right.get(key).is_some_and(|rhs| rhs != *bits))
        .count();
    print!(
        "{label}: stored {}/{}, native-only {}, wheel-only {}, changed {changed}/{shared}",
        native.bits.len(),
        wheel.bits.len(),
        left.len() - shared,
        right.len() - shared
    );
    if native.indptr.is_empty() {
        let slots = native.rows.iter().zip(&native.columns).zip(&native.bits);
        let wheel_slots = wheel.rows.iter().zip(&wheel.columns).zip(&wheel.bits);
        let (mut coordinates, mut triples) = (0, 0);
        for ((left_pos, left_bits), (right_pos, right_bits)) in slots.zip(wheel_slots) {
            coordinates += usize::from(left_pos == right_pos);
            triples += usize::from(left_pos == right_pos && left_bits == right_bits);
        }
        print!(", ordered coordinates {coordinates}, ordered triples {triples}");
    }
    println!();
}

fn distances(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn report_rank_match(label: &str, native: &RankedNeighbors, wheel: &Snapshot) {
    let ranks = native.indices.iter().zip(&native.distances);
    let expected = wheel.indices.iter().zip(&wheel.bits);
    let (mut columns, mut bits) = (0, 0);
    for ((a, distance), (b, expected_bits)) in ranks.zip(expected) {
        columns += usize::from(a == b);
        bits += usize::from(a == b && distance.to_bits() == *expected_bits);
    }
    println!(
        "  {label}: aligned {columns} columns and {bits} pairs of {}",
        wheel.indices.len()
    );
}

#[test]
#[ignore = "requires the public-wheel capture from reference/probe_approximate_search.py"]
fn compare_checked_cases_with_public_wheel() {
    let path = std::env::var_os("APPROXIMATE_SEARCH_OBSERVATIONS").unwrap();
    let observations: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(!observations.cases.is_empty());
    for (name, case) in observations.cases {
        let metric = match case.metric.as_str() {
            "euclidean" => DistanceMetric::Euclidean,
            "cosine" => DistanceMetric::Cosine,
            "correlation" => DistanceMetric::Correlation,
            metric => panic!("unknown fixture metric {metric}"),
        };
        let cutoff = case
            .cutoff
            .or(metric.default_disconnection_distance())
            .unwrap_or(f32::INFINITY);
        let options = GraphOptions::default();
        let training = distances(&case.training_bits);
        let queries = distances(&case.query_bits);
        let graph_fit = |ranks: &RankedNeighbors| {
            Graph::csr(
                fit_supplied_neighbors_graph(
                    &ranks.indices,
                    &ranks.distances,
                    case.n_rows,
                    ranks.width,
                    ranks.width,
                    options,
                    cutoff,
                )
                .unwrap(),
            )
        };
        let index = DenseApproximateIndex::build(
            &training,
            case.n_rows,
            case.n_features,
            case.width,
            metric,
            42,
        )
        .unwrap();
        let native_fit = graph_fit(index.training_ranks());
        let query_ranks = index.query(&queries, case.n_query_rows).unwrap();
        let native_query = Graph::coo(
            transform_ranked_neighbors_graph(
                &query_ranks.indices,
                &query_ranks.distances,
                case.n_query_rows,
                case.n_rows,
                query_ranks.width,
                index.query_scale_neighbors(),
                cutoff,
                options,
            )
            .unwrap(),
        );
        println!("{name}");
        compare("  native approximate CSR", &native_fit, &case.forced.fit);
        compare(
            "  native approximate COO",
            &native_query,
            &case.forced.query,
        );
        let epsilon = if metric == DistanceMetric::Euclidean {
            "0.12"
        } else {
            "0.24"
        };
        if let Some(wheel) = case.query_epsilon_sweep.get(epsilon) {
            let query = Graph::coo(
                transform_ranked_neighbors_graph(
                    &wheel.indices,
                    &distances(&wheel.bits),
                    case.n_query_rows,
                    case.n_rows,
                    case.width,
                    case.fitted_k,
                    cutoff,
                    options,
                )
                .unwrap(),
            );
            compare("  wheel ranks replay COO", &query, &case.forced.query);
        }

        let zero = &case.snapshots[&0];
        for rounds in [0, 1, case.iterations] {
            let refined = refine_training_neighbors(
                &training,
                case.n_rows,
                case.n_features,
                case.fitted_k,
                metric,
                &zero.indices,
                rounds,
            )
            .unwrap();
            let wheel = &case.snapshots[&rounds];
            report_rank_match(&format!("wheel seeds round {rounds}"), &refined, wheel);
            if let Some(restart) = case.restarts.get(&rounds) {
                report_rank_match(&format!("wheel restart round {rounds}"), &refined, restart);
                if rounds == 1 {
                    assert_eq!(
                        refined.indices, restart.indices,
                        "{name}, restarted first round"
                    );
                }
            }
            if rounds == case.iterations && case.fitted_k == case.width {
                compare(
                    "  wheel seeds refined CSR",
                    &graph_fit(&refined),
                    &case.forced.fit,
                );
                let wheel_ranks = RankedNeighbors {
                    width: case.width,
                    indices: wheel.indices.clone(),
                    distances: distances(&wheel.bits),
                };
                compare(
                    "  wheel ranks replay CSR",
                    &graph_fit(&wheel_ranks),
                    &case.forced.fit,
                );
            }
        }

        if let Some(automatic) = case.automatic {
            let exact_query = Graph::coo(
                transform_dense_graph_with_cutoff(
                    &training,
                    case.n_rows,
                    &queries,
                    case.n_query_rows,
                    case.n_features,
                    case.fitted_k,
                    metric,
                    cutoff,
                    options,
                )
                .unwrap(),
            );
            let exhaustive = exact_query.entries();
            let outside_exact = |graph: &Graph| {
                graph
                    .entries()
                    .keys()
                    .filter(|key| !exhaustive.contains_key(key))
                    .count()
            };
            println!(
                "  approximate COO coordinates outside exhaustive ranks: native {}, wheel {}",
                outside_exact(&native_query),
                outside_exact(&case.forced.query)
            );
            if use_approximate_search(case.n_rows, false) {
                compare("  native automatic CSR", &native_fit, &automatic.fit);
                compare("  native automatic COO", &native_query, &automatic.query);
            } else {
                let (fit, _) = fit_dense_graph_with_cutoff(
                    &training,
                    case.n_rows,
                    case.n_features,
                    case.fitted_k,
                    metric,
                    cutoff,
                    options,
                )
                .unwrap();
                compare("  native automatic CSR", &Graph::csr(fit), &automatic.fit);
                compare("  native automatic COO", &exact_query, &automatic.query);
            }
        }
    }
}
