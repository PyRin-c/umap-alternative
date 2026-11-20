use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use serde::Deserialize;
use umap_core::approximate_neighbors::{
    ApproximateSearchError, DenseApproximateQueryIndex, RankedNeighbors,
};

#[test]
fn live_offers_leave_the_public_draw_limit_in_place() {
    let data: Vec<f32> = (0..25).map(|row| (row * row) as f32).collect();
    let index =
        DenseApproximateQueryIndex::from_training_graph(&data, 25, 1, &[-1; 75], 3, 42, None)
            .unwrap();
    let expected = index.query(&[82.5, 85.0], 2, 5, 0.12).unwrap();
    let widened = index
        .with_live_training_offers(&[-1; 100], 4, None)
        .unwrap();
    assert_eq!(widened.query(&[82.5, 85.0], 2, 5, 0.12).unwrap(), expected);
    assert!(matches!(
        widened.with_live_training_offers(&[-1; 99], 4, None),
        Err(ApproximateSearchError::RankShape)
    ));
}

#[test]
fn a_vacant_prepared_heap_stops_after_the_nearest_rows_links() {
    let data = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let mut graph = [-1; 10];
    graph[1] = 9;
    let prepared =
        DenseApproximateQueryIndex::from_training_graph(&data, 10, 1, &graph, 1, 42, None)
            .unwrap()
            .with_projection_words(&[[0, 5]], NonZeroUsize::new(6).unwrap(), 1)
            .unwrap();
    let zero = [0, 1, 2, 3, -1, -1, -1, -1, -1, -1];
    for epsilon in [0.0, f64::from_bits(1)] {
        assert_eq!(
            prepared.query(&[0.0], 1, 10, epsilon).unwrap().indices,
            zero
        );
    }
    assert_eq!(
        prepared
            .query(&[0.0], 1, 10, f64::from(f32::from_bits(1)))
            .unwrap()
            .indices,
        [0, 1, 2, 3, 9, -1, -1, -1, -1, -1]
    );
}

#[derive(Deserialize)]
struct Observations {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    d: usize,
    caller: u32,
    cap: NonZeroUsize,
    depth: u8,
    words: Vec<[u64; 2]>,
    orientation_scores: Vec<usize>,
    train_bits: Vec<u32>,
    query_bits: Vec<u32>,
    graph: Vec<i32>,
    live: Vec<i32>,
    prepared: Vec<Sweep>,
    wide: Vec<Sweep>,
    plain: Vec<Sweep>,
    wheel_ns: u64,
    wheel_plain_ns: u64,
}

#[derive(Deserialize)]
struct Sweep {
    width: usize,
    epsilon: f64,
    indices: Vec<i32>,
    bits: Vec<u32>,
}

type RankResidual = (usize, f64, usize, Vec<i32>, Vec<i32>);

#[derive(Default)]
struct Counts {
    slots: usize,
    ranks: usize,
    bits: usize,
    shared: usize,
    shared_bits: usize,
    wheel_only: usize,
    rust_only: usize,
    first_residual: Option<RankResidual>,
}

impl Counts {
    fn compare(&mut self, observed: RankedNeighbors, expected: &Sweep) {
        self.slots += expected.indices.len();
        self.ranks += observed
            .indices
            .iter()
            .zip(&expected.indices)
            .filter(|(left, right)| left == right)
            .count();
        self.bits += observed
            .distances
            .iter()
            .zip(&expected.bits)
            .filter(|(left, right)| left.to_bits() == **right)
            .count();
        let w = expected.width;
        for row in 0..expected.indices.len() / w {
            let range = row * w..(row + 1) * w;
            let wheel = &expected.indices[range.clone()];
            let rust = &observed.indices[range.clone()];
            let mut wheel_only = Vec::new();
            for (position, &id) in wheel.iter().enumerate().filter(|(_, id)| **id >= 0) {
                if let Some(other) = rust.iter().position(|&vertex| vertex == id) {
                    self.shared += 1;
                    self.shared_bits += usize::from(
                        observed.distances[range.start + other].to_bits()
                            == expected.bits[range.start + position],
                    );
                } else {
                    wheel_only.push(id);
                }
            }
            let rust_only: Vec<_> = rust
                .iter()
                .copied()
                .filter(|&id| id >= 0 && !wheel.contains(&id))
                .collect();
            self.wheel_only += wheel_only.len();
            self.rust_only += rust_only.len();
            if self.first_residual.is_none() && self.ranks < self.slots && wheel != rust {
                self.first_residual = Some((w, expected.epsilon, row, wheel_only, rust_only));
            }
        }
    }

    fn print(&self, label: &str) {
        println!(
            "{label} ranks {}/{}, bits {}/{}, shared bits {}/{}, wheel-only {}, Rust-only {}, first residual {:?}",
            self.ranks,
            self.slots,
            self.bits,
            self.slots,
            self.shared_bits,
            self.shared,
            self.wheel_only,
            self.rust_only,
            self.first_residual
        );
    }

    fn assert_exact(&self, name: &str) {
        assert_eq!(self.ranks, self.slots, "{name}");
        assert_eq!(self.bits, self.slots, "{name}");
    }
}

#[test]
#[ignore = "requires reference/probe_prepared_traversal.py output"]
fn compare_prepared_traversal_with_wheel() {
    let path = std::env::var_os("PREPARED_TRAVERSAL_OBSERVATIONS").unwrap();
    let observations: Observations = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in observations.cases {
        let decode = |raw: Vec<u32>| raw.into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let data = decode(case.train_bits);
        let queries = decode(case.query_bits);
        let count = queries.len() / case.d;
        let plain = || {
            DenseApproximateQueryIndex::from_training_graph(
                &data,
                case.n,
                case.d,
                &case.graph,
                5,
                case.caller,
                None,
            )
            .unwrap()
        };
        let control = plain();
        let exported = plain()
            .with_projection_words(&case.words, case.cap, case.depth)
            .unwrap();
        let prepared = plain()
            .with_live_training_offers(&case.live, 6, None)
            .unwrap()
            .with_projection_words(&case.words, case.cap, case.depth)
            .unwrap();
        let measure = |index: &DenseApproximateQueryIndex<'_>, sweeps: &[Sweep]| {
            let mut total = Counts::default();
            for run in sweeps {
                total.compare(
                    index
                        .query(&queries, count, run.width, run.epsilon)
                        .unwrap(),
                    run,
                );
            }
            total
        };
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
        println!("{} packing scores {:?}", case.name, case.orientation_scores);
        measure(&exported, &case.prepared).print("exported ordinary");
        measure(&exported, &case.wide).print("exported wide");
        let ordinary = measure(&prepared, &case.prepared);
        let wide = measure(&prepared, &case.wide);
        let graph_only = measure(&control, &case.plain);
        ordinary.print("live ordinary");
        wide.print("live wide");
        graph_only.print("graph-only");
        println!(
            "{count} queries, warm ns exported {}, prepared Rust {} wheel {}, graph-only Rust {} wheel {}",
            timing(&exported),
            timing(&prepared),
            case.wheel_ns,
            timing(&control),
            case.wheel_plain_ns
        );
        ordinary.assert_exact(&case.name);
        wide.assert_exact(&case.name);
        graph_only.assert_exact(&case.name);
    }
}
