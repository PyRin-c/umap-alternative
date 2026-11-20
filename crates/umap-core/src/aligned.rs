//! Serial optimization of related fuzzy graphs, including frozen slices on append.

use std::collections::HashSet;

use crate::fuzzy_graph::CsrGraph;
use crate::layout::DistanceReduction;
use crate::rng::{Mt19937, sample_vertex};
use crate::sampling::seeded_train_positive_interval;

// The joint schedule has its own curve and sample rate. Mapper settings
// determine the graphs but leave these fixed in the public wheel probes.
const A: f64 = 1.576_943_460_196_219_6;
const B: f64 = 0.895_060_878_125_481_2;
const NEGATIVE_RATE: f64 = 5.0;

/// Settings for the joint layout sweep.
pub struct AlignedLayoutConfig {
    pub n_components: usize,
    pub n_epochs: usize,
    pub first_moving: usize,
    pub window: usize,
    pub regularisation: f64,
    pub seed: [i64; 3],
}

#[derive(Clone, Copy)]
struct Anchor {
    slice: usize,
    row: usize,
    weight: f64,
}

struct Edge {
    head: usize,
    tail: usize,
    interval: f64,
    negative_interval: f32,
    next_positive: f64,
    next_negative: f32,
}

impl Edge {
    fn new(head: usize, tail: usize, interval: f64) -> Self {
        let negative_interval = (interval / NEGATIVE_RATE) as f32;
        Self {
            head,
            tail,
            interval,
            negative_interval,
            next_positive: interval,
            next_negative: negative_interval,
        }
    }

    fn take_samples(&mut self, epoch: f64) -> Option<usize> {
        if self.next_positive > epoch {
            return None;
        }
        let count =
            ((epoch - f64::from(self.next_negative)) / f64::from(self.negative_interval)) as usize;
        self.next_positive += self.interval;
        self.next_negative = (f64::from(self.next_negative)
            + count as f64 * f64::from(self.negative_interval)) as f32;
        Some(count)
    }
}

fn row_indices(graph: &CsrGraph, row: usize) -> &[usize] {
    &graph.indices[graph.indptr[row]..graph.indptr[row + 1]]
}

fn compose(previous: &[Option<usize>], adjacent: &[Option<usize>]) -> Vec<Option<usize>> {
    previous
        .iter()
        .map(|row| row.and_then(|row| adjacent[row]))
        .collect()
}

fn weights_for_pair(
    source: &CsrGraph,
    target: &CsrGraph,
    mapping: &[Option<usize>],
    target_index: usize,
    distance: usize,
    rows: &mut [Vec<Anchor>],
) {
    let targets: HashSet<_> = mapping.iter().copied().flatten().collect();
    let decay = (1.0 - distance as f64).exp();
    for (source_row, target_row) in mapping.iter().copied().enumerate() {
        let Some(target_row) = target_row else {
            continue;
        };
        let left: HashSet<_> = row_indices(source, source_row)
            .iter()
            .filter_map(|&row| mapping[row])
            .collect();
        let right: HashSet<_> = row_indices(target, target_row)
            .iter()
            .copied()
            .filter(|row| targets.contains(row))
            .collect();
        let shared = left.intersection(&right).count();
        if shared > 0 {
            let union = left.len() + right.len() - shared;
            rows[source_row].push(Anchor {
                slice: target_index,
                row: target_row,
                weight: decay * shared as f64 / union as f64,
            });
        }
    }
}

fn alignment_anchors(
    graphs: &[CsrGraph],
    relations: &[Vec<(usize, usize)>],
    config: &AlignedLayoutConfig,
) -> Result<Vec<Vec<Vec<Anchor>>>, &'static str> {
    let mut forward = Vec::with_capacity(relations.len());
    let mut backward = Vec::with_capacity(relations.len());
    for (index, links) in relations.iter().enumerate() {
        let mut left = vec![None; graphs[index].n_rows];
        let mut right = vec![None; graphs[index + 1].n_rows];
        for &(source, target) in links {
            if source >= left.len() || target >= right.len() {
                return Err("relation row is outside the corresponding slice");
            }
            left[source] = Some(target);
            right[target] = Some(source);
        }
        forward.push(left);
        backward.push(right);
    }
    let mut anchors = Vec::with_capacity(graphs.len() - config.first_moving);
    for i in config.first_moving..graphs.len() {
        let mut rows = vec![Vec::new(); graphs[i].n_rows];
        let start = i.saturating_sub(config.window);
        let end = i
            .saturating_add(config.window)
            .saturating_add(1)
            .min(graphs.len());
        let identity: Vec<_> = (0..graphs[i].n_rows).map(Some).collect();
        if config.regularisation != 0.0 {
            let mut mapping = identity.clone();
            for j in (start..i).rev() {
                mapping = compose(&mapping, &backward[j]);
                weights_for_pair(&graphs[i], &graphs[j], &mapping, j, i - j, &mut rows);
            }
            // Keep the accumulation order from earlier to later slices.
            for row in &mut rows {
                row.reverse();
            }
            mapping = identity;
            for j in i + 1..end {
                mapping = compose(&mapping, &forward[j - 1]);
                weights_for_pair(&graphs[i], &graphs[j], &mapping, j, j - i, &mut rows);
            }
        }
        anchors.push(rows);
    }
    Ok(anchors)
}

fn squared(left: &[f32], right: &[f32]) -> f64 {
    f64::from(DistanceReduction::SeededCompatibility.squared(left, right))
}

fn attraction(distance: f64) -> f64 {
    if distance > 0.0 {
        -2.0 * A * B * distance.powf(B - 1.0) / (1.0 + A * distance.powf(B))
    } else {
        0.0
    }
}

fn repulsion(distance: f64) -> f64 {
    if distance > 0.0 {
        2.0 * B / ((0.001 + distance) * (1.0 + A * distance.powf(B)))
    } else {
        0.0
    }
}

fn row(layout: &[f32], index: usize, dim: usize) -> &[f32] {
    &layout[index * dim..(index + 1) * dim]
}

fn alignment_gradient(
    layouts: &[Vec<f32>],
    anchors: &[Anchor],
    component: usize,
    coordinate: f32,
    config: &AlignedLayoutConfig,
) -> f64 {
    anchors.iter().fold(0.0, |sum, anchor| {
        let target = layouts[anchor.slice][anchor.row * config.n_components + component];
        if target.is_finite() {
            sum + f64::from(target - coordinate) * anchor.weight * config.regularisation
        } else {
            sum
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn update_edge(
    layouts: &mut [Vec<f32>],
    anchors: &[Vec<Anchor>],
    slice: usize,
    edge: &Edge,
    negative_count: usize,
    alpha: f64,
    seed: &mut [i64; 3],
    config: &AlignedLayoutConfig,
) {
    let dim = config.n_components;
    let head = edge.head * dim;
    let tail = edge.tail * dim;
    let factor = attraction(squared(
        row(&layouts[slice], edge.head, dim),
        row(&layouts[slice], edge.tail, dim),
    ));
    for component in 0..dim {
        let current = layouts[slice][head + component];
        let other = layouts[slice][tail + component];
        let base = (factor * f64::from(current - other)).clamp(-4.0, 4.0);
        let head_regularisation =
            alignment_gradient(layouts, &anchors[edge.head], component, current, config);
        layouts[slice][head + component] =
            (f64::from(current) + alpha * (base + head_regularisation)) as f32;
        if config.first_moving == 0 {
            let tail_regularisation =
                alignment_gradient(layouts, &anchors[edge.tail], component, other, config);
            layouts[slice][tail + component] =
                (f64::from(other) + alpha * (-base + tail_regularisation)) as f32;
        }
    }

    let n_rows = layouts[slice].len() / dim;
    for _ in 0..negative_count {
        let sample = sample_vertex(seed, n_rows);
        if sample == edge.head {
            continue;
        }
        let factor = repulsion(squared(
            row(&layouts[slice], edge.head, dim),
            row(&layouts[slice], sample, dim),
        ));
        let other_start = sample * dim;
        for component in 0..dim {
            let current = layouts[slice][head + component];
            let other = layouts[slice][other_start + component];
            let base = (factor * f64::from(current - other)).clamp(-4.0, 4.0);
            let regularisation =
                alignment_gradient(layouts, &anchors[edge.head], component, current, config);
            layouts[slice][head + component] =
                (f64::from(current) + alpha * (base + regularisation)) as f32;
        }
    }
}

fn validate(
    graphs: &[CsrGraph],
    layouts: &[Vec<f32>],
    relations: &[Vec<(usize, usize)>],
    config: &AlignedLayoutConfig,
) -> Result<(), &'static str> {
    if graphs.is_empty()
        || graphs.len() != layouts.len()
        || relations.len() != graphs.len() - 1
        || config.first_moving >= graphs.len()
        || config.n_components == 0
        || !config.regularisation.is_finite()
    {
        return Err("inconsistent aligned graph, relation, or layout dimensions");
    }
    for (slice, (graph, layout)) in graphs.iter().zip(layouts).enumerate() {
        if graph.n_rows == 0
            || graph.n_rows != graph.n_cols
            || graph.n_rows.checked_mul(config.n_components) != Some(layout.len())
            || graph.indptr.len() != graph.n_rows + 1
            || graph.indptr.first() != Some(&0)
            || graph.indptr.last() != Some(&graph.indices.len())
            || graph.indptr.windows(2).any(|x| x[0] > x[1])
            || graph.indices.len() != graph.data.len()
            || graph.indices.iter().any(|&col| col >= graph.n_rows)
            || graph
                .data
                .iter()
                .any(|&weight| !(0.0..=1.0).contains(&weight))
        {
            return Err("invalid aligned layout or CSR graph");
        }
        for index in 0..graph.n_rows {
            if row_indices(graph, index).contains(&index) {
                return Err("aligned CSR contains a self edge");
            }
        }
        if slice >= config.first_moving && layout.iter().any(|value| !value.is_finite()) {
            return Err("moving aligned layouts must contain finite coordinates");
        }
    }
    Ok(())
}

fn shuffled_slice_order(order: &mut [usize], seed: i64) -> Mt19937 {
    let mut rng = Mt19937::new(seed.unsigned_abs() as u32);
    for i in (1..order.len()).rev() {
        let mask = (i + 1).next_power_of_two() - 1;
        let j = loop {
            let candidate = (rng.next_u32() as usize) & mask;
            if candidate <= i {
                break candidate;
            }
        };
        order.swap(i, j);
    }
    rng
}

/// Jointly optimize CSR graph layouts. Slices before `first_moving` are anchors.
/// Return the epoch shuffle stream at the point used by the next append.
pub fn optimize_aligned_layout(
    graphs: &[CsrGraph],
    layouts: &mut [Vec<f32>],
    relations: &[Vec<(usize, usize)>],
    config: &AlignedLayoutConfig,
) -> Result<Option<Mt19937>, &'static str> {
    validate(graphs, layouts, relations, config)?;
    let anchors = alignment_anchors(graphs, relations, config)?;
    if config.n_epochs == 0 {
        return Ok(None);
    }
    let mut schedules: Vec<Vec<Edge>> = Vec::with_capacity(anchors.len());
    for graph in &graphs[config.first_moving..] {
        let max = graph.data.iter().copied().fold(0.0, f32::max);
        let mut edges = Vec::with_capacity(graph.data.len());
        for head in 0..graph.n_rows {
            for position in graph.indptr[head]..graph.indptr[head + 1] {
                if let Some(interval) =
                    seeded_train_positive_interval(config.n_epochs, max, graph.data[position])
                {
                    edges.push(Edge::new(head, graph.indices[position], interval));
                }
            }
        }
        schedules.push(edges);
    }
    let mut seed = config.seed;
    let mut alpha = 1.0;
    let mut order = (0..graphs.len()).collect::<Vec<_>>();
    let mut shuffle_rng = shuffled_slice_order(&mut order, seed[0]);
    for epoch in 1..config.n_epochs {
        order
            .iter_mut()
            .enumerate()
            .for_each(|(i, slice)| *slice = i);
        shuffle_rng = shuffled_slice_order(&mut order, seed[0]);
        for slice in order.iter().copied().filter(|&i| i >= config.first_moving) {
            let moving = slice - config.first_moving;
            for edge in &mut schedules[moving] {
                if let Some(count) = edge.take_samples(epoch as f64) {
                    update_edge(
                        layouts,
                        &anchors[moving],
                        slice,
                        edge,
                        count,
                        alpha,
                        &mut seed,
                        config,
                    );
                }
            }
        }
        alpha = 1.0 - epoch as f64 / config.n_epochs as f64;
    }
    if layouts[config.first_moving..]
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("aligned optimization produced non-finite coordinates");
    }
    Ok(Some(shuffle_rng))
}
