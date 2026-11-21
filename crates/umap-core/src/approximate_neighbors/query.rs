use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

use super::projection::{
    PreparedProjection, angular_margin, partition_rows, prepared_margin, projection_pivots,
};
use super::{
    ApproximateSearchError, DenseRows, RankedNeighbors, TrainingNeighbor, Tree,
    approximate_squared_euclidean, cosine,
};
use crate::neighbors::{DistanceMetric, Neighbor};
use crate::rng::{Mt19937, draw_negative_seed, sample_absolute_vertex};

/// Approximate Euclidean queries over a supplied training graph.
///
/// The index uses an unpruned symmetric union of the supplied directed links.
/// Entry points are random by default. A caller may also prepare one native
/// projection tree to seed the same walk. The output fits the ranked query
/// handoff.
pub struct DenseApproximateQueryIndex<'a> {
    data: DenseRows<'a>,
    search: PreparedQuery,
}

pub(super) struct PreparedQuery {
    edges: Vec<Vec<usize>>,
    metric: DistanceMetric,
    training_width: usize,
    edge_floor: f64,
    rng: [i64; 3],
    tree: Option<Tree>,
    prepared_positions: Option<Vec<usize>>,
}

impl<'a> DenseApproximateQueryIndex<'a> {
    /// Build a query index from a flat, row-major training neighbor graph.
    ///
    /// `-1` denotes a missing neighbor. Self links are omitted and repeated
    /// edges coalesce. The training width also caps the number of draws used
    /// to seed each held-out query. If supplied, `squared_distances` must align
    /// with the indices and uses the same units as the search priority. With
    /// `None`, the index computes its minimum edge distance from the features.
    pub fn from_training_graph(
        training: &'a [f32],
        n_rows: usize,
        n_features: usize,
        indices: &[i32],
        training_width: usize,
        seed: u32,
        squared_distances: Option<&[f32]>,
    ) -> Result<Self, ApproximateSearchError> {
        let data = DenseRows::training(training, n_rows, n_features)?;
        let (edges, edge_floor) = build_edges(data, indices, training_width, squared_distances)?;

        let mut search = PreparedQuery {
            edges,
            metric: DistanceMetric::Euclidean,
            training_width,
            edge_floor,
            rng: query_stream(seed, n_rows),
            tree: None,
            prepared_positions: None,
        };
        search.order_edges();
        Ok(Self { data, search })
    }

    /// Replace the walk's links with live training offers.
    ///
    /// The live offer width controls the layout of these rows. It leaves the
    /// entry draw limit from the exported training graph in place. Self links,
    /// vacancies and optional squared distances follow [Self::from_training_graph].
    pub fn with_live_training_offers(
        mut self,
        indices: &[i32],
        offer_width: usize,
        squared_distances: Option<&[f32]>,
    ) -> Result<Self, ApproximateSearchError> {
        (self.search.edges, self.search.edge_floor) =
            build_edges(self.data, indices, offer_width, squared_distances)?;
        self.search.order_edges();
        Ok(self)
    }

    /// Prepare one Euclidean projection tree with a caller-selected seed.
    ///
    /// The native tree is built independently of the supplied training links.
    /// Its pivots use the Rust projection builder's random stream. Queries
    /// offer the selected leaf's rows to the result heap and frontier.
    pub fn with_projection_tree(mut self, seed: u32, leaf_size: NonZeroUsize) -> Self {
        let mut rng = Mt19937::new(seed);
        self.search.tree = Some(Tree::build(
            self.data,
            DistanceMetric::Euclidean,
            (0..self.data.n_rows).collect(),
            leaf_size.get(),
            &mut rng,
        ));
        if self.search.prepared_positions.take().is_some() {
            self.search.order_edges();
        }
        self
    }

    /// Prepare one Euclidean tree from pairs of unsigned projection words.
    ///
    /// The root uses slot zero. The children of slot `i` use slots
    /// `2 * i + 1` and `2 * i + 2`. A leaf at capacity or the depth limit
    /// leaves its slot unused and preserves the sibling's position.
    ///
    /// A degenerate split becomes a leaf. Supply the words for the desired
    /// tree, since training and prepared-query pivots can differ. The split
    /// direction sets the prepared graph order: visit left leaves first,
    /// with original row order within each leaf.
    ///
    /// Each query batch restarts its tie stream. When it reaches a split
    /// with a float32 projection margin of magnitude at most 1e-8, it
    /// uses the low bit of the next draw to choose the branch.
    pub fn with_projection_words(
        mut self,
        words: &[[u64; 2]],
        leaf_size: NonZeroUsize,
        max_depth: u8,
    ) -> Result<Self, ApproximateSearchError> {
        let tree = build_word_tree(
            self.data,
            (0..self.data.n_rows).collect(),
            leaf_size.get(),
            max_depth,
            words,
            0,
        )?;
        self.search.set_prepared_tree(tree, self.data.n_rows);
        Ok(self)
    }

    /// Search the supplied graph using a squared-distance priority queue.
    ///
    /// The entry stream restarts at the beginning of each call and advances
    /// across its rows. Repeated entry points consume attempts. Missing ranks
    /// are returned as `(-1, +inf)` so the result can feed
    /// `transform_ranked_neighbors_graph`.
    ///
    /// A frontier point must beat the current cutoff strictly. Epsilon
    /// expands the worst retained squared distance by a fraction of its gap
    /// from the smallest squared non-self training edge. The cutoff rounds
    /// to the same precision as the squared priorities.
    pub fn query(
        &self,
        queries: &[f32],
        n_query_rows: usize,
        width: usize,
        epsilon: f64,
    ) -> Result<RankedNeighbors, ApproximateSearchError> {
        let queries = DenseRows::new(queries, n_query_rows, self.data.n_features)?;
        self.search.query(self.data, queries, width, epsilon)
    }
}

impl PreparedQuery {
    pub(super) fn from_refinement(
        data: DenseRows<'_>,
        ranks: &[Vec<TrainingNeighbor>],
        training_width: usize,
        rng: [i64; 3],
        metric: DistanceMetric,
    ) -> Self {
        let distance = |left, right| query_distance(metric, data.row(left), data.row(right));
        let mut in_degree = vec![0_usize; data.n_rows];
        let mut edges = vec![Vec::new(); data.n_rows];
        let mut edge_floor = f32::INFINITY;
        let mut retained = Vec::with_capacity(training_width);
        for (row, candidates) in ranks.iter().enumerate() {
            retained.clear();
            for candidate in candidates {
                let other = candidate.index();
                if other == row {
                    continue;
                }
                in_degree[other] += 1;
                let edge = distance(row, other);
                if retained.iter().any(|&prior| distance(prior, other) < edge) {
                    continue;
                }
                edge_floor = edge_floor.min(edge);
                retained.push(other);
                edges[row].push(other);
                edges[other].push(row);
            }
        }
        let degree_limit = training_width
            .saturating_add(1)
            .saturating_mul(3)
            .div_ceil(2);
        for (row, neighbors) in edges.iter_mut().enumerate() {
            neighbors.sort_unstable();
            neighbors.dedup();
            if neighbors.len() > degree_limit {
                neighbors.select_nth_unstable_by_key(degree_limit, |&other| {
                    (distance(row, other).to_bits(), other)
                });
                neighbors.truncate(degree_limit);
            }
        }
        let mut search = Self {
            edges,
            metric,
            training_width,
            edge_floor: if edge_floor.is_finite() {
                f64::from(edge_floor)
            } else {
                0.0
            },
            rng: warm_query_stream(rng, data.n_rows),
            tree: None,
            prepared_positions: None,
        };
        let tree = build_search_tree(data, &in_degree, (0..data.n_rows).collect(), metric);
        search.set_prepared_tree(tree, data.n_rows);
        search
    }

    fn order_edges(&mut self) {
        let positions = self.prepared_positions.as_deref();
        for row in &mut self.edges {
            row.sort_unstable_by_key(|&index| positions.map_or(index, |map| map[index]));
            row.dedup();
        }
    }

    fn set_prepared_tree(&mut self, mut tree: Tree, n_rows: usize) {
        let mut positions = vec![0; n_rows];
        let mut next = 0;
        pack_leaves(&mut tree, &mut positions, &mut next);
        self.tree = Some(tree);
        self.prepared_positions = Some(positions);
        self.order_edges();
    }

    pub(super) fn query(
        &self,
        data: DenseRows<'_>,
        queries: DenseRows<'_>,
        width: usize,
        epsilon: f64,
    ) -> Result<RankedNeighbors, ApproximateSearchError> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(ApproximateSearchError::InvalidEpsilon);
        }
        if width == 0 {
            return Err(ApproximateSearchError::RankShape);
        }
        // A prepared search treats epsilon values that round to zero at
        // priority precision as zero, including in the underfilled walk.
        let epsilon = if self.tree.is_some() && epsilon as f32 == 0.0 {
            0.0
        } else {
            epsilon
        };

        let mut ranks = RankedNeighbors::with_capacity(queries.n_rows, width)?;
        if queries.n_rows == 0 {
            return Ok(ranks);
        }
        let mut rng = self.rng;
        let mut seen = vec![usize::MAX; data.n_rows];
        let missing = Neighbor {
            index: usize::MAX,
            distance: f32::INFINITY,
        };
        let mut result_heap = vec![missing; width];
        let mut frontier = BinaryHeap::new();
        let mut vacant_candidates = Vec::new();
        let mut unseen_edges = Vec::new();
        let groups_of_four = data.batches_cosine();
        let positions = self.prepared_positions.as_deref();
        let priority = |point: Neighbor| {
            let position = positions.map_or(point.index, |map| map[point.index]);
            Reverse((point.distance.to_bits(), position, point.index))
        };
        let search_bound = |distance: f32| {
            if distance.is_infinite() {
                distance
            } else {
                let distance = f64::from(distance);
                (distance + epsilon * (distance - self.edge_floor)) as f32
            }
        };
        let margin = margin_for(self.metric);
        let normalized;
        let query_data = if self.metric == DistanceMetric::Cosine {
            normalized = cosine::normalized_queries(queries.data, queries.n_features);
            normalized.as_slice()
        } else {
            queries.data
        };
        let mut last_packed_row = None;
        for (row, query) in query_data.chunks_exact(queries.n_features).enumerate() {
            if self.metric == DistanceMetric::Cosine
                && query.iter().all(|&value| value == 0.0)
                && cosine::squared_query_norm(queries.row(row)) == 0.0
            {
                // The empty heap's -1 index addresses the final packed training row.
                let last = *last_packed_row.get_or_insert_with(|| {
                    positions
                        .and_then(|map| map.iter().position(|&p| p == data.n_rows - 1))
                        .unwrap_or(data.n_rows - 1) as i32
                });
                ranks.indices.extend(std::iter::repeat_n(last, width));
                ranks.distances.extend(std::iter::repeat_n(1.0, width));
                continue;
            }
            result_heap.fill(missing);
            frontier.clear();
            let visit = |seen: &mut [usize], index: usize, batched: bool| {
                if seen[index] == row {
                    return Ok(None);
                }
                seen[index] = row;
                let distance = if batched {
                    cosine::batched_priority(query, data.row(index))
                } else {
                    query_distance(self.metric, query, data.row(index))
                };
                if !distance.is_finite() {
                    return Err(ApproximateSearchError::NonFiniteDistance {
                        row,
                        candidate: index,
                    });
                }
                Ok(Some(Neighbor { index, distance }))
            };
            if let Some(tree) = &self.tree {
                let leaf = if positions.is_some() {
                    prepared_leaf(tree, data, query, &mut rng, margin)
                } else {
                    tree.leaf(data, DistanceMetric::Euclidean, query)
                };
                for &index in leaf {
                    if let Some(point) = visit(&mut seen, index, false)? {
                        if point.distance < result_heap[0].distance {
                            sift_down(&mut result_heap, point);
                        }
                        if leaf.len() != data.n_rows {
                            frontier.push(priority(point));
                        }
                    }
                }
            } else {
                for _ in 0..width.min(self.training_width) {
                    let index = sample_absolute_vertex(&mut rng, data.n_rows);
                    if let Some(point) = visit(&mut seen, index, false)? {
                        sift_down(&mut result_heap, point);
                        frontier.push(priority(point));
                    }
                }
            }
            let mut bound = search_bound(result_heap[0].distance);
            while let Some(Reverse((distance, _, index))) = frontier.pop() {
                if f32::from_bits(distance) >= bound {
                    break;
                }
                let edges = &self.edges[index];
                let (candidates, batched_len) = if groups_of_four {
                    unseen_edges.clear();
                    unseen_edges.extend(
                        edges
                            .iter()
                            .copied()
                            .filter(|&candidate| seen[candidate] != row),
                    );
                    (unseen_edges.as_slice(), unseen_edges.len() / 4 * 4)
                } else {
                    (edges.as_slice(), 0)
                };
                if epsilon == 0.0 && bound.is_infinite() {
                    vacant_candidates.clear();
                    for (i, &candidate) in candidates.iter().enumerate() {
                        if let Some(point) = visit(&mut seen, candidate, i < batched_len)? {
                            vacant_candidates.push(point);
                        }
                    }
                    // The zero-epsilon, underfilled walk offers the first two
                    // candidates of each complete group of four, and every
                    // tail candidate. Once full, normal offers resume.
                    let grouped = vacant_candidates.len() / 4 * 4;
                    for (position, &point) in vacant_candidates.iter().enumerate() {
                        if bound.is_infinite() && position < grouped && position % 4 >= 2 {
                            continue;
                        }
                        if point.distance < bound {
                            sift_down(&mut result_heap, point);
                            bound = result_heap[0].distance;
                            frontier.push(priority(point));
                        }
                    }
                    // Prepared entry ends this pass even if its nearest row
                    // had no unseen links. Random entry can try another row.
                    if bound.is_infinite() && (positions.is_some() || !vacant_candidates.is_empty())
                    {
                        break;
                    }
                    continue;
                }
                for (i, &candidate) in candidates.iter().enumerate() {
                    if let Some(point) = visit(&mut seen, candidate, i < batched_len)?
                        && point.distance < bound
                    {
                        if point.distance < result_heap[0].distance {
                            sift_down(&mut result_heap, point);
                            bound = search_bound(result_heap[0].distance);
                        }
                        frontier.push(priority(point));
                    }
                }
            }
            finish_query_row(&mut ranks, &mut result_heap, self.metric);
        }
        Ok(ranks)
    }
}

fn query_stream(seed: u32, n_rows: usize) -> [i64; 3] {
    let mut caller = Mt19937::new(seed);
    draw_negative_seed(|| caller.next_u32());
    warm_query_stream(draw_negative_seed(|| caller.next_u32()), n_rows)
}

fn warm_query_stream(mut rng: [i64; 3], n_rows: usize) -> [i64; 3] {
    for _ in 0..10 {
        sample_absolute_vertex(&mut rng, n_rows);
    }
    rng
}

fn query_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Euclidean => approximate_squared_euclidean(a, b),
        DistanceMetric::Cosine => cosine::priority(a, b),
        _ => metric.distance(a, b),
    }
}

fn finish_query_row(ranks: &mut RankedNeighbors, heap: &mut [Neighbor], metric: DistanceMetric) {
    for end in (1..heap.len()).rev() {
        let replacement = heap[end];
        heap[end] = heap[0];
        sift_down(&mut heap[..end], replacement);
    }
    for neighbor in heap {
        ranks.indices.push(if neighbor.index == usize::MAX {
            -1
        } else {
            neighbor.index as i32
        });
        ranks.distances.push(match metric {
            DistanceMetric::Euclidean => neighbor.distance.sqrt(),
            DistanceMetric::Cosine => cosine::distance(neighbor.distance),
            _ => neighbor.distance,
        });
    }
}

fn build_edges(
    data: DenseRows<'_>,
    indices: &[i32],
    width: usize,
    squared_distances: Option<&[f32]>,
) -> Result<(Vec<Vec<usize>>, f64), ApproximateSearchError> {
    let n_rows = data.n_rows;
    if width == 0
        || n_rows.checked_mul(width) != Some(indices.len())
        || squared_distances.is_some_and(|distances| distances.len() != indices.len())
    {
        return Err(ApproximateSearchError::RankShape);
    }
    let mut edges = vec![Vec::new(); n_rows];
    let mut edge_floor = f32::INFINITY;
    for (row, neighbors) in indices.chunks_exact(width).enumerate() {
        for (offset, &index) in neighbors.iter().enumerate() {
            if index == -1 || index == row as i32 {
                continue;
            }
            let position = row * width + offset;
            if index < 0 || index as usize >= n_rows {
                return Err(ApproximateSearchError::InvalidSeed { position, index });
            }
            let column = index as usize;
            let distance = squared_distances.map_or_else(
                || approximate_squared_euclidean(data.row(row), data.row(column)),
                |distances| distances[position],
            );
            if !distance.is_finite() || distance < 0.0 {
                return Err(ApproximateSearchError::InvalidEdgeDistance { position });
            }
            edge_floor = edge_floor.min(distance);
            edges[row].push(column);
            edges[column].push(row);
        }
    }
    Ok((
        edges,
        if edge_floor.is_finite() {
            f64::from(edge_floor)
        } else {
            0.0
        },
    ))
}

fn pack_leaves(tree: &mut Tree, positions: &mut [usize], next: &mut usize) {
    match tree {
        Tree::Leaf(rows) => {
            rows.sort_unstable();
            for &row in rows.iter() {
                positions[row] = *next;
                *next += 1;
            }
        }
        Tree::Split { left, right, .. } | Tree::Balanced { left, right } => {
            pack_leaves(left, positions, next);
            pack_leaves(right, positions, next);
        }
    }
}

type ProjectionMargin = fn(&[f32], &[f32], &[f32]) -> f32;

fn margin_for(metric: DistanceMetric) -> ProjectionMargin {
    if metric.is_angular() {
        angular_margin
    } else {
        prepared_margin
    }
}

fn build_search_tree(
    data: DenseRows<'_>,
    in_degree: &[usize],
    mut rows: Vec<usize>,
    metric: DistanceMetric,
) -> Tree {
    const QUERY_LEAF_SIZE: usize = 30;
    if rows.len() <= QUERY_LEAF_SIZE {
        return Tree::Leaf(rows);
    }

    // The three hubs are ordered by global incoming degree, then row ID.
    let key = |row: usize| (Reverse(in_degree[row]), row);
    let mut hubs = Vec::with_capacity(4);
    for &row in &rows {
        let slot = hubs.partition_point(|&prior| key(prior) < key(row));
        if slot < 3 {
            hubs.insert(slot, row);
            hubs.truncate(3);
        }
    }
    let ((a, b), plane) = [(hubs[0], hubs[1]), (hubs[0], hubs[2]), (hubs[1], hubs[2])]
        .into_iter()
        .map(|(a, b)| {
            (
                (a, b),
                PreparedProjection::new(data.row(a), data.row(b), metric.is_angular()),
            )
        })
        .min_by_key(|(_, plane)| {
            let left = rows
                .iter()
                .filter(|&&row| plane.margin(data.row(row)) > 0.0)
                .count();
            left.abs_diff(rows.len() - left)
        })
        .expect("three hub pairs");
    let margin = |row| plane.margin(data.row(row));
    let mut boundary = partition_rows(&mut rows, |row| margin(row) > 0.0);
    let mut threshold = 0.0;
    if boundary.min(rows.len() - boundary) < rows.len().div_ceil(8) {
        rows.sort_unstable_by(|&left, &right| {
            margin(right)
                .total_cmp(&margin(left))
                .then(left.cmp(&right))
        });
        boundary = rows.len() / 2;
        threshold = margin(rows[boundary]);
    }
    let right = rows.split_off(boundary);
    Tree::Split {
        a,
        b,
        threshold,
        left: Box::new(build_search_tree(data, in_degree, rows, metric)),
        right: Box::new(build_search_tree(data, in_degree, right, metric)),
    }
}

fn prepared_leaf<'a>(
    tree: &'a Tree,
    data: DenseRows<'_>,
    query: &[f32],
    rng: &mut [i64; 3],
    projection: ProjectionMargin,
) -> &'a [usize] {
    let Tree::Split {
        a,
        b,
        threshold,
        left,
        right,
    } = tree
    else {
        return tree.leaf(data, DistanceMetric::Euclidean, query);
    };
    let margin = projection(data.row(*a), data.row(*b), query) - threshold;
    let go_left = if !margin.is_finite() {
        return tree.leaf(data, DistanceMetric::Euclidean, query);
    } else if margin.abs() <= 1.0e-8 {
        sample_absolute_vertex(rng, 2) == 0
    } else {
        margin > 0.0
    };
    prepared_leaf(
        if go_left { left } else { right },
        data,
        query,
        rng,
        projection,
    )
}

fn build_word_tree(
    data: DenseRows<'_>,
    mut rows: Vec<usize>,
    leaf_size: usize,
    depth: u8,
    words: &[[u64; 2]],
    slot: usize,
) -> Result<Tree, ApproximateSearchError> {
    if rows.len() <= leaf_size || depth == 0 {
        return Ok(Tree::Leaf(rows));
    }
    let &pair = words
        .get(slot)
        .ok_or(ApproximateSearchError::ProjectionWordsExhausted)?;
    let (a, b) = projection_pivots(&rows, pair);

    let metric = DistanceMetric::Euclidean;
    let boundary = partition_rows(&mut rows, |row| {
        let point = data.row(row);
        metric.distance(point, data.row(a)) <= metric.distance(point, data.row(b))
    });
    if boundary == 0 || boundary == rows.len() {
        return Ok(Tree::Leaf(rows));
    }
    let right = rows.split_off(boundary);
    Ok(Tree::Split {
        a,
        b,
        threshold: 0.0,
        left: Box::new(build_word_tree(
            data,
            rows,
            leaf_size,
            depth - 1,
            words,
            2 * slot + 1,
        )?),
        right: Box::new(build_word_tree(
            data,
            right,
            leaf_size,
            depth - 1,
            words,
            2 * slot + 2,
        )?),
    })
}

// The farthest retained candidate is at the root. On equal child keys the
// left child is selected, and a replacement stops above an equal key.
fn sift_down(heap: &mut [Neighbor], replacement: Neighbor) {
    let mut parent = 0;
    while parent < heap.len() / 2 {
        let left = 2 * parent + 1;
        let right = left + 1;
        let child = if right < heap.len() && heap[left].distance < heap[right].distance {
            right
        } else {
            left
        };
        if replacement.distance >= heap[child].distance {
            break;
        }
        heap[parent] = heap[child];
        parent = child;
    }
    heap[parent] = replacement;
}

#[cfg(test)]
#[path = "projection_words_tests.rs"]
mod projection_words_tests;

#[cfg(test)]
#[path = "projection_ties_tests.rs"]
mod projection_ties_tests;
