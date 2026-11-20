use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;

use super::projection::{partition_rows, prepared_margin, projection_pivots};
use super::{
    ApproximateSearchError, DenseRows, RankedNeighbors, Tree, approximate_squared_euclidean,
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
    edges: Vec<Vec<usize>>,
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

        // With no search trees the entry stream comes from the caller's
        // second signed three-word draw, advanced ten times before the batch.
        let mut seed = Mt19937::new(seed);
        draw_negative_seed(|| seed.next_u32());
        let mut rng = draw_negative_seed(|| seed.next_u32());
        for _ in 0..10 {
            sample_absolute_vertex(&mut rng, n_rows);
        }
        let mut index = Self {
            data,
            edges,
            training_width,
            edge_floor,
            rng,
            tree: None,
            prepared_positions: None,
        };
        index.order_edges();
        Ok(index)
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
        (self.edges, self.edge_floor) =
            build_edges(self.data, indices, offer_width, squared_distances)?;
        self.order_edges();
        Ok(self)
    }

    /// Prepare one Euclidean projection tree with a caller-selected seed.
    ///
    /// The native tree is built independently of the supplied training links.
    /// Its pivots use the Rust projection builder's random stream. Queries
    /// offer the selected leaf's rows to the result heap and frontier.
    pub fn with_projection_tree(mut self, seed: u32, leaf_size: NonZeroUsize) -> Self {
        let mut rng = Mt19937::new(seed);
        self.tree = Some(Tree::build(
            self.data,
            DistanceMetric::Euclidean,
            (0..self.data.n_rows).collect(),
            leaf_size.get(),
            &mut rng,
        ));
        if self.prepared_positions.take().is_some() {
            self.order_edges();
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
        let mut tree = build_word_tree(
            self.data,
            (0..self.data.n_rows).collect(),
            leaf_size.get(),
            max_depth,
            words,
            0,
        )?;
        let mut positions = vec![0; self.data.n_rows];
        let mut next_position = 0;
        pack_leaves(&mut tree, &mut positions, &mut next_position);
        self.tree = Some(tree);
        self.prepared_positions = Some(positions);
        self.order_edges();
        Ok(self)
    }

    fn order_edges(&mut self) {
        let positions = self.prepared_positions.as_deref();
        for row in &mut self.edges {
            row.sort_unstable_by_key(|&index| positions.map_or(index, |map| map[index]));
            row.dedup();
        }
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

        let mut ranks = RankedNeighbors::with_capacity(n_query_rows, width)?;
        if n_query_rows == 0 {
            return Ok(ranks);
        }
        let mut rng = self.rng;
        let mut seen = vec![usize::MAX; self.data.n_rows];
        let missing = Neighbor {
            index: usize::MAX,
            distance: f32::INFINITY,
        };
        let mut result_heap = vec![missing; width];
        let mut frontier = BinaryHeap::new();
        let mut vacant_candidates = Vec::new();
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
        for (row, query) in queries.data.chunks_exact(queries.n_features).enumerate() {
            result_heap.fill(missing);
            frontier.clear();
            let mut visit = |index: usize| {
                if seen[index] == row {
                    return Ok(None);
                }
                seen[index] = row;
                let distance = approximate_squared_euclidean(query, self.data.row(index));
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
                    prepared_leaf(tree, self.data, query, &mut rng)
                } else {
                    tree.leaf(self.data, DistanceMetric::Euclidean, query)
                };
                for &index in leaf {
                    if let Some(point) = visit(index)? {
                        if point.distance < result_heap[0].distance {
                            sift_down(&mut result_heap, point);
                        }
                        if leaf.len() != self.data.n_rows {
                            frontier.push(priority(point));
                        }
                    }
                }
            } else {
                for _ in 0..width.min(self.training_width) {
                    let index = sample_absolute_vertex(&mut rng, self.data.n_rows);
                    if let Some(point) = visit(index)? {
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
                if epsilon == 0.0 && bound.is_infinite() {
                    vacant_candidates.clear();
                    for &candidate in &self.edges[index] {
                        if let Some(point) = visit(candidate)? {
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
                for &candidate in &self.edges[index] {
                    if let Some(point) = visit(candidate)?
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
            for end in (1..width).rev() {
                let replacement = result_heap[end];
                result_heap[end] = result_heap[0];
                sift_down(&mut result_heap[..end], replacement);
            }
            for neighbor in &result_heap {
                ranks.indices.push(if neighbor.index == usize::MAX {
                    -1
                } else {
                    neighbor.index as i32
                });
                ranks.distances.push(neighbor.distance.sqrt());
            }
        }
        Ok(ranks)
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

fn prepared_leaf<'a>(
    tree: &'a Tree,
    data: DenseRows<'_>,
    query: &[f32],
    rng: &mut [i64; 3],
) -> &'a [usize] {
    let Tree::Split { a, b, left, right } = tree else {
        return tree.leaf(data, DistanceMetric::Euclidean, query);
    };
    let margin = prepared_margin(data.row(*a), data.row(*b), query);
    let go_left = if !margin.is_finite() {
        return tree.leaf(data, DistanceMetric::Euclidean, query);
    } else if margin.abs() <= 1.0e-8 {
        sample_absolute_vertex(rng, 2) == 0
    } else {
        margin > 0.0
    };
    prepared_leaf(if go_left { left } else { right }, data, query, rng)
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
