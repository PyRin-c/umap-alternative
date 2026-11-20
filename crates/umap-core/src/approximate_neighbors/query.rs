use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{ApproximateSearchError, DenseRows, RankedNeighbors, approximate_squared_euclidean};
use crate::neighbors::Neighbor;
use crate::rng::{Mt19937, draw_negative_seed, sample_absolute_vertex};

/// Tree-free approximate Euclidean queries over a supplied training graph.
///
/// The index uses an unpruned symmetric union of the supplied directed links.
/// It isolates random entry points and the graph walk from projection-tree
/// construction. The caller may pass the output to the ranked query handoff.
pub struct DenseApproximateQueryIndex<'a> {
    data: DenseRows<'a>,
    edges: Vec<Vec<usize>>,
    training_width: usize,
    edge_floor: f64,
    rng: [i64; 3],
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
        if training_width == 0
            || n_rows.checked_mul(training_width) != Some(indices.len())
            || squared_distances.is_some_and(|distances| distances.len() != indices.len())
        {
            return Err(ApproximateSearchError::RankShape);
        }
        let mut edges = vec![Vec::new(); n_rows];
        let mut edge_floor = f32::INFINITY;
        for (row, neighbors) in indices.chunks_exact(training_width).enumerate() {
            for (offset, &index) in neighbors.iter().enumerate() {
                if index == -1 || index == row as i32 {
                    continue;
                }
                let position = row * training_width + offset;
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
        for row in &mut edges {
            row.sort_unstable();
            row.dedup();
        }

        // With no search trees the entry stream comes from the caller's
        // second signed three-word draw, advanced ten times before the batch.
        let mut seed = Mt19937::new(seed);
        draw_negative_seed(|| seed.next_u32());
        let mut rng = draw_negative_seed(|| seed.next_u32());
        for _ in 0..10 {
            sample_absolute_vertex(&mut rng, n_rows);
        }
        Ok(Self {
            data,
            edges,
            training_width,
            edge_floor: if edge_floor.is_finite() {
                f64::from(edge_floor)
            } else {
                0.0
            },
            rng,
        })
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
    /// from the smallest squared non-self training edge.
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
        let search_bound = |distance: f32| {
            let distance = f64::from(distance);
            // Preserve the unbounded search radius when no full result exists,
            // including when epsilon is zero.
            if distance.is_infinite() {
                distance
            } else {
                distance + epsilon * (distance - self.edge_floor)
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
            for _ in 0..width.min(self.training_width) {
                let index = sample_absolute_vertex(&mut rng, self.data.n_rows);
                if let Some(point) = visit(index)? {
                    sift_down(&mut result_heap, point);
                    frontier.push(Reverse((point.distance.to_bits(), point.index)));
                }
            }
            let mut bound = search_bound(result_heap[0].distance);
            while let Some(Reverse((distance, index))) = frontier.pop() {
                if f64::from(f32::from_bits(distance)) >= bound {
                    break;
                }
                for &candidate in &self.edges[index] {
                    if let Some(point) = visit(candidate)?
                        && f64::from(point.distance) < bound
                    {
                        if point.distance < result_heap[0].distance {
                            sift_down(&mut result_heap, point);
                            bound = search_bound(result_heap[0].distance);
                        }
                        frontier.push(Reverse((point.distance.to_bits(), point.index)));
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
