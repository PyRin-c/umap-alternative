//! Seeded dense approximate ranks for the training and held-out graph handoffs.
//!
//! The supplied-seed and tree-free query entry points allow refinement and
//! held-out queries to be measured independently of tree construction.
//! Exhaustive search remains available through `neighbors`.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use crate::neighbors::{DistanceMetric, KnnError, Neighbor};
use crate::rng::{Mt19937, draw_negative_seed, sample_absolute_vertex};

mod cosine;
mod cosine_refine;
mod projection;
mod query;
pub use projection::training_projection_leaves;
pub use query::DenseApproximateQueryIndex;

const SMALL_DATA_LIMIT: usize = 4096;
const MAX_CANDIDATES: usize = 60;
const MIN_LEAF_SIZE: usize = 60;

/// Draw the caller's projection scalar after consuming the two signed states.
///
/// `caller` must be at the start of approximate-index initialization. The
/// signed draws can reject raw words, so the scalar has no fixed byte offset
/// in the caller stream. The resulting value is in `0..2^62`.
pub fn draw_projection_scalar(caller: &mut Mt19937) -> u64 {
    ApproximateStreams::draw(caller).projection
}

struct ApproximateStreams {
    fill: [i64; 3],
    query: [i64; 3],
    projection: u64,
}

impl ApproximateStreams {
    fn draw(caller: &mut Mt19937) -> Self {
        let fill = draw_negative_seed(|| caller.next_u32());
        let query = draw_negative_seed(|| caller.next_u32());
        let projection =
            (u64::from(caller.next_u32() & 0x3fff_ffff) << 32) | u64::from(caller.next_u32());
        Self {
            fill,
            query,
            projection,
        }
    }
}

/// Observed dense feature routing after any deduplication.
///
/// The caller handles singletons and any fallback from supplied neighbors before
/// using this predicate. The exact and approximate search APIs remain separate.
pub fn use_approximate_search(n_distinct_rows: usize, force: bool) -> bool {
    n_distinct_rows > 1 && (force || n_distinct_rows >= SMALL_DATA_LIMIT)
}

/// Flat distance-ranked rows for the graph handoffs.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedNeighbors {
    /// Number of slots per row, including missing ranks.
    pub width: usize,
    /// Training indices in rank order. Missing ranks contain `-1`.
    pub indices: Vec<i32>,
    /// Distances in rank order. Missing ranks contain positive infinity.
    pub distances: Vec<f32>,
}

/// Errors from dense approximate search or refinement.
#[derive(Clone, Debug, PartialEq)]
pub enum ApproximateSearchError {
    /// Matrix shape or value validation failed.
    Matrix(KnnError),
    /// The index requires at least two training rows and one feature, with
    /// row indices representable as `i32`.
    Dimensions,
    /// Rank width is invalid, storage would overflow, or the seeds have the wrong size.
    RankShape,
    /// Search epsilon must be finite and nonnegative.
    InvalidEpsilon,
    /// A supplied seed is neither a valid training index nor the missing marker.
    InvalidSeed { position: usize, index: i32 },
    /// A supplied squared distance for a non-self graph edge is invalid.
    InvalidEdgeDistance { position: usize },
    /// A requested projection split has no pivot words at its slot.
    ProjectionWordsExhausted,
    /// The metric returned a non-finite distance for a ranked candidate.
    NonFiniteDistance { row: usize, candidate: usize },
}

impl fmt::Display for ApproximateSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(error) => error.fmt(formatter),
            Self::Dimensions => write!(formatter, "invalid approximate index dimensions"),
            Self::RankShape => write!(formatter, "invalid approximate rank buffer shape"),
            Self::InvalidEpsilon => {
                write!(formatter, "search epsilon must be finite and nonnegative")
            }
            Self::InvalidSeed { position, index } => {
                write!(
                    formatter,
                    "invalid seed index {index} at position {position}"
                )
            }
            Self::InvalidEdgeDistance { position } => {
                write!(
                    formatter,
                    "invalid squared edge distance at position {position}"
                )
            }
            Self::ProjectionWordsExhausted => write!(formatter, "projection pivot words exhausted"),
            Self::NonFiniteDistance { row, candidate } => {
                write!(
                    formatter,
                    "non-finite distance from row {row} to candidate {candidate}"
                )
            }
        }
    }
}

impl std::error::Error for ApproximateSearchError {}

/// A dense index with a bounded candidate build and graph-guided queries.
///
/// Fitting ranks are built from co-membership in seeded projection-tree leaves,
/// random fill for undersized rows, and forward/reverse neighbor refinement.
/// Queries descend a retained tree and walk the refined graph. The caller
/// selects exhaustive search explicitly.
pub struct DenseApproximateIndex<'a> {
    data: Cow<'a, [f32]>,
    n_rows: usize,
    n_features: usize,
    metric: DistanceMetric,
    seed: u32,
    fit_ranks: RankedNeighbors,
    search: QuerySearch,
}

enum QuerySearch {
    Prepared(query::PreparedQuery),
    Generic { edges: Vec<Vec<usize>>, tree: Tree },
}

impl<'a> DenseApproximateIndex<'a> {
    /// Build using a caller-selected seed for the projection pivots and random fill.
    ///
    /// The output has up to `min(width, n_rows - 1)` finite ranks per row,
    /// including self. Before that output is formed, refinement can use one
    /// more non-self candidate. The ranked buffers use the requested width.
    pub fn build(
        data: &'a [f32],
        n_rows: usize,
        n_features: usize,
        width: usize,
        metric: DistanceMetric,
        seed: u32,
    ) -> Result<Self, ApproximateSearchError> {
        Self::build_index(
            data,
            n_rows,
            n_features,
            width,
            metric,
            seed,
            &mut Mt19937::new(seed),
        )
    }

    /// Build from the caller's current MT19937 state and advance it past the
    /// training, query, and projection seeds for subsequent layout initialization.
    pub fn build_with_rng(
        data: &'a [f32],
        n_rows: usize,
        n_features: usize,
        width: usize,
        metric: DistanceMetric,
        caller: &mut Mt19937,
    ) -> Result<Self, ApproximateSearchError> {
        let fallback_seed = if metric == DistanceMetric::Euclidean || metric.is_angular() {
            0
        } else {
            caller.clone().next_u32()
        };
        Self::build_index(
            data,
            n_rows,
            n_features,
            width,
            metric,
            fallback_seed,
            caller,
        )
    }

    fn build_index(
        data: &'a [f32],
        n_rows: usize,
        n_features: usize,
        width: usize,
        metric: DistanceMetric,
        seed: u32,
        caller: &mut Mt19937,
    ) -> Result<Self, ApproximateSearchError> {
        let input = DenseRows::training(data, n_rows, n_features)?;
        if width < 2 {
            return Err(ApproximateSearchError::RankShape);
        }
        let features = if metric == DistanceMetric::Cosine {
            Cow::Owned(cosine::normalized_training(input.data, n_features))
        } else {
            Cow::Borrowed(input.data)
        };
        let data = DenseRows {
            data: features.as_ref(),
            normalized_cosine: metric == DistanceMetric::Cosine,
            ..input
        };
        let mut fit_ranks = RankedNeighbors::with_capacity(n_rows, width)?;
        let k = width.min(n_rows - 1);
        let rounds = (((n_rows as f64).log2()).round() as usize).max(5);
        let mut streams = ApproximateStreams::draw(caller);
        let (tree, mut ranks) = seed_forest(data, metric, k, seed, &mut streams)?;
        let search = if let Some(tree) = tree {
            refine_supplied(data, metric, &mut ranks, k, rounds)?;
            QuerySearch::Generic {
                edges: incident(&ranks),
                tree,
            }
        } else {
            if data.batches_cosine() {
                cosine_refine::refine(
                    data,
                    &mut ranks,
                    &mut streams.fill,
                    rounds,
                    n_rows * k / 1_000,
                );
            } else {
                refine(data, metric, &mut ranks, k, rounds, n_rows * k / 1_000)?;
            }
            QuerySearch::Prepared(query::PreparedQuery::from_refinement(
                data,
                &ranks,
                k,
                streams.query,
                metric,
            ))
        };
        for (row, rank) in ranks.iter().enumerate() {
            fit_ranks.extend_training(row, rank, k);
        }
        if metric == DistanceMetric::Cosine {
            for distance in &mut fit_ranks.distances {
                *distance = cosine::distance(*distance);
            }
        }
        Ok(Self {
            data: features,
            n_rows,
            n_features,
            metric,
            seed,
            fit_ranks,
            search,
        })
    }

    /// Requested-width arrays for `fit_supplied_neighbors_graph`.
    pub fn training_ranks(&self) -> &RankedNeighbors {
        &self.fit_ranks
    }

    /// Effective fitted count for `transform_ranked_neighbors_graph`.
    pub fn query_scale_neighbors(&self) -> usize {
        self.fit_ranks.width.min(self.n_rows - 1)
    }

    /// Retain the feature matrix and fitted search data independently of the caller's buffer.
    pub fn into_owned(self) -> DenseApproximateIndex<'static> {
        DenseApproximateIndex {
            data: Cow::Owned(self.data.into_owned()),
            n_rows: self.n_rows,
            n_features: self.n_features,
            metric: self.metric,
            seed: self.seed,
            fit_ranks: self.fit_ranks,
            search: self.search,
        }
    }

    /// Search each query independently, returning the requested width.
    ///
    /// Epsilon is 0.24 for the angular metrics and 0.12 otherwise.
    /// Missing slots have the graph handoff's sentinel pair.
    pub fn query(
        &self,
        queries: &[f32],
        n_query_rows: usize,
    ) -> Result<RankedNeighbors, ApproximateSearchError> {
        self.query_with_width(queries, n_query_rows, self.fit_ranks.width)
    }

    /// Search using the requested result width for entry and cutoff decisions.
    pub fn query_with_width(
        &self,
        queries: &[f32],
        n_query_rows: usize,
        width: usize,
    ) -> Result<RankedNeighbors, ApproximateSearchError> {
        if !(1..=self.fit_ranks.width).contains(&width) {
            return Err(ApproximateSearchError::RankShape);
        }
        let data = DenseRows {
            data: self.data.as_ref(),
            n_rows: self.n_rows,
            n_features: self.n_features,
            normalized_cosine: self.metric == DistanceMetric::Cosine,
        };
        let queries = DenseRows::new(queries, n_query_rows, data.n_features)?;
        let (edges, tree) = match &self.search {
            QuerySearch::Prepared(search) => {
                let epsilon = if self.metric.is_angular() { 0.24 } else { 0.12 };
                return search.query(data, queries, width, epsilon);
            }
            QuerySearch::Generic { edges, tree } => (edges, tree),
        };
        let mut result = RankedNeighbors::with_capacity(n_query_rows, width)?;
        let k = width.min(data.n_rows);
        let factor = if self.metric.is_angular() { 1.24 } else { 1.12 };
        let mut visited = vec![usize::MAX; data.n_rows];
        for (row, query) in queries.data.chunks_exact(queries.n_features).enumerate() {
            let mut best = Vec::with_capacity(k);
            let mut frontier = BinaryHeap::new();
            let mut visit =
                |index: usize, best: &mut Vec<Neighbor>, frontier: &mut BinaryHeap<_>| {
                    if visited[index] != row {
                        visited[index] = row;
                        let distance =
                            checked_distance(self.metric, query, data.row(index), row, index)?;
                        if distance <= bound(best, k, factor) {
                            frontier.push(Reverse((distance.to_bits(), index)));
                            offer(best, Neighbor { index, distance }, k, |point| *point);
                        }
                    }
                    Ok(())
                };
            for &index in tree.leaf(data, self.metric, query) {
                visit(index, &mut best, &mut frontier)?;
            }
            // Small leaves can need additional entry points.
            if best.len() < k {
                let mut rng = Mt19937::new(self.seed);
                for candidate in sample_rows(data.n_rows, &mut rng) {
                    visit(candidate, &mut best, &mut frontier)?;
                    if best.len() == k {
                        break;
                    }
                }
            }
            while let Some(Reverse((distance, index))) = frontier.pop() {
                if f32::from_bits(distance) > bound(&best, k, factor) {
                    break;
                }
                for &candidate in &edges[index] {
                    visit(candidate, &mut best, &mut frontier)?;
                }
            }
            result.extend(&best);
        }
        Ok(result)
    }
}

fn seed_forest(
    data: DenseRows<'_>,
    metric: DistanceMetric,
    k: usize,
    seed: u32,
    streams: &mut ApproximateStreams,
) -> Result<(Option<Tree>, Vec<Vec<TrainingNeighbor>>), ApproximateSearchError> {
    let n_rows = data.n_rows;
    let trees_from_rows = (n_rows as f64).sqrt() / 20.0;
    let mut ranks = if data.batches_cosine() {
        vec![vec![TrainingNeighbor::VACANT; k]; n_rows]
    } else {
        vec![Vec::new(); n_rows]
    };
    let tree = if metric == DistanceMetric::Euclidean || metric.is_angular() {
        let tree_count = 5 + trees_from_rows.round_ties_even() as usize;
        let leaf_size = MIN_LEAF_SIZE.max(k.saturating_mul(5));
        let tree_count = if n_rows <= leaf_size {
            1
        } else {
            tree_count.min(64)
        };
        #[cfg(target_arch = "x86_64")]
        let mut cutoffs = (metric == DistanceMetric::Euclidean
            && data.n_features == 32
            && data.n_rows >= SMALL_DATA_LIMIT
            && std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("fma"))
        .then(|| vec![f32::INFINITY; n_rows]);
        for ordinal in 0..tree_count {
            for leaf in projection::training_leaves(
                data,
                streams.projection,
                ordinal,
                leaf_size,
                64,
                metric.is_angular(),
            ) {
                #[cfg(target_arch = "x86_64")]
                if let Some(cutoffs) = cutoffs.as_mut() {
                    // SAFETY: allocating the cutoffs required both CPU features
                    // and the fixed row width used by the leaf kernel.
                    unsafe { seed_euclidean_leaf_32(data, &leaf, &mut ranks, cutoffs, k)? };
                    continue;
                }
                Tree::Leaf(leaf).seed(data, metric, &mut ranks, k)?;
            }
        }
        None
    } else {
        let tree_count = 5 + trees_from_rows.round() as usize;
        let leaf_size = MIN_LEAF_SIZE.max(k);
        let mut rng = Mt19937::new(seed);
        let mut trees = (0..tree_count.min(64))
            .map(|_| Tree::build(data, metric, (0..n_rows).collect(), leaf_size, &mut rng));
        let tree = trees.next().expect("positive tree count");
        tree.seed(data, metric, &mut ranks, k)?;
        for other in trees {
            other.seed(data, metric, &mut ranks, k)?;
        }
        Some(tree)
    };
    fill_missing(data, metric, &mut ranks, k, &mut streams.fill)?;
    Ok((tree, ranks))
}

/// Isolate refinement from seed generation using supplied row-major indices.
///
/// Duplicate seeds coalesce and `-1` marks an empty rank. A supplied self
/// index occupies a seed slot. Self is added to the output after refinement,
/// so a row supplied without self can use all seed slots to discover edges.
/// A round joins fresh incident neighbors against one another and against the
/// previous round's candidates, including evicted neighbors.
pub fn refine_training_neighbors(
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    width: usize,
    metric: DistanceMetric,
    seed_indices: &[i32],
    rounds: usize,
) -> Result<RankedNeighbors, ApproximateSearchError> {
    let data = DenseRows::training(data, n_rows, n_features)?;
    if !(1..=n_rows).contains(&width) || n_rows.checked_mul(width) != Some(seed_indices.len()) {
        return Err(ApproximateSearchError::RankShape);
    }
    let mut ranks = vec![Vec::new(); n_rows];
    for (row, seeds) in seed_indices.chunks_exact(width).enumerate() {
        for (offset, &index) in seeds.iter().enumerate() {
            if index == -1 {
                continue;
            }
            if index < 0 || index as usize >= n_rows {
                return Err(ApproximateSearchError::InvalidSeed {
                    position: row * width + offset,
                    index,
                });
            }
            let column = index as usize;
            offer_training(
                &mut ranks[row],
                column,
                data.distance(metric, row, column)?,
                width,
            );
        }
    }
    refine_supplied(data, metric, &mut ranks, width, rounds)?;
    let mut result = RankedNeighbors::with_capacity(n_rows, width)?;
    for (row, rank) in ranks.iter().enumerate() {
        result.extend_training(row, rank, width);
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct DenseRows<'a> {
    data: &'a [f32],
    n_rows: usize,
    n_features: usize,
    normalized_cosine: bool,
}

impl<'a> DenseRows<'a> {
    fn training(
        data: &'a [f32],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Self, ApproximateSearchError> {
        if n_rows < 2 || n_rows > i32::MAX as usize || n_features == 0 {
            return Err(ApproximateSearchError::Dimensions);
        }
        Self::new(data, n_rows, n_features)
    }

    fn new(
        data: &'a [f32],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Self, ApproximateSearchError> {
        if n_rows.checked_mul(n_features) != Some(data.len()) {
            return Err(ApproximateSearchError::Matrix(KnnError::InvalidShape {
                n_rows,
                n_features,
                values: data.len(),
            }));
        }
        if let Some((index, &value)) = data
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ApproximateSearchError::Matrix(KnnError::NonFiniteValue {
                index,
                value,
            }));
        }
        Ok(Self {
            data,
            n_rows,
            n_features,
            normalized_cosine: false,
        })
    }

    fn row(self, index: usize) -> &'a [f32] {
        let start = index * self.n_features;
        &self.data[start..start + self.n_features]
    }

    fn batches_cosine(self) -> bool {
        self.normalized_cosine && self.n_features >= 16
    }

    fn distance(
        self,
        metric: DistanceMetric,
        row: usize,
        candidate: usize,
    ) -> Result<f32, ApproximateSearchError> {
        if row == candidate {
            Ok(0.0)
        } else if self.normalized_cosine {
            Ok(cosine::priority(self.row(row), self.row(candidate)))
        } else {
            checked_distance(metric, self.row(row), self.row(candidate), row, candidate)
        }
    }
}

fn checked_distance(
    metric: DistanceMetric,
    left: &[f32],
    right: &[f32],
    row: usize,
    candidate: usize,
) -> Result<f32, ApproximateSearchError> {
    let distance = match metric {
        DistanceMetric::Euclidean => approximate_euclidean(left, right),
        _ => metric.distance(left, right),
    };
    if !distance.is_finite() {
        return Err(ApproximateSearchError::NonFiniteDistance { row, candidate });
    }
    Ok(if distance > 0.0 { distance } else { 0.0 })
}

// The approximate Euclidean wheel path first rounds the feature difference
// to f32. Its square accumulation has 32-coordinate blocks, four-coordinate
// blocks, then a scalar tail. Keep the horizontal reduction order explicit.
fn approximate_euclidean(left: &[f32], right: &[f32]) -> f32 {
    approximate_squared_euclidean(left, right).sqrt()
}

fn approximate_squared_euclidean(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the runtime check covers both enabled target features.
        return unsafe { approximate_squared_euclidean_fma(left, right) };
    }
    approximate_squared_euclidean_portable(left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
fn approximate_squared_euclidean_fma(left: &[f32], right: &[f32]) -> f32 {
    if let (Ok(left), Ok(right)) = (<&[f32; 32]>::try_from(left), <&[f32; 32]>::try_from(right)) {
        return squared_euclidean_32(left, right);
    }
    approximate_squared_euclidean_portable(left, right)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn squared_euclidean_32(left: &[f32; 32], right: &[f32; 32]) -> f32 {
    let lanes = std::array::from_fn(|coordinate| {
        let difference = left[coordinate] - right[coordinate];
        difference * difference
    });
    reduce_32_lanes(&lanes)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn checked_euclidean_32(
    a: &[f32; 32],
    b: &[f32; 32],
    row: usize,
    candidate: usize,
) -> Result<f32, ApproximateSearchError> {
    let distance = squared_euclidean_32(a, b).sqrt();
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(ApproximateSearchError::NonFiniteDistance { row, candidate })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
fn seed_euclidean_leaf_32(
    data: DenseRows<'_>,
    rows: &[usize],
    ranks: &mut [Vec<TrainingNeighbor>],
    cutoffs: &mut [f32],
    k: usize,
) -> Result<(), ApproximateSearchError> {
    for (position, &left) in rows.iter().enumerate() {
        let a: &[f32; 32] = data.row(left).try_into().expect("32 training features");
        for &right in &rows[position + 1..] {
            let b: &[f32; 32] = data.row(right).try_into().expect("32 training features");
            let distance = checked_euclidean_32(a, b, left, right)?;
            offer_below_cutoff(&mut ranks[left], &mut cutoffs[left], right, distance, k);
            offer_below_cutoff(&mut ranks[right], &mut cutoffs[right], left, distance, k);
        }
    }
    Ok(())
}

#[inline(always)]
fn approximate_squared_euclidean_portable(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let (blocks_left, left) = left.as_chunks::<32>();
    let (blocks_right, right) = right.as_chunks::<32>();
    let mut sum = 0.0;
    if !blocks_left.is_empty() {
        let mut wide = [0.0; 32];
        for (left, right) in blocks_left.iter().zip(blocks_right) {
            for ((lane, left), right) in wide.iter_mut().zip(left).zip(right) {
                let delta = left - right;
                *lane = delta.mul_add(delta, *lane);
            }
        }
        sum = reduce_32_lanes(&wide);
    }
    let (blocks_left, left) = left.as_chunks::<4>();
    let (blocks_right, right) = right.as_chunks::<4>();
    let mut lanes = [sum, 0.0, 0.0, 0.0];
    for (left, right) in blocks_left.iter().zip(blocks_right) {
        for ((lane, left), right) in lanes.iter_mut().zip(left).zip(right) {
            let delta = left - right;
            *lane = delta.mul_add(delta, *lane);
        }
    }
    sum = reduce_4_lanes(&lanes);
    for (left, right) in left.iter().zip(right) {
        let delta = left - right;
        sum = delta.mul_add(delta, sum);
    }
    sum
}

#[inline]
fn reduce_32_lanes(lanes: &[f32; 32]) -> f32 {
    let pair = |i| (lanes[i] + lanes[i + 8]) + (lanes[i + 16] + lanes[i + 24]);
    let cross = |i| pair(i) + pair(i + 4);
    (cross(0) + cross(2)) + (cross(1) + cross(3))
}

#[inline]
fn reduce_4_lanes(lanes: &[f32; 4]) -> f32 {
    (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
}

#[cfg(test)]
fn approximate_euclidean_portable(left: &[f32], right: &[f32]) -> f32 {
    approximate_squared_euclidean_portable(left, right).sqrt()
}

impl RankedNeighbors {
    fn with_capacity(rows: usize, width: usize) -> Result<Self, ApproximateSearchError> {
        let slots = rows
            .checked_mul(width)
            .ok_or(ApproximateSearchError::RankShape)?;
        let mut indices = Vec::new();
        let mut distances = Vec::new();
        indices
            .try_reserve_exact(slots)
            .map_err(|_| ApproximateSearchError::RankShape)?;
        distances
            .try_reserve_exact(slots)
            .map_err(|_| ApproximateSearchError::RankShape)?;
        Ok(Self {
            width,
            indices,
            distances,
        })
    }

    fn extend(&mut self, row: &[Neighbor]) {
        self.indices
            .extend(row.iter().map(|neighbor| neighbor.index as i32));
        self.distances
            .extend(row.iter().map(|neighbor| neighbor.distance));
        self.indices
            .extend(std::iter::repeat_n(-1, self.width - row.len()));
        self.distances
            .extend(std::iter::repeat_n(f32::INFINITY, self.width - row.len()));
    }

    fn extend_training(&mut self, row: usize, rank: &[TrainingNeighbor], k: usize) {
        let start = self.indices.len();
        self.indices.push(row as i32);
        self.distances.push(0.0);
        for neighbor in rank
            .iter()
            .map(|entry| entry.point())
            .filter(|neighbor| neighbor.index != row)
            .take(k - 1)
        {
            self.indices.push(neighbor.index as i32);
            self.distances.push(neighbor.distance);
        }
        self.indices.resize(start + self.width, -1);
        self.distances.resize(start + self.width, f32::INFINITY);
    }
}

fn bound(best: &[Neighbor], k: usize, factor: f32) -> f32 {
    if best.len() < k {
        f32::INFINITY
    } else {
        best[k - 1].distance * factor
    }
}

fn offer<T>(row: &mut Vec<T>, candidate: T, k: usize, point: impl Fn(&T) -> Neighbor) -> bool {
    let Neighbor { index, distance } = point(&candidate);
    if (row.len() == k && distance >= point(&row[k - 1]).distance)
        || row.iter().any(|entry| point(entry).index == index)
    {
        return false;
    }
    let position = row.partition_point(|entry| point(entry).distance <= distance);
    if row.len() == k {
        row.pop();
    }
    row.insert(position, candidate);
    true
}

#[derive(Clone, Copy)]
struct TrainingNeighbor {
    // Training row IDs fit in i32. The vacant row uses -1.
    row: i32,
    distance: f32,
    fresh: bool,
}

impl TrainingNeighbor {
    #[inline]
    fn index(self) -> usize {
        self.row as isize as usize
    }
    #[inline]
    fn point(self) -> Neighbor {
        Neighbor {
            index: self.index(),
            distance: self.distance,
        }
    }

    const VACANT: Self = Self {
        row: -1,
        distance: f32::INFINITY,
        fresh: false,
    };
}

fn offer_training(row: &mut Vec<TrainingNeighbor>, index: usize, distance: f32, k: usize) -> bool {
    offer(
        row,
        TrainingNeighbor {
            row: index as i32,
            distance,
            fresh: true,
        },
        k,
        |entry| entry.point(),
    )
}

fn offer_training_heap(row: &mut [TrainingNeighbor], index: usize, distance: f32) -> bool {
    if distance >= row[0].distance || row.iter().any(|entry| entry.index() == index) {
        return false;
    }
    let mut place = 0;
    while 2 * place + 1 < row.len() {
        let mut child = 2 * place + 1;
        if child + 1 < row.len() && row[child + 1].distance > row[child].distance {
            child += 1;
        }
        if row[child].distance <= distance {
            break;
        }
        row[place] = row[child];
        place = child;
    }
    row[place] = TrainingNeighbor {
        row: index as i32,
        distance,
        fresh: true,
    };
    true
}

fn fill_missing(
    data: DenseRows<'_>,
    metric: DistanceMetric,
    ranks: &mut [Vec<TrainingNeighbor>],
    k: usize,
    rng: &mut [i64; 3],
) -> Result<(), ApproximateSearchError> {
    for (row, neighbors) in ranks.iter_mut().enumerate() {
        let mut missing = if data.batches_cosine() {
            neighbors
                .iter()
                .filter(|entry| entry.index() == usize::MAX)
                .count()
        } else {
            k - neighbors.len()
        };
        // Only self-draws retry. Bound them even if the RNG has a degenerate state.
        let attempts = missing.saturating_mul(2).saturating_add(32);
        for _ in 0..attempts {
            if missing == 0 {
                break;
            }
            let candidate = sample_absolute_vertex(rng, data.n_rows);
            if candidate != row {
                let distance = data.distance(metric, row, candidate)?;
                if data.batches_cosine() {
                    offer_training_heap(neighbors, candidate, distance);
                } else {
                    offer_training(neighbors, candidate, distance, k);
                }
                missing -= 1;
            }
        }
    }
    Ok(())
}

fn incident(ranks: &[Vec<TrainingNeighbor>]) -> Vec<Vec<usize>> {
    let mut edges = vec![Vec::new(); ranks.len()];
    for (row, neighbors) in ranks.iter().enumerate() {
        for entry in neighbors {
            let neighbor = entry.point();
            if row != neighbor.index {
                edges[row].push(neighbor.index);
                edges[neighbor.index].push(row);
            }
        }
    }
    for edge in &mut edges {
        edge.sort_unstable();
        edge.dedup();
    }
    edges
}

fn refine_supplied(
    data: DenseRows<'_>,
    metric: DistanceMetric,
    ranks: &mut [Vec<TrainingNeighbor>],
    k: usize,
    rounds: usize,
) -> Result<(), ApproximateSearchError> {
    let mut previous = vec![Vec::new(); ranks.len()];
    let mut new = Vec::new();
    for _ in 0..rounds {
        let mut candidates = incident(ranks);
        let mut changes = 0;
        for (neighbors, old) in candidates.iter_mut().zip(&previous) {
            neighbors.truncate(MAX_CANDIDATES);
            new.clear();
            new.extend(
                neighbors
                    .iter()
                    .copied()
                    .filter(|index| old.binary_search(index).is_err()),
            );
            changes += compare_candidates(data, metric, ranks, &new, old, k)?;
        }
        if changes == 0 {
            break;
        }
        previous = candidates;
    }
    Ok(())
}

fn refine(
    data: DenseRows<'_>,
    metric: DistanceMetric,
    ranks: &mut [Vec<TrainingNeighbor>],
    k: usize,
    rounds: usize,
    update_limit: usize,
) -> Result<(), ApproximateSearchError> {
    let mut euclidean = (metric == DistanceMetric::Euclidean && data.n_rows >= SMALL_DATA_LIMIT)
        .then(|| EuclideanRefinement {
            distances: vec![(0, 0.0); 32_768],
            cutoffs: ranks.iter().map(|row| training_cutoff(row, k)).collect(),
        });
    let compare_euclidean = euclidean_comparator(data.n_features);
    let mut fresh = vec![Vec::new(); ranks.len()];
    let mut retained = vec![Vec::new(); ranks.len()];
    for _ in 0..rounds {
        for (row, neighbors) in ranks.iter_mut().enumerate() {
            for entry in neighbors {
                let other = entry.index();
                if row != other {
                    let candidates = if entry.fresh {
                        &mut fresh
                    } else {
                        &mut retained
                    };
                    candidates[row].push(other);
                    candidates[other].push(row);
                }
                entry.fresh = false;
            }
        }
        let mut changes = 0;
        for (new, old) in fresh.iter_mut().zip(&mut retained) {
            for candidates in [&mut *new, &mut *old] {
                candidates.sort_unstable();
                candidates.dedup();
                candidates.truncate(MAX_CANDIDATES);
            }
            changes += if let Some(refinement) = &mut euclidean {
                // SAFETY: the selected kernel matches this width and CPU.
                unsafe { compare_euclidean(data, ranks, new, old, k, refinement)? }
            } else {
                compare_candidates(data, metric, ranks, new, old, k)?
            };
            new.clear();
            old.clear();
        }
        if changes <= update_limit {
            break;
        }
    }
    Ok(())
}

struct EuclideanRefinement {
    distances: Vec<(u64, f32)>,
    cutoffs: Vec<f32>,
}

fn training_cutoff(row: &[TrainingNeighbor], k: usize) -> f32 {
    if row.len() == k {
        row[k - 1].distance
    } else {
        f32::INFINITY
    }
}

#[inline]
fn offer_below_cutoff(
    row: &mut Vec<TrainingNeighbor>,
    cutoff: &mut f32,
    index: usize,
    distance: f32,
    k: usize,
) -> bool {
    if distance >= *cutoff || !offer_training(row, index, distance, k) {
        return false;
    }
    *cutoff = training_cutoff(row, k);
    true
}

fn compare_candidates(
    data: DenseRows<'_>,
    metric: DistanceMetric,
    ranks: &mut [Vec<TrainingNeighbor>],
    new: &[usize],
    old: &[usize],
    k: usize,
) -> Result<usize, ApproximateSearchError> {
    let mut changes = 0;
    for (position, &left) in new.iter().enumerate() {
        for &right in new[position + 1..].iter().chain(old) {
            if left == right {
                continue;
            }
            let distance = data.distance(metric, left, right)?;
            changes += usize::from(offer_training(&mut ranks[left], right, distance, k));
            changes += usize::from(offer_training(&mut ranks[right], left, distance, k));
        }
    }
    Ok(changes)
}

type EuclideanComparator = unsafe fn(
    DenseRows<'_>,
    &mut [Vec<TrainingNeighbor>],
    &[usize],
    &[usize],
    usize,
    &mut EuclideanRefinement,
) -> Result<usize, ApproximateSearchError>;

fn euclidean_comparator(width: usize) -> EuclideanComparator {
    #[cfg(target_arch = "x86_64")]
    if width == 32 && std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        return compare_euclidean_32;
    }
    let _ = width;
    compare_euclidean_portable
}

fn compare_euclidean_portable(
    data: DenseRows<'_>,
    ranks: &mut [Vec<TrainingNeighbor>],
    new: &[usize],
    old: &[usize],
    k: usize,
    refinement: &mut EuclideanRefinement,
) -> Result<usize, ApproximateSearchError> {
    refinement.compare(ranks, new, old, k, |left, right| {
        data.distance(DistanceMetric::Euclidean, left, right)
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
fn compare_euclidean_32(
    data: DenseRows<'_>,
    ranks: &mut [Vec<TrainingNeighbor>],
    new: &[usize],
    old: &[usize],
    k: usize,
    refinement: &mut EuclideanRefinement,
) -> Result<usize, ApproximateSearchError> {
    refinement.compare(ranks, new, old, k, |left, right| {
        let a: &[f32; 32] = data.row(left).try_into().expect("32 training features");
        let b: &[f32; 32] = data.row(right).try_into().expect("32 training features");
        checked_euclidean_32(a, b, left, right)
    })
}

impl EuclideanRefinement {
    #[inline(always)]
    fn compare(
        &mut self,
        ranks: &mut [Vec<TrainingNeighbor>],
        new: &[usize],
        old: &[usize],
        k: usize,
        compute: impl Fn(usize, usize) -> Result<f32, ApproximateSearchError>,
    ) -> Result<usize, ApproximateSearchError> {
        let mut changes = 0;
        for (position, &left) in new.iter().enumerate() {
            for &right in new[position + 1..].iter().chain(old) {
                if left == right {
                    continue;
                }
                // Euclidean candidate distances are symmetric and features
                // are fixed during refinement. Keep offer order unchanged.
                let key = ((left.min(right) as u64) << 32) | right.max(left) as u64;
                let slot = ((key ^ (key >> 32)).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ key.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    >> 49;
                let entry = &mut self.distances[slot as usize];
                if entry.0 != key {
                    *entry = (key, compute(left, right)?);
                }
                let distance = entry.1;
                changes += usize::from(offer_below_cutoff(
                    &mut ranks[left],
                    &mut self.cutoffs[left],
                    right,
                    distance,
                    k,
                ));
                changes += usize::from(offer_below_cutoff(
                    &mut ranks[right],
                    &mut self.cutoffs[right],
                    left,
                    distance,
                    k,
                ));
            }
        }
        Ok(changes)
    }
}

fn draw(rng: &mut Mt19937, upper: usize) -> usize {
    ((u64::from(rng.next_u32()) * upper as u64) >> 32) as usize
}

// Partial Fisher-Yates sampling visits each index at most once.
fn sample_rows(n_rows: usize, rng: &mut Mt19937) -> impl Iterator<Item = usize> + '_ {
    let mut rows: Vec<_> = (0..n_rows).collect();
    (0..n_rows).map(move |position| {
        let pivot = position + draw(rng, n_rows - position);
        rows.swap(position, pivot);
        rows[position]
    })
}

enum Tree {
    Leaf(Vec<usize>),
    Split {
        a: usize,
        b: usize,
        threshold: f32,
        left: Box<Tree>,
        right: Box<Tree>,
    },
    Balanced {
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

impl Tree {
    fn build(
        data: DenseRows<'_>,
        metric: DistanceMetric,
        mut rows: Vec<usize>,
        leaf_size: usize,
        rng: &mut Mt19937,
    ) -> Self {
        if rows.len() <= leaf_size {
            return Self::Leaf(rows);
        }
        let first = draw(rng, rows.len());
        let second = draw(rng, rows.len() - 1);
        let (a, b) = (rows[first], rows[second + usize::from(second >= first)]);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &index in &rows {
            let da = metric.distance(data.row(index), data.row(a));
            let db = metric.distance(data.row(index), data.row(b));
            match da.total_cmp(&db) {
                std::cmp::Ordering::Less => left.push(index),
                std::cmp::Ordering::Greater => right.push(index),
                std::cmp::Ordering::Equal if draw(rng, 2) == 0 => left.push(index),
                std::cmp::Ordering::Equal => right.push(index),
            }
        }
        // Keep tree depth bounded for skewed and degenerate projections.
        if left.len().min(right.len()) < rows.len().div_ceil(16) {
            let right = rows.split_off(rows.len() / 2);
            return Self::Balanced {
                left: Box::new(Self::build(data, metric, rows, leaf_size, rng)),
                right: Box::new(Self::build(data, metric, right, leaf_size, rng)),
            };
        }
        Self::Split {
            a,
            b,
            threshold: 0.0,
            left: Box::new(Self::build(data, metric, left, leaf_size, rng)),
            right: Box::new(Self::build(data, metric, right, leaf_size, rng)),
        }
    }

    fn seed(
        &self,
        data: DenseRows<'_>,
        metric: DistanceMetric,
        ranks: &mut [Vec<TrainingNeighbor>],
        k: usize,
    ) -> Result<(), ApproximateSearchError> {
        match self {
            Self::Leaf(rows) => {
                for (position, &left) in rows.iter().enumerate() {
                    for &right in &rows[position + 1..] {
                        let distance = data.distance(metric, left, right)?;
                        if data.batches_cosine() {
                            offer_training_heap(&mut ranks[left], right, distance);
                            offer_training_heap(&mut ranks[right], left, distance);
                        } else {
                            offer_training(&mut ranks[left], right, distance, k);
                            offer_training(&mut ranks[right], left, distance, k);
                        }
                    }
                }
            }
            Self::Split { left, right, .. } | Self::Balanced { left, right } => {
                left.seed(data, metric, ranks, k)?;
                right.seed(data, metric, ranks, k)?;
            }
        }
        Ok(())
    }

    fn leaf(&self, data: DenseRows<'_>, metric: DistanceMetric, query: &[f32]) -> &[usize] {
        match self {
            Self::Leaf(rows) => rows,
            Self::Balanced { left, .. } => left.leaf(data, metric, query),
            Self::Split {
                a, b, left, right, ..
            } => {
                let da = metric.distance(query, data.row(*a));
                let db = metric.distance(query, data.row(*b));
                if da <= db { left } else { right }.leaf(data, metric, query)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fixed_width_leaf_offers_preserve_distance_bits_tie_order_and_overflow() {
        if !std::is_x86_feature_detected!("avx") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let data: Vec<f32> = [-3.0, 2.0, -1.0, 1.0, 1.0, 0.0, 5.0]
            .into_iter()
            .flat_map(|value| [value; 32])
            .collect();
        let input = DenseRows::training(&data, 7, 32).unwrap();
        let groups = [vec![3, 2, 0, 6, 1, 4], vec![5, 4, 3, 0, 6, 2]];
        let mut expected = vec![Vec::new(); 7];
        let mut actual = vec![Vec::new(); 7];
        let mut cutoffs = vec![f32::INFINITY; 7];
        for group in groups {
            // SAFETY: both target features and the fixed width were checked.
            unsafe { seed_euclidean_leaf_32(input, &group, &mut actual, &mut cutoffs, 3) }.unwrap();
            Tree::Leaf(group)
                .seed(input, DistanceMetric::Euclidean, &mut expected, 3)
                .unwrap();
        }
        for (actual, expected) in actual.iter().zip(&expected) {
            let bits = |row: &[TrainingNeighbor]| {
                row.iter()
                    .map(|entry| (entry.index(), entry.distance.to_bits(), entry.fresh))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(actual), bits(expected));
        }

        let data = [[f32::MAX; 32], [-f32::MAX; 32]].concat();
        let input = DenseRows::training(&data, 2, 32).unwrap();
        let mut ranks = vec![Vec::new(); 2];
        // SAFETY: both target features and the fixed width were checked.
        let error = unsafe {
            seed_euclidean_leaf_32(input, &[0, 1], &mut ranks, &mut [f32::INFINITY; 2], 1)
        }
        .unwrap_err();
        assert_eq!(
            error,
            ApproximateSearchError::NonFiniteDistance {
                row: 0,
                candidate: 1
            }
        );
    }

    #[test]
    fn approximate_euclidean_preserves_the_four_lane_reduction() {
        let left = [0x3f14_0085, 0x3f9a_cf41, 0xbe46_3ac3, 0x3e0e_f583].map(f32::from_bits);
        let right = [0x3f4e_977e, 0x3f71_adfd, 0x3ed8_a165, 0xbd64_49e2].map(f32::from_bits);
        assert_eq!(
            approximate_euclidean_portable(&left, &right).to_bits(),
            0x3f3c_57ce
        );
        assert_eq!(approximate_euclidean(&left, &right).to_bits(), 0x3f3c_57ce);
    }

    #[test]
    fn random_fill_caps_self_retries() {
        let data = DenseRows::training(&[0.0, 1.0], 2, 1).unwrap();
        let mut rows = vec![Vec::new(); 2];
        fill_missing(data, DistanceMetric::Euclidean, &mut rows, 1, &mut [0; 3]).unwrap();
        assert!(rows[0].is_empty());
        assert_eq!(
            rows[1]
                .iter()
                .map(|entry| entry.point())
                .collect::<Vec<_>>(),
            [Neighbor {
                index: 0,
                distance: 1.0
            }]
        );
    }

    #[derive(Deserialize)]
    struct ConstructionProbes {
        cases: BTreeMap<String, Inputs>,
        scheduling: BTreeMap<String, Scheduling>,
        partial_fill: PartialFill,
    }

    #[derive(Deserialize)]
    struct Inputs {
        n_rows: usize,
        n_features: usize,
        width: usize,
        metric: String,
        seed: Option<u32>,
        training_bits: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct Scheduling {
        random_only: [Ranks; 2],
    }

    #[derive(Deserialize)]
    struct Ranks {
        indices: Vec<i32>,
        bits: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct PartialFill {
        n_rows: usize,
        n_features: usize,
        width: usize,
        seed: u32,
        training_bits: Vec<u32>,
        left: Vec<bool>,
        expected: Ranks,
    }

    #[derive(Deserialize)]
    struct PivotProbes {
        cases: BTreeMap<String, SeededCase>,
        #[serde(default)]
        distance_controls: BTreeMap<usize, DistanceControl>,
        #[serde(default)]
        caller_seeds: BTreeMap<u32, CallerSeed>,
    }

    #[derive(Deserialize)]
    struct SeededCase {
        #[serde(flatten)]
        input: Inputs,
        snapshots: BTreeMap<usize, Ranks>,
    }

    #[derive(Deserialize)]
    struct DistanceControl {
        data: Vec<u32>,
        #[serde(flatten)]
        ranks: Ranks,
    }

    #[derive(Deserialize)]
    struct CallerSeed {
        first_state: [i64; 3],
        second_state: [i64; 3],
        scalar: u64,
    }

    #[test]
    #[ignore = "requires output from a public construction or pivot probe"]
    fn measure_native_seed_ranks() {
        let path = std::env::var_os("APPROXIMATE_SEARCH_OBSERVATIONS").unwrap();
        let probes: PivotProbes = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(!probes.cases.is_empty());
        for (seed, expected) in probes.caller_seeds {
            let mut rng = Mt19937::new(seed);
            assert_eq!(draw_negative_seed(|| rng.next_u32()), expected.first_state);
            assert_eq!(draw_negative_seed(|| rng.next_u32()), expected.second_state);
            let scalar = draw_projection_scalar(&mut Mt19937::new(seed));
            assert_eq!(scalar, expected.scalar);
            println!("caller seed {seed}: both signed draws and projection scalar {scalar} match");
        }
        for (name, case) in probes.cases {
            let input = case.input;
            let values: Vec<_> = input
                .training_bits
                .into_iter()
                .map(f32::from_bits)
                .collect();
            let data = DenseRows::training(&values, input.n_rows, input.n_features).unwrap();
            let metric = match input.metric.as_str() {
                "euclidean" => DistanceMetric::Euclidean,
                "cosine" => DistanceMetric::Cosine,
                "correlation" => DistanceMetric::Correlation,
                metric => panic!("unknown metric: {metric}"),
            };
            let k = input.width;
            let seed = input.seed.unwrap_or(42);
            let mut streams = ApproximateStreams::draw(&mut Mt19937::new(seed));
            let (_tree, mut native) = seed_forest(data, metric, k, seed, &mut streams).unwrap();
            for round in [0, 1] {
                let mut ranks = RankedNeighbors::with_capacity(input.n_rows, k).unwrap();
                for (row, neighbors) in native.iter().enumerate() {
                    ranks.extend_training(row, neighbors, k);
                }
                let wheel = &case.snapshots[&round];
                let mut shared = 0;
                let mut aligned = 0;
                let mut bits = 0;
                let mut wide_bits = 0;
                for (row, columns) in ranks.indices.chunks_exact(k).enumerate() {
                    let start = row * k;
                    let expected = &wheel.indices[start..start + k];
                    for (slot, &col) in columns
                        .iter()
                        .enumerate()
                        .skip(1)
                        .filter(|(_, col)| **col >= 0)
                    {
                        if let Some(position) =
                            expected.iter().position(|&candidate| candidate == col)
                        {
                            shared += 1;
                            aligned += usize::from(slot == position);
                            let expected_bits = wheel.bits[start + position];
                            bits += usize::from(
                                ranks.distances[start + slot].to_bits() == expected_bits,
                            );
                            let wide = metric.distance(data.row(row), data.row(col as usize));
                            wide_bits += usize::from(wide.to_bits() == expected_bits);
                        }
                    }
                }
                println!(
                    "{name} native round {round}: shared {shared}, aligned {aligned}, bits {wide_bits} -> {bits} of {shared}"
                );
                if round == 0 {
                    refine(data, metric, &mut native, k, 1, 0).unwrap();
                }
            }
        }

        for (dimension, control) in probes.distance_controls {
            let values: Vec<_> = control.data.into_iter().map(f32::from_bits).collect();
            let n = values.len() / dimension;
            let data = DenseRows::training(&values, n, dimension).unwrap();
            let mut wide = 0;
            let mut validated = 0;
            for (slot, (&col, &bits)) in control
                .ranks
                .indices
                .iter()
                .zip(&control.ranks.bits)
                .enumerate()
            {
                let row = slot / n;
                let (left, right) = (data.row(row), data.row(col as usize));
                wide +=
                    usize::from(DistanceMetric::Euclidean.distance(left, right).to_bits() == bits);
                let selected = approximate_euclidean(left, right).to_bits();
                assert_eq!(
                    approximate_euclidean_portable(left, right).to_bits(),
                    selected
                );
                validated += usize::from(selected == bits);
            }
            println!(
                "dimension {dimension}: {wide} -> {validated} exact distance bits of {}",
                control.ranks.bits.len()
            );
            assert_eq!(validated, control.ranks.bits.len(), "dimension {dimension}");
        }
    }

    #[test]
    #[ignore = "requires reference/probe_approximate_construction.py output"]
    fn replay_initial_offers() {
        let path = std::env::var_os("APPROXIMATE_SEARCH_OBSERVATIONS").unwrap();
        let probes: ConstructionProbes =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(!probes.scheduling.is_empty());
        for (name, run) in probes.scheduling {
            let input = &probes.cases[&name];
            let values: Vec<_> = input
                .training_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect();
            let data = DenseRows::training(&values, input.n_rows, input.n_features).unwrap();
            let metric = match input.metric.as_str() {
                "euclidean" => DistanceMetric::Euclidean,
                "cosine" => DistanceMetric::Cosine,
                "correlation" => DistanceMetric::Correlation,
                metric => panic!("unknown metric: {metric}"),
            };
            let k = input.width;
            let mut rng = Mt19937::new(input.seed.unwrap_or(42));
            let mut rng = draw_negative_seed(|| rng.next_u32());
            let mut ranks = vec![Vec::new(); input.n_rows];
            fill_missing(data, metric, &mut ranks, k, &mut rng).unwrap();
            for (round, expected) in run.random_only.into_iter().enumerate() {
                let mut observed = RankedNeighbors::with_capacity(input.n_rows, k).unwrap();
                for (row, rank) in ranks.iter().enumerate() {
                    observed.extend_training(row, rank, k);
                }
                let mut columns = 0;
                let mut pairs = 0;
                for (j, (&actual, &want)) in
                    observed.indices.iter().zip(&expected.indices).enumerate()
                {
                    columns += usize::from(actual == want);
                    pairs += usize::from(
                        actual == want && observed.distances[j].to_bits() == expected.bits[j],
                    );
                }
                println!(
                    "{name} no-tree round {round}: {columns} columns, {pairs} pairs of {}",
                    observed.indices.len()
                );
                if round == 1 || metric != DistanceMetric::Cosine {
                    assert_eq!(observed.indices, expected.indices, "{name} round {round}");
                }
                refine(data, metric, &mut ranks, k, 1, 0).unwrap();
            }
        }

        let run = probes.partial_fill;
        let values: Vec<_> = run.training_bits.into_iter().map(f32::from_bits).collect();
        let data = DenseRows::training(&values, run.n_rows, run.n_features).unwrap();
        let (left, right): (Vec<_>, Vec<_>) = (0..run.n_rows).partition(|&row| run.left[row]);
        let mut ranks = vec![Vec::new(); run.n_rows];
        let metric = DistanceMetric::Euclidean;
        for leaf in [left, right] {
            Tree::Leaf(leaf)
                .seed(data, metric, &mut ranks, run.width)
                .unwrap();
        }
        let mut rng = Mt19937::new(run.seed);
        let mut rng = draw_negative_seed(|| rng.next_u32());
        fill_missing(data, metric, &mut ranks, run.width, &mut rng).unwrap();
        let mut observed = RankedNeighbors::with_capacity(run.n_rows, run.width).unwrap();
        for (row, rank) in ranks.iter().enumerate() {
            observed.extend_training(row, rank, run.width);
        }
        assert_eq!(observed.indices, run.expected.indices, "partial tree fill");
        println!(
            "partial tree fill: all {} columns match",
            observed.indices.len()
        );
    }
}
