//! Exhaustive neighbor search and fuzzy graph fitting for sorted CSR feature matrices.

use std::cmp::Ordering;
use std::fmt;

use crate::fit::{build_fitted_graph, build_graph_with_cutoff};
use crate::fuzzy_graph::{CsrGraph, GraphError, GraphOptions};
use crate::neighbors::{
    DistanceMetric, Neighbor, boolean_distance_pairs, bray_curtis_distance_pairs,
    canberra_distance_pairs, collect_neighbors, cosine_distance_pairs, euclidean_distance_pairs,
    manhattan_distance_pairs, manhattan_distance_pairs_wide,
};
use crate::transform::{DenseGraphTransformError, QueryTieRanking};

mod euclidean_index;
use euclidean_index::SparseEuclideanIndex;

/// Borrowed CSR buffers with their declared shape.
#[derive(Clone, Copy, Debug)]
pub struct CsrFeatures<'a> {
    /// Row offsets, including the terminal offset.
    pub indptr: &'a [usize],
    /// Column index for each stored feature.
    pub indices: &'a [usize],
    /// Finite stored feature values.
    pub data: &'a [f32],
    /// Matrix shape as (rows, features).
    pub shape: (usize, usize),
}

impl<'a> CsrFeatures<'a> {
    fn row(self, row: usize) -> CsrRow<'a> {
        let entries = self.indptr[row]..self.indptr[row + 1];
        CsrRow {
            columns: &self.indices[entries.clone()],
            values: &self.data[entries],
        }
    }
}

/// Distinct stored rows and the map back to the original row order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrRowGroups {
    /// First original row in each group, in lexicographic row-key order.
    pub first: Vec<usize>,
    /// Group index for each original row.
    pub inverse: Vec<usize>,
    /// Number of original rows in each group.
    pub counts: Vec<usize>,
}

/// Group canonical CSR rows by their stored column indices and values.
///
/// Row keys start with all stored column indices, followed by the stored
/// values. Explicit zeros participate in the key. Positive and negative
/// zero compare equal, with the first occurrence chosen as representative.
pub fn unique_csr_rows(input: CsrFeatures<'_>) -> Result<CsrRowGroups, SparseKnnError> {
    validate_csr_features(input, None, false)?;
    let mut order: Vec<_> = (0..input.shape.0).collect();
    order.sort_unstable_by(|&left, &right| {
        compare_row_keys(input.row(left), input.row(right)).then(left.cmp(&right))
    });
    let mut groups = CsrRowGroups {
        first: Vec::new(),
        inverse: vec![0; order.len()],
        counts: Vec::new(),
    };
    for row in order {
        let distinct = groups.first.last().is_none_or(|&previous| {
            compare_row_keys(input.row(previous), input.row(row)) != Ordering::Equal
        });
        if distinct {
            groups.first.push(row);
            groups.counts.push(1);
        } else if let Some(count) = groups.counts.last_mut() {
            *count += 1;
        }
        groups.inverse[row] = groups.first.len() - 1;
    }
    Ok(groups)
}

enum RowKeyElement {
    Column(usize),
    Value(f32),
}

fn compare_row_keys(left: CsrRow<'_>, right: CsrRow<'_>) -> Ordering {
    let left_key = row_key(left);
    let right_key = row_key(right);
    for (left, right) in left_key.zip(right_key) {
        use RowKeyElement::{Column, Value};
        let ordering = match (left, right) {
            (Column(left), Column(right)) => left.cmp(&right),
            (Value(left), Value(right)) => left.partial_cmp(&right).expect("validated values"),
            (Column(column), Value(value)) => compare_column_to_value(column, value),
            (Value(value), Column(column)) => compare_column_to_value(column, value).reverse(),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.columns.len().cmp(&right.columns.len())
}

fn row_key(row: CsrRow<'_>) -> impl Iterator<Item = RowKeyElement> {
    row.columns
        .iter()
        .copied()
        .map(RowKeyElement::Column)
        .chain(row.values.iter().copied().map(RowKeyElement::Value))
}

fn compare_column_to_value(column: usize, value: f32) -> Ordering {
    // A large integer can round onto an exactly represented feature value.
    (column as f64)
        .partial_cmp(&f64::from(value))
        .expect("validated value")
        .then_with(|| (column as u128).cmp(&(value as u128)))
}

/// Distance metrics for exhaustive sparse feature search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseDistanceMetric {
    /// A metric shared with dense neighbor search.
    Dense(DistanceMetric),
    /// One minus the dot product of L2-normalized rows, clipped to `0..=2`.
    Cosine,
}

impl From<DistanceMetric> for SparseDistanceMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::Cosine,
            _ => Self::Dense(metric),
        }
    }
}

/// Input validation failures for [`exhaustive_csr_knn`].
#[derive(Clone, Debug, PartialEq)]
pub enum SparseKnnError {
    /// Row pointer, column index, and stored value lengths must agree with the shape.
    InvalidShape {
        n_rows: usize,
        n_features: usize,
        offsets: usize,
        indices: usize,
        values: usize,
    },
    /// The row pointers must begin at zero and end at the stored entry count.
    InvalidEndpoints {
        first: usize,
        last: usize,
        entries: usize,
    },
    /// Row pointers must be nondecreasing and within the stored entry count.
    InvalidRowOffsets {
        row: usize,
        start: usize,
        end: usize,
        entries: usize,
    },
    /// Stored column indices must be in `0..n_features`.
    InvalidColumnIndex {
        row: usize,
        position: usize,
        column: usize,
        n_features: usize,
    },
    /// Column indices must be sorted and unique when the path requires canonical CSR.
    UnorderedColumnIndices {
        row: usize,
        position: usize,
        previous: usize,
        column: usize,
    },
    /// `k` must be at least one and no greater than the number of rows.
    InvalidK { k: usize, n_rows: usize },
    /// A stored feature value must be finite.
    NonFiniteValue { position: usize, value: f32 },
    /// Correlation requires mean-centering the implicit coordinates.
    UnsupportedCorrelation,
    /// This distance requires a separate sparse feature path.
    UnsupportedFeatureMetric(DistanceMetric),
}

impl fmt::Display for SparseKnnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_rows,
                n_features,
                offsets,
                indices,
                values,
            } => write!(
                formatter,
                "CSR shape {n_rows}x{n_features} does not match {offsets} row pointers, \
                 {indices} column indices, and {values} stored values"
            ),
            Self::InvalidEndpoints {
                first,
                last,
                entries,
            } => write!(
                formatter,
                "CSR row pointers must start at 0 and end at {entries}, got {first} and {last}"
            ),
            Self::InvalidRowOffsets {
                row,
                start,
                end,
                entries,
            } => write!(
                formatter,
                "CSR row {row} has invalid offsets {start}..{end} for {entries} stored entries"
            ),
            Self::InvalidColumnIndex {
                row,
                position,
                column,
                n_features,
            } => write!(
                formatter,
                "CSR row {row}, entry {position} has column {column}, outside 0..{n_features}"
            ),
            Self::UnorderedColumnIndices {
                row,
                position,
                previous,
                column,
            } => write!(
                formatter,
                "CSR row {row}, entry {position} has column {column}, expected greater than {previous}"
            ),
            Self::InvalidK { k, n_rows } => {
                write!(formatter, "k={k} is outside the valid range 1..={n_rows}")
            }
            Self::NonFiniteValue { position, value } => {
                write!(
                    formatter,
                    "CSR value at entry {position} is not finite: {value}"
                )
            }
            Self::UnsupportedCorrelation => {
                formatter.write_str("correlation is unsupported for CSR features")
            }
            Self::UnsupportedFeatureMetric(metric) => {
                write!(
                    formatter,
                    "{metric:?} is unsupported for this CSR feature path"
                )
            }
        }
    }
}

impl std::error::Error for SparseKnnError {}

/// Errors returned while fitting a sparse fuzzy graph.
#[derive(Clone, Debug, PartialEq)]
pub enum SparseGraphFitError {
    /// CSR validation or neighbor count validation failed.
    Knn(SparseKnnError),
    /// Fuzzy simplicial-set construction rejected the neighbor rows.
    Graph(GraphError),
    /// The tie-break callback returned invalid ranks.
    Ranking(DenseGraphTransformError),
}

impl fmt::Display for SparseGraphFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Knn(error) => write!(formatter, "sparse graph neighbor search failed: {error}"),
            Self::Graph(error) => {
                write!(formatter, "sparse fuzzy graph construction failed: {error}")
            }
            Self::Ranking(error) => write!(formatter, "sparse neighbor ranking failed: {error}"),
        }
    }
}

impl std::error::Error for SparseGraphFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Knn(error) => Some(error),
            Self::Graph(error) => Some(error),
            Self::Ranking(error) => Some(error),
        }
    }
}

/// Invalid CSR input or distance ranking in a held-out sparse search.
#[derive(Clone, Debug, PartialEq)]
pub enum SparseQueryRankError {
    /// The fitted feature matrix or requested count is invalid.
    Training(SparseKnnError),
    /// The query feature matrix is invalid.
    Queries(SparseKnnError),
    /// The CSR query matrix is wider than the training matrix.
    FeatureCount { training: usize, queries: usize },
    /// A selected distance or tie-break rank is invalid.
    Ranking(DenseGraphTransformError),
}

impl fmt::Display for SparseQueryRankError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Training(error) => write!(formatter, "invalid CSR training input: {error}"),
            Self::Queries(error) => write!(formatter, "invalid CSR query input: {error}"),
            Self::FeatureCount { training, queries } => {
                write!(
                    formatter,
                    "expected at most {training} CSR features, got {queries}"
                )
            }
            Self::Ranking(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SparseQueryRankError {}

/// Find the `k` nearest rows for every row of a sorted CSR feature matrix.
///
/// `indptr` contains `n_rows + 1` nondecreasing offsets, starting at zero and
/// ending at `indices.len() == data.len()`. Each row has ascending column
/// indices less than `n_features`. Stored values must be finite. Missing
/// entries represent zero, and empty rows and explicit signed zeros are
/// allowed. Cosine distance uses stored values separately when normalizing
/// rows with repeated columns. For the other metrics, combine duplicate
/// columns before calling this API.
///
/// Distances use a two-pointer merge in ascending column order, visiting only
/// stored coordinates. Euclidean distances sum squared `f64` differences,
/// round the sum to `f32`, and take its square root in `f32`. Manhattan distances
/// sum absolute `f32` differences in `f64`, and Chebyshev distances take their
/// maximum. Both round the final result to `f32`. Computed distances can overflow to
/// positive infinity.
///
/// For cosine distance, each row is normalized once. The norm uses float32
/// squares and a float64 sum and square root. The normalized values, dot
/// products and distances use float32. A zero-norm row is left unchanged,
/// and each row's distance to itself is exactly zero.
///
/// Every row is considered as a candidate, including the query itself, and
/// `k` must be in `1..=n_rows`. As in the dense search, rows are ranked by
/// [`f32::total_cmp`] distance and then by ascending candidate index.
pub fn exhaustive_csr_knn(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: impl Into<SparseDistanceMetric>,
) -> Result<Vec<Vec<Neighbor>>, SparseKnnError> {
    let input = CsrFeatures {
        indptr,
        indices,
        data,
        shape: (n_rows, n_features),
    };
    csr_knn_counting_cutoff(input, k, metric.into(), f32::INFINITY, false)
        .map(|(neighbors, _)| neighbors)
}

fn csr_knn_counting_cutoff(
    input: CsrFeatures<'_>,
    k: usize,
    metric: SparseDistanceMetric,
    cutoff: f32,
    allow_duplicates: bool,
) -> Result<(Vec<Vec<Neighbor>>, usize), SparseKnnError> {
    let metric = match metric {
        SparseDistanceMetric::Dense(DistanceMetric::Cosine) => SparseDistanceMetric::Cosine,
        SparseDistanceMetric::Dense(DistanceMetric::Correlation) => {
            return Err(SparseKnnError::UnsupportedCorrelation);
        }
        metric => metric,
    };
    validate_csr_features(
        input,
        Some(k),
        allow_duplicates || metric == SparseDistanceMetric::Cosine,
    )?;
    let normalized_data = (metric == SparseDistanceMetric::Cosine)
        .then(|| normalize_sparse_rows(input.indptr, input.data));
    let input = CsrFeatures {
        data: normalized_data.as_deref().unwrap_or(input.data),
        ..input
    };
    let canonical_euclidean = metric == SparseDistanceMetric::Dense(DistanceMetric::Euclidean)
        && (!allow_duplicates
            || input.indptr.windows(2).all(|offsets| {
                input.indices[offsets[0]..offsets[1]]
                    .windows(2)
                    .all(|columns| columns[0] != columns[1])
            }));
    let mut euclidean_index = if canonical_euclidean {
        SparseEuclideanIndex::new(input, input.shape.0)
    } else {
        None
    };
    let euclidean_pairs = if canonical_euclidean && euclidean_index.is_none() {
        cache_euclidean_pairs(input)
    } else {
        None
    };
    let mut removed = 0;
    let neighbors = collect_neighbors(input.shape.0, k, |query_index, candidates| {
        if let Some(index) = &mut euclidean_index {
            index.fill(input.row(query_index), candidates);
        } else if let Some(pairs) = &euclidean_pairs {
            fill_euclidean_candidates(candidates, query_index, input.shape.0, pairs);
        } else {
            let query = input.row(query_index);
            for index in 0..input.shape.0 {
                let distance = match metric {
                    SparseDistanceMetric::Dense(DistanceMetric::Euclidean)
                        if canonical_euclidean =>
                    {
                        canonical_euclidean_distance(query, input.row(index))
                    }
                    SparseDistanceMetric::Dense(metric) => {
                        sparse_distance(query, input.row(index), metric, input.shape.1)
                    }
                    SparseDistanceMetric::Cosine if query_index == index => 0.0,
                    SparseDistanceMetric::Cosine => sparse_cosine_distance(query, input.row(index)),
                };
                removed += usize::from(distance >= cutoff);
                candidates.push(Neighbor { index, distance });
            }
            return;
        }
        removed += candidates
            .iter()
            .filter(|neighbor| neighbor.distance >= cutoff)
            .count();
    });
    Ok((neighbors, removed))
}

fn validate_csr_features(
    CsrFeatures {
        indptr,
        indices,
        data,
        shape: (n_rows, n_features),
    }: CsrFeatures<'_>,
    k: Option<usize>,
    allow_duplicates: bool,
) -> Result<(), SparseKnnError> {
    if n_rows.checked_add(1) != Some(indptr.len()) || indices.len() != data.len() {
        return Err(SparseKnnError::InvalidShape {
            n_rows,
            n_features,
            offsets: indptr.len(),
            indices: indices.len(),
            values: data.len(),
        });
    }

    let first = indptr[0];
    let last = indptr[n_rows];
    if first != 0 || last != data.len() {
        return Err(SparseKnnError::InvalidEndpoints {
            first,
            last,
            entries: data.len(),
        });
    }

    if let Some(k) = k
        && !(1..=n_rows).contains(&k)
    {
        return Err(SparseKnnError::InvalidK { k, n_rows });
    }

    for (row, offsets) in indptr.windows(2).enumerate() {
        let (start, end) = (offsets[0], offsets[1]);
        if start > end || end > data.len() {
            return Err(SparseKnnError::InvalidRowOffsets {
                row,
                start,
                end,
                entries: data.len(),
            });
        }

        for position in start..end {
            let column = indices[position];
            if column >= n_features {
                return Err(SparseKnnError::InvalidColumnIndex {
                    row,
                    position,
                    column,
                    n_features,
                });
            }
            if position > start
                && (indices[position - 1] > column
                    || (indices[position - 1] == column && !allow_duplicates))
            {
                return Err(SparseKnnError::UnorderedColumnIndices {
                    row,
                    position,
                    previous: indices[position - 1],
                    column,
                });
            }
            let value = data[position];
            if !value.is_finite() {
                return Err(SparseKnnError::NonFiniteValue { position, value });
            }
        }
    }

    Ok(())
}

/// Select compact row-major Euclidean ranks from canonical sparse features.
///
/// A narrower query uses the leading training columns and the dense metric's
/// float32 product rounding. The callback receives one distance row in
/// training-column order when selected ranks tie.
pub fn rank_sparse_euclidean_queries(
    training: CsrFeatures<'_>,
    queries: CsrFeatures<'_>,
    k: usize,
    mut rank_ties: impl FnMut(&[f32], usize) -> Vec<usize>,
) -> Result<Vec<Neighbor>, SparseQueryRankError> {
    rank_sparse_feature_queries(
        training,
        queries,
        k,
        DistanceMetric::Euclidean,
        |distances, k| {
            let SparseTieDistances::Float32(distances) = distances else {
                unreachable!("Euclidean CSR ranks use float32")
            };
            rank_ties(distances, k)
        },
    )
}

/// A row of candidate distances in its search precision, for resolving ties.
pub enum SparseTieDistances<'a> {
    /// Euclidean, Chebyshev, and cosine rank distances in float32.
    Float32(&'a [f32]),
    /// Manhattan ranks the accumulated values before rounding to float32.
    Float64(&'a [f64]),
}

/// Return selected Euclidean, Manhattan, Chebyshev, or cosine ranks from CSR rows.
///
/// Manhattan queries round each difference in float32 and sum the absolute
/// differences in float64. Narrow queries use the leading training columns
/// and rank in float32. Cosine ranks matching-width queries using normalized
/// stored values. A narrower query uses the cosine of the shared prefix and
/// requires canonical CSR buffers without repeated entries.
pub fn rank_sparse_feature_queries(
    training: CsrFeatures<'_>,
    queries: CsrFeatures<'_>,
    k: usize,
    metric: DistanceMetric,
    mut rank_ties: impl FnMut(SparseTieDistances<'_>, usize) -> Vec<usize>,
) -> Result<Vec<Neighbor>, SparseQueryRankError> {
    use SparseQueryRankError as Error;
    validate_feature_metric(metric).map_err(Error::Training)?;
    if queries.shape.1 > training.shape.1 {
        return Err(Error::FeatureCount {
            training: training.shape.1,
            queries: queries.shape.1,
        });
    }
    let width = (queries.shape.1 < training.shape.1).then_some(queries.shape.1);
    let normalize = metric == DistanceMetric::Cosine && width.is_none();
    validate_csr_features(training, Some(k), normalize).map_err(Error::Training)?;
    validate_csr_features(queries, None, normalize).map_err(Error::Queries)?;
    let normalized_training =
        normalize.then(|| normalize_sparse_rows(training.indptr, training.data));
    let normalized_queries = normalize.then(|| normalize_sparse_rows(queries.indptr, queries.data));
    let training = CsrFeatures {
        data: normalized_training.as_deref().unwrap_or(training.data),
        ..training
    };
    let queries = CsrFeatures {
        data: normalized_queries.as_deref().unwrap_or(queries.data),
        ..queries
    };
    let prefix_cosine = metric == DistanceMetric::Cosine && width.is_some();
    let mut ranks = Vec::new();
    let mut ranking = SparseFeatureRanking::new(training, metric);
    let mut euclidean_index = if metric == DistanceMetric::Euclidean && width.is_none() {
        SparseEuclideanIndex::new(training, queries.shape.0)
    } else {
        None
    };
    for query in 0..queries.shape.0 {
        if let Some(index) = &mut euclidean_index {
            index.fill(queries.row(query), &mut ranking.candidates);
        } else {
            ranking.fill(queries.row(query), width);
        }
        ranking
            .select(query, k, &mut rank_ties)
            .map_err(Error::Ranking)?;
        for &neighbor in &ranking.candidates[..k] {
            if !neighbor.distance.is_finite() {
                return Err(Error::Ranking(
                    DenseGraphTransformError::NonFiniteDistance {
                        query,
                        training: neighbor.index,
                        distance: neighbor.distance,
                    },
                ));
            }
            // Prefix cosine can round slightly below zero. Keep those values
            // in the sort order, then pass zero distance to graph smoothing.
            ranks.push(if prefix_cosine && neighbor.distance < 0.0 {
                Neighbor {
                    distance: 0.0,
                    ..neighbor
                }
            } else {
                neighbor
            });
        }
    }
    Ok(ranks)
}

struct SparseFeatureRanking<'a> {
    training: CsrFeatures<'a>,
    metric: DistanceMetric,
    candidates: Vec<Neighbor>,
    wide_distances: Vec<f64>,
    tie_ranking: QueryTieRanking,
}

impl<'a> SparseFeatureRanking<'a> {
    fn new(training: CsrFeatures<'a>, metric: DistanceMetric) -> Self {
        Self {
            training,
            metric,
            candidates: Vec::with_capacity(training.shape.0),
            wide_distances: Vec::new(),
            tie_ranking: QueryTieRanking::default(),
        }
    }

    fn fill(&mut self, row: CsrRow<'_>, width: Option<usize>) {
        self.candidates.clear();
        self.wide_distances.clear();
        let n_features = width.unwrap_or(self.training.shape.1);
        for index in 0..self.training.shape.0 {
            let mut candidate = self.training.row(index);
            if let Some(width) = width {
                candidate = candidate.prefix(width);
            }
            let distance = match self.metric {
                DistanceMetric::Euclidean if width.is_some() => {
                    euclidean_distance_pairs(merged_features(row, candidate)) as f32
                }
                DistanceMetric::Euclidean => canonical_euclidean_distance(row, candidate),
                DistanceMetric::Manhattan if width.is_none() => {
                    let distance = manhattan_distance_pairs_wide(merged_features(row, candidate));
                    self.wide_distances.push(distance);
                    distance as f32
                }
                DistanceMetric::Cosine if width.is_some() => {
                    cosine_distance_pairs(merged_features(row, candidate))
                }
                DistanceMetric::Cosine => sparse_cosine_distance(row, candidate),
                _ => sparse_distance(row, candidate, self.metric, n_features),
            };
            self.candidates.push(Neighbor { index, distance });
        }
    }

    fn select(
        &mut self,
        query: usize,
        k: usize,
        rank_ties: impl FnOnce(SparseTieDistances<'_>, usize) -> Vec<usize>,
    ) -> Result<(), DenseGraphTransformError> {
        let wide = &self.wide_distances;
        self.tie_ranking
            .select(query, &mut self.candidates, k, false, |distances, count| {
                // Increasing float64 distances remain ordered after rounding.
                // The full-precision row is needed only if that rounding produces a tie.
                let distances = if wide.is_empty() {
                    SparseTieDistances::Float32(distances)
                } else {
                    SparseTieDistances::Float64(wide)
                };
                rank_ties(distances, count)
            })
    }

    fn cut(&self, neighbor: &Neighbor, cutoff: f64) -> bool {
        if self.metric == DistanceMetric::Manhattan {
            self.wide_distances[neighbor.index] >= cutoff
        } else {
            neighbor.distance >= cutoff as f32
        }
    }
}

fn validate_feature_metric(metric: DistanceMetric) -> Result<(), SparseKnnError> {
    if metric == DistanceMetric::Correlation {
        Err(SparseKnnError::UnsupportedFeatureMetric(metric))
    } else {
        Ok(())
    }
}

/// Fit a square fuzzy graph from a sorted CSR feature matrix.
///
/// The input and `k` follow [`exhaustive_csr_knn`]. The graph is canonical CSR
/// with strictly ascending columns per row and no diagonal entries. Graph
/// fitting requires finite selected distances and a self neighbor in every
/// selected row. A tie at zero distance can exclude the self neighbor and
/// produce [`GraphError::MissingSelf`].
pub fn fit_sparse_graph(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: impl Into<SparseDistanceMetric>,
) -> Result<CsrGraph, SparseGraphFitError> {
    fit_sparse_graph_with_options(indptr, indices, data, n_rows, n_features, k, metric, None)
}

/// Fit a sorted CSR feature graph with optional fuzzy graph settings.
///
/// The inputs follow [`fit_sparse_graph`]. The optional settings follow
/// [`crate::fit::fit_dense_graph_with_options`].
#[allow(clippy::too_many_arguments)]
pub fn fit_sparse_graph_with_options(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_rows: usize,
    n_features: usize,
    k: usize,
    metric: impl Into<SparseDistanceMetric>,
    options: Option<GraphOptions>,
) -> Result<CsrGraph, SparseGraphFitError> {
    let neighbors = exhaustive_csr_knn(indptr, indices, data, n_rows, n_features, k, metric)
        .map_err(SparseGraphFitError::Knn)?;
    build_fitted_graph(&neighbors, options).map_err(SparseGraphFitError::Graph)
}

/// Fit canonical CSR from Euclidean feature ranks, counting all ordered
/// pairs at or above the cutoff. Repeated feature entries are coalesced in
/// float64 before measuring distances. Zero-distance ties may exclude self.
pub fn fit_sparse_euclidean_graph_with_cutoff(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    shape: (usize, usize),
    k: usize,
    cutoff: f32,
    options: GraphOptions,
) -> Result<(CsrGraph, usize), SparseGraphFitError> {
    let input = CsrFeatures {
        indptr,
        indices,
        data,
        shape,
    };
    let (graph, _, removed) =
        fit_index_ranked_sparse_graph(input, k, DistanceMetric::Euclidean, cutoff, options)?;
    Ok((graph, removed))
}

/// Rebuild an uncut Euclidean graph and retain the neighbor columns of the
/// appended rows for layout initialization. Ranks break ties by row index.
pub fn fit_sparse_euclidean_update_graph(
    input: CsrFeatures<'_>,
    n_existing: usize,
    k: usize,
    options: GraphOptions,
) -> Result<(CsrGraph, Vec<usize>), SparseGraphFitError> {
    fit_sparse_update_graph(input, n_existing, k, DistanceMetric::Euclidean, options)
}

/// Rebuild an uncut feature graph and return the appended rows' neighbor columns.
/// Manhattan orders its float64 sums before rounding the selected distances.
/// Training ties in the search precision follow ascending row index.
pub fn fit_sparse_update_graph(
    input: CsrFeatures<'_>,
    n_existing: usize,
    k: usize,
    metric: DistanceMetric,
    options: GraphOptions,
) -> Result<(CsrGraph, Vec<usize>), SparseGraphFitError> {
    validate_feature_metric(metric).map_err(SparseGraphFitError::Knn)?;
    let (graph, neighbors, _) = if metric == DistanceMetric::Manhattan {
        fit_sparse_graph_with_ranker(
            input,
            k,
            metric,
            f64::INFINITY,
            options,
            rank_manhattan_ties,
        )?
    } else {
        fit_index_ranked_sparse_graph(input, k, metric, f32::INFINITY, options)?
    };
    let appended_ranks = neighbors
        .into_iter()
        .skip(n_existing)
        .flatten()
        .map(|rank| rank.index)
        .collect();
    Ok((graph, appended_ranks))
}

type RankedSparseGraph = (CsrGraph, Vec<Vec<Neighbor>>, usize);

fn fit_index_ranked_sparse_graph(
    input: CsrFeatures<'_>,
    k: usize,
    metric: DistanceMetric,
    cutoff: f32,
    options: GraphOptions,
) -> Result<RankedSparseGraph, SparseGraphFitError> {
    let allow_duplicates = metric == DistanceMetric::Euclidean;
    let (neighbors, removed) =
        csr_knn_counting_cutoff(input, k, metric.into(), cutoff, allow_duplicates)
            .map_err(SparseGraphFitError::Knn)?;
    let graph = build_graph_with_cutoff(&neighbors, k, cutoff, options)
        .map_err(SparseGraphFitError::Graph)?;
    Ok((graph, neighbors, removed))
}

/// Fit from sorted CSR, keeping Manhattan distances wide through selection
/// and the inclusive fit cutoff. Cosine accepts repeated columns and breaks
/// fit ties by training index. Return the count of cut ordered pairs.
pub fn fit_sparse_feature_graph_with_cutoff(
    input: CsrFeatures<'_>,
    k: usize,
    metric: DistanceMetric,
    cutoff: f64,
    options: GraphOptions,
    rank_ties: impl FnMut(SparseTieDistances<'_>, usize) -> Vec<usize>,
) -> Result<(CsrGraph, usize), SparseGraphFitError> {
    validate_feature_metric(metric).map_err(SparseGraphFitError::Knn)?;
    if metric == DistanceMetric::Cosine {
        let (graph, _, removed) =
            fit_index_ranked_sparse_graph(input, k, metric, cutoff as f32, options)?;
        return Ok((graph, removed));
    }
    let (graph, _, removed) =
        fit_sparse_graph_with_ranker(input, k, metric, cutoff, options, rank_ties)?;
    Ok((graph, removed))
}

fn fit_sparse_graph_with_ranker(
    input: CsrFeatures<'_>,
    k: usize,
    metric: DistanceMetric,
    cutoff: f64,
    options: GraphOptions,
    mut rank_ties: impl FnMut(SparseTieDistances<'_>, usize) -> Vec<usize>,
) -> Result<RankedSparseGraph, SparseGraphFitError> {
    use SparseGraphFitError as Error;
    validate_csr_features(input, Some(k), false).map_err(Error::Knn)?;
    let mut euclidean_index = if metric == DistanceMetric::Euclidean {
        SparseEuclideanIndex::new(input, input.shape.0)
    } else {
        None
    };
    let euclidean_pairs = if metric == DistanceMetric::Euclidean && euclidean_index.is_none() {
        cache_euclidean_pairs(input)
    } else {
        None
    };
    let mut ranking = SparseFeatureRanking::new(input, metric);
    let mut neighbors = Vec::with_capacity(input.shape.0);
    let mut removed = 0;
    for row in 0..input.shape.0 {
        if let Some(index) = &mut euclidean_index {
            index.fill(input.row(row), &mut ranking.candidates);
        } else if let Some(pairs) = &euclidean_pairs {
            ranking.candidates.clear();
            fill_euclidean_candidates(&mut ranking.candidates, row, input.shape.0, pairs);
        } else {
            ranking.fill(input.row(row), None);
        }
        removed += ranking
            .candidates
            .iter()
            .filter(|neighbor| ranking.cut(neighbor, cutoff))
            .count();
        ranking
            .select(row, k, &mut rank_ties)
            .map_err(Error::Ranking)?;
        neighbors.push(
            ranking.candidates[..k]
                .iter()
                .map(|&neighbor| Neighbor {
                    distance: if ranking.cut(&neighbor, cutoff) {
                        f32::INFINITY
                    } else {
                        neighbor.distance
                    },
                    ..neighbor
                })
                .collect(),
        );
    }
    let graph =
        build_graph_with_cutoff(&neighbors, k, f32::INFINITY, options).map_err(Error::Graph)?;
    Ok((graph, neighbors, removed))
}

fn cache_euclidean_pairs(input: CsrFeatures<'_>) -> Option<Vec<f32>> {
    // Keep the exact row-by-row search for training sets whose pair table exceeds 8 MiB.
    const MAX_CACHED_PAIRS: usize = (8 * 1024 * 1024) / std::mem::size_of::<f32>();
    let rows = input.shape.0;
    let pair_count = rows.checked_mul(rows.saturating_sub(1))? / 2;
    if pair_count > MAX_CACHED_PAIRS {
        return None;
    }

    let mut pairs = Vec::new();
    pairs.try_reserve_exact(pair_count).ok()?;
    // Save the lower triangle in row order. For finite inputs each squared
    // difference has the same bits after exchanging its two operands.
    for row in 1..rows {
        let left = input.row(row);
        for column in 0..row {
            pairs.push(canonical_euclidean_distance(left, input.row(column)));
        }
    }
    Some(pairs)
}

fn fill_euclidean_candidates(
    candidates: &mut Vec<Neighbor>,
    row: usize,
    n_rows: usize,
    pairs: &[f32],
) {
    let row_start = row * row.saturating_sub(1) / 2;
    candidates.extend(
        pairs[row_start..row_start + row]
            .iter()
            .enumerate()
            .map(|(index, &distance)| Neighbor { index, distance }),
    );
    candidates.push(Neighbor {
        index: row,
        distance: 0.0,
    });
    candidates.extend((row + 1..n_rows).map(|index| Neighbor {
        index,
        distance: pairs[index * (index - 1) / 2 + row],
    }));
}

fn rank_manhattan_ties(distances: SparseTieDistances<'_>, k: usize) -> Vec<usize> {
    let SparseTieDistances::Float64(distances) = distances else {
        unreachable!("Manhattan update ranks use float64 distances")
    };
    let mut columns: Vec<_> = (0..distances.len()).collect();
    let compare = |&left: &usize, &right: &usize| {
        distances[left]
            .total_cmp(&distances[right])
            .then(left.cmp(&right))
    };
    if k < columns.len() {
        columns.select_nth_unstable_by(k, compare);
        columns.truncate(k);
    }
    columns.sort_unstable_by(compare);
    columns
}

#[derive(Clone, Copy)]
struct CsrRow<'a> {
    columns: &'a [usize],
    values: &'a [f32],
}

impl<'a> CsrRow<'a> {
    fn prefix(self, width: usize) -> Self {
        let end = self.columns.partition_point(|&column| column < width);
        Self {
            columns: &self.columns[..end],
            values: &self.values[..end],
        }
    }

    fn entries(self) -> impl Iterator<Item = (usize, f32)> + 'a {
        self.columns
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    fn wide_entries(self) -> impl Iterator<Item = (usize, f64)> + 'a {
        let mut entries = self.entries().peekable();
        std::iter::from_fn(move || {
            let (column, value) = entries.next()?;
            let mut sum = f64::from(value);
            while let Some((_, duplicate)) = entries.next_if(|&(next, _)| next == column) {
                sum += f64::from(duplicate);
            }
            Some((column, sum))
        })
    }
}

fn normalize_sparse_rows(indptr: &[usize], data: &[f32]) -> Vec<f32> {
    let mut normalized = data.to_vec();
    for offsets in indptr.windows(2) {
        let row = &mut normalized[offsets[0]..offsets[1]];
        let squared_norm = row
            .iter()
            .fold(0.0_f64, |sum, &value| sum + f64::from(value * value));
        if squared_norm > 0.0 {
            let norm = squared_norm.sqrt();
            for value in row {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }
    normalized
}

fn sparse_cosine_distance(left: CsrRow<'_>, right: CsrRow<'_>) -> f32 {
    let (mut left_position, mut right_position) = (0, 0);
    let mut dot = 0.0_f32;
    while let (Some(left_column), Some(right_column)) = (
        left.columns.get(left_position),
        right.columns.get(right_position),
    ) {
        match left_column.cmp(right_column) {
            Ordering::Less => left_position += 1,
            Ordering::Greater => right_position += 1,
            Ordering::Equal => {
                let mut right_end = right_position + 1;
                while right.columns.get(right_end) == Some(right_column) {
                    right_end += 1;
                }
                while left.columns.get(left_position) == Some(left_column) {
                    for &right_value in &right.values[right_position..right_end] {
                        dot += left.values[left_position] * right_value;
                    }
                    left_position += 1;
                }
                right_position = right_end;
            }
        }
    }
    (1.0 - dot).clamp(0.0, 2.0)
}

fn merged_features<'a>(
    left: CsrRow<'a>,
    right: CsrRow<'a>,
) -> impl Iterator<Item = (f32, f32)> + 'a {
    merged_entries(left.entries(), right.entries())
}

fn merged_entries<T: Default + Copy>(
    left: impl Iterator<Item = (usize, T)>,
    right: impl Iterator<Item = (usize, T)>,
) -> impl Iterator<Item = (T, T)> {
    let mut left = left.peekable();
    let mut right = right.peekable();
    std::iter::from_fn(move || {
        // Validated feature indices are always less than usize::MAX.
        let left_column = left.peek().map_or(usize::MAX, |&(column, _)| column);
        let right_column = right.peek().map_or(usize::MAX, |&(column, _)| column);
        Some(match left_column.cmp(&right_column) {
            Ordering::Less => (left.next()?.1, T::default()),
            Ordering::Greater => (T::default(), right.next()?.1),
            Ordering::Equal if left_column == usize::MAX => return None,
            Ordering::Equal => (left.next()?.1, right.next()?.1),
        })
    })
}

// Both rows have strictly increasing columns. Preserve the original column
// order for the f64 sum and round it to f32 before taking the square root.
fn canonical_euclidean_distance(left: CsrRow<'_>, right: CsrRow<'_>) -> f32 {
    let (mut left_position, mut right_position) = (0, 0);
    let mut sum = 0.0_f64;
    while left_position < left.columns.len() && right_position < right.columns.len() {
        let difference = match left.columns[left_position].cmp(&right.columns[right_position]) {
            Ordering::Less => {
                let value = f64::from(left.values[left_position]);
                left_position += 1;
                value
            }
            Ordering::Greater => {
                let value = -f64::from(right.values[right_position]);
                right_position += 1;
                value
            }
            Ordering::Equal => {
                let value =
                    f64::from(left.values[left_position]) - f64::from(right.values[right_position]);
                left_position += 1;
                right_position += 1;
                value
            }
        };
        sum += difference * difference;
    }
    for &value in left.values[left_position..]
        .iter()
        .chain(&right.values[right_position..])
    {
        let wide = f64::from(value);
        sum += wide * wide;
    }
    (sum as f32).sqrt()
}

fn sparse_distance(
    left: CsrRow<'_>,
    right: CsrRow<'_>,
    metric: DistanceMetric,
    n_features: usize,
) -> f32 {
    if metric == DistanceMetric::Manhattan {
        return manhattan_distance_pairs(merged_features(left, right));
    }
    match metric {
        DistanceMetric::Euclidean => {
            let squared_sum = merged_entries(left.wide_entries(), right.wide_entries()).fold(
                0.0,
                |sum, (left, right)| {
                    let difference = left - right;
                    sum + difference * difference
                },
            );
            (squared_sum as f32).sqrt()
        }
        DistanceMetric::Chebyshev => merged_features(left, right)
            .fold(0.0_f64, |maximum, (left, right)| {
                maximum.max((f64::from(left) - f64::from(right)).abs())
            }) as f32,
        DistanceMetric::Canberra => canberra_distance_pairs(merged_features(left, right)) as f32,
        DistanceMetric::BrayCurtis => {
            bray_curtis_distance_pairs(merged_features(left, right)) as f32
        }
        DistanceMetric::Manhattan | DistanceMetric::Cosine | DistanceMetric::Correlation => {
            unreachable!("CSR metric was dispatched before merging coordinates")
        }
        metric => boolean_distance_pairs(metric, merged_features(left, right), n_features),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_euclidean_pairs_preserve_distances_ties_and_cutoff_counts() {
        let input = CsrFeatures {
            indptr: &[0, 0, 3, 5, 7, 9],
            indices: &[0, 1, 4, 1, 4, 0, 6, 2, 6],
            data: &[
                -0.0,
                f32::from_bits(1),
                3.0,
                f32::from_bits(1),
                -3.0,
                1.0,
                5.0,
                -0.0,
                -5.0,
            ],
            shape: (5, 7),
        };
        let metric = DistanceMetric::Euclidean;
        let pairs = cache_euclidean_pairs(input).unwrap();
        let mut candidates = Vec::new();
        let distance_rows: Vec<Vec<_>> = (0..input.shape.0)
            .map(|row| {
                (0..input.shape.0)
                    .map(|column| sparse_distance(input.row(row), input.row(column), metric, 7))
                    .collect()
            })
            .collect();
        for (row, expected) in distance_rows.iter().enumerate() {
            candidates.clear();
            fill_euclidean_candidates(&mut candidates, row, input.shape.0, &pairs);
            for (column, candidate) in candidates.iter().enumerate() {
                assert_eq!(candidate.index, column);
                assert_eq!(candidate.distance.to_bits(), expected[column].to_bits());
            }
        }

        for cutoff in [3.0, f32::INFINITY] {
            let options = GraphOptions::default();
            let k = 3;
            let ranks = collect_neighbors(input.shape.0, k, |row, candidates| {
                candidates.extend(
                    distance_rows[row]
                        .iter()
                        .enumerate()
                        .map(|(index, &distance)| Neighbor { index, distance }),
                );
            });
            let expected_removed = distance_rows
                .iter()
                .flatten()
                .filter(|&&distance| distance >= cutoff)
                .count();
            let expected_graph = build_graph_with_cutoff(&ranks, k, cutoff, options).unwrap();
            let (graph, removed) = fit_sparse_euclidean_graph_with_cutoff(
                input.indptr,
                input.indices,
                input.data,
                input.shape,
                k,
                cutoff,
                options,
            )
            .unwrap();
            assert_eq!(removed, expected_removed);
            assert_eq!(graph, expected_graph);
            let mut tied_rows =
                [&distance_rows[0], &distance_rows[3], &distance_rows[4]].into_iter();
            let (graph, removed) = fit_sparse_feature_graph_with_cutoff(
                input,
                k,
                metric,
                f64::from(cutoff),
                options,
                |distances, count| {
                    let SparseTieDistances::Float32(distances) = distances else {
                        panic!("expected Euclidean distances in float32");
                    };
                    assert_eq!(distances, tied_rows.next().unwrap());
                    let mut columns: Vec<_> = (0..distances.len()).collect();
                    columns.sort_by(|&left, &right| distances[left].total_cmp(&distances[right]));
                    columns.truncate(count);
                    columns
                },
            )
            .unwrap();
            assert!(tied_rows.next().is_none());
            assert_eq!(removed, expected_removed);
            assert_eq!(graph, expected_graph);
        }
    }

    #[test]
    fn index_ranked_euclidean_fit_keeps_wide_duplicate_reduction() {
        let input = CsrFeatures {
            indptr: &[0, 3, 3, 4],
            indices: &[0, 0, 0, 0],
            data: &[16_777_216.0, 1.0, -16_777_216.0, 3.0],
            shape: (3, 1),
        };
        let (ranks, removed) =
            csr_knn_counting_cutoff(input, 3, DistanceMetric::Euclidean.into(), 2.0, true).unwrap();
        assert_eq!(
            ranks[0],
            [
                Neighbor {
                    index: 0,
                    distance: 0.0
                },
                Neighbor {
                    index: 1,
                    distance: 1.0
                },
                Neighbor {
                    index: 2,
                    distance: 2.0
                },
            ]
        );
        assert_eq!(removed, 4);
    }

    #[test]
    fn cached_euclidean_pairs_keep_overflow_errors() {
        let input = CsrFeatures {
            indptr: &[0, 1, 2, 3],
            indices: &[0, 0, 0],
            data: &[f32::MAX, 0.0, -f32::MAX],
            shape: (3, 1),
        };
        let error = fit_sparse_feature_graph_with_cutoff(
            input,
            2,
            DistanceMetric::Euclidean,
            f64::INFINITY,
            GraphOptions::default(),
            |distances, _| {
                let SparseTieDistances::Float32(distances) = distances else {
                    panic!("expected Euclidean distances in float32");
                };
                assert_eq!(distances, &[0.0, f32::INFINITY, f32::INFINITY]);
                vec![0, 1]
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            SparseGraphFitError::Ranking(DenseGraphTransformError::NonFiniteDistance {
                query: 0,
                training: 1,
                distance: f32::INFINITY,
            })
        );
    }

    #[test]
    fn euclidean_pair_cache_falls_back_beyond_its_memory_budget() {
        for rows in [2049, usize::MAX] {
            let input = CsrFeatures {
                indptr: &[],
                indices: &[],
                data: &[],
                shape: (rows, 0),
            };
            assert!(cache_euclidean_pairs(input).is_none());
        }
    }
}
