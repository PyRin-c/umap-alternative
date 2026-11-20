//! Neighbors of dense rows of packed bytes.

use crate::fit::{DenseGraphFitError, build_graph_with_cutoff};
use crate::fuzzy_graph::{CsrGraph, GraphOptions};
use crate::neighbors::{KnnError, Neighbor, collect_neighbors};
use crate::transform::{DenseGraphTransformError, QueryTieRanking};

/// A distance over the individual bits in a packed byte row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedBitMetric {
    /// Count the set bits in the exclusive-or of the rows.
    Hamming,
    /// Negative log of the fraction of shared bits in the union.
    Jaccard,
}

impl PackedBitMetric {
    /// Measure two packed rows. When widths differ, compare their shared prefix.
    pub fn distance(self, left: &[u8], right: &[u8]) -> f32 {
        let pairs = left.iter().zip(right);
        match self {
            Self::Hamming => pairs.map(|(a, b)| (a ^ b).count_ones() as u64).sum::<u64>() as f32,
            Self::Jaccard => {
                let (mut shared, mut union) = (0_u64, 0_u64);
                for (a, b) in pairs {
                    shared += u64::from((a & b).count_ones());
                    union += u64::from((a | b).count_ones());
                }
                if union == 0 {
                    0.0
                } else if shared == 0 {
                    f32::INFINITY
                } else {
                    -(shared as f32 / union as f32).ln()
                }
            }
        }
    }
}

/// A row-major matrix of packed bytes.
#[derive(Clone, Copy, Debug)]
pub struct PackedRows<'a> {
    /// Contiguous row data.
    pub data: &'a [u8],
    /// Number of rows and number of bytes per row.
    pub shape: (usize, usize),
}

impl PackedRows<'_> {
    fn validate(self) -> Result<(), KnnError> {
        let (n_rows, n_features) = self.shape;
        if n_features == 0 || n_rows.checked_mul(n_features) != Some(self.data.len()) {
            return Err(KnnError::InvalidShape {
                n_rows,
                n_features,
                values: self.data.len(),
            });
        }
        Ok(())
    }
}

/// Fit a fuzzy graph from packed rows, breaking distance ties by row index.
/// The cutoff is inclusive and the removed count covers all ordered pairs.
pub fn fit_packed_graph(
    training: PackedRows<'_>,
    k: usize,
    metric: PackedBitMetric,
    cutoff: f64,
    options: GraphOptions,
) -> Result<(CsrGraph, usize), DenseGraphFitError> {
    training.validate()?;
    let (n_rows, width) = training.shape;
    if !(1..=n_rows).contains(&k) {
        return Err(KnnError::InvalidK { k, n_rows }.into());
    }
    let mut removed = 0;
    let neighbors = collect_neighbors(n_rows, k, |row, candidates| {
        let query = &training.data[row * width..(row + 1) * width];
        for (index, candidate) in training.data.chunks_exact(width).enumerate() {
            let distance = metric.distance(query, candidate);
            let cut = f64::from(distance) >= cutoff;
            removed += usize::from(cut);
            let distance = if cut { f32::INFINITY } else { distance };
            candidates.push(Neighbor { index, distance });
        }
    });
    Ok((
        build_graph_with_cutoff(&neighbors, k, f32::INFINITY, options)?,
        removed,
    ))
}

/// Select compact held-out ranks against dense packed features.
///
/// Queries can compare a prefix of the fitted width. A disjoint Jaccard rank
/// carries positive infinity and can be passed to graph construction as missing.
pub fn rank_packed_queries(
    training: PackedRows<'_>,
    queries: PackedRows<'_>,
    k: usize,
    metric: PackedBitMetric,
    mut rank_ties: impl FnMut(&[f32], usize) -> Vec<usize>,
) -> Result<Vec<Neighbor>, DenseGraphTransformError> {
    use DenseGraphTransformError as Error;
    training.validate().map_err(Error::Training)?;
    queries.validate().map_err(Error::Queries)?;
    let (n_training_rows, width) = training.shape;
    let (n_queries, query_width) = queries.shape;
    if !(1..=n_training_rows).contains(&k) {
        return Err(Error::Training(KnnError::InvalidK {
            k,
            n_rows: n_training_rows,
        }));
    }
    if query_width > width {
        return Err(Error::Queries(KnnError::InvalidShape {
            n_rows: n_queries,
            n_features: width,
            values: queries.data.len(),
        }));
    }
    let mut ranks = Vec::new();
    let mut candidates = Vec::with_capacity(n_training_rows);
    let mut tie_ranking = QueryTieRanking::default();
    for (row, query) in queries.data.chunks_exact(query_width).enumerate() {
        candidates.clear();
        candidates.extend(
            training
                .data
                .chunks_exact(width)
                .enumerate()
                .map(|(index, data)| Neighbor {
                    index,
                    distance: metric.distance(query, data),
                }),
        );
        tie_ranking.select(row, &mut candidates, k, true, &mut rank_ties)?;
        ranks.extend_from_slice(&candidates[..k]);
    }
    Ok(ranks)
}
