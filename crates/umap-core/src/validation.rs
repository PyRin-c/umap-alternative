//! Trustworthiness scores from the ranks of the same samples in two spaces.

use std::collections::HashMap;
use std::fmt;

use crate::neighbors::{euclidean_distance_wide, manhattan_distance_pairs_wide};

/// Source pair metrics used with ranked embedded neighbors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustworthinessMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl TrustworthinessMetric {
    pub fn distance_f32(self, left: &[f32], right: &[f32]) -> f64 {
        match self {
            Self::Euclidean => euclidean_distance_wide(left, right),
            Self::Manhattan => {
                manhattan_distance_pairs_wide(left.iter().copied().zip(right.iter().copied()))
            }
            Self::Chebyshev => self.reduce(
                left.iter()
                    .zip(right)
                    .map(|(&a, &b)| f64::from((a - b).abs())),
            ),
        }
    }

    pub fn distance_f64(self, left: &[f64], right: &[f64]) -> f64 {
        self.reduce(left.iter().zip(right).map(|(&a, &b)| {
            let delta = a - b;
            match self {
                Self::Euclidean => delta * delta,
                _ => delta.abs(),
            }
        }))
    }

    pub fn distance_i64(self, left: &[i64], right: &[i64]) -> f64 {
        self.reduce(left.iter().zip(right).map(|(&a, &b)| {
            let delta = a.wrapping_sub(b);
            match self {
                Self::Euclidean => delta.wrapping_mul(delta) as f64,
                _ => delta.wrapping_abs() as f64,
            }
        }))
    }

    pub fn distance_u64(self, left: &[u64], right: &[u64]) -> f64 {
        self.reduce(left.iter().zip(right).map(|(&a, &b)| {
            let delta = a.wrapping_sub(b);
            match self {
                Self::Euclidean => delta.wrapping_mul(delta) as f64,
                _ => delta as f64,
            }
        }))
    }

    fn reduce(self, terms: impl Iterator<Item = f64>) -> f64 {
        match self {
            Self::Euclidean => terms.fold(0.0, |sum, term| sum + term).sqrt(),
            Self::Manhattan => terms.fold(0.0, |sum, term| sum + term),
            Self::Chebyshev => terms.fold(0.0, f64::max),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustworthinessError {
    Shape,
    MissingRank,
    ZeroDenominator,
}

impl fmt::Display for TrustworthinessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Shape => "incompatible trustworthiness input shapes",
            Self::MissingRank => "embedded neighbor was not ranked in the source row",
            Self::ZeroDenominator => "division by zero when normalizing trustworthiness",
        })
    }
}

impl std::error::Error for TrustworthinessError {}

/// Accumulates trustworthiness for every neighborhood size from one through `max_k`.
pub struct Trustworthiness {
    n_samples: usize,
    max_k: usize,
    rows: usize,
    penalty_changes: Vec<(i128, i128)>,
    order: Vec<usize>,
    ranks: Vec<usize>,
    ranked_ids: HashMap<i64, usize>,
}

impl Trustworthiness {
    pub fn new(n_samples: usize, max_k: usize) -> Result<Self, TrustworthinessError> {
        let size = max_k.checked_add(2).ok_or(TrustworthinessError::Shape)?;
        Ok(Self {
            n_samples,
            max_k,
            rows: 0,
            penalty_changes: vec![(0, 0); size],
            order: Vec::new(),
            ranks: Vec::new(),
            ranked_ids: HashMap::new(),
        })
    }

    /// Add one source row of ordered identifiers and its ranked embedded neighbors.
    /// The source rank is the zero-based position of the first matching identifier.
    pub fn add_ranked_row(
        &mut self,
        source: &[i64],
        embedded: &[i64],
    ) -> Result<(), TrustworthinessError> {
        let embedded = self.check_row(embedded)?;
        self.ranked_ids.clear();
        for (position, &id) in source.iter().enumerate() {
            self.ranked_ids.entry(id).or_insert(position);
        }
        if embedded.iter().any(|id| !self.ranked_ids.contains_key(id)) {
            return Err(TrustworthinessError::MissingRank);
        }
        for (position, id) in embedded.iter().enumerate() {
            self.add_penalty(position, self.ranked_ids[id]);
        }
        self.rows += 1;
        Ok(())
    }

    /// Add one source distance row in sample index order and its ranked embedded neighbors.
    pub fn add_distance_row(
        &mut self,
        source: &[f64],
        embedded: &[usize],
    ) -> Result<(), TrustworthinessError> {
        let embedded = self.check_row(embedded)?;
        if source.len() != self.n_samples {
            return Err(TrustworthinessError::Shape);
        }
        if embedded.iter().any(|&id| id >= self.n_samples) {
            return Err(TrustworthinessError::MissingRank);
        }
        self.order.clear();
        self.order.extend(0..self.n_samples);
        sort_source_row(source, &mut self.order);
        self.ranks.resize(self.n_samples, 0);
        for (position, &id) in self.order.iter().enumerate() {
            self.ranks[id] = position;
        }
        for (position, &id) in embedded.iter().enumerate() {
            self.add_penalty(position, self.ranks[id]);
        }
        self.rows += 1;
        Ok(())
    }

    /// Finish the curve. The zero-neighbor entry is set by the caller's convention.
    pub fn finish(self, zero_neighbors: f64) -> Result<Vec<f64>, TrustworthinessError> {
        if self.rows != self.n_samples {
            return Err(TrustworthinessError::Shape);
        }
        let n_samples = self.n_samples as f64;
        let (mut rank_sum, mut rank_count) = (0, 0);
        let mut scores = vec![zero_neighbors];
        for (neighbors, (sum_change, count_change)) in self
            .penalty_changes
            .into_iter()
            .enumerate()
            .skip(1)
            .take(self.max_k)
        {
            rank_sum += sum_change;
            rank_count += count_change;
            let penalty = rank_sum - rank_count * neighbors as i128;
            let k = neighbors as f64;
            let denominator = n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0);
            if denominator == 0.0 {
                return Err(TrustworthinessError::ZeroDenominator);
            }
            scores.push(1.0 - (2.0 / denominator) * penalty as f64);
        }
        Ok(scores)
    }

    fn check_row<'a, T>(&self, embedded: &'a [T]) -> Result<&'a [T], TrustworthinessError> {
        if self.rows == self.n_samples || embedded.len() < self.max_k {
            return Err(TrustworthinessError::Shape);
        }
        Ok(&embedded[..self.max_k])
    }

    fn add_penalty(&mut self, query_position: usize, source_rank: usize) {
        let start = query_position + 1;
        if source_rank <= start {
            return;
        }
        // A ranked neighbor contributes (source_rank - k) for this entire range.
        let stop = source_rank.min(self.max_k + 1);
        self.penalty_changes[start].0 += source_rank as i128;
        self.penalty_changes[start].1 += 1;
        self.penalty_changes[stop].0 -= source_rank as i128;
        self.penalty_changes[stop].1 -= 1;
    }
}

/// Score a dense source matrix using an explicit pair metric and embedded neighbor indices.
pub fn trustworthiness_from_dense<T>(
    source: &[T],
    n_features: usize,
    embedded: &[Vec<usize>],
    max_k: usize,
    metric: impl Fn(&[T], &[T]) -> f64,
) -> Result<Vec<f64>, TrustworthinessError> {
    if embedded.len().checked_mul(n_features) != Some(source.len()) {
        return Err(TrustworthinessError::Shape);
    }
    let mut scores = Trustworthiness::new(embedded.len(), max_k)?;
    let mut distances = Vec::with_capacity(embedded.len());
    let source_row = |row| &source[row * n_features..(row + 1) * n_features];
    for (row, neighbors) in embedded.iter().enumerate() {
        let point = source_row(row);
        distances.clear();
        distances.extend((0..embedded.len()).map(|other| metric(point, source_row(other))));
        scores.add_distance_row(&distances, neighbors)?;
    }
    scores.finish(1.0)
}

fn sort_source_row(distances: &[f64], mut order: &mut [usize]) {
    let less = |a: usize, b: usize| {
        distances[a] < distances[b] || (!distances[a].is_nan() && distances[b].is_nan())
    };
    // The reference ranks ties via insertion through size 15 and a median pivot
    // above it. Keep equal values in the observed partition order.
    while order.len() > 15 {
        let last = order.len() - 1;
        let mid = last / 2;
        for (left, right) in [(0, mid), (mid, last), (0, mid)] {
            if less(order[right], order[left]) {
                order.swap(left, right);
            }
        }
        order.swap(mid, last);
        let pivot = order[last];
        let (mut left, mut right) = (0, last);
        while left < right {
            while less(order[left], pivot) {
                left += 1;
            }
            while left < right && less(pivot, order[right - 1]) {
                right -= 1;
            }
            if left + 1 >= right {
                break;
            }
            order.swap(left, right - 1);
            left += 1;
            right -= 1;
        }
        order.swap(left, last);
        let (prefix, suffix) = order.split_at_mut(left);
        let suffix = &mut suffix[1..];
        // Recurse on the smaller side to bound stack depth for uneven partitions.
        if prefix.len() < suffix.len() {
            sort_source_row(distances, prefix);
            order = suffix;
        } else {
            sort_source_row(distances, suffix);
            order = prefix;
        }
    }
    for index in 1..order.len() {
        let mut position = index;
        while position > 0 && less(order[position], order[position - 1]) {
            order.swap(position, position - 1);
            position -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_change_partition_order_at_sixteen_samples() {
        let mut order: Vec<_> = (0..16).collect();
        sort_source_row(&[0.0; 16], &mut order[..15]);
        assert!(order.iter().copied().eq(0..16));
        sort_source_row(&[0.0; 16], &mut order);
        assert!(order.iter().copied().eq((0..15).rev().chain([15])));
    }

    #[test]
    fn rank_intervals_match_the_direct_penalty() {
        let source = [9, -3, 2, 12, 5, 17, 20];
        let embedded = [20, 17, 9, 5, 2, -3, 12];
        let mut scores = Trustworthiness::new(7, 6).unwrap();
        for _ in 0..7 {
            scores.add_ranked_row(&source, &embedded).unwrap();
        }
        let scores = scores.finish(0.0).unwrap();
        for (k, &score) in scores.iter().enumerate().skip(1) {
            let penalty: usize = embedded[..k]
                .iter()
                .map(|id| {
                    source
                        .iter()
                        .position(|position| position == id)
                        .unwrap()
                        .saturating_sub(k)
                })
                .sum();
            let k = k as f64;
            let expected = 1.0 - (2.0 / (7.0 * k * (14.0 - 3.0 * k - 1.0))) * (penalty * 7) as f64;
            assert_eq!(score.to_bits(), expected.to_bits());
        }
    }
}
