use std::num::NonZeroUsize;

use super::{ApproximateSearchError, DenseRows, reduce_4_lanes, reduce_32_lanes};
use crate::neighbors::DistanceMetric;

/// Replay one dense Euclidean training tree from a caller projection scalar.
///
/// Obtain the scalar with [super::draw_projection_scalar]. Trees use a zero-based
/// ordinal within the forest. The output lists left leaves before right leaves,
/// preserving each leaf's in-place partition order. This stage supplies leaf
/// membership for training candidate offers.
pub fn training_projection_leaves(
    training: &[f32],
    n_rows: usize,
    n_features: usize,
    caller_scalar: u64,
    tree_ordinal: usize,
    leaf_size: NonZeroUsize,
    max_depth: u8,
) -> Result<Vec<Vec<usize>>, ApproximateSearchError> {
    let data = DenseRows::training(training, n_rows, n_features)?;
    Ok(training_leaves(
        data,
        caller_scalar,
        tree_ordinal,
        leaf_size.get(),
        max_depth,
        false,
    ))
}

pub(super) fn training_leaves(
    data: DenseRows<'_>,
    caller_scalar: u64,
    tree_ordinal: usize,
    leaf_size: usize,
    max_depth: u8,
    angular: bool,
) -> Vec<Vec<usize>> {
    let key = tree_key(caller_scalar, tree_ordinal);
    let mut leaves = Vec::new();
    collect_training_leaves(
        data,
        (0..data.n_rows).collect(),
        key,
        leaf_size,
        max_depth,
        angular,
        &mut leaves,
    );
    leaves
}

fn collect_training_leaves(
    data: DenseRows<'_>,
    mut rows: Vec<usize>,
    key: u64,
    leaf_size: usize,
    depth: u8,
    angular: bool,
    leaves: &mut Vec<Vec<usize>>,
) {
    if rows.len() <= leaf_size || depth == 0 {
        leaves.push(rows);
        return;
    }
    let (a, b) = projection_pivots(&rows, pivot_words(key));
    let (left, right) = (data.row(a), data.row(b));
    let (normal, offset) = if angular {
        (angular_normal(left, right).collect(), 0.0)
    } else {
        let normal: Vec<_> = left.iter().zip(right).map(|(a, b)| a - b).collect();
        let offset = accumulate_projection(data.n_features, 0.0, |i, accumulator| {
            normal[i].mul_add(-(left[i] + right[i]) * 0.5, accumulator)
        });
        (normal, offset)
    };
    let boundary = partition_rows(&mut rows, |row| {
        let point = data.row(row);
        let margin = accumulate_projection(data.n_features, offset, |i, accumulator| {
            normal[i].mul_add(point[i], accumulator)
        });
        if !margin.is_finite() {
            let metric = if angular {
                DistanceMetric::Cosine
            } else {
                DistanceMetric::Euclidean
            };
            metric.distance(point, left) <= metric.distance(point, right)
        } else if margin.abs() <= 1.0e-8 {
            splitmix64(key ^ row as u64) & 1 == 0
        } else {
            margin > 0.0
        }
    });
    if boundary == 0 || boundary == rows.len() {
        leaves.push(rows);
        return;
    }
    let right = rows.split_off(boundary);
    collect_training_leaves(
        data,
        rows,
        child_key(key, 0),
        leaf_size,
        depth - 1,
        angular,
        leaves,
    );
    collect_training_leaves(
        data,
        right,
        child_key(key, 1),
        leaf_size,
        depth - 1,
        angular,
        leaves,
    );
}

fn angular_normal<'a>(left: &'a [f32], right: &'a [f32]) -> impl Iterator<Item = f32> + 'a {
    let left_norm = projection_norm(left);
    let right_norm = projection_norm(right);
    left.iter()
        .zip(right)
        .map(move |(&a, &b)| a / left_norm - b / right_norm)
}

pub(super) fn angular_margin(left: &[f32], right: &[f32], query: &[f32]) -> f32 {
    let left_norm = projection_norm(left);
    let right_norm = projection_norm(right);
    accumulate_projection(query.len(), 0.0, |i, sum| {
        (left[i] / left_norm - right[i] / right_norm).mul_add(query[i], sum)
    })
}

fn projection_norm(point: &[f32]) -> f32 {
    let norm =
        accumulate_projection(point.len(), 0.0, |i, sum| point[i].mul_add(point[i], sum)).sqrt();
    if norm == 0.0 { 1.0 } else { norm }
}

fn tree_key(scalar: u64, ordinal: usize) -> u64 {
    splitmix64(scalar ^ splitmix64((ordinal as u64).wrapping_add(1)))
}

fn child_key(key: u64, side: u64) -> u64 {
    splitmix64((key << 1) | side)
}

fn pivot_words(key: u64) -> [u64; 2] {
    [splitmix64(key), splitmix64(key ^ 0x5851_f42d_4c95_7f2d)]
}

fn splitmix64(key: u64) -> u64 {
    let word = key.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

pub(super) fn projection_pivots(rows: &[usize], [first, second]: [u64; 2]) -> (usize, usize) {
    let first = (first % rows.len() as u64) as usize;
    let second = (second % (rows.len() - 1) as u64) as usize;
    (rows[first], rows[second + usize::from(second >= first)])
}

pub(super) fn partition_rows(rows: &mut [usize], goes_left: impl Fn(usize) -> bool) -> usize {
    let mut cursor = 0;
    let mut boundary = rows.len();
    while cursor < boundary {
        if goes_left(rows[cursor]) {
            cursor += 1;
        } else {
            boundary -= 1;
            rows.swap(cursor, boundary);
        }
    }
    boundary
}

pub(super) fn prepared_margin(left: &[f32], right: &[f32], query: &[f32]) -> f32 {
    // Prepared splits round each centered product before summing in f64.
    let offset = left.iter().zip(right).fold(0.0_f64, |offset, (&a, &b)| {
        offset - f64::from((a - b) * ((a + b) * 0.5))
    });
    accumulate_projection(query.len(), offset as f32, |i, accumulator| {
        (left[i] - right[i]).mul_add(query[i], accumulator)
    })
}

#[inline]
pub(super) fn accumulate_projection(
    width: usize,
    mut sum: f32,
    add: impl Fn(usize, f32) -> f32,
) -> f32 {
    let wide_end = width / 32 * 32;
    if wide_end != 0 {
        let mut wide = [0.0; 32];
        wide[0] = sum;
        for start in (0..wide_end).step_by(32) {
            for (lane, accumulator) in wide.iter_mut().enumerate() {
                *accumulator = add(start + lane, *accumulator);
            }
        }
        sum = reduce_32_lanes(&wide);
    }
    let block_end = width / 4 * 4;
    let mut lanes = [sum, 0.0, 0.0, 0.0];
    for start in (wide_end..block_end).step_by(4) {
        for (lane, accumulator) in lanes.iter_mut().enumerate() {
            *accumulator = add(start + lane, *accumulator);
        }
    }
    sum = reduce_4_lanes(&lanes);
    for i in block_end..width {
        sum = add(i, sum);
    }
    sum
}

#[cfg(test)]
#[path = "training_projection_tests.rs"]
mod tests;
