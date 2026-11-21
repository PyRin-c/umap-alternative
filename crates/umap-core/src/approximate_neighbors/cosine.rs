use super::{projection::accumulate_projection, reduce_4_lanes, reduce_32_lanes};

pub(super) fn normalized_training(data: &[f32], width: usize) -> Vec<f32> {
    normalize(data, width, false)
}

pub(super) fn normalized_queries(data: &[f32], width: usize) -> Vec<f32> {
    normalize(data, width, true)
}

fn normalize(data: &[f32], width: usize, query: bool) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());
    for row in data.chunks_exact(width) {
        if query {
            let norm = squared_query_norm(row).sqrt();
            let inverse = if norm == 0.0 { 0.0 } else { 1.0 / norm };
            result.extend(row.iter().map(|value| value * inverse));
        } else {
            let norm = squared_norm(row).sqrt();
            let norm = if norm < 10.0 * f32::EPSILON {
                1.0
            } else {
                norm
            };
            result.extend(row.iter().map(|value| value / norm));
        }
    }
    result
}

pub(super) fn squared_query_norm(row: &[f32]) -> f32 {
    let (blocks, tail) = row.as_chunks::<32>();
    let mut wide = [0.0; 32];
    for block in blocks {
        for (sum, &value) in wide.iter_mut().zip(block) {
            *sum += value * value;
        }
    }
    let (blocks, tail) = tail.as_chunks::<4>();
    let mut lanes = [reduce_32_lanes(&wide), 0.0, 0.0, 0.0];
    for block in blocks {
        for (sum, &value) in lanes.iter_mut().zip(block) {
            *sum += value * value;
        }
    }
    tail.iter()
        .fold(reduce_4_lanes(&lanes), |sum, value| sum + value * value)
}

fn squared_norm(row: &[f32]) -> f32 {
    if row.len() < 4 {
        return row.iter().map(|value| value * value).sum();
    }
    let mut lanes = [0.0; 4];
    // Training normalization matches the row-wise float32 contraction used
    // by the reference. Four vectors from each full block enter in reverse.
    let (blocks, tail) = row.as_chunks::<16>();
    for block in blocks {
        for vector in block.as_chunks::<4>().0.iter().rev() {
            for (sum, &value) in lanes.iter_mut().zip(vector) {
                *sum += value * value;
            }
        }
    }
    let (vectors, tail) = tail.as_chunks::<4>();
    for block in vectors {
        for (sum, &value) in lanes.iter_mut().zip(block) {
            *sum += value * value;
        }
    }
    for (sum, &value) in lanes.iter_mut().zip(tail) {
        *sum += value * value;
    }
    let [a, b, c, d] = lanes;
    (a + b) + (c + d)
}

pub(super) fn priority(left: &[f32], right: &[f32]) -> f32 {
    let dot = accumulate_projection(left.len(), 0.0, |i, sum| left[i].mul_add(right[i], sum));
    if dot >= 1.0 {
        0.0
    } else if dot > 0.0 {
        -dot.log2()
    } else {
        f32::MAX
    }
}

pub(super) fn distance(priority: f32) -> f32 {
    if priority.is_finite() {
        (1.0 - (-f64::from(priority)).exp2()) as f32
    } else {
        priority
    }
}
