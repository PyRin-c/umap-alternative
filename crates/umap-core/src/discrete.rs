//! Distances for ordered counts and strings used as discrete targets.

use std::collections::HashMap;

fn count_log(value: f64) -> f64 {
    // The pinned Linux reference produces a positive quiet NaN for a negative
    // float64 logarithm. Its float32 logarithm and the Windows f64 path use
    // the platform result, which is negative on the verified hosts.
    if cfg!(target_os = "linux") && value < 0.0 {
        f64::NAN
    } else {
        value.ln()
    }
}

/// Unnormalized signed Poisson count distance on the interval between two counts.
/// `poisson_lambda.1` selects float32 evaluation of the logarithm.
pub fn count_distance(x: i64, y: i64, poisson_lambda: (f64, bool)) -> f64 {
    let low = x.min(y);
    let high = x.max(y);
    let (lambda, single) = poisson_lambda;
    let log_lambda = if single {
        f64::from((lambda as f32).ln())
    } else {
        count_log(lambda)
    };
    let mut log_factorial = if low < 10 {
        (2..low).map(|i| (i as f64).ln()).sum()
    } else {
        let count = low as f64 + 1.0;
        let ln_count = count.ln();
        (count * ln_count - count)
            + 0.5 * ((2.0 * std::f64::consts::PI).ln() - ln_count)
            + 1.0 / (12.0 * count)
    };
    let mut distance = 0.0;
    for count in low..high {
        distance += (count as f64 * log_lambda - lambda) - log_factorial;
        log_factorial += count_log(count as f64);
    }
    distance
}

/// Levenshtein edit distance in Unicode code points. The row-based cutoff
/// stops once every prefix of the right string exceeds the specified bound.
pub fn levenshtein(left: &[u32], right: &[u32], max_distance: f64) -> f64 {
    if left.len().abs_diff(right.len()) as f64 > max_distance {
        return max_distance;
    }
    if left.is_empty() || right.is_empty() {
        return left.len().max(right.len()) as f64;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; previous.len()];
    for (row, &first) in left.iter().enumerate() {
        current[0] = row + 1;
        let mut minimum = current[0];
        for (column, &second) in right.iter().enumerate() {
            let value = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(first != second));
            current[column + 1] = value;
            minimum = minimum.min(value);
        }
        if minimum as f64 > max_distance {
            return max_distance;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()] as f64
}

/// Myers edit distance, capped at `max_distance` on inputs below 64 code points.
/// At the 64-point boundary the reference takes the row-reduction path instead.
pub fn myers(left: &[u32], right: &[u32], max_distance: f64) -> f64 {
    if left.len().max(right.len()) >= 64 {
        return levenshtein(left, right, max_distance);
    }
    if left.len().abs_diff(right.len()) as f64 > max_distance {
        return max_distance;
    }
    let (pattern, text) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    if pattern.is_empty() {
        return text.len() as f64;
    }
    let mut masks = HashMap::with_capacity(pattern.len());
    for (position, &code) in pattern.iter().enumerate() {
        *masks.entry(code).or_insert(0_u64) |= 1 << position;
    }
    let last_bit = 1 << (pattern.len() - 1);
    let mut distance = pattern.len();
    let (mut positive, mut negative) = (!0_u64, 0_u64);
    for code in text {
        let equal = masks.get(code).copied().unwrap_or(0);
        let vertical = equal | negative;
        let horizontal = (((equal & positive).wrapping_add(positive)) ^ positive) | equal;
        let increase = negative | !(horizontal | positive);
        let decrease = positive & horizontal;
        distance += usize::from(increase & last_bit != 0);
        distance -= usize::from(decrease & last_bit != 0);
        let increase = (increase << 1) | 1;
        let decrease = decrease << 1;
        positive = decrease | !(vertical | increase);
        negative = increase & vertical;
    }
    (distance as f64).min(max_distance)
}
