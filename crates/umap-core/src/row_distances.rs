//! Standalone real feature distances with binary64 results.

use crate::neighbors::{DistanceMetric, bray_curtis_sums, correlation_products, cosine_products};

/// Coordinates retain their product and difference precision before summation.
#[derive(Clone, Copy, Debug)]
pub enum MetricRow<'a> {
    /// Binary32 coordinates.
    F32(&'a [f32]),
    /// Binary64 coordinates.
    F64(&'a [f64]),
    /// Signed integers with 64-bit arithmetic.
    Signed(&'a [i64]),
    /// Unsigned integers with 64-bit arithmetic.
    Unsigned(&'a [u64]),
}

impl MetricRow<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Signed(values) => values.len(),
            Self::Unsigned(values) => values.len(),
        }
    }

    fn floats(self) -> impl Iterator<Item = f64> {
        (0..self.len()).map(move |i| match self {
            Self::F32(values) => f64::from(values[i]),
            Self::F64(values) => values[i],
            Self::Signed(values) => values[i] as f64,
            Self::Unsigned(values) => values[i] as f64,
        })
    }

    fn signed(self) -> impl Iterator<Item = i64> {
        (0..self.len()).map(move |i| match self {
            Self::Signed(values) => values[i],
            Self::Unsigned(values) => values[i] as i64,
            _ => unreachable!("only integer rows use signed arithmetic"),
        })
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }

    fn squared_norm(self) -> f64 {
        match self {
            Self::F32(row) => row.iter().map(|&x| f64::from(x * x)).sum(),
            Self::F64(row) => row.iter().map(|&x| x * x).sum(),
            Self::Signed(row) => row.iter().map(|&x| x.wrapping_mul(x) as f64).sum(),
            Self::Unsigned(row) => row.iter().map(|&x| x.wrapping_mul(x) as i64 as f64).sum(),
        }
    }
}

/// Evaluate cosine, correlation, Canberra or Bray–Curtis on two equal-width rows.
///
/// The result retains binary64 reduction precision. Empty correlation,
/// unequal widths and unsupported metrics return `None`.
pub fn row_distance(
    metric: DistanceMetric,
    left: MetricRow<'_>,
    right: MetricRow<'_>,
) -> Option<f64> {
    if left.len() != right.len() {
        return None;
    }
    let distance = match metric {
        DistanceMetric::Cosine => cosine(left, right),
        DistanceMetric::Correlation if left.len() != 0 => correlation(left, right),
        DistanceMetric::Canberra | DistanceMetric::BrayCurtis => ratios(metric, left, right),
        _ => return None,
    };
    Some(distance)
}

fn cosine(left: MetricRow<'_>, right: MetricRow<'_>) -> f64 {
    let (dot, xx, yy) = if let (MetricRow::F32(left), MetricRow::F32(right)) = (left, right) {
        cosine_products(left, right)
    } else {
        let dot = match (left, right) {
            (MetricRow::Unsigned(x), MetricRow::Unsigned(y)) => x
                .iter()
                .zip(y)
                .map(|(&a, &b)| a.wrapping_mul(b) as f64)
                .sum(),
            _ if left.is_float() || right.is_float() => {
                left.floats().zip(right.floats()).map(|(a, b)| a * b).sum()
            }
            _ => left
                .signed()
                .zip(right.signed())
                .map(|(a, b)| a.wrapping_mul(b) as f64)
                .sum(),
        };
        (dot, left.squared_norm(), right.squared_norm())
    };
    match (xx == 0.0, yy == 0.0) {
        (true, true) => 0.0,
        (true, false) | (false, true) => 1.0,
        (false, false) if dot.is_nan() => cosine_nan(left, right),
        (false, false) => 1.0 - dot / (xx * yy).sqrt(),
    }
}

fn cosine_nan(left: MetricRow<'_>, right: MetricRow<'_>) -> f64 {
    // A NaN in the running dot product keeps its payload. At each coordinate
    // the left operand has priority if both values are already NaNs.
    let both_f32 = matches!((left, right), (MetricRow::F32(_), MetricRow::F32(_)));
    let invalid = f64::from_bits(0xfff8_0000_0000_0000);
    let mut dot = 0.0;
    for (a, b) in left.floats().zip(right.floats()) {
        if a.is_nan() {
            return f64::from_bits(a.to_bits() | (1 << 51));
        }
        if b.is_nan() {
            return f64::from_bits(b.to_bits() | (1 << 51));
        }
        let product = if both_f32 {
            f64::from((a as f32) * (b as f32))
        } else {
            a * b
        };
        dot += product;
        if dot.is_nan() {
            return invalid;
        }
    }
    invalid
}

fn correlation(left: MetricRow<'_>, right: MetricRow<'_>) -> f64 {
    let ((dot, xx, yy), _) = match (left, right) {
        (MetricRow::F32(x), MetricRow::F32(y)) => correlation_products(x, y),
        (MetricRow::F64(x), MetricRow::F64(y)) => correlation_products(x, y),
        _ => {
            let (x, y): (Vec<_>, Vec<_>) = left.floats().zip(right.floats()).unzip();
            correlation_products(&x, &y)
        }
    };
    if xx == 0.0 && yy == 0.0 {
        0.0
    } else if dot == 0.0 {
        1.0
    } else {
        1.0 - dot / (xx * yy).sqrt()
    }
}

fn ratios(metric: DistanceMetric, left: MetricRow<'_>, right: MetricRow<'_>) -> f64 {
    let canberra = metric == DistanceMetric::Canberra;
    match (left, right) {
        (MetricRow::F32(x), MetricRow::F32(y)) if canberra => x
            .iter()
            .zip(y)
            .map(|(&a, &b)| {
                let divisor = a.abs() + b.abs();
                if divisor > 0.0 {
                    f64::from((a - b).abs() / divisor)
                } else {
                    0.0
                }
            })
            .fold(0.0, |total, value| total + value),
        (MetricRow::F32(x), MetricRow::F32(y)) => {
            let (numerator, denominator) =
                bray_curtis_sums(x.iter().copied().zip(y.iter().copied()));
            positive_ratio(numerator, denominator)
        }
        _ if left.is_float() || right.is_float() => {
            let pairs = left.floats().zip(right.floats());
            if canberra {
                pairs
                    .map(|(a, b)| positive_ratio((a - b).abs(), a.abs() + b.abs()))
                    .fold(0.0, |total, value| total + value)
            } else {
                ratio_of_sums(pairs.map(|(a, b)| ((a - b).abs(), (a + b).abs())))
            }
        }
        (MetricRow::Unsigned(x), MetricRow::Unsigned(y)) => {
            let pairs = x
                .iter()
                .zip(y)
                .map(|(&a, &b)| (a.wrapping_sub(b), a.wrapping_add(b)));
            if canberra {
                pairs
                    .map(|(a, b)| positive_ratio(a as f64, b as f64))
                    .fold(0.0, |total, value| total + value)
            } else {
                ratio_of_sums(pairs.map(|(a, b)| (a as f64, b as f64)))
            }
        }
        (MetricRow::Signed(signed), MetricRow::Unsigned(unsigned))
        | (MetricRow::Unsigned(unsigned), MetricRow::Signed(signed))
            if canberra =>
        {
            signed
                .iter()
                .zip(unsigned)
                .map(|(&signed, &unsigned)| mixed_canberra_term(signed, unsigned))
                .fold(0.0, |total, value| total + value)
        }
        _ => {
            let pairs = left.signed().zip(right.signed());
            if canberra {
                pairs
                    .map(|(a, b)| signed_canberra_term(a, b))
                    .fold(0.0, |total, value| total + value)
            } else {
                ratio_of_sums(pairs.map(|(a, b)| {
                    (
                        signed_difference(a, b) as f64,
                        a.wrapping_add(b).wrapping_abs() as f64,
                    )
                }))
            }
        }
    }
}

fn signed_difference(a: i64, b: i64) -> i64 {
    a.max(b).wrapping_sub(a.min(b))
}

fn signed_canberra_term(a: i64, b: i64) -> f64 {
    let denominator = a.wrapping_abs().wrapping_add(b.wrapping_abs());
    signed_canberra_ratio(a, b, denominator)
}

fn mixed_canberra_term(signed: i64, unsigned: u64) -> f64 {
    let right = unsigned as i64;
    // The unsigned absolute value is the original word, even when the
    // mixed numerator and divisor interpret its high bit as a sign.
    let denominator = signed.wrapping_abs().wrapping_add(right);
    if right < 0 && denominator < 0 {
        0.0
    } else {
        signed_canberra_ratio(signed, right, denominator)
    }
}

fn signed_canberra_ratio(a: i64, b: i64, denominator: i64) -> f64 {
    if a == i64::MIN || b == i64::MIN || denominator == 0 {
        0.0
    } else {
        signed_difference(a, b) as f64 / denominator as f64
    }
}

fn ratio_of_sums(terms: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (numerator, denominator) = terms.fold((0.0, 0.0), |(num, den), (a, b)| (num + a, den + b));
    positive_ratio(numerator, denominator)
}

fn positive_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}
