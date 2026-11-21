//! Exponential membership attenuation for supervised fuzzy graphs.

#[derive(Clone, Copy)]
struct Multiplier {
    value: f64,
    single: bool,
}

impl Multiplier {
    fn new(exponent: f64, single: bool) -> Self {
        let value = if single {
            f64::from((exponent as f32).exp())
        } else {
            exponent.exp()
        };
        Self { value, single }
    }

    fn apply(self, weight: f64, single_weight: bool) -> f64 {
        if self.single && single_weight {
            f64::from(weight as f32 * self.value as f32)
        } else {
            weight * self.value
        }
    }
}

pub(super) struct CategoricalAttenuation {
    unknown: Multiplier,
    different: Multiplier,
}

impl CategoricalAttenuation {
    pub(super) fn new(unknown: (f64, bool), far: (f64, bool)) -> Self {
        Self {
            unknown: Multiplier::new(-unknown.0, unknown.1),
            different: Multiplier::new(-far.0, far.1),
        }
    }

    pub(super) fn apply(
        &self,
        unknown: bool,
        different: bool,
        weight: f64,
        single_weight: bool,
    ) -> f64 {
        if unknown {
            self.unknown.apply(weight, single_weight)
        } else if different {
            self.different.apply(weight, single_weight)
        } else {
            weight
        }
    }
}

/// Attenuate graph memberships by endpoint labels. Label -1 denotes an unknown
/// class, and `i64::MIN` denotes a label that never compares equal (such as NaN).
/// The booleans in `unknown` and `far` select float32 exponent arithmetic.
pub fn attenuate_categorical_edges(
    (rows, columns): (&[usize], &[usize]),
    weights: &mut [f64],
    labels: &[i64],
    unknown: (f64, bool),
    far: (f64, bool),
    single_weights: bool,
) -> Result<(), &'static str> {
    if rows.len() != weights.len()
        || columns.len() != weights.len()
        || rows
            .iter()
            .chain(columns)
            .any(|&index| index >= labels.len())
    {
        return Err("invalid categorical edge coordinates");
    }
    let attenuation = CategoricalAttenuation::new(unknown, far);
    for ((&row, &column), weight) in rows.iter().zip(columns).zip(weights) {
        let (left, right) = (labels[row], labels[column]);
        *weight = attenuation.apply(
            left == -1 || right == -1,
            left != right || left == i64::MIN,
            *weight,
            single_weights,
        );
    }
    Ok(())
}

/// Multiply each weight by the exponential of its supplied signed exponent.
/// Each exponent is paired with its exp precision. A float32 weight is rounded
/// during multiplication only when its exponential factor is also float32.
pub fn attenuate_metric_edges(
    weights: &mut [f64],
    exponents: &[(f64, bool)],
    single_weights: bool,
) -> Result<(), &'static str> {
    if weights.len() != exponents.len() {
        return Err("edge weights and exponents must have equal lengths");
    }
    for (weight, &(exponent, single)) in weights.iter_mut().zip(exponents) {
        *weight = Multiplier::new(exponent, single).apply(*weight, single_weights);
    }
    Ok(())
}
