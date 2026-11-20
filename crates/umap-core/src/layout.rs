//! Small deterministic layout-update primitives.
//!
//! This module contains the attractive part of the UMAP objective for one
//! edge.  If `q = ||y_i - y_j||^2`, the edge loss is
//! `log(1 + a * q^b)`.  Its gradient with respect to `y_i` is
//!
//! `2 * a * b * q^(b - 1) * (y_i - y_j) / (1 + a * q^b)`.
//!
//! The public update functions apply one simultaneous, symmetric gradient
//! descent step in `f32` arithmetic.  They are intentionally a primitive:
//! scheduling, edge sampling, and repulsive updates belong to a later layer.

use std::fmt;

/// Input or arithmetic failures while applying one attractive edge update.
#[derive(Clone, Debug, PartialEq)]
pub enum LayoutError {
    /// The row-major embedding does not have a positive component count or a
    /// length divisible by that count.
    InvalidShape {
        /// Number of components claimed for each row.
        n_components: usize,
        /// Number of values supplied by the caller.
        values: usize,
    },
    /// The two endpoint slices do not describe the same number of components.
    InvalidEndpointShape {
        /// Number of values in the first endpoint.
        head: usize,
        /// Number of values in the second endpoint.
        tail: usize,
    },
    /// An endpoint row is outside the row-major embedding.
    InvalidRow {
        /// Requested row index.
        row: usize,
        /// Number of rows in the embedding.
        n_rows: usize,
    },
    /// A parameter is non-finite or outside the domain of the attractive
    /// objective (`a > 0`, `b > 0`, and `learning_rate >= 0`).
    InvalidParameter {
        /// Parameter name (`"a"`, `"b"`, or `"learning_rate"`).
        parameter: &'static str,
        /// Supplied parameter value.
        value: f32,
    },
    /// An endpoint contains a non-finite coordinate.
    NonFiniteValue {
        /// Endpoint name (`"head"` or `"tail"`).
        endpoint: &'static str,
        /// Coordinate index within that endpoint.
        component: usize,
        /// Invalid coordinate value.
        value: f32,
    },
    /// The squared distance or objective derivative overflowed or became
    /// undefined in `f32` arithmetic.
    NonFiniteArithmetic {
        /// Description of the intermediate that was not finite.
        quantity: &'static str,
        /// Intermediate value observed by the kernel.
        value: f32,
    },
    /// Applying the computed step would produce a non-finite coordinate.
    NonFiniteUpdate {
        /// Endpoint name (`"head"` or `"tail"`).
        endpoint: &'static str,
        /// Coordinate index within that endpoint.
        component: usize,
        /// Candidate coordinate value.
        value: f32,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_components,
                values,
            } => write!(
                formatter,
                "layout shape with {n_components} components does not match {values} values"
            ),
            Self::InvalidEndpointShape { head, tail } => write!(
                formatter,
                "attractive edge endpoints have different lengths: head={head}, tail={tail}"
            ),
            Self::InvalidRow { row, n_rows } => {
                write!(formatter, "layout row {row} is outside 0..{n_rows}")
            }
            Self::InvalidParameter { parameter, value } => {
                write!(
                    formatter,
                    "invalid attractive parameter {parameter}={value}"
                )
            }
            Self::NonFiniteValue {
                endpoint,
                component,
                value,
            } => write!(
                formatter,
                "{endpoint} coordinate {component} is not finite: {value}"
            ),
            Self::NonFiniteArithmetic { quantity, value } => {
                write!(
                    formatter,
                    "attractive update {quantity} is not finite: {value}"
                )
            }
            Self::NonFiniteUpdate {
                endpoint,
                component,
                value,
            } => write!(
                formatter,
                "attractive update would make {endpoint} coordinate {component} non-finite: {value}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Apply one symmetric attractive update to two endpoint vectors.
///
/// The endpoint vectors are interpreted as Euclidean coordinates.  The loss
/// for this edge is `log(1 + a * d^(2b))`, where `d` is their Euclidean
/// distance.  The update uses its exact derivative in `f32` arithmetic and
/// moves both endpoints toward one another using the same step computed from
/// the original coordinates.  A zero-distance edge is a valid no-op.
///
/// No force clipping is applied.  If a finite input nevertheless overflows a
/// distance, derivative, or resulting coordinate in `f32`, the function
/// returns [`LayoutError::NonFiniteArithmetic`] or
/// [`LayoutError::NonFiniteUpdate`] without modifying either endpoint.
pub fn attractive_update_pair(
    head: &mut [f32],
    tail: &mut [f32],
    a: f32,
    b: f32,
    learning_rate: f32,
) -> Result<(), LayoutError> {
    if head.is_empty() || head.len() != tail.len() {
        return Err(if head.len() != tail.len() {
            LayoutError::InvalidEndpointShape {
                head: head.len(),
                tail: tail.len(),
            }
        } else {
            LayoutError::InvalidShape {
                n_components: 0,
                values: head.len(),
            }
        });
    }

    validate_parameters(a, b, learning_rate)?;
    validate_endpoint(head, "head")?;
    validate_endpoint(tail, "tail")?;

    // Compute the squared distance in a fixed component order.  Keeping this
    // and all subsequent arithmetic in f32 makes the primitive deterministic
    // for a given f32 input and avoids an implicit precision contract.
    let mut distance_squared = 0.0_f32;
    for (&head_value, &tail_value) in head.iter().zip(tail.iter()) {
        let difference = tail_value - head_value;
        distance_squared += difference * difference;
    }

    if !distance_squared.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "squared distance",
            value: distance_squared,
        });
    }

    // At zero distance the derivative expression contains q^(b - 1), which
    // is singular for some valid b values.  The positional gradient is still
    // exactly zero because every coordinate difference is zero.
    if distance_squared == 0.0 || learning_rate == 0.0 {
        return Ok(());
    }

    let distance_power_b = distance_squared.powf(b);
    let denominator = 1.0_f32 + a * distance_power_b;
    if !distance_power_b.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "distance power",
            value: distance_power_b,
        });
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "objective denominator",
            value: denominator,
        });
    }
    let gradient_scale = (2.0_f32 * a * b * distance_squared.powf(b - 1.0_f32)) / denominator;
    if !gradient_scale.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "gradient scale",
            value: gradient_scale,
        });
    }
    let step = learning_rate * gradient_scale;
    if !step.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "gradient step",
            value: step,
        });
    }

    for (component, (&head_value, &tail_value)) in head.iter().zip(tail.iter()).enumerate() {
        let difference = tail_value - head_value;
        let new_head = head_value + step * difference;
        let new_tail = tail_value - step * difference;

        if !new_head.is_finite() {
            return Err(LayoutError::NonFiniteUpdate {
                endpoint: "head",
                component,
                value: new_head,
            });
        }
        if !new_tail.is_finite() {
            return Err(LayoutError::NonFiniteUpdate {
                endpoint: "tail",
                component,
                value: new_tail,
            });
        }
    }

    for component in 0..head.len() {
        let difference = tail[component] - head[component];
        head[component] += step * difference;
        tail[component] -= step * difference;
    }

    Ok(())
}

/// Apply one symmetric attractive update to two rows in a row-major matrix.
///
/// `embedding` has shape `n_rows x n_components`; `head_row` and `tail_row`
/// identify the edge endpoints.  Equal rows are treated as a valid no-op after
/// validating their coordinates and parameters.
pub fn attractive_update(
    embedding: &mut [f32],
    n_components: usize,
    head_row: usize,
    tail_row: usize,
    a: f32,
    b: f32,
    learning_rate: f32,
) -> Result<(), LayoutError> {
    if n_components == 0 || !embedding.len().is_multiple_of(n_components) {
        return Err(LayoutError::InvalidShape {
            n_components,
            values: embedding.len(),
        });
    }

    validate_parameters(a, b, learning_rate)?;

    let n_rows = embedding.len() / n_components;
    if head_row >= n_rows {
        return Err(LayoutError::InvalidRow {
            row: head_row,
            n_rows,
        });
    }
    if tail_row >= n_rows {
        return Err(LayoutError::InvalidRow {
            row: tail_row,
            n_rows,
        });
    }

    if head_row == tail_row {
        let start = head_row * n_components;
        validate_endpoint(&embedding[start..start + n_components], "head")?;
        return Ok(());
    }

    let head_start = head_row * n_components;
    let tail_start = tail_row * n_components;
    if head_start < tail_start {
        let (before_tail, after_tail) = embedding.split_at_mut(tail_start);
        let head = &mut before_tail[head_start..head_start + n_components];
        let tail = &mut after_tail[..n_components];
        attractive_update_pair(head, tail, a, b, learning_rate)
    } else {
        let (before_head, after_head) = embedding.split_at_mut(head_start);
        let tail = &mut before_head[tail_start..tail_start + n_components];
        let head = &mut after_head[..n_components];
        attractive_update_pair(head, tail, a, b, learning_rate)
    }
}

fn validate_parameters(a: f32, b: f32, learning_rate: f32) -> Result<(), LayoutError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "a",
            value: a,
        });
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "b",
            value: b,
        });
    }
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "learning_rate",
            value: learning_rate,
        });
    }
    Ok(())
}

fn validate_endpoint(endpoint: &[f32], name: &'static str) -> Result<(), LayoutError> {
    if let Some((component, value)) = endpoint
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LayoutError::NonFiniteValue {
            endpoint: name,
            component,
            value,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LayoutError, attractive_update, attractive_update_pair};

    #[test]
    fn analytic_two_dimensional_update_is_symmetric() {
        let mut head = [0.0_f32, 0.0];
        let mut tail = [2.0_f32, 0.0];

        attractive_update_pair(&mut head, &mut tail, 1.0, 1.0, 0.1).unwrap();

        // q = 4, so the gradient scale is 2 / (1 + 4) = 0.4.  The
        // displacement of each endpoint is 0.1 * 0.4 * 2 = 0.08.
        assert!((head[0] - 0.08).abs() < 1.0e-6);
        assert!((tail[0] - 1.92).abs() < 1.0e-6);
        assert_eq!(head[1], 0.0);
        assert_eq!(tail[1], 0.0);
        assert!((head[0] + tail[0] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_distance_is_a_no_op() {
        let mut head = [1.25_f32, -2.5];
        let original = head;
        let mut tail = original;

        attractive_update_pair(&mut head, &mut tail, 1.0, 0.25, 10.0).unwrap();

        assert_eq!(head, original);
        assert_eq!(tail, original);
    }

    #[test]
    fn row_major_update_targets_only_the_requested_rows() {
        let mut embedding = [0.0_f32, 0.0, 2.0, 0.0, 9.0, 9.0];

        attractive_update(&mut embedding, 2, 1, 0, 1.0, 1.0, 0.1).unwrap();

        assert!((embedding[0] - 0.08).abs() < 1.0e-6);
        assert!((embedding[2] - 1.92).abs() < 1.0e-6);
        assert_eq!(&embedding[4..], &[9.0, 9.0]);
    }

    #[test]
    fn invalid_inputs_are_rejected_without_modification() {
        let mut head = [0.0_f32, 1.0];
        let mut tail = [2.0_f32, 3.0];
        let original_head = head;
        let original_tail = tail;

        let error = attractive_update_pair(&mut head, &mut tail, 1.0, 0.0, 0.1).unwrap_err();

        assert!(matches!(
            error,
            LayoutError::InvalidParameter { parameter: "b", .. }
        ));
        assert_eq!(head, original_head);
        assert_eq!(tail, original_tail);
    }
}
