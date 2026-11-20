//! Small deterministic layout-update primitives.
//!
//! This module contains the attractive part of the UMAP objective for one
//! edge, one explicit negative-example update, and a deterministic
//! attractive-only epoch scheduler.  If
//! `q = ||y_i - y_j||^2`, the edge loss is
//! `log(1 + a * q^b)`.  Its gradient with respect to `y_i` is
//!
//! `2 * a * b * q^(b - 1) * (y_i - y_j) / (1 + a * q^b)`.
//!
//! Attractive edge updates are symmetric and store coordinates in `f32`.
//! A negative example updates only the supplied current endpoint. Both
//! updates compute distances in `f32` and apply forces in `f64` before
//! storing the result in `f32`. The scheduler applies the attractive
//! edge-sampling rule.

use std::fmt;

use crate::fuzzy_graph::CsrGraph;

/// Input or arithmetic failures while applying one layout update.
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
    /// The CSR graph shape or backing-array lengths are inconsistent with one
    /// another, or the graph is not square.
    InvalidGraphShape {
        /// Number of graph rows.
        n_rows: usize,
        /// Number of graph columns.
        n_cols: usize,
        /// Length of the row-offset array.
        indptr: usize,
        /// Length of the column-index array.
        indices: usize,
        /// Length of the membership array.
        data: usize,
    },
    /// A CSR row has an invalid or non-monotonic pair of offsets.
    InvalidGraphOffsets {
        /// Row whose `[start, end)` range is invalid.
        row: usize,
        /// Start offset observed for that row.
        start: usize,
        /// End offset observed for that row.
        end: usize,
        /// Number of available edge entries.
        entries: usize,
    },
    /// A CSR column index is outside the square graph.
    InvalidGraphIndex {
        /// Source row of the invalid edge.
        row: usize,
        /// Position within that row's CSR range.
        position: usize,
        /// Invalid destination row.
        index: usize,
        /// Number of graph rows and columns.
        n_rows: usize,
    },
    /// CSR columns must be strictly ascending so sweep order is explicit and
    /// duplicate edges cannot silently apply their force twice.
    InvalidGraphOrder {
        /// Source row of the out-of-order edge.
        row: usize,
        /// Position within that row's CSR range.
        position: usize,
        /// Previous column index in the row.
        previous: usize,
        /// Current column index in the row.
        current: usize,
    },
    /// A CSR membership weight is not a finite fuzzy-set value in `[0, 1]`.
    InvalidGraphWeight {
        /// Source row of the invalid edge.
        row: usize,
        /// Position within that row's CSR range.
        position: usize,
        /// Invalid membership weight.
        weight: f32,
    },
    /// A parameter is non-finite or outside the supported layout-update
    /// domain (`a > 0`, `b > 0`, and `learning_rate >= 0`; the negative
    /// example's `repulsion_strength` is finite and non-negative).
    InvalidParameter {
        /// Parameter name (`"a"`, `"b"`, `"repulsion_strength"`, or
        /// `"learning_rate"`).
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
    /// undefined.
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
                "layout-update endpoints have different lengths: head={head}, tail={tail}"
            ),
            Self::InvalidRow { row, n_rows } => {
                write!(formatter, "layout row {row} is outside 0..{n_rows}")
            }
            Self::InvalidGraphShape {
                n_rows,
                n_cols,
                indptr,
                indices,
                data,
            } => write!(
                formatter,
                "CSR graph shape {n_rows}x{n_cols} has lengths indptr={indptr}, indices={indices}, data={data}"
            ),
            Self::InvalidGraphOffsets {
                row,
                start,
                end,
                entries,
            } => write!(
                formatter,
                "CSR graph row {row} has invalid offsets {start}..{end} for {entries} entries"
            ),
            Self::InvalidGraphIndex {
                row,
                position,
                index,
                n_rows,
            } => write!(
                formatter,
                "CSR graph row {row}, position {position} has index {index}, outside 0..{n_rows}"
            ),
            Self::InvalidGraphOrder {
                row,
                position,
                previous,
                current,
            } => write!(
                formatter,
                "CSR graph row {row}, position {position} is not strictly ascending: {previous} then {current}"
            ),
            Self::InvalidGraphWeight {
                row,
                position,
                weight,
            } => write!(
                formatter,
                "CSR graph row {row}, position {position} has invalid weight {weight}"
            ),
            Self::InvalidParameter { parameter, value } => {
                write!(formatter, "invalid layout parameter {parameter}={value}")
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
                write!(formatter, "layout update {quantity} is not finite: {value}")
            }
            Self::NonFiniteUpdate {
                endpoint,
                component,
                value,
            } => write!(
                formatter,
                "layout update would make {endpoint} coordinate {component} non-finite: {value}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Apply one symmetric attractive update to two endpoint vectors.
///
/// The endpoint vectors are interpreted as Euclidean coordinates.  The loss
/// for this edge is `log(1 + a * d^(2b))`, where `d` is their Euclidean
/// distance. Differences are computed in `f32`, and each squared component
/// is added with one rounding using `f32::mul_add`. Each gradient component
/// is clipped to `[-4, 4]` before scaling by the learning rate. The curve,
/// gradient, and coordinate addition use `f64`, with the result rounded to
/// `f32` for storage. Both endpoints use the same step from the original
/// coordinates.
/// A zero-distance edge is a valid no-op.
///
/// A non-finite distance, derivative, or stored coordinate returns
/// [`LayoutError::NonFiniteArithmetic`] or [`LayoutError::NonFiniteUpdate`]
/// before either endpoint is modified.
pub fn attractive_update_pair(
    head: &mut [f32],
    tail: &mut [f32],
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
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

    let (a, b, learning_rate) = (a.into(), b.into(), learning_rate.into());
    validate_parameters(a, b, learning_rate)?;

    // Accumulate in component order with one rounding per squared component.
    let mut distance_squared = 0.0_f32;
    for (&head_value, &tail_value) in head.iter().zip(tail.iter()) {
        let difference = tail_value - head_value;
        distance_squared = difference.mul_add(difference, distance_squared);
    }

    if !distance_squared.is_finite() {
        // A non-finite coordinate propagates through the distance. Check the
        // endpoints here to distinguish invalid input from distance overflow.
        validate_endpoint(head, "head")?;
        validate_endpoint(tail, "tail")?;
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

    let distance_squared = f64::from(distance_squared);
    let distance_power_b = distance_squared.powf(b);
    if !distance_power_b.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "distance power",
            value: distance_power_b as f32,
        });
    }
    let denominator = 1.0 + a * distance_power_b;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "objective denominator",
            value: denominator as f32,
        });
    }
    let gradient_scale = (2.0 * a * b * distance_squared.powf(b - 1.0)) / denominator;
    if !gradient_scale.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "gradient scale",
            value: gradient_scale as f32,
        });
    }

    let updated_endpoints = |head: f32, tail: f32| {
        let difference = f64::from(tail - head);
        let gradient = (gradient_scale * difference).clamp(-4.0, 4.0);
        let displacement = learning_rate * gradient;
        (
            (f64::from(head) + displacement) as f32,
            (f64::from(tail) - displacement) as f32,
        )
    };
    for (component, (&head_value, &tail_value)) in head.iter().zip(tail.iter()).enumerate() {
        let (new_head, new_tail) = updated_endpoints(head_value, tail_value);

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

    for (head, tail) in head.iter_mut().zip(tail.iter_mut()) {
        (*head, *tail) = updated_endpoints(*head, *tail);
    }

    Ok(())
}

/// Apply one one-sided negative-example update to a current endpoint.
///
/// `current` is the only coordinate vector that may be modified; `other` is
/// an explicitly selected negative example and remains unchanged. For
/// `s = ||current - other||^2`, the repulsive coefficient is
/// `2 * repulsion_strength * b / ((0.001 + s) * (1 + a * s^b))`.
/// Each component of the coefficient times `current - other` is clipped to
/// `[-4, 4]` before scaling by the learning rate.
///
/// Differences and fused squared-distance accumulation use `f32`. The force,
/// learning rate, and coordinate addition use `f64` before each coordinate
/// is rounded to `f32` for storage. A coincident pair is a validated no-op.
///
/// Coordinates and parameters must be finite, with `a > 0`, `b > 0`,
/// `repulsion_strength >= 0`, and `learning_rate >= 0`. Arithmetic or
/// candidate-coordinate overflow returns an error before the endpoint is
/// modified.
pub fn negative_example_update(
    current: &mut [f32],
    other: &[f32],
    a: impl Into<f64>,
    b: impl Into<f64>,
    repulsion_strength: impl Into<f64>,
    learning_rate: impl Into<f64>,
) -> Result<(), LayoutError> {
    if current.is_empty() || current.len() != other.len() {
        return Err(if current.len() != other.len() {
            LayoutError::InvalidEndpointShape {
                head: current.len(),
                tail: other.len(),
            }
        } else {
            LayoutError::InvalidShape {
                n_components: 0,
                values: current.len(),
            }
        });
    }

    let (a, b, repulsion_strength, learning_rate) = (
        a.into(),
        b.into(),
        repulsion_strength.into(),
        learning_rate.into(),
    );
    validate_parameters(a, b, learning_rate)?;
    validate_repulsion_strength(repulsion_strength)?;
    validate_endpoint(current, "current")?;
    validate_endpoint(other, "other")?;

    let mut distance_squared = 0.0_f32;
    for (&current_value, &other_value) in current.iter().zip(other.iter()) {
        let difference = current_value - other_value;
        distance_squared = difference.mul_add(difference, distance_squared);
    }

    if !distance_squared.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "squared distance",
            value: distance_squared,
        });
    }

    if distance_squared == 0.0 || repulsion_strength == 0.0 || learning_rate == 0.0 {
        return Ok(());
    }

    let distance_squared = f64::from(distance_squared);
    let distance_power_b = distance_squared.powf(b);
    if !distance_power_b.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "distance power",
            value: distance_power_b as f32,
        });
    }

    let objective_denominator = 1.0 + a * distance_power_b;
    if !objective_denominator.is_finite() || objective_denominator <= 0.0 {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "objective denominator",
            value: objective_denominator as f32,
        });
    }

    let distance_denominator = (0.001 + distance_squared) * objective_denominator;
    if !distance_denominator.is_finite() || distance_denominator <= 0.0 {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "negative-example denominator",
            value: distance_denominator as f32,
        });
    }

    let numerator = 2.0 * repulsion_strength * b;
    if !numerator.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "negative-example numerator",
            value: numerator as f32,
        });
    }

    let gradient_scale = numerator / distance_denominator;
    if !gradient_scale.is_finite() {
        return Err(LayoutError::NonFiniteArithmetic {
            quantity: "negative-example gradient scale",
            value: gradient_scale as f32,
        });
    }

    let updated_coordinate = |current: f32, other: f32| {
        let difference = f64::from(current - other);
        let gradient = (gradient_scale * difference).clamp(-4.0, 4.0);
        (f64::from(current) + learning_rate * gradient) as f32
    };
    // Validate every candidate before mutating any coordinate, preserving the
    // attractive kernel's all-or-nothing behavior for a single update.
    for (component, (&current_value, &other_value)) in current.iter().zip(other.iter()).enumerate()
    {
        let new_current = updated_coordinate(current_value, other_value);
        if !new_current.is_finite() {
            return Err(LayoutError::NonFiniteUpdate {
                endpoint: "current",
                component,
                value: new_current,
            });
        }
    }

    for (current_value, &other_value) in current.iter_mut().zip(other.iter()) {
        *current_value = updated_coordinate(*current_value, other_value);
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
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
) -> Result<(), LayoutError> {
    if n_components == 0 || !embedding.len().is_multiple_of(n_components) {
        return Err(LayoutError::InvalidShape {
            n_components,
            values: embedding.len(),
        });
    }

    let (a, b, learning_rate) = (a.into(), b.into(), learning_rate.into());
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

/// Apply one ascending CSR sweep of symmetric attractive updates.
///
/// The graph is traversed in row order and then in the stored ascending
/// column order.  Each CSR membership is one directed edge entry and is
/// applied once; if a graph stores both directions of an undirected edge,
/// both entries therefore contribute to the sweep.  A nonzero membership
/// selects an edge, but the supplied learning rate is fixed for every
/// selected edge; the membership is not multiplied into the force.  In a
/// scheduled optimizer, membership values belong in edge due-time/sample
/// scheduling, which is intentionally outside this static sweep.
///
/// The graph is validated completely before the first coordinate is changed:
/// it must be square, have consistent CSR arrays, contain in-range strictly
/// ascending columns, and contain finite weights in `[0, 1]`.  The embedding
/// must have exactly `graph.n_rows * n_components` finite values.
pub fn attractive_update_csr(
    embedding: &mut [f32],
    n_components: usize,
    graph: &CsrGraph,
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
) -> Result<(), LayoutError> {
    let (a, b, learning_rate) = (a.into(), b.into(), learning_rate.into());
    validate_csr_inputs(embedding, n_components, graph, a, b, learning_rate)?;

    for row in 0..graph.n_rows {
        for position in graph.indptr[row]..graph.indptr[row + 1] {
            if graph.data[position] > 0.0 {
                attractive_update(
                    embedding,
                    n_components,
                    row,
                    graph.indices[position],
                    a,
                    b,
                    learning_rate,
                )?;
            }
        }
    }

    Ok(())
}

/// Apply deterministic attractive updates over explicit optimization epochs.
///
/// Each positive CSR membership is treated as one directed edge.  Its next
/// sample is initially due at `1 / weight` and advances by the same interval
/// after each update, so an edge with weight `w` is sampled at approximately
/// `w` updates per epoch.  The scheduler traverses CSR rows and columns in
/// their stored order, uses one shared learning rate for each epoch, and
/// linearly decays that rate *after* an epoch has been processed:
///
/// ```text
/// alpha <- initial_learning_rate * (1 - epoch / n_epochs)
/// ```
///
/// This ordering follows the public UMAP epoch schedule: with an edge weight
/// of one, `n_epochs = 1` performs no update (its first due epoch is one),
/// while `n_epochs = 2` performs one update at the initial learning rate.
/// `n_epochs = 0` is a validated no-op. Due times and the decayed rate are
/// kept in `f64` across epochs. Coordinates are updated by
/// [`attractive_update`].
///
/// This function intentionally implements only attractive updates.  It does
/// not draw negative samples and does not apply repulsive forces.  The graph
/// and embedding are validated completely before the first update, but an
/// arithmetic failure discovered while applying a later edge may follow
/// earlier successful updates, just like [`attractive_update_csr`].
pub fn attractive_optimize_csr(
    embedding: &mut [f32],
    n_components: usize,
    graph: &CsrGraph,
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
    n_epochs: usize,
) -> Result<(), LayoutError> {
    let (a, b, learning_rate) = (a.into(), b.into(), learning_rate.into());
    validate_csr_inputs(embedding, n_components, graph, a, b, learning_rate)?;

    // UMAP's edge sampling interval is n_epochs / (n_epochs * weight), which
    // simplifies to 1 / weight for every positive weight.  Keeping due times
    // in f64 also preserves fractional schedules when weights are f32 values.
    let mut epoch_of_next_sample = graph
        .data
        .iter()
        .map(|&weight| {
            if weight == 0.0 {
                f64::INFINITY
            } else {
                1.0 / f64::from(weight)
            }
        })
        .collect::<Vec<_>>();

    let mut alpha = learning_rate;
    for epoch in 0..n_epochs {
        let epoch_as_f64 = epoch as f64;
        for row in 0..graph.n_rows {
            let start = graph.indptr[row];
            let end = graph.indptr[row + 1];
            for (offset, (&index, next_sample)) in graph.indices[start..end]
                .iter()
                .zip(epoch_of_next_sample[start..end].iter_mut())
                .enumerate()
            {
                let position = start + offset;
                if *next_sample <= epoch_as_f64 {
                    attractive_update(embedding, n_components, row, index, a, b, alpha)?;
                    let weight = f64::from(graph.data[position]);
                    *next_sample += 1.0 / weight;
                }
            }
        }

        // The current alpha is used for this whole epoch.  Updating after the
        // body intentionally makes the first due epoch use the initial rate.
        alpha = learning_rate * (1.0 - epoch_as_f64 / n_epochs as f64);
    }

    Ok(())
}

pub(crate) fn validate_csr_inputs(
    embedding: &[f32],
    n_components: usize,
    graph: &CsrGraph,
    a: impl Into<f64>,
    b: impl Into<f64>,
    learning_rate: impl Into<f64>,
) -> Result<(), LayoutError> {
    if n_components == 0 {
        return Err(LayoutError::InvalidShape {
            n_components,
            values: embedding.len(),
        });
    }
    let expected_values = graph.n_rows.checked_mul(n_components);
    if graph.n_cols != graph.n_rows || expected_values != Some(embedding.len()) {
        return Err(LayoutError::InvalidGraphShape {
            n_rows: graph.n_rows,
            n_cols: graph.n_cols,
            indptr: graph.indptr.len(),
            indices: graph.indices.len(),
            data: graph.data.len(),
        });
    }
    let expected_indptr = graph.n_rows.checked_add(1);
    if expected_indptr != Some(graph.indptr.len())
        || graph.indptr.first().copied() != Some(0)
        || graph.indices.len() != graph.data.len()
    {
        return Err(LayoutError::InvalidGraphShape {
            n_rows: graph.n_rows,
            n_cols: graph.n_cols,
            indptr: graph.indptr.len(),
            indices: graph.indices.len(),
            data: graph.data.len(),
        });
    }

    validate_parameters(a.into(), b.into(), learning_rate.into())?;
    if let Some((index, value)) = embedding
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LayoutError::NonFiniteValue {
            endpoint: "embedding",
            component: index,
            value,
        });
    }

    for row in 0..graph.n_rows {
        let start = graph.indptr[row];
        let end = graph.indptr[row + 1];
        if start > end || end > graph.indices.len() {
            return Err(LayoutError::InvalidGraphOffsets {
                row,
                start,
                end,
                entries: graph.indices.len(),
            });
        }

        let mut previous = None;
        for position in start..end {
            let index = graph.indices[position];
            if index >= graph.n_rows {
                return Err(LayoutError::InvalidGraphIndex {
                    row,
                    position,
                    index,
                    n_rows: graph.n_rows,
                });
            }
            if let Some(previous) = previous
                && index <= previous
            {
                return Err(LayoutError::InvalidGraphOrder {
                    row,
                    position,
                    previous,
                    current: index,
                });
            }
            previous = Some(index);

            let weight = graph.data[position];
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(LayoutError::InvalidGraphWeight {
                    row,
                    position,
                    weight,
                });
            }
        }
    }

    Ok(())
}

fn validate_parameters(a: f64, b: f64, learning_rate: f64) -> Result<(), LayoutError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "a",
            value: a as f32,
        });
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "b",
            value: b as f32,
        });
    }
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "learning_rate",
            value: learning_rate as f32,
        });
    }
    Ok(())
}

pub(crate) fn validate_repulsion_strength(repulsion_strength: f64) -> Result<(), LayoutError> {
    if !repulsion_strength.is_finite() || repulsion_strength < 0.0 {
        return Err(LayoutError::InvalidParameter {
            parameter: "repulsion_strength",
            value: repulsion_strength as f32,
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
    use super::{
        LayoutError, attractive_optimize_csr, attractive_update, attractive_update_pair,
        negative_example_update,
    };
    use crate::fuzzy_graph::CsrGraph;

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
    fn negative_example_update_matches_stabilized_gradient() {
        let mut current = [1.0_f32, 2.0];
        let other = [4.0_f32, 6.0];

        negative_example_update(&mut current, &other, 1.0_f32, 0.5_f32, 0.75_f32, 0.4).unwrap();

        // The squared distance is 25, so the regularized denominator is
        // 25.001 * (1 + 5). Neither component reaches the clipping limit.
        let step = 0.4 * (2.0 * 0.75 * 0.5 / (25.001 * 6.0));
        assert_eq!(
            current,
            [(1.0 - 3.0 * step) as f32, (2.0 - 4.0 * step) as f32]
        );
        assert_eq!(other, [4.0, 6.0]);
    }

    #[test]
    fn negative_example_zero_distance_is_a_no_op() {
        let mut current = [1.25_f32, -2.5];
        let original = current;
        let other = original;

        negative_example_update(&mut current, &other, 1.0, 0.25, 1.0, 10.0).unwrap();

        assert_eq!(current, original);
        assert_eq!(other, original);
    }

    #[test]
    fn negative_example_rejects_signed_or_nonfinite_inputs_before_modifying() {
        let mut current = [0.0_f32, 1.0];
        let original = current;
        let other = [2.0_f32, 3.0];

        let error = negative_example_update(&mut current, &other, 1.0, 1.0, -1.0, 0.1).unwrap_err();
        assert!(matches!(
            error,
            LayoutError::InvalidParameter {
                parameter: "repulsion_strength",
                ..
            }
        ));
        assert_eq!(current, original);

        let mut current = [0.0_f32, 1.0];
        let other = [f32::NAN, 3.0];
        let error = negative_example_update(&mut current, &other, 1.0, 1.0, 1.0, 0.1).unwrap_err();
        assert!(matches!(
            error,
            LayoutError::NonFiniteValue {
                endpoint: "other",
                component: 0,
                ..
            }
        ));
        assert_eq!(current, [0.0, 1.0]);
    }

    #[test]
    fn negative_example_checks_every_candidate_before_writing() {
        let mut current = [0.0_f32, 0.0];
        let error = negative_example_update(&mut current, &[0.0001, 1.0], 1.0, 1.0, 1.0, 1.0e40)
            .unwrap_err();

        assert!(matches!(
            error,
            LayoutError::NonFiniteUpdate {
                endpoint: "current",
                component: 1,
                ..
            }
        ));
        assert_eq!(current, [0.0, 0.0]);
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

    #[test]
    fn csr_sweep_uses_each_stored_entry_in_row_order() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![0.25, 0.75],
        };
        let mut embedding = [0.0_f32, 2.0];

        super::attractive_update_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1).unwrap();

        // The second directed entry sees the result of the first one; a CSR
        // sweep is deliberately in-place rather than a batch update.
        assert!((embedding[0] - 0.16391098).abs() < 1.0e-6);
        assert!((embedding[1] - 1.836089).abs() < 1.0e-6);
    }

    #[test]
    fn csr_sweep_rejects_bad_edges_before_modifying_embedding() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 2, 2],
            indices: vec![1, 1],
            data: vec![1.0, 1.0],
        };
        let mut embedding = [0.0_f32, 2.0];
        let original = embedding;

        let error =
            super::attractive_update_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1).unwrap_err();

        assert!(matches!(error, LayoutError::InvalidGraphOrder { .. }));
        assert_eq!(embedding, original);
    }

    #[test]
    fn csr_zero_membership_skips_an_edge_without_scaling_nonzero_edges() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![0.0, 0.25],
        };
        let mut embedding = [0.0_f32, 2.0];

        super::attractive_update_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1).unwrap();

        // Only the second entry is selected, and its 0.25 membership does not
        // reduce the fixed direct-sweep learning rate.
        assert!((embedding[0] - 0.08).abs() < 1.0e-6);
        assert!((embedding[1] - 1.92).abs() < 1.0e-6);
    }

    #[test]
    fn attractive_optimizer_delays_first_sample_and_decays_after_epoch() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 1],
            indices: vec![1],
            data: vec![1.0],
        };

        let mut one_epoch = [0.0_f32, 2.0];
        attractive_optimize_csr(&mut one_epoch, 1, &graph, 1.0, 1.0, 0.1, 1).unwrap();
        assert_eq!(one_epoch, [0.0, 2.0]);

        let mut scheduled = [0.0_f32, 2.0];
        attractive_optimize_csr(&mut scheduled, 1, &graph, 1.0, 1.0, 0.1, 3).unwrap();

        let mut expected = [0.0_f32, 2.0];
        attractive_update(&mut expected, 1, 0, 1, 1.0, 1.0, 0.1).unwrap();
        let decayed_rate = 0.1 * (1.0 - 1.0 / 3.0);
        attractive_update(&mut expected, 1, 0, 1, 1.0, 1.0, decayed_rate).unwrap();
        assert_eq!(scheduled, expected);
    }

    #[test]
    fn attractive_optimizer_uses_membership_as_sample_frequency() {
        let graph = CsrGraph {
            n_rows: 3,
            n_cols: 3,
            indptr: vec![0, 2, 2, 2],
            indices: vec![1, 2],
            data: vec![1.0, 0.5],
        };
        let mut embedding = [0.0_f32, 2.0, 4.0];

        // The unit edge is due at epoch one.  The half-weight edge is due at
        // epoch two, so it must remain untouched in a two-epoch run.
        attractive_optimize_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1, 2).unwrap();

        assert!((embedding[0] - 0.08).abs() < 1.0e-6);
        assert!((embedding[1] - 1.92).abs() < 1.0e-6);
        assert_eq!(embedding[2], 4.0);
    }

    #[test]
    fn attractive_optimizer_validates_before_modifying_embedding() {
        let graph = CsrGraph {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 2, 2],
            indices: vec![1, 1],
            data: vec![1.0, 1.0],
        };
        let mut embedding = [0.0_f32, 2.0];
        let original = embedding;

        let error =
            attractive_optimize_csr(&mut embedding, 1, &graph, 1.0, 1.0, 0.1, 2).unwrap_err();

        assert!(matches!(error, LayoutError::InvalidGraphOrder { .. }));
        assert_eq!(embedding, original);
    }
}
