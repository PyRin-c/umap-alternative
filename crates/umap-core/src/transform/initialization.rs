//! Initial layouts computed from supplied graph rows or neighbor ranks.

use std::fmt;

use crate::numeric::{pairwise_sum, pairwise_sum_by};

/// A shape or fitted row index is invalid for neighborhood interpolation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeighborhoodInitializationError {
    Shape,
    Index,
}

impl fmt::Display for NeighborhoodInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => formatter.write_str("incompatible transform initialization shapes"),
            Self::Index => formatter.write_str("neighbor index is outside the fitted embedding"),
        }
    }
}

impl std::error::Error for NeighborhoodInitializationError {}

use NeighborhoodInitializationError as Error;

/// Form weighted sums of fitted coordinates in the supplied rank order.
///
/// Weights are used as supplied. `add_product` implements the multiplication and
/// accumulation precision for one input dtype pair. Negative neighbor indices
/// address rows from the end of the fitted matrix.
pub fn initialize_ranked_transform<W: Copy, C: Copy>(
    indices: &[i64],
    shape: (usize, usize),
    weights: &[W],
    embedding: (&[C], (usize, usize)),
    add_product: impl Fn(f32, W, C) -> f32,
) -> Result<Vec<f32>, Error> {
    let (n_rows, width) = shape;
    let (embedding, (fitted_rows, components)) = embedding;
    if n_rows.checked_mul(width) != Some(indices.len())
        || indices.len() != weights.len()
        || fitted_rows.checked_mul(components) != Some(embedding.len())
    {
        return Err(Error::Shape);
    }
    let mut coordinates = vec![0.0; output_len(n_rows, components)?];
    for row in 0..n_rows {
        let out = &mut coordinates[row * components..(row + 1) * components];
        for entry in row * width..(row + 1) * width {
            let index = indices[entry];
            let index = if index >= 0 {
                usize::try_from(index)
                    .ok()
                    .filter(|&index| index < fitted_rows)
            } else {
                usize::try_from(index.unsigned_abs())
                    .ok()
                    .and_then(|back| fitted_rows.checked_sub(back))
            }
            .ok_or(Error::Index)?;
            let fitted = &embedding[index * components..(index + 1) * components];
            for (coordinate, &value) in out.iter_mut().zip(fitted) {
                *coordinate = add_product(*coordinate, weights[entry], value);
            }
        }
    }
    Ok(coordinates)
}

/// Initialize from positive CSR memberships with distinct sorted columns.
///
/// Empty rows become NaN. A unit membership copies the first matching fitted
/// row. Otherwise the sum adds the first weight after the pairwise tail sum,
/// and the sparse product accumulates in reverse column order.
/// `precision` selects float32 arithmetic for normalization and for the
/// normalized sparse product, respectively. Inputs carry their original
/// values in float64, and the output is float32 in either mode.
pub fn initialize_csr_transform(
    graph: (&[usize], &[usize], &[f64], usize),
    embedding: (&[f64], (usize, usize)),
    precision: (bool, bool),
) -> Result<Vec<f32>, Error> {
    let (offsets, indices, weights, graph_columns) = graph;
    let (embedding, (fitted_rows, components)) = embedding;
    let (single_weights, single_product) = precision;
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&weights.len())
        || !offsets.is_sorted()
        || indices.len() != weights.len()
        || fitted_rows.checked_mul(components) != Some(embedding.len())
    {
        return Err(Error::Shape);
    }
    let mut coordinates = vec![f32::NAN; output_len(offsets.len() - 1, components)?];
    let mut wide = vec![0.0; components];
    let fitted = |index: usize| {
        if index >= fitted_rows {
            return Err(Error::Index);
        }
        Ok(&embedding[index * components..(index + 1) * components])
    };
    for (row, bounds) in offsets.windows(2).enumerate() {
        let weights = &weights[bounds[0]..bounds[1]];
        let Some((&first, tail)) = weights.split_first() else {
            continue;
        };
        let columns = &indices[bounds[0]..bounds[1]];
        let out = &mut coordinates[row * components..(row + 1) * components];
        if let Some(index) = weights.iter().position(|&weight| weight == 1.0) {
            for (coordinate, &value) in out.iter_mut().zip(fitted(columns[index])?) {
                *coordinate = value as f32;
            }
            continue;
        }
        if fitted_rows != graph_columns {
            return Err(Error::Shape);
        }
        let scale = if single_weights {
            f64::from((pairwise_sum_by(tail, |&weight| weight as f32) + first as f32).recip())
        } else {
            (pairwise_sum(tail) + first).recip()
        };
        out.fill(0.0);
        wide.fill(0.0);
        for (&column, &weight) in columns.iter().zip(weights).rev() {
            let weight = if single_weights {
                f64::from(weight as f32 * scale as f32)
            } else {
                weight * scale
            };
            if weight == 0.0 {
                continue;
            }
            let fitted = fitted(column)?;
            if single_product {
                for (coordinate, &value) in out.iter_mut().zip(fitted) {
                    *coordinate += weight as f32 * value as f32;
                }
            } else {
                for (coordinate, &value) in wide.iter_mut().zip(fitted) {
                    *coordinate += weight * value;
                }
            }
        }
        if !single_product {
            for (coordinate, &value) in out.iter_mut().zip(&wide) {
                *coordinate = value as f32;
            }
        }
    }
    Ok(coordinates)
}

fn output_len(rows: usize, components: usize) -> Result<usize, Error> {
    rows.checked_mul(components)
        .filter(|&size| size <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or(Error::Shape)
}
