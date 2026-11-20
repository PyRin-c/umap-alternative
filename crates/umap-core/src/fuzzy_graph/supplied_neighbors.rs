use std::{borrow::Cow, fmt};

use super::{
    CsrGraph, GraphOptions, MIN_SCALE_MULTIPLIER, batch_scale_floor, fuzzy_union, local_offset,
    mean_distance, membership, search_scale, symmetrize,
};

/// Invalid input to [`fit_supplied_neighbors_graph`].
#[derive(Clone, Debug, PartialEq)]
pub enum SuppliedNeighborsGraphError {
    /// At least one row is required and both arrays must contain `n_rows * width` elements.
    InvalidShape {
        n_rows: usize,
        width: usize,
        indices: usize,
        distances: usize,
    },
    /// The requested neighbor count must be in `1..=width`.
    InvalidNeighborCount { width: usize, k: usize },
    /// Each pair must be a valid index and finite non-negative distance or `(-1, +inf)`.
    InvalidNeighbor {
        row: usize,
        position: usize,
        index: i32,
        distance: f32,
    },
    /// The disconnection distance must be non-negative and may be positive infinity.
    InvalidDisconnectionDistance { distance: f32 },
}

impl fmt::Display for SuppliedNeighborsGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_rows,
                width,
                indices,
                distances,
            } => write!(
                formatter,
                "supplied neighbors need two {n_rows} x {width} arrays with at least one row, got {indices} indices and {distances} distances"
            ),
            Self::InvalidNeighborCount { width, k } => write!(
                formatter,
                "requested neighbor count {k} is outside 1..={width}"
            ),
            Self::InvalidNeighbor {
                row,
                position,
                index,
                distance,
            } => write!(
                formatter,
                "supplied neighbor at row {row}, column {position} has invalid pair ({index}, {distance})"
            ),
            Self::InvalidDisconnectionDistance { distance } => write!(
                formatter,
                "disconnection distance {distance} must be non-negative"
            ),
        }
    }
}

impl std::error::Error for SuppliedNeighborsGraphError {}

/// Fit canonical CSR from supplied row-major nearest-neighbor arrays.
///
/// Supply two buffers of shape `n_rows x width` with at least one row. Each
/// pair is either a zero-based row index with a finite, non-negative distance
/// or the missing marker `(-1, f32::INFINITY)`. Column order is significant.
/// Self indices may occur anywhere or be absent. Every self index is omitted
/// from the graph but its distance still participates in scale estimation.
///
/// `k` is the requested neighbor count used for the `log2(k)` scale target.
/// The supplied arrays can have more columns than rows, and `k` may exceed
/// `n_rows`.
/// Every column in `width` contributes to local connectivity and can emit
/// an edge. The scale search omits column zero regardless of its index.
/// Local connectivity uses the positive finite distances in supplied order.
/// Missing distances do not contribute to the scale search or mean floors.
///
/// Finite distances at or above `disconnection_distance` are treated as
/// missing before estimating scales or computing memberships. Pass
/// `f32::INFINITY` when the caller has already applied its cutoff or when
/// there is no finite cutoff. The buffers are borrowed immutably.
///
/// Repeated directed coordinates are summed in column order before applying
/// the fuzzy set operation. Duplicate sums can yield weights outside
/// `[0, 1]`. The result is symmetric, omits diagonal and zero entries, and
/// stores columns in strictly ascending order in each row. Public wheel
/// results for wide rows with many duplicates can round asymmetrically.
///
/// # Errors
///
/// Returns [`SuppliedNeighborsGraphError`] for a mismatched or overflowing
/// shape, invalid requested count or cutoff, or an invalid index-distance
/// pair, even if that pair would have been cut. Callers resolve estimator
/// validation, fallback, cutoff defaults, array
/// mutation and search-index adapters before or around this graph stage.
pub fn fit_supplied_neighbors_graph(
    indices: &[i32],
    distances: &[f32],
    n_rows: usize,
    width: usize,
    k: usize,
    options: GraphOptions,
    disconnection_distance: f32,
) -> Result<CsrGraph, SuppliedNeighborsGraphError> {
    use SuppliedNeighborsGraphError as Error;

    if n_rows == 0
        || n_rows.checked_mul(width) != Some(indices.len())
        || indices.len() != distances.len()
    {
        return Err(Error::InvalidShape {
            n_rows,
            width,
            indices: indices.len(),
            distances: distances.len(),
        });
    }
    if !(1..=width).contains(&k) {
        return Err(Error::InvalidNeighborCount { width, k });
    }
    if !(0.0..=f32::INFINITY).contains(&disconnection_distance) {
        return Err(Error::InvalidDisconnectionDistance {
            distance: disconnection_distance,
        });
    }
    for (entry, (&index, &distance)) in indices.iter().zip(distances).enumerate() {
        let valid = if index == -1 {
            distance == f32::INFINITY
        } else {
            usize::try_from(index).is_ok_and(|index| index < n_rows)
                && (0.0..=f32::MAX).contains(&distance)
        };
        if !valid {
            return Err(Error::InvalidNeighbor {
                row: entry / width,
                position: entry % width,
                index,
                distance,
            });
        }
    }

    let distances = if disconnection_distance.is_finite()
        && distances
            .iter()
            .any(|&distance| distance.is_finite() && distance >= disconnection_distance)
    {
        Cow::Owned(
            distances
                .iter()
                .map(|&distance| {
                    if distance >= disconnection_distance {
                        f32::INFINITY
                    } else {
                        distance
                    }
                })
                .collect(),
        )
    } else {
        Cow::Borrowed(distances)
    };

    Ok(build_ranked_graph(
        indices.iter().map(|&index| index as usize),
        &distances,
        n_rows,
        width,
        k,
        options,
    ))
}

// The caller validates the aligned flat buffers. Infinite distances mark
// missing entries, and position zero is omitted from the scale search.
pub(crate) fn build_ranked_graph(
    indices: impl Iterator<Item = usize>,
    distances: &[f32],
    n_rows: usize,
    width: usize,
    k: usize,
    options: GraphOptions,
) -> CsrGraph {
    let target = (k as f64).log2();
    let batch_floor = with_finite_distances(distances, |finite| {
        if finite.is_empty() {
            0.0
        } else {
            batch_scale_floor(finite)
        }
    });
    let mut directed = Vec::with_capacity(n_rows);
    let mut positive = Vec::with_capacity(width);
    let mut offsets = Vec::with_capacity(width - 1);
    let mut indices = indices;
    for (row, distances) in distances.chunks_exact(width).enumerate() {
        positive.clear();
        positive.extend(
            distances
                .iter()
                .copied()
                .filter(|&distance| distance > 0.0 && distance.is_finite()),
        );
        let rho = local_offset(&positive, options.local_connectivity);
        offsets.clear();
        offsets.extend(
            distances[1..]
                .iter()
                .filter(|distance| distance.is_finite())
                .map(|&distance| (distance - rho).max(0.0)),
        );
        let floor = if rho > 0.0 {
            with_finite_distances(distances, |finite| {
                (MIN_SCALE_MULTIPLIER * mean_distance(finite)) as f32
            })
        } else {
            batch_floor
        };
        let sigma = search_scale(&offsets, target, false).max(floor);

        let mut memberships = indices
            .by_ref()
            .take(width)
            .zip(distances)
            .filter(|&(index, &distance)| distance.is_finite() && index != row)
            .map(|(index, &distance)| (index, membership(distance, rho, sigma)))
            .collect::<Vec<_>>();
        // Stable ordering keeps duplicate summation in the original column order.
        memberships.sort_by_key(|&(index, _)| index);
        memberships.dedup_by(|(index, weight), (previous, sum)| {
            if index == previous {
                *sum += *weight;
                true
            } else {
                false
            }
        });
        directed.push(memberships);
    }

    let mix = options.set_op_mix_ratio as f32;
    let intersection_mix = (1.0 - options.set_op_mix_ratio) as f32;
    symmetrize(&directed, false, |forward, reverse| {
        let union = fuzzy_union(forward, reverse);
        let intersection = forward * reverse;
        mix * union + intersection_mix * intersection
    })
}

fn with_finite_distances(distances: &[f32], calculate: impl FnOnce(&[f32]) -> f32) -> f32 {
    if distances.iter().all(|distance| distance.is_finite()) {
        calculate(distances)
    } else {
        let finite = distances
            .iter()
            .copied()
            .filter(|distance| distance.is_finite())
            .collect::<Vec<_>>();
        calculate(&finite)
    }
}
