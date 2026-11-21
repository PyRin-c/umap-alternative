use super::{CsrFeatures, CsrRow, canonical_euclidean_distance};
use crate::neighbors::Neighbor;

#[derive(Clone, Copy, Default)]
struct Posting {
    row: usize,
    value: f64,
}

/// Inverted index for validated CSR with unique, ascending column indices.
pub(super) struct SparseEuclideanIndex<'a> {
    training: CsrFeatures<'a>,
    offsets: Vec<usize>,
    postings: Vec<Posting>,
    norms: Vec<f64>,
    products: Vec<f64>,
    max_row_entries: usize,
}

impl<'a> SparseEuclideanIndex<'a> {
    pub(super) fn new(training: CsrFeatures<'a>, query_rows: usize) -> Option<Self> {
        const MAX_INDEX_BYTES: usize = 8 * 1024 * 1024;
        let (rows, columns) = training.shape;
        // Use the merge kernel for small batches and densely populated CSR.
        if rows < 32
            || query_rows < 8
            || training.data.len().checked_mul(4)? >= rows.checked_mul(columns)?
        {
            return None;
        }
        let offset_count = columns.checked_add(1)?;
        let bytes = offset_count
            .checked_mul(2 * std::mem::size_of::<usize>())?
            .checked_add(
                training
                    .data
                    .len()
                    .checked_mul(std::mem::size_of::<Posting>())?,
            )?
            .checked_add(rows.checked_mul(2 * std::mem::size_of::<f64>())?)?;
        if bytes > MAX_INDEX_BYTES {
            return None;
        }

        let mut offsets = vec![0; offset_count];
        for &column in training.indices {
            offsets[column + 1] += 1;
        }
        for column in 0..columns {
            offsets[column + 1] += offsets[column];
        }
        let mut next = offsets[..columns].to_vec();
        let mut postings = vec![Posting::default(); training.data.len()];
        let mut norms = Vec::with_capacity(rows);
        let mut max_row_entries = 0;
        for row in 0..rows {
            let features = training.row(row);
            max_row_entries = max_row_entries.max(features.values.len());
            norms.push(squared_norm(features.values));
            for (&column, &value) in features.columns.iter().zip(features.values) {
                postings[next[column]] = Posting {
                    row,
                    value: f64::from(value),
                };
                next[column] += 1;
            }
        }
        Some(Self {
            training,
            offsets,
            postings,
            norms,
            products: vec![0.0; rows],
            max_row_entries,
        })
    }

    pub(super) fn fill(&mut self, query: CsrRow<'_>, candidates: &mut Vec<Neighbor>) {
        self.products.fill(0.0);
        for (&column, &value) in query.columns.iter().zip(query.values) {
            let value = f64::from(value);
            for posting in &self.postings[self.offsets[column]..self.offsets[column + 1]] {
                self.products[posting.row] += value * posting.value;
            }
        }

        let query_norm = squared_norm(query.values);
        let max_terms = query.values.len() + self.max_row_entries;
        // Products of two f32 values are exact in f64. If S is the sum of
        // the two exact norms, the absolute sum of dot product terms is at
        // most S/2, and the exact squared distance is at most 2*S. Each sum
        // uses at most max_terms additions. Bound the errors from both sum
        // orders and the final arithmetic, including rounding the interval
        // endpoints. The index size cap keeps max_terms * EPSILON small.
        let relative_error = (max_terms as f64 + 8.0) * (32.0 * f64::EPSILON);
        candidates.clear();
        candidates.extend(self.norms.iter().zip(&self.products).enumerate().map(
            |(index, (&norm, &product))| {
                let norm_sum = query_norm + norm;
                let squared = norm_sum - 2.0 * product;
                let error = relative_error * norm_sum;
                let lower = (squared - error).max(0.0) as f32;
                let upper = (squared + error) as f32;
                // Use the shortcut only if the whole interval rounds to
                // the same float32 as the merge kernel's squared sum.
                let distance = if lower.to_bits() == upper.to_bits() {
                    upper.sqrt()
                } else {
                    canonical_euclidean_distance(query, self.training.row(index))
                };
                Neighbor { index, distance }
            },
        ));
    }
}

fn squared_norm(values: &[f32]) -> f64 {
    values.iter().fold(0.0, |sum, &value| {
        let value = f64::from(value);
        sum + value * value
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_distance_keeps_round_to_even_at_the_float32_midpoint() {
        let mut indptr = [2; 33];
        indptr[0] = 0;
        let training = CsrFeatures {
            indptr: &indptr,
            indices: &[0, 1],
            data: &[1.0, 1.0 / 4096.0],
            shape: (32, 8),
        };
        let mut index = SparseEuclideanIndex::new(training, 32).unwrap();
        let mut candidates = Vec::new();
        index.fill(training.row(1), &mut candidates);
        // The exact squared sum is 1 + 2^-24, halfway to the next f32.
        assert_eq!(candidates[0].distance.to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn indexed_distances_match_merging_through_cancellation_and_overflow() {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in 0..64 {
            indices.extend([0, 1, 2 + row, 126, 127]);
            data.extend([
                1.0e20,
                row as f32 * 0.125,
                if row % 2 == 0 {
                    -0.0
                } else {
                    f32::from_bits(1)
                },
                f32::from_bits(((row + 4) as u32) << 23),
                if row < 32 { -1.0 } else { f32::MAX },
            ]);
            indptr.push(data.len());
        }
        let training = CsrFeatures {
            indptr: &indptr,
            indices: &indices,
            data: &data,
            shape: (64, 128),
        };
        let mut index = SparseEuclideanIndex::new(training, 64).unwrap();
        let mut distances = Vec::new();
        for row in 0..training.shape.0 {
            let query = training.row(row);
            index.fill(query, &mut distances);
            assert_eq!(distances.len(), training.shape.0);
            for (column, candidate) in distances.iter().enumerate() {
                let expected = canonical_euclidean_distance(query, training.row(column));
                assert_eq!(candidate.index, column);
                assert_eq!(
                    candidate.distance.to_bits(),
                    expected.to_bits(),
                    "{row}, {column}"
                );
            }
        }
        index.fill(training.row(0), &mut distances);
        assert_eq!(distances[1].distance, 0.125);
        assert_eq!(distances[32].distance, f32::INFINITY);
    }

    #[test]
    fn indexed_distances_match_the_merge_kernel_for_mixed_sparse_outer_products() {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut bits = 17_u32;
        for _ in 0..80 {
            for column in 0..256 {
                bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if bits >> 28 == 0 {
                    indices.push(column);
                    data.push(((bits & 0xfff) as i32 - 2048) as f32 * 0.03125);
                }
            }
            indptr.push(data.len());
        }
        let training = CsrFeatures {
            indptr: &indptr,
            indices: &indices,
            data: &data,
            shape: (80, 256),
        };
        let mut index = SparseEuclideanIndex::new(training, 80).unwrap();
        let mut candidates = Vec::new();
        for row in 0..training.shape.0 {
            index.fill(training.row(row), &mut candidates);
            for (column, candidate) in candidates.iter().enumerate() {
                assert_eq!(
                    candidate.distance.to_bits(),
                    canonical_euclidean_distance(training.row(row), training.row(column)).to_bits(),
                );
            }
        }
    }
}
