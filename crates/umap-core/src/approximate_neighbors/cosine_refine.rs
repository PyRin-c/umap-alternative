use super::{DenseRows, MAX_CANDIDATES, TrainingNeighbor, cosine, offer_training_heap};
use crate::rng::next_negative_word;

pub(super) fn refine(
    data: DenseRows<'_>,
    ranks: &mut [Vec<TrainingNeighbor>],
    rng: &mut [i64; 3],
    rounds: usize,
    update_limit: usize,
) {
    let empty = [TrainingNeighbor::VACANT; MAX_CANDIDATES];
    let mut fresh = vec![empty; ranks.len()];
    let mut retained = vec![empty; ranks.len()];
    let mut live_old = Vec::with_capacity(MAX_CANDIDATES);
    for _ in 0..rounds {
        fresh.fill(empty);
        retained.fill(empty);
        for (row, rank) in ranks.iter_mut().enumerate() {
            for entry in rank {
                let other = entry.index();
                if other == usize::MAX {
                    continue;
                }
                let candidates = if entry.fresh {
                    &mut fresh
                } else {
                    &mut retained
                };
                offer_training_heap(
                    &mut candidates[row],
                    other,
                    next_negative_word(rng).unsigned_abs() as f32,
                );
                offer_training_heap(
                    &mut candidates[other],
                    row,
                    next_negative_word(rng).unsigned_abs() as f32,
                );
                entry.fresh = false;
            }
        }
        let mut changes = 0;
        for (new, old) in fresh.iter().zip(&retained) {
            // Only cross-checks against retained candidates compact away vacant slots.
            live_old.clear();
            live_old.extend(
                old.iter()
                    .copied()
                    .filter(|entry| entry.index() != usize::MAX),
            );
            for (position, entry) in new.iter().enumerate() {
                let left = entry.index();
                if left != usize::MAX {
                    changes += compare(data, ranks, left, &new[position + 1..]);
                    changes += compare(data, ranks, left, &live_old);
                }
            }
        }
        if changes <= update_limit {
            break;
        }
    }
    for rank in ranks {
        rank.retain(|entry| entry.index() != usize::MAX);
        rank.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    }
}

fn compare(
    data: DenseRows<'_>,
    ranks: &mut [Vec<TrainingNeighbor>],
    left: usize,
    candidates: &[TrainingNeighbor],
) -> usize {
    let mut changes = 0;
    let runs = candidates.split(|entry| entry.index() == usize::MAX || entry.index() == left);
    for run in runs {
        // A vacant heap slot ends the current run of four-candidate offers.
        let batched_len = run.len() / 4 * 4;
        for (i, entry) in run.iter().enumerate() {
            let right = entry.index();
            let distance = if i < batched_len {
                cosine::batched_priority(data.row(left), data.row(right))
            } else {
                cosine::priority(data.row(left), data.row(right))
            };
            changes += usize::from(offer_training_heap(&mut ranks[left], right, distance));
            changes += usize::from(offer_training_heap(&mut ranks[right], left, distance));
        }
    }
    changes
}
