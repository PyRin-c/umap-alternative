use std::ops::{Add, AddAssign};

/// Sort indices with the median-pivot order observed in the reference.
pub(crate) fn sort_indices(mut order: &mut [usize], less: impl Fn(usize, usize) -> bool + Copy) {
    // The reference uses insertion sort through size 15. Keep equal values
    // in the observed partition order for larger buffers.
    while order.len() > 15 {
        let last = order.len() - 1;
        let mid = last / 2;
        for (left, right) in [(0, mid), (mid, last), (0, mid)] {
            if less(order[right], order[left]) {
                order.swap(left, right);
            }
        }
        order.swap(mid, last);
        let pivot = order[last];
        let (mut left, mut right) = (0, last);
        while left < right {
            while less(order[left], pivot) {
                left += 1;
            }
            while left < right && less(pivot, order[right - 1]) {
                right -= 1;
            }
            if left + 1 >= right {
                break;
            }
            order.swap(left, right - 1);
            left += 1;
            right -= 1;
        }
        order.swap(left, last);
        let (prefix, suffix) = order.split_at_mut(left);
        let suffix = &mut suffix[1..];
        // Recurse on the smaller side to bound stack depth for uneven partitions.
        if prefix.len() < suffix.len() {
            sort_indices(prefix, less);
            order = suffix;
        } else {
            sort_indices(suffix, less);
            order = prefix;
        }
    }
    for index in 1..order.len() {
        let id = order[index];
        let mut position = index;
        while position > 0 && less(id, order[position - 1]) {
            order[position] = order[position - 1];
            position -= 1;
        }
        order[position] = id;
    }
}

// Contiguous reductions combine complete groups of eight before adding any tail.
pub(crate) fn pairwise_sum<T>(values: &[T]) -> T
where
    T: Copy + From<i8> + Add<Output = T> + AddAssign,
{
    pairwise_sum_by(values, |&value| value)
}

pub(crate) fn pairwise_sum_by<S, T>(values: &[S], value: impl Fn(&S) -> T + Copy) -> T
where
    T: Copy + From<i8> + Add<Output = T> + AddAssign,
{
    if values.len() < 8 {
        return values.iter().map(value).fold(T::from(0), Add::add);
    }
    if values.len() > 128 {
        let (left, right) = values.split_at((values.len() / 2) & !7);
        return pairwise_sum_by(left, value) + pairwise_sum_by(right, value);
    }
    let (first, rest) = values
        .split_first_chunk::<8>()
        .expect("at least eight values");
    let mut lanes = std::array::from_fn(|index| value(&first[index]));
    let (blocks, tail) = rest.as_chunks::<8>();
    for block in blocks {
        for (lane, item) in lanes.iter_mut().zip(block) {
            *lane += value(item);
        }
    }
    let [a, b, c, d, e, f, g, h] = lanes;
    tail.iter()
        .map(value)
        .fold(((a + b) + (c + d)) + ((e + f) + (g + h)), Add::add)
}
