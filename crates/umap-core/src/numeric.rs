use std::ops::{Add, AddAssign};

// Contiguous reductions combine complete groups of eight before adding any tail.
pub(crate) fn pairwise_sum<T>(values: &[T]) -> T
where
    T: Copy + From<i8> + Add<Output = T> + AddAssign,
{
    if values.len() < 8 {
        return values.iter().copied().fold(T::from(0), Add::add);
    }
    if values.len() > 128 {
        let (left, right) = values.split_at((values.len() / 2) & !7);
        return pairwise_sum(left) + pairwise_sum(right);
    }
    let (first, rest) = values
        .split_first_chunk::<8>()
        .expect("at least eight values");
    let mut lanes = *first;
    let (blocks, tail) = rest.as_chunks::<8>();
    for block in blocks {
        for (lane, &value) in lanes.iter_mut().zip(block) {
            *lane += value;
        }
    }
    let [a, b, c, d, e, f, g, h] = lanes;
    tail.iter()
        .copied()
        .fold(((a + b) + (c + d)) + ((e + f) + (g + h)), Add::add)
}
