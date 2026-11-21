use crate::layout::squared_delta_2d;

/// Two-coordinate updates for an already validated, density-free layout.
#[derive(Clone, Copy)]
pub(super) struct SeededLayout2d {
    a: f64,
    b: f64,
    attraction_numerator: f64,
    repulsion_numerator: f64,
    has_repulsion: bool,
}

impl SeededLayout2d {
    pub(super) fn new(a: f64, b: f64, repulsion_strength: f64) -> Option<Self> {
        // A finite nonzero f32 squared distance is in [2^-149, 2^128).
        // These bounds keep the powers, numerators, and denominators finite
        // in f64 over that entire range, including for b - 1 in [-1, 3].
        (a <= 65_536.0 && b <= 4.0 && repulsion_strength <= 65_536.0).then_some(Self {
            a,
            b,
            attraction_numerator: 2.0 * a * b,
            repulsion_numerator: 2.0 * repulsion_strength * b,
            has_repulsion: repulsion_strength != 0.0,
        })
    }

    // A false return leaves both rows untouched for the checked kernel to
    // report the same error and component as on the general path.
    #[inline(always)]
    pub(super) fn attract(
        self,
        embedding: &mut [f32],
        head: usize,
        tail: usize,
        alpha: f64,
    ) -> bool {
        if head == tail {
            return true;
        }
        let (rows, _) = embedding.as_chunks_mut::<2>();
        let [hx, hy] = rows[head];
        let [tx, ty] = rows[tail];
        let (dx, dy) = (tx - hx, ty - hy);
        let distance = squared_delta_2d(dx, dy);
        if !distance.is_finite() {
            return false;
        }
        if distance == 0.0 || alpha == 0.0 {
            return true;
        }
        let q = f64::from(distance);
        let denominator = 1.0 + self.a * q.powf(self.b);
        let scale = (self.attraction_numerator * q.powf(self.b - 1.0)) / denominator;
        let step_x = displacement(dx, scale, alpha);
        let step_y = displacement(dy, scale, alpha);
        let new_hx = (f64::from(hx) + step_x) as f32;
        let new_hy = (f64::from(hy) + step_y) as f32;
        let new_tx = (f64::from(tx) - step_x) as f32;
        let new_ty = (f64::from(ty) - step_y) as f32;
        if !(new_hx.is_finite() && new_hy.is_finite() && new_tx.is_finite() && new_ty.is_finite()) {
            return false;
        }
        rows[head] = [new_hx, new_hy];
        rows[tail] = [new_tx, new_ty];
        true
    }

    #[inline(always)]
    pub(super) fn repel(self, embedding: &mut [f32], head: usize, tail: usize, alpha: f64) -> bool {
        let (rows, _) = embedding.as_chunks_mut::<2>();
        let [hx, hy] = rows[head];
        let [tx, ty] = rows[tail];
        let (dx, dy) = (hx - tx, hy - ty);
        let distance = squared_delta_2d(dx, dy);
        if !distance.is_finite() {
            return false;
        }
        if distance == 0.0 || !self.has_repulsion || alpha == 0.0 {
            return true;
        }
        let q = f64::from(distance);
        let denominator = (0.001 + q) * (1.0 + self.a * q.powf(self.b));
        let scale = self.repulsion_numerator / denominator;
        let new_x = (f64::from(hx) + displacement(dx, scale, alpha)) as f32;
        let new_y = (f64::from(hy) + displacement(dy, scale, alpha)) as f32;
        if !(new_x.is_finite() && new_y.is_finite()) {
            return false;
        }
        rows[head] = [new_x, new_y];
        true
    }
}

#[inline(always)]
fn displacement(delta: f32, scale: f64, alpha: f64) -> f64 {
    alpha * (scale * f64::from(delta)).clamp(-4.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::SeededLayout2d;
    use crate::layout::{
        DistanceReduction, attractive_update_with_distance, negative_update_fixed_tail,
    };

    #[test]
    fn packed_updates_preserve_checked_coordinate_bits() {
        for b in [f64::MIN_POSITIVE, 0.25, 0.875, 1.0, 4.0] {
            for repulsion in [0.0, f64::MIN_POSITIVE, 1.0] {
                let layout = SeededLayout2d::new(1.5, b, repulsion).unwrap();
                let alpha = 2.25;
                for input in [
                    [-0.0, 1.0, 4.5, -0.5],
                    [0.0, -0.0, -0.0, 0.0],
                    [0.125, -20.0, 0.125, 16_384.0],
                    [f32::from_bits(1), 0.00001, -f32::from_bits(1), 0.00002],
                ] {
                    let mut checked = input;
                    let mut packed = input;
                    attractive_update_with_distance(
                        &mut checked,
                        2,
                        0,
                        1,
                        1.5,
                        b,
                        alpha,
                        DistanceReduction::SeededCompatibility,
                        |_, _| Ok(None),
                    )
                    .unwrap();
                    assert!(layout.attract(&mut packed, 0, 1, alpha));
                    assert_eq!(packed.map(f32::to_bits), checked.map(f32::to_bits));
                    let (head, tail) = checked.split_at_mut(2);
                    negative_update_fixed_tail(head, tail, 1.5, b, repulsion, alpha).unwrap();
                    assert!(layout.repel(&mut packed, 0, 1, alpha));
                    assert_eq!(packed.map(f32::to_bits), checked.map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn packed_updates_leave_rows_for_the_checked_overflow_path() {
        let layout = SeededLayout2d::new(1.0, 1.0, 1.0).unwrap();
        let mut input = [f32::MAX, 0.0, -f32::MAX, 0.0];
        let original = input;
        assert!(!layout.attract(&mut input, 0, 1, 1.0));
        assert!(!layout.repel(&mut input, 0, 1, 1.0));
        assert_eq!(input, original);

        let mut input = [0.125, 1.0, 0.0, 0.0];
        let original = input;
        assert!(!layout.attract(&mut input, 0, 1, 1.0e40));
        assert!(!layout.repel(&mut input, 0, 1, 1.0e40));
        assert_eq!(input, original);
        assert!(SeededLayout2d::new(1.0, 5.0, 1.0).is_none());
    }
}
