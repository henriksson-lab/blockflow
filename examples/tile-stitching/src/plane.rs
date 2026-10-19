// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The in-plane geometry of one overlapping pair: which rectangle of each tile
//! the correlation reads, which shifts it can reach, and how a shift maps onto
//! a lag of [`blockflow::ops::fft`].
//!
//! Both the rigid correlation and the per-slice measurement compare two planes
//! placed at nominal positions, over a range of shifts of the second. Only the
//! part of each plane that the other can reach under some shift matters, so the
//! planes are cut to that rectangle first: it is what the landscape is computed
//! on, and (for the per-slice measurement) what the normalisation statistics
//! are taken over. The shift range is then clamped to the shifts at which the
//! two cuts overlap at all, so no candidate in the landscape has an empty
//! overlap.

use blockflow::error::{Error, Result};
use blockflow::ops::fft::ShiftWindow;

/// The cut and the reachable shift window for two plane regions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneGeometry {
    /// Half-open local ranges into the first plane.
    pub cut_a: [(usize, usize); 2],
    /// And into the second.
    pub cut_b: [(usize, usize); 2],
    /// The lowest reachable shift per axis.
    pub shift_min: [isize; 2],
    /// One past the highest reachable shift per axis.
    pub shift_max: [isize; 2],
    /// `lag = shift + lag_offset`: a shift is measured between the two planes'
    /// nominal positions, a lag between the two cuts' own origins.
    pub lag_offset: [isize; 2],
}

impl PlaneGeometry {
    /// The geometry for two plane regions and an **inclusive** `[low, high]`
    /// shift range per axis.
    pub fn new(
        position_a: [isize; 2],
        shape_a: [usize; 2],
        position_b: [isize; 2],
        shape_b: [usize; 2],
        max_shifts: [[isize; 2]; 2],
    ) -> Result<Self> {
        let mut cut_a = [(0usize, 0usize); 2];
        let mut cut_b = [(0usize, 0usize); 2];
        let mut shift_min = [0isize; 2];
        let mut shift_max = [0isize; 2];
        let mut lag_offset = [0isize; 2];
        for axis in 0..2 {
            let (low, high) = (max_shifts[axis][0], max_shifts[axis][1]);
            if high < low {
                return Err(Error::invalid(format!(
                    "the shift range on plane axis {axis} runs from {low} to {high}, which is empty"
                )));
            }
            let (pa, pb) = (position_a[axis], position_b[axis]);
            let (sa, sb) = (shape_a[axis] as isize, shape_b[axis] as isize);
            // The second plane swept over the whole range, intersected with the
            // first, and the mirror image.
            let a_low = pa.max(pb + low);
            let a_high = (pa + sa).min(pb + sb + high);
            let b_low = pb.max(pa - high);
            let b_high = (pb + sb).min(pa + sa - low);
            if a_high <= a_low || b_high <= b_low {
                return Err(Error::invalid(format!(
                    "the two tiles never overlap on plane axis {axis} within the shift range; \
                     widen the shift range or correct the nominal positions"
                )));
            }
            cut_a[axis] = ((a_low - pa) as usize, (a_high - pa) as usize);
            cut_b[axis] = ((b_low - pb) as usize, (b_high - pb) as usize);
            let low = low.max(a_low - b_high + 1);
            let high = (high + 1).min(a_high - b_low);
            if high <= low {
                return Err(Error::invalid(format!(
                    "plane axis {axis}'s shift range is empty once clamped to the overlap"
                )));
            }
            shift_min[axis] = low;
            shift_max[axis] = high;
            lag_offset[axis] = b_low - a_low;
        }
        Ok(Self {
            cut_a,
            cut_b,
            shift_min,
            shift_max,
            lag_offset,
        })
    }

    /// The shape of a landscape over the reachable shifts.
    pub fn landscape_shape(&self) -> [usize; 2] {
        std::array::from_fn(|axis| (self.shift_max[axis] - self.shift_min[axis]) as usize)
    }

    pub fn cut_shape_a(&self) -> [usize; 2] {
        std::array::from_fn(|axis| self.cut_a[axis].1 - self.cut_a[axis].0)
    }

    pub fn cut_shape_b(&self) -> [usize; 2] {
        std::array::from_fn(|axis| self.cut_b[axis].1 - self.cut_b[axis].0)
    }

    /// The lag window the landscape is planned over. Its index `[i, j]` is the
    /// shift `shift_min + [i, j]`.
    pub fn window(&self) -> Result<ShiftWindow> {
        ShiftWindow::new(
            std::array::from_fn(|axis| self.shift_min[axis] + self.lag_offset[axis]),
            std::array::from_fn(|axis| self.shift_max[axis] - 1 + self.lag_offset[axis]),
        )
    }

    /// The shift a landscape index names.
    pub fn shift_at(&self, index: [usize; 2]) -> [isize; 2] {
        std::array::from_fn(|axis| self.shift_min[axis] + index[axis] as isize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The overlap of a 404-wide plane and one placed at 338, searched over
    /// +/-30: the cut is the reachable 96 columns of each, and every reachable
    /// shift keeps some overlap.
    #[test]
    fn the_cut_is_what_the_shift_range_can_reach() {
        let geometry = PlaneGeometry::new(
            [0, 0],
            [404, 100],
            [338, 0],
            [404, 100],
            [[-30, 30], [-5, 5]],
        )
        .expect("a geometry");
        assert_eq!(geometry.cut_a[0], (308, 404));
        assert_eq!(geometry.cut_b[0], (0, 96));
        assert_eq!(geometry.shift_min, [-30, -5]);
        assert_eq!(geometry.shift_max, [31, 6]);
        assert_eq!(geometry.landscape_shape(), [61, 11]);
        assert_eq!(geometry.lag_offset, [30, 0]);
    }

    /// A search wider than the planes is clamped to the shifts that overlap.
    #[test]
    fn a_search_wider_than_the_plane_is_clamped() {
        let geometry =
            PlaneGeometry::new([0, 0], [20, 18], [0, 0], [20, 18], [[-30, 30], [-20, 20]])
                .expect("a geometry");
        assert_eq!(geometry.shift_min, [-19, -17]);
        assert_eq!(geometry.shift_max, [20, 18]);
    }
}
