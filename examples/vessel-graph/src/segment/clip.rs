// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The clip: a raw intensity range rescaled onto `0 ..= bins - 1`, and the three
//! masks of that range that the binarisation arms read.
//!
//! Everything here is a voxelwise map, so it is `ops::voxelwise` throughout:
//! a widen, a map or a predicate, and for the clipped volume a narrowing back
//! to `u16`. The masks are evaluated on the **raw** volume rather than on the
//! clipped one, because every sub-floor voxel and every voxel exactly at the
//! floor land in bin 0 and could no longer be told apart.

use blockflow::op::Chain;
use blockflow::ops::local::{Narrowing, Rounding};
use blockflow::ops::voxelwise::{NarrowOp, VoxelwiseMapOp, VoxelwiseMaskOp};
use blockflow::{Dtype, Error, Result};

/// The raw range a channel is clipped to and the number of bins it is spread
/// over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipParams {
    /// `(low, high)` in the source's own units. A voxel below `low` is "low",
    /// one at or above `high` is saturated.
    pub range: (f64, f64),
    /// The clipped volume holds bins `0 ..= bins - 1`.
    pub bins: u32,
}

impl Default for ClipParams {
    /// A 16-bit light-sheet vasculature channel: `(400, 60000)` over 4096 bins.
    fn default() -> Self {
        Self {
            range: (400.0, 60000.0),
            bins: 4096,
        }
    }
}

impl ClipParams {
    pub fn validate(&self) -> Result<()> {
        let (low, high) = self.range;
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(Error::invalid(format!(
                "clip: the range is ({low}, {high}); both ends must be finite with the low end \
                 strictly below the high end"
            )));
        }
        if self.bins < 2 || self.bins > 65536 {
            return Err(Error::invalid(format!(
                "clip: {} bins; the clipped volume is u16, so between 2 and 65536",
                self.bins
            )));
        }
        Ok(())
    }

    /// The top bin, `bins - 1`, as the value the clipped volume reaches.
    pub fn top(&self) -> f64 {
        f64::from(self.bins) - 1.0
    }

    /// One raw value to its bin, rounded to the nearest.
    pub fn map(&self, value: f64) -> f64 {
        let (low, high) = self.range;
        let clamped = value.clamp(low, high);
        ((clamped - low) * self.top() / (high - low)).round()
    }

    /// `value >= low && value < high`: the population the lightsheet, equalize
    /// and vessel-background estimates are taken over.
    pub fn in_range(&self) -> impl Fn(f64) -> bool + Copy + Send + Sync + 'static {
        let (low, high) = self.range;
        move |value| value >= low && value < high
    }

    /// The clipped volume, from an `f64` stream: map, then narrow to `u16`.
    pub fn clip_nodes(&self) -> Result<Vec<Chain>> {
        let params = *self;
        Ok(vec![
            Chain::op(VoxelwiseMapOp::new("clip", move |value: f64| {
                params.map(value)
            })),
            Chain::op(NarrowOp::new(
                "clip bins",
                Narrowing::new(Dtype::U16, Rounding::ToNearest)?,
            )),
        ])
    }

    /// `!(low || high)`, as a `Bool` image, from an `f64` stream.
    pub fn in_range_node(&self) -> Chain {
        Chain::op(VoxelwiseMaskOp::new("in clip range", self.in_range()))
    }

    /// `value >= low`: the median's population, which keeps saturated voxels.
    pub fn not_low_node(&self) -> Chain {
        let low = self.range.0;
        Chain::op(VoxelwiseMaskOp::new("not low", move |value: f64| {
            value >= low
        }))
    }

    /// `value >= high`: the saturated voxels, which seed the binary mask.
    pub fn high_node(&self) -> Chain {
        Chain::op(VoxelwiseMaskOp::at_or_above("clip high", self.range.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_map_spans_the_bins_and_clamps_outside_the_range() {
        let clip = ClipParams {
            range: (100.0, 1100.0),
            bins: 1001,
        };
        assert_eq!(clip.map(0.0), 0.0);
        assert_eq!(clip.map(100.0), 0.0);
        assert_eq!(clip.map(600.4), 500.0);
        assert_eq!(clip.map(600.6), 501.0);
        assert_eq!(clip.map(1100.0), 1000.0);
        assert_eq!(clip.map(5000.0), 1000.0);
    }

    #[test]
    fn the_masks_are_the_three_sides_of_the_range() {
        let clip = ClipParams {
            range: (10.0, 20.0),
            bins: 16,
        };
        let inside = clip.in_range();
        assert!(!inside(9.9) && inside(10.0) && inside(19.9) && !inside(20.0));
    }

    #[test]
    fn a_degenerate_range_is_refused() {
        for range in [(5.0, 5.0), (6.0, 5.0), (f64::NAN, 5.0)] {
            assert!(ClipParams { range, bins: 16 }.validate().is_err());
        }
        assert!(ClipParams {
            range: (0.0, 1.0),
            bins: 1
        }
        .validate()
        .is_err());
        assert!(ClipParams::default().validate().is_ok());
    }
}
