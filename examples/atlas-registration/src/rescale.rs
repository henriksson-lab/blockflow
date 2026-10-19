// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Axis-wise affine maps between voxel frames.
//!
//! A voxel frame is an OME-NGFF `scale` plus `translation`: voxel index `i`
//! sits at physical `scale * i + translation`, with voxel **centres** at the
//! integers. A [`Rescale`] carries a point from one frame's indices to
//! another's, and the centred convention is the same one `ops::Resample` uses,
//! so a point and the volume it was found in move together.

use blockflow::error::{Error, Result};

/// Where voxel indices sit in physical space: `p = scale * i + translation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub scale: [f64; 3],
    pub translation: [f64; 3],
}

impl Frame {
    pub fn new(scale: [f64; 3], translation: [f64; 3]) -> Result<Self> {
        if scale.iter().any(|s| !(*s > 0.0) || !s.is_finite())
            || translation.iter().any(|t| !t.is_finite())
        {
            return Err(Error::InvalidArgument(format!(
                "voxel frame: scale {scale:?} must be finite and positive, and translation \
                 {translation:?} finite"
            )));
        }
        Ok(Self { scale, translation })
    }

    pub fn unit() -> Self {
        Self {
            scale: [1.0; 3],
            translation: [0.0; 3],
        }
    }

    pub fn to_physical(&self, index: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| self.scale[axis] * index[axis] + self.translation[axis])
    }

    pub fn to_index(&self, physical: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| (physical[axis] - self.translation[axis]) / self.scale[axis])
    }

    /// The frame of a lattice of `spacing` voxels laid over `extent` voxels of
    /// this frame, starting at the same physical edge.
    ///
    /// Voxel 0 of this frame covers `translation - scale/2 ..`, so the new
    /// lattice's first centre is half of its own spacing in from that edge.
    pub fn relattice(&self, spacing: [f64; 3]) -> Result<Self> {
        Self::new(
            spacing,
            std::array::from_fn(|axis| {
                self.translation[axis] + (spacing[axis] - self.scale[axis]) / 2.0
            }),
        )
    }
}

/// `y = factor * x + offset`, per axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rescale {
    factor: [f64; 3],
    offset: [f64; 3],
}

impl Rescale {
    pub fn identity() -> Self {
        Self {
            factor: [1.0; 3],
            offset: [0.0; 3],
        }
    }

    /// From a volume of `source` voxels to one of `target` voxels covering the
    /// same extent: the centred map `y = (x + 0.5) * target/source - 0.5`.
    ///
    /// A zero extent is refused rather than producing an infinity that survives
    /// into every coordinate.
    pub fn between(source: [usize; 3], target: [usize; 3]) -> Result<Self> {
        if source.contains(&0) || target.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "rescale: the source shape is {source:?} and the target shape is {target:?}, and \
                 an extent of zero on either has no scale"
            )));
        }
        let factor: [f64; 3] = std::array::from_fn(|a| target[a] as f64 / source[a] as f64);
        Ok(Self {
            factor,
            offset: std::array::from_fn(|a| 0.5 * (factor[a] - 1.0)),
        })
    }

    /// From one frame's indices to another's, through the physical space they
    /// share.
    pub fn frames(from: &Frame, to: &Frame) -> Self {
        Self {
            factor: std::array::from_fn(|a| from.scale[a] / to.scale[a]),
            offset: std::array::from_fn(|a| {
                (from.translation[a] - to.translation[a]) / to.scale[a]
            }),
        }
    }

    /// A frame's indices to its physical coordinates.
    pub fn to_physical(frame: &Frame) -> Self {
        Self {
            factor: frame.scale,
            offset: frame.translation,
        }
    }

    /// Physical coordinates to a frame's indices.
    pub fn to_index(frame: &Frame) -> Self {
        Self::to_physical(frame).inverse()
    }

    pub fn factor(&self) -> [f64; 3] {
        self.factor
    }

    pub fn offset(&self) -> [f64; 3] {
        self.offset
    }

    pub fn inverse(&self) -> Self {
        Self {
            factor: std::array::from_fn(|a| 1.0 / self.factor[a]),
            offset: std::array::from_fn(|a| -self.offset[a] / self.factor[a]),
        }
    }

    /// `self` first, then `next`.
    pub fn then(&self, next: &Self) -> Self {
        Self {
            factor: std::array::from_fn(|a| next.factor[a] * self.factor[a]),
            offset: std::array::from_fn(|a| next.factor[a] * self.offset[a] + next.offset[a]),
        }
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.factor[a] * point[a] + self.offset[a])
    }

    /// The whole list, **in order**.
    pub fn apply_all(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|point| self.apply(*point)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn between_is_the_centred_resample_map() {
        let rescale = Rescale::between([10, 10, 10], [5, 20, 10]).unwrap();
        // Halving: output 0 sits between inputs 0 and 1.
        assert_eq!(rescale.inverse().apply([0.0, 0.0, 0.0])[0], 0.5);
        // Doubling: input 0 covers outputs 0 and 1, so its centre is 0.5.
        assert_eq!(rescale.apply([0.0, 0.0, 0.0])[1], 0.5);
        assert_eq!(rescale.apply([3.0, 3.0, 3.0])[2], 3.0);
        assert!(Rescale::between([0, 1, 1], [1, 1, 1]).is_err());
    }

    #[test]
    fn frames_agree_with_the_physical_round_trip() {
        let a = Frame::new([6.0, 5.0, 5.0], [0.0, 0.0, 0.0]).unwrap();
        let b = a.relattice([25.0, 25.0, 25.0]).unwrap();
        let rescale = Rescale::frames(&a, &b);
        let point = [12.0, 40.5, -3.0];
        let direct = b.to_index(a.to_physical(point));
        let mapped = rescale.apply(point);
        for axis in 0..3 {
            assert!((direct[axis] - mapped[axis]).abs() < 1e-12);
        }
        let back = rescale.inverse().apply(mapped);
        for axis in 0..3 {
            assert!((back[axis] - point[axis]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_pyramid_level_frame_is_a_relattice() {
        // OME-Zarr level 1 of a 2x pyramid over 6 x 5 x 5 voxels.
        let level0 = Frame::new([6.0, 5.0, 5.0], [0.0; 3]).unwrap();
        let level1 = level0.relattice([12.0, 10.0, 10.0]).unwrap();
        assert_eq!(level1.translation, [3.0, 2.5, 2.5]);
        assert_eq!(
            Rescale::frames(&level0, &level1),
            Rescale::between([10, 10, 10], [5, 5, 5]).unwrap()
        );
    }

    #[test]
    fn composition_applies_in_order() {
        let a = Rescale::between([4, 4, 4], [8, 8, 8]).unwrap();
        let b = Rescale::to_physical(&Frame::new([2.0, 3.0, 4.0], [1.0, 1.0, 1.0]).unwrap());
        let point = [1.5, 2.0, -1.0];
        assert_eq!(a.then(&b).apply(point), b.apply(a.apply(point)));
    }
}
