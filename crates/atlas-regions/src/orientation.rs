// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Signed axis permutations: how a sample's axes map onto the atlas's, for
//! volumes, points, shapes and resolutions.
//!
//! An [`Orientation`] says, for each **output** axis in `[z, y, x]` order,
//! which input axis it is read from and whether it is reversed: transpose
//! first, then flip, the order the reference's `orientation.py` uses.
//!
//! The reference writes an orientation as a tuple of signed 1-based axes in
//! its own `(x, y, z)` array order, such as `(1, -2, 3)`: output axis `d` is
//! input axis `|o[d]| - 1`, reversed when `o[d] < 0`. Its arrays are the
//! transpose of ours (its axis 1 is our `x`, its axis 3 our `z`), so
//! [`Orientation::from_xyz`] reads its tuple as
//!
//! ```text
//! our output axis a (0 = z, 1 = y, 2 = x)  <-  reference entry o[2 - a]
//! source[a] = 3 - |o[2 - a]|,   flip[a] = o[2 - a] < 0
//! ```
//!
//! so `(1, -2, 3)` is `source = [0, 1, 2]` with `y` flipped, and `(2, -1, 3)`
//! swaps `y` and `x` and flips the new `y`. Tests check every operation against
//! the reference's own functions on small arrays.

use blockflow::error::{Error, Result};
use blockflow::voxels::Voxels;
use ndarray::{Array3, Axis};

/// A signed permutation of the three axes, in `[z, y, x]` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Orientation {
    source: [usize; 3],
    flip: [bool; 3],
}

impl Default for Orientation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Orientation {
    /// Every axis kept in place.
    pub const IDENTITY: Self = Self {
        source: [0, 1, 2],
        flip: [false; 3],
    };

    /// Output axis `a` is input axis `source[a]`, reversed when `flip[a]`.
    pub fn new(source: [usize; 3], flip: [bool; 3]) -> Result<Self> {
        let mut seen = [false; 3];
        for &axis in &source {
            if axis > 2 || std::mem::replace(&mut seen[axis], true) {
                return Err(Error::invalid(format!(
                    "orientation source axes {source:?} are not a permutation of 0, 1, 2"
                )));
            }
        }
        Ok(Self { source, flip })
    }

    /// The reference's tuple of signed 1-based axes in `(x, y, z)` order, such
    /// as `[1, -2, 3]` (see the module documentation for the mapping).
    pub fn from_xyz(tuple: [i32; 3]) -> Result<Self> {
        let mut source = [0; 3];
        let mut flip = [false; 3];
        for a in 0..3 {
            let entry = tuple[2 - a];
            if !(1..=3).contains(&entry.unsigned_abs()) {
                return Err(Error::invalid(format!(
                    "orientation {tuple:?}: each entry must be a signed axis 1, 2 or 3"
                )));
            }
            source[a] = 3 - entry.unsigned_abs() as usize;
            flip[a] = entry < 0;
        }
        Self::new(source, flip)
            .map_err(|_| Error::invalid(format!("orientation {tuple:?} repeats an axis")))
    }

    /// The reference's tuple for this orientation; the inverse of
    /// [`Self::from_xyz`].
    pub fn to_xyz(&self) -> [i32; 3] {
        let mut tuple = [0; 3];
        for a in 0..3 {
            let axis = 3 - self.source[a] as i32;
            tuple[2 - a] = if self.flip[a] { -axis } else { axis };
        }
        tuple
    }

    /// Parses the reference's spellings: `"1,-2,3"`, `"(1, -2, 3)"`, `"left"`
    /// (the identity) or `"right"` (`(-1, 2, 3)`). The tuple is in `(x, y, z)`
    /// order, as [`Self::from_xyz`].
    pub fn parse(spec: &str) -> Result<Self> {
        let trimmed = spec.trim();
        match trimmed {
            "left" => return Ok(Self::IDENTITY),
            "right" => return Self::from_xyz([-1, 2, 3]),
            _ => {}
        }
        let inner = trimmed
            .trim_start_matches(['(', '['])
            .trim_end_matches([')', ']']);
        let entries = inner
            .split(',')
            .map(|entry| entry.trim().parse::<i32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::invalid(format!("orientation {spec:?}: {error}")))?;
        let tuple: [i32; 3] = entries.try_into().map_err(|_| {
            Error::invalid(format!(
                "orientation {spec:?}: expected three signed axes such as 1,-2,3, or left/right"
            ))
        })?;
        Self::from_xyz(tuple)
    }

    /// For each output axis, the input axis it is read from.
    pub fn source(&self) -> [usize; 3] {
        self.source
    }

    /// For each output axis, whether it is reversed.
    pub fn flip(&self) -> [bool; 3] {
        self.flip
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// The orientation that undoes this one.
    pub fn inverse(&self) -> Self {
        let mut inverse = Self::IDENTITY;
        for a in 0..3 {
            inverse.source[self.source[a]] = a;
            inverse.flip[self.source[a]] = self.flip[a];
        }
        inverse
    }

    /// This orientation followed by `next`.
    pub fn then(&self, next: &Self) -> Self {
        let mut composed = Self::IDENTITY;
        for a in 0..3 {
            let middle = next.source[a];
            composed.source[a] = self.source[middle];
            composed.flip[a] = next.flip[a] ^ self.flip[middle];
        }
        composed
    }

    /// The shape of an oriented volume whose unoriented shape is `shape`.
    pub fn orient_shape(&self, shape: [usize; 3]) -> [usize; 3] {
        self.source.map(|axis| shape[axis])
    }

    /// A per-axis voxel size (or any per-axis value), permuted; flips do not
    /// change it.
    pub fn orient_resolution(&self, resolution: [f64; 3]) -> [f64; 3] {
        self.source.map(|axis| resolution[axis])
    }

    /// A point in the voxel coordinates of a volume of `shape` (unoriented,
    /// voxel centres at integers), in the oriented volume's coordinates. A
    /// reversed axis maps `i` to `n - 1 - i`.
    pub fn orient_point(&self, point: [f64; 3], shape: [usize; 3]) -> [f64; 3] {
        [0, 1, 2].map(|a| {
            let axis = self.source[a];
            if self.flip[a] {
                (shape[axis] as f64 - 1.0) - point[axis]
            } else {
                point[axis]
            }
        })
    }

    /// [`Self::orient_point`] on every point.
    pub fn orient_points(&self, points: &[[f64; 3]], shape: [usize; 3]) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|&point| self.orient_point(point, shape))
            .collect()
    }

    /// An owned, standard-layout copy of `volume`, oriented.
    pub fn orient_voxels(&self, volume: &Voxels) -> Voxels {
        macro_rules! arm {
            ($($variant:ident),*) => {
                match volume {
                    $(Voxels::$variant(array) => Voxels::$variant(self.orient_array(array)),)*
                }
            };
        }
        arm!(Bool, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64)
    }

    /// [`Self::orient_voxels`] on a plain array.
    pub fn orient_array<T: Clone>(&self, array: &Array3<T>) -> Array3<T> {
        if self.is_identity() {
            return array.clone();
        }
        let mut view = array.view().permuted_axes(self.source);
        for a in 0..3 {
            if self.flip[a] {
                view.invert_axis(Axis(a));
            }
        }
        view.as_standard_layout().into_owned()
    }

    /// The unoriented shape of an oriented volume of `oriented_shape`.
    pub fn unorient_shape(&self, oriented_shape: [usize; 3]) -> [usize; 3] {
        self.inverse().orient_shape(oriented_shape)
    }

    /// [`Self::orient_resolution`] undone.
    pub fn unorient_resolution(&self, resolution: [f64; 3]) -> [f64; 3] {
        self.inverse().orient_resolution(resolution)
    }

    /// A point of the oriented volume back in the coordinates of the
    /// unoriented volume of `shape` — the same shape [`Self::orient_point`]
    /// takes, as the reference's `orient_points(..., inverse=True)` does.
    pub fn unorient_point(&self, point: [f64; 3], shape: [usize; 3]) -> [f64; 3] {
        self.inverse().orient_point(point, self.orient_shape(shape))
    }

    /// [`Self::unorient_point`] on every point.
    pub fn unorient_points(&self, points: &[[f64; 3]], shape: [usize; 3]) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|&point| self.unorient_point(point, shape))
            .collect()
    }

    /// An oriented volume back in the unoriented frame.
    pub fn unorient_voxels(&self, volume: &Voxels) -> Voxels {
        self.inverse().orient_voxels(volume)
    }
}

impl std::fmt::Display for Orientation {
    /// The reference's tuple, `(x, y, z)` order: `(1,-2,3)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [a, b, c] = self.to_xyz();
        write!(f, "({a},{b},{c})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<Orientation> {
        let mut out = Vec::new();
        for source in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for bits in 0..8 {
                let flip = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
                out.push(Orientation::new(source, flip).unwrap());
            }
        }
        out
    }

    #[test]
    fn the_reference_tuple_maps_onto_z_y_x() {
        let o = Orientation::from_xyz([1, -2, 3]).unwrap();
        assert_eq!(o.source(), [0, 1, 2]);
        assert_eq!(o.flip(), [false, true, false]);
        let o = Orientation::from_xyz([2, -1, 3]).unwrap();
        assert_eq!(o.source(), [0, 2, 1]);
        assert_eq!(o.flip(), [false, true, false]);
        let o = Orientation::from_xyz([-1, 2, 3]).unwrap();
        assert_eq!(o.flip(), [false, false, true]);
        assert_eq!(Orientation::parse("right").unwrap(), o);
        assert_eq!(Orientation::parse("left").unwrap(), Orientation::IDENTITY);
        assert_eq!(
            Orientation::parse("(2, -1, 3)").unwrap().to_string(),
            "(2,-1,3)"
        );
    }

    #[test]
    fn bad_tuples_are_refused() {
        assert!(Orientation::from_xyz([1, 1, 3]).is_err());
        assert!(Orientation::from_xyz([0, 2, 3]).is_err());
        assert!(Orientation::from_xyz([1, 2, 4]).is_err());
        assert!(Orientation::parse("1,2").is_err());
        assert!(Orientation::new([0, 0, 1], [false; 3]).is_err());
    }

    #[test]
    fn every_orientation_round_trips() {
        let shape = [4, 3, 2];
        let volume = Voxels::I64(Array3::from_shape_fn((4, 3, 2), |(z, y, x)| {
            (z * 100 + y * 10 + x) as i64
        }));
        let point = [1.25, -0.5, 7.0];
        for o in all() {
            assert_eq!(Orientation::from_xyz(o.to_xyz()).unwrap(), o);
            assert_eq!(o.then(&o.inverse()), Orientation::IDENTITY);
            assert_eq!(o.inverse().then(&o), Orientation::IDENTITY);
            let oriented = o.orient_voxels(&volume);
            assert_eq!(oriented.shape(), o.orient_shape(shape));
            assert_eq!(o.unorient_voxels(&oriented), volume);
            assert_eq!(o.unorient_shape(o.orient_shape(shape)), shape);
            let back = o.unorient_point(o.orient_point(point, shape), shape);
            assert_eq!(back, point, "{o}");
        }
    }

    #[test]
    fn a_point_follows_its_voxel() {
        let shape = [4, 3, 2];
        let volume = Array3::from_shape_fn((4, 3, 2), |(z, y, x)| [z, y, x]);
        for o in all() {
            let oriented = o.orient_array(&volume);
            for (index, original) in volume.indexed_iter() {
                let p = o.orient_point([index.0 as f64, index.1 as f64, index.2 as f64], shape);
                let at = p.map(|c| c as usize);
                assert_eq!(&oriented[at], original, "{o}");
            }
        }
    }

    #[test]
    fn composition_matches_applying_twice() {
        let volume = Array3::from_shape_fn((4, 3, 2), |(z, y, x)| z * 100 + y * 10 + x);
        let orientations = all();
        for a in orientations.iter().step_by(5) {
            for b in orientations.iter().step_by(7) {
                let twice = b.orient_array(&a.orient_array(&volume));
                assert_eq!(a.then(b).orient_array(&volume), twice);
            }
        }
    }
}
