// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Destriping: removing curtaining (FIB milling stripes) from each z-plane by
// suppressing a wedge of its 2-D spectrum.
//
// The idea
// --------
// A stripe pattern that is constant *along* a direction `d` in the plane varies
// only across it, so all of its energy lies on the line through the origin of
// the spectrum that is **perpendicular to `d`**. Vertical stripes — constant
// down the rows, varying across the columns — sit on the column-frequency axis.
// Multiplying the spectrum by a filter that is zero in a narrow wedge around
// that line, and one elsewhere, removes the stripes and costs the image only the
// small fraction of its own energy that happens to fall in the wedge.
//
// Three refinements, each a parameter with no default:
//
// * **half-width** of the wedge (radians): the stripes are never perfectly
//   straight or perfectly aligned, so their energy spreads a little off the line.
// * **taper** (radians): outside the half-width the suppression falls off as a
//   Gaussian in the angle, rather than stepping, because a hard edge in the
//   spectrum rings in the image. Zero means a hard wedge.
// * **minimum frequency** (cycles per pixel): frequencies below it are kept
//   whatever their angle, so large-scale content — a gradient across the plane,
//   the shape of the sample — is not touched. It is a disc in cycles per pixel,
//   so it means the same thing whatever the padded size.
//
// Padding
// -------
// The transform is circular, and a plane whose left and right edges differ
// would wrap into a bright seam that the wedge then smears. Each plane is
// therefore extended by a caller-chosen **mirror margin** on each side
// (`ridge::Boundary::Reflect`, which keeps stripes stripes) and the extended
// length is rounded up to the next `5`-smooth size with
// [`fft::next_smooth_length`](super::fft::next_smooth_length) — `ops::fft`'s
// header measures what a hostile length costs. The filter is applied, the plane
// transformed back, and the margin discarded.
//
// Not a stencil
// -------------
// A Fourier coefficient reads the whole plane, so [`DestripeOp`]'s reach is the
// whole of axes 1 and 2 ([`AxisReach::All`]) and nothing on axis 0: the planes
// are independent, so a lattice may cut z freely but must keep each plane whole.

use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::reach::{AxisReach, Reach};
use crate::voxels::Voxels;

use super::fft::{next_smooth_length, RealTransform2, Spectrum};
use super::ridge::Boundary;
use super::shapes_agree;

/// The wedge, the taper, the kept disc and the padding of a destriping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DestripeParams {
    angle: f64,
    half_width: f64,
    taper: f64,
    min_frequency: f64,
    margin: [usize; 2],
}

impl DestripeParams {
    /// * `angle`: the direction the stripes **run along**, in radians, measured
    ///   in the plane from axis 0 (rows) towards axis 1 (columns). Stripes that
    ///   are constant down each column (vertical curtains) are `0`.
    /// * `half_width`: the wedge's half-width in radians, `>= 0`.
    /// * `taper`: the Gaussian angular fall-off beyond it, radians, `>= 0`.
    /// * `min_frequency`: radius of the kept disc, cycles per pixel, `>= 0`.
    /// * `margin`: mirror margin per in-plane axis, in pixels.
    pub fn new(
        angle: f64,
        half_width: f64,
        taper: f64,
        min_frequency: f64,
        margin: [usize; 2],
    ) -> Result<Self> {
        for (what, value) in [
            ("half-width", half_width),
            ("taper", taper),
            ("minimum frequency", min_frequency),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "the destripe {what} must be finite and not negative; got {value}"
                )));
            }
        }
        if !angle.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "the stripe angle must be finite; got {angle}"
            )));
        }
        Ok(Self {
            angle,
            half_width,
            taper,
            min_frequency,
            margin,
        })
    }

    /// The factor the spectrum is multiplied by at frequency `(f0, f1)`, in
    /// cycles per pixel along axes 0 and 1. In `[0, 1]`, even in the frequency.
    pub fn weight(&self, frequency: [f64; 2]) -> f64 {
        let radius = frequency[0].hypot(frequency[1]);
        if radius == 0.0 || radius < self.min_frequency {
            return 1.0;
        }
        // The stripes' energy lies along the normal to their direction.
        let normal = [-self.angle.sin(), self.angle.cos()];
        let along = (frequency[0] * normal[0] + frequency[1] * normal[1]).abs() / radius;
        let deviation = along.clamp(0.0, 1.0).acos();
        let suppression = if deviation <= self.half_width {
            1.0
        } else if self.taper > 0.0 {
            let beyond = (deviation - self.half_width) / self.taper;
            (-0.5 * beyond * beyond).exp()
        } else {
            0.0
        };
        1.0 - suppression
    }
}

/// A planned destriping of planes of one shape: the transform, the filter and
/// the buffers, built once and reused per plane.
#[derive(Debug, Clone)]
pub struct Destriper {
    shape: [usize; 2],
    margin: [usize; 2],
    transform: RealTransform2,
    filter: Array2<f64>,
    spectrum: Spectrum,
    padded: Array2<f64>,
    result: Array2<f64>,
}

impl Destriper {
    pub fn new(shape: [usize; 2], params: &DestripeParams) -> Result<Self> {
        if shape[0] == 0 || shape[1] == 0 {
            return Err(Error::InvalidArgument(format!(
                "destriping needs a non-empty plane, got {shape:?}"
            )));
        }
        let margin = params.margin;
        let padded_shape = [
            next_smooth_length(shape[0] + 2 * margin[0]),
            next_smooth_length(shape[1] + 2 * margin[1]),
        ];
        let transform = RealTransform2::new(padded_shape)?;
        let [rows, width] = transform.spectrum_shape();
        let filter = Array2::from_shape_fn((rows, width), |(r, c)| {
            let n0 = padded_shape[0] as isize;
            let signed = if (r as isize) <= n0 / 2 {
                r as isize
            } else {
                r as isize - n0
            };
            params.weight([
                signed as f64 / padded_shape[0] as f64,
                c as f64 / padded_shape[1] as f64,
            ])
        });
        Ok(Self {
            shape,
            margin,
            spectrum: transform.spectrum(),
            padded: Array2::zeros((padded_shape[0], padded_shape[1])),
            result: Array2::zeros((padded_shape[0], padded_shape[1])),
            transform,
            filter,
        })
    }

    /// The transformed plane's shape: the input plus its margins, rounded up.
    pub fn padded_shape(&self) -> [usize; 2] {
        self.transform.shape()
    }

    /// Destripe one plane of [`Self`]'s shape into `out`.
    pub fn destripe_plane_into<T>(
        &mut self,
        input: ArrayView2<'_, T>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        shapes_agree(input.shape(), &self.shape, "destripe (plane)")?;
        shapes_agree(input.shape(), out.shape(), "destripe (output)")?;
        let [m0, m1] = [self.margin[0] as isize, self.margin[1] as isize];
        for ((r, c), slot) in self.padded.indexed_iter_mut() {
            let i = Boundary::Reflect.index(r as isize - m0, self.shape[0]);
            let j = Boundary::Reflect.index(c as isize - m1, self.shape[1]);
            *slot = input[[i, j]].into();
        }
        self.transform
            .forward(self.padded.view(), &mut self.spectrum)?;
        self.spectrum
            .zip_mut_with(&self.filter, |value, &weight| *value *= weight);
        self.transform
            .inverse(&mut self.spectrum, &mut self.result)?;
        let (m0, m1) = (self.margin[0], self.margin[1]);
        for ((i, j), slot) in out.indexed_iter_mut() {
            *slot = self.result[[i + m0, j + m1]];
        }
        Ok(())
    }
}

/// Destripe every z-plane of `input` into `out`.
pub fn destripe_into<T>(
    input: ArrayView3<'_, T>,
    params: &DestripeParams,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "destripe_into")?;
    let (planes, rows, cols) = input.dim();
    if planes == 0 || rows == 0 || cols == 0 {
        return Ok(());
    }
    let mut destriper = Destriper::new([rows, cols], params)?;
    for (source, target) in input.outer_iter().zip(out.outer_iter_mut()) {
        destriper.destripe_plane_into(source, target)?;
    }
    Ok(())
}

/// [`destripe_into`] as a [`BlockOp`]: reach nothing on z and the whole of
/// each plane, `f64` out.
pub struct DestripeOp {
    name: &'static str,
    params: DestripeParams,
    cost: f64,
}

impl DestripeOp {
    pub fn new(name: &'static str, params: DestripeParams) -> Self {
        Self {
            name,
            params,
            cost: DESTRIPE_COST,
        }
    }

    pub fn params(&self) -> &DestripeParams {
        &self.params
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for DestripeOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The whole in-plane axes, stated as their extent; [`Self::reach_spec`]
    /// says [`AxisReach::All`].
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        if axis == 0 {
            0
        } else {
            volume_len
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([AxisReach::none(), AxisReach::All, AxisReach::All])
    }

    fn slicing(&self) -> Slicing {
        Slicing::Whole(
            "each plane's spectrum is a sum over the whole plane, so a cut inside a plane \
             changes every output voxel of it",
        )
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let shape = input.shape();
        for axis in 1..3 {
            if at.offset[axis] != 0 || shape[axis] != at.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "{}: destriping transforms whole planes, so the buffer must span axis \
                     {axis} (all {} voxels from 0); it starts at {} and spans {}",
                    self.name, at.volume[axis], at.offset[axis], shape[axis]
                )));
            }
        }
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            input,
            Error::InvalidArgument(format!("{}: no buffer holds half-precision", self.name)),
            |view| { destripe_into(view, &self.params, out) }
        )
    }

    /// Not declared: the inverse transform of a filtered constant is that
    /// constant only to a rounding, not to the bit.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Estimated, not measured: a forward and an inverse real transform per padded
/// pixel, of the order of `ops::convolve`'s transform path.
const DESTRIPE_COST: f64 = 40.0;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// An isotropic texture: white noise smoothed by a Gaussian, so its
    /// spectrum has no preferred direction.
    pub(super) fn texture(dim: (usize, usize, usize), seed: u64) -> Array3<f64> {
        let mut rng = Rng(seed);
        let noise = Array3::from_shape_fn(dim, |_| rng.next() - 0.5);
        let mut out = Array3::zeros(dim);
        let kernels =
            [0.0, 1.0, 1.0].map(|sigma| crate::ops::ridge::gaussian_weights(sigma, 4.0).unwrap());
        crate::ops::ridge::gaussian_smooth_into(noise.view(), &kernels, out.view_mut()).unwrap();
        out.mapv_inplace(|v| 20.0 * v);
        out
    }

    fn rms(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
        let n = a.len() as f64;
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / n).sqrt()
    }

    fn params() -> DestripeParams {
        DestripeParams::new(
            0.0,
            0.5f64.to_radians(),
            1.0f64.to_radians(),
            0.003,
            [16, 16],
        )
        .unwrap()
    }

    #[test]
    fn the_wedge_is_on_the_normal_to_the_stripes() {
        let p = params();
        // vertical stripes vary along axis 1: frequency (0, f) is removed
        assert_eq!(p.weight([0.0, 0.2]), 0.0);
        assert_eq!(p.weight([0.0, -0.2]), 0.0);
        assert!(p.weight([0.2, 0.0]) > 0.999);
        assert_eq!(p.weight([0.0, 0.002]), 1.0, "inside the kept disc");
        assert_eq!(p.weight([0.0, 0.0]), 1.0);
        let turned =
            DestripeParams::new(std::f64::consts::FRAC_PI_2, 0.01, 0.0, 0.0, [0, 0]).unwrap();
        assert_eq!(turned.weight([0.3, 0.0]), 0.0);
        assert_eq!(turned.weight([0.0, 0.3]), 1.0);
        assert!(DestripeParams::new(0.0, -0.1, 0.0, 0.0, [0, 0]).is_err());
    }

    /// Vertical curtains added to a texture are removed: the error against the
    /// clean image drops by a large factor.
    #[test]
    fn vertical_stripes_are_removed() {
        let dim = (2usize, 256usize, 240usize);
        let clean = texture(dim, 7);
        let mut rng = Rng(11);
        let stripes: Vec<f64> = (0..dim.2).map(|_| 6.0 * (rng.next() - 0.5)).collect();
        let striped = Array3::from_shape_fn(dim, |(z, i, j)| clean[[z, i, j]] + stripes[j]);
        let mut out = Array3::zeros(dim);
        destripe_into(striped.view(), &params(), out.view_mut()).unwrap();
        let before = rms(&striped, &clean);
        let after = rms(&out, &clean);
        assert!(after * 4.0 < before, "before {before}, after {after}");
    }

    /// Without stripes, the image is changed little: what is lost is the
    /// texture's own energy inside the wedge, about two percent here (a line
    /// through the origin of a `288 x 256` spectrum, and the smoothing puts
    /// more of the energy near the origin than white noise would).
    #[test]
    fn an_unstriped_image_is_changed_little() {
        let dim = (2usize, 256usize, 240usize);
        let clean = texture(dim, 3);
        let mut out = Array3::zeros(dim);
        destripe_into(clean.view(), &params(), out.view_mut()).unwrap();
        let signal = (clean.iter().map(|v| v * v).sum::<f64>() / clean.len() as f64).sqrt();
        let change = rms(&out, &clean);
        assert!(change < 0.2 * signal, "change {change}, signal {signal}");
    }

    #[test]
    fn the_op_wants_whole_planes_and_matches_the_kernel() {
        let dim = (3usize, 40usize, 50usize);
        let input = texture(dim, 5);
        let op = DestripeOp::new("destripe", params());
        assert_eq!(op.reach(0, 3), 0);
        assert_eq!(op.reach(1, 40), 40);
        let volume = [3, 40, 50];
        let voxels: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
        op.apply(&voxels, &mut out, &Anchor::whole(volume)).unwrap();
        let mut want = Array3::zeros(dim);
        destripe_into(input.view(), &params(), want.view_mut()).unwrap();
        assert_eq!(out.view::<f64>().unwrap(), want.view());
        // a z-slab gives the same planes
        let slab: Voxels = input.slice(ndarray::s![1..3, .., ..]).to_owned().into();
        let mut part = Voxels::zeros(Dtype::F64, [2, 40, 50]).unwrap();
        op.apply(&slab, &mut part, &Anchor::new([1, 0, 0], volume))
            .unwrap();
        assert_eq!(
            part.view::<f64>().unwrap(),
            want.slice(ndarray::s![1..3, .., ..])
        );
        // a cut plane is refused
        let cut: Voxels = input.slice(ndarray::s![.., 0..20, ..]).to_owned().into();
        let mut part = Voxels::zeros(Dtype::F64, [3, 20, 50]).unwrap();
        assert!(op
            .apply(&cut, &mut part, &Anchor::new([0, 0, 0], volume))
            .is_err());
    }
}
