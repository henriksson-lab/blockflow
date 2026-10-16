// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Per-slice intensity normalisation: making every z-plane of a stack use the
// same brightness and contrast.
//
// Why per slice
// -------------
// A serial-section or FIB-SEM stack is acquired one plane at a time, and the
// detector's gain and offset drift between planes. The drift is a property of
// the *plane*, not of a neighbourhood, so `super::normalise`'s local estimates
// are the wrong tool: they would also flatten real contrast within a plane.
// What is wanted is one affine map per plane, `out = gain[z] * value +
// offset[z]`, chosen so that the plane's robust low and high levels land where
// the stack as a whole puts them.
//
// The three steps
// ---------------
// 1. **Statistics** ([`slice_statistics`]): per plane, a low and a high
//    percentile, optionally only within a foreground mask and optionally
//    ignoring exact zeros (the padding a registered or cropped stack carries).
//    Percentiles rather than min/max or mean/std because a few saturated or
//    dead pixels must not move the map.
// 2. **Target** ([`slice_mapping`]): either the per-plane statistics smoothed
//    along z by a running median ([`SliceTarget::Smoothed`]) — which follows a
//    slow, genuine change in the sample while removing plane-to-plane jumps —
//    or a fixed output range ([`SliceTarget::Fixed`]).
// 3. **Apply** ([`apply_slice_mapping_into`], [`SliceNormaliseOp`]): the affine
//    map of the plane a voxel is in. Voxelwise, so reach 0 and a stencil; the
//    only position it reads is the voxel's global z, from the [`Anchor`].
//
// Steps 1 and 2 are a **whole-volume** pass here: a percentile is not a sum, so
// it has no exact block-wise partial. A caller with a volume too large to hold
// can run [`slice_statistics`] over z-slabs (the statistic of a plane needs only
// that plane) and concatenate the tables; a block-reduced histogram form is not
// provided.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::voxels::Voxels;

use super::shapes_agree;

/// What [`slice_statistics`] measures per plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceStatisticsParams {
    low: f64,
    high: f64,
    exclude_zeros: bool,
    stride: usize,
}

impl SliceStatisticsParams {
    /// `low` and `high` are percentiles in `[0, 100]` with `low < high`.
    /// `exclude_zeros` drops samples that are exactly zero (padding). `stride`
    /// subsamples each plane on a regular `stride x stride` grid for speed; `1`
    /// reads every pixel.
    pub fn new(low: f64, high: f64, exclude_zeros: bool, stride: usize) -> Result<Self> {
        if !(0.0..=100.0).contains(&low) || !(0.0..=100.0).contains(&high) || !(low < high) {
            return Err(Error::InvalidArgument(format!(
                "slice percentiles must satisfy 0 <= low < high <= 100; got {low} and {high}"
            )));
        }
        if stride == 0 {
            return Err(Error::InvalidArgument(
                "a sampling stride of zero samples nothing".into(),
            ));
        }
        Ok(Self {
            low,
            high,
            exclude_zeros,
            stride,
        })
    }
}

/// One plane's robust levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceLevels {
    pub low: f64,
    pub high: f64,
    /// How many samples the two percentiles were taken over.
    pub count: usize,
}

/// The `p`-th percentile (`0..=100`) of `values`, by linear interpolation
/// between the two bracketing order statistics. Reorders `values`.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    let n = values.len();
    let rank = p / 100.0 * (n - 1) as f64;
    let below = rank.floor() as usize;
    let fraction = rank - below as f64;
    let (_, &mut lower, rest) = values.select_nth_unstable_by(below, f64::total_cmp);
    if fraction == 0.0 || rest.is_empty() {
        return lower;
    }
    let upper = rest.iter().copied().min_by(f64::total_cmp).unwrap();
    lower + fraction * (upper - lower)
}

/// Per z-plane low and high percentiles of `volume`.
///
/// `mask`, if given, must have `volume`'s shape; only voxels where it is true
/// are counted. A plane with no counted sample — all padding, or all outside
/// the mask — is `None`. Non-finite values are never counted.
pub fn slice_statistics<T>(
    volume: ArrayView3<'_, T>,
    mask: Option<ArrayView3<'_, bool>>,
    params: &SliceStatisticsParams,
) -> Result<Vec<Option<SliceLevels>>>
where
    T: Copy + Into<f64>,
{
    if let Some(mask) = mask.as_ref() {
        shapes_agree(volume.shape(), mask.shape(), "slice_statistics (mask)")?;
    }
    let (planes, rows, cols) = volume.dim();
    let stride = params.stride;
    let mut samples = Vec::with_capacity(rows.div_ceil(stride) * cols.div_ceil(stride));
    let mut out = Vec::with_capacity(planes);
    for z in 0..planes {
        samples.clear();
        for i in (0..rows).step_by(stride) {
            for j in (0..cols).step_by(stride) {
                if let Some(mask) = mask.as_ref() {
                    if !mask[[z, i, j]] {
                        continue;
                    }
                }
                let value: f64 = volume[[z, i, j]].into();
                if !value.is_finite() || (params.exclude_zeros && value == 0.0) {
                    continue;
                }
                samples.push(value);
            }
        }
        if samples.is_empty() {
            out.push(None);
            continue;
        }
        let low = percentile(&mut samples, params.low);
        let high = percentile(&mut samples, params.high);
        out.push(Some(SliceLevels {
            low,
            high,
            count: samples.len(),
        }));
    }
    Ok(out)
}

/// The median of each centred window of `width` (odd) over the present values,
/// clipped at the ends. A window with no present value is `None`.
pub fn running_median(values: &[Option<f64>], width: usize) -> Result<Vec<Option<f64>>> {
    if width == 0 || width % 2 == 0 {
        return Err(Error::InvalidArgument(format!(
            "a running median needs an odd, positive width so that it is centred; got {width}"
        )));
    }
    let half = width / 2;
    let mut window = Vec::with_capacity(width);
    Ok((0..values.len())
        .map(|z| {
            window.clear();
            let from = z.saturating_sub(half);
            let to = (z + half + 1).min(values.len());
            window.extend(values[from..to].iter().flatten().copied());
            if window.is_empty() {
                None
            } else {
                Some(percentile(&mut window, 50.0))
            }
        })
        .collect())
}

/// Where each plane's levels are mapped to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceTarget {
    /// The per-plane levels smoothed along z by a running median of `width`
    /// planes (odd). Follows slow change, removes plane-to-plane jumps.
    Smoothed { width: usize },
    /// Every plane's low level to `low` and high level to `high`.
    Fixed { low: f64, high: f64 },
}

/// One affine map per plane: `out = gain[z] * value + offset[z]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceMapping {
    gain: Vec<f64>,
    offset: Vec<f64>,
}

impl SliceMapping {
    /// A mapping from explicit per-plane gains and offsets. Both must be finite
    /// and of the same length.
    pub fn new(gain: Vec<f64>, offset: Vec<f64>) -> Result<Self> {
        if gain.len() != offset.len() {
            return Err(Error::InvalidArgument(format!(
                "a slice mapping needs one gain and one offset per plane; got {} and {}",
                gain.len(),
                offset.len()
            )));
        }
        if gain
            .iter()
            .chain(offset.iter())
            .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidArgument(
                "slice gains and offsets must be finite".into(),
            ));
        }
        Ok(Self { gain, offset })
    }

    /// The identity on `planes` planes.
    pub fn identity(planes: usize) -> Self {
        Self {
            gain: vec![1.0; planes],
            offset: vec![0.0; planes],
        }
    }

    pub fn planes(&self) -> usize {
        self.gain.len()
    }

    pub fn gain(&self) -> &[f64] {
        &self.gain
    }

    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    #[inline]
    pub fn map(&self, plane: usize, value: f64) -> f64 {
        self.gain[plane] * value + self.offset[plane]
    }
}

/// The map that sends `from` levels to `to` levels. A plane whose two levels
/// coincide (a flat plane) gets a gain of one and is only shifted.
fn affine(from: (f64, f64), to: (f64, f64)) -> (f64, f64) {
    let span = from.1 - from.0;
    let gain = if span > 0.0 {
        (to.1 - to.0) / span
    } else {
        1.0
    };
    (gain, to.0 - gain * from.0)
}

/// Turn per-plane levels into a [`SliceMapping`] onto `target`.
///
/// A plane without levels (`None`) takes the map of the **nearest plane that
/// has them**, ties to the lower plane; a stack where no plane has levels is an
/// error.
pub fn slice_mapping(levels: &[Option<SliceLevels>], target: SliceTarget) -> Result<SliceMapping> {
    let planes = levels.len();
    let present: Vec<usize> = (0..planes).filter(|&z| levels[z].is_some()).collect();
    if present.is_empty() {
        return Err(Error::InvalidArgument(
            "no plane has any counted sample, so there is nothing to normalise against".into(),
        ));
    }
    let targets: Vec<(f64, f64)> = match target {
        SliceTarget::Fixed { low, high } => {
            if !low.is_finite() || !high.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "a fixed slice target must be finite; got {low} and {high}"
                )));
            }
            vec![(low, high); planes]
        }
        SliceTarget::Smoothed { width } => {
            let lows: Vec<Option<f64>> = levels.iter().map(|l| l.map(|l| l.low)).collect();
            let highs: Vec<Option<f64>> = levels.iter().map(|l| l.map(|l| l.high)).collect();
            let lows = running_median(&lows, width)?;
            let highs = running_median(&highs, width)?;
            lows.into_iter()
                .zip(highs)
                .map(|(low, high)| (low.unwrap_or(f64::NAN), high.unwrap_or(f64::NAN)))
                .collect()
        }
    };
    let mut gain = vec![1.0; planes];
    let mut offset = vec![0.0; planes];
    for &z in &present {
        let level = levels[z].unwrap();
        (gain[z], offset[z]) = affine((level.low, level.high), targets[z]);
    }
    for z in 0..planes {
        if levels[z].is_some() {
            continue;
        }
        let nearest = *present.iter().min_by_key(|&&p| (p.abs_diff(z), p)).unwrap();
        gain[z] = gain[nearest];
        offset[z] = offset[nearest];
    }
    SliceMapping::new(gain, offset)
}

/// Apply `mapping` to `input`, whose first plane is plane `first_plane` of the
/// stack the mapping was made for. With `keep_zeros`, an input of exactly zero
/// (padding) stays zero.
pub fn apply_slice_mapping_into<T>(
    input: ArrayView3<'_, T>,
    mapping: &SliceMapping,
    first_plane: usize,
    keep_zeros: bool,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "apply_slice_mapping_into")?;
    let planes = input.dim().0;
    if first_plane + planes > mapping.planes() {
        return Err(Error::InvalidArgument(format!(
            "planes {first_plane}..{} are not all in a mapping of {} planes",
            first_plane + planes,
            mapping.planes()
        )));
    }
    for (z, (source, mut target)) in input.outer_iter().zip(out.outer_iter_mut()).enumerate() {
        let plane = first_plane + z;
        target.zip_mut_with(&source, |out, &value| {
            let value: f64 = value.into();
            *out = if keep_zeros && value == 0.0 {
                0.0
            } else {
                mapping.map(plane, value)
            };
        });
    }
    Ok(())
}

/// [`apply_slice_mapping_into`] as a [`BlockOp`]: reach 0, a stencil, `f64` out.
///
/// The mapping is a parameter, computed beforehand by [`slice_statistics`] and
/// [`slice_mapping`] (or given directly), and must have one entry per plane of
/// the volume the op runs on.
pub struct SliceNormaliseOp {
    name: &'static str,
    mapping: SliceMapping,
    keep_zeros: bool,
    cost: f64,
}

impl SliceNormaliseOp {
    pub fn new(name: &'static str, mapping: SliceMapping, keep_zeros: bool) -> Self {
        Self {
            name,
            mapping,
            keep_zeros,
            cost: SLICE_NORMALISE_COST,
        }
    }

    pub fn mapping(&self) -> &SliceMapping {
        &self.mapping
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for SliceNormaliseOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Voxelwise.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// Each output voxel is a function of its own input voxel and its global
    /// plane index, which a cut does not change.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        if at.volume[0] != self.mapping.planes() {
            return Err(Error::InvalidArgument(format!(
                "{}: the mapping has {} planes and the volume {}",
                self.name,
                self.mapping.planes(),
                at.volume[0]
            )));
        }
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            input,
            Error::InvalidArgument(format!("{}: no buffer holds half-precision", self.name)),
            |view| {
                apply_slice_mapping_into(view, &self.mapping, at.offset[0], self.keep_zeros, out)
            }
        )
    }

    /// Not declared: a constant block maps to a different value in each plane.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Estimated, not measured: one multiply-add and a branch per voxel, about the
/// cost of the other voxelwise affine maps in `ops`.
const SLICE_NORMALISE_COST: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// A stack whose planes are one texture under a different gain and offset
    /// each, plus zero padding.
    fn drifting_stack() -> (Array3<f64>, Array3<f64>) {
        let dim = (9usize, 20usize, 24usize);
        let clean = Array3::from_shape_fn(dim, |(_, i, j)| {
            100.0 + 40.0 * ((i as f64) * 0.7).sin() * ((j as f64) * 0.45).cos()
        });
        let mut drifted = clean.clone();
        for (z, mut plane) in drifted.outer_iter_mut().enumerate() {
            let gain = 1.0 + 0.15 * ((z as f64) * 1.3).sin();
            let offset = 20.0 * ((z as f64) * 2.1).cos();
            plane.mapv_inplace(|v| gain * v + offset);
            // padding stripe
            for i in 0..dim.1 {
                plane[[i, 0]] = 0.0;
                plane[[i, 1]] = 0.0;
            }
        }
        (clean, drifted)
    }

    #[test]
    fn the_percentile_interpolates_between_order_statistics() {
        let mut values = vec![4.0, 1.0, 3.0, 2.0, 5.0];
        assert_eq!(percentile(&mut values, 0.0), 1.0);
        assert_eq!(percentile(&mut values, 100.0), 5.0);
        assert_eq!(percentile(&mut values, 50.0), 3.0);
        assert_eq!(percentile(&mut values, 12.5), 1.5);
        let medians =
            running_median(&[Some(1.0), Some(9.0), None, Some(2.0), Some(3.0)], 3).unwrap();
        assert_eq!(
            medians,
            vec![Some(5.0), Some(5.0), Some(5.5), Some(2.5), Some(2.5)]
        );
        assert!(running_median(&[Some(1.0)], 2).is_err());
    }

    /// Fixed target: every plane's percentiles land on the target, so the drift
    /// is gone and all planes agree with each other outside the padding.
    #[test]
    fn a_fixed_target_removes_per_plane_gain_and_offset() {
        let (_, drifted) = drifting_stack();
        let params = SliceStatisticsParams::new(2.0, 98.0, true, 1).unwrap();
        let levels = slice_statistics(drifted.view(), None, &params).unwrap();
        assert!(levels.iter().all(|l| l.unwrap().count == 20 * 22));
        let mapping = slice_mapping(
            &levels,
            SliceTarget::Fixed {
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap();
        let mut out = Array3::zeros(drifted.dim());
        apply_slice_mapping_into(drifted.view(), &mapping, 0, true, out.view_mut()).unwrap();
        for z in 1..drifted.dim().0 {
            for i in 0..20 {
                assert_eq!(out[[z, i, 0]], 0.0, "padding stays zero");
                for j in 2..24 {
                    assert!((out[[z, i, j]] - out[[0, i, j]]).abs() < 1e-9);
                }
            }
        }
        let after = slice_statistics(out.view(), None, &params).unwrap();
        for level in after {
            let level = level.unwrap();
            assert!(level.low.abs() < 1e-9 && (level.high - 1.0).abs() < 1e-9);
        }
    }

    /// Smoothed target: an outlier plane is pulled to its neighbours, which
    /// share one gain, and planes already agreeing with them are left alone.
    #[test]
    fn a_smoothed_target_corrects_an_outlier_plane_and_keeps_the_rest() {
        let dim = (7usize, 16usize, 16usize);
        let clean = Array3::from_shape_fn(dim, |(_, i, j)| (i * 16 + j) as f64);
        let mut drifted = clean.clone();
        drifted
            .index_axis_mut(ndarray::Axis(0), 3)
            .mapv_inplace(|v| 2.0 * v + 50.0);
        let params = SliceStatisticsParams::new(1.0, 99.0, false, 1).unwrap();
        let levels = slice_statistics(drifted.view(), None, &params).unwrap();
        let mapping = slice_mapping(&levels, SliceTarget::Smoothed { width: 3 }).unwrap();
        let mut out = Array3::zeros(dim);
        apply_slice_mapping_into(drifted.view(), &mapping, 0, false, out.view_mut()).unwrap();
        for ((z, i, j), &value) in out.indexed_iter() {
            if z == 2 || z == 4 {
                // their windows include the outlier, but the median ignores it
                assert!((value - clean[[z, i, j]]).abs() < 1e-9);
            }
            if z == 3 {
                assert!((value - clean[[z, i, j]]).abs() < 1e-9, "{value}");
            }
        }
    }

    /// The mask restricts the statistics; an empty plane borrows its nearest
    /// neighbour's map.
    #[test]
    fn the_mask_restricts_and_an_empty_plane_borrows_its_neighbours_map() {
        let (_, drifted) = drifting_stack();
        let mut mask = Array3::from_elem(drifted.dim(), true);
        mask.index_axis_mut(ndarray::Axis(0), 4).fill(false);
        let params = SliceStatisticsParams::new(5.0, 95.0, true, 2).unwrap();
        let levels = slice_statistics(drifted.view(), Some(mask.view()), &params).unwrap();
        assert!(levels[4].is_none());
        let mapping = slice_mapping(
            &levels,
            SliceTarget::Fixed {
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap();
        assert_eq!(mapping.gain()[4], mapping.gain()[3]);
        assert!(slice_mapping(&[None, None], SliceTarget::Smoothed { width: 3 }).is_err());
    }

    /// The op on a sub-block, anchored part way down the stack, reproduces the
    /// whole-volume answer: the plane index is global.
    #[test]
    fn the_op_reads_the_global_plane_and_is_cut_invariant() {
        let (_, drifted) = drifting_stack();
        let dim = drifted.dim();
        let params = SliceStatisticsParams::new(2.0, 98.0, true, 1).unwrap();
        let levels = slice_statistics(drifted.view(), None, &params).unwrap();
        let mapping = slice_mapping(&levels, SliceTarget::Smoothed { width: 5 }).unwrap();
        let op = SliceNormaliseOp::new("slice", mapping, true);
        assert_eq!(op.reach(0, dim.0), 0);
        let volume = [dim.0, dim.1, dim.2];
        let whole_in: Voxels = drifted.clone().into();
        let mut whole = Voxels::zeros(Dtype::F64, volume).unwrap();
        op.apply(&whole_in, &mut whole, &Anchor::whole(volume))
            .unwrap();
        let whole = whole.view::<f64>().unwrap().to_owned();

        let part = drifted.slice(ndarray::s![3..7, 5..15, ..]).to_owned();
        let part_in: Voxels = part.into();
        let mut out = Voxels::zeros(Dtype::F64, [4, 10, dim.2]).unwrap();
        op.apply(&part_in, &mut out, &Anchor::new([3, 5, 0], volume))
            .unwrap();
        assert_eq!(
            out.view::<f64>().unwrap(),
            whole.slice(ndarray::s![3..7, 5..15, ..])
        );

        // a u16 input reaches the same kernel
        let as_u16: Voxels = drifted.mapv(|v| v.round() as u16).into();
        let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
        op.apply(&as_u16, &mut out, &Anchor::whole(volume)).unwrap();
        // and a mapping of the wrong length is refused
        let wrong = SliceNormaliseOp::new("slice", SliceMapping::identity(3), true);
        assert!(wrong
            .apply(&whole_in, &mut out, &Anchor::whole(volume))
            .is_err());
    }
}
