// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Tract maps: a fibre-tract channel binarised at a clip range taken from the
//! volume's own intensity distribution, the mask as points, and the points as
//! a density.
//!
//! ```text
//! run 1:  raw -> sampled values -> percentiles (p_lo, p_hi)           = clip range
//! run 2:  raw -> binarize(clip range) -> mask -> points -> ball density
//! ```
//!
//! The range is global, so it is a barrier `reduce` over a sample of the
//! volume (`ops::threshold::SampledValuesOp` and `GlobalThreshold::Percentile`,
//! numpy's linear-interpolated percentile) rather than anything per block. It
//! is computed in a run of its own because every mask of the binarisation is
//! stated in terms of it when the second plan is built. The density is
//! `ops::voxelize::VoxelizeOp` over the mask's points, at the source
//! resolution; mapping the points into an atlas frame belongs to registration
//! ([`Rescale`] is the shape-ratio half of that).

use std::collections::BTreeMap;

use blockflow::assemble::{Phase, PlanBuilder};
use blockflow::env::Environment;
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    PhaseView, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::SourceInput;
use blockflow::ops::threshold::{sampled_thresholds, GlobalThreshold};
use blockflow::ops::voxelize::{ball, VoxelizeOp};
use blockflow::ops::{SampledValuesOp, ValueSampling, WidenLabelsOp};
use blockflow::points::{decode_points, encode_points, Point};
use blockflow::sidecar::{FragmentKey, Lifecycle};
use blockflow::voxels::Voxels;
use blockflow::{Chain, Error, Result};

use super::binarize::{append_binarize_phases, BinarizeParams};
use super::clip::ClipParams;
use super::plan::{read_back, run_in_memory, Cursor, Image};

const SAMPLES_STREAM: &str = "tract.samples";
const RANGE_STREAM: &str = "tract.range";
const POINTS_STREAM: &str = "tract.points";
const RANGE_MAGIC: &[u8] = b"vessel-graph tract range v1";

/// How a tract map is made.
#[derive(Debug, Clone, PartialEq)]
pub struct TractParams {
    /// The clip range's two percentiles of the raw intensities, `0 ..= 100`.
    pub percentiles: (f64, f64),
    /// One voxel in this many (by linear index) is sampled for the percentiles.
    pub sample_every: usize,
    /// The binarisation, whose clip range is replaced by the measured one.
    pub binarize: BinarizeParams,
    /// The density kernel: a ball of this radius per axis.
    pub density_radius: [usize; 3],
}

impl Default for TractParams {
    fn default() -> Self {
        Self {
            percentiles: (70.0, 99.999),
            sample_every: 1000,
            binarize: tract_binarize(ClipParams::default()),
            density_radius: [5, 5, 5],
        }
    }
}

/// The binarisation a tract channel takes: the clip, the lightsheet correction,
/// the median and the vessel arm with a sensitive threshold. A tract is a
/// bright tube against a dim background, so the arms built for faint
/// capillaries — deconvolution, adaptive threshold, equalisation — are off.
pub fn tract_binarize(clip: ClipParams) -> BinarizeParams {
    let mut params = BinarizeParams {
        clip,
        ..BinarizeParams::default()
    };
    params.deconvolve = None;
    params.adaptive = None;
    params.equalize = None;
    if let Some(vesselize) = params.vesselize.as_mut() {
        vesselize.threshold = 1.0;
        vesselize.sigma = 1.0;
    }
    params
}

impl TractParams {
    pub fn validate(&self) -> Result<()> {
        let (low, high) = self.percentiles;
        GlobalThreshold::percentile(low)?;
        GlobalThreshold::percentile(high)?;
        if low >= high {
            return Err(Error::invalid(format!(
                "tract: the percentiles ({low}, {high}) must increase"
            )));
        }
        if self.sample_every == 0 {
            return Err(Error::invalid(
                "tract: a sample spacing of zero samples nothing",
            ));
        }
        Ok(())
    }
}

/// The barrier that turns the sampled values into the clip range, and hands it
/// to every block as one small fragment so it can be read after the run.
struct ClipRangeOp {
    samples_phase: usize,
    percentiles: (f64, f64),
}

impl FragmentOp for ClipRangeOp {
    fn name(&self) -> &'static str {
        "tract clip range"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(SAMPLES_STREAM, self.samples_phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            RANGE_STREAM,
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut range = [0.0; 2];
        for (slot, percentile) in range
            .iter_mut()
            .zip([self.percentiles.0, self.percentiles.1])
        {
            *slot =
                sampled_thresholds(at, SAMPLES_STREAM, GlobalThreshold::percentile(percentile)?)?
                    [0];
        }
        let mut bytes = RANGE_MAGIC.to_vec();
        for value in range {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(RANGE_STREAM, at.reduced.to_vec()))
    }
}

fn decode_range(bytes: &[u8]) -> Result<(f64, f64)> {
    let body = bytes
        .strip_prefix(RANGE_MAGIC)
        .filter(|body| body.len() == 16)
        .ok_or_else(|| Error::invalid("tract: a clip-range fragment of the wrong shape"))?;
    let low = f64::from_le_bytes(body[..8].try_into().expect("eight bytes"));
    let high = f64::from_le_bytes(body[8..].try_into().expect("eight bytes"));
    Ok((low, high))
}

/// Append the sampling and the barrier that reduces it; read the answer with
/// [`read_clip_range`].
pub fn append_clip_range_phases(
    plan: &mut PlanBuilder,
    raw: Image,
    params: &TractParams,
) -> Result<Phase> {
    params.validate()?;
    let image = if raw.id.is_supplied() {
        SourceInput::voxelwise(raw.id).holding(raw.dtype)
    } else {
        SourceInput::voxelwise(raw.id)
    };
    let samples = plan.fragments(
        SampledValuesOp::new("tract samples", SAMPLES_STREAM, image).on(
            ValueSampling::LinearIndex {
                every: params.sample_every,
            },
        )?,
    )?;
    plan.fragments(ClipRangeOp {
        samples_phase: samples.index(),
        percentiles: params.percentiles,
    })
}

/// The clip range a finished run of [`append_clip_range_phases`] computed.
pub fn read_clip_range(env: &dyn Environment) -> Result<(f64, f64)> {
    let mut range = None;
    fold_fragments(env, RANGE_STREAM, &mut |_key, bytes| {
        range = Some(decode_range(bytes)?);
        Ok(())
    })?;
    range.ok_or_else(|| Error::invalid("tract: the run published no clip range"))
}

/// Emit each block's set voxels as unit-weight points, for [`VoxelizeOp`].
struct MaskPointsOp;

impl FragmentOp for MaskPointsOp {
    fn name(&self) -> &'static str {
        "tract points"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            POINTS_STREAM,
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let pixels = at.pixels()?.as_array()?.widened();
        let mut points = Vec::new();
        for ((i, j, k), &value) in pixels.indexed_iter() {
            let global = [
                at.read.start[0] + i,
                at.read.start[1] + j,
                at.read.start[2] + k,
            ];
            let inside = (0..3).all(|axis| {
                global[axis] >= at.core.start[axis]
                    && global[axis] < at.core.start[axis] + at.core.shape[axis]
            });
            if inside && value != 0.0 {
                points.push(Point::unit(global));
            }
        }
        Ok(BlockOutput::fragment(POINTS_STREAM, encode_points(&points)))
    }
}

/// Where a tract plan's answers are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TractImages {
    /// The mask as `uint64` 0/1, ready to publish as a label image.
    pub mask: Image,
    /// Points per ball around each voxel, `uint64`.
    pub density: Image,
}

/// Append the binarisation at `range`, the points and the density.
pub fn append_tract_phases(
    c: &mut Cursor<'_>,
    raw: Image,
    params: &TractParams,
    range: (f64, f64),
) -> Result<TractImages> {
    let mut binarize = params.binarize.clone();
    binarize.clip.range = range;
    let binarized = append_binarize_phases(c, raw, &binarize)?;
    let mask = c.pixels(c.on(binarized.binary, vec![Chain::op(WidenLabelsOp)]))?;
    let points = c.plan.fragments(MaskPointsOp)?;
    let grid = c.plan.grid().clone();
    let density = c.plan.fragments(VoxelizeOp::new(
        "tract density",
        POINTS_STREAM,
        points.index(),
        ball(params.density_radius),
        &grid,
    )?)?;
    let density = c.wrote(density)?;
    Ok(TractImages { mask, density })
}

/// Every point a finished tract run emitted, in no particular order.
pub fn read_points(env: &dyn Environment) -> Result<Vec<[usize; 3]>> {
    let mut points = Vec::new();
    fold_fragments(env, POINTS_STREAM, &mut |_key, bytes| {
        points.extend(decode_points(bytes)?.into_iter().map(|point| point.at));
        Ok(())
    })?;
    points.sort_unstable();
    Ok(points)
}

/// Every point a finished tract run emitted, one row of blocks (one block
/// edge of z) at a time and sorted within it, the rows in z order: together
/// the order [`read_points`] gives, holding one row's points at a time.
pub fn for_each_point_slab(
    env: &dyn Environment,
    visit: &mut dyn FnMut(&[[usize; 3]]) -> Result<()>,
) -> Result<()> {
    let mut rows: BTreeMap<usize, Vec<FragmentKey>> = BTreeMap::new();
    for key in env.sidecar_keys(POINTS_STREAM)? {
        rows.entry(key.block[0]).or_default().push(key);
    }
    for keys in rows.values() {
        let mut points = Vec::new();
        for key in keys {
            let bytes = env
                .read_sidecar(&key.stream, key.phase, key.block)?
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "tract: the points of {key:?} were listed and are gone"
                    ))
                })?;
            points.extend(decode_points(&bytes)?.into_iter().map(|point| point.at));
        }
        // A block's points are its own core's, so a row of blocks holds every
        // point of its planes.
        points.sort_unstable();
        visit(&points)?;
    }
    Ok(())
}

/// How many points a finished tract run emitted, without holding them.
pub fn count_points(env: &dyn Environment) -> Result<usize> {
    let mut count = 0;
    fold_fragments(env, POINTS_STREAM, &mut |_key, bytes| {
        count += decode_points(bytes)?.len();
        Ok(())
    })?;
    Ok(count)
}

/// An in-memory tract map's answers.
#[derive(Debug, Clone)]
pub struct TractMap {
    pub range: (f64, f64),
    pub mask: Voxels,
    pub density: Voxels,
    pub points: Vec<[usize; 3]>,
}

/// The clip range of a volume held in memory.
pub fn clip_range(volume: &Voxels, params: &TractParams, block: [usize; 3]) -> Result<(f64, f64)> {
    let shape = volume.shape();
    let mut plan = PlanBuilder::new(shape, volume.dtype(), BlockGrid::new(shape, block)?);
    append_clip_range_phases(&mut plan, Image::new(0usize, volume.dtype()), params)?;
    let assembly = plan.finish()?;
    let (_, env) = run_in_memory(&assembly, vec![volume.clone()], &[])?;
    read_clip_range(&env)
}

/// A tract map of a volume held in memory: both runs.
pub fn tract_map(volume: Voxels, params: &TractParams, block: [usize; 3]) -> Result<TractMap> {
    let range = clip_range(&volume, params, block)?;
    let shape = volume.shape();
    let source = Image::new(0usize, volume.dtype());
    let mut plan = PlanBuilder::new(shape, volume.dtype(), BlockGrid::new(shape, block)?);
    let mut c = Cursor::new(&mut plan, source);
    let images = append_tract_phases(&mut c, source, params, range)?;
    let assembly = plan.finish()?;
    let (_, env) = run_in_memory(
        &assembly,
        vec![volume],
        &[images.mask.id, images.density.id],
    )?;
    Ok(TractMap {
        range,
        mask: read_back(&env, images.mask)?,
        density: read_back(&env, images.density)?,
        points: read_points(&env)?,
    })
}

/// A per-axis scale between two shapes, `target / source`: the affine half of
/// mapping source points into an atlas frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rescale {
    factor: [f64; 3],
}

impl Rescale {
    pub fn between(source: [usize; 3], target: [usize; 3]) -> Result<Self> {
        if source.contains(&0) || target.contains(&0) {
            return Err(Error::invalid(format!(
                "rescale: {source:?} to {target:?}; an extent of zero has no scale"
            )));
        }
        Ok(Self {
            factor: std::array::from_fn(|axis| target[axis] as f64 / source[axis] as f64),
        })
    }

    pub fn factor(&self) -> [f64; 3] {
        self.factor
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| point[axis] * self.factor[axis])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rescale_is_the_shape_ratio() {
        let rescale = Rescale::between([100, 200, 50], [25, 25, 25]).unwrap();
        assert_eq!(rescale.factor(), [0.25, 0.125, 0.5]);
        assert_eq!(rescale.apply([8.0, 8.0, 8.0]), [2.0, 1.0, 4.0]);
        assert!(Rescale::between([0, 1, 1], [1, 1, 1]).is_err());
    }

    #[test]
    fn a_range_fragment_round_trips_and_a_foreign_one_is_refused() {
        let mut bytes = RANGE_MAGIC.to_vec();
        bytes.extend_from_slice(&3.5f64.to_le_bytes());
        bytes.extend_from_slice(&900.0f64.to_le_bytes());
        assert_eq!(decode_range(&bytes).unwrap(), (3.5, 900.0));
        assert!(decode_range(b"not a range").is_err());
        bytes.pop();
        assert!(decode_range(&bytes).is_err());
    }

    #[test]
    fn bad_percentiles_are_refused() {
        let mut params = TractParams {
            percentiles: (90.0, 10.0),
            ..TractParams::default()
        };
        assert!(params.validate().is_err());
        params.percentiles = (10.0, 101.0);
        assert!(params.validate().is_err());
        params.percentiles = (10.0, 90.0);
        params.sample_every = 0;
        assert!(params.validate().is_err());
    }
}
