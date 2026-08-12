// SPDX-License-Identifier: MIT
//
// Original work for this crate. These are synthetic ops: the "noop ops and
// noop loader" the design calls for.
//
// They exist so that the *framework* — chain,
// planner, executor, valid-region derivation — can be proven before a single
// real kernel is attached to it, which is what makes a seam failure
// attributable rather than merely observed.
//
// They are compiled unconditionally rather than under `#[cfg(test)]` because
// they are also the oracle a future rewrite-rule system checks against, and
// because a probe that only exists in test builds cannot be pointed at a real
// run to isolate a regression.
//
// The four kinds, and what each is for
// ------------------------------------
// * `IdentityOp`  — the design's `NoopOp`. Declares an arbitrary reach and cost
//   but computes the identity, so *the expected output is the input*, with no
//   reference implementation anywhere. This is what catches wrong valid
//   regions, off-by-one halos, mis-stitched blocks and dropped writes.
// * `AffineOp`    — `out = in * scale + offset`, whose `constant_maps_to` is
//   exactly computable, so the empty-block short circuit can be checked to
//   produce *the same answer* as doing the work rather than merely a plausible
//   one.
// * `OpaqueOp`    — identity, but declares **no** constant mapping. One of
//   these anywhere in a phase must disable the short circuit; that is the
//   default-`None` safety property, made visible.
// * `WindowSumOp` — output at each voxel is a function of its **whole declared
//   reach**, so a short halo diverges at block edges. An identity op cannot
//   show that, because it never reads its halo.

use ndarray::{ArrayD, Axis};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::voxels::Voxels;

use super::op::{Anchor, BlockConstraint, BlockOp, Output, SideBlock};

/// The design's `NoopOp`: an arbitrary declared reach, cost and traversal
/// preference over an operation that computes the identity.
///
/// The divergence between what it *declares* and what it *does* is the point.
/// Declared reach drives the plan and the valid region; the identity gives a
/// known-correct answer to compare against. A framework bug therefore shows up
/// as output that is not the input.
pub struct IdentityOp {
    name: &'static str,
    reach: [usize; 3],
    cost: f64,
    order: Option<[usize; 3]>,
}

impl IdentityOp {
    pub fn new(name: &'static str, reach: [usize; 3]) -> Self {
        Self {
            name,
            reach,
            cost: 1.0,
            order: None,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_order(mut self, order: [usize; 3]) -> Self {
        self.order = Some(order);
        self
    }
}

impl BlockOp for IdentityOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    /// **Every element type**, because the identity is the one operation that
    /// does not have to know which. That makes it the probe for a run in a
    /// narrow type: the expected output is still the input, bit for bit,
    /// whatever the input was held as.
    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.order
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// `out = in * scale + offset`, voxelwise.
///
/// Voxelwise arithmetic with an exact `constant_maps_to`, so a short-circuited
/// block and a computed block can be asserted **equal**, not merely both
/// plausible.
pub struct AffineOp {
    name: &'static str,
    scale: f64,
    offset: f64,
    reach: [usize; 3],
    cost: f64,
    order: Option<[usize; 3]>,
}

impl AffineOp {
    pub fn new(name: &'static str, scale: f64, offset: f64, reach: [usize; 3]) -> Self {
        Self {
            name,
            scale,
            offset,
            reach,
            cost: 1.0,
            order: None,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_order(mut self, order: [usize; 3]) -> Self {
        self.order = Some(order);
        self
    }
}

impl BlockOp for AffineOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        ndarray::Zip::from(&mut out)
            .and(input)
            .for_each(|out, &value| *out = value * self.scale + self.offset);
        Ok(())
    }

    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.order
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value * self.scale + self.offset)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// The identity, declaring **no** constant mapping.
///
/// The point of this op is what it refuses to say. `constant_maps_to` returns
/// `None`, so one of these anywhere in a phase must stop the empty-block short
/// circuit — safely, because `None` is the trait default and therefore what an
/// op that has never thought about the question also says.
pub struct OpaqueOp {
    name: &'static str,
    reach: [usize; 3],
    cost: f64,
}

impl OpaqueOp {
    pub fn new(name: &'static str, reach: [usize; 3]) -> Self {
        Self {
            name,
            reach,
            cost: 1.0,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for OpaqueOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Sum over the `radius`-box around each voxel, clamped to the array it is
/// handed.
///
/// This is the halo detector. Its output at a voxel is a function of the whole
/// declared reach, so:
///
/// * given a sufficient halo, a block-edge voxel's window lies entirely inside
///   the block's read extent, and the block answer equals the whole-volume
///   answer;
/// * given a short halo, the window is truncated by the *block* rather than by
///   the *volume*, and the answer differs — silently, which is the failure mode
///   the whole design exists to convert into a loud one.
///
/// Clamping is to the handed array's bounds, which is correct at a real volume
/// boundary (there is nothing beyond to read) and wrong at a block seam (there
/// is). That asymmetry is exactly what makes the op a detector.
///
/// `declared_reach` is separate from `radius` so that **under-declaring reach**
/// can be provoked too. An op that reads further than it admits produces a
/// valid region wider than what was actually trustworthy: the tiling check
/// passes and the values are wrong. That is the failure a fused op with an
/// under-stated reach would cause, and it is worth having a probe for it.
pub struct WindowSumOp {
    name: &'static str,
    radius: [usize; 3],
    declared_reach: [usize; 3],
    cost: f64,
    order: Option<[usize; 3]>,
}

impl WindowSumOp {
    /// An honest op: declared reach equals what it reads.
    pub fn new(name: &'static str, radius: [usize; 3]) -> Self {
        Self {
            name,
            radius,
            declared_reach: radius,
            cost: 1.0,
            order: None,
        }
    }

    /// A dishonest op, for provoking the over-trust failure.
    pub fn under_declaring(name: &'static str, radius: [usize; 3], declared: [usize; 3]) -> Self {
        Self {
            name,
            radius,
            declared_reach: declared,
            cost: 1.0,
            order: None,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_order(mut self, order: [usize; 3]) -> Self {
        self.order = Some(order);
        self
    }

    /// The box sum is separable, so three clamped 1-D passes give it in O(n)
    /// rather than O(n * w^3). Separability holds because the window is a
    /// product of intervals and the clamp is applied per axis independently —
    /// `box ∩ bounds` is itself a product of intervals.
    fn window_sum_axis(data: &mut ndarray::ArrayViewMut3<'_, f64>, axis: usize, radius: usize) {
        if radius == 0 {
            return;
        }
        let len = data.shape()[axis];
        let mut prefix = vec![0.0_f64; len + 1];
        let mut lane = vec![0.0_f64; len];
        for mut line in data.lanes_mut(Axis(axis)) {
            for (slot, value) in lane.iter_mut().zip(line.iter()) {
                *slot = *value;
            }
            for index in 0..len {
                prefix[index + 1] = prefix[index] + lane[index];
            }
            for (index, value) in line.iter_mut().enumerate() {
                let low = index.saturating_sub(radius);
                let high = (index + radius + 1).min(len);
                *value = prefix[high] - prefix[low];
            }
        }
    }
}

impl BlockOp for WindowSumOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.declared_reach[axis]
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        out.assign(&input);
        for axis in 0..3 {
            Self::window_sum_axis(&mut out, axis, self.radius[axis]);
        }
        Ok(())
    }

    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.order
    }

    /// A window sum of a constant is a constant only where the window is not
    /// clamped, and the op cannot tell from `value` alone whether it will be.
    /// So: `None` for anything but zero, where the sum is zero regardless of
    /// how much of the window falls outside.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ------------------------------------------------------- fragment probes --
//
// The same idea one layer over: synthetic ops that prove the *fragment*
// machinery — declaration, gathering, reach, coverage, the store — with no real
// kernel anywhere. One per shape, and between them they close the set.
//
// All three encode their fragment as little-endian `u64`s through
// `fragment::pack_u64`, which is a probe choosing an encoding rather than the
// store knowing one.

/// `volume -> fragments`. Reads its block's pixels, writes one fragment.
///
/// The payload is `[i, j, k, valid voxels, sum of the block as f64 bits, tag]`,
/// and the reach is zero, so the read extent is the core and a reduction over
/// the stream reproduces two facts about the whole volume that no single
/// fragment holds: the blocks tile it, and their values sum to its sum. A
/// simulated environment holds no data, so the sum is zero there and the voxel
/// count is still exact — which is the part a data-free run can honestly
/// assert.
///
/// `tag` is a number the *caller* stamps on every fragment this instance
/// writes, and it is here for one reason: it is how a reader can tell which
/// producer a fragment came from. A distributed worker sets it to something
/// that identifies the process, and a consumer that counts distinct tags is
/// then measuring **cross-process** fragment visibility rather than assuming
/// it. Zero by default, which means "nobody said".
pub struct BlockSummaryOp {
    name: &'static str,
    stream: String,
    lifecycle: crate::sidecar::Lifecycle,
    tag: u64,
}

impl BlockSummaryOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: crate::sidecar::Lifecycle,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            tag: 0,
        }
    }

    /// Stamp every fragment this instance writes with `tag`. See the type's
    /// header for what it is for.
    pub fn with_tag(mut self, tag: u64) -> Self {
        self.tag = tag;
        self
    }

    /// The payload's fields, for whoever merges: block, valid voxels, the
    /// block's sum, and the producer's tag.
    pub fn read(bytes: &[u8]) -> Result<([usize; 3], usize, f64, u64)> {
        let values = crate::fragment::unpack_u64(bytes)?;
        if values.len() != 6 {
            return Err(Error::InvalidArgument(format!(
                "a block summary is six words; this one is {}",
                values.len()
            )));
        }
        Ok((
            [values[0] as usize, values[1] as usize, values[2] as usize],
            values[3] as usize,
            f64::from_bits(values[4]),
            values[5],
        ))
    }
}

impl crate::fragment::FragmentOp for BlockSummaryOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &crate::fragment::BlockView<'_>) -> Result<crate::fragment::BlockOutput> {
        let sum = match at.pixels()? {
            crate::env::BlockBuf::Array(array) => array.view::<f64>()?.iter().sum::<f64>(),
            crate::env::BlockBuf::Accounted { .. } => 0.0,
        };
        Ok(crate::fragment::BlockOutput::fragment(
            self.stream.clone(),
            crate::fragment::pack_u64(&[
                at.index[0] as u64,
                at.index[1] as u64,
                at.index[2] as u64,
                at.valid.voxels() as u64,
                sum.to_bits(),
                self.tag,
            ]),
        ))
    }
}

/// `fragments -> fragments`. **No pixel IO at all.**
///
/// Reads its declared neighbourhood of one stream and writes
/// `[i, j, k, fragments seen, voxels summed over them, distinct producer tags]`.
/// The count of fragments seen is the measured fetch count, and it must equal
/// `fragment::neighbourhood_size` for the declared reach — one for a zero-reach
/// phase, and no more than the lattice allows at an edge. The distinct tag count
/// is the evidence that a fragment crossed a process boundary: greater than one
/// means this block folded fragments written by more than one producer.
pub struct NeighbourFoldOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    reach: [usize; 3],
    stream: String,
    lifecycle: crate::sidecar::Lifecycle,
}

impl NeighbourFoldOp {
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        reach: [usize; 3],
        stream: impl Into<String>,
        lifecycle: crate::sidecar::Lifecycle,
    ) -> Self {
        Self {
            name,
            input: input.into(),
            input_phase,
            reach,
            stream: stream.into(),
            lifecycle,
        }
    }

    /// The payload's fields: the block, how many fragments it saw, their summed
    /// voxel counts, and how many distinct producers wrote them.
    pub fn read(bytes: &[u8]) -> Result<([usize; 3], usize, usize, usize)> {
        let values = crate::fragment::unpack_u64(bytes)?;
        if values.len() != 6 {
            return Err(Error::InvalidArgument(format!(
                "a neighbour fold is six words; this one is {}",
                values.len()
            )));
        }
        Ok((
            [values[0] as usize, values[1] as usize, values[2] as usize],
            values[3] as usize,
            values[4] as usize,
            values[5] as usize,
        ))
    }
}

impl crate::fragment::FragmentOp for NeighbourFoldOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<crate::fragment::FragmentInput> {
        vec![
            crate::fragment::FragmentInput::own(self.input.clone(), self.input_phase)
                .with_reach(self.reach),
        ]
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &crate::fragment::BlockView<'_>) -> Result<crate::fragment::BlockOutput> {
        let mut seen = 0u64;
        let mut voxels = 0u64;
        let mut tags = std::collections::BTreeSet::new();
        for (_, bytes) in at.fragments(&self.input) {
            let (_, block_voxels, _, tag) = BlockSummaryOp::read(bytes)?;
            seen += 1;
            voxels += block_voxels as u64;
            tags.insert(tag);
        }
        Ok(crate::fragment::BlockOutput::fragment(
            self.stream.clone(),
            crate::fragment::pack_u64(&[
                at.index[0] as u64,
                at.index[1] as u64,
                at.index[2] as u64,
                seen,
                voxels,
                tags.len() as u64,
            ]),
        ))
    }
}

/// `fragments -> volume`. The global reduction, as a phase.
///
/// Declares a reach of the whole volume — which is what makes every block read
/// the whole extent, makes a short halo collapse the valid regions and fail the
/// tiling check, and drives the cost model towards one block. It **streams** its
/// input rather than gathering it, because gathering a whole-lattice reach would
/// put every fragment in memory once per block.
///
/// It writes the summed voxel count into every voxel of its block, so the output
/// level is a constant volume whose value is knowable without a reference
/// implementation.
pub struct FragmentReduceOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    lattice: [usize; 3],
}

impl FragmentReduceOp {
    /// `lattice` is the input phase's blocks per axis: a bounded stand-in for
    /// "every block", because an unbounded reach would overflow the halo
    /// arithmetic rather than mean anything.
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        lattice: [usize; 3],
    ) -> Self {
        Self {
            name,
            input: input.into(),
            input_phase,
            lattice,
        }
    }
}

impl crate::fragment::FragmentOp for FragmentReduceOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<crate::fragment::FragmentInput> {
        vec![
            crate::fragment::FragmentInput::own(self.input.clone(), self.input_phase)
                .with_reach(self.lattice),
        ]
    }

    fn apply(&self, at: &crate::fragment::BlockView<'_>) -> Result<crate::fragment::BlockOutput> {
        let mut total = 0u64;
        let mut error: Option<Error> = None;
        at.stream_fragments(&self.input, &mut |_, bytes| {
            match BlockSummaryOp::read(bytes) {
                Ok((_, voxels, _, _)) => total += voxels as u64,
                Err(err) => error = Some(err),
            }
            Ok(())
        })?;
        if let Some(err) = error {
            return Err(err);
        }
        Ok(crate::fragment::BlockOutput::nothing().with_pixels(at.output_buffer(total as f64)?))
    }
}

// ------------------------------------------------- several outputs, and --
// -------------------------------------- a decomposition an op refuses --

/// An op that writes arrays **beside** its primary result, of stated element
/// types and stated ranks.
///
/// The primary result is the identity, on the same argument as [`IdentityOp`]:
/// the expected value is the input, so a wrong valid region or a mis-stitched
/// block shows up without a reference implementation anywhere. The side outputs
/// are the part under test, and each is described by four numbers:
///
/// * `dtype` — its element type, which is what the byte accounting turns on;
/// * `channels` — `0` for an array of the volume's own rank, and `n > 0` for one
///   of **rank 4**, `[v0, v1, v2 / fold, n]`. An op emitting one value per class
///   per output position is the ordinary case, and it is not rank 3;
/// * `fold` — how many voxels of the last axis one output element covers, so a
///   side output may be *smaller* than the volume rather than a copy of it. This
///   is what makes a byte ratio interesting: at `fold = 3` a side output is a
///   third of the elements, and a run's real cost is the sum rather than the
///   primary alone.
///
/// The value written is a mean over the folded voxels plus the channel index —
/// a function of the trustworthy voxels and of nothing else, so it is the same
/// under every decomposition, which is the property the tests assert.
pub struct SideOutputOp {
    name: &'static str,
    reach: [usize; 3],
    /// `(suffix, dtype, channels, fold)`
    sides: Vec<(&'static str, Dtype, usize, usize)>,
}

impl SideOutputOp {
    pub fn new(name: &'static str, reach: [usize; 3]) -> Self {
        Self {
            name,
            reach,
            sides: Vec::new(),
        }
    }

    /// Declare one more array beside the primary result. See the type's own
    /// documentation for what the four numbers mean.
    pub fn with_side(
        mut self,
        suffix: &'static str,
        dtype: Dtype,
        channels: usize,
        fold: usize,
    ) -> Self {
        self.sides.push((suffix, dtype, channels, fold.max(1)));
        self
    }

    /// The shape of side output `which` over `volume`.
    fn shape(&self, which: usize, volume: [usize; 3]) -> Vec<usize> {
        let (_, _, channels, fold) = self.sides[which];
        let mut shape = vec![volume[0], volume[1], volume[2] / fold];
        if channels > 0 {
            shape.push(channels);
        }
        shape
    }
}

impl BlockOp for SideOutputOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    /// Any element type: the primary result is the identity, and the side
    /// outputs are computed through `Voxels::widened`, which every type answers.
    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn side_outputs(&self, volume: [usize; 3]) -> Vec<Output> {
        self.sides
            .iter()
            .enumerate()
            .map(|(which, (suffix, dtype, _, _))| {
                Output::new(
                    format!("{}.{suffix}", self.name),
                    *dtype,
                    &self.shape(which, volume),
                )
            })
            .collect()
    }

    fn side_region(&self, which: usize, valid: &Region, _volume: [usize; 3]) -> Result<Region> {
        let (suffix, _, channels, fold) = self.sides[which];
        // The fold is exact or it is refused. A block whose valid region does
        // not divide by it would write a partial element, and there is no such
        // thing — the guard downstream would report it as a hole, which is a
        // true statement about a wrong cause.
        if !valid.start[2].is_multiple_of(fold) || !valid.shape[2].is_multiple_of(fold) {
            return Err(Error::InvalidArgument(format!(
                "{}.{suffix}: this block is trusted for {}..{} of the last axis, which does \
                 not divide by the fold of {fold}. Every output element covers {fold} voxels, \
                 so a block must be cut on a multiple of it.",
                self.name,
                valid.start[2],
                valid.start[2] + valid.shape[2]
            )));
        }
        let mut start = vec![valid.start[0], valid.start[1], valid.start[2] / fold];
        let mut shape = vec![valid.shape[0], valid.shape[1], valid.shape[2] / fold];
        if channels > 0 {
            start.push(0);
            shape.push(channels);
        }
        Ok(Region::new(&start, &shape))
    }

    fn apply_side(
        &self,
        _input: &Voxels,
        primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        // A side output here is a mean, so the shell widens rather than asking
        // the level to be `f64`. See `Voxels::widened` for why that is a
        // deliberate copy and not the default.
        let primary = primary.widened();
        // The trustworthy sub-box of the buffer. A side output is written once
        // per output voxel; the halo is read and not written.
        let mut trusted = primary.view();
        for (axis, (&start, &len)) in block
            .within
            .start
            .iter()
            .zip(block.within.shape.iter())
            .enumerate()
        {
            trusted.slice_axis_inplace(Axis(axis), ndarray::Slice::from(start..start + len));
        }
        let mut produced = Vec::with_capacity(self.sides.len());
        for (which, (_, _, channels, fold)) in self.sides.iter().enumerate() {
            let region = &block.regions[which];
            let mut array = ArrayD::zeros(ndarray::IxDyn(&region.shape));
            let planes = trusted.shape()[2] / fold;
            for i in 0..trusted.shape()[0] {
                for j in 0..trusted.shape()[1] {
                    for k in 0..planes {
                        let mut total = 0.0;
                        for step in 0..*fold {
                            total += trusted[[i, j, k * fold + step]];
                        }
                        let mean = total / *fold as f64;
                        match channels {
                            0 => array[[i, j, k]] = mean,
                            n => {
                                for channel in 0..*n {
                                    array[[i, j, k, channel]] = mean + channel as f64;
                                }
                            }
                        }
                    }
                }
            }
            produced.push(array);
        }
        Ok(produced)
    }

    fn cost_per_voxel(&self) -> f64 {
        1.0
    }
}

/// An op that accepts exactly one block extent, and nothing else.
///
/// The identity again, so that a plan which *does* satisfy the mandate produces
/// a checkable answer rather than only running. What is under test is the
/// refusal: `Constraints` offers a list of scalar edges, so an anisotropic
/// extent is not expressible as a candidate at all, and before an op could state
/// one both shipped planners returned a decomposition that checked clean and
/// could not run.
pub struct MandatedExtentOp {
    name: &'static str,
    extent: [usize; 3],
}

impl MandatedExtentOp {
    pub fn new(name: &'static str, extent: [usize; 3]) -> Self {
        Self { name, extent }
    }
}

impl BlockOp for MandatedExtentOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        if input.shape() != self.extent {
            return Err(Error::InvalidArgument(format!(
                "{}: handed {:?}, accepts only {:?}",
                self.name,
                input.shape(),
                self.extent
            )));
        }
        out.assign(input)
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn block_constraint(&self, _volume: [usize; 3]) -> Option<BlockConstraint> {
        Some(BlockConstraint::Extent(self.extent))
    }
}

/// An op whose blocks are a lattice **no block grid produces**: `count` windows
/// of fixed `edge`, spread evenly over axis 0, so that they overlap and are
/// unevenly spaced.
///
/// This is the awkward case stated as a constraint rather than left implicit.
/// `BlockGrid::cores` builds `start = index * block` — disjoint and evenly
/// strided — so no candidate, no budget and no cost model will ever produce
/// these regions, and every shipped strategy refuses. What can produce them is a
/// phase whose blocks carry an explicit fetch region: the block lattice stays the
/// unit grid of an index space and the overlap lives in the mapping, which is
/// the representation such a lattice has anyway.
///
/// The positions are a function of the **global** extent, deliberately: that is
/// what makes handing this op a block instead of the whole array move every
/// window, and a halo does not fix it.
pub struct SpreadLatticeOp {
    name: &'static str,
    count: usize,
    edge: usize,
}

impl SpreadLatticeOp {
    pub fn new(name: &'static str, count: usize, edge: usize) -> Self {
        Self {
            name,
            count: count.max(2),
            edge,
        }
    }

    /// The windows over `volume`, in block order. Evenly spread and rounded, so
    /// the step is fractional in general.
    pub fn windows(&self, volume: [usize; 3]) -> Vec<Region> {
        let span = volume[0].saturating_sub(self.edge);
        (0..self.count)
            .map(|index| {
                let start = (index * span + (self.count - 1) / 2) / (self.count - 1);
                Region::new(&[start, 0, 0], &[self.edge, volume[1], volume[2]])
            })
            .collect()
    }
}

impl BlockOp for SpreadLatticeOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn block_constraint(&self, volume: [usize; 3]) -> Option<BlockConstraint> {
        Some(BlockConstraint::Regions(self.windows(volume)))
    }
}

/// An op that **changes the element type**: any input, `bool` out, set where the
/// input is non-zero.
///
/// The probe for `produces`. Everything else in this module hands its input type
/// on unchanged, so nothing else exercises a level allocated at a width its
/// input was not — which is the case the element type became a tag for. The
/// predicate is the crate's one mask convention (`ops::voxelwise::is_set`), so a
/// `bool` result read back through `VoxelElement::into_f64` is the same mask.
pub struct NonZeroOp {
    name: &'static str,
    reach: [usize; 3],
}

impl NonZeroOp {
    pub fn new(name: &'static str, reach: [usize; 3]) -> Self {
        Self { name, reach }
    }
}

impl BlockOp for NonZeroOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `bool`, whatever it read. The declaration the level is allocated from.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.widened();
        let mut out = out.view_mut::<bool>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = value != 0.0);
        Ok(())
    }

    /// Exactly the predicate, which is pure and reads one voxel.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(if value != 0.0 { 1.0 } else { 0.0 })
    }
}

/// An op that **produces a smaller block than it is handed**: every `factor`-th
/// voxel along axis 0, and nothing else touched.
///
/// This is the probe for the wall change 5 closed. Before an op could declare
/// its output shape, `Environment::apply` allocated the output as the input's
/// own extent and `run_task` refused outright any plan whose fetch extent was
/// not its write extent — so a phase could translate its read but never resize
/// it, and a level that changes size was unrunnable however the plan was built.
/// Decimation is the smallest operation that has the property, and the *value*
/// is a value that was read, so a decimated volume can be checked against its
/// source with no reference implementation.
///
/// It is deliberately not resampling: an interpolating op would make the test
/// about the interpolation. What is under test is the **shape declaration**.
pub struct DecimateOp {
    name: &'static str,
    factor: usize,
}

impl DecimateOp {
    /// `factor` of 1 is the identity, and anything larger shrinks axis 0.
    pub fn new(name: &'static str, factor: usize) -> Self {
        Self {
            name,
            factor: factor.max(1),
        }
    }
}

impl BlockOp for DecimateOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero: a decimation reads the voxels it keeps and no others. The
    /// **stride** is not a reach — reach is measured in the space the op reads,
    /// and the voxels it skips are not read at all.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    /// The declaration this probe exists for. Rounded down, so a block whose
    /// extent is not a multiple of the factor drops the remainder rather than
    /// producing a partial sample.
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        [input[0] / self.factor, input[1], input[2]]
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let kept: Vec<usize> = (0..out.shape()[0]).map(|i| i * self.factor).collect();
        for (target, source) in kept.iter().enumerate() {
            let from = input.slice_region(&Region::new(
                &[*source, 0, 0],
                &[1, input.shape()[1], input.shape()[2]],
            ))?;
            out.assign_region(
                &Region::new(&[target, 0, 0], &[1, input.shape()[1], input.shape()[2]]),
                &from,
            )?;
        }
        Ok(())
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn ramp(shape: [usize; 3]) -> Voxels {
        let mut array = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        for (flat, value) in array.iter_mut().enumerate() {
            *value = (flat % 17) as f64 + 1.0;
        }
        array.into()
    }

    /// The separable implementation must agree with the definition, or the
    /// halo detector detects its own arithmetic rather than the halo.
    #[test]
    fn separable_window_sum_matches_the_naive_definition() {
        let shape = [7, 6, 5];
        let input = ramp(shape);
        let radius = [2usize, 1, 3];
        let op = WindowSumOp::new("window", radius);
        let mut got = Voxels::zeros(Dtype::F64, shape).unwrap();
        op.apply(&input, &mut got, &Anchor::whole(shape)).unwrap();

        let source = input.view::<f64>().unwrap();
        let mut want = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let mut total = 0.0;
                    for a in i.saturating_sub(radius[0])..(i + radius[0] + 1).min(shape[0]) {
                        for b in j.saturating_sub(radius[1])..(j + radius[1] + 1).min(shape[1]) {
                            for c in k.saturating_sub(radius[2])..(k + radius[2] + 1).min(shape[2])
                            {
                                total += source[[a, b, c]];
                            }
                        }
                    }
                    want[[i, j, k]] = total;
                }
            }
        }
        assert_eq!(got, want.into());
    }

    #[test]
    fn identity_op_declares_reach_it_does_not_use() {
        let op = IdentityOp::new("identity", [9, 9, 9]);
        assert_eq!(op.reach(1, 1000), 9);
        let input = ramp([4, 4, 4]);
        let mut out = Voxels::zeros(Dtype::F64, [4, 4, 4]).unwrap();
        op.apply(&input, &mut out, &Anchor::whole([4, 4, 4]))
            .unwrap();
        assert_eq!(out, input);
    }

    /// The output shape is what the op **declared**, and the values are the
    /// values it was handed. Both halves, because a shape declaration nobody
    /// checks the contents against is a shape declaration that can be wrong.
    #[test]
    fn a_decimating_op_declares_a_smaller_output_and_fills_it_from_its_input() {
        let op = DecimateOp::new("half", 2);
        assert_eq!(op.output_shape([8, 3, 2]), [4, 3, 2]);
        assert_eq!(op.output_shape([7, 3, 2]), [3, 3, 2]);
        assert_eq!(op.reach(0, 1000), 0);

        let input = ramp([8, 3, 2]);
        let mut out = Voxels::zeros(Dtype::F64, [4, 3, 2]).unwrap();
        op.apply(&input, &mut out, &Anchor::whole([8, 3, 2]))
            .unwrap();
        let source = input.view::<f64>().unwrap();
        let got = out.view::<f64>().unwrap();
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    assert_eq!(got[[i, j, k]], source[[i * 2, j, k]]);
                }
            }
        }
    }

    /// A decimation over a narrow element type keeps the type, which is what
    /// makes it usable as the probe for a resizing phase in any of them.
    #[test]
    fn a_decimating_op_keeps_the_element_type_it_was_handed() {
        let op = DecimateOp::new("half", 2);
        let input: Voxels = Array3::from_shape_fn((4, 1, 1), |(i, _, _)| i % 2 == 0).into();
        let mut out = Voxels::zeros(Dtype::Bool, [2, 1, 1]).unwrap();
        op.apply(&input, &mut out, &Anchor::whole([4, 1, 1]))
            .unwrap();
        assert_eq!(op.produces(Dtype::Bool), Dtype::Bool);
        assert!(out.view::<bool>().unwrap()[[0, 0, 0]]);
        assert!(out.view::<bool>().unwrap()[[1, 0, 0]]);
    }

    #[test]
    fn opaque_op_declares_no_constant_mapping() {
        assert_eq!(
            OpaqueOp::new("opaque", [0, 0, 0]).constant_maps_to(0.0),
            None
        );
        assert_eq!(
            WindowSumOp::new("window", [1, 1, 1]).constant_maps_to(0.0),
            Some(0.0)
        );
        assert_eq!(
            WindowSumOp::new("window", [1, 1, 1]).constant_maps_to(1.0),
            None
        );
    }
}
