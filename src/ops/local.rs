// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Windowed statistics evaluated on a coarse lattice and interpolated back to
// full resolution, and the threshold that compares each voxel against one.
//
// This is the family with the **global-anchoring defect class**, and everything
// below is arranged around it.
//
// The defect
// ----------
// The usual construction lays a sample lattice out relative to the array it is
// handed — `count = shape / spacing`, `first = (shape - (count - 1) * spacing) / 2`
// — and derives the upsampling factor from the same shape. Handed a block, it
// therefore lays out a *different* lattice, and the same voxel gets a different
// answer depending on which block it landed in. Not at the seams: **throughout
// the block**. And a halo does not help, because widening the read does not
// change that the lattice was laid out relative to the array passed in.
// `XY_BLOCK_SPLITTING.md` records this as the reason two axes of a real pipeline
// cannot be cut at all.
//
// Four grids, and only one of them belongs to this operation
// -----------------------------------------------------------
// The defect above is what happens when two of these are confused, so they are
// named here and kept apart by construction:
//
// | grid | whose | what it decides |
// |---|---|---|
// | the **sample lattice** | *this operation's* | where the statistic is evaluated. A parameter of the op, in volume coordinates, fixed before any decomposition exists. |
// | the **block decomposition** | the planner's | how the work is cut into tasks. Chosen for memory and parallelism, and free to change between runs of the same pipeline. |
// | the **chunk grid** | the storage's | how bytes are grouped on disk. An IO and compression concern; `chunks_touched` prices it and nothing else reads it. |
// | the originating tool's **own blocks** | nobody's, here | a fact about how some other implementation happened to run. |
//
// Only the first is an input to the answer. The other three must not be able to
// change a voxel's value, and the way that is guaranteed is that **the lattice
// is materialised as an explicit set of positions in volume coordinates** — see
// [`Sampling`] and [`SampleLattice`] — before anything is cut. A block then
// evaluates *the same lattice points* the whole volume would, at the same global
// coordinates, over the same clamped windows, gathered in the same order; the
// sums are therefore identical floating-point sums and the output is identical
// bit for bit.
//
// The fourth row is worth its own sentence, because it is where the defect came
// from. The established implementations of this technique lay the grid out
// relative to the array they are handed, so under their own block processing the
// grid becomes a function of *their* block size. That is a defect and not a
// specification, and it has been measured downstream rather than argued: a
// re-anchored run is byte-identical to a single whole-volume run at all 22
// compared stages, while the block-relative run differs by thousands of voxels
// and raising the block overlap does not reduce the difference at all. So
// another tool's block size is never something to reproduce here; the
// whole-volume lattice is the algorithm, correctly placed.
//
// A tool's *layout convention* is a different matter, and that one is worth
// matching exactly — so it is a parameter, [`Sampling`], and not a constant.
//
// There is no unanchored path in this file to fall back to. The kernel does not
// take a shape; it takes an `Anchor` and a lattice, and neither can describe
// something laid out relative to a block.
//
// What the reach has to cover, and why it is bigger than the window
// -----------------------------------------------------------------
// Two terms, because there are two steps:
//
// * the **interpolation** reaches to the lattice points bracketing the voxel,
//   which are up to a spacing short of one spacing away — and at the two ends of
//   an axis, to the nearest lattice point, which is `first` or
//   `volume - 1 - last` away;
// * the **window** reaches the element's radius beyond each of those lattice
//   points.
//
// So `reach = lattice distance + element radius`, and an implementation that
// declared only the element's radius would be short by a spacing everywhere.
// [`SampleLattice::max_distance`] computes the first term exactly from the
// volume and the spacing, so there is nothing to configure and nothing to get
// wrong independently of the lattice it describes.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp};
use crate::reach::{AxisReach, Reach};
use crate::region::Region;
use crate::voxels::Voxels;

use super::element::{select_nth, Rank, StructuringElement, Total};
use super::shapes_agree;
use super::voxelwise::combine_into;

// ------------------------------------------------------------- lattice --

/// How the sample positions are chosen along each axis.
///
/// **The convention is a parameter, not a constant.** Two tools that lay out a
/// grid differently are two values of this enum rather than two code paths, and
/// a caller that needs to match one says which. Nothing downstream branches on
/// it: everything reads the positions [`Self::positions`] produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sampling {
    /// Every `spacing` voxels, **centred**: the unsampled margins at the two
    /// ends of an axis are equal to within a voxel.
    ///
    /// A spacing of one samples every voxel, which is the no-lattice case and a
    /// legitimate parameter rather than a special one.
    Centred { spacing: [usize; 3] },
    /// `count` samples with the first on voxel `0` and the last on the last
    /// voxel, spread as evenly as integer positions allow.
    ///
    /// This is `scipy.ndimage.zoom`'s convention — its coordinate map is
    /// `i * (src - 1) / (dst - 1)`, which pins both endpoints — and it is here
    /// because a caller matching a tool built on `zoom` needs it. It is *not*
    /// the same lattice as [`Self::Centred`] and the two give different numbers.
    Endpoints { count: [usize; 3] },
    /// Stated positions, in volume coordinates, per axis.
    ///
    /// The escape hatch, and the reason the other two are not an exhaustive
    /// list: an irregular lattice is expressible, its reach is computed from the
    /// positions like every other, and nothing needs to know how it was chosen.
    At { positions: [Vec<usize>; 3] },
}

impl Sampling {
    /// Every `spacing` voxels on every axis.
    pub fn every(spacing: [usize; 3]) -> Self {
        Sampling::Centred { spacing }
    }

    /// The sample positions along `axis` of an axis `volume_len` voxels long.
    ///
    /// **Per axis, because the lattice is separable**, which is what lets
    /// [`Self::max_distance`] answer `BlockOp::reach`'s one-axis question at all.
    pub fn positions(&self, axis: usize, volume_len: usize) -> Result<Vec<usize>> {
        if volume_len == 0 {
            return Err(Error::InvalidArgument(
                "a sample lattice needs a non-empty axis".to_string(),
            ));
        }
        match self {
            Sampling::Centred { spacing } => {
                let spacing = spacing[axis];
                if spacing == 0 {
                    return Err(Error::InvalidArgument(format!(
                        "a sample spacing must be at least one voxel; got {spacing}"
                    )));
                }
                let count = (volume_len / spacing).max(1);
                let first = (volume_len - (count - 1) * spacing) / 2;
                Ok((0..count).map(|index| first + index * spacing).collect())
            }
            Sampling::Endpoints { count } => {
                let count = count[axis];
                if count == 0 {
                    return Err(Error::InvalidArgument(
                        "an endpoint lattice needs at least one sample".to_string(),
                    ));
                }
                if count == 1 || volume_len == 1 {
                    return Ok(vec![0]);
                }
                let last = (volume_len - 1) as f64;
                let steps = (count - 1) as f64;
                let mut positions: Vec<usize> = (0..count)
                    .map(|index| (index as f64 * last / steps).round() as usize)
                    .collect();
                positions.dedup();
                Ok(positions)
            }
            Sampling::At { positions } => Ok(positions[axis].clone()),
        }
    }

    /// The interpolation term of the reach along `axis`; see
    /// [`SampleLattice::max_distance`], which is where it is computed.
    pub fn max_distance(&self, axis: usize, volume_len: usize) -> usize {
        self.positions(axis, volume_len)
            .map(|positions| span_max_distance(&positions, volume_len))
            .unwrap_or(0)
    }

    /// Samples per voxel, where it is knowable without a volume.
    ///
    /// `Some` only for [`Self::Centred`], whose density is `1 / prod(spacing)`
    /// however long the axes are. The other two need the extent, and
    /// `cost_per_voxel` is not given one — so they answer `None` and the cost
    /// model charges the window at full density, which **over**-prices them.
    /// That is the safe direction (a planner isolates an op it thinks is dear),
    /// it is stated rather than hidden, and `LocalStatisticOp::with_cost` is the
    /// override for a caller who knows better.
    pub fn samples_per_voxel(&self) -> Option<f64> {
        match self {
            Sampling::Centred { spacing } => {
                Some(1.0 / (spacing[0] as f64 * spacing[1] as f64 * spacing[2] as f64).max(1.0))
            }
            _ => None,
        }
    }
}

/// Where the samples sit, in **volume** coordinates: an explicit, materialised
/// set of positions.
///
/// **Explicit is the point.** The type holds positions rather than a rule for
/// generating them, so once a lattice exists there is nothing left that could
/// re-derive it from a smaller array. That is what makes the operation
/// independent of every decomposition rather than carefully consistent with one:
/// a block does not compute *its* lattice, it looks up which of these positions
/// it needs.
///
/// It is also what admits an irregular lattice at no extra cost — the reach is
/// the widest gap rather than a spacing, and every consumer already reads
/// positions.
///
/// The positions are strictly increasing and inside the volume. Both are checked
/// at construction, because every consumer below assumes them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleLattice {
    volume: [usize; 3],
    positions: [Vec<usize>; 3],
}

impl SampleLattice {
    /// Materialise the lattice a [`Sampling`] describes over `volume`.
    ///
    /// **This is where a convention becomes a set of numbers**, and it happens
    /// once, before any decomposition exists.
    pub fn of(sampling: &Sampling, volume: [usize; 3]) -> Result<Self> {
        let positions = [
            sampling.positions(0, volume[0])?,
            sampling.positions(1, volume[1])?,
            sampling.positions(2, volume[2])?,
        ];
        Self::at(volume, positions)
    }

    /// The centred lattice of a given spacing — [`Sampling::Centred`], named for
    /// the callers that want exactly it.
    pub fn centred(volume: [usize; 3], spacing: [usize; 3]) -> Result<Self> {
        Self::of(&Sampling::Centred { spacing }, volume)
    }

    /// Stated positions, validated.
    pub fn at(volume: [usize; 3], positions: [Vec<usize>; 3]) -> Result<Self> {
        for axis in 0..3 {
            if volume[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "a sample lattice needs a non-empty volume; got {volume:?}"
                )));
            }
            let axis_positions = &positions[axis];
            if axis_positions.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "axis {axis} has no sample positions; there would be nothing to \
                     interpolate from"
                )));
            }
            if axis_positions.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(Error::InvalidArgument(format!(
                    "the sample positions on axis {axis} are not strictly increasing: \
                     {axis_positions:?}. Every lookup below is a search over them, and a \
                     repeated position is a bracket of width zero."
                )));
            }
            if let Some(&last) = axis_positions.last() {
                if last >= volume[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "axis {axis} has a sample at {last} in a volume of {}. A sample \
                         outside the volume has no window to gather.",
                        volume[axis]
                    )));
                }
            }
        }
        Ok(Self { volume, positions })
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    /// The sample positions along `axis`, in volume coordinates.
    pub fn positions(&self, axis: usize) -> &[usize] {
        &self.positions[axis]
    }

    /// How many samples along `axis`. At least one.
    pub fn count(&self, axis: usize) -> usize {
        self.positions[axis].len()
    }

    /// The volume coordinate of sample `index` along `axis`.
    pub fn centre(&self, axis: usize, index: usize) -> usize {
        let positions = &self.positions[axis];
        positions[index.min(positions.len() - 1)]
    }

    /// The greatest sample index whose centre is at or below `coordinate`,
    /// clamped into range at both ends.
    pub fn index_below(&self, axis: usize, coordinate: usize) -> usize {
        let positions = &self.positions[axis];
        positions
            .partition_point(|&position| position <= coordinate)
            .saturating_sub(1)
    }

    /// The sample indices the interpolation reads at `coordinate`, and how far
    /// between them it lands.
    ///
    /// Two decisions live here, and both are load-bearing.
    ///
    /// **The weight is applied as `a + t * (b - a)`**, not `(1 - t) * a + t * b`:
    /// where `a == b` the first form returns `a` **exactly**, whatever `t` is.
    /// That is what makes the constant algebra of a rank statistic exactly true
    /// rather than nearly true, and it is the difference between being able to
    /// declare `constant_maps_to` and having to withhold it.
    ///
    /// **A sample whose weight would be zero is not returned at all.** At a
    /// coordinate sitting exactly on a sample there is a second sample a gap
    /// away that a naive bracket would hand back with weight zero — and then the
    /// op would *read* a voxel a full gap out while its answer did not depend on
    /// it. `reach` would have to cover that read or under-declare, and an
    /// under-declared reach is the failure this crate exists to remove.
    /// Returning the degenerate bracket instead makes the two agree: the op
    /// reads what it declares and declares what it reads.
    pub fn bracket(&self, axis: usize, coordinate: usize) -> (usize, usize, f64) {
        let positions = &self.positions[axis];
        let low = self.index_below(axis, coordinate);
        let centre = positions[low];
        let high = (low + 1).min(positions.len() - 1);
        if high == low || coordinate <= centre {
            return (low, low, 0.0);
        }
        let gap = positions[high] - centre;
        (low, high, (coordinate - centre) as f64 / gap as f64)
    }

    /// The furthest a voxel's interpolation reaches for a sample, along `axis`.
    ///
    /// This is the first of the two terms of the reach, and it is computed from
    /// the **positions the interpolation actually uses**, so it cannot describe
    /// a different lattice from the one that runs. An irregular lattice is
    /// therefore priced by its widest gap and not by an average.
    pub fn max_distance(&self, axis: usize) -> usize {
        span_max_distance(&self.positions[axis], self.volume[axis])
    }

    // ------------------------------------------------------ as a grid --
    //
    // The lattice read as a *coordinate space of its own*, rather than as
    // positions inside the fine volume. This is what lets the statistic be a
    // phase whose output volume is the lattice, with the interpolation back to
    // full resolution a separate phase reading it.
    //
    // **Why that separation is worth having**, since one fused op already
    // computes both: the fused form's reach is `lattice distance + element
    // radius` against the *fine* volume, and the two terms belong to different
    // dependencies. Split, the statistic reaches the element and nothing more,
    // and the interpolation reaches one *coarse* voxel. At a spacing of 25 the
    // interpolation term is 25 fine voxels of halo per side per block that the
    // fused form fetches and the split form does not.
    //
    // The blocks of such a phase are cut on **this** grid, not on the fine one,
    // which is the arrangement `ResampleOp` already uses and the reason it is
    // right: a phase's valid regions must tile its own output, and every chunk
    // of that output must fall inside exactly one of them
    // (`check_chunk_exclusive_writes`). Cutting the fine grid and deriving
    // lattice counts would put block boundaries wherever the arithmetic landed
    // and cut chunks; cutting this grid cannot. What varies instead is how much
    // of the fine level each block reads, and that is stated per block in
    // `BlockGeometry::source`, which exists for exactly this.

    /// The shape of the coarse level this lattice defines: one voxel per sample.
    pub fn lattice_volume(&self) -> [usize; 3] {
        [self.count(0), self.count(1), self.count(2)]
    }

    /// The fine voxels that lattice points `start .. start + count` of `axis`
    /// read, given a window reaching `lo` below and `hi` above each sample.
    ///
    /// Returned as a half-open range **clamped to the fine axis**, because the
    /// window is clamped there too: a sample near the volume's edge reads what
    /// is there and the whole-volume reference does the same, so a fetch wider
    /// than the array would be asking for voxels no answer depends on.
    ///
    /// An empty `count` gives an empty range at `start`'s own position rather
    /// than at zero, so a degenerate block does not silently claim the origin.
    pub fn source_span(
        &self,
        axis: usize,
        start: usize,
        count: usize,
        lo: usize,
        hi: usize,
    ) -> (usize, usize) {
        let extent = self.volume[axis];
        if count == 0 {
            let at = self.positions[axis]
                .get(start.min(self.count(axis).saturating_sub(1)))
                .copied()
                .unwrap_or(0)
                .min(extent);
            return (at, at);
        }
        let last = start + count - 1;
        let low = self.centre(axis, start).saturating_sub(lo);
        let high = (self.centre(axis, last) + hi + 1).min(extent);
        (low, high.max(low))
    }

    /// The gap between samples on `axis`, if every gap is the same one.
    ///
    /// `None` says the lattice is irregular, and the caller that asks is
    /// [`Self::source_window`] — which needs a constant width and therefore a
    /// constant gap. A single sample has no gap and answers `Some(1)`: there is
    /// nothing to be inconsistent about, and one is the value that makes the
    /// width arithmetic below degenerate correctly.
    pub fn uniform_gap(&self, axis: usize) -> Option<usize> {
        let positions = &self.positions[axis];
        if positions.len() <= 1 {
            return Some(1);
        }
        let gap = positions[1].checked_sub(positions[0])?;
        positions
            .windows(2)
            .all(|pair| pair[1].checked_sub(pair[0]) == Some(gap))
            .then_some(gap)
    }

    /// [`Self::source_span`], widened to a **constant extent** and slid inward
    /// at the volume's ends instead of being clipped.
    ///
    /// **Why a second span rather than a wider first one.** `source_span` says
    /// what a block's samples actually read, which is what a coverage guard must
    /// compare against. This says what the block should be *handed*, and the two
    /// differ at the volume's ends on purpose: `BlockOp::output_shape` maps a
    /// block's input shape to its output shape and sees nothing else, so a block
    /// at the end holding the same number of samples as an interior one must be
    /// handed the same number of voxels or its output count could not be derived
    /// at all. Sliding inward keeps every sample's own window inside the buffer,
    /// so the extra voxels are read and simply not needed — the same trade
    /// `Reach::window` already makes for `BlockConstraint::Extent`.
    ///
    /// `None` when the lattice is irregular, because there is then no constant
    /// width to slide, or when the volume is narrower than one block's span.
    /// Both are facts about the arguments rather than failures to try, and a
    /// caller that gets `None` has the fused `LocalStatisticOp`, which needs no
    /// inversion because it writes at full resolution.
    pub fn source_window(
        &self,
        axis: usize,
        start: usize,
        count: usize,
        lo: usize,
        hi: usize,
    ) -> Option<(usize, usize)> {
        if count == 0 {
            return None;
        }
        let gap = self.uniform_gap(axis)?;
        let extent = self.volume[axis];
        let width = (count - 1).checked_mul(gap)?.checked_add(lo + hi + 1)?;
        if width > extent {
            return None;
        }
        let ideal = self.centre(axis, start).saturating_sub(lo);
        // Slid, not clipped: pulled back just far enough that the full width
        // fits, which leaves every sample of the block inside it.
        let low = ideal.min(extent - width);
        Some((low, low + width))
    }

    /// How many samples a block handed `extent` voxels on `axis` holds.
    ///
    /// The inverse of [`Self::source_window`]'s width, and the reason that
    /// function insists on a constant one. `None` for an irregular lattice, or
    /// for an extent that is not a whole number of gaps past the window.
    pub fn samples_for_window(
        &self,
        axis: usize,
        extent: usize,
        lo: usize,
        hi: usize,
    ) -> Option<usize> {
        let gap = self.uniform_gap(axis)?;
        let window = lo + hi + 1;
        let spanned = extent.checked_sub(window)?;
        (gap > 0 && spanned % gap == 0).then(|| spanned / gap + 1)
    }

    /// [`Self::source_span`] on every axis: the fine region a block of the
    /// coarse grid must be handed.
    ///
    /// `read` is stated in **lattice indices** — it is a region of
    /// [`Self::lattice_volume`], which is what the phase's blocks are cut from —
    /// and the region returned is in fine voxels. The two coordinate systems are
    /// the whole reason this function exists, and mixing them up is the mistake
    /// it is arranged to make impossible: neither argument nor result can be
    /// passed for the other without the extents disagreeing.
    pub fn source_region(&self, read: &Region, window: [(usize, usize); 3]) -> Result<Region> {
        if read.start.len() != 3 || read.shape.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "a sample lattice is 3-D; this region has rank {}",
                read.start.len()
            )));
        }
        let mut start = [0usize; 3];
        let mut shape = [0usize; 3];
        for axis in 0..3 {
            if read.start[axis] + read.shape[axis] > self.count(axis) {
                return Err(Error::InvalidArgument(format!(
                    "lattice region {:?}+{:?} runs past axis {axis} of the lattice, which holds \
                     {} sample(s). This region is stated in lattice indices, not in voxels.",
                    read.start,
                    read.shape,
                    self.count(axis)
                )));
            }
            let (low, high) = self.source_span(
                axis,
                read.start[axis],
                read.shape[axis],
                window[axis].0,
                window[axis].1,
            );
            start[axis] = low;
            shape[axis] = high - low;
        }
        Ok(Region::new(&start, &shape))
    }
}

/// The interpolation term of the reach, from a list of positions.
///
/// Three cases, and the maximum of them:
///
/// * a voxel before the first sample reads that sample, `first` away;
/// * a voxel after the last reads that one, `volume - 1 - last` away;
/// * a voxel strictly between two samples reads both, and the further of the
///   pair is `gap - 1` away at worst. It is not `gap`: a voxel sitting exactly
///   on a sample gets the degenerate bracket, so the sample a full gap away is
///   never read. See [`SampleLattice::bracket`], which is where that is arranged
///   and where it has to stay arranged for this number to be true.
///
/// A lattice of every voxel therefore contributes nothing at all, which is right
/// — every voxel is its own sample and there is no interpolation to reach for.
fn span_max_distance(positions: &[usize], volume_len: usize) -> usize {
    let (Some(&first), Some(&last)) = (positions.first(), positions.last()) else {
        return 0;
    };
    let mut distance = first.max(volume_len.saturating_sub(1).saturating_sub(last));
    for pair in positions.windows(2) {
        distance = distance.max(pair[1] - pair[0] - 1);
    }
    distance
}

/// The interpolation term of the reach for a **centred** lattice, along one
/// axis. See [`span_max_distance`] for the derivation.
pub fn axis_max_distance(volume_len: usize, spacing: usize) -> usize {
    if volume_len == 0 || spacing == 0 {
        return 0;
    }
    Sampling::Centred {
        spacing: [spacing; 3],
    }
    .max_distance(0, volume_len)
}

// ------------------------------------------------------------- kernels --

/// Evaluate `reduce` over `element` at every lattice point, then interpolate
/// back to every voxel of the buffer.
///
/// Generic over the element type, which the algorithm allows because the gather
/// only copies; what it may do with the copies is the reducer's business.
///
/// `at` is not decoration. It is where the lattice comes from — `at.volume`
/// gives the extent the lattice is laid out over, `at.offset` says which part of
/// it this buffer holds — and it is why the same voxel gets the same answer
/// under every decomposition.
///
/// A window is clamped to the buffer. At a real volume boundary that is the
/// global clamp and is right; short of a sufficient halo it is a truncation the
/// whole volume would not have made, and the values differ — which is how the
/// halo guard is *seen* to matter rather than merely asserted.
pub fn local_statistic_into<T, F>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    element: &StructuringElement,
    lattice: &SampleLattice,
    mut reduce: F,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy,
    // `FnMut` rather than `Fn`, so the caller can hand in a closure holding a
    // scratch buffer it reuses across lattice points. Nothing here calls
    // `reduce` more than once per point or in any order but the lattice's, so
    // the looser bound costs no guarantee.
    F: FnMut(&mut [T]) -> f64,
{
    shapes_agree(input.shape(), out.shape(), "local_statistic_into")?;
    if lattice.volume() != at.volume {
        return Err(Error::InvalidArgument(format!(
            "local_statistic_into: the lattice is over {:?} but the anchor says {:?}",
            lattice.volume(),
            at.volume
        )));
    }
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    for axis in 0..3 {
        if at.offset[axis] + shape[axis] > at.volume[axis] {
            return Err(Error::InvalidArgument(format!(
                "local_statistic_into: a buffer of {shape:?} at {:?} does not fit a volume of {:?}",
                at.offset, at.volume
            )));
        }
    }
    if shape.contains(&0) {
        return Ok(());
    }

    // Which lattice points any voxel of this buffer can read. Everything
    // outside this range is somebody else's block's business.
    let mut low = [0usize; 3];
    let mut high = [0usize; 3];
    for axis in 0..3 {
        let start = at.offset[axis];
        let end = start + shape[axis] - 1;
        // Asked of `bracket` rather than derived beside it, so that the samples
        // computed are exactly the samples read — the same agreement `reach`
        // depends on, made in one place.
        low[axis] = lattice.bracket(axis, start).0;
        high[axis] = lattice.bracket(axis, end).1;
    }

    // The sample grid, in global lattice indices offset by `low`.
    let grid_shape = (
        high[0] - low[0] + 1,
        high[1] - low[1] + 1,
        high[2] - low[2] + 1,
    );
    let mut grid = Array3::<f64>::zeros(grid_shape);
    let mut window: Vec<T> = Vec::with_capacity(element.len());
    for p in 0..grid_shape.0 {
        for q in 0..grid_shape.1 {
            for r in 0..grid_shape.2 {
                let centre = [
                    lattice.centre(0, low[0] + p) as isize,
                    lattice.centre(1, low[1] + q) as isize,
                    lattice.centre(2, low[2] + r) as isize,
                ];
                window.clear();
                for offset in element.offsets() {
                    let mut index = [0usize; 3];
                    let mut inside = true;
                    for axis in 0..3 {
                        // The window is stated in volume coordinates and read
                        // out of the buffer. The buffer lies inside the volume,
                        // so this is the global clamp intersected with what is
                        // held — and with a sufficient halo the intersection is
                        // the global clamp itself.
                        let global = centre[axis] + offset[axis];
                        let local = global - at.offset[axis] as isize;
                        if local < 0 || local >= shape[axis] as isize {
                            inside = false;
                            break;
                        }
                        index[axis] = local as usize;
                    }
                    if inside {
                        window.push(input[index]);
                    }
                }
                grid[[p, q, r]] = reduce(&mut window);
            }
        }
    }

    // Interpolate back. The brackets are per axis, so they are computed once per
    // plane rather than once per voxel.
    let brackets: Vec<Vec<(usize, usize, f64)>> = (0..3)
        .map(|axis| {
            (0..shape[axis])
                .map(|step| {
                    let (a, b, t) = lattice.bracket(axis, at.offset[axis] + step);
                    (a - low[axis], b - low[axis], t)
                })
                .collect()
        })
        .collect();

    for i in 0..shape[0] {
        let (a0, b0, t0) = brackets[0][i];
        for j in 0..shape[1] {
            let (a1, b1, t1) = brackets[1][j];
            for k in 0..shape[2] {
                let (a2, b2, t2) = brackets[2][k];
                let at_edge = |p: usize, q: usize| lerp(grid[[p, q, a2]], grid[[p, q, b2]], t2);
                let at_face = |p: usize| lerp(at_edge(p, a1), at_edge(p, b1), t1);
                out[[i, j, k]] = lerp(at_face(a0), at_face(b0), t0);
            }
        }
    }
    Ok(())
}

/// `a + t * (b - a)`, which returns `a` exactly where `a == b`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Compare each voxel against a co-located threshold.
///
/// Generic over `PartialOrd`, which is all a comparison needs, and over the
/// output type, which the comparison never inspects.
pub fn threshold_against_into<T, U>(
    input: ArrayView3<'_, T>,
    threshold: ArrayView3<'_, T>,
    above: U,
    below: U,
    out: ArrayViewMut3<'_, U>,
) -> Result<()>
where
    T: PartialOrd,
    U: Copy,
{
    combine_into(input, threshold, out, |value, level| {
        if value > level {
            above
        } else {
            below
        }
    })
}

// ---------------------------------------------------------- statistics --

/// A reduction over a gathered window, implementable **outside this crate**.
///
/// This is the extension point for a windowed statistic the crate does not
/// ship. `Statistic` is otherwise a closed enum, and a closed enum is the wrong
/// shape for something a caller may reasonably need to extend — `Isodata` had to
/// be added *inside* the crate for exactly that reason.
///
/// **Why a trait object here, when `MapFn` is a type parameter.** A voxelwise
/// map runs once per voxel, so an indirect call there is the whole cost and
/// `ops::voxelwise` is generic to avoid it. A reduction runs once per **lattice
/// point**, after a gather of the window — 27 to 729 voxels for the elements
/// this crate is used with — so one virtual call is amortised by that factor and
/// costs nothing measurable. Paying a generic parameter on a type that is
/// stored, cloned, compared and hashed everywhere would buy nothing back.
///
/// **Why `f64` rather than generic.** A trait with a generic method is not
/// object-safe. Nothing is lost: `LocalStatistic::evaluate_into` is stated in
/// `f64` on both sides already, so the op path never had another element type to
/// offer. The genericity of [`Statistic::reduce`] survives for the shipped
/// variants and for direct callers of the kernel.
pub trait Reducer: Send + Sync + 'static {
    /// Reduce the gathered window to one value.
    ///
    /// `window` is in the element's canonical order and may be reordered — a
    /// selection is allowed to permute it. `full` is the element's size before
    /// any clamping at a volume boundary, so a rank-like reduction can tell a
    /// truncated window from a whole one; `window.len()` is what actually landed
    /// inside.
    fn reduce(&self, window: &mut [f64], full: usize) -> f64;

    /// What this gives for a window that is uniformly `value`, where that is
    /// **exactly** true.
    ///
    /// `None` is the default and is always safe: it disables the short circuit,
    /// so a block is computed rather than assumed. Declaring something wrong
    /// here is not safe, which is why silence is the default — see `ops/mod.rs`.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    /// Cost of one evaluation over a window of `window` voxels, in the units of
    /// `ops::cost` — beyond the gather itself, which the caller already charges.
    fn cost_per_sample(&self, _window: usize) -> f64 {
        0.0
    }

    /// A stable identity, so that a `Statistic` holding this can still be
    /// compared and hashed.
    ///
    /// `Statistic` is `Eq + Hash` because plans are compared and fingerprinted,
    /// and a trait object has no derivable identity. Two reducers that would
    /// compute the same answer must return the same key; two that would not,
    /// must not. Returning a constant makes every instance of your type equal,
    /// which is right for a parameterless reducer and wrong for one with fields.
    fn key(&self) -> (&'static str, u64);
}

/// What to reduce a window to.
#[derive(Debug, Clone)]
pub enum Statistic {
    /// The arithmetic mean.
    Mean,
    /// The population standard deviation, computed in two passes so that it
    /// cannot go negative under the square root.
    Deviation,
    /// An order statistic, through the same [`Rank`] the rank filter uses.
    Rank(Rank),
    /// The **isodata** threshold of the window's histogram: the value that is
    /// its own two-class midpoint. See [`Isodata`], which holds the two
    /// parameters and states the rule.
    ///
    /// This is not an approximation of any of the three above and is not
    /// approximated by them. A mean answers "what is the level here"; this
    /// answers "where does this window separate into two classes", and on a
    /// bimodal window the two are far apart — the mean sits wherever the mass
    /// is, the isodata threshold sits between the modes.
    Isodata(Isodata),
    /// A reduction supplied by the caller. See [`Reducer`].
    ///
    /// `Arc` rather than `Box` so that `Statistic` stays `Clone`, which
    /// `LocalStatistic` and every op holding one rely on.
    Custom(std::sync::Arc<dyn Reducer>),
}

/// Compared and hashed **by identity, not by behaviour**, because a trait object
/// has no derivable equality. The shipped variants compare structurally as they
/// always did; a custom one compares by the key its [`Reducer`] states.
impl PartialEq for Statistic {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Statistic::Mean, Statistic::Mean) => true,
            (Statistic::Deviation, Statistic::Deviation) => true,
            (Statistic::Rank(left), Statistic::Rank(right)) => left == right,
            (Statistic::Isodata(left), Statistic::Isodata(right)) => left == right,
            (Statistic::Custom(left), Statistic::Custom(right)) => left.key() == right.key(),
            _ => false,
        }
    }
}

impl Eq for Statistic {}

impl std::hash::Hash for Statistic {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Statistic::Mean | Statistic::Deviation => {}
            Statistic::Rank(rank) => rank.hash(state),
            Statistic::Isodata(isodata) => isodata.hash(state),
            Statistic::Custom(reducer) => reducer.key().hash(state),
        }
    }
}

impl std::fmt::Debug for dyn Reducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (name, tag) = self.key();
        write!(f, "Reducer({name}#{tag})")
    }
}

impl Statistic {
    /// Reduce a gathered window. The window is in the element's canonical order,
    /// so a summation over it is the same summation everywhere.
    pub fn reduce<T>(&self, window: &mut [T], full: usize) -> f64
    where
        T: Ord + Copy + Into<f64>,
    {
        let mut scratch = Vec::new();
        self.reduce_with(window, full, &mut scratch)
    }

    /// [`Self::reduce`], over a scratch buffer the caller owns.
    ///
    /// **The buffer is the whole of the difference, and it only matters for a
    /// custom reducer.** A [`Reducer`] is stated in `f64` — it has to be, since
    /// a trait object cannot be generic over the element type, and being
    /// injectable without recompiling this crate is the entire point of it — so
    /// the gathered window is widened before the call. Done inside
    /// [`Self::reduce`] that is an allocation per window; with a buffer that
    /// lives outside the loop it is none, which is the same arrangement
    /// `rank_filter_into` uses for the window it gathers.
    ///
    /// The virtual call itself is not worth avoiding here and the asymmetry with
    /// `MapFn` is deliberate: a `MapFn` runs per voxel on about one arithmetic
    /// operation, which is why it is generic and monomorphised, and a `Reducer`
    /// runs once per *window* on `|element|` of them. Amortised over a window,
    /// an indirect call is not measurable; the allocation was.
    ///
    /// The shipped variants take the generic path and touch `scratch` not at
    /// all, so a caller using them pays nothing for its existence.
    pub fn reduce_with<T>(&self, window: &mut [T], full: usize, scratch: &mut Vec<f64>) -> f64
    where
        T: Ord + Copy + Into<f64>,
    {
        // `Isodata` is asked before the emptiness test rather than after,
        // because it has an answer for an empty window that is a *parameter* —
        // the value it falls back to when the histogram yields no threshold —
        // and the shared `0.0` below would silently override it. The other
        // three keep the exit they have always had: a mean of nothing is
        // `0 / 0`, and zero is the answer this module has always given for it.
        if let Statistic::Isodata(isodata) = self {
            return isodata.of(window.iter().map(|&value| value.into()));
        }
        if let Statistic::Custom(reducer) = self {
            scratch.clear();
            scratch.extend(window.iter().map(|&value| value.into()));
            return reducer.reduce(scratch, full);
        }
        if window.is_empty() {
            return 0.0;
        }
        match self {
            Statistic::Mean => mean_of(window),
            Statistic::Deviation => {
                let mean = mean_of(window);
                let total: f64 = window
                    .iter()
                    .map(|&value| {
                        let centred = value.into() - mean;
                        centred * centred
                    })
                    .sum();
                (total / window.len() as f64).sqrt()
            }
            Statistic::Rank(rank) => {
                let index = rank.resolve(full, window.len());
                select_nth(window, index).map(Into::into).unwrap_or(0.0)
            }
            // Both answered above, before the emptiness test.
            Statistic::Isodata(_) | Statistic::Custom(_) => {
                unreachable!("answered before the emptiness test")
            }
        }
    }

    /// What this statistic gives for a window that is uniformly `value`, when
    /// that is **exactly** what it gives and not merely what it gives in real
    /// arithmetic.
    ///
    /// * A **rank** selects a value that was read, so the answer is `value`
    ///   itself, bit for bit, at any truncation.
    /// * A **mean** does not. `(v + v + ... + v) / m` is not `v` in binary
    ///   floating point for a general `v` and `m` — `0.1` summed three times and
    ///   divided by three is `0.10000000000000002` — so declaring `Some(value)`
    ///   would make a short-circuited block differ from a computed one in the
    ///   last bit. It is exact at zero, where every partial sum is zero, and
    ///   that is the case declared.
    /// * A **deviation** is exact at zero for the same reason and, for the same
    ///   reason as the mean, is *not* exactly zero elsewhere: the mean it
    ///   subtracts is already off by an ulp, so the residuals are not zero.
    /// * An **isodata** threshold of a constant window is that constant, and it
    ///   is exact for a reason that is arithmetic rather than lucky: the
    ///   histogram's bin width is `(max - min) / bins`, a constant window has
    ///   `max == min`, and the rule returns the maximum without building a
    ///   histogram at all. `max` over equal values is one of those values. See
    ///   [`Isodata::of`], where that exit is. It is declared only for a
    ///   **finite** constant, because a window of infinities or `NaN`s has no
    ///   finite value to histogram and takes the fallback instead — which is a
    ///   parameter and not the constant.
    pub fn constant_maps_to(&self, value: f64) -> Option<f64> {
        match self {
            Statistic::Rank(_) => Some(value),
            Statistic::Isodata(_) => value.is_finite().then_some(value),
            Statistic::Mean | Statistic::Deviation => (value == 0.0).then_some(0.0),
            Statistic::Custom(reducer) => reducer.constant_maps_to(value),
        }
    }

    /// What this reducer costs per sample **beyond the gather**, over a window
    /// of `window` voxels, relative to a voxelwise map.
    ///
    /// Zero for the three that walk the gathered window once and stop: their
    /// whole cost is proportional to the window, which `cost_for` already
    /// charges per element voxel at the gather's own rate. An isodata sweep is
    /// the exception and it has **two** terms, which is what no existing
    /// constant could have said: it walks the window again to bin it, and then
    /// walks the histogram, and the second of those is a function of the bin
    /// count alone. A 256-bin threshold over a 27-voxel window is dominated by
    /// its bins; the same threshold over a 729-voxel window is not.
    pub(super) fn cost_per_sample(&self, window: usize) -> f64 {
        match self {
            Statistic::Isodata(isodata) => {
                ISODATA_COST_PER_ELEMENT_VOXEL * window as f64
                    + ISODATA_COST_PER_BIN * isodata.bins() as f64
            }
            Statistic::Custom(reducer) => reducer.cost_per_sample(window),
            Statistic::Mean | Statistic::Deviation | Statistic::Rank(_) => 0.0,
        }
    }
}

/// The isodata (Ridler-Calvard) threshold of a window's histogram, and the two
/// things about it that have to be stated rather than assumed.
///
/// The rule
/// --------
/// Bin the window's finite values into `bins` equal-width bins spanning
/// `[min, max]`, with centres at `min + (i + 0.5) * width`. Then a bin centre
/// `c_i` is **a** threshold when the mean of everything at or below it and the
/// mean of everything strictly above it average to a value that lands inside
/// bin `i` itself:
///
/// ```text
/// m = (mean of values in bins 0..=i  +  mean of values in bins i+1..) / 2
/// c_i is a threshold  <=>  0 <= m - c_i < width
/// ```
///
/// The answer is the **last** such centre. Three consequences worth having in
/// front of you, because they are where two implementations of "isodata" differ:
///
/// * **It is not iterated to a fixed point.** The classical presentation starts
///   from a guess, splits, re-averages and repeats. This is the direct form: it
///   tests every bin for the fixed-point property in one pass and reports the
///   ones that have it. Where the histogram admits several, an iteration would
///   return whichever one its starting guess fell into; this returns the
///   highest, always, whatever the data. That is a **tie rule**, not a detail —
///   a bimodal window with a long bright tail can easily admit two.
/// * **Bin `i` is compared with a half-open interval**, `>= 0` on one side and
///   `< width` on the other. A midpoint landing exactly on `c_i` is a
///   threshold; one landing exactly on `c_i + width` belongs to the next bin
///   and is not.
/// * **The top bin is never a threshold.** The loop stops one short, where
///   everything strictly above is empty by construction; bins whose lower or
///   upper class is empty are skipped for the same reason, since their class
///   mean would be `0 / 0`.
///
/// The two parameters, and why neither has a default here
/// ------------------------------------------------------
/// **`bins`** is the resolution of the histogram, and the answer is a bin
/// centre — so it is also the quantisation of the answer. It is a parameter
/// because the implementations this rule is shared with choose it by inspecting
/// the source array's *element type*: one bin per integer value for an integer
/// array, a fixed count for a float one. A statistic in this module is handed
/// `f64` on both sides (see [`LocalStatisticOp::apply`]) and cannot see an
/// element type to dispatch on, and dispatching on whether the values *look*
/// integral is a documented way to get this wrong — so the choice is stated by
/// the caller instead. A caller reproducing the float-array behaviour of those
/// implementations says `256`.
///
/// **`fallback`** is what to answer when the histogram admits no threshold at
/// all — an empty window, a window with no finite value in it, or a histogram
/// in which no bin has the property above. There is no value derivable from the
/// data for that case: the rule genuinely has no answer, and the choice of what
/// to say instead belongs to whoever will consume the number. It is a parameter
/// for exactly that reason.
#[derive(Debug, Clone, Copy)]
pub struct Isodata {
    bins: usize,
    fallback: f64,
}

impl Isodata {
    /// `bins` equal-width bins, and the value to answer where the rule has no
    /// answer. Both are described on [`Isodata`].
    ///
    /// `bins` must be at least one and `fallback` must be finite. The finiteness
    /// is not fussiness: a `NaN` fallback would make a window that took it
    /// compare unequal to itself, and every byte-identity check downstream in
    /// this crate would stop meaning anything.
    pub fn new(bins: usize, fallback: f64) -> Result<Self> {
        if bins == 0 {
            return Err(Error::InvalidArgument(
                "an isodata histogram needs at least one bin; with none there is nothing to \
                 take a threshold from"
                    .to_string(),
            ));
        }
        if !fallback.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "an isodata fallback must be finite, got {fallback}. It is a value that can \
                 end up in the output volume, and a NaN there compares unequal to itself — \
                 which turns every decomposition check in this crate into a pass that means \
                 nothing."
            )));
        }
        // `-0.0` is normalised to `+0.0` so that the equality below agrees with
        // the hash: the two compare equal as numbers and differ as bits, and a
        // `Hash` taken over the bits would then break `Hash`'s contract with
        // `Eq`. Normalising once here is cheaper than a hash that has to
        // remember the exception.
        let fallback = if fallback == 0.0 { 0.0 } else { fallback };
        Ok(Self { bins, fallback })
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    pub fn fallback(&self) -> f64 {
        self.fallback
    }

    /// The rule, over whatever `values` yields. See [`Isodata`] for what it is.
    ///
    /// Non-finite values are dropped rather than binned: they have no bin, and
    /// a `min` or a `max` taken over them would be either infinite (making every
    /// bin width infinite) or `NaN`-poisoned.
    ///
    /// The iterator is consumed once and the finite values are collected,
    /// because the rule needs three passes over them — the extremes, then the
    /// counts — and a caller's iterator is not required to be cheap or
    /// repeatable.
    pub fn of<I>(&self, values: I) -> f64
    where
        I: Iterator<Item = f64>,
    {
        let values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
        if values.is_empty() {
            return self.fallback;
        }
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let width = (maximum - minimum) / self.bins as f64;
        if !(width > 0.0) {
            // Every value is the same one. There is no histogram to build and
            // no two classes to separate, and the answer is that value —
            // **exactly**, because `max` returns one of its operands rather
            // than computing anything. This is the exit `constant_maps_to`
            // rests on, and it is also what a `bins` so large that the width
            // underflows would take, which is the right answer for that too.
            return maximum;
        }

        let mut counts = vec![0.0f64; self.bins];
        for &value in &values {
            // The cast truncates towards zero and the value is at or above the
            // minimum, so the index is never negative; the clamp is for the
            // maximum itself, whose exact quotient is `bins`.
            let bin = (((value - minimum) / width) as usize).min(self.bins - 1);
            counts[bin] += 1.0;
        }
        let centres: Vec<f64> = (0..self.bins)
            .map(|index| minimum + (index as f64 + 0.5) * width)
            .collect();
        if self.bins == 1 {
            return centres[0];
        }

        // Cumulative count and cumulative intensity, in bin order. Both are the
        // "at or below" halves; the "strictly above" halves are the totals less
        // these, which is what makes the whole sweep two passes rather than
        // `bins` passes.
        let mut below = Vec::with_capacity(self.bins);
        let mut running = 0.0;
        for count in &counts {
            running += count;
            below.push(running);
        }
        let total = running;
        let mut intensity = Vec::with_capacity(self.bins);
        let mut running = 0.0;
        for (count, centre) in counts.iter().zip(&centres) {
            running += count * centre;
            intensity.push(running);
        }
        let total_intensity = running;

        // The bin width is taken as the difference of two centres rather than
        // as `width`: the comparison below is against the gap between the
        // centres it is comparing, and recomputing it from the same subtraction
        // the centres came from keeps the two exactly consistent where
        // `minimum + 1.5 * width - (minimum + 0.5 * width)` is not exactly
        // `width`.
        let step = centres[1] - centres[0];
        let mut found = self.fallback;
        for index in 0..self.bins - 1 {
            let above = total - below[index];
            if below[index] == 0.0 || above == 0.0 {
                continue;
            }
            let lower = intensity[index] / below[index];
            let higher = (total_intensity - intensity[index]) / above;
            let midpoint = (lower + higher) / 2.0;
            let distance = midpoint - centres[index];
            if distance >= 0.0 && distance < step {
                // The last one wins; see [`Isodata`] on why that is the tie
                // rule and not an artefact of the sweep's direction.
                found = centres[index];
            }
        }
        found
    }
}

/// Equality by **bits** on the fallback, so that it agrees with [`Hash`].
///
/// The constructor refuses a non-finite fallback and normalises `-0.0`, so on
/// every value that can exist here bitwise equality and numeric equality are the
/// same relation — which is what makes deriving `Eq` legitimate at all for a
/// type holding an `f64`, and it is worth having: [`Statistic`] is `Eq + Hash`
/// and a variant that could not be would take that away from every caller.
impl PartialEq for Isodata {
    fn eq(&self, other: &Self) -> bool {
        self.bins == other.bins && self.fallback.to_bits() == other.fallback.to_bits()
    }
}

impl Eq for Isodata {}

impl std::hash::Hash for Isodata {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bins.hash(state);
        self.fallback.to_bits().hash(state);
    }
}

fn mean_of<T: Copy + Into<f64>>(window: &[T]) -> f64 {
    let total: f64 = window.iter().map(|&value| value.into()).sum();
    total / window.len() as f64
}

impl From<Total> for f64 {
    fn from(value: Total) -> f64 {
        value.0
    }
}

/// A statistic, a window and a [`Sampling`]: everything the lattice needs except
/// the volume, which comes from the anchor.
///
/// The sampling is held as a **convention**, not as a materialised lattice,
/// because a `LocalStatistic` is built before any volume is known — it is a
/// parameter of a chain, and a chain outlives the array it runs on. The lattice
/// itself is materialised once per call from `Anchor::volume`, which is the
/// global extent, and never from the buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalStatistic {
    element: StructuringElement,
    sampling: Sampling,
    statistic: Statistic,
}

impl LocalStatistic {
    /// The centred lattice of a given spacing, which is the ordinary case.
    ///
    /// `spacing` of `[1, 1, 1]` samples every voxel, which is the no-lattice
    /// case and is a legitimate parameter rather than a special one.
    pub fn new(
        element: StructuringElement,
        spacing: [usize; 3],
        statistic: Statistic,
    ) -> Result<Self> {
        Self::sampled(element, Sampling::Centred { spacing }, statistic)
    }

    /// The same, with the sampling convention stated.
    pub fn sampled(
        element: StructuringElement,
        sampling: Sampling,
        statistic: Statistic,
    ) -> Result<Self> {
        if element.is_empty() {
            return Err(Error::InvalidArgument(
                "a local statistic needs a non-empty window".to_string(),
            ));
        }
        // Materialised once here, over a nominal extent, so that a sampling that
        // cannot produce positions is refused when it is *stated* rather than at
        // the first block. The positions themselves are discarded: the real ones
        // come from the anchor.
        for axis in 0..3 {
            sampling.positions(axis, 1)?;
        }
        Ok(Self {
            element,
            sampling,
            statistic,
        })
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    /// Cloned rather than borrowed so that every existing caller keeps
    /// compiling; a `Statistic` is a small enum or an `Arc` bump, and this is
    /// called once per `evaluate_into` rather than per voxel.
    pub fn statistic(&self) -> Statistic {
        self.statistic.clone()
    }

    /// The two terms: how far the interpolation reaches for a sample, plus how
    /// far that sample's window reaches beyond it.
    ///
    /// **The worst case over the volume**, which is what a symmetric per-axis
    /// integer can say. A globally anchored lattice does not line up with block
    /// boundaries — that is the whole point of it — so the samples a *particular*
    /// block needs sit at a different distance outside its core for every block,
    /// and most blocks reach less than this. Two things say tighter things:
    /// [`Self::reach_sides`], which drops the symmetry an even window does not
    /// have and is what `LocalStatisticOp::reach_spec` declares, and
    /// [`Self::halo`], which drops the "over the volume" and is a table.
    pub fn reach(&self, axis: usize, volume_len: usize) -> usize {
        let (lo, hi) = self.reach_sides(axis, volume_len);
        lo.max(hi)
    }

    /// The same two terms, per side.
    ///
    /// The lattice term is symmetric — a sample may be that far away in either
    /// direction — and the window term is not, so an element with an even
    /// extent makes the whole thing asymmetric by exactly the element's own
    /// asymmetry. Stated here so that [`Self::halo`] and both op shells derive
    /// it from one place; two derivations of one quantity is the arrangement
    /// this crate keeps out of its geometry.
    pub fn reach_sides(&self, axis: usize, volume_len: usize) -> (usize, usize) {
        let distance = self.sampling.max_distance(axis, volume_len);
        let (lo, hi) = self.element.sides(axis);
        (distance + lo, distance + hi)
    }

    /// [`Self::reach_sides`] on every axis, as the crate's `Reach`.
    pub fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        Reach::asymmetric([
            self.reach_sides(0, volume[0]),
            self.reach_sides(1, volume[1]),
            self.reach_sides(2, volume[2]),
        ])
    }

    /// The halo each block actually needs, as a table.
    ///
    /// **This is the consequence of global anchoring, priced.** The lattice is
    /// fixed in volume coordinates and knows nothing about the decomposition, so
    /// its points do not line up with block boundaries — a block may hold three
    /// samples or four, and the nearest sample outside its core may be one voxel
    /// away or a whole gap away. `reach` has to be the **worst** case over every
    /// block, because it is a symmetric per-axis integer; most blocks need less,
    /// and this says how much less for each.
    ///
    /// Derived per block from the brackets its own voxels take: the lowest
    /// sample the first voxel of the core interpolates from, the highest the last
    /// voxel does, each widened by the window's radius. Where every entry agrees
    /// it collapses to the uniform form, so a plan that does not need a table
    /// does not carry one.
    ///
    /// This is a **granted** halo and is deliberately not what [`Self::reach`]
    /// returns — `Resample::halo` draws the same line for the same reason.
    /// `reach` comes from [`Sampling::max_distance`], which never sees a grid;
    /// deriving one from the other would make the tiling guard compare a number
    /// against itself.
    pub fn halo(&self, grid: &BlockGrid) -> Result<Reach> {
        let volume = grid.volume();
        let block = grid.block();
        let counts = grid.blocks_per_axis();
        let lattice = SampleLattice::of(&self.sampling, volume)?;
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        for axis in 0..3 {
            // Per side: the window below the lowest sample a core voxel reads
            // from, and above the highest. An element with an even extent
            // reaches one further below than above, and granting the wider side
            // in both directions would hand every block a plane it never
            // touches.
            let (window_lo, window_hi) = self.element.sides(axis);
            let mut table = Vec::with_capacity(counts[axis]);
            for index in 0..counts[axis] {
                let core_lo = index * block[axis];
                let core_hi = (core_lo + block[axis]).min(volume[axis]);
                if core_hi <= core_lo {
                    table.push((0, 0));
                    continue;
                }
                let (low, _, _) = lattice.bracket(axis, core_lo);
                let (_, high, _) = lattice.bracket(axis, core_hi - 1);
                let lowest = lattice.centre(axis, low).saturating_sub(window_lo);
                let highest = lattice.centre(axis, high) + window_hi;
                table.push((
                    core_lo.saturating_sub(lowest),
                    highest.saturating_sub(core_hi - 1),
                ));
            }
            let first = table[0];
            axes[axis] = if table.iter().all(|entry| *entry == first) {
                AxisReach::Bounded {
                    lo: first.0,
                    hi: first.1,
                }
            } else {
                AxisReach::PerBlock(table)
            };
        }
        Ok(Reach::per_axis(axes))
    }

    /// Evaluate over a `f64` buffer.
    ///
    /// The lattice is built here, from `at.volume`, and that single line is the
    /// global anchoring: no caller supplies it, and there is no parameter that
    /// could make it a function of the buffer.
    pub fn evaluate_into(
        &self,
        input: ArrayView3<'_, f64>,
        at: &Anchor,
        out: ArrayViewMut3<'_, f64>,
    ) -> Result<()> {
        let lattice = SampleLattice::of(&self.sampling, at.volume)?;
        let ordered = input.mapv(Total);
        let statistic = &self.statistic;
        let full = self.element.len();
        // Lives across every lattice point of this buffer, so a custom reducer
        // widens into the same allocation rather than a fresh one per point.
        // Untouched by the shipped statistics; see `Statistic::reduce_with`.
        let mut scratch: Vec<f64> = Vec::with_capacity(full);
        local_statistic_into(
            ordered.view(),
            at,
            &self.element,
            &lattice,
            |window| statistic.reduce_with(window, full, &mut scratch),
            out,
        )
    }
}

// ------------------------------------------------------------ adapters --

/// Write the local statistic itself.
pub struct LocalStatisticOp {
    name: &'static str,
    statistic: LocalStatistic,
    cost: f64,
}

impl LocalStatisticOp {
    pub fn new(name: &'static str, statistic: LocalStatistic) -> Self {
        let cost = cost_for(&statistic);
        Self {
            name,
            statistic,
            cost,
        }
    }

    pub fn statistic(&self) -> &LocalStatistic {
        &self.statistic
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for LocalStatisticOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Lattice distance plus window radius. Both terms come from the parameters;
    /// neither is settable. The bound; [`Self::reach_spec`] is exact.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.statistic.reach(axis, volume_len)
    }

    /// The same two terms per side, so that a window with an even extent is
    /// planned as the off-centre window it is.
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        self.statistic.reach_spec(volume)
    }

    /// `f64`, and the reason is worth stating rather than assuming.
    ///
    /// `local_statistic_into` is generic over `T: Copy` with an **`f64`
    /// accumulator** — a mean and a deviation must be summed in something wider
    /// than a `u8`, and that is a legitimate place `f64` stays whatever the
    /// input is. What is *not* generic is `LocalStatistic::evaluate_into`, which
    /// is stated in `f64` on both sides. Widening it is kernel work rather than
    /// shell work, so this shell declares what it can actually bridge instead of
    /// promising a conversion it would have to invent.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        self.statistic
            .evaluate_into(input.view::<f64>()?, at, out.view_mut::<f64>()?)
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.statistic.statistic().constant_maps_to(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Compare each voxel against `scale * statistic + offset` at that voxel.
///
/// The affine adjustment is the parameterisation: `scale` alone gives "a
/// fraction of the local level", `offset` alone gives "a fixed margin above it",
/// and a statistic of [`Statistic::Deviation`] with both gives the usual
/// mean-plus-k-deviations form when it is composed after a mean. Which of those
/// a caller wants is the caller's business.
pub struct AdaptiveThresholdOp {
    name: &'static str,
    statistic: LocalStatistic,
    scale: f64,
    offset: f64,
    above: f64,
    below: f64,
    cost: f64,
}

impl AdaptiveThresholdOp {
    pub fn new(name: &'static str, statistic: LocalStatistic, scale: f64, offset: f64) -> Self {
        let cost = cost_for(&statistic) + super::voxelwise::COMBINE_COST;
        Self {
            name,
            statistic,
            scale,
            offset,
            above: 1.0,
            below: 0.0,
            cost,
        }
    }

    /// What to write on each side of the comparison. `1.0` / `0.0` by default,
    /// which is this module's mask convention.
    pub fn with_levels(mut self, above: f64, below: f64) -> Self {
        self.above = above;
        self.below = below;
        self
    }

    pub fn statistic(&self) -> &LocalStatistic {
        &self.statistic
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for AdaptiveThresholdOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The statistic's reach. The comparison itself is voxelwise and adds
    /// nothing.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.statistic.reach(axis, volume_len)
    }

    /// The statistic's reach per side, for the same reason and by the same
    /// derivation: a voxelwise comparison adds nothing on either side.
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        self.statistic.reach_spec(volume)
    }

    /// `f64`, for [`LocalStatisticOp::apply`]'s reason and one more of its own:
    /// the comparison is `T: PartialOrd` against a threshold **of the same
    /// type**, and the threshold is the statistic's `f64` output. A narrower
    /// input would have to be widened to be compared against it, which is a
    /// conversion this shell would be choosing rather than adapting.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let out = out.view_mut::<f64>()?;
        let mut level =
            Array3::<f64>::zeros((input.shape()[0], input.shape()[1], input.shape()[2]));
        self.statistic.evaluate_into(input, at, level.view_mut())?;
        let scale = self.scale;
        let offset = self.offset;
        level.map_inplace(|value| *value = scale * *value + offset);
        threshold_against_into(input, level.view(), self.above, self.below, out)
    }

    /// Exactly true wherever the statistic's own mapping is.
    ///
    /// The composition is `value > scale * s + offset`, evaluated with the same
    /// expression the kernel evaluates, so where `s` is exact the comparison and
    /// its answer are too. Where the statistic withholds its mapping — a mean of
    /// a non-zero constant — this withholds as well, by construction rather than
    /// by a second judgement.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        let statistic = self.statistic.statistic().constant_maps_to(value)?;
        let level = self.scale * statistic + self.offset;
        Some(if value > level {
            self.above
        } else {
            self.below
        })
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured; see `super::COST_MEASUREMENT`.
///
/// The lattice divides the work: a window is evaluated once per sample, not once
/// per voxel, so the per-voxel cost falls with the cube of the spacing. The
/// interpolation is the term that does not, and it is charged flat.
pub(super) fn cost_for(statistic: &LocalStatistic) -> f64 {
    // `None` means the density needs a volume this function is not given; see
    // `Sampling::samples_per_voxel`. Charging full density over-prices, which is
    // the safe direction.
    let samples_per_voxel = statistic.sampling().samples_per_voxel().unwrap_or(1.0);
    // Two terms per sample: the gather, which scales with the window, and
    // whatever the reducer does afterwards that does *not*. Only one statistic
    // has the second — a histogram sweep is a function of the bin count and not
    // of how many values went into it — and it is zero for the other three, so
    // this is bit-for-bit the expression it replaces wherever it was already
    // right.
    let window = statistic.element().len();
    let per_sample = SAMPLE_COST_PER_ELEMENT_VOXEL * window as f64
        + statistic.statistic().cost_per_sample(window);
    per_sample * samples_per_voxel + INTERPOLATION_COST
}

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const SAMPLE_COST_PER_ELEMENT_VOXEL: f64 = 1.96;
/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const INTERPOLATION_COST: f64 = 2.7;
/// Measured; see `super::COST_MEASUREMENT` and the table on
/// [`ISODATA_COST_PER_BIN`]. The isodata rule's **second** walk over the
/// gathered window, per value, relative to a voxelwise map: the collect, the two
/// extreme folds and the binning division. It sits on top of
/// [`SAMPLE_COST_PER_ELEMENT_VOXEL`], which prices the gather itself, and it is
/// about the same size — which is the useful thing it says.
pub(super) const ISODATA_COST_PER_ELEMENT_VOXEL: f64 = 1.87;
/// Measured. One bin of an isodata sweep, per sample, relative to a voxelwise
/// map — the passes the rule makes over the histogram (the cumulative count, the
/// cumulative intensity, the midpoint test), which cost the same whether the
/// window held twenty-seven values or a thousand.
///
/// **It is large**, and that is the figure a planner most needs: at 256 bins
/// this term alone is over a thousand voxelwise maps per *sample*, so an isodata
/// threshold on a dense lattice is an expensive op and on a coarse one is not. A
/// model that priced it as "a windowed statistic" would be out by an order of
/// magnitude in whichever direction the lattice happened to fall.
///
/// **The measurement.** `ops::cost::report([96, 64, 64], 5)`, `--release`, on
/// the machine this crate was developed on. Each isodata row is read against the
/// mean row with the same window and lattice, because the gather is identical
/// between them and the difference is exactly this reducer; the figures below
/// are that difference, multiplied by the 512 voxels per sample.
///
/// ```text
/// window   bins   relative   mean at the same window   difference, per sample
///     729     64       8.95                      5.52                    1756
///     729    256      10.36                      5.52                    2478
///      27    256       5.35                      2.76                    1326
/// ```
///
/// The two constants are the least-squares fit of
/// `per sample = element * window + bin * bins` over those three rows — two
/// predictors and no intercept, because the model *is* the shape of the work:
/// one pass over the window, one over the histogram. It reproduces all three to
/// within 5%, which is the standard `docs/design/BLOCK_OPS.md` §"Simulating
/// strategies" sets: trust "A beats B", distrust "A takes 40 minutes".
pub(super) const ISODATA_COST_PER_BIN: f64 = 4.71;

#[cfg(test)]
mod tests {
    use super::super::element::ElementShape;
    use super::*;

    /// The layout is centred, and it is a function of the volume alone.
    #[test]
    fn the_lattice_is_centred_and_derived_from_the_volume() {
        let lattice = SampleLattice::centred([10, 10, 10], [4, 4, 4]).unwrap();
        assert_eq!(lattice.count(0), 2);
        assert_eq!(lattice.centre(0, 0), 3);
        assert_eq!(lattice.centre(0, 1), 7);
        // equal margins to within a voxel: 3 before, 2 after
        assert_eq!(10 - 1 - lattice.centre(0, 1), 2);
    }

    #[test]
    fn an_axis_shorter_than_a_spacing_still_has_one_sample() {
        let lattice = SampleLattice::centred([3, 3, 3], [8, 8, 8]).unwrap();
        assert_eq!(lattice.count(0), 1);
        assert_eq!(lattice.centre(0, 0), 1);
        assert_eq!(lattice.bracket(0, 2), (0, 0, 0.0));
    }

    /// **The property this whole file exists for**, with the defect it avoids
    /// spelled out first so the assertion is not vacuous.
    ///
    /// One fixed voxel, two decompositions that both contain it with room for
    /// the derived reach. Laid out over the block, the sample it interpolates
    /// from is a *different voxel* in each — that is the defect, and it is shown
    /// here rather than described. Laid out over the volume, the value is the
    /// whole-volume value in both.
    #[test]
    fn an_unanchored_lattice_moves_with_the_block_and_an_anchored_one_does_not() {
        let volume = [37usize, 12, 11];
        let spacing = [5usize, 3, 4];
        let voxel = [20usize, 5, 6];
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let local = LocalStatistic::new(element, spacing, Statistic::Mean).unwrap();
        let input = ramp(volume);
        let extents = [(14usize, 14usize), (8, 20)];
        for &(offset, len) in &extents {
            assert!(
                voxel[0] - offset >= local.reach(0, volume[0])
                    && offset + len - voxel[0] > local.reach(0, volume[0]),
                "the voxel must be trustworthy in both extents or the test proves nothing"
            );
        }

        // The defect: a lattice laid out over the array handed in.
        let unanchored_sample = |offset: usize, len: usize| {
            let block = SampleLattice::centred([len, volume[1], volume[2]], spacing).unwrap();
            offset + block.centre(0, block.index_below(0, voxel[0] - offset))
        };
        assert_ne!(
            unanchored_sample(extents[0].0, extents[0].1),
            unanchored_sample(extents[1].0, extents[1].1),
            "if these agreed there would be no defect and the rest of this test \
             would be measuring nothing"
        );

        // The kernel: a lattice laid out over the volume, reached through the
        // anchor, giving the whole-volume answer from either extent.
        let mut whole = Array3::zeros((volume[0], volume[1], volume[2]));
        local
            .evaluate_into(input.view(), &Anchor::whole(volume), whole.view_mut())
            .unwrap();
        for &(offset, len) in &extents {
            let piece = input
                .slice(ndarray::s![offset..offset + len, .., ..])
                .to_owned();
            let mut got = Array3::zeros(piece.dim());
            local
                .evaluate_into(
                    piece.view(),
                    &Anchor::new([offset, 0, 0], volume),
                    got.view_mut(),
                )
                .unwrap();
            assert_eq!(
                got[[voxel[0] - offset, voxel[1], voxel[2]]],
                whole[voxel],
                "extent {offset}..{}",
                offset + len
            );
        }
    }

    #[test]
    fn the_reach_carries_both_terms() {
        let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 5]).unwrap();
        let statistic = LocalStatistic::new(element, [8, 8, 8], Statistic::Mean).unwrap();
        // volume 64, spacing 8 -> count 8, first 4, last 60; distances 4, 3, 7
        assert_eq!(axis_max_distance(64, 8), 7);
        assert_eq!(statistic.reach(0, 64), 7 + 2);
        // a spacing of one is the no-lattice case: every voxel is its own
        // sample, so the reach is the window and nothing more
        let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 5]).unwrap();
        let dense = LocalStatistic::new(element, [1, 1, 1], Statistic::Mean).unwrap();
        assert_eq!(axis_max_distance(64, 1), 0);
        assert_eq!(dense.reach(0, 64), 2);
    }

    /// The lattice term is symmetric and the window term is not, so an even
    /// window makes the whole reach asymmetric by exactly the window's own
    /// asymmetry — and the halo table has to carry that through per side.
    ///
    /// The halo is what a block is actually handed; granting the wider side in
    /// both directions would give every block a plane no voxel of its core
    /// reads, on every block of the lattice, which is the cost this crate keeps
    /// per-side reaches in order not to pay.
    #[test]
    fn an_even_window_makes_both_the_reach_and_the_halo_asymmetric() {
        let element = StructuringElement::from_size(ElementShape::Box, [6, 5, 5]).unwrap();
        assert_eq!(element.sides(0), (3, 2));
        let statistic = LocalStatistic::new(element, [8, 8, 8], Statistic::Mean).unwrap();
        assert_eq!(statistic.reach_sides(0, 64), (7 + 3, 7 + 2));
        assert_eq!(statistic.reach(0, 64), 10, "the bound is the wider side");
        assert_eq!(statistic.reach_spec([64, 64, 64]).at(0, 0, 64), (10, 9));

        // And the granted halo, which is derived from the brackets rather than
        // from the reach, is asymmetric on the same axis and by the same voxel.
        //
        // Taken on the **no-lattice** sampling and on the **interior** blocks,
        // for two separate reasons. A spacing wide enough to bracket dominates
        // the window on both sides — with samples eight apart the nearest one
        // outside the core is far enough away that three and two below it round
        // to the same halo — so a lattice term would hide the property rather
        // than exercise it. And the first and last block have their halo
        // clamped by the volume's own edge, where there is nothing beyond to
        // fetch, so the asymmetry is a fact about seams.
        let element = StructuringElement::from_size(ElementShape::Box, [6, 5, 5]).unwrap();
        let dense = LocalStatistic::new(element, [1, 1, 1], Statistic::Mean).unwrap();
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        let halo = dense.halo(&grid).unwrap();
        assert_eq!(
            halo.at(0, 0, 64),
            (0, 2),
            "clamped at the volume's own edge"
        );
        for index in 1..4 {
            assert_eq!(
                halo.at(0, index, 64),
                (3, 2),
                "block {index} must be granted the window's two sides and not its \
                 bounding box"
            );
        }
    }

    /// The interpolation term is exactly what it claims: sweeping every voxel of
    /// an axis, no sample further than `axis_max_distance` is ever read.
    ///
    /// This is the assertion that keeps `bracket` and `reach` from drifting.
    /// Tighten one without the other and this fails, which is the point — the
    /// alternative is a reach that is correct only because it is generous, and a
    /// generosity nobody records is a generosity somebody later removes.
    #[test]
    fn no_voxel_reads_a_sample_further_than_the_declared_distance() {
        for volume_len in [1usize, 2, 7, 16, 17, 64, 65] {
            for spacing in [1usize, 2, 3, 5, 8, 13, 100] {
                let lattice = SampleLattice::centred([volume_len, 4, 4], [spacing, 1, 1]).unwrap();
                let bound = axis_max_distance(volume_len, spacing);
                for coordinate in 0..volume_len {
                    let (low, high, t) = lattice.bracket(0, coordinate);
                    for index in [low, high] {
                        let centre = lattice.centre(0, index);
                        let distance = centre.abs_diff(coordinate);
                        assert!(
                            distance <= bound,
                            "volume {volume_len}, spacing {spacing}, voxel \
                             {coordinate}: read sample {index} at {centre}, \
                             {distance} away, but declared {bound}"
                        );
                    }
                    assert!((0.0..1.0).contains(&t));
                    if low == high {
                        assert_eq!(t, 0.0);
                    }
                }
            }
        }
    }

    /// The interpolation must be exact on a constant, or the constant algebra of
    /// a rank statistic would be a near miss rather than a declaration.
    #[test]
    fn interpolating_a_constant_grid_is_exact() {
        for value in [0.1_f64, -7.25, 1e-17, 3.0] {
            assert_eq!(lerp(value, value, 0.0), value);
            assert_eq!(lerp(value, value, 0.37), value);
            assert_eq!(lerp(value, value, 1.0), value);
        }
    }

    fn ramp(shape: [usize; 3]) -> Array3<f64> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64
        })
    }

    /// The whole-volume answer restricted to a window equals the window's own
    /// answer, for every window and every statistic — which is
    /// decomposition-invariance asserted at the kernel, before any executor is
    /// involved.
    #[test]
    fn a_sub_buffer_reproduces_the_whole_volume_answer_inside_its_trustworthy_extent() {
        let volume = [29usize, 13, 11];
        let input = ramp(volume);
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        for spacing in [[1usize, 1, 1], [3, 2, 4], [7, 13, 5]] {
            for statistic in [
                Statistic::Mean,
                Statistic::Deviation,
                Statistic::Rank(Rank::median(&element)),
            ] {
                let local =
                    LocalStatistic::new(element.clone(), spacing, statistic.clone()).unwrap();
                let mut whole = Array3::zeros((volume[0], volume[1], volume[2]));
                local
                    .evaluate_into(input.view(), &Anchor::whole(volume), whole.view_mut())
                    .unwrap();

                // a sub-buffer with a halo of the derived reach on axis 0
                let reach = local.reach(0, volume[0]);
                let core = 10usize..20;
                let start = core.start.saturating_sub(reach);
                let end = (core.end + reach).min(volume[0]);
                let piece = input.slice(ndarray::s![start..end, .., ..]).to_owned();
                let mut got = Array3::zeros(piece.dim());
                local
                    .evaluate_into(
                        piece.view(),
                        &Anchor::new([start, 0, 0], volume),
                        got.view_mut(),
                    )
                    .unwrap();
                for i in core.clone() {
                    for j in 0..volume[1] {
                        for k in 0..volume[2] {
                            assert_eq!(
                                got[[i - start, j, k]],
                                whole[[i, j, k]],
                                "spacing {spacing:?} {statistic:?} at {i},{j},{k}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The mean's mapping is withheld for a reason, and the reason is
    /// demonstrable rather than theoretical.
    #[test]
    fn the_mean_of_a_constant_is_not_the_constant_which_is_why_it_is_not_declared() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let local = LocalStatistic::new(element, [1, 1, 1], Statistic::Mean).unwrap();
        let input = Array3::from_elem((5, 1, 1), 0.1_f64);
        let mut out = Array3::zeros(input.dim());
        local
            .evaluate_into(input.view(), &Anchor::whole([5, 1, 1]), out.view_mut())
            .unwrap();
        assert_ne!(
            out[[2, 0, 0]],
            0.1_f64,
            "if this ever becomes exact the declaration can be widened, but it \
             must be widened deliberately"
        );
        assert_eq!(Statistic::Mean.constant_maps_to(0.1), None);
        assert_eq!(Statistic::Mean.constant_maps_to(0.0), Some(0.0));
        // a rank is exact at every constant, and says so
        assert_eq!(
            Statistic::Rank(Rank::Nth(3)).constant_maps_to(0.1),
            Some(0.1)
        );
    }

    #[test]
    fn a_rank_statistic_of_a_constant_computes_what_it_declares() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let local = LocalStatistic::new(
            element.clone(),
            [3, 3, 3],
            Statistic::Rank(Rank::median(&element)),
        )
        .unwrap();
        let input = Array3::from_elem((9, 9, 9), 0.1_f64);
        let mut out = Array3::zeros(input.dim());
        local
            .evaluate_into(input.view(), &Anchor::whole([9, 9, 9]), out.view_mut())
            .unwrap();
        assert!(
            out.iter().all(|&value| value == 0.1_f64),
            "a declared constant must be the computed one, bit for bit"
        );
    }

    #[test]
    fn an_adaptive_threshold_declares_only_what_its_statistic_does() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let mean = AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(element.clone(), [4, 4, 4], Statistic::Mean).unwrap(),
            1.0,
            0.5,
        );
        assert_eq!(mean.constant_maps_to(0.0), Some(0.0));
        assert_eq!(mean.constant_maps_to(3.0), None);

        let ranked = AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(
                element.clone(),
                [4, 4, 4],
                Statistic::Rank(Rank::median(&element)),
            )
            .unwrap(),
            0.5,
            0.0,
        );
        // 3.0 > 0.5 * 3.0, so a uniform 3.0 volume is entirely above its own
        // local median scaled by a half
        assert_eq!(ranked.constant_maps_to(3.0), Some(1.0));
        assert_eq!(ranked.constant_maps_to(0.0), Some(0.0));
    }

    // ------------------------------------------------------- isodata --

    /// Everything that cannot produce a usable threshold is refused where it is
    /// **stated**, rather than at the first window that reaches it.
    #[test]
    fn an_isodata_that_could_not_produce_a_threshold_is_refused_at_construction() {
        assert!(Isodata::new(0, 1.0).is_err());
        for fallback in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = Isodata::new(256, fallback).unwrap_err().to_string();
            assert!(err.contains("must be finite"), "got: {err}");
        }
        assert!(Isodata::new(1, 0.0).is_ok());
        assert!(Isodata::new(256, -3.5).is_ok());
    }

    /// The fallback is a **parameter**, and it is what an empty window and a
    /// window with nothing finite in it get. Both of those are the same case —
    /// there is no histogram — and neither is the crate's business to name.
    #[test]
    fn a_window_with_nothing_to_histogram_takes_the_stated_fallback() {
        for fallback in [1.0f64, 0.0, -3.5, 1e12] {
            let isodata = Isodata::new(256, fallback).unwrap();
            assert_eq!(isodata.of(std::iter::empty()), fallback);
            assert_eq!(
                isodata.of([f64::NAN, f64::INFINITY, f64::NEG_INFINITY].into_iter()),
                fallback
            );
            // and a window that *does* have values does not take it
            assert_eq!(isodata.of([2.0, 2.0, 2.0].into_iter()), 2.0);
        }
    }

    /// `Eq` and `Hash` agree, which is what makes holding an `f64` in a
    /// `Statistic` legitimate. `-0.0` is the only value where the two could
    /// disagree and the constructor normalises it.
    #[test]
    fn two_isodata_parameters_are_equal_exactly_when_they_hash_alike() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash = |value: &Isodata| {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        };
        let plus = Isodata::new(256, 0.0).unwrap();
        let minus = Isodata::new(256, -0.0).unwrap();
        assert_eq!(plus, minus);
        assert_eq!(hash(&plus), hash(&minus));
        assert_eq!(plus.fallback().to_bits(), 0.0f64.to_bits());

        let other = Isodata::new(64, 0.0).unwrap();
        assert_ne!(plus, other);
        assert_eq!(
            Statistic::Isodata(plus),
            Statistic::Isodata(Isodata::new(256, 0.0).unwrap())
        );
        assert_ne!(Statistic::Isodata(plus), Statistic::Isodata(other));
    }

    /// A single bin has one centre and the rule returns it — the degenerate
    /// case, stated so that it is a decision rather than an accident of the
    /// loop bound below it.
    #[test]
    fn one_bin_answers_its_only_centre() {
        let isodata = Isodata::new(1, 99.0).unwrap();
        // centre of [0, 4] is 2
        assert_eq!(isodata.of([0.0, 1.0, 4.0].into_iter()), 2.0);
    }

    /// The cost model sees the shape of this reducer's work: the bin count, not
    /// the window.
    #[test]
    fn an_isodata_statistic_is_priced_by_its_bins_and_a_mean_is_not() {
        let window = StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap();
        let of = |statistic| {
            cost_for(&LocalStatistic::new(window.clone(), [1, 1, 1], statistic).unwrap())
        };
        let mean = of(Statistic::Mean);
        let narrow = of(Statistic::Isodata(Isodata::new(64, 1.0).unwrap()));
        let wide = of(Statistic::Isodata(Isodata::new(256, 1.0).unwrap()));
        assert!(narrow > mean, "a histogram sweep is not free");
        assert!(wide > narrow, "four times the bins is more work");
        // and the three that were already here are unchanged, bit for bit
        assert_eq!(
            mean,
            SAMPLE_COST_PER_ELEMENT_VOXEL * window.len() as f64 * 1.0 + INTERPOLATION_COST
        );
        assert_eq!(of(Statistic::Deviation), mean);
        assert_eq!(of(Statistic::Rank(Rank::lowest())), mean);
    }

    #[test]
    fn the_kernel_refuses_a_buffer_that_does_not_fit_its_anchor() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let local = LocalStatistic::new(element, [2, 2, 2], Statistic::Mean).unwrap();
        let input = Array3::from_elem((8, 8, 8), 1.0);
        let mut out = Array3::zeros(input.dim());
        let err = local
            .evaluate_into(
                input.view(),
                &Anchor::new([4, 0, 0], [8, 8, 8]),
                out.view_mut(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not fit a volume"), "got: {err}");
    }
}

#[cfg(test)]
mod lattice_tests {
    use super::*;
    use crate::ops::element::ElementShape;

    /// The lattice is a set of positions in **volume** coordinates, so cutting
    /// the volume differently cannot move a sample. Two decompositions see the
    /// same points; what differs is only which of them each block holds.
    #[test]
    fn the_positions_are_the_same_whatever_the_blocks_are() {
        let sampling = Sampling::every([7, 7, 7]);
        let lattice = SampleLattice::of(&sampling, [64, 64, 64]).unwrap();
        // count = 64/7 = 9, span = 56, first = (64 - 56)/2 = 4
        assert_eq!(lattice.positions(0), &[4, 11, 18, 25, 32, 39, 46, 53, 60]);

        // the same lattice, however the volume is later cut
        for block in [[64, 64, 64], [8, 64, 64], [13, 20, 5], [7, 7, 7]] {
            let grid = BlockGrid::new([64, 64, 64], block).unwrap();
            assert_eq!(
                SampleLattice::of(&sampling, grid.volume()).unwrap(),
                lattice,
                "block {block:?} changed the lattice"
            );
        }
    }

    /// ...and the samples therefore land at a **different local offset in every
    /// block**, which is the consequence worth seeing written down.
    ///
    /// A spacing of 7 cut into blocks of 8 drifts by one voxel per block — local
    /// 4, then 3, then 2, then 1 — and then a block holds **two** samples where
    /// each of its neighbours holds one. Nothing about a block's own geometry
    /// predicts either. That is what "the lattice is independent of the
    /// decomposition" costs and buys: the answer is the same under every cut,
    /// and no block can assume where its samples are.
    #[test]
    fn a_global_lattice_lands_differently_inside_every_block() {
        let lattice = SampleLattice::centred([64, 64, 64], [7, 7, 7]).unwrap();
        let grid = BlockGrid::new([64, 64, 64], [8, 64, 64]).unwrap();

        let mut per_block = Vec::new();
        for index in 0..grid.blocks_per_axis()[0] {
            let core_lo = index * 8;
            let core_hi = (core_lo + 8).min(64);
            let local: Vec<usize> = lattice
                .positions(0)
                .iter()
                .filter(|&&position| position >= core_lo && position < core_hi)
                .map(|&position| position - core_lo)
                .collect();
            per_block.push(local);
        }
        assert_eq!(
            per_block,
            vec![
                vec![4],
                vec![3],
                vec![2],
                vec![1],
                vec![0, 7],
                vec![6],
                vec![5],
                vec![4],
            ],
            "the samples do not line up with the blocks, and must not"
        );
        // and the counts genuinely differ, which is what makes the per-block
        // halo below a table rather than one number
        assert!(per_block.iter().any(|block| block.len() == 2));
        assert!(per_block.iter().any(|block| block.len() == 1));
    }

    /// The granted halo is a table, is tighter than the symmetric requirement
    /// for at least one block, and covers what every block actually reads.
    #[test]
    fn the_granted_halo_is_per_block_and_never_short() {
        let element = StructuringElement::from_radius(ElementShape::Box, [2, 1, 1]);
        let statistic = LocalStatistic::new(element.clone(), [7, 7, 7], Statistic::Mean).unwrap();
        let volume = [40usize, 40, 40];
        let grid = BlockGrid::new(volume, [8, 40, 40]).unwrap();
        let halo = statistic.halo(&grid).unwrap();

        let AxisReach::PerBlock(table) = halo.axis(0) else {
            panic!(
                "an unevenly landing lattice must produce a table, got {:?}",
                halo.axis(0)
            );
        };
        let first = table[0];
        assert!(
            table.iter().any(|entry| *entry != first),
            "a table whose entries all agree should have collapsed to the uniform form"
        );

        // never wider than the worst case the requirement states...
        let worst = statistic.reach(0, volume[0]);
        for (index, &(lo, hi)) in table.iter().enumerate() {
            assert!(
                lo <= worst && hi <= worst,
                "block {index} exceeds the requirement"
            );
        }
        // ...and at least one block genuinely needs less, which is the saving
        assert!(table.iter().any(|&(lo, hi)| lo < worst || hi < worst));

        // and never short: every sample a block's own voxels bracket, plus that
        // sample's window, is inside the granted read.
        let lattice = SampleLattice::centred(volume, [7, 7, 7]).unwrap();
        for (index, &(lo, hi)) in table.iter().enumerate() {
            let core_lo = index * 8;
            let core_hi = (core_lo + 8).min(volume[0]);
            let read_lo = core_lo.saturating_sub(lo);
            let read_hi = (core_hi - 1 + hi).min(volume[0] - 1);
            for voxel in core_lo..core_hi {
                let (a, b, _) = lattice.bracket(0, voxel);
                for sample in [a, b] {
                    let centre = lattice.centre(0, sample);
                    assert!(
                        centre.saturating_sub(element.reach(0)) >= read_lo
                            || centre < element.reach(0),
                        "block {index} would read below its granted halo"
                    );
                    assert!(
                        centre + element.reach(0) <= read_hi
                            || centre + element.reach(0) >= volume[0],
                        "block {index} would read above its granted halo"
                    );
                }
            }
        }
    }

    /// The two shipped conventions are different lattices, and the endpoint one
    /// pins both ends. A caller matching a tool built on `ndimage.zoom` needs the
    /// second; a caller who wants equal margins needs the first.
    #[test]
    fn the_two_conventions_disagree_and_each_is_what_it_says() {
        let centred = SampleLattice::of(&Sampling::every([4, 4, 4]), [16, 16, 16]).unwrap();
        assert_eq!(centred.positions(0), &[2, 6, 10, 14]);

        let ends =
            SampleLattice::of(&Sampling::Endpoints { count: [4, 4, 4] }, [16, 16, 16]).unwrap();
        assert_eq!(ends.positions(0), &[0, 5, 10, 15]);
        assert_ne!(centred, ends);

        // the endpoint convention reaches nothing at the ends, because a sample
        // sits on the first and last voxel
        assert_eq!(ends.max_distance(0), 4);
        assert_eq!(centred.max_distance(0), 3);
    }

    /// An irregular lattice is expressible, and its reach is the **widest gap**
    /// rather than an average — which is the property a spacing could not state.
    #[test]
    fn an_irregular_lattice_is_priced_by_its_widest_gap() {
        let positions = [vec![0, 1, 2, 20], vec![0], vec![0]];
        let lattice = SampleLattice::at([32, 4, 4], positions).unwrap();
        assert_eq!(lattice.count(0), 4);
        // the 0..1 and 1..2 gaps contribute nothing; the 2..20 gap contributes
        // 17, and the tail past the last sample contributes 31 - 20 = 11
        assert_eq!(lattice.max_distance(0), 17);
    }

    /// Every way of stating a lattice that cannot be interpolated from is
    /// refused where it is stated.
    #[test]
    fn a_lattice_that_could_not_be_interpolated_from_is_refused() {
        assert!(SampleLattice::at([8, 8, 8], [vec![], vec![0], vec![0]]).is_err());
        assert!(SampleLattice::at([8, 8, 8], [vec![0, 0], vec![0], vec![0]]).is_err());
        assert!(SampleLattice::at([8, 8, 8], [vec![4, 2], vec![0], vec![0]]).is_err());
        assert!(SampleLattice::at([8, 8, 8], [vec![0, 8], vec![0], vec![0]]).is_err());
        assert!(SampleLattice::at([0, 8, 8], [vec![0], vec![0], vec![0]]).is_err());
        assert!(SampleLattice::centred([8, 8, 8], [0, 1, 1]).is_err());
        assert!(SampleLattice::at([8, 8, 8], [vec![0, 4], vec![0], vec![0]]).is_ok());
    }
}
