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
    reduce: F,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy,
    F: Fn(&mut [T]) -> f64,
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

/// What to reduce a window to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistic {
    /// The arithmetic mean.
    Mean,
    /// The population standard deviation, computed in two passes so that it
    /// cannot go negative under the square root.
    Deviation,
    /// An order statistic, through the same [`Rank`] the rank filter uses.
    Rank(Rank),
}

impl Statistic {
    /// Reduce a gathered window. The window is in the element's canonical order,
    /// so a summation over it is the same summation everywhere.
    pub fn reduce<T>(&self, window: &mut [T], full: usize) -> f64
    where
        T: Ord + Copy + Into<f64>,
    {
        if window.is_empty() {
            return 0.0;
        }
        match *self {
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
    pub fn constant_maps_to(&self, value: f64) -> Option<f64> {
        match self {
            Statistic::Rank(_) => Some(value),
            Statistic::Mean | Statistic::Deviation => (value == 0.0).then_some(0.0),
        }
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

    pub fn statistic(&self) -> Statistic {
        self.statistic
    }

    /// The two terms: how far the interpolation reaches for a sample, plus how
    /// far that sample's window reaches beyond it.
    ///
    /// **The worst case over the volume**, which is what a symmetric per-axis
    /// integer can say. A globally anchored lattice does not line up with block
    /// boundaries — that is the whole point of it — so the samples a *particular*
    /// block needs sit at a different distance outside its core for every block,
    /// and most blocks reach less than this. See `LocalStatisticOp::reach_spec`,
    /// which says the tighter thing.
    pub fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.sampling.max_distance(axis, volume_len) + self.element.reach(axis)
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
            let window = self.element.reach(axis);
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
                let lowest = lattice.centre(axis, low).saturating_sub(window);
                let highest = lattice.centre(axis, high) + window;
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
        let statistic = self.statistic;
        let full = self.element.len();
        local_statistic_into(
            ordered.view(),
            at,
            &self.element,
            &lattice,
            |window| statistic.reduce(window, full),
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
    /// neither is settable.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.statistic.reach(axis, volume_len)
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
    SAMPLE_COST_PER_ELEMENT_VOXEL * statistic.element().len() as f64 * samples_per_voxel
        + INTERPOLATION_COST
}

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const SAMPLE_COST_PER_ELEMENT_VOXEL: f64 = 1.96;
/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const INTERPOLATION_COST: f64 = 2.7;

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
                let local = LocalStatistic::new(element.clone(), spacing, statistic).unwrap();
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
