// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The conventional arrangement derives a block's valid region from the block
// plan that produced it, which is why a tiling check over those regions can
// only confirm that the plan is self-consistent — it compares a number against
// itself. This file inverts that: what is trustworthy is *derived from reach*,
// and the tiling check (`tiling`) becomes a real guard on the halo.
//
// The inversion, per axis
// -----------------------
//     read extent        = core grown by halo, clamped to the volume
//     trustworthy extent = read extent shrunk by reach, except where the read
//                          was clamped by the volume itself
//     valid region       = core ∩ trustworthy
//
// | | outcome |
// |---|---|
// | `halo >= reach` | valid == core, regions tile, correct |
// | `halo <  reach` | valid **shrinks below core** -> gap -> tiling check fires |
// | `halo >> reach` | valid == core, tiles, redundant compute — performance only |
//
// There is deliberately **no** `assert!(halo >= reach)` anywhere in this
// module. The design asks for the guard to be the tiling check that already
// exists and already runs, rather than a new assertion that a future call site
// can forget.
//
// The clamp exception is load-bearing. A voxel at a real volume boundary is
// trustworthy even though its reach ran off the end, because there is nothing
// beyond the end to have read — the op saw everything that exists. Without the
// exception every volume-edge block would report a hole and the guard would cry
// wolf on every correct run, which is how guards get deleted.
//
// The same sentence is what grants it in the *source* frame, on one axis
// -----------------------------------------------------------------------
// A reach stated against the image below (`Frame::Source`) is otherwise denied
// the exception, because a **cropping** phase's edge is an interior position of
// the array it reads: a neighbour exists there, and a halo could have reached
// it. Correct, and it is why the frame exists.
//
// **That reasoning does not hold for an axis the op consumes entirely.** There
// is no beyond, and so no neighbour a halo could have reached into.
// `AxisReach::All` is not a distance that ran off the end; it is the statement
// that the end is where the op stops. So the exception is granted per axis, on
// the two conditions that make it mean something: the reach on that axis is
// `All`, and this block's read spans the whole of the axis.
//
// The second condition is what keeps it honest, and it pays for itself. Where
// the axis *is* cut with a finite halo, no block spans it, every block stays
// degenerate and the tiling check fires as before — so the declaration becomes
// the **whole-axis mandate it always implied**: leave the axis whole, or grant a
// whole-axis halo, and there is no third option. That is a free partial answer
// to **G9** in `docs/ops-survey/README.md`'s register
// (`BlockConstraint::FullExtent(axis)`), obtained without a constraint type —
// declared by the op rather than configured by the planner, and enforced by the
// guard that already ran.
//
// What this file cannot check is that the block *fetched* the axis it declared.
// That is a fact about `BlockGeometry::source`, which is attached after this
// runs, and `Decomposition::check` is where the claim is met. Without that half
// the declaration would be decoration; with it, saying the truth is worth more
// than the `Space::source_index()` escape rather than merely as much — the
// escape records *that* a dependency exists, this records what would satisfy it.
//
// One thing neither half changes: `All` stated in the phase's own frame is
// **vacuous** on a collapsed axis. `AxisReach::is_whole` requires `extent > 1`,
// so against an axis of extent 1 the words are accepted without being a
// statement of anything.

use crate::error::{Error, Result};
use crate::reach::{AxisReach, Reach};
use crate::region::Region;

/// A grid of block cores over a 3-D volume.
///
/// Per phase, not per chain: a phase boundary is already a materialisation, so
/// the grid may change there for free (`docs/design/BLOCK_OPS.md` §"Block size
/// may differ per phase"). Nothing in this type ties it to the whole workflow.
///
/// This is *not* `block_processing::split_into_blocks`. That returns valid
/// regions computed from the plan; this returns cores only, and validity is
/// derived from reach in `BlockGeometry::derive`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockGrid {
    volume: [usize; 3],
    block: [usize; 3],
    split_axes: Vec<usize>,
}

impl BlockGrid {
    pub fn new(volume: [usize; 3], block: [usize; 3]) -> Result<Self> {
        for axis in 0..3 {
            if volume[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "block grid: volume axis {axis} is empty"
                )));
            }
            if block[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "block grid: block axis {axis} is zero"
                )));
            }
        }
        let block = [
            block[0].min(volume[0]),
            block[1].min(volume[1]),
            block[2].min(volume[2]),
        ];
        let split_axes = (0..3).filter(|&axis| block[axis] < volume[axis]).collect();
        Ok(Self {
            volume,
            block,
            split_axes,
        })
    }

    /// One block spanning the whole volume: no seams, no halo reasoning. This
    /// is the geometry the `Trivial` oracle strategy uses.
    pub fn whole(volume: [usize; 3]) -> Result<Self> {
        Self::new(volume, volume)
    }

    /// Split only `axes`; every other axis spans the whole volume. The z-only
    /// default is `axes = [2]`; `docs/design/XY_BLOCK_SPLITTING.md` is the same
    /// call with more axes.
    pub fn along(volume: [usize; 3], axes: &[usize], edge: usize) -> Result<Self> {
        let mut block = volume;
        for &axis in axes {
            if axis >= 3 {
                return Err(Error::InvalidArgument(format!(
                    "block grid: axis {axis} out of bounds"
                )));
            }
            block[axis] = edge;
        }
        Self::new(volume, block)
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn block(&self) -> [usize; 3] {
        self.block
    }

    /// Axes actually cut. A halo on any other axis buys nothing, because the
    /// block already spans it.
    pub fn split_axes(&self) -> &[usize] {
        &self.split_axes
    }

    pub fn blocks_per_axis(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for axis in 0..3 {
            counts[axis] = self.volume[axis].div_ceil(self.block[axis]);
        }
        counts
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks_per_axis().iter().product()
    }

    /// Voxels in an interior block's core — the unit the infinite-grid cost
    /// model is stated in.
    ///
    /// This is the **widest** block, and a grid whose edge does not divide its
    /// volume has none of that size at its boundary. Use it where over-stating
    /// is the safe direction — a residency budget, a buffer size. For anything
    /// that *compares two grids*, use [`BlockGrid::mean_core_voxels`] and read
    /// its note on why.
    pub fn core_voxels(&self) -> f64 {
        self.block.iter().map(|&edge| edge as f64).product()
    }

    /// Voxels in the **average** block's core: the volume, over the blocks that
    /// cover it.
    ///
    /// Exact, not an estimate. [`BlockGrid::cores`] clips every core to the
    /// volume and the cores tile it without overlap, so the cores sum to the
    /// volume and this is that sum divided by [`BlockGrid::n_blocks`]. It
    /// equals [`BlockGrid::core_voxels`] exactly when the block divides the
    /// volume on every split axis, and is smaller otherwise.
    ///
    /// # Why a price that compares grids must use this one
    ///
    /// Charging every block at the widest is a deliberate over-charge, and as a
    /// statement about *one* grid it is harmless — it is the same direction of
    /// error a generous halo has. What makes it harmful is that the planner
    /// uses the price to **choose between** grids, and the size of the
    /// over-charge is a property of the candidate rather than of the machine:
    /// it is the grid's padding, `n_blocks * core_voxels / volume`, and that
    /// ratio moves with the edge because the boundary remainder does.
    ///
    /// Measured on the pipeline's `404 x 1304 x 3369` tile, over the four edges
    /// `skeletonize`'s ladder offers there:
    ///
    /// ```text
    /// edge | blocks | n x core_voxels / volume
    ///  512 |     21 | 1.253
    ///  256 |    168 | 1.588
    ///  128 |   1188 | 1.404
    ///   64 |   7791 | 1.151
    /// ```
    ///
    /// A price built on the widest block therefore charges the 256 grid **38%
    /// more per voxel of real work than the 64 grid**, for no reason connected
    /// to the work. That is an order of magnitude larger than the margins the
    /// search decides on — it inverted 256 against 128, which the model
    /// preferred by 4.1% and which measures 38% slower — and it is not a
    /// coefficient anybody can measure away, because it is not a cost. It is
    /// the model quoting a different unit for each candidate.
    ///
    /// Nothing here is tuned: the mean is a fact about the grid and there is no
    /// constant to pick.
    ///
    /// **This is one instance of a pattern, not a one-off.** The same failure —
    /// a term wrong by an amount that varies with the candidate, which is a bias
    /// and not a conservative approximation — was found a second time in the
    /// same expression, in the read charge. The rule and both measurements are
    /// stated once on [`PhaseCost`](crate::decomposition::PhaseCost).
    pub fn mean_core_voxels(&self) -> f64 {
        self.volume.iter().map(|&edge| edge as f64).product::<f64>() / self.n_blocks() as f64
    }

    /// The mean over this grid's blocks of the voxels one block **reads** at
    /// `halo`, exactly, with the volume boundary clamped.
    ///
    /// The counterpart of [`Self::mean_core_voxels`] for the read extent, and it
    /// exists for the same reason and was arrived at by the same route. The core
    /// was charged at the widest block until that was measured to be a bias; the
    /// read was charged on the **infinite grid** — every block assumed interior,
    /// so every block paying a full halo on both sides — and that is the same
    /// mistake with a different justification attached to it.
    ///
    /// **Why the infinite-grid charge was defensible and then was not.** On its
    /// own it over-states by the boundary fraction, 1.3% to 9.4% over the grids
    /// measured in `tests/phase_pricing.rs`, always in the direction the cost
    /// model is declared safe in. What removed the defence is that the
    /// over-statement is a function of the candidate — a wider halo has a larger
    /// boundary fraction, and a phase reading three arrays pays the fraction
    /// three times — so once a phase's traffic entered the price the model began
    /// ranking partitions on the size of its own error. A sibling crate's
    /// partition suite caught it: at a 32-cube block the search preferred a
    /// four-phase plan over a three-phase one that reads 3.2% **fewer** voxels
    /// and writes one intermediate fewer, which is worse by both of the
    /// quantities the model is made of. See
    /// [`PhaseCost`](crate::decomposition::PhaseCost) for the rule.
    ///
    /// **This is exact, and that is checkable rather than claimed.** Blocks are
    /// a Cartesian product of per-axis positions and a read extent is a product
    /// of per-axis lengths, so the mean of the product is the product of the
    /// means — no independence assumption, just the factorisation. Multiplied by
    /// [`Self::n_blocks`] it therefore equals
    /// `Decomposition::exact_read_voxels` for one image, to the voxel, and
    /// `tests/phase_pricing.rs` asserts that equality rather than a band around
    /// it.
    ///
    /// Nothing here is tuned. There was a constant to pick while the charge was
    /// approximate; there is none now.
    pub fn mean_read_voxels(&self, halo: &Reach) -> f64 {
        (0..3)
            .map(|axis| {
                let (lo, hi) = halo.axis(axis).bound(self.volume[axis]);
                self.mean_read_extent(axis, lo, hi)
            })
            .product()
    }

    /// The mean read length along one axis, clamped at both ends of the volume.
    ///
    /// Split out because [`crate::decomposition::price_phase`] charges some axes
    /// on the infinite grid deliberately — a full-reach axis, and a single-block
    /// phase whose halo spans the axis — and needs the two charges side by side
    /// to do it.
    pub fn mean_read_extent(&self, axis: usize, lo: usize, hi: usize) -> f64 {
        let volume = self.volume[axis];
        let block = self.block[axis];
        let blocks = self.blocks_per_axis()[axis];
        let mut total = 0usize;
        for index in 0..blocks {
            let core_lo = index * block;
            let core_hi = ((index + 1) * block).min(volume);
            total += (core_hi + hi).min(volume) - core_lo.saturating_sub(lo);
        }
        total as f64 / blocks as f64
    }

    /// Every core, in natural (axis 0 slowest) order.
    pub fn cores(&self) -> Vec<BlockCore> {
        let counts = self.blocks_per_axis();
        let mut cores = Vec::with_capacity(self.n_blocks());
        let mut flat = 0usize;
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let index = [i, j, k];
                    let mut start = [0usize; 3];
                    let mut shape = [0usize; 3];
                    for axis in 0..3 {
                        start[axis] = index[axis] * self.block[axis];
                        shape[axis] =
                            (start[axis] + self.block[axis]).min(self.volume[axis]) - start[axis];
                    }
                    cores.push(BlockCore {
                        index,
                        flat,
                        core: Region::new(&start, &shape),
                    });
                    flat += 1;
                }
            }
        }
        cores
    }
}

/// One block's core, before any halo is decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCore {
    pub index: [usize; 3],
    pub flat: usize,
    pub core: Region,
}

/// A block's core, what is read for it, and what of it may be trusted.
///
/// `core`, `read` and `valid` are all in **this phase's own** coordinate space —
/// the space its `BlockGrid` is cut from, which is also the space its output
/// lands in. `source` is the odd one out: it is the region actually fetched from
/// the image below, in **that** image's space, and it is equal to `read`
/// wherever the two spaces are the same, which is every phase this crate shipped
/// before it existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockGeometry {
    pub index: [usize; 3],
    pub flat: usize,
    pub core: Region,
    pub read: Region,
    /// What the task fetches from the image below, in the image below's
    /// coordinate space. Defaults to `read`.
    ///
    /// **Why this is here rather than in an `Environment`.** A cross-grid read
    /// can be expressed today by writing an environment that quietly maps one
    /// region to another, and that works — but the plan then does not say what
    /// the run will move, so nothing prices it and nothing checks it. A measured
    /// case predicted `121 x payload` and moved four times the spatial volume,
    /// with no term anywhere for the difference. Stated here it is in the
    /// binding half: `Decomposition::exact_read_voxels` counts it,
    /// `Decomposition::check` refuses one that leaves the image it reads,
    /// `TaskGraph` builds dependencies from it, and the fingerprint records it.
    pub source: Region,
    /// `core ∩ trustworthy`. **Smaller than `core` iff the halo was short**,
    /// which is what the tiling check turns into an error.
    pub valid: Region,
}

impl BlockGeometry {
    /// The symmetric form: one integer per axis for the halo and one for the
    /// reach, in the phase's own voxels.
    ///
    /// Kept as the name every call site uses, because that is what most of them
    /// mean. It is [`Self::derive_with`] over the same numbers lifted into
    /// [`Reach`], not a second implementation — a guard with two derivations
    /// behind it is a guard that can be right in one of them.
    pub fn derive(
        core: &BlockCore,
        volume: [usize; 3],
        halo: [usize; 3],
        reach: [usize; 3],
    ) -> Self {
        Self::derive_with(core, volume, &Reach::from(halo), &Reach::from(reach))
    }

    /// The general form: a halo and a reach that may be asymmetric, per-block,
    /// whole-axis, and stated in a named coordinate space.
    ///
    /// Both must already be in the phase's own voxels — `Reach::in_voxels`
    /// converts, and `PhaseDecomposition::derive` is where that happens, because
    /// it is the first place a grid exists to convert with.
    ///
    /// **Which side of the block each number applies to.** To produce the voxel
    /// at `v` the operation reads `v - lo ..= v + hi`; a block reads
    /// `core_lo - halo_lo .. core_hi + halo_hi`. So the trustworthy extent is
    /// the read shrunk by `lo` at the bottom and `hi` at the top — the same
    /// inversion this module has always done, with the two sides no longer
    /// forced to be one number.
    ///
    /// **The clamp exception is conditional on the space, and on one axis on
    /// what the reach says.** A read clamped at the phase's own volume edge is
    /// trustworthy when that edge is a real edge of the array; a phase that
    /// crops or regrids has edges that are interior positions of the image
    /// below, and a reach stated in that frame (`Frame::Source`) is therefore
    /// not granted the exception. That is the direction of error the crate
    /// accepts: a seam that might be real is treated as real, so the guard fires
    /// rather than trusting a voxel whose context was never fetched.
    ///
    /// The exception to that exception is an axis the reach declares it consumes
    /// entirely — see the module header. There is no seam to be wrong about, so
    /// the grant is restored, per axis, for a block whose read spans the axis.
    pub fn derive_with(core: &BlockCore, volume: [usize; 3], halo: &Reach, reach: &Reach) -> Self {
        let mut read_start = [0usize; 3];
        let mut read_shape = [0usize; 3];
        let mut valid_start = [0usize; 3];
        let mut valid_shape = [0usize; 3];
        let mut degenerate = false;
        let trust_the_edge = reach.space().clamp_is_an_edge();

        for axis in 0..3 {
            let core_lo = core.core.start[axis];
            let core_hi = core_lo + core.core.shape[axis];
            let (halo_lo, halo_hi) = halo.at(axis, core.index[axis], volume[axis]);
            let (reach_lo, reach_hi) = reach.at(axis, core.index[axis], volume[axis]);

            let read_lo = core_lo.saturating_sub(halo_lo);
            let read_hi = (core_hi + halo_hi).min(volume[axis]);
            read_start[axis] = read_lo;
            read_shape[axis] = read_hi - read_lo;

            // **A consumed axis is not a seam.** `Frame::Source` is denied
            // the clamp exception because a cropping phase's edge is an
            // interior position of the array it reads — a neighbour exists
            // there and a halo could have reached it. An axis the op consumes
            // entirely has no beyond and no such neighbour, so the grant is
            // restored on it: the reach there is `All`, and this block's read
            // spans the whole of the axis.
            //
            // Both conditions are needed. Without the second, a block that read
            // part of a cut axis would be trusted for the part it never saw;
            // with it, no block of a cut axis is trusted under a finite halo,
            // every block stays degenerate, and the tiling check turns the
            // declaration into the whole-axis mandate it always implied.
            //
            // `Frame::Phase` is untouched — `trust_the_edge` is already true
            // there — so no plan that checked before this existed moves.
            let consumed = matches!(reach.axis(axis), AxisReach::All)
                && read_lo == 0
                && read_hi == volume[axis];
            let trust_axis = trust_the_edge || consumed;

            let trust_lo = if read_lo == 0 && trust_axis {
                0
            } else {
                read_lo + reach_lo
            };
            let trust_hi = if read_hi == volume[axis] && trust_axis {
                volume[axis]
            } else {
                read_hi.saturating_sub(reach_hi)
            };

            let lo = core_lo.max(trust_lo);
            let hi = core_hi.min(trust_hi);
            if lo >= hi {
                degenerate = true;
            } else {
                valid_start[axis] = lo;
                valid_shape[axis] = hi - lo;
            }
        }

        // A block with no trustworthy voxel is pinned to its core's lower
        // corner with zero extent, so it stays disjoint from every neighbour by
        // construction and the tiling check reports the *coverage* hole rather
        // than a spurious overlap. A misleading message from a guard is nearly
        // as bad as no guard.
        if degenerate {
            valid_start = [core.core.start[0], core.core.start[1], core.core.start[2]];
            valid_shape = [0, 0, 0];
        }

        let read = Region::new(&read_start, &read_shape);
        Self {
            index: core.index,
            flat: core.flat,
            core: core.core.clone(),
            source: read.clone(),
            read,
            valid: Region::new(&valid_start, &valid_shape),
        }
    }

    /// Read this block from `source` of the image below instead of from `read`.
    ///
    /// The halo arithmetic above is untouched: `read` stays the extent this
    /// phase's own reach and halo produced, `valid` stays derived from it, and
    /// the tiling guard still runs over `valid`. All this says is *where the
    /// bytes come from*, and it says it in the plan rather than in an
    /// environment — see the field's own documentation.
    ///
    /// It cannot be checked here, because what `source` must lie inside is the
    /// **previous** phase's volume and a `BlockGeometry` does not know its
    /// neighbours. `Decomposition::check` is where that lives.
    pub fn with_source(mut self, source: Region) -> Self {
        self.source = source;
        self
    }

    /// Whether this block reads a region other than its own read extent.
    ///
    /// The predicate the fingerprint and the wire format branch on, so that a
    /// plan which does not use the feature is byte-for-byte the plan it was
    /// before the feature existed.
    pub fn reads_across_grids(&self) -> bool {
        self.source != self.read
    }

    /// Where `valid` sits inside `read` — what the executor slices out of the
    /// block buffer before writing.
    pub fn valid_within_read(&self) -> Region {
        let mut start = [0usize; 3];
        for axis in 0..3 {
            start[axis] = self.valid.start[axis] - self.read.start[axis];
        }
        Region::new(&start, &self.valid.shape)
    }

    /// Whether every voxel of the core is trustworthy.
    ///
    /// Not itself a guard — the tiling check is the guard — but it is how a
    /// diagnostic says *which* block lost its valid region once the guard has
    /// fired.
    pub fn valid_covers_core(&self) -> bool {
        self.valid == self.core
    }
}

/// `region` lies inside `shape`.
///
/// `Region::check_within` in `region_io` says the same thing but is private to
/// that module; this is a local copy rather than a visibility change there,
/// because `region_io` is not this task's ground.
pub fn region_within(region: &Region, shape: &[usize], what: &str) -> Result<()> {
    if region.ndim() != shape.len() {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: region.shape.clone(),
        });
    }
    for axis in 0..shape.len() {
        if region.start[axis] + region.shape[axis] > shape[axis] {
            return Err(Error::InvalidArgument(format!(
                "{what}: region axis {axis} spans {}..{} of {}",
                region.start[axis],
                region.start[axis] + region.shape[axis],
                shape[axis]
            )));
        }
    }
    Ok(())
}

/// Do two regions share a voxel?
pub fn regions_intersect(left: &Region, right: &Region) -> bool {
    (0..left.start.len()).all(|axis| {
        let (a_lo, a_hi) = (left.start[axis], left.start[axis] + left.shape[axis]);
        let (b_lo, b_hi) = (right.start[axis], right.start[axis] + right.shape[axis]);
        a_lo < b_hi && b_lo < a_hi
    })
}

/// Chunks of a `chunk`-sized grid that a read of `region` must touch.
///
/// This is the loader-side cost the design asks to be counted: "a plan that
/// says fuse these three and an executor that materialises between them agree
/// on output while disagreeing on everything the plan exists to control".
pub fn chunks_touched(region: &Region, chunk: &[usize]) -> u64 {
    (0..region.start.len())
        .map(|axis| {
            let edge = chunk.get(axis).copied().unwrap_or(1).max(1);
            if region.shape[axis] == 0 {
                return 0u64;
            }
            let first = region.start[axis] / edge;
            let last = (region.start[axis] + region.shape[axis] - 1) / edge;
            (last - first + 1) as u64
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sufficient_halo_leaves_the_valid_region_equal_to_the_core() {
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        for core in grid.cores() {
            let geometry = BlockGeometry::derive(&core, [64, 8, 8], [4, 0, 0], [4, 0, 0]);
            assert_eq!(geometry.valid, geometry.core, "block {:?}", core.index);
        }
    }

    #[test]
    fn a_generous_halo_costs_reads_and_nothing_else() {
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        for core in grid.cores() {
            let tight = BlockGeometry::derive(&core, [64, 8, 8], [4, 0, 0], [4, 0, 0]);
            let generous = BlockGeometry::derive(&core, [64, 8, 8], [12, 0, 0], [4, 0, 0]);
            assert_eq!(generous.valid, generous.core);
            assert_eq!(generous.valid, tight.valid);
            assert!(generous.read.voxels() >= tight.read.voxels());
        }
    }

    #[test]
    fn a_short_halo_shrinks_the_valid_region_below_the_core() {
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        let cores = grid.cores();
        // interior block: both seams are real, so both sides shrink by r - h
        let interior = BlockGeometry::derive(&cores[1], [64, 8, 8], [2, 0, 0], [5, 0, 0]);
        assert_eq!((interior.core.start[0], interior.core.shape[0]), (16, 16));
        assert_eq!((interior.valid.start[0], interior.valid.shape[0]), (19, 10));
        assert!(!interior.valid_covers_core());

        // the volume-edge side of block 0 stays valid: nothing beyond to read
        let first = BlockGeometry::derive(&cores[0], [64, 8, 8], [2, 0, 0], [5, 0, 0]);
        assert_eq!((first.valid.start[0], first.valid.shape[0]), (0, 13));
    }

    #[test]
    fn a_block_with_no_trustworthy_voxel_is_pinned_to_its_core_corner() {
        let grid = BlockGrid::new([48, 4, 4], [16, 4, 4]).unwrap();
        let cores = grid.cores();
        let geometry = BlockGeometry::derive(&cores[1], [48, 4, 4], [0, 0, 0], [40, 0, 0]);
        assert_eq!(geometry.valid.shape, vec![0, 0, 0]);
        assert_eq!(geometry.valid.start, geometry.core.start);
    }

    #[test]
    fn a_single_block_grid_has_no_seams_so_no_halo_can_be_short() {
        let grid = BlockGrid::whole([32, 16, 8]).unwrap();
        assert_eq!(grid.n_blocks(), 1);
        assert!(grid.split_axes().is_empty());
        let core = &grid.cores()[0];
        // even a reach of 1000 with a halo of 0 leaves the whole volume valid
        let geometry = BlockGeometry::derive(core, [32, 16, 8], [0, 0, 0], [1000, 1000, 1000]);
        assert_eq!(geometry.valid, geometry.core);
        assert_eq!(geometry.read, geometry.core);
    }

    /// The default is what keeps every existing plan the plan it was: a block
    /// reads its own read extent until somebody says otherwise.
    #[test]
    fn a_derived_block_reads_its_own_read_extent() {
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        for core in grid.cores() {
            let geometry = BlockGeometry::derive(&core, [64, 8, 8], [4, 0, 0], [4, 0, 0]);
            assert_eq!(geometry.source, geometry.read);
            assert!(!geometry.reads_across_grids());
        }
    }

    /// And a source only moves the *fetch*: the halo arithmetic, the valid
    /// region and therefore the tiling guard are untouched by it.
    #[test]
    fn a_source_moves_the_fetch_and_nothing_else() {
        let grid = BlockGrid::new([32, 8, 8], [16, 8, 8]).unwrap();
        let core = &grid.cores()[1];
        let plain = BlockGeometry::derive(core, [32, 8, 8], [2, 0, 0], [2, 0, 0]);
        let moved = plain
            .clone()
            .with_source(Region::new(&[100, 0, 0], &[18, 8, 8]));
        assert!(moved.reads_across_grids());
        assert_eq!(moved.read, plain.read);
        assert_eq!(moved.valid, plain.valid);
        assert_eq!(moved.valid_within_read(), plain.valid_within_read());
        assert_eq!(moved.source.voxels(), 18 * 8 * 8);
    }

    #[test]
    fn chunk_accounting_counts_partial_chunks_at_both_ends() {
        // 10..30 over a chunk edge of 8 touches chunks 1, 2, 3
        assert_eq!(chunks_touched(&Region::new(&[10], &[20]), &[8]), 3);
        assert_eq!(chunks_touched(&Region::new(&[0], &[8]), &[8]), 1);
        assert_eq!(chunks_touched(&Region::new(&[0, 0], &[9, 9]), &[8, 8]), 4);
    }
}
