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

use crate::error::{Error, Result};
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
    pub fn core_voxels(&self) -> f64 {
        self.block.iter().map(|&edge| edge as f64).product()
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
/// the level below, in **that** level's space, and it is equal to `read`
/// wherever the two spaces are the same, which is every phase this crate shipped
/// before it existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockGeometry {
    pub index: [usize; 3],
    pub flat: usize,
    pub core: Region,
    pub read: Region,
    /// What the task fetches from the level below, in the level below's
    /// coordinate space. Defaults to `read`.
    ///
    /// **Why this is here rather than in an `Environment`.** A cross-grid read
    /// can be expressed today by writing an environment that quietly maps one
    /// region to another, and that works — but the plan then does not say what
    /// the run will move, so nothing prices it and nothing checks it. A measured
    /// case predicted `121 x payload` and moved four times the spatial volume,
    /// with no term anywhere for the difference. Stated here it is in the
    /// binding half: `Decomposition::exact_read_voxels` counts it,
    /// `Decomposition::check` refuses one that leaves the level it reads,
    /// `TaskGraph` builds dependencies from it, and the fingerprint records it.
    pub source: Region,
    /// `core ∩ trustworthy`. **Smaller than `core` iff the halo was short**,
    /// which is what the tiling check turns into an error.
    pub valid: Region,
}

impl BlockGeometry {
    pub fn derive(
        core: &BlockCore,
        volume: [usize; 3],
        halo: [usize; 3],
        reach: [usize; 3],
    ) -> Self {
        let mut read_start = [0usize; 3];
        let mut read_shape = [0usize; 3];
        let mut valid_start = [0usize; 3];
        let mut valid_shape = [0usize; 3];
        let mut degenerate = false;

        for axis in 0..3 {
            let core_lo = core.core.start[axis];
            let core_hi = core_lo + core.core.shape[axis];

            let read_lo = core_lo.saturating_sub(halo[axis]);
            let read_hi = (core_hi + halo[axis]).min(volume[axis]);
            read_start[axis] = read_lo;
            read_shape[axis] = read_hi - read_lo;

            let trust_lo = if read_lo == 0 {
                0
            } else {
                read_lo + reach[axis]
            };
            let trust_hi = if read_hi == volume[axis] {
                volume[axis]
            } else {
                read_hi.saturating_sub(reach[axis])
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

    /// Read this block from `source` of the level below instead of from `read`.
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
