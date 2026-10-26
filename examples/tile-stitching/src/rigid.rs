// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The rigid alignment of each neighbouring pair, from max projections of the
//! facing ends.
//!
//! For one pair, with `axis` the volume axis the two tiles are tiled along:
//!
//! ```text
//! slab_pre  = pre [shape[axis] - depth[axis] ..]      the facing end
//! slab_post = post[.. depth[axis]]                    the facing end
//! mip_pre, mip_post = max over axis of each slab      3-D -> 2-D
//! shift = argmin over shifts of the mean squared difference of the two
//!         projections, placed at the tiles' current positions
//! ```
//!
//! The shift goes back into 3-D with a zero on `axis`: the grid spacing is
//! taken as correct along the tiling axis and only the two transverse axes are
//! corrected, which falls out of projecting that axis away.
//!
//! The window is the **declared** overlap (`depth`), not the one the positions
//! imply. The score is the sum of squared differences divided by the number of
//! pixels that overlap at that shift: without the divide the overlap shrinks as
//! the shift grows and the answer is biased towards the edge of the search. It
//! is evaluated at every shift at once through
//! [`blockflow::ops::fft::SquaredDifference`]; among equal scores the smallest
//! shift (lexicographically) wins.
//!
//! Only the facing slabs are read and they are reduced to 2-D at once, so the
//! cost is the projection area times the number of candidate shifts, not the
//! tile volume.

use blockflow::error::{Error, Result};
use blockflow::ops::fft::{Padding, SquaredDifference};
use blockflow::voxels::VoxelElement;
use ndarray::{s, Array2, ArrayView2, ArrayView3};

use crate::layout::TileLayout;
use crate::plane::PlaneGeometry;
use crate::tiles::{admitted, bounding, chunk_pieces, BoxAt, KeptBox, KeptPair, TileBoxes};

/// Which end of a tile the correlation window is taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    /// The last `depth` planes: the `pre` tile's facing end.
    Trailing,
    /// The first `depth` planes: the `post` tile's facing end.
    Leading,
}

/// When a pair is too dark to align.
///
/// A rejected pair gets `quality = -inf` and is dropped by the placement, so
/// both tiles keep their nominal positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Background {
    /// No check.
    None,
    /// Reject when every value in either window is below `level`.
    AllBelow { level: f64 },
    /// Reject when fewer than `minimum` values in either window reach `level`.
    Counted { level: f64, minimum: MinimumCount },
}

/// How many pixels above the background level are enough.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MinimumCount {
    Absolute(usize),
    /// A fraction of the window's size.
    Fraction(f64),
}

/// One pair's answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairAlignment {
    /// The correction to the grid's guess, zero on the tiling axis.
    pub shift: [isize; 3],
    /// Minus the minimised score, or `-inf` when the background check refused
    /// the pair.
    pub quality: f64,
}

/// One pair's answer in the projected frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairAlignment2 {
    pub shift: [isize; 2],
    pub quality: f64,
}

/// The volume axis alignment `index` is tiled along: for grid neighbours,
/// the grid axis their grid positions differ on; for neighbours found from
/// boxes, the axis their boxes were found offset along.
pub fn mip_axis(layout: &TileLayout, index: usize) -> Result<usize> {
    layout
        .tiling_axis(index)
        .ok_or_else(|| Error::invalid(format!("no alignment {index}")))
}

/// The plane range the correlation window is taken from, clamped to the tile.
pub fn face_range(shape: usize, depth: usize, face: Face) -> std::ops::Range<usize> {
    match face {
        Face::Trailing => shape.saturating_sub(depth)..shape,
        Face::Leading => 0..depth.min(shape),
    }
}

/// Max-project a slab along one axis.
///
/// Compared with `total_cmp`, so `+0.0` beats `-0.0` and the projection does
/// not depend on the order the planes are visited in.
pub fn project_max<T: VoxelElement>(slab: ArrayView3<T>, axis: usize) -> Result<Array2<f64>> {
    if axis > 2 {
        return Err(Error::invalid(format!(
            "axis {axis} is not one of a 3-D slab's three"
        )));
    }
    let shape = slab.shape();
    if shape[axis] == 0 {
        return Err(Error::invalid(
            "an empty slab has no maximum; the correlation window has zero depth",
        ));
    }
    let kept: Vec<usize> = (0..3).filter(|other| *other != axis).collect();
    let mut projection = Array2::from_elem((shape[kept[0]], shape[kept[1]]), f64::NEG_INFINITY);
    if axis == 2 {
        // Along the last axis a plane is strided and a lane is contiguous:
        // each output value is its lane's maximum (the same maximum, in any
        // order, by `total_cmp`).
        for (target, lane) in projection.iter_mut().zip(slab.lanes(ndarray::Axis(2))) {
            for value in lane.iter() {
                let value = value.into_f64();
                if value.total_cmp(target).is_gt() {
                    *target = value;
                }
            }
        }
        return Ok(projection);
    }
    for plane in slab.axis_iter(ndarray::Axis(axis)) {
        for (target, value) in projection.iter_mut().zip(plane.iter()) {
            let value = value.into_f64();
            if value.total_cmp(target).is_gt() {
                *target = value;
            }
        }
    }
    Ok(projection)
}

/// Fold a block's partial projection into the running one. A per-column
/// maximum is associative and commutative, so any cut of the slab folds to the
/// same answer.
pub fn fold_projection(accumulator: &mut Array2<f64>, partial: ArrayView2<f64>) -> Result<()> {
    if accumulator.shape() != partial.shape() {
        return Err(Error::ShapeMismatch {
            expected: accumulator.shape().to_vec(),
            got: partial.shape().to_vec(),
        });
    }
    accumulator.zip_mut_with(&partial, |target, &value| {
        if value.total_cmp(target).is_gt() {
            *target = value;
        }
    });
    Ok(())
}

fn clip(window: ArrayView2<f64>, bounds: Option<(Option<f64>, Option<f64>)>) -> Array2<f64> {
    let mut prepared = window.to_owned();
    if let Some((low, high)) = bounds {
        prepared.mapv_inplace(|value| {
            let value = low.map_or(value, |low| value.max(low));
            high.map_or(value, |high| value.min(high))
        });
    }
    prepared
}

fn background_rejects(window: ArrayView2<f64>, background: Background) -> bool {
    match background {
        Background::None => false,
        Background::AllBelow { level } => window.iter().all(|value| *value < level),
        Background::Counted { level, minimum } => {
            let above = window.iter().filter(|value| **value >= level).count();
            let needed = match minimum {
                MinimumCount::Absolute(count) => count as f64,
                MinimumCount::Fraction(fraction) => window.len() as f64 * fraction,
            };
            (above as f64) < needed
        }
    }
}

/// Subtract the mean and divide by the norm.
fn normalise(window: &mut Array2<f64>) {
    if window.is_empty() {
        return;
    }
    let mean = window.mean().unwrap_or(0.0);
    window.mapv_inplace(|value| value - mean);
    let norm = window.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        window.mapv_inplace(|value| value / norm);
    }
}

/// The correlation of two projections placed at `pre_position` and
/// `post_position`, over an inclusive shift range per axis.
pub fn align_projections(
    pre: ArrayView2<f64>,
    pre_position: [isize; 2],
    post: ArrayView2<f64>,
    post_position: [isize; 2],
    max_shifts: [(isize, isize); 2],
    params: &RigidParams,
) -> Result<PairAlignment2> {
    let geometry = PlaneGeometry::new(
        pre_position,
        [pre.shape()[0], pre.shape()[1]],
        post_position,
        [post.shape()[0], post.shape()[1]],
        max_shifts.map(|(low, high)| [low, high]),
    )?;
    let cut = |plane: ArrayView2<f64>, ranges: [(usize, usize); 2]| {
        clip(
            plane.slice(s![ranges[0].0..ranges[0].1, ranges[1].0..ranges[1].1]),
            params.clip,
        )
    };
    let mut a = cut(pre, geometry.cut_a);
    let mut b = cut(post, geometry.cut_b);
    if background_rejects(a.view(), params.background)
        || background_rejects(b.view(), params.background)
    {
        return Ok(PairAlignment2 {
            shift: [0, 0],
            quality: f64::NEG_INFINITY,
        });
    }
    if params.normalize {
        normalise(&mut a);
        normalise(&mut b);
    }
    let mut plan = SquaredDifference::new(
        geometry.cut_shape_a(),
        geometry.cut_shape_b(),
        geometry.window()?,
        Padding::Smooth,
    )?;
    let landscape = plan.landscape(a.view(), b.view())?;
    let (lag, value) = landscape
        .argmin()
        .ok_or_else(|| Error::invalid("no candidate shift left the two windows overlapping"))?;
    Ok(PairAlignment2 {
        shift: std::array::from_fn(|axis| lag[axis] - geometry.lag_offset[axis]),
        quality: -value,
    })
}

/// One pair, from its two projections along the tiling axis.
pub fn align_pair_projected(
    layout: &TileLayout,
    index: usize,
    pre: ArrayView2<f64>,
    post: ArrayView2<f64>,
    params: &RigidParams,
) -> Result<PairAlignment> {
    let axis = mip_axis(layout, index)?;
    let pair = &layout.alignments()[index];
    let transverse: Vec<usize> = (0..3).filter(|other| *other != axis).collect();
    let pre_position = std::array::from_fn(|i| layout.positions()[pair.pre][transverse[i]]);
    let post_position = std::array::from_fn(|i| layout.positions()[pair.post][transverse[i]]);
    let max_shifts = std::array::from_fn(|i| params.max_shifts[transverse[i]]);
    let flat = align_projections(pre, pre_position, post, post_position, max_shifts, params)?;
    let mut shift = [0isize; 3];
    for (i, &target) in transverse.iter().enumerate() {
        shift[target] = flat.shift[i];
    }
    Ok(PairAlignment {
        shift,
        quality: flat.quality,
    })
}

/// One pair, from the two whole tiles: the slabs, the projections and the
/// correlation.
pub fn align_pair<T: VoxelElement>(
    layout: &TileLayout,
    index: usize,
    pre_volume: ArrayView3<T>,
    post_volume: ArrayView3<T>,
    params: &RigidParams,
) -> Result<PairAlignment> {
    let axis = mip_axis(layout, index)?;
    let (axis, pre_range, post_range) = faces(
        layout,
        index,
        pre_volume.shape()[axis],
        post_volume.shape()[axis],
        params,
    )?;
    let pre = project_max(
        pre_volume.slice_axis(ndarray::Axis(axis), pre_range.into()),
        axis,
    )?;
    let post = project_max(
        post_volume.slice_axis(ndarray::Axis(axis), post_range.into()),
        axis,
    )?;
    align_pair_projected(layout, index, pre.view(), post.view(), params)
}

/// Every pair of the layout, aligned and written back into it.
pub fn align_layout<T: VoxelElement>(
    layout: &mut TileLayout,
    volumes: &[ArrayView3<T>],
    params: &RigidParams,
) -> Result<()> {
    align_layout_boxes(layout, volumes, params)
}

/// The facing ends pair `index` is aligned from: the tiling axis and the
/// plane ranges of `pre` and `post` along it.
fn faces(
    layout: &TileLayout,
    index: usize,
    pre_extent: usize,
    post_extent: usize,
    params: &RigidParams,
) -> Result<(usize, std::ops::Range<usize>, std::ops::Range<usize>)> {
    let axis = mip_axis(layout, index)?;
    let depth = params.depth[axis].ok_or_else(|| {
        Error::invalid(format!(
            "no projection depth is set for axis {axis}, which this pair is tiled along; set it \
             to the nominal overlap (`RigidParams::from_overlaps`)"
        ))
    })?;
    if depth == 0 {
        return Err(Error::invalid(format!(
            "the projection depth for axis {axis} is zero, which selects no planes"
        )));
    }
    Ok((
        axis,
        face_range(pre_extent, depth, Face::Trailing),
        face_range(post_extent, depth, Face::Leading),
    ))
}

/// [`align_layout`] reading each pair's facing ends from `tiles`, pairs in
/// parallel as the memory cap allows ([`crate::tiles::admitted`]): a pair
/// holds its two slabs while it projects them.
pub fn align_layout_boxes<T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    layout: &mut TileLayout,
    tiles: &B,
    params: &RigidParams,
) -> Result<()> {
    align_layout_keeping(layout, tiles, params, &[]).map(drop)
}

/// [`align_layout_boxes`], also reading for each pair `index` the boxes
/// `later[index]` names (of `pre`, then of `post`) where that decodes fewer
/// stored chunks than reading them apart: the facing slab and that box are
/// read as one, the slab projected from it, and the box kept, leased from the
/// process budget, for the step that wants it. A pair whose boxes would not
/// save a decode, or that has no room under the cap, reads its slabs alone.
pub fn align_layout_keeping<'a, T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    layout: &mut TileLayout,
    tiles: &'a B,
    params: &RigidParams,
    later: &[Option<[BoxAt; 2]>],
) -> Result<Vec<Option<KeptPair<'a, T>>>> {
    if tiles.len() != layout.sources().len() {
        return Err(Error::invalid(format!(
            "the layout has {} tiles and {} volumes were handed in",
            layout.sources().len(),
            tiles.len()
        )));
    }
    // The slab of `tile` along `axis` over `range`, whole across it.
    let slab = |tile: usize, axis: usize, range: &std::ops::Range<usize>| {
        let mut start = [0usize; 3];
        let mut shape = tiles.shape(tile);
        start[axis] = range.start;
        shape[axis] = range.len();
        (start, shape)
    };
    let view: &TileLayout = layout;
    let bytes = |&index: &usize| -> u64 {
        let pair = &view.alignments()[index];
        let (pre_extent, post_extent) = (tiles.shape(pair.pre), tiles.shape(pair.post));
        let Ok(axis) = mip_axis(view, index) else {
            return 0;
        };
        let Ok((_, pre_range, post_range)) =
            faces(view, index, pre_extent[axis], post_extent[axis], params)
        else {
            return 0;
        };
        let (pre_start, pre_shape) = slab(pair.pre, axis, &pre_range);
        let (post_start, post_shape) = slab(pair.post, axis, &post_range);
        // The largest piece of a slab as read (one at a time), and the two
        // projections in `f64` with the correlation's padded copies. (A pair
        // that keeps its boxes leases them apart.)
        let plane =
            |shape: [usize; 3]| (shape.iter().product::<usize>() / shape[axis].max(1)) as u64;
        let piece = |tile: usize, (start, shape): BoxAt| {
            let split = split_axis(shape, axis);
            let row = tiles.read_bytes(shape) / shape[split].max(1) as u64;
            chunk_pieces(
                tiles,
                tile,
                split,
                start[split]..start[split] + shape[split],
                row,
                SLAB_PIECE_BYTES,
            )
            .iter()
            .map(|range| row * range.len() as u64)
            .max()
            .unwrap_or(0)
        };
        piece(pair.pre, (pre_start, pre_shape)).max(piece(pair.post, (post_start, post_shape)))
            + 8 * 8 * (plane(pre_shape) + plane(post_shape))
    };
    // What the kept boxes hold between them: three quarters of what the later
    // step's boxes take with a pair on every worker, since its pairs start
    // and finish staggered and never hold all of that at once (keeping all of
    // it raised osf-sa3x8's peak 3% over reading twice; three quarters lowered
    // it 10%, and osf-cztq2's 15%, at the same CPU time); and no more than
    // holding every tile (the ceiling), nor half the cap, so the pairs in
    // flight keep room. The later step measures pairs in order, so the first
    // pairs are the ones worth keeping.
    let budget = blockflow::budget::process_budget();
    let in_flight: u64 = later
        .iter()
        .take(rayon::current_num_threads().max(1))
        .flatten()
        .map(|boxes| boxes.iter().map(|b| tiles.read_bytes(b.1)).sum::<u64>())
        .sum();
    let keep_limit = (in_flight / 4 * 3)
        .min(tiles.ceiling())
        .min(budget.total() / 2);
    let kept_bytes = std::sync::atomic::AtomicU64::new(0);
    let results = admitted(
        (0..view.alignments().len()).collect(),
        bytes,
        tiles.ceiling(),
        |index| {
            let pair = &view.alignments()[index];
            let (pre_extent, post_extent) = (tiles.shape(pair.pre), tiles.shape(pair.post));
            let axis = mip_axis(view, index)?;
            let (axis, pre_range, post_range) =
                faces(view, index, pre_extent[axis], post_extent[axis], params)?;
            let slabs = [
                slab(pair.pre, axis, &pre_range),
                slab(pair.post, axis, &post_range),
            ];
            let tile = [pair.pre, pair.post];
            // Read as one where that decodes less than reading apart.
            let joint = later.get(index).copied().flatten().and_then(|boxes| {
                let joint: [BoxAt; 2] = std::array::from_fn(|i| bounding(slabs[i], boxes[i]));
                let decoded = |i: usize, b: BoxAt| tiles.decoded(tile[i], b.0, b.1);
                let apart: u64 = (0..2)
                    .map(|i| decoded(i, slabs[i]) + decoded(i, boxes[i]))
                    .sum();
                let together: u64 = (0..2).map(|i| decoded(i, joint[i])).sum();
                (together < apart).then_some(joint)
            });
            let lease = joint.and_then(|joint| {
                let want = joint.iter().map(|b| tiles.read_bytes(b.1)).sum::<u64>();
                let held = kept_bytes.fetch_add(want, std::sync::atomic::Ordering::Relaxed);
                let lease = (held + want <= keep_limit)
                    .then(|| budget.try_acquire_reclaiming(want))
                    .flatten();
                if lease.is_none() {
                    kept_bytes.fetch_sub(want, std::sync::atomic::Ordering::Relaxed);
                }
                lease.map(|lease| (joint, lease))
            });
            match lease {
                Some((joint, lease)) => {
                    let [pre, post] = [0, 1].map(|i| -> Result<KeptBox<'a, T>> {
                        Ok(KeptBox {
                            at: joint[i],
                            data: tiles.read(tile[i], joint[i].0, joint[i].1)?,
                        })
                    });
                    let (pre, post) = (pre?, post?);
                    let pre_projection = project_max(pre.view_of(slabs[0]), axis)?;
                    let post_projection = project_max(post.view_of(slabs[1]), axis)?;
                    let result = align_pair_projected(
                        view,
                        index,
                        pre_projection.view(),
                        post_projection.view(),
                        params,
                    )?;
                    Ok((result, Some(KeptPair { pre, post, lease })))
                }
                None => {
                    let pre = project_slab(tiles, pair.pre, slabs[0], axis, SLAB_PIECE_BYTES)?;
                    let post = project_slab(tiles, pair.post, slabs[1], axis, SLAB_PIECE_BYTES)?;
                    let result =
                        align_pair_projected(view, index, pre.view(), post.view(), params)?;
                    Ok((result, None))
                }
            }
        },
    )?;
    let mut kept = Vec::with_capacity(results.len());
    for (index, (result, pair)) in results.into_iter().enumerate() {
        layout.set_shift(index, result.shift, result.quality)?;
        kept.push(pair);
    }
    Ok(kept)
}

/// About how much of a slab a pair reads at once when it does not keep it.
const SLAB_PIECE_BYTES: u64 = 8 << 20;

/// The axis a slab projected along `axis` is read in pieces along: its
/// longest other axis (the first on a tie), each piece projecting to whole
/// rows of the projection.
fn split_axis(shape: [usize; 3], axis: usize) -> usize {
    let mut best = usize::from(axis == 0);
    for a in 0..3 {
        if a != axis && shape[a] > shape[best] {
            best = a;
        }
    }
    best
}

/// The projection along `axis` of tile `tile`'s box `slab`, read in pieces of
/// whole stored chunks of about `target` bytes ([`chunk_pieces`]) so a pair
/// never holds its slab whole: each projected value comes from one piece, so
/// it is the projection of the slab read at once.
fn project_slab<T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    tiles: &B,
    tile: usize,
    (start, shape): BoxAt,
    axis: usize,
    target: u64,
) -> Result<Array2<f64>> {
    let split = split_axis(shape, axis);
    let row = tiles.read_bytes(shape) / shape[split].max(1) as u64;
    let pieces = chunk_pieces(
        tiles,
        tile,
        split,
        start[split]..start[split] + shape[split],
        row,
        target,
    );
    if pieces.len() <= 1 {
        return project_max(tiles.read(tile, start, shape)?.view(), axis);
    }
    let kept: Vec<usize> = (0..3).filter(|other| *other != axis).collect();
    let along = usize::from(kept[1] == split);
    let mut projection = Array2::from_elem((shape[kept[0]], shape[kept[1]]), f64::NEG_INFINITY);
    for range in pieces {
        let (mut at, mut extent) = (start, shape);
        at[split] = range.start;
        extent[split] = range.len();
        let part = project_max(tiles.read(tile, at, extent)?.view(), axis)?;
        projection
            .slice_axis_mut(
                ndarray::Axis(along),
                (range.start - start[split]..range.end - start[split]).into(),
            )
            .assign(&part);
    }
    Ok(projection)
}

/// The rigid correlation's parameters, per volume axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidParams {
    /// The correlation window's thickness along each tiling axis; `None` for an
    /// axis no pair is tiled along.
    pub depth: [Option<usize>; 3],
    /// The inclusive search range per volume axis. The tiling axis's entry is
    /// unused, since the projection removes it.
    pub max_shifts: [(isize, isize); 3],
    /// `(low, high)`, either half optional. Clipping a saturated tail stops a
    /// few blown-out pixels from dominating the squared difference.
    pub clip: Option<(Option<f64>, Option<f64>)>,
    pub background: Background,
    /// Subtract the mean and divide by the norm of each window first. Useful
    /// when the two tiles were acquired at different gains.
    pub normalize: bool,
}

impl Default for RigidParams {
    /// A `(z, y, x)` tile set tiled in x and y: +/-30 voxels in-plane and
    /// +/-20 along z. `depth` is unset; [`RigidParams::from_overlaps`] fills it.
    fn default() -> Self {
        Self {
            depth: [None; 3],
            max_shifts: [(-20, 20), (-30, 30), (-30, 30)],
            clip: Some((None, Some(25_000.0))),
            background: Background::Counted {
                level: 0.0,
                minimum: MinimumCount::Absolute(100),
            },
            normalize: false,
        }
    }
}

impl RigidParams {
    /// The defaults, with the window set to the nominal overlap along each
    /// tiling axis: `overlaps[d]` along `grid_axes[d]`.
    pub fn from_overlaps(overlaps: &[usize], grid_axes: &[usize]) -> Self {
        let mut params = Self::default();
        for (grid_axis, &axis) in grid_axes.iter().enumerate() {
            params.depth[axis] = overlaps.get(grid_axis).or(overlaps.last()).copied();
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::TileBox;

    /// Resident tiles that say they are stored in chunks, so a slab is read
    /// in pieces.
    struct Chunked<'a>(Vec<ArrayView3<'a, u16>>);

    impl TileBoxes<u16> for Chunked<'_> {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn shape(&self, tile: usize) -> [usize; 3] {
            let shape = self.0[tile].shape();
            [shape[0], shape[1], shape[2]]
        }

        fn read(
            &self,
            tile: usize,
            start: [usize; 3],
            shape: [usize; 3],
        ) -> Result<TileBox<'_, u16>> {
            self.0[..].read(tile, start, shape)
        }

        fn read_bytes(&self, shape: [usize; 3]) -> u64 {
            2 * shape.iter().product::<usize>() as u64
        }

        fn chunking(&self, _tile: usize) -> Option<([usize; 3], [usize; 3])> {
            Some(([4, 8, 8], [0, 3, 0]))
        }
    }

    #[test]
    fn a_slab_projected_in_pieces_is_the_slab_projected_whole() {
        let volume = ndarray::Array3::from_shape_fn((37, 21, 19), |(z, y, x)| {
            ((z * 131 + y * 17 + x * 7) % 251) as u16
        });
        let tiles = Chunked(vec![volume.view()]);
        for axis in 0..3 {
            let mut start = [1, 2, 3];
            let mut shape = [33, 17, 15];
            start[axis] = 5;
            shape[axis] = 6;
            let whole = project_max(
                volume.slice(s![
                    start[0]..start[0] + shape[0],
                    start[1]..start[1] + shape[1],
                    start[2]..start[2] + shape[2]
                ]),
                axis,
            )
            .unwrap();
            // A target of one row gives pieces of one stored chunk each.
            let pieces = project_slab(&tiles, 0, (start, shape), axis, 1).unwrap();
            assert_eq!(pieces, whole, "axis {axis}");
        }
    }

    #[test]
    fn a_projection_along_the_last_axis_is_the_one_plane_by_plane() {
        let volume = ndarray::Array3::from_shape_fn((9, 7, 5), |(z, y, x)| {
            ((z * 31 + y * 7 + x * 3) % 17) as f32 - 8.0
        });
        let lanes = project_max(volume.view(), 2).unwrap();
        let planes = Array2::from_shape_fn((9, 7), |(z, y)| {
            (0..5)
                .map(|x| f64::from(volume[[z, y, x]]))
                .fold(f64::NEG_INFINITY, f64::max)
        });
        assert_eq!(lanes, planes);
    }
}
