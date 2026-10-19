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
    let pre_range = face_range(pre_volume.shape()[axis], depth, Face::Trailing);
    let post_range = face_range(post_volume.shape()[axis], depth, Face::Leading);
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
    if volumes.len() != layout.sources().len() {
        return Err(Error::invalid(format!(
            "the layout has {} tiles and {} volumes were handed in",
            layout.sources().len(),
            volumes.len()
        )));
    }
    let results = (0..layout.alignments().len())
        .map(|index| {
            let pair = &layout.alignments()[index];
            align_pair(
                layout,
                index,
                volumes[pair.pre].view(),
                volumes[pair.post].view(),
                params,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, result) in results.into_iter().enumerate() {
        layout.set_shift(index, result.shift, result.quality)?;
    }
    Ok(())
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
