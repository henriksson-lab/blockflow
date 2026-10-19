// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Fusion as a planned, out-of-core phase: the output is computed in slabs of
//! whole chunk rows along one axis, and each slab reads only the boxes of the
//! tiles it needs.
//!
//! **The plan.** Before anything is read, every tile's box for the whole
//! output region is known ([`crate::fuse`]'s `needed_box`: the planes usable
//! there and, per plane, the in-plane part that falls inside the region at that
//! plane's own position). The region is then cut along the sweep axis into
//! slabs of whole level-0 chunk rows of the output image, and each slab's box
//! of each tile is known the same way.
//!
//! **Reads.** A tile is held as one buffer: its whole-region box across the
//! sweep axis, and along it the rows the current slab needs. Moving to the next
//! slab keeps the rows still needed and reads the rest, up to the next chunk
//! boundary of the tile's own storage; so every stored tile chunk is decoded
//! once (unless a tile's needs move backwards, which takes a wobble larger than
//! a slab, and costs a re-read rather than a wrong answer). The next slab's
//! reads run while the current slab is fused and written.
//!
//! **Fusion.** A slab's output chunks are fused in parallel by the same kernel
//! as the resident fusion, reading from the buffers instead of whole tiles, so
//! every output voxel is the same function of the same tile voxels in the same
//! order, and the result is identical to [`crate::fuse::fuse_region`] over the
//! whole region (`tests/stitch_planned.rs`).
//!
//! **Why not a `blockflow` plan.** A phase's supplied inputs must share the
//! plan's volume and lattice (`ZarrEnvironment::attach` refuses an input of
//! another shape), and a phase's per-block source regions are a stencil or a
//! table over that one lattice. Here every tile has its own shape and sits at
//! its own translation per plane, so the per-block read of each input is a box
//! of a differently shaped image; that is what this module plans by hand.

use std::sync::Arc;
use std::time::Instant;

use blockflow::error::{Error, Result};
use blockflow::region::Region;
use blockflow::zarr_env::AttachedImage;
use ndarray::{s, Array3, Axis};
use rayon::prelude::*;

use crate::fuse::{fuse_with, needed_box, FuseParams, Placement};
use crate::pyramid::PyramidElement;

/// Where one tile's voxels are read from.
#[derive(Debug, Clone)]
pub struct TileSource {
    /// The tile's level, as a rank-three image.
    pub image: AttachedImage,
    /// Tile coordinates are stored voxel minus this (the stitched planes' first
    /// stored plane along z).
    pub offset: [usize; 3],
    /// The stored chunk shape, `[z, y, x]`; reads are extended to its
    /// boundaries along the sweep axis.
    pub chunk: [usize; 3],
}

/// What a planned fusion did.
#[derive(Debug, Clone, Default)]
pub struct PlannedStats {
    pub slabs: usize,
    /// Tile voxels read from the store.
    pub voxels_read: u64,
    /// Seconds spent fusing and writing (the reads overlap them).
    pub fuse_seconds: f64,
}

/// The chunk shape stored for `level` of the OME-Zarr image at `root`, its
/// last three axes (the spatial ones, by the NGFF axis order).
pub fn stored_chunk(root: &std::path::Path, level_path: &str) -> [usize; 3] {
    let read = || -> Option<[usize; 3]> {
        let bytes = std::fs::read(root.join(level_path).join("zarr.json")).ok()?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let shape = json
            .pointer("/chunk_grid/configuration/chunk_shape")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize))
            .collect::<Option<Vec<_>>>()?;
        let n = shape.len();
        (n >= 3).then(|| [shape[n - 3], shape[n - 2], shape[n - 1]])
    };
    read().unwrap_or([1, 1, 1])
}

/// One tile's buffer: tile coordinates `start..start + data.shape()`.
struct Buffer<T> {
    start: [usize; 3],
    data: Array3<T>,
}

impl<T> Buffer<T> {
    fn stop(&self, axis: usize) -> usize {
        self.start[axis] + self.data.len_of(Axis(axis))
    }
}

type Buffers<T> = Vec<Option<Arc<Buffer<T>>>>;

/// The buffers for a slab whose needs are `needs`, from the previous slab's
/// `held`: kept where they cover, extended or re-read where they do not.
fn prepare<T: PyramidElement>(
    held: &Buffers<T>,
    needs: &[Option<([usize; 3], [usize; 3])>],
    whole: &[Option<([usize; 3], [usize; 3])>],
    sources: &[TileSource],
    axis: usize,
) -> Result<(Buffers<T>, u64)> {
    let results = (0..sources.len())
        .into_par_iter()
        .map(|tile| -> Result<(Option<Arc<Buffer<T>>>, u64)> {
            let held = held.get(tile).cloned().flatten();
            let Some((need_start, need_stop)) = needs[tile] else {
                // Not needed here; kept for a later slab.
                return Ok((held, 0));
            };
            let (whole_start, whole_stop) = whole[tile].expect("a slab's need is in the region's");
            let (a, b) = (need_start[axis], need_stop[axis]);
            if let Some(buffer) = &held {
                if buffer.start[axis] <= a && b <= buffer.stop(axis) {
                    return Ok((Some(buffer.clone()), 0));
                }
            }
            // Extend the read to the tile's next stored chunk boundary.
            let c = sources[tile].chunk[axis].max(1);
            let offset = sources[tile].offset[axis];
            let aligned = (b + offset).div_ceil(c) * c - offset;
            let stop = aligned.min(whole_stop[axis]).max(b);
            let keep = held
                .as_ref()
                .filter(|buffer| buffer.start[axis] <= a && a < buffer.stop(axis));
            let read_from = keep.map_or(a, |buffer| buffer.stop(axis));
            let mut start = whole_start;
            let mut shape: [usize; 3] = std::array::from_fn(|d| whole_stop[d] - whole_start[d]);
            start[axis] = read_from;
            shape[axis] = stop - read_from;
            let file_start: [usize; 3] =
                std::array::from_fn(|d| start[d] + sources[tile].offset[d]);
            let read = sources[tile].image.read_region(file_start, shape)?;
            let read = read
                .view::<T>()
                .map_err(|_| Error::invalid(format!("tile {tile} changed element type")))?
                .to_owned();
            let voxels = read.len() as u64;
            let data = match keep {
                None => read,
                Some(buffer) => {
                    let from = a - buffer.start[axis];
                    let kept = buffer
                        .data
                        .slice_axis(Axis(axis), ndarray::Slice::from(from..));
                    ndarray::concatenate(Axis(axis), &[kept, read.view()])
                        .map_err(Error::backend)?
                }
            };
            // Either way the buffer now starts at the slab's first needed row.
            let mut begin = whole_start;
            begin[axis] = a;
            Ok((Some(Arc::new(Buffer { start: begin, data })), voxels))
        })
        .collect::<Result<Vec<_>>>()?;
    let read = results.iter().map(|r| r.1).sum();
    Ok((results.into_iter().map(|r| r.0).collect(), read))
}

/// Fuse `region` (fused coordinates) of `placement`, reading tile `i` from
/// `sources[i]`, and hand the result to `sink` in slabs of `rows` rows along
/// `axis`, in order. `chunk` is the output's chunk shape (region-local):
/// slabs are fused chunk by chunk in parallel.
#[allow(clippy::too_many_arguments)]
pub fn fuse_planned<T: PyramidElement>(
    params: &FuseParams,
    placement: &Placement,
    sources: &[TileSource],
    region: &Region,
    axis: usize,
    rows: usize,
    chunk: [usize; 3],
    mut sink: impl FnMut(Array3<T>) -> Result<()> + Send,
) -> Result<PlannedStats> {
    if sources.len() != placement.sources().len() {
        return Err(Error::ShapeMismatch {
            expected: vec![placement.sources().len()],
            got: vec![sources.len()],
        });
    }
    region.check_within(&placement.shape()?, "fused region")?;
    let rows = rows.max(1);
    let whole = needed_box(placement, region);
    let extent = region.shape[axis];
    let slab_region = |index: usize| {
        let lo = index * rows;
        let hi = ((index + 1) * rows).min(extent);
        let mut start = [region.start[0], region.start[1], region.start[2]];
        let mut shape = [region.shape[0], region.shape[1], region.shape[2]];
        start[axis] += lo;
        shape[axis] = hi - lo;
        Region::new(&start, &shape)
    };
    let slabs = extent.div_ceil(rows);
    let mut stats = PlannedStats {
        slabs,
        ..PlannedStats::default()
    };
    let (mut current, read) = prepare::<T>(
        &vec![None; sources.len()],
        &needed_box(placement, &slab_region(0)),
        &whole,
        sources,
        axis,
    )?;
    stats.voxels_read += read;
    for index in 0..slabs {
        let slab = slab_region(index);
        let (fused, next) = rayon::join(
            || -> Result<f64> {
                let started = Instant::now();
                let out = fuse_slab(params, placement, &current, &slab, region, chunk)?;
                sink(out)?;
                Ok(started.elapsed().as_secs_f64())
            },
            || -> Result<Option<(Buffers<T>, u64)>> {
                if index + 1 == slabs {
                    return Ok(None);
                }
                let needs = needed_box(placement, &slab_region(index + 1));
                prepare(&current, &needs, &whole, sources, axis).map(Some)
            },
        );
        stats.fuse_seconds += fused?;
        if let Some((next, read)) = next? {
            current = next;
            stats.voxels_read += read;
        }
    }
    Ok(stats)
}

/// One slab, its output chunks (`chunk`, on the grid of `region`) fused in
/// parallel.
fn fuse_slab<T: PyramidElement>(
    params: &FuseParams,
    placement: &Placement,
    buffers: &Buffers<T>,
    slab: &Region,
    region: &Region,
    chunk: [usize; 3],
) -> Result<Array3<T>> {
    let mut ranges: [Vec<(usize, usize)>; 3] = Default::default();
    for (a, ranges) in ranges.iter_mut().enumerate() {
        // Region-local, on the region's chunk grid.
        let lo = slab.start[a] - region.start[a];
        let hi = lo + slab.shape[a];
        let c = chunk[a].max(1);
        let mut at = lo;
        while at < hi {
            let next = ((at / c + 1) * c).min(hi);
            ranges.push((at, next));
            at = next;
        }
    }
    let mut boxes = Vec::new();
    for &z in &ranges[0] {
        for &y in &ranges[1] {
            for &x in &ranges[2] {
                boxes.push([z, y, x]);
            }
        }
    }
    let sample = |tile: usize, at: [usize; 3]| -> f64 {
        let buffer = buffers[tile]
            .as_ref()
            .expect("a tile the plan reads is held");
        buffer.data[[
            at[0] - buffer.start[0],
            at[1] - buffer.start[1],
            at[2] - buffer.start[2],
        ]]
        .into_f64()
    };
    let pieces = boxes
        .into_par_iter()
        .map(|b| {
            let start: [usize; 3] = std::array::from_fn(|a| region.start[a] + b[a].0);
            let shape: [usize; 3] = std::array::from_fn(|a| b[a].1 - b[a].0);
            let piece: Array3<T> =
                fuse_with(params, sample, placement, &Region::new(&start, &shape));
            (b, piece)
        })
        .collect::<Vec<_>>();
    let mut out = Array3::from_elem(
        (slab.shape[0], slab.shape[1], slab.shape[2]),
        T::from_f64(0.0),
    );
    let base: [usize; 3] = std::array::from_fn(|a| slab.start[a] - region.start[a]);
    for (b, piece) in pieces {
        out.slice_mut(s![
            b[0].0 - base[0]..b[0].1 - base[0],
            b[1].0 - base[1]..b[1].1 - base[1],
            b[2].0 - base[2]..b[2].1 - base[2]
        ])
        .assign(&piece);
    }
    Ok(out)
}
