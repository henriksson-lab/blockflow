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
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis};
use rayon::prelude::*;

use crate::fuse::{fuse_into, needed_box, FuseParams, Placement};
use crate::pyramid::PyramidElement;
use crate::tiles::{StoredTiles, TileBox, TileBoxes};

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

/// One tile's buffer: tile coordinates `start..start + data.shape()`. Held as
/// the box read (or joined), so a read is never copied to be kept; for a
/// resident tile, a view.
struct Buffer<'a, T> {
    start: [usize; 3],
    shape: [usize; 3],
    data: TileBox<'a, T>,
}

impl<T: PyramidElement> Buffer<'_, T> {
    fn view(&self) -> ArrayView3<'_, T> {
        self.data.view()
    }

    fn stop(&self, axis: usize) -> usize {
        self.start[axis] + self.shape[axis]
    }
}

type Buffers<'a, T> = Vec<Option<Arc<Buffer<'a, T>>>>;

/// The buffers for a slab whose needs are `needs`, from the previous slab's
/// `held`: kept where they cover, extended or re-read where they do not.
/// Returns them and the stored voxels read.
#[allow(clippy::too_many_arguments)]
fn prepare<'a, T: PyramidElement, B: TileBoxes<T> + ?Sized>(
    held: &Buffers<'a, T>,
    needs: &[Option<([usize; 3], [usize; 3])>],
    whole: &[Option<([usize; 3], [usize; 3])>],
    later: &[bool],
    tiles: &'a B,
    axis: usize,
    extend: bool,
) -> Result<(Buffers<'a, T>, u64)> {
    let results = (0..tiles.len())
        .into_par_iter()
        .map(|tile| -> Result<(Option<Arc<Buffer<'a, T>>>, u64)> {
            let held = held.get(tile).cloned().flatten();
            let Some((need_start, need_stop)) = needs[tile] else {
                // Not needed here: kept for a later slab that needs it, and
                // dropped once none does (a tile the sweep has passed would
                // otherwise be held to the end).
                return Ok((held.filter(|_| later[tile]), 0));
            };
            let (whole_start, whole_stop) = whole[tile].expect("a slab's need is in the region's");
            let (a, b) = (need_start[axis], need_stop[axis]);
            if let Some(buffer) = &held {
                if buffer.start[axis] <= a && b <= buffer.stop(axis) {
                    return Ok((Some(buffer.clone()), 0));
                }
            }
            let mut start = whole_start;
            let mut shape: [usize; 3] = std::array::from_fn(|d| whole_stop[d] - whole_start[d]);
            let Some((chunk, offset)) = tiles.chunking(tile) else {
                // A resident tile: a view of the rows needed, nothing read.
                start[axis] = a;
                shape[axis] = b - a;
                let data = tiles.read(tile, start, shape)?;
                return Ok((Some(Arc::new(Buffer { start, shape, data })), 0));
            };
            // Extend the read to the tile's next stored chunk boundary (unless
            // the memory cap has no room for the extra rows).
            let c = if extend { chunk[axis].max(1) } else { 1 };
            let offset = offset[axis];
            let aligned = (b + offset).div_ceil(c) * c - offset;
            let stop = aligned.min(whole_stop[axis]).max(b);
            let keep = held
                .as_ref()
                .filter(|buffer| buffer.start[axis] <= a && a < buffer.stop(axis));
            let read_from = keep.map_or(a, |buffer| buffer.stop(axis));
            start[axis] = read_from;
            shape[axis] = stop - read_from;
            let read = tiles.read(tile, start, shape)?;
            let voxels = shape.iter().product::<usize>() as u64;
            let data = match keep {
                None => read,
                Some(buffer) => {
                    let from = a - buffer.start[axis];
                    let held = buffer.view();
                    let kept = held.slice_axis(Axis(axis), ndarray::Slice::from(from..));
                    TileBox::Read(T::wrap(
                        ndarray::concatenate(Axis(axis), &[kept, read.view()])
                            .map_err(Error::backend)?,
                    ))
                }
            };
            // Either way the buffer now starts at the slab's first needed row.
            let mut begin = whole_start;
            begin[axis] = a;
            let mut extent = shape;
            extent[axis] = stop - a;
            Ok((
                Some(Arc::new(Buffer {
                    start: begin,
                    shape: extent,
                    data,
                })),
                voxels,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let read = results.iter().map(|r| r.1).sum();
    Ok((results.into_iter().map(|r| r.0).collect(), read))
}

/// The tiles of `sources` as boxes to read, each of the extent `placement`
/// gives it.
pub(crate) fn stored<T: PyramidElement>(
    placement: &Placement,
    sources: &[TileSource],
) -> StoredTiles {
    StoredTiles {
        sources: sources.to_vec(),
        shapes: placement.sources().iter().map(|s| s.shape()).collect(),
        dtype: T::DTYPE,
    }
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
    sink: impl FnMut(Array3<T>) -> Result<()> + Send,
) -> Result<PlannedStats> {
    fuse_planned_reading(
        params, placement, sources, region, axis, rows, chunk, true, sink,
    )
}

/// [`fuse_planned`], saying whether a tile read is `extend`ed to the tile's
/// next stored chunk boundary (each stored chunk decoded once, at the cost of
/// holding up to a chunk's rows more of every tile).
#[allow(clippy::too_many_arguments)]
pub fn fuse_planned_reading<T: PyramidElement>(
    params: &FuseParams,
    placement: &Placement,
    sources: &[TileSource],
    region: &Region,
    axis: usize,
    rows: usize,
    chunk: [usize; 3],
    extend: bool,
    mut sink: impl FnMut(Array3<T>) -> Result<()> + Send,
) -> Result<PlannedStats> {
    if sources.len() != placement.sources().len() {
        return Err(Error::ShapeMismatch {
            expected: vec![placement.sources().len()],
            got: vec![sources.len()],
        });
    }
    fuse_planned_boxes(
        params,
        placement,
        &stored::<T>(placement, sources),
        region,
        axis,
        rows,
        chunk,
        extend,
        |slab| sink(slab.to_owned()),
    )
}

/// [`fuse_planned_reading`] over any source of tile boxes: stored tiles
/// ([`StoredTiles`]) or resident ones (views, which are never copied). Each
/// slab is fused into one buffer, reused from slab to slab, and lent to
/// `sink`.
#[allow(clippy::too_many_arguments)]
pub fn fuse_planned_boxes<T: PyramidElement, B: TileBoxes<T> + ?Sized>(
    params: &FuseParams,
    placement: &Placement,
    tiles: &B,
    region: &Region,
    axis: usize,
    rows: usize,
    chunk: [usize; 3],
    extend: bool,
    mut sink: impl FnMut(ArrayView3<'_, T>) -> Result<()> + Send,
) -> Result<PlannedStats> {
    if tiles.len() != placement.sources().len() {
        return Err(Error::ShapeMismatch {
            expected: vec![placement.sources().len()],
            got: vec![tiles.len()],
        });
    }
    for (index, source) in placement.sources().iter().enumerate() {
        if tiles.shape(index) != source.shape() {
            return Err(Error::ShapeMismatch {
                expected: source.shape().to_vec(),
                got: tiles.shape(index).to_vec(),
            });
        }
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
    // The last slab each tile is needed by, so its buffer goes once the sweep
    // has passed it.
    let mut last_need = vec![None; tiles.len()];
    for index in 0..slabs {
        for (tile, need) in needed_box(placement, &slab_region(index))
            .iter()
            .enumerate()
        {
            if need.is_some() {
                last_need[tile] = Some(index);
            }
        }
    }
    let later = |index: usize| -> Vec<bool> {
        last_need
            .iter()
            .map(|last| last.is_some_and(|last| last > index))
            .collect()
    };
    let (mut current, read) = prepare::<T, B>(
        &vec![None; tiles.len()],
        &needed_box(placement, &slab_region(0)),
        &whole,
        &later(0),
        tiles,
        axis,
        extend,
    )?;
    stats.voxels_read += read;
    // The buffers are leased from the process budget while they are held, so
    // the writer's chunk encoding and any cache see the room they take.
    let budget = blockflow::budget::process_budget();
    let lease_for = |buffers: &Buffers<T>| {
        let bytes = buffers
            .iter()
            .flatten()
            .map(|buffer| match buffer.data {
                TileBox::View(_) => 0,
                TileBox::Read(_) => {
                    (buffer.shape.iter().product::<usize>() * std::mem::size_of::<T>()) as u64
                }
            })
            .sum::<u64>();
        budget
            .try_acquire_reclaiming(bytes)
            .unwrap_or_else(blockflow::budget::Lease::unbudgeted)
    };
    let mut lease = lease_for(&current);
    // One slab's worth of output, fused into slab after slab: a fresh buffer
    // per slab would be mapped and faulted in afresh each time.
    let mut out = {
        let mut shape = [region.shape[0], region.shape[1], region.shape[2]];
        shape[axis] = rows.min(extent);
        Array3::from_elem(shape, T::from_f64(0.0))
    };
    for index in 0..slabs {
        let slab = slab_region(index);
        let (fused, next) = rayon::join(
            || -> Result<f64> {
                let started = Instant::now();
                let mut part =
                    out.slice_axis_mut(Axis(axis), ndarray::Slice::from(0..slab.shape[axis]));
                fuse_slab(
                    params,
                    placement,
                    &current,
                    &slab,
                    region,
                    chunk,
                    part.view_mut(),
                );
                sink(part.view())?;
                Ok(started.elapsed().as_secs_f64())
            },
            || -> Result<Option<(Buffers<T>, u64)>> {
                if index + 1 == slabs {
                    return Ok(None);
                }
                let needs = needed_box(placement, &slab_region(index + 1));
                prepare(
                    &current,
                    &needs,
                    &whole,
                    &later(index + 1),
                    tiles,
                    axis,
                    extend,
                )
                .map(Some)
            },
        );
        stats.fuse_seconds += fused?;
        if let Some((next, read)) = next? {
            current = next;
            stats.voxels_read += read;
            lease = lease_for(&current);
        }
    }
    drop(lease);
    Ok(stats)
}

/// One slab, its output chunks (`chunk`, on the grid of `region`) fused in
/// parallel, each straight into its part of `out`.
fn fuse_slab<T: PyramidElement>(
    params: &FuseParams,
    placement: &Placement,
    buffers: &Buffers<T>,
    slab: &Region,
    region: &Region,
    chunk: [usize; 3],
    out: ArrayViewMut3<'_, T>,
) {
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
    let views: Vec<Option<([usize; 3], ArrayView3<'_, T>)>> = buffers
        .iter()
        .map(|buffer| buffer.as_ref().map(|buffer| (buffer.start, buffer.view())))
        .collect();
    let sample = |tile: usize, at: [usize; 3]| -> f64 {
        let (start, data) = views[tile].as_ref().expect("a tile the plan reads is held");
        data[[at[0] - start[0], at[1] - start[1], at[2] - start[2]]].into_f64()
    };
    // The slab cut into the boxes' parts, in the boxes' order.
    let mut parts = vec![out];
    for (a, ranges) in ranges.iter().enumerate() {
        let mut cut = Vec::with_capacity(parts.len() * ranges.len());
        for part in parts {
            let mut rest = part;
            for &(lo, hi) in &ranges[..ranges.len() - 1] {
                let (head, tail) = rest.split_at(Axis(a), hi - lo);
                cut.push(head);
                rest = tail;
            }
            cut.push(rest);
        }
        parts = cut;
    }
    boxes.into_par_iter().zip(parts).for_each(|(b, part)| {
        let start: [usize; 3] = std::array::from_fn(|a| region.start[a] + b[a].0);
        let shape: [usize; 3] = std::array::from_fn(|a| b[a].1 - b[a].0);
        fuse_into(
            params,
            sample,
            placement,
            &Region::new(&start, &shape),
            part,
        );
    });
}

/// How many rows along `axis` a slab of `region` fuses at once under the
/// process's memory cap, and whether tile reads extend to the tiles' stored
/// chunk boundaries. Whole output chunk rows: as few as give every worker an
/// output chunk to fuse (each slab holds every tile's rows for it, so a
/// thicker slab only costs memory), at most [`crate::pyramid::slab_rows`],
/// and fewer when a slab's tile buffers (the current and the next slab's, as
/// read and as kept, with the extension) and its fused rows (fused, and the
/// writer's copy) would hold more than half of what the cap leaves; the
/// extension is dropped when even one row of output chunks does not fit with
/// it. One row at least.
pub fn fused_slab_rows(
    placement: &Placement,
    region: &Region,
    sources: &[TileSource],
    axis: usize,
    chunk: [usize; 3],
    dtype: blockflow::Dtype,
) -> (usize, bool) {
    let shape = [region.shape[0], region.shape[1], region.shape[2]];
    let c = chunk[axis].max(1);
    let per_row: usize = (0..3)
        .filter(|&a| a != axis)
        .map(|a| shape[a].div_ceil(chunk[a].max(1)))
        .product();
    let busy = c * rayon::current_num_threads().div_ceil(per_row.max(1));
    let rows = crate::pyramid::slab_rows(shape, chunk, axis, dtype).min(busy);
    let element = dtype.size_of().max(1) as u64;
    let extent = shape[axis].max(1) as u64;
    let out_row = (shape.iter().product::<usize>() as u64 / extent) * element;
    let tile_row = needed_box(placement, region)
        .iter()
        .flatten()
        .map(|(start, stop)| (0..3).map(|a| (stop[a] - start[a]) as u64).product::<u64>())
        .sum::<u64>()
        / extent
        * element;
    let extension = sources
        .iter()
        .map(|source| source.chunk[axis])
        .max()
        .unwrap_or(1) as u64;
    let room = blockflow::budget::process_budget().total() / 2;
    // The fused rows twice; the tile rows of the current slab and the next
    // one's, each once as read and once as kept.
    let held = |rows: u64, extend: bool| {
        2 * out_row * rows + 4 * tile_row * (rows + if extend { extension } else { 0 })
    };
    let affordable = |extend: bool| {
        let fixed = held(0, extend);
        (room.saturating_sub(fixed) / (2 * out_row + 4 * tile_row).max(1)) as usize / c * c
    };
    let extend = affordable(true) >= c;
    let rows = rows.min(affordable(extend).max(c)).min(shape[axis].max(1));
    (rows, extend)
}

/// Fuse `region` of `placement` from `sources` straight into a multiscale
/// image at `root` (which must not exist) of the decimation `factors`: slab by
/// slab ([`fused_slab_rows`]), never holding the output whole. Byte for byte
/// the image [`crate::io::write_volume_at`] writes for the resident fusion of
/// the same region.
#[allow(clippy::too_many_arguments)]
pub fn fuse_to_pyramid(
    params: &FuseParams,
    placement: &Placement,
    sources: &[TileSource],
    dtype: blockflow::Dtype,
    region: &Region,
    root: &std::path::Path,
    factors: &[[usize; 3]],
    frame: blockflow::zarr_env::PhysicalFrame,
) -> Result<PlannedStats> {
    if root.exists() {
        return Err(Error::invalid(format!(
            "{} exists; remove it or choose another output",
            root.display()
        )));
    }
    let shape = [region.shape[0], region.shape[1], region.shape[2]];
    let axis = crate::pyramid::sweep_axis(shape, dtype);
    blockflow::dtype_dispatch!(
        dtype,
        |Element| {
            let mut writer = crate::pyramid::PyramidWriter::<Element>::create(
                root,
                shape,
                factors,
                axis,
                Some(frame),
            )?;
            let chunk = writer.chunk();
            let (rows, extend) = fused_slab_rows(placement, region, sources, axis, chunk, dtype);
            let stats = fuse_planned_boxes::<Element, _>(
                params,
                placement,
                &stored::<Element>(placement, sources),
                region,
                axis,
                rows,
                chunk,
                extend,
                |slab| writer.push_view(slab),
            )?;
            writer.finish()?;
            Ok(stats)
        },
        f16 => Err(Error::invalid("float16 tiles cannot be fused"))
    )
}
