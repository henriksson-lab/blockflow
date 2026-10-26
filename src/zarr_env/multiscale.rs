// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! A multiscale image written **slab by slab, chunks in parallel**: the
//! writer behind [`ZarrEnvironment::build_multiscale`], and public for a
//! caller whose image arrives in pieces (a fusion computed slab by slab).
//!
//! The image is handed over in slabs along one axis (the sweep axis), in
//! order. Each level keeps the rows it has been given but not yet written or
//! downsampled; as soon as a whole row of chunks is present it is encoded and
//! stored, chunks in parallel, and as soon as the windows of the next level's
//! rows are present those rows are computed (rows in parallel) and handed on.
//! So a level is never read back from the store, and no level is held whole
//! unless its own chunks span the whole sweep axis.
//!
//! **Bytes.** Every chunk is written once, whole, through the same
//! `StoredArray` the environment writes with, so the arrays are those the
//! earlier builder wrote (one chunk at a time, each level re-read from the
//! store to build the next): the same chunk shape per level, codec, fill
//! value, `floor(n / f)` level extents and window means summed in the same
//! order (or the window's first voxel for `bool`).
//!
//! **Memory.** What the writer holds is the rows a level has not yet written
//! or downsampled — less than one chunk row of each level, plus a downsampling
//! window — and the chunks being encoded. A slab is copied to be held only
//! when the level holds rows it has not yet written; rows it holds only for
//! the next level's window (one per slab, when a slab's rows are odd) are
//! joined with the slab one window at a time. A resident volume
//! ([`MultiscaleWriter::write_volume`]) is stored from where it lies, and its
//! level 1 computed from it in batches of whole level-1 chunk rows, so that
//! level never gathers rows towards a chunk row. Those lease from the process budget
//! ([`crate::budget::process_budget`]): as many chunks are encoded at once as
//! the cap leaves room for, and always one. The slabs are the caller's;
//! [`MultiscaleWriter::slab_rows`] sizes them under the cap.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{s, Array3, ArrayView3, Axis, CowArray, Ix3};
use rayon::prelude::*;
use zarrs::array::Element;
use zarrs::filesystem::FilesystemStore;

use super::{
    chunk_for_extent, downsampled_shape, write_multiscales_metadata, zarr_data_type, AttachedImage,
    ChunkLocks, CompressionPolicy, MultiscaleImage, PhysicalFrame, PyramidSpec, StoredArray,
};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::voxels::VoxelElement;

/// What a multiscale image's elements must be to be written slab by slab.
pub trait PyramidElement: VoxelElement + Element + Send + Sync + Copy + 'static {}
impl<T: VoxelElement + Element + Send + Sync + Copy + 'static> PyramidElement for T {}

/// One level: its array and the rows it holds that are not yet written or not
/// yet downsampled.
struct Level<T> {
    array: StoredArray,
    shape: [usize; 3],
    chunk: [usize; 3],
    /// The factor to the next level, if there is one.
    factor: Option<[usize; 3]>,
    /// Rows `pending_start..pending_start + pending.len_of(sweep)`.
    pending_start: usize,
    pending: Option<Array3<T>>,
    /// Rows `0..written` are stored.
    written: usize,
    /// The next row of the next level to compute.
    next: usize,
}

/// A multiscale image written from slabs handed over in order along one axis.
pub struct MultiscaleWriter<T: PyramidElement> {
    root: PathBuf,
    axis: usize,
    levels: Vec<Level<T>>,
    scales: Vec<[usize; 3]>,
    physical: Option<PhysicalFrame>,
    locks: ChunkLocks,
}

/// The axis a volume of `shape` chunked as `chunk` is best handed over along:
/// the one its chunks cut most often (the first such axis on a tie), so a slab
/// of whole chunk rows is as thin as the chunking allows.
pub fn sweep_axis(shape: [usize; 3], chunk: [usize; 3]) -> usize {
    let count = |a: usize| shape[a].div_ceil(chunk[a].max(1));
    let mut best = 0;
    for a in 1..3 {
        if count(a) > count(best) {
            best = a;
        }
    }
    best
}

/// A level's chunk shape under `spec`: its stated chunk, else
/// [`chunk_for_extent`] of the level's shape.
pub fn level_chunk(spec: &PyramidSpec, shape: [usize; 3], dtype: Dtype) -> [usize; 3] {
    spec.chunk.unwrap_or_else(|| chunk_for_extent(shape, dtype))
}

impl<T: PyramidElement> MultiscaleWriter<T> {
    /// Create the arrays of a pyramid of `shape` under `root`, built by
    /// `spec`, to be handed over along `axis` (see [`sweep_axis`]).
    pub fn create(
        root: impl Into<PathBuf>,
        shape: [usize; 3],
        spec: &PyramidSpec,
        compression: &CompressionPolicy,
        axis: usize,
    ) -> Result<Self> {
        if axis >= 3 {
            return Err(Error::InvalidArgument(format!(
                "sweep axis {axis} of a volume"
            )));
        }
        let root = root.into();
        let dtype = T::DTYPE;
        std::fs::create_dir_all(&root).map_err(Error::backend)?;
        let store = Arc::new(FilesystemStore::new(&root).map_err(Error::backend)?);
        let mut levels = Vec::with_capacity(spec.factors.len() + 1);
        let mut scales = vec![[1usize; 3]];
        let mut level_shape = shape;
        for index in 0..=spec.factors.len() {
            if index > 0 {
                let factor = spec.factors[index - 1];
                level_shape = downsampled_shape(level_shape, factor, index)?;
                let mut cumulative = *scales.last().expect("level 0 exists");
                for a in 0..3 {
                    cumulative[a] *= factor[a];
                }
                scales.push(cumulative);
            }
            let chunk = level_chunk(spec, level_shape, dtype);
            let array = StoredArray::create(
                &store,
                &format!("/{index}"),
                index as u64,
                dtype,
                &level_shape,
                &chunk,
                compression.at(index, dtype),
            )?;
            levels.push(Level {
                array,
                shape: level_shape,
                chunk,
                factor: spec.factors.get(index).copied(),
                pending_start: 0,
                pending: None,
                written: 0,
                next: 0,
            });
        }
        Ok(Self {
            root,
            axis,
            levels,
            scales,
            physical: spec.physical.clone(),
            locks: ChunkLocks::new(),
        })
    }

    /// The sweep axis.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Level 0's chunk shape.
    pub fn chunk(&self) -> [usize; 3] {
        self.levels[0].chunk
    }

    /// Level 0's shape.
    pub fn shape(&self) -> [usize; 3] {
        self.levels[0].shape
    }

    /// How many rows along the sweep axis a slab handed to [`Self::push`]
    /// should hold: whole level-0 chunk rows, enough chunks to keep every
    /// worker busy twice over and about 256 MiB, but no more than an eighth of
    /// what the process's memory cap leaves (one chunk row at least), and at
    /// most the volume.
    pub fn slab_rows(&self) -> usize {
        let budget = crate::budget::process_budget().total();
        slab_rows_under(self.shape(), self.chunk(), self.axis, T::DTYPE, budget)
    }

    /// Hand over the next slab of level 0: rows from where the last one ended,
    /// the whole extent across the sweep axis.
    pub fn push(&mut self, slab: ArrayView3<'_, T>) -> Result<()> {
        self.push_level(0, CowArray::from(slab))
    }

    fn push_level(&mut self, index: usize, slab: CowArray<'_, T, Ix3>) -> Result<()> {
        let axis = self.axis;
        let level = &mut self.levels[index];
        let mut expected = level.shape;
        expected[axis] = slab.len_of(Axis(axis));
        if slab.shape() != expected {
            return Err(Error::ShapeMismatch {
                expected: expected.to_vec(),
                got: slab.shape().to_vec(),
            });
        }
        let held = level.pending.as_ref().map_or(0, |p| p.len_of(Axis(axis)));
        let end = level.pending_start + held;
        if end + expected[axis] > level.shape[axis] {
            return Err(Error::InvalidArgument(format!(
                "level {index} has {} rows along axis {axis} and was handed rows {end}..{}",
                level.shape[axis],
                end + expected[axis]
            )));
        }
        if let Some(coarser) = self.push_beside_held(index, &slab)? {
            drop(slab);
            return match coarser {
                Some(coarser) => self.push_level(index + 1, CowArray::from(coarser)),
                None => Ok(()),
            };
        }
        let level = &mut self.levels[index];
        // What is held, then the slab: borrowed when nothing is held, which is
        // the usual case for a slab of whole chunk rows.
        let rows: CowArray<'_, T, Ix3> = match level.pending.take() {
            None => {
                level.pending_start = end;
                slab
            }
            Some(held) => CowArray::from(
                ndarray::concatenate(Axis(axis), &[held.view(), slab.view()])
                    .map_err(Error::backend)?,
            ),
        };
        let end = end + expected[axis];

        // Every whole chunk row now present (or the last, short one).
        let c = level.chunk[axis];
        let mut stop = level.written;
        while stop < level.shape[axis] && (stop + c <= end || end == level.shape[axis]) {
            stop = (stop + c).min(level.shape[axis]);
        }
        if stop > level.written {
            let origin = level.pending_start;
            write_rows(
                level,
                rows.view(),
                origin,
                axis,
                level.written,
                stop,
                &self.locks,
            )?;
            level.written = stop;
        }

        // The next level's rows whose windows are now present.
        let mut coarser = None;
        if let Some(factor) = level.factor {
            let f = factor[axis];
            let next_shape = self.levels[index + 1].shape;
            let level = &mut self.levels[index];
            let ready = (end / f).min(next_shape[axis]);
            if ready > level.next {
                let first = level.next * f - level.pending_start;
                let window = rows.slice_axis(
                    Axis(axis),
                    ndarray::Slice::from(first..first + (ready - level.next) * f),
                );
                let mut shape = next_shape;
                shape[axis] = ready - level.next;
                coarser = Some(downsample(window, factor, shape));
                level.next = ready;
            }
        }

        // Keep only the rows something still needs.
        let level = &mut self.levels[index];
        let keep = match level.factor {
            Some(factor) => level.written.min(level.next * factor[axis]),
            None => level.written,
        };
        let from = keep.max(level.pending_start) - level.pending_start;
        if from < rows.len_of(Axis(axis)) {
            level.pending = Some(
                rows.slice_axis(Axis(axis), ndarray::Slice::from(from..))
                    .to_owned(),
            );
        }
        level.pending_start = level.pending_start.max(keep);
        drop(rows);

        if let Some(coarser) = coarser {
            self.push_level(index + 1, CowArray::from(coarser))?;
        }
        Ok(())
    }

    /// [`Self::push_level`] when level `index` holds only rows it has already
    /// written, kept for the next level's windows (a level whose rows along
    /// the sweep axis are not a multiple of the factor keeps one at each
    /// slab's end): the slab is written and downsampled where it lies, and
    /// only the one window across the held rows and the slab is joined,
    /// rather than the whole slab copied to follow them. `None` when that is
    /// not the case (nothing held, unwritten rows held, or a slab too thin to
    /// leave the held rows behind); otherwise the next level's new rows, if
    /// any.
    #[allow(clippy::type_complexity)]
    fn push_beside_held(
        &mut self,
        index: usize,
        slab: &CowArray<'_, T, Ix3>,
    ) -> Result<Option<Option<Array3<T>>>> {
        let axis = self.axis;
        let next_shape = self.levels.get(index + 1).map(|level| level.shape);
        let level = &mut self.levels[index];
        let Some(held) = level.pending.as_ref() else {
            return Ok(None);
        };
        let start = level.pending_start;
        let end = start + held.len_of(Axis(axis));
        if level.written != end {
            return Ok(None);
        }
        let n = slab.len_of(Axis(axis));
        let new_end = end + n;
        let c = level.chunk[axis];
        let mut stop = level.written;
        while stop < level.shape[axis] && (stop + c <= new_end || new_end == level.shape[axis]) {
            stop = (stop + c).min(level.shape[axis]);
        }
        let (ready, keep) = match (level.factor, next_shape) {
            (Some(factor), Some(next_shape)) => {
                let f = factor[axis];
                let ready = (new_end / f).min(next_shape[axis]).max(level.next);
                (ready, stop.min(ready * f))
            }
            _ => (level.next, stop),
        };
        // The held rows must all be left behind once this slab is in.
        if keep < end {
            return Ok(None);
        }
        let held = level.pending.take().expect("checked above");
        if stop > level.written {
            write_rows(
                level,
                slab.view(),
                end,
                axis,
                level.written,
                stop,
                &self.locks,
            )?;
            level.written = stop;
        }
        let mut coarser = None;
        if let (Some(factor), Some(next_shape)) = (level.factor, next_shape) {
            let f = factor[axis];
            let mut parts = Vec::new();
            let mut row = level.next;
            if row < ready && row * f < end {
                // The window across the held rows and the slab, joined.
                let joined = ndarray::concatenate(
                    Axis(axis),
                    &[
                        held.slice_axis(Axis(axis), ndarray::Slice::from(row * f - start..)),
                        slab.slice_axis(Axis(axis), ndarray::Slice::from(0..row * f + f - end)),
                    ],
                )
                .map_err(Error::backend)?;
                let mut shape = next_shape;
                shape[axis] = 1;
                parts.push(downsample(joined.view(), factor, shape));
                row += 1;
            }
            if row < ready {
                let window = slab.slice_axis(
                    Axis(axis),
                    ndarray::Slice::from(row * f - end..ready * f - end),
                );
                let mut shape = next_shape;
                shape[axis] = ready - row;
                parts.push(downsample(window, factor, shape));
            }
            coarser = match parts.len() {
                0 => None,
                1 => parts.pop(),
                _ => Some(
                    ndarray::concatenate(
                        Axis(axis),
                        &parts.iter().map(|part| part.view()).collect::<Vec<_>>(),
                    )
                    .map_err(Error::backend)?,
                ),
            };
            level.next = ready;
        }
        drop(held);
        // Keep only the rows something still needs: all of them the slab's.
        let from = keep - end;
        if from < n {
            level.pending = Some(
                slab.slice_axis(Axis(axis), ndarray::Slice::from(from..))
                    .to_owned(),
            );
        }
        level.pending_start = keep;
        Ok(Some(coarser))
    }

    /// Check every level is complete, write the group's metadata, and return
    /// the image.
    pub fn finish(self) -> Result<MultiscaleImage> {
        for (index, level) in self.levels.iter().enumerate() {
            if level.written != level.shape[self.axis] {
                return Err(Error::InvalidArgument(format!(
                    "pyramid level {index} has {} of its {} rows along axis {}",
                    level.written, level.shape[self.axis], self.axis
                )));
            }
        }
        write_multiscales_metadata(&self.root, &self.scales, self.physical.as_ref(), T::DTYPE)?;
        let levels = (0..self.levels.len())
            .map(|index| AttachedImage::at(self.root.join(index.to_string())))
            .collect();
        MultiscaleImage::new(levels, self.scales)
    }

    /// Write all of `volume` (level 0) through this writer. Level 0 is
    /// stored straight from `volume`, and level 1 computed from `volume` in
    /// batches of whole level-1 chunk rows ([`batch_rows_under`]): the volume is resident, so level 0 holds
    /// nothing back for level 1, and level 1 holds no more than one batch
    /// (where slabs handed over one at a time would leave it gathering rows
    /// towards a chunk row, and copying them to join the next).
    pub fn write_volume(mut self, volume: ArrayView3<'_, T>) -> Result<MultiscaleImage> {
        let axis = self.axis;
        let shape = self.shape();
        if volume.shape() != shape {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: volume.shape().to_vec(),
            });
        }
        // Level 0 in one call: its rows are all present, and the store
        // encodes their chunks in parallel itself.
        let extent = shape[axis];
        write_rows(&self.levels[0], volume, 0, axis, 0, extent, &self.locks)?;
        self.levels[0].written = extent;
        if let Some(factor) = self.levels[0].factor {
            let level = &self.levels[1];
            let (next_shape, f) = (level.shape, factor[axis]);
            let batch = batch_rows_under(
                next_shape,
                level.chunk,
                axis,
                T::DTYPE,
                crate::budget::process_budget().total(),
            );
            let mut row = 0;
            while row < next_shape[axis] {
                let stop = (row + batch).min(next_shape[axis]);
                let window = volume.slice_axis(Axis(axis), ndarray::Slice::from(row * f..stop * f));
                let mut rows = next_shape;
                rows[axis] = stop - row;
                let coarser = downsample(window, factor, rows);
                self.push_level(1, CowArray::from(coarser))?;
                row = stop;
            }
            self.levels[0].next = next_shape[axis];
        }
        self.finish()
    }
}

/// How many rows along `axis` a level of `shape` is computed in at once when
/// its source is resident: whole chunk rows, as few as give every worker a
/// chunk to encode, and no more than [`slab_rows_under`] (nor the level).
pub(crate) fn batch_rows_under(
    shape: [usize; 3],
    chunk: [usize; 3],
    axis: usize,
    dtype: Dtype,
    budget: u64,
) -> usize {
    let c = chunk[axis].max(1);
    let chunks_per_row: usize = (0..3)
        .filter(|&a| a != axis)
        .map(|a| shape[a].div_ceil(chunk[a].max(1)))
        .product();
    let busy = c * rayon::current_num_threads().div_ceil(chunks_per_row.max(1));
    busy.min(slab_rows_under(shape, chunk, axis, dtype, budget))
        .max(1)
        .min(shape[axis].max(1))
}

/// [`MultiscaleWriter::slab_rows`] for a cap leaving `budget` bytes.
pub fn slab_rows_under(
    shape: [usize; 3],
    chunk: [usize; 3],
    axis: usize,
    dtype: Dtype,
    budget: u64,
) -> usize {
    let c = chunk[axis].max(1);
    let row_bytes = shape
        .iter()
        .enumerate()
        .filter(|&(a, _)| a != axis)
        .map(|(_, &n)| n)
        .product::<usize>()
        * dtype.size_of().max(1);
    let chunks_per_row: usize = (0..3)
        .filter(|&a| a != axis)
        .map(|a| shape[a].div_ceil(chunk[a].max(1)))
        .product();
    let workers = rayon::current_num_threads();
    let target = (256usize << 20).min(usize::try_from(budget / 8).unwrap_or(usize::MAX));
    let by_bytes = target.div_ceil(row_bytes.max(1) * c).max(1);
    // Enough chunks for the workers, but never more than twice the byte
    // target: with large chunks a row of them is already most of it.
    let by_workers = (2 * workers)
        .div_ceil(chunks_per_row.max(1))
        .min(2 * by_bytes);
    let affordable = usize::try_from(budget / 8).unwrap_or(usize::MAX) / (row_bytes.max(1) * c);
    let rows = by_bytes.max(by_workers).min(affordable).max(1) * c;
    rows.min(shape[axis])
}

/// Store the chunks of rows `start..stop` (whole chunk rows) of `level` from
/// `rows`, which holds rows from `origin`, as many at once as the process
/// budget has room for.
fn write_rows<T: PyramidElement>(
    level: &Level<T>,
    rows: ArrayView3<'_, T>,
    origin: usize,
    axis: usize,
    start: usize,
    stop: usize,
    locks: &ChunkLocks,
) -> Result<()> {
    let chunk = level.chunk;
    let shape = level.shape;
    // Rows along the first axis of a standard-layout buffer are one
    // contiguous run: stored with one call, which splits and encodes the
    // chunks in parallel itself, with no copy of each chunk first.
    if axis == 0 {
        let local = rows.slice_axis(Axis(0), ndarray::Slice::from(start - origin..stop - origin));
        if let Some(data) = local.as_slice() {
            let region = Region::new(&[start, 0, 0], &[stop - start, shape[1], shape[2]]);
            level.array.write_as::<T>(&region, data, Some(locks))?;
            return Ok(());
        }
    }
    let mut ranges: [Vec<(usize, usize)>; 3] = Default::default();
    for a in 0..3 {
        let (lo, hi) = if a == axis {
            (start, stop)
        } else {
            (0, shape[a])
        };
        let mut at = lo;
        while at < hi {
            let next = ((at / chunk[a] + 1) * chunk[a]).min(hi);
            ranges[a].push((at, next));
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
    // A chunk being encoded holds its elements, the encoded bytes and the
    // codec's scratch: three chunks' worth, leased per chunk in flight.
    let per_chunk = (chunk.iter().product::<usize>() * T::DTYPE.size_of().max(1)) as u64 * 3;
    let workers = rayon::current_num_threads().min(boxes.len()).max(1);
    let budget = crate::budget::process_budget();
    let mut parallel = workers;
    let lease = loop {
        if let Some(granted) = budget.try_acquire_reclaiming(per_chunk * parallel as u64) {
            break granted;
        }
        if parallel == 1 {
            // No room at all: one chunk at a time, which is what it costs.
            break crate::budget::Lease::unbudgeted();
        }
        parallel = parallel.div_ceil(2);
    };
    let group = boxes.len().div_ceil(parallel.max(1));
    let outcome = boxes.par_chunks(group.max(1)).try_for_each(|group| {
        for b in group {
            let mut local = *b;
            local[axis] = (b[axis].0 - origin, b[axis].1 - origin);
            let piece = rows.slice(s![
                local[0].0..local[0].1,
                local[1].0..local[1].1,
                local[2].0..local[2].1
            ]);
            let data = piece.iter().copied().collect::<Vec<T>>();
            let region = Region::new(
                &[b[0].0, b[1].0, b[2].0],
                &[b[0].1 - b[0].0, b[1].1 - b[1].0, b[2].1 - b[2].0],
            );
            level.array.write_as::<T>(&region, &data, Some(locks))?;
        }
        Ok::<(), Error>(())
    });
    drop(lease);
    outcome
}

/// The block means of `input` (or first voxels, for `bool`), `shape` of them,
/// summed in the order the builder always summed them (the first axis
/// innermost).
fn downsample<T: PyramidElement>(
    input: ArrayView3<'_, T>,
    factor: [usize; 3],
    shape: [usize; 3],
) -> Array3<T> {
    let mut out = Array3::from_elem(shape, T::from_f64(0.0));
    let [f0, f1, f2] = factor;
    let count = (f0 * f1 * f2) as f64;
    // Planes of the first axis in parallel; each voxel is computed alone.
    // Within a row, the input lanes along the last axis the row reads are
    // taken once (`lanes[dy * f0 + dx]`), and each voxel sums its window from
    // them in the builder's order.
    let plane = shape[1] * shape[2];
    out.as_slice_mut()
        .expect("a fresh array is contiguous")
        .par_chunks_mut(plane.max(1))
        .enumerate()
        .for_each(|(x, out_plane)| {
            let mut lanes = Vec::with_capacity(f0 * f1);
            for (y, out_row) in out_plane.chunks_mut(shape[2].max(1)).enumerate() {
                if T::DTYPE == Dtype::Bool {
                    let lane = input.slice(s![x * f0, y * f1, ..]);
                    for (z, value) in out_row.iter_mut().enumerate() {
                        *value = lane[z * f2];
                    }
                    continue;
                }
                lanes.clear();
                for dy in 0..f1 {
                    for dx in 0..f0 {
                        lanes.push(input.slice(s![x * f0 + dx, y * f1 + dy, ..]));
                    }
                }
                for (z, value) in out_row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for dz in 0..f2 {
                        let at = z * f2 + dz;
                        for lane in &lanes {
                            sum += lane[at].into_f64();
                        }
                    }
                    *value = T::from_f64(sum / count);
                }
            }
        });
    out
}

/// Write a resident volume as a multiscale image at `root`: the body of
/// [`super::ZarrEnvironment::build_multiscale_with_compression`].
pub(super) fn write_resident(
    root: &Path,
    input: &crate::voxels::Voxels,
    spec: &PyramidSpec,
    compression: &CompressionPolicy,
) -> Result<MultiscaleImage> {
    let shape = input.shape();
    // Along the first axis: the volume is resident, so a slab is a view and
    // costs nothing to hold, and its rows are one contiguous run that is
    // stored without copying a chunk at a time.
    let axis = 0;
    by_dtype!(input.dtype(), |Element| {
        let view = input.view::<Element>()?;
        MultiscaleWriter::<Element>::create(root, shape, spec, compression, axis)?
            .write_volume(view)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxels::Voxels;
    use crate::zarr_env::ZarrEnvironment;

    fn scratch(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "blockflow-multiscale-{}-{name}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// Every file under `root`, relative path and bytes, sorted.
    fn files(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let bytes = std::fs::read(&path).unwrap();
                    out.push((path.strip_prefix(root).unwrap().to_path_buf(), bytes));
                }
            }
        }
        out.sort();
        out
    }

    fn volume(dtype: Dtype, shape: [usize; 3]) -> Voxels {
        let value = |(z, y, x): (usize, usize, usize)| ((z * 131 + y * 17 + x * 7) % 251) as f64;
        let dims = (shape[0], shape[1], shape[2]);
        match dtype {
            Dtype::U8 => Voxels::from(Array3::from_shape_fn(dims, |at| value(at) as u8)),
            Dtype::U16 => Voxels::from(Array3::from_shape_fn(dims, |at| value(at) as u16 * 3)),
            Dtype::F32 => Voxels::from(Array3::from_shape_fn(dims, |at| value(at) as f32 / 7.0)),
            Dtype::Bool => Voxels::from(Array3::from_shape_fn(dims, |at| value(at) > 120.0)),
            other => panic!("no fixture for {other:?}"),
        }
    }

    fn same(what: &str, a: &Path, b: &Path) {
        let (a, b) = (files(a), files(b));
        assert_eq!(
            a.iter().map(|f| &f.0).collect::<Vec<_>>(),
            b.iter().map(|f| &f.0).collect::<Vec<_>>(),
            "{what}: different files"
        );
        for (x, y) in a.iter().zip(&b) {
            assert!(x.1 == y.1, "{what}: {} differs", x.0.display());
        }
    }

    #[test]
    fn the_slab_writer_stores_the_bytes_the_chunk_by_chunk_builder_stored() {
        let frame = PhysicalFrame::new([2.0, 0.5, 0.5], Some("micrometer"));
        let cases: Vec<(Dtype, [usize; 3], PyramidSpec)> = vec![
            (
                Dtype::U16,
                [37, 61, 53],
                PyramidSpec::powers_of_two(3).unwrap(),
            ),
            (
                Dtype::F32,
                [40, 64, 48],
                PyramidSpec::new(vec![[1, 2, 2], [2, 2, 2]])
                    .unwrap()
                    .with_physical(frame.clone()),
            ),
            (
                Dtype::U8,
                [33, 70, 45],
                PyramidSpec::powers_of_two(3)
                    .unwrap()
                    .with_chunk([8, 16, 16]),
            ),
            (
                Dtype::Bool,
                [24, 50, 50],
                PyramidSpec::new(vec![[2, 3, 2]]).unwrap(),
            ),
            (
                Dtype::U16,
                [9, 300, 400],
                PyramidSpec::powers_of_two(4).unwrap(),
            ),
            // Odd chunk rows: a slab of whole chunk rows leaves a row behind
            // for the next level's window.
            (
                Dtype::U16,
                [45, 39, 33],
                PyramidSpec::powers_of_two(3)
                    .unwrap()
                    .with_chunk([9, 13, 11]),
            ),
        ];
        for (index, (dtype, shape, spec)) in cases.into_iter().enumerate() {
            let input = volume(dtype, shape);
            let reference = scratch(&format!("reference-{index}"));
            ZarrEnvironment::build_multiscale_reference(
                &reference,
                &input,
                &spec,
                CompressionPolicy::derived(),
            )
            .unwrap();
            let built = scratch(&format!("built-{index}"));
            ZarrEnvironment::build_multiscale(&built, &input, &spec).unwrap();
            same(&format!("case {index}"), &reference, &built);
            // Slabs of any size, on any axis, give the same bytes.
            for axis in 0..3 {
                let chunk = level_chunk(&spec, shape, dtype)[axis];
                for step in [1usize, 5, 16, chunk, 3 * chunk] {
                    let pushed = scratch(&format!("pushed-{index}-{axis}-{step}"));
                    crate::dtype_dispatch!(dtype, |Element| {
                        let view = input.view::<Element>().unwrap();
                        let mut writer = MultiscaleWriter::<Element>::create(
                            &pushed,
                            shape,
                            &spec,
                            &CompressionPolicy::derived(),
                            axis,
                        )
                        .unwrap();
                        let mut at = 0;
                        while at < shape[axis] {
                            let stop = (at + step).min(shape[axis]);
                            writer
                                .push(view.slice_axis(Axis(axis), ndarray::Slice::from(at..stop)))
                                .unwrap();
                            at = stop;
                        }
                        writer.finish().unwrap();
                        Ok::<(), Error>(())
                    }, f16 => unreachable!("no float16 fixture"))
                    .unwrap();
                    same(
                        &format!("case {index}, axis {axis}, step {step}"),
                        &reference,
                        &pushed,
                    );
                    std::fs::remove_dir_all(&pushed).unwrap();
                }
            }
            std::fs::remove_dir_all(&reference).unwrap();
            std::fs::remove_dir_all(&built).unwrap();
        }
    }

    #[test]
    fn a_writer_refuses_rows_past_the_end_and_an_unfinished_level() {
        let shape = [4, 8, 8];
        let spec = PyramidSpec::powers_of_two(2).unwrap();
        let root = scratch("refuses");
        let input = Array3::<u16>::zeros((4, 8, 8));
        let mut writer =
            MultiscaleWriter::<u16>::create(&root, shape, &spec, &CompressionPolicy::derived(), 0)
                .unwrap();
        writer.push(input.slice(s![0..2, .., ..])).unwrap();
        assert!(writer.push(input.slice(s![0..3, .., ..])).is_err());
        let mut writer =
            MultiscaleWriter::<u16>::create(&root, shape, &spec, &CompressionPolicy::derived(), 0)
                .unwrap();
        writer.push(input.slice(s![0..2, .., ..])).unwrap();
        assert!(writer.finish().is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_slab_is_whole_chunk_rows_and_shrinks_under_a_cap() {
        let shape = [1000, 2000, 2000];
        let chunk = [64, 64, 64];
        let roomy = slab_rows_under(shape, chunk, 0, Dtype::U16, 64 << 30);
        assert_eq!(roomy % 64, 0);
        let tight = slab_rows_under(shape, chunk, 0, Dtype::U16, 512 << 20);
        assert!(tight <= roomy && tight % 64 == 0 && tight >= 64);
        // One chunk row when even that is over the cap's share.
        assert_eq!(slab_rows_under(shape, chunk, 0, Dtype::U16, 1 << 20), 64);
    }
}

/// Timing: `BLOCKFLOW_PYRAMID_TIMING=reference|writer` and
/// `BLOCKFLOW_PYRAMID_CASE=atlas|density|fused` pick the builder and the
/// volume; run each in its own process (under `/usr/bin/time -v`) to compare
/// wall, CPU and peak. Needs about 4 GB of memory and of scratch space under
/// `$BLOCKFLOW_PYRAMID_DIR` (default the temporary directory).
#[cfg(test)]
mod timing {
    use super::*;
    use crate::voxels::Voxels;
    use crate::zarr_env::ZarrEnvironment;

    #[test]
    #[ignore = "timing; see the module comment"]
    fn time_a_multiscale_build() {
        let which = std::env::var("BLOCKFLOW_PYRAMID_TIMING").unwrap_or_else(|_| "writer".into());
        let case = std::env::var("BLOCKFLOW_PYRAMID_CASE").unwrap_or_else(|_| "atlas".into());
        // An atlas-space f32 volume as atlas registration writes it (five
        // levels), a single-level density map with 128-voxel chunks as
        // cell-density and the vessel graph write it, and a fused u16 volume.
        let (input, spec) = match case.as_str() {
            "atlas" => (
                Voxels::from(Array3::from_shape_fn((320, 456, 528), |(z, y, x)| {
                    ((z * 7 + y * 3 + x) % 1000) as f32 * 0.25
                })),
                PyramidSpec::powers_of_two(5).unwrap(),
            ),
            "density" => (
                Voxels::from(Array3::from_shape_fn((400, 1200, 900), |(z, y, x)| {
                    ((z * 7 + y * 3 + x) % 1000) as f32 * 0.25
                })),
                PyramidSpec::new(Vec::new())
                    .unwrap()
                    .with_chunk([128, 128, 128]),
            ),
            _ => (
                Voxels::from(Array3::from_shape_fn((521, 1810, 1141), |(z, y, x)| {
                    ((z * 7 + y * 3 + x) % 4000) as u16
                })),
                PyramidSpec::new(vec![[2, 2, 2], [2, 2, 2], [2, 2, 2]]).unwrap(),
            ),
        };
        let dir = std::env::var_os("BLOCKFLOW_PYRAMID_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(format!(
                "blockflow-pyramid-timing-{which}-{case}-{}",
                std::process::id()
            ));
        let _ = std::fs::remove_dir_all(&dir);
        let started = std::time::Instant::now();
        if which == "reference" {
            ZarrEnvironment::build_multiscale_reference(
                &dir,
                &input,
                &spec,
                CompressionPolicy::derived(),
            )
            .unwrap();
        } else {
            ZarrEnvironment::build_multiscale(&dir, &input, &spec).unwrap();
        }
        eprintln!("{which} {case}: {:.2} s", started.elapsed().as_secs_f64());
        if std::env::var_os("BLOCKFLOW_PYRAMID_KEEP").is_none() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
