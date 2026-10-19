// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The fused image's multiscale OME-Zarr pyramid, written slab by slab and
//! chunk-parallel.
//!
//! The image is handed over in **slabs** along one axis (the sweep axis), in
//! order. Each level keeps the rows it has been given but not yet written or
//! downsampled; as soon as a whole row of chunks is present it is encoded and
//! stored, chunks in parallel, and as soon as the windows of the next level's
//! rows are present those rows are computed (rows in parallel) and handed on.
//! So a level is never read back from the store, and no level is held whole
//! unless its own chunks span the whole sweep axis.
//!
//! **Bytes.** The arrays, their metadata and the group's metadata are those
//! `blockflow::zarr_env::ZarrEnvironment::build_multiscale` writes for a
//! resident volume and the same [`PyramidSpec`]-style factors: the chunk shape
//! of every level is `chunk_for_block` of its shape, the codec is
//! `Compression::for_dtype`, the fill value is the type's unwritten sentinel,
//! a level is `floor(n / f)` per axis, and a coarser voxel is the mean of its
//! window summed in the same order (or the window's first voxel for `bool`).
//! Chunks are encoded by the same `zarrs` codecs, so every stored byte agrees.
//! That builder encodes one chunk at a time for every level but the first;
//! this one encodes every ready chunk at once.

use std::path::Path;
use std::sync::Arc;

use blockflow::error::{Error, Result};
use blockflow::voxels::{VoxelElement, Voxels};
use blockflow::zarr_env::{chunk_for_block, zarr_data_type, Compression, PhysicalFrame};
use blockflow::Dtype;
use ndarray::{s, Array3, ArrayView3, Axis};
use rayon::prelude::*;
use zarrs::array::codec::GzipCodec;
use zarrs::array::{Array as ZarrArray, ArrayBuilder, ArraySubset, Element, FillValue};
use zarrs::filesystem::FilesystemStore;

type Stored = ZarrArray<FilesystemStore>;

/// What a slab's elements must be to be stored.
pub trait PyramidElement: VoxelElement + Element + Send + Sync + Copy + 'static {}
impl<T: VoxelElement + Element + Send + Sync + Copy + 'static> PyramidElement for T {}

/// The decimation factors `write_volume_at` has always used: y and x halved per
/// level, z too while it is at least 64, until both in-plane extents are under
/// 64 or `levels` levels exist (level 0 included).
pub fn pyramid_factors(shape: [usize; 3], levels: usize) -> Vec<[usize; 3]> {
    let mut factors = Vec::new();
    let mut extent = shape;
    for _ in 1..levels {
        if extent[1] < 64 && extent[2] < 64 {
            break;
        }
        let z = if extent[0] >= 64 { 2 } else { 1 };
        factors.push([z, 2, 2]);
        extent = [extent[0] / z, extent[1] / 2, extent[2] / 2];
    }
    factors
}

/// The axis a volume of `shape` is best handed over along: the one its
/// level-0 chunks cut most often (the first such axis on a tie), so a slab of
/// whole chunk rows is as thin as the chunking allows.
pub fn sweep_axis(shape: [usize; 3], dtype: Dtype) -> usize {
    let chunk = chunk_for_block(shape, dtype);
    let count = |a: usize| shape[a].div_ceil(chunk[a]);
    let mut best = 0;
    for a in 1..3 {
        if count(a) > count(best) {
            best = a;
        }
    }
    best
}

/// The type's unwritten sentinel, as `zarr_env` stores it.
fn unwritten_fill(dtype: Dtype) -> Result<FillValue> {
    Ok(match dtype {
        Dtype::Bool => FillValue::from(true),
        Dtype::U8 => FillValue::from(u8::MAX),
        Dtype::U16 => FillValue::from(u16::MAX),
        Dtype::U32 => FillValue::from(u32::MAX),
        Dtype::U64 => FillValue::from(u64::MAX),
        Dtype::I8 => FillValue::from(i8::MAX),
        Dtype::I16 => FillValue::from(i16::MAX),
        Dtype::I32 => FillValue::from(i32::MAX),
        Dtype::I64 => FillValue::from(i64::MAX),
        Dtype::F32 => FillValue::from(f32::NAN),
        Dtype::F64 => FillValue::from(f64::NAN),
        Dtype::F16 => return Err(Error::invalid("float16 images cannot be written")),
    })
}

fn create_array(
    store: &Arc<FilesystemStore>,
    path: &str,
    dtype: Dtype,
    shape: [usize; 3],
    chunk: [usize; 3],
) -> Result<Stored> {
    let mut builder = ArrayBuilder::new(
        shape.iter().map(|&v| v as u64).collect::<Vec<_>>(),
        chunk.iter().map(|&v| v.max(1) as u64).collect::<Vec<_>>(),
        zarr_data_type(dtype)?,
        unwritten_fill(dtype)?,
    );
    match Compression::for_dtype(dtype) {
        Compression::None => {}
        Compression::Gzip(level) => {
            let codec = GzipCodec::new(level.min(9)).map_err(Error::backend)?;
            builder.bytes_to_bytes_codecs(vec![Arc::new(codec)]);
        }
        other => {
            return Err(Error::invalid(format!(
                "no writer here for {other:?}, which Compression::for_dtype does not choose"
            )))
        }
    }
    let array = builder.build(store.clone(), path).map_err(Error::backend)?;
    array.store_metadata().map_err(Error::backend)?;
    Ok(array)
}

/// One level: its array and the rows it holds that are not yet written or
/// not yet downsampled.
struct Level<T> {
    array: Stored,
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
pub struct PyramidWriter<T: PyramidElement> {
    root: std::path::PathBuf,
    axis: usize,
    levels: Vec<Level<T>>,
    scales: Vec<[usize; 3]>,
    physical: Option<PhysicalFrame>,
}

impl<T: PyramidElement> PyramidWriter<T> {
    /// Create the arrays of a pyramid of `shape` under `root` (which must not
    /// exist), built by `factors`, handed over along `axis`.
    pub fn create(
        root: &Path,
        shape: [usize; 3],
        factors: &[[usize; 3]],
        axis: usize,
        physical: Option<PhysicalFrame>,
    ) -> Result<Self> {
        if axis >= 3 {
            return Err(Error::invalid(format!("sweep axis {axis} of a volume")));
        }
        let dtype = T::DTYPE;
        std::fs::create_dir_all(root).map_err(Error::backend)?;
        let store = Arc::new(FilesystemStore::new(root).map_err(Error::backend)?);
        let mut levels = Vec::with_capacity(factors.len() + 1);
        let mut scales = vec![[1usize; 3]];
        let mut level_shape = shape;
        for index in 0..=factors.len() {
            if index > 0 {
                let factor = factors[index - 1];
                let next: [usize; 3] = std::array::from_fn(|a| level_shape[a] / factor[a]);
                if next.contains(&0) {
                    return Err(Error::invalid(format!(
                        "pyramid level {index} would shrink {level_shape:?} by {factor:?} to \
                         zero voxels"
                    )));
                }
                level_shape = next;
                let mut cumulative = *scales.last().expect("level 0");
                for a in 0..3 {
                    cumulative[a] *= factor[a];
                }
                scales.push(cumulative);
            }
            let chunk = chunk_for_block(level_shape, dtype);
            let array = create_array(&store, &format!("/{index}"), dtype, level_shape, chunk)?;
            levels.push(Level {
                array,
                shape: level_shape,
                chunk,
                factor: factors.get(index).copied(),
                pending_start: 0,
                pending: None,
                written: 0,
                next: 0,
            });
        }
        Ok(Self {
            root: root.to_path_buf(),
            axis,
            levels,
            scales,
            physical,
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

    /// Hand over the next slab of level 0: rows from where the last one ended,
    /// the whole extent across the sweep axis.
    pub fn push(&mut self, slab: Array3<T>) -> Result<()> {
        self.push_level(0, slab)
    }

    fn push_level(&mut self, index: usize, slab: Array3<T>) -> Result<()> {
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
        let end = level.pending_start + level.pending.as_ref().map_or(0, |p| p.len_of(Axis(axis)));
        if end + expected[axis] > level.shape[axis] {
            return Err(Error::invalid(format!(
                "level {index} has {} rows along axis {axis} and was handed rows {end}..{}",
                level.shape[axis],
                end + expected[axis]
            )));
        }
        level.pending = Some(match level.pending.take() {
            None => {
                level.pending_start = end;
                slab
            }
            Some(held) => ndarray::concatenate(Axis(axis), &[held.view(), slab.view()])
                .map_err(Error::backend)?,
        });
        let end = end + expected[axis];

        // Every whole chunk row now present (or the last, short one).
        let c = level.chunk[axis];
        let mut stop = level.written;
        while stop < level.shape[axis] && (stop + c <= end || end == level.shape[axis]) {
            stop = (stop + c).min(level.shape[axis]);
        }
        if stop > level.written {
            let pending = level.pending.as_ref().expect("rows were just added");
            write_rows(level, pending.view(), axis, level.written, stop)?;
            level.written = stop;
        }

        // The next level's rows whose windows are now present.
        let mut coarser = None;
        if let Some(factor) = level.factor {
            let f = factor[axis];
            let next_shape = self.levels[index + 1].shape;
            let rows = next_shape[axis];
            let level = &mut self.levels[index];
            let ready = (end / f).min(rows);
            if ready > level.next {
                let pending = level.pending.as_ref().expect("rows were just added");
                let first = level.next * f - level.pending_start;
                let window = pending.slice_axis(
                    Axis(axis),
                    ndarray::Slice::from(first..first + (ready - level.next) * f),
                );
                let mut shape = next_shape;
                shape[axis] = ready - level.next;
                coarser = Some(downsample(window, factor, shape));
                level.next = ready;
            }
        }

        // Drop the rows nothing needs any more.
        let level = &mut self.levels[index];
        let keep = match level.factor {
            Some(factor) => level.written.min(level.next * factor[axis]),
            None => level.written,
        };
        if keep > level.pending_start {
            let pending = level.pending.take().expect("rows were just added");
            let held = pending.len_of(Axis(axis));
            let drop = keep - level.pending_start;
            if drop < held {
                level.pending = Some(
                    pending
                        .slice_axis(Axis(axis), ndarray::Slice::from(drop..))
                        .to_owned(),
                );
            }
            level.pending_start = keep;
        }

        if let Some(coarser) = coarser {
            self.push_level(index + 1, coarser)?;
        }
        Ok(())
    }

    /// Check every level is complete and write the group's metadata.
    pub fn finish(self) -> Result<()> {
        for (index, level) in self.levels.iter().enumerate() {
            if level.written != level.shape[self.axis] {
                return Err(Error::invalid(format!(
                    "pyramid level {index} has {} of its {} rows along axis {}",
                    level.written, level.shape[self.axis], self.axis
                )));
            }
        }
        write_multiscales_metadata(&self.root, &self.scales, self.physical.as_ref())
    }
}

/// Store the chunks of rows `start..stop` (whole chunk rows) of `level` from
/// `pending`, which holds rows from `level.pending_start`.
fn write_rows<T: PyramidElement>(
    level: &Level<T>,
    pending: ArrayView3<T>,
    axis: usize,
    start: usize,
    stop: usize,
) -> Result<()> {
    let chunk = level.chunk;
    let shape = level.shape;
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
    boxes.into_par_iter().try_for_each(|b| {
        let mut local = b;
        local[axis] = (
            b[axis].0 - level.pending_start,
            b[axis].1 - level.pending_start,
        );
        let piece = pending.slice(s![
            local[0].0..local[0].1,
            local[1].0..local[1].1,
            local[2].0..local[2].1
        ]);
        let data = piece.iter().copied().collect::<Vec<T>>();
        let subset = ArraySubset::new_with_start_shape(
            b.iter().map(|r| r.0 as u64).collect(),
            b.iter().map(|r| (r.1 - r.0) as u64).collect(),
        )
        .map_err(Error::backend)?;
        level
            .array
            .store_array_subset(&subset, data.as_slice())
            .map_err(Error::backend)
    })
}

/// The block means of `input` (or first voxels, for `bool`), `shape` of them,
/// summed in the order `zarr_env`'s builder sums them.
fn downsample<T: PyramidElement>(
    input: ArrayView3<T>,
    factor: [usize; 3],
    shape: [usize; 3],
) -> Array3<T> {
    let mut out = Array3::from_elem(shape, T::from_f64(0.0));
    // Rows of the first axis in parallel; each voxel is computed alone.
    let plane = shape[1] * shape[2];
    out.as_slice_mut()
        .expect("a fresh array is contiguous")
        .par_chunks_mut(plane.max(1))
        .enumerate()
        .for_each(|(x, row)| {
            for (i, value) in row.iter_mut().enumerate() {
                let (y, z) = (i / shape[2], i % shape[2]);
                if T::DTYPE == Dtype::Bool {
                    *value = input[[x * factor[0], y * factor[1], z * factor[2]]];
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0usize;
                for dz in 0..factor[2] {
                    for dy in 0..factor[1] {
                        for dx in 0..factor[0] {
                            sum += input
                                [[x * factor[0] + dx, y * factor[1] + dy, z * factor[2] + dz]]
                            .into_f64();
                            count += 1;
                        }
                    }
                }
                *value = T::from_f64(sum / count as f64);
            }
        });
    out
}

/// The group's `zarr.json`, as `zarr_env` writes it.
fn write_multiscales_metadata(
    root: &Path,
    scales: &[[usize; 3]],
    physical: Option<&PhysicalFrame>,
) -> Result<()> {
    let datasets = scales
        .iter()
        .enumerate()
        .map(|(index, scale)| {
            let transformations = match physical {
                None => serde_json::json!([{"type": "scale", "scale": scale}]),
                Some(frame) => {
                    let size: [f64; 3] =
                        std::array::from_fn(|axis| scale[axis] as f64 * frame.voxel_size[axis]);
                    let at: [f64; 3] = std::array::from_fn(|axis| {
                        frame.origin[axis]
                            + (scale[axis] as f64 - 1.0) / 2.0 * frame.voxel_size[axis]
                    });
                    serde_json::json!([
                        {"type": "scale", "scale": size},
                        {"type": "translation", "translation": at}
                    ])
                }
            };
            serde_json::json!({
                "path": index.to_string(),
                "coordinateTransformations": transformations
            })
        })
        .collect::<Vec<_>>();
    let axis = |name: &str| match physical {
        None => serde_json::json!({"name": name, "type": "space", "unit": "voxel"}),
        Some(PhysicalFrame {
            unit: Some(unit), ..
        }) => serde_json::json!({"name": name, "type": "space", "unit": unit}),
        Some(PhysicalFrame { unit: None, .. }) => {
            serde_json::json!({"name": name, "type": "space"})
        }
    };
    let metadata = serde_json::json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "version": "0.5",
                        "axes": [axis("z"), axis("y"), axis("x")],
                        "datasets": datasets
                    }
                ]
            }
        }
    });
    let bytes = serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?;
    std::fs::write(root.join("zarr.json"), bytes).map_err(Error::backend)
}

/// Write a resident volume through a [`PyramidWriter`], in slabs of whole
/// level-0 chunk rows along [`sweep_axis`].
pub fn write_resident(
    root: &Path,
    volume: &Voxels,
    factors: &[[usize; 3]],
    physical: Option<PhysicalFrame>,
) -> Result<()> {
    blockflow::dtype_dispatch!(
        volume.dtype(),
        |Element| {
            let view = volume.view::<Element>()?;
            let shape = volume.shape();
            let axis = sweep_axis(shape, volume.dtype());
            let mut writer = PyramidWriter::<Element>::create(root, shape, factors, axis, physical)?;
            let step = slab_rows(shape, writer.chunk(), axis, volume.dtype());
            let mut at = 0;
            while at < shape[axis] {
                let stop = (at + step).min(shape[axis]);
                writer.push(view.slice_axis(Axis(axis), ndarray::Slice::from(at..stop)).to_owned())?;
                at = stop;
            }
            writer.finish()
        },
        f16 => Err(Error::invalid("float16 images cannot be written"))
    )
}

/// How many rows along `axis` a slab of a volume of `shape` holds: whole
/// level-0 chunk rows, at least enough chunks to keep every worker busy twice
/// over and about 256 MiB, at most the volume.
pub fn slab_rows(shape: [usize; 3], chunk: [usize; 3], axis: usize, dtype: Dtype) -> usize {
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
        .map(|a| shape[a].div_ceil(chunk[a]))
        .product();
    let workers = rayon::current_num_threads();
    let by_bytes = (256usize << 20).div_ceil(row_bytes.max(1) * c);
    let by_workers = (2 * workers).div_ceil(chunks_per_row.max(1));
    let rows = by_bytes.max(by_workers).max(1) * c;
    rows.min(shape[axis])
}
