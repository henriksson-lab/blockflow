// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The fused image's multiscale OME-Zarr pyramid: this crate's decimation
//! factors, written slab by slab by `blockflow`'s
//! [`MultiscaleWriter`](blockflow::zarr_env::MultiscaleWriter).
//!
//! The writer is core's, so the image is byte for byte what
//! `ZarrEnvironment::build_multiscale` writes for the resident volume and the
//! same factors (it is the same writer): the chunk shape of every level is
//! `chunk_for_extent` of its shape, the codec is `Compression::for_dtype`, the
//! fill value is the type's unwritten sentinel, a level is `floor(n / f)` per
//! axis, and a coarser voxel is the mean of its window (or the window's first
//! voxel for `bool`). Chunks are encoded in parallel as the memory cap allows.

use std::path::Path;

use blockflow::error::Result;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{
    chunk_for_extent, slab_rows_under, CompressionPolicy, MultiscaleWriter, PhysicalFrame,
    PyramidSpec, ZarrEnvironment,
};
use blockflow::Dtype;
use ndarray::Array3;

pub use blockflow::zarr_env::PyramidElement;

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
/// level-0 chunks cut most often (the first such axis on a tie).
pub fn sweep_axis(shape: [usize; 3], dtype: Dtype) -> usize {
    blockflow::zarr_env::sweep_axis(shape, chunk_for_extent(shape, dtype))
}

fn spec(factors: &[[usize; 3]], physical: Option<PhysicalFrame>) -> Result<PyramidSpec> {
    let spec = PyramidSpec::new(factors.to_vec())?;
    Ok(match physical {
        Some(frame) => spec.with_physical(frame),
        None => spec,
    })
}

/// A multiscale image written from slabs handed over in order along one axis:
/// core's writer, with this crate's way of stating the pyramid.
pub struct PyramidWriter<T: PyramidElement> {
    inner: MultiscaleWriter<T>,
}

impl<T: PyramidElement> PyramidWriter<T> {
    /// Create the arrays of a pyramid of `shape` under `root`, built by
    /// `factors`, handed over along `axis`.
    pub fn create(
        root: &Path,
        shape: [usize; 3],
        factors: &[[usize; 3]],
        axis: usize,
        physical: Option<PhysicalFrame>,
    ) -> Result<Self> {
        Ok(Self {
            inner: MultiscaleWriter::create(
                root,
                shape,
                &spec(factors, physical)?,
                &CompressionPolicy::derived(),
                axis,
            )?,
        })
    }

    /// The sweep axis.
    pub fn axis(&self) -> usize {
        self.inner.axis()
    }

    /// Level 0's chunk shape.
    pub fn chunk(&self) -> [usize; 3] {
        self.inner.chunk()
    }

    /// Level 0's shape.
    pub fn shape(&self) -> [usize; 3] {
        self.inner.shape()
    }

    /// Hand over the next slab of level 0.
    pub fn push(&mut self, slab: Array3<T>) -> Result<()> {
        self.inner.push(slab.view())
    }

    /// [`Self::push`], lent.
    pub fn push_view(&mut self, slab: ndarray::ArrayView3<'_, T>) -> Result<()> {
        self.inner.push(slab)
    }

    /// Check every level is complete and write the group's metadata.
    pub fn finish(self) -> Result<()> {
        self.inner.finish().map(drop)
    }
}

/// Write a resident volume as a pyramid of `factors`.
pub fn write_resident(
    root: &Path,
    volume: &Voxels,
    factors: &[[usize; 3]],
    physical: Option<PhysicalFrame>,
) -> Result<()> {
    ZarrEnvironment::build_multiscale(root, volume, &spec(factors, physical)?).map(drop)
}

/// How many rows along `axis` a slab of a volume of `shape` holds: whole
/// level-0 chunk rows, sized under the process's memory cap
/// ([`MultiscaleWriter::slab_rows`]).
pub fn slab_rows(shape: [usize; 3], chunk: [usize; 3], axis: usize, dtype: Dtype) -> usize {
    slab_rows_under(
        shape,
        chunk,
        axis,
        dtype,
        blockflow::budget::process_budget().total(),
    )
}
