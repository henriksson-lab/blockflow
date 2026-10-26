//! Single-level OME-Zarr output in bounded whole chunks, in natural ZYX order.
//! Reuses the environment's native typed encoder and physical metadata writer.
use super::{write_multiscales_metadata, AttachedImage, Compression, PhysicalFrame, StoredArray};
use crate::{
    error::{Error, Result},
    region::Region,
    voxels::VoxelElement,
};
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};
use zarrs::{array::Element, filesystem::FilesystemStore};

pub struct TileWriter<T: VoxelElement + Element> {
    root: PathBuf,
    array: StoredArray,
    shape: [usize; 3],
    chunk: [usize; 3],
    counts: [usize; 3],
    total: usize,
    next: usize,
    poisoned: bool,
    physical: Option<PhysicalFrame>,
    element: PhantomData<T>,
}
impl<T: VoxelElement + Element> TileWriter<T> {
    /// Fresh roots only. No metadata describing a complete OME image is
    /// installed until finish. One clipped chunk per write, no image/row buffer.
    pub fn create(
        root: impl Into<PathBuf>,
        shape: [usize; 3],
        chunk: [usize; 3],
        physical: Option<PhysicalFrame>,
        compression: Compression,
    ) -> Result<Self> {
        if shape.contains(&0)
            || chunk.contains(&0)
            || shape
                .iter()
                .try_fold(1usize, |n, v| n.checked_mul(*v))
                .is_none()
        {
            return Err(Error::invalid("invalid tiled image shape/chunk"));
        }
        if physical.as_ref().is_some_and(|p| {
            p.voxel_size.iter().any(|v| !v.is_finite() || *v <= 0.)
                || p.origin.iter().any(|v| !v.is_finite())
                || p.unit.as_ref().is_some_and(|u| u.trim().is_empty())
        }) {
            return Err(Error::invalid("invalid tiled image physical frame"));
        }
        let chunk = std::array::from_fn(|i| chunk[i].min(shape[i]));
        chunk
            .iter()
            .try_fold(T::DTYPE.size_of(), |n, v| n.checked_mul(*v))
            .ok_or_else(|| Error::invalid("tile byte count overflow"))?;
        let counts = std::array::from_fn(|i| shape[i].div_ceil(chunk[i]));
        let total = counts
            .iter()
            .try_fold(1usize, |n, v| n.checked_mul(*v))
            .ok_or_else(|| Error::invalid("tile count overflow"))?;
        let root = root.into();
        if let Some(parent) = root.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(Error::backend)?;
        }
        // create_dir is exclusive; no overwriting an existing image, even
        // between an existence check and creation.
        std::fs::create_dir(&root).map_err(Error::backend)?;
        let store = Arc::new(FilesystemStore::new(&root).map_err(Error::backend)?);
        let array = StoredArray::create(&store, "/0", 0, T::DTYPE, &shape, &chunk, compression)?;
        Ok(Self {
            root,
            array,
            shape,
            chunk,
            counts,
            total,
            next: 0,
            poisoned: false,
            physical,
            element: PhantomData,
        })
    }
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }
    pub fn chunk(&self) -> [usize; 3] {
        self.chunk
    }
    pub fn chunks_written(&self) -> usize {
        self.next
    }
    pub fn total_chunks(&self) -> usize {
        self.total
    }
    pub fn next_tile(&self) -> Option<([usize; 3], [usize; 3])> {
        if self.next == self.total || self.poisoned {
            return None;
        }
        let index = [
            self.next / (self.counts[1] * self.counts[2]),
            (self.next / self.counts[2]) % self.counts[1],
            self.next % self.counts[2],
        ];
        let start = std::array::from_fn(|i| index[i] * self.chunk[i]);
        Some((
            start,
            std::array::from_fn(|i| self.chunk[i].min(self.shape[i] - start[i])),
        ))
    }
    pub fn write_next(&mut self, values: &[T]) -> Result<()> {
        let (start, shape) = self
            .next_tile()
            .ok_or_else(|| Error::invalid("tiled writer complete or poisoned"))?;
        if values.len() != shape.iter().product::<usize>() {
            return Err(Error::invalid(
                "tile payload does not match next clipped chunk",
            ));
        }
        self.poisoned = true;
        self.array
            .write_as(&Region::new(&start, &shape), values, None)?;
        self.next += 1;
        self.poisoned = false;
        Ok(())
    }
    /// Publishes single-level metadata only after every chunk succeeds.
    /// No fsync/power-loss durability guarantee is made. A failed write poisons
    /// this writer; callers retain its incomplete fresh directory for diagnosis.
    pub fn finish(self) -> Result<AttachedImage> {
        if self.poisoned || self.next != self.total {
            return Err(Error::invalid(
                "cannot finish an incomplete/poisoned tiled image",
            ));
        }
        write_multiscales_metadata(&self.root, &[[1; 3]], self.physical.as_ref(), T::DTYPE)?;
        Ok(AttachedImage::at(self.root.join("0")))
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
}
