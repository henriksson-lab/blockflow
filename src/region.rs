// SPDX-License-Identifier: MIT
//
// Original work. Extracted from `clearmap_rs::parallel_processing::region_io`,
// which retains the backend implementations that depend on that crate's IO
// layer. What moved here is exactly the part with no backend in it: the box,
// the two traits, and the in-memory implementations that make byte-identity
// between a streamed and an unstreamed run testable.
//
// The npy backend was one of the ones that stayed, and that was a mistake — the
// format is not this application's, and leaving it behind cost twenty private
// re-implementations of it in the crates that consume this one. It is `npy` in
// this crate now, implementing the two traits below over a file rather than
// over an array; see that module's header for the count and for what it does
// about memory order. The Zarr backend's position is unchanged.
//
// The split is the point of the crate boundary. A `RegionSource` is what the
// executor, the cache and the prefetcher are written against; where the bytes
// come from is somebody else's problem, and now it is somebody else's *crate*.
//
// Deliberate shape of the traits, and why
// ---------------------------------------
// * **Typed (`ArrayD<T>`), not bytes.** Every compute kernel consumes an array.
//   A byte-oriented trait would force a dtype dispatch and a reinterpret at
//   every call site, and would not even save the copy — a decoding backend hands
//   back an owned buffer regardless. The cache does work in bytes internally
//   (`cache::ChunkFetcher`), because it must hold seven arrays of four dtypes
//   under one budget; that is a different layer with a different reason.
// * **`&self`, not `&mut self`.** Backends whose reads need an exclusive handle
//   own a handle pool behind `&self` rather than pushing `&mut` up through
//   every caller. This keeps `Arc<dyn RegionSource<T>>` usable, which the
//   prefetcher requires: it reads the same source from several threads at once.
//
// `is_known_empty` is the one addition made during extraction. It is defaulted
// to `None` — "I do not know" — so no existing implementation changes, and it
// exists because the prefetcher can act on it: a region a backend already knows
// is empty need not be read, and must not be coalesced across.

use std::sync::Mutex;

use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::error::{Error, Result};

/// An axis-aligned box, in voxels.
///
/// Half-open per axis: `start[i] .. start[i] + shape[i]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Region {
    pub start: Vec<usize>,
    pub shape: Vec<usize>,
}

impl Region {
    pub fn new(start: &[usize], shape: &[usize]) -> Self {
        Self {
            start: start.to_vec(),
            shape: shape.to_vec(),
        }
    }

    /// The whole of `shape`.
    pub fn whole(shape: &[usize]) -> Self {
        Self {
            start: vec![0; shape.len()],
            shape: shape.to_vec(),
        }
    }

    /// From per-axis `(lo, hi)` ranges, which is how the block planner speaks.
    pub fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        Self {
            start: ranges.iter().map(|&(lo, _)| lo).collect(),
            shape: ranges
                .iter()
                .map(|&(lo, hi)| hi.saturating_sub(lo))
                .collect(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.start.len()
    }

    /// Voxels in the region.
    /// The extent as a fixed array, defaulting a missing axis to `1`.
    ///
    /// For callers that need the *shape* rather than the voxel count — a
    /// surface term is not a function of a volume, and
    /// the six-faces shape `crate::fragment::SidecarSize::block_faces`
    /// describes is one.
    pub fn shape3(&self) -> [usize; 3] {
        let mut shape = [1usize; 3];
        for axis in 0..3 {
            shape[axis] = self.shape.get(axis).copied().unwrap_or(1);
        }
        shape
    }

    pub fn voxels(&self) -> usize {
        self.shape.iter().product()
    }

    /// Exclusive upper corner.
    pub fn end(&self) -> Vec<usize> {
        self.start
            .iter()
            .zip(self.shape.iter())
            .map(|(&start, &len)| start + len)
            .collect()
    }

    /// Per-axis `(lo, hi)` ranges — the planner's spelling.
    pub fn ranges(&self) -> Vec<(usize, usize)> {
        self.start
            .iter()
            .zip(self.shape.iter())
            .map(|(&start, &len)| (start, start + len))
            .collect()
    }

    /// The box common to both, or `None` when they do not meet.
    ///
    /// Used by the cache to work out which part of a chunk a request wants.
    pub fn intersect(&self, other: &Region) -> Option<Region> {
        if self.ndim() != other.ndim() {
            return None;
        }
        let mut start = Vec::with_capacity(self.ndim());
        let mut shape = Vec::with_capacity(self.ndim());
        for axis in 0..self.ndim() {
            let lo = self.start[axis].max(other.start[axis]);
            let hi =
                (self.start[axis] + self.shape[axis]).min(other.start[axis] + other.shape[axis]);
            if hi <= lo {
                return None;
            }
            start.push(lo);
            shape.push(hi - lo);
        }
        Some(Region { start, shape })
    }

    /// Guard: this region lies wholly inside `shape`, and has its rank.
    ///
    /// Public because backend implementations in other crates need exactly this
    /// check and a second implementation of it is how off-by-ones get in.
    /// `what` names the caller, so the message says which source refused.
    pub fn check_within(&self, shape: &[usize], what: &str) -> Result<()> {
        if self.ndim() != shape.len() {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: self.shape.clone(),
            });
        }
        for (axis, ((&start, &len), &dim)) in self
            .start
            .iter()
            .zip(self.shape.iter())
            .zip(shape.iter())
            .enumerate()
        {
            if start + len > dim {
                return Err(Error::InvalidArgument(format!(
                    "{what}: region axis {axis} spans {start}..{} of {dim}",
                    start + len
                )));
            }
        }
        Ok(())
    }
}

/// Something a rectangular region can be read out of without materialising the
/// rest.
pub trait RegionSource<T>: Send + Sync {
    /// Shape of the whole volume, in voxels.
    fn shape(&self) -> &[usize];

    /// Read one region. The returned array has exactly `region.shape`.
    fn read_region(&self, region: &Region) -> Result<ArrayD<T>>;

    /// A name for diagnostics.
    fn describe(&self) -> String {
        format!("region source {:?}", self.shape())
    }

    /// Fast path: `Some(true)` if the region is known empty without reading it —
    /// an absent chunk, an all-fill shard, a sparse index miss.
    ///
    /// `None` means "I cannot tell cheaply", which is what a backend without the
    /// capability must say. It is never a licence to guess: a `Some(true)` that
    /// is wrong silently zeroes real data, so a backend that is unsure says
    /// `None` and pays for the read.
    fn is_known_empty(&self, _region: &Region) -> Option<bool> {
        None
    }
}

/// Something a rectangular region can be written into without holding the rest.
///
/// Writes are order-independent and, for the streaming loop, disjoint — the loop
/// only ever writes a block's *valid* region, and those tile the volume exactly.
/// That is why `&self` suffices and why no sequencing is needed.
pub trait RegionSink<T>: Send + Sync {
    fn shape(&self) -> &[usize];

    /// Write `data` with its lower corner at `start`.
    fn write_region(&self, start: &[usize], data: &ArrayD<T>) -> Result<()>;

    /// Stage barrier: everything written is durable.
    fn finish(&self) -> Result<()> {
        Ok(())
    }

    fn describe(&self) -> String {
        format!("region sink {:?}", self.shape())
    }
}

// ------------------------------------------------------------------- arrays --

/// An in-memory volume, read by region.
///
/// This is what makes byte-identity testable: a streaming loop pointed at an
/// `ArrayRegionSource` sees exactly the values a whole-volume run slices out of
/// the same array, so any difference between a streamed and an unstreamed run is
/// the loop's, not the storage layer's. The cache's correctness sweep leans on
/// the same property.
pub struct ArrayRegionSource<'a, T> {
    volume: &'a ArrayD<T>,
}

impl<'a, T> ArrayRegionSource<'a, T> {
    pub fn new(volume: &'a ArrayD<T>) -> Self {
        Self { volume }
    }
}

impl<T> RegionSource<T> for ArrayRegionSource<'_, T>
where
    T: Clone + Send + Sync,
{
    fn shape(&self) -> &[usize] {
        self.volume.shape()
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        region.check_within(self.volume.shape(), "array region source")?;
        let mut view = self.volume.view();
        for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
            view.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
        }
        Ok(view.to_owned())
    }

    fn describe(&self) -> String {
        format!("in-memory array {:?}", self.volume.shape())
    }
}

/// An in-memory volume, written by region.
///
/// Holds the whole output, so it is *not* a streaming sink — it exists as the
/// control in the byte-identity comparison and for volumes small enough that the
/// output was never the problem.
pub struct ArrayRegionSink<T> {
    shape: Vec<usize>,
    volume: Mutex<ArrayD<T>>,
}

impl<T> ArrayRegionSink<T>
where
    T: Clone + Default,
{
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            volume: Mutex::new(ArrayD::from_elem(IxDyn(shape), T::default())),
        }
    }

    /// Take the assembled volume.
    pub fn into_inner(self) -> ArrayD<T> {
        self.volume
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl<T> RegionSink<T> for ArrayRegionSink<T>
where
    T: Clone + Default + Send + Sync,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn write_region(&self, start: &[usize], data: &ArrayD<T>) -> Result<()> {
        let region = Region::new(start, data.shape());
        region.check_within(&self.shape, "array region sink")?;
        let mut volume = self
            .volume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut target = volume.view_mut();
        for (axis, (&lo, &len)) in start.iter().zip(data.shape().iter()).enumerate() {
            target.slice_axis_inplace(Axis(axis), Slice::from(lo..lo + len));
        }
        target.assign(data);
        Ok(())
    }

    fn describe(&self) -> String {
        format!("in-memory sink {:?}", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn ramp(shape: [usize; 3]) -> ArrayD<f64> {
        let mut next = 0.0;
        Array3::from_shape_fn(shape, |_| {
            next += 1.0;
            next
        })
        .into_dyn()
    }

    #[test]
    fn a_region_describes_its_own_box() {
        let region = Region::from_ranges(&[(2, 6), (0, 3)]);
        assert_eq!(region.start, vec![2, 0]);
        assert_eq!(region.shape, vec![4, 3]);
        assert_eq!(region.end(), vec![6, 3]);
        assert_eq!(region.voxels(), 12);
        assert_eq!(region.ranges(), vec![(2, 6), (0, 3)]);
    }

    #[test]
    fn intersection_is_the_common_box_or_nothing() {
        let left = Region::new(&[0, 0], &[4, 4]);
        let right = Region::new(&[2, 3], &[4, 4]);
        assert_eq!(left.intersect(&right), Some(Region::new(&[2, 3], &[2, 1])));
        assert_eq!(left.intersect(&Region::new(&[4, 0], &[1, 1])), None);
        assert_eq!(left.intersect(&Region::new(&[0], &[1])), None);
        // Symmetric.
        assert_eq!(left.intersect(&right), right.intersect(&left));
    }

    #[test]
    fn an_array_source_returns_exactly_the_box() {
        let volume = ramp([4, 5, 6]);
        let source = ArrayRegionSource::new(&volume);
        let region = Region::new(&[1, 2, 3], &[2, 2, 2]);
        let got = source.read_region(&region).unwrap();
        assert_eq!(got.shape(), &[2, 2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(got[[i, j, k]], volume[[1 + i, 2 + j, 3 + k]]);
                }
            }
        }
    }

    #[test]
    fn a_region_outside_the_volume_is_refused_rather_than_clamped() {
        let volume = ramp([4, 4, 4]);
        let source = ArrayRegionSource::new(&volume);
        assert!(source
            .read_region(&Region::new(&[3, 0, 0], &[2, 4, 4]))
            .is_err());
        assert!(source.read_region(&Region::new(&[0, 0], &[4, 4])).is_err());
    }

    #[test]
    fn array_round_trip_through_the_sink_reassembles_the_volume() {
        let volume = ramp([4, 5, 6]);
        let source = ArrayRegionSource::new(&volume);
        let sink = ArrayRegionSink::<f64>::zeros(&[4, 5, 6]);
        // Two disjoint halves along axis 1.
        for (start, shape) in [([0, 0, 0], [4, 2, 6]), ([0, 2, 0], [4, 3, 6])] {
            let region = Region::new(&start, &shape);
            let data = source.read_region(&region).unwrap();
            sink.write_region(&start, &data).unwrap();
        }
        sink.finish().unwrap();
        assert_eq!(sink.into_inner(), volume);
    }

    #[test]
    fn a_source_that_cannot_tell_says_so_rather_than_guessing() {
        let volume = ramp([4, 4, 4]);
        let source = ArrayRegionSource::new(&volume);
        assert_eq!(source.is_known_empty(&Region::whole(&[4, 4, 4])), None);
    }
}
