// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The IO half of the event stream, wired at **one seam**.
//
// `region_io`'s `RegionSource` / `RegionSink` is the narrow waist the design
// already has: every streamed read and write in this crate goes through one of
// the two, whether the storage underneath is an in-memory array, a Zarr store
// or a TIFF stack. So instrumentation belongs here as a *decorator* rather than
// scattered through `io/` — one wrapper covers every implementation, present
// and future, and `io/` keeps not knowing that anything is watching.
//
// What is emitted, and at what rate
// ---------------------------------
// **One event per `read_region` / `write_region` call** — that is, at the
// granularity the *caller* asked for, not per chunk. A block spanning a 4x4x4
// neighbourhood of chunks would emit 64 events per read at chunk granularity;
// at call granularity it emits one, carrying `chunks` as a *count*. Measured on
// the framework's own suite the call-granularity rate is one read and one write
// per task, against five to fifteen `OpApplied` events per task, so IO events
// are a **minority** of the stream rather than the 10-100x majority per-chunk
// emission would have made them. If per-chunk detail is ever wanted it should
// be a separate opt-in variant, so that the default rate stays bounded by the
// task count.
//
// `duration_ns` is wall time around the inner call, which for a real store is
// dominated by the storage round trip and is the number a cost model wants.
// `bytes` is `voxels * size_of::<T>()`, i.e. the **uncompressed** size in
// memory — not the on-disk size, which the source layer does not report.
//
// Correctness
// -----------
// The wrapper forwards `Result` untouched and emits *after* the inner call, so
// a failed read emits nothing and a listener cannot influence what was read. A
// listener that panics is contained exactly as in the executor: the wrapper
// dispatches through the same `catch_unwind` path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::ArrayD;

use crate::error::Result;
use crate::region::{Region, RegionSink, RegionSource};

use super::listener::EventListener;
use super::log::Event;

/// Shared fan-out for the IO decorators.
///
/// Deliberately a *separate* small struct rather than the executor's `Dispatch`:
/// a source may outlive any one run, and it holds its listeners by `Arc` rather
/// than borrowing them.
struct IoDispatch {
    listeners: Vec<Arc<dyn EventListener>>,
    faulted: Vec<AtomicBool>,
}

impl IoDispatch {
    fn new(listeners: Vec<Arc<dyn EventListener>>) -> Self {
        Self {
            faulted: listeners.iter().map(|_| AtomicBool::new(false)).collect(),
            listeners,
        }
    }

    fn emit(&self, event: Event) {
        for (slot, listener) in self.listeners.iter().enumerate() {
            if self.faulted[slot].load(Ordering::Relaxed) {
                continue;
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                listener.on_event(&event)
            }));
            if outcome.is_err() {
                self.faulted[slot].store(true, Ordering::Relaxed);
            }
        }
    }
}

/// A `RegionSource` that reports every read it serves.
pub struct ObservedSource<T, S: RegionSource<T>> {
    inner: S,
    dispatch: IoDispatch,
    /// Resolved once: `describe()` may format, and formatting per read would be
    /// an allocation on the IO path for a string that never changes.
    name: String,
    image: usize,
    chunk: Option<Vec<usize>>,
    marker: std::marker::PhantomData<fn() -> T>,
}

impl<T, S: RegionSource<T>> ObservedSource<T, S> {
    pub fn new(inner: S, listeners: Vec<Arc<dyn EventListener>>) -> Self {
        let name = inner.describe();
        Self {
            inner,
            dispatch: IoDispatch::new(listeners),
            name,
            image: 0,
            chunk: None,
            marker: std::marker::PhantomData,
        }
    }

    /// Which storage image this source is, for a multi-phase run. Default 0.
    pub fn at_image(mut self, image: usize) -> Self {
        self.image = image;
        self
    }

    /// The store's chunk shape, if known. Without it `chunks` is reported as 0
    /// — an honest "not known" rather than a fabricated count.
    pub fn with_chunk_shape(mut self, chunk: &[usize]) -> Self {
        self.chunk = Some(chunk.to_vec());
        self
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }
}

/// Chunks a region touches, for an arbitrary-rank chunk shape. `0` means the
/// chunking is not known.
fn chunks_touched(region: &Region, chunk: Option<&Vec<usize>>) -> u64 {
    let Some(chunk) = chunk else { return 0 };
    if chunk.len() != region.ndim() || chunk.contains(&0) {
        return 0;
    }
    let mut total: u64 = 1;
    for axis in 0..region.ndim() {
        let start = region.start[axis];
        let end = start + region.shape[axis];
        if region.shape[axis] == 0 {
            return 0;
        }
        let first = start / chunk[axis];
        let last = (end - 1) / chunk[axis];
        total *= (last - first + 1) as u64;
    }
    total
}

impl<T: Send + Sync, S: RegionSource<T>> RegionSource<T> for ObservedSource<T, S> {
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        let started = Instant::now();
        let data = self.inner.read_region(region)?;
        let duration_ns = started.elapsed().as_nanos() as u64;
        let voxels = region.voxels();
        self.dispatch.emit(Event::RegionRead {
            source: self.name.clone(),
            image: self.image,
            index: None,
            region: region.clone(),
            voxels,
            bytes: voxels as u64 * std::mem::size_of::<T>() as u64,
            chunks: chunks_touched(region, self.chunk.as_ref()),
            duration_ns,
        });
        Ok(data)
    }

    fn describe(&self) -> String {
        self.inner.describe()
    }
}

/// A `RegionSink` that reports every write it accepts, and the materialisation
/// at `finish`.
pub struct ObservedSink<T, S: RegionSink<T>> {
    inner: S,
    dispatch: IoDispatch,
    name: String,
    image: usize,
    phase: usize,
    intermediate: bool,
    chunk: Option<Vec<usize>>,
    bytes: std::sync::atomic::AtomicU64,
    marker: std::marker::PhantomData<fn() -> T>,
}

impl<T, S: RegionSink<T>> ObservedSink<T, S> {
    pub fn new(inner: S, listeners: Vec<Arc<dyn EventListener>>) -> Self {
        let name = inner.describe();
        Self {
            inner,
            dispatch: IoDispatch::new(listeners),
            name,
            image: 0,
            phase: 0,
            intermediate: false,
            chunk: None,
            bytes: std::sync::atomic::AtomicU64::new(0),
            marker: std::marker::PhantomData,
        }
    }

    /// Which phase writes here, and whether the result is an intermediate
    /// rather than the workflow's output. Both only affect the `Materialised`
    /// event's fields.
    pub fn at_phase(mut self, phase: usize, intermediate: bool) -> Self {
        self.phase = phase;
        self.image = phase + 1;
        self.intermediate = intermediate;
        self
    }

    pub fn with_chunk_shape(mut self, chunk: &[usize]) -> Self {
        self.chunk = Some(chunk.to_vec());
        self
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<T: Send + Sync, S: RegionSink<T>> RegionSink<T> for ObservedSink<T, S> {
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn write_region(&self, start: &[usize], data: &ArrayD<T>) -> Result<()> {
        let started = Instant::now();
        self.inner.write_region(start, data)?;
        let duration_ns = started.elapsed().as_nanos() as u64;
        let region = Region::new(start, data.shape());
        let voxels = region.voxels();
        let bytes = voxels as u64 * std::mem::size_of::<T>() as u64;
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.dispatch.emit(Event::RegionWritten {
            sink: self.name.clone(),
            image: self.image,
            index: None,
            region: region.clone(),
            voxels,
            bytes,
            chunks: chunks_touched(&region, self.chunk.as_ref()),
            duration_ns,
        });
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        self.inner.finish()?;
        self.dispatch.emit(Event::Materialised {
            phase: self.phase,
            image: self.image,
            bytes: self.bytes.load(Ordering::Relaxed),
            intermediate: self.intermediate,
        });
        Ok(())
    }

    fn describe(&self) -> String {
        self.inner.describe()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use super::*;
    use crate::listener::LatestOpPerChunk;
    use crate::region::{ArrayRegionSink, ArrayRegionSource};

    use std::sync::Mutex;

    #[derive(Default)]
    struct Collect(Mutex<Vec<Event>>);
    impl EventListener for Collect {
        fn on_event(&self, event: &Event) {
            self.0.lock().unwrap().push(event.clone());
        }
    }

    #[test]
    fn a_source_reports_one_event_per_call_not_one_per_chunk() {
        let volume: ArrayD<f64> = ArrayD::zeros(IxDyn(&[16, 16, 16]));
        let collect = Arc::new(Collect::default());
        let source = ObservedSource::new(
            ArrayRegionSource::new(&volume),
            vec![collect.clone() as Arc<dyn EventListener>],
        )
        .with_chunk_shape(&[4, 4, 4]);

        let region = Region::new(&[0, 0, 0], &[8, 8, 16]);
        let data = source.read_region(&region).unwrap();
        assert_eq!(data.shape(), &[8, 8, 16]);

        let events = collect.0.lock().unwrap().clone();
        assert_eq!(events.len(), 1, "one call, one event: {events:?}");
        match &events[0] {
            Event::RegionRead {
                voxels,
                bytes,
                chunks,
                region: at,
                index,
                ..
            } => {
                assert_eq!(*voxels, 8 * 8 * 16);
                assert_eq!(*bytes, 8 * 8 * 16 * 8);
                // 2 x 2 x 4 chunks of 4^3 — a count, not 16 separate events
                assert_eq!(*chunks, 16);
                assert_eq!(at, &region);
                assert_eq!(*index, None, "a bare source knows no block index");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_sink_reports_writes_and_the_materialisation() {
        let collect = Arc::new(Collect::default());
        let sink = ObservedSink::new(
            ArrayRegionSink::<f64>::zeros(&[8, 8, 8]),
            vec![collect.clone() as Arc<dyn EventListener>],
        )
        .at_phase(1, true);

        let block: ArrayD<f64> = ArrayD::zeros(IxDyn(&[4, 8, 8]));
        sink.write_region(&[0, 0, 0], &block).unwrap();
        sink.write_region(&[4, 0, 0], &block).unwrap();
        sink.finish().unwrap();

        let events = collect.0.lock().unwrap().clone();
        assert_eq!(events.len(), 3);
        match &events[2] {
            Event::Materialised {
                phase,
                image,
                bytes,
                intermediate,
            } => {
                assert_eq!((*phase, *image, *intermediate), (1, 2, true));
                assert_eq!(*bytes, 8 * 8 * 8 * 8, "both writes, in bytes");
            }
            other => panic!("{other:?}"),
        }
    }

    /// The IO decorator and the executor's dispatcher must contain a panicking
    /// listener the same way, or "a listener cannot fail a run" would hold in
    /// one layer and not the other.
    #[test]
    fn a_panicking_listener_does_not_fail_a_read() {
        struct Exploding;
        impl EventListener for Exploding {
            fn on_event(&self, _event: &Event) {
                panic!("boom");
            }
        }
        let volume: ArrayD<f64> = ArrayD::zeros(IxDyn(&[4, 4, 4]));
        let latest = Arc::new(LatestOpPerChunk::new());
        let source = ObservedSource::new(
            ArrayRegionSource::new(&volume),
            vec![
                Arc::new(Exploding) as Arc<dyn EventListener>,
                latest.clone() as Arc<dyn EventListener>,
            ],
        );
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let data = source.read_region(&Region::new(&[0, 0, 0], &[4, 4, 4]));
        std::panic::set_hook(previous);
        assert!(data.is_ok(), "an observer must not break the read");
    }

    #[test]
    fn an_unknown_chunk_shape_reports_zero_rather_than_a_guess() {
        let volume: ArrayD<f64> = ArrayD::zeros(IxDyn(&[4, 4, 4]));
        let collect = Arc::new(Collect::default());
        let source = ObservedSource::new(
            ArrayRegionSource::new(&volume),
            vec![collect.clone() as Arc<dyn EventListener>],
        );
        source
            .read_region(&Region::new(&[0, 0, 0], &[4, 4, 4]))
            .unwrap();
        let events = collect.0.lock().unwrap().clone();
        match &events[0] {
            Event::RegionRead { chunks, .. } => assert_eq!(*chunks, 0),
            other => panic!("{other:?}"),
        }
    }
}
