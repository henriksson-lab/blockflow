// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// One cache, keyed `(array, chunk)`, one LRU, one byte budget
// ============================================================
//
// Why this is ours rather than a storage library's
// ------------------------------------------------
// Every chunk cache the Zarr layer offers binds to **one array**:
// `{ array: Arc<Array<..>>, .. }`, constructed `new(array, capacity)`. There is
// no shared LRU across arrays. That is fine for a program that reads one array;
// it is the wrong shape for a pipeline, which holds source, binary, smoothed,
// filled, combined, output and skeleton open **at once**, with demand that is
// uneven and moves over time. Per-array caches strand memory in whichever array
// is cold at exactly the moment the hot one needs it, and no amount of tuning
// the individual capacities fixes it, because the right split changes during
// the run.
//
// So: one cache above the sources, keyed by `(array, chunk)`, evicting across
// all of them from a single recency order. Capacity moves to demand because
// nothing in the structure ties it down. `cross_array_pressure_moves_capacity`
// in `cache_tests` is the test a per-array cache cannot pass.
//
// The key is a **canonical lattice index**, never a request shape
// ---------------------------------------------------------------
// A cache keyed by the extent a caller asked for is a bad cache: two callers
// wanting overlapping boxes produce different keys over the same data, entries
// duplicate, hit rate collapses, and a request for one chunk cannot be served
// from a cached span that contains it. Here every request is decomposed onto
// the array's chunk lattice, and the lattice index is the key. Two differently
// shaped reads over the same data hit the same entries, by construction.
//
// The chunk shape is the caller's to choose (`register`). It is usually the
// storage unit — a Zarr chunk, a shard, a plane run — because anything else
// decodes twice. It is a parameter because "what is the decode unit" is a
// property of the backend and the site, not of this file.
//
// Two tiers, one eviction order
// -----------------------------
// | tier | holds | a hit costs | why it exists |
// |---|---|---|---|
// | `Decoded` | uncompressed bytes | one `memcpy` | **a hit must be free.** If a hit still had to decompress, prefetching would have moved the decode off the read's critical path and back onto the reader's — which is not what it was for. |
// | `Encoded` | compressed bytes | `memcpy` + decompress | at the measured **19.7x** on `bool` volumes, retaining one costs ~1/20 of the space. Speculative and retained entries are worth keeping at that price and not at full price. |
//
// Both tiers live in **one** LRU and are evicted against each other by byte
// cost, so an encoded entry survives roughly twenty times longer than the
// decoded entry it displaced — which is the whole point of having the second
// tier rather than a second cache.
//
// **Which tier a stage's chunks land in is a planner decision, so it is a
// parameter** (`ArrayPolicy`), settable per array and changeable during a run
// (`set_policy`). This file has no policy in it and must not grow one. What a
// planner needs in order to choose well is written up in the `Tier` docs.
//
// Nothing here can starve compute
// -------------------------------
// Retention takes an `Opportunistic` lease per entry from the shared
// `MemoryBudget` and **fails fast**: refused means the chunk is served and not
// retained, which is the cache degrading into a pass-through rather than into
// an error. Because the lease lives *in* the entry, "cache contents never
// exceed the lease" is not a check anybody has to run — the two are the same
// number, and dropping an entry returns its bytes on every path.
//
// Concurrency
// -----------
// One `Mutex` over the entry map and the recency order, held only for map
// operations and for the `memcpy` that serves a decoded hit. **Never held
// across a source read, a compression, or a listener call** — a slow backend or
// a slow listener must not be able to serialise the cache.
//
// Concurrent demand for the same chunk costs one read, not `N`: a fetch marks
// its chunks pending, and a second reader waits for the first rather than
// duplicating the work. The wait is bounded and falls back to fetching, so a
// panicking fetcher costs a duplicated read rather than a hang.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};

use crate::budget::{Lease, MemoryBudget};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::listener::EventListener;
use crate::log::{Event, PrefetchWaste};
use crate::region::{Region, RegionSource};

/// How long a reader waits for a fetch already in flight before doing it
/// itself.
///
/// Not load-bearing for correctness — the fallback is a duplicated read, which
/// is merely wasteful. It exists so that a fetcher that panics mid-fetch costs
/// a stall rather than a hang, and it is generous because the thing being
/// waited for is a storage read.
const PENDING_WAIT: Duration = Duration::from_millis(200);

/// Default cap on how many lattice-consecutive chunks are fetched in one
/// request.
///
/// Coalescing trades over-read for fewer round trips, which is the right trade
/// on high-latency storage and the wrong one on a local NVMe. Hence a
/// parameter (`with_max_coalesce`) rather than a constant, with a default that
/// is useful and not aggressive.
const DEFAULT_MAX_COALESCE: usize = 8;

// ------------------------------------------------------------------ tiers --

/// Which form a cached chunk is held in.
///
/// # What a planner needs in order to choose
///
/// The question is not "which is faster" — `Decoded` always serves faster — but
/// **how many chunks fit**, and that depends on three things the planner knows
/// and this file does not:
///
/// * **Compression ratio of the stage's data.** Measured at 19.7x for the
///   `bool` volumes in this pipeline; near 1x for noisy `f32`. At 19.7x an
///   encoded entry buys twenty times the residency for the same bytes; at 1.1x
///   it buys nothing and costs a decompression per hit. The ratio is
///   observable — `CacheStats::encoded_ratio` reports what the cache actually
///   achieved — so this is a measurement, not a guess.
/// * **Reuse distance.** A chunk read once by one block and then never again
///   should not be retained in either tier. A halo chunk read by every one of
///   its neighbours wants `Decoded`. A chunk that *might* be revisited several
///   phases later wants `Encoded`: keeping it at 1/20 cost is nearly free, and
///   the alternative is re-reading it from storage.
/// * **Whether the hit is on the critical path.** A prefetched chunk exists
///   precisely so that the reader who wants it does no work. Putting it in
///   `Encoded` moves the decompression onto that reader, which is the cost the
///   prefetch was buying off. So: prefetched-for-imminent-use is `Decoded`;
///   retained-in-case is `Encoded`. That is why `ArrayPolicy` has two fields
///   and not one.
///
/// # Measured, on one machine, release build, 64 KiB chunks
///
/// `measured_cache_and_prefetch_report` in `cache_tests` prints these; they are
/// the shape of the trade rather than a promise about your hardware.
///
/// | | cost |
/// |---|---|
/// | decoded hit | **9.9 us** |
/// | encoded hit (deflate level 1) | **962 us** — ~100x a decoded hit |
/// | uncached read from a 40 ms source | **40 000 us** |
///
/// So the ordering that matters is not "decoded beats encoded" but *both beat
/// storage by orders of magnitude*: an encoded hit is still ~40x cheaper than
/// the read it replaced. `Encoded` is the wrong tier for a chunk read every few
/// microseconds and the right one for a chunk that would otherwise have to come
/// back off disk. Compression on a sparse `bool` chunk measured **107x** at
/// 256 KiB, so that residency is nearly free; on the `u16` ramp used for the
/// timing above it is close to 1x, which is why the ratio must be measured per
/// stage and not assumed.
///
/// The default is `Decoded` for both, because it is the one that cannot be
/// slower than no cache at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Uncompressed. A hit is a copy.
    Decoded,
    /// Compressed with the cache's [`Codec`]. A hit is a copy plus a decode.
    Encoded,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Decoded => "decoded",
            Self::Encoded => "encoded",
        }
    }
}

/// Per-array caching policy. **The planner's parameter; this crate has no
/// default policy beyond "do the safe thing".**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayPolicy {
    /// Tier for a chunk brought in by a demand read.
    pub demand_tier: Tier,
    /// Tier for a chunk brought in by the prefetcher.
    pub prefetch_tier: Tier,
    /// Whether to retain at all. `false` makes this array a pass-through — the
    /// right setting for a stage read strictly once, where retention is pure
    /// eviction pressure on the arrays that do get reused.
    pub retain: bool,
}

impl Default for ArrayPolicy {
    fn default() -> Self {
        Self {
            demand_tier: Tier::Decoded,
            prefetch_tier: Tier::Decoded,
            retain: true,
        }
    }
}

impl ArrayPolicy {
    /// Retain nothing. Reads pass through to the source.
    pub fn pass_through() -> Self {
        Self {
            retain: false,
            ..Self::default()
        }
    }

    /// Prefetch into `Decoded` so an imminent read is free; retain demand
    /// misses `Encoded` because they may not be wanted again soon.
    pub fn prefetch_hot_retain_cold() -> Self {
        Self {
            demand_tier: Tier::Encoded,
            prefetch_tier: Tier::Decoded,
            retain: true,
        }
    }

    pub fn all(tier: Tier) -> Self {
        Self {
            demand_tier: tier,
            prefetch_tier: tier,
            retain: true,
        }
    }
}

// ------------------------------------------------------------------ codec --

/// How the `Encoded` tier compresses. A parameter, because the right codec is a
/// property of the data and the site.
pub trait Codec: Send + Sync {
    fn name(&self) -> &'static str;
    fn encode(&self, decoded: &[u8]) -> Vec<u8>;
    /// `decoded_len` is known exactly (chunk voxels x element size), so a
    /// correct decode never needs to grow a buffer and a wrong one is caught
    /// here rather than downstream.
    fn decode(&self, encoded: &[u8], decoded_len: usize) -> Result<Vec<u8>>;
}

/// The built-in codec: raw deflate.
///
/// Chosen for being pure Rust, dependency-light and unsurprising, not for being
/// the fastest — a site that cares should supply zstd or blosc through
/// [`Codec`]. On the `bool` volumes this pipeline produces it reaches the
/// ratios the encoded tier is premised on, because such a volume is mostly runs
/// of one byte.
pub struct DeflateCodec {
    level: u32,
}

impl DeflateCodec {
    /// `level` is 0-9. The default is 1: the encoded tier's job is to hold more
    /// chunks, and past level 1 the extra ratio on run-heavy data is small
    /// while the CPU cost is not.
    pub fn new(level: u32) -> Self {
        Self {
            level: level.min(9),
        }
    }
}

impl Default for DeflateCodec {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Codec for DeflateCodec {
    fn name(&self) -> &'static str {
        "deflate"
    }

    fn encode(&self, decoded: &[u8]) -> Vec<u8> {
        use flate2::write::DeflateEncoder;
        use std::io::Write;
        let mut encoder = DeflateEncoder::new(Vec::new(), flate2::Compression::new(self.level));
        // Writing to a `Vec` cannot fail, and neither can finishing it; the
        // `Result`s are an artefact of the `io::Write` interface.
        let _ = encoder.write_all(decoded);
        encoder.finish().unwrap_or_default()
    }

    fn decode(&self, encoded: &[u8], decoded_len: usize) -> Result<Vec<u8>> {
        use flate2::write::DeflateDecoder;
        use std::io::Write;
        let mut decoder = DeflateDecoder::new(Vec::with_capacity(decoded_len));
        decoder
            .write_all(encoded)
            .map_err(|err| Error::invalid(format!("cache decode: {err}")))?;
        let out = decoder
            .finish()
            .map_err(|err| Error::invalid(format!("cache decode: {err}")))?;
        if out.len() != decoded_len {
            return Err(Error::ShapeMismatch {
                expected: vec![decoded_len],
                got: vec![out.len()],
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- element --

/// An element type the cache can hold.
///
/// The cache works in **bytes**, not in `T`, because it holds chunks from
/// several arrays of several dtypes under one byte budget — a generic parameter
/// would make that impossible. This trait is the narrow conversion, and it is
/// exact in both directions: little-endian for the numeric types, one byte of
/// `0`/`1` for `bool`. Round-tripping is byte-identity, which is what makes the
/// correctness claim ("a cached read equals an uncached read") checkable.
pub trait CacheElement: Copy + Send + Sync + 'static {
    const DTYPE: Dtype;
    fn encode_into(values: &[Self], into: &mut Vec<u8>);
    fn decode_from(bytes: &[u8]) -> Result<Vec<Self>>;
}

macro_rules! numeric_element {
    ($type:ty, $dtype:expr) => {
        impl CacheElement for $type {
            const DTYPE: Dtype = $dtype;

            fn encode_into(values: &[Self], into: &mut Vec<u8>) {
                into.reserve(values.len() * std::mem::size_of::<Self>());
                for value in values {
                    into.extend_from_slice(&value.to_le_bytes());
                }
            }

            fn decode_from(bytes: &[u8]) -> Result<Vec<Self>> {
                let width = std::mem::size_of::<Self>();
                if bytes.len() % width != 0 {
                    return Err(Error::invalid(format!(
                        "cached buffer of {} bytes is not a whole number of {}-byte elements",
                        bytes.len(),
                        width
                    )));
                }
                Ok(bytes
                    .chunks_exact(width)
                    .map(|chunk| {
                        let mut word = [0u8; std::mem::size_of::<Self>()];
                        word.copy_from_slice(chunk);
                        Self::from_le_bytes(word)
                    })
                    .collect())
            }
        }
    };
}

numeric_element!(u8, Dtype::U8);
numeric_element!(u16, Dtype::U16);
numeric_element!(u32, Dtype::U32);
numeric_element!(u64, Dtype::U64);
numeric_element!(i8, Dtype::I8);
numeric_element!(i16, Dtype::I16);
numeric_element!(i32, Dtype::I32);
numeric_element!(i64, Dtype::I64);
numeric_element!(f32, Dtype::F32);
numeric_element!(f64, Dtype::F64);

impl CacheElement for bool {
    const DTYPE: Dtype = Dtype::Bool;

    fn encode_into(values: &[Self], into: &mut Vec<u8>) {
        into.extend(values.iter().map(|&value| u8::from(value)));
    }

    fn decode_from(bytes: &[u8]) -> Result<Vec<Self>> {
        // Any non-zero byte is `true`. Nothing this crate writes produces one,
        // but a foreign encoder might, and silently returning `false` for `2`
        // would be the worse failure.
        Ok(bytes.iter().map(|&byte| byte != 0).collect())
    }
}

// ---------------------------------------------------------------- fetcher --

/// What the cache calls on a miss. Byte-oriented and type-erased, so one cache
/// serves arrays of different element types.
pub trait ChunkFetcher: Send + Sync {
    /// Read `region` and return its elements in C order, in this crate's
    /// element encoding (see [`CacheElement`]).
    fn fetch(&self, region: &Region) -> Result<Vec<u8>>;

    /// Fast path: `Some(true)` if the region is known empty without reading it.
    ///
    /// The cache acts on this in two ways: it does not read such a region, and
    /// it **does not coalesce across it** — a run of chunks being fetched
    /// together is broken where one is known empty, so the saving is not given
    /// straight back as over-read.
    fn is_known_empty(&self, _region: &Region) -> Option<bool> {
        None
    }
}

/// Any [`RegionSource`] as a [`ChunkFetcher`].
pub struct RegionSourceFetcher<T, S> {
    source: S,
    element: std::marker::PhantomData<fn() -> T>,
}

impl<T, S> RegionSourceFetcher<T, S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            element: std::marker::PhantomData,
        }
    }

    pub fn inner(&self) -> &S {
        &self.source
    }
}

impl<T, S> ChunkFetcher for RegionSourceFetcher<T, S>
where
    T: CacheElement,
    S: RegionSource<T>,
{
    fn fetch(&self, region: &Region) -> Result<Vec<u8>> {
        let array = self.source.read_region(region)?;
        let standard = array.as_standard_layout();
        let values = standard.as_slice().ok_or_else(|| {
            Error::invalid("region source returned an array with no contiguous layout")
        })?;
        let mut bytes = Vec::with_capacity(values.len() * T::DTYPE.size_of());
        T::encode_into(values, &mut bytes);
        Ok(bytes)
    }

    fn is_known_empty(&self, region: &Region) -> Option<bool> {
        self.source.is_known_empty(region)
    }
}

// -------------------------------------------------------------------- key --

/// A registered array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArrayId(pub u32);

/// The cache key: an array, and a chunk's index in that array's lattice.
///
/// Row-major over the chunk grid, so the index is small, `Copy` and totally
/// ordered — and consecutive indices are lattice-adjacent along the last axis,
/// which is what makes coalescing a scan rather than a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub array: ArrayId,
    pub chunk: u64,
}

struct Registered {
    id: ArrayId,
    name: String,
    shape: Vec<usize>,
    chunk: Vec<usize>,
    grid: Vec<usize>,
    dtype: Dtype,
    fetcher: Arc<dyn ChunkFetcher>,
}

impl Registered {
    /// The region one lattice index covers, clamped to the array.
    fn chunk_region(&self, index: u64) -> Region {
        let ndim = self.grid.len();
        let mut start = vec![0usize; ndim];
        let mut shape = vec![0usize; ndim];
        let mut rest = index as usize;
        for axis in (0..ndim).rev() {
            let coord = rest % self.grid[axis];
            rest /= self.grid[axis];
            let lo = coord * self.chunk[axis];
            start[axis] = lo;
            shape[axis] = self.chunk[axis].min(self.shape[axis] - lo);
        }
        Region { start, shape }
    }

    /// Every lattice index `region` touches, in row-major order.
    fn covering(&self, region: &Region) -> Vec<u64> {
        let ndim = self.grid.len();
        let mut per_axis: Vec<Vec<usize>> = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            if region.shape[axis] == 0 {
                return Vec::new();
            }
            let first = region.start[axis] / self.chunk[axis];
            let last = (region.start[axis] + region.shape[axis] - 1) / self.chunk[axis];
            per_axis.push((first..=last).collect());
        }
        let mut out = vec![0u64];
        for axis in 0..ndim {
            let stride = self.grid[axis] as u64;
            let mut next = Vec::with_capacity(out.len() * per_axis[axis].len());
            for &prefix in &out {
                for &coord in &per_axis[axis] {
                    next.push(prefix * stride + coord as u64);
                }
            }
            out = next;
        }
        out.sort_unstable();
        out
    }

    /// The bounding region of a run of consecutive indices.
    fn run_region(&self, run: &[u64]) -> Region {
        let first = self.chunk_region(run[0]);
        let last = self.chunk_region(run[run.len() - 1]);
        let start = first.start.clone();
        let shape = start
            .iter()
            .zip(last.start.iter().zip(last.shape.iter()))
            .map(|(&lo, (&last_lo, &last_len))| last_lo + last_len - lo)
            .collect();
        Region { start, shape }
    }
}

// ------------------------------------------------------------------ state --

struct Entry {
    tier: Tier,
    payload: Vec<u8>,
    decoded_len: usize,
    /// The bytes this entry holds against the budget. Dropping the entry
    /// returns them, on every path including a panic.
    _lease: Lease,
    tick: u64,
    from_prefetch: bool,
    used: bool,
}

#[derive(Default)]
struct State {
    entries: HashMap<ChunkKey, Entry>,
    /// Recency order, oldest first. `BTreeMap` rather than an intrusive list
    /// because eviction wants the minimum and a hit wants a re-key, both
    /// `O(log n)`, and `n` is thousands.
    recency: BTreeMap<u64, ChunkKey>,
    clock: u64,
    bytes: u64,
    /// Chunks somebody is fetching right now. Keeps concurrent demand for one
    /// chunk down to one read.
    pending: HashSet<ChunkKey>,
}

#[derive(Default)]
struct Counters {
    hits_decoded: AtomicU64,
    hits_encoded: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    refusals: AtomicU64,
    known_empty: AtomicU64,
    source_reads: AtomicU64,
    source_bytes: AtomicU64,
    decoded_bytes_seen: AtomicU64,
    encoded_bytes_stored: AtomicU64,
    prefetch_issued: AtomicU64,
    prefetch_used: AtomicU64,
    prefetch_wasted_evicted: AtomicU64,
    prefetch_wasted_refused: AtomicU64,
    prefetch_declined: AtomicU64,
}

/// What the cache did, as counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits_decoded: u64,
    pub hits_encoded: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Chunks served but not retained because the budget refused the bytes.
    pub refusals: u64,
    /// Chunks a backend said were empty, so were never read.
    pub known_empty: u64,
    /// Calls into a `ChunkFetcher`. Lower than `misses` when coalescing worked.
    pub source_reads: u64,
    pub source_bytes: u64,
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted_evicted: u64,
    pub prefetch_wasted_refused: u64,
    /// Prefetches not attempted because compute was waiting on the budget.
    pub prefetch_declined: u64,
    pub resident_bytes: u64,
    pub resident_chunks: u64,
    decoded_bytes_seen: u64,
    encoded_bytes_stored: u64,
}

impl CacheStats {
    pub fn hits(&self) -> u64 {
        self.hits_decoded + self.hits_encoded
    }

    /// Hits as a fraction of chunk lookups. `None` before anything was looked
    /// up, rather than a fabricated zero.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits() + self.misses;
        (total > 0).then(|| self.hits() as f64 / total as f64)
    }

    /// The compression ratio the encoded tier actually achieved, over every
    /// chunk it encoded. This is the number a planner needs to decide whether
    /// `Tier::Encoded` is worth it for a given stage, and it is measured rather
    /// than assumed.
    pub fn encoded_ratio(&self) -> Option<f64> {
        (self.encoded_bytes_stored > 0)
            .then(|| self.decoded_bytes_seen as f64 / self.encoded_bytes_stored as f64)
    }
}

// ------------------------------------------------------------------ cache --

/// One `(array, chunk)`-keyed LRU over every registered array.
pub struct ChunkCache {
    arrays: RwLock<Vec<Arc<Registered>>>,
    policies: RwLock<Vec<ArrayPolicy>>,
    state: Mutex<State>,
    /// Signalled whenever a pending fetch resolves.
    resolved: Condvar,
    budget: MemoryBudget,
    capacity: u64,
    max_coalesce: usize,
    codec: Arc<dyn Codec>,
    listeners: Vec<Arc<dyn EventListener>>,
    counters: Counters,
}

impl ChunkCache {
    /// A cache holding at most `capacity_bytes`, leasing them opportunistically
    /// from `budget`.
    ///
    /// Two bounds, not one, and they do different jobs: `capacity_bytes` is
    /// this cache's own ceiling, so it cannot take the whole machine even when
    /// the budget is idle; the budget is the global guard that keeps it from
    /// competing with compute.
    pub fn new(budget: MemoryBudget, capacity_bytes: u64) -> Self {
        Self {
            arrays: RwLock::new(Vec::new()),
            policies: RwLock::new(Vec::new()),
            state: Mutex::new(State::default()),
            resolved: Condvar::new(),
            budget,
            capacity: capacity_bytes,
            max_coalesce: DEFAULT_MAX_COALESCE,
            codec: Arc::new(DeflateCodec::default()),
            listeners: Vec::new(),
            counters: Counters::default(),
        }
    }

    pub fn with_codec(mut self, codec: Arc<dyn Codec>) -> Self {
        self.codec = codec;
        self
    }

    /// How many lattice-consecutive chunks may be fetched in one request.
    /// `1` disables coalescing.
    pub fn with_max_coalesce(mut self, chunks: usize) -> Self {
        self.max_coalesce = chunks.max(1);
        self
    }

    /// Register listeners. Fixed for the cache's lifetime, like the executor's:
    /// the event path is hot and a registry lock on it would be the contention
    /// the cache exists to remove.
    pub fn with_listeners(mut self, listeners: Vec<Arc<dyn EventListener>>) -> Self {
        self.listeners = listeners;
        self
    }

    /// Register an array. `chunk` is the lattice — usually the storage unit.
    pub fn register(
        &self,
        name: &str,
        shape: &[usize],
        chunk: &[usize],
        dtype: Dtype,
        policy: ArrayPolicy,
        fetcher: Arc<dyn ChunkFetcher>,
    ) -> Result<ArrayId> {
        if chunk.len() != shape.len() {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: chunk.to_vec(),
            });
        }
        if chunk.iter().any(|&extent| extent == 0) || shape.iter().any(|&extent| extent == 0) {
            return Err(Error::invalid(format!(
                "{name}: shape {shape:?} and chunk {chunk:?} must both be positive on every axis"
            )));
        }
        let grid: Vec<usize> = shape
            .iter()
            .zip(chunk.iter())
            .map(|(&dim, &extent)| dim.div_ceil(extent))
            .collect();

        let mut arrays = self.arrays.write().unwrap_or_else(|p| p.into_inner());
        let mut policies = self.policies.write().unwrap_or_else(|p| p.into_inner());
        let id = ArrayId(arrays.len() as u32);
        arrays.push(Arc::new(Registered {
            id,
            name: name.to_string(),
            shape: shape.to_vec(),
            chunk: chunk.to_vec(),
            grid,
            dtype,
            fetcher,
        }));
        policies.push(policy);
        Ok(id)
    }

    /// Change an array's policy mid-run. The planner's lever: a stage that has
    /// stopped being read can be demoted without dropping what it already holds.
    pub fn set_policy(&self, array: ArrayId, policy: ArrayPolicy) -> Result<()> {
        let mut policies = self.policies.write().unwrap_or_else(|p| p.into_inner());
        let slot = policies
            .get_mut(array.0 as usize)
            .ok_or_else(|| Error::invalid(format!("no array {}", array.0)))?;
        *slot = policy;
        Ok(())
    }

    pub fn policy(&self, array: ArrayId) -> Option<ArrayPolicy> {
        self.policies
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(array.0 as usize)
            .copied()
    }

    fn registered(&self, array: ArrayId) -> Result<Arc<Registered>> {
        self.arrays
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(array.0 as usize)
            .cloned()
            .ok_or_else(|| Error::invalid(format!("no array {}", array.0)))
    }

    /// The shape an array was registered with.
    pub fn shape_of(&self, array: ArrayId) -> Option<Vec<usize>> {
        self.registered(array).ok().map(|reg| reg.shape.clone())
    }

    fn emit(&self, events: Vec<Event>) {
        if self.listeners.is_empty() {
            return;
        }
        for event in events {
            for listener in &self.listeners {
                // A listener may observe; it may never influence. A panicking
                // one is contained here for the same reason the executor
                // contains it: a broken observer must not break the run.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    listener.on_event(&event)
                }));
            }
        }
    }

    // ----------------------------------------------------------- reading --

    /// Read `region` of `array`, serving whatever the cache already holds.
    ///
    /// The returned bytes are byte-identical to what the array's fetcher would
    /// return for the same region, whichever tier served them and whatever the
    /// cache's state — which is the whole correctness claim, and is swept in
    /// `cache_tests`.
    pub fn read_region_bytes(&self, array: ArrayId, region: &Region) -> Result<Vec<u8>> {
        let reg = self.registered(array)?;
        region.check_within(&reg.shape, &reg.name)?;
        let element = reg.dtype.size_of();
        let mut out = vec![0u8; region.voxels() * element];
        let covering = reg.covering(region);
        let mut missing = Vec::new();
        for chunk in covering {
            if !self.serve_from_cache(&reg, chunk, region, &mut out, element)? {
                missing.push(chunk);
            }
        }
        if !missing.is_empty() {
            self.fill(&reg, &missing, Some((region, &mut out)), false)?;
        }
        Ok(out)
    }

    /// Read a region as a typed array. What [`CachingSource`] is built on.
    pub fn read_region<T: CacheElement>(
        &self,
        array: ArrayId,
        region: &Region,
    ) -> Result<ArrayD<T>> {
        let bytes = self.read_region_bytes(array, region)?;
        let values = T::decode_from(&bytes)?;
        ArrayD::from_shape_vec(IxDyn(&region.shape), values)
            .map_err(|err| Error::invalid(format!("cached region does not fit its shape: {err}")))
    }

    /// Try to serve one chunk from the cache into `out`. `Ok(false)` is a miss.
    fn serve_from_cache(
        &self,
        reg: &Registered,
        chunk: u64,
        want: &Region,
        out: &mut [u8],
        element: usize,
    ) -> Result<bool> {
        let key = ChunkKey {
            array: reg.id,
            chunk,
        };
        let chunk_region = reg.chunk_region(chunk);
        let mut events = Vec::new();

        // Under the lock: touch recency, and for the decoded tier do the copy
        // right here. That copy is the *whole cost* of a decoded hit, and doing
        // it here rather than after cloning the payload out is what keeps the
        // claim "a hit is one memcpy" true.
        let encoded_payload = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let Some(entry) = state.entries.get(&key) else {
                return Ok(false);
            };
            let tier = entry.tier;
            let was_unused_prefetch = entry.from_prefetch && !entry.used;
            let bytes = entry.payload.len() as u64;
            let payload = match tier {
                Tier::Decoded => {
                    copy_overlap(&entry.payload, &chunk_region, out, want, element);
                    None
                }
                Tier::Encoded => Some((entry.payload.clone(), entry.decoded_len)),
            };

            let tick = state.clock + 1;
            state.clock = tick;
            let previous = state.entries.get_mut(&key).map(|entry| {
                let previous = entry.tick;
                entry.tick = tick;
                entry.used = true;
                previous
            });
            if let Some(previous) = previous {
                state.recency.remove(&previous);
                state.recency.insert(tick, key);
            }

            match tier {
                Tier::Decoded => self.counters.hits_decoded.fetch_add(1, Ordering::Relaxed),
                Tier::Encoded => self.counters.hits_encoded.fetch_add(1, Ordering::Relaxed),
            };
            if was_unused_prefetch {
                self.counters.prefetch_used.fetch_add(1, Ordering::Relaxed);
                events.push(Event::PrefetchUsed {
                    array: reg.name.clone(),
                    chunk,
                    waited_ns: 0,
                });
            }
            events.push(Event::CacheHit {
                array: reg.name.clone(),
                chunk,
                tier,
                bytes,
                decode_ns: 0,
            });
            payload
        };

        // Outside the lock: decompress, then copy. Holding the cache mutex
        // across a decompression would make one slow entry everybody's problem.
        if let Some((payload, decoded_len)) = encoded_payload {
            let started = Instant::now();
            let decoded = self.codec.decode(&payload, decoded_len)?;
            let decode_ns = started.elapsed().as_nanos() as u64;
            copy_overlap(&decoded, &chunk_region, out, want, element);
            if let Some(Event::CacheHit {
                decode_ns: slot, ..
            }) = events.last_mut()
            {
                *slot = decode_ns;
            }
        }

        self.emit(events);
        Ok(true)
    }

    /// Fetch `chunks`, optionally copying them into a caller's buffer.
    ///
    /// `target` is `None` for a prefetch, which wants the entries in the cache
    /// and has nowhere to put the bytes.
    fn fill(
        &self,
        reg: &Registered,
        chunks: &[u64],
        mut target: Option<(&Region, &mut Vec<u8>)>,
        from_prefetch: bool,
    ) -> Result<()> {
        let element = reg.dtype.size_of();
        let policy = self.policy(reg.id).unwrap_or_default();
        let tier = if from_prefetch {
            policy.prefetch_tier
        } else {
            policy.demand_tier
        };

        // 1. Drop what the backend already knows is empty. The output buffer is
        //    zeroed, so an empty chunk needs no copy — and, more importantly,
        //    removing it here is what breaks the coalescing run at step 3, so
        //    the saving is not handed straight back as over-read.
        let mut wanted = Vec::with_capacity(chunks.len());
        let mut events = Vec::new();
        for &chunk in chunks {
            let region = reg.chunk_region(chunk);
            if reg.fetcher.is_known_empty(&region) == Some(true) {
                self.counters.known_empty.fetch_add(1, Ordering::Relaxed);
                events.push(Event::CacheKnownEmpty {
                    array: reg.name.clone(),
                    chunk,
                    bytes: (region.voxels() * element) as u64,
                });
                continue;
            }
            wanted.push(chunk);
        }
        self.emit(std::mem::take(&mut events));

        // 2. Claim what nobody else is already fetching.
        let (claimed, waited) = self.claim(reg.id, &wanted);

        // 3. Fetch the claimed chunks, coalescing lattice-consecutive runs.
        for run in runs(&claimed, reg.grid[reg.grid.len() - 1], self.max_coalesce) {
            let region = reg.run_region(&run);
            let started = Instant::now();
            let fetched = match reg.fetcher.fetch(&region) {
                Ok(bytes) => bytes,
                Err(err) => {
                    self.release(reg.id, &run);
                    return Err(err);
                }
            };
            let duration_ns = started.elapsed().as_nanos() as u64;
            self.counters.source_reads.fetch_add(1, Ordering::Relaxed);
            self.counters
                .source_bytes
                .fetch_add(fetched.len() as u64, Ordering::Relaxed);

            let expected = region.voxels() * element;
            if fetched.len() != expected {
                self.release(reg.id, &run);
                return Err(Error::ShapeMismatch {
                    expected: vec![expected],
                    got: vec![fetched.len()],
                });
            }

            let per_chunk = duration_ns / run.len().max(1) as u64;
            for &chunk in &run {
                let chunk_region = reg.chunk_region(chunk);
                // One buffer per chunk, because that is the unit of eviction.
                // When the run is a single chunk this is a move, not a copy.
                let decoded = if run.len() == 1 {
                    fetched.clone()
                } else {
                    let mut buffer = vec![0u8; chunk_region.voxels() * element];
                    copy_overlap(&fetched, &region, &mut buffer, &chunk_region, element);
                    buffer
                };
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                self.emit(vec![Event::CacheMiss {
                    array: reg.name.clone(),
                    chunk,
                    bytes: decoded.len() as u64,
                    duration_ns: per_chunk,
                }]);
                if let Some((want, out)) = target.as_mut() {
                    copy_overlap(&decoded, &chunk_region, out, want, element);
                }
                self.retain(reg, chunk, decoded, tier, from_prefetch);
            }
        }

        // 4. Chunks somebody else was already fetching. Waiting is the point —
        //    it is how concurrent demand for one chunk costs one read — but it
        //    is bounded, so a fetcher that dies mid-flight costs a duplicated
        //    read rather than a hang.
        for chunk in waited {
            let key = ChunkKey {
                array: reg.id,
                chunk,
            };
            let started = Instant::now();
            {
                let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                while state.pending.contains(&key) {
                    let (next, timeout) = self
                        .resolved
                        .wait_timeout(state, PENDING_WAIT)
                        .unwrap_or_else(|p| p.into_inner());
                    state = next;
                    if timeout.timed_out() {
                        state.pending.remove(&key);
                        break;
                    }
                }
            }
            let waited_ns = started.elapsed().as_nanos() as u64;
            let served = if let Some((want, out)) = target.as_mut() {
                self.serve_from_cache(reg, chunk, want, out, element)?
            } else {
                self.state
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .entries
                    .contains_key(&key)
            };
            if served {
                if waited_ns > 0 {
                    self.emit(vec![Event::PrefetchUsed {
                        array: reg.name.clone(),
                        chunk,
                        waited_ns,
                    }]);
                }
                continue;
            }
            // The other fetch did not leave anything behind — refused by the
            // budget, evicted immediately, or it died. Do it ourselves.
            let (mine, _) = self.claim(reg.id, &[chunk]);
            if mine.is_empty() {
                // Somebody claimed it again in the meantime; serve it straight
                // from the source rather than waiting a second time.
                let region = reg.chunk_region(chunk);
                let decoded = reg.fetcher.fetch(&region)?;
                if let Some((want, out)) = target.as_mut() {
                    copy_overlap(&decoded, &region, out, want, element);
                }
                continue;
            }
            self.fill_one(reg, chunk, &mut target, tier, from_prefetch, element)?;
        }
        Ok(())
    }

    fn fill_one(
        &self,
        reg: &Registered,
        chunk: u64,
        target: &mut Option<(&Region, &mut Vec<u8>)>,
        tier: Tier,
        from_prefetch: bool,
        element: usize,
    ) -> Result<()> {
        let region = reg.chunk_region(chunk);
        let started = Instant::now();
        let decoded = match reg.fetcher.fetch(&region) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.release(reg.id, &[chunk]);
                return Err(err);
            }
        };
        self.counters.source_reads.fetch_add(1, Ordering::Relaxed);
        self.counters
            .source_bytes
            .fetch_add(decoded.len() as u64, Ordering::Relaxed);
        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        self.emit(vec![Event::CacheMiss {
            array: reg.name.clone(),
            chunk,
            bytes: decoded.len() as u64,
            duration_ns: started.elapsed().as_nanos() as u64,
        }]);
        if let Some((want, out)) = target.as_mut() {
            copy_overlap(&decoded, &region, out, want, element);
        }
        self.retain(reg, chunk, decoded, tier, from_prefetch);
        Ok(())
    }

    /// Mark chunks pending. Returns `(claimed, already pending elsewhere)`.
    fn claim(&self, array: ArrayId, chunks: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let mut claimed = Vec::new();
        let mut waited = Vec::new();
        for &chunk in chunks {
            let key = ChunkKey { array, chunk };
            // Somebody may have inserted it since the caller looked.
            if state.entries.contains_key(&key) {
                continue;
            }
            if state.pending.insert(key) {
                claimed.push(chunk);
            } else {
                waited.push(chunk);
            }
        }
        (claimed, waited)
    }

    fn release(&self, array: ArrayId, chunks: &[u64]) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        for &chunk in chunks {
            state.pending.remove(&ChunkKey { array, chunk });
        }
        self.resolved.notify_all();
    }

    /// Offer a freshly fetched chunk to the cache, and clear its claim.
    ///
    /// Everything about admission happens under one lock — eviction, the
    /// opportunistic lease, and the insert — so the resident byte count and the
    /// leased byte count cannot disagree even for an instant.
    fn retain(
        &self,
        reg: &Registered,
        chunk: u64,
        decoded: Vec<u8>,
        tier: Tier,
        from_prefetch: bool,
    ) {
        let key = ChunkKey {
            array: reg.id,
            chunk,
        };
        let policy = self.policy(reg.id).unwrap_or_default();
        let decoded_len = decoded.len();

        // Compression happens outside the lock: it is the one part of admission
        // whose cost is proportional to the data.
        let payload = match tier {
            Tier::Decoded => decoded,
            Tier::Encoded => {
                let encoded = self.codec.encode(&decoded);
                self.counters
                    .decoded_bytes_seen
                    .fetch_add(decoded_len as u64, Ordering::Relaxed);
                self.counters
                    .encoded_bytes_stored
                    .fetch_add(encoded.len() as u64, Ordering::Relaxed);
                encoded
            }
        };
        let cost = payload.len() as u64;

        let mut events = Vec::new();
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.pending.remove(&key);

            let admissible = policy.retain && cost > 0 && cost <= self.capacity;
            if admissible {
                while state.bytes + cost > self.capacity {
                    if !self.evict_one(&mut state, &reg.name, &mut events) {
                        break;
                    }
                }
            }
            let lease = if admissible && state.bytes + cost <= self.capacity {
                self.budget.try_acquire_opportunistic(cost)
            } else {
                None
            };
            match lease {
                Some(lease) => {
                    let tick = state.clock + 1;
                    state.clock = tick;
                    state.recency.insert(tick, key);
                    state.bytes += cost;
                    state.entries.insert(
                        key,
                        Entry {
                            tier,
                            payload,
                            decoded_len,
                            _lease: lease,
                            tick,
                            from_prefetch,
                            used: false,
                        },
                    );
                }
                None => {
                    if policy.retain {
                        self.counters.refusals.fetch_add(1, Ordering::Relaxed);
                        events.push(Event::CacheRefused {
                            array: reg.name.clone(),
                            chunk,
                            bytes: cost,
                        });
                        if from_prefetch {
                            self.counters
                                .prefetch_wasted_refused
                                .fetch_add(1, Ordering::Relaxed);
                            events.push(Event::PrefetchWasted {
                                array: reg.name.clone(),
                                chunk,
                                reason: PrefetchWaste::Refused,
                            });
                        }
                    }
                }
            }
            self.resolved.notify_all();
        }
        self.emit(events);
    }

    /// Drop the least recently used entry, whichever array it belongs to.
    ///
    /// **This is the line the whole file exists for.** There is nothing here
    /// that consults `key.array`, which is precisely why capacity follows
    /// demand across arrays instead of being partitioned between them up front.
    fn evict_one(&self, state: &mut State, for_array: &str, events: &mut Vec<Event>) -> bool {
        let Some((&tick, &key)) = state.recency.iter().next() else {
            return false;
        };
        state.recency.remove(&tick);
        let Some(entry) = state.entries.remove(&key) else {
            return false;
        };
        let bytes = entry.payload.len() as u64;
        state.bytes = state.bytes.saturating_sub(bytes);
        let prefetched_unused = entry.from_prefetch && !entry.used;
        let name = self.name_of(key.array);
        self.counters.evictions.fetch_add(1, Ordering::Relaxed);
        events.push(Event::CacheEvicted {
            array: name.clone(),
            chunk: key.chunk,
            tier: entry.tier,
            bytes,
            for_array: for_array.to_string(),
            prefetched_unused,
        });
        if prefetched_unused {
            self.counters
                .prefetch_wasted_evicted
                .fetch_add(1, Ordering::Relaxed);
            events.push(Event::PrefetchWasted {
                array: name,
                chunk: key.chunk,
                reason: PrefetchWaste::Evicted,
            });
        }
        // `entry` drops here, returning its lease's bytes to the budget.
        true
    }

    fn name_of(&self, array: ArrayId) -> String {
        self.arrays
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(array.0 as usize)
            .map(|reg| reg.name.clone())
            .unwrap_or_default()
    }

    // ---------------------------------------------------------- prefetch --

    /// Populate the cache with `region` of `array`, on behalf of a plan.
    ///
    /// Returns how many chunks were fetched. **Declines rather than waits**
    /// whenever compute is queueing for the budget, which is the property that
    /// makes the prefetcher safe to run at any depth.
    pub fn prefetch_region(&self, array: ArrayId, region: &Region, rank: u32) -> Result<usize> {
        let reg = self.registered(array)?;
        region.check_within(&reg.shape, &reg.name)?;
        if self.budget.revoking() {
            self.counters
                .prefetch_declined
                .fetch_add(1, Ordering::Relaxed);
            return Ok(0);
        }
        let missing: Vec<u64> = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            reg.covering(region)
                .into_iter()
                .filter(|&chunk| {
                    let key = ChunkKey { array, chunk };
                    !state.entries.contains_key(&key) && !state.pending.contains(&key)
                })
                .collect()
        };
        if missing.is_empty() {
            return Ok(0);
        }
        self.counters
            .prefetch_issued
            .fetch_add(missing.len() as u64, Ordering::Relaxed);
        self.emit(
            missing
                .iter()
                .map(|&chunk| Event::PrefetchIssued {
                    array: reg.name.clone(),
                    chunk,
                    rank,
                })
                .collect(),
        );
        self.fill(&reg, &missing, None, true)?;
        Ok(missing.len())
    }

    /// Abandon the prefetched-but-unread entries of an array.
    ///
    /// What a cancelled plan leaves behind. Counted as waste, because that is
    /// what it is — the depth was wrong, or the plan was.
    pub fn drop_unused_prefetches(&self, array: ArrayId) -> usize {
        let name = self.name_of(array);
        let mut events = Vec::new();
        let mut dropped = 0;
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let doomed: Vec<ChunkKey> = state
                .entries
                .iter()
                .filter(|(key, entry)| key.array == array && entry.from_prefetch && !entry.used)
                .map(|(key, _)| *key)
                .collect();
            for key in doomed {
                if let Some(entry) = state.entries.remove(&key) {
                    state.recency.remove(&entry.tick);
                    state.bytes = state.bytes.saturating_sub(entry.payload.len() as u64);
                    dropped += 1;
                    self.counters
                        .prefetch_wasted_evicted
                        .fetch_add(1, Ordering::Relaxed);
                    events.push(Event::PrefetchWasted {
                        array: name.clone(),
                        chunk: key.chunk,
                        reason: PrefetchWaste::Cancelled,
                    });
                }
            }
        }
        self.emit(events);
        dropped
    }

    // ------------------------------------------------------------- state --

    pub fn stats(&self) -> CacheStats {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let counters = &self.counters;
        CacheStats {
            hits_decoded: counters.hits_decoded.load(Ordering::Relaxed),
            hits_encoded: counters.hits_encoded.load(Ordering::Relaxed),
            misses: counters.misses.load(Ordering::Relaxed),
            evictions: counters.evictions.load(Ordering::Relaxed),
            refusals: counters.refusals.load(Ordering::Relaxed),
            known_empty: counters.known_empty.load(Ordering::Relaxed),
            source_reads: counters.source_reads.load(Ordering::Relaxed),
            source_bytes: counters.source_bytes.load(Ordering::Relaxed),
            prefetch_issued: counters.prefetch_issued.load(Ordering::Relaxed),
            prefetch_used: counters.prefetch_used.load(Ordering::Relaxed),
            prefetch_wasted_evicted: counters.prefetch_wasted_evicted.load(Ordering::Relaxed),
            prefetch_wasted_refused: counters.prefetch_wasted_refused.load(Ordering::Relaxed),
            prefetch_declined: counters.prefetch_declined.load(Ordering::Relaxed),
            resident_bytes: state.bytes,
            resident_chunks: state.entries.len() as u64,
            decoded_bytes_seen: counters.decoded_bytes_seen.load(Ordering::Relaxed),
            encoded_bytes_stored: counters.encoded_bytes_stored.load(Ordering::Relaxed),
        }
    }

    /// Bytes currently held. Equal, always, to the bytes leased.
    pub fn resident_bytes(&self) -> u64 {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).bytes
    }

    /// How many of `array`'s chunks are resident. The cross-array eviction
    /// evidence is read off this.
    pub fn resident_chunks(&self, array: ArrayId) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .keys()
            .filter(|key| key.array == array)
            .count()
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Drop everything, returning every lease.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.entries.clear();
        state.recency.clear();
        state.bytes = 0;
    }
}

// ------------------------------------------------------------ read-through --

/// A [`RegionSource`] that reads through the cache.
///
/// The seam that lets an existing consumer gain caching without knowing it has:
/// same trait, same regions, same bytes.
pub struct CachingSource<T> {
    cache: Arc<ChunkCache>,
    array: ArrayId,
    shape: Vec<usize>,
    name: String,
    element: std::marker::PhantomData<fn() -> T>,
}

impl<T: CacheElement> CachingSource<T> {
    /// Register `source` with `cache` and return a source that reads through it.
    pub fn attach<S: RegionSource<T> + 'static>(
        cache: Arc<ChunkCache>,
        name: &str,
        chunk: &[usize],
        policy: ArrayPolicy,
        source: S,
    ) -> Result<Self> {
        let shape = source.shape().to_vec();
        let array = cache.register(
            name,
            &shape,
            chunk,
            T::DTYPE,
            policy,
            Arc::new(RegionSourceFetcher::<T, S>::new(source)),
        )?;
        Ok(Self {
            cache,
            array,
            shape,
            name: name.to_string(),
            element: std::marker::PhantomData,
        })
    }

    pub fn array(&self) -> ArrayId {
        self.array
    }

    pub fn cache(&self) -> &Arc<ChunkCache> {
        &self.cache
    }
}

impl<T: CacheElement> RegionSource<T> for CachingSource<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        self.cache.read_region::<T>(self.array, region)
    }

    fn describe(&self) -> String {
        format!("cached {} {:?}", self.name, self.shape)
    }
}

// ----------------------------------------------------------------- boxes --

/// Row-major strides, in elements.
fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// Copy the part of `src` (laid out as `src_region`) that lies inside
/// `dst_region` into `dst`.
///
/// Row-wise: the last axis of both buffers is contiguous, so the inner
/// operation is a `memcpy` of `overlap.shape[last]` elements and the loop is
/// over everything above it. That is the only shape of copy this file needs,
/// and writing it once is what keeps region arithmetic in one place.
fn copy_overlap(
    src: &[u8],
    src_region: &Region,
    dst: &mut [u8],
    dst_region: &Region,
    element: usize,
) {
    let Some(overlap) = src_region.intersect(dst_region) else {
        return;
    };
    let ndim = overlap.ndim();
    if ndim == 0 || overlap.voxels() == 0 {
        return;
    }
    let last = ndim - 1;
    let src_strides = c_strides(&src_region.shape);
    let dst_strides = c_strides(&dst_region.shape);
    let row = overlap.shape[last] * element;

    let outer: usize = overlap.shape[..last].iter().product();
    let mut counter = vec![0usize; last];
    for _ in 0..outer {
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;
        for axis in 0..last {
            let coord = overlap.start[axis] + counter[axis];
            src_offset += (coord - src_region.start[axis]) * src_strides[axis];
            dst_offset += (coord - dst_region.start[axis]) * dst_strides[axis];
        }
        src_offset += (overlap.start[last] - src_region.start[last]) * src_strides[last];
        dst_offset += (overlap.start[last] - dst_region.start[last]) * dst_strides[last];
        let src_byte = src_offset * element;
        let dst_byte = dst_offset * element;
        dst[dst_byte..dst_byte + row].copy_from_slice(&src[src_byte..src_byte + row]);

        for axis in (0..last).rev() {
            counter[axis] += 1;
            if counter[axis] < overlap.shape[axis] {
                break;
            }
            counter[axis] = 0;
        }
    }
}

/// Group sorted lattice indices into runs that are consecutive **and do not
/// wrap the last axis**, capped at `limit`.
///
/// Consecutive indices are adjacent along the last axis by construction of the
/// row-major key, so a run is a single box and one request fetches it. The wrap
/// check matters: index `k*width - 1` and `k*width` are consecutive numbers and
/// opposite corners of the array.
///
/// `chunks` arriving here has already had known-empty entries removed, so a run
/// **cannot span a region a backend has ruled out** — the coalescing and the
/// empty-block fast path do not fight each other.
fn runs(chunks: &[u64], width: usize, limit: usize) -> Vec<Vec<u64>> {
    let width = width.max(1) as u64;
    let mut sorted = chunks.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<Vec<u64>> = Vec::new();
    for chunk in sorted {
        match out.last_mut() {
            Some(run)
                if *run.last().expect("runs are never empty") + 1 == chunk
                    && chunk % width != 0
                    && run.len() < limit =>
            {
                run.push(chunk);
            }
            _ => out.push(vec![chunk]),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_strides(&[5]), vec![1]);
        assert_eq!(c_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn a_copy_of_the_whole_box_is_the_identity() {
        let src: Vec<u8> = (0..24).collect();
        let region = Region::new(&[0, 0, 0], &[2, 3, 4]);
        let mut dst = vec![0u8; 24];
        copy_overlap(&src, &region, &mut dst, &region, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn a_copy_lands_the_overlap_at_the_right_offsets() {
        // A 4x4 source at the origin, a 2x2 destination at (1,1).
        let src: Vec<u8> = (0..16).collect();
        let src_region = Region::new(&[0, 0], &[4, 4]);
        let dst_region = Region::new(&[1, 1], &[2, 2]);
        let mut dst = vec![0u8; 4];
        copy_overlap(&src, &src_region, &mut dst, &dst_region, 1);
        assert_eq!(dst, vec![5, 6, 9, 10]);
    }

    #[test]
    fn a_copy_between_offset_boxes_takes_only_the_overlap() {
        // Source covers rows 2..4, destination rows 3..5: one row in common.
        let src: Vec<u8> = vec![10, 11, 12, 13, 20, 21, 22, 23];
        let src_region = Region::new(&[2, 0], &[2, 4]);
        let dst_region = Region::new(&[3, 1], &[2, 2]);
        let mut dst = vec![0u8; 4];
        copy_overlap(&src, &src_region, &mut dst, &dst_region, 1);
        assert_eq!(dst, vec![21, 22, 0, 0]);
    }

    #[test]
    fn a_copy_of_wide_elements_moves_whole_elements() {
        let src: Vec<u8> = vec![1, 0, 2, 0, 3, 0, 4, 0];
        let src_region = Region::new(&[0], &[4]);
        let dst_region = Region::new(&[1], &[2]);
        let mut dst = vec![0u8; 4];
        copy_overlap(&src, &src_region, &mut dst, &dst_region, 2);
        assert_eq!(dst, vec![2, 0, 3, 0]);
    }

    #[test]
    fn disjoint_boxes_copy_nothing() {
        let src = vec![1u8; 4];
        let mut dst = vec![0u8; 4];
        copy_overlap(
            &src,
            &Region::new(&[0, 0], &[2, 2]),
            &mut dst,
            &Region::new(&[9, 9], &[2, 2]),
            1,
        );
        assert_eq!(dst, vec![0; 4]);
    }

    #[test]
    fn runs_are_consecutive_and_never_wrap_the_last_axis() {
        // Width 4: 3 and 4 are consecutive numbers but opposite edges.
        assert_eq!(
            runs(&[0, 1, 2, 3, 4, 5], 4, 8),
            vec![vec![0, 1, 2, 3], vec![4, 5]]
        );
        assert_eq!(runs(&[0, 2, 4], 4, 8), vec![vec![0], vec![2], vec![4]]);
        assert_eq!(runs(&[5, 6, 7], 4, 8), vec![vec![5, 6, 7]]);
    }

    #[test]
    fn the_coalesce_limit_caps_a_run() {
        assert_eq!(runs(&[0, 1, 2, 3], 16, 2), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(
            runs(&[0, 1, 2, 3], 16, 1),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn every_element_type_round_trips_its_bytes_exactly() {
        fn check<T: CacheElement + PartialEq + std::fmt::Debug>(values: &[T]) {
            let mut bytes = Vec::new();
            T::encode_into(values, &mut bytes);
            assert_eq!(bytes.len(), values.len() * T::DTYPE.size_of());
            assert_eq!(T::decode_from(&bytes).unwrap(), values.to_vec());
        }
        check(&[true, false, true]);
        check(&[0u8, 1, 255]);
        check(&[0u16, 1, 65535]);
        check(&[0u32, 1, u32::MAX]);
        check(&[0u64, 1, u64::MAX]);
        check(&[i8::MIN, 0, i8::MAX]);
        check(&[i16::MIN, 0, i16::MAX]);
        check(&[i32::MIN, 0, i32::MAX]);
        check(&[i64::MIN, 0, i64::MAX]);
        check(&[f32::MIN, 0.0, f32::MAX, -0.0]);
        check(&[f64::MIN, 0.0, f64::MAX, -0.0]);
    }

    #[test]
    fn the_default_codec_round_trips_and_compresses_a_run_heavy_buffer() {
        let codec = DeflateCodec::default();
        // What a `bool` volume looks like: mostly one value.
        let mut decoded = vec![0u8; 1 << 16];
        for index in (0..decoded.len()).step_by(997) {
            decoded[index] = 1;
        }
        let encoded = codec.encode(&decoded);
        assert_eq!(codec.decode(&encoded, decoded.len()).unwrap(), decoded);
        let ratio = decoded.len() as f64 / encoded.len() as f64;
        assert!(
            ratio > 10.0,
            "a run-heavy buffer should compress hard, got {ratio:.1}x"
        );
    }

    #[test]
    fn a_decode_of_the_wrong_length_is_an_error_rather_than_a_short_buffer() {
        let codec = DeflateCodec::default();
        let encoded = codec.encode(&[1u8, 2, 3, 4]);
        assert!(codec.decode(&encoded, 8).is_err());
    }
}
