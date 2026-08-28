// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What has to be true of a cache, and how each claim is checked
// =============================================================
//
// | claim | why it is the claim that matters | how |
// |---|---|---|
// | **A cached read is byte-identical to an uncached one.** | Correctness is invisible: a cache that returns *almost* the right voxels produces a complete, well-formed, wrong volume. Nothing downstream would notice. | Swept over both tiers, cold and warm, aligned and unaligned, against the same source read directly. |
// | **Eviction crosses arrays.** | This is the entire reason for building our own rather than using a per-array one. A cache that cannot move capacity between arrays is the thing we already had. | A hot array and a cold one over one capacity; the cold one's entries must go. |
// | **The budget is respected, and starvation degrades rather than fails.** | A cache that can exceed its lease is a memory leak with extra steps; one that errors under pressure turns a memory setting into a correctness cliff. | Resident bytes are compared to the budget's own opportunistic total, and a deliberately starved budget is asserted to produce *misses*, not errors. |
// | **Prefetch never blocks compute.** | Measurable, so it is measured. An assertion that it "does not block" would be a comment. | A slow source, a deep queue, and a wall-clock measurement of a demand read taken while the queue is full. |
//
// Two of these are properties a *wrong* implementation would still appear to
// satisfy under a lenient test, so they are set up to fail loudly:
// `cross_array_pressure_moves_capacity` fails on any per-array partitioning,
// and `a_starved_budget_degrades_to_misses_rather_than_failing` fails if
// retention is attempted without a lease.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};

use crate::budget::MemoryBudget;
use crate::cache::{ArrayId, ArrayPolicy, CachingSource, ChunkCache, RegionSourceFetcher, Tier};
use crate::dtype::Dtype;
use crate::error::Result;
use crate::listener::EventListener;
use crate::log::Event;
use crate::prefetch::{BlockPlan, PlanHandle, Prefetcher, RegionRequest};
use crate::region::{Region, RegionSource};

// ------------------------------------------------------------- test double --

/// An in-memory source that counts its reads, can be made slow, and can be
/// taught that some regions are empty.
///
/// Owning rather than borrowing because the cache holds its fetchers in an
/// `Arc<dyn ChunkFetcher>` and therefore needs `'static`.
struct Probe<T> {
    volume: ArrayD<T>,
    reads: AtomicUsize,
    voxels_read: AtomicUsize,
    delay: Duration,
    /// Regions whose lower corner is in here are reported empty without a read.
    empty_at: Vec<Vec<usize>>,
}

impl<T: Clone + Send + Sync> Probe<T> {
    fn new(volume: ArrayD<T>) -> Self {
        Self {
            volume,
            reads: AtomicUsize::new(0),
            voxels_read: AtomicUsize::new(0),
            delay: Duration::ZERO,
            empty_at: Vec::new(),
        }
    }

    fn slow(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn empty_at(mut self, starts: &[&[usize]]) -> Self {
        self.empty_at = starts.iter().map(|start| start.to_vec()).collect();
        self
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::Relaxed)
    }
}

impl<T: Clone + Send + Sync> RegionSource<T> for Probe<T> {
    fn shape(&self) -> &[usize] {
        self.volume.shape()
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        region.check_within(self.volume.shape(), "probe")?;
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.voxels_read
            .fetch_add(region.voxels(), Ordering::Relaxed);
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        let mut view = self.volume.view();
        for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
            view.slice_axis_inplace(
                ndarray::Axis(axis),
                ndarray::Slice::from(start..start + len),
            );
        }
        Ok(view.to_owned())
    }

    fn describe(&self) -> String {
        format!("probe {:?}", self.volume.shape())
    }

    fn is_known_empty(&self, region: &Region) -> Option<bool> {
        if self.empty_at.is_empty() {
            return None;
        }
        Some(self.empty_at.contains(&region.start))
    }
}

/// A shared handle onto a probe, so a test can read its counters after handing
/// ownership to the cache.
fn probe_pair<T: Clone + Send + Sync + 'static>(probe: Probe<T>) -> (Arc<Probe<T>>, Arc<Probe<T>>) {
    let shared = Arc::new(probe);
    (Arc::clone(&shared), shared)
}

impl<T: Clone + Send + Sync> RegionSource<T> for Arc<Probe<T>> {
    fn shape(&self) -> &[usize] {
        (**self).shape()
    }
    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        (**self).read_region(region)
    }
    fn describe(&self) -> String {
        (**self).describe()
    }
    fn is_known_empty(&self, region: &Region) -> Option<bool> {
        (**self).is_known_empty(region)
    }
}

fn ramp_u16(shape: [usize; 3]) -> ArrayD<u16> {
    let total: usize = shape.iter().product();
    ArrayD::from_shape_vec(
        IxDyn(&shape),
        (0..total).map(|index| (index % 60013) as u16).collect(),
    )
    .expect("ramp fits its shape")
}

fn sparse_bool(shape: [usize; 3], every: usize) -> ArrayD<bool> {
    let total: usize = shape.iter().product();
    ArrayD::from_shape_vec(
        IxDyn(&shape),
        (0..total).map(|index| index % every == 0).collect(),
    )
    .expect("volume fits its shape")
}

/// A listener that keeps every event, for the tests that assert on the stream.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<Event>>,
}

impl Recorder {
    fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn count(&self, predicate: impl Fn(&Event) -> bool) -> usize {
        self.events()
            .iter()
            .filter(|event| predicate(event))
            .count()
    }
}

impl EventListener for Recorder {
    fn on_event(&self, event: &Event) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }
}

/// A generous budget, for tests that are not about the budget.
fn roomy() -> MemoryBudget {
    MemoryBudget::new(1 << 30)
}

// ------------------------------------------------------ correctness sweep --

/// The claim the whole thing rests on, swept.
///
/// Both tiers, cold and warm, and regions that are aligned to the lattice,
/// straddle it, sit inside one chunk, and cover the whole volume. Every one is
/// compared against the *same source read directly*, so the comparison cannot
/// be satisfied by both paths being wrong in the same way.
#[test]
fn a_cached_read_is_byte_identical_to_an_uncached_one_over_both_tiers() {
    let shape = [9usize, 7, 11];
    let volume = ramp_u16(shape);
    let chunk = [4usize, 3, 5];

    let regions = vec![
        // whole volume
        Region::whole(&shape),
        // exactly one chunk
        Region::new(&[0, 0, 0], &[4, 3, 5]),
        // exactly one interior chunk
        Region::new(&[4, 3, 5], &[4, 3, 5]),
        // straddling two chunks on every axis
        Region::new(&[2, 1, 3], &[5, 4, 6]),
        // wholly inside one chunk, offset
        Region::new(&[1, 1, 1], &[2, 1, 3]),
        // a single voxel
        Region::new(&[8, 6, 10], &[1, 1, 1]),
        // a slab spanning the last axis
        Region::new(&[3, 0, 0], &[1, 7, 11]),
        // the short chunks at the far corner
        Region::new(&[8, 6, 10], &[1, 1, 1]),
        // an odd box crossing the ragged edge
        Region::new(&[6, 4, 8], &[3, 3, 3]),
    ];

    for tier in [Tier::Decoded, Tier::Encoded] {
        let truth = Probe::new(volume.clone());
        let cache = Arc::new(ChunkCache::new(roomy(), 1 << 24));
        let cached = CachingSource::<u16>::attach(
            Arc::clone(&cache),
            "sweep",
            &chunk,
            ArrayPolicy::all(tier),
            Probe::new(volume.clone()),
        )
        .expect("attach");

        for pass in 0..2 {
            for region in &regions {
                let expected = truth.read_region(region).expect("uncached read");
                let got = cached.read_region(region).expect("cached read");
                assert_eq!(
                    got,
                    expected,
                    "{tier:?} tier, pass {pass} ({}), region {region:?}",
                    if pass == 0 { "cold" } else { "warm" }
                );
            }
        }

        // The second pass must actually have been served from the cache,
        // otherwise the warm half of the sweep proved nothing.
        let stats = cache.stats();
        assert!(
            stats.hits() > 0,
            "{tier:?}: the warm pass produced no hits, so it was not a warm pass"
        );
        match tier {
            Tier::Decoded => assert_eq!(stats.hits_encoded, 0),
            Tier::Encoded => assert_eq!(stats.hits_decoded, 0),
        }
    }
}

/// A `bool` volume is the case the encoded tier was designed around, and it is
/// also the one where an encoding bug would be least visible — every voxel is
/// one of two values, so a corrupted decode still looks like plausible data.
#[test]
fn a_bool_volume_round_trips_through_the_encoded_tier_exactly() {
    let shape = [8usize, 8, 32];
    let volume = sparse_bool(shape, 29);
    let truth = Probe::new(volume.clone());
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 24));
    let cached = CachingSource::<bool>::attach(
        Arc::clone(&cache),
        "bool",
        &[8, 8, 16],
        ArrayPolicy::all(Tier::Encoded),
        Probe::new(volume.clone()),
    )
    .expect("attach");

    for region in [
        Region::whole(&shape),
        Region::new(&[1, 1, 1], &[6, 6, 30]),
        Region::new(&[7, 7, 31], &[1, 1, 1]),
    ] {
        assert_eq!(
            cached.read_region(&region).unwrap(),
            truth.read_region(&region).unwrap()
        );
        // Again, warm.
        assert_eq!(
            cached.read_region(&region).unwrap(),
            truth.read_region(&region).unwrap()
        );
    }

    // And the premise of the tier: a sparse `bool` volume compresses hard.
    //
    // The bound is well below the 19.7x measured on real volumes, and
    // deliberately so: these chunks are 1 KiB, where deflate's window and
    // header are a visible fraction of the output. `the_default_codec_...` in
    // `cache.rs` exercises the ratio at a realistic size. What this assertion
    // is for is that the encoded tier is *storing compressed bytes at all* —
    // a tier that silently fell back to storing decoded bytes would pass every
    // other assertion in this test.
    let ratio = cache
        .stats()
        .encoded_ratio()
        .expect("the encoded tier stored something");
    assert!(
        ratio > 5.0,
        "a sparse bool volume should compress well; got {ratio:.1}x"
    );
}

/// The key is the lattice index, not the request shape — so two differently
/// shaped reads over overlapping data share entries instead of duplicating
/// them. This is what makes the cache worth having at all; a request-keyed
/// cache would report a miss here.
#[test]
fn two_differently_shaped_reads_share_the_chunks_they_overlap_on() {
    let shape = [4usize, 4, 16];
    let (probe, handle) = probe_pair(Probe::new(ramp_u16(shape)));
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 24));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "shared",
        &[4, 4, 4],
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    // Chunks 0 and 1 along the last axis.
    source
        .read_region(&Region::new(&[0, 0, 0], &[4, 4, 8]))
        .unwrap();
    let after_first = handle.reads();
    let misses_first = cache.stats().misses;
    assert_eq!(misses_first, 2);

    // Chunks 1 and 2 — a different box, overlapping on chunk 1.
    source
        .read_region(&Region::new(&[0, 0, 5], &[4, 4, 7]))
        .unwrap();
    let stats = cache.stats();
    assert_eq!(stats.hits(), 1, "chunk 1 should have been reused");
    assert_eq!(stats.misses, 3, "only chunk 2 was new");
    assert!(
        handle.reads() > after_first,
        "chunk 2 genuinely came from the source"
    );
}

// -------------------------------------------------- cross-array eviction --

/// **The test a per-array cache cannot pass, and the reason this cache exists.**
///
/// Two arrays share one capacity. The cold one fills it first; then the hot one
/// is read repeatedly. Capacity must follow the demand — the cold array's
/// entries are evicted to make room for the hot array's, and the eviction
/// events must name who took the space.
///
/// A per-array cache, however tuned, holds the cold array's entries for the
/// whole run: each array has its own capacity and its own LRU, and neither can
/// see the other's demand. That is the failure this replaces.
#[test]
fn cross_array_pressure_moves_capacity_from_the_cold_array_to_the_hot_one() {
    let shape = [1usize, 1, 64];
    let chunk = [1usize, 1, 8]; // 8 chunks per array, 16 u16 bytes each
    let recorder = Arc::new(Recorder::default());
    // Room for about six chunks in total across *both* arrays, so holding all
    // of one array's eight is already impossible.
    let capacity = 6 * 8 * 2;
    let cache = Arc::new(
        ChunkCache::new(roomy(), capacity as u64)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );

    let cold = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "cold",
        &chunk,
        ArrayPolicy::default(),
        Probe::new(ramp_u16(shape)),
    )
    .expect("attach cold");
    let hot = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "hot",
        &chunk,
        ArrayPolicy::default(),
        Probe::new(ramp_u16(shape)),
    )
    .expect("attach hot");

    // Phase 1: the cold array fills whatever it can.
    for index in 0..8 {
        cold.read_region(&Region::new(&[0, 0, index * 8], &[1, 1, 8]))
            .unwrap();
    }
    let cold_resident_before = cache.resident_chunks(cold.array());
    assert!(
        cold_resident_before > 0,
        "the cold array must have taken the capacity first, or the test proves nothing"
    );

    // Phase 2: the hot array is read, and read again. Nothing tells the cache
    // which array is hot; recency is the only signal, and it is enough.
    for _ in 0..3 {
        for index in 0..6 {
            hot.read_region(&Region::new(&[0, 0, index * 8], &[1, 1, 8]))
                .unwrap();
        }
    }

    let cold_after = cache.resident_chunks(cold.array());
    let hot_after = cache.resident_chunks(hot.array());
    assert_eq!(
        cold_after, 0,
        "the cold array still holds {cold_after} chunks; capacity did not move"
    );
    assert!(
        hot_after >= 5,
        "the hot array holds only {hot_after} chunks; it did not get the capacity"
    );

    // The eviction events must record that one array's space went to another —
    // which is the observable form of the property.
    let cross: usize = recorder.count(|event| {
        matches!(event, Event::CacheEvicted { array, for_array, .. } if array == "cold" && for_array == "hot")
    });
    assert!(
        cross > 0,
        "no eviction was attributed across arrays; events were {:?}",
        recorder
            .events()
            .iter()
            .filter(|event| matches!(event, Event::CacheEvicted { .. }))
            .collect::<Vec<_>>()
    );

    // And the hot array is genuinely being served from cache by the end.
    assert!(cache.stats().hits() > 6, "{:?}", cache.stats());
}

/// Eviction across arrays must not damage anything: the cold array still reads
/// correctly afterwards, it just pays for it.
#[test]
fn an_evicted_array_still_reads_correctly_it_only_reads_again() {
    let shape = [1usize, 1, 32];
    let volume = ramp_u16(shape);
    let truth = Probe::new(volume.clone());
    let cache = Arc::new(ChunkCache::new(roomy(), 32));
    let cold = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "cold",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(volume.clone()),
    )
    .expect("attach");
    let hot = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "hot",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(volume.clone()),
    )
    .expect("attach");

    let region = Region::new(&[0, 0, 0], &[1, 1, 8]);
    let first = cold.read_region(&region).unwrap();
    for index in 0..4 {
        hot.read_region(&Region::new(&[0, 0, index * 8], &[1, 1, 8]))
            .unwrap();
    }
    assert_eq!(cache.resident_chunks(cold.array()), 0);
    let second = cold.read_region(&region).unwrap();
    assert_eq!(first, second);
    assert_eq!(second, truth.read_region(&region).unwrap());
}

// ------------------------------------------------------------- the budget --

/// Contents never exceed the lease, because they *are* the lease.
#[test]
fn resident_bytes_always_equal_the_bytes_leased_from_the_budget() {
    let budget = MemoryBudget::new(1 << 20);
    let cache = Arc::new(ChunkCache::new(budget.clone(), 4096));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "leased",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 512])),
    )
    .expect("attach");

    for index in 0..64 {
        source
            .read_region(&Region::new(&[0, 0, index * 8], &[1, 1, 8]))
            .unwrap();
        assert_eq!(
            cache.resident_bytes(),
            budget.opportunistic_in_use(),
            "the cache's own count and the budget's disagree after read {index}"
        );
        assert!(
            cache.resident_bytes() <= cache.capacity(),
            "the cache exceeded its own ceiling"
        );
    }

    // Clearing returns every byte.
    cache.clear();
    assert_eq!(cache.resident_bytes(), 0);
    assert_eq!(budget.opportunistic_in_use(), 0);
}

/// A budget with no slack must produce **misses**, not errors and not an
/// over-allocation. This is the "pressure reduces performance, never
/// correctness" rule applied to the cache.
#[test]
fn a_starved_budget_degrades_to_misses_rather_than_failing() {
    let budget = MemoryBudget::new(1024);
    // Compute takes essentially all of it and holds on.
    let _compute = budget.acquire(1000);

    let shape = [1usize, 1, 256];
    let volume = ramp_u16(shape);
    let truth = Probe::new(volume.clone());
    let cache = Arc::new(ChunkCache::new(budget.clone(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "starved",
        &[1, 1, 32],
        ArrayPolicy::default(),
        Probe::new(volume.clone()),
    )
    .expect("attach");

    for _pass in 0..3 {
        for index in 0..8 {
            let region = Region::new(&[0, 0, index * 32], &[1, 1, 32]);
            assert_eq!(
                source
                    .read_region(&region)
                    .expect("a starved cache still reads"),
                truth.read_region(&region).unwrap()
            );
        }
    }

    let stats = cache.stats();
    assert!(
        stats.refusals > 0,
        "the budget should have refused something: {stats:?}"
    );
    assert_eq!(stats.hits(), 0, "nothing could be retained, so nothing hit");
    assert!(
        cache.resident_bytes() <= 24,
        "the cache retained {} bytes out of a 24-byte slack",
        cache.resident_bytes()
    );
    assert!(budget.in_use() <= budget.total());
}

/// A pass-through array retains nothing at all — the setting a planner uses for
/// a stage read exactly once, so that it does not evict the stages that are
/// reused.
#[test]
fn a_pass_through_policy_retains_nothing_and_still_reads_correctly() {
    let shape = [1usize, 1, 64];
    let volume = ramp_u16(shape);
    let truth = Probe::new(volume.clone());
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "once",
        &[1, 1, 8],
        ArrayPolicy::pass_through(),
        Probe::new(volume.clone()),
    )
    .expect("attach");

    let region = Region::whole(&shape);
    assert_eq!(
        source.read_region(&region).unwrap(),
        truth.read_region(&region).unwrap()
    );
    source.read_region(&region).unwrap();
    let stats = cache.stats();
    assert_eq!(cache.resident_bytes(), 0);
    assert_eq!(stats.hits(), 0);
    assert_eq!(stats.refusals, 0, "declining to retain is not a refusal");
}

/// The planner may change its mind mid-run.
#[test]
fn a_policy_change_takes_effect_on_the_next_fetch() {
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "switch",
        &[1, 1, 8],
        ArrayPolicy::all(Tier::Decoded),
        Probe::new(ramp_u16([1, 1, 64])),
    )
    .expect("attach");

    source
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .unwrap();
    source
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .unwrap();
    assert_eq!(cache.stats().hits_decoded, 1);

    cache
        .set_policy(source.array(), ArrayPolicy::all(Tier::Encoded))
        .unwrap();
    source
        .read_region(&Region::new(&[0, 0, 8], &[1, 1, 8]))
        .unwrap();
    source
        .read_region(&Region::new(&[0, 0, 8], &[1, 1, 8]))
        .unwrap();
    assert_eq!(cache.stats().hits_encoded, 1);
    assert_eq!(cache.stats().hits_decoded, 1, "the old entry is unchanged");
}

// ------------------------------------------------------ empty-region fast --

/// A region a backend has already ruled out is neither read **nor coalesced
/// across**.
///
/// The second half is the part that is easy to get wrong: fetching chunks 0-4
/// in one coalesced request would "skip" the empty chunk while reading it
/// anyway, handing the saving straight back as over-read. Two source reads
/// rather than one is the evidence that the run was broken at the empty chunk.
#[test]
fn a_known_empty_chunk_is_not_read_and_the_coalescing_run_breaks_at_it() {
    let shape = [1usize, 1, 20];
    let chunk = [1usize, 1, 4];
    // Chunk 2 covers z 8..12; the volume is genuinely zero there, so "empty"
    // is the truth and byte-identity is still checkable.
    let mut volume = ramp_u16(shape);
    for z in 8..12 {
        volume[[0, 0, z]] = 0;
    }
    let truth = Probe::new(volume.clone());
    let (probe, handle) = probe_pair(Probe::new(volume.clone()).empty_at(&[&[0, 0, 8]]));

    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20).with_max_coalesce(16));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "sparse",
        &chunk,
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    let whole = Region::whole(&shape);
    assert_eq!(
        source.read_region(&whole).unwrap(),
        truth.read_region(&whole).unwrap(),
        "the empty chunk must come back as the zeros it is"
    );

    assert_eq!(
        handle.reads(),
        2,
        "expected two runs (chunks 0-1 and 3-4) split by the empty chunk, got {} read(s)",
        handle.reads()
    );
    assert_eq!(cache.stats().known_empty, 1);
    assert_eq!(
        cache.stats().misses,
        4,
        "four chunks were genuinely fetched"
    );
    // And no capacity was spent on a chunk we never read.
    assert_eq!(cache.resident_chunks(source.array()), 4);
}

/// Coalescing is a lever, and turning it off is visible.
#[test]
fn the_coalesce_limit_controls_how_many_requests_a_region_costs() {
    let shape = [1usize, 1, 32];
    for (limit, expected_reads) in [(1usize, 4usize), (2, 2), (8, 1)] {
        let (probe, handle) = probe_pair(Probe::new(ramp_u16(shape)));
        let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20).with_max_coalesce(limit));
        let source = CachingSource::<u16>::attach(
            Arc::clone(&cache),
            "coalesce",
            &[1, 1, 8],
            ArrayPolicy::default(),
            probe,
        )
        .expect("attach");
        source.read_region(&Region::whole(&shape)).unwrap();
        assert_eq!(
            handle.reads(),
            expected_reads,
            "coalesce limit {limit} should cost {expected_reads} request(s)"
        );
        assert_eq!(cache.stats().misses, 4, "still four chunks either way");
    }
}

// ------------------------------------------------------------------ events --

/// Everything the cache does reaches the existing listener trait.
#[test]
fn the_cache_reports_hits_misses_and_evictions_through_the_listener_trait() {
    let recorder = Arc::new(Recorder::default());
    let cache = Arc::new(
        ChunkCache::new(roomy(), 32)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "watched",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 64])),
    )
    .expect("attach");

    for index in 0..4 {
        source
            .read_region(&Region::new(&[0, 0, index * 8], &[1, 1, 8]))
            .unwrap();
    }
    // Re-read the most recent one: a hit.
    source
        .read_region(&Region::new(&[0, 0, 24], &[1, 1, 8]))
        .unwrap();

    assert!(recorder.count(|event| matches!(event, Event::CacheMiss { .. })) >= 4);
    assert_eq!(
        recorder.count(|event| matches!(event, Event::CacheHit { .. })),
        1
    );
    assert!(recorder.count(|event| matches!(event, Event::CacheEvicted { .. })) > 0);

    // A decoded hit costs no decode, and the event says so — which is how the
    // "a hit must be free" claim is checkable rather than merely stated.
    let decoded_hits: Vec<u64> = recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::CacheHit {
                tier: Tier::Decoded,
                decode_ns,
                ..
            } => Some(*decode_ns),
            _ => None,
        })
        .collect();
    assert_eq!(decoded_hits, vec![0]);
}

/// An encoded hit does pay a decode, and the event records it. Stated as its
/// own test because it is the *counterpart* of the claim above: the tiers
/// differ in exactly this way and nowhere else.
#[test]
fn an_encoded_hit_records_the_decode_it_paid_for() {
    let recorder = Arc::new(Recorder::default());
    let cache = Arc::new(
        ChunkCache::new(roomy(), 1 << 20)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );
    let source = CachingSource::<bool>::attach(
        Arc::clone(&cache),
        "encoded",
        &[1, 1, 4096],
        ArrayPolicy::all(Tier::Encoded),
        Probe::new(sparse_bool([1, 1, 4096], 7)),
    )
    .expect("attach");

    let region = Region::whole(&[1, 1, 4096]);
    source.read_region(&region).unwrap();
    source.read_region(&region).unwrap();

    let decode_ns: Vec<u64> = recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::CacheHit { decode_ns, .. } => Some(*decode_ns),
            _ => None,
        })
        .collect();
    assert_eq!(decode_ns.len(), 1);
    assert!(
        decode_ns[0] > 0,
        "an encoded hit decompressed 4 KiB in zero nanoseconds"
    );
}

/// A panicking listener is contained, exactly as it is on the executor's path.
#[test]
fn a_panicking_listener_does_not_break_a_read() {
    struct Angry;
    impl EventListener for Angry {
        fn on_event(&self, _event: &Event) {
            panic!("this listener is broken");
        }
    }
    let cache = Arc::new(
        ChunkCache::new(roomy(), 1 << 20)
            .with_listeners(vec![Arc::new(Angry) as Arc<dyn EventListener>]),
    );
    let volume = ramp_u16([1, 1, 16]);
    let truth = Probe::new(volume.clone());
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "angry",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(volume.clone()),
    )
    .expect("attach");
    let region = Region::whole(&[1, 1, 16]);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let got = source.read_region(&region);
    std::panic::set_hook(previous);
    assert_eq!(got.unwrap(), truth.read_region(&region).unwrap());
}

// ---------------------------------------------------------------- prefetch --

fn plan_over(array: ArrayId, chunks: usize, extent: usize) -> BlockPlan {
    BlockPlan::in_order(
        array,
        (0..chunks).map(|index| Region::new(&[0, 0, index * extent], &[1, 1, extent])),
    )
}

/// Prefetch populates the cache, and a later read is a hit rather than a fetch.
#[test]
fn prefetch_populates_the_cache_and_the_read_that_follows_is_a_hit() {
    let shape = [1usize, 1, 64];
    let volume = ramp_u16(shape);
    let truth = Probe::new(volume.clone());
    let (probe, handle) = probe_pair(Probe::new(volume.clone()));
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "ahead",
        &[1, 1, 8],
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 2);
    prefetcher.submit(&plan_over(source.array(), 8, 8));
    prefetcher.drain();

    let reads_after_prefetch = handle.reads();
    assert!(reads_after_prefetch > 0, "the prefetcher read nothing");
    assert_eq!(cache.resident_chunks(source.array()), 8);

    for index in 0..8 {
        let region = Region::new(&[0, 0, index * 8], &[1, 1, 8]);
        assert_eq!(
            source.read_region(&region).unwrap(),
            truth.read_region(&region).unwrap()
        );
    }
    assert_eq!(
        handle.reads(),
        reads_after_prefetch,
        "the demand reads went back to the source; the prefetch bought nothing"
    );
    let stats = cache.stats();
    assert_eq!(stats.hits(), 8);
    assert_eq!(stats.prefetch_used, 8, "{stats:?}");
}

/// **Measured, not asserted.** With the prefetch queue saturated against a slow
/// source, a demand read for something else must still complete in about one
/// source read — not behind the backlog.
///
/// The numbers: the source takes `DELAY` per read and the plan is 24 chunks, so
/// the queue holds well over a second of work at depth 2. If a demand read were
/// serialised behind it the measurement would be hundreds of milliseconds. The
/// bound is deliberately loose (4 delays) because this runs on a shared CI
/// machine; the failure it is built to catch is an order of magnitude away from
/// the threshold, not a few percent.
#[test]
fn a_demand_read_does_not_queue_behind_a_saturated_prefetcher() {
    const DELAY: Duration = Duration::from_millis(40);
    let shape = [1usize, 1, 256];
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 22));

    let prefetched = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "queued",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16(shape)).slow(DELAY),
    )
    .expect("attach");
    let urgent = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "urgent",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16(shape)).slow(DELAY),
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 2);
    let handle = prefetcher.submit(&plan_over(prefetched.array(), 24, 8));

    // A read of a *different* array, so it can share nothing with the queue.
    let started = Instant::now();
    let got = urgent
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .expect("the demand read must not fail");
    let elapsed = started.elapsed();

    assert_eq!(got.len(), 8);
    assert!(
        prefetcher.queued() > 0 || prefetcher.stats().started < prefetcher.stats().submitted,
        "the prefetch queue drained before the measurement; the test proved nothing"
    );
    assert!(
        elapsed < DELAY * 4,
        "a demand read took {elapsed:?} with a saturated prefetch queue; \
         one source read is {DELAY:?}, so it queued behind the prefetcher"
    );

    prefetcher.cancel(handle);
}

/// Submission itself must be cheap — a worker declaring its future reads is on
/// the critical path even though the reads are not.
#[test]
fn submitting_a_plan_returns_without_doing_any_io() {
    const DELAY: Duration = Duration::from_millis(50);
    let (probe, handle) = probe_pair(Probe::new(ramp_u16([1, 1, 256])).slow(DELAY));
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 22));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "submit",
        &[1, 1, 8],
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 1);
    let started = Instant::now();
    prefetcher.submit(&plan_over(source.array(), 32, 8));
    let elapsed = started.elapsed();
    assert!(
        elapsed < DELAY,
        "submit took {elapsed:?}, which is at least one source read; it is doing IO"
    );
    let _ = handle.reads();
}

/// A cancelled plan stops fetching. Anything already in flight finishes — that
/// is stated in `cancel`'s docs and is what this measures, rather than pretending
/// a storage read can be interrupted.
#[test]
fn a_cancelled_plan_stops_fetching_and_its_waste_is_counted() {
    const DELAY: Duration = Duration::from_millis(20);
    let (probe, handle) = probe_pair(Probe::new(ramp_u16([1, 1, 512])).slow(DELAY));
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 22));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "abandoned",
        &[1, 1, 8],
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 1);
    let plan = prefetcher.submit(&plan_over(source.array(), 64, 8));
    std::thread::sleep(DELAY * 2);
    prefetcher.cancel(plan);
    prefetcher.drain();

    let stats = prefetcher.stats();
    assert_eq!(stats.submitted, 64);
    assert!(
        stats.cancelled > 40,
        "most of the plan should have been dropped unrun: {stats:?}"
    );
    assert!(
        handle.reads() < 20,
        "the source was read {} times after a cancel at ~2 reads in",
        handle.reads()
    );
}

/// The stronger cancel: an abandoned plan gives back the entries nobody read,
/// and they are labelled as the waste they are.
#[test]
fn abandoning_a_plan_releases_what_it_speculatively_loaded() {
    let recorder = Arc::new(Recorder::default());
    let cache = Arc::new(
        ChunkCache::new(roomy(), 1 << 20)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "speculative",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 64])),
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 2);
    let plan = prefetcher.submit(&plan_over(source.array(), 8, 8));
    prefetcher.drain();
    assert_eq!(cache.resident_chunks(source.array()), 8);

    // One of them does get read, and must survive.
    source
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .unwrap();

    let dropped = prefetcher.cancel_and_release(plan, &[source.array()]);
    assert_eq!(dropped, 7, "the seven unread entries should go");
    assert_eq!(
        cache.resident_chunks(source.array()),
        1,
        "the one that was read must stay"
    );
    assert_eq!(
        recorder.count(|event| matches!(
            event,
            Event::PrefetchWasted {
                reason: crate::log::PrefetchWaste::Cancelled,
                ..
            }
        )),
        7
    );
    assert_eq!(cache.resident_bytes(), 16);
}

/// Waste from over-deep prefetching is visible. **This is the number that says
/// the depth is wrong** — nothing else in the system reports it.
#[test]
fn prefetching_deeper_than_the_capacity_reports_the_waste_it_causes() {
    let recorder = Arc::new(Recorder::default());
    // Room for four chunks; the plan is sixteen.
    let cache = Arc::new(
        ChunkCache::new(roomy(), 4 * 16)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "toodeep",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 128])),
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 1);
    prefetcher.submit(&plan_over(source.array(), 16, 8));
    prefetcher.drain();

    let stats = cache.stats();
    assert!(
        stats.prefetch_wasted_evicted >= 8,
        "prefetching 16 chunks into room for 4 should waste most of them: {stats:?}"
    );
    assert!(
        recorder.count(|event| matches!(
            event,
            Event::PrefetchWasted {
                reason: crate::log::PrefetchWaste::Evicted,
                ..
            }
        )) > 0
    );
    // The survivors are the *last* ranks, because eviction is LRU and the
    // prefetcher went in rank order. That is the wrong end of the plan to keep,
    // and is exactly what a depth sweep is supposed to find.
    assert_eq!(cache.resident_chunks(source.array()), 4);
}

/// Under budget pressure the prefetcher stops spending IO, not just memory.
#[test]
fn the_prefetcher_declines_while_compute_is_queueing_for_the_budget() {
    let budget = MemoryBudget::new(4096);
    let _held = budget.acquire(4000);
    let waiter = budget.clone();
    let blocked = std::thread::spawn(move || {
        // Cannot be granted until `_held` drops, so `waiting_reserved` stays
        // non-zero and every opportunistic request is refused outright.
        let _lease = waiter.acquire(4096);
    });
    let mut spins = 0;
    while budget.waiting() == 0 && spins < 2000 {
        std::thread::sleep(Duration::from_millis(1));
        spins += 1;
    }
    assert_eq!(budget.waiting(), 1, "the budget never registered a waiter");

    let (probe, handle) = probe_pair(Probe::new(ramp_u16([1, 1, 128])));
    let cache = Arc::new(ChunkCache::new(budget.clone(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "pressured",
        &[1, 1, 8],
        ArrayPolicy::default(),
        probe,
    )
    .expect("attach");

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 2);
    prefetcher.submit(&plan_over(source.array(), 16, 8));
    prefetcher.drain();

    assert_eq!(
        handle.reads(),
        0,
        "the prefetcher did IO while a compute worker was queueing for memory"
    );
    assert!(cache.stats().prefetch_declined > 0);
    assert_eq!(cache.resident_bytes(), 0);

    drop(_held);
    blocked.join().unwrap();
}

/// Concurrent demand for one chunk costs one read, not `N`.
#[test]
fn concurrent_readers_of_one_chunk_cause_a_single_source_read() {
    const DELAY: Duration = Duration::from_millis(30);
    let volume = ramp_u16([1, 1, 8]);
    let (probe, handle) = probe_pair(Probe::new(volume.clone()).slow(DELAY));
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let source = Arc::new(
        CachingSource::<u16>::attach(
            Arc::clone(&cache),
            "shared",
            &[1, 1, 8],
            ArrayPolicy::default(),
            probe,
        )
        .expect("attach"),
    );

    let region = Region::new(&[0, 0, 0], &[1, 1, 8]);
    let readers: Vec<_> = (0..6)
        .map(|_| {
            let source = Arc::clone(&source);
            let region = region.clone();
            std::thread::spawn(move || source.read_region(&region).unwrap())
        })
        .collect();
    let results: Vec<_> = readers.into_iter().map(|r| r.join().unwrap()).collect();

    let truth = Probe::new(volume);
    let expected = truth.read_region(&region).unwrap();
    for result in &results {
        assert_eq!(result, &expected);
    }
    assert_eq!(
        handle.reads(),
        1,
        "six concurrent readers of one chunk caused {} source reads",
        handle.reads()
    );
}

/// A fetcher that fails propagates its error to the reader and leaves no claim
/// behind — a second read must not hang waiting for the fetch that died.
#[test]
fn a_failing_fetch_surfaces_and_does_not_wedge_the_next_read() {
    struct Broken;
    impl RegionSource<u16> for Broken {
        fn shape(&self) -> &[usize] {
            &[1, 1, 8]
        }
        fn read_region(&self, _region: &Region) -> Result<ArrayD<u16>> {
            Err(crate::error::Error::backend("the disk is on fire"))
        }
    }
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let array = cache
        .register(
            "broken",
            &[1, 1, 8],
            &[1, 1, 8],
            Dtype::U16,
            ArrayPolicy::default(),
            Arc::new(RegionSourceFetcher::<u16, Broken>::new(Broken)),
        )
        .expect("register");

    let region = Region::new(&[0, 0, 0], &[1, 1, 8]);
    for _ in 0..2 {
        let started = Instant::now();
        let err = cache.read_region_bytes(array, &region).unwrap_err();
        assert!(err.to_string().contains("on fire"), "{err}");
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "the second read waited for a claim the failed fetch should have released"
        );
    }
}

/// Registration is checked, so a mis-specified lattice fails at registration
/// rather than as a wrong read later.
#[test]
fn a_mis_specified_lattice_is_refused_at_registration() {
    let cache = ChunkCache::new(roomy(), 1 << 20);
    let fetcher = Arc::new(RegionSourceFetcher::<u16, Probe<u16>>::new(Probe::new(
        ramp_u16([1, 1, 8]),
    )));
    assert!(cache
        .register(
            "rank",
            &[1, 1, 8],
            &[1, 8],
            Dtype::U16,
            ArrayPolicy::default(),
            fetcher.clone()
        )
        .is_err());
    assert!(cache
        .register(
            "zero",
            &[1, 1, 8],
            &[1, 1, 0],
            Dtype::U16,
            ArrayPolicy::default(),
            fetcher
        )
        .is_err());
}

/// A request outside the array is refused rather than clamped, exactly as the
/// uncached source refuses it.
#[test]
fn a_region_outside_the_array_is_refused_by_the_cache_too() {
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "bounds",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 16])),
    )
    .expect("attach");
    assert!(source
        .read_region(&Region::new(&[0, 0, 12], &[1, 1, 8]))
        .is_err());
    assert!(source.read_region(&Region::new(&[0, 0], &[1, 8])).is_err());
}

/// A plan handle from one prefetcher is a plain identifier; cancelling an
/// unknown one is harmless. Stated because `cancel` is called from teardown
/// paths where the plan may already have completed.
#[test]
fn cancelling_a_finished_or_unknown_plan_is_harmless() {
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 20));
    let prefetcher = Prefetcher::new(Arc::clone(&cache), 1);
    prefetcher.cancel(PlanHandle::from_raw(999));
    prefetcher.drain();
    assert_eq!(prefetcher.stats().submitted, 0);
}

/// Requests are honoured in rank order, so what a worker needs soonest arrives
/// first. Checked with one worker, because with several the order in which they
/// *finish* is not the order they started.
#[test]
fn a_single_worker_fetches_in_rank_order() {
    let recorder = Arc::new(Recorder::default());
    let cache = Arc::new(
        ChunkCache::new(roomy(), 1 << 20)
            .with_listeners(vec![Arc::clone(&recorder) as Arc<dyn EventListener>]),
    );
    let source = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "ordered",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 64])),
    )
    .expect("attach");

    // Submitted deliberately out of order.
    let requests: Vec<RegionRequest> = [5u32, 0, 7, 2, 1]
        .iter()
        .map(|&rank| {
            RegionRequest::new(
                source.array(),
                Region::new(&[0, 0, rank as usize * 8], &[1, 1, 8]),
                rank,
            )
        })
        .collect();
    let prefetcher = Prefetcher::new(Arc::clone(&cache), 1);
    prefetcher.submit(&requests);
    prefetcher.drain();

    let issued: Vec<u64> = recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::PrefetchIssued { chunk, .. } => Some(*chunk),
            _ => None,
        })
        .collect();
    assert_eq!(issued, vec![0, 1, 2, 5, 7]);
}

/// Numbers rather than bounds: what the cache and prefetcher actually cost on
/// whatever machine this is.
///
/// `#[ignore]`d and printed rather than asserted, for the same reason
/// `dispatch_overhead_report` is: absolute times are machine-dependent, a
/// threshold on them would be a flaky test, and the *shape* of the numbers is
/// what anybody tuning depth or tier needs to see.
///
/// `cargo test -p blockflow measured_cache_and_prefetch_report -- --ignored --nocapture`
#[test]
#[ignore]
fn measured_cache_and_prefetch_report() {
    const DELAY: Duration = Duration::from_millis(40);

    // --- hit cost, per tier, on a 64 KiB chunk.
    for tier in [Tier::Decoded, Tier::Encoded] {
        let shape = [1usize, 1, 32768];
        let cache = Arc::new(ChunkCache::new(roomy(), 1 << 24));
        let source = CachingSource::<u16>::attach(
            Arc::clone(&cache),
            "hit",
            &[1, 1, 32768],
            ArrayPolicy::all(tier),
            Probe::new(ramp_u16(shape)),
        )
        .unwrap();
        let region = Region::whole(&shape);
        source.read_region(&region).unwrap();
        let started = Instant::now();
        for _ in 0..50 {
            source.read_region(&region).unwrap();
        }
        println!(
            "hit {:>8?} tier: {:>8.1} us per 64 KiB chunk",
            tier,
            started.elapsed().as_secs_f64() * 1e6 / 50.0
        );
    }

    // --- compression achieved on a sparse bool volume at a realistic chunk.
    let shape = [64usize, 64, 64];
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 26));
    let source = CachingSource::<bool>::attach(
        Arc::clone(&cache),
        "ratio",
        &[64, 64, 64],
        ArrayPolicy::all(Tier::Encoded),
        Probe::new(sparse_bool(shape, 29)),
    )
    .unwrap();
    source.read_region(&Region::whole(&shape)).unwrap();
    println!(
        "encoded ratio on a 256 KiB sparse bool chunk: {:.1}x",
        cache.stats().encoded_ratio().unwrap_or(0.0)
    );

    // --- demand-read latency with the prefetch queue saturated.
    let cache = Arc::new(ChunkCache::new(roomy(), 1 << 24));
    let queued = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "queued",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 512])).slow(DELAY),
    )
    .unwrap();
    let urgent = CachingSource::<u16>::attach(
        Arc::clone(&cache),
        "urgent",
        &[1, 1, 8],
        ArrayPolicy::default(),
        Probe::new(ramp_u16([1, 1, 512])).slow(DELAY),
    )
    .unwrap();

    let alone = Instant::now();
    urgent
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .unwrap();
    let alone = alone.elapsed();

    let prefetcher = Prefetcher::new(Arc::clone(&cache), 4);
    let plan = prefetcher.submit(&plan_over(queued.array(), 60, 8));
    let under_load = Instant::now();
    urgent
        .read_region(&Region::new(&[0, 0, 8], &[1, 1, 8]))
        .unwrap();
    let under_load = under_load.elapsed();
    let still_queued = prefetcher.queued();
    prefetcher.cancel(plan);

    println!(
        "demand read: {alone:?} idle, {under_load:?} with {still_queued} prefetches still queued \
         (source read is {DELAY:?})"
    );

    // --- what a hit saves against the same slow source.
    let hit = Instant::now();
    urgent
        .read_region(&Region::new(&[0, 0, 0], &[1, 1, 8]))
        .unwrap();
    println!("cached re-read of the same region: {:?}", hit.elapsed());
}
