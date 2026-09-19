//! **The chunk cache, wired to the one environment that should have it.**
//!
//! `docs/design/cache-and-prefetch.md` §0.1 opens with the fact this file
//! closes: neither `ChunkCache` nor `Prefetcher` is constructed outside a
//! `#[cfg(test)]` module, so **no `Environment::read` is served from one**. §1.2
//! settles where it belongs — below `Voxels`, at the per-array chunk lattice,
//! because a cache keyed by the extent a caller asked for produces "different
//! keys over the same data" and a halo re-read is precisely two overlapping
//! boxes — and observes that the only missing piece is an adapter from an
//! environment's storage call to `RegionSource<T>`.
//!
//! # What this file has to establish, in order
//!
//! 1. **A cached read is the same read.** Byte-identical voxels, cache on
//!    against cache off, over regions that overlap.
//! 2. **The cache is actually serving them.** Identical voxels is *necessary
//!    and not sufficient* — it is also exactly what a cache that never serves
//!    anything produces, which is this crate's own empty-sink trap. `hits` is
//!    what separates the two, and it is asserted non-zero.
//! 3. **A written image is cached and a write invalidates it**, because serving
//!    a stale chunk would be a wrong answer rather than a slow one — which is
//!    what `ChunkCache::invalidate` is for.
#![cfg(feature = "zarr")]

use blockflow::cache::CacheStats;
use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::Environment;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::voxelwise::VoxelwiseMapOp;
use blockflow::region::Region;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::Dtype;
use blockflow::Voxels;

mod support;

use support::scratch::ScratchDir;
use support::single_phase;
use support::volume::{modular_mask_bool, xorshift_u16, xorshift_unit_f64};

const VOLUME: [usize; 3] = [64, 64, 64];
const CHUNK: [usize; 3] = [16, 16, 16];

fn source() -> Voxels {
    Voxels::U16(xorshift_u16(VOLUME, 0x2545_F491_4F6C_DD1D))
}

/// Overlapping windows: a `24^3` core on a `16` stride, so consecutive reads
/// share chunks. **The overlap is the experiment** — a cache cannot show
/// anything against a traversal that never revisits a chunk.
fn regions() -> Vec<Region> {
    let mut out = Vec::new();
    for z in (0..VOLUME[0] - 24).step_by(16) {
        for y in (0..VOLUME[1] - 24).step_by(16) {
            for x in (0..VOLUME[2] - 24).step_by(16) {
                out.push(Region {
                    start: vec![z, y, x],
                    shape: vec![24, 24, 24],
                });
            }
        }
    }
    out
}

fn read_all(env: &ZarrEnvironment, regions: &[Region]) -> Vec<Voxels> {
    regions
        .iter()
        .map(|region| match env.read(0, region).expect("image 0 reads") {
            blockflow::env::BlockBuf::Array(voxels) => voxels,
            other => panic!("a real environment must return voxels, got {other:?}"),
        })
        .collect()
}

fn root(tag: &str) -> std::path::PathBuf {
    ScratchDir::new("zarr-cache", tag).keep()
}

struct CacheFixture {
    regions: Vec<Region>,
}

impl CacheFixture {
    fn new() -> Self {
        Self { regions: regions() }
    }

    fn regions(&self) -> &[Region] {
        &self.regions
    }

    fn uncached(&self, tag: &str) -> ZarrEnvironment {
        ZarrEnvironment::create(root(tag), &source(), CHUNK)
            .expect("a store")
            .without_cache()
    }

    fn cached_u16(&self, tag: &str, capacity: u64) -> ZarrEnvironment {
        ZarrEnvironment::create(root(tag), &source(), CHUNK)
            .expect("a store")
            .with_cache(capacity)
    }

    fn single_phase_threshold(dtype: Dtype) -> (Workflow, Decomposition) {
        let chain: Chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.5, 1.0, 0.0));
        let workflow = Workflow::new(chain, VOLUME, dtype);
        let grid = BlockGrid::along(VOLUME, &[0, 1, 2], CHUNK[0]).expect("a grid");
        let plan = single_phase::plan_on_grid(&workflow, VOLUME, grid);
        (workflow, plan)
    }

    fn read_all(&self, env: &ZarrEnvironment) -> Vec<Voxels> {
        read_all(env, &self.regions)
    }

    fn uncached_truth(&self, tag: &str) -> Vec<Voxels> {
        self.read_all(&self.uncached(tag))
    }

    fn run_offset_readers(
        &self,
        warm: &std::sync::Arc<ZarrEnvironment>,
        want: std::sync::Arc<Vec<Voxels>>,
        rounds: usize,
        before_round: impl Fn(&std::sync::Arc<ZarrEnvironment>, &std::sync::Arc<Vec<Region>>),
    ) -> usize {
        let regions = std::sync::Arc::new(self.regions.clone());
        let mut wrong = 0usize;
        for round in 0..rounds {
            before_round(warm, &regions);
            let mut handles = Vec::new();
            for thread in 0..4 {
                let warm = std::sync::Arc::clone(warm);
                let regions = std::sync::Arc::clone(&regions);
                let want = std::sync::Arc::clone(&want);
                handles.push(std::thread::spawn(move || {
                    let mut bad = 0usize;
                    for step in 0..regions.len() {
                        let which = (step + thread * 3 + round) % regions.len();
                        let got = match warm
                            .read(0, &regions[which])
                            .expect("a cached read must not fail")
                        {
                            blockflow::env::BlockBuf::Array(voxels) => voxels,
                            other => panic!("expected voxels, got {other:?}"),
                        };
                        if got != want[which] {
                            bad += 1;
                        }
                    }
                    bad
                }));
            }
            for handle in handles {
                wrong += handle.join().expect("no thread may panic");
            }
        }
        wrong
    }
}

/// **Same voxels, and the cache really served them.**
#[test]
fn a_cached_read_is_the_same_read_and_the_cache_serves_it() {
    let fixture = CacheFixture::new();
    assert!(
        fixture.regions().len() > 8,
        "the fixture needs a traversal to revisit"
    );

    // **The control is `without_cache`, where it used to be the default.** The
    // argument for opt-in was that a cache changes what a read costs and what
    // the process holds without anyone saying so; it held until
    // `a_bigger_cache_reads_strictly_fewer_bytes_from_the_store` below measured
    // what one is worth — 3.09x fewer bytes off the store — so the default is on
    // and the opt-out is explicit.
    let cold = fixture.uncached("cold");
    assert!(
        cold.cache_stats().is_none(),
        "`without_cache` must actually leave the environment without one, or the \
         comparison below is a cache against a cache"
    );
    let plain = fixture.read_all(&cold);

    // Sixteen chunks of `16^3` `u16` — big enough to hold a neighbourhood
    // and far too small to hold the volume, so it must evict and the hits
    // that remain are real reuse rather than "everything fits".
    let warm = fixture.cached_u16("warm", 16 * 16 * 16 * 16 * 2);
    let cached = fixture.read_all(&warm);

    assert_eq!(plain.len(), cached.len());
    for (index, (a, b)) in plain.iter().zip(cached.iter()).enumerate() {
        assert_eq!(
            a.dtype(),
            b.dtype(),
            "region {index} came back as {:?} cached and {:?} uncached",
            b.dtype(),
            a.dtype()
        );
        assert!(
            a == b,
            "region {index} differs between the cached and uncached reads. A cache that changes \
             a voxel is not a cache"
        );
    }

    // **The control.** Everything above passes just as well against a cache
    // that is never consulted, which is what the unwired state looked like.
    let stats: CacheStats = warm
        .cache_stats()
        .expect("the warm environment has a cache");
    assert!(
        stats.hits() > 0,
        "the cache served {} hits against {} misses. Identical voxels is necessary and not \
         sufficient — a cache that never serves anything produces them too",
        stats.hits(),
        stats.misses
    );
    eprintln!(
        "cache: {} hits, {} misses, {} resident bytes",
        stats.hits(),
        stats.misses,
        stats.resident_bytes
    );
}

/// **A written image is cached, and a write invalidates the chunks it covers.**
///
/// Both halves are asserted, because either alone passes for the wrong reason: a
/// read after a write returning what was written is also what an image nobody
/// caches produces, and a hit on a written image is only safe if the write threw
/// away what the cache held. This writes image 1, reads it twice for the hit,
/// then overwrites and reads again for the invalidation.
#[test]
fn a_written_image_is_cached_and_a_write_invalidates_it() {
    let path = root("written");
    let env = ZarrEnvironment::create(&path, &source(), CHUNK)
        .expect("a store")
        .with_cache(1 << 24);
    // A one-phase plan, stated directly rather than searched for: what this
    // test needs from it is only that image 1 exists.
    let (_workflow, plan) = CacheFixture::single_phase_threshold(Dtype::U16);
    env.prepare(&plan).expect("image 1 is created");

    let region = Region {
        start: vec![0, 0, 0],
        shape: vec![16, 16, 16],
    };
    // Warm the source first, so the counters below are known to be capable of
    // moving at all — without this, "no new hits" is what a dead cache says.
    let _ = env.read(0, &region).expect("image 0 reads");
    let _ = env.read(0, &region).expect("image 0 reads again");
    let baseline = env.cache_stats().expect("a cache");
    assert!(
        baseline.hits() > 0,
        "reading the source twice produced no hit, so this fixture cannot tell a cached image \
         from an uncached one and the assertion below means nothing"
    );

    // **A written image is cached, and a write to it invalidates.** The
    // property that matters is not "no hits" — it is that a read after a write
    // returns what was written, never a chunk cached before it.
    let ones = Voxels::from(ndarray::Array3::<u16>::from_elem((16, 16, 16), 1));
    env.write(1, &region, &region, &blockflow::env::BlockBuf::Array(ones))
        .expect("image 1 is writable");
    let first = env.read(1, &region).expect("image 1 reads");
    let again = env.read(1, &region).expect("image 1 reads again");
    assert_eq!(
        first.as_array().unwrap(),
        again.as_array().unwrap(),
        "two reads of one written image disagreed"
    );
    assert!(
        first
            .as_array()
            .unwrap()
            .view::<u16>()
            .unwrap()
            .iter()
            .all(|&value| value == 1),
        "the read did not return what was written"
    );
    let after = env.cache_stats().expect("a cache");
    assert!(
        after.hits() > baseline.hits(),
        "reading a written image twice produced no hit, so intermediates are still not \
         cached and the invalidation below has nothing to protect"
    );

    // Now overwrite it and read again: the cached chunk from the read above must
    // not survive. This is the whole of what `ChunkCache::invalidate` buys, and
    // without it this read returns ones.
    let twos = Voxels::from(ndarray::Array3::<u16>::from_elem((16, 16, 16), 2));
    env.write(1, &region, &region, &blockflow::env::BlockBuf::Array(twos))
        .expect("image 1 is writable again");
    let overwritten = env
        .read(1, &region)
        .expect("image 1 reads after the overwrite");
    assert!(
        overwritten
            .as_array()
            .unwrap()
            .view::<u16>()
            .unwrap()
            .iter()
            .all(|&value| value == 2),
        "a read after an overwrite returned the chunk cached before it — a stale chunk is \
         a wrong answer rather than a slow one, which is why this image was uncacheable \
         until `ChunkCache::invalidate` existed"
    );
    // And it does reach the cache: the read after the write was a miss, which
    // is what an invalidated chunk produces and what "not cached at all" would
    // also produce — which is why the hit above is asserted too. The pair is the
    // claim: cached, and invalidated.
    assert!(
        after.misses > baseline.misses,
        "the written image produced no miss, so it never reached the cache and the \
         invalidation above protected nothing"
    );
}

/// The element type is the array's, and the cache is registered per image.
///
/// A `bool` volume goes through the same path as a `u16` one — the registry is
/// type-erased and downcast where `by_dtype!` has already fixed the element —
/// so a width whose `CacheElement` packing differs must still round-trip.
#[test]
fn a_bool_volume_round_trips_through_the_cache() {
    let path = root("bool");
    let mask = Voxels::Bool(modular_mask_bool(VOLUME, 3, 0));
    let env = ZarrEnvironment::create(&path, &mask, CHUNK)
        .expect("a store")
        .with_cache(1 << 24);
    let region = Region {
        start: vec![8, 8, 8],
        shape: vec![24, 24, 24],
    };
    let read = |region: &Region| match env.read(0, region).expect("a read") {
        blockflow::env::BlockBuf::Array(voxels) => voxels,
        other => panic!("expected voxels, got {other:?}"),
    };
    let (first, second) = (read(&region), read(&region));
    assert!(first == second, "the same region read twice must agree");
    assert!(
        env.cache_stats().expect("a cache").hits() > 0,
        "the second read of the same region served nothing from the cache"
    );
}

/// **The prefetch sweep, in the shape `docs/design/cache-and-prefetch.md` §4.2
/// asks for.**
///
/// That note rejects the obvious assertion first: *"waste must be non-zero
/// somewhere in the suite" is the right instinct and the wrong assertion*,
/// because a run that wastes nothing may simply have a cache large enough,
/// which is a **good** outcome. So the assertable form is a sweep with four
/// parts, and all four are here:
///
/// 1. **Depth 0 is the control** — nothing issued, and the answer identical to
///    every other depth. Without it the sweep measures the prefetcher against
///    nothing.
/// 2. **Something prefetched is actually consumed** at the shallow end. A
///    prefetcher whose reads are never used is fetching the *wrong* things,
///    which is a different defect from fetching too many.
/// 3. **Waste rises with depth**, at a cache size held fixed and below the
///    plan's footprint.
/// 4. **A liveness control that fails if the sweep never reaches the regime** —
///    because a sweep whose every depth fits in the cache is the empty sink
///    wearing different clothes, and this file has already been caught by that
///    once.
#[test]
#[ignore = "prefetch economics sweep; concurrent correctness tests run in default CI"]
fn the_prefetch_sweep_has_a_control_at_both_ends() {
    // `f64`, because `threshold` states the element types it accepts and
    // `uint16` is not one — the plan is refused when it is made rather than
    // when a block reaches the op, which is the crate working as intended.
    let volume = Voxels::F64(xorshift_unit_f64(VOLUME, 0x9E37_79B9_7F4A_7C15));
    let (workflow, plan) = CacheFixture::single_phase_threshold(Dtype::F64);

    // Eight chunks of `16^3 f64`, against a volume of sixty-four of them. **The
    // cache must be well below the plan's footprint** or nothing can be evicted
    // and part 3 has nothing to measure.
    let cache_bytes = 8 * 16 * 16 * 16 * 8;
    let run = |lookahead: usize| {
        let path = root(&format!("sweep-{lookahead}"));
        let env = ZarrEnvironment::create(&path, &volume, CHUNK)
            .expect("a store")
            .with_cache(cache_bytes)
            .with_prefetch(1, lookahead)
            .expect("a cache to prefetch into");
        env.prepare(&plan).expect("the plan prepares");
        let hints = Hints {
            prefetch_depth: lookahead,
            ..Hints::default()
        };
        execute("sweep", &workflow, &plan, &hints, &env).expect("the run");
        // The prefetcher's threads are asynchronous; drain so the counters are
        // about a finished run rather than about when this line was reached.
        // **This line said so before it existed**, and the counters were a race
        // in the meantime: a fast machine had the fetch landed by the time it
        // read `prefetch_issued`, and a loaded CI runner did not.
        env.drain_prefetch();
        let stats = env.cache_stats().expect("a cache");
        let issued = env.prefetch_stats().expect("a prefetcher").submitted;
        (issued, stats)
    };

    let (issued_none, none) = run(0);
    let (issued_shallow, shallow) = run(1);
    let (issued_deep, deep) = run(48);

    // 1. The control.
    assert_eq!(
        issued_none, 0,
        "depth zero submitted {issued_none} requests; it must be the arm that prefetches nothing"
    );
    assert_eq!(
        none.prefetch_issued, 0,
        "depth zero issued {} prefetches into the cache",
        none.prefetch_issued
    );

    // 4. The liveness control, and it comes before the claims that need it: if
    // the deep arm never reached past what the cache would have held anyway,
    // everything below is vacuous.
    assert!(
        issued_deep > issued_shallow,
        "the deep arm submitted {issued_deep} against the shallow arm's {issued_shallow}. The \
         sweep never reached the regime it exists to measure"
    );
    assert!(
        deep.evictions > 0,
        "the deep arm evicted nothing, so the cache held everything it was given and this sweep \
         is the empty sink in different clothes"
    );

    // 2. Something fetched ahead was consumed.
    //
    // **On the deep arm, because the shallow one is a race and loses it on a
    // loaded machine.** At a lookahead of one the prefetcher is exactly one
    // block in front of the demand path; where the machine is slow enough that
    // the reader reaches the chunk first, the prefetch is not refused and not
    // wasted — the prefetcher simply finds it resident and does nothing, so
    // `prefetch_issued` stays at zero with no counter recording why. That is a
    // legitimate outcome of a depth of one rather than a defect, and asserting
    // against it made this test pass on a fast machine and fail on a hosted
    // runner. `env.drain_prefetch()` above removes the *other* half of that race
    // — a prefetch submitted and not yet finished — and cannot remove this half,
    // because there is nothing pending to wait for.
    //
    // The deep arm carries the claim structurally: with a lookahead of 48 the
    // prefetcher is far enough ahead that its targets are blocks the reader has
    // not reached, so what it fetches is genuinely fetched *ahead*.
    assert!(
        deep.prefetch_issued > 0,
        "the deep arm issued no prefetch into the cache at all, having submitted \
         {issued_deep}. With a lookahead of 48 the prefetcher is ahead of the reader by \
         construction, so this is not the race the shallow arm has."
    );

    // 3. Waste rises with depth.
    let waste = |stats: &CacheStats| stats.prefetch_wasted_evicted + stats.prefetch_wasted_refused;
    assert!(
        waste(&deep) >= waste(&shallow),
        "waste fell as the lookahead grew: {} deep against {} shallow. Waste is the cost of \
         depth and is what tells you the depth is wrong; nothing else in the system will",
        waste(&deep),
        waste(&shallow)
    );

    eprintln!(
        "sweep: none issued {issued_none}; shallow issued {issued_shallow} used {} wasted {}; \
         deep issued {issued_deep} used {} wasted {} evictions {}",
        shallow.prefetch_used,
        waste(&shallow),
        deep.prefetch_used,
        waste(&deep),
        deep.evictions
    );
}

// ------------------------------------------ what the cache is actually worth --

/// **What a chunk cache saves on a real blocked run**, which nothing had
/// measured.
///
/// The tests above establish that the cache is *correct* — a cached read is the
/// same read, it really serves hits, a written image is invalidated on write, a
/// `bool` volume round-trips. None of them says what it is **for**, and that gap
/// is load-bearing well outside this file:
///
/// * `simulate::Machine::cache_bytes` is the simulator's central lever, and
///   ordering-changes-hit-rate is the mechanism the whole module exists to
///   rank;
/// * `distributed::placement` and `distributed::cache_model` both model a
///   worker's cache, and neither had a measured figure for what one saves.
///
/// So this is that measurement: the same plan through the same environment, at a
/// range of budgets, reporting the bytes that actually left the store.
///
/// **The claim asserted here is byte counts, not time.** The store reads are
/// deterministic; a wall clock on a shared machine is not, and this crate does
/// not assert on durations. `print_what_the_cache_saves` beside it prints the
/// timing for a human.
#[test]
#[ignore = "cache economics measurement; answer-preservation tests run in default CI"]
fn a_bigger_cache_reads_strictly_fewer_bytes_from_the_store() {
    let regions = regions();
    let bytes_at = |capacity: u64| -> (u64, u64, u64) {
        let path = root(&format!("worth-{capacity}"));
        let env = ZarrEnvironment::create(&path, &source(), CHUNK)
            .expect("a store")
            .with_cache(capacity);
        read_all(&env, &regions);
        let stats = env.cache_stats().expect("a cache");
        (stats.source_bytes, stats.hits(), stats.misses)
    };

    // One chunk of `16^3` `u16` is 8192 bytes. The sweep runs from a cache that
    // can hold a single chunk — no reuse is possible across the overlapping
    // windows, which is the no-cache baseline in everything but name — to one
    // that holds the whole volume.
    let chunk_bytes = (CHUNK.iter().product::<usize>() * 2) as u64;
    let volume_bytes = (VOLUME.iter().product::<usize>() * 2) as u64;
    println!(
        "{:>12} {:>14} {:>8} {:>8}  chunk {chunk_bytes} B, volume {volume_bytes} B",
        "capacity", "store bytes", "hits", "misses"
    );
    let mut previous: Option<(u64, u64)> = None;
    let mut smallest = 0u64;
    let mut largest = 0u64;
    for multiple in [1u64, 4, 16, 64, 256] {
        let capacity = chunk_bytes * multiple;
        let (source_bytes, hits, misses) = bytes_at(capacity);
        println!("{capacity:>12} {source_bytes:>14} {hits:>8} {misses:>8}");
        if multiple == 1 {
            smallest = source_bytes;
        }
        largest = source_bytes;
        if let Some((before, _)) = previous {
            assert!(
                source_bytes <= before,
                "a cache of {capacity} bytes read {source_bytes} from the store against the \
                 smaller cache's {before}. More room must not cost more reads."
            );
        }
        previous = Some((source_bytes, hits));
    }

    assert!(
        smallest > largest,
        "the store read {smallest} bytes at one chunk of capacity and {largest} at 256 — the \
         cache saved nothing, so either this traversal does not revisit a chunk or the cache \
         is not on the read path"
    );
    let saved = smallest as f64 / largest as f64;
    println!(
        "a cache that holds the volume reads {saved:.2}x fewer bytes than one that holds a chunk"
    );
    assert!(
        saved > 1.5,
        "the cache saved only {saved:.2}x, where the overlapping windows this file reads \
         should share far more than that"
    );
}

/// The same capacity sweep with a clock on it.
///
/// Ignored because this crate gates on deterministic byte counts, not host
/// timings. The recorded calibration table lives in
/// `docs/design/cache-and-prefetch.md` §5.1.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_what_the_cache_saves() {
    use std::time::Instant;

    let regions = regions();
    let chunk_bytes = (CHUNK.iter().product::<usize>() * 2) as u64;
    println!(
        "{:>14} {:>12} {:>14}",
        "capacity", "wall (ms)", "store bytes"
    );
    for capacity in [0u64, chunk_bytes, chunk_bytes * 16, chunk_bytes * 256] {
        let path = root(&format!("worth-timed-{capacity}"));
        let mut env = ZarrEnvironment::create(&path, &source(), CHUNK).expect("a store");
        if capacity > 0 {
            env = env.with_cache(capacity);
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let started = Instant::now();
            read_all(&env, &regions);
            best = best.min(started.elapsed().as_secs_f64());
        }
        let bytes = env
            .cache_stats()
            .map(|stats| stats.source_bytes.to_string())
            .unwrap_or_else(|| "uncached".to_string());
        println!(
            "{:>14} {:>12.1} {bytes:>14}",
            if capacity == 0 {
                "none".to_string()
            } else {
                capacity.to_string()
            },
            best * 1e3
        );
    }
}

/// **Concurrent cached reads must return what concurrent uncached reads do.**
///
/// This is the property that was never checked, and the one that fails. Every
/// other test in this file reads on one thread; the executor does not, and the
/// moment `ZarrEnvironment` cached by default,
/// `tests/zarr_env.rs`'s `concurrent_execution_through_storage_is_still_byte_identical`
/// began failing intermittently at concurrency 4 — a different voxel each run,
/// which is the signature of a race rather than of a mis-computed answer.
///
/// This isolates it from the executor: no plan, no ops, no blocks. Many threads
/// read overlapping regions of one **immutable** array through one cache, and
/// every read must equal the same read served without a cache. An immutable
/// array is the easy case — there is no invalidation to get wrong — so a
/// disagreement here is the cache's fill protocol and nothing else.
#[test]
fn concurrent_reads_through_the_cache_return_what_uncached_reads_do() {
    use std::sync::Arc;

    let fixture = CacheFixture::new();
    assert!(
        fixture.regions().len() > 8,
        "the fixture needs a traversal to revisit"
    );

    // The truth, read once with no cache and no concurrency.
    let want = fixture.uncached_truth("concurrent-plain");

    // **A capacity that must evict.** Sixteen chunks of the volume's several
    // hundred, so the fill path is exercised repeatedly rather than warming
    // once and answering from memory forever — which is the state in which a
    // race in the claim protocol can be reached at all.
    let warm = Arc::new(fixture.cached_u16(
        "concurrent-warm",
        16 * CHUNK.iter().product::<usize>() as u64 * 2,
    ));
    let wrong = fixture.run_offset_readers(&warm, Arc::new(want), 8, |_, _| {});
    assert_eq!(
        wrong, 0,
        "{wrong} concurrent cached reads disagreed with the uncached read of the same \
         region. The array is never written, so there is no invalidation to get wrong — \
         this is the fill protocol serving bytes that are not the chunk's."
    );

    let stats = warm.cache_stats().expect("a cache");
    assert!(
        stats.hits() > 0 && stats.misses > 0,
        "the capacity did not make the fixture both hit and evict, so the fill path was \
         not exercised: {stats:?}"
    );
}

/// **Prefetching must not change what a read returns, under concurrency.**
///
/// The sibling of `concurrent_reads_through_the_cache_return_what_uncached_reads_do`,
/// and written for the reason that one exists: the bug it caught lived in
/// `ChunkCache::claim`, which the prefetcher drives too — through
/// `fill(.., None, true)`, on its own threads, against the same shared state.
///
/// The existing concurrency coverage of that path is narrower than it looks.
/// `cache_tests::six_concurrent_readers_of_one_chunk_cause_one_source_read`
/// spawns six threads at **one region**, which exercises "concurrent demand for
/// one chunk costs one read" and never reaches varied overlapping regions under
/// eviction — which is exactly the shape that produced 23 wrong reads before the
/// fix.
///
/// So: demand reads on several threads, a prefetcher filling the same cache
/// underneath them, and a capacity small enough that it must evict while both
/// are running. Every read must equal the uncached read of the same region.
#[test]
fn prefetching_under_concurrent_demand_reads_changes_no_answer() {
    use std::sync::Arc;

    let fixture = CacheFixture::new();
    assert!(
        fixture.regions().len() > 8,
        "the fixture needs a traversal to revisit"
    );

    let want = Arc::new(fixture.uncached_truth("prefetch-plain"));

    // Eight chunks: enough that prefetched chunks land and are used, far too
    // few to hold the traversal, so the fill path runs throughout.
    let capacity = 8 * CHUNK.iter().product::<usize>() as u64 * 2;
    let warm = Arc::new(
        fixture
            .cached_u16("prefetch-warm", capacity)
            .with_prefetch(2, 6)
            .expect("a prefetcher needs the cache above"),
    );

    let wrong = fixture.run_offset_readers(&warm, want, 6, |warm, regions| {
        // Ask for the whole traversal to be warmed while the readers run, so
        // the prefetch threads and the demand reads contend for one cache.
        warm.prefetch(0, regions).expect("a prefetch submission");
    });
    warm.drain_prefetch();

    assert_eq!(
        wrong, 0,
        "{wrong} reads disagreed with the uncached read of the same region while a \
         prefetcher was filling the same cache. The array is never written, so this is \
         the fill protocol handing out bytes that are not the chunk's."
    );

    // And the prefetcher was actually doing something — without this the test
    // passes for a prefetcher that never ran, which is this crate's empty-sink
    // trap and the exact reason the bug above went unseen for so long.
    let stats = warm.cache_stats().expect("a cache");
    assert!(
        stats.hits() > 0 && stats.misses > 0,
        "the capacity did not make the fixture both hit and evict: {stats:?}"
    );
    let prefetch = warm.prefetch_stats().expect("a prefetcher");
    assert!(
        prefetch.started > 0,
        "the prefetcher started nothing, so this test is a concurrency test with one \
         fewer thread than it claims: {prefetch:?}"
    );
}

/// Print what caching written intermediates buys on a two-reaching-phase plan.
///
/// Read hit counts, not `store bytes`: source-byte counts cover different image
/// sets when only image 0 is cacheable. The recorded comparison lives in
/// `docs/design/cache-and-prefetch.md` §1.5.
///
/// ```text
/// cargo test --release --features zarr --test zarr_cache -- --ignored --nocapture what_caching_the_intermediates
/// ```
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_what_caching_the_intermediates_saves() {
    use blockflow::ops::smooth::{Gaussian, SmoothOp};

    // Two reaching phases, so both the source and the intermediate are read with
    // a halo and both have something to reuse.
    let chain = Chain::sequence(vec![
        Chain::op(SmoothOp::new(
            "first",
            Gaussian::isotropic(1.0, 2.0).expect("a kernel"),
        )),
        Chain::op(SmoothOp::new(
            "second",
            Gaussian::isotropic(1.0, 2.0).expect("a kernel"),
        )),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::U16);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], 16).expect("a grid");
    // One phase per op, so there is a real intermediate image between them.
    let phases: Vec<PhaseDecomposition> = (0..slots.len())
        .map(|slot| {
            let reach = slots[slot].reach3(&VOLUME);
            PhaseDecomposition::derive(
                vec![slot],
                vec![names[slot].clone()],
                reach,
                reach,
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&VOLUME),
    };
    // A smoothing widens `u16` to `f64`, and a plan that does not say so is
    // refused — see `Decomposition::declare_dtypes`, which is the thing to call
    // rather than a width written here by hand.
    plan.declare_dtypes(&workflow.chain).expect("the widths");
    assert_eq!(plan.n_phases(), 2, "the measurement needs an intermediate");

    let chunk_bytes = (CHUNK.iter().product::<usize>() * 2) as u64;
    println!(
        "{:>16} {:>14} {:>10} {:>10}",
        "capacity", "store bytes", "hits", "misses"
    );
    for capacity in [0u64, chunk_bytes * 8, chunk_bytes * 64, chunk_bytes * 512] {
        let path = root(&format!("intermediates-{capacity}"));
        let mut env = ZarrEnvironment::create(&path, &source(), CHUNK).expect("a store");
        env = if capacity > 0 {
            env.with_cache(capacity)
        } else {
            env.without_cache()
        };
        execute("cached", &workflow, &plan, &Hints::default(), &env).expect("a run");
        match env.cache_stats() {
            Some(stats) => println!(
                "{capacity:>16} {:>14} {:>10} {:>10}",
                stats.source_bytes,
                stats.hits(),
                stats.misses
            ),
            None => println!("{:>16} {:>14} {:>10} {:>10}", "none", "-", "-", "-"),
        }
    }
}
