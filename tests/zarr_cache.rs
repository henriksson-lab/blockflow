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
//! 3. **A written image is not cached**, because `ChunkCache` has no per-array
//!    invalidation and serving a stale chunk would be a wrong answer rather
//!    than a slow one.
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
use ndarray::Array3;

const VOLUME: [usize; 3] = [64, 64, 64];
const CHUNK: [usize; 3] = [16, 16, 16];

fn source() -> Voxels {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    Voxels::U16(Array3::from_shape_fn(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 48) as u16
        },
    ))
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
    let path =
        std::env::temp_dir().join(format!("blockflow-zarr-cache-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// **Same voxels, and the cache really served them.**
#[test]
fn a_cached_read_is_the_same_read_and_the_cache_serves_it() {
    let regions = regions();
    assert!(
        regions.len() > 8,
        "the fixture needs a traversal to revisit"
    );

    let cold_root = root("cold");
    let cold = ZarrEnvironment::create(&cold_root, &source(), CHUNK).expect("a store");
    assert!(
        cold.cache_stats().is_none(),
        "a cache must be asked for; an environment that acquires one by upgrading changes what \
         a read costs and what the process holds without anyone saying so"
    );
    let plain = read_all(&cold, &regions);

    let warm_root = root("warm");
    let warm = ZarrEnvironment::create(&warm_root, &source(), CHUNK)
        .expect("a store")
        // Sixteen chunks of `16^3` `u16` — big enough to hold a neighbourhood
        // and far too small to hold the volume, so it must evict and the hits
        // that remain are real reuse rather than "everything fits".
        .with_cache(16 * 16 * 16 * 16 * 2);
    let cached = read_all(&warm, &regions);

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

/// **A written image is never cached**, because a stale chunk is a wrong answer.
///
/// `ChunkCache` has no per-array invalidation — only `clear`, which throws the
/// source's chunks away too — so an image that a phase writes and the phase
/// above reads must be read directly. This writes image 1 through the
/// environment, reads it twice, and asserts the cache gained nothing: a hit
/// there would be a chunk served from before the write.
#[test]
fn a_written_image_is_not_cached() {
    let path = root("written");
    let env = ZarrEnvironment::create(&path, &source(), CHUNK)
        .expect("a store")
        .with_cache(1 << 24);
    // A one-phase plan, stated directly rather than searched for: what this
    // test needs from it is only that image 1 exists.
    let chain: Chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.5, 1.0, 0.0));
    let workflow = Workflow::new(chain, VOLUME, Dtype::U16);
    let reach = workflow.chain.reach3(&VOLUME);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], CHUNK[0]).expect("a grid");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names,
            reach,
            reach,
            grid,
        )],
        chain_reach: reach,
    };
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

    let block =
        blockflow::env::BlockBuf::Array(Voxels::zeros(Dtype::U16, [16, 16, 16]).expect("a block"));
    env.write(1, &region, &region, &block)
        .expect("image 1 is writable");
    let _ = env.read(1, &region).expect("image 1 reads");
    let _ = env.read(1, &region).expect("image 1 reads again");
    let after = env.cache_stats().expect("a cache");
    assert_eq!(
        after.hits(),
        baseline.hits(),
        "reading a written image twice added {} hits. `ChunkCache` cannot invalidate one array, \
         so a chunk cached before a write would be served after it — a wrong answer rather than \
         a slow one",
        after.hits() - baseline.hits()
    );
    assert_eq!(
        after.misses, baseline.misses,
        "a written image should not reach the cache at all, as a miss or otherwise"
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
    let mask = Voxels::Bool(Array3::from_shape_fn(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        |(z, y, x)| (z + y + x) % 3 == 0,
    ));
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
fn the_prefetch_sweep_has_a_control_at_both_ends() {
    // `f64`, because `threshold` states the element types it accepts and
    // `uint16` is not one — the plan is refused when it is made rather than
    // when a block reaches the op, which is the crate working as intended.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let volume = Voxels::F64(Array3::from_shape_fn(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        },
    ));
    let chain: Chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.5, 1.0, 0.0));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let reach = workflow.chain.reach3(&VOLUME);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], CHUNK[0]).expect("a grid");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names,
            reach,
            reach,
            grid,
        )],
        chain_reach: reach,
    };

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
/// same read, it really serves hits, a written image is excluded, a `bool`
/// volume round-trips. None of them says what it is **for**, and that gap is
/// load-bearing well outside this file:
///
/// * `simulate::Machine::cache_bytes` is the simulator's central lever, and
///   ordering-changes-hit-rate is the mechanism the whole module exists to
///   rank;
/// * `HandoutPolicy::CacheModelled` and `HandoutPolicy::Coalescing` are both
///   **refused at the caller boundary** on the grounds that "`cache::ChunkCache`
///   has no non-test construction site, so no `Environment::read` is served from
///   one";
/// * `distributed::placement` and `distributed::cache_model` repeat the same
///   sentence.
///
/// That sentence is exactly true and narrower than it sounds. `ChunkCache` *is*
/// constructed on the read path — `ZarrEnvironment::with_cache` does it, and
/// `read` is served through it for image 0 and supplied images. What is missing
/// is a **caller**: the only four call sites of `with_cache` in the repository
/// are in this file. So a cache exists, is reachable, is opt-in, and nobody opts
/// in.
///
/// The reason nobody opts in is that its value was never measured, and this is
/// that measurement: the same plan through the same environment, at a range of
/// budgets, reporting the bytes that actually left the store.
///
/// **The claim asserted here is byte counts, not time.** The store reads are
/// deterministic; a wall clock on a shared machine is not, and this crate does
/// not assert on durations. `print_what_the_cache_saves` beside it prints the
/// timing for a human.
#[test]
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

/// The same sweep with a clock on it, for a human deciding whether to turn a
/// cache on. **Ignored, because it is a measurement**: this crate asserts on
/// byte counts and never on durations, and the assertion above is the part that
/// belongs in a suite.
///
/// Recorded, release, best of three over this file's overlapping windows:
///
/// ```text
///     capacity        wall (ms)   store bytes
///     none                  7.9      uncached
///     one chunk            17.9     4 866 048
///     sixteen chunks        9.6     2 359 296
///     the whole volume      1.5       524 288
/// ```
///
/// Two things, and the second is the one nobody would guess:
///
/// * a cache that holds the working set is **5.3x faster** than none, and reads
///   each chunk exactly once — 524 288 bytes is the volume, so the halo re-reads
///   are entirely absorbed;
/// * a cache that is **too small is worse than none at all**: 17.9 ms against
///   7.9, because it pays the bookkeeping on every read and never reuses
///   anything. The same shape as every other locality mechanism this crate has
///   measured — below a threshold set by the working set, the machinery costs
///   more than the sharing saves.
///
/// So "turn the cache on" is not the whole recommendation. It is "turn it on
/// with a budget that holds a block's read extent times the blocks in flight",
/// and below that leave it off.
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
