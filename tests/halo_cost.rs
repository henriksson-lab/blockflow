// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What a halo re-read actually costs, in time.**
//
// Every halo figure this project has produced is in **voxels** — read
// amplification, redundancy, traffic — and every one of them is priced as though
// re-reading a halo costs what reading a core costs. `CostModel` has a single
// `read_cost_per_voxel` and multiplies it by the read extent, so a voxel fetched
// for the first time and a voxel fetched again because it fell in a neighbour's
// halo are charged the same.
//
// They need not cost the same. A halo is, by construction, the region a
// *neighbouring* block just read, so it is the part of the volume most likely to
// be warm — in the CPU's cache on an in-memory path, in the page cache on a file
// one. This file measures that rather than assuming it, in either direction.
//
// The two questions are separate and are kept separate:
//
//  1. **Is a halo voxel cheaper than a core voxel on the same path?** Measured by
//     holding the block grid fixed and growing the halo, and comparing the
//     *time* ratio against the *voxel* ratio. If time grows more slowly than
//     voxels, the halo is cheaper and every figure in voxels overstates.
//  2. **Does the answer survive a cold cache?** The page cache is what makes a
//     file re-read cheap, and `src/npy.rs`'s coalescing measurement is this
//     project's precedent for the shape of that answer: 1.36x cold, 0.70x warm,
//     and a refusal to model a direction that depends on something the reader
//     cannot know.
//
// The discipline is that record's: **arms interleaved**, so a ratio is
// trustworthy even while absolutes move with the machine's afternoon; the best
// of several repetitions rather than a mean, so a scheduling hiccup cannot make
// an arm look slow; and no assertion anywhere on an absolute time.
//
// **The premise does not hold on this path, and that is the first finding.**
// `ArrayEnvironment::read` is a slice copy out of a resident `Array3` plus a
// fresh allocation — no file, no mmap, no page cache. `env.rs` says so itself
// about the chunk grid it carries: "an accounting fiction, used by
// `chunks_touched` to price IO that is not happening". So on the environment
// nearly every measurement in this project was taken through, "given io cache"
// has nothing to apply to; the halo's cost is memory bandwidth and an allocator,
// and neither is discounted by a page cache. What *can* discount it is the CPU's
// own cache, which is what `residency_report` isolates.
//
// **This does not re-measure the flat-file case, and that is deliberate.** A
// second worker measured a 1 GiB volume of 512 planes at halo 5 with
// `posix_fadvise(DONTNEED)` for the cold arm and reports `3.48x` the bytes for
// `1.32x` the time cold and `2.98x` warm, with cold `42x` warm at one block. That
// is the sequential layout; this file covers the two layouts that one does not —
// a resident array and a chunked store — and the three results only make sense
// together. `docs/ops-survey/README.md`'s G20 row reconciles them.
//
// What it measured
// ----------------
// ```text
// [512, 512, 512] u16, one region shape per row, best of 5
// region      cold source        warm source       warm/cold
// [32,32,32]  1.746 ns/voxel     0.540 ns/voxel        0.309
// [64,64,64]  0.651 ns/voxel     0.447 ns/voxel        0.687
// ```
//
// A warm source really is cheaper — by `1.5x` to `3.2x` — so a halo voxel *is*
// worth less than a core voxel here. Two things bound that, and both are in the
// table. There is a **floor**: the allocation and the write of the destination
// are paid whether the source was warm or cold, which is why the warm column
// does not approach zero. And the discount is **itself a function of the block
// shape** — `0.31` at `32^3` against `0.69` at `64^3` — so replacing one weight
// that is wrong per candidate with another weight that is wrong per candidate is
// not obviously progress. The halo sweep in `report` agrees from the other side:
// at edge 128, where it is monotone and trustworthy, `2.065x` the voxels cost
// `1.764x` the time, which is an effective halo weight of `0.72`.
//
// The sweep at edges 32 and 64 came out **non-monotone** on a machine at twice
// its core count in load and is not evidence of anything; it is kept because the
// residency measurement below exists to replace it and the reader should be able
// to see why.

use std::time::Instant;

use ndarray::Array3;

use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::region::Region;
use blockflow::voxels::Voxels;

/// Large enough that the source cannot sit in any last-level cache: `256^3`
/// `u16` is 32 MiB, and the read arms below walk all of it many times.
const VOLUME: [usize; 3] = [256, 256, 256];

fn source() -> Voxels {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 48) as u16
    })
    .into()
}

/// The read regions a grid of `edge` with a symmetric halo of `halo` produces,
/// in the order a scheduler visits them — which is the order that decides
/// whether a halo is warm.
fn read_regions(edge: usize, halo: usize) -> Vec<Region> {
    let grid = BlockGrid::new(VOLUME, [edge; 3]).expect("a grid");
    grid.cores()
        .iter()
        .map(|core| {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                let lo = core.core.start[axis].saturating_sub(halo);
                let hi = (core.core.start[axis] + core.core.shape[axis] + halo).min(VOLUME[axis]);
                start[axis] = lo;
                shape[axis] = hi - lo;
            }
            Region::new(&start, &shape)
        })
        .collect()
}

fn voxels_of(regions: &[Region]) -> u64 {
    regions.iter().map(|r| r.voxels() as u64).sum()
}

/// One pass over every block's read, timed. Returns seconds.
fn time_pass(env: &ArrayEnvironment, regions: &[Region]) -> f64 {
    let started = Instant::now();
    let mut sink = 0u64;
    for region in regions {
        let buf = env.read(0, region).expect("a read");
        if let blockflow::env::BlockBuf::Array(voxels) = buf {
            // Touch the result so the copy cannot be optimised away and so the
            // cost includes landing it, which is what a block op pays.
            sink = sink.wrapping_add(voxels.len() as u64);
        }
    }
    std::hint::black_box(sink);
    started.elapsed().as_secs_f64()
}

/// Best of `repeats`, arms interleaved by the caller.
fn best(env: &ArrayEnvironment, regions: &[Region], repeats: usize) -> f64 {
    let mut seen = f64::INFINITY;
    for _ in 0..repeats.max(1) {
        let elapsed = time_pass(env, regions);
        if elapsed.total_cmp(&seen).is_lt() {
            seen = elapsed;
        }
    }
    seen
}

/// The table: for one block edge, what each halo costs in voxels and in time.
pub fn report(edge: usize, repeats: usize) -> String {
    let env = ArrayEnvironment::new(source(), 1, [64, 64, 64]).expect("an environment");
    let halos = [0usize, 2, 4, 8, 16, 32, 35];
    let plans: Vec<(usize, Vec<Region>)> = halos
        .iter()
        .map(|&halo| (halo, read_regions(edge, halo)))
        .collect();
    let base_voxels = voxels_of(&plans[0].1) as f64;

    // **Interleaved.** Every arm is timed once before any arm is timed twice, so
    // a machine that gets busier part-way through spoils all the arms equally
    // instead of one of them.
    let mut times: Vec<f64> = vec![f64::INFINITY; plans.len()];
    for _ in 0..repeats.max(1) {
        for (index, (_, regions)) in plans.iter().enumerate() {
            let elapsed = time_pass(&env, regions);
            if elapsed.total_cmp(&times[index]).is_lt() {
                times[index] = elapsed;
            }
        }
    }

    let mut out = format!(
        "ArrayEnvironment, {VOLUME:?} u16, block edge {edge}, best of {repeats}, interleaved\n\
         halo  read voxels   voxel x   time s    time x   ns/read voxel   ns/core voxel\n"
    );
    for (index, (halo, regions)) in plans.iter().enumerate() {
        let voxels = voxels_of(regions) as f64;
        out.push_str(&format!(
            "{halo:4}  {:11}   {:7.3}  {:7.4}  {:7.3}  {:14.3}  {:14.3}\n",
            voxels as u64,
            voxels / base_voxels,
            times[index],
            times[index] / times[0],
            times[index] * 1e9 / voxels,
            times[index] * 1e9 / base_voxels,
        ));
    }
    out
}

#[test]
#[ignore = "a measurement, not an assertion"]
fn what_a_halo_costs_in_memory() {
    for edge in [32usize, 64, 128] {
        println!("{}", report(edge, 5));
    }
}

// --------------------------------------------- source residency, isolated --

/// **The same read, twice, differing only in whether its source is resident.**
///
/// The sweep above varies the halo, which varies the region size, the region
/// shape, the allocation size and the number of regions all at once — and on a
/// machine at twice its core count in load it came out non-monotone, which means
/// it is not evidence. This isolates the one variable the "halo is warm" claim
/// is about: **where the source bytes are**, holding the region size, the
/// allocation and the count fixed.
///
/// * **cold source** — regions spread across a volume far larger than any
///   last-level cache, each one read once, in an order that gives the prefetcher
///   nothing.
/// * **warm source** — the *same* region read again and again, so its source is
///   as resident as a source can be. This is a better warm case than a real halo
///   ever gets.
///
/// If a warm source is much cheaper per voxel than a cold one, a halo voxel is
/// cheaper than a core voxel on this path and every figure in voxels overstates.
/// If it is not, they do not.
pub fn residency_report(edge: usize, repeats: usize) -> String {
    // Well past this machine's 27.5 MiB of L3, so a "cold" source really is one.
    let big: [usize; 3] = [512, 512, 512];
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let source: Voxels = Array3::from_shape_fn((big[0], big[1], big[2]), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 48) as u16
    })
    .into();
    let env = ArrayEnvironment::new(source, 1, [64, 64, 64]).expect("an environment");

    // One region shape, used by both arms.
    let shape = [edge; 3];
    let per_axis = big[0] / edge;
    let spread: Vec<Region> = (0..per_axis)
        .flat_map(|a| {
            (0..per_axis).flat_map(move |b| {
                (0..per_axis).map(move |c| {
                    let start = [a * edge, b * edge, c * edge];
                    Region::new(&start, &shape)
                })
            })
        })
        .collect();
    let one = vec![Region::new(&[0, 0, 0], &shape); spread.len()];

    let mut cold = f64::INFINITY;
    let mut warm = f64::INFINITY;
    for _ in 0..repeats.max(1) {
        let elapsed = time_pass(&env, &spread);
        if elapsed.total_cmp(&cold).is_lt() {
            cold = elapsed;
        }
        let elapsed = time_pass(&env, &one);
        if elapsed.total_cmp(&warm).is_lt() {
            warm = elapsed;
        }
    }
    let voxels = (spread.len() * edge * edge * edge) as f64;
    format!(
        "ArrayEnvironment source residency, {big:?} u16, region {shape:?}, {} reads, best of {repeats}\n\
         cold source  {cold:8.4} s   {:7.3} ns/voxel\n\
         warm source  {warm:8.4} s   {:7.3} ns/voxel\n\
         warm/cold    {:8.3}\n",
        spread.len(),
        cold * 1e9 / voxels,
        warm * 1e9 / voxels,
        warm / cold,
    )
}

#[test]
#[ignore = "a measurement, not an assertion"]
fn what_source_residency_is_worth_in_memory() {
    for edge in [32usize, 64] {
        println!("{}", residency_report(edge, 5));
    }
}

// ---------------------------------------- the premise, as a property --

/// **A halo voxel is counted twice, by design — and this is where that stops
/// being a detail and becomes a statement.**
///
/// Everything above is a wall clock. This is not: it asserts the fact the
/// pricing rests on, which is that the crate's accounting makes no distinction
/// between a voxel fetched for the first time and one fetched again because it
/// fell in a neighbour's halo. `EnvCounters::read_voxels` is a monotone
/// `fetch_add` per read with no residency set, so overlapping reads are summed.
///
/// That is the right accounting for *bytes moved* and the wrong one for *time*,
/// and `docs/ops-survey/README.md`'s G20 row carries the measurements of how
/// wrong, in both directions, on three paths. **If a halo weight is ever
/// introduced, this assertion is what has to be inverted** — not deleted, since
/// the counter would still be counting bytes and only the price would change.
#[test]
fn the_accounting_counts_an_overlapping_read_twice() {
    use std::sync::atomic::Ordering;

    let env = ArrayEnvironment::new(source(), 1, [64, 64, 64]).expect("an environment");
    let whole = Region::new(&[0, 0, 0], &[64, 64, 64]);
    let before = env.counters().read_voxels.load(Ordering::SeqCst);
    env.read(0, &whole).expect("a read");
    let once = env.counters().read_voxels.load(Ordering::SeqCst) - before;
    env.read(0, &whole).expect("the same read again");
    let twice = env.counters().read_voxels.load(Ordering::SeqCst) - before;
    assert_eq!(once, 64 * 64 * 64, "one read counts its own voxels");
    assert_eq!(
        twice,
        2 * once,
        "the second read of the same region must be counted in full: the accounting is of bytes \
         moved, and it has no notion of a voxel already held"
    );

    // **Liveness.** The equality above would hold for a counter that ignored the
    // region and added a constant. A different-sized read must move it by that
    // different size.
    let smaller = Region::new(&[0, 0, 0], &[32, 32, 32]);
    let before = env.counters().read_voxels.load(Ordering::SeqCst);
    env.read(0, &smaller).expect("a read");
    assert_eq!(
        env.counters().read_voxels.load(Ordering::SeqCst) - before,
        32 * 32 * 32,
        "the counter must follow the region it was given"
    );
}
