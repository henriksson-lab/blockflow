// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What a halo re-read costs through storage, warm and cold.**
//
// The claim this exists to test: a halo is by construction the region a
// neighbouring block just read, so it is the part of the volume most likely to
// be in the page cache, and every figure this project states in *voxels* may
// therefore overstate what a halo costs in *time*.
//
// It is a claim about IO, so it has to be tested on a path that does IO —
// `ArrayEnvironment` does none, and `tests/halo_cost.rs` says what it does
// instead. `ZarrEnvironment` is the crate's storage environment and this is its
// measurement.
//
// **The design separates the two things the page cache is being credited with.**
// A read of a compressed chunk is a file read *plus a decompression*, and the
// page cache can only remove the first. So every arm below is run at two
// compressions:
//
// * `Compression::None` — the bytes codec verbatim. Here a warm re-read really
//   is close to free, and if the halo is ever cheap it is cheap here.
// * `Compression::Gzip(1)` — which is **the crate's default for every integer
//   dtype** (`Compression::for_dtype`), so it is what a run actually pays. A
//   warm re-read still pays the inflate.
//
// crossed with cold and warm, at halo 0 and at a halo the ops in this crate
// really declare. The number to read is the **time ratio against the voxel
// ratio**: if a halo of `r` costs `k` times the voxels and `k` times the time,
// the voxel figures are honest; if it costs `k` times the voxels and much less
// than `k` times the time, they overstate.
//
// **Cold is produced the way `src/npy.rs`'s coalescing measurement produced
// it** — `posix_fadvise(POSIX_FADV_DONTNEED)` on the store's files, here through
// `dd oflag=nocache`, which is that call. No root, no `drop_caches`, and only
// this run's own files are evicted.
//
// **Arms are interleaved and reported as ratios**, because this measurement was
// taken on a machine whose load average was twice its core count. Absolutes move
// with the afternoon; a ratio between two arms timed side by side does not.
// Nothing here asserts on an absolute time.
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
// ZarrEnvironment, [256, 256, 256] u16, chunk [64, 64, 64], block edge 64, best of 2
//
// compression: none
// halo  read voxels   voxel x   cold s   cold x   warm s   warm x   warm/cold  unaligned
//    0     16777216     1.000    0.015    1.000    0.014    1.000      0.909          0
//    4     21952000     1.308    2.430  159.019    1.106   79.616      0.455         64
//    8     28094464     1.675    2.126  139.127    0.877   63.137      0.412         64
//   16     43614208     2.600    2.042  133.605    0.954   68.717      0.467         64
//   64    262144000    15.625    0.669   43.783    0.402   28.970      0.601          0
//
// compression: gzip(1) — the crate's default for integers
// halo  read voxels   voxel x   cold s   cold x   warm s   warm x   warm/cold  unaligned
//    0     16777216     1.000    0.281    1.000    0.283    1.000      1.004          0
//    4     21952000     1.308    1.501    5.332    1.249    4.421      0.832         64
//    8     28094464     1.675    1.538    5.466    1.202    4.256      0.782         64
//   16     43614208     2.600    1.582    5.621    1.297    4.591      0.820         64
//   64    262144000    15.625    1.679    5.966    1.567    5.547      0.933          0
// ```
//
// **The control row is the whole answer, and it points the other way from the
// question.** A halo of 64 fetches **15.6x** the voxels of no halo and **12x**
// the voxels of a halo of 4 — and costs **3.6x less** than the halo of 4
// uncompressed, and about the same under gzip. The voxel count does not explain
// the cost. What explains it is the `unaligned` column: a halo of 4, 8 or 16
// makes every read land part-way into a chunk, and a halo of a whole chunk lands
// on chunk boundaries again.
//
// So on a chunked store a halo is **dearer** than its voxel count, not cheaper.
// Under the crate's default codec a halo of 4 costs `5.3x` the time for `1.31x`
// the voxels — a four-fold *under*-charge by the voxel model, in the direction
// nobody was worried about.
//
// **And the page cache buys little of it.** `warm/cold` under gzip is `0.78` to
// `1.00`: a second pass over data that is entirely in the page cache is barely
// faster, because the inflate is paid again and there is no in-process chunk
// cache to skip it — `src/cache.rs`'s `ChunkCache` has **no non-test
// construction site anywhere in the crate**. Uncompressed, where there is
// nothing to re-decode, the page cache is worth about `2x` (`0.41`-`0.61`), which
// is the only place in this table the original hypothesis holds.

#![cfg(feature = "zarr")]

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use ndarray::Array3;

use blockflow::env::Environment;
use blockflow::geometry::BlockGrid;
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{Compression, CompressionPolicy, ZarrEnvironment};

const VOLUME: [usize; 3] = [256, 256, 256];
const CHUNK: [usize; 3] = [64, 64, 64];

/// **Compressible, but not trivially so.** A field of pure noise would make
/// `Gzip` cost its inflate and buy nothing, and a constant field would make a
/// chunk a handful of bytes; either would answer a different question from the
/// one a real volume asks. This is a smooth ramp with a little noise on it,
/// which is what an acquisition looks like to a compressor.
fn source() -> Voxels {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(a, b, c)| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state >> 58) as u16;
        ((a + b + c) as u16).wrapping_mul(7).wrapping_add(noise)
    })
    .into()
}

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

/// `posix_fadvise(POSIX_FADV_DONTNEED)` over every file under `root`, which is
/// exactly what `dd oflag=nocache conv=notrunc,fdatasync count=0` issues.
///
/// Returns the number of files it evicted, so a caller can assert it did
/// something — a cold arm that silently evicted nothing is a warm arm wearing
/// its name, which is the failure this whole file would not survive.
fn evict(root: &Path) -> usize {
    let mut evicted = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let status = Command::new("dd")
                    .arg(format!("of={}", path.display()))
                    .arg("oflag=nocache")
                    .arg("conv=notrunc,fdatasync")
                    .arg("count=0")
                    .arg("status=none")
                    .status();
                if matches!(status, Ok(code) if code.success()) {
                    evicted += 1;
                }
            }
        }
    }
    evicted
}

fn time_pass(env: &ZarrEnvironment, regions: &[Region]) -> f64 {
    let started = Instant::now();
    let mut sink = 0u64;
    for region in regions {
        let buf = env.read(0, region).expect("a read");
        if let blockflow::env::BlockBuf::Array(voxels) = buf {
            sink = sink.wrapping_add(voxels.len() as u64);
        }
    }
    std::hint::black_box(sink);
    started.elapsed().as_secs_f64()
}

struct Arm {
    halo: usize,
    voxels: u64,
    cold: f64,
    warm: f64,
    /// Reads whose stored region did not land on whole chunks, which
    /// `ZarrEnvironment` counts for itself. **The control on this file's
    /// explanation**: if a halo is dear because it destroys chunk alignment
    /// rather than because it fetches more voxels, this is the column that says
    /// so, and a halo of a whole chunk must bring it back to zero.
    unaligned: u64,
}

fn measure(
    root: &Path,
    compression: Compression,
    edge: usize,
    halos: &[usize],
    repeats: usize,
) -> Vec<Arm> {
    let _ = std::fs::remove_dir_all(root);
    let env = ZarrEnvironment::create_with_compression(
        root,
        &source(),
        CHUNK,
        CompressionPolicy::uniform(compression),
    )
    .expect("a store");
    let plans: Vec<(usize, Vec<Region>)> = halos
        .iter()
        .map(|&halo| (halo, read_regions(edge, halo)))
        .collect();
    let mut cold = vec![f64::INFINITY; plans.len()];
    let mut warm = vec![f64::INFINITY; plans.len()];
    let mut unaligned = vec![0u64; plans.len()];
    for _ in 0..repeats.max(1) {
        for (index, (_, regions)) in plans.iter().enumerate() {
            // Cold: evict, then one pass. The eviction is asserted to have
            // touched files by the caller.
            assert!(evict(root) > 0, "the cold arm evicted no files");
            let before = env.unaligned_reads();
            let elapsed = time_pass(&env, regions);
            unaligned[index] = env.unaligned_reads() - before;
            if elapsed.total_cmp(&cold[index]).is_lt() {
                cold[index] = elapsed;
            }
            // Warm: the pass immediately after, so every byte it wants was just
            // read by the pass before it. This is the most favourable warm case
            // there is, which is the right way to test a claim that warmth is
            // what makes a halo cheap.
            let elapsed = time_pass(&env, regions);
            if elapsed.total_cmp(&warm[index]).is_lt() {
                warm[index] = elapsed;
            }
        }
    }
    plans
        .iter()
        .enumerate()
        .map(|(index, (halo, regions))| Arm {
            halo: *halo,
            voxels: voxels_of(regions),
            cold: cold[index],
            warm: warm[index],
            unaligned: unaligned[index],
        })
        .collect()
}

pub fn report(edge: usize, repeats: usize) -> String {
    let base = std::env::var("BLOCKFLOW_HALO_IO_DIR").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("blockflow-halo-io")
            .display()
            .to_string()
    });
    // **`64` is the control and it is the point of the list.** The chunk is
    // `64` on every axis, so a halo of 4, 8 or 16 makes every read land part-way
    // into a chunk while a halo of a whole chunk lands on chunk boundaries
    // again — and fetches *far more* voxels than any of them. If the halo's cost
    // tracked its voxels, 64 would be the dearest row by a distance. If it
    // tracks alignment, it will not be.
    let halos = [0usize, 4, 8, 16, 64];
    let mut out = format!(
        "ZarrEnvironment, {VOLUME:?} u16, chunk {CHUNK:?}, block edge {edge}, best of {repeats}\n"
    );
    for (name, compression) in [
        ("none", Compression::None),
        (
            "gzip(1) — the crate's default for integers",
            Compression::Gzip(1),
        ),
    ] {
        let root = Path::new(&base).join(name.split_whitespace().next().unwrap());
        let arms = measure(&root, compression, edge, &halos, repeats);
        let base_voxels = arms[0].voxels as f64;
        out.push_str(&format!(
            "\ncompression: {name}\n\
             halo  read voxels   voxel x   cold s   cold x   warm s   warm x   warm/cold  unaligned\n"
        ));
        for arm in &arms {
            out.push_str(&format!(
                "{:4}  {:11}   {:7.3}  {:7.3}  {:7.3}  {:7.3}  {:7.3}  {:9.3}  {:9}\n",
                arm.halo,
                arm.voxels,
                arm.voxels as f64 / base_voxels,
                arm.cold,
                arm.cold / arms[0].cold,
                arm.warm,
                arm.warm / arms[0].warm,
                arm.warm / arm.cold,
                arm.unaligned,
            ));
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    out
}

#[test]
#[ignore = "a measurement, not an assertion"]
fn what_a_halo_costs_through_storage() {
    println!("{}", report(64, 2));
}

/// **The durable half of this file's finding, with no timing in it at all.**
///
/// The measurement above is a wall clock on a shared machine and says so. This
/// is the same finding as a property: on a chunked store a halo's first effect
/// is not that it fetches more voxels, it is that it **stops the read landing on
/// whole chunks** — and `ZarrEnvironment` counts that for itself, so the claim
/// can be asserted rather than timed.
///
/// The control is the halo of a whole chunk. It fetches far more voxels than a
/// halo of four, and it is *aligned*, so if the cost of a halo tracked its voxel
/// count this would be the worst case and it is instead the same as no halo at
/// all.
#[test]
fn a_halo_stops_a_read_landing_on_whole_chunks_and_a_chunk_wide_halo_does_not() {
    let root = std::env::temp_dir().join(format!("blockflow-halo-align-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let small: [usize; 3] = [128, 128, 128];
    let chunk = [32usize, 32, 32];
    let mut state = 0x51ED_270F_A2C1_0003u64;
    let voxels: Voxels = Array3::from_shape_fn((small[0], small[1], small[2]), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 52) as u16
    })
    .into();
    let env = ZarrEnvironment::create(&root, &voxels, chunk).expect("a store");

    let regions = |halo: usize| -> Vec<Region> {
        let grid = BlockGrid::new(small, [32; 3]).expect("a grid");
        grid.cores()
            .iter()
            .map(|core| {
                let mut start = [0usize; 3];
                let mut shape = [0usize; 3];
                for axis in 0..3 {
                    let lo = core.core.start[axis].saturating_sub(halo);
                    let hi =
                        (core.core.start[axis] + core.core.shape[axis] + halo).min(small[axis]);
                    start[axis] = lo;
                    shape[axis] = hi - lo;
                }
                Region::new(&start, &shape)
            })
            .collect()
    };
    let unaligned_for = |halo: usize| -> (u64, usize, u64) {
        let before = env.unaligned_reads();
        let list = regions(halo);
        for region in &list {
            env.read(0, region).expect("a read");
        }
        (
            env.unaligned_reads() - before,
            list.len(),
            list.iter().map(|r| r.voxels() as u64).sum(),
        )
    };

    let (none, blocks, base_voxels) = unaligned_for(0);
    assert_eq!(
        none, 0,
        "with the block grid equal to the chunk grid every read is whole chunks"
    );
    let (four, _, four_voxels) = unaligned_for(4);
    assert_eq!(
        four, blocks as u64,
        "a halo of four must make every one of the {blocks} reads partial"
    );
    let (whole, _, whole_voxels) = unaligned_for(32);
    assert_eq!(
        whole, 0,
        "a halo of a whole chunk lands on chunk boundaries again"
    );
    // **The control's teeth.** The aligned halo fetches strictly more voxels
    // than the unaligned one, so "more voxels" and "more partial reads" point in
    // opposite directions here — which is what makes the counter, and not the
    // voxel count, the thing that explains the timings above.
    assert!(
        whole_voxels > four_voxels && four_voxels > base_voxels,
        "voxels must rise with the halo — {base_voxels}, {four_voxels}, {whole_voxels} — or this \
         control is not opposing anything"
    );
    let _ = std::fs::remove_dir_all(&root);
}

// ------------------------------------- what buying chunk alignment costs --

/// The read region of a block, **rounded outwards to whole chunks**.
///
/// This is the only way to buy alignment for a halo the op did not choose: the
/// read starts at `core - halo`, so it lands on a chunk boundary exactly when
/// the *halo* is a whole number of chunks, and a halo is the op's reach. So the
/// lattice cannot be adjusted to fix it — only the fetch can, by over-reading to
/// the boundary.
fn rounded_regions(volume: [usize; 3], edge: usize, halo: usize, chunk: [usize; 3]) -> Vec<Region> {
    let grid = BlockGrid::new(volume, [edge; 3]).expect("a grid");
    grid.cores()
        .iter()
        .map(|core| {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                let lo = core.core.start[axis].saturating_sub(halo);
                let hi = (core.core.start[axis] + core.core.shape[axis] + halo).min(volume[axis]);
                let lo = lo - lo % chunk[axis];
                let hi = hi.div_ceil(chunk[axis]) * chunk[axis];
                let hi = hi.min(volume[axis]);
                start[axis] = lo;
                shape[axis] = hi - lo;
            }
            Region::new(&start, &shape)
        })
        .collect()
}

/// **What buying chunk alignment costs, as a function of how many chunks wide a
/// block is.**
///
/// ```text
/// [256, 256, 256] u16, chunk [32, 32, 32], halo 4, best of 2
/// compression: none                          compression: gzip(1)
/// edge  chunks  arm      voxel x  cold s     cold s   unaligned
///   32       1  plain      1.000   2.330      2.464         512
///   32       1  rounded   11.488   4.841      3.956           0
///   64       2  plain      1.000  13.844      1.127          64
///   64       2  rounded    4.096   2.478      0.984           0
///  128       4  plain      1.000   4.880      0.374           8
///  128       4  rounded    1.781   1.182      0.242           0
/// ```
///
/// **The lever is real, conditional, and smaller than it looked.** Rounding out
/// pays only when a block is **at least two chunks wide** — at one chunk per
/// edge the over-read is `11.5x` the voxels and costs `1.6x`-`2.1x` *more*. At
/// two and four chunks it wins: `1.15x` and `1.55x` under the default codec,
/// `5.6x` and `4.1x` uncompressed.
///
/// **And it only pays cold.** Warm, rounding out is consistently a small loss —
/// `2.206`-`2.495`, `0.433`-`0.479`, `0.125`-`0.132` under gzip — because it is
/// strictly more bytes and there is no IO left to save. So the *sign* of this
/// lever depends on page-cache residency, which G20 established is the one thing
/// neither the planner nor the caller can know.
///
/// **Which relocates it.** A fetch that rounds itself out to chunk boundaries
/// needs no `Constraints` field and no plan change: `ZarrEnvironment::read`
/// already holds both the requested region and `array.chunk`, and the condition
/// — is this block at least two chunks wide — is visible there too. It is an
/// environment decision, not a cost-model one, and G20's row records that the
/// recommendation moved.
///
/// The claim this tests is the one the halo table points at: on a chunked store
/// the cost tracks partial reads rather than voxels, so over-reading to the
/// chunk boundary should buy back the partial reads at the price of more bytes.
/// Whether that is a trade worth making depends entirely on **how large the core
/// is against the chunk** — round a four-voxel halo out to a 32-voxel boundary
/// on a 32-wide block and the read triples; do it on a 128-wide block and it
/// grows by a third.
pub fn alignment_report(repeats: usize) -> String {
    let base = std::env::var("BLOCKFLOW_HALO_IO_DIR").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("blockflow-halo-io")
            .display()
            .to_string()
    });
    let chunk = [32usize, 32, 32];
    let halo = 4usize;
    let mut out = format!(
        "ZarrEnvironment, {VOLUME:?} u16, chunk {chunk:?}, halo {halo}, best of {repeats}\n\
         the two arms are the same blocks: the plain read, and the same read grown out to whole chunks\n"
    );
    for (name, compression) in [("none", Compression::None), ("gzip1", Compression::Gzip(1))] {
        let root = Path::new(&base).join(format!("align-{name}"));
        let _ = std::fs::remove_dir_all(&root);
        let env = ZarrEnvironment::create_with_compression(
            &root,
            &source(),
            chunk,
            CompressionPolicy::uniform(compression),
        )
        .expect("a store");
        out.push_str(&format!(
            "\ncompression: {name}\n\
             edge  chunks/edge  arm       read voxels   voxel x   cold s   warm s   unaligned\n"
        ));
        for edge in [32usize, 64, 128] {
            let plain = read_regions(edge, halo);
            let rounded = rounded_regions(VOLUME, edge, halo, chunk);
            let base_voxels = voxels_of(&plain) as f64;
            for (arm, regions) in [("plain", &plain), ("rounded", &rounded)] {
                let mut cold = f64::INFINITY;
                let mut warm = f64::INFINITY;
                let mut unaligned = 0u64;
                for _ in 0..repeats.max(1) {
                    assert!(evict(&root) > 0, "the cold arm evicted no files");
                    let before = env.unaligned_reads();
                    let elapsed = time_pass(&env, regions);
                    unaligned = env.unaligned_reads() - before;
                    if elapsed.total_cmp(&cold).is_lt() {
                        cold = elapsed;
                    }
                    let elapsed = time_pass(&env, regions);
                    if elapsed.total_cmp(&warm).is_lt() {
                        warm = elapsed;
                    }
                }
                out.push_str(&format!(
                    "{edge:4}  {:11}  {arm:8}  {:11}   {:7.3}  {:7.3}  {:7.3}  {:9}\n",
                    edge / chunk[0],
                    voxels_of(regions),
                    voxels_of(regions) as f64 / base_voxels,
                    cold,
                    warm,
                    unaligned,
                ));
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    out
}

#[test]
#[ignore = "a measurement, not an assertion"]
fn what_buying_chunk_alignment_costs() {
    println!("{}", alignment_report(2));
}

/// **The op's stride and the store's chunk constrain two different lattices, and
/// that is why they do not simply compose.**
///
/// `AxisReach::Aligned` constrains the **core** lattice: it wants the block edge
/// to be a whole number of the op's stride, because `BlockGrid::cores` builds
/// `start = index * edge` and so an edge that divides makes every core start
/// land on a tile boundary. Chunk alignment constrains the **read** lattice, and
/// the read starts at `core - halo`. So an aligned edge is not enough: the halo
/// has to be a whole number of chunks too, and a halo is the op's reach, chosen
/// for a kernel and never for a store.
///
/// This asserts exactly that, on counters rather than a clock: **every row below
/// has a block edge that is a whole number of chunks**, and only the halo
/// differs. If edge alignment were the binding constraint they would all be
/// aligned; instead the two whose halo divides the chunk are aligned and the two
/// whose halo does not are not.
#[test]
fn an_aligned_edge_is_not_enough_because_the_halo_shifts_the_read() {
    let root = std::env::temp_dir().join(format!("blockflow-halo-compose-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let small: [usize; 3] = [128, 128, 128];
    let chunk = [32usize, 32, 32];
    let edge = 64usize; // two whole chunks: the core lattice is aligned throughout
    assert_eq!(
        edge % chunk[0],
        0,
        "the edge must divide the chunk in every row"
    );
    let mut state = 0xDEAD_BEEF_CAFE_0011u64;
    let voxels: Voxels = Array3::from_shape_fn((small[0], small[1], small[2]), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 52) as u16
    })
    .into();
    let env = ZarrEnvironment::create(&root, &voxels, chunk).expect("a store");

    let unaligned_for = |halo: usize| -> u64 {
        let grid = BlockGrid::new(small, [edge; 3]).expect("a grid");
        let before = env.unaligned_reads();
        for core in grid.cores() {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                let lo = core.core.start[axis].saturating_sub(halo);
                let hi = (core.core.start[axis] + core.core.shape[axis] + halo).min(small[axis]);
                start[axis] = lo;
                shape[axis] = hi - lo;
            }
            env.read(0, &Region::new(&start, &shape)).expect("a read");
        }
        env.unaligned_reads() - before
    };

    let blocks = BlockGrid::new(small, [edge; 3]).expect("a grid").n_blocks() as u64;
    // A halo that is a whole number of chunks keeps the read aligned.
    assert_eq!(
        unaligned_for(0),
        0,
        "no halo: the read is the core and the core is aligned"
    );
    assert_eq!(
        unaligned_for(32),
        0,
        "a halo of a whole chunk keeps it aligned"
    );
    // A halo that is not makes every read partial, on the same core lattice.
    assert_eq!(
        unaligned_for(4),
        blocks,
        "a four-voxel halo shifts every read off the grid"
    );
    assert_eq!(unaligned_for(8), blocks, "and so does an eight-voxel one");
    // **Liveness.** The edge really is aligned, so these results are about the
    // halo and not about a lattice that was never on the chunk grid.
    assert_eq!(
        unaligned_for(0),
        0,
        "if the core lattice were itself unaligned this whole comparison would say nothing"
    );
    let _ = std::fs::remove_dir_all(&root);
}
