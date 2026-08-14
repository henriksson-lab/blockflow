// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The claim that matters: the storage layer is invisible to the answer.**
//
// Everything this crate had proven before `ZarrEnvironment` existed was proven
// against arrays already in memory. That is a real gap, and the way to close it
// is not to test the storage layer in isolation — a round trip proves a round
// trip — but to run the *existing* ops, over the *existing* synthetic data,
// through the *existing* executor, and assert the result is what the in-memory
// environment produced, voxel for voxel.
//
// So `ArrayEnvironment` is the oracle here, exactly as the whole-volume kernels
// are the oracle in `image_ops.rs`: the older, slower, less general
// implementation is the thing that says what "correct" means. If a chunk grid,
// a partial-chunk write or a fill value were wrong, this is where it shows.
//
// Four properties, in order of what they would catch:
//
// 1. **Byte identity.** Every op family, through storage, equals the same op
//    through memory.
// 2. **Under concurrency.** The same, with the executor writing blocks from a
//    pool — which is the configuration the partial-chunk race lives in.
// 3. **Decomposition invariance survives storage.** The property the framework
//    rests on does not become weaker for having been written to a disk.
// 4. **Chunk-exclusive writing is a mandate.** Every chunk of a level is
//    written by exactly one task. A level nobody outside the run reads derives
//    its chunking from the block grid that writes it, so the invariant is free;
//    a plan whose blocks straddle a chunk grid a caller *dictated* is refused,
//    naming the chunk. Alignment on the **read** side stays a performance fact
//    and nothing else — a halo straddles chunks by definition.
// 5. **Compression is invisible to the answer, and visible on the disk.** The
//    same chain compressed, uncompressed and in memory agrees voxel for voxel;
//    what changes is `stored_bytes`, and by how much is measured here rather
//    than asserted from a document.

#![cfg(feature = "zarr")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain, Output};
use blockflow::ops::{
    AdaptiveThresholdOp, ElementShape, LocalStatistic, LocalStatisticOp, Morphology, MorphologyOp,
    Rank, RankFilterOp, Statistic, StructuringElement, VoxelwiseMapOp,
};
use blockflow::probes::{NonZeroOp, SideOutputOp};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{chunk_for_block, Compression, CompressionPolicy, ZarrEnvironment};
use blockflow::{Dtype, Region};

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

/// A directory nobody else is using. Removed by [`Scratch`]'s `Drop`, so a test
/// that panics still cleans up after itself.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "blockflow-zarr-env-{}-{name}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A generated volume with structure at several scales, so that a filter has
/// something to do and a chunk seam has something to get wrong.
///
/// `Scene`'s region generation is byte-identical to the corresponding cut of a
/// whole generation, which is what makes it valid data for a decomposition test
/// rather than merely convenient.
fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250812)
            .with_objects(40)
            .with_radius(1.5, 4.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                array[[i, j, k]] = rendered.intensity[[i, j, k]];
            }
        }
    }
    array
}

fn mask(input: &Array3<f64>, level: f64) -> Array3<f64> {
    input.mapv(|value| if value > level { 1.0 } else { 0.0 })
}

fn box_element(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, radius)
}

fn ball(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, radius)
}

/// One phase holding the whole chain, at a given block edge, split on every
/// axis. Built from the chain's **own** reach: nothing here supplies one, so
/// nothing here can hide one that is wrong.
fn plan(workflow: &Workflow, block: usize) -> Decomposition {
    let reach = workflow.chain.reach3(&VOLUME);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

/// The same run, in memory.
fn through_memory(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    hints: &Hints,
) -> Voxels {
    let env = ArrayEnvironment::for_decomposition(input.clone(), decomposition, [8, 8, 8]).unwrap();
    execute("memory", workflow, decomposition, hints, &env).unwrap();
    env.output()
}

/// The same run, through Zarr arrays on a disk, at whatever compression the
/// environment's default is.
fn through_storage(
    root: &PathBuf,
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    chunk: [usize; 3],
    hints: &Hints,
) -> (Voxels, ZarrEnvironment) {
    through_storage_with(
        root,
        workflow,
        decomposition,
        input,
        chunk,
        hints,
        CompressionPolicy::derived(),
    )
}

/// The same run, with the codec of every level said explicitly, and the wall
/// clock it took.
///
/// The elapsed time covers `prepare` and the execution and nothing else — not
/// the scene generation, not the read-back — because what is being compared
/// across policies is the cost of encoding and decoding, and everything outside
/// that is common to all of them.
fn through_storage_timed(
    root: &PathBuf,
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    chunk: [usize; 3],
    hints: &Hints,
    compression: CompressionPolicy,
) -> (Voxels, ZarrEnvironment, std::time::Duration) {
    let env = ZarrEnvironment::create_with_compression(root, input, chunk, compression).unwrap();
    let started = std::time::Instant::now();
    execute("storage", workflow, decomposition, hints, &env).unwrap();
    let elapsed = started.elapsed();
    let output = env.output().unwrap();
    (output, env, elapsed)
}

fn through_storage_with(
    root: &PathBuf,
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    chunk: [usize; 3],
    hints: &Hints,
    compression: CompressionPolicy,
) -> (Voxels, ZarrEnvironment) {
    let (output, env, _) = through_storage_timed(
        root,
        workflow,
        decomposition,
        input,
        chunk,
        hints,
        compression,
    );
    (output, env)
}

/// Equality that says where it failed, and treats two NaNs as equal.
///
/// `Voxels` is `PartialEq`, but a level's unwritten sentinel is `f64::NAN` and
/// `NaN != NaN` — so a bare `assert_eq!` would report a difference between two
/// runs that had agreed perfectly about which voxels nobody wrote. Comparing
/// them as equal is the honest thing: "unwritten here and unwritten there" is
/// agreement.
fn assert_same(left: &Voxels, right: &Voxels, what: &str) {
    assert_eq!(left.dtype(), right.dtype(), "{what}: element type");
    assert_eq!(left.shape(), right.shape(), "{what}: shape");
    let left_values = left.widened();
    let right_values = right.widened();
    for (index, (a, b)) in left_values.iter().zip(right_values.iter()).enumerate() {
        let agree = a == b || (a.is_nan() && b.is_nan());
        assert!(
            agree,
            "{what}: element {index} is {a} in memory and {b} in storage"
        );
    }
}

/// One chain per op family, with the element type its `accepts` requires.
fn cases(input: &Array3<f64>) -> Vec<(&'static str, Chain, Voxels)> {
    let masked = mask(input, 0.35);
    let box3 = box_element([1, 1, 1]);
    let ball2 = ball([2, 1, 1]);
    let window = box_element([1, 2, 1]);

    vec![
        (
            "voxelwise threshold",
            Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0)),
            input.clone().into(),
        ),
        (
            "median, box element",
            Chain::op(RankFilterOp::median("median", box3.clone())),
            input.clone().into(),
        ),
        (
            "rank 1/4, ellipsoid element",
            Chain::op(RankFilterOp::new(
                "rank",
                ball2.clone(),
                Rank::Nth(ball2.len() / 4),
            )),
            input.clone().into(),
        ),
        (
            "open",
            Chain::op(MorphologyOp::new("open", Morphology::Open, box3.clone())),
            masked.clone().into(),
        ),
        (
            "close",
            Chain::op(MorphologyOp::new("close", Morphology::Close, ball2)),
            masked.clone().into(),
        ),
        (
            "local deviation on a lattice",
            Chain::op(LocalStatisticOp::new(
                "deviation",
                LocalStatistic::new(window.clone(), [8, 8, 8], Statistic::Deviation).unwrap(),
            )),
            input.clone().into(),
        ),
        (
            "adaptive threshold against a local mean",
            Chain::op(AdaptiveThresholdOp::new(
                "adaptive",
                LocalStatistic::new(window.clone(), [5, 4, 3], Statistic::Mean).unwrap(),
                1.0,
                0.02,
            )),
            input.clone().into(),
        ),
        (
            "a chain: median, adaptive threshold, open",
            Chain::sequence(vec![
                Chain::op(RankFilterOp::median("median", box3.clone())),
                Chain::op(AdaptiveThresholdOp::new(
                    "adaptive",
                    LocalStatistic::new(window, [5, 4, 3], Statistic::Mean).unwrap(),
                    1.0,
                    0.01,
                )),
                Chain::op(MorphologyOp::new("open", Morphology::Open, box3)),
            ]),
            input.clone().into(),
        ),
        (
            "a not, to reach the bool path",
            Chain::op(VoxelwiseMapOp::not("not")),
            masked.into(),
        ),
    ]
}

// ----------------------------------------------- 1. the claim that matters --

/// Every op family, run through Zarr arrays on a disk, produces exactly what
/// the same op produces through arrays in memory — at two block edges and two
/// chunk shapes, one of which does not divide the other.
#[test]
fn every_op_through_storage_is_byte_identical_to_the_same_op_in_memory() {
    let input = intensities();
    for (name, chain, source) in cases(&input) {
        let workflow = Workflow::new(chain, VOLUME, source.dtype());
        for block in [8, 12] {
            for chunk in [[8, 8, 8], [5, 7, 6]] {
                let decomposition = plan(&workflow, block);
                let hints = Hints::default();
                let memory = through_memory(&workflow, &decomposition, &source, &hints);

                let scratch = Scratch::new("identity");
                let (storage, env) = through_storage(
                    scratch.path(),
                    &workflow,
                    &decomposition,
                    &source,
                    chunk,
                    &hints,
                );
                assert_same(
                    &memory,
                    &storage,
                    &format!("{name} at block {block}, chunk {chunk:?}"),
                );
                // And the run really did go through storage, rather than
                // through some path that quietly did nothing.
                let (read_bytes, written_bytes) = env.counters().byte_snapshot();
                assert!(read_bytes > 0 && written_bytes > 0, "{name}: nothing moved");
            }
        }
    }
}

// ------------------------------------ 1b. and compression changes none of it --

/// **The claim compression is allowed to make, and the only one.** Every op
/// family, run through a compressed store, an uncompressed store and memory,
/// gives one answer.
///
/// Four policies rather than two, because "compressed" is not one thing: the
/// derived default leaves a `float64` level raw, so a run that only compared
/// `derived` against `None` would never have compressed the levels these chains
/// mostly produce. The `uniform` arms compress every level whatever it holds,
/// which is what actually exercises the codec on `float64`.
#[test]
fn compression_is_invisible_to_the_answer_for_every_op_family() {
    let input = intensities();
    for (name, chain, source) in cases(&input) {
        let workflow = Workflow::new(chain, VOLUME, source.dtype());
        let decomposition = plan(&workflow, 12);
        let hints = Hints::default();
        let memory = through_memory(&workflow, &decomposition, &source, &hints);

        for policy in [
            ("none", CompressionPolicy::uniform(Compression::None)),
            ("derived", CompressionPolicy::derived()),
            ("gzip1", CompressionPolicy::uniform(Compression::Gzip(1))),
            ("gzip9", CompressionPolicy::uniform(Compression::Gzip(9))),
        ] {
            let (what, compression) = policy;
            let scratch = Scratch::new("compressed-identity");
            // An input chunk shape that divides neither the volume nor the block
            // grid, so reads decompress chunks they keep a corner of, and the
            // written level's own chunks overhang the volume's far edge — which
            // makes those writes decompress-patch-recompress, the path a codec
            // is most likely to get wrong.
            let (storage, env) = through_storage_with(
                scratch.path(),
                &workflow,
                &decomposition,
                &source,
                [5, 7, 6],
                &hints,
                compression,
            );
            assert_same(&memory, &storage, &format!("{name} under {what}"));
            assert!(
                env.serialised_writes() > 0 && env.unaligned_reads() > 0,
                "{name} under {what}: nothing took the decompress-patch-recompress path, so this \
                 is not testing what it was written to test"
            );
        }
    }
}

/// The same output, from stores that are *not* the same on disk.
///
/// Byte identity of the answer would be a weak claim if the two runs had written
/// the same files — so this asserts they did not: the compressed store is
/// materially smaller, and the answer is identical anyway.
#[test]
fn a_compressed_store_and_a_raw_one_differ_on_disk_and_agree_in_the_answer() {
    let input = intensities();
    let source: Voxels = mask(&input, 0.35).into();
    let chain = Chain::op(NonZeroOp::new("non zero", [0, 0, 0]));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let mut decomposition = plan(&workflow, 8);
    decomposition.declare_dtypes(&workflow.chain).unwrap();
    assert_eq!(decomposition.dtype_at(1), Dtype::Bool);

    let scratch = Scratch::new("raw");
    let (raw, raw_env) = through_storage_with(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [8, 8, 8],
        &Hints::default(),
        CompressionPolicy::uniform(Compression::None),
    );
    let scratch = Scratch::new("gzipped");
    let (gzipped, gzip_env) = through_storage_with(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [8, 8, 8],
        &Hints::default(),
        CompressionPolicy::derived(),
    );

    assert_same(&raw, &gzipped, "raw store against compressed store");
    assert_eq!(raw_env.compression_at(1).unwrap(), Compression::None);
    assert_eq!(gzip_env.compression_at(1).unwrap(), Compression::Gzip(1));
    assert!(
        gzip_env.stored_bytes(1).unwrap() * 4 < raw_env.stored_bytes(1).unwrap(),
        "the bool level stored {} bytes compressed against {} raw",
        gzip_env.stored_bytes(1).unwrap(),
        raw_env.stored_bytes(1).unwrap()
    );
    // And the derived policy left the `float64` input alone, which is the half
    // of the default that saves CPU rather than bytes.
    assert_eq!(gzip_env.compression_at(0).unwrap(), Compression::None);
    assert_eq!(
        gzip_env.stored_bytes(0).unwrap(),
        raw_env.stored_bytes(0).unwrap()
    );
}

/// **The measurement.** Stored bytes and elapsed time, per level, under six
/// policies, over a `synthetic::Scene` volume with a `bool` level in it — and a
/// second, shorter table for a `uint16` level, which no chain here produces.
///
/// This is the evidence for `Compression::for_dtype`, and it is a test rather
/// than a note in a document so that it is re-run rather than remembered. It
/// prints the table — `cargo test --features zarr --release compression_pays --
/// --nocapture` — and asserts only the two directions the defaults depend on:
///
/// * the `bool` level compresses **a great deal**, so the default for `bool`
///   must be to compress it;
/// * the `float64` level compresses **hardly at all**, so the default for the
///   floats must be to leave them alone rather than pay deflate to discover
///   that.
///
/// The thresholds are loose on purpose. A tight ratio would be an assertion
/// about `synthetic::Scene`'s parameters, which are not what is being claimed;
/// what is being claimed is that the two element types are in different
/// regimes, and that is what the assertions say.
#[test]
fn compression_pays_for_bool_and_not_for_float() {
    // Larger than the rest of this file's volumes, so that gzip's 18-byte
    // header and trailer per chunk are a rounding error on the ratio rather
    // than a term in it.
    const BIG: [usize; 3] = [64, 64, 64];
    const CHUNK: [usize; 3] = [32, 32, 32];

    let scene = Scene::new(
        SceneSpec::new(BIG, 20250812)
            .with_objects(300)
            .with_radius(1.5, 5.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((BIG[0], BIG[1], BIG[2]));
    for i in 0..BIG[0] {
        for j in 0..BIG[1] {
            for k in 0..BIG[2] {
                array[[i, j, k]] = rendered.intensity[[i, j, k]];
            }
        }
    }
    let source: Voxels = array.into();

    // One phase, so two levels: level 0 is the `float64` scene as written, and
    // level 1 is the `bool` mask derived from it. Both are levels this framework
    // really produces, and they are the two ends of the compressibility range.
    //
    // The threshold in front of the `non zero` is load-bearing for the
    // measurement rather than for the chain: the scene carries noise, so *every*
    // voxel of it is non-zero, and the mask of that is uniformly `true` — which
    // is the `bool` fill value, which `zarrs` stores as no file at all. That is
    // a real and free win for masks, but it is not the win compression is being
    // asked about, and measuring it here would flatter the codec with somebody
    // else's work.
    let chain = Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::threshold("threshold", 0.35, 1.0, 0.0)),
        Chain::op(NonZeroOp::new("non zero", [0, 0, 0])),
    ]);
    let workflow = Workflow::new(chain, BIG, Dtype::F64);
    let reach = workflow.chain.reach3(&BIG);
    let grid = BlockGrid::along(BIG, &[0, 1, 2], 32).unwrap();
    let mut decomposition = Decomposition {
        volume: BIG,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            (0..workflow.chain.slots().len()).collect(),
            workflow
                .chain
                .slots()
                .iter()
                .map(|slot| slot.display_name())
                .collect(),
            reach,
            reach,
            grid,
        )],
        chain_reach: reach,
    };
    decomposition.declare_dtypes(&workflow.chain).unwrap();
    assert_eq!(decomposition.dtype_at(0), Dtype::F64);
    assert_eq!(decomposition.dtype_at(1), Dtype::Bool);

    let memory = through_memory(&workflow, &decomposition, &source, &Hints::default());

    // The last three are the question the *per level* design exists to ask: if
    // the `bool` level is where the ratio is, is it worth turning that one level
    // up while leaving the `float64` one alone? The table answers it.
    let policies = [
        (
            "bytes (no compression)",
            CompressionPolicy::uniform(Compression::None),
        ),
        (
            "gzip1 everywhere",
            CompressionPolicy::uniform(Compression::Gzip(1)),
        ),
        (
            "gzip9 everywhere",
            CompressionPolicy::uniform(Compression::Gzip(9)),
        ),
        ("derived (the default)", CompressionPolicy::derived()),
        (
            "derived, bool at gzip6",
            CompressionPolicy::derived().with_level(1, Compression::Gzip(6)),
        ),
        (
            "derived, bool at gzip9",
            CompressionPolicy::derived().with_level(1, Compression::Gzip(9)),
        ),
    ];
    let mut measured = Vec::new();
    for (what, policy) in policies {
        let scratch = Scratch::new("measure");
        let (storage, env, elapsed) = through_storage_timed(
            scratch.path(),
            &workflow,
            &decomposition,
            &source,
            CHUNK,
            &Hints::default(),
            policy,
        );
        // Every policy, against the in-memory oracle. A measurement of a wrong
        // answer is worth nothing.
        assert_same(&memory, &storage, what);
        measured.push((
            what,
            env.stored_bytes(0).unwrap(),
            env.stored_bytes(1).unwrap(),
            elapsed,
        ));
    }

    eprintln!(
        "\ncompression over a {BIG:?} synthetic::Scene, chunk {CHUNK:?}, \
         {} build\n  level 0 = float64 intensities ({} B raw), level 1 = bool mask ({} B raw)",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        BIG[0] * BIG[1] * BIG[2] * 8,
        BIG[0] * BIG[1] * BIG[2],
    );
    eprintln!(
        "{:<24} {:>12} {:>8} {:>12} {:>8} {:>10} {:>12}",
        "policy", "level0 B", "ratio", "level1 B", "ratio", "run", "break-even"
    );
    let (_, raw0, raw1, raw_time) = measured[0];
    for &(what, level0, level1, elapsed) in &measured {
        // The number that decides the trade, and the reason this prints a
        // throughput rather than a ratio: compression pays exactly when the
        // bytes it saved would have taken longer to move than the CPU it spent
        // saving them. Below this store speed, compress; above it, do not.
        let saved = (raw0 + raw1).saturating_sub(level0 + level1) as f64;
        let extra = elapsed.saturating_sub(raw_time).as_secs_f64();
        let break_even = if extra > 0.0 {
            format!("{:.1} MB/s", saved / extra / 1e6)
        } else {
            "-".to_string()
        };
        eprintln!(
            "{what:<24} {level0:>12} {:>7.2}x {level1:>12} {:>7.2}x {:>9.0?} {break_even:>12}",
            raw0 as f64 / level0 as f64,
            raw1 as f64 / level1 as f64,
            elapsed
        );
    }
    eprintln!();

    // ---- and the same question for an integer level, which needs no chain --
    //
    // The `uint16` case is the third row of `Compression::for_dtype`'s table and
    // it is not reachable from the chain above, so it is measured directly: a
    // twelve-bit quantisation of the same scene, written as level 0 and never
    // read. That is exactly the shape of a raw acquisition, which is the input
    // this framework is usually handed.
    let quantised = Voxels::from(
        source
            .view::<f64>()
            .unwrap()
            .mapv(|value| (value.clamp(0.0, 1.0) * 4095.0) as u16),
    );
    eprintln!("a uint16 quantisation of the same scene, written as level 0:");
    let mut integer = Vec::new();
    for compression in [
        Compression::None,
        Compression::Gzip(1),
        Compression::Gzip(6),
        Compression::Gzip(9),
    ] {
        let scratch = Scratch::new("measure-uint16");
        let started = std::time::Instant::now();
        let env = ZarrEnvironment::create_with_compression(
            scratch.path(),
            &quantised,
            CHUNK,
            CompressionPolicy::uniform(compression),
        )
        .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(env.level(0).unwrap(), quantised, "{}", compression.name());
        integer.push((compression, env.stored_bytes(0).unwrap(), elapsed));
    }
    let (_, raw16, _) = integer[0];
    for &(compression, bytes, elapsed) in &integer {
        eprintln!(
            "  {:<10} {bytes:>12} {:>7.2}x {:>9.0?}",
            compression.name(),
            raw16 as f64 / bytes as f64,
            elapsed
        );
    }
    eprintln!();
    assert!(
        integer[1].1 < raw16,
        "the uint16 level did not compress at all ({} B against {raw16} B); the integer default \
         assumes it does",
        integer[1].1
    );

    let gzip1 = measured[1];
    // The `bool` level: this is the case the defaults are built around.
    assert!(
        gzip1.2 * 5 < raw1,
        "the bool level compressed only {:.2}x ({} B against {} B); the default for bool is \
         premised on it compressing a great deal, and if it no longer does the default should \
         change rather than this threshold",
        raw1 as f64 / gzip1.2 as f64,
        gzip1.2,
        raw1
    );
    // The `float64` level: this is the case the defaults *decline*.
    assert!(
        gzip1.1 * 4 > raw0 * 3,
        "the float64 level compressed {:.2}x ({} B against {} B), which is more than the \
         `Compression::for_dtype` default assumes. If float data on this framework really does \
         compress, the default to leave it raw is wrong and should be revisited on this number \
         rather than defended by loosening this assertion.",
        raw0 as f64 / gzip1.1 as f64,
        gzip1.1,
        raw0
    );
}

/// A plan whose levels want different answers gets them, and the override is
/// heard over the derivation.
///
/// This is what "per level, not per run" buys, made mechanical: the `float64`
/// level is compressed because the caller asked, and the `bool` level is left
/// raw because the caller asked — both against what the derivation would have
/// done, so a policy that was quietly ignored would fail here.
#[test]
fn a_per_level_override_reaches_the_arrays_it_names() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0)),
        Chain::op(NonZeroOp::new("non zero", [0, 0, 0])),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let mut phases = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let reach = slots[index].reach3(&VOLUME);
        let grid = BlockGrid::along(VOLUME, &[0, 1, 2], 12).unwrap();
        phases.push(PhaseDecomposition::derive(
            vec![index],
            vec![name.clone()],
            reach,
            reach,
            grid,
        ));
    }
    let mut decomposition = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: workflow.chain.reach3(&VOLUME),
    };
    decomposition.declare_dtypes(&workflow.chain).unwrap();
    assert_eq!(decomposition.dtype_at(2), Dtype::Bool);

    // Both levels against the grain of the derivation.
    let policy = CompressionPolicy::derived()
        .with_level(0, Compression::Gzip(6))
        .with_level(2, Compression::None);
    // Every level pinned: this test inspects the *intermediates*, which are
    // otherwise erased the moment the phase that reads them finishes. That is
    // what `keep_levels` is for, and saying so here is cheaper than a test that
    // silently stopped checking level 1.
    let keep_all = Hints {
        keep_levels: (0..decomposition.n_levels()).collect(),
        ..Hints::default()
    };
    let memory = through_memory(&workflow, &decomposition, &source, &keep_all);
    let scratch = Scratch::new("mixed");
    let (storage, env) = through_storage_with(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [8, 8, 8],
        &keep_all,
        policy,
    );
    assert_same(&memory, &storage, "a mixed per-level policy");

    assert_eq!(env.compression_at(0).unwrap(), Compression::Gzip(6));
    // Level 1 was not named, so it kept the derivation for its own type.
    assert_eq!(env.level_dtype(1).unwrap(), Dtype::F64);
    assert_eq!(env.compression_at(1).unwrap(), Compression::None);
    assert_eq!(env.level_dtype(2).unwrap(), Dtype::Bool);
    assert_eq!(env.compression_at(2).unwrap(), Compression::None);
    // And the level that was told to compress really did.
    assert!(
        env.stored_bytes(2).unwrap() > 0,
        "the bool level was written and stored nothing"
    );
}

/// Compression does not change what the alignment counters mean.
///
/// Two runs with every level compressed: a plan aligned on both sides is zero on
/// both counters, and a plan whose blocks straddle the **input**'s chunk grid is
/// not. That matters because compression is precisely what makes an unaligned
/// read expensive — a decompress of a whole chunk to keep a face of it — so the
/// counter has to keep pointing at it.
///
/// The *write* half of this comparison moved to
/// `a_block_grid_that_straddles_a_dictated_chunk_grid_is_refused_and_names_the_chunk`:
/// a straddling write is no longer something to count, it is something to
/// refuse. What is left on the write side here is the far-edge overhang, which
/// `a_conforming_plan_serialises_no_write_and_over_reads_no_chunk` pins down.
#[test]
fn the_alignment_counters_still_mean_what_they_meant_under_compression() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let everything = CompressionPolicy::uniform(Compression::Gzip(1));

    let scratch = Scratch::new("aligned-gzip");
    let (aligned, aligned_env) = through_storage_with(
        scratch.path(),
        &workflow,
        &plan(&workflow, 4),
        &source,
        [4, 4, 4],
        &Hints::default(),
        everything.clone(),
    );
    assert_eq!(aligned_env.serialised_writes(), 0);
    assert_eq!(aligned_env.unaligned_reads(), 0);

    let scratch = Scratch::new("straddling-gzip");
    let (straddling, straddling_env) = through_storage_with(
        scratch.path(),
        &workflow,
        &plan(&workflow, 6),
        &source,
        [4, 4, 4],
        &Hints::default(),
        everything,
    );
    assert!(
        straddling_env.unaligned_reads() > 0,
        "six-voxel blocks reading a four-voxel chunk grid must decompress chunks they keep only \
         part of; that is legal, and it is what the counter is for"
    );

    assert_same(&aligned, &straddling, "aligned against straddling, gzipped");
}

// ----------------------------------------------- 2. under concurrency --

/// The same, with the executor writing blocks from a pool.
///
/// This used to be the configuration the partial-chunk race lived in — several
/// threads writing valid regions that tile the volume but not the chunk grid —
/// and it deliberately chose a coprime block edge and chunk shape so that every
/// chunk was shared. Under the chunk-exclusive invariant that arrangement is no
/// longer expressible for a level: the written level takes its chunking from the
/// block grid, so no two tasks can meet in a chunk however many threads run.
///
/// So what this measures now is the *other* half — that concurrency does not
/// change the answer — and the race itself is measured where it can still be
/// provoked: `zarr_env::tests::concurrent_partial_chunk_writes_lose_data_...`,
/// which writes two halves of one chunk directly and watches the loss.
#[test]
fn concurrent_execution_through_storage_is_still_byte_identical() {
    let input = intensities();
    let source: Voxels = input.clone().into();
    let chain = Chain::sequence(vec![
        Chain::op(RankFilterOp::median("median", box_element([1, 1, 1]))),
        Chain::op(AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(box_element([1, 2, 1]), [5, 4, 3], Statistic::Mean).unwrap(),
            1.0,
            0.01,
        )),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let decomposition = plan(&workflow, 7);
    let serial = through_memory(&workflow, &decomposition, &source, &Hints::default());

    for concurrency in [2, 4, 8] {
        let hints = Hints {
            concurrency,
            ..Hints::default()
        };
        let scratch = Scratch::new("concurrent");
        let (storage, env) = through_storage(
            scratch.path(),
            &workflow,
            &decomposition,
            &source,
            [5, 5, 5],
            &hints,
        );
        assert_same(&serial, &storage, &format!("at concurrency {concurrency}"));
        // And the plan really is chunk-exclusive, rather than merely lucky: the
        // levels these blocks wrote are chunked from the blocks themselves.
        for level in 1..decomposition.n_levels() {
            assert_eq!(
                env.chunk_at(level).unwrap(),
                chunk_for_block(decomposition.phases[level - 1].grid.block(), Dtype::F64),
                "level {level} at concurrency {concurrency}"
            );
        }
    }
}

// ------------------------------- 3. decomposition invariance, through storage --

/// The property the framework rests on does not weaken for having been written
/// to a disk: the same chain over the same volume gives one answer whatever the
/// block grid, and now, whatever the chunk grid too.
#[test]
fn decomposition_invariance_survives_storage() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::op(RankFilterOp::median("median", box_element([1, 1, 1])));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    let mut reference: Option<Voxels> = None;
    for (block, chunk) in [
        (32, [32, 24, 20]),
        (16, [8, 8, 8]),
        (12, [8, 8, 8]),
        (10, [4, 4, 4]),
        (8, [5, 7, 6]),
        (6, [16, 16, 16]),
    ] {
        let decomposition = plan(&workflow, block);
        let scratch = Scratch::new("invariance");
        let (storage, _) = through_storage(
            scratch.path(),
            &workflow,
            &decomposition,
            &source,
            chunk,
            &Hints::default(),
        );
        match &reference {
            None => reference = Some(storage),
            Some(first) => assert_same(
                first,
                &storage,
                &format!("block {block}, chunk {chunk:?} against the first decomposition"),
            ),
        }
    }
}

// ------------------------------- 4. chunk-exclusive writing is a mandate --

/// **A block grid that straddles a dictated chunk grid is refused**, and the
/// refusal names the phase and the chunk two blocks would have shared.
///
/// This test asserted the opposite until the chunk-exclusive invariant landed:
/// it ran the straddling plan, checked the answer was identical, and reported
/// the cost in `serialised_writes`. That was true and is still true of the
/// *answer* — a partial-chunk write is correct, one thread at a time — but it
/// was permitting a hazard rather than removing it. Two tasks writing one chunk
/// lose each other's bytes under `zarrs` 0.23.13, and a chunk with several
/// owners has no lifetime a cache tier can hold it by. So the plan is now
/// refused, and what used to be a counter to watch is a property to rely on.
///
/// What the old test checked independently is checked here still, in the last
/// arm: the answer does not depend on the block grid. Six-voxel blocks remain
/// perfectly legal — they are only illegal against a chunk grid somebody else
/// fixed, and with nothing dictated the level derives its chunking from them.
#[test]
fn a_block_grid_that_straddles_a_dictated_chunk_grid_is_refused_and_names_the_chunk() {
    let input = intensities();
    let source: Voxels = input.into();
    // Reach 0, so the valid regions are the cores and the block edge is exactly
    // what lands on the chunk grid or does not.
    let chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    // 32, 24 and 20 are all multiples of 4, so a 4-voxel block on a 4-voxel
    // chunk grid covers every chunk it touches from edge to edge — including the
    // last one on each axis, which is the case that is easy to get wrong.
    let aligned_plan = plan(&workflow, 4);
    let scratch = Scratch::new("aligned");
    let aligned_env = ZarrEnvironment::create(scratch.path(), &source, [4, 4, 4])
        .unwrap()
        .with_output_chunk([4, 4, 4]);
    execute(
        "storage",
        &workflow,
        &aligned_plan,
        &Hints::default(),
        &aligned_env,
    )
    .unwrap();
    let aligned = aligned_env.output().unwrap();
    assert_eq!(
        aligned_env.serialised_writes(),
        0,
        "a 4-voxel block on a 4-voxel chunk grid should never read-modify-write"
    );
    assert_eq!(
        aligned_env.unaligned_reads(),
        0,
        "and should never decode a chunk to throw most of it away"
    );

    // Six-voxel blocks against the same dictated four-voxel chunks: block
    // [0,0,0] writes 0..6 of axis 0 and block [1,0,0] writes 6..12, so chunk
    // [1,0,0] — voxels 4..8 — has two writers.
    let straddling_plan = plan(&workflow, 6);
    let scratch = Scratch::new("straddling-dictated");
    let dictated = ZarrEnvironment::create(scratch.path(), &source, [4, 4, 4])
        .unwrap()
        .with_output_chunk([4, 4, 4]);
    let err = dictated.prepare(&straddling_plan).unwrap_err().to_string();
    assert!(err.contains("phase 0"), "got: {err}");
    assert!(err.contains("chunk [1, 0, 0]"), "got: {err}");
    assert!(err.contains("[4, 0, 0]..[8, 4, 4]"), "got: {err}");
    assert!(
        err.contains("block [0, 0, 0]") && err.contains("block [1, 0, 0]"),
        "got: {err}"
    );
    assert!(err.contains("exactly one task"), "got: {err}");
    // And it names the constraint the caller chose, because that is the one
    // they can drop — together with the one they may not be able to.
    assert!(err.contains("with_output_chunk"), "got: {err}");
    assert!(err.contains("BlockConstraint::Extent"), "got: {err}");
    // The refusal is not something `execute` can walk past: `prepare` is the
    // first thing it does.
    assert!(execute(
        "storage",
        &workflow,
        &straddling_plan,
        &Hints::default(),
        &dictated
    )
    .is_err());

    // The same block grid with nothing dictated: legal, chunked from the blocks,
    // and the same answer to the voxel.
    let scratch = Scratch::new("straddling-derived");
    let (derived, derived_env) = through_storage(
        scratch.path(),
        &workflow,
        &straddling_plan,
        &source,
        [4, 4, 4],
        &Hints::default(),
    );
    assert_eq!(
        derived_env.chunk_at(1).unwrap(),
        [6, 6, 6],
        "the level a phase writes takes its chunking from that phase's blocks"
    );
    // Level 0 keeps the layout it arrived with, and reads of it straddle — which
    // is legal, is what a halo does, and is still counted.
    assert_eq!(derived_env.chunk_at(0).unwrap(), [4, 4, 4]);
    assert_same(&aligned, &derived, "four-voxel blocks against six-voxel");
}

/// **`serialised_writes` is zero for a conforming plan**, which is the property
/// the counter was only ever a hint at.
///
/// Under the invariant no two tasks share a chunk, so the read-modify-write path
/// is unreachable for a level write — with one exception that is worth stating
/// rather than hiding: `zarrs` compares a subset against the **unclipped** chunk
/// extent, so a chunk that overhangs the volume's far edge takes the slow path
/// even though it holds no voxel anybody else can write. The block edges here
/// divide all three axes of the volume, so there is no overhang and the counter
/// is exactly zero; at a block edge that does not divide the volume it is the
/// count of edge-touching blocks, and no hazard.
#[test]
fn a_conforming_plan_serialises_no_write_and_over_reads_no_chunk() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    // gcd(32, 24, 20) is 4, so these are the block edges that divide the volume
    // on every axis.
    for block in [2usize, 4] {
        let decomposition = plan(&workflow, block);
        let scratch = Scratch::new("conforming");
        let (_, env) = through_storage(
            scratch.path(),
            &workflow,
            &decomposition,
            &source,
            [block, block, block],
            &Hints::default(),
        );
        assert_eq!(
            env.serialised_writes(),
            0,
            "block {block}: a chunk-exclusive plan over a volume the chunk divides must never \
             read-modify-write"
        );
        assert_eq!(env.unaligned_reads(), 0, "block {block}");
    }
}

// ------------------------------------------------ levels are per-phase --

/// A phase that changes the element type gets a level of that type on disk, and
/// the run still agrees with the one in memory.
///
/// This is what `prepare` is for: level `p+1` is created at phase `p`'s
/// `volume_at` and `dtype_at`, not at the workflow's. A storage environment that
/// created every level at the input's type would write a `float64` array where
/// the plan says `bool` — an eight-fold overcount that every reader downstream
/// would be right to believe.
#[test]
fn a_phase_that_changes_element_type_gets_a_level_of_that_type_on_disk() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::sequence(vec![
        Chain::op(RankFilterOp::median("median", box_element([1, 1, 1]))),
        Chain::op(NonZeroOp::new("non zero", [0, 0, 0])),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    // Two phases, so the intermediate is materialised: phase 0 writes `float64`,
    // phase 1 writes `bool`.
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let mut phases = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let reach = slots[index].reach3(&VOLUME);
        let grid = BlockGrid::along(VOLUME, &[0, 1, 2], 12).unwrap();
        phases.push(PhaseDecomposition::derive(
            vec![index],
            vec![name.clone()],
            reach,
            reach,
            grid,
        ));
    }
    let mut decomposition = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: workflow.chain.reach3(&VOLUME),
    };
    decomposition.declare_dtypes(&workflow.chain).unwrap();
    assert_eq!(decomposition.dtype_at(1), Dtype::F64);
    assert_eq!(decomposition.dtype_at(2), Dtype::Bool);

    // Pinned for the same reason as the per-level policy test: what is being
    // checked here is the width of an intermediate on disk.
    let keep_all = Hints {
        keep_levels: (0..decomposition.n_levels()).collect(),
        ..Hints::default()
    };
    let memory = through_memory(&workflow, &decomposition, &source, &keep_all);
    let scratch = Scratch::new("dtype-change");
    let (storage, env) = through_storage(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [8, 8, 8],
        &keep_all,
    );
    assert_eq!(env.level_dtype(0).unwrap(), Dtype::F64);
    assert_eq!(env.level_dtype(1).unwrap(), Dtype::F64);
    assert_eq!(env.level_dtype(2).unwrap(), Dtype::Bool);
    assert_same(&memory, &storage, "a phase that changes element type");

    // The `bool` level really is one byte a voxel on disk, which is the whole
    // reason the element type became a level's own business.
    assert_eq!(env.level(2).unwrap().dtype(), Dtype::Bool);
}

// --------------------------------------------------------- side outputs --

/// The arrays an op writes beside its result go to storage too, and hold what
/// the in-memory environment holds.
#[test]
fn side_outputs_through_storage_hold_what_the_in_memory_environment_holds() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::op(SideOutputOp::new("side", [0, 0, 0]).with_side("sums", Dtype::F64, 0, 1));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let decomposition = plan(&workflow, 8);

    let memory =
        ArrayEnvironment::for_decomposition(source.clone(), &decomposition, [8, 8, 8]).unwrap();
    execute(
        "memory",
        &workflow,
        &decomposition,
        &Hints::default(),
        &memory,
    )
    .unwrap();

    let scratch = Scratch::new("side");
    let storage = ZarrEnvironment::create(scratch.path(), &source, [8, 8, 8]).unwrap();
    execute(
        "storage",
        &workflow,
        &decomposition,
        &Hints::default(),
        &storage,
    )
    .unwrap();

    assert_eq!(memory.side_output_names(), storage.side_output_names());
    assert!(
        !storage.side_output_names().is_empty(),
        "this op declares a side output; if it stopped, the test proves nothing"
    );
    for name in storage.side_output_names() {
        let want = memory.side_output(&name).unwrap();
        let got = storage.side_output(&name).unwrap().unwrap();
        assert_eq!(want.shape(), got.shape(), "side output {name:?}: shape");
        for (index, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            let agree = a == b || (a.is_nan() && b.is_nan());
            assert!(agree, "side output {name:?} element {index}: {a} vs {b}");
        }
    }
    assert_eq!(
        memory.counters().side_snapshot(),
        storage.counters().side_snapshot()
    );
}

// ------------------------------------------------------------- the seams --

/// The whole-volume kernels, called once, are still the final oracle: a run
/// through storage agrees with a run that never blocked the volume at all.
///
/// `ArrayEnvironment` is the day-to-day comparison above because it is in
/// process and stage-by-stage. It is only as good as its own agreement with the
/// kernels, so this closes that loop rather than assuming it.
#[test]
fn a_run_through_storage_agrees_with_the_whole_volume_kernels() {
    let input = intensities();
    let source: Voxels = input.into();
    let chain = Chain::op(RankFilterOp::median("median", box_element([1, 1, 1])));
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    let mut reference = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    workflow
        .chain
        .apply(&source, &mut reference, &Anchor::whole(VOLUME))
        .unwrap();

    let decomposition = plan(&workflow, 9);
    let scratch = Scratch::new("oracle");
    let (storage, _) = through_storage(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [7, 7, 7],
        &Hints::default(),
    );
    assert_same(&reference, &storage, "whole-volume kernels against storage");
}

/// A read outside the level is refused rather than clamped, and says which axis.
#[test]
fn a_region_outside_a_level_is_refused_by_name() {
    let scratch = Scratch::new("bounds");
    let source = Voxels::zeros(Dtype::U8, [4, 4, 4]).unwrap();
    let env = ZarrEnvironment::create(scratch.path(), &source, [2, 2, 2]).unwrap();
    let err = env
        .read(0, &Region::new(&[3, 0, 0], &[2, 4, 4]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("axis 0"), "got: {err}");
}

/// The fragments an op writes beside its blocks survive on the same store, so a
/// strategy that produces them is runnable here and not only in memory.
#[test]
fn sidecar_fragments_round_trip_through_the_store() {
    use blockflow::sidecar::Lifecycle;

    let scratch = Scratch::new("sidecars");
    let source = Voxels::zeros(Dtype::U8, [4, 4, 4]).unwrap();
    let env = ZarrEnvironment::create(scratch.path(), &source, [2, 2, 2]).unwrap();
    env.declare_sidecar("counts", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("counts", 0, [1, 2, 3], b"seventeen")
        .unwrap();
    assert_eq!(
        env.read_sidecar("counts", 0, [1, 2, 3]).unwrap().as_deref(),
        Some(&b"seventeen"[..])
    );
    assert_eq!(env.sidecar_keys("counts").unwrap().len(), 1);
}

/// Declaring the same side output twice with a different shape is refused, on
/// the same argument `ArrayEnvironment` refuses it: a plan whose outputs depend
/// on which op ran last is the class of silent wrongness this crate removes.
#[test]
fn two_disagreeing_side_output_declarations_are_refused() {
    let scratch = Scratch::new("declare-twice");
    let source = Voxels::zeros(Dtype::F64, [4, 4, 4]).unwrap();
    let env = ZarrEnvironment::create(scratch.path(), &source, [2, 2, 2]).unwrap();
    let first = Output::new("table", Dtype::F64, &[4, 2]);
    env.declare_side_output(&first).unwrap();
    env.declare_side_output(&first).unwrap();
    let second = Output::new("table", Dtype::F64, &[4, 3]);
    let err = env.declare_side_output(&second).unwrap_err().to_string();
    assert!(err.contains("table"), "got: {err}");
}

/// An environment given a plan it cannot host says so, naming the level.
#[test]
fn a_plan_that_disagrees_with_level_zero_is_refused_by_prepare() {
    let scratch = Scratch::new("prepare");
    let source = Voxels::zeros(Dtype::F64, [4, 4, 4]).unwrap();
    let env = ZarrEnvironment::create(scratch.path(), &source, [2, 2, 2]).unwrap();
    let chain: Chain = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.5, 1.0, 0.0));
    let workflow = Workflow::new(chain, [4, 4, 4], Dtype::U16);
    let slots = workflow.chain.slots();
    let grid = BlockGrid::along([4, 4, 4], &[0], 2).unwrap();
    let decomposition = Decomposition {
        volume: [4, 4, 4],
        dtype: Dtype::U16,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            vec!["threshold".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid,
        )],
        chain_reach: [0, 0, 0],
    };
    let err = env.prepare(&decomposition).unwrap_err().to_string();
    assert!(
        err.contains("float64") && err.contains("uint16"),
        "got: {err}"
    );
}

/// A side output written before it was declared is refused rather than creating
/// an array nobody planned.
#[test]
fn a_side_output_written_before_it_was_declared_is_refused() {
    use blockflow::voxels::SideBuf;

    let scratch = Scratch::new("undeclared");
    let source = Voxels::zeros(Dtype::F64, [4, 4, 4]).unwrap();
    let env = ZarrEnvironment::create(scratch.path(), &source, [2, 2, 2]).unwrap();
    let output = Output::new("nowhere", Dtype::F64, &[2, 2]);
    let region = Region::whole(&[2, 2]);
    let err = env
        .write_side(&output, 0, &region, &SideBuf::zeros(&region))
        .unwrap_err()
        .to_string();
    assert!(err.contains("declared"), "got: {err}");
}

// ------------------------------------------------------- level lifetime --

/// An intermediate level is **erased from the store** once the phase that reads
/// it has finished, and the answer is unchanged.
///
/// This is where the level-lifetime work is measured in the units that matter.
/// In memory it is an allocation; here it is a directory of chunks on a disk
/// that, at the scale this crate exists for, has no room for `N` copies of it.
///
/// Three things asserted, and the third is what makes the first two safe:
/// the directory is gone, the output is byte-identical to a run that kept
/// everything, and reading the freed level fails with a message about the plan.
#[test]
fn an_intermediate_level_is_erased_from_the_store_and_the_answer_is_unchanged() {
    let input = intensities();
    let source: Voxels = input.clone().into();
    let chain = Chain::sequence(vec![
        Chain::op(RankFilterOp::median("median", box_element([1, 1, 1]))),
        Chain::op(MorphologyOp::new(
            "dilate",
            Morphology::Dilate,
            box_element([1, 1, 1]),
        )),
        Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0)),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    // one phase per slot, so every stage really materialises a level
    let slots = workflow.chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 16).unwrap();
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: (0..slots.len())
            .map(|slot| {
                PhaseDecomposition::derive(
                    vec![slot],
                    vec![slots[slot].display_name()],
                    slots[slot].reach3(&VOLUME),
                    slots[slot].reach3(&VOLUME),
                    grid.clone(),
                )
            })
            .collect(),
        chain_reach: workflow.chain.reach3(&VOLUME),
    };

    let kept_root = Scratch::new("level-lifetime-kept");
    let keep_all = Hints {
        keep_levels: (0..plan.n_levels()).collect(),
        ..Hints::default()
    };
    let (kept, kept_env) = through_storage(
        kept_root.path(),
        &workflow,
        &plan,
        &source,
        [8, 8, 8],
        &keep_all,
    );

    let freed_root = Scratch::new("level-lifetime-freed");
    let (freed, freed_env) = through_storage(
        freed_root.path(),
        &workflow,
        &plan,
        &source,
        [8, 8, 8],
        &Hints::default(),
    );

    assert_same(
        &freed,
        &kept,
        "freeing the intermediates changed the answer",
    );

    // the directories: present in the run that kept them, gone in the other
    for level in 1..plan.n_levels() - 1 {
        assert!(
            kept_root.path().join(format!("level{level}")).exists(),
            "level {level} should still be on disk when it was pinned"
        );
        assert!(
            !freed_root.path().join(format!("level{level}")).exists(),
            "level {level} should have been erased"
        );
        assert!(freed_env.is_discarded(level));
        assert!(!kept_env.is_discarded(level));
    }

    // level 0 is somebody else's array and the output is what the run is for
    assert!(freed_root.path().join("level0").exists());
    assert!(freed_root
        .path()
        .join(format!("level{}", plan.n_levels() - 1))
        .exists());
    assert!(!freed_env.is_discarded(0));
    assert!(!freed_env.is_discarded(plan.n_levels() - 1));

    // and the freed level is loud rather than empty
    let message = freed_env
        .read(1, &blockflow::region::Region::new(&[0, 0, 0], &[4, 4, 4]))
        .unwrap_err()
        .to_string();
    assert!(message.contains("discarded"), "{message}");
    assert!(message.contains("keep_levels"), "{message}");
}
