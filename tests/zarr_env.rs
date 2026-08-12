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
// 4. **Alignment is a performance fact, not a correctness one.** A block grid
//    that straddles the chunk grid gives the same answer and says so in the
//    counter.

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
use blockflow::zarr_env::ZarrEnvironment;
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

/// The same run, through Zarr arrays on a disk.
fn through_storage(
    root: &PathBuf,
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    chunk: [usize; 3],
    hints: &Hints,
) -> (Voxels, ZarrEnvironment) {
    let env = ZarrEnvironment::create(root, input, chunk).unwrap();
    execute("storage", workflow, decomposition, hints, &env).unwrap();
    let output = env.output().unwrap();
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

// ----------------------------------------------- 2. under concurrency --

/// The same, with the executor writing blocks from a pool.
///
/// This is the configuration the partial-chunk race lives in: several threads
/// writing valid regions that tile the volume but do not tile the chunk grid.
/// A guard that only works when nothing is contending is not a guard, so the
/// block edge and the chunk shape here are deliberately coprime — every write
/// straddles chunks and every chunk is shared between blocks.
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
        assert!(
            env.serialised_writes() > 0,
            "at concurrency {concurrency} no write took the guarded path, so this test is not \
             exercising the race it was written for"
        );
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

// --------------------------- 4. alignment is performance, not correctness --

/// A block grid that lands on the chunk grid takes no locks; one that straddles
/// it takes them, is slower, and gives the identical answer.
///
/// This is the crate's founding principle made mechanical for the storage
/// layer: *a mistake in which chunks we fetch costs performance, never
/// correctness.*
#[test]
fn a_block_grid_that_straddles_the_chunk_grid_costs_locks_and_not_the_answer() {
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
    let (aligned, aligned_env) = through_storage(
        scratch.path(),
        &workflow,
        &aligned_plan,
        &source,
        [4, 4, 4],
        &Hints::default(),
    );
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

    let straddling_plan = plan(&workflow, 6);
    let scratch = Scratch::new("straddling");
    let (straddling, straddling_env) = through_storage(
        scratch.path(),
        &workflow,
        &straddling_plan,
        &source,
        [4, 4, 4],
        &Hints::default(),
    );
    assert!(straddling_env.serialised_writes() > 0);
    assert!(straddling_env.unaligned_reads() > 0);

    assert_same(&aligned, &straddling, "aligned against straddling");
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

    let memory = through_memory(&workflow, &decomposition, &source, &Hints::default());
    let scratch = Scratch::new("dtype-change");
    let (storage, env) = through_storage(
        scratch.path(),
        &workflow,
        &decomposition,
        &source,
        [8, 8, 8],
        &Hints::default(),
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
