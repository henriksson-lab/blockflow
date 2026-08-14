// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A diamond, executed as a diamond.**
//
// `docs/design/BLOCK_OPS.md` records the failure this file exists to make
// impossible: a real fan-in — one input, two arms, one sink — was modelled as
// `Chain::Alternative`, and **903 comparisons passed**. They passed because
// `Alternative` folds reach as the max across branches, and the max is the
// correct budget whether one branch runs or all of them. No reach test can tell
// the two readings apart.
//
// What tells them apart is what *executes* and what is *produced*, so that is
// what is asserted here, four ways, each of which the reach comparisons could
// not have caught:
//
// 1. **Both arms ran.** Counted, per arm, through an op that records its own
//    invocations — and equal to the block count, so it is not "at least once
//    somewhere" but "once per block on each arm".
// 2. **Both arms contributed.** The combined result differs from either arm
//    computed alone. This is the value-level statement of (1), and it is the
//    one that fails if a future change quietly re-selects a single branch.
// 3. **The same chain written as an `Alternative` gives a different answer** —
//    the exact bug, reproduced beside the fix, so the distinction is a
//    measurement in this file rather than a claim in a comment.
// 4. **The answer is decomposition-invariant**, against a whole-volume
//    reference computed by calling the same kernels once, across four block
//    sizes and three split-axis sets, in memory and through Zarr storage.
//
// Plus the guards, seen firing: a halo short of the fan-in's own reach, and a
// combine handed an element type it cannot join.
//
// The data is `synthetic::Scene`, whose region generation is byte-identical to
// the corresponding cut of a whole generation — which is what makes it valid
// data for a decomposition test rather than merely convenient.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    AdaptiveThresholdOp, ElementShape, LocalStatistic, Logic, LogicCombine, Morphology,
    MorphologyOp, RankFilterOp, Statistic, StructuringElement, VoxelwiseMapOp,
};
use blockflow::probes::SideOutputOp;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Result};

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250811)
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

fn box_element(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, radius)
}

/// An identity that counts how many times it was applied.
///
/// The whole content of the bug this file is about is *which subtrees run*, and
/// a value test can only see a branch that both ran and changed something. This
/// sees the run itself. It is an identity so that inserting one at the head of
/// an arm changes nothing about the answer — the counter is the only
/// observation it makes.
struct CountingOp {
    name: &'static str,
    calls: Arc<AtomicUsize>,
}

impl CountingOp {
    fn new(name: &'static str) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                name,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl BlockOp for CountingOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, so that inserting one does not change any plan's halo.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        out.assign(input)
    }
}

// --------------------------------------------------------------- the arms --

/// Arm A: a global threshold. Reach 0, so its answer is exact everywhere.
fn arm_a(counter: Option<CountingOp>) -> Chain {
    let threshold = Chain::op(VoxelwiseMapOp::threshold("threshold", 0.30, 1.0, 0.0));
    match counter {
        None => threshold,
        Some(count) => Chain::sequence(vec![Chain::op(count), threshold]),
    }
}

/// Arm B: a locally-anchored threshold, then an opening. Reach is **not** zero,
/// so the fan-in's halo comes from this arm and a short one is visible.
fn arm_b(counter: Option<CountingOp>) -> Chain {
    let inner = Chain::sequence(vec![
        Chain::op(AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(box_element([1, 1, 1]), [4, 4, 4], Statistic::Mean).unwrap(),
            1.0,
            0.0,
        )),
        Chain::op(MorphologyOp::new(
            "open",
            Morphology::Open,
            box_element([1, 1, 1]),
        )),
    ]);
    match counter {
        None => inner,
        Some(count) => Chain::sequence(vec![Chain::op(count), inner]),
    }
}

/// The stem both arms consume: a median filter.
fn stem() -> Chain {
    Chain::op(RankFilterOp::median("median", box_element([1, 1, 1])))
}

/// The chain this whole file is about: `median > (A & B) > or`.
fn diamond(count_a: Option<CountingOp>, count_b: Option<CountingOp>) -> Chain {
    Chain::sequence(vec![
        stem(),
        Chain::parallel(
            vec![arm_a(count_a), arm_b(count_b)],
            Box::new(LogicCombine::new("or", Logic::Or)),
        )
        .unwrap(),
    ])
}

/// The same shape written the way the design document says it was written
/// before: an exclusive choice. Structurally accepted, numerically wrong.
fn the_bug(taken: usize) -> Chain {
    Chain::sequence(vec![
        stem(),
        Chain::alternative(vec![arm_a(None), arm_b(None)], taken).unwrap(),
    ])
}

// ------------------------------------------------------------- machinery --

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the whole chain, built from the chain's **own** reach —
/// nothing here supplies one, so nothing here can hide one that is wrong.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    plan_with_reach(workflow, block, split_axes, workflow.chain.reach3(&VOLUME))
}

fn plan_with_reach(
    workflow: &Workflow,
    block: usize,
    split_axes: &[usize],
    reach: [usize; 3],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

/// The oracle: the same kernels, called once, over the whole array.
fn reference(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn run(workflow: &Workflow, decomposition: &Decomposition, input: &Array3<f64>) -> Array3<f64> {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    execute("fan-in", workflow, decomposition, &Hints::default(), &env).unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

fn set_voxels(array: &Array3<f64>) -> usize {
    array.iter().filter(|&&value| value != 0.0).count()
}

// ------------------------------------------------- both arms, demonstrably --

/// The assertion the 903 passing comparisons could not make.
#[test]
fn both_arms_of_a_diamond_run_once_per_block_and_both_reach_the_sink() {
    let input = intensities();

    // (2) both arms contribute: the OR is strictly larger than either arm.
    let only_a = reference(&Chain::sequence(vec![stem(), arm_a(None)]), &input);
    let only_b = reference(&Chain::sequence(vec![stem(), arm_b(None)]), &input);
    let both = reference(&diamond(None, None), &input);
    assert!(
        set_voxels(&only_a) > 0 && set_voxels(&only_b) > 0,
        "an arm that sets nothing cannot demonstrate anything: a={} b={}",
        set_voxels(&only_a),
        set_voxels(&only_b)
    );
    assert_ne!(both, only_a, "the combined result equals arm A alone");
    assert_ne!(both, only_b, "the combined result equals arm B alone");
    assert!(
        set_voxels(&both) > set_voxels(&only_a).max(set_voxels(&only_b)),
        "an OR of two arms must set more than either: a={} b={} both={}",
        set_voxels(&only_a),
        set_voxels(&only_b),
        set_voxels(&both)
    );

    // (3) the bug, reproduced: written as an alternation, whichever branch is
    // live is the whole answer and the other arm is dropped.
    assert_eq!(reference(&the_bug(0), &input), only_a);
    assert_eq!(reference(&the_bug(1), &input), only_b);
    assert_ne!(reference(&the_bug(0), &input), both);
    assert_ne!(reference(&the_bug(1), &input), both);

    // (1) both arms ran, once per block, through the executor.
    let (count_a, calls_a) = CountingOp::new("count_a");
    let (count_b, calls_b) = CountingOp::new("count_b");
    let workflow = workflow(diamond(Some(count_a), Some(count_b)));
    let decomposition = plan(&workflow, 8, &[0, 1, 2]);
    let blocks = decomposition.n_tasks();
    assert!(blocks > 1, "one block cannot show a per-block count");
    assert_eq!(run(&workflow, &decomposition, &input), both);
    assert_eq!(
        (
            calls_a.load(Ordering::SeqCst),
            calls_b.load(Ordering::SeqCst)
        ),
        (blocks, blocks),
        "each arm must be applied exactly once per block; this is the \
         observation that distinguishes a fan-in from an alternation"
    );
}

// -------------------------------------------------- decomposition invariance --

/// The standing bar: one answer, twelve decompositions, against the whole-volume
/// reference.
#[test]
fn a_diamond_is_decomposition_invariant_in_memory() {
    let input = intensities();
    let want = reference(&diamond(None, None), &input);
    assert!(
        want.iter().any(|&value| value == 1.0) && want.iter().any(|&value| value == 0.0),
        "the diamond produced a constant, so it is not discriminating anything"
    );

    for block in [4usize, 7, 13, 64] {
        for split_axes in [vec![0], vec![1, 2], vec![0, 1, 2]] {
            let workflow = workflow(diamond(None, None));
            let decomposition = plan(&workflow, block, &split_axes);
            decomposition.check().unwrap();
            assert_eq!(
                run(&workflow, &decomposition, &input),
                want,
                "block {block}, axes {split_axes:?}"
            );
        }
    }
}

/// The same claim through the storage backend that landed today.
///
/// A fan-in exercises it in a way a linear chain does not: the branch results
/// never touch the environment, so what reaches storage is the combine's answer
/// and only that. If the branch buffers were ever routed through a level — the
/// arrangement this node deliberately avoids — this is where the extra
/// materialisation would appear.
#[cfg(feature = "zarr")]
#[test]
fn a_diamond_through_storage_is_byte_identical_to_the_same_diamond_in_memory() {
    use blockflow::zarr_env::ZarrEnvironment;

    let input = intensities();
    let want = reference(&diamond(None, None), &input);
    let source: Voxels = input.clone().into();

    let root = std::env::temp_dir().join(format!(
        "blockflow-fan-in-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();

    for (index, block) in [4usize, 7, 13].into_iter().enumerate() {
        for chunk in [[8usize, 8, 8], [5, 5, 5]] {
            let workflow = workflow(diamond(None, None));
            let decomposition = plan(&workflow, block, &[0, 1, 2]);
            let store = root.join(format!("b{index}-c{}", chunk[0]));
            let env = ZarrEnvironment::create(&store, &source, chunk).unwrap();
            execute(
                "storage",
                &workflow,
                &decomposition,
                &Hints::default(),
                &env,
            )
            .unwrap();
            let produced = env.output().unwrap();
            assert_eq!(
                produced.view::<f64>().unwrap().to_owned(),
                want,
                "block {block}, chunk {chunk:?}"
            );
            drop(env);
            std::fs::remove_dir_all(&store).ok();
        }
    }
    std::fs::remove_dir_all(&root).ok();
}

// ----------------------------------------------- side outputs, from all arms --

/// Both arms' side outputs are declared, allocated and filled.
///
/// This is the fold `docs/design/BLOCK_OPS.md` §"Step 2" warns about from the
/// other direction: over-declaring an output is not safe the way over-declaring
/// reach is, so `Alternative` declares only the live branch's. A `Parallel` must
/// declare **every** branch's, because every branch writes — and an undeclared
/// one is not a wasted array, it is a hole the coverage guard reports.
#[test]
fn a_diamond_writes_the_side_outputs_of_every_arm() {
    let input = intensities();

    /// The diamond again, with an identity at the head of each arm that writes
    /// one array beside its result. The primary is unaffected, so the answer is
    /// still the plain diamond's.
    fn side_diamond() -> Chain {
        Chain::sequence(vec![
            stem(),
            Chain::parallel(
                vec![
                    Chain::sequence(vec![
                        Chain::op(SideOutputOp::new("a", [0, 0, 0]).with_side(
                            "m",
                            Dtype::F32,
                            0,
                            1,
                        )),
                        arm_a(None),
                    ]),
                    Chain::sequence(vec![
                        Chain::op(SideOutputOp::new("b", [0, 0, 0]).with_side(
                            "m",
                            Dtype::F32,
                            0,
                            1,
                        )),
                        arm_b(None),
                    ]),
                ],
                Box::new(LogicCombine::new("or", Logic::Or)),
            )
            .unwrap(),
        ])
    }

    assert_eq!(
        workflow(side_diamond())
            .outputs()
            .iter()
            .map(|output| output.name.clone())
            .collect::<Vec<_>>(),
        vec!["output".to_string(), "a.m".to_string(), "b.m".to_string()],
        "both arms of a fan-in declare, because both write"
    );

    let want = reference(&diamond(None, None), &input);
    for block in [7usize, 13] {
        let workflow = workflow(side_diamond());
        let decomposition = plan(&workflow, block, &[0, 1, 2]);
        decomposition.check().unwrap();
        let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
        execute("sides", &workflow, &decomposition, &Hints::default(), &env).unwrap();

        // The primary is unchanged by the identities that carry the side
        // outputs, so it is still the diamond's answer.
        assert_eq!(
            env.output().view::<f64>().unwrap().to_owned(),
            want,
            "block {block}"
        );
        assert_eq!(env.side_output_names(), ["a.m", "b.m"]);
        // Both arrays exist and neither is a hole: the coverage guard inside
        // `execute` has already refused anything that did not tile, so reading
        // them back is the confirmation that they were tiled by *both* arms.
        for name in ["a.m", "b.m"] {
            let side = env.side_output(name).unwrap();
            assert_eq!(side.shape(), &[VOLUME[0], VOLUME[1], VOLUME[2]]);
        }
    }
}

// ------------------------------------------------- the guards, seen firing --

/// A halo short of the fan-in's own reach must be refused.
///
/// The reach in question comes from the **widest arm** — arm A reaches zero —
/// so this is specifically the `Parallel` fold being load-bearing: a node that
/// folded reach as its first branch's, or as the combine's alone, would grant a
/// halo of zero here and produce a well-formed wrong volume.
#[test]
fn a_halo_short_of_the_fan_ins_reach_is_refused() {
    let workflow = workflow(diamond(None, None));
    let reach = workflow.chain.reach3(&VOLUME);
    assert!(
        reach.iter().any(|&value| value > 0),
        "a fan-in whose arms all reach zero cannot exercise the halo guard"
    );

    // The plan states the true reach and grants one voxel less halo than it
    // needs, which is what makes the guard able to fire at all: a halo that fed
    // the reach would be compared against itself.
    let short = [reach[0] - 1, reach[1], reach[2]];
    let broken = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..workflow.chain.slots().len()).collect(),
            workflow
                .chain
                .slots()
                .iter()
                .map(|slot| slot.display_name())
                .collect(),
            reach,
            short,
            BlockGrid::along(VOLUME, &[0, 1, 2], 8).unwrap(),
        )],
        chain_reach: reach,
    };
    let err = broken
        .check()
        .expect_err("a halo below the fan-in's reach must not check")
        .to_string();
    assert!(!err.is_empty(), "the refusal must say something");

    // The same plan with the halo the fan-in's reach implies does check, so the
    // refusal above is about the missing voxel and not about the shape of the
    // plan.
    plan_with_reach(&workflow, 8, &[0, 1, 2], reach)
        .check()
        .expect("the same plan with a sufficient halo must check");

    let input = intensities();
    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    assert!(
        execute("short", &workflow, &broken, &Hints::default(), &env).is_err(),
        "the executor must refuse a plan whose halo is short of the fan-in's reach"
    );
}

/// A malformed fan-in is refused when the plan is made.
///
/// Two ways to be malformed, both of which reach the same place: `Chain::
/// produces`, which `check_dtypes` calls before any block runs.
#[test]
fn a_combine_that_cannot_join_its_branches_is_refused_before_anything_runs() {
    // A combine handed two different element types. Arm A writes `f64`;
    // `NonZeroOp` narrows to `bool`.
    let chain = Chain::sequence(vec![
        stem(),
        Chain::parallel(
            vec![
                arm_a(None),
                Chain::op(blockflow::probes::NonZeroOp::new("mask", [0, 0, 0])),
            ],
            Box::new(LogicCombine::new("or", Logic::Or)),
        )
        .unwrap(),
    ]);
    let err = chain
        .produces(Dtype::F64)
        .expect_err("a combine must refuse a pair of types it cannot join")
        .to_string();
    assert!(
        err.contains("does not accept [float64, bool]"),
        "got: {err}"
    );

    // And a fan-in with nothing to fan into.
    let empty = match Chain::parallel(Vec::new(), Box::new(LogicCombine::new("or", Logic::Or))) {
        Ok(_) => panic!("a fan-in with no branches was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(empty.contains("no branches"), "got: {empty}");
}
