// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops`. The bar is **decomposition
// invariance**, not "it runs", and `docs/design/BLOCK_OPS.md` says why in one
// line: a run that agrees at a single block size is exactly how a too-small halo
// passes.
//
// So every op here is put through four things, and each catches something the
// others cannot:
//
// 1. **A whole-volume reference**, computed by calling the same `apply` once on
//    the entire array. Not a second implementation and not a statistic — the
//    same kernel, so the comparison is exact and a disagreement is a
//    decomposition bug rather than a modelling difference.
// 2. **Several decompositions**, varying block edge and split axes, each
//    asserted byte-identical to that reference.
// 3. **The halo guard seen to fire**, structurally: a halo forced below the
//    derived reach must make the valid regions stop tiling and must make the
//    executor refuse the plan.
// 4. **The same failure seen behaviourally**: a phase that *understates* its
//    reach tiles perfectly and produces wrong values, which is the silent
//    version of the same bug and the reason the structural guard is worth
//    having. A guard nobody has watched fail is not known to work.
//
// The data is `synthetic::Scene`, whose region generation is byte-identical to
// the corresponding cut of a whole generation — which is what makes it valid
// data for a decomposition test rather than merely convenient.

use std::sync::Arc;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    AdaptiveThresholdOp, Boundary, CombineOp, ElementShape, Gaussian, LocalStatistic,
    LocalStatisticOp, Logic, Morphology, MorphologyOp, Rank, RankFilterOp, SmoothOp, Statistic,
    StructuringElement, VoxelwiseMapOp,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

/// A generated volume with structure at several scales, so that a filter has
/// something to do and a seam has something to get wrong.
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

/// A mask, for the ops that take one.
fn mask(input: &Array3<f64>, level: f64) -> Array3<f64> {
    input.mapv(|value| if value > level { 1.0 } else { 0.0 })
}

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the whole chain, at a given block edge and split axes.
///
/// Built from the chain's **own** reach, which is the only honest way: nothing
/// here supplies a reach, so nothing here can hide one that is wrong.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    plan_with_reach(workflow, block, split_axes, workflow.chain.reach3(&VOLUME))
}

/// The same, with the reach stated rather than derived — for provoking the
/// silent failure, and for nothing else.
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
    run_reporting(workflow, decomposition, input).0
}

/// The output and how the run got there. The second half matters exactly once —
/// where what is being asserted is that blocks were *skipped* and the answer is
/// still the reference's.
fn run_reporting(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (Array3<f64>, usize) {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    let stats = execute("ops", workflow, decomposition, &Hints::default(), &env).unwrap();
    (
        env.output().view::<f64>().unwrap().to_owned(),
        stats.tasks_short_circuited,
    )
}

// ------------------------------------------------------------- the ops --

fn box_element(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, radius)
}

fn ball(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, radius)
}

/// Every op in the module, each with the input it takes and the reach it
/// derives. One list, used by all four properties, so an op cannot be added to
/// one check and forgotten in another.
fn cases(input: &Array3<f64>) -> Vec<(&'static str, Chain, Array3<f64>, [usize; 3])> {
    let masked = mask(input, 0.35);
    let operand: Arc<Voxels> = Arc::new(mask(input, 0.5).into());
    let box3 = box_element([1, 1, 1]);
    let ball2 = ball([2, 1, 1]);
    let window = box_element([1, 2, 1]);

    vec![
        (
            "voxelwise map",
            Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0)),
            input.clone(),
            [0, 0, 0],
        ),
        (
            "voxelwise not",
            Chain::op(VoxelwiseMapOp::not("not")),
            masked.clone(),
            [0, 0, 0],
        ),
        (
            "combine AND",
            Chain::op(CombineOp::new("and", Logic::And, operand.clone())),
            masked.clone(),
            [0, 0, 0],
        ),
        (
            "combine OR",
            Chain::op(CombineOp::new("or", Logic::Or, operand.clone())),
            masked.clone(),
            [0, 0, 0],
        ),
        (
            "combine XOR",
            Chain::op(CombineOp::new("xor", Logic::Xor, operand)),
            masked.clone(),
            [0, 0, 0],
        ),
        (
            "median, box element",
            Chain::op(RankFilterOp::median("median", box3.clone())),
            input.clone(),
            [1, 1, 1],
        ),
        (
            "rank 1/4, ellipsoid element",
            Chain::op(RankFilterOp::new(
                "rank",
                ball2.clone(),
                Rank::Nth(ball2.len() / 4),
            )),
            input.clone(),
            [2, 1, 1],
        ),
        (
            "erode",
            Chain::op(MorphologyOp::new("erode", Morphology::Erode, box3.clone())),
            masked.clone(),
            [1, 1, 1],
        ),
        (
            "dilate",
            Chain::op(MorphologyOp::new(
                "dilate",
                Morphology::Dilate,
                ball2.clone(),
            )),
            masked.clone(),
            [2, 1, 1],
        ),
        (
            "open",
            Chain::op(MorphologyOp::new("open", Morphology::Open, box3.clone())),
            masked.clone(),
            [2, 2, 2],
        ),
        (
            "close",
            Chain::op(MorphologyOp::new("close", Morphology::Close, ball2.clone())),
            masked.clone(),
            [4, 2, 2],
        ),
        (
            "local mean on a lattice",
            Chain::op(LocalStatisticOp::new(
                "mean",
                LocalStatistic::new(window.clone(), [5, 4, 3], Statistic::Mean).unwrap(),
            )),
            input.clone(),
            [5, 5, 3],
        ),
        (
            "local deviation on a lattice",
            Chain::op(LocalStatisticOp::new(
                "deviation",
                LocalStatistic::new(window.clone(), [8, 8, 8], Statistic::Deviation).unwrap(),
            )),
            input.clone(),
            [8, 9, 8],
        ),
        (
            "local median on a lattice",
            Chain::op(LocalStatisticOp::new(
                "local median",
                LocalStatistic::new(
                    window.clone(),
                    [4, 4, 4],
                    Statistic::Rank(Rank::median(&window)),
                )
                .unwrap(),
            )),
            input.clone(),
            [4, 5, 4],
        ),
        // Anisotropic on purpose. A smoothing is the first op here whose reach
        // is a *different* number on each axis for a reason other than an
        // element's shape, and `ceil(3 * 1.5) = 5` against `ceil(3 * 0.6) = 2`
        // is exactly the case a per-axis bug survives when every axis is equal.
        (
            "gaussian smooth, anisotropic",
            Chain::op(SmoothOp::new(
                "smooth",
                Gaussian::new([1.0, 1.5, 0.6], 3.0).unwrap(),
            )),
            input.clone(),
            [3, 5, 2],
        ),
        // The same op with the other boundary convention, because a rule about
        // what lies outside the array is exactly the kind of thing that is right
        // whole-volume and wrong under decomposition: the convention belongs to
        // the *volume's* face, and a block whose read extent was not clipped
        // there would apply it at a seam instead. The reach is the same reach,
        // which is a claim this list also checks.
        (
            "gaussian smooth, reflected boundary",
            Chain::op(SmoothOp::new(
                "smooth",
                Gaussian::new([1.0, 1.5, 0.6], 3.0)
                    .unwrap()
                    .with_boundary(Boundary::Reflect),
            )),
            input.clone(),
            [3, 5, 2],
        ),
        (
            "a chain: smooth then threshold",
            Chain::sequence(vec![
                Chain::op(SmoothOp::new(
                    "smooth",
                    Gaussian::isotropic(1.0, 3.0).unwrap(),
                )),
                Chain::op(VoxelwiseMapOp::threshold("threshold", 0.4, 1.0, 0.0)),
            ]),
            input.clone(),
            [3, 3, 3],
        ),
        (
            "adaptive threshold against a local mean",
            Chain::op(AdaptiveThresholdOp::new(
                "adaptive",
                LocalStatistic::new(window.clone(), [5, 4, 3], Statistic::Mean).unwrap(),
                1.0,
                0.02,
            )),
            input.clone(),
            [5, 5, 3],
        ),
        (
            "adaptive threshold against a local median",
            Chain::op(AdaptiveThresholdOp::new(
                "adaptive",
                LocalStatistic::new(
                    window.clone(),
                    [6, 6, 6],
                    Statistic::Rank(Rank::median(&window)),
                )
                .unwrap(),
                1.1,
                0.0,
            )),
            input.clone(),
            [6, 7, 6],
        ),
        (
            "a chain: median, adaptive threshold, open",
            Chain::sequence(vec![
                Chain::op(RankFilterOp::median("median", box3.clone())),
                Chain::op(AdaptiveThresholdOp::new(
                    "adaptive",
                    LocalStatistic::new(window.clone(), [5, 4, 3], Statistic::Mean).unwrap(),
                    1.0,
                    0.01,
                )),
                Chain::op(MorphologyOp::new("open", Morphology::Open, box3)),
            ]),
            input.clone(),
            [8, 8, 6],
        ),
    ]
}

// ------------------------------------------------- 1 & 2. the main bar --

/// The reaches the ops derive are the reaches this suite expects, which is what
/// keeps the sweep below from silently testing a weaker property than it names.
#[test]
fn every_op_derives_the_reach_its_parameters_imply() {
    let input = intensities();
    for (name, chain, _, want) in cases(&input) {
        assert_eq!(
            chain.reach3(&VOLUME),
            want,
            "{name} derived a reach this suite did not expect"
        );
        for axis in 0..3 {
            assert!(
                want[axis] < VOLUME[axis],
                "{name} reaches the whole of axis {axis}, which makes it a planning \
                 barrier rather than a local op and the sweep meaningless"
            );
        }
    }
}

/// **The bar.** Byte-identical output against the whole-volume reference, under
/// every decomposition, for every op.
#[test]
fn every_op_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, chain, source, _) in cases(&input) {
        let want = reference(&chain, &source);
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [3usize, 7, 16, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition.check().unwrap_or_else(|err| {
                    panic!("{name}: an honestly derived plan must tile: {err}")
                });
                assert_eq!(
                    run(&workflow, &decomposition, &source),
                    want,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the \
                     whole-volume reference"
                );
                ran += 1;
            }
        }
        assert!(ran >= 16, "{name}: the sweep did not run");
    }
}

/// The same property stated the other way round: all the decompositions agree
/// with **each other**, so nothing rests on the reference being special.
#[test]
fn no_two_decompositions_disagree() {
    let input = intensities();
    for (name, chain, source, _) in cases(&input) {
        let workflow = workflow(chain);
        let first = run(&workflow, &plan(&workflow, 4, &[0]), &source);
        for block in [6usize, 11, 32] {
            for split_axes in [vec![1], vec![2], vec![0, 2]] {
                assert_eq!(
                    run(&workflow, &plan(&workflow, block, &split_axes), &source),
                    first,
                    "{name}: block {block}, axes {split_axes:?}"
                );
            }
        }
    }
}

/// **A boundary convention belongs to the volume's face, not to a block's.**
///
/// The sweep above already asserts byte-identity for the reflected smoothing, so
/// this test exists to say what that identity *means* and to show it is not
/// vacuous. Three things, in order:
///
/// 1. the two conventions really do disagree, and disagree at the faces and
///    nowhere else — so a run that reflected where it should have clamped would
///    be caught by a comparison of values rather than only of code;
/// 2. reflecting about a **block's** edge would give a different answer, shown
///    by taking the same op over an interior cut of the volume and watching its
///    own new faces produce values the whole-volume run does not have;
/// 3. and the decomposed run has the whole-volume values at every voxel of every
///    face, at block sizes small enough that the faces fall in several different
///    blocks.
///
/// The third is the property; the first two are what stop it from being provable
/// by an op that ignored the parameter.
#[test]
fn a_boundary_convention_is_applied_at_the_volumes_face_and_not_at_a_blocks() {
    fn apply_whole(op: &SmoothOp, input: &Array3<f64>) -> Array3<f64> {
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        op.apply(&source, &mut out, &Anchor::whole(shape))
            .expect("the whole-array reference must run");
        out.view::<f64>().unwrap().to_owned()
    }

    let input = intensities();
    let sigma = [1.0, 1.5, 0.6];
    let radius = [3usize, 5, 2];
    let clamped = SmoothOp::new("clamp", Gaussian::new(sigma, 3.0).unwrap());
    let reflected = SmoothOp::new(
        "reflect",
        Gaussian::new(sigma, 3.0)
            .unwrap()
            .with_boundary(Boundary::Reflect),
    );
    for axis in 0..3 {
        assert_eq!(
            clamped.reach(axis, VOLUME[axis]),
            reflected.reach(axis, VOLUME[axis]),
            "the convention must not move the reach"
        );
    }

    // 1. they differ, at the faces, and only there
    let under_clamp = apply_whole(&clamped, &input);
    let under_reflect = apply_whole(&reflected, &input);
    let mut differing = 0;
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                let at = [i, j, k];
                let interior = (0..3)
                    .all(|axis| at[axis] >= radius[axis] && at[axis] + radius[axis] < VOLUME[axis]);
                if interior {
                    assert_eq!(
                        under_clamp[at].to_bits(),
                        under_reflect[at].to_bits(),
                        "at {at:?} no tap leaves the volume, so the convention cannot show"
                    );
                } else if under_clamp[at] != under_reflect[at] {
                    differing += 1;
                }
            }
        }
    }
    assert!(
        differing > 1000,
        "only {differing} voxels moved; a convention nothing reads would also pass \
         the identity below"
    );

    // 2. the same op over an interior cut invents samples at its *own* new
    // faces, and they are not the volume's values
    let cut = input.slice(ndarray::s![8..24, .., ..]).to_owned();
    let over_cut = apply_whole(&reflected, &cut);
    let mut wrong_at_the_cut = 0;
    for j in 0..VOLUME[1] {
        for k in 0..VOLUME[2] {
            if over_cut[[0, j, k]] != under_reflect[[8, j, k]] {
                wrong_at_the_cut += 1;
            }
        }
    }
    assert!(
        wrong_at_the_cut > 0,
        "reflecting about an interior edge must give a different answer, or this \
         test is asserting nothing about where the reflection happens"
    );

    // 3. and every decomposition has the volume's answer, faces included
    for op in [&clamped, &reflected] {
        let workflow = workflow(Chain::op(SmoothOp::new(op.name(), op.gaussian().clone())));
        let want = if op.name() == "clamp" {
            &under_clamp
        } else {
            &under_reflect
        };
        for block in [3usize, 5, 7, 16] {
            for split_axes in [vec![0], vec![1], vec![2], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition.check().unwrap();
                let got = run(&workflow, &decomposition, &input);
                assert_eq!(
                    &got,
                    want,
                    "{}: block {block}, axes {split_axes:?} reflected about the wrong edge",
                    op.name()
                );
            }
        }
    }
}

// ------------------------------------------------------ 3. the guard --

/// **The guard, seen firing, once per op.**
///
/// A halo one voxel short of the derived reach must make the valid regions stop
/// tiling, and the executor must refuse the plan for the same reason. Ops with
/// no reach are excluded, and excluded *explicitly*: a voxelwise op cannot have
/// a short halo, so a test that included it would report a pass it had not
/// earned.
#[test]
fn a_halo_short_of_the_derived_reach_is_caught_for_every_op_that_has_one() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, source, reach) in cases(&input) {
        let axis = match (0..3).find(|&axis| reach[axis] > 0) {
            Some(axis) => axis,
            None => {
                assert_eq!(reach, [0, 0, 0], "{name}");
                continue;
            }
        };
        let workflow = workflow(chain);
        let honest = plan(&workflow, 8, &[axis]);
        honest.check().unwrap();

        let mut short = [0usize; 3];
        short[axis] = reach[axis] - 1;
        let forced = honest.with_forced_halo(short);

        let err = forced
            .check()
            .expect_err(&format!("{name}: a short halo must not check out"))
            .to_string();
        assert!(
            err.contains("do not tile the volume exactly"),
            "{name}: expected the tiling guard, got: {err}"
        );

        let env = ArrayEnvironment::new(source.clone().into(), 1, [4, 4, 4]).unwrap();
        let err = execute("short", &workflow, &forced, &Hints::default(), &env)
            .expect_err(&format!("{name}: the executor must refuse a short halo"))
            .to_string();
        assert!(
            err.contains("do not tile the volume exactly"),
            "{name}: got {err}"
        );
        provoked += 1;
    }
    assert!(
        provoked >= 10,
        "only {provoked} ops have a reach to be short of; the guard was barely provoked"
    );
}

/// **The silent version of the same failure**, and the evidence that each op's
/// reach is tight rather than merely safe.
///
/// A phase that *understates* its reach by one voxel tiles perfectly — `check`
/// is happy, because the geometry is self-consistent — and the values are wrong
/// at the seams. Every op here must be shown to notice that, or its declared
/// reach is bigger than the data its answer depends on and the halo it costs is
/// being paid for nothing.
///
/// **Why a search over decompositions rather than one.** `reach` is a bound over
/// every block placement, and a *particular* placement need not attain it: with
/// a sample spacing of five and a block edge of eight, no seam happens to land
/// where the worst case lives, and a one-voxel understatement is invisible at
/// that one decomposition. That is not the reach being loose — it is the same
/// reason a single decomposition proves nothing about a halo, arriving from the
/// other direction. So the search is over block edges as well as axes.
///
/// **Why a single op is held to a stricter standard than a chain.** For one op
/// the reach is derived from that op's own parameters and is asked to be
/// *tight*: understating it by exactly one voxel must be visible somewhere. For
/// a chain, `Chain::reach` **adds** the members' reaches, which is a correct
/// upper bound on the composition's dependency cone but not necessarily an
/// attained one — three ops would have to hit their individual worst cases at
/// the same seam. Over-declaring is safe and merely costs halo, so the chain is
/// asked only that an understatement bite eventually, and how far it had to be
/// understated is reported rather than assumed.
#[test]
fn an_understated_reach_tiles_perfectly_and_produces_wrong_values() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, source, reach) in cases(&input) {
        let axes: Vec<usize> = (0..3).filter(|&axis| reach[axis] > 0).collect();
        if axes.is_empty() {
            assert_eq!(reach, [0, 0, 0], "{name}");
            continue;
        }
        let composite = chain.slots().len() > 1;
        let want = reference(&chain, &source);
        let workflow = workflow(chain);

        // The smallest understatement, in voxels, that any decomposition here
        // makes visible.
        let mut smallest_visible = None;
        'search: for short_by in 1..=*reach.iter().max().unwrap() {
            for &axis in &axes {
                if short_by > reach[axis] {
                    continue;
                }
                let mut understated = reach;
                understated[axis] = reach[axis] - short_by;
                for block in [5usize, 7, 8, 11, 13, 16] {
                    let lying = plan_with_reach(&workflow, block, &[axis], understated);
                    lying
                        .check()
                        .expect("an understated reach still tiles — that is the danger");
                    let got = run(&workflow, &lying, &source);
                    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                    if differing > 0 {
                        assert!(
                            differing < got.len(),
                            "{name}: everything differs at axis {axis}, block {block}, \
                             so this is not a seam effect"
                        );
                        smallest_visible = Some(short_by);
                        break 'search;
                    }
                }
            }
        }

        let smallest_visible = smallest_visible.unwrap_or_else(|| {
            panic!(
                "{name}: no understatement of reach {reach:?} produced a wrong value \
                 under any decomposition here, so nothing this op reads matters and \
                 its reach describes something else"
            )
        });
        if composite {
            assert!(
                smallest_visible <= 3,
                "{name}: the summed reach {reach:?} had to be cut by \
                 {smallest_visible} voxels before it mattered, which is more slack \
                 than a sum of three reaches should carry"
            );
        } else {
            assert_eq!(
                smallest_visible, 1,
                "{name}: reach {reach:?} is not tight — cutting it by one voxel \
                 changed nothing under any decomposition here, so the halo it costs \
                 is wider than what the answer depends on"
            );
        }
        provoked += 1;
    }
    assert!(provoked >= 10, "only {provoked} ops were provoked");
}

// ------------------------------------------- 4. the constant algebra --

/// A block that is short-circuited must produce **exactly** what computing it
/// would have. The input here is uniformly empty over most of its blocks, so the
/// executor takes the short circuit for some and not others, and the whole
/// output is still the reference's.
#[test]
fn a_short_circuited_block_produces_what_computing_it_would_have() {
    let mut input = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 12..20 {
        for j in 8..16 {
            for k in 6..14 {
                input[[i, j, k]] = 1.0;
            }
        }
    }

    let box3 = box_element([1, 1, 1]);
    let window = box_element([1, 1, 1]);
    let declaring: Vec<(&str, Chain)> = vec![
        (
            "median",
            Chain::op(RankFilterOp::median("median", box3.clone())),
        ),
        (
            "open",
            Chain::op(MorphologyOp::new("open", Morphology::Open, box3.clone())),
        ),
        (
            "local median",
            Chain::op(LocalStatisticOp::new(
                "local median",
                LocalStatistic::new(
                    window.clone(),
                    [4, 4, 4],
                    Statistic::Rank(Rank::median(&window)),
                )
                .unwrap(),
            )),
        ),
        (
            "adaptive threshold against a local mean",
            Chain::op(AdaptiveThresholdOp::new(
                "adaptive",
                LocalStatistic::new(window, [4, 4, 4], Statistic::Mean).unwrap(),
                1.0,
                0.5,
            )),
        ),
        (
            "AND against an operand",
            Chain::op(CombineOp::new(
                "and",
                Logic::And,
                Arc::new(Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 1.0).into()),
            )),
        ),
    ];

    for (name, chain) in declaring {
        assert!(
            chain.constant_maps_to(0.0).is_some(),
            "{name} declares nothing for an empty block, so this test would prove nothing"
        );
        let want = reference(&chain, &input);
        let workflow = workflow(chain);
        let mut skipped = 0;
        for block in [4usize, 8, 12] {
            let (got, short_circuited) =
                run_reporting(&workflow, &plan(&workflow, block, &[0, 1, 2]), &input);
            assert_eq!(got, want, "{name}: block {block}");
            skipped += short_circuited;
        }
        assert!(
            skipped > 0,
            "{name}: no block was short-circuited, so this test asserted equality \
             against a run that did all the work anyway"
        );
    }
}

/// The other half of the asymmetry: an op that has not stated a mapping is never
/// skipped, and a chain containing one is not either.
#[test]
fn what_is_not_exactly_true_is_not_declared() {
    let window = box_element([1, 1, 1]);
    let mean = LocalStatisticOp::new(
        "mean",
        LocalStatistic::new(window, [4, 4, 4], Statistic::Mean).unwrap(),
    );
    // exact at zero, and withheld everywhere else, because the mean of m copies
    // of a general v is not v in binary floating point
    assert_eq!(mean.constant_maps_to(0.0), Some(0.0));
    assert_eq!(mean.constant_maps_to(0.1), None);

    // and a two-input op answers only where its connective absorbs
    let operand: Arc<Voxels> =
        Arc::new(Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 1.0).into());
    let and = CombineOp::new("and", Logic::And, operand.clone());
    assert_eq!(and.constant_maps_to(0.0), Some(0.0));
    assert_eq!(and.constant_maps_to(1.0), None);
    let xor = CombineOp::new("xor", Logic::Xor, operand);
    assert_eq!(xor.constant_maps_to(0.0), None);
    assert_eq!(xor.constant_maps_to(1.0), None);
}

// ------------------------------------------------------- the diamond --

/// The shape the crate could not express: one input, two arms, one combine.
///
/// `Chain` still has no fan-in node, so the second arm is materialised whole and
/// handed to the combine as its operand — which is the arrangement described in
/// `ops::voxelwise`. What is being asserted is not that the plumbing is elegant
/// but that the **combined** result is decomposition-invariant, which is the
/// property a fan-in node would have to preserve when it arrives.
#[test]
fn a_diamond_combines_two_arms_and_stays_decomposition_invariant() {
    let input = intensities();
    let box3 = box_element([1, 1, 1]);
    let window = box_element([1, 1, 1]);

    // arm B, computed whole: a local threshold, opened.
    let arm_b = Chain::sequence(vec![
        Chain::op(AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(window, [4, 4, 4], Statistic::Mean).unwrap(),
            1.0,
            0.0,
        )),
        Chain::op(MorphologyOp::new("open", Morphology::Open, box3.clone())),
    ]);
    let operand: Arc<Voxels> = Arc::new(reference(&arm_b, &input).into());

    // arm A, and the sink that combines it with B.
    let diamond = Chain::sequence(vec![
        Chain::op(RankFilterOp::median("median", box3)),
        Chain::op(VoxelwiseMapOp::threshold("threshold", 0.3, 1.0, 0.0)),
        Chain::op(CombineOp::new("or", Logic::Or, operand)),
    ]);
    let want = reference(&diamond, &input);
    assert!(
        want.iter().any(|&value| value == 1.0) && want.iter().any(|&value| value == 0.0),
        "the diamond produced a constant, so it is not discriminating anything"
    );

    let workflow = workflow(diamond);
    for block in [4usize, 7, 13, 64] {
        for split_axes in [vec![0], vec![1, 2], vec![0, 1, 2]] {
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

// --------------------------------------------------- global anchoring --

/// The defect class this batch's last two ops are exposed to, asserted at the
/// image a user would notice it.
///
/// An unanchored sample lattice gives the same voxel a different value depending
/// on which block it landed in — **throughout** the block, not at its seams — so
/// the symptom is not a thin band of disagreement but a large fraction of the
/// volume differing. That is what this test would see if the lattice were ever
/// re-derived from a buffer, and it is checked at three spacings because the
/// defect scales with the spacing and would be invisible at one.
#[test]
fn a_sampled_op_is_invariant_at_every_spacing_and_not_merely_at_one() {
    let input = intensities();
    let window = box_element([1, 1, 1]);
    for spacing in [[2usize, 2, 2], [5, 3, 7], [9, 9, 9], [13, 11, 9]] {
        for statistic in [
            Statistic::Mean,
            Statistic::Deviation,
            Statistic::Rank(Rank::median(&window)),
        ] {
            let chain = Chain::op(LocalStatisticOp::new(
                "sampled",
                LocalStatistic::new(window.clone(), spacing, statistic.clone()).unwrap(),
            ));
            let reach = chain.reach3(&VOLUME);
            if (0..3).any(|axis| reach[axis] >= VOLUME[axis]) {
                // A reach that spans an axis is a planning barrier, not a local
                // op; skipping it here is a statement about the fixture, and the
                // spacings above are chosen so that it does not happen.
                panic!("spacing {spacing:?} makes this a barrier on this volume");
            }
            let want = reference(&chain, &input);
            let workflow = workflow(chain);
            for block in [4usize, 9, 21] {
                for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                    let decomposition = plan(&workflow, block, &split_axes);
                    decomposition.check().unwrap();
                    let got = run(&workflow, &decomposition, &input);
                    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing,
                        0,
                        "spacing {spacing:?} {statistic:?}: block {block}, axes \
                         {split_axes:?} disagreed at {differing} of {} voxels — an \
                         unanchored lattice is the usual cause",
                        got.len()
                    );
                }
            }
        }
    }
}
