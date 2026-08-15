// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops::normalise` — removing a locally
// estimated level, and normalising against a local centre and spread.
//
// These two ops are the crate's **unanchored-grid case**. Every estimate they
// remove is sampled on a lattice and interpolated back, and
// `docs/design/XY_BLOCK_SPLITTING.md` records what happens when such a lattice
// is laid out relative to the array the op was handed rather than to the volume:
// the same voxel gets a different estimate depending on which block it landed
// in, *throughout* the block rather than at its seams, and no halo repairs it
// because widening the read does not move the lattice. So the bar here is
// **byte-identical output across several decompositions**, and it is a bar
// rather than a nicety: output that is not byte-identical is output whose
// lattice moved.
//
// Five properties, each catching something the others cannot:
//
// 1. **A whole-volume reference** — the same kernels, called once, over the
//    whole array. Not a second implementation, so a disagreement is a
//    decomposition bug and nothing else.
// 2. **Several decompositions** — block edge and split axes varied, each
//    asserted equal to that reference, and to each other.
// 3. **The defect demonstrated, then the invariance asserted.** A block-derived
//    lattice is *constructed* here and shown to move a fixed voxel's sample from
//    one global coordinate to another, and to move it for most of the volume
//    rather than near a seam. Only then is the op asserted invariant — so the
//    assertion is known not to be vacuous.
// 4. **The halo guard seen to fire** — a halo below the derived reach must stop
//    the valid regions tiling and must make the executor refuse the plan.
// 5. **The reach shown tight to one voxel** — understating it by exactly one
//    must produce visibly wrong seams under some decomposition, or the halo it
//    costs is wider than what the answer depends on.
//
// The data is `synthetic::Scene` with its gradient turned up. That field is two
// octaves of lattice noise, which is precisely the shape a sampled estimate can
// get wrong in a scale-dependent way: a block that laid its own lattice out
// would track the field at one scale and the whole volume at another.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    AdaptiveThresholdOp, ElementShape, LevelCorrectionOp, LocalContrastOp, LocalStatistic, Rank,
    Removal, SampleLattice, Statistic, StructuringElement,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

/// A generated volume with a **strong two-octave background field** under the
/// objects, which is what these ops exist to estimate and remove.
///
/// The gradient is turned well up from the default and given more cycles than
/// the default, deliberately: a level that varies at a scale comparable to the
/// sample spacing is the case a lattice can get wrong, and a flat one would let
/// a badly anchored estimate agree with a well anchored one by luck.
fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250812)
            .with_objects(40)
            .with_radius(1.5, 4.0)
            .with_gradient(0.6, 3.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                // Lifted clear of zero so that a division has a level to divide
                // by everywhere and the floor is not what is being tested.
                array[[i, j, k]] = rendered.intensity[[i, j, k]] + 1.0;
            }
        }
    }
    array
}

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the whole chain, at a given block edge and split axes,
/// built from the chain's **own** reach — nothing here supplies one, so nothing
/// here can hide one that is wrong.
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

fn run_reporting(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (Array3<f64>, usize) {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    let stats = execute(
        "normalise",
        workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    (
        env.output().view::<f64>().unwrap().to_owned(),
        stats.tasks_short_circuited,
    )
}

// ------------------------------------------------------------- the ops --

fn element(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, radius)
}

fn statistic(radius: [usize; 3], spacing: [usize; 3], statistic: Statistic) -> LocalStatistic {
    LocalStatistic::new(element(radius), spacing, statistic).unwrap()
}

/// Every configuration under test, with the input it takes and the reach it
/// derives. One list, used by all five properties, so a case cannot be added to
/// one check and forgotten in another.
fn cases(input: &Array3<f64>) -> Vec<(&'static str, Chain, Array3<f64>, [usize; 3])> {
    let low = || Statistic::Rank(Rank::lowest());
    let high = || Statistic::Rank(Rank::highest(&element([1, 1, 1])));

    vec![
        (
            "subtract a sampled low order statistic",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 2, 1], [5, 4, 3], low()),
                    Removal::Subtract,
                )
                .unwrap(),
            ),
            input.clone(),
            [5, 5, 3],
        ),
        (
            "divide by a sampled mean on a coarse lattice",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 2, 1], [8, 8, 8], Statistic::Mean),
                    Removal::Divide { floor: 0.25 },
                )
                .unwrap(),
            ),
            input.clone(),
            [8, 9, 8],
        ),
        (
            "subtract a sampled mean, dense lattice",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 1, 1], [2, 2, 2], Statistic::Mean),
                    Removal::Subtract,
                )
                .unwrap(),
            ),
            input.clone(),
            [2, 2, 2],
        ),
        (
            "contrast against a mean and a deviation on two lattices",
            Chain::op(
                LocalContrastOp::new(
                    "contrast",
                    statistic([1, 3, 1], [5, 4, 3], Statistic::Mean),
                    statistic([1, 1, 1], [9, 3, 3], Statistic::Deviation),
                    0.05,
                )
                .unwrap(),
            ),
            input.clone(),
            // per axis the larger of [5, 6, 3] and [9, 3, 3], never the sum
            [9, 6, 3],
        ),
        (
            "contrast against two order statistics",
            Chain::op(
                LocalContrastOp::new(
                    "contrast",
                    statistic([1, 1, 1], [4, 4, 4], low()),
                    statistic([1, 1, 1], [4, 4, 4], high()),
                    0.05,
                )
                .unwrap(),
            ),
            input.clone(),
            [4, 4, 4],
        ),
        (
            "a chain: remove a level, then threshold against a local mean",
            Chain::sequence(vec![
                Chain::op(
                    LevelCorrectionOp::new(
                        "level",
                        statistic([1, 2, 1], [5, 4, 3], low()),
                        Removal::Subtract,
                    )
                    .unwrap(),
                ),
                Chain::op(AdaptiveThresholdOp::new(
                    "adaptive",
                    statistic([1, 2, 1], [5, 4, 3], Statistic::Mean),
                    1.0,
                    0.01,
                )),
            ]),
            input.clone(),
            [10, 10, 6],
        ),
    ]
}

// ------------------------------------------------- 1 & 2. the main bar --

/// The reaches the ops derive are the reaches this suite expects, which keeps
/// the sweeps below from silently testing a weaker property than they name.
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
/// every decomposition, for every configuration.
#[test]
fn every_op_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, chain, source, _) in cases(&input) {
        let want = reference(&chain, &source);
        assert!(
            want.iter().any(|&value| value != want[[0, 0, 0]]),
            "{name} produced a constant volume, so byte-identity would prove nothing"
        );
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [3usize, 7, 16, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition.check().unwrap_or_else(|err| {
                    panic!("{name}: an honestly derived plan must tile: {err}")
                });
                let got = run(&workflow, &decomposition, &source);
                let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing,
                    0,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the \
                     whole-volume reference at {differing} of {} voxels. A sampled \
                     estimate that disagrees everywhere rather than at the seams is an \
                     unanchored lattice.",
                    got.len()
                );
                ran += 1;
            }
        }
        assert!(ran >= 16, "{name}: the sweep did not run");
    }
}

/// The same property stated the other way round: the decompositions agree with
/// **each other**, so nothing rests on the reference being special.
#[test]
fn no_two_decompositions_disagree() {
    let input = intensities();
    for (name, chain, source, _) in cases(&input) {
        let workflow = workflow(chain);
        let first = run(&workflow, &plan(&workflow, 5, &[0]), &source);
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

// --------------------------------------------------- 3. the anchoring --

/// **Demonstrate, then assert.**
///
/// The defect first, constructed rather than described: for each block edge, a
/// lattice is laid out over *the block* the way `XY_BLOCK_SPLITTING.md` records,
/// and compared against the lattice laid out over the volume. A fixed voxel's
/// sample moves from one global coordinate to another; and the count of voxels
/// whose sample moves is most of the volume, which is the part that makes this a
/// different failure from a seam artefact — a halo cannot help with it.
///
/// Then the ops, over the same block edges, asserted byte-identical to the
/// whole-volume answer. Because the first half showed the defect is real for
/// exactly these decompositions, the second half is known not to be asserting
/// something that was true anyway.
#[test]
fn a_block_derived_lattice_moves_and_these_ops_are_asserted_against_that() {
    let spacing = [5usize, 4, 3];
    let anchored = SampleLattice::centred(VOLUME, spacing).unwrap();

    // --- demonstrate -----------------------------------------------------
    // One fixed voxel, two block edges: the sample it interpolates from sits at
    // two different global coordinates.
    let sample_under = |block: usize, voxel: usize| {
        let start = (voxel / block) * block;
        let len = (start + block).min(VOLUME[0]) - start;
        let local = SampleLattice::centred([len, VOLUME[1], VOLUME[2]], spacing).unwrap();
        start + local.centre(0, local.index_below(0, voxel - start))
    };
    assert_eq!(anchored.centre(0, anchored.index_below(0, 20)), 18);
    assert_eq!(sample_under(8, 20), 20);
    assert_eq!(sample_under(7, 20), 17);
    assert_ne!(
        sample_under(8, 20),
        sample_under(7, 20),
        "if these agreed there would be no defect and the rest of this test would be \
         measuring nothing"
    );

    // And it is not one voxel: most of the volume's voxels would read a
    // different sample. That is what "throughout the block, not at its seams"
    // means, and it is why a halo is not the answer.
    for block in [7usize, 8, 11] {
        let moved = (0..VOLUME[0])
            .filter(|&voxel| {
                sample_under(block, voxel) != anchored.centre(0, anchored.index_below(0, voxel))
            })
            .count();
        assert!(
            moved * 2 > VOLUME[0],
            "block {block}: only {moved} of {} voxels would read a different sample, \
             which is too few for this to demonstrate the defect",
            VOLUME[0]
        );
    }

    // --- assert ----------------------------------------------------------
    let input = intensities();
    for statistic in [
        Statistic::Mean,
        Statistic::Deviation,
        Statistic::Rank(Rank::lowest()),
    ] {
        for removal in [Removal::Subtract, Removal::Divide { floor: 0.25 }] {
            let chain = Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    LocalStatistic::new(element([1, 1, 1]), spacing, statistic.clone()).unwrap(),
                    removal,
                )
                .unwrap(),
            );
            let want = reference(&chain, &input);
            let workflow = workflow(chain);
            for block in [7usize, 8, 11] {
                for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                    let decomposition = plan(&workflow, block, &split_axes);
                    decomposition.check().unwrap();
                    let got = run(&workflow, &decomposition, &input);
                    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing,
                        0,
                        "{statistic:?} {removal:?}: block {block}, axes {split_axes:?} \
                         disagreed at {differing} of {} voxels — an unanchored lattice is \
                         the usual cause",
                        got.len()
                    );
                }
            }
        }
    }
}

/// The invariance holds at **every** spacing rather than at one, which matters
/// because the defect's size scales with the spacing: a lattice that moved by a
/// voxel at spacing two moves by seven at spacing nine.
#[test]
fn the_invariance_holds_at_every_spacing_and_not_merely_at_one() {
    let input = intensities();
    for spacing in [[2usize, 2, 2], [5, 3, 7], [9, 9, 9], [13, 11, 9]] {
        let chain = Chain::op(
            LocalContrastOp::new(
                "contrast",
                LocalStatistic::new(element([1, 1, 1]), spacing, Statistic::Mean).unwrap(),
                LocalStatistic::new(element([1, 1, 1]), spacing, Statistic::Deviation).unwrap(),
                0.05,
            )
            .unwrap(),
        );
        let reach = chain.reach3(&VOLUME);
        for axis in 0..3 {
            assert!(
                reach[axis] < VOLUME[axis],
                "spacing {spacing:?} makes this a barrier on this volume rather than a \
                 local op, so the sweep would prove nothing"
            );
        }
        let want = reference(&chain, &input);
        let workflow = workflow(chain);
        for block in [4usize, 9, 21] {
            for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition.check().unwrap();
                assert_eq!(
                    run(&workflow, &decomposition, &input),
                    want,
                    "spacing {spacing:?}: block {block}, axes {split_axes:?}"
                );
            }
        }
    }
}

// ------------------------------------------------------- 4. the guard --

/// **The guard, seen firing, once per configuration.**
///
/// A halo one voxel short of the derived reach must make the valid regions stop
/// tiling, and the executor must refuse the plan for the same reason.
#[test]
fn a_halo_short_of_the_derived_reach_is_caught_for_every_op() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, source, reach) in cases(&input) {
        let axis = (0..3)
            .find(|&axis| reach[axis] > 0)
            .unwrap_or_else(|| panic!("{name} has no reach on any axis"));
        let workflow = workflow(chain);
        let honest = plan(&workflow, 12, &[axis]);
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
        provoked >= 6,
        "only {provoked} configurations were provoked"
    );
}

// ------------------------------------------------- 5. the reach, tight --

/// **The silent version of the same failure**, and the evidence that each
/// derived reach is tight rather than merely safe.
///
/// A phase that *understates* its reach by one voxel tiles perfectly — the
/// geometry is self-consistent — and the values are wrong at the seams. Every
/// configuration must be shown to notice that, or the halo it costs is wider
/// than what its answer depends on.
///
/// **Why a search over block edges as well as axes.** `reach` is a bound over
/// every block placement and a particular placement need not attain it: with a
/// sample spacing of five and a block edge of eight, no seam lands where the
/// worst case lives. That is not slack in the reach — it is the same reason a
/// single decomposition proves nothing about a halo, arriving from the other
/// side.
///
/// **Why a chain is held to a weaker standard.** For one op the reach is derived
/// from that op's own parameters and is asked to be tight. For a chain,
/// `Chain::reach` **adds** its members' reaches, which is a correct upper bound
/// on the composition's dependency cone but not necessarily an attained one, so
/// the chain is asked only that an understatement bite eventually and how far it
/// had to be cut is reported rather than assumed.
#[test]
fn an_understated_reach_tiles_perfectly_and_produces_wrong_values() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, source, reach) in cases(&input) {
        let axes: Vec<usize> = (0..3).filter(|&axis| reach[axis] > 0).collect();
        let composite = chain.slots().len() > 1;
        let want = reference(&chain, &source);
        let workflow = workflow(chain);

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
                            "{name}: everything differs at axis {axis}, block {block}, so \
                             this is not a seam effect"
                        );
                        smallest_visible = Some(short_by);
                        break 'search;
                    }
                }
            }
        }

        let smallest_visible = smallest_visible.unwrap_or_else(|| {
            panic!(
                "{name}: no understatement of reach {reach:?} produced a wrong value under \
                 any decomposition here, so nothing this op reads matters and its reach \
                 describes something else"
            )
        });
        if composite {
            assert!(
                smallest_visible <= 3,
                "{name}: the summed reach {reach:?} had to be cut by {smallest_visible} \
                 voxels before it mattered, which is more slack than a sum of two reaches \
                 should carry"
            );
        } else {
            assert_eq!(
                smallest_visible, 1,
                "{name}: reach {reach:?} is not tight — cutting it by one voxel changed \
                 nothing under any decomposition here, so the halo it costs is wider than \
                 what the answer depends on"
            );
        }
        provoked += 1;
    }
    assert!(
        provoked >= 6,
        "only {provoked} configurations were provoked"
    );
}

// ------------------------------------------- the constant algebra --

/// A block that is short-circuited must produce **exactly** what computing it
/// would have. The input is uniformly zero over most of its blocks, so the
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

    let low = || Statistic::Rank(Rank::lowest());
    let declaring: Vec<(&str, Chain)> = vec![
        (
            "subtract an order statistic",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 1, 1], [4, 4, 4], low()),
                    Removal::Subtract,
                )
                .unwrap(),
            ),
        ),
        (
            "divide by an order statistic",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 1, 1], [4, 4, 4], low()),
                    Removal::Divide { floor: 0.25 },
                )
                .unwrap(),
            ),
        ),
        (
            "subtract a mean",
            Chain::op(
                LevelCorrectionOp::new(
                    "level",
                    statistic([1, 1, 1], [4, 4, 4], Statistic::Mean),
                    Removal::Subtract,
                )
                .unwrap(),
            ),
        ),
        (
            "contrast against a mean and a deviation",
            Chain::op(
                LocalContrastOp::new(
                    "contrast",
                    statistic([1, 1, 1], [4, 4, 4], Statistic::Mean),
                    statistic([1, 1, 1], [4, 4, 4], Statistic::Deviation),
                    0.25,
                )
                .unwrap(),
            ),
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
            "{name}: no block was short-circuited, so this test asserted equality against \
             a run that did all the work anyway"
        );
    }
}

/// The other half of the asymmetry: what the statistic underneath withholds,
/// these ops withhold, and a chain containing one is not skipped either.
#[test]
fn what_is_not_exactly_true_is_not_declared() {
    let mean = LevelCorrectionOp::new(
        "level",
        statistic([1, 1, 1], [4, 4, 4], Statistic::Mean),
        Removal::Subtract,
    )
    .unwrap();
    assert_eq!(mean.constant_maps_to(0.0), Some(0.0));
    assert_eq!(mean.constant_maps_to(0.1), None);

    // an order statistic selects a value that was read, so subtracting it is
    // exactly zero and dividing by it is exactly one, at every constant
    let low = statistic([1, 1, 1], [4, 4, 4], Statistic::Rank(Rank::lowest()));
    let subtract = LevelCorrectionOp::new("level", low.clone(), Removal::Subtract).unwrap();
    assert_eq!(subtract.constant_maps_to(0.1), Some(0.0));
    let divide = LevelCorrectionOp::new("level", low, Removal::Divide { floor: 0.01 }).unwrap();
    assert_eq!(divide.constant_maps_to(0.1), Some(1.0));

    // and one statistic withholding withholds the pair
    let mixed = LocalContrastOp::new(
        "contrast",
        statistic([1, 1, 1], [4, 4, 4], Statistic::Rank(Rank::lowest())),
        statistic([1, 1, 1], [4, 4, 4], Statistic::Deviation),
        0.25,
    )
    .unwrap();
    assert_eq!(mixed.constant_maps_to(0.0), Some(0.0));
    assert_eq!(mixed.constant_maps_to(0.1), None);
}
