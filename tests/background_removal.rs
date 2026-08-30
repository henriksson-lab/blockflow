// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops::background` — estimating a
// large-scale background by a grey opening, and removing it.
//
// This op is **the crate's first real diamond in `src/ops/`**: one input, an
// identity arm and an estimate arm, a voxelwise subtraction at the sink. So the
// suite has to make two different kinds of statement, and they are separated
// below because they catch different failures.
//
// What the reach tests catch, and what they cannot
// -----------------------------------------------
// `docs/design/BLOCK_OPS.md` records a real fan-in modelled as a
// `Chain::Alternative` that passed **903 comparisons**: reach folds as the max
// across branches whether one arm runs or all of them, so no comparison of
// reaches, halos or block sizes can tell a diamond from a choice. Every property
// in the first half of this file would pass for a chain that quietly dropped the
// estimate arm and returned the original. That is why
// `both_arms_of_the_diamond_run_once_per_block` counts applications **per arm**
// and compares them against the task count — not "at least once somewhere".
//
// The seven properties:
//
// 1. **The reach is what the parameters imply**, from the chain's own fold and
//    from `background_reach`, which derive it two different ways.
// 2. **Byte-identical to a whole-volume reference** — the same kernels called
//    once over the whole array — under sixteen decompositions per configuration,
//    and identical to each other.
// 3. **The halo guard seen firing**: one voxel short of the derived reach must
//    stop the valid regions tiling and must make the executor refuse the plan.
// 4. **The reach shown tight to one voxel**: understating it by exactly one
//    tiles perfectly and produces wrong values at the seams, so the halo it costs
//    is no wider than what the answer depends on.
// 5. **Both arms ran**, counted per arm, per block.
// 6. **The background is removed**, by a stated measure rather than by running:
//    a field the objects sit on is put in and taken out, and the two figures are
//    computed and printed.
// 7. **The constant algebra**, in bits, with blocks actually short-circuited.
//
// The data is `synthetic::Scene`. Its two-octave background field is exactly what
// this op exists to remove, and — critically for (6) — the field's **amplitude**
// is a knob that does not disturb the object placement or the noise, so the same
// scene can be rendered with and without it and the difference between the two
// renderings *is* the field.

use std::sync::atomic::Ordering;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::background::{
    background_estimate, background_reach, remove_background, DifferenceCombine,
};
use blockflow::ops::{ElementShape, StructuringElement, VoxelwiseMapOp};
use blockflow::probes::CountingIdentityOp;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [28, 20, 16];
const SEED: u64 = 20250812;

/// How far the field swings, and how fast.
///
/// **The estimator can only remove what is slow compared with its own element**,
/// which is a property of the operation rather than a weakness of this test: an
/// opening keeps whatever the element fits inside. One cycle across the longest
/// axis puts the field's two octaves on lattices of 28 and 14 voxels, against an
/// element seven voxels across in property (6). The relationship is stated here
/// so that a reader can see the test is not quietly choosing a field the op
/// cannot fail on.
const FIELD_AMPLITUDE: f64 = 1.2;
const FIELD_CYCLES: f64 = 1.0;

// ------------------------------------------------------------- fixtures --

/// The same scene at a chosen field amplitude.
///
/// `gradient` scales the field at render time and nothing else — object
/// placement is a function of the seed, the shape and the size knobs, and the
/// noise is a function of the seed and the global coordinate. So `render(a)`
/// minus `render(0.0)` is `a` times the field, exactly, which is what property
/// (6) measures against.
fn render(gradient: f64) -> (Array3<f64>, Array3<u32>) {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, SEED)
            .with_objects(25)
            .with_radius(1.0, 1.8)
            .with_touching(0.0, 0.0)
            .with_gradient(gradient, FIELD_CYCLES)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    (rendered.intensity, rendered.labels)
}

fn intensities() -> Array3<f64> {
    render(FIELD_AMPLITUDE).0
}

fn element(shape: ElementShape, radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(shape, radius)
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
        "background",
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

/// Every configuration under test, with the reach it derives. One list, used by
/// properties (1) to (4), so a case cannot be added to one check and forgotten
/// in another.
///
/// The last entry is the **estimate alone**, which is a `Sequence` of two rank
/// filters rather than a fan-in: the same halo arithmetic through a different
/// node, so a change that broke one and not the other is visible.
fn cases() -> Vec<(&'static str, Chain, [usize; 3])> {
    vec![
        (
            "remove, box element of radius one",
            remove_background(&element(ElementShape::Box, [1, 1, 1])).unwrap(),
            [2, 2, 2],
        ),
        (
            "remove, ellipsoid of radius two",
            remove_background(&element(ElementShape::Ellipsoid, [2, 2, 2])).unwrap(),
            [4, 4, 4],
        ),
        (
            "remove, an element flat on one axis",
            remove_background(&element(ElementShape::Box, [3, 1, 0])).unwrap(),
            [6, 2, 0],
        ),
        (
            "the estimate alone",
            background_estimate(&element(ElementShape::Ellipsoid, [2, 2, 2])),
            [4, 4, 4],
        ),
    ]
}

// ------------------------------------------------------- 1. the reaches --

/// The reach every configuration derives is the reach this suite expects, from
/// both statements of it, so the sweeps below cannot silently be testing a
/// weaker property than they name.
#[test]
fn every_configuration_derives_the_reach_its_element_implies() {
    for (name, chain, want) in cases() {
        assert_eq!(
            chain.reach3(&VOLUME),
            want,
            "{name} derived a reach this suite did not expect"
        );
        for axis in 0..3 {
            assert!(
                want[axis] < VOLUME[axis],
                "{name} reaches the whole of axis {axis}, which makes it a planning barrier \
                 rather than a local op and the sweep meaningless"
            );
        }
    }

    // and the second statement of the same quantity agrees with the fold, for
    // every element the cases use
    for (shape, radius) in [
        (ElementShape::Box, [1, 1, 1]),
        (ElementShape::Ellipsoid, [2, 2, 2]),
        (ElementShape::Box, [3, 1, 0]),
    ] {
        let element = element(shape, radius);
        assert_eq!(
            remove_background(&element).unwrap().reach3(&VOLUME),
            background_reach(&element),
            "{shape:?} {radius:?}"
        );
    }
}

// ------------------------------------------------------------ 2. the bar --

/// **The bar.** Byte-identical output against the whole-volume reference, under
/// every decomposition, for every configuration.
#[test]
fn every_configuration_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, chain, _) in cases() {
        let want = reference(&chain, &input);
        assert!(
            want.iter().any(|&value| value != want[[0, 0, 0]]),
            "{name} produced a constant volume, so byte-identity would prove nothing"
        );
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [3usize, 7, 12, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition.check().unwrap_or_else(|err| {
                    panic!("{name}: an honestly derived plan must tile: {err}")
                });
                let got = run(&workflow, &decomposition, &input);
                let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing,
                    0,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the whole-volume \
                     reference at {differing} of {} voxels",
                    got.len()
                );
                ran += 1;
            }
        }
        assert_eq!(ran, 16, "{name}: the sweep did not run");
    }
}

/// The same property stated the other way round: the decompositions agree with
/// **each other**, so nothing rests on the reference being special.
#[test]
fn no_two_decompositions_disagree() {
    let input = intensities();
    for (name, chain, _) in cases() {
        let workflow = workflow(chain);
        let first = run(&workflow, &plan(&workflow, 5, &[0]), &input);
        for block in [6usize, 9, 28] {
            for split_axes in [vec![1], vec![2], vec![0, 2]] {
                assert_eq!(
                    run(&workflow, &plan(&workflow, block, &split_axes), &input),
                    first,
                    "{name}: block {block}, axes {split_axes:?}"
                );
            }
        }
    }
}

// ------------------------------------------------------- 3. the guard --

/// **The guard, seen firing, once per configuration.**
///
/// A halo one voxel short of the derived reach must make the valid regions stop
/// tiling, and the executor must refuse the plan for the same reason. The reach
/// itself is stated honestly in the plan, so the guard is comparing the halo
/// against a number the chain derived rather than against itself.
#[test]
fn a_halo_short_of_the_derived_reach_is_caught_for_every_configuration() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, reach) in cases() {
        let axis = (0..3)
            .find(|&axis| reach[axis] > 0)
            .unwrap_or_else(|| panic!("{name} has no reach on any axis"));
        let workflow = workflow(chain);
        let honest = plan(&workflow, 9, &[axis]);
        honest.check().unwrap();

        let mut short = reach;
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

        let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
        let err = execute("short", &workflow, &forced, &Hints::default(), &env)
            .expect_err(&format!("{name}: the executor must refuse a short halo"))
            .to_string();
        assert!(
            err.contains("do not tile the volume exactly"),
            "{name}: got {err}"
        );
        provoked += 1;
    }
    assert_eq!(provoked, 4, "only {provoked} configurations were provoked");
}

// ------------------------------------------------ 4. the reach, tight --

/// **The silent version of the same failure**, and the evidence that the derived
/// reach is tight rather than merely safe.
///
/// A phase that *understates* its reach by one voxel tiles perfectly — the
/// geometry is self-consistent — and the values are wrong at the seams. Every
/// configuration must be shown to notice a cut of exactly one, or the halo it
/// costs is wider than what the answer depends on.
///
/// The search runs over block edges as well as axes because `reach` is a bound
/// over every block placement and a particular placement need not attain it.
#[test]
fn an_understated_reach_tiles_perfectly_and_produces_wrong_values() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, reach) in cases() {
        let axes: Vec<usize> = (0..3).filter(|&axis| reach[axis] > 0).collect();
        let want = reference(&chain, &input);
        let workflow = workflow(chain);

        let mut smallest_visible = None;
        'search: for short_by in 1..=*reach.iter().max().unwrap() {
            for &axis in &axes {
                if short_by > reach[axis] {
                    continue;
                }
                let mut understated = reach;
                understated[axis] = reach[axis] - short_by;
                for block in [5usize, 7, 9, 12] {
                    let lying = plan_with_reach(&workflow, block, &[axis], understated);
                    lying
                        .check()
                        .expect("an understated reach still tiles — that is the danger");
                    let got = run(&workflow, &lying, &input);
                    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                    if differing > 0 {
                        assert!(
                            differing < got.len(),
                            "{name}: everything differs at axis {axis}, block {block}, so this \
                             is not a seam effect"
                        );
                        smallest_visible = Some(short_by);
                        break 'search;
                    }
                }
            }
        }

        assert_eq!(
            smallest_visible,
            Some(1),
            "{name}: reach {reach:?} is not tight — cutting it by one voxel changed nothing \
             under any decomposition here, so the halo it costs is wider than what the answer \
             depends on"
        );
        provoked += 1;
    }
    assert_eq!(provoked, 4, "only {provoked} configurations were provoked");
}

// ------------------------------------------------------- 5. both arms --

/// **The assertion the 903 passing comparisons could not make.**
///
/// The diamond rebuilt with a counter at the head of each arm — the same two
/// arms `remove_background` assembles, in the same order, through the same sink
/// — and the counts compared against the number of tasks. Not "at least once
/// somewhere": once per block, on each arm.
///
/// Three things are asserted together, and each is needed:
///
/// * the counted chain produces exactly what `remove_background` produces, so
///   the counters are watching the op under test and not a lookalike;
/// * **no block was short-circuited**, or a count below the task count would
///   have an innocent explanation;
/// * both counts equal the task count.
///
/// Plus the value-level statement of the same thing: the answer differs from
/// either arm computed alone, so neither arm is being dropped on the floor.
#[test]
fn both_arms_of_the_diamond_run_once_per_block() {
    let input = intensities();
    let element = element(ElementShape::Ellipsoid, [2, 2, 2]);

    // the value-level statement first
    let plain = remove_background(&element).unwrap();
    let both = reference(&plain, &input);
    let only_original = reference(&Chain::op(VoxelwiseMapOp::new("id", |value| value)), &input);
    let only_estimate = reference(&background_estimate(&element), &input);
    assert_ne!(both, only_original, "the answer is the original arm alone");
    assert_ne!(both, only_estimate, "the answer is the estimate arm alone");
    assert!(
        both.iter().any(|&value| value > 0.0),
        "the difference is zero everywhere, so the two arms agree and nothing is demonstrated"
    );

    // and then the arms themselves, counted
    let (count_a, calls_a) = CountingIdentityOp::new("count_original");
    let (count_b, calls_b) = CountingIdentityOp::new("count_estimate");
    let counted = Chain::parallel(
        vec![
            Chain::sequence(vec![
                Chain::op(count_a),
                Chain::op(VoxelwiseMapOp::new("original", |value| value)),
            ]),
            Chain::sequence(vec![Chain::op(count_b), background_estimate(&element)]),
        ],
        Box::new(DifferenceCombine::new("difference")),
    )
    .unwrap();
    assert_eq!(
        counted.reach3(&VOLUME),
        plain.reach3(&VOLUME),
        "the counters must not change the plan"
    );

    let workflow = workflow(counted);
    let decomposition = plan(&workflow, 7, &[0, 1, 2]);
    let tasks = decomposition.n_tasks();
    assert!(tasks > 1, "one block cannot show a per-block count");

    let (got, short_circuited) = run_reporting(&workflow, &decomposition, &input);
    assert_eq!(got, both, "the counted chain must be the op under test");
    assert_eq!(
        short_circuited, 0,
        "a short-circuited block would explain a count below the task count"
    );
    assert_eq!(
        (
            calls_a.load(Ordering::SeqCst),
            calls_b.load(Ordering::SeqCst)
        ),
        (tasks, tasks),
        "each arm must be applied exactly once per block; this is the observation that \
         distinguishes a fan-in from an alternation"
    );
}

// -------------------------------------------- 6. the background removed --

/// Every voxel where `labels` says there is no object.
fn empty_voxels(labels: &Array3<u32>) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                if labels[[i, j, k]] == 0 {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

fn spread(array: &Array3<f64>, at: &[[usize; 3]]) -> (f64, f64) {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for position in at {
        let value = array[*position];
        low = low.min(value);
        high = high.max(value);
    }
    (low, high)
}

/// **The measurement.** The field goes in, and the numbers say it came out.
///
/// Three figures, each a different way of being wrong if the op did not work,
/// and all three printed so that a reader of the test output sees the size of the
/// effect rather than a bare pass:
///
/// 1. **The field itself.** The same scene rendered with and without the
///    amplitude differs by up to `gap_in` — that difference *is* the field, since
///    nothing else in the render depends on the amplitude. After removal the two
///    answers differ by at most `gap_out`. The ratio is what the op removed.
/// 2. **The spread over object-free voxels.** Before removal the empty parts of
///    the volume span a range set by the field; after removal they span a range
///    set by the noise.
/// 3. **Separability, which is what the removal is *for*.** Before removal no
///    single global level separates the object interiors from the empty voxels —
///    a dim object over a low part of the field sits below a bright part of the
///    field with nothing in it — so the margin is negative. After removal it is
///    positive, and one threshold does.
///
/// **What it measured**, on this scene, with a 123-voxel element of radius three:
///
/// ```text
/// field:             0.6769 in,   0.0790 left,  8.6x
/// empty voxels:      0.9673 span in,   0.1201 span out
/// separating margin: -0.1444 before,   0.2856 after
/// ```
///
/// The bars asserted are looser than those figures — a fivefold reduction where
/// 8.6 was measured — because the number that matters is the one printed and a
/// test that fails on a two-percent drift is a test that gets deleted. What the
/// bars refuse is a *change of kind*: an op that stopped removing the field, or
/// one that flattened the objects along with it.
#[test]
fn the_removal_takes_the_field_out_and_leaves_the_objects() {
    let element = element(ElementShape::Ellipsoid, [3, 3, 3]);
    let chain = remove_background(&element).unwrap();
    assert_eq!(chain.reach3(&VOLUME), [6, 6, 6]);

    let (with_field, labels) = render(FIELD_AMPLITUDE);
    let (flat, flat_labels) = render(0.0);
    assert_eq!(
        labels, flat_labels,
        "the amplitude moved an object, so the two renderings are not the same scene"
    );

    let out_with = reference(&chain, &with_field);
    let out_flat = reference(&chain, &flat);

    // (1) how much of the field survived
    let gap_in = with_field
        .iter()
        .zip(flat.iter())
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
    let gap_out = out_with
        .iter()
        .zip(out_flat.iter())
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
    println!(
        "field: {gap_in:.4} in, {gap_out:.4} left, {:.1}x",
        gap_in / gap_out
    );
    assert!(
        gap_in > 0.5,
        "the field is only {gap_in:.4} deep, so there is nothing much to remove"
    );
    assert!(
        gap_out * 5.0 < gap_in,
        "the removal left {gap_out:.4} of a {gap_in:.4} field, which is less than a \
         fivefold reduction"
    );

    // (2) how flat the object-free parts became
    let empty = empty_voxels(&labels);
    assert!(
        empty.len() * 2 > with_field.len(),
        "most of the volume must be empty"
    );
    let (low_in, high_in) = spread(&with_field, &empty);
    let (low_out, high_out) = spread(&out_with, &empty);
    println!(
        "empty voxels: {:.4} span in, {:.4} span out",
        high_in - low_in,
        high_out - low_out
    );
    assert!(
        (high_out - low_out) * 4.0 < high_in - low_in,
        "the empty voxels spanned {:.4} before and {:.4} after",
        high_in - low_in,
        high_out - low_out
    );

    // (3) separability: one global level, before and after
    let bright: Vec<[usize; 3]> = {
        let mut out = Vec::new();
        for i in 0..VOLUME[0] {
            for j in 0..VOLUME[1] {
                for k in 0..VOLUME[2] {
                    // an object's interior, taken from the field-free rendering
                    // so that the field cannot decide what counts as an object
                    if labels[[i, j, k]] != 0 && flat[[i, j, k]] > 0.5 {
                        out.push([i, j, k]);
                    }
                }
            }
        }
        out
    };
    assert!(bright.len() > 100, "only {} object voxels", bright.len());
    let margin = |array: &Array3<f64>| {
        let (dimmest_object, _) = spread(array, &bright);
        let (_, brightest_empty) = spread(array, &empty);
        dimmest_object - brightest_empty
    };
    let before = margin(&with_field);
    let after = margin(&out_with);
    println!("separating margin: {before:.4} before, {after:.4} after");
    assert!(
        before < 0.0,
        "a single global level already separated the objects before removal ({before:.4}), so \
         this measurement is vacuous"
    );
    assert!(
        after > 0.0,
        "no single global level separates the objects after removal ({after:.4})"
    );
}

// ------------------------------------------- 7. the constant algebra --

/// A block that is short-circuited must produce **exactly** what computing it
/// would have, and here that is `+0.0` — in bits, not "about zero".
///
/// The input is uniformly one value over most of its blocks, so the executor
/// takes the short circuit for some and computes others, and the whole output is
/// still the reference's.
#[test]
fn a_short_circuited_block_produces_the_positive_zero_it_declared() {
    let element = element(ElementShape::Ellipsoid, [2, 2, 2]);
    let chain = remove_background(&element).unwrap();
    assert_eq!(
        chain.constant_maps_to(2.5).map(f64::to_bits),
        Some(0.0_f64.to_bits()),
        "a top-hat of a constant field is exactly positive zero"
    );

    let mut input = Array3::<f64>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 2.5);
    for i in 10..18 {
        for j in 6..14 {
            for k in 4..12 {
                input[[i, j, k]] = 3.5;
            }
        }
    }

    let want = reference(&chain, &input);
    let workflow = workflow(chain);
    let mut skipped = 0;
    for block in [4usize, 6, 9] {
        let (got, short_circuited) =
            run_reporting(&workflow, &plan(&workflow, block, &[0, 1, 2]), &input);
        assert_eq!(got, want, "block {block}");
        for (position, value) in got.indexed_iter() {
            if want[position] == 0.0 {
                assert_eq!(
                    value.to_bits(),
                    0.0_f64.to_bits(),
                    "block {block} wrote a zero of the wrong sign at {position:?}"
                );
            }
        }
        skipped += short_circuited;
    }
    assert!(
        skipped > 0,
        "no block was short-circuited, so this asserted equality against a run that did all \
         the work anyway"
    );
}
