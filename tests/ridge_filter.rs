// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops::ridge`.
//
// The bar is the module's standing one and not "it runs": a run that agrees at a
// single block size is exactly how a too-small halo passes. So the filter is put
// through the same four properties every other op here is held to, plus three
// that are its own:
//
// 1. **A whole-volume reference** — the same `apply`, called once on the entire
//    array. Not a second implementation, so a disagreement is a decomposition
//    bug rather than a modelling difference.
// 2. **Several decompositions**, varying block edge and split axes, each
//    byte-identical to it. Including the side output, which is positioned from
//    the volume and would move with the block if it were not.
// 3. **The halo guard seen to fire**, structurally, through
//    `Decomposition::with_forced_halo`.
// 4. **The same failure seen behaviourally**: understating the reach by exactly
//    one voxel tiles perfectly and produces wrong values. That is what makes the
//    derived reach *tight* rather than merely safe, and the `+ 1` the derivative
//    stencil contributes is precisely a one-voxel term — so this is the test
//    that would have caught leaving it out.
// 5. **The filter answers**: the response is higher on the elongated objects
//    `synthetic::Scene` places than on the background between them. A filter
//    that is decomposition-invariant and responds to nothing would pass 1-4.
// 6. **The scale map is a real choice**: the argmax over scales is not the same
//    scale everywhere, or the side output carries no information.
// 7. **`constant_maps_to` is exactly true**, checked through the executor rather
//    than argued: blocks are seen to be skipped, and the volume is still
//    byte-identical to the reference. This op declares a mapping the local mean
//    withholds, so the declaration has to be earned rather than asserted.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{Polarity, RidgeFilterOp, RidgeResponse, ScaleSpace};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [26, 20, 16];

// ------------------------------------------------------------- fixtures --

/// Elongated objects on a gently varying background, which is what a ridge
/// filter is supposed to find and what a seam has something to get wrong.
///
/// `elongation` is turned up from the spec's default: the objects a ridge filter
/// should answer for are the ones that are longer than they are wide, and the
/// default scene is closer to round.
fn scene() -> Scene {
    let mut spec = SceneSpec::new(VOLUME, 20250812)
        .with_objects(18)
        .with_radius(1.2, 2.6)
        .with_noise(0.01);
    spec.elongation = 3.5;
    spec.gradient = 0.1;
    Scene::new(spec).unwrap()
}

fn intensities() -> Array3<f64> {
    scene().render().intensity
}

/// One scale, and the smallest reach the filter can have: radius 2 plus the
/// stencil's 1.
fn narrow() -> RidgeFilterOp {
    RidgeFilterOp::new(
        "narrow",
        ScaleSpace::isotropic(&[1.0], 2.0, 1.0).unwrap(),
        response(),
    )
}

/// Two scales and an **anisotropic** one, so the reach differs per axis and a
/// per-axis mistake cannot hide behind a cube.
fn wide() -> RidgeFilterOp {
    RidgeFilterOp::new(
        "wide",
        ScaleSpace::new(vec![[1.0, 1.0, 1.0], [2.0, 1.4, 0.6]], 2.0, 1.0).unwrap(),
        response(),
    )
}

fn response() -> RidgeResponse {
    RidgeResponse::new(0.5, 0.5, 0.05, Polarity::Ridge).unwrap()
}

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the chain, built from the chain's **own** reach — nothing
/// here supplies one, so nothing here can hide one that is wrong.
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

/// The oracle: the same kernel, called once, over the whole array.
fn reference(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn run(workflow: &Workflow, decomposition: &Decomposition, input: &Array3<f64>) -> Array3<f64> {
    run_with_env(workflow, decomposition, input).0
}

fn run_with_env(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (Array3<f64>, ArrayEnvironment) {
    let (out, env, _) = run_reporting(workflow, decomposition, input);
    (out, env)
}

/// The output, the environment, and how many blocks the executor skipped. The
/// last of those matters exactly once — where what is asserted is that blocks
/// *were* skipped and the answer is still the reference's.
fn run_reporting(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (Array3<f64>, ArrayEnvironment, usize) {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    let stats = execute("ridge", workflow, decomposition, &Hints::default(), &env).unwrap();
    let out = env.output().view::<f64>().unwrap().to_owned();
    (out, env, stats.tasks_short_circuited)
}

/// The configurations the sweep runs, with the reach each one's parameters
/// imply. One list, used by every property below, so a case cannot be added to
/// one check and forgotten in another.
fn cases() -> Vec<(&'static str, fn() -> Chain, [usize; 3])> {
    vec![
        ("one scale", || Chain::op(narrow()), [3, 3, 3]),
        ("two scales, anisotropic", || Chain::op(wide()), [5, 4, 3]),
        (
            "with a scale map",
            || Chain::op(wide().with_scale_map("scale")),
            [5, 4, 3],
        ),
    ]
}

// ------------------------------------------------------- 1 & 2. the bar --

/// The reaches the parameters imply are the reaches the suite expects, which is
/// what keeps the sweep from silently testing a weaker property than it names.
#[test]
fn the_reach_is_the_widest_gaussian_radius_plus_the_derivative_stencil() {
    for (name, chain, want) in cases() {
        assert_eq!(chain().reach3(&VOLUME), want, "{name}");
        for axis in 0..3 {
            assert!(
                want[axis] < VOLUME[axis],
                "{name} reaches the whole of axis {axis}, which makes the sweep meaningless"
            );
        }
    }
    // and the terms are visible separately: truncate 2.0 on sigma 1.0 gives a
    // radius of 2, and the reach is 3.
    assert_eq!(narrow().scales().radius(0, 0), 2);
    assert_eq!(narrow().reach(0, VOLUME[0]), 3);
}

/// **The bar.** Byte-identical output against the whole-volume reference, under
/// every decomposition, for every configuration.
#[test]
fn every_configuration_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, chain, _) in cases() {
        let chain = chain();
        let want = reference(&chain, &input);
        assert!(
            want.iter().any(|&value| value > 0.0),
            "{name}: a reference of nothing but zeros would make this vacuous"
        );
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [6usize, 9, 13, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition
                    .check()
                    .unwrap_or_else(|err| panic!("{name}: an honest plan must tile: {err}"));
                assert_eq!(
                    run(&workflow, &decomposition, &input),
                    want,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the \
                     whole-volume reference"
                );
                ran += 1;
            }
        }
        assert_eq!(ran, 16, "{name}: the sweep did not run");
    }
}

/// The side output under the same bar. It is positioned from the volume, and a
/// side output positioned from the block is the failure this catches — the
/// primary would still agree.
#[test]
fn the_scale_map_is_the_same_array_under_every_decomposition() {
    let input = intensities();
    let op = wide().with_scale_map("scale");
    let name = op.side_outputs(VOLUME)[0].name.clone();
    assert_eq!(name, "wide.scale");
    let workflow = workflow(Chain::op(op));

    let mut first: Option<ndarray::ArrayD<f64>> = None;
    for block in [6usize, 11, 64] {
        for split_axes in [vec![0], vec![1, 2], vec![0, 1, 2]] {
            let decomposition = plan(&workflow, block, &split_axes);
            let (_, env) = run_with_env(&workflow, &decomposition, &input);
            let got = env.side_output(&name).expect("the scale map was declared");
            assert!(
                got.iter().all(|value| value.is_finite()),
                "{block}/{split_axes:?}: a hole in the scale map is a NaN"
            );
            match &first {
                None => first = Some(got),
                Some(want) => assert_eq!(&got, want, "block {block}, axes {split_axes:?}"),
            }
        }
    }
    let map = first.expect("the sweep ran");
    // 6. it is a real choice: both scales win somewhere, or the map carries
    // nothing and the test above is comparing two constant arrays.
    assert!(map.iter().any(|&value| value == 0.0), "{map:?}");
    assert!(
        map.iter().any(|&value| value == 1.0),
        "no voxel preferred the second scale, so the argmax is not a choice"
    );
}

/// The same property stated the other way round: the decompositions agree with
/// **each other**, so nothing rests on the reference being special.
#[test]
fn no_two_decompositions_disagree() {
    let input = intensities();
    for (name, chain, _) in cases() {
        let workflow = workflow(chain());
        let first = run(&workflow, &plan(&workflow, 7, &[0]), &input);
        for block in [8usize, 12] {
            for split_axes in [vec![1], vec![0, 2]] {
                assert_eq!(
                    run(&workflow, &plan(&workflow, block, &split_axes), &input),
                    first,
                    "{name}: block {block}, axes {split_axes:?}"
                );
            }
        }
    }
}

// -------------------------------------------------------- 3. the guard --

/// **The guard, seen firing.**
///
/// A halo one voxel short of the derived reach must make the valid regions stop
/// tiling, and the executor must refuse the plan for the same reason.
#[test]
fn a_halo_short_of_the_derived_reach_is_caught() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, reach) in cases() {
        for axis in 0..3 {
            assert!(
                reach[axis] > 0,
                "{name}: axis {axis} has no reach to be short of"
            );
            let workflow = workflow(chain());
            let honest = plan(&workflow, 9, &[axis]);
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

            let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
            let err = execute("short", &workflow, &forced, &Hints::default(), &env)
                .expect_err(&format!("{name}: the executor must refuse a short halo"))
                .to_string();
            assert!(
                err.contains("do not tile the volume exactly"),
                "{name}: {err}"
            );
            provoked += 1;
        }
    }
    assert_eq!(
        provoked, 9,
        "the guard was not provoked on every axis of every case"
    );
}

// ---------------------------------------------------- 4. the tightness --

/// **The reach is tight to one voxel**, which is the property the whole
/// derivation rests on.
///
/// A phase that understates its reach by exactly one tiles perfectly — `check`
/// is happy, because the geometry is self-consistent — and the values are wrong
/// at the seams. Every axis of every configuration must show it, on the first
/// understatement rather than eventually: the derivative stencil contributes
/// exactly one voxel, so an implementation that forgot it would be short by
/// exactly this much, and the test has to be able to tell.
#[test]
fn understating_the_reach_by_exactly_one_voxel_changes_the_answer_on_every_axis() {
    let input = intensities();
    for (name, chain, reach) in cases() {
        let want = reference(&chain(), &input);
        let workflow = workflow(chain());
        for axis in 0..3 {
            let mut understated = reach;
            understated[axis] = reach[axis] - 1;

            let mut seen = false;
            for block in [7usize, 9, 11] {
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
                    seen = true;
                    break;
                }
            }
            assert!(
                seen,
                "{name}: cutting the reach on axis {axis} from {} to {} changed nothing \
                 under any block size here, so the declared reach is wider than what the \
                 answer depends on",
                reach[axis], understated[axis]
            );
        }
    }
}

// ------------------------------------------------- 5. the filter answers --

/// **The filter responds to what it is for.** A decomposition-invariant filter
/// that answered zero everywhere would pass every test above.
///
/// The comparison is between the objects `Scene` placed and the background
/// between them, using the label volume as ground truth — the same generator,
/// so the two sets are exactly the ones that were drawn.
///
/// The margin asserted is on the **mean over each set**, not on every voxel, and
/// that is deliberate rather than a weakened claim: a ridge filter's response
/// peaks along an object's centre line and falls to nothing in the middle of a
/// wide one, and it is non-zero just outside an object where the smoothed
/// intensity still curves. A per-voxel claim would be false for this filter and
/// for any other of its kind.
#[test]
fn the_response_is_higher_on_the_objects_than_on_the_background() {
    let rendered = scene().render();
    assert!(
        rendered.labels.iter().any(|&label| label > 0),
        "the scene placed nothing, so there is nothing to be higher on"
    );
    for (name, chain, _) in cases() {
        let got = reference(&chain(), &rendered.intensity);

        let mut on = (0.0f64, 0usize);
        let mut off = (0.0f64, 0usize);
        for (value, &label) in got.iter().zip(rendered.labels.iter()) {
            let bucket = if label > 0 { &mut on } else { &mut off };
            bucket.0 += value;
            bucket.1 += 1;
        }
        let on_mean = on.0 / on.1 as f64;
        let off_mean = off.0 / off.1 as f64;
        assert!(
            on_mean > 3.0 * off_mean,
            "{name}: mean response {on_mean} on {} object voxels against {off_mean} on \
             {} background voxels",
            on.1,
            off.1
        );
        // and the strongest response in the volume is on an object, not on the
        // background — the statement the mean cannot make
        let (best, _) = got
            .indexed_iter()
            .zip(rendered.labels.iter())
            .max_by(|left, right| left.0 .1.total_cmp(right.0 .1))
            .unwrap();
        assert!(
            rendered.labels[best.0] > 0,
            "{name}: the strongest response in the volume is background at {:?}",
            best.0
        );
    }
}

// ------------------------------------------- the constant declaration --

/// **`constant_maps_to` is exactly true, checked through the executor.**
///
/// `BlockOp::constant_maps_to` licenses the executor to skip a block whose input
/// is uniform and write the declared value instead, so a skipped block must
/// produce *exactly* what a computed one would have. The local mean cannot make
/// this claim — `(v + ... + v) / m` is not `v` in binary floating point — and
/// this op can, for the reason set out on `RidgeFilterOp::constant_maps_to`.
///
/// Both halves are asserted, because either alone would be worthless: that
/// blocks were **actually skipped**, and that the volume is byte-identical to
/// the whole-volume reference. A run that skipped nothing would pass the second.
#[test]
fn a_constant_volume_is_short_circuited_and_still_matches_the_reference() {
    for constant in [0.0, 0.25, -3.5, 1e6] {
        let input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), constant);
        for (name, chain, _) in cases() {
            let chain = chain();
            let declares_a_side_output = !chain.side_outputs(VOLUME).is_empty();
            let want = reference(&chain, &input);
            assert!(
                want.iter().all(|&value| value == 0.0),
                "{name}: the whole-volume answer for a constant must be exactly zero"
            );
            let workflow = workflow(chain);
            let decomposition = plan(&workflow, 9, &[0, 1]);
            let (got, _, skipped) = run_reporting(&workflow, &decomposition, &input);
            assert_eq!(got, want, "{name} at {constant}");
            if declares_a_side_output {
                // The framework refuses to skip a phase that writes a side
                // output, because the algebra that licenses the skip is about
                // the primary result only. Asserted rather than assumed, so
                // that the count below is not silently explained away.
                assert_eq!(skipped, 0, "{name}");
            } else {
                assert!(
                    skipped > 0,
                    "{name} at {constant}: nothing was skipped, so this test is not \
                     exercising the declaration it exists to check"
                );
            }
        }
    }
}
