// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for the three ops `docs/design/pixel-classification.md`
// needed and the crate did not have: the structure tensor's eigenvalues, the
// Hessian's eigenvalues unfolded, and the Gaussian gradient magnitude.
//
// **One file for three ops in two modules**, which is unusual here and is
// deliberate. They are the same shape — smooth at a scale, take differences,
// and answer a number per voxel — and every property below is about that shape
// rather than about any one of them. Testing them together is what makes the
// sweep able to say *which* op a seam bug is in, because the same fixture runs
// through all three; testing them apart would have been three copies of this
// file differing in a constructor.
//
// The module's own tests hold the *mathematics* — the ramp's closed form, the
// axis relabelling, the coherence of an edge against a texture. Those all run on
// one whole array and would pass for an op whose declared reach was nonsense.
// This file holds the other half, the one the crate exists for: that the answer
// does not depend on how the volume was cut up.
//
// The same five properties `tests/ridge_filter.rs` states, and one that is this
// op's own:
//
// 1. **A whole-volume reference** — the same `apply`, called once on the entire
//    array, so a disagreement is a decomposition bug and not a second
//    implementation's opinion.
// 2. **Several decompositions**, varying block edge and split axes, each
//    byte-identical to it.
// 3. **The halo guard seen to fire**, through `Decomposition::with_forced_halo`.
// 4. **The same failure seen behaviourally**: one voxel less of reach tiles
//    perfectly and gives wrong values.
// 5. **`constant_maps_to` earned rather than asserted** — blocks seen to be
//    skipped, and the volume still byte-identical.
// 6. **The reach is a sum and not a maximum**, which is this op's own trap and
//    the one place it differs in kind from `ops::ridge`. Ridge's scales are
//    alternatives folded by a maximum; here the derivative smoothing, the
//    stencil and the integration smoothing are stages applied in turn, so they
//    add. An implementation that copied ridge's fold would declare
//    `max(2, 2) + 1 = 3` where the truth is `2 + 1 + 2 = 5` — and property 4,
//    run at *that* understatement rather than at one voxel, is what catches it.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    Eigenvalue, GradientMagnitudeOp, HessianEigenvalueOp, StructureTensor, StructureTensorOp,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [26, 20, 16];

// ------------------------------------------------------------- fixtures --

/// Elongated objects on a gently varying background. A structure tensor wants
/// *oriented* content — a scene of spheres would make every eigenvalue
/// comparable everywhere and property 5's margin would be measuring noise.
fn scene() -> Scene {
    let mut spec = SceneSpec::new(VOLUME, 20260831)
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

/// Isotropic, gamma 1: derivative radius 2, the stencil's 1, integration radius
/// 2. Reach 5 on every axis, and the two radii are equal — which is exactly the
/// case where confusing the sum with the maximum is a factor of nearly two.
fn isotropic(which: Eigenvalue) -> StructureTensorOp {
    StructureTensorOp::new(
        "isotropic",
        StructureTensor::at_gamma([1.0, 1.0, 1.0], 1.0, 2.0).unwrap(),
        which,
    )
}

/// Anisotropic, and a wider integration scale than derivative scale throughout —
/// Labkit's gamma 3 in spirit. Reach `[2+1+4, 2+1+3, 1+1+2] = [7, 6, 4]`,
/// different on every axis, so a per-axis mistake cannot hide behind a cube.
fn anisotropic(which: Eigenvalue) -> StructureTensorOp {
    StructureTensorOp::new(
        "anisotropic",
        StructureTensor::new([1.0, 1.0, 0.5], [2.0, 1.5, 1.0], 2.0).unwrap(),
        which,
    )
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
    run_reporting(workflow, decomposition, input).1
}

/// The output and how many blocks the executor skipped. The second matters
/// exactly once — where what is asserted is that blocks *were* skipped and the
/// answer is still the reference's.
fn run_reporting(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (usize, Array3<f64>) {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    let stats = execute(
        "structure_tensor",
        workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    (
        stats.tasks_short_circuited,
        env.output().view::<f64>().unwrap().to_owned(),
    )
}

/// The configurations the sweep runs, with the reach each one's parameters
/// imply. One list, used by every property below, so a case cannot be added to
/// one check and forgotten in another.
///
/// All three eigenvalues appear, because they come out of the same
/// decomposition but not out of the same arithmetic: the largest is a simple
/// root and the smaller two are the ones the closed form loses digits on, so a
/// seam effect too small to see in the first could still show in the third.
fn cases() -> Vec<(String, Box<dyn Fn() -> Chain>, [usize; 3])> {
    let mut cases: Vec<(String, Box<dyn Fn() -> Chain>, [usize; 3])> = Vec::new();
    for which in Eigenvalue::ALL {
        cases.push((
            format!("structure tensor isotropic/{}", which.as_str()),
            Box::new(move || Chain::op(isotropic(which))),
            [5, 5, 5],
        ));
        cases.push((
            format!("structure tensor anisotropic/{}", which.as_str()),
            Box::new(move || Chain::op(anisotropic(which))),
            [7, 6, 4],
        ));
        // `radius(1.0, 2.0) + 1`, the second-difference stencil's one voxel on
        // top of the smoothing — and *not* a sum of two scales, because there is
        // only one. The two rules living in one sweep is the point.
        cases.push((
            format!("hessian/{}", which.as_str()),
            Box::new(move || {
                Chain::op(HessianEigenvalueOp::new("hessian", [1.0, 1.0, 1.0], 2.0, which).unwrap())
            }),
            [3, 3, 3],
        ));
    }
    cases.push((
        "gradient magnitude isotropic".to_string(),
        Box::new(|| Chain::op(GradientMagnitudeOp::new("grad", [1.0; 3], 2.0).unwrap())),
        [3, 3, 3],
    ));
    cases.push((
        "gradient magnitude anisotropic".to_string(),
        Box::new(|| Chain::op(GradientMagnitudeOp::new("grad", [1.0, 1.0, 0.5], 2.0).unwrap())),
        [3, 3, 2],
    ));
    cases
}

// --------------------------------------------- 6. the reach is a sum --

/// The reaches the parameters imply are the reaches the suite expects.
///
/// For the structure tensor each is strictly larger than the maximum-fold an
/// implementation copying `ops::ridge` would have produced; for the other two
/// there is nothing to fold, and that difference is the reason the three are
/// swept together.
#[test]
fn the_reach_adds_the_two_scales_rather_than_folding_them_by_a_maximum() {
    for (name, chain, want) in cases() {
        assert_eq!(chain().reach3(&VOLUME), want, "{name}");
        for axis in 0..3 {
            assert!(
                want[axis] < VOLUME[axis] / 2,
                "{name}: axis {axis} reaches half the volume, which makes the sweep \
                 meaningless"
            );
        }
    }
    // The terms are visible separately. Derivative radius 2, stencil 1,
    // integration radius 2: a maximum-fold would say 3 where the truth is 5.
    let tensor = StructureTensor::at_gamma([1.0, 1.0, 1.0], 1.0, 2.0).unwrap();
    assert_eq!(tensor.reach(0), 5);
    assert_eq!(isotropic(Eigenvalue::Largest).reach(0, VOLUME[0]), 5);
}

// ------------------------------------------------------- 1 & 2. the bar --

/// **The bar.** Byte-identical output against the whole-volume reference, under
/// every decomposition, for every configuration.
#[test]
fn every_configuration_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, chain, _) in cases() {
        let want = reference(&chain(), &input);
        assert!(
            want.iter().any(|&value| value != 0.0),
            "{name}: the reference is all zeros, so nothing below is being tested"
        );
        let workflow = workflow(chain());
        for block in [6usize, 9, 13] {
            for split_axes in [&[0usize][..], &[1][..], &[0, 1][..], &[0, 1, 2][..]] {
                let decomposition = plan(&workflow, block, split_axes);
                decomposition
                    .check()
                    .unwrap_or_else(|err| panic!("{name} at {block}/{split_axes:?}: {err}"));
                assert_eq!(
                    run(&workflow, &decomposition, &input),
                    want,
                    "{name} at block {block}, split {split_axes:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------- 3. the halo guard --

/// A halo one voxel short of the derived reach must make the valid regions stop
/// tiling, and the executor must refuse the plan for the same reason.
#[test]
fn a_halo_short_of_the_derived_reach_is_caught() {
    let input = intensities();
    let mut provoked = 0;
    for (name, chain, reach) in cases() {
        for axis in 0..3 {
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
        provoked,
        cases().len() * 3,
        "the guard was not provoked on every axis of every case"
    );
}

// ---------------------------------------------------- 4. the tightness --

/// **The reach is tight to one voxel.**
///
/// A phase that understates its reach by exactly one tiles perfectly — `check`
/// is happy, because the geometry is self-consistent — and the values are wrong
/// at the seams. Every axis of every configuration must show it.
///
/// This is also what would catch the maximum-fold: understating the isotropic
/// case's reach from 5 to 3 is two of these, and if one voxel already shows,
/// two certainly do.
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

/// **The maximum-fold, run.** Not argued from the test above but executed: the
/// reach `ops::ridge`'s rule would have produced, in a plan that tiles, gives a
/// different volume. This is the regression test for the one mistake this op is
/// most likely to acquire.
#[test]
fn the_reach_a_maximum_fold_would_have_given_is_visibly_wrong() {
    let input = intensities();
    let chain = || Chain::op(isotropic(Eigenvalue::Largest));
    let want = reference(&chain(), &input);
    let workflow = workflow(chain());

    // `max(radius(sigma), radius(rho)) + 1` = 3, against the truthful 5.
    let folded = plan_with_reach(&workflow, 9, &[0, 1], [3, 3, 3]);
    folded.check().expect("the wrong reach still tiles");
    let got = run(&workflow, &folded, &input);
    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
    assert!(
        differing > 0,
        "a maximum-fold reach reproduced the reference, so this op's reach rule is \
         not what its documentation claims"
    );
    assert!(
        differing < got.len(),
        "everything differs; not a seam effect"
    );
}

// ------------------------------------------------- 5. the short circuit --

/// **`constant_maps_to` earned rather than asserted.** Both halves: that blocks
/// were actually skipped, and that the volume is byte-identical to the
/// whole-volume reference. A run that skipped nothing would pass the second.
#[test]
fn a_constant_volume_is_short_circuited_and_still_matches_the_reference() {
    for constant in [0.0, 0.25, -3.5, 1e6] {
        let input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), constant);
        for (name, chain, _) in cases() {
            let want = reference(&chain(), &input);
            assert!(
                want.iter().all(|&value| value == 0.0),
                "{name}: the whole-volume answer for a constant must be exactly zero"
            );
            let workflow = workflow(chain());
            let decomposition = plan(&workflow, 9, &[0, 1]);
            let (skipped, got) = run_reporting(&workflow, &decomposition, &input);
            assert_eq!(got, want, "{name} at {constant}");
            assert!(
                skipped > 0,
                "{name} at {constant}: nothing was skipped, so this test is not \
                 exercising the declaration it exists to check"
            );
        }
    }
}

// ------------------------------------------------- the op is a feature --

/// **The three eigenvalues are three different features**, which is the premise
/// of asking for six of them per scale in `docs/design/pixel-classification.md`.
/// If two of the three carried the same information the stack would be paying
/// for a duplicate, and this suite — every property of which is about
/// *agreement* — would not notice.
#[test]
fn the_three_eigenvalues_are_not_the_same_image() {
    let input = intensities();
    let outputs: Vec<Array3<f64>> = Eigenvalue::ALL
        .iter()
        .map(|&which| reference(&Chain::op(isotropic(which)), &input))
        .collect();
    for (left, right) in [(0, 1), (1, 2), (0, 2)] {
        let differing = outputs[left]
            .iter()
            .zip(outputs[right].iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > outputs[left].len() / 2,
            "{} and {} agree on {} of {} voxels",
            Eigenvalue::ALL[left].as_str(),
            Eigenvalue::ALL[right].as_str(),
            outputs[left].len() - differing,
            outputs[left].len()
        );
    }
    // Descending, everywhere, which is what `Eigenvalue`'s naming promises.
    for at in 0..outputs[0].len() {
        let (large, middle, small) = (
            outputs[0].as_slice().unwrap()[at],
            outputs[1].as_slice().unwrap()[at],
            outputs[2].as_slice().unwrap()[at],
        );
        assert!(
            large >= middle && middle >= small,
            "{large} {middle} {small}"
        );
    }
    // And a structure tensor is positive semi-definite — it is a smoothed sum
    // of outer products — so the smallest is never negative beyond what the
    // closed form loses at a repeated root.
    let scale = outputs[0].iter().cloned().fold(0.0f64, f64::max);
    let floor = -f64::EPSILON.sqrt() * scale;
    assert!(
        outputs[2].iter().all(|&value| value >= floor),
        "an eigenvalue below {floor:e} means the tensor came out indefinite"
    );
}
