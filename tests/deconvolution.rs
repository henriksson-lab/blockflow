// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops::deconvolve`.
//
// The bar is the module's standing one and not "it runs": a run that agrees at
// a single block size is exactly how a too-small halo passes. So the op is put
// through the four properties every op here is held to, plus three that belong
// to this one:
//
// 1. **A whole-volume reference** — the same `apply`, called once on the entire
//    array. Not a second implementation, so a disagreement is a decomposition
//    bug rather than a modelling difference.
// 2. **Several decompositions**, varying block edge and split axes, each
//    byte-identical to it.
// 3. **The halo guard seen to fire**, structurally, through
//    `Decomposition::with_forced_halo`.
// 4. **The same failure seen behaviourally**: understating the reach by exactly
//    one voxel tiles perfectly and produces wrong values. That is what makes the
//    derived reach *tight* rather than merely safe. It matters more here than
//    for a one-pass filter, because the reach is a product — `2 * radius *
//    iterations` — and every factor of it is a way to be short by a whole
//    multiple rather than by one.
// 5. **It actually deconvolves**: a `synthetic::Scene` is blurred with a known
//    kernel and restored with the same one, and the restored volume is measurably
//    closer to the original than the blurred one is. An op that returned its
//    input would pass 1-4 without complaint.
// 6. **The restoration survives decomposition**: the same improvement, measured
//    on a volume assembled block by block, byte-identical to the whole-volume
//    answer. Properties 2 and 5 are each half of the claim; this is both at once.
// 7. **`constant_maps_to` is exactly true where it is declared and absent where
//    it is not**, checked through the executor rather than argued: blocks are
//    seen to be skipped for a zero volume and the result is byte-identical, and
//    no block is skipped for a non-zero constant — which is the honest answer,
//    because this op does not reproduce one exactly.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::deconvolve::{blur_into, Deconvolution, DeconvolveOp, PointSpread};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [26, 20, 16];

// ------------------------------------------------------------- fixtures --

/// Bright objects on a positive background. Positive throughout, which is what
/// the multiplicative iteration is defined over, and structured, which is what
/// a blur has something to destroy and a seam something to get wrong.
fn scene() -> Scene {
    Scene::new(
        SceneSpec::new(VOLUME, 20250812)
            .with_objects(14)
            .with_radius(1.4, 3.0)
            .with_noise(0.005),
    )
    .unwrap()
}

fn intensities() -> Array3<f64> {
    scene().render().intensity
}

fn kernel(sigma: [f64; 3]) -> PointSpread {
    PointSpread::gaussian(sigma, 2.0).unwrap()
}

fn parameters(sigma: [f64; 3], iterations: usize) -> Deconvolution {
    Deconvolution::new(kernel(sigma), iterations).unwrap()
}

fn chain_for(parameters: &Deconvolution) -> Chain {
    Chain::op(DeconvolveOp::new("deconvolve", parameters.clone()))
}

/// The configurations the sweep runs, with the reach each one's parameters
/// imply. One list, used by every property below, so a case cannot be added to
/// one check and forgotten in another.
///
/// The three exercise the three ways the reach can be wrong: the radius alone
/// (one iteration), the iteration count (three of them, so a reach that forgot
/// to multiply would be a third of what it should be), and a per-axis radius
/// that a cube cannot hide.
fn cases() -> Vec<(&'static str, Deconvolution, [usize; 3])> {
    vec![
        ("one iteration", parameters([0.5, 0.5, 0.5], 1), [2, 2, 2]),
        (
            "three iterations",
            parameters([0.5, 0.5, 0.5], 3),
            [6, 6, 6],
        ),
        (
            "two iterations, anisotropic",
            parameters([1.0, 0.5, 0.5], 2),
            [8, 4, 4],
        ),
    ]
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
    run_reporting(workflow, decomposition, input).0
}

/// The output and how many blocks the executor skipped. The second matters
/// exactly where what is asserted is that blocks *were* skipped and the answer
/// is still the reference's.
fn run_reporting(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
) -> (Array3<f64>, usize) {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    let stats = execute(
        "deconvolve",
        workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    let out = env.output().view::<f64>().unwrap().to_owned();
    (out, stats.tasks_short_circuited)
}

/// Root-mean-square difference between two volumes of the same shape. The
/// measure every restoration claim below is stated in, so that "closer" is one
/// number and not an impression.
fn rms_difference(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    let total: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    (total / left.len() as f64).sqrt()
}

// ------------------------------------------------------- 1 & 2. the bar --

/// The reaches the parameters imply are the reaches the suite expects, which is
/// what keeps the sweep from silently testing a weaker property than it names.
#[test]
fn the_reach_is_twice_the_radius_times_the_iteration_count() {
    for (name, parameters, want) in cases() {
        let chain = chain_for(&parameters);
        assert_eq!(chain.reach3(&VOLUME), want, "{name}");
        for axis in 0..3 {
            assert_eq!(
                want[axis],
                2 * parameters.spread().radius(axis) * parameters.iterations(),
                "{name}: axis {axis}"
            );
            assert!(
                want[axis] < VOLUME[axis],
                "{name} reaches the whole of axis {axis}, which makes the sweep meaningless"
            );
        }
    }
    // and the factors are visible separately: a Gaussian of sigma 0.5 truncated
    // at two standard deviations has radius 1, so one iteration reaches 2 and
    // three reach 6.
    assert_eq!(kernel([0.5; 3]).radius(0), 1);
    assert_eq!(parameters([0.5; 3], 1).reach(0), 2);
    assert_eq!(parameters([0.5; 3], 3).reach(0), 6);
}

/// **The bar.** Byte-identical output against the whole-volume reference, under
/// every decomposition, for every configuration.
#[test]
fn every_configuration_reproduces_its_whole_volume_reference_under_every_decomposition() {
    let input = intensities();
    for (name, parameters, _) in cases() {
        let want = reference(&chain_for(&parameters), &input);
        assert!(
            want.iter().any(|&value| value > 0.0),
            "{name}: a reference of nothing but zeros would make this vacuous"
        );
        assert!(
            want.iter().all(|value| value.is_finite()),
            "{name}: a non-finite reference compares unequal to itself and makes every \
             check below vacuous"
        );
        let workflow = workflow(chain_for(&parameters));
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

/// The same property stated the other way round: the decompositions agree with
/// **each other**, so nothing rests on the reference being special.
#[test]
fn no_two_decompositions_disagree() {
    let input = intensities();
    for (name, parameters, _) in cases() {
        let workflow = workflow(chain_for(&parameters));
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
    for (name, parameters, reach) in cases() {
        for axis in 0..3 {
            assert!(
                reach[axis] > 0,
                "{name}: axis {axis} has no reach to be short of"
            );
            let workflow = workflow(chain_for(&parameters));
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
/// understatement rather than eventually.
#[test]
fn understating_the_reach_by_exactly_one_voxel_changes_the_answer_on_every_axis() {
    let input = intensities();
    for (name, parameters, reach) in cases() {
        let want = reference(&chain_for(&parameters), &input);
        let workflow = workflow(chain_for(&parameters));
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

// ------------------------------------------------ 5. it deconvolves --

/// **The op undoes a blur it is told about.**
///
/// The one measurement that says this is a deconvolution and not a filter that
/// runs: a scene is blurred with a known kernel, the same kernel is handed to
/// the op, and what comes back is compared to the original the blur was applied
/// to. The measure is the root-mean-square difference over the whole volume,
/// stated as a ratio so it does not depend on the scene's brightness.
///
/// The margin asserted is deliberately conservative — a *quarter* off the
/// distance — while the number this configuration actually reaches is printed in
/// the failure message. Iterative deconvolution approaches its answer rather
/// than computing it, so the honest claim is "measurably closer", and the
/// measurement is here rather than a promise that it converges.
///
/// **What it measures today**: the blur leaves the volume 0.0977 from the
/// original in root-mean-square, and twelve iterations bring that to 0.0672 — a
/// ratio of **0.688**, against the 0.75 asserted. The gap between the two is the
/// room a change is allowed to move this before the suite says so.
#[test]
fn a_blurred_scene_is_restored_closer_to_the_original_than_the_blur_left_it() {
    let truth = intensities();
    let blur = kernel([1.2, 1.2, 1.2]);
    let mut blurred = Array3::<f64>::zeros(truth.raw_dim());
    blur_into(truth.view(), &blur, blurred.view_mut()).unwrap();

    let before = rms_difference(&blurred, &truth);
    assert!(
        before > 0.0,
        "the kernel did not blur anything, so there is nothing to restore"
    );

    // The whole-volume claim, at an iteration count whose reach (2 * 3 * 12 =
    // 72) is wider than this volume — which is the trade this family makes and
    // is why the decomposed claim below is stated at a smaller count.
    let restored = reference(
        &chain_for(&Deconvolution::new(blur.clone(), 12).unwrap()),
        &blurred,
    );
    let after = rms_difference(&restored, &truth);
    assert!(
        after < 0.75 * before,
        "12 iterations left the volume {after} from the original, against the blurred \
         volume's {before} — a ratio of {:.3}, and no restoration worth the name",
        after / before
    );
    assert!(
        restored.iter().all(|value| value.is_finite()),
        "a non-finite voxel in the restored volume"
    );
}

/// **The restoration and the decomposition, together.**
///
/// Property 5 measures a real improvement whole-volume; property 2 shows the
/// blocks agree with the whole. This asserts both of one run: an iteration count
/// whose reach fits inside this volume, restoring measurably, byte-identical
/// across six decompositions.
///
/// **What it measures today**: 0.0693 from the original after the blur, 0.0504
/// after four iterations — a ratio of **0.727**, against the 0.9 asserted. The
/// smaller improvement than the twelve-iteration case above is the trade this
/// family makes, priced in one number: the reach is `2 * radius * iterations`,
/// so restoring further is a wider halo and there is no iteration count that is
/// free.
#[test]
fn the_restoration_is_the_same_volume_however_it_is_cut() {
    let truth = intensities();
    let blur = kernel([0.8, 0.8, 0.8]);
    let mut blurred = Array3::<f64>::zeros(truth.raw_dim());
    blur_into(truth.view(), &blur, blurred.view_mut()).unwrap();

    let parameters = Deconvolution::new(blur, 4).unwrap();
    // reach 2 * 2 * 4 = 16 on every axis, which fits inside every axis of this
    // volume — so the sweep below is decomposing something, not running one block
    assert_eq!(chain_for(&parameters).reach3(&VOLUME), [16, 16, 16]);

    let before = rms_difference(&blurred, &truth);
    let want = reference(&chain_for(&parameters), &blurred);
    let after = rms_difference(&want, &truth);
    assert!(
        after < 0.9 * before,
        "4 iterations left {after} against the blur's {before} (ratio {:.3})",
        after / before
    );

    let workflow = workflow(chain_for(&parameters));
    for block in [6usize, 9, 20] {
        for split_axes in [vec![0], vec![1, 2]] {
            let decomposition = plan(&workflow, block, &split_axes);
            decomposition.check().unwrap();
            let got = run(&workflow, &decomposition, &blurred);
            assert_eq!(
                got, want,
                "block {block}, axes {split_axes:?}: the restored volume moved when the \
                 blocks did"
            );
        }
    }
}

// ------------------------------------------- 7. the constant declaration --

/// **`constant_maps_to` is exactly true, checked through the executor.**
///
/// `BlockOp::constant_maps_to` licenses the executor to skip a block whose input
/// is uniform and write the declared value instead, so a skipped block must
/// produce *exactly* what a computed one would have.
///
/// Both halves are asserted, because either alone would be worthless: that
/// blocks were **actually skipped**, and that the volume is byte-identical to
/// the whole-volume reference. A run that skipped nothing would pass the second.
#[test]
fn a_zero_volume_is_short_circuited_and_still_matches_the_reference() {
    let input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0f64);
    for (name, parameters, _) in cases() {
        let want = reference(&chain_for(&parameters), &input);
        assert!(
            want.iter().all(|value| value.to_bits() == 0.0f64.to_bits()),
            "{name}: the whole-volume answer for a zero volume must be exactly +0.0, in \
             the bits — a negative zero would compare equal to what the skip writes and \
             differ from it"
        );
        let workflow = workflow(chain_for(&parameters));
        let decomposition = plan(&workflow, 9, &[0, 1]);
        let (got, skipped) = run_reporting(&workflow, &decomposition, &input);
        assert_eq!(got, want, "{name}");
        // and in the bits, because `-0.0 == 0.0` and this is the one comparison
        // in the suite where a skipped block and a computed one could differ
        // without `assert_eq!` noticing
        assert!(
            got.iter().all(|value| value.to_bits() == 0.0f64.to_bits()),
            "{name}: a skipped block wrote something that is equal to +0.0 and is not it"
        );
        assert!(
            skipped > 0,
            "{name}: nothing was skipped, so this test is not exercising the declaration \
             it exists to check"
        );
    }
}

/// The half of the declaration that is an absence: a non-zero constant is **not**
/// declared, so nothing is skipped and the answer is computed. This op cannot
/// claim a non-zero fixed point — the normalised weights do not sum to exactly
/// one — and the test that it does not claim one is what keeps a later
/// convenience from being added silently.
#[test]
fn a_non_zero_constant_is_computed_rather_than_skipped() {
    for constant in [0.25f64, 4.0, 1e6] {
        let input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), constant);
        for (name, parameters, _) in cases() {
            let op = DeconvolveOp::new("deconvolve", parameters.clone());
            assert_eq!(op.constant_maps_to(constant), None, "{name} at {constant}");

            let want = reference(&chain_for(&parameters), &input);
            let workflow = workflow(chain_for(&parameters));
            let decomposition = plan(&workflow, 9, &[0, 1]);
            let (got, skipped) = run_reporting(&workflow, &decomposition, &input);
            assert_eq!(got, want, "{name} at {constant}");
            assert_eq!(
                skipped, 0,
                "{name} at {constant}: a block was skipped for a constant this op does \
                 not declare a mapping for"
            );
        }
    }
}
