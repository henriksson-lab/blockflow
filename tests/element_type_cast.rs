// SPDX-License-Identifier: MIT
//
// **A plan that changes the width of what it holds.**
//
// `ops::voxelwise::WidenOp` was the step from a level held at its producer's
// width to a kernel stated in `f64`. `ops::voxelwise::NarrowOp` is the step
// back, and it is what makes a *narrow* level reachable from inside a plan at
// all: `VoxelwiseMapOp` is an `f64 -> f64` map in an `f64` buffer, so a chain
// could compute `value > level` and had nowhere to put the answer but eight
// bytes a voxel. A `bool` level was therefore unreachable however the plan was
// written.
//
// What each test is for
// ---------------------
// * **The gap, closed, through a real plan.** A threshold followed by a cast,
//   producing a `bool` level, compared against the comparison itself — computed
//   here from the definition, not recorded from a run of this code.
// * **Decomposition invariance.** Several cuts, byte for byte. A voxelwise op
//   has reach 0 and so has nothing to get wrong at a seam *in principle*, which
//   is the kind of claim that stops holding the moment somebody adds a fast path
//   keyed on the buffer being whole — which this op's sibling has.
// * **The existing map op is byte-unchanged.** The risk of adding an output
//   element type was that it would be added by teaching `VoxelwiseMapOp` to
//   narrow, which would have moved every existing chain's answer. The same
//   chains are run here and compared against the definition.
// * **The round trip.** Narrowing then widening returns the values the
//   narrowing kept, which is what says the two ops are inverses where they
//   agree — and only there.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::local::{Narrowing, Rounding};
use blockflow::ops::{NarrowOp, VoxelwiseMapOp, WidenOp};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [24, 18, 12];

/// A real volume rather than a ramp, because the interesting property of a
/// threshold is where the data actually sits relative to the level.
fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250811)
            .with_objects(25)
            .with_radius(1.5, 3.5)
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

/// One phase per chain slot, so an intermediate really is materialised at the
/// element type the slot before it declared.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let reach = workflow.chain.reach3(&VOLUME);
    let phases = workflow
        .chain
        .slots()
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            PhaseDecomposition::derive(
                vec![index],
                vec![slot.display_name()],
                reach,
                reach,
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: reach,
    };
    plan.declare_dtypes(&workflow.chain).unwrap();
    plan
}

fn run(chain: Chain, block: usize, split_axes: &[usize], input: &Array3<f64>) -> Voxels {
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let decomposition = plan(&workflow, block, split_axes);
    let env = ArrayEnvironment::for_decomposition(input.clone().into(), &decomposition, [4, 4, 4])
        .unwrap();
    execute("cast", &workflow, &decomposition, &Hints::default(), &env).unwrap();
    env.output().clone()
}

fn whole(chain: &Chain, input: &Array3<f64>) -> Voxels {
    let source: Voxels = input.clone().into();
    let produces = chain.produces(Dtype::F64).expect("the chain must assemble");
    let mut out = Voxels::zeros(produces, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out
}

/// The cuts. Two edge lengths that divide the volume and two that do not, over
/// three sets of split axes, so that a partial block is exercised on each.
const BLOCKS: [usize; 4] = [4, 7, 12, 24];
const AXES: [&[usize]; 3] = [&[0], &[0, 2], &[0, 1, 2]];

const LEVEL: f64 = 0.30;

// --------------------------------------------------------------- the tests --

/// **The gap, closed.** A comparison computed in `f64` reaching a `bool` level
/// through a plan, and the answer is the comparison.
#[test]
fn a_threshold_reaches_a_bool_level_through_a_plan() {
    let input = intensities();
    let chain = || {
        Chain::sequence(vec![
            Chain::op(VoxelwiseMapOp::threshold("step", LEVEL, 1.0, 0.0)),
            Chain::op(NarrowOp::to_mask("mask")),
        ])
    };
    assert_eq!(
        chain().produces(Dtype::F64).unwrap(),
        Dtype::Bool,
        "the plan must declare a bool level, which is what could not be said before"
    );

    let want = whole(&chain(), &input);
    let mask = want.view::<bool>().unwrap();
    // Against the definition, not against a recorded output.
    for (value, &set) in input.iter().zip(mask.iter()) {
        assert_eq!(set, *value > LEVEL, "at {value}");
    }
    // and the fixture discriminates: the threshold is inside the data's range
    assert!(mask.iter().any(|&set| set), "nothing was above the level");
    assert!(mask.iter().any(|&set| !set), "nothing was below the level");

    for block in BLOCKS {
        for axes in AXES {
            let got = run(chain(), block, axes, &input);
            assert_eq!(got.dtype(), Dtype::Bool);
            assert_eq!(
                got.view::<bool>().unwrap(),
                mask,
                "block {block} on axes {axes:?}"
            );
        }
    }
}

/// The reason the narrow level is worth reaching, in bytes.
#[test]
fn the_bool_level_a_plan_can_now_write_is_an_eighth_of_the_f64_one() {
    let comparison = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let mask = Voxels::zeros(Dtype::Bool, VOLUME).unwrap();
    assert_eq!(comparison.bytes(), mask.bytes() * 8);
}

/// The op states its output element type, which is the whole of what gap 2
/// asked for — and it states its *input* type too, which is `f64` and nothing
/// else, because `WidenOp` is the way in.
#[test]
fn a_narrowing_states_its_target_and_takes_only_f64() {
    let to_mask = NarrowOp::to_mask("mask");
    assert_eq!(to_mask.to(), Dtype::Bool);
    assert_eq!(to_mask.produces(Dtype::F64), Dtype::Bool);
    assert_eq!(to_mask.narrowing(), None);
    assert_eq!(to_mask.reach(0, 64), 0);

    for target in [
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::U64,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F32,
        Dtype::F64,
    ] {
        let op = NarrowOp::new("narrow", Narrowing::to(target).unwrap());
        assert_eq!(op.produces(Dtype::F64), target, "{target:?}");
        assert!(op.accepts(Dtype::F64), "{target:?}");
        for refused in [Dtype::Bool, Dtype::U8, Dtype::U16, Dtype::F32] {
            assert!(!op.accepts(refused), "{target:?} should refuse {refused:?}");
        }
    }

    // The two element types a narrowing has nothing to say about are refused
    // where the rule lives, and the message is that rule's rather than a second
    // one written here.
    assert!(
        Narrowing::to(Dtype::Bool).is_err(),
        "a comparison, not a rounding"
    );
    assert!(Narrowing::to(Dtype::F16).is_err(), "no buffer holds it");
}

/// Narrowing then widening returns the values the narrowing kept — over a plan,
/// under every cut. What it says is that the two ops are inverses on the values
/// they agree about, which is the strongest statement available: they are not
/// inverses in general, and the fixture below is deliberately inside the range
/// where they are.
#[test]
fn a_widening_after_a_narrowing_returns_what_the_narrowing_kept() {
    let mut input = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = (flat % 200) as f64;
    }
    let chain = || {
        Chain::sequence(vec![
            Chain::op(NarrowOp::new(
                "narrow",
                Narrowing::new(Dtype::U8, Rounding::ToNearest).unwrap(),
            )),
            Chain::op(WidenOp::new("widen")),
        ])
    };
    assert_eq!(chain().produces(Dtype::F64).unwrap(), Dtype::F64);

    let want = whole(&chain(), &input);
    let round_tripped = want.view::<f64>().unwrap();
    for (before, after) in input.iter().zip(round_tripped.iter()) {
        assert_eq!(before.to_bits(), after.to_bits(), "at {before}");
    }
    // and not vacuous: something in there needed more than one bit
    assert!(input.iter().any(|&value| value > 1.0));

    for block in BLOCKS {
        for axes in AXES {
            let got = run(chain(), block, axes, &input);
            for (a, b) in got.view::<f64>().unwrap().iter().zip(round_tripped.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "block {block} on axes {axes:?}");
            }
        }
    }
}

/// **`VoxelwiseMapOp` is byte-unchanged.**
///
/// The same chains that existed before a width cast did, over the same volume
/// and the same cuts, compared against the definition of what each map computes.
/// A map op that had been taught to narrow would fail this at the first
/// `produces`.
#[test]
fn the_existing_voxelwise_map_still_passes_its_input_width_through_unchanged() {
    let input = intensities();
    let cases: [(&str, fn() -> Chain, fn(f64) -> f64); 4] = [
        (
            "threshold",
            || Chain::op(VoxelwiseMapOp::threshold("step", LEVEL, 1.0, 0.0)),
            |value| if value > LEVEL { 1.0 } else { 0.0 },
        ),
        (
            "at_or_above",
            || Chain::op(VoxelwiseMapOp::at_or_above("step", LEVEL, 1.0, 0.0)),
            |value| if value >= LEVEL { 1.0 } else { 0.0 },
        ),
        (
            "identity",
            || Chain::op(VoxelwiseMapOp::identity("identity")),
            |value| value,
        ),
        (
            "not",
            || Chain::op(VoxelwiseMapOp::not("complement")),
            |value| if value != 0.0 { 0.0 } else { 1.0 },
        ),
    ];
    for (what, chain, want) in cases {
        assert_eq!(
            chain().produces(Dtype::F64).unwrap(),
            Dtype::F64,
            "{what}: the map op still passes its input width through"
        );
        let reference = whole(&chain(), &input);
        let reference = reference.view::<f64>().unwrap();
        for (value, got) in input.iter().zip(reference.iter()) {
            assert_eq!(got.to_bits(), want(*value).to_bits(), "{what} at {value}");
        }
        for block in BLOCKS {
            for axes in AXES {
                let got = run(chain(), block, axes, &input);
                assert_eq!(got.dtype(), Dtype::F64, "{what}");
                for (a, b) in got.view::<f64>().unwrap().iter().zip(reference.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{what}: block {block} on axes {axes:?}"
                    );
                }
            }
        }
    }
}
