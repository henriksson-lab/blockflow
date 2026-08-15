// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Fusing two voxelwise maps into one op is the same computation.**
//
// `ops::voxelwise::Compose` lets a caller replace two chain slots with one, so
// that a block is read once, mapped twice in registers, and written once —
// rather than read, mapped, written, read, mapped, written, with a whole
// intermediate level in between. The saving is real and the planner can see it
// in `cost_per_voxel`. What makes the saving *legitimate* is that the two
// arrangements produce the same numbers, and that is what this file asserts.
//
// Byte for byte, not approximately. A fused map and a two-phase chain that
// agreed only to a tolerance would be two different answers, and the choice
// between them would be a numerical decision disguised as a cost decision.
// `f64` equality is compared through `to_bits` so that `-0.0` and `0.0` count as
// different, since a sign of zero is exactly the sort of thing a rearranged
// expression loses.
//
// Over a real volume and several decompositions, because a voxelwise op has
// reach 0 and therefore nothing to get wrong at a block seam *in principle* —
// which is the kind of statement that stops being true when somebody adds a
// fast path keyed on the buffer being whole. The fused op takes a contiguous
// slice path when it can; a block is a different shape from a volume; so the
// grids here are not decoration.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::voxelwise::{Compose, Identity, Not, Threshold};
use blockflow::ops::VoxelwiseMapOp;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [32, 24, 20];

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

/// One phase per slot, so a two-slot chain really does materialise its
/// intermediate and a one-slot chain really does not. That difference is the
/// thing being justified, so the plan must not quietly erase it.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    let slots = workflow.chain.slots();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let reach = workflow.chain.reach3(&VOLUME);
    let phases = slots
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
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases,
        chain_reach: reach,
    }
}

fn run(chain: Chain, block: usize, split_axes: &[usize], input: &Array3<f64>) -> Array3<f64> {
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let decomposition = plan(&workflow, block, split_axes);
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    execute("fusion", &workflow, &decomposition, &Hints::default(), &env).unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

fn whole(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

#[track_caller]
fn identical(left: &Array3<f64>, right: &Array3<f64>, what: &str) {
    assert_eq!(left.shape(), right.shape(), "{what}: shapes");
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: voxel {index} is {a} fused and {b} in two phases"
        );
    }
}

/// The two phases, as a chain, and the one fused phase that replaces them.
fn split() -> Chain {
    Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::threshold("step", 0.30, 1.0, 0.0)),
        Chain::op(VoxelwiseMapOp::not("complement")),
    ])
}

fn fused() -> Chain {
    Chain::op(VoxelwiseMapOp::from_map(
        "fused",
        Compose::new(Threshold::above(0.30, 1.0, 0.0), Not),
    ))
}

/// The equality that justifies fusing at all, over a real volume and eight
/// decompositions of it.
#[test]
fn a_fused_map_equals_the_two_phases_it_replaces_under_every_decomposition() {
    let input = intensities();
    let want = whole(&split(), &input);
    identical(&whole(&fused(), &input), &want, "whole volume");

    for block in [4usize, 7, 13, 32] {
        for split_axes in [&[0usize][..], &[0, 2][..], &[0, 1, 2][..]] {
            let two = run(split(), block, split_axes, &input);
            let one = run(fused(), block, split_axes, &input);
            identical(
                &two,
                &want,
                &format!("two phases, block {block} on {split_axes:?}"),
            );
            identical(
                &one,
                &want,
                &format!("fused, block {block} on {split_axes:?}"),
            );
        }
    }
}

/// The fused chain is **one** slot where the split chain is two, which is the
/// whole point: one phase, one level, one pass. Asserted rather than assumed,
/// because if the plan gave them the same shape the test above would be
/// comparing two identical arrangements.
#[test]
fn fusing_removes_a_slot_and_the_level_that_goes_with_it() {
    assert_eq!(split().slots().len(), 2);
    assert_eq!(fused().slots().len(), 1);

    let two = Workflow::new(split(), VOLUME, Dtype::F64);
    let one = Workflow::new(fused(), VOLUME, Dtype::F64);
    assert_eq!(plan(&two, 8, &[0, 1, 2]).n_phases(), 2);
    assert_eq!(plan(&one, 8, &[0, 1, 2]).n_phases(), 1);
}

/// Three decompositions of the *same* map into (first, then), all of which must
/// give the identical volume — including the ones with an identity in them,
/// which is the decomposition a caller writes when a stage is switched off.
#[test]
fn every_way_of_splitting_one_map_gives_the_same_volume() {
    let input = intensities();
    let level = 0.30;
    let want = whole(
        &Chain::op(VoxelwiseMapOp::threshold("step", level, 1.0, 0.0)),
        &input,
    );

    let arrangements: Vec<(&str, Chain)> = vec![
        (
            "identity then threshold",
            Chain::op(VoxelwiseMapOp::from_map(
                "fused",
                Compose::new(Identity, Threshold::above(level, 1.0, 0.0)),
            )),
        ),
        (
            "threshold then identity",
            Chain::op(VoxelwiseMapOp::from_map(
                "fused",
                Compose::new(Threshold::above(level, 1.0, 0.0), Identity),
            )),
        ),
        (
            "as two phases with an identity between",
            Chain::sequence(vec![
                Chain::op(VoxelwiseMapOp::identity("id")),
                Chain::op(VoxelwiseMapOp::threshold("step", level, 1.0, 0.0)),
            ]),
        ),
    ];

    for (what, chain) in arrangements {
        identical(&whole(&chain, &input), &want, what);
        identical(&run(chain, 7, &[0, 1, 2], &input), &want, what);
    }
}
