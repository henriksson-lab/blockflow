// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A level is written by one phase, read by exactly one phase, and then dead.
// Until now nothing acted on that: every level of an `N`-phase plan was
// allocated at full volume and held for the whole run, so a twenty-stage chain
// kept twenty-one copies of the data resident while at most two of them were
// live. This suite is the evidence that it no longer does, and the evidence that
// freeing them changed no answer.
//
// Three properties, and the third is the one that makes the first two worth
// having:
//
// 1. **The saving is real and is a number.** Residency after a long chain is
//    counted, not described.
// 2. **The output is unchanged.** Byte-identical to the same chain run with
//    every level pinned, which is the old behaviour exactly.
// 3. **Reading a freed level fails.** A level that came back as zeros would be
//    the same class of defect as an unwritten level reading as zeros, which this
//    crate fills with NaN precisely so that it cannot happen quietly.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition, Visibility};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::{ElementShape, Morphology, MorphologyOp, StructuringElement, VoxelwiseMapOp};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];
const STAGES: usize = 20;

fn input() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 31 + j * 17 + k * 7) % 11) as f64
    })
}

/// A chain long enough that holding every level is obviously the wrong thing:
/// twenty stages, each a cheap voxelwise map so the test measures residency
/// rather than arithmetic.
fn long_chain() -> Chain {
    Chain::sequence(
        (0..STAGES)
            .map(|stage| {
                let bias = stage as f64;
                Chain::op(VoxelwiseMapOp::new("step", move |value| value + bias * 0.0))
            })
            .collect(),
    )
}

/// One phase per slot, so every stage really does materialise a level.
fn one_phase_per_slot(chain: &Chain) -> Decomposition {
    let slots = chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [0, 0, 0],
                [0, 0, 0],
                grid.clone(),
            )
        })
        .collect();
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [0, 0, 0],
    }
}

fn run(hints: &Hints) -> (Array3<f64>, ArrayEnvironment) {
    let chain = long_chain();
    let plan = one_phase_per_slot(&chain);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    execute("levels", &workflow, &plan, hints, &env).expect("a run");
    let out = env.output().view::<f64>().unwrap().to_owned();
    (out, env)
}

/// Everything pinned: the behaviour before this existed, kept as the reference.
fn keep_everything() -> Hints {
    Hints {
        keep_levels: (0..=STAGES).collect(),
        ..Hints::default()
    }
}

/// End of run: the input and the output, and nothing else.
///
/// Note what this is *not* measuring. Peak residency during the run is one
/// higher — while phase `p` runs it holds level `p` and level `p+1`, plus level
/// 0, which is never freed — so the bound is three levels at any moment against
/// `N + 1` before. What is asserted here is the end state, because it is the one
/// a test can read without sampling the run.
#[test]
fn a_long_chain_ends_holding_two_levels_instead_of_all_of_them() {
    let (_, env) = run(&Hints::default());
    assert_eq!(env.resident_levels(), 2, "the input and the output");

    // and the old behaviour, for the comparison to be against something real
    let (_, kept) = run(&keep_everything());
    assert_eq!(kept.resident_levels(), STAGES + 1);
}

#[test]
fn freeing_the_intermediates_changes_no_voxel() {
    let (freed, _) = run(&Hints::default());
    let (kept, _) = run(&keep_everything());
    assert_eq!(freed, kept);
    // and it is the answer, not merely two runs agreeing
    assert_eq!(freed, input());
}

/// Level 0 is somebody else's array and the output is what the run is for.
/// Neither is ever a candidate, whatever the hints say.
#[test]
fn the_input_and_the_output_are_never_freed() {
    let chain = long_chain();
    let plan = one_phase_per_slot(&chain);
    assert_eq!(plan.level_visibility(0), Visibility::Published);
    assert_eq!(
        plan.level_visibility(plan.n_levels() - 1),
        Visibility::Published
    );
    for level in 1..plan.n_levels() - 1 {
        assert_eq!(
            plan.level_visibility(level),
            Visibility::Internal,
            "level {level}"
        );
    }

    let (_, env) = run(&Hints::default());
    assert!(!env.is_discarded(0));
    assert!(!env.is_discarded(STAGES));
}

/// A freed level must be **loud**, not empty. Silence here would be the same
/// defect as an unwritten level reading as zeros.
#[test]
fn reading_a_freed_level_fails_and_says_why() {
    let (_, env) = run(&Hints::default());
    assert!(env.is_discarded(1));

    let region = blockflow::region::Region::new(&[0, 0, 0], &[4, 4, 4]);
    let message = env.read(1, &region).unwrap_err().to_string();
    assert!(message.contains("discarded"), "{message}");
    assert!(message.contains("keep_levels"), "{message}");
}

/// Pinning one level keeps exactly that one.
#[test]
fn pinning_a_level_keeps_it_and_nothing_else() {
    let hints = Hints {
        keep_levels: [7usize].into_iter().collect(),
        ..Hints::default()
    };
    let (out, env) = run(&hints);
    assert_eq!(out, input());
    assert!(!env.is_discarded(7));
    assert!(env.is_discarded(6));
    assert!(env.is_discarded(8));
    assert_eq!(
        env.resident_levels(),
        3,
        "the input, the output, and the pinned one"
    );
}

/// The saving is in *bytes*, and a chain that changes element type makes the
/// point sharper than a uniform one: an intermediate held at eight bytes a voxel
/// costs eight times a mask level, and holding twenty of them is what this
/// stopped doing.
#[test]
fn the_saving_is_proportional_to_the_chain_length() {
    let per_level = (VOLUME[0] * VOLUME[1] * VOLUME[2] * std::mem::size_of::<f64>()) as u64;
    let (_, freed) = run(&Hints::default());
    let (_, kept) = run(&keep_everything());

    let freed_bytes = freed.resident_levels() as u64 * per_level;
    let kept_bytes = kept.resident_levels() as u64 * per_level;
    assert_eq!(kept_bytes / freed_bytes, 10, "21 levels against 2");
    assert!(kept_bytes - freed_bytes > 0);
}

/// An op with a real reach, so the chain is not only voxelwise maps — the
/// lifetime rule must not depend on a phase reading exactly its own core.
#[test]
fn a_chain_with_halos_frees_its_intermediates_too() {
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let chain = Chain::sequence(
        (0..4)
            .map(|_| {
                Chain::op(MorphologyOp::new(
                    "dilate",
                    Morphology::Dilate,
                    element.clone(),
                ))
            })
            .collect(),
    );
    let slots = chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [1, 1, 1],
                [1, 1, 1],
                grid.clone(),
            )
        })
        .collect();
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [4, 4, 4],
    };
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0);
    mask[[8, 6, 5]] = 1.0;
    let source: Voxels = mask.into();

    let env = ArrayEnvironment::for_decomposition(source.clone(), &plan, [4, 4, 4]).unwrap();
    execute("halos", &workflow, &plan, &Hints::default(), &env).expect("a run");
    let freed = env.output().view::<f64>().unwrap().to_owned();
    assert_eq!(env.resident_levels(), 2);

    let kept_env = ArrayEnvironment::for_decomposition(source, &plan, [4, 4, 4]).unwrap();
    let hints = Hints {
        keep_levels: (0..=4).collect(),
        ..Hints::default()
    };
    execute("halos", &workflow, &plan, &hints, &kept_env).expect("a run");
    assert_eq!(freed, kept_env.output().view::<f64>().unwrap().to_owned());
    assert_eq!(kept_env.resident_levels(), 5);
}
