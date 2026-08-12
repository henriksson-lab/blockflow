// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The element type is a tag and a level is rank 3. This file is the acceptance
// suite for what that bought and for what it had to be refused to buy it.
//
// Five things, and each catches something the others cannot
// ---------------------------------------------------------
// 1. **The tax, measured.** The same chain over the same extent, once with the
//    level held as `f64` and once as `bool`, with the byte figure taken from the
//    environment's own counter. Not an argument about widths — the run's own
//    number, and the answer must be exactly 8x.
// 2. **The narrow run is still the right run.** A `bool` workload agrees with
//    its whole-volume reference under several decompositions, which is the bar
//    `docs/design/BLOCK_OPS.md` sets everywhere: a run that agrees at one block
//    size is how a short halo passes.
// 3. **An op refuses a type at plan time.** `decompose` fails, naming the op and
//    the type, before any block exists. A refusal that arrives when a block
//    reaches the op is the failure this replaces.
// 4. **A phase may change the element type**, and the level is allocated at what
//    it writes rather than at what the workflow declared.
// 5. **A phase may change the shape**, which is the wall `apply` declaring its
//    output shape closed — and the guard that used to refuse every such plan is
//    provoked here to show it still fires when the declaration and the plan
//    disagree.

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::env::Environment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::{ElementShape, Morphology, MorphologyOp, StructuringElement, VoxelwiseMapOp};
use blockflow::region::Region;
use blockflow::strategy::{execute, Hints, Strategy, Trivial, Workflow};
use blockflow::voxels::Voxels;
use blockflow::{DecimateOp, Dtype, IdentityOp, NonZeroOp};
use ndarray::Array3;

const VOLUME: [usize; 3] = [24, 12, 12];

/// A mask with structure at two scales, so an opening has something to remove
/// and a seam has something to get wrong.
fn speckle() -> Array3<bool> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        (i * 31 + j * 17 + k * 7) % 5 < 2 || (i / 4 + j / 4 + k / 4) % 3 == 0
    })
}

fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(MorphologyOp::new("open", Morphology::Open, element())),
        Chain::op(MorphologyOp::new("dilate", Morphology::Dilate, element())),
    ])
}

/// One phase holding the whole chain, at a given block edge and split axes,
/// with the halo derived from the chain's own reach and from nothing else.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach3(&VOLUME);
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names,
            reach,
            reach,
            grid,
        )],
        chain_reach: reach,
    };
    plan.declare_dtypes(&workflow.chain).unwrap();
    plan
}

// -------------------------------------------------------- 1. the measurement --

/// The 8x, taken from the environment's counters rather than argued from the
/// widths.
///
/// Both runs are the *same chain over the same extent with the same
/// decomposition*; the only difference is what the level holds. Before the
/// element type became a tag the `bool` column was not expressible at all — a
/// `BlockBuf` was an `ArrayD<f64>` — so the `f64` figure is what this workload
/// cost and the `bool` figure is what it costs now.
#[test]
fn the_same_workload_moves_an_eighth_of_the_bytes_as_bool() {
    let mask = speckle();

    let wide_workflow = Workflow::new(chain(), VOLUME, Dtype::F64);
    let wide_plan = plan(&wide_workflow, 8, &[0]);
    let wide_input: Voxels = mask.mapv(|value| if value { 1.0 } else { 0.0 }).into();
    let wide = ArrayEnvironment::new(wide_input, 1, [4, 4, 4]).unwrap();
    execute("wide", &wide_workflow, &wide_plan, &Hints::default(), &wide).unwrap();

    let narrow_workflow = Workflow::new(chain(), VOLUME, Dtype::Bool);
    let narrow_plan = plan(&narrow_workflow, 8, &[0]);
    let narrow = ArrayEnvironment::new(mask.clone().into(), 1, [4, 4, 4]).unwrap();
    execute(
        "narrow",
        &narrow_workflow,
        &narrow_plan,
        &Hints::default(),
        &narrow,
    )
    .unwrap();

    // The two runs did identical work: same blocks, same voxels. The
    // *fingerprints* differ, and must — the element type is parity-visible, so
    // "the same plan in a different type" is a different plan and says so.
    assert_ne!(wide_plan.fingerprint(), narrow_plan.fingerprint());
    assert_eq!(
        format!("{:?}", wide_plan.phases[0].blocks),
        format!("{:?}", narrow_plan.phases[0].blocks),
        "the two runs must cut the volume identically, or the byte ratio is a \
         comparison of two different decompositions"
    );
    assert_eq!(
        wide_plan.exact_read_voxels(),
        narrow_plan.exact_read_voxels()
    );
    assert_eq!(
        wide.counters().snapshot().2,
        narrow.counters().snapshot().2,
        "the two runs must read the same voxels, or the byte ratio means nothing"
    );

    let (wide_read, wide_written) = wide.counters().byte_snapshot();
    let (narrow_read, narrow_written) = narrow.counters().byte_snapshot();
    assert_eq!(wide_read, narrow_read * 8);
    assert_eq!(wide_written, narrow_written * 8);
    // 331 776 against 41 472 for this extent, which is the ratio the 9.5 GB of
    // read buffers at 40 threads scales from.
    assert_eq!(narrow_read, wide_read / 8);
    assert_eq!(
        wide.counters()
            .peak_resident_bytes
            .load(std::sync::atomic::Ordering::SeqCst),
        narrow
            .counters()
            .peak_resident_bytes
            .load(std::sync::atomic::Ordering::SeqCst)
            * 8,
        "residency is the figure a budget is stated in, so it must move too"
    );
}

// ------------------------------------------- 2. the narrow run is still right --

/// Every decomposition of the `bool` chain reproduces the whole-volume answer,
/// bit for bit.
#[test]
fn a_bool_workload_agrees_with_its_whole_volume_reference_under_every_decomposition() {
    let mask = speckle();
    let source: Voxels = mask.clone().into();
    let chain = chain();

    let mut want = Voxels::zeros(Dtype::Bool, VOLUME).unwrap();
    chain
        .apply(&source, &mut want, &Anchor::whole(VOLUME))
        .unwrap();
    let want = want.view::<bool>().unwrap().to_owned();
    assert!(
        want.iter().any(|&value| value) && want.iter().any(|&value| !value),
        "the reference is constant, so it discriminates nothing"
    );

    let workflow = Workflow::new(self::chain(), VOLUME, Dtype::Bool);
    for block in [4usize, 5, 8, 32] {
        for split_axes in [vec![0], vec![1, 2], vec![0, 1, 2]] {
            let decomposition = plan(&workflow, block, &split_axes);
            decomposition.check().unwrap();
            let env = ArrayEnvironment::new(source.clone(), 1, [4, 4, 4]).unwrap();
            execute("bool", &workflow, &decomposition, &Hints::default(), &env).unwrap();
            assert_eq!(
                env.output().view::<bool>().unwrap().to_owned(),
                want,
                "block {block}, axes {split_axes:?}"
            );
        }
    }
}

/// The halo guard still fires on a `bool` phase. It is reached through the same
/// code as the `f64` one, but the buffers are a different type on the way in, so
/// this is a guard **seen** to fire rather than one assumed to still be there.
#[test]
fn a_short_halo_is_caught_on_a_bool_phase_too() {
    let workflow = Workflow::new(chain(), VOLUME, Dtype::Bool);
    let honest = plan(&workflow, 8, &[0]);
    honest.check().unwrap();

    let forced = honest.with_forced_halo([1, 0, 0]);
    let message = forced.check().unwrap_err().to_string();
    assert!(
        message.contains("do not tile the volume exactly"),
        "{message}"
    );

    let env = ArrayEnvironment::new(speckle().into(), 1, [4, 4, 4]).unwrap();
    let message = execute("short", &workflow, &forced, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("do not tile the volume exactly"),
        "{message}"
    );
}

// ------------------------------------------------- 3. refused at plan time --

/// An op that cannot work in the type it would be handed stops the *plan*.
///
/// `VoxelwiseMapOp` holds an `f64 -> f64` closure the caller supplied, so it
/// accepts `f64` and nothing else. Over a `bool` workflow every shipped strategy
/// must refuse, by name, with no environment in sight and no block created.
#[test]
fn an_op_that_cannot_take_the_element_type_is_refused_when_the_plan_is_made() {
    let workflow = Workflow::new(
        Chain::op(VoxelwiseMapOp::threshold("threshold", 0.5, 1.0, 0.0)),
        VOLUME,
        Dtype::Bool,
    );
    let message = Trivial
        .decompose(&workflow, &Default::default())
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("threshold") && message.contains("bool"),
        "{message}"
    );
}

/// And a plan built by hand that slipped past the planner is refused by the
/// executor, because a plan may arrive from anywhere.
///
/// The same arrangement `check_block_constraints` uses, and for the same reason:
/// a `Decomposition` records op *names*, so nothing but the executor holds both
/// the plan and the implementations.
#[test]
fn a_plan_whose_level_is_the_wrong_width_is_refused_by_the_executor() {
    let workflow = Workflow::new(
        Chain::op(NonZeroOp::new("set", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    );
    let mut hand_built = plan(&workflow, 8, &[0]);
    // Undo what `declare_dtypes` recorded: a plan that says its level is `f64`
    // while its op writes `bool` is precisely the plan whose level is the wrong
    // width, and it is the one a wire or an older planner could hand over.
    hand_built.phases[0].dtype = None;
    hand_built.check().unwrap();

    let env =
        ArrayEnvironment::new(Voxels::zeros(Dtype::F64, VOLUME).unwrap(), 1, [4, 4, 4]).unwrap();
    let message = execute("mistyped", &workflow, &hand_built, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("its ops write bool") && message.contains("allocates level 1 as float64"),
        "{message}"
    );
}

// ------------------------------------------- 4. a phase changes the type --

/// A workflow that reads `f64` and writes `bool`: the plan says so, the level is
/// allocated one byte wide, and the answer is the predicate.
#[test]
fn a_phase_may_narrow_the_element_type_and_the_level_follows_the_plan() {
    let workflow = Workflow::new(
        Chain::op(NonZeroOp::new("set", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    );
    let decomposition = Trivial.decompose(&workflow, &Default::default()).unwrap();
    assert_eq!(decomposition.dtype_at(0), Dtype::F64);
    assert_eq!(decomposition.dtype_at(1), Dtype::Bool);
    assert_eq!(decomposition.uniform_dtype(), None);

    let mut input = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = (flat % 3) as f64;
    }
    let env = ArrayEnvironment::for_decomposition(input.clone().into(), &decomposition, [4, 4, 4])
        .unwrap();
    execute(
        "narrowing",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    assert_eq!(env.level_dtype(1), Dtype::Bool);
    let out = env.output();
    assert_eq!(out.bytes(), VOLUME.iter().product::<usize>() as u64);
    let out = out.view::<bool>().unwrap().to_owned();
    assert_eq!(out, input.mapv(|value| value != 0.0));

    // `ArrayEnvironment::new` gives every level the input's type, which cannot
    // host this plan — and says so rather than converting at the write.
    let flat = ArrayEnvironment::new(input.into(), 1, [4, 4, 4]).unwrap();
    let message = execute("flat", &workflow, &decomposition, &Hints::default(), &flat)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("holds level 1 as float64") && message.contains("as bool"),
        "{message}"
    );
}

// ------------------------------------------ 5. a phase changes the shape --

/// The output volume of a decimating phase, in the decomposition.
///
/// `read` is the phase's own (output) extent; `source` is the extent in the
/// level below that the op must be handed to produce it. Stating the second is
/// what change 2 added; declaring that the op turns one into the other is what
/// change 5 added, and the two together are what make this plan runnable.
fn decimating_plan(workflow: &Workflow, factor: usize, block: usize) -> Decomposition {
    let out_volume = [VOLUME[0] / factor, VOLUME[1], VOLUME[2]];
    let phase = PhaseDecomposition::derive(
        vec![0],
        vec![workflow.chain.display_name()],
        [0, 0, 0],
        [0, 0, 0],
        BlockGrid::along(out_volume, &[0], block).unwrap(),
    )
    .with_sources(|block| {
        Region::new(
            &[block.read.start[0] * factor, 0, 0],
            &[block.read.shape[0] * factor, VOLUME[1], VOLUME[2]],
        )
    });
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    }
}

/// A level that is **half the size of the one below it**, checked and run.
///
/// This is the wall `strategy.rs` used to refuse outright: `apply` wrote an
/// output the shape of its input, so a cross-grid fetch could translate but
/// never resize, and the refusal named what was missing. What was missing was an
/// op saying what shape it produces.
#[test]
fn a_phase_may_resize_the_level_when_its_op_declares_the_shape() {
    for factor in [2usize, 3] {
        let workflow = Workflow::new(
            Chain::op(DecimateOp::new("decimate", factor)),
            VOLUME,
            Dtype::F64,
        );
        let mut input = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
        for (flat, value) in input.iter_mut().enumerate() {
            *value = (flat % 101) as f64 + 1.0;
        }

        let want =
            Array3::from_shape_fn((VOLUME[0] / factor, VOLUME[1], VOLUME[2]), |(i, j, k)| {
                input[[i * factor, j, k]]
            });

        for block in [2usize, 4, 12] {
            let decomposition = decimating_plan(&workflow, factor, block);
            decomposition.check().unwrap();
            assert_eq!(
                decomposition.output_volume(),
                [VOLUME[0] / factor, VOLUME[1], VOLUME[2]]
            );
            assert_eq!(decomposition.uniform_volume(), None);

            let env = ArrayEnvironment::for_decomposition(
                input.clone().into(),
                &decomposition,
                [2, 2, 2],
            )
            .unwrap();
            execute(
                "decimate",
                &workflow,
                &decomposition,
                &Hints::default(),
                &env,
            )
            .unwrap();
            assert_eq!(
                env.output().view::<f64>().unwrap().to_owned(),
                want,
                "factor {factor}, block {block}"
            );
        }
    }
}

/// The same plan with an op that does **not** resize: the executor names the two
/// extents that disagree instead of producing a volume that is off by the
/// difference everywhere.
///
/// This is the guard the old blanket refusal became. It has to be provoked to be
/// known to work, and provoking it is exactly the mistake it exists to catch —
/// a plan whose grid says one thing and whose op says another.
#[test]
fn a_phase_whose_op_does_not_resize_is_refused_and_names_both_extents() {
    let workflow = Workflow::new(
        Chain::op(IdentityOp::new("identity", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    );
    let decomposition = decimating_plan(&workflow, 2, 4);
    // The plan itself is fine: every fetch lies inside the level it reads, which
    // is all a plan can say. What it cannot say is what the op will do with it.
    decomposition.check().unwrap();

    let env = ArrayEnvironment::for_decomposition(
        Voxels::zeros(Dtype::F64, VOLUME).unwrap(),
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let message = execute(
        "mismatch",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("has nowhere to land")
            && message.contains("its ops turn that into [8, 12, 12]")
            && message.contains("read extent of [4, 12, 12]"),
        "{message}"
    );
}
