// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// One input, several outputs — each with its own element type and its own rank.
//
// Why this file exists, in one number
// -----------------------------------
// An operation that produces more than one result had nowhere to put the extra
// ones. A `Workflow` names a single output of a single element type and the
// executor writes one buffer to one image, so the rest were written on the side
// by whatever happened to be holding the storage, and nothing in the framework
// knew they existed. Measured on the patch-lattice harness: the run wrote
// **158.6 MB** and the framework counted **95.2 MB**, short by a factor of
// **1.67**. A plan is checked against the framework's figure, so that shortfall
// is not a reporting nicety — it is the planner being lied to.
//
// What is asserted here, and what each part catches
// -------------------------------------------------
// 1. **The accounting reconciles**, on exactly the shape that was short: the
//    same 121 blocks, the same two extra arrays at a third of an element per
//    output position, the same dtypes. Two independently-derived numbers — what
//    the environment counted, and the size of the arrays that were declared —
//    must agree exactly, and their sum must be the 158.6 MB the run really
//    moves.
// 2. **Decomposition invariance**, which is this crate's standing bar and the
//    only thing that catches a side output positioned from the block rather than
//    from the volume. Byte-identical arrays under eight decompositions, against a
//    whole-volume reference computed by the same code.
// 3. **The coverage guard seen to fire.** `BlockOp::side_region` defaults to the
//    block's valid region, which is safe only because a wrong mapping is caught:
//    a mapping that overlaps, that leaves a hole, or that is of the wrong rank
//    must fail the run.
// 4. **The halo guard still fires** on a phase that has side outputs, because
//    that phase reaches the guard through code the halo tests do not cover.
// 5. **A phase with a side output is never short-circuited**, because the
//    algebra that licenses the skip is about the primary result only.

use std::sync::Arc;

use ndarray::{Array3, ArrayD, IxDyn};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::{AccountingEnvironment, ArrayEnvironment};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, Output, SideBlock, SourceInputs};
use blockflow::region::Region;
use blockflow::strategy::{execute, execute_observed, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::{AffineOp, Dtype, Event, OrderLog, SideOutputOp};

// --------------------------------------------------------------- helpers --

fn one_phase(
    workflow: &Workflow,
    volume: [usize; 3],
    grid: BlockGrid,
    reach: [usize; 3],
    halo: [usize; 3],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names = slots.iter().map(|slot| slot.display_name()).collect();
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names,
            reach,
            halo,
            grid,
        )],
        chain_reach: reach,
    }
}

// ------------------------------------------------- the accounting itself --

/// The measured shortfall, reproduced and closed.
///
/// The geometry is the harness's: a `[11, 11, payload]` array in patch-index
/// space, one block per patch, 121 blocks, `f32`. The two extra arrays carry one
/// element per output position rather than one per input voxel — a third, since
/// the payload folds three classes — one `int32` and one `float32`.
///
/// Before, the framework's figure was the primary alone: 95 158 272 bytes. The
/// run wrote 158 597 120. Both numbers are pinned here, and so is the identity
/// between what the environment counted and what the declarations imply, which
/// is the part that makes this a reconciliation rather than a restatement.
#[test]
fn the_byte_accounting_reconciles_with_what_the_environment_wrote() {
    const CLASSES: usize = 3;
    const PAYLOAD: usize = 196_608;
    let volume = [11, 11, PAYLOAD];

    let chain = Chain::op(
        SideOutputOp::new("gather", [0, 0, 0])
            .with_side("label", Dtype::I32, 0, CLASSES)
            .with_side("score", Dtype::F32, 0, CLASSES),
    );
    let workflow = Workflow::new(chain, volume, Dtype::F32);

    // Every array this workflow writes, primary first. Derived off the chain, so
    // it cannot disagree with what the executor will declare.
    let outputs = workflow.outputs();
    assert_eq!(
        outputs
            .iter()
            .map(|output| (output.name.as_str(), output.dtype, output.shape.clone()))
            .collect::<Vec<_>>(),
        vec![
            ("output", Dtype::F32, vec![11, 11, PAYLOAD]),
            ("gather.label", Dtype::I32, vec![11, 11, PAYLOAD / CLASSES]),
            ("gather.score", Dtype::F32, vec![11, 11, PAYLOAD / CLASSES]),
        ]
    );

    let grid = BlockGrid::new(volume, [1, 1, PAYLOAD]).unwrap();
    let decomposition = one_phase(&workflow, volume, grid, [0, 0, 0], [0, 0, 0]);
    assert_eq!(decomposition.n_tasks(), 121);

    let env = AccountingEnvironment::new(volume, [1, 1, PAYLOAD], Dtype::F32.size_of() as u64);
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    // What the framework used to count: the one output the workflow names.
    let primary_bytes = stats.write_voxels * Dtype::F32.size_of() as u64;
    assert_eq!(primary_bytes, 95_158_272);

    // What the environment counted for the rest.
    assert_eq!(stats.side_writes, 121 * 2);
    assert_eq!(stats.side_bytes_written, 63_438_848);

    // The independent derivation: the arrays that were declared, times their
    // element sizes. The regions tile each array exactly — the guard below says
    // so — therefore the two figures must be equal, and any disagreement is
    // either a write nobody counted or a count for a write nobody made.
    let declared: u64 = outputs.iter().skip(1).map(Output::bytes).sum();
    assert_eq!(declared, stats.side_bytes_written);

    let real = primary_bytes + stats.side_bytes_written;
    assert_eq!(real, 158_597_120);
    assert_eq!(
        (real as f64 / primary_bytes as f64 * 100.0).round(),
        167.0,
        "the ratio this test exists to close"
    );
}

// ------------------------------------------------ decomposition invariance --

const VOLUME: [usize; 3] = [12, 6, 6];

fn ramp() -> Voxels {
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = (flat % 37) as f64 + 0.5;
    }
    array.into()
}

/// A chain whose op writes two arrays beside its result: one of rank 4 with a
/// channel axis, one of rank 3 at a third of the last axis.
fn side_chain() -> Chain {
    Chain::op(
        SideOutputOp::new("pass", [1, 1, 0])
            .with_side("channels", Dtype::I32, 2, 1)
            .with_side("folded", Dtype::F32, 0, 3),
    )
}

/// The oracle: the same code, called once over the whole array.
fn reference(chain: &Chain, input: &Voxels) -> Vec<ArrayD<f64>> {
    let at = Anchor::whole(VOLUME);
    let mut primary = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain.apply(input, &mut primary, &at).unwrap();
    let whole = Region::whole(&VOLUME);
    let regions: Vec<Region> = chain
        .side_outputs(VOLUME)
        .iter()
        .enumerate()
        .map(|(which, _)| chain.side_region(which, &whole, VOLUME).unwrap())
        .collect();
    let within = Region::whole(&VOLUME);
    chain
        .apply_side(
            input,
            SourceInputs::none(),
            &primary,
            &SideBlock {
                at: &at,
                within: &within,
                regions: &regions,
            },
        )
        .unwrap()
}

/// Eight decompositions, one answer.
///
/// The bar `docs/design/BLOCK_OPS.md` sets everywhere else, applied to the
/// arrays that were previously outside the framework: a side output positioned
/// from the block rather than from the volume agrees with the reference at one
/// block size and disagrees at the next, which is exactly what running eight
/// catches and running one does not. The op reaches one voxel on two axes, so
/// the halo is real and the trustworthy sub-box is not the whole buffer.
#[test]
fn every_decomposition_writes_the_same_side_outputs() {
    let input = ramp();
    let chain = side_chain();
    let wanted = reference(&chain, &input);
    let names = ["pass.channels", "pass.folded"];
    assert_eq!(wanted[0].shape(), &[12, 6, 6, 2]);
    assert_eq!(wanted[1].shape(), &[12, 6, 2]);

    let cases: [(usize, &[usize]); 8] = [
        (12, &[0]),
        (6, &[0]),
        (4, &[0]),
        (3, &[0]),
        (2, &[1]),
        (3, &[1]),
        (4, &[0, 1]),
        (2, &[0, 1]),
    ];
    for (edge, axes) in cases {
        let workflow = Workflow::new(side_chain(), VOLUME, Dtype::F64);
        let reach = workflow.chain.reach3(&VOLUME);
        let grid = BlockGrid::along(VOLUME, axes, edge).unwrap();
        let decomposition = one_phase(&workflow, VOLUME, grid, reach, reach);
        let env = ArrayEnvironment::new(input.clone(), 1, [2, 2, 2]).unwrap();
        execute("t", &workflow, &decomposition, &Hints::default(), &env).unwrap();

        assert_eq!(
            env.output(),
            input,
            "the primary result at edge {edge} over {axes:?}"
        );
        assert_eq!(env.side_output_names(), names);
        for (name, want) in names.iter().zip(&wanted) {
            assert_eq!(
                &env.side_output(name).unwrap(),
                want,
                "side output {name:?} at edge {edge} over {axes:?}"
            );
        }
    }
}

// ------------------------------------------------- the guards, seen firing --

/// A side output whose per-block regions do not tile it fails the run.
///
/// This is what makes the default mapping — "the block's valid region" — safe to
/// have: it is checked, not assumed. Three ways to get it wrong are provoked
/// here, and each must be refused rather than silently half-filling an array.
struct MisplacingOp {
    /// How the mapping is wrong: 0 overlaps, 1 leaves a hole, 2 is of the wrong
    /// rank.
    fault: usize,
}

impl BlockOp for MisplacingOp {
    fn name(&self) -> &'static str {
        "misplacing"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> blockflow::Result<()> {
        out.assign(input)
    }

    fn side_outputs(&self, volume: [usize; 3]) -> Vec<Output> {
        vec![Output::new("misplaced", Dtype::F64, &volume)]
    }

    fn side_region(
        &self,
        _which: usize,
        valid: &Region,
        volume: [usize; 3],
    ) -> blockflow::Result<Region> {
        Ok(match self.fault {
            // every block writes the same corner: overlapping
            0 => Region::new(&[0, 0, 0], &[1, volume[1], volume[2]]),
            // every block writes one plane short: a hole
            1 => Region::new(
                &valid.start.clone(),
                &[
                    valid.shape[0].saturating_sub(1),
                    valid.shape[1],
                    valid.shape[2],
                ],
            ),
            // rank 2 against a rank-3 array
            _ => Region::new(&[valid.start[0], 0], &[valid.shape[0], volume[1]]),
        })
    }

    fn apply_side(
        &self,
        _input: &Voxels,
        _sources: SourceInputs<'_>,
        _primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> blockflow::Result<Vec<ArrayD<f64>>> {
        Ok(vec![ArrayD::zeros(IxDyn(&block.regions[0].shape))])
    }
}

#[test]
fn a_side_output_whose_regions_do_not_tile_it_is_refused() {
    // Two of the three are caught by the coverage guard after the phase; the
    // third — a mapping of the wrong rank — never gets that far, because a
    // rank-2 region of a rank-3 array is refused by the write itself. Both are
    // refusals, and the earlier one is the better message.
    for (fault, expected) in [
        (0usize, "overlap"),
        (1, "do not tile the volume exactly"),
        (2, "shape mismatch"),
    ] {
        let workflow = Workflow::new(Chain::op(MisplacingOp { fault }), VOLUME, Dtype::F64);
        let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
        let decomposition = one_phase(&workflow, VOLUME, grid, [0, 0, 0], [0, 0, 0]);
        // The plan itself is fine: the guard is on what was *written*, in the
        // side output's own space, which no part of the plan describes.
        decomposition.check().unwrap();
        let env = ArrayEnvironment::new(ramp(), 1, [2, 2, 2]).unwrap();
        let message = execute("t", &workflow, &decomposition, &Hints::default(), &env)
            .unwrap_err()
            .to_string();
        assert!(message.contains(expected), "fault {fault} gave: {message}");
        assert!(
            fault == 2 || message.contains("side output \"misplaced\""),
            "fault {fault} gave: {message}"
        );
    }
}

/// The halo guard, on a phase that also has side outputs.
///
/// It is reached through code the halo tests do not exercise — the side-output
/// production sits between the op and the write — so it is provoked here rather
/// than assumed to have come along.
#[test]
fn the_halo_guard_still_fires_on_a_phase_with_side_outputs() {
    let workflow = Workflow::new(side_chain(), VOLUME, Dtype::F64);
    let reach = workflow.chain.reach3(&VOLUME);
    assert_eq!(reach, [1, 1, 0]);
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let good = one_phase(&workflow, VOLUME, grid, reach, reach);
    good.check().unwrap();

    let short = good.with_forced_halo([0, 0, 0]);
    assert_eq!(short.phases[0].reach, reach, "the reach is untouched");
    let message = short.check().unwrap_err().to_string();
    assert!(
        message.contains("do not tile the volume exactly") && message.contains("halo [0, 0, 0]"),
        "{message}"
    );
    let env = ArrayEnvironment::new(ramp(), 1, [2, 2, 2]).unwrap();
    assert!(execute("t", &workflow, &short, &Hints::default(), &env).is_err());
}

/// The short circuit is licensed by an algebra over the primary result, so a
/// phase with a side output does not get it.
///
/// Without this, a uniform block would be skipped, its side outputs never
/// written, and the coverage guard would report a hole — a true statement about
/// the wrong cause.
#[test]
fn a_phase_with_a_side_output_is_never_short_circuited() {
    let uniform: Voxels = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 4.0).into();

    // The same chain without the side outputs *is* short-circuited, so the
    // assertion below is about the side outputs and not about the data.
    let plain = Workflow::new(
        Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    );
    assert!(short_circuits(&plain, &uniform) > 0);

    let with_sides = Workflow::new(
        Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
            Chain::op(SideOutputOp::new("pass", [0, 0, 0]).with_side("folded", Dtype::F32, 0, 3)),
        ]),
        VOLUME,
        Dtype::F64,
    );
    assert_eq!(short_circuits(&with_sides, &uniform), 0);
}

fn short_circuits(workflow: &Workflow, input: &Voxels) -> usize {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let decomposition = one_phase(workflow, VOLUME, grid, [0, 0, 0], [0, 0, 0]);
    let env = ArrayEnvironment::new(input.clone(), 1, [2, 2, 2]).unwrap();
    let log = Arc::new(OrderLog::new());
    let stats = execute_observed(
        "t",
        workflow,
        &decomposition,
        &Hints::default(),
        &env,
        &[log.clone() as Arc<dyn blockflow::EventListener>],
    )
    .unwrap();
    stats
        .log
        .events()
        .into_iter()
        .filter(|event| matches!(event, Event::BlockShortCircuited { .. }))
        .count()
}

/// A side output *is* a resize, and it is the one that works today.
///
/// `docs/design/BLOCK_OPS.md` records the wall as "a cross-grid fetch may
/// translate but not resize", and this is the half of it that these outputs
/// close: an array a third the size of the volume, and one of rank 4, both
/// written per block, both in the plan's accounting and both under the coverage
/// guard. What is still not resizable is the **image** — `BlockOp::apply` writes
/// an output the shape of its input, so the buffer that goes back to image `p+1`
/// is the shape of the one that came off image `p`, and that is what the
/// executor still refuses by name.
#[test]
fn a_side_output_may_be_a_different_size_and_a_different_rank_from_the_image() {
    let input = ramp();
    let chain = side_chain();
    let produced = reference(&chain, &input);
    assert_eq!(produced[0].ndim(), 4, "a rank the volume does not have");
    assert_eq!(
        produced[1].len() * 3,
        input.len(),
        "a third of the elements"
    );

    // and the image itself still may not resize, said plainly
    let workflow = Workflow::new(side_chain(), VOLUME, Dtype::F64);
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let mut decomposition = one_phase(&workflow, VOLUME, grid, [0, 0, 0], [0, 0, 0]);
    decomposition.phases[0] = decomposition.phases[0]
        .clone()
        .with_sources(|block| Region::new(&block.read.start.clone(), &[1, 1, 1]));
    let env = ArrayEnvironment::new(input, 1, [2, 2, 2]).unwrap();
    let message = execute("t", &workflow, &decomposition, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(message.contains("has nowhere to land"), "{message}");
}
