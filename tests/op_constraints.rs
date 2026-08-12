// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// An op saying what it will accept, and the framework refusing everything else.
//
// The failure this closes
// -----------------------
// `BlockOp` had no way to constrain the decomposition. With a mandated block
// shape absent from the candidate list, **both shipped strategies returned a
// decomposition that `check`s clean and is unrunnable** — the tiling held, the
// dependencies covered the reads, the fingerprint was stable, and the first
// block handed to the op was the wrong shape. A plan that passes its own guard
// and cannot run is the worst failure mode available, because every signal the
// framework has says it is fine.
//
// Two things this file is careful about
// -------------------------------------
// **The hook is consulted by `decompose` and re-checked in `execute`.** A
// constraint honoured only at planning time is not a constraint: this crate's
// whole design allows `Greedy::run` to be handed `Trivial::decompose`'s plan, or
// a plan off a wire, and both must be refused if they do not fit. Both ends are
// provoked here.
//
// **It is not only the easy case.** A mandate that *is* a block grid — one
// anisotropic extent — is the easy half, and the planners now build it directly
// rather than choosing from a list of scalar edges that cannot express it. The
// hard half is a lattice that is not a grid at all: windows spread evenly across
// an extent are overlapping and unevenly spaced, and `BlockGrid::cores` builds
// `start = index * block`. That is stated too, and what happens then is a
// **refusal** rather than a silent fallback to the easy case.

use blockflow::decomposition::{
    check_block_constraints, Constraints, Decomposition, PhaseDecomposition,
};
use blockflow::env::{AccountingEnvironment, ArrayEnvironment};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockConstraint, BlockOp, Chain};
use blockflow::strategy::{execute, Enumerating, Greedy, Hints, Strategy, Trivial};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, IdentityOp, MandatedExtentOp, SpreadLatticeOp, Workflow};

/// The patch-lattice shape, at a size a test can hold: an array in patch-index
/// space, one block per patch, and a payload axis the block must span whole.
const PATCHES: [usize; 3] = [11, 11, 12];
const ONE_PATCH: [usize; 3] = [1, 1, 12];

fn candidates_without_the_mandate() -> Constraints {
    Constraints {
        // The mandated extent is deliberately absent, and could not be present:
        // `block_candidates` is a list of scalar edges and `[1, 1, 12]` is not a
        // cube.
        block_candidates: vec![2, 4],
        split_axes: vec![0, 1],
        ..Constraints::default()
    }
}

fn patch_workflow() -> Workflow {
    Workflow::new(
        Chain::op(MandatedExtentOp::new("gather", ONE_PATCH)),
        PATCHES,
        Dtype::F32,
    )
}

// ---------------------------------------------------- consulted, at planning --

/// Both planners now produce the mandated grid, from a candidate set that does
/// not contain it and could not.
///
/// This is the exact case that used to return an unrunnable plan. The mandate
/// **replaces** the candidate list rather than filtering it, which is the only
/// thing that can work: a scalar edge cannot name an anisotropic block.
#[test]
fn both_planners_produce_the_mandated_block_shape() {
    let workflow = patch_workflow();
    let constraints = candidates_without_the_mandate();
    for strategy in [
        &Enumerating::default() as &dyn Strategy,
        &Greedy::default() as &dyn Strategy,
    ] {
        let decomposition = strategy.decompose(&workflow, &constraints).unwrap();
        assert_eq!(
            decomposition.phases[0].grid.block(),
            ONE_PATCH,
            "{}",
            strategy.name()
        );
        assert_eq!(decomposition.n_tasks(), 121, "{}", strategy.name());
        decomposition.check().unwrap();
        check_block_constraints(&workflow.chain, &decomposition).unwrap();

        // and it runs, which is the whole difference
        let env = AccountingEnvironment::new(PATCHES, ONE_PATCH, 4);
        let stats = strategy.run(&workflow, &decomposition, &env).unwrap();
        assert_eq!(stats.tasks, 121);
    }
}

/// The oracle has one plan to offer, so it refuses rather than offering it.
///
/// `Trivial` is one block spanning the volume. An op that accepts one patch is
/// told so, instead of being handed the single grid this strategy has.
#[test]
fn the_oracle_refuses_a_mandate_it_cannot_meet() {
    let message = Trivial
        .decompose(&patch_workflow(), &Constraints::default())
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("accepts exactly [1, 1, 12]") && message.contains("[11, 11, 12]"),
        "{message}"
    );
}

/// A mandate no grid over this volume satisfies is refused at planning by both
/// planners, and the refusal says which constraint did it.
///
/// `[4, 4, 12]` over `[11, 11, 12]` tiles with ragged edge blocks, so three
/// blocks in each row are handed something smaller than the mandate. That is a
/// property of the *volume*, not of the candidate list, so no amount of
/// searching finds a plan and the planner must say so.
#[test]
fn a_mandate_the_volume_cannot_satisfy_is_refused_at_planning() {
    let workflow = Workflow::new(
        Chain::op(MandatedExtentOp::new("gather", [4, 4, 12])),
        PATCHES,
        Dtype::F32,
    );
    let constraints = candidates_without_the_mandate();

    let message = Enumerating::default()
        .decompose(&workflow, &constraints)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("accepts exactly [4, 4, 12]") && message.contains("not the budget"),
        "{message}"
    );

    let message = Greedy::default()
        .decompose(&workflow, &constraints)
        .unwrap_err()
        .to_string();
    assert!(message.contains("accepts exactly [4, 4, 12]"), "{message}");
}

/// Two ops that mandate different blocks are cut into two phases rather than
/// refused.
///
/// A conflict is a fact about a *partition*, not about the chain: the same two
/// ops in two phases are fine. `Enumerating` drops the partitions that fuse them
/// and keeps searching; `Greedy` cuts on the change as it walks.
#[test]
fn two_ops_that_mandate_different_blocks_are_cut_apart() {
    let volume = [12, 12, 12];
    let build = || {
        Chain::sequence(vec![
            Chain::op(MandatedExtentOp::new("a", [1, 12, 12])),
            Chain::op(MandatedExtentOp::new("b", [12, 1, 12])),
        ])
    };
    // The fold says so directly, before any planner is involved.
    let message = build().block_constraint(volume).unwrap_err().to_string();
    assert!(message.contains("cannot be fused"), "{message}");

    for strategy in [
        &Enumerating::default() as &dyn Strategy,
        &Greedy::default() as &dyn Strategy,
    ] {
        let workflow = Workflow::new(build(), volume, Dtype::F64);
        let decomposition = strategy
            .decompose(&workflow, &candidates_without_the_mandate())
            .unwrap();
        assert_eq!(decomposition.n_phases(), 2, "{}", strategy.name());
        assert_eq!(decomposition.phases[0].grid.block(), [1, 12, 12]);
        assert_eq!(decomposition.phases[1].grid.block(), [12, 1, 12]);
        check_block_constraints(&workflow.chain, &decomposition).unwrap();
    }
}

// --------------------------------------------------- re-checked, at execution --

/// A plan that violates a mandate is refused by the executor, whatever produced
/// it.
///
/// The plan here is hand-built and passes every other guard: its valid regions
/// tile, its dependencies cover its reads, its fingerprint is stable. Only the
/// ops know it is wrong, and the executor is the first place that holds both.
#[test]
fn a_plan_that_violates_a_mandate_is_refused_at_execution() {
    let workflow = patch_workflow();
    let slots = workflow.chain.slots();
    let grid = BlockGrid::new(PATCHES, [2, 2, 12]).unwrap();
    let decomposition = Decomposition {
        volume: PATCHES,
        dtype: workflow.dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            slots.iter().map(|slot| slot.display_name()).collect(),
            [0, 0, 0],
            [0, 0, 0],
            grid,
        )],
        chain_reach: [0, 0, 0],
    };
    // Every guard the framework had before this change is satisfied by it.
    decomposition.check().unwrap();

    let env = AccountingEnvironment::new(PATCHES, ONE_PATCH, 4);
    let message = execute(
        "foreign",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("decomposition phase 0") && message.contains("accepts exactly [1, 1, 12]"),
        "{message}"
    );

    // and a *foreign* run of it is refused identically — this crate lets any
    // strategy run any decomposition, so the guard cannot live in a planner
    let message = Greedy::default()
        .run(&workflow, &decomposition, &env)
        .unwrap_err()
        .to_string();
    assert!(message.contains("accepts exactly [1, 1, 12]"), "{message}");
}

// ------------------------------------------- the lattice that is not a grid --

const SPREAD: [usize; 3] = [17, 4, 4];

/// Five overlapping windows of eight, spread evenly over an extent of
/// seventeen: `0, 2, 5, 7, 9`. Overlapping, and unevenly spaced because the step
/// is fractional.
#[test]
fn the_spread_lattice_is_overlapping_and_unevenly_spaced() {
    let op = SpreadLatticeOp::new("spread", 5, 8);
    let starts: Vec<usize> = op
        .windows(SPREAD)
        .iter()
        .map(|region| region.start[0])
        .collect();
    assert_eq!(starts, vec![0, 2, 5, 7, 9]);
    let steps: Vec<usize> = starts.windows(2).map(|pair| pair[1] - pair[0]).collect();
    assert_eq!(steps, vec![2, 3, 2, 2], "not one stride");
    assert!(steps.iter().all(|&step| step < 8), "and they overlap");
}

/// No shipped strategy can produce it, and each says so instead of producing
/// something else.
///
/// This is the answer to "should the hook return a grid?". If it did, this
/// lattice would be inexpressible and the op would get the easy case silently.
/// Stating it as regions means the framework knows exactly what it cannot build.
#[test]
fn a_lattice_no_grid_produces_is_refused_by_every_shipped_strategy() {
    let workflow = Workflow::new(
        Chain::op(SpreadLatticeOp::new("spread", 5, 8)),
        SPREAD,
        Dtype::F64,
    );
    for strategy in [
        &Trivial as &dyn Strategy,
        &Enumerating::default() as &dyn Strategy,
        &Greedy::default() as &dyn Strategy,
    ] {
        let message = strategy
            .decompose(&workflow, &Constraints::default())
            .unwrap_err()
            .to_string();
        // All three name the same fact in the same words: this lattice is not a
        // block grid, and a grid is all any of them can build.
        assert!(
            message.contains("block grid"),
            "{}: {message}",
            strategy.name()
        );
    }
}

/// It **is** satisfiable, and only one way: the block lattice stays the unit
/// grid of an index space and the overlap lives in the per-block fetch region.
///
/// That is not a workaround, it is the representation such a lattice has. The
/// plan below checks clean *and* satisfies the mandate, which is the whole
/// claim: the constraint is stated in the source space and matched against what
/// a block is actually handed.
///
/// What it cannot yet do is **run**, and the reason is the wall
/// `docs/design/BLOCK_OPS.md` records: a cross-grid fetch may translate, but not
/// resize. The op is handed an `[8, 4, 4]` window and `BlockOp::apply` writes an
/// output the shape of its input, so there is nowhere for a `[1, 1, 1]` slot of
/// the index array to come from. The executor refuses by name rather than
/// producing a wrong volume, and that refusal is asserted here so the wall is
/// recorded where it is hit rather than only in a document.
#[test]
fn the_only_plan_that_satisfies_it_carries_the_lattice_as_a_fetch_region() {
    let op = SpreadLatticeOp::new("spread", 5, 8);
    let windows = op.windows(SPREAD);
    let workflow = Workflow::new(Chain::op(op), SPREAD, Dtype::F64);

    // The index space: one slot per window, and the windows themselves as the
    // regions each slot is served from.
    let index_space = [windows.len(), 1, 1];
    let phase = PhaseDecomposition::derive(
        vec![0],
        vec!["spread".to_string()],
        [0, 0, 0],
        [0, 0, 0],
        BlockGrid::new(index_space, [1, 1, 1]).unwrap(),
    )
    .with_sources(|block| windows[block.index[0]].clone());
    let decomposition = Decomposition {
        volume: SPREAD,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    };

    decomposition.check().unwrap();
    check_block_constraints(&workflow.chain, &decomposition).unwrap();
    // and the plan says what will really be read, which is more than the volume
    // holds, because the windows overlap
    assert_eq!(decomposition.exact_read_voxels(), vec![5 * 8 * 4 * 4]);
    assert!(decomposition.exact_read_voxels()[0] > SPREAD.iter().product::<usize>());

    let env = ArrayEnvironment::for_decomposition(
        Voxels::zeros(Dtype::F64, SPREAD).unwrap(),
        &decomposition,
        [4, 4, 4],
    )
    .unwrap();
    let message = execute("t", &workflow, &decomposition, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("has nowhere to land") && message.contains("[8, 4, 4]"),
        "{message}"
    );
}

// ------------------------------------------------- what is still in tension --

/// A mandated extent and a halo cannot both be satisfied, and the planner says
/// so rather than producing a plan whose edge blocks are the wrong shape.
///
/// The op is handed `core` grown by the halo and clamped at the volume, so a
/// single extent cannot hold for both an interior block and an edge one. This is
/// the coordinate-space problem `docs/design/BLOCK_OPS.md` reserves for the reach
/// representation: until a reach carries the space it is stated in, "the extent I
/// accept" and "the extent I need around it" are the same number and cannot both
/// be honoured.
struct ReachingMandateOp;

impl BlockOp for ReachingMandateOp {
    fn name(&self) -> &'static str {
        "reaching"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        usize::from(axis == 0)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> blockflow::Result<()> {
        out.assign(input)
    }

    fn block_constraint(&self, _volume: [usize; 3]) -> Option<BlockConstraint> {
        Some(BlockConstraint::Extent([4, 4, 4]))
    }
}

#[test]
fn a_mandated_extent_and_a_halo_are_not_jointly_satisfiable() {
    let volume = [12, 4, 4];
    let workflow = Workflow::new(Chain::op(ReachingMandateOp), volume, Dtype::F64);
    let message = Enumerating::default()
        .decompose(&workflow, &Constraints::default())
        .unwrap_err()
        .to_string();
    assert!(message.contains("accepts exactly [4, 4, 4]"), "{message}");

    // The unconstrained op beside it plans as it always did, so the refusal is
    // about the mandate and not about the volume.
    let plain = Workflow::new(
        Chain::op(IdentityOp::new("plain", [1, 0, 0])),
        volume,
        Dtype::F64,
    );
    Enumerating::default()
        .decompose(&plain, &Constraints::default())
        .unwrap();
}
