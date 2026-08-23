//! **A fragment phase, reported through `Stats`.**
//!
//! `Stats::ops_applied` counts applications of a *chain slot* and
//! `Stats::blocks_visited` is derived from the `OpApplied` events a chain emits.
//! A fragment phase owns no chain slot and emits no such event, so a plan made
//! entirely of fragment phases reports zero in both — and a downstream crate
//! reading `Stats` could see nothing a fragment phase did except the pixels it
//! happened to touch.
//!
//! Those two zeroes are **kept**, and the questions they conflated are split
//! instead: `Stats::fragment_applications` is where a fragment phase's work is
//! counted, and `Stats::blocks_admitted` is how many blocks the plan touched
//! whatever kind of phase touched them. Making a fragment phase emit `OpApplied`
//! would have moved `ops_applied`, the exported log and
//! `ExecutionLog::recomputed_margin_voxels`, which measures a *chain's* halo
//! redundancy from the regions those events carry.
//!
//! That is what these tests hold. `tests/fragment_join_barrier.rs` measures the
//! same two placements of the same merge by *asking the op* — it wraps the op in
//! a counting shim and reads the shim. This file measures them the way somebody
//! outside the crate has to: run the plan, read `Stats`, and nothing else. The
//! two arms differ by `Merge` alone; they declare the same barrier, grant the
//! same halo, read the same pixels and produce the same mask, and every counter
//! that separates them is one of the fragment counters below.

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::fragment::{fragment_phase, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::log::Stats;
use blockflow::op::Chain;
use blockflow::ops::components::Merge;
use blockflow::ops::fill::{FillHolesOp, LabelBackgroundOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [16, 32, 32];
const STREAM: &str = "stats.fill.faces";

/// A mask with sealed cavities in it, and one that drains through a face.
fn mask_scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    fill_box(&mut mask, [1, 2, 2], [14, 8, 8], true);
    fill_box(&mut mask, [2, 3, 3], [13, 7, 7], false);
    fill_box(&mut mask, [1, 12, 12], [14, 20, 20], true);
    fill_box(&mut mask, [2, 13, 13], [13, 19, 19], false);
    mask
}

fn fill_box(mask: &mut Array3<bool>, low: [usize; 3], high: [usize; 3], value: bool) {
    for i in low[0]..=high[0] {
        for j in low[1]..=high[1] {
            for k in low[2]..=high[2] {
                mask[[i, j, k]] = value;
            }
        }
    }
}

/// `ops::fill`'s two phases at `block`, with the merge placed by `merge`.
fn run(block: [usize; 3], merge: Merge) -> (Stats, Array3<bool>, usize) {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let blocks = grid.n_blocks();
    let label = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let fill = FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid).merging(merge);

    let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    labelling.dtype = Some(Dtype::U32);
    let mut filling = fragment_phase(&fill, grid).expect("phase 1");
    filling.dtype = Some(Dtype::Bool);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![labelling, filling],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("the plan tiles");

    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(mask_scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    let hints = Hints {
        concurrency: 1,
        ..Hints::default()
    };
    let stats = execute_phases(
        "fragment-stats",
        &workflow,
        &plan,
        &hints,
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
    )
    .expect("a run");
    let output = env.output().view::<bool>().expect("a mask").to_owned();
    (stats, output, blocks)
}

/// The two placements, told apart by `Stats` alone.
///
/// This is the whole point of the counters: the arms agree on the answer and on
/// every pixel figure, and the fragment figures are what carry the difference.
/// The hoisted arm reads the set for the phase, so its reads are `O(blocks)`;
/// the per-block arm reads the set in every block, so they are `O(blocks²)`.
#[test]
fn the_merge_placement_is_visible_in_stats() {
    let block = [8usize, 8, 8];
    let (hoisted, hoisted_mask, blocks) = run(block, Merge::OnceForThePhase);
    let (per_block, per_block_mask, again) = run(block, Merge::PerBlockBehindABarrier);
    assert_eq!(blocks, again);
    assert_eq!(blocks, 2 * 4 * 4);

    // Same answer. Without this the cost comparison is between two different
    // computations and means nothing.
    assert_eq!(
        hoisted_mask, per_block_mask,
        "the two placements must fill the same mask"
    );

    // Same pixels — which is exactly why nothing outside could see the
    // difference before. Both arms declare the barrier, so both are relieved of
    // the whole-volume halo.
    assert_eq!(
        hoisted.read_voxels, per_block.read_voxels,
        "both arms are behind a barrier, so both read the same pixels. If this ever differs \
         the arms differ by more than the merge's placement and the comparison below is not \
         about hoisting."
    );
    assert_eq!(hoisted.write_voxels, per_block.write_voxels);
    assert_eq!(hoisted.tasks, per_block.tasks);
    assert_eq!(hoisted.phases, per_block.phases);

    // And the figures that do carry it.
    assert_eq!(
        hoisted.sidecar_reads,
        2 * blocks as u64,
        "the set is read once for the phase, and once more for the `SeamFold::Unordered` check"
    );
    assert_eq!(
        per_block.sidecar_reads,
        (blocks * blocks) as u64,
        "every block reads the whole set"
    );
    assert_eq!(
        hoisted.sidecar_writes, per_block.sidecar_writes,
        "both arms write one fragment per block from the labelling phase; the placement of the \
         merge changes what is read, not what is written"
    );
    assert!(
        per_block.fragment_applications > hoisted.fragment_applications,
        "under a per-block merge every block is handed the whole fragment set, so the \
         `SeamFold::Unordered` order check costs a second application per block: {} against {}",
        per_block.fragment_applications,
        hoisted.fragment_applications
    );
    println!(
        "{blocks} blocks: hoisted {} fragment reads / {} applications, per-block {} / {}; both \
         read {} voxels",
        hoisted.sidecar_reads,
        hoisted.fragment_applications,
        per_block.sidecar_reads,
        per_block.fragment_applications,
        hoisted.read_voxels,
    );
}

/// **Zero, distinguished from absent.**
///
/// `ops_applied` is zero for this run and it is not the case that nothing
/// happened. `fragment_applications` is where the work is, and
/// `Stats::applications` is the sum a reader should consult when the question is
/// "did this run apply anything at all".
#[test]
fn a_fragment_only_run_reports_zero_chain_work_and_says_where_its_work_went() {
    let (stats, _, blocks) = run([8, 8, 8], Merge::OnceForThePhase);
    assert_eq!(
        stats.ops_applied, 0,
        "a fragment phase owns no chain slot, so it can only contribute zero here"
    );
    assert_eq!(
        stats.blocks_visited, 0,
        "`blocks_visited` is derived from the `OpApplied` events a chain emits, and a fragment \
         phase emits none"
    );
    // **The question `blocks_visited` cannot answer, answered.** `TaskAdmitted`
    // is emitted for every task of every phase, so this is the block count a
    // reader means by "how many blocks did this run touch" — and it separates
    // the two zeroes above from a plan that genuinely did nothing.
    assert_eq!(
        stats.blocks_admitted, blocks,
        "every block of the lattice was admitted, in both phases; a block is counted once"
    );
    assert!(
        stats.blocks_admitted > stats.blocks_visited,
        "the whole point of the second field: {} blocks ran and {} applied a chain slot",
        stats.blocks_admitted,
        stats.blocks_visited
    );
    assert_eq!(
        stats.tasks,
        2 * blocks,
        "`tasks` counts (phase, block) pairs and `blocks_admitted` counts blocks, so the two \
         differ by the phase count and neither is a substitute for the other"
    );
    assert_eq!(
        stats.fragment_applications,
        2 * blocks as u64,
        "two fragment phases, one application per block each"
    );
    assert_eq!(
        stats.applications(),
        stats.fragment_applications,
        "`applications` is the sum, and it is the figure that separates 'nothing ran' from \
         'not counted in the field you read'"
    );
    assert!(stats.applications() > 0);
}

/// The listing cost a barrier's completeness check pays, given a name.
///
/// `strategy::reduce_phase` verifies that the fragment set is complete before it
/// reduces over it, by listing the producing stream's keys. That is `O(blocks)`
/// and moves no bytes, so it appears in neither `sidecar_reads` nor
/// `sidecar_bytes_read` — and its multiplier is the block count, which a caller
/// raises to make a stage fit in memory. Counted so that a caller can see it
/// rise.
#[test]
fn the_completeness_check_costs_listings_and_no_bytes() {
    let (coarse, _, coarse_blocks) = run([16, 32, 32], Merge::OnceForThePhase);
    let (fine, _, fine_blocks) = run([8, 8, 8], Merge::OnceForThePhase);
    assert_eq!(coarse_blocks, 1);
    assert_eq!(fine_blocks, 32);
    // **Both listings, named.** The labelling phase declares one every-block
    // stream, and it is listed twice: once by `execute_phases` when that phase's
    // last task completes, and once by `strategy::reduce_phase` before the
    // barrier reduces over it. The second is the copy the distributed path
    // depends on — a worker never reaches the end-of-phase moment — and keeping
    // it in the single-node path too is a decision recorded on `reduce_phase`.
    // Pinned to a number so that a change which added a third would have to say
    // so here.
    assert_eq!(
        coarse.sidecar_listings, 2,
        "one listing at the end of the producing phase and one before the reduction"
    );
    assert_eq!(
        coarse.sidecar_listings, fine.sidecar_listings,
        "the number of listings is a property of the plan's streams, not of the lattice"
    );
    // The block count, which is what the two listings are each multiplied by,
    // and the reason the pair is measured at all.
    assert_eq!(coarse.blocks_admitted, coarse_blocks);
    assert_eq!(fine.blocks_admitted, fine_blocks);
    assert_eq!(
        coarse.sidecar_keys_listed,
        coarse.sidecar_listings * coarse_blocks as u64,
        "each listing returns one key per block"
    );
    assert_eq!(
        fine.sidecar_keys_listed,
        fine.sidecar_listings * fine_blocks as u64,
        "and so the keys listed rise with the block count, while the bytes do not move"
    );
    println!(
        "1 block: {} listings, {} keys; {fine_blocks} blocks: {} listings, {} keys",
        coarse.sidecar_listings,
        coarse.sidecar_keys_listed,
        fine.sidecar_listings,
        fine.sidecar_keys_listed,
    );
}
