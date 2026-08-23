// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What the barrier's completeness check costs, in listings.**
//
// `strategy::reduce_phase` verifies that the fragment set it is about to reduce
// over is complete, by listing the producing streams' keys. That is the guard
// against the one failure mode a distributed barrier has — a sidecar store that
// is not in fact shared, where every worker reduces over its own fragments and
// answers plausibly and differently on each machine — so it is not something to
// trade away. What is worth measuring is its *multiplier*, because a listing
// returns one key per block and the block count is the figure a caller raises to
// make a stage fit in memory.
//
// Two multipliers were in it, and they are different things:
//
// * **The producing phase's stream count.** `check_fragment_coverage` lists
//   every output stream of the op it is handed, so one call already covers all
//   of a producer's streams. Run once per *input*, a barrier joining two streams
//   from one producer ran the whole check twice — every listing in it repeated,
//   for nothing. That is removed: the check is grouped by producing phase and
//   runs once per phase. `two_streams_from_one_producer_cost_one_check` is the
//   measurement and `a_hole_is_still_refused_after_the_grouping` is the liveness
//   control beside it, because a deduplication that quietly dropped the check
//   would pass the first test perfectly.
// * **The single-node repetition.** `execute_phases` runs the same check on a
//   fragment phase's outputs when that phase completes, and a later barrier runs
//   it again from `reduce_phase`. That one is **kept** — see `reduce_phase`'s
//   doc for the two alternatives and why both are worse — and what it costs is
//   pinned here rather than described: one extra listing per producing phase,
//   `O(blocks)` keys, against a phase that irreducibly writes `blocks` fragments
//   and reads at least `blocks` more.

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    fragment_phase, pack_u64, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp,
    FragmentOutput, PhaseView, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::reduce_phase;
use blockflow::voxels::Voxels;
use ndarray::Array3;

const VOLUME: [usize; 3] = [16, 4, 4];
const BLOCK: [usize; 3] = [4, 4, 4];
const BLOCKS: usize = 4;
const LEFT: &str = "listings.left";
const RIGHT: &str = "listings.right";
const RESULT: &str = "listings.result";

/// Phase 0: one pixel read, two every-block streams. The producer whose streams
/// the barrier below joins.
struct TwoStreamOp;

impl FragmentOp for TwoStreamOp {
    fn name(&self) -> &'static str {
        "two-streams"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(LEFT, Lifecycle::DeleteOnExit, Coverage::EveryBlock),
            FragmentOutput::new(RIGHT, Lifecycle::DeleteOnExit, Coverage::EveryBlock),
        ]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let word = at.index[0] as u64;
        Ok(BlockOutput::fragment(LEFT, pack_u64(&[word]))
            .with_fragment(RIGHT, pack_u64(&[word + 100])))
    }
}

/// Phase 1: a barrier that reduces over `LEFT`, and reads either one stream of
/// phase 0 or both.
///
/// **The two arms differ in `inputs` and in nothing else.** `reduce` folds
/// `LEFT` in both, so the fold is the same work, produces the same bytes and
/// makes the same reads; the only thing the second declaration can change is how
/// many times the completeness check runs.
struct JoinOp {
    both: bool,
}

impl FragmentOp for JoinOp {
    fn name(&self) -> &'static str {
        "join"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        let mut inputs = vec![FragmentInput::own(LEFT, 0)];
        if self.both {
            inputs.push(FragmentInput::own(RIGHT, 0));
        }
        inputs
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            RESULT,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut total = 0u64;
        at.stream_fragments(LEFT, &mut |_, bytes| {
            total += blockflow::fragment::unpack_u64(bytes)?[0];
            Ok(())
        })?;
        Ok(pack_u64(&[total]))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            RESULT,
            pack_u64(&[at.index[0] as u64]),
        ))
    }
}

fn grid() -> BlockGrid {
    BlockGrid::new(VOLUME, BLOCK).expect("a lattice")
}

fn plan() -> Decomposition {
    let producer = TwoStreamOp;
    let join = JoinOp { both: false };
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![
            fragment_phase(&producer, grid()).expect("phase 0"),
            fragment_phase(&join, grid()).expect("phase 1"),
        ],
        chain_reach: [0, 0, 0],
    }
}

/// A prepared environment with both streams whole: one fragment per block per
/// stream, written directly rather than by running the phase, so the fragment
/// set is the fixture and not an outcome.
fn env_with(plan: &Decomposition, holes: &[(&str, [usize; 3])]) -> ArrayEnvironment {
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]))),
        plan,
        VOLUME,
    )
    .expect("an environment");
    for stream in [LEFT, RIGHT, RESULT] {
        env.declare_sidecar(stream, Lifecycle::DeleteOnExit)
            .expect("declared");
    }
    for core in plan.phases[0].grid.cores() {
        for stream in [LEFT, RIGHT] {
            if holes.contains(&(stream, core.index)) {
                continue;
            }
            env.write_sidecar(stream, 0, core.index, &pack_u64(&[core.index[0] as u64]))
                .expect("a fragment");
        }
    }
    env
}

/// **The check is run once per producing phase, not once per input stream.**
///
/// The two arms declare the same barrier over the same fragment set and fold the
/// same stream; the second one additionally names `RIGHT` in `inputs`. Both
/// producer streams are checked either way — `check_fragment_coverage` lists
/// every output of the op it is handed — so the second declaration must not buy
/// a second pass over the same keys.
#[test]
fn two_streams_from_one_producer_cost_one_check() {
    let plan = plan();
    let producer = TwoStreamOp;

    let one = JoinOp { both: false };
    let work_one = [PhaseWork::Fragments(&producer), PhaseWork::Fragments(&one)];
    let env_one = env_with(&plan, &[]);
    let before = env_one.counters().listing_snapshot();
    let blob_one = reduce_phase(&plan, 1, &work_one, &env_one).expect("a complete set reduces");
    let after = env_one.counters().listing_snapshot();
    let (listings_one, keys_one) = (after.0 - before.0, after.1 - before.1);

    let two = JoinOp { both: true };
    let work_two = [PhaseWork::Fragments(&producer), PhaseWork::Fragments(&two)];
    let env_two = env_with(&plan, &[]);
    let before = env_two.counters().listing_snapshot();
    let blob_two = reduce_phase(&plan, 1, &work_two, &env_two).expect("a complete set reduces");
    let after = env_two.counters().listing_snapshot();
    let (listings_two, keys_two) = (after.0 - before.0, after.1 - before.1);

    assert_eq!(
        blob_one, blob_two,
        "the arms differ only in what they declare, so they must reduce to the same bytes; \
         without this the listing comparison is between two different computations"
    );
    assert_eq!(
        listings_one, listings_two,
        "a barrier that names two streams of one producer must not run the producer's \
         completeness check twice: {listings_one} listing(s) for one input against \
         {listings_two} for two"
    );
    assert_eq!(keys_one, keys_two);
    // And the absolute figure, named rather than left as a ratio: one listing
    // per output stream of the producing phase, each returning one key per
    // block. The producer declares two streams over a four-block lattice.
    assert_eq!(
        listings_one, 2,
        "one listing per stream of the checked phase"
    );
    assert_eq!(
        keys_one,
        2 * BLOCKS as u64,
        "each listing returns one key per block"
    );
    println!(
        "one input: {listings_one} listings / {keys_one} keys; two inputs: {listings_two} / \
         {keys_two}"
    );
}

/// **The liveness control for the test above.**
///
/// Grouping the check by producing phase would pass
/// `two_streams_from_one_producer_cost_one_check` perfectly if it had removed
/// the check instead of deduplicating it — both arms would list nothing and
/// agree. This is the same program with one thing changed: a fragment missing
/// from a stream the barrier reads. Both arms must refuse, including the arm
/// whose hole is in the stream it is *not* the first input of.
#[test]
fn a_hole_is_still_refused_after_the_grouping() {
    let plan = plan();
    let producer = TwoStreamOp;
    let two = JoinOp { both: true };
    let work = [PhaseWork::Fragments(&producer), PhaseWork::Fragments(&two)];

    // The hole is in `RIGHT`, the second declared input — the one whose own
    // check is the one the grouping removed. It is still caught, because the
    // single remaining call lists every stream the producer declares.
    let env = env_with(&plan, &[(RIGHT, [2, 0, 0])]);
    let err = reduce_phase(&plan, 1, &work, &env).expect_err("an incomplete set is refused");
    let text = err.to_string();
    assert!(text.contains("is not complete"), "{text}");
    assert!(text.contains(RIGHT), "the failing stream is named: {text}");
    assert!(
        text.contains("not actually shared between nodes"),
        "the distributed failure mode is named: {text}"
    );

    // A hole in the first input is refused too, so the pass above is not the
    // grouping happening to keep only the stream that was broken.
    let env = env_with(&plan, &[(LEFT, [0, 0, 0])]);
    let err = reduce_phase(&plan, 1, &work, &env).expect_err("an incomplete set is refused");
    assert!(err.to_string().contains(LEFT), "{err}");

    // And the positive arm, so that "refused" is not simply what this fixture
    // always does.
    let env = env_with(&plan, &[]);
    reduce_phase(&plan, 1, &work, &env).expect("a complete set reduces");
}
