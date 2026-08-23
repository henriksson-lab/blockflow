// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **`PartitionSearch::SingleGroup`: the block edge chosen, the fusion given.**
//
// It exists because a caller may already have decided its chain is one phase for
// a reason no cost model can see — a consumer's thinning stage fuses `n` passes
// per round because the phase it really runs is an `IterativeOp` with **one**
// substage, and the chain it hands a planner is a stand-in with `n` slots. Asking
// the DP about that stand-in gets a correct answer to a question the stage cannot
// act on: it cuts between passes and buys a materialisation the op has no way to
// perform. So such a caller hand-rolled the candidate sweep, and a hand-rolled
// sweep prices grids the search would never offer the moment either drifts.
//
// What has to be true for this to be worth having, and is asserted below:
//
// 1. **It is the same sweep**, not a second one — at one group the DP and this
//    must return the identical grid, because both are `PhasePricer` picking an
//    argmin with the same tie-break.
// 2. **It really declines to cut**, where the DP would have. Without this the
//    mode could be the DP under another name and every test would still pass.
// 3. **A barrier is refused by name**, not fused over. A full-reach op must be
//    alone in its phase, so no block candidate makes one group legal, and saying
//    "nothing fits the budget" would send a caller to widen a budget that was
//    never the problem.

use blockflow::decomposition::{Constraints, CostModel};
use blockflow::op::Chain;
use blockflow::probes::{OpaqueOp, WindowSumOp};
use blockflow::strategy::{Enumerating, PartitionSearch, Strategy, Workflow};
use blockflow::Dtype;

const VOLUME: [usize; 3] = [64, 64, 64];

fn constraints() -> Constraints {
    Constraints {
        budget_bytes: None,
        expected_concurrency: 1,
        model: CostModel::default(),
        block_candidates: vec![64, 32, 16, 8],
        split_axes: vec![0, 1, 2],
        ..Default::default()
    }
}

fn enumerating(search: PartitionSearch, concurrency: usize) -> Enumerating {
    Enumerating {
        concurrency,
        search,
        ..Enumerating::default()
    }
}

/// A chain the DP has a reason to cut: several windowed ops, so fusing them adds
/// their reaches and widens the halo every block re-reads.
fn cuttable() -> Chain {
    Chain::sequence(
        (0..4)
            .map(|i| Chain::op(WindowSumOp::new("window", [2 + i, 2 + i, 2 + i]).with_cost(4.0)))
            .collect(),
    )
}

/// **It declines to cut where the DP cuts.** The point of the mode, and the thing
/// that would silently not be true if it were the DP under another name.
#[test]
fn it_gives_one_phase_where_the_dp_chooses_several() {
    let workflow = Workflow::new(cuttable(), VOLUME, Dtype::F64);
    let constraints = constraints();
    let dp = enumerating(PartitionSearch::Dp, 8)
        .decompose(&workflow, &constraints)
        .expect("the DP plans");
    let one = enumerating(PartitionSearch::SingleGroup, 8)
        .decompose(&workflow, &constraints)
        .expect("one group plans");
    assert_eq!(
        one.n_phases(),
        1,
        "SingleGroup returned {} phases",
        one.n_phases()
    );
    assert_eq!(
        one.phases[0].slots,
        vec![0, 1, 2, 3],
        "and the one phase owns every slot"
    );
    assert!(
        dp.n_phases() > 1,
        "the fixture stopped being one the DP cuts, so this test no longer distinguishes \
         anything: the DP returned {} phase(s)",
        dp.n_phases()
    );
}

/// **It is the same sweep.** Where the DP itself chooses one group, the two must
/// agree on the grid to the axis — same `PhasePricer`, same `price_phase`, same
/// `phase_makespan`, same tie-break.
#[test]
fn at_one_group_it_returns_the_grid_the_dp_returns() {
    let constraints = constraints();
    for concurrency in [1usize, 4, 16] {
        // A single slot: there is no partition to choose, so the DP's answer is
        // one group by construction and only the edge is in question.
        let workflow = Workflow::new(
            Chain::op(WindowSumOp::new("window", [3, 3, 3]).with_cost(6.0)),
            VOLUME,
            Dtype::F64,
        );
        let dp = enumerating(PartitionSearch::Dp, concurrency)
            .decompose(&workflow, &constraints)
            .expect("the DP plans");
        let one = enumerating(PartitionSearch::SingleGroup, concurrency)
            .decompose(&workflow, &constraints)
            .expect("one group plans");
        assert_eq!(dp.n_phases(), 1);
        assert_eq!(
            one.phases[0].grid.block(),
            dp.phases[0].grid.block(),
            "at concurrency {concurrency} the two searches chose different grids"
        );
        assert_eq!(one.fingerprint(), dp.fingerprint());
    }
}

/// **A planning barrier is refused by name**, because no block candidate could
/// ever make one group legal and a budget message would send the caller the wrong
/// way. The negative control is the same chain with the barrier removed.
#[test]
fn a_barrier_inside_the_group_is_refused_and_names_the_cut_rather_than_the_budget() {
    let constraints = constraints();
    let with_barrier = Workflow::new(
        Chain::sequence(vec![
            Chain::op(WindowSumOp::new("window", [1, 1, 1])),
            // Reaches the whole of axis 0: a planning barrier, which must be
            // alone in its phase.
            Chain::op(OpaqueOp::new("whole", [VOLUME[0], 0, 0])),
            Chain::op(WindowSumOp::new("window", [1, 1, 1])),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let message = enumerating(PartitionSearch::SingleGroup, 4)
        .decompose(&with_barrier, &constraints)
        .expect_err("one group across a barrier is not a legal plan")
        .to_string();
    assert!(
        message.contains("full-reach op forces a cut"),
        "the refusal should name the barrier: {message}"
    );
    assert!(
        !message.contains("budget"),
        "the refusal must not blame a budget that was never the problem: {message}"
    );

    // The control: the same shape without the barrier plans in one group, so the
    // refusal above is about the barrier and not about the chain's length.
    let without = Workflow::new(
        Chain::sequence(vec![
            Chain::op(WindowSumOp::new("window", [1, 1, 1])),
            Chain::op(WindowSumOp::new("window", [1, 1, 1])),
            Chain::op(WindowSumOp::new("window", [1, 1, 1])),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let plan = enumerating(PartitionSearch::SingleGroup, 4)
        .decompose(&without, &constraints)
        .expect("no barrier, one group");
    assert_eq!(plan.n_phases(), 1);
}
