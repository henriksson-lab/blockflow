// SPDX-License-Identifier: MIT
//
// **One plan, two block sizes**, and the objective that makes that possible.
//
// The finding this file exists to pin down came from a tile-scale run of a real
// chain: one stage wants a **single block** and the stage next to it wants
// **many**, and no single number serves both.
//
// * a fragment-and-join stage read the volume `N + 1` times for `N` blocks —
//   22x at 21 blocks, 169x at 168. It wants one.
// * a local stage ran **12.0x** faster at a 256 cut than at one block: 415.9 s
//   against 5002.8 s, over the same 89 rounds and the same 1 850 209 voxels,
//   with 0 differing. It wants many.
// * and at one block the chain had **one task**, so a 40-worker pool ran it on
//   one thread with the rest parked in `futex_wait`.
//
// Where the mechanism already was, and where it was not
// -----------------------------------------------------
// `PhaseDecomposition::grid` has always been per-phase — a boundary is a
// materialisation, so two phases need not share a grid — and `Enumerating`'s
// candidate sweep has always been *inside* the per-run price, choosing an edge
// for each contiguous run on its own. **The search was already per-phase.**
//
// What was uniform was the **objective**. It was the phase's serial work,
// `cost_per_block x n_blocks`, which is `volume x redundancy x per-voxel`: the
// block count cancels, `redundancy >= 1` falls monotonically as the block grows,
// and so the sweep answered *"the largest candidate that fits"* in every phase of
// every chain. The freedom existed and could never be exercised.
//
// So this file is about the objective. It is `strategy::phase_makespan` inside
// the search and `strategy::predicted_makespan` read back off a finished plan,
// and both are:
//
// ```text
// max( cost_per_block x ceil(n_blocks / workers) ,  read x read_cost + core x write )
//      \------------- the pool -----------------/   \--------- the channel -------/
// ```
//
// Both halves are load-bearing and the tests below separate them:
//
// * without the **pool** term the search cannot see that the unit of parallelism
//   is the block, and takes one block everywhere —
//   `a_search_that_cannot_see_task_count_takes_one_block_everywhere`;
// * without the **channel** term it thinks workers multiply bandwidth, and buys
//   parallelism with read amplification — the exact trade
//   `decomposition::cuttable_axes` was written to forbid after a 336-block plan
//   ran past fifteen minutes. `read_amplification_is_not_bought_with_workers`
//   holds the line.
//
// | test | what it settles |
// |---|---|
// | `one_plan_now_carries_two_block_sizes` | the headline: the join phase keeps one block, the local phase gets eight |
// | `a_search_that_cannot_see_task_count_takes_one_block_everywhere` | the negative control, which is `concurrency == 1` |
// | `the_local_phase_is_six_times_faster_for_no_extra_reading` | the win as a number, with the read volume unchanged |
// | `read_amplification_is_not_bought_with_workers` | the channel bound, on a volume where the floor *would* admit the cut |
// | `no_worker_count_moves_a_voxel` | the acceptance bar: every plan the search now produces computes the same bytes |
// | `residency_does_not_rise_with_the_worker_count` | the priced peak, per phase and summed |
// | `the_plan_is_the_minimum_of_the_objective_it_claims` | the chosen edge really is the argmin of `predicted_makespan` |
// | `the_dp_and_the_enumeration_agree_under_the_makespan_objective` | the DP's licence survives the new term |
// | `a_whole_axis_declaration_is_never_cut_at_any_worker_count` | the search cannot propose a lattice `Decomposition::check` will refuse |
// | `the_account_names_every_candidate_it_dropped` | no silent cap |

use blockflow::decomposition::{
    predicted_cost, Constraints, CostModel, Decomposition, PhaseDecomposition,
};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::geometry::BlockGrid;
use blockflow::graph::TaskGraph;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::probes::{AffineOp, WindowSumOp};
use blockflow::reach::{AxisReach, Reach};
use blockflow::strategy::{
    predicted_makespan, Enumerating, PartitionSearch, SchedulePriority, SearchAccount, Strategy,
    Trivial, Workflow,
};
use blockflow::voxels::Voxels;
use ndarray::Array3;

// ------------------------------------------------------------- the fixture --

/// Long on the split axis and small on the others, so the block count is a
/// question about one axis and the arithmetic below is readable.
const VOLUME: [usize; 3] = [64, 16, 16];

/// A ladder of edges over that axis, every one of which divides it.
const CANDIDATES: [usize; 5] = [4, 8, 16, 32, 64];

/// **The join.** A radius of 30 on an axis of 64 means `lo + hi = 60`, so
/// `edge + 60 < 64` holds for no candidate: `cuttable_axes` refuses every cut,
/// because none of them narrows what a block reads. This is the shape of the
/// fragment-and-join stage — one block, and the plan says so structurally rather
/// than by price.
const JOIN_RADIUS: [usize; 3] = [30, 0, 0];

/// **The local op.** Reach zero, so its redundancy is `1.0` at every grid and it
/// reads the same total however it is cut; ten units of compute a voxel, so what
/// it has to gain from being cut is the pool.
const LOCAL_COST: f64 = 10.0;

fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(WindowSumOp::new("join", JOIN_RADIUS)),
        Chain::op(AffineOp::new("local", 2.0, 1.0, [0, 0, 0]).with_cost(LOCAL_COST)),
    ])
}

fn constraints() -> Constraints {
    Constraints {
        budget_bytes: None,
        expected_concurrency: 1,
        model: CostModel::default(),
        block_candidates: CANDIDATES.to_vec(),
        // One axis, so a block edge is a block count and nothing else.
        split_axes: vec![0],
    }
}

fn strategy(workers: usize, search: PartitionSearch) -> Enumerating {
    Enumerating {
        concurrency: workers,
        priority: SchedulePriority::PhaseMajor,
        search,
    }
}

fn workflow(chain: Chain, volume: [usize; 3]) -> Workflow {
    Workflow::new(chain, volume, Dtype::F64)
}

fn plan(workers: usize) -> (Workflow, Decomposition, SearchAccount) {
    let workflow = workflow(chain(), VOLUME);
    let (decomposition, account) = strategy(workers, PartitionSearch::Dp)
        .decompose_accounted(&workflow, &constraints())
        .expect("a plan");
    (workflow, decomposition, account)
}

/// The edge of each phase's block on the split axis, which on this fixture is
/// the whole of what the per-phase choice can say.
fn edges(decomposition: &Decomposition) -> Vec<usize> {
    decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect()
}

fn block_counts(decomposition: &Decomposition) -> Vec<usize> {
    decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.n_blocks())
        .collect()
}

/// Structure at several scales, so a seam has something to get wrong.
fn input(shape: [usize; 3]) -> Voxels {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
    })
    .into()
}

/// Plan with `strategy`, run it, and hand back the output image.
fn run(strategy: &dyn Strategy, chain: Chain, constraints: &Constraints) -> Voxels {
    let workflow = workflow(chain, VOLUME);
    let decomposition = strategy.decompose(&workflow, constraints).expect("a plan");
    let env = ArrayEnvironment::for_decomposition(input(VOLUME), &decomposition, [4, 4, 4])
        .expect("an environment");
    strategy
        .run(&workflow, &decomposition, &env)
        .expect("a run");
    env.image(decomposition.n_phases())
}

// ============================================ 1. one plan, two block sizes ==

/// **The headline.** At eight workers the two phases of one plan are cut
/// differently, and each is cut the way its own arithmetic wants.
#[test]
fn one_plan_now_carries_two_block_sizes() {
    let (_, decomposition, account) = plan(8);
    assert_eq!(
        decomposition.n_phases(),
        2,
        "the join and the local op should not be fused: {:?}",
        decomposition
            .phases
            .iter()
            .map(|p| &p.names)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        edges(&decomposition),
        vec![64, 8],
        "one plan, two block sizes"
    );
    assert_eq!(
        block_counts(&decomposition),
        vec![1, 8],
        "the join takes one block and the local op takes one per worker"
    );
    // And the account says so in the plan's own terms.
    let chosen: Vec<(usize, usize, usize)> = account
        .chosen
        .iter()
        .map(|phase| (phase.block[0], phase.n_blocks, phase.rounds))
        .collect();
    assert_eq!(chosen, vec![(64, 1, 1), (8, 8, 1)]);
    assert_eq!(account.workers, 8);

    // **And the symptom this began as.** The unit of parallelism is the task and
    // the unit of a task is the block, so a plan of one block is a plan of one
    // task — which is what a 40-worker pool found when it ran the measured chain
    // on one thread with the rest parked in `futex_wait`.
    let (_, control, _) = plan(1);
    assert_eq!(
        tasks(&control),
        1,
        "the control plans one task for the chain"
    );
    assert_eq!(
        tasks(&decomposition),
        9,
        "one for the join, eight for the local op"
    );
}

/// Tasks in the DAG the executor will schedule: one per block per phase.
fn tasks(decomposition: &Decomposition) -> usize {
    let graph = TaskGraph::build(decomposition);
    (0..decomposition.n_phases())
        .map(|phase| graph.tasks_in_phase(phase).len())
        .sum()
}

/// The join phase is one block because **no cut narrows what it reads**, not
/// because a weight came out that way. `cuttable_axes` is exact arithmetic on
/// the declared reach, so this holds at every worker count.
#[test]
fn the_join_phase_is_one_block_at_every_worker_count() {
    for workers in [1usize, 2, 4, 8, 16, 40] {
        let (_, decomposition, _) = plan(workers);
        assert_eq!(
            decomposition.phases[0].grid.n_blocks(),
            1,
            "workers {workers}"
        );
        assert!(
            decomposition.phases[0].grid.split_axes().is_empty(),
            "workers {workers}: the join's axis was cut"
        );
    }
}

// ================================================= 2. the negative control ==

/// **A search that cannot see task count picks one block everywhere.**
///
/// That search is not a straw man built for this test: it is `concurrency == 1`,
/// which makes `ceil(n / workers)` equal `n_blocks` and the objective the serial
/// work total — the objective this crate searched under until now, still
/// reachable, still the default.
///
/// On this fixture it fuses the whole chain into **one phase of one whole-volume
/// block**, which is the cheapest plan there is by work: no materialisation, and
/// a reach of 30 that no cut would narrow, so redundancy `1.0`. It is also one
/// task. A pool of eight runs it on one thread.
#[test]
fn a_search_that_cannot_see_task_count_takes_one_block_everywhere() {
    let (_, control, _) = plan(1);
    assert_eq!(control.n_phases(), 1, "the work objective fuses everything");
    assert_eq!(block_counts(&control), vec![1]);
    assert_eq!(edges(&control), vec![64]);

    let (_, informed, _) = plan(8);
    assert_eq!(
        block_counts(&informed),
        vec![1, 8],
        "the makespan objective cuts the chain and then cuts the half that gains"
    );
    assert_ne!(block_counts(&control), block_counts(&informed));

    // **And on read volume the control really does look optimal.** It reads the
    // volume once and the plan the search prefers reads it twice, because a
    // phase boundary is a materialisation and somebody has to read the
    // intermediate back. A search scored on bytes moved would keep the control
    // and be wrong by 2.6x in wall clock — which is the whole reason the
    // objective is not bytes moved.
    let total = |d: &Decomposition| d.exact_read_voxels().iter().sum::<usize>();
    assert_eq!(total(&control), 16384);
    assert_eq!(total(&informed), 32768);
    let model = constraints().model;
    let (workflow, _, _) = plan(8);
    let control_time = predicted_makespan(&workflow.chain, &control, &[], &model, 8).unwrap();
    let informed_time = predicted_makespan(&workflow.chain, &informed, &[], &model, 8).unwrap();
    assert_eq!(control_time / informed_time, 2.6);
}

/// **The win, as a number, twice — once holding the partition fixed so nothing
/// else can be credited for it, and once against the plan the control actually
/// returns.**
///
/// Holding the cuts fixed is the sharper of the two. The local phase has reach
/// zero, so its redundancy is `1.0` at every grid: the plan at edge 64 and the
/// plan at edge 8 read *the same voxels*, do the same work, write the same
/// bytes, and differ only in how many tasks there are to do it with. The
/// makespan falls **6.0x** and the serial work total does not move at all —
/// which is exactly why the old objective could not prefer either.
///
/// That is the shape of the measured case in this file's header: same voxels,
/// same rounds, 0 differing, 12.0x.
#[test]
fn the_local_phase_is_six_times_faster_for_no_extra_reading() {
    let constraints = constraints();
    let (workflow, informed, _) = plan(8);

    // The same partition, with the local phase given the block the work
    // objective would have chosen for it.
    let uncut = with_edge(&informed, 1, 64).expect("the whole volume is always a grid");
    assert_eq!(block_counts(&uncut), vec![1, 1]);

    assert_eq!(
        informed.exact_read_voxels(),
        uncut.exact_read_voxels(),
        "the cut moved no bytes, which is the point"
    );
    assert_eq!(
        predicted_cost(&workflow.chain, &informed, &[], &constraints.model).unwrap(),
        predicted_cost(&workflow.chain, &uncut, &[], &constraints.model).unwrap(),
        "and it changed no work, which is why the old objective was blind to it"
    );

    let cut = predicted_makespan(&workflow.chain, &informed, &[], &constraints.model, 8).unwrap();
    let whole = predicted_makespan(&workflow.chain, &uncut, &[], &constraints.model, 8).unwrap();
    assert_eq!(whole, 245760.0);
    assert_eq!(cut, 81920.0);
    // The plan-level figure is diluted by the join phase, which is one block
    // either way; the local phase on its own is the 6.0x.
    assert_eq!(whole - 49152.0, 196608.0, "the local phase, uncut");
    assert_eq!(cut - 49152.0, 32768.0, "the local phase, at eight blocks");
    assert_eq!((whole - 49152.0) / (cut - 49152.0), 6.0);

    // And against the plan the control really returns — one fused phase, which
    // saves a materialisation and pays for it in wall clock.
    let (_, control, _) = plan(1);
    let control_makespan =
        predicted_makespan(&workflow.chain, &control, &[], &constraints.model, 8).unwrap();
    assert_eq!(control_makespan, 212992.0);
    assert_eq!(control_makespan / cut, 2.6);
}

// ============================================ 3. the channel bound is real ==

/// **The search will not buy parallelism with read amplification.**
///
/// This is the failure the pool term has on its own, and it is not
/// hypothetical: with `max(pool, channel)` reduced to `pool`, this fixture's
/// wide phase is handed **32 blocks at 16x the read volume** for a predicted
/// 2.5x, which is precisely the trade that put a 336-block plan past fifteen
/// minutes and put `cuttable_axes` in the crate.
///
/// The volume here is chosen so the floor *would* allow the cut — a reach of 120
/// on an axis of 256 leaves `edge + 240 < 256` true at an edge of 8 — so nothing
/// structural is doing the work. The channel bound is.
#[test]
fn read_amplification_is_not_bought_with_workers() {
    const WIDE: [usize; 3] = [256, 64, 64];
    let constraints = Constraints {
        block_candidates: vec![8, 16, 32, 64, 128, 256],
        split_axes: vec![0],
        ..constraints()
    };
    let wide = Chain::sequence(vec![
        Chain::op(WindowSumOp::new("join", [120, 0, 0])),
        Chain::op(AffineOp::new("local", 2.0, 1.0, [0, 0, 0]).with_cost(LOCAL_COST)),
    ]);
    let workflow = workflow(wide, WIDE);

    // The floor really does admit a cut here, or this test proves nothing.
    let admitted =
        blockflow::decomposition::cuttable_axes(&[0], &Reach::symmetric([120, 0, 0]), WIDE, 8);
    assert_eq!(admitted, vec![0], "the reach-derived floor allows edge 8");

    for workers in [1usize, 8, 40] {
        let decomposition = strategy(workers, PartitionSearch::Dp)
            .decompose(&workflow, &constraints)
            .expect("a plan");
        let grid = &decomposition.phases[0].grid;
        assert_eq!(
            grid.n_blocks(),
            1,
            "workers {workers}: the wide phase was cut into {} blocks of {:?}, \
             which reads the volume many times over",
            grid.n_blocks(),
            grid.block()
        );
    }
}

// ==================================================== 4. no answer may move ==

/// **The acceptance bar.** Every plan the new objective produces computes what
/// the one-block, one-phase oracle computes, byte for byte.
#[test]
fn no_worker_count_moves_a_voxel() {
    let constraints = constraints();
    let oracle = run(&Trivial, chain(), &constraints);
    for workers in [1usize, 2, 3, 4, 8, 16, 40] {
        for search in [PartitionSearch::Dp, PartitionSearch::Exhaustive] {
            let got = run(&strategy(workers, search), chain(), &constraints);
            assert_eq!(got, oracle, "workers {workers}, {search:?}");
        }
    }
}

// ======================================================= 5. residency ======

/// **Residency does not rise, and the reason it cannot is structural.**
///
/// The block choice enters the resident figure through
/// `PhaseCost::working_set_bytes_per_block`, which is the clamped read extent
/// times the element size: a phase the makespan objective moves to a *smaller*
/// block has a strictly smaller working set, and one it leaves alone has the
/// same. The blocks in flight are the pool, which is a caller's number rather
/// than a planner's, so the peak is `max over phases` of that figure — phases
/// are sequential, a boundary being a materialisation.
///
/// The partition may move, so the comparison is on the peak rather than
/// phase-by-phase. It is flat: the join phase reads its whole halo at one block
/// under every objective, and every phase the search re-cuts gets *lighter*.
///
/// **The one place residency can rise, stated rather than found later.** It is
/// not the block, it is the *partition*: a plan with more phases has more
/// intermediate images alive, and here the makespan objective buys a cut the
/// work objective did not, so the plan goes from two images to three. That is a
/// whole-volume image, not a block, and it is the term
/// `docs/design/images-and-phases.md` counts. It is priced — the cut is taken
/// only because it pays for its own `materialise_cost_per_voxel` — but it is
/// priced in *time*, and a caller whose ceiling is bytes rather than seconds
/// should say so with `budget_bytes` or keep `concurrency` at one. Asserted
/// below so the trade is visible.
#[test]
fn residency_does_not_rise_with_the_worker_count() {
    let (_, control, _) = plan(1);
    let baseline = peak(&control);
    assert_eq!(baseline, 262144.0);
    for workers in [2usize, 4, 8, 16, 40] {
        let (_, decomposition, _) = plan(workers);
        assert!(
            peak(&decomposition) <= baseline,
            "workers {workers}: peak resident bytes rose {baseline} -> {}",
            peak(&decomposition)
        );
        // No phase of any plan exceeds the control's peak either.
        for (index, bytes) in resident(&decomposition).into_iter().enumerate() {
            assert!(
                bytes <= baseline,
                "workers {workers}, phase {index}: {bytes}"
            );
        }
    }
    // The phase the search re-cut is eight times lighter than it was.
    let (_, informed, _) = plan(8);
    assert_eq!(resident(&informed), vec![262144.0, 32768.0]);
    let uncut = with_edge(&informed, 1, 64).expect("a grid");
    assert_eq!(resident(&uncut), vec![262144.0, 262144.0]);

    // And the trade the paragraph above names: one more image alive, because
    // one more phase. `n_phases + 1` images — the input and one per phase.
    assert_eq!(control.n_phases() + 1, 2);
    assert_eq!(informed.n_phases() + 1, 3);
}

/// The peak over a plan: phases are sequential, so it is the largest phase.
fn peak(decomposition: &Decomposition) -> f64 {
    resident(decomposition).into_iter().fold(0.0_f64, f64::max)
}

/// Bytes resident while one block of each phase is in flight — the figure
/// `budget_bytes` is checked against.
fn resident(decomposition: &Decomposition) -> Vec<f64> {
    decomposition
        .phases
        .iter()
        .enumerate()
        .map(|(index, phase)| {
            let bytes = decomposition.dtype_at(index).size_of() as f64;
            let mut voxels = 1.0_f64;
            let reach = phase.reach.in_voxels(phase.grid.block());
            for axis in 0..3 {
                let (lo, hi) = reach.axis(axis).bound(decomposition.volume[axis]);
                let grown = phase.grid.block()[axis] as f64 + lo as f64 + hi as f64;
                voxels *= grown.min(decomposition.volume[axis] as f64);
            }
            voxels * bytes * 2.0
        })
        .collect()
}

// ================================== 6. the objective is the one it claims ==

/// **The chosen grid is the argmin.** For every phase of the chosen plan, every
/// other candidate edge is rebuilt into that phase and the whole plan re-priced;
/// none of them is cheaper, and the one that ties is the smaller edge the
/// tie-break declines.
#[test]
fn the_plan_is_the_minimum_of_the_objective_it_claims() {
    let constraints = constraints();
    for workers in [1usize, 2, 4, 8, 16, 40] {
        let (workflow, chosen, _) = plan(workers);
        let best =
            predicted_makespan(&workflow.chain, &chosen, &[], &constraints.model, workers).unwrap();
        for index in 0..chosen.n_phases() {
            for &edge in &CANDIDATES {
                let Some(variant) = with_edge(&chosen, index, edge) else {
                    continue;
                };
                let cost =
                    predicted_makespan(&workflow.chain, &variant, &[], &constraints.model, workers)
                        .unwrap();
                assert!(
                    cost >= best,
                    "workers {workers}, phase {index} at edge {edge}: {cost} < {best}"
                );
            }
        }
    }
}

/// `decomposition` with phase `index` re-cut at `edge`, or `None` where that is
/// not a grid the planner would have offered.
fn with_edge(decomposition: &Decomposition, index: usize, edge: usize) -> Option<Decomposition> {
    let phase = &decomposition.phases[index];
    let axes =
        blockflow::decomposition::cuttable_axes(&[0], &phase.reach, decomposition.volume, edge);
    let grid = BlockGrid::along(decomposition.volume, &axes, edge).ok()?;
    let mut variant = decomposition.clone();
    variant.phases[index] = PhaseDecomposition::derive(
        phase.slots.clone(),
        phase.names.clone(),
        phase.reach.clone(),
        phase.halo.clone(),
        grid,
    );
    variant.phases[index].dtype = phase.dtype;
    Some(variant)
}

// ============================== 7. the DP's licence survives the new term ==

/// **`ceil(n / workers)` is a function of one group and nothing else**, so it
/// does not join `PartitionSearch`'s list of what would break the dynamic
/// program. Asserted the way this crate asserts it everywhere: the DP and the
/// `2^(n-1)` enumeration must return the *same* partition and the *same* grids,
/// not merely the same cost.
#[test]
fn the_dp_and_the_enumeration_agree_under_the_makespan_objective() {
    let mut compared = 0usize;
    let mut multi_phase = 0usize;
    let mut mixed_blocks = 0usize;
    for seed in 0..400u64 {
        let mut rng = Rng::new(seed);
        let n = 2 + rng.below(5);
        let volume = *rng.pick(&[[64, 16, 16], [48, 12, 12], [96, 8, 8]]);
        let slots: Vec<(usize, f64)> = (0..n)
            .map(|_| {
                (
                    *rng.pick(&[0usize, 1, 2, 5, 12, 30]),
                    *rng.pick(&[0.25f64, 1.0, 4.0, 16.0]),
                )
            })
            .collect();
        let build = || {
            Chain::sequence(
                slots
                    .iter()
                    .enumerate()
                    .map(|(i, &(reach, cost))| {
                        Chain::op(AffineOp::new(NAMES[i], 1.0, 0.0, [reach, 0, 0]).with_cost(cost))
                    })
                    .collect(),
            )
        };
        let constraints = Constraints {
            block_candidates: vec![4, 8, 16, 32, 64],
            split_axes: vec![0],
            ..constraints()
        };
        for workers in [1usize, 2, 8, 40] {
            let dp = strategy(workers, PartitionSearch::Dp)
                .decompose(&workflow(build(), volume), &constraints);
            let ex = strategy(workers, PartitionSearch::Exhaustive)
                .decompose(&workflow(build(), volume), &constraints);
            match (dp, ex) {
                (Ok(dp), Ok(ex)) => {
                    compared += 1;
                    assert_eq!(
                        dp.phases
                            .iter()
                            .map(|p| p.slots.clone())
                            .collect::<Vec<_>>(),
                        ex.phases
                            .iter()
                            .map(|p| p.slots.clone())
                            .collect::<Vec<_>>(),
                        "seed {seed}, workers {workers}: different cuts"
                    );
                    assert_eq!(
                        edges(&dp),
                        edges(&ex),
                        "seed {seed}, workers {workers}: different grids"
                    );
                    if dp.n_phases() > 1 {
                        multi_phase += 1;
                        let first = edges(&dp)[0];
                        if edges(&dp).iter().any(|&edge| edge != first) {
                            mixed_blocks += 1;
                        }
                    }
                }
                (Err(_), Err(_)) => {}
                (dp, ex) => panic!("seed {seed}, workers {workers}: {dp:?} vs {ex:?}"),
            }
        }
    }
    assert!(compared > 1000, "the sweep did not sweep: {compared}");
    assert!(multi_phase > 100, "nothing was cut: {multi_phase}");
    assert!(
        mixed_blocks > 20,
        "no generated chain got two block sizes: {mixed_blocks}"
    );
}

const NAMES: [&str; 8] = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a, T>(&mut self, from: &'a [T]) -> &'a T {
        let index = self.below(from.len());
        &from[index]
    }
}

// ================================== 8. the search cannot propose a refusal ==

/// An op that consumes a whole axis, in the source frame — the declaration
/// `tests/collapsing_phase.rs` established is a **whole-axis mandate**: a plan
/// may leave the axis whole or grant it a whole-axis halo, and there is no third
/// option.
struct WholeAxisOp;

impl BlockOp for WholeAxisOp {
    fn name(&self) -> &'static str {
        "whole-axis"
    }
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        if axis == 0 {
            volume_len
        } else {
            0
        }
    }
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()])
    }
    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
}

/// **The refusal is the net, not the plan.** A phase declaring `AxisReach::All`
/// mandates that the axis is left whole or given a whole-axis halo, and
/// `splittable_axes` drops such an axis before a candidate is priced — so no
/// worker count can talk the search into a lattice `Decomposition::check` would
/// then refuse.
#[test]
fn a_whole_axis_declaration_is_never_cut_at_any_worker_count() {
    let constraints = Constraints {
        split_axes: vec![0, 1, 2],
        ..constraints()
    };
    let chain = || {
        Chain::sequence(vec![
            Chain::op(WholeAxisOp),
            Chain::op(AffineOp::new("local", 2.0, 1.0, [0, 0, 0]).with_cost(LOCAL_COST)),
        ])
    };
    for workers in [1usize, 8, 40, 4096] {
        let decomposition = strategy(workers, PartitionSearch::Dp)
            .decompose(&workflow(chain(), VOLUME), &constraints)
            .expect("a plan");
        // `decompose` runs `Decomposition::check`, so reaching here is already
        // the claim. This says *which* axis was protected.
        let barrier = decomposition
            .phases
            .iter()
            .find(|phase| phase.names.iter().any(|name| name == "whole-axis"))
            .expect("the barrier is somebody's phase");
        assert!(
            !barrier.grid.split_axes().contains(&0),
            "workers {workers}: the consumed axis was cut"
        );
        assert_eq!(barrier.grid.block()[0], VOLUME[0], "workers {workers}");
    }
}

// ================================================ 9. nothing dropped silently ==

/// **What was searched, and what was thrown away.** A cap nobody can read is a
/// cap nobody can argue with, so the search hands its own accounting back.
#[test]
fn the_account_names_every_candidate_it_dropped() {
    let (_, _, account) = plan(8);
    assert_eq!(account.slots, 2);
    assert_eq!(account.workers, 8);
    assert_eq!(account.candidates_offered_per_run, CANDIDATES.len());
    // Three contiguous runs of two slots: 0..1, 0..2, 1..2.
    assert_eq!(account.runs_priced, 3);
    assert_eq!(account.runs_refused, 0);
    assert_eq!(account.runs_forbidden_by_barrier, 0);
    assert_eq!(account.candidates.offered, 3 * CANDIDATES.len());
    assert_eq!(
        account.candidates.priced + account.candidates.no_grid + account.candidates.over_budget,
        account.candidates.offered,
        "every offered candidate is either priced or accounted for as dropped"
    );
    assert_eq!(account.chosen.len(), 2);

    // A budget that bites is reported as a budget, not as silence. The join's
    // halo makes its phase 256 KiB at every edge, so the chain that shows this
    // is a local one, where the resident figure really is the block.
    let tight = Constraints {
        budget_bytes: Some(200_000),
        ..constraints()
    };
    let local_chain = || {
        Chain::sequence(vec![
            Chain::op(AffineOp::new("a", 2.0, 1.0, [0, 0, 0])),
            Chain::op(AffineOp::new("b", 3.0, 0.0, [0, 0, 0]).with_cost(LOCAL_COST)),
        ])
    };
    let (plan, tight_account) = strategy(8, PartitionSearch::Dp)
        .decompose_accounted(&workflow(local_chain(), VOLUME), &tight)
        .expect("a plan");
    assert!(
        tight_account.candidates.over_budget > 0,
        "a 200 kB budget dropped nothing: {tight_account:?}"
    );
    assert_eq!(
        tight_account.candidates.priced
            + tight_account.candidates.no_grid
            + tight_account.candidates.over_budget,
        tight_account.candidates.offered
    );
    assert!(plan.phases.iter().all(|phase| phase.grid.block()[0] < 64));
}
