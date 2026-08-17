// SPDX-License-Identifier: MIT
//
// The acceptance criteria for `strategy::Materialising`, and the measurement it
// exists to make possible.
//
// The strategy is three things at once and this file is organised as those three
// things:
//
// 1. **The pessimistic baseline.** One phase per slot, fully blocked. It must
//    compute what `Trivial` computes, byte for byte, which is the only claim
//    every strategy in this crate is held to.
// 2. **The per-op measurement instrument, which is the point.** A fused
//    `Sequence` hides its members' costs inside one phase and `Chain::Parallel`
//    is a single `apply`, so the measurable unit is the *slot*. Under full
//    materialisation every op is its own phase, so `Event::OpApplied` times each
//    one separately and `statistics::Recorder`'s attribution is exact — with no
//    fusion that has to be broken in order to measure it.
// 3. **A free incumbent bound.** Its cost is `O(n)` and it is always feasible,
//    so it is an upper bound a pruned search can start from. Asserted here
//    against the search that exists.
//
// Reading the measurement, and the caveat that decides how
// -------------------------------------------------------
// `measured_against_declared` is `#[ignore]`d and prints a table. **Every
// absolute nanosecond figure it prints is unreliable**, and not by a little: it
// was developed on a machine carrying a load average of 39 on 40 cores, where a
// per-voxel figure is a measurement of the queue as much as of the code. What
// survives contention far better is the *ratio between two ops timed in the same
// run*, because both were queued behind the same thing. So the table's last two
// columns — each op's cost per unit of its own declared cost relative to the
// run's mean, and how far the repetitions moved — are the ones to read, and the
// ns/voxel column beside them is context, not evidence.
//
// The one figure that survived every reading of the table is the ratio between
// the two rank filters, whose elements differ by exactly the factor the constant
// claims the cost is linear in. It repeats to within a few per cent across
// separate processes, and it is a ratio of two stable rows in one run. Every
// other row has a caveat on it, and the caveats are in the tests themselves.
//
// **Nothing in this file asserts on a duration.** The assertions are all
// relationships that hold at any speed: an accounted fraction is a ratio of two
// numbers from the same run, a plan's shape is data-blind, and output equality is
// output equality. That is the same discipline `tests/statistics.rs` states, and
// it is why the ignored test *prints* rather than asserts.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array3;

use blockflow::decomposition::{predicted_cost, Constraints, CostModel};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::listener::EventListener;
use blockflow::log::Event;
use blockflow::op::Chain;
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::rank::RankFilterOp;
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::ops::voxelwise::VoxelwiseMapOp;
use blockflow::probes::IdentityOp;
use blockflow::statistics::{observed_nanos, Recorder, Term};
use blockflow::strategy::{Enumerating, Materialising, Strategy, Trivial, Workflow};
use blockflow::voxels::Voxels;

/// Small enough that the acceptance tests are fast, large enough that the
/// planner has several blocks to cut on every axis.
const VOLUME: [usize; 3] = [24, 16, 20];

/// Where the measurement runs. Bigger, because a per-voxel figure taken over a
/// block that fits in L2 is a measurement of L2.
const MEASURED_VOLUME: [usize; 3] = [64, 64, 96];

fn box3() -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).expect("an element")
}

fn box5() -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, [5, 5, 5]).expect("an element")
}

/// Four cost *shapes*, not four ops: a flat voxelwise map, a separable
/// convolution priced per tap, and a rank filter at two element sizes, priced
/// per element voxel. One `compute_scale` has to serve all of them, which is the
/// thing the table measures.
///
/// **The two rank filters are the sharpest question in the chain.**
/// `RANK_COST_PER_ELEMENT_VOXEL` asserts that the cost is linear in the number
/// of voxels in the element, so 125 of them should cost 125/27 times what 27 do.
/// Two sizes in one run is the cheapest possible test of that, and it is a
/// *ratio within one run*, which is the only kind of figure this file is willing
/// to believe on a loaded machine.
///
/// A grey-level neighbourhood op would have been a fifth shape and there is
/// deliberately none: `MorphologyOp` is a **mask** op — its kernel is a `bool`
/// kernel and it threshold-maps anything else — so putting it in an `f64` chain
/// flattens the volume to zeros and ones, after which the executor short-circuits
/// every remaining block and the instrument measures nothing at all. That is not
/// a defect in the op; it is a warning about assembling a measurement chain by
/// picking interesting-sounding names.
///
/// **Every slot is uniquely named on purpose.** `Recorder` keys its per-family
/// compute term on `Chain::display_name`, which for an op is the instance name
/// the caller gave it — so distinct names make the per-family attribution a
/// per-*slot* attribution, which is the exactness this file claims. Two slots
/// sharing a name would be summed together and the claim would quietly weaken to
/// "exact per name".
fn measured_chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::new("scale", |value| value * 0.5)),
        Chain::op(SmoothOp::new(
            "smooth",
            Gaussian::isotropic(1.0, 3.0).expect("a gaussian"),
        )),
        Chain::op(RankFilterOp::median("median3", box3())),
        Chain::op(RankFilterOp::median("median5", box5())),
        Chain::op(VoxelwiseMapOp::new("offset", |value| value + 1.0)),
    ])
}

/// The same shapes with a planning barrier in the middle, so the strategies
/// disagree about everything except the answer.
fn barrier_chain(shape: [usize; 3]) -> Chain {
    Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::new("scale", |value| value * 0.5)),
        Chain::op(SmoothOp::new(
            "smooth",
            Gaussian::isotropic(1.0, 3.0).expect("a gaussian"),
        )),
        Chain::op(IdentityOp::new("whole", shape)),
        Chain::op(VoxelwiseMapOp::new("offset", |value| value + 1.0)),
    ])
}

fn workflow(chain: Chain, shape: [usize; 3]) -> Workflow {
    Workflow::new(chain, shape, Dtype::F64)
}

/// **Structured, and that is not decoration.** The obvious pseudo-random fill —
/// `(flat * 7919) % 1013` — is what `tests/statistics.rs` uses and it is wrong
/// for this file: a sequence that decorrelates between adjacent voxels is flat
/// after a Gaussian and uniform after a dilation, so the executor legitimately
/// short-circuits every block of the last two phases and the instrument measures
/// nothing. A sawtooth at a scale the filters cannot erase does not have that
/// problem, and `Attribution::short_circuited` is asserted zero so that a future
/// edit to this function cannot quietly hollow the measurement out again.
fn input(shape: [usize; 3]) -> Voxels {
    let mut array = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for ((i, j, k), value) in array.indexed_iter_mut() {
        *value =
            ((i * 37 + j * 17 + k * 5) % 251) as f64 + (((i + 1) * (j + 2) * (k + 3)) % 31) as f64;
    }
    array.into()
}

fn constraints(candidates: Vec<usize>) -> Constraints {
    Constraints {
        block_candidates: candidates,
        split_axes: vec![0, 1, 2],
        ..Constraints::default()
    }
}

/// Plan with `strategy`, run it, and hand back the output level.
fn run(
    strategy: &dyn Strategy,
    chain: Chain,
    shape: [usize; 3],
    constraints: &Constraints,
) -> Voxels {
    let workflow = workflow(chain, shape);
    let decomposition = strategy.decompose(&workflow, constraints).expect("a plan");
    let env = ArrayEnvironment::for_decomposition(input(shape), &decomposition, [4, 4, 4])
        .expect("an environment");
    strategy
        .run(&workflow, &decomposition, &env)
        .expect("a run");
    env.level(decomposition.n_phases())
}

// ==================================================== 1. the baseline is right ==

/// **The answer does not change.** The whole point of `Trivial` is that it is
/// obviously right — one block, no seams — so a new strategy is admitted by
/// agreeing with it byte for byte, and by nothing else.
#[test]
fn the_answer_does_not_change() {
    let plain = constraints(vec![4, 8]);
    let oracle = run(&Trivial, measured_chain(), VOLUME, &plain);
    let materialised = run(&Materialising::default(), measured_chain(), VOLUME, &plain);
    assert_eq!(
        materialised, oracle,
        "materialising disagreed with the trivial oracle"
    );
}

/// And on the chain the planners disagree most about: the oracle fuses all four
/// slots into one unblocked pass, materialising cuts three times *and* blocks
/// each piece, and the barrier sits between them. Every intermediate is written
/// to a level and read back, which is three round trips the oracle never makes.
#[test]
fn the_answer_does_not_change_across_a_barrier() {
    let plain = constraints(vec![4, 8]);
    let oracle = run(&Trivial, barrier_chain(VOLUME), VOLUME, &plain);
    let materialised = run(
        &Materialising::default(),
        barrier_chain(VOLUME),
        VOLUME,
        &plain,
    );
    assert_eq!(materialised, oracle);
}

/// A concurrent, block-major materialising run is still the same answer. The
/// decomposition is binding and the hints are not, so raising concurrency may
/// reorder the walk and must not move a voxel — the property that makes
/// `concurrency: 1` a *measurement* decision rather than a correctness one.
#[test]
fn concurrency_does_not_change_the_answer() {
    let plain = constraints(vec![4, 8]);
    let oracle = run(&Trivial, measured_chain(), VOLUME, &plain);
    let busy = Materialising {
        concurrency: 4,
        priority: blockflow::strategy::SchedulePriority::BlockMajor,
    };
    assert_eq!(run(&busy, measured_chain(), VOLUME, &plain), oracle);
}

// ================================================ 2. it really is one per slot ==

/// One phase per slot, and each phase holds exactly its own slot in order.
///
/// The second half is what makes it an assertion rather than an arithmetic
/// coincidence: `n_phases == n_slots` is also true of a plan that put slot 3 in
/// phase 0 and slot 0 in phase 3.
#[test]
fn every_slot_is_its_own_phase() {
    let workflow = workflow(measured_chain(), VOLUME);
    let plan = Materialising::default()
        .decompose(&workflow, &constraints(vec![4, 8]))
        .expect("a plan");
    let slots = workflow.chain.slots();
    assert_eq!(plan.n_phases(), slots.len());
    for (index, phase) in plan.phases.iter().enumerate() {
        assert_eq!(
            phase.slots,
            vec![index],
            "phase {index} is not slot {index} alone"
        );
    }
}

/// And it is genuinely blocked, which is the half that distinguishes it from
/// `Trivial`. A baseline that materialised everything into one block per phase
/// would have no halo to re-read and would measure the wrong program.
#[test]
fn every_phase_is_blocked() {
    let workflow = workflow(measured_chain(), VOLUME);
    let plan = Materialising::default()
        .decompose(&workflow, &constraints(vec![4, 8]))
        .expect("a plan");
    for (index, phase) in plan.phases.iter().enumerate() {
        assert!(
            phase.grid.n_blocks() > 1,
            "phase {index} got a single block, so nothing about halos is being measured"
        );
    }
}

/// **A barrier does not merge with a neighbour**, which under one-phase-per-slot
/// is a statement about a rule that is not being relied on: a full-reach op is
/// alone in its phase because *every* op is, not because a special case put it
/// there. The test is here so that a future edit which starts fusing "cheap
/// adjacent slots" is caught by the barrier as well as by the count.
///
#[test]
fn a_barrier_does_not_merge_with_a_neighbour() {
    let workflow = workflow(barrier_chain(VOLUME), VOLUME);
    let plain = constraints(vec![4, 8]);
    let plan = Materialising::default()
        .decompose(&workflow, &plain)
        .expect("a plan");

    let slots = workflow.chain.slots();
    assert_eq!(plan.n_phases(), slots.len());
    // The barrier is slot 2, and it is alone.
    assert_eq!(plan.phases[2].slots, vec![2]);
    assert_eq!(plan.phases[2].names, vec!["whole".to_string()]);
    // So are its neighbours, separately.
    assert_eq!(plan.phases[1].slots, vec![1]);
    assert_eq!(plan.phases[3].slots, vec![3]);
}

/// **And the corner really was missing.** On a chain with no barrier in it at
/// all, the search fuses and this does not — strictly fewer phases for the same
/// answer. Without this the whole file could be describing a strategy the
/// planner already had under another name.
#[test]
fn the_search_fuses_where_this_does_not() {
    let workflow = workflow(measured_chain(), VOLUME);
    let plain = constraints(vec![4, 8]);
    let materialised = Materialising::default()
        .decompose(&workflow, &plain)
        .expect("a plan");
    let searched = Enumerating::default()
        .decompose(&workflow, &plain)
        .expect("a plan");
    assert_eq!(materialised.n_phases(), workflow.chain.slots().len());
    assert!(
        searched.n_phases() < materialised.n_phases(),
        "the search planned {} phases and materialising planned {}",
        searched.n_phases(),
        materialised.n_phases()
    );
}

// ============================================== 3. the incumbent, and its bound ==

/// The incumbent is an upper bound on the search, and the search beats it.
///
/// Both directions matter. That `predicted_cost` of the searched plan is no
/// dearer proves the bound is sound — a branch-and-bound pruning against
/// `incumbent_cost` cannot discard the optimum. That it is strictly cheaper
/// proves the bound is not the answer, so there is something to search for.
#[test]
fn the_incumbent_bounds_the_search_from_above() {
    let workflow = workflow(measured_chain(), VOLUME);
    let plain = constraints(vec![4, 8]);
    let incumbent = Materialising::default()
        .incumbent_cost(&workflow, &plain)
        .expect("a price");

    let searched = Enumerating::default()
        .decompose(&workflow, &plain)
        .expect("a plan");
    let best = predicted_cost(&workflow.chain, &searched, &plain.model).expect("a price");

    assert!(
        best <= incumbent,
        "the search found {best}, dearer than the free incumbent {incumbent} — the bound is \
         unsound and pruning against it would discard the optimum"
    );
    assert!(
        best < incumbent,
        "the search found exactly the materialised plan, so this chain proves nothing about \
         the bound being slack"
    );
}

/// The incumbent agrees with pricing the plan by hand: it is the same number
/// `predicted_cost` gives for the decomposition it hands out, not a second
/// estimate that could drift from it.
#[test]
fn the_incumbent_is_the_price_of_the_plan_it_hands_out() {
    let workflow = workflow(measured_chain(), VOLUME);
    let plain = constraints(vec![4, 8]);
    let strategy = Materialising::default();
    let plan = strategy.decompose(&workflow, &plain).expect("a plan");
    let by_hand = predicted_cost(&workflow.chain, &plan, &plain.model).expect("a price");
    let quoted = strategy.incumbent_cost(&workflow, &plain).expect("a price");
    assert_eq!(by_hand, quoted);
}

// ==================================== 4. the per-op attribution really is exact ==

/// What one measured run yielded, per slot.
struct Attribution {
    /// Slot order, with the name the chain gave it.
    names: Vec<String>,
    declared: Vec<f64>,
    /// Nanoseconds inside `apply`, per slot.
    nanos: Vec<f64>,
    /// Voxels the op was applied over, per slot. The block's **read** extent, so
    /// halo margins are counted once per block that recomputes them — which is
    /// the honest denominator, since that is the work that happened.
    voxels: Vec<f64>,
    /// Nanoseconds the whole `run_observed` call took.
    wall: f64,
    /// Read, write, materialise and compute totals, from the `Recorder`.
    terms: BTreeMap<Term, (f64, f64)>,
    /// Blocks the executor skipped because their input was uniform and every op
    /// in the phase declared what it maps a constant to.
    ///
    /// **Carried because it invalidates the measurement rather than because it
    /// is a bug.** A short-circuited block did no work and took no time, so it
    /// contributes to neither the numerator nor the denominator of a coefficient
    /// — `Recorder` says so in as many words — but it *did* consume wall clock,
    /// and a phase that skipped every block contributes a slot with no time at
    /// all. Either way the table would be measuring a different program from the
    /// one it names.
    short_circuited: usize,
}

impl Attribution {
    fn accounted(&self) -> f64 {
        let ops: f64 = self.nanos.iter().sum();
        let io: f64 = [Term::Read, Term::Write, Term::Materialise]
            .iter()
            .filter_map(|term| self.terms.get(term))
            .map(|(_, nanos)| nanos)
            .sum();
        (ops + io) / self.wall
    }
}

/// Plan with `Materialising`, run it with a `Recorder` and an `ExecutionLog`
/// attached, and take the per-slot figures out of both.
///
/// Both are needed and neither is redundant. The `Recorder` gives the aggregated
/// terms — read, write, materialise, and compute per family — which is what
/// calibration consumes. The raw log gives the per-slot *and per-phase* split,
/// which is what proves the attribution is exact rather than merely available.
fn attribute(chain: Chain, shape: [usize; 3], candidates: Vec<usize>) -> Attribution {
    let workflow = workflow(chain, shape);
    let strategy = Materialising::default();
    let plan = strategy
        .decompose(&workflow, &constraints(candidates))
        .expect("a plan");
    let slots = workflow.chain.slots();
    assert_eq!(plan.n_phases(), slots.len());

    let env = ArrayEnvironment::for_decomposition(input(shape), &plan, [8, 8, 8])
        .expect("an environment");
    let recorder = Arc::new(Recorder::new(&workflow.chain, &plan));
    let listeners: [Arc<dyn EventListener>; 1] = [recorder.clone()];

    let started = Instant::now();
    let stats = strategy
        .run_observed(&workflow, &plan, &env, &listeners)
        .expect("a run");
    let wall = started.elapsed().as_nanos() as f64;

    let mut nanos = vec![0.0; slots.len()];
    let mut voxels = vec![0.0; slots.len()];
    let mut short_circuited = 0usize;
    for event in stats.log.events() {
        match event {
            Event::OpApplied {
                phase,
                slot,
                over,
                duration_ns,
                ..
            } => {
                // **The exactness claim, in one line.** Under this strategy a
                // slot and its phase are the same integer, so nothing the
                // executor did could have folded two ops' time into one bucket.
                assert_eq!(phase, slot, "slot {slot} was applied in phase {phase}");
                nanos[slot] += duration_ns as f64;
                voxels[slot] += over.voxels() as f64;
            }
            Event::BlockShortCircuited { .. } => short_circuited += 1,
            _ => {}
        }
    }

    let observations = recorder.observations();
    let terms = observations
        .terms
        .iter()
        .map(|(term, observed)| (term.clone(), (observed.units, observed.nanos)))
        .collect();

    Attribution {
        names: slots.iter().map(|slot| slot.display_name()).collect(),
        declared: slots.iter().map(|slot| slot.cost_per_voxel()).collect(),
        nanos,
        voxels,
        wall,
        terms,
        short_circuited,
    }
}

/// **Every slot separately timed, and the sum accounts for the run.**
///
/// Three separate claims, and the third is the one that needed the strategy:
///
/// * every slot has a non-zero time of its own, so none was folded into a
///   neighbour;
/// * the `Recorder`'s per-family compute term exists for each slot's name, with
///   the same nanoseconds the raw log attributes to that slot — the aggregate
///   agrees with the events it was built from;
/// * the accounted fraction — op time plus read, write and materialise time,
///   over wall clock — is a fraction of one, which it can only be if the timed
///   intervals are disjoint. At `concurrency: 1` they are.
///
/// No duration is asserted on. The accounted fraction is a ratio of two figures
/// from the same run and is as true on a loaded machine as on an idle one.
#[test]
fn every_slot_is_timed_separately_and_the_sum_accounts_for_the_run() {
    let run = attribute(measured_chain(), VOLUME, vec![4, 8]);
    assert_eq!(
        run.short_circuited, 0,
        "a skipped block is unmeasured work; see `Attribution::short_circuited`"
    );

    for (index, name) in run.names.iter().enumerate() {
        assert!(
            run.nanos[index] > 0.0,
            "slot {index} ({name}) was never timed"
        );
        assert!(
            run.voxels[index] > 0.0,
            "slot {index} ({name}) covered nothing"
        );
        let family = run
            .terms
            .get(&Term::ComputeOf(name.clone()))
            .unwrap_or_else(|| panic!("the recorder has no compute term for {name}"));
        assert_eq!(
            family.1, run.nanos[index],
            "the recorder and the log disagree about how long {name} took"
        );
    }

    let accounted = run.accounted();
    println!(
        "accounted fraction: {:.3} of {:.0} ns wall",
        accounted, run.wall
    );
    assert!(
        accounted > 0.0 && accounted <= 1.0,
        "accounted fraction {accounted} is not a fraction — the intervals overlap, so the run \
         was not serial and the attribution is not a partition of its time"
    );
}

/// The same, with a barrier in the chain. Worth its own case because a barrier
/// phase is the one place a planner is tempted to treat a slot specially, and a
/// slot that got no phase of its own would be a slot whose cost is attributed to
/// whoever it was fused with.
#[test]
fn a_barrier_is_timed_separately_too() {
    let run = attribute(barrier_chain(VOLUME), VOLUME, vec![4, 8]);
    assert_eq!(run.names[2], "whole");
    assert!(run.nanos[2] > 0.0, "the barrier was never timed on its own");
}

// ================================================= the measurement, and its table ==

/// How many times the chain is run before anything is reported.
///
/// **The minimum of several, not the mean of one**, and the reason is the same
/// one `ops/cost.rs` gives for its own harness: contamination on a shared machine
/// is *one-sided*, so a sample can only be slowed, never sped, and the low-order
/// statistic is the one that is a measurement of the code rather than of the
/// queue. `Recorder` takes the work-weighted mean instead, and it is right to —
/// it is producing a coefficient that must *reproduce a total*, contention
/// included. These two are answering different questions with the same events.
///
/// The number is also what makes the `spread` column possible, and that column
/// turned out to matter more than any single figure: the first repetition pays
/// first-touch page faults on every level the environment allocated, and on the
/// cheap voxelwise slots that is worth more than a factor of two. A one-shot
/// table reports it as the op's cost.
const REPETITIONS: usize = 5;

/// **The deliverable: measured against declared, per op.**
///
/// Ignored because it does real work over a real volume, and because it prints
/// rather than asserts. Run it with
///
/// ```text
/// cargo test --release --test materialising -- --ignored --nocapture
/// ```
///
/// # How to read the table, and what not to believe
///
/// `ns/voxel` is what the op cost per voxel it was applied over, best of
/// [`REPETITIONS`]. **Do not quote it.** It is contended, it is a debug or
/// release figure depending on how the test was run, and it moves by a third
/// with codegen-unit partitioning alone — `ops/mod.rs` records that measurement,
/// and it is a warning about every absolute number in this crate, not only about
/// that module's.
///
/// `ns/unit` is that divided by the op's *declared* `cost_per_voxel`, which is
/// the quantity `CostModel::compute_scale` is a single value of. If the shipped
/// constants had every family's relative cost right, this column would be
/// **flat**. It is not, and `vs mean` says by how much: the ratio of each op's
/// `ns/unit` to the work-weighted mean over the run. A value of 3 means the
/// planner charges that op three times what it costs relative to the rest of the
/// chain; a value of 0.3 means it is charged a third of what it costs.
///
/// `spread` is the worst repetition over the best. **Read it before believing
/// the row.** A row with a spread near 1 was measured; a row with a spread over
/// 2 is a row whose op is cheap enough that what was timed is mostly memory
/// traffic and scheduling.
///
/// The I/O rows are measured through `ArrayEnvironment`, which is memory. So the
/// `materialise` figure is what a materialisation costs *as a memory pass plus
/// the cache it displaces*, which is the regime this crate's tests run in and is
/// emphatically not what it costs against storage. `CostModel` prices it as I/O;
/// the gap between those two regimes is the point, and closing it needs a run
/// against a real store, not a bigger array here.
///
/// # What it said, and how much of it to believe
///
/// Recorded from four `--release` runs of this test on a 40-core host under a
/// load average of about 39, so **every figure here is a ratio between rows of
/// one run**; the absolutes are omitted on purpose. Best-of-5 within a process,
/// then compared across processes.
///
/// * **The rank filter's element-size law is very slightly convex, and this is
///   the one figure worth acting on.** `RANK_COST_PER_ELEMENT_VOXEL` is charged
///   linearly in the element's voxel count, so `median5` (125) should cost
///   `125/27 = 4.63` times `median3` (27). Measured: **3.78, 3.88, 3.92, 3.97**
///   across four processes — call it 3.9, so the linear law **over-prices the
///   larger element by about 1.19x**. Both rows have a spread of 1.01-1.07, both
///   were timed in the same run against the same contention, and the figure
///   repeats to a few per cent. Believe it.
/// * **One `compute_scale` cannot carry this chain.**
///   `Snapshot::family_spread` reports **6.7x** between the dearest and cheapest
///   family per unit of declared cost, and even taking every unstable row at its
///   *worst* repetition the spread stays above 2.8x. The direction is not in
///   doubt: rank filtering is charged far too little relative to smoothing.
/// * **`SmoothOp` is over-priced, and by how much cannot be pinned here.**
///   `SMOOTH_COST_PER_TAP` values 21 taps at 28.77 map-units. Measured against
///   `offset` as the map unit it is worth about 17.6, and against `scale` about
///   11 — so over-priced by somewhere between **1.6x and 2.6x**. The range is
///   not measurement noise on the smoothing; it is the *reference* moving, see
///   the next point. A quiet machine would not fix this. A voxelwise map is too
///   cheap to be a stable unit of anything at this volume, and pinning
///   `SMOOTH_COST_PER_TAP` needs a reference op that is neither memory-bound nor
///   contended.
/// * **Two identical voxelwise maps disagree by a stable 1.6x**, and this is the
///   finding that limits every other row. `scale` and `offset` are both
///   `VoxelwiseMapOp`, both declared 1.0, both applied over exactly 393216
///   voxels with no halo. Measured ratio: **1.58, 1.62, 1.59, 1.59**. It is
///   reproducible, so it is not noise — it is *position in the chain*, and
///   `CostModel` has no term for it. Nothing that rests on a voxelwise map as
///   its unit can claim better than a factor of 1.6.
/// * **`materialise_cost_per_voxel = 1.0` is about right in this regime, which
///   is not the regime it was written for.** Measured against the output write
///   in the same runs, a materialisation costs **1.18-1.26x** a write — near
///   enough the same number, because through `ArrayEnvironment` both are memory.
///   The doc comment on the field is about *compression* against storage, and
///   nothing here tests that. What this does say is that the in-memory test
///   suite cannot be used to justify the constant either way.
/// * **Compute is under-priced against I/O by roughly 1.8x.** Measured
///   `compute.ns_per_unit` was 1.96-1.99 across processes with a spread of
///   1.01-1.06 — the most stable number in the whole table — against
///   `read.ns_per_voxel` at 1.08-1.20. The model ships both at 1.0. Again: an
///   in-memory statement, and a storage-backed run would move the read term by
///   orders of magnitude and this ratio with it.
///
/// **Not measured, and needing a quiet machine:** any absolute nanoseconds per
/// voxel; `SMOOTH_COST_PER_TAP` itself; `MAP_COST` as a unit; and everything
/// about `order_conflict_penalty`, which nothing in the event stream can see and
/// which no run can calibrate — see `Snapshot::calibrate`, which scales it by the
/// anchor rather than measuring it, deliberately.
#[test]
#[ignore = "measures; run with --ignored --nocapture and read the table"]
fn measured_against_declared() {
    let runs: Vec<Attribution> = (0..REPETITIONS)
        .map(|_| attribute(measured_chain(), MEASURED_VOLUME, vec![16, 32]))
        .collect();
    for run in &runs {
        assert_eq!(
            run.short_circuited, 0,
            "a skipped block is unmeasured work, and a table built over one is a table about a \
             different program"
        );
    }

    let names = &runs[0].names;
    let declared = &runs[0].declared;
    // Data-blind, so identical across repetitions — asserted rather than assumed,
    // because the per-voxel figures are divided by it.
    for run in &runs {
        assert_eq!(&run.voxels, &runs[0].voxels);
    }
    let voxels = &runs[0].voxels;

    let best = |index: usize| -> f64 {
        runs.iter()
            .map(|run| run.nanos[index])
            .fold(f64::INFINITY, f64::min)
    };
    let worst =
        |index: usize| -> f64 { runs.iter().map(|run| run.nanos[index]).fold(0.0, f64::max) };

    let total_units: f64 = declared
        .iter()
        .zip(voxels)
        .map(|(declared, voxels)| declared * voxels)
        .sum();
    let total_nanos: f64 = (0..names.len()).map(best).sum();
    let mean = total_nanos / total_units;

    println!(
        "\nvolume {MEASURED_VOLUME:?}, {} slots, one phase each, concurrency 1, best of \
         {REPETITIONS}",
        names.len()
    );
    println!(
        "EVERY ABSOLUTE FIGURE BELOW IS UNRELIABLE: taken on a machine under load. Read the \
         last two columns — a ratio within one run, and how far the repetitions moved."
    );
    println!(
        "\n{:<10} {:>10} {:>12} {:>14} {:>12} {:>10} {:>8}",
        "op", "declared", "voxels", "ns/voxel", "ns/unit", "vs mean", "spread"
    );
    for index in 0..names.len() {
        let per_unit = best(index) / (declared[index] * voxels[index]);
        println!(
            "{:<10} {:>10.2} {:>12.0} {:>14.3} {:>12.4} {:>10.2} {:>8.2}",
            names[index],
            declared[index],
            voxels[index],
            best(index) / voxels[index],
            per_unit,
            per_unit / mean,
            worst(index) / best(index),
        );
    }
    println!(
        "{:<10} {:>10} {:>12.0} {:>14} {:>12.4} {:>10.2}",
        "-- mean", "", total_units, "", mean, 1.0
    );

    println!(
        "\n{:<26} {:>14} {:>14} {:>10} {:>8}",
        "term", "units", "ns/unit", "declared", "spread"
    );
    let shipped = CostModel::default();
    for (term, quoted) in [
        (Term::Read, shipped.read_cost_per_voxel),
        (Term::Write, shipped.write_cost_per_voxel),
        (Term::Materialise, shipped.materialise_cost_per_voxel),
        (Term::Compute, shipped.compute_scale),
    ] {
        let rates: Vec<f64> = runs
            .iter()
            .filter_map(|run| run.terms.get(&term))
            .map(|(units, nanos)| nanos / units)
            .collect();
        match (
            rates.iter().copied().fold(f64::INFINITY, f64::min),
            rates.len(),
        ) {
            (_, 0) => println!(
                "{:<26} {:>14} {:>14} {:>10.2}",
                term.key(),
                "-",
                "-",
                quoted
            ),
            (low, _) => println!(
                "{:<26} {:>14.0} {:>14.4} {:>10.2} {:>8.2}",
                term.key(),
                runs[0].terms[&term].0,
                low,
                quoted,
                rates.iter().copied().fold(0.0, f64::max) / low,
            ),
        }
    }

    let accounted: Vec<f64> = runs.iter().map(Attribution::accounted).collect();
    println!(
        "\naccounted fraction: {:.4} to {:.4} over {REPETITIONS} repetitions",
        accounted.iter().copied().fold(f64::INFINITY, f64::min),
        accounted.iter().copied().fold(0.0, f64::max),
    );
    println!(
        "  best-of op time {:.0} ns against a wall clock of {:.0} ns",
        total_nanos, runs[0].wall
    );
}

/// The same measurement, through the store rather than through the raw log, so
/// that what is reported is exactly what a planner would consume.
///
/// `Snapshot::family_spread` is the one number that says what `CostModel` cannot
/// express: every family's coefficient is nanoseconds per unit of *declared*
/// cost, so a spread of 1.0 would mean the constants have the relative costs
/// right and one `compute_scale` serves all of them. Anything else is the size of
/// the correction the model has nowhere to put.
#[test]
#[ignore = "measures; run with --ignored --nocapture"]
fn the_store_sees_what_the_log_sees() {
    let workflow = workflow(measured_chain(), MEASURED_VOLUME);
    let strategy = Materialising::default();
    let plain = constraints(vec![16, 32]);
    let plan = strategy.decompose(&workflow, &plain).expect("a plan");

    let mut store = blockflow::statistics::Statistics::new();
    let mut machine = None;
    let mut last = None;
    // `REPETITIONS` runs recorded, because that is what the store is *for*: a
    // coefficient is the work-weighted mean over the retained history, so one
    // run recorded here would report the first-touch page faults as the cost of
    // the cheap slots exactly as a one-shot table would.
    for _ in 0..REPETITIONS {
        let env = ArrayEnvironment::for_decomposition(input(MEASURED_VOLUME), &plan, [8, 8, 8])
            .expect("an environment");
        let recorder = Arc::new(Recorder::new(&workflow.chain, &plan));
        let listeners: [Arc<dyn EventListener>; 1] = [recorder.clone()];
        let stats = strategy
            .run_observed(&workflow, &plan, &env, &listeners)
            .expect("a run");
        store.record(&recorder.observations());
        machine = Some(recorder.machine().clone());
        last = Some(stats);
    }
    let stats = last.expect("a run");
    let snapshot = store.snapshot(&machine.expect("a machine"));
    println!("\n{}", snapshot.describe());
    match snapshot.family_spread() {
        Some(spread) => println!(
            "family spread: {spread:.2}x — the dearest family costs that many times the \
             cheapest, per unit of declared cost, and `CostModel` has one scalar for both"
        ),
        None => println!("family spread: not enough families"),
    }
    println!(
        "calibrated model: {:?}",
        snapshot.calibrate(&CostModel::default())
    );
    println!(
        "observed {:.0} ns over the event stream",
        observed_nanos(&stats.log)
    );
}
