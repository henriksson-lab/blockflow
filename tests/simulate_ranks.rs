//! **Does the simulator rank anything?**
//!
//! The module's own acceptance bar, asserted. It does not predict runtimes —
//! `Rates` carries one byte-to-seconds coefficient where the calibration corpus
//! measures a spread above an order of magnitude — so a comparison against a
//! wall clock is a category error and there is none here.
//!
//! What is testable is the thing it exists to do: **a change known to be an
//! improvement must rank as one, a change known to be neutral must rank as
//! neutral, and a quantity known to be invariant must come out invariant.** A
//! simulator that fails those ranks nothing, however plausible its numbers.
use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Constraints;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{
    simulate, BoundedHorizonThroughput, ExecutorOrder, Machine, PlanOrder, RateBasis, Rates,
    ReleaseAware, RunAhead, Scheduler, WarmestFirst,
};
use blockflow::Dtype;
use std::collections::BTreeSet;

const VOLUME: [usize; 3] = [64, 64, 64];

/// A three-phase all-pixel chain, cut on `edge`.
fn plan(edge: usize) -> blockflow::assemble::Assembly {
    let grid = BlockGrid::new(VOLUME, [edge, edge, edge]).expect("a grid");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    // A reach of one on every axis, so a block's read extent genuinely exceeds
    // its core and the cut has something to over-fetch — a zero-reach chain
    // would make every cut free and the cache term unobservable.
    for name in ["first", "second", "third"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
            .expect("a pixel phase");
    }
    builder.finish().expect("an assembly")
}

/// Chunks small enough that the volume is many of them.
///
/// **Not a detail.** At the default `64^3` chunk this volume is one chunk per
/// image, every task reads exactly it, and a cache of capacity one serves the
/// whole phase — so no cache size and no ordering can change a single fetch,
/// and three tests that looked like they were measuring the cache were
/// measuring nothing. A cache term is only observable when there is more to
/// hold than fits.
fn rates() -> Rates {
    Rates {
        chunk: [16, 16, 16],
        chunk_bytes: 16 * 16 * 16 * 8,
        ..Rates::default()
    }
}

fn run(
    edge: usize,
    machine: Machine,
    scheduler: &mut dyn Scheduler,
) -> blockflow::simulate::Outcome {
    let assembly = plan(edge);
    simulate(
        &assembly.decomposition,
        &assembly.work(),
        &machine,
        &rates(),
        &BTreeSet::new(),
        &BTreeSet::new(),
        &TILE_PHASE_RATES,
        scheduler,
    )
    .expect("a simulable plan")
}

/// Three phases at the tile run's own measured spread.
///
/// **Taken from a ruler and not invented.** The `2026-08-23` sequential tile
/// run measured `combine (union)` at `3.541`, `smooth` at `98.329` and
/// `skeletonize` at `201.397` nanoseconds per voxel — a factor of **57** across
/// one chain. A simulator run at one uniform rate cannot see a throughput term
/// at all, so a fixture that used one would be testing the term against a
/// machine where it is definitionally inert.
const TILE_PHASE_RATES: [f64; 3] = [3.541, 98.329, 201.397];

/// **Adding workers cannot lengthen the run, and must shorten a cuttable one.**
///
/// The direction is not in doubt for any real scheduler, so a simulator that
/// got it wrong would be reporting on its own event loop rather than on the
/// plan. Asserted as a strict improvement at a cut that has blocks to spare and
/// as exact equality at one block, where there is nothing to parallelise —
/// **the second half is what makes the first a finding**: a loop that simply
/// divided by the worker count would pass the strict half and fail this one.
#[test]
fn workers_shorten_a_cuttable_run_and_do_nothing_to_a_single_block() {
    let one = |w: usize| {
        run(
            64,
            Machine {
                workers: w,
                ..Machine::default()
            },
            &mut PlanOrder,
        )
        .makespan_ns
    };
    assert_eq!(
        one(1),
        one(8),
        "a one-block plan has one task per phase and nothing to run beside it, so eight workers \
         must finish exactly when one does"
    );

    let cut = |w: usize| {
        run(
            16,
            Machine {
                workers: w,
                ..Machine::default()
            },
            &mut PlanOrder,
        )
        .makespan_ns
    };
    assert!(
        cut(8) < cut(1),
        "eight workers took {} against one worker's {} on a 4x4x4 block grid",
        cut(8),
        cut(1)
    );
}

/// **Total compute is invariant to the schedule; IO is not.**
///
/// This is the premise the whole IO-penalty argument rests on — *ordering does
/// not change the work, it changes the memory and the IO* — so it is asserted
/// in both directions rather than assumed.
///
/// **Isolating it needs `io_ns_per_byte: 0`, and finding that out was the
/// point.** The first version of this test set `cache_bytes: 0` and expected
/// two schedulers to agree, on the reasoning that a cache nobody can warm
/// cannot reorder anything. `ModelledCache::new` clamps capacity to **at least
/// one chunk**, so a zero budget is a one-chunk cache and not a missing one;
/// the two schedulers thrashed it differently and disagreed by 8%. Zeroing the
/// *price* of a byte is the way to say "compute only"; zeroing the cache is
/// not.
#[test]
fn ordering_does_not_change_the_work() {
    let machine = Machine {
        workers: 1,
        cache_bytes: 0,
        prefetch_depth: 0,
    };
    let compute_only = |scheduler: &mut dyn Scheduler| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &machine,
            &Rates {
                io_ns_per_byte: 0.0,
                ..rates()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
            &TILE_PHASE_RATES,
            scheduler,
        )
        .expect("a simulable plan")
    };
    let plan_order = compute_only(&mut PlanOrder);
    let warmest = compute_only(&mut WarmestFirst);
    assert_eq!(
        plan_order.makespan_ns, warmest.makespan_ns,
        "with a byte priced at zero the makespan is pure compute, and two orderings of the same \
         tasks must cost the same"
    );
    assert_eq!(plan_order.tasks_run, warmest.tasks_run);

    // The other direction, and it is what makes the first half a finding rather
    // than a statement about an inert model: once a byte has a price, the same
    // two schedulers over the same tasks **do** differ.
    let priced_plan = run(16, machine, &mut PlanOrder);
    let priced_warm = run(16, machine, &mut WarmestFirst);
    assert_ne!(
        priced_plan.cache_misses, priced_warm.cache_misses,
        "with a one-chunk cache the two orderings induced the same misses, so this fixture \
         cannot see an ordering's effect on IO at all"
    );
}

/// **A cache that fits changes the answer; one that does not, does not.**
///
/// The control is the point. A cache term that improved the makespan at *every*
/// size would be a term that is not modelling a cache.
#[test]
fn a_cache_only_helps_when_it_is_large_enough_to_hold_something() {
    let cold = run(
        16,
        Machine {
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
        },
        &mut PlanOrder,
    );
    let warm = run(
        16,
        Machine {
            workers: 1,
            cache_bytes: 64 * 1024 * 1024 * 1024,
            prefetch_depth: 0,
        },
        &mut PlanOrder,
    );
    assert!(
        warm.fetched_bytes < cold.fetched_bytes,
        "a cache big enough for the whole volume fetched {} bytes against a cacheless run's {}",
        warm.fetched_bytes,
        cold.fetched_bytes
    );
    // **Not `cold.cache_hits == 0`.** `ModelledCache::new` clamps capacity to
    // one chunk, so a zero budget is the smallest cache and not the absence of
    // one — a run at `cache_bytes: 0` still hits whenever two consecutive tasks
    // share their last chunk. The claim is the ratio, and it is large.
    assert!(
        warm.cache_hits > cold.cache_hits * 4,
        "a cache holding the whole volume hit {} times against a one-chunk cache's {}; if these \
         are close, the cache size is not reaching the model",
        warm.cache_hits,
        cold.cache_hits
    );
}

/// **A finer cut does not lower the image floor.**
///
/// The session's central finding, restated inside the simulator so that a
/// scheduler tuned in here cannot be tuned against a machine where re-cutting
/// buys residency. Only the in-flight term moves.
#[test]
fn a_finer_cut_does_not_lower_the_image_floor() {
    let machine = Machine {
        workers: 1,
        ..Machine::default()
    };
    let coarse = run(64, machine, &mut PlanOrder);
    let fine = run(16, machine, &mut PlanOrder);
    assert!(
        fine.peak_bytes < coarse.peak_bytes,
        "the in-flight term should shrink with the cut: {} against {}",
        fine.peak_bytes,
        coarse.peak_bytes
    );
    // And what remains is the images, which are the same at both cuts. The
    // reference is `peak_image_bytes` — the walk that `residency_bar.rs`
    // calibrates against a ruler — rather than a count of images written here,
    // because intermediates are freed after their last reader and a plan does
    // **not** hold one image per phase.
    let assembly = plan(16);
    let floor = assembly
        .decomposition
        .peak_image_bytes(&assembly.work())
        .expect("an all-pixel plan");
    assert!(
        fine.peak_bytes >= floor,
        "the image walk says {floor} bytes are alive at the worst phase and the simulated run \
         peaked at {} — the simulator is freeing something the walk keeps",
        fine.peak_bytes
    );

    // **And the other side of the bracket**, which is the half that catches the
    // opposite error. The simulator keeps its own incremental image trace —
    // `peak_image_bytes_with` reports a peak and a scheduler needs the value at
    // every instant — so the two walks are separate code and could disagree in
    // either direction. Above the floor it is holding something the walk frees;
    // more than the floor plus everything that can be in flight at once, and it
    // is holding something neither accounts for.
    let ceiling = assembly
        .decomposition
        .residency(
            &Constraints {
                expected_concurrency: 1,
                ..Constraints::default()
            },
            &assembly.work(),
            &BTreeSet::new(),
            &BTreeSet::new(),
        )
        .expect("a residency")
        .total_bytes();
    assert!(
        fine.peak_bytes <= ceiling,
        "the simulated run peaked at {} against a floor of {floor} plus all the working set \
         `Decomposition::residency` allows, {ceiling}. The two image walks have diverged",
        fine.peak_bytes
    );
}

/// **Every task runs exactly once, and the run ends.**
///
/// The loop's own invariant. A scheduler that returns an out-of-range index is
/// clamped rather than trusted, so this also pins that a misbehaving scheduler
/// cannot lose or duplicate work.
#[test]
fn a_misbehaving_scheduler_cannot_lose_work() {
    /// Always asks for a slot far past the end.
    struct OutOfRange;
    impl Scheduler for OutOfRange {
        fn name(&self) -> &'static str {
            "out-of-range"
        }
        fn pick(&mut self, _decision: &blockflow::simulate::Decision<'_>) -> usize {
            usize::MAX
        }
    }
    let sane = run(
        16,
        Machine {
            workers: 4,
            ..Machine::default()
        },
        &mut PlanOrder,
    );
    let silly = run(
        16,
        Machine {
            workers: 4,
            ..Machine::default()
        },
        &mut OutOfRange,
    );
    assert_eq!(
        sane.tasks_run, silly.tasks_run,
        "both schedulers ran the same plan, so the same tasks were run. Cache misses are \
         deliberately **not** compared here: ordering changes those, which is the whole reason a \
         scheduler can matter"
    );
    assert!(silly.tasks_run > 0 && silly.makespan_ns > 0);
}

/// **Prefetch pays at depth one and is a cliff after it.**
///
/// Swept over depths 0/1/2/4/8/16/64 at three worker counts and three cache
/// sizes: **depth 1 is optimal in every configuration where prefetch can act at
/// all**, gaining `1.4%` to `3.1%`; depth 2 is already behind it; depth 4 and up
/// are worse than not prefetching. At depth 64 the run takes between `+23.7%`
/// and `+121.7%` longer.
///
/// Two mechanisms, both visible in the counters. A prefetched chunk **evicts one
/// that was about to be used** — misses rise from 1266 to 5646 at depth 64 on a
/// sixteen-chunk cache — and a run of fetches ahead makes the next **demand**
/// fetch queue behind all of it, which `io_wait_ns` shows rising four-fold.
/// Where the cache is large enough that nothing is evicted, only the second
/// mechanism operates and the cliff is much gentler.
///
/// **The model had to be fixed before this could be seen.** Its first form
/// re-tested "is the channel free" before each fetch, so the first prefetch made
/// the channel busy and every depth above 1 was inert: a sweep of seven depths
/// returned the identical row seven times, and a test asserting that deeper
/// "stops helping" passed against numbers that could not differ. A prefetcher
/// with a queue issues a run when it finds the channel free; it does not re-ask
/// after each one.
#[test]
fn prefetch_pays_at_depth_one_and_is_a_cliff_after_it() {
    let at = |depth: usize| {
        run(
            16,
            Machine {
                workers: 1,
                cache_bytes: rates().chunk_bytes * 64,
                prefetch_depth: depth,
            },
            &mut PlanOrder,
        )
    };
    let (none, one, two, deep) = (at(0), at(1), at(2), at(64));

    assert_eq!(
        none.prefetched_bytes, 0,
        "depth zero must fetch nothing ahead"
    );
    assert!(
        one.makespan_ns < none.makespan_ns,
        "depth one took {} against no prefetch at {}; if it has stopped paying, this fixture \
         never has an idle channel and the whole sweep is unobservable",
        one.makespan_ns,
        none.makespan_ns
    );
    assert!(
        two.makespan_ns > one.makespan_ns,
        "depth two took {} against depth one's {}. **If these are equal the model has regressed \
         to re-testing the channel per fetch**, which makes every depth above one inert and this \
         test vacuous",
        two.makespan_ns,
        one.makespan_ns
    );
    assert!(
        deep.makespan_ns > none.makespan_ns * 2,
        "depth 64 took {} against no prefetch at all at {}; the cliff should be steep",
        deep.makespan_ns,
        none.makespan_ns
    );
    // And the pollution half, separately from the queueing half.
    assert!(
        deep.cache_misses > none.cache_misses * 4,
        "deep prefetch induced {} misses against {}; a prefetched chunk should be evicting one \
         that was about to be used",
        deep.cache_misses,
        none.cache_misses
    );
}

/// **What the three schedulers actually do, side by side.**
///
/// The comparison the simulator exists for. Printed as well as asserted,
/// because the numbers are the finding and the assertion is only the part that
/// must not regress.
#[test]
fn the_schedulers_compared() {
    // **Sixteen chunks, and the size is the whole experiment.**
    //
    // The first version of this test used an 8 MiB cache. At a 16^3 `f64`
    // chunk that is 256 chunks, and the fixture's three phases touch 192
    // distinct ones — so the cache held *everything*, every chunk missed
    // exactly once whatever the order, and all three schedulers reported 192
    // misses. The comparison looked clean and was measuring nothing: an
    // ordering cannot show its effect on IO in a cache that never evicts.
    let chunk_bytes = rates().chunk_bytes;
    let machine = Machine {
        workers: 4,
        cache_bytes: chunk_bytes * 16,
        prefetch_depth: 0,
    };
    let horizon =
        BoundedHorizonThroughput::new(BoundedHorizonThroughput::floor_ns(&rates()) * 4, &rates())
            .expect("a horizon above the floor");

    let naive = BoundedHorizonThroughput::with_basis(
        horizon.horizon_ns(),
        &rates(),
        RateBasis::PerBlockReadExtent,
    )
    .expect("a horizon above the floor");
    let mut schedulers: Vec<Box<dyn Scheduler>> = vec![
        Box::new(PlanOrder),
        Box::new(WarmestFirst),
        Box::new(naive),
        Box::new(horizon),
        Box::new(ReleaseAware),
        Box::new(RunAhead),
        Box::new(ExecutorOrder::phase_major()),
        Box::new(ExecutorOrder::block_major()),
    ];
    eprintln!(
        "\nscheduler                     basis    makespan      misses   io_wait   idle%     peak MiB"
    );
    let mut results = Vec::new();
    for scheduler in schedulers.iter_mut() {
        let outcome = run(16, machine, scheduler.as_mut());
        eprintln!(
            "{:<26} {:>7} {:>11} {:>11} {:>9} {:>6.1}",
            scheduler.name(),
            if results.len() == 2 {
                "block"
            } else if results.len() == 3 {
                "phase"
            } else {
                "-"
            },
            outcome.makespan_ns,
            outcome.cache_misses,
            outcome.io_wait_ns,
            outcome.idle_fraction(machine.workers) * 100.0,
        );
        eprintln!("{:>92.2}", outcome.peak_bytes as f64 / 1048576.0);
        results.push((scheduler.name(), outcome));
    }

    // The conservation law holds across all three, which is what makes the
    // differences above differences in *schedule* and not in work done.
    let tasks = results[0].1.tasks_run;
    for (name, outcome) in &results {
        assert_eq!(
            outcome.tasks_run, tasks,
            "{name} ran {} tasks against {tasks}; the schedules are not over the same plan",
            outcome.tasks_run
        );
    }

    let plan_order = results[0].1;
    let warmest = results[1].1;
    let naive_rate = results[2].1;
    let per_phase = results[3].1;

    // **The finding, in two halves.**
    //
    // *Half one: the obvious throughput proxy is worse than doing nothing.*
    // `RateBasis::PerBlockReadExtent` ranks a task by its own output over its
    // own read extent — and **a read extent clamps at the volume boundary
    // while a core does not**. An interior block of this fixture reads `18^3`
    // for a `16^3` core; a face block reads less for the same core, so it
    // scores *higher*. The scheduler therefore prefers boundary blocks, which
    // are exactly the blocks with the fewest neighbours to share a cached
    // chunk with, and it scatters the traversal over the volume's surface. It
    // is not wrong about any single task — a face block really does retire
    // more output per voxel read. It is wrong about the run.
    assert!(
        naive_rate.cache_misses > plan_order.cache_misses,
        "the per-block rate induced {} misses against plan order's {}. If it has stopped losing \
         to doing nothing, the boundary preference documented here is gone",
        naive_rate.cache_misses,
        plan_order.cache_misses
    );

    // *Half two: charging the term where the work actually differs fixes it.*
    // `RateBasis::PerPhaseCost` is constant within a phase — every block of a
    // phase does the same work — so the geometric artefact above cannot enter,
    // and the term speaks only across phases, where the tile run measures a
    // **57x** spread. It then beats both the tie-break alone and plan order.
    //
    // **This half is only meaningful because `TILE_PHASE_RATES` is not flat.**
    // Under one uniform rate `PerPhaseCost` is constant everywhere, every task
    // ties, and the scheduler degenerates to `WarmestFirst` — correctly, but
    // measuring nothing. The fixture carries the measured spread for that
    // reason.
    assert!(
        per_phase.cache_misses < warmest.cache_misses,
        "the per-phase rate induced {} misses against the tie-break alone at {}; the throughput \
         term is meant to add something once it is charged where work differs",
        per_phase.cache_misses,
        warmest.cache_misses
    );
    assert!(
        per_phase.cache_misses < naive_rate.cache_misses,
        "the two bases agree ({} against {}), so this fixture cannot tell them apart",
        per_phase.cache_misses,
        naive_rate.cache_misses
    );
    assert!(
        warmest.cache_misses < plan_order.cache_misses,
        "preferring a warm task induced {} misses against plan order's {}",
        warmest.cache_misses,
        plan_order.cache_misses
    );

    // **The control.** All of the above is a finding only if the fixture could
    // have shown a difference. The first version of this test used an 8 MiB
    // cache — 256 chunks against the 192 distinct ones these three phases touch
    // — so the cache held everything, every chunk missed exactly once whatever
    // the order, and all schedulers reported precisely 192 misses. It looked
    // clean and measured nothing.
    let touches = plan_order.cache_hits + plan_order.cache_misses;
    assert!(
        per_phase.cache_misses > 192,
        "only {} misses against 192 distinct chunks and {touches} touches — the cache is holding \
         the whole volume, so no ordering can change a fetch and this comparison is vacuous",
        per_phase.cache_misses
    );
}

/// A horizon too short to contain one fetch is refused by name.
///
/// **The nonsense-strategy guard.** A scheduler that cannot see a fetch finish
/// cannot see caching pay, so it orders the run as though re-reading were free.
/// Refusing is better than scheduling that, and refusing *by name* is better
/// than refusing silently.
#[test]
fn a_horizon_below_one_fetch_is_refused() {
    let rates = rates();
    let floor = BoundedHorizonThroughput::floor_ns(&rates);
    assert!(BoundedHorizonThroughput::new(floor, &rates).is_ok());
    let err = BoundedHorizonThroughput::new(floor - 1, &rates)
        .expect_err("a horizon below the floor must be refused");
    let message = format!("{err}");
    assert!(
        message.contains("shorter than") && message.contains("fetch"),
        "the refusal should say what is too short and why: {message}"
    );
}

/// **When ordering can move the peak, and when it cannot.**
///
/// Stage 4 of the residency plan asks the scheduler to "ensure that data can be
/// released by doing things in a sensible order". This measures how much that
/// is worth, against `RunAhead` — a scheduler built to be bad at exactly this,
/// which starts the next phase the instant one block unblocks it rather than
/// finishing the phase it is in.
///
/// | workers | plan-order | run-ahead | gap |
/// |---|---|---|---|
/// | 1 | 6.09 MiB | 8.09 MiB | **33%** |
/// | 2 | 8.15 | 8.18 | 0.4% |
/// | 4 | 8.31 | 8.35 | 0.5% |
/// | 16 | 9.28 | 9.34 | 0.6% |
///
/// **Ordering is a real residency lever at one worker and almost none above
/// two.** With two or more workers the block-level dependencies force a phase
/// to overlap its successor whatever the scheduler prefers — phase `p + 1`'s
/// first block becomes ready long before phase `p`'s last one finishes — so
/// every schedule ends up holding the same images and the choice is spent.
///
/// What *does* move the peak on that axis is the **worker count itself**:
/// `6.09` to `9.28 MiB`, up 52%, and all of it working set. That is memory
/// being read, not held dead, so it is the trade the plan's §0 says to make
/// rather than the defect it says to fix.
///
/// The consequence for the plan is that **Stage 4's leverage is in the
/// partition and not in the scheduler** at the concurrency this project targets.
#[test]
fn ordering_moves_the_peak_only_at_low_concurrency() {
    let peak = |w: usize, sched: &mut dyn Scheduler| {
        run(
            16,
            Machine {
                workers: w,
                cache_bytes: rates().chunk_bytes * 16,
                prefetch_depth: 0,
            },
            sched,
        )
        .peak_bytes as f64
    };

    // At one worker the scheduler owns the interleaving completely, and a bad
    // one costs a third of the peak.
    let (one_plan, one_ahead) = (peak(1, &mut PlanOrder), peak(1, &mut RunAhead));
    assert!(
        one_ahead > one_plan * 1.2,
        "at one worker a deliberately bad order reached {one_ahead} against {one_plan}; if these \
         are close, this fixture cannot show an ordering effect on residency at all and every \
         invariance below is vacuous"
    );

    // Above two workers the choice is spent: concurrency forces the overlap the
    // scheduler was choosing.
    for w in [2usize, 4, 8, 16] {
        let (good, bad) = (peak(w, &mut PlanOrder), peak(w, &mut RunAhead));
        assert!(
            bad < good * 1.05,
            "at {w} workers the bad order reached {bad} against {good} — more than 5% apart, so \
             ordering still has residency leverage here and the finding this test records is \
             wrong"
        );
    }

    // And the axis that does move it. All working set, which is memory being
    // read rather than held dead.
    assert!(
        peak(16, &mut PlanOrder) > peak(1, &mut PlanOrder) * 1.4,
        "sixteen workers held {} against one worker's {}; the working-set term should grow with \
         the tiles in flight",
        peak(16, &mut PlanOrder),
        peak(1, &mut PlanOrder)
    );
}

/// **`SchedulePriority::BlockMajor` does not have the smaller working set, and
/// the shipped default is the right one.**
///
/// Its doc reads "Fusion, and the smaller working set". Neither half holds in
/// this executor.
///
/// *The working set.* Block-major advances one block as far through the phases
/// as its dependencies allow, so **every phase's image is live at once from
/// early in the run**. Phase-major finishes a phase — and frees what dies with
/// it — before starting the next. Measured at one worker: `8.48 MiB` against
/// `6.38`, a third **more**, not less.
///
/// *The fusion.* There is none to have. `SchedulePriority` reorders a heap; the
/// environment still allocates an image per phase either way. The claim
/// describes an execution model this crate does not implement.
///
/// *And it is slower.* Across a sweep of two cuts, four worker counts and four
/// cache sizes, block-major loses in twelve of the sixteen configurations that
/// can tell them apart, by up to **41%**. It wins by 1-2% only at forty
/// workers, where the earlier finding in this file says ordering barely matters
/// at all.
///
/// The four configurations that *cannot* tell them apart are the coarse cut —
/// at eight blocks the two orders are byte-identical in all sixteen rows,
/// because the dependency structure leaves no choice. That is why this test
/// uses the finer cut, and it is the same empty-sink trap as the cache that
/// never evicts.
#[test]
fn block_major_does_not_have_the_smaller_working_set() {
    let at = |w: usize, sched: &mut dyn Scheduler| {
        run(
            16,
            Machine {
                workers: w,
                cache_bytes: rates().chunk_bytes * 16,
                prefetch_depth: 0,
            },
            sched,
        )
    };

    // The peak, where the doc's claim is exactly backwards.
    let phase = at(1, &mut ExecutorOrder::phase_major());
    let block = at(1, &mut ExecutorOrder::block_major());
    assert!(
        block.peak_bytes > phase.peak_bytes,
        "block-major peaked at {} against phase-major's {}; if it has become the smaller of the \
         two, its doc's claim has come true and this test is what should be deleted",
        block.peak_bytes,
        phase.peak_bytes
    );

    // And the clock, where the default also wins.
    let phase4 = at(4, &mut ExecutorOrder::phase_major());
    let block4 = at(4, &mut ExecutorOrder::block_major());
    assert!(
        block4.makespan_ns > phase4.makespan_ns,
        "block-major finished in {} against phase-major's {}",
        block4.makespan_ns,
        phase4.makespan_ns
    );

    // **The default is what is being defended**, so pin that the baseline this
    // whole file compares against really is the shipped one: `PlanOrder` picks
    // the lowest ready task id, and task ids are assigned phase-major, so the
    // two must agree in every column.
    let baseline = at(4, &mut PlanOrder);
    assert_eq!(baseline.makespan_ns, phase4.makespan_ns);
    assert_eq!(baseline.cache_misses, phase4.cache_misses);
    assert_eq!(baseline.peak_bytes, phase4.peak_bytes);
}
