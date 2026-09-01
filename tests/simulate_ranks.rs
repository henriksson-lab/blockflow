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
use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::Constraints;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{
    phase_rates_from_snapshot, simulate, BoundedHorizonThroughput, ExecutorOrder, Machine,
    PerPhase, PlanOrder, RateBasis, Rates, ReleaseAware, RunAhead, Scheduler, WarmestFirst,
    MEASURED_CONTENTION,
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
        PerPhase {
            ns_per_voxel: &TILE_PHASE_RATES,
            ..PerPhase::default()
        },
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
        ..Machine::default()
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
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                ..PerPhase::default()
            },
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
            ..Machine::default()
        },
        &mut PlanOrder,
    );
    let warm = run(
        16,
        Machine {
            workers: 1,
            cache_bytes: 64 * 1024 * 1024 * 1024,
            prefetch_depth: 0,
            ..Machine::default()
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
            &assembly.workflow.chain,
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
                ..Machine::default()
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
    // **The cliff, and why the threshold is a tenth rather than a doubling.**
    //
    // It was `> 2x` and that figure was measuring a defect. The ahead-loop used
    // to skip only tasks already *started*, never asking whether a task's
    // dependencies were met — and because `TaskGraph::build` lays tasks out
    // phase-major, every depth past the end of the current phase was fetching
    // chunks of an intermediate image the producing phase had not written. Those
    // fetches cost channel time and evicted live chunks, which is most of where
    // a doubling came from; they also banked hits against data that did not
    // exist. With the loop gated on readiness the cliff is made of the two
    // things that are real — queueing ahead of demand, and eviction — and it
    // measures **1.18x** here, against depth one's 0.95x.
    //
    // Measured on this fixture: none `116.60 ms`, depth 1 `111.29 ms`, depth 2
    // `112.34 ms`, depth 64 `137.26 ms`; misses 192 against 1205.
    assert!(
        deep.makespan_ns > none.makespan_ns + none.makespan_ns / 10,
        "depth 64 took {} against no prefetch at all at {}; the cliff should be steep — a tenth          is already well past the noise floor of a deterministic simulation, which has none",
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
/// **A prefetch never fetches an image whose producing phase has not written
/// it.** The acceptance for the readiness gate in the ahead-loop.
///
/// The fixture is the one that makes the claim decidable rather than merely
/// plausible: **one block per phase**, so at any dispatch the only task that is
/// ready is the one being dispatched — every later task is a phase behind a
/// dependency that has not finished. A prefetcher gated on readiness therefore
/// issues **nothing at all**, at any depth. One gated only on "not already
/// started", which is what this used to be, would issue for every remaining
/// task in the plan and warm chunks of intermediates that do not exist yet.
///
/// So `prefetched_bytes == 0` is not a statement that prefetch is broken; it is
/// the difference between the two rules, isolated.
#[test]
fn a_prefetch_never_fetches_an_image_its_producing_phase_has_not_written() {
    let deep = run(
        64,
        Machine {
            workers: 1,
            cache_bytes: rates().chunk_bytes * 64,
            prefetch_depth: 64,
            ..Machine::default()
        },
        &mut PlanOrder,
    );
    assert_eq!(
        deep.prefetched_bytes, 0,
        "with one block per phase nothing is ever ready but the task being dispatched, so a          prefetcher that asks whether a task's dependencies are met has nothing to fetch.          Bytes here mean it is fetching an intermediate the producing phase has not written."
    );
    // And the run still happened, so the zero above is a property of the
    // prefetcher rather than of an empty plan.
    assert_eq!(deep.tasks_run, 3, "one block, three phases");
}

/// **Stored bytes are a property of the plan, not of the schedule** — the same
/// conservation law `tasks_run` carries, and the reason it is worth asserting is
/// that cache misses are *not* such a quantity.
#[test]
fn what_a_run_stores_does_not_depend_on_who_scheduled_it() {
    let machine = Machine {
        workers: 4,
        cache_bytes: rates().chunk_bytes * 8,
        prefetch_depth: 0,
        ..Machine::default()
    };
    let mut schedulers: Vec<Box<dyn Scheduler>> = vec![
        Box::new(PlanOrder),
        Box::new(WarmestFirst),
        Box::new(ExecutorOrder::block_major()),
        Box::new(ExecutorOrder::phase_major()),
        Box::new(RunAhead),
        Box::new(ReleaseAware),
    ];
    let first = run(16, machine, schedulers[0].as_mut());
    assert!(
        first.written_bytes > 0,
        "the fixture must actually store something, or this asserts nothing"
    );
    assert!(
        first.materialised_bytes > 0,
        "a three-phase chain has two intermediates, and they are stored too"
    );
    for scheduler in schedulers.iter_mut() {
        let outcome = run(16, machine, scheduler.as_mut());
        assert_eq!(
            (outcome.written_bytes, outcome.materialised_bytes),
            (first.written_bytes, first.materialised_bytes),
            "{} stored a different amount from {}, and what a plan stores is not the \
             scheduler's to change",
            scheduler.name(),
            schedulers[0].name()
        );
    }
}

/// **A store costs time, and a plan that stores more finishes later.**
///
/// Asserted through the rate rather than through two plans, so that nothing else
/// moves: the same plan, the same order, the same fetches, and only the price of
/// a stored byte different. A simulator that charged nothing for a write — which
/// this one did — returns the identical makespan for both, which is the failure
/// this catches.
#[test]
fn storing_costs_time_and_the_two_destinations_are_priced_apart() {
    let machine = Machine {
        workers: 1,
        cache_bytes: rates().chunk_bytes * 64,
        prefetch_depth: 0,
        ..Machine::default()
    };
    let at = |write: f64, materialise: f64| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &machine,
            &Rates {
                write_ns_per_byte: write,
                materialise_ns_per_byte: materialise,
                ..rates()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };

    let free = at(0.0, 0.0);
    let output_only = at(4.0, 0.0);
    let intermediates_only = at(0.0, 4.0);
    let both = at(4.0, 4.0);

    assert!(
        output_only.makespan_ns > free.makespan_ns,
        "pricing the output's bytes must lengthen the run: {} against {}",
        output_only.makespan_ns,
        free.makespan_ns
    );
    assert!(
        intermediates_only.makespan_ns > free.makespan_ns,
        "and so must pricing the intermediates': {} against {}",
        intermediates_only.makespan_ns,
        free.makespan_ns
    );
    assert!(
        both.makespan_ns > output_only.makespan_ns
            && both.makespan_ns > intermediates_only.makespan_ns,
        "the two are separate terms and both are paid: {} against {} and {}",
        both.makespan_ns,
        output_only.makespan_ns,
        intermediates_only.makespan_ns
    );
    // The byte counts are the same in all four: only the price moved.
    for outcome in [output_only, intermediates_only, both] {
        assert_eq!(
            (outcome.written_bytes, outcome.materialised_bytes),
            (free.written_bytes, free.materialised_bytes),
            "a rate change must not change what is stored"
        );
    }
    // A three-phase chain over this volume stores every block's valid region
    // once per phase, so the total is the volume in bytes, per phase.
    let volume_bytes = (VOLUME.iter().product::<usize>() * 8) as u64;
    assert_eq!(
        free.written_bytes, volume_bytes,
        "the output is written exactly once"
    );
    assert_eq!(
        free.materialised_bytes,
        volume_bytes * 2,
        "and the two intermediates once each"
    );
}

/// **A phase that reads two images is charged for two.**
///
/// The fixture is the smallest one that isolates the term: the same three-phase
/// plan twice, with only the middle phase's chain different — an identity, or
/// that identity folded with one **supplied** array. Same volume, same cut, same
/// order, same task count, and — the part a first version of this test got
/// wrong — the same **reach**, since a fan-in of two reach-zero arms reaches
/// zero and comparing it against a reach-one identity would be measuring the
/// halo instead of the second array.
///
/// Before `PhaseDecomposition::images_read` reached this loop the simulator
/// fetched the phase's own image whatever the phase read, which under-priced
/// exactly the fused multi-input arrangement a scheduler would otherwise prefer.
#[test]
fn a_phase_that_reads_two_images_is_charged_for_two() {
    let machine = Machine {
        workers: 1,
        cache_bytes: 0,
        prefetch_depth: 0,
        ..Machine::default()
    };
    let with_middle = |middle: Chain| {
        let grid = BlockGrid::new(VOLUME, [16, 16, 16]).expect("a grid");
        let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
        builder
            .pixels(Chain::op(IdentityOp::new("first", [1, 1, 1])))
            .expect("a pixel phase");
        builder.pixels(middle).expect("the phase under test");
        builder
            .pixels(Chain::op(IdentityOp::new("third", [1, 1, 1])))
            .expect("a pixel phase");
        let assembly = builder.finish().expect("an assembly");
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &machine,
            &rates(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase::default(),
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };

    let kept = || Chain::op(IdentityOp::new("kept", [0, 0, 0]));
    let one_input = with_middle(kept());
    // The phase's own input beside one supplied array, folded. The first branch
    // is an identity over the handed image rather than a second source leaf, so
    // the phase reads its input the way every other phase does.
    let two_inputs = with_middle(
        Chain::parallel(
            vec![
                kept(),
                Chain::source(ImageId::supplied(0).index(), Dtype::F64),
            ],
            Box::new(blockflow::ops::voxelwise::LogicCombine::new(
                "or",
                blockflow::ops::voxelwise::Logic::Or,
            )),
        )
        .expect("a fan-in"),
    );

    assert_eq!(
        two_inputs.tasks_run, one_input.tasks_run,
        "the plans must differ in what a block reads and in nothing else"
    );
    assert!(
        two_inputs.fetched_bytes > one_input.fetched_bytes,
        "a phase reading two arrays really does traverse two: {} against {}",
        two_inputs.fetched_bytes,
        one_input.fetched_bytes
    );
    assert!(
        two_inputs.peak_bytes > one_input.peak_bytes,
        "and holds two input tiles while it does: {} against {}",
        two_inputs.peak_bytes,
        one_input.peak_bytes
    );
}

/// **One chunk grid per image, built from that image's own extent.**
///
/// Asserted through a scheduler, because `Decision` is where the grids are
/// visible and a scheduler is the supported way to look at one. The claim is
/// exactly item 7's: `ChunkGrid::volume()` for image `i` is
/// `Decomposition::volume_at(i)`, for every image any phase reads.
///
/// The plan decimates, so the images genuinely differ in extent — against
/// `decomposition.volume` for all of them, which is what the loop used to build.
/// The failure that motivates it is an image **larger** than the plan's nominal
/// volume: `ChunkGrid::keys` clamps to `counts`, so chunks past the end of an
/// undersized lattice collapse onto the last one and are reported as hits they
/// never were.
#[test]
fn every_image_is_keyed_against_its_own_extent() {
    struct GridsSeen {
        volumes: std::sync::Arc<std::sync::Mutex<Vec<(usize, [usize; 3])>>>,
    }
    impl Scheduler for GridsSeen {
        fn name(&self) -> &'static str {
            "grids-seen"
        }
        fn pick(&mut self, decision: &blockflow::simulate::Decision<'_>) -> usize {
            let mut seen = self.volumes.lock().expect("a lock");
            for (&image, grid) in decision.grids.iter() {
                seen.push((image, grid.volume()));
            }
            0
        }
    }

    let volume = [32, 32, 32];
    let grid = BlockGrid::new(volume, [16, 16, 16]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder
        .pixels(Chain::op(blockflow::probes::DecimateOp::new("decimate", 2)))
        .expect("a decimating phase");
    // The phase after it works on the decimated lattice: `PlanBuilder` sizes a
    // phase from the grid it holds, so a shape change is a `regrid` and not
    // something an op's `output_shape` can do behind the builder's back.
    builder.regrid(BlockGrid::new([16, 32, 32], [16, 16, 16]).expect("a half-height grid"));
    builder
        .pixels(Chain::op(IdentityOp::new(
            "on the smaller lattice",
            [1, 1, 1],
        )))
        .expect("a pixel phase");
    // A third phase, so that the smaller image is **read** and not merely
    // written: `volume_at(i)` is the volume of the phase that wrote image `i`,
    // so without a reader the smaller extent never reaches a grid at all and
    // the test would pass without exercising anything.
    builder
        .pixels(Chain::op(IdentityOp::new(
            "reads the smaller one",
            [1, 1, 1],
        )))
        .expect("a pixel phase");
    let assembly = builder.finish().expect("an assembly");
    let plan = &assembly.decomposition;

    let volumes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    simulate(
        plan,
        &assembly.work(),
        &Machine::default(),
        &rates(),
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        &mut GridsSeen {
            volumes: volumes.clone(),
        },
    )
    .expect("a simulable plan");

    let seen = volumes.lock().expect("a lock");
    assert!(
        !seen.is_empty(),
        "the scheduler must have been asked at least once"
    );
    for &(image, grid_volume) in seen.iter() {
        assert_eq!(
            grid_volume,
            plan.volume_at(image),
            "image {image} is keyed against {grid_volume:?} but its extent is {:?}",
            plan.volume_at(image)
        );
    }
    // And the fixture earns its keep: the extents really are different, so a
    // single grid over `decomposition.volume` would have been wrong for one.
    let distinct: std::collections::BTreeSet<[usize; 3]> = seen.iter().map(|&(_, v)| v).collect();
    assert!(
        distinct.len() > 1,
        "a decimating plan must have images of two extents, or this test passes vacuously"
    );
}

// ------------------------------------------- the sweep over rate-space --

/// The byte-to-seconds coefficients this file's conclusions are checked over.
///
/// **Taken from the spread the corpus measures, not invented.** `simulate`'s own
/// header states that the coefficient relating bytes to seconds runs from `0.31`
/// to above `4` depending on layout, warmth and codec — which is why the module
/// ranks rather than predicts. A ranking asserted at one point inside an
/// order-of-magnitude spread is a ranking about one machine on one day; the
/// question worth answering is **over what region of that spread does it hold**.
const IO_SWEEP: &[f64] = &[0.31, 0.5, 1.0, 2.0, 4.0];

/// And the store coefficients, on the same argument. `CostModel` keeps `Write`
/// and `Materialise` apart because an intermediate compresses differently from
/// an output, so the two are swept independently rather than together.
const STORE_SWEEP: &[f64] = &[0.25, 1.0, 4.0];

/// Every `Rates` in the grid: `IO_SWEEP x STORE_SWEEP x STORE_SWEEP`.
fn rate_space() -> Vec<Rates> {
    let mut all = Vec::new();
    for &io in IO_SWEEP {
        for &write in STORE_SWEEP {
            for &materialise in STORE_SWEEP {
                all.push(Rates {
                    io_ns_per_byte: io,
                    write_ns_per_byte: write,
                    materialise_ns_per_byte: materialise,
                    ..rates()
                });
            }
        }
    }
    all
}

/// The points of the grid at which `judge` held, and the total.
fn holds_over_rate_space(mut judge: impl FnMut(&Rates) -> bool) -> (usize, usize) {
    let space = rate_space();
    let total = space.len();
    let held = space.iter().filter(|rates| judge(rates)).count();
    (held, total)
}

/// Run one plan at one point of rate-space.
fn at_rates(
    edge: usize,
    machine: Machine,
    rates: &Rates,
    scheduler: &mut dyn Scheduler,
) -> blockflow::simulate::Outcome {
    let assembly = plan(edge);
    simulate(
        &assembly.decomposition,
        &assembly.work(),
        &machine,
        rates,
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase {
            ns_per_voxel: &TILE_PHASE_RATES,
            ..PerPhase::default()
        },
        scheduler,
    )
    .expect("a simulable plan")
}

/// **Every ranking in this file states the region of rate-space it holds in.**
///
/// The module's acceptance bar is that a change known to be an improvement ranks
/// as one. That was asserted at a single point — one `io_ns_per_byte`, one set of
/// phase rates — which cannot distinguish a finding from an artefact of the
/// coefficient it was measured at. This runs the same comparisons over the whole
/// grid and asserts the region, so that a conclusion which quietly stops holding
/// at half the byte cost fails here instead of being believed.
///
/// **A comparison that holds on only part of the grid is documented, not
/// deleted.** Two of the five below are of that kind and say so.
#[test]
fn every_ranking_states_the_region_of_rate_space_it_holds_in() {
    let cached = |depth: usize| Machine {
        workers: 1,
        cache_bytes: rates().chunk_bytes * 64,
        prefetch_depth: depth,
        ..Machine::default()
    };

    // 1. More workers finish a cuttable plan sooner. Structural: it is about
    //    queueing, and no coefficient should be able to overturn it.
    let (held, total) = holds_over_rate_space(|r| {
        let one = at_rates(
            16,
            Machine {
                workers: 1,
                ..Machine::default()
            },
            r,
            &mut PlanOrder,
        );
        let eight = at_rates(
            16,
            Machine {
                workers: 8,
                ..Machine::default()
            },
            r,
            &mut PlanOrder,
        );
        eight.makespan_ns < one.makespan_ns
    });
    assert_eq!(
        (held, total),
        (total, total),
        "workers must shorten a cuttable run everywhere in rate-space, and held at only \
         {held} of {total} points — a queueing property that depends on the byte cost is not a \
         queueing property"
    );

    // 2. Ordering does not change what a run stores. A conservation law, so it
    //    holds at every point or it is not one.
    let (held, total) = holds_over_rate_space(|r| {
        let plan_order = at_rates(16, cached(0), r, &mut PlanOrder);
        let warmest = at_rates(16, cached(0), r, &mut WarmestFirst);
        plan_order.written_bytes == warmest.written_bytes
            && plan_order.materialised_bytes == warmest.materialised_bytes
            && plan_order.tasks_run == warmest.tasks_run
    });
    assert_eq!(
        (held, total),
        (total, total),
        "what a plan stores is not the scheduler's to change, and that held at only {held} of \
         {total} points"
    );

    // 3. Depth one pays. **Everywhere, measured** — 45 of 45 — which was not
    //    the expectation: a prefetch buys idle channel time, and how much there
    //    is to buy is exactly what `io_ns_per_byte` sets, so this looked like
    //    the conclusion most likely to be an artefact of one coefficient. It is
    //    not, and the assertion is the whole grid so that the day it becomes one
    //    is a failure here.
    let (held, total) = holds_over_rate_space(|r| {
        let none = at_rates(16, cached(0), r, &mut PlanOrder);
        let one = at_rates(16, cached(1), r, &mut PlanOrder);
        one.makespan_ns < none.makespan_ns
    });
    assert_eq!(
        (held, total),
        (total, total),
        "depth-one prefetch paid at only {held} of {total} points. It held at all of them when \
         this was written, so a shortfall is a change in the model rather than a fixture that \
         never had an idle channel — find which end of the sweep lost it before relaxing this."
    );

    // 4. Deep prefetch is a cliff. Also 45 of 45, and for the same reason it
    //    is worth pinning: the cliff shrank from `2x` to `1.18x` when the
    //    ahead-loop stopped fetching images their producing phase had not
    //    written, so its *size* is known to be sensitive. Its *sign* is not.
    let (held, total) = holds_over_rate_space(|r| {
        let none = at_rates(16, cached(0), r, &mut PlanOrder);
        let deep = at_rates(16, cached(64), r, &mut PlanOrder);
        deep.makespan_ns > none.makespan_ns
    });
    assert_eq!(
        (held, total),
        (total, total),
        "deep prefetch hurt at only {held} of {total} points, so the cliff this file records \
         has become an artefact of one coefficient rather than a finding"
    );

    // 5. Storing costs time. Structural once a store is priced at all, so the
    //    only points it may fail at are the ones where both store rates are the
    //    smallest the grid carries and the run is compute-bound — which is why
    //    this is asserted against a *free* store rather than across the grid.
    let (held, total) = holds_over_rate_space(|r| {
        let free = at_rates(
            16,
            cached(0),
            &Rates {
                write_ns_per_byte: 0.0,
                materialise_ns_per_byte: 0.0,
                ..*r
            },
            &mut PlanOrder,
        );
        let priced = at_rates(16, cached(0), r, &mut PlanOrder);
        priced.makespan_ns > free.makespan_ns
    });
    assert_eq!(
        (held, total),
        (total, total),
        "a priced store must lengthen the run everywhere, and did so at only {held} of {total}"
    );
}

/// **A substage count multiplies the phase's compute and nothing else.**
///
/// Asserted as an *identity* rather than an inequality, which is the strongest
/// form available and needs no threshold: running phase 1 for two substages must
/// produce exactly the run that pricing phase 1 at twice the rate produces, to
/// the nanosecond. Anything else means the count is reaching a term it should
/// not — the fetch, the store, or another phase.
///
/// The shape being asserted is `iterate`'s own: `S x (read + compute) + write`,
/// where its `read` is the block re-traversing two private buffers rather than a
/// second trip to storage, because `run_iterative_phase` writes only after the
/// loop. So the fetch happens once, the store happens once, and what repeats is
/// the compute.
#[test]
fn a_substage_count_multiplies_the_compute_and_nothing_else() {
    let machine = Machine {
        workers: 1,
        cache_bytes: rates().chunk_bytes * 8,
        prefetch_depth: 0,
        ..Machine::default()
    };
    let run_with = |ns: [f64; 3], substages: [usize; 3]| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &machine,
            &rates(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &ns,
                substages: &substages,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };

    let one = TILE_PHASE_RATES;
    let doubled_middle = [one[0], one[1] * 2.0, one[2]];

    let twice_over = run_with(one, [1, 2, 1]);
    let twice_the_rate = run_with(doubled_middle, [1, 1, 1]);
    assert_eq!(
        twice_over.makespan_ns, twice_the_rate.makespan_ns,
        "two substages must cost exactly what twice the rate costs: {} against {}",
        twice_over.makespan_ns, twice_the_rate.makespan_ns
    );

    // And the terms it must *not* have touched.
    let once = run_with(one, [1, 1, 1]);
    for repeated in [twice_over, twice_the_rate] {
        assert_eq!(
            (repeated.fetched_bytes, repeated.cache_misses),
            (once.fetched_bytes, once.cache_misses),
            "the substages ping-pong private buffers; the storage read happens once"
        );
        assert_eq!(
            (repeated.written_bytes, repeated.materialised_bytes),
            (once.written_bytes, once.materialised_bytes),
            "and the image is written once, after the loop"
        );
    }
    assert!(
        twice_over.makespan_ns > once.makespan_ns,
        "iterating twice must cost more than iterating once"
    );

    // Zero means one: a phase that is not an iteration reports `0` in
    // `Stats::substages`, and reading that as "no work" would price it away.
    assert_eq!(
        run_with(one, [0, 0, 0]).makespan_ns,
        once.makespan_ns,
        "a reported zero is a phase that is not an iteration, not a phase that computes nothing"
    );
}

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
        ..Machine::default()
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
                ..Machine::default()
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
                ..Machine::default()
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

// ------------------------------------- the rates come from a measurement --

/// A `Statistics` with a coefficient recorded enough times to be believed.
///
/// `REPRODUCTIONS` runs of the same figure, because `Snapshot::believable`
/// treats "seen once" exactly as "never seen" — a distinction this helper has to
/// respect or every assertion below would be about the fallback.
fn measured(terms: &[(blockflow::statistics::Term, f64, f64)]) -> blockflow::statistics::Snapshot {
    use blockflow::statistics::{MachineKey, Observed, RunObservations, Statistics};
    let machine = MachineKey::detect();
    let mut stats = Statistics::new();
    for _ in 0..blockflow::statistics::REPRODUCTIONS {
        let mut observed = std::collections::BTreeMap::new();
        for (term, units, nanos) in terms {
            observed.insert(
                term.clone(),
                Observed {
                    units: *units,
                    nanos: *nanos,
                },
            );
        }
        stats.record(&RunObservations {
            machine: machine.clone(),
            terms: observed,
        });
    }
    stats.snapshot(&machine)
}

/// **`Rates` come from a recorded measurement, not from a literal.**
///
/// The acceptance for closing the calibration loop. What is asserted is the
/// mapping — which term feeds which field, and what happens when a term is
/// missing or has not been reproduced — because the mapping is the part a
/// caller cannot check for itself.
#[test]
fn rates_are_read_off_a_snapshot_and_fall_back_where_it_is_silent() {
    use blockflow::statistics::Term;
    let seed = Rates::default();

    // Nothing recorded: every field is the seed, unchanged.
    let empty = measured(&[]);
    assert_eq!(Rates::from_snapshot(&empty, &seed, 8.0), seed);

    // `ReadBytes` is already nanoseconds per byte and is used as it stands.
    let snapshot = measured(&[
        (Term::ReadBytes, 1_000.0, 2_500.0),
        (Term::Write, 1_000.0, 32_000.0),
        (Term::Materialise, 1_000.0, 8_000.0),
        (Term::Compute, 1_000.0, 7_000.0),
    ]);
    let rates = Rates::from_snapshot(&snapshot, &seed, 8.0);
    assert_eq!(rates.io_ns_per_byte, 2.5, "ReadBytes is per byte already");
    assert_eq!(
        rates.write_ns_per_byte, 4.0,
        "Write is per voxel, so it is divided by the element width: 32 ns/voxel over 8 bytes"
    );
    assert_eq!(rates.materialise_ns_per_byte, 1.0);
    assert_eq!(rates.compute_ns_per_voxel, 7.0);
    // The chunk geometry is not a measurement and is carried from the seed.
    assert_eq!(
        (rates.chunk, rates.chunk_bytes),
        (seed.chunk, seed.chunk_bytes)
    );

    // Seen once is not seen: `believable` wants `REPRODUCTIONS` runs, and a
    // coefficient below that is treated exactly as one that was never recorded.
    let mut once = blockflow::statistics::Statistics::new();
    let machine = blockflow::statistics::MachineKey::detect();
    once.record(&blockflow::statistics::RunObservations {
        machine: machine.clone(),
        terms: [(
            Term::ReadBytes,
            blockflow::statistics::Observed {
                units: 1_000.0,
                nanos: 99_000.0,
            },
        )]
        .into_iter()
        .collect(),
    });
    assert_eq!(
        Rates::from_snapshot(&once.snapshot(&machine), &seed, 8.0).io_ns_per_byte,
        seed.io_ns_per_byte,
        "an unreproduced coefficient must not reach a rate; `Provenance::Unreproduced` is a \
         distinct state precisely so that it does not"
    );
}

/// **A phase's compute rate is its own family's measurement, not the run's
/// average.**
///
/// `Term::ComputeOf` was recorded and never read, and the spread it records is
/// the reason to read it: the tile run measured phases a factor of 57 apart, and
/// one uniform rate makes every throughput term inert across ready tasks.
#[test]
fn per_phase_rates_come_from_the_per_family_coefficients() {
    use blockflow::statistics::Term;
    let assembly = plan(16);
    let slots = assembly.workflow.chain.slots();
    let names: Vec<String> = assembly.decomposition.phases[0].names.clone();
    assert_eq!(names.len(), 1, "one slot per phase in this fixture");

    // One family measured, the others silent, so the fallback is visible beside
    // the measurement rather than inferred.
    let first = assembly.decomposition.phases[0].names[0].clone();
    let snapshot = measured(&[
        (Term::Compute, 1_000.0, 3_000.0),
        (Term::ComputeOf(first), 1_000.0, 50_000.0),
    ]);
    let rates = phase_rates_from_snapshot(&snapshot, &assembly.decomposition, &slots, 1.0);
    assert_eq!(rates.len(), assembly.decomposition.n_phases());
    assert!(
        rates[0] > rates[1] && rates[1] == rates[2],
        "the measured family must stand apart from the two that fell back: {rates:?}"
    );
    // And the declared cost carries through: the rate is `declared x measured`.
    let declared = slots[0].cost_per_voxel();
    assert_eq!(rates[0], declared * 50.0);
    assert_eq!(rates[1], slots[1].cost_per_voxel() * 3.0);
}

// ------------------------------------------------------ machine terms --

/// **Contention: forty workers do not go forty times faster.**
///
/// The tile run realised `2.41x` against forty requested, and the simulator
/// scaled near-linearly, so its worker axis was about a machine nobody has —
/// which is why the module header said not to read it. With
/// `MEASURED_CONTENTION` set, the axis is bounded by the figure it was fitted
/// to, and the header's warning can be replaced by a number.
#[test]
fn contention_bounds_the_worker_axis_at_what_was_measured() {
    let at = |workers: usize, contention: f64| {
        run(
            16,
            Machine {
                workers,
                cache_bytes: rates().chunk_bytes * 64,
                contention,
                ..Machine::default()
            },
            &mut PlanOrder,
        )
        .makespan_ns as f64
    };

    let ideal = at(1, 0.0) / at(40, 0.0);
    let realised = at(1, MEASURED_CONTENTION) / at(40, MEASURED_CONTENTION);
    assert!(
        ideal > 8.0,
        "with no contention the simulator should scale nearly linearly — that is the behaviour \
         the header warns about — and it reached only {ideal:.2}x"
    );
    assert!(
        (2.0..=3.0).contains(&realised),
        "at the fitted coefficient forty workers should realise about the measured 2.41x, and \
         reached {realised:.2}x"
    );
    // The coefficient is off by default, so every figure recorded about this
    // simulator before it existed still reproduces.
    assert_eq!(Machine::default().contention, 0.0);
}

/// **A per-request cost puts a floor under a small chunk.**
///
/// With cost proportional to bytes alone, halving the chunk halves the
/// over-fetch and nothing pays for the extra objects, so a chunk-size sweep
/// improves monotonically toward zero — a statement about a device with no
/// per-request cost, which is no device. With latency charged per chunk the
/// sweep has an interior optimum, which is what a real chunk-size decision has.
#[test]
fn a_chunk_size_sweep_has_an_interior_optimum_once_a_request_costs_something() {
    let sweep = |latency: f64| {
        [4usize, 8, 16, 32]
            .into_iter()
            .map(|edge| {
                let assembly = plan(16);
                let outcome = simulate(
                    &assembly.decomposition,
                    &assembly.work(),
                    &Machine {
                        workers: 1,
                        cache_bytes: 0,
                        ..Machine::default()
                    },
                    &Rates {
                        chunk: [edge, edge, edge],
                        chunk_bytes: (edge * edge * edge * 8) as u64,
                        io_latency_ns: latency,
                        ..rates()
                    },
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                    PerPhase {
                        ns_per_voxel: &TILE_PHASE_RATES,
                        ..PerPhase::default()
                    },
                    &mut PlanOrder,
                )
                .expect("a simulable plan");
                outcome.io_wait_ns
            })
            .collect::<Vec<_>>()
    };

    let free_requests = sweep(0.0);
    assert!(
        free_requests.windows(2).all(|w| w[0] <= w[1]),
        "with no per-request cost the smallest chunk must be the cheapest, monotonically: \
         {free_requests:?}"
    );

    let costly_requests = sweep(200_000.0);
    let best = costly_requests
        .iter()
        .enumerate()
        .min_by_key(|&(_, wait)| *wait)
        .map(|(index, _)| index)
        .expect("a sweep");
    assert!(
        best > 0,
        "once a request costs something the smallest chunk must stop being free, and the sweep \
         still preferred it: {costly_requests:?}"
    );
}

/// **Channels: concurrent fetches overlap, and that changes what prefetch is
/// allowed to do.**
///
/// One serial channel says parallel requests never help, which is false of every
/// filesystem and is the whole design of object storage. Two consequences, and
/// the second was the opposite of what this test first asserted:
///
/// 1. On an IO-bound plan, widening the channel shortens the run.
/// 2. The **prefetch cliff grows** with channel count, rather than softening.
///    The prefetcher's rule is "only into idle channel time"; with one channel
///    and demand fetches saturating it there is no idle time and depth 64 issues
///    *nothing at all* — measured, a cliff of exactly `1.000x`. Eight channels
///    leave idle time, the prefetcher uses it, and the pollution it causes shows
///    up: `2.185x`. So the cliff is not a property of depth alone; it is a
///    property of depth on a channel wide enough to permit it.
#[test]
fn channels_overlap_fetches_and_widen_the_prefetch_cliff() {
    let io_bound = |channels: usize, depth: usize| {
        run(
            16,
            Machine {
                workers: 4,
                cache_bytes: 0,
                prefetch_depth: depth,
                io_channels: channels,
                ..Machine::default()
            },
            &mut PlanOrder,
        )
    };

    // 1. An IO-bound plan finishes sooner when fetches overlap.
    let serial = io_bound(1, 0);
    let parallel = io_bound(8, 0);
    assert_eq!(
        serial.fetched_bytes, parallel.fetched_bytes,
        "the channel count must change when bytes move, not how many"
    );
    assert!(
        parallel.makespan_ns < serial.makespan_ns,
        "eight channels took {} against one channel's {}; concurrency has to buy something or \
         the model still says a device serves one request at a time",
        parallel.makespan_ns,
        serial.makespan_ns
    );

    // 2. And the cliff is a property of depth *on a wide enough channel*.
    let cliff = |channels: usize| {
        io_bound(channels, 64).makespan_ns as f64 / io_bound(channels, 0).makespan_ns as f64
    };
    assert_eq!(
        io_bound(1, 64).prefetched_bytes,
        0,
        "with one saturated channel the prefetcher finds no idle time and issues nothing"
    );
    assert!(
        io_bound(8, 64).prefetched_bytes > 0,
        "with eight it does, which is what makes the cliff visible at all"
    );
    assert!(
        cliff(8) > cliff(1),
        "the cliff must grow with the channel count that permits the prefetching that causes \
         it: {:.3}x on eight against {:.3}x on one",
        cliff(8),
        cliff(1)
    );
}

/// **A decode makes a fetched byte dearer, so a codec is a trade.**
///
/// A cache hit and a cache miss both cost their bytes' decode in reality; the
/// simulator charges the miss path here and the tiered hit path is the
/// two-tier-cache item. What this pins is that the term exists and is
/// proportional to what was fetched — with `0.0` shipped, so nothing recorded
/// before it moved.
#[test]
fn decoding_costs_what_was_fetched() {
    let at = |decode: f64| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &Machine {
                workers: 1,
                cache_bytes: 0,
                ..Machine::default()
            },
            &Rates {
                decode_ns_per_byte: decode,
                ..rates()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };
    let free = at(0.0);
    let costly = at(2.0);
    assert_eq!(
        free.fetched_bytes, costly.fetched_bytes,
        "a decode rate must not change what was fetched"
    );
    assert_eq!(
        costly.makespan_ns - free.makespan_ns,
        (free.fetched_bytes as f64 * 2.0) as u64,
        "and must cost exactly two nanoseconds for each byte of it"
    );
    assert_eq!(Rates::default().decode_ns_per_byte, 0.0);
}

/// **A short-circuited block skips its work but not its bytes.**
///
/// The acceptance for modelling `constant_maps_to`. The sequence being copied is
/// `strategy`'s own: the uniformity test is `env.uniform(&buf)`, so the block's
/// own image is fetched **before** anything can be skipped; the source images
/// are read inside `if !short_circuited` and are not; the compute is skipped;
/// and the block is still written, because "the block the work would have
/// produced" is a constant block that has to exist.
///
/// The reason this needs a model of the data rather than a measured scalar is
/// the last assertion: the skipped fraction is a function of the grid as well as
/// the volume, so one figure does not transfer between two decompositions — and
/// a block ladder sweeps exactly that.
#[test]
fn a_short_circuited_block_skips_the_compute_and_still_moves_its_bytes() {
    let at = |fraction: [f64; 3]| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &Machine {
                workers: 1,
                cache_bytes: 0,
                ..Machine::default()
            },
            &rates(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                constant_fraction: &fraction,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };

    let none = at([0.0, 0.0, 0.0]);
    let half = at([0.0, 0.5, 0.0]);
    let all = at([0.0, 1.0, 0.0]);

    assert_eq!(
        none.tasks_short_circuited, 0,
        "nothing skips at a fraction of zero"
    );
    assert!(
        (0 < half.tasks_short_circuited)
            && (half.tasks_short_circuited < all.tasks_short_circuited),
        "a half fraction must skip some blocks and not all: {} against {}",
        half.tasks_short_circuited,
        all.tasks_short_circuited
    );
    assert_eq!(
        all.tasks_short_circuited,
        (all.tasks_run) / 3,
        "at a fraction of one, every block of the one phase named skips"
    );

    // The work goes.
    assert!(
        all.makespan_ns < none.makespan_ns,
        "skipping a whole phase's compute has to be cheaper: {} against {}",
        all.makespan_ns,
        none.makespan_ns
    );
    // The bytes stay: the block is read to find out it is uniform, and written
    // because the constant block must exist.
    assert_eq!(
        (all.fetched_bytes, all.written_bytes, all.materialised_bytes),
        (
            none.fetched_bytes,
            none.written_bytes,
            none.materialised_bytes
        ),
        "a short circuit skips the work, not the traffic"
    );
    // And it is a property of the plan and the data, not of the schedule.
    let by_warmth = {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &Machine {
                workers: 1,
                cache_bytes: 0,
                ..Machine::default()
            },
            &rates(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                constant_fraction: &[0.0, 0.5, 0.0],
                ..PerPhase::default()
            },
            &mut WarmestFirst,
        )
        .expect("a simulable plan")
    };
    assert_eq!(
        by_warmth.tasks_short_circuited, half.tasks_short_circuited,
        "which blocks are uniform is not the scheduler's to change"
    );
}

/// **The encoded tier holds more and charges for it.**
///
/// The acceptance for two tiers. `cache.rs` sizes the trade — an encoded entry
/// survives roughly twenty times longer for the same bytes, and an encoded hit
/// is **962 us, ~100x a decoded hit**, still ~40x cheaper than storage — and the
/// consequence is a cache whose curve has a knee rather than one that improves
/// monotonically for ever.
///
/// What is asserted is the shape, not the constants: giving part of the budget
/// to the encoded tier turns fetches into decodes, which is cheaper than storage
/// and dearer than a decoded hit. The three-way ordering is the finding a
/// single-tier model could not have.
#[test]
fn the_encoded_tier_trades_fetches_for_decodes() {
    let at = |encoded_fraction: f64| {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &Machine {
                workers: 1,
                // Small enough that the decoded tier alone cannot hold the
                // working set, or there is nothing for a second tier to buy.
                cache_bytes: rates().chunk_bytes * 4,
                ..Machine::default()
            }
            .with_encoded_fraction(encoded_fraction),
            &Rates {
                decode_ns_per_byte: 0.05,
                ..rates()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };

    let one_tier = at(0.0);
    let two_tiers = at(0.5);

    assert_eq!(
        one_tier.encoded_hits, 0,
        "with no encoded share there is no encoded tier to hit"
    );
    assert!(
        two_tiers.encoded_hits > 0,
        "half the budget as encoded chunks must serve something"
    );
    assert!(
        two_tiers.cache_misses < one_tier.cache_misses,
        "the point of the encoded tier is residency: {} misses against {}",
        two_tiers.cache_misses,
        one_tier.cache_misses
    );
    assert!(
        two_tiers.fetched_bytes < one_tier.fetched_bytes,
        "and a chunk it holds is a chunk not fetched"
    );
    // Ordering, which is the whole finding: an encoded hit is dearer than a
    // decoded one and cheaper than a fetch. Priced by making the decode
    // ruinous — if an encoded hit were free this would not move.
    let ruinous_decode = {
        let assembly = plan(16);
        simulate(
            &assembly.decomposition,
            &assembly.work(),
            &Machine {
                workers: 1,
                cache_bytes: rates().chunk_bytes * 4,
                ..Machine::default()
            }
            .with_encoded_fraction(0.5),
            &Rates {
                decode_ns_per_byte: 50.0,
                ..rates()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
            PerPhase {
                ns_per_voxel: &TILE_PHASE_RATES,
                ..PerPhase::default()
            },
            &mut PlanOrder,
        )
        .expect("a simulable plan")
    };
    assert!(
        ruinous_decode.makespan_ns > two_tiers.makespan_ns,
        "an encoded hit must cost its decode, or the second tier is free residency and the \
         sweep is monotone again"
    );
    assert_eq!(
        Machine::default().encoded_fraction,
        0.0,
        "off by default, so every figure recorded before the tier existed still reproduces"
    );
}

/// **The cache budget is what physically serves a re-read, and is not a planner
/// lever.** The decision recorded on `Machine::cache_bytes`, asserted where it
/// is observable: `with_page_cache` takes it from the machine rather than from
/// the plan.
#[test]
fn the_cache_budget_comes_from_the_machine_not_the_plan() {
    let asked = Machine::default().with_page_cache();
    // Not compared against a second call: `default_budget_bytes` reads
    // `/proc/meminfo`, so two calls a moment apart legitimately differ. What is
    // asserted is that the figure came from the machine at all.
    assert!(
        asked.cache_bytes > 0,
        "the budget is free memory, by the same rule `budget` uses everywhere else, and came \
         back as nothing"
    );
    assert_eq!(
        Machine::default().cache_bytes,
        0,
        "and it is not the default, because a figure from whatever machine a test ran on would \
         make every recorded simulator number unreproducible"
    );
}

/// **A sidecar costs bytes, and a barrier holds every block's at once.**
///
/// The acceptance for sidecar traffic. `probes::BlockSummaryOp` declares
/// `SidecarSize::PerBlock(48)` — `pack_u64` of six `u64`s, whatever the block
/// holds — which is a *tight* bound and therefore one the executor checks at the
/// write site.
///
/// The gather is recorded as its own peak rather than folded into `peak_bytes`,
/// because `Residency` has no term for it yet: a figure the byte budget does not
/// know about must not silently start moving the number strategies are compared
/// on.
#[test]
fn a_fragment_phase_writes_sidecar_bytes_and_a_barrier_holds_them_all() {
    let volume = [16, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder
        .pixels(Chain::op(IdentityOp::new("before", [0, 0, 0])))
        .expect("a pixel phase");
    builder
        .fragments(blockflow::probes::BlockSummaryOp::new(
            "summary",
            "summary",
            blockflow::sidecar::Lifecycle::DeleteOnExit,
        ))
        .expect("a fragment phase");
    let assembly = builder.finish().expect("an assembly");
    let blocks = assembly.decomposition.phases[1].blocks.len() as u64;

    let outcome = simulate(
        &assembly.decomposition,
        &assembly.work(),
        &Machine {
            workers: 1,
            ..Machine::default()
        },
        &rates(),
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        &mut PlanOrder,
    )
    .expect("a simulable plan");

    assert_eq!(
        outcome.sidecar_bytes_written,
        blocks * 48,
        "every block of the fragment phase writes its declared 48 bytes"
    );
    assert_eq!(
        outcome.sidecar_gather_peak,
        blocks * 48,
        "and the gather holds all of them at once, which is the peak nothing was budgeting"
    );
    // A plan with no fragment phase declares no sidecar and is charged none —
    // so the term is the fragment phase's and not an overhead on everything.
    let pixels_only = run(
        16,
        Machine {
            workers: 1,
            ..Machine::default()
        },
        &mut PlanOrder,
    );
    assert_eq!(
        (
            pixels_only.sidecar_bytes_written,
            pixels_only.sidecar_gather_peak
        ),
        (0, 0)
    );
}

/// **An op that exceeds its declared bound is refused, by name, as it does it.**
///
/// The check that stops `SidecarSize` becoming a second copy of a truth. Without
/// it a declaration is a comment: it would disagree with the bytes the op writes
/// and nothing would notice until a budget built on it was wrong.
#[test]
fn writing_past_the_declared_sidecar_bound_is_refused() {
    use blockflow::fragment::{BlockOutput, BlockView, FragmentOp, FragmentOutput, SidecarSize};

    struct Overrun;
    impl FragmentOp for Overrun {
        fn name(&self) -> &'static str {
            "overrun"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        // No pixels, so the phase before it is free to have its image collected
        // and this fixture is about the sidecar bound and nothing else.
        fn reads_pixels(&self) -> bool {
            false
        }
        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "overrun",
                blockflow::sidecar::Lifecycle::DeleteOnExit,
                blockflow::fragment::Coverage::EveryBlock,
            )
            .sized(SidecarSize::per_block(8))]
        }
        fn apply(&self, _at: &BlockView<'_>) -> blockflow::Result<BlockOutput> {
            // Nine bytes against a declared eight.
            Ok(BlockOutput::fragment("overrun", vec![0u8; 9]))
        }
    }

    let volume = [8, 8, 8];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    // The fragment phase alone, with no pixel phase in front of it: this test is
    // about the sidecar bound, and an intermediate image would drag image
    // lifetime into it.
    builder.fragments(Overrun).expect("a fragment phase");
    let assembly = builder.finish().expect("an assembly");

    // **Not uniform.** A block whose read extent is constant short-circuits
    // before `apply` is reached, so a uniform fixture would test the short
    // circuit and report that no bound was broken.
    let input =
        blockflow::voxels::Voxels::F64(ndarray::Array3::from_shape_fn(volume, |(z, y, x)| {
            (z * 100 + y * 10 + x) as f64
        }));
    let env = blockflow::env::ArrayEnvironment::for_decomposition(
        input,
        &assembly.decomposition,
        [8, 8, 8],
    )
    .expect("an environment");
    // `execute_phases` and not `execute`: `execute` hands every phase
    // `PhaseWork::Pixels`, so a fragment op reached through it is never applied
    // at all — the block is read and written as if the phase were a chain.
    let error = blockflow::strategy::execute_phases(
        "bound",
        &assembly.workflow,
        &assembly.decomposition,
        &blockflow::strategy::Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect_err("nine bytes against a declared eight must be refused");
    let message = error.to_string();
    assert!(
        message.contains("declares at most 8") && message.contains("wrote 9"),
        "the refusal must name the bound and what was written: {message}"
    );
    assert!(
        message.contains("overrun"),
        "and the op, so a reader knows whose declaration to fix: {message}"
    );
}

/// **The handout policy is ranked by the quantity it exists to reduce.**
///
/// The acceptance for the distributed dimension. Three things have to be true at
/// once for this to mean anything, and each was missing before:
///
/// * the cache is **per worker**, because a chunk two nodes both read costs two
///   fetches whatever either caches, and one shared pool cannot express that;
/// * `Outcome::duplicated_fetches` counts exactly that;
/// * the scheduler is the **real** `handout::choose`, not a transcription of it
///   — the `priority_key` precedent, for the same reason.
///
/// The assertion is on the **sign**, matching
/// `nearest_first_handout_costs_fewer_duplicated_fetches_than_naive_pull` in
/// `src/distributed/tests.rs`, which asserts an ordering and reports the ratio.
#[test]
fn a_locality_handout_duplicates_fewer_fetches_than_naive_pull() {
    use blockflow::distributed::handout::HandoutPolicy;

    let with = |policy: HandoutPolicy| {
        run(
            8,
            Machine {
                workers: 4,
                cache_bytes: rates().chunk_bytes * 32,
                cache_shared: false,
                ..Machine::default()
            },
            &mut blockflow::simulate::Handout::new(policy),
        )
    };

    let naive = with(HandoutPolicy::Naive);
    let nearest = with(HandoutPolicy::NearestFirst);

    assert_eq!(
        naive.tasks_run, nearest.tasks_run,
        "a handout policy chooses the order, never the work"
    );
    assert!(
        naive.duplicated_fetches > 0,
        "with per-worker caches a naive pull has to duplicate something, or this fixture \
         cannot tell the policies apart"
    );
    assert!(
        nearest.duplicated_fetches < naive.duplicated_fetches,
        "nearest-first duplicated {} fetches against naive pull's {}; the sign is what \
         `src/distributed/tests.rs` measures on the real coordinator",
        nearest.duplicated_fetches,
        naive.duplicated_fetches
    );
}

/// **A shared cache cannot see a duplicated fetch**, which is why the field
/// above had to exist before a handout policy could be ranked at all.
#[test]
fn a_shared_cache_reports_no_duplication_by_construction() {
    let shared = run(
        8,
        Machine {
            workers: 4,
            cache_bytes: rates().chunk_bytes * 32,
            cache_shared: true,
            ..Machine::default()
        },
        &mut PlanOrder,
    );
    assert_eq!(
        shared.duplicated_fetches, 0,
        "one pool means one fetch per chunk, which is the optimistic reading `cache_bytes` \
         documents and the reason it cannot rank a handout"
    );
    assert!(Machine::default().cache_shared, "and it is the default");
}

/// **The barrier gather is a residency term, and it is the one that grows as
/// the cut gets finer.**
///
/// `Residency` had two terms and both fall with the block edge, so a planner
/// shrinking the working set was free to grow a peak nothing was watching. A
/// fixed per-block payload — which is what most fragment streams have, a header
/// and a per-object body — totals `n_blocks x payload`, and that rises.
#[test]
fn the_gather_is_budgeted_and_rises_as_the_cut_gets_finer() {
    use blockflow::decomposition::Constraints;

    let residency_at = |edge: usize| {
        let volume = [32, 32, 32];
        let grid = BlockGrid::new(volume, [edge, edge, edge]).expect("a grid");
        let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
        builder
            .pixels(Chain::op(IdentityOp::new("before", [0, 0, 0])))
            .expect("a pixel phase");
        builder
            .fragments(blockflow::probes::BlockSummaryOp::new(
                "summary",
                "summary",
                blockflow::sidecar::Lifecycle::DeleteOnExit,
            ))
            .expect("a fragment phase");
        let assembly = builder.finish().expect("an assembly");
        let blocks = assembly.decomposition.phases[1].blocks.len() as u64;
        let residency = assembly
            .decomposition
            .residency(
                &assembly.workflow.chain,
                &Constraints::default(),
                &assembly.work(),
                &BTreeSet::new(),
                &BTreeSet::new(),
            )
            .expect("a residency");
        (residency, blocks)
    };

    let (coarse, coarse_blocks) = residency_at(32);
    let (fine, fine_blocks) = residency_at(8);

    assert!(
        fine_blocks > coarse_blocks,
        "the finer cut must have more blocks"
    );
    assert_eq!(
        coarse.sidecar_bytes,
        coarse_blocks * 48,
        "`BlockSummaryOp` declares `PerBlock(48)`, so the gather is one per block"
    );
    assert!(
        fine.sidecar_bytes > coarse.sidecar_bytes,
        "the gather grows with the block count: {} at edge 8 against {} at edge 32",
        fine.sidecar_bytes,
        coarse.sidecar_bytes
    );
    assert!(
        fine.working_set_bytes < coarse.working_set_bytes,
        "while the working set falls — which is the whole reason this term had to exist"
    );
    assert!(
        fine.total_bytes() >= fine.image_bytes + fine.working_set_bytes,
        "and the total carries it"
    );

    // A plan with no fragment phase has no gather, so the term is the fragment
    // phase's rather than an overhead on everything.
    let volume = [32, 32, 32];
    let grid = BlockGrid::new(volume, [16, 16, 16]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder
        .pixels(Chain::op(IdentityOp::new("only", [0, 0, 0])))
        .expect("a pixel phase");
    let pixels_only = builder.finish().expect("an assembly");
    assert_eq!(
        pixels_only
            .decomposition
            .residency(
                &pixels_only.workflow.chain,
                &Constraints::default(),
                &pixels_only.work(),
                &BTreeSet::new(),
                &BTreeSet::new()
            )
            .expect("a residency")
            .sidecar_bytes,
        0
    );
}

/// **A barrier phase starts when its barrier clears, and the simulator's ready
/// set has to notice.**
///
/// `TaskGraph::is_barrier` is checked in the event loop rather than encoded as
/// edges — *"a barrier changes when a phase may start, not what it depends
/// on"* — so a barrier phase's tasks can have every dependency satisfied and
/// still not be runnable. The ready set is maintained rather than rescanned, so
/// those tasks are parked when they are admitted and released when the phase
/// they wait on completes, and **that release is a code path no other fixture
/// in this file reaches**: every plan here is pixel phases, and a pixel phase
/// is never a barrier.
///
/// It is not a hypothetical path. The first version of the maintained set
/// released phases `0..finished_phases` where the rule is `finished_phases >=
/// phase`, which is off by exactly one and is off by it in the **ordinary**
/// case — a barrier cleared by the phase immediately before it, whose tasks are
/// admitted during that phase's last completion, while `finished_phases` still
/// reads the old value. Nothing here caught it, because nothing here had a
/// barrier. Now something does.
///
/// Two assertions and a third that comes for free:
///
/// * the run **completes**. With the tasks parked for ever the loop reaches
///   "none is ready and none is running" and `simulate` returns an error, so
///   this is the assertion that would have caught the defect;
/// * every task of the barrier phase is picked **after** every task of the
///   phase before it, which is what the barrier means;
/// * and in a debug build the maintained set is compared against the full scan
///   at every dispatch, so this fixture is also what puts that oracle over a
///   barrier.
#[test]
fn a_barrier_phase_is_released_when_the_phase_before_it_completes() {
    use blockflow::decomposition::{Decomposition, PhaseDecomposition};
    use blockflow::reach::Reach;
    use std::cell::RefCell;
    use std::rc::Rc;

    let volume = [16usize, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let phase = |slot: usize, barrier: bool| {
        PhaseDecomposition::derive(
            vec![slot],
            vec![format!("phase {slot}")],
            Reach::symmetric([0, 0, 0]),
            Reach::symmetric([0, 0, 0]),
            grid.clone(),
        )
        .with_barrier(barrier)
    };
    let decomposition = Decomposition {
        volume,
        dtype: Dtype::F64,
        // The middle phase waits for the whole of the first, and the last waits
        // for nothing — so the fixture has a barrier to clear *and* a phase
        // after it, which is what makes the release observable rather than
        // merely terminal.
        phases: vec![phase(0, false), phase(1, true), phase(2, false)],
        chain_reach: [0, 0, 0],
    };
    decomposition.check().expect("an honest plan must tile");
    let blocks = grid.n_blocks();

    /// Records what it picked, and picks in plan order.
    struct Recording {
        picked: Rc<RefCell<Vec<(usize, usize)>>>,
    }
    impl Scheduler for Recording {
        fn name(&self) -> &'static str {
            "recording"
        }
        fn pick(&mut self, decision: &blockflow::simulate::Decision<'_>) -> usize {
            let id = decision.ready[0];
            self.picked
                .borrow_mut()
                .push((decision.graph.tasks[id].phase, id));
            0
        }
    }

    let picked = Rc::new(RefCell::new(Vec::new()));
    let outcome = simulate(
        &decomposition,
        &[blockflow::fragment::PhaseWork::Pixels; 3],
        &Machine {
            workers: 4,
            ..Machine::default()
        },
        &rates(),
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        &mut Recording {
            picked: picked.clone(),
        },
    )
    .expect("a plan whose barrier clears must run to completion");
    assert_eq!(outcome.tasks_run as usize, 3 * blocks);

    let picked = picked.borrow();
    let last_of_first = picked
        .iter()
        .rposition(|&(phase, _)| phase == 0)
        .expect("the first phase ran");
    let first_of_barrier = picked
        .iter()
        .position(|&(phase, _)| phase == 1)
        .expect("the barrier phase ran");
    assert!(
        last_of_first < first_of_barrier,
        "a block of the barrier phase started at pick {first_of_barrier}, before the phase it \
         waits on finished dispatching at {last_of_first}"
    );
}
