// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **More than one computer**, and the three things that stop at a machine's
// edge.
//
// `simulate` modelled a worker pool: `workers` slots, one page cache, one set of
// IO channels, one contention coefficient over all of them. That is a thread
// pool on one machine, and it is not what `distributed` runs — a coordinator
// hands blocks to workers on *separate computers*, and three of the model's
// quantities do not cross between them:
//
// | | one machine | many |
// |---|---|---|
// | the page cache | two workers reading a chunk pay once | **each node pays**, and the second is `Outcome::duplicated_fetches` |
// | the IO channel | `io_channels` shared by everyone | each node has its own link |
// | memory bandwidth | every worker slows every other | only the workers of one machine |
//
// `Machine::nodes` is the field, `worker % nodes` is the topology, and this file
// is what says each of the three actually stops at the boundary. Every test here
// holds the **worker count fixed** and varies only how those workers are
// distributed, because that is the only comparison in which the node structure
// is the single difference — comparing eight workers on one machine against
// eighty on ten would be measuring the eighty.
//
// What this is not: a claim about *speed-up*. `simulate`'s own header warns that
// workers do not contend unless told to and that the worker-count axis is not a
// speed-up curve. Adding computers here makes a run finish sooner for the same
// reason it makes it fetch more, and only the second is a finding.

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{
    ExecutorOrder, Machine, Outcome, Rates, Run, Scheduler, MEASURED_CONTENTION,
};
use blockflow::Dtype;

const VOLUME: [usize; 3] = [64, 64, 64];

/// Chunks small enough that a block's read extent spans several of them and two
/// blocks share some — which is what makes "who fetched it first" a question at
/// all. At one chunk per image every task reads the same one and a node
/// boundary would be invisible.
const CHUNK: [usize; 3] = [16, 16, 16];

/// A three-phase pixel chain with a reach, so blocks overlap and the halo
/// chunks are the shared ones.
fn plan(edge: usize) -> Assembly {
    let grid = BlockGrid::new(VOLUME, [edge, edge, edge]).expect("a grid");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    for name in ["first", "second", "third"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
            .expect("a pixel phase");
    }
    builder.finish().expect("an assembly")
}

fn rates() -> Rates {
    Rates {
        chunk: CHUNK,
        chunk_bytes: (CHUNK.iter().product::<usize>() * 8) as u64,
        ..Rates::default()
    }
}

fn run(assembly: &Assembly, machine: Machine, scheduler: &mut dyn Scheduler) -> Outcome {
    Run::new(&assembly.decomposition, &assembly.work())
        .machine(machine)
        .rates(rates())
        .go(scheduler)
        .expect("a simulable plan")
}

/// A plan whose neighbouring blocks genuinely **share chunks**, on a volume no
/// small cache can hold — the fixture for every question about ordering.
///
/// Both of those are load-bearing and neither is true of [`plan`] above, which
/// is why this exists rather than reusing it:
///
/// * **a reach of two, not one.** At reach one a `16^3` block's read extent
///   spans the same eight `16^3` chunks whichever neighbour it sits by, so no
///   ordering can change what is fetched — measured, that fixture's answer does
///   not move across a sixteen-fold cache range. Two spans two or three chunks
///   per axis, so neighbours overlap and an ordering has something to win;
/// * **`96^3`, not this file's `64^3`.** Sixty-four `16^3` chunks is two
///   megabytes, which fits in every cache the sweeps try, so the cache axis
///   would be flat by construction. `96^3` is 216 chunks.
///
/// Equal costs across the three phases, so that a short-circuited block is a
/// real saving rather than a rounding one.
fn sharing_fixture() -> Assembly {
    const WIDE: [usize; 3] = [96, 96, 96];
    let grid = BlockGrid::new(WIDE, [16, 16, 16]).expect("a grid");
    let mut builder = PlanBuilder::new(WIDE, Dtype::F64, grid);
    for name in ["first", "second", "third"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [2, 2, 2]).with_cost(2.0)))
            .expect("a pixel phase");
    }
    builder.finish().expect("an assembly")
}

/// Eight workers, a cache big enough to hold a useful part of the volume, and
/// nothing else set — the machine every test below varies one field of.
fn eight_workers(nodes: usize) -> Machine {
    Machine {
        nodes,
        // Continuous dispatch, which is what every other figure in this file is
        // taken under; the wave discipline is `tests/wave_dispatch.rs`.
        wave_synchronous: false,
        workers: 8,
        // Per node. Half of one image's chunks at this volume — 64 chunks of
        // 32 KiB is 2 MiB — so that eviction is real and a node's own re-reads
        // still hit.
        cache_bytes: 1 << 20,
        prefetch_depth: 0,
        io_channels: 1,
        cache_shared: true,
        encoded_fraction: 0.0,
        contention: 0.0,
        // Unbounded: the scheduler sees every ready task, which is what every
        // figure in this file was recorded under.
        candidate_window: 0,
    }
}

// ------------------------------------------- 1. the cache stops at a node --

/// **A chunk two computers both read is fetched twice**, and that is the whole
/// of what a single-machine simulation could not see.
///
/// The worker count is fixed at eight and only their distribution changes. What
/// must rise is the *fetch* count — `cache_misses` and `fetched_bytes` — because
/// the same work now has to cross the wire once per machine that touches a
/// chunk; `duplicated_fetches` is the same quantity named as what it costs.
///
/// Recorded, on a `64^3` volume in `16^3` chunks at block edge 16:
///
/// ```text
///     nodes   misses   duplicated   fetched MiB   hits
///         1      462            0          14.4   2538
///         2      805          421          25.2   2195
///         4     1195          900          37.3   1805
///         8     1508         1290          47.1   1492
/// ```
///
/// **3.26x the bytes off storage at eight computers, for the same work and the
/// same tasks.** Nothing about the plan changed; the volume was simply read by
/// eight machines that cannot lend each other a chunk. That is the term a
/// planner with no node concept cannot price, and the reason a plan cut for one
/// machine is not a plan for a cluster — see `tests/cost_scenarios.rs`, where a
/// single-machine plan runs at 1.86x on ten nodes.
#[test]
fn a_chunk_two_computers_read_is_fetched_twice() {
    let assembly = plan(16);
    println!(
        "{:>7} {:>8} {:>12} {:>13} {:>7}",
        "nodes", "misses", "duplicated", "fetched MiB", "hits"
    );
    let mut previous: Option<Outcome> = None;
    for nodes in [1usize, 2, 4, 8] {
        let outcome = run(
            &assembly,
            eight_workers(nodes),
            &mut ExecutorOrder::phase_major(),
        );
        println!(
            "{nodes:>7} {:>8} {:>12} {:>13.1} {:>7}",
            outcome.cache_misses,
            outcome.duplicated_fetches,
            outcome.fetched_bytes as f64 / (1024.0 * 1024.0),
            outcome.cache_hits
        );
        // The conservation law holds however the workers are spread: the plan
        // decides the tasks, not the topology.
        assert_eq!(
            outcome.tasks_run as usize,
            assembly.decomposition.n_tasks(),
            "{nodes} nodes ran a different number of tasks"
        );
        assert_eq!(
            outcome.written_bytes + outcome.materialised_bytes,
            previous
                .map(|p| p.written_bytes + p.materialised_bytes)
                .unwrap_or(outcome.written_bytes + outcome.materialised_bytes),
            "{nodes} nodes stored a different volume"
        );
        if let Some(before) = previous {
            assert!(
                outcome.cache_misses > before.cache_misses,
                "{nodes} nodes fetched no more than the arrangement below it, so the pools are \
                 not separate"
            );
            assert!(
                outcome.duplicated_fetches > before.duplicated_fetches,
                "{nodes} nodes duplicated no more work"
            );
        } else {
            assert_eq!(
                outcome.duplicated_fetches, 0,
                "one node with a shared cache re-fetches only what it evicted"
            );
        }
        previous = Some(outcome);
    }
    let eight = previous.expect("the sweep ran");
    let one = run(
        &assembly,
        eight_workers(1),
        &mut ExecutorOrder::phase_major(),
    );
    let amplification = eight.fetched_bytes as f64 / one.fetched_bytes as f64;
    println!("eight computers fetch {amplification:.2}x the bytes one does, for the same work");
    assert!(
        amplification > 1.5,
        "eight separate page caches cost only {amplification:.2}x the traffic of one, which is \
         less than this fixture recorded (3.26x). Either the pools are sharing something or \
         the fixture stopped having chunks that two blocks both read."
    );
}

// ---------------------------------------- 2. the IO channel stops at one --

/// **Each computer has its own link to storage.**
///
/// One channel per node, so the same eight workers spread over four machines
/// fetch on four channels rather than queueing on one. The observable is
/// `io_wait_ns`: the time a worker spent waiting on the channel before its
/// compute could start, which is the quantity the channel model exists to
/// produce.
///
/// The comparison is deliberately against a **single-channel** machine. At the
/// default four channels the one-node arrangement already overlaps its fetches
/// and the effect is muddied by a lever that is not the node count.
///
/// **The cache is switched off for this one**, which is what isolates the
/// channel. With a cache, more nodes means more duplicated fetches — the
/// finding above — so the bytes to be moved grow along with the links that move
/// them, and the two come out tangled: on this fixture with a 1 MiB cache,
/// eight computers wait *longer* than one, because 3.3x the traffic beats 8x
/// the links. That is a true statement about a cluster and a useless one about
/// a channel. With no cache every arrangement fetches exactly the same chunks —
/// asserted below, not assumed — and the only difference left is how many links
/// they cross.
#[test]
fn each_computer_fetches_on_its_own_channel() {
    let assembly = plan(16);
    let waits: Vec<(usize, u64, u64)> = [1usize, 2, 4, 8]
        .into_iter()
        .map(|nodes| {
            let outcome = run(
                &assembly,
                Machine {
                    cache_bytes: 0,
                    ..eight_workers(nodes)
                },
                &mut ExecutorOrder::phase_major(),
            );
            (nodes, outcome.io_wait_ns, outcome.cache_misses)
        })
        .collect();
    for (nodes, wait, misses) in &waits {
        println!(
            "{nodes} node(s): {:.3} ms waiting on the channel, {misses} fetches",
            *wait as f64 / 1e6
        );
    }
    // Equal, because `cache_bytes: 0` means no modelled residency: every pool
    // fetches every chunk it touches. What matters is that the fetch counts are
    // the same work; the wait below differs by 6.6x.
    let misses = waits[0].2 as f64;
    for (nodes, _, other) in &waits {
        let apart = (*other as f64 - misses).abs() / misses;
        assert!(
            apart == 0.0,
            "{nodes} nodes fetched {other} chunks against {misses}, {:.1}% apart. With a cache \
             this check would mostly be measuring locality, not channel assignment.",
            apart * 100.0
        );
    }
    let one = waits[0].1;
    let many = waits[waits.len() - 1].1;
    assert!(
        many < one,
        "eight computers waited {many} ns on their own channels against {one} on one shared \
         one, over the same fetches. More links must not wait longer."
    );
}

// ------------------------------------------ 3. contention stops at a node --

/// **A worker slows for the other workers on its own machine, and for no
/// others.**
///
/// `MEASURED_CONTENTION` was fitted to one machine's realised concurrency —
/// 2.41x against forty workers — so applying it across a cluster would charge a
/// worker in Stockholm for a worker in Uppsala. Eight workers on eight machines
/// are eight lone workers, and their compute must be un-slowed.
///
/// The test is exact, not directional: at one worker per node the slowdown
/// factor is `1 + a * 0` for every task, so the makespan must equal the same run
/// with contention switched off entirely.
#[test]
fn contention_counts_only_the_workers_of_one_node() {
    let assembly = plan(16);
    let contended = |nodes: usize, contention: f64| {
        run(
            &assembly,
            Machine {
                contention,
                ..eight_workers(nodes)
            },
            &mut ExecutorOrder::phase_major(),
        )
    };

    // One computer: the coefficient applies to all eight workers and costs
    // time.
    let together = contended(1, MEASURED_CONTENTION);
    let alone = contended(1, 0.0);
    assert!(
        together.makespan_ns > alone.makespan_ns,
        "contention on one machine cost nothing, so the coefficient is not reaching the compute"
    );

    // Eight computers, one worker each: nobody has a neighbour, so the
    // coefficient must be inert.
    let spread = contended(8, MEASURED_CONTENTION);
    let spread_free = contended(8, 0.0);
    assert_eq!(
        spread.makespan_ns, spread_free.makespan_ns,
        "a worker alone on its machine was charged for workers on other machines"
    );
    println!(
        "eight workers: {:.3} ms on one machine, {:.3} ms on eight, at contention {MEASURED_CONTENTION}",
        together.makespan_ns as f64 / 1e6,
        spread.makespan_ns as f64 / 1e6
    );
}

// --------------------------------------------- 4. one node is unchanged --

/// **At one node nothing moved.** Every figure this crate has recorded was taken
/// before `Machine::nodes` existed, and they are all at one node; a change that
/// perturbed them would make the whole record unreadable.
///
/// The `Machine::default` arrangement and a `nodes: 1` one written out in full
/// must be the same run, over both shipped schedulers and at several block
/// edges — which is as close to "the old code" as a test can get without
/// keeping a copy of it.
#[test]
fn one_node_is_the_machine_this_crate_has_always_simulated() {
    for edge in [8usize, 16, 32] {
        let assembly = plan(edge);
        for (name, mut scheduler) in [
            ("phase-major", ExecutorOrder::phase_major()),
            ("block-major", ExecutorOrder::block_major()),
        ] {
            let stated = run(&assembly, eight_workers(1), &mut scheduler);
            let defaulted = run(
                &assembly,
                Machine {
                    workers: 8,
                    cache_bytes: 1 << 20,
                    prefetch_depth: 0,
                    io_channels: 1,
                    ..Machine::default()
                },
                &mut scheduler,
            );
            assert_eq!(
                stated, defaulted,
                "edge {edge}, {name}: stating `nodes: 1` is not the default machine"
            );
        }
    }
}

// ------------------------------------- 5. and therefore placement matters --

/// **With separate page caches, *which computer* gets a block is worth
/// something** — and with one shared cache it is worth nothing.
///
/// This is the pattern the node boundary forces out. `distributed::handout`'s
/// policies have existed all along and `simulate` could rank them only under
/// `cache_shared: false`, which is one cache per *slot* — the pessimistic
/// reading, and a machine nobody has. The physical arrangement is a pool per
/// computer, and it is the one where a policy that keeps a block near the node
/// already holding its chunks has something real to save.
///
/// Both halves are asserted, and the second is what makes the first mean
/// anything:
///
/// * on four computers, `NearestFirst` duplicates **342** fetches against
///   `Naive`'s **900** — 62% of the duplication removed by choosing which
///   machine gets which block, with the plan, the tasks and the cache size
///   untouched;
/// * on one computer with the same eight workers and the same total cache,
///   both policies duplicate **nothing**, because there is one pool and the
///   question does not arise.
#[test]
fn which_computer_gets_a_block_matters_only_when_they_do_not_share_a_cache() {
    use blockflow::distributed::handout::HandoutPolicy;
    use blockflow::simulate::Handout;

    let assembly = plan(16);
    let with = |nodes: usize, policy: HandoutPolicy| {
        run(&assembly, eight_workers(nodes), &mut Handout::new(policy))
    };

    let naive = with(4, HandoutPolicy::Naive);
    let nearest = with(4, HandoutPolicy::NearestFirst);
    assert_eq!(
        naive.tasks_run, nearest.tasks_run,
        "a handout policy chooses the order, never the work"
    );
    assert!(
        naive.duplicated_fetches > 0,
        "four separate page caches must duplicate something under a naive pull, or this \
         fixture cannot tell the policies apart"
    );
    println!(
        "four computers: naive duplicates {}, nearest-first {}",
        naive.duplicated_fetches, nearest.duplicated_fetches
    );
    assert!(
        nearest.duplicated_fetches < naive.duplicated_fetches,
        "nearest-first duplicated {} fetches against naive pull's {}. With a pool per computer \
         a locality policy has something to save, and the sign is what \
         `src/distributed/tests.rs` measures on the real coordinator.",
        nearest.duplicated_fetches,
        naive.duplicated_fetches
    );

    for policy in [HandoutPolicy::Naive, HandoutPolicy::NearestFirst] {
        assert_eq!(
            with(1, policy).duplicated_fetches,
            0,
            "{policy:?} duplicated a fetch on one computer, where there is one pool and \
             nothing to duplicate"
        );
    }
}

// ------------------------------ 6. and when spreading them apart pays --

/// **Seeding the computers apart pays across this sweep, but not monotonically.**
///
/// `HandoutPolicy::NearestFirst` — farthest-point seeding, then
/// nearest-unclaimed — is the coordinator's default and is exactly the right
/// shape for the boundary above: put each machine in its own region, so no
/// chunk is wanted by two of them. Against plan order, on four computers over a
/// `96^3` volume in `16^3` chunks, as the speed-up of seeded over bunched:
///
/// ```text
///     threads per computer   2 MiB   8 MiB   32 MiB
///                        1   1.239   1.034    1.035
///                        2   1.400   1.212    1.212
///                        4   1.186   1.197    1.218
///                       10   1.013   1.157    1.184
/// ```
///
/// Read it as **cache per thread**, which is the quantity that orders the whole
/// table:
///
/// * **2 MiB and one thread** — two megabytes for one thread's working set — is
///   the policy's best case, and it is worth 1.24x and two thirds of the
///   traffic. The pool holds what the thread is working on, so keeping the
///   machines apart is pure saving;
/// * **2 MiB and two threads** is the peak after the simulator started pricing
///   the plan's real fetch floor, at 1.40x;
/// * **2 MiB and ten threads** used to invert. It is now a small win, 1.013x,
///   so the old "cache per thread" story was partly an artefact of the
///   one-chunk floor;
/// * **32 MiB** now pays at ten threads (1.184) but flattens the gain at one
///   (1.035): when a pool holds the one-thread working set, there is little
///   left for a policy to save.
///
/// So the rule is not "spread the workers out". It is **spread the computers
/// apart, then remeasure when the simulator's floor changes**, because the
/// ordering depends on whether the floor is one chunk or the plan's whole fetch
/// set.
///
/// **What this test is not.** It is not a claim that one policy is better. It is
/// the boundary between them, measured, because "spread the workers out" is the
/// first thing anyone proposes about a cluster and it is right only on one side
/// of a line nobody had drawn.
///
/// **A prediction that failed, recorded because it was expensive to give up.**
/// The inversion looked like a policy bug: `Handout` seeded from every *other
/// worker's* anchor, so with ten threads per machine it was separating threads
/// that share a cache as eagerly as machines that do not. Keying the seeds and
/// the view anchor by **node** instead — which is what `Decision::node_anchors`
/// is for, and which is right on its own terms — changed the inverted case by
/// nothing at all. The axis is the cache, not the seeding target.
#[test]
fn seeding_the_computers_apart_pays_when_each_can_hold_what_its_threads_touch() {
    use blockflow::distributed::handout::HandoutPolicy;
    use blockflow::simulate::Handout;

    let assembly = sharing_fixture();
    let at = |threads: usize, cache_bytes: u64| {
        let machine = Machine {
            nodes: 4,
            workers: 4 * threads,
            cache_bytes,
            contention: MEASURED_CONTENTION,
            ..eight_workers(4)
        };
        let bunched = run(&assembly, machine, &mut ExecutorOrder::phase_major());
        let seeded = run(
            &assembly,
            machine,
            &mut Handout::new(HandoutPolicy::NearestFirst),
        );
        assert_eq!(
            bunched.tasks_run, seeded.tasks_run,
            "a handout policy chooses the order, never the work"
        );
        bunched.makespan_ns as f64 / seeded.makespan_ns as f64
    };

    println!(
        "{:>8} {:>8} {:>8} {:>8}",
        "threads", "2 MiB", "8 MiB", "32 MiB"
    );
    let mut table = Vec::new();
    for threads in [1usize, 2, 4, 10] {
        let row: Vec<f64> = [2u64 << 20, 8 << 20, 32 << 20]
            .into_iter()
            .map(|cache| at(threads, cache))
            .collect();
        println!(
            "{threads:>8} {:>8.3} {:>8.3} {:>8.3}",
            row[0], row[1], row[2]
        );
        table.push(row);
    }

    // One thread per computer, a cache that holds its working set: the case the
    // policy exists for, and the largest number in the table.
    assert!(
        table[0][0] > 1.15,
        "seeding four computers apart is worth only {:.3}x at one thread each on a tight \
         cache, where the recorded figure is 1.239",
        table[0][0]
    );
    // Ten threads on the same cache used to invert under the earlier timing
    // floor. With the plan-aware fetch floor it still pays, but only barely.
    assert!(
        table[3][0] > 1.0,
        "ten threads on a 2 MiB cache lost to seeding at {:.3}x; the recorded figure is \
         now 1.013 after the simulator started pricing the plan's real fetch floor",
        table[3][0]
    );
    // The axis is the cache: give those ten threads room and it comes back.
    assert!(
        table[3][2] > table[3][0] && table[3][2] > 0.99,
        "a 32 MiB cache did not recover the inverted case: {:.3} against {:.3} at 2 MiB. The \
         axis is the cache, not the thread count.",
        table[3][2],
        table[3][0]
    );
    assert!(
        table[1][0] > table[0][0] && table[1][0] > table[2][0],
        "the tight-cache peak moved away from two threads per computer: one={:.3}, \
         two={:.3}, four={:.3}",
        table[0][0],
        table[1][0],
        table[2][0]
    );
}

// ------------------------------------- 7. a policy that reads the pool --

/// **`HandoutPolicy::Coalescing`: score the candidates, don't prescribe a
/// route** — and the term that earns its place is the one no existing policy
/// has.
///
/// A locality-preserving traversal (Hilbert, Morton) is the textbook answer to
/// "what order should a node walk its region in", and it is the wrong shape
/// here: a prescribed curve marches into holes left by stolen blocks, blocks
/// that short-circuited, and neighbours that finished early. What survives noise
/// is a *score* recomputed from the state at hand, and the curve is then
/// whatever the scores trace out.
///
/// The score is two counts of chunks and no weight:
///
/// * chunks this block needs that my node's pool **does not hold**;
/// * minus those a task my node **already has in flight** is bringing in —
///   because the workers of one computer share a page cache, so a block beside
///   what a neighbour is doing is nearly free.
///
/// Distance to the node's anchor breaks ties, and only ties: it has no cost
/// units, so it cannot be added to a chunk count without a constant, and there
/// is no evidence for one.
///
/// Measured against plan order and against `NearestFirst`, four computers over
/// a `96^3` volume in `16^3` chunks — as the speed-up over plan order, so above
/// one is better:
///
/// ```text
///     threads   cache    nearest-first   coalescing
///           1   2 MiB            1.239        1.277
///           1   8 MiB            1.034        1.070
///           1  32 MiB            1.035        1.070
///           2   2 MiB            1.400        1.324
///           2   8 MiB            1.212        1.057
///           4   2 MiB            1.186        1.261
///           4   8 MiB            1.197        0.987
///          10   2 MiB            1.013        1.181
///          10   8 MiB            1.157        0.932
///          10  32 MiB            1.184        0.932
/// ```
///
/// **It is conditional, and it still helps the tight ten-thread cell.** The
/// case that used to invert no longer does, but coalescing still moves ten
/// threads on two megabytes from 1.013 to 1.181. At larger caches and high
/// thread counts, the in-flight warmth term over-clusters the run and loses.
///
/// **What it costs.** 122.8 microseconds a handout against `NearestFirst`'s
/// 8.1 — fifteen times more, and the cause is the chunk-key walk over *every*
/// ready candidate, not the arithmetic. Two readings, and they differ:
///
/// * on a real coordinator it is free. A block is tens to hundreds of
///   milliseconds of work, so a hundred microseconds to place it is under a
///   tenth of a percent;
/// * in **this simulator** it is not, because a handout is simulated in
///   nanoseconds. The scheduler's scan of the ready set is already 98% of a
///   large simulation, and this multiplies it.
///
/// The bound, if that ever matters, is to score a shortlist rather than the
/// whole ready set — the same fix the dispatch loop wants for the same reason.
/// Not done here: it would be tuning a cost nobody is paying yet.
#[test]
fn scoring_the_candidates_beats_prescribing_a_route() {
    use blockflow::distributed::handout::HandoutPolicy;
    use blockflow::simulate::Handout;

    let assembly = sharing_fixture();
    let against_plan_order = |threads: usize, cache_bytes: u64, policy: HandoutPolicy| {
        let machine = Machine {
            nodes: 4,
            workers: 4 * threads,
            cache_bytes,
            contention: MEASURED_CONTENTION,
            ..eight_workers(4)
        };
        let bunched = run(&assembly, machine, &mut ExecutorOrder::phase_major());
        let scored = run(&assembly, machine, &mut Handout::new(policy));
        assert_eq!(
            bunched.tasks_run, scored.tasks_run,
            "a handout policy chooses the order, never the work"
        );
        bunched.makespan_ns as f64 / scored.makespan_ns as f64
    };

    println!(
        "{:>8} {:>8} {:>14} {:>12}",
        "threads", "cacheMiB", "nearest-first", "coalescing"
    );
    let mut beaten = 0usize;
    for threads in [1usize, 2, 4, 10] {
        for cache in [2u64 << 20, 8 << 20, 32 << 20] {
            let nearest = against_plan_order(threads, cache, HandoutPolicy::NearestFirst);
            let coalescing = against_plan_order(threads, cache, HandoutPolicy::Coalescing);
            println!(
                "{threads:>8} {:>8} {nearest:>14.3} {coalescing:>12.3}",
                cache >> 20
            );
            beaten += usize::from(coalescing > nearest + 0.01);
        }
    }
    assert!(
        beaten >= 5,
        "coalescing beat nearest-first in only {beaten} of twelve cells, where the current \
         record keeps it useful in five cells"
    );

    // The cell that originally justified the policy no longer inverts, but the
    // in-flight warmth term should still improve it.
    let inverted = against_plan_order(10, 2 << 20, HandoutPolicy::NearestFirst);
    let fixed = against_plan_order(10, 2 << 20, HandoutPolicy::Coalescing);
    assert!(
        inverted > 1.0,
        "ten threads on a 2 MiB pool no longer benefits under nearest-first ({inverted:.3}); \
         the current simulator records it as a small win rather than the old inversion"
    );
    assert!(
        fixed > inverted + 0.01,
        "coalescing scored {fixed:.3} in the tight ten-thread cell against nearest-first's \
         {inverted:.3}; the imminent-warmth term should still help there."
    );
}
