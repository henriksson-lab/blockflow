// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What can be checked without a socket, and one measurement that answers a
// question the design left open on purpose.
//
// The design says of the handout policy: *"this is what the simulator is for.
// `AccountingEnvironment` already counts reads, so a handout policy can be
// scored by duplicate fetches across workers with no data and no cluster …
// Comparing naive pull, nearest-first and static territories is a test, not a
// discussion."* So the locality question is settled here, by running the real
// coordinator's real handout over a real decomposition with N virtual workers
// and counting.
//
// The process-level verification — byte-identical output, coverage over a
// merged stream, a worker dying — is in `tests/local_multi_node.rs`, because it
// starts the actual binaries.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::env::{AccountingEnvironment, Environment};
use crate::strategy::execute_task;

use super::cache_model::ChunkGrid;
use super::coordinator::Job;
use super::handout::HandoutPolicy;
use super::protocol::Handout;
use super::spec::{probe_job, ChainSpec, OpSpec, ProbeWorkflows, WorkflowFactory};

// ------------------------------------------------- the protocol is insulated --

/// The dtype and `Array3` work rewrote `apply` and the environment. **The
/// protocol was untouched**, which is what this asserted in advance.
///
/// The protocol deals in block indices, phases and regions, so it should be
/// insulated from that — and "should be" is exactly the kind of claim that is
/// true until the day somebody needs one field. Asserted rather than trusted,
/// because the failure would have been invisible until the rewrite landed and
/// then would have been a redesign rather than a fix.
///
/// The forbidden spellings live *here* rather than in the file being checked,
/// so that the check cannot pass by containing its own counterexample.
#[test]
fn no_message_in_the_protocol_names_an_element_type() {
    let leaks = [
        format!("Arr{}", "ayD"),
        format!("Arr{}", "ay3"),
        format!("f{}", 64),
        format!("Block{}", "Buf"),
        format!("nd{}", "array"),
        format!("Dt{}", "ype"),
    ];
    for leak in &leaks {
        assert!(
            !without_tests(include_str!("protocol.rs")).contains(leak.as_str()),
            "the protocol mentions {leak:?}. It should carry block indices, phases and \
             regions and nothing about elements — see the module header."
        );
    }
    // The handout is allowed a real number, because a block's position is a
    // voxel *coordinate* and a coordinate is not a voxel. What it may not have
    // is an array or an element-type tag.
    let handout = without_tests(include_str!("handout.rs"));
    for leak in leaks.iter().filter(|leak| !leak.ends_with("64")) {
        assert!(
            !handout.contains(leak.as_str()),
            "the handout mentions {leak:?}"
        );
    }
}

/// The code, without its own test module. A fixture is allowed to name a
/// concrete element type; the thing being fixtured is not.
fn without_tests(source: &str) -> &str {
    match source.find("#[cfg(test)]") {
        Some(at) => &source[..at],
        None => source,
    }
}

/// Nothing about cache state may introduce a point where one side waits for the
/// other. The strongest cheap form of that: there is **no message** by which a
/// worker could report cache state, so no handout can be waiting for one.
#[test]
fn there_is_no_way_for_a_worker_to_report_its_cache() {
    let protocol = include_str!("protocol.rs");
    for forbidden in ["cache", "resident", "evict"] {
        assert!(
            !protocol.to_lowercase().contains(forbidden),
            "the protocol mentions {forbidden:?}. The coordinator *models* each worker's \
             cache from what it assigned and is never told — see `cache_model` for why a \
             report would turn an advisory estimate into a scheduling constraint."
        );
    }
    // And the model's own interface takes assignments, never reports.
    let model = include_str!("cache_model.rs");
    assert!(model.contains("pub fn note_assigned"));
    assert!(
        !model.contains("pub fn note_reported") && !model.contains("pub fn update_from_worker"),
        "the cache model grew a way to be told rather than to derive"
    );
}

// ------------------------------------------------------------- locality --

/// What N workers cost each other, over one decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Locality {
    /// Distinct `(image, chunk)` the whole job touches. What one node with an
    /// unbounded cache would fetch.
    distinct: usize,
    /// Summed over workers, the distinct chunks each one touches. What N nodes
    /// with unbounded caches fetch between them.
    fetched: usize,
    /// Chunks more than one worker had to fetch. The seam, counted.
    duplicated: usize,
    /// Chunk reads actually performed, as counted by the workers' own
    /// environments. Higher than `fetched` because it includes a worker
    /// re-reading a halo chunk for successive blocks — real work, not a seam.
    chunk_reads: u64,
    tasks: usize,
}

impl Locality {
    /// Fetches per distinct chunk. 1.0 is perfect: nobody fetched anything
    /// anybody else did.
    fn redundancy(&self) -> f64 {
        self.fetched as f64 / self.distinct.max(1) as f64
    }
}

/// Run a whole job over `workers` virtual workers under one policy, and count.
///
/// The workers are virtual in one sense only: they are round-robin rather than
/// racing, so the schedule is deterministic and two policies are compared over
/// the same task order. Everything else is real — the real `Job`, the real
/// handout, the real dependency release, and a real `execute_task` per task
/// against a per-worker environment that counts what it read.
fn measure(policy: HandoutPolicy, workers: usize, blocks: usize, phases: usize) -> Locality {
    let (mut spec, decomposition) = probe_job(blocks, phases, ChainSpec::identity());
    spec.policy = policy;
    let chunk = spec.workflow.chunk;
    let volume = spec.workflow.shape;
    let chain = ProbeWorkflows.chain(&spec.workflow).expect("a probe chain");
    let mut job = Job::new(spec, decomposition.clone()).expect("a valid job");
    let chunks = ChunkGrid::new(volume, chunk);

    // One environment per worker: separate caches, separate counters, which is
    // the whole point — there is no shared cache across nodes and there should
    // not be.
    let environments: Vec<AccountingEnvironment> = (0..workers)
        .map(|_| AccountingEnvironment::new(volume, chunk, 8))
        .collect();
    for environment in &environments {
        environment.prepare(&decomposition).expect("prepared");
    }

    let mut touched: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); workers];
    let mut tasks = 0usize;
    let mut idle = 0usize;
    loop {
        let mut progressed = false;
        for (index, environment) in environments.iter().enumerate() {
            let name = format!("worker-{index}");
            match job.pull(&name) {
                Handout::Task(assignment) => {
                    touched[index].extend(chunks.keys(assignment.phase, &assignment.read));
                    let task = &job.graph().tasks[assignment.task];
                    execute_task(&chain, &decomposition, task, environment, &[])
                        .expect("a probe task runs");
                    job.completed(&name, assignment.task).expect("completed");
                    tasks += 1;
                    progressed = true;
                }
                Handout::Wait { .. } => {}
                Handout::Finished => {}
            }
        }
        if job.finished() {
            break;
        }
        if !progressed {
            idle += 1;
            assert!(idle < 1000, "the job stopped making progress");
        }
    }

    let mut union: BTreeSet<u64> = BTreeSet::new();
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for set in &touched {
        for &key in set {
            union.insert(key);
            *counts.entry(key).or_default() += 1;
        }
    }
    Locality {
        distinct: union.len(),
        fetched: touched.iter().map(BTreeSet::len).sum(),
        duplicated: counts.values().filter(|&&count| count > 1).count(),
        chunk_reads: environments
            .iter()
            .map(|environment| environment.counters().snapshot().4)
            .sum(),
        tasks,
    }
}

/// The design's claim, measured: **naive global pull destroys locality.**
///
/// If adjacent blocks land on different workers the halo between them is
/// fetched twice, once into each worker's cache, and neither gets the reuse
/// that made the halo cheap on one node. Seeding far apart and then taking the
/// nearest unclaimed task is supposed to fix that without a territory to
/// maintain. This is the number that says whether it does.
#[test]
fn nearest_first_handout_costs_fewer_duplicated_fetches_than_naive_pull() {
    let workers = 4;
    let naive = measure(HandoutPolicy::Naive, workers, 32, 1);
    let nearest = measure(HandoutPolicy::NearestFirst, workers, 32, 1);
    let modelled = measure(HandoutPolicy::CacheModelled, workers, 32, 1);

    assert_eq!(naive.tasks, nearest.tasks);
    assert_eq!(naive.distinct, nearest.distinct);
    assert!(
        nearest.duplicated < naive.duplicated,
        "nearest-first duplicated {} chunks, naive {} — over {} distinct chunks and {} \
         tasks on {workers} workers",
        nearest.duplicated,
        naive.duplicated,
        naive.distinct,
        naive.tasks
    );
    assert!(
        nearest.redundancy() < naive.redundancy(),
        "redundancy: naive {:.3}, nearest-first {:.3}, cache-modelled {:.3}",
        naive.redundancy(),
        nearest.redundancy(),
        modelled.redundancy()
    );
    // Recorded rather than asserted tightly: the point of the measurement is
    // the ordering, and the magnitude depends on the decomposition.
    println!(
        "locality over {} tasks, {workers} workers, {} distinct chunks:\n  \
         naive          duplicated {:4}  redundancy {:.3}  chunk reads {}\n  \
         nearest-first  duplicated {:4}  redundancy {:.3}  chunk reads {}\n  \
         cache-modelled duplicated {:4}  redundancy {:.3}  chunk reads {}",
        naive.tasks,
        naive.distinct,
        naive.duplicated,
        naive.redundancy(),
        naive.chunk_reads,
        nearest.duplicated,
        nearest.redundancy(),
        nearest.chunk_reads,
        modelled.duplicated,
        modelled.redundancy(),
        modelled.chunk_reads,
    );
}

/// The advantage should grow with the number of workers, because duplicated
/// fetches are proportional to the *surface* between workers' regions and naive
/// pull maximises that surface.
#[test]
fn the_locality_advantage_survives_more_workers() {
    for workers in [2, 4, 8] {
        let naive = measure(HandoutPolicy::Naive, workers, 32, 1);
        let nearest = measure(HandoutPolicy::NearestFirst, workers, 32, 1);
        assert!(
            nearest.duplicated <= naive.duplicated,
            "{workers} workers: nearest-first {} duplicated, naive {}",
            nearest.duplicated,
            naive.duplicated
        );
        println!(
            "{workers} workers: naive {} duplicated, nearest-first {} duplicated",
            naive.duplicated, nearest.duplicated
        );
    }
}

/// Whatever the policy, the job is the same job: every task once, in dependency
/// order, and the acceptance criterion holds over the merged stream.
///
/// This is the property that makes the handout *advisory*. If a policy could
/// change what ran, none of the above would be a free choice.
#[test]
fn the_handout_policy_changes_who_does_what_and_never_what_is_done() {
    for policy in [
        HandoutPolicy::Naive,
        HandoutPolicy::NearestFirst,
        HandoutPolicy::CacheModelled,
    ] {
        let measured = measure(policy, 3, 16, 2);
        let (_, decomposition) = probe_job(16, 2, ChainSpec::identity());
        assert_eq!(
            measured.tasks,
            decomposition.n_tasks(),
            "{}: {} tasks executed, {} in the decomposition",
            policy.as_str(),
            measured.tasks,
            decomposition.n_tasks()
        );
    }
}

// -------------------------------------------------- placing a barrier phase --

/// What one barrier phase cost, and what it could have cost.
///
/// A full-reach op is a planning barrier and resolves to a **single block**, so
/// a barrier phase is one task: one worker does it and the rest of the cluster
/// idles through it. Which worker gets it is therefore the only placement
/// decision the phase has, and it is worth the whole phase's read cost.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BarrierPlacement {
    /// Distinct chunks the barrier task reads. The whole of its image.
    read_chunks: usize,
    /// Per worker, how many of those it had **not** had through its hands when
    /// the barrier became ready — what the phase would have cost had it landed
    /// there.
    costs: Vec<usize>,
    /// Which worker got it, and how many blocks of the previous phase each ran.
    worker: usize,
    shares: Vec<usize>,
}

impl BarrierPlacement {
    /// What the placement actually cost.
    fn refetched(&self) -> usize {
        self.costs[self.worker]
    }

    fn best(&self) -> usize {
        self.costs.iter().copied().min().expect("a worker")
    }

    fn worst(&self) -> usize {
        self.costs.iter().copied().max().expect("a worker")
    }

    /// What an *arbitrary* worker costs, which is what "whichever worker happens
    /// to claim it" is: the handout without this rule is a race between poll
    /// timers, and no worker is favoured by it. Reported as a mean rather than as
    /// one particular draw, because a single draw of a race is not a measurement.
    fn mean(&self) -> f64 {
        self.costs.iter().sum::<usize>() as f64 / self.costs.len() as f64
    }
}

/// Run a job whose last phase is a barrier, and see where the barrier landed.
///
/// `weights` is how many pulls each worker attempts per round — an **unequal**
/// cluster, which is the ordinary case rather than a contrived one: the design
/// records that 37-41 % of blocks are empty and that empty regions *cluster*, so
/// two workers over one volume do unequal amounts of work whatever the handout
/// does. All-ones is the balanced cluster, and it is measured too because the
/// answer there is the interesting negative.
fn place_barrier(
    policy: HandoutPolicy,
    weights: &[usize],
    blocks: usize,
    scarce_placement: bool,
) -> BarrierPlacement {
    let workers = weights.len();
    let shape = [8 * blocks, 8, 8];
    // A second op reaching the whole of every axis. `is_planning_barrier` is an
    // exact `reach >= extent`, so this decomposes to N blocks and then one.
    let chain = ChainSpec(vec![
        OpSpec::new("identity", "first", [2, 0, 0]),
        OpSpec::new("identity", "merge", shape),
    ]);
    let (mut spec, decomposition) = probe_job(blocks, 2, chain);
    spec.policy = policy;
    let chunk = spec.workflow.chunk;
    let volume = spec.workflow.shape;
    let chain = ProbeWorkflows.chain(&spec.workflow).expect("a probe chain");
    let mut job = Job::new(spec, decomposition.clone())
        .expect("a valid job")
        .with_scarce_placement(scarce_placement);
    let chunks = ChunkGrid::new(volume, chunk);

    let last = decomposition.n_phases() - 1;
    assert_eq!(
        decomposition.phases[last].blocks.len(),
        1,
        "the point of this measurement is a phase of one block"
    );

    let environments: Vec<AccountingEnvironment> = (0..workers)
        .map(|_| AccountingEnvironment::new(volume, chunk, 8))
        .collect();
    for environment in &environments {
        environment.prepare(&decomposition).expect("prepared");
    }

    // Ground truth, not the coordinator's model: what each worker has actually
    // had through its hands, read or written. The model is asserted against this
    // rather than trusted to be it.
    let mut held: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); workers];
    let mut shares = vec![0usize; workers];
    let mut placement: Option<BarrierPlacement> = None;
    let mut idle = 0usize;
    let mut round = 0usize;
    loop {
        let mut progressed = false;
        // Rotate who asks first. Without this the heaviest worker is also
        // always the first to ask after a phase boundary, which would hand the
        // baseline the right answer for a reason that has nothing to do with
        // locality — the harness would be measuring itself.
        for step in 0..workers {
            let index = (step + round) % workers;
            let environment = &environments[index];
            let name = format!("worker-{index}");
            for _ in 0..weights[index] {
                let Handout::Task(assignment) = job.pull(&name) else {
                    continue;
                };
                if assignment.phase == last {
                    let keys = chunks.keys(assignment.phase, &assignment.read);
                    placement = Some(BarrierPlacement {
                        read_chunks: keys.len(),
                        costs: held
                            .iter()
                            .map(|set| keys.iter().filter(|key| !set.contains(key)).count())
                            .collect(),
                        worker: index,
                        shares: shares.clone(),
                    });
                } else {
                    shares[index] += 1;
                }
                held[index].extend(chunks.keys(assignment.phase, &assignment.read));
                held[index].extend(chunks.keys(assignment.phase + 1, &assignment.valid));
                let task = &job.graph().tasks[assignment.task];
                execute_task(&chain, &decomposition, task, environment, &[])
                    .expect("a probe task runs");
                job.completed(&name, assignment.task).expect("completed");
                progressed = true;
            }
        }
        round += 1;
        if job.finished() {
            break;
        }
        if !progressed {
            idle += 1;
            assert!(idle < 1000, "the job stopped making progress");
        }
    }
    placement.expect("the barrier phase ran")
}

/// The measurement this was built for: **what the barrier phase re-reads**, with
/// and without locality-aware selection, over the same job.
///
/// Reported the way the 62-vs-6 locality figure is — the ordering asserted, the
/// magnitudes printed, because the magnitude depends on the decomposition and
/// the claim is about the direction.
#[test]
fn locality_aware_selection_cuts_what_a_barrier_phase_re_reads() {
    // An unequal cluster: worker 0 gets three pulls a round to the others' one.
    let weights = [3usize, 1, 1, 1];
    let blocks = 32;
    let without = place_barrier(HandoutPolicy::NearestFirst, &weights, blocks, false);
    let with = place_barrier(HandoutPolicy::NearestFirst, &weights, blocks, true);

    assert_eq!(without.read_chunks, with.read_chunks);
    assert_eq!(
        with.refetched(),
        with.best(),
        "the selection should land the barrier on the best-placed worker; costs {:?}, \
         it went to {}",
        with.costs,
        with.worker
    );
    assert!(
        (with.refetched() as f64) < without.mean(),
        "locality-aware {} against an arbitrary worker's {:.1}",
        with.refetched(),
        without.mean()
    );
    // Each run is scored against its own spread, because the rule also fires on
    // the *tail* of the ordinary phase — where fewer tasks remain than there are
    // workers — so the shares the barrier then chooses between are themselves
    // slightly different. Comparing one run's outcome with the other run's best
    // would be comparing two different jobs.
    println!(
        "barrier over {} chunks, {} workers:\n  \
         claim order     worker {} re-read {:3}; an arbitrary worker {:5.1} \
         (best {}, worst {}, shares {:?})\n  \
         locality-aware  worker {} re-read {:3}; an arbitrary worker {:5.1} \
         (best {}, worst {}, shares {:?})",
        with.read_chunks,
        weights.len(),
        without.worker,
        without.refetched(),
        without.mean(),
        without.best(),
        without.worst(),
        without.shares,
        with.worker,
        with.refetched(),
        with.mean(),
        with.best(),
        with.worst(),
        with.shares,
    );
}

/// How the saving moves with the cluster size, since one number over one
/// decomposition is a data point rather than a result.
#[test]
fn the_barrier_saving_holds_across_cluster_sizes() {
    for workers in [2usize, 4, 8] {
        let mut weights = vec![1usize; workers];
        weights[0] = 3;
        let without = place_barrier(HandoutPolicy::NearestFirst, &weights, 32, false);
        let with = place_barrier(HandoutPolicy::NearestFirst, &weights, 32, true);
        assert_eq!(with.refetched(), with.best());
        assert!(
            (with.refetched() as f64) <= without.mean(),
            "{workers} workers: {} against {:.1}",
            with.refetched(),
            without.mean()
        );
        println!(
            "{workers} workers, shares {:?}: barrier re-reads {} of {} \
             (an arbitrary worker {:.1}, worst {})",
            with.shares,
            with.refetched(),
            with.read_chunks,
            without.mean(),
            without.worst()
        );
    }
}

/// The negative, stated rather than hidden: on a **balanced** cluster the
/// selection buys nothing.
///
/// Every worker ran the same number of blocks of the previous phase, so every
/// worker holds the same share of the image the barrier reads, no worker is
/// strictly better than any other, and the rule declines to have an opinion —
/// which is the same fallback a cold start takes. The saving is real only where
/// the shares are unequal, and it is worth having only because they usually are.
#[test]
fn on_a_balanced_cluster_the_selection_has_nothing_to_choose_between() {
    let weights = [1usize, 1, 1, 1];
    let without = place_barrier(HandoutPolicy::NearestFirst, &weights, 32, false);
    let with = place_barrier(HandoutPolicy::NearestFirst, &weights, 32, true);
    assert_eq!(
        with.refetched(),
        without.refetched(),
        "shares {:?} against {:?}",
        with.shares,
        without.shares
    );
    assert_eq!(with.best(), with.worst(), "no worker is better placed");
    println!(
        "balanced cluster, shares {:?}: barrier re-reads {} of {} either way, and every \
         worker would have cost the same ({}..{})",
        with.shares,
        with.refetched(),
        with.read_chunks,
        with.best(),
        with.worst()
    );
}

/// Whatever else it does, it must not stall. Every worker count from one to
/// eight, and every policy, over a job with a barrier in it.
#[test]
fn a_barrier_phase_is_always_claimed_by_somebody() {
    for workers in 1..=8usize {
        for policy in [
            HandoutPolicy::Naive,
            HandoutPolicy::NearestFirst,
            HandoutPolicy::CacheModelled,
        ] {
            let weights = vec![1usize; workers];
            let placed = place_barrier(policy, &weights, 8, true);
            assert!(placed.refetched() <= placed.read_chunks);
            assert_eq!(placed.shares.iter().sum::<usize>(), 8);
        }
    }
}

/// The coordinator's model of a worker's overlap must agree with what that
/// worker actually had through its hands, or the selection is choosing on
/// fiction. It is an *estimate* by design, so this is asserted where the
/// estimate should be exact: a budget nothing is evicted from and every
/// assignment honoured.
#[test]
fn the_modelled_overlap_matches_what_the_worker_really_holds() {
    let placed = place_barrier(HandoutPolicy::CacheModelled, &[3, 1, 1, 1], 32, true);
    assert_eq!(
        placed.refetched(),
        placed.best(),
        "the model picked a worker the ground truth disagrees with; costs {:?}",
        placed.costs
    );
}

/// Coverage, from a merged stream, with several workers and no network.
///
/// The same `check_coverage_and_order` a single-node run is asserted with. That
/// it applies unchanged is the point: a distributed run's log is an
/// `ExecutionLog`, not a distributed-specific artefact.
#[test]
fn the_merged_event_stream_satisfies_the_single_node_acceptance_criterion() {
    let (mut spec, decomposition) = probe_job(12, 2, ChainSpec::identity());
    spec.policy = HandoutPolicy::NearestFirst;
    let volume = spec.workflow.shape;
    let chunk = spec.workflow.chunk;
    let chain = ProbeWorkflows.chain(&spec.workflow).unwrap();
    let mut job = Job::new(spec, decomposition.clone()).unwrap();

    let outbox: Arc<crate::log::ExecutionLog> = Arc::new(crate::log::ExecutionLog::new());
    let listeners: Vec<Arc<dyn crate::listener::EventListener>> = vec![outbox.clone()];
    let environment = AccountingEnvironment::new(volume, chunk, 8);
    environment.prepare(&decomposition).unwrap();

    let mut idle = 0;
    while !job.finished() {
        let mut progressed = false;
        for worker in 0..3 {
            let name = format!("w{worker}");
            if let Handout::Task(assignment) = job.pull(&name) {
                let task = &job.graph().tasks[assignment.task];
                execute_task(&chain, &decomposition, task, &environment, &listeners).unwrap();
                job.completed(&name, assignment.task).unwrap();
                progressed = true;
            }
        }
        if !progressed {
            idle += 1;
            assert!(idle < 1000, "stalled");
        }
    }
    // Feed the coordinator the stream the way a worker would, one event at a
    // time, and assert from what the coordinator ended up holding.
    for event in outbox.events() {
        job.report(&crate::export::event_json(&event));
    }
    assert_eq!(job.unknown_events(), 0);
    job.check_coverage()
        .expect("every block, every op, in order");
    assert!(
        job.log().duplicate_applications().is_empty(),
        "a block was executed twice with nobody dying"
    );
}
