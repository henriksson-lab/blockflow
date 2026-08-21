// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Which **worker** gets a task, when there are not enough tasks to go round.
//
// `handout` answers the ordinary question — given a worker, which unclaimed
// task? This module answers the dual, and it only becomes interesting in one
// situation.
//
// The situation
// -------------
// A **full-reach op is a planning barrier** and resolves to a single block: the
// cost model drives it there, because N blocks each reading everything is
// N-times redundant. So a barrier phase is *one task*, one worker does it, and
// the rest of the cluster idles through it.
//
// For an ordinary phase, who gets which block hardly matters — there are many
// tasks, and `handout`'s locality bias spreads them well (measured: naive pull
// duplicated 62 chunks against 6 for nearest-first). For a barrier it is the
// **only placement decision in the phase**, so getting it right is worth the
// whole phase's read cost, and getting it wrong wastes a full-volume read.
//
// Today that task goes to whichever worker happens to ask first, which is a
// race between poll timers. It should go to the worker that already holds the
// most of what the task will read.
//
// The rule
// --------
// Given the asking worker and the workers it could be handed to *instead*:
//
// 1. **Scarcity, of the phase and not of the moment.** A task is contestable
//    only if **its own phase** has fewer tasks than there are workers able to
//    take one. A barrier phase has one task however large the volume is, so it
//    is always contestable with two or more workers; a sixteen-block phase never
//    is, however few of its blocks happen to be left.
//
//    That distinction was **measured, not reasoned**. Scoring on the momentary
//    ready set instead — "fewer tasks left than workers" — fires in the tail of
//    every ordinary phase, and it broke
//    `the_work_list_stays_at_least_one_task_ahead_of_what_is_being_computed`:
//    a refused worker's list ran empty while the coordinator still had work,
//    which is precisely the failure `DISTRIBUTION.md` says must not happen,
//    because it would stop prefetch multi-node while every single-node test kept
//    passing. In a phase that is scarce *by construction* the idling is the
//    barrier's, not the rule's — N-1 workers were going to idle whatever the
//    coordinator said.
// 2. **Strictly better.** The asker is refused a task only if some contender is
//    modelled as holding **strictly more** of that task's read set. Equal
//    overlap leaves the asker entitled, so a cold start — where nobody holds
//    anything and every score is identical — is first-come-first-served, which
//    is the behaviour without this module at all.
// 3. **Nothing else.** Distance, plan order and worker identity play no part in
//    the refusal. They are `handout`'s business, and they decide among the
//    tasks this module leaves.
//
// So this is a **filter over the ready set**, applied before the score. That
// shape is deliberate — see "What a capability filter would need".
//
// A wrong choice costs a re-read, never correctness
// -------------------------------------------------
// The founding property of the distributed layer, and it applies exactly here.
// The cache model is approximate by design (`cache_model`), so a worker chosen
// for overlap it does not really have simply fetches the chunks, as it would
// have anyway. Nothing about the output, the coverage or the ordering depends
// on which worker runs a task — the same argument that lets a killed worker's
// task be reissued and executed twice.
//
// No synchronisation, and no new timer
// ------------------------------------
// Everything here is derived from what the coordinator already holds:
// assignments it made and completions it was told about. No worker is asked
// anything, and no worker waits — a refused pull returns the existing
// `Handout::Wait`, which is the same answer it gets when everything ready is
// claimed, and it is answered immediately.
//
// The liveness bound is worth stating because it is *not* a timer this module
// introduces. A task can be withheld from the asker only while some other
// worker is a **contender**, and the coordinator's definition of that already
// requires the worker to be idle and to have been seen recently. A worker that
// dies stops being seen and drops out; a worker that is alive is polling, and
// the highest-overlap contender is by construction entitled to the task, so it
// takes it on its next poll. Deferral therefore ends either when the better
// worker claims the task or when it stops looking alive — never later.
//
// What a capability filter would need
// -----------------------------------
// Classes of nodes — heterogeneous hardware, some with an accelerator and some
// without — turn placement into a *capability filter* (which nodes may run this
// op at all) composed with the locality score this module adds. That composes
// in front of `entitled` without restructuring anything: the pipeline in
// `Job::pull` is `ready -> [capability] -> entitled -> handout::choose`, and a
// capability filter is one more `&[usize] -> [usize]` stage over the same ready
// set. Three things it would need that do not exist yet: an op-side declaration
// of what a slot requires (a `BlockOp` method, folded over a phase the way
// `reach` already is), a worker-side declaration of what a node offers (a field
// on the join message, which is the one place a worker legitimately describes
// itself), and a decision about *feasibility* — a capability filter can empty
// the ready set permanently, which locality never can, so it needs a job-level
// check at submission that every phase has at least one node able to run it.
// That third point is why it is not folded in here: this module may only ever
// express a preference, and a capability is a constraint.

use std::borrow::Cow;

use crate::graph::TaskGraph;

use super::cache_model::{ChunkGrid, ModelledCache};

/// What one worker is modelled as holding, for the purpose of placing a scarce
/// task.
///
/// Two tiers, because a barrier reads an image that was only ever **written**.
/// Phase `p` reads image `p` and writes image `p + 1`, so the image a barrier at
/// phase `p` reads is the one phase `p - 1` produced and nobody read. A model
/// fed only by reads therefore scores every worker identically for a barrier and
/// has nothing to say at exactly the moment it matters most; `produced` is what
/// gives it something to say. See `Job::admit` for what that second tier claims
/// physically and what it does not.
pub struct Residency<'a> {
    /// Which worker this is. Carried for diagnostics; the rule never breaks a
    /// tie on it, because an equal score means the asker keeps the task.
    pub id: &'a str,
    /// Chunks the coordinator models this worker as having **read**.
    pub resident: Option<&'a ModelledCache>,
    /// Chunks the coordinator models this worker as having **written**.
    pub produced: Option<&'a ModelledCache>,
}

impl Residency<'_> {
    /// How many of `keys` this worker would **not** have to fetch.
    ///
    /// A chunk in both tiers counts once: the question is whether the read is
    /// served from the node, not how many places on the node could serve it.
    pub fn overlap(&self, keys: &[u64]) -> usize {
        keys.iter()
            .filter(|&&key| {
                self.resident.is_some_and(|cache| cache.holds(key))
                    || self.produced.is_some_and(|cache| cache.holds(key))
            })
            .count()
    }
}

/// The chunks a task will fetch, as the cache model keys them.
///
/// `source` rather than `read`: the two differ only for a phase that reads
/// across grids, and a model keyed on the wrong region would be scoring chunks
/// nobody asks for. This is the same read set `handout` scores against, on
/// purpose — one definition of "what this task fetches", used by both halves of
/// the decision.
pub fn read_keys(graph: &TaskGraph, chunks: &ChunkGrid, task: usize) -> Vec<u64> {
    let entry = &graph.tasks[task];
    chunks.keys(entry.phase, &entry.geometry.source)
}

/// Which of `ready` this worker may claim now.
///
/// `phase_tasks` is how many tasks each phase has, in phase order — the *plan's*
/// shape, fixed at submission, not what happens to be left. See rule 1.
///
/// Borrowed unchanged whenever no ready task belongs to a scarce phase, which is
/// every handout of an ordinary run.
pub fn entitled<'a>(
    ready: &'a [usize],
    graph: &TaskGraph,
    chunks: &ChunkGrid,
    phase_tasks: &[usize],
    asker: &Residency<'_>,
    contenders: &[Residency<'_>],
) -> Cow<'a, [usize]> {
    let scarce = |task: usize| {
        let phase = graph.tasks[task].phase;
        is_scarce(
            phase_tasks.get(phase).copied().unwrap_or(usize::MAX),
            contenders.len(),
        )
    };
    if !ready.iter().copied().any(scarce) {
        return Cow::Borrowed(ready);
    }
    let kept: Vec<usize> = ready
        .iter()
        .copied()
        .filter(|&task| {
            if !scarce(task) {
                return true;
            }
            let keys = read_keys(graph, chunks, task);
            let mine = asker.overlap(&keys);
            // Strictly better, so an uninformative model refuses nobody.
            !contenders
                .iter()
                .any(|contender| contender.overlap(&keys) > mine)
        })
        .collect();
    Cow::Owned(kept)
}

/// Whether a phase is small enough that placing its tasks is worth a wait.
///
/// Strictly fewer tasks in the phase than workers able to take one — the asker
/// plus its contenders. At the threshold every worker can be busy, and holding a
/// task back for a better owner would idle a worker that had somewhere to be.
pub fn is_scarce(phase_tasks: usize, contenders: usize) -> bool {
    phase_tasks > 0 && phase_tasks <= contenders
}

/// The worker with the most modelled overlap with a task's read set.
///
/// Not used by the handout — the handout expresses the same preference as a
/// filter, one asking worker at a time, because that is what a pull-based
/// coordinator can do without asking anybody anything. This is here for tests
/// and diagnostics, which want the answer directly rather than as the fixed
/// point of a sequence of pulls.
pub fn best_worker<'a>(
    graph: &TaskGraph,
    chunks: &ChunkGrid,
    task: usize,
    workers: &'a [Residency<'a>],
) -> Option<(&'a str, usize)> {
    let keys = read_keys(graph, chunks, task);
    let mut best: Option<(&str, usize)> = None;
    for worker in workers {
        let overlap = worker.overlap(&keys);
        if best.is_none_or(|(_, current)| overlap > current) {
            best = Some((worker.id, overlap));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposition::{Decomposition, PhaseDecomposition};
    use crate::dtype::Dtype;
    use crate::geometry::BlockGrid;
    use crate::region::Region;

    /// Two phases over a strip: eight ordinary blocks, then one block that reads
    /// everything. The shape a full-reach op decomposes to.
    fn barrier() -> (TaskGraph, ChunkGrid) {
        let volume = [64usize, 8, 8];
        let fine = BlockGrid::new(volume, [8, 8, 8]).unwrap();
        let whole = BlockGrid::new(volume, volume).unwrap();
        let decomposition = Decomposition {
            volume,
            dtype: Dtype::U16,
            phases: vec![
                PhaseDecomposition::derive(
                    vec![0],
                    vec!["fine".to_string()],
                    [0, 0, 0],
                    [0, 0, 0],
                    fine,
                ),
                PhaseDecomposition::derive(
                    vec![1],
                    vec!["merge".to_string()],
                    [64, 8, 8],
                    [64, 8, 8],
                    whole,
                ),
            ],
            chain_reach: [64, 8, 8],
        };
        (
            TaskGraph::build(&decomposition),
            ChunkGrid::new(volume, [8, 8, 8]),
        )
    }

    /// Tasks per phase for [`barrier`]: eight ordinary blocks, then one.
    fn phase_tasks() -> Vec<usize> {
        vec![8, 1]
    }

    fn cache() -> ModelledCache {
        let chunk_bytes = 8 * 8 * 8 * 2;
        ModelledCache::new(chunk_bytes * 4096, chunk_bytes)
    }

    /// The single task of the last phase — the barrier.
    fn barrier_task(graph: &TaskGraph) -> usize {
        let last = graph.tasks.iter().map(|task| task.phase).max().unwrap();
        let tasks: Vec<usize> = (0..graph.len())
            .filter(|&t| graph.tasks[t].phase == last)
            .collect();
        assert_eq!(tasks.len(), 1, "the barrier phase should be one block");
        tasks[0]
    }

    /// Note that a worker was assigned block `index` of phase 0, which is what
    /// the coordinator does on handout: it reads image 0 and writes image 1.
    fn ran_first_phase(resident: &mut ModelledCache, produced: &mut ModelledCache, index: usize) {
        let chunks = ChunkGrid::new([64, 8, 8], [8, 8, 8]);
        let block = Region::new(&[index * 8, 0, 0], &[8, 8, 8]);
        resident.note_assigned(&chunks.keys(0, &block));
        produced.note_assigned(&chunks.keys(1, &block));
    }

    #[test]
    fn with_work_for_everybody_nothing_is_withheld() {
        let (graph, chunks) = barrier();
        let ready: Vec<usize> = (0..8).collect();
        let asker = Residency {
            id: "a",
            resident: None,
            produced: None,
        };
        let contenders = [
            Residency {
                id: "b",
                resident: None,
                produced: None,
            },
            Residency {
                id: "c",
                resident: None,
                produced: None,
            },
        ];
        assert_eq!(
            entitled(&ready, &graph, &chunks, &phase_tasks(), &asker, &contenders).as_ref(),
            ready.as_slice()
        );
    }

    #[test]
    fn a_barrier_goes_to_the_worker_that_produced_most_of_what_it_reads() {
        let (graph, chunks) = barrier();
        let task = barrier_task(&graph);

        let (mut a_read, mut a_wrote) = (cache(), cache());
        let (mut b_read, mut b_wrote) = (cache(), cache());
        // b ran six of the eight blocks of the previous phase; a ran two.
        for index in 0..2 {
            ran_first_phase(&mut a_read, &mut a_wrote, index);
        }
        for index in 2..8 {
            ran_first_phase(&mut b_read, &mut b_wrote, index);
        }
        let a = Residency {
            id: "a",
            resident: Some(&a_read),
            produced: Some(&a_wrote),
        };
        let b = Residency {
            id: "b",
            resident: Some(&b_read),
            produced: Some(&b_wrote),
        };

        let ready = vec![task];
        assert!(
            entitled(
                &ready,
                &graph,
                &chunks,
                &phase_tasks(),
                &a,
                std::slice::from_ref(&b)
            )
            .is_empty(),
            "the worker holding a quarter of the volume must not take the barrier while the \
             one holding three quarters is idle"
        );
        assert_eq!(
            entitled(
                &ready,
                &graph,
                &chunks,
                &phase_tasks(),
                &b,
                std::slice::from_ref(&a)
            )
            .as_ref(),
            ready.as_slice(),
            "the worker holding the most of it is entitled to it"
        );
        assert_eq!(
            best_worker(&graph, &chunks, task, &[a, b]).map(|(id, _)| id),
            Some("b")
        );
    }

    /// The fallback that matters: a cold start, and a worker that has just
    /// joined. Nobody holds anything, every score is equal, and equal leaves the
    /// asker entitled — so the handout is exactly what it was without this.
    #[test]
    fn an_uninformative_model_withholds_nothing() {
        let (graph, chunks) = barrier();
        let task = barrier_task(&graph);
        let empty = cache();
        let cold = Residency {
            id: "cold",
            resident: Some(&empty),
            produced: Some(&empty),
        };
        let unknown = Residency {
            id: "new",
            resident: None,
            produced: None,
        };
        let ready = vec![task];
        for asker in [&cold, &unknown] {
            for contender in [&cold, &unknown] {
                assert_eq!(
                    entitled(
                        &ready,
                        &graph,
                        &chunks,
                        &phase_tasks(),
                        asker,
                        std::slice::from_ref(contender)
                    )
                    .as_ref(),
                    ready.as_slice(),
                    "{} vs {}: with no information there is no reason to prefer anybody",
                    asker.id,
                    contender.id
                );
            }
        }
    }

    /// Whatever the scores, somebody may claim the task — otherwise a barrier
    /// would deadlock the run rather than merely misplace it.
    #[test]
    fn somebody_is_always_entitled() {
        let (graph, chunks) = barrier();
        let task = barrier_task(&graph);
        let mut caches: Vec<(ModelledCache, ModelledCache)> =
            (0..4).map(|_| (cache(), cache())).collect();
        // Four workers with four different shares, including one with none.
        for (worker, share) in [(0usize, 0usize..0), (1, 0..1), (2, 1..4), (3, 4..8)] {
            for index in share {
                let (read, wrote) = &mut caches[worker];
                ran_first_phase(read, wrote, index);
            }
        }
        let all: Vec<Residency<'_>> = caches
            .iter()
            .enumerate()
            .map(|(index, (read, wrote))| Residency {
                id: ["w0", "w1", "w2", "w3"][index],
                resident: Some(read),
                produced: Some(wrote),
            })
            .collect();
        let ready = vec![task];
        let entitled_count = (0..all.len())
            .filter(|&asker| {
                let contenders: Vec<Residency<'_>> = (0..all.len())
                    .filter(|&other| other != asker)
                    .map(|other| Residency {
                        id: all[other].id,
                        resident: all[other].resident,
                        produced: all[other].produced,
                    })
                    .collect();
                !entitled(
                    &ready,
                    &graph,
                    &chunks,
                    &phase_tasks(),
                    &all[asker],
                    &contenders,
                )
                .is_empty()
            })
            .count();
        assert_eq!(
            entitled_count, 1,
            "exactly the best worker should be entitled, and at least one must be"
        );
    }

    /// The wall this rule hit, kept as a test because the fix is invisible
    /// otherwise.
    ///
    /// Scoring scarcity on the *ready set* rather than on the phase makes the
    /// last few blocks of every ordinary phase contestable, and refusing a
    /// worker there empties its work list while the coordinator still has work —
    /// which is exactly the failure `DISTRIBUTION.md` says must not happen,
    /// because prefetch would stop working multi-node while every single-node
    /// test kept passing. It broke
    /// `the_work_list_stays_at_least_one_task_ahead_of_what_is_being_computed`
    /// before the phase-shaped test replaced it.
    #[test]
    fn the_tail_of_an_ordinary_phase_is_never_contested() {
        let (graph, chunks) = barrier();
        let (mut read, mut wrote) = (cache(), cache());
        for index in 0..7 {
            ran_first_phase(&mut read, &mut wrote, index);
        }
        let rich = Residency {
            id: "rich",
            resident: Some(&read),
            produced: Some(&wrote),
        };
        let empty = cache();
        let poor = Residency {
            id: "poor",
            resident: Some(&empty),
            produced: Some(&empty),
        };
        // One block of the eight-block phase left, and three idle workers.
        let ready: Vec<usize> = (0..graph.len())
            .filter(|&task| graph.tasks[task].phase == 0)
            .take(1)
            .collect();
        let contenders = [
            Residency {
                id: "rich",
                resident: rich.resident,
                produced: rich.produced,
            },
            Residency {
                id: "other",
                resident: None,
                produced: None,
            },
        ];
        assert_eq!(
            entitled(&ready, &graph, &chunks, &phase_tasks(), &poor, &contenders).as_ref(),
            ready.as_slice(),
            "a phase of eight blocks is not scarce because seven of them are gone"
        );
    }

    #[test]
    fn scarcity_is_strictly_fewer_tasks_in_the_phase_than_workers() {
        // three workers: the asker and two contenders
        assert!(is_scarce(1, 2), "a barrier phase, three workers");
        assert!(is_scarce(2, 2));
        assert!(!is_scarce(3, 2), "work for everybody");
        assert!(!is_scarce(4, 2));
        // one worker on its own contends with nobody
        assert!(!is_scarce(1, 0), "a barrier phase, one worker");
        assert!(!is_scarce(0, 4));
    }

    #[test]
    fn overlap_counts_a_chunk_held_in_both_tiers_once() {
        let chunks = ChunkGrid::new([64, 8, 8], [8, 8, 8]);
        let (mut read, mut wrote) = (cache(), cache());
        let block = Region::new(&[0, 0, 0], &[8, 8, 8]);
        read.note_assigned(&chunks.keys(0, &block));
        wrote.note_assigned(&chunks.keys(0, &block));
        let worker = Residency {
            id: "w",
            resident: Some(&read),
            produced: Some(&wrote),
        };
        assert_eq!(worker.overlap(&chunks.keys(0, &block)), 1);
    }
}
