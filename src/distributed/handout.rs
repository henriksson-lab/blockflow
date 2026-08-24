// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Which unclaimed task goes to which worker.
//
// The problem
// -----------
// Caches are per worker; there is no shared cache across nodes and there should
// not be. So **naive global pull destroys locality**: if adjacent blocks land on
// different workers the halo between them is fetched twice, once into each
// worker's cache, and neither gets the reuse that made the halo cheap on one
// node. Halo redundancy is already 1.35-2.22x from the decomposition alone;
// splitting neighbours across workers pays it again per seam.
//
// Why not territories
// -------------------
// Because a static spatial partition fights load balance, and hard. Empty
// regions **cluster** rather than scattering uniformly, so one worker gets a
// mostly-empty territory and another a dense one — which is exactly the
// imbalance pull-based handout exists to fix.
//
// The resolution: one queue, a locality-aware *choice*
// ----------------------------------------------------
// * **Seed workers far apart.** Two workers on one volume should start at
//   opposite edges and grow towards each other, not interleave. Maximally
//   distant seeding is a farthest-point choice over the ready set, and needs no
//   partition to maintain.
// * **Then nearest-unclaimed.** The coordinator knows what each worker last
//   completed, so it hands out the nearest ready task to that point.
// * **Degrades gracefully.** When a worker's neighbourhood is exhausted it takes
//   the nearest remaining task, wherever that is. Locality by default, balance
//   under pressure, and nothing to rebalance.
// * **Compact regions fall out.** Duplicated fetches are proportional to the
//   *surface* between workers' regions, so a worker growing a cube duplicates
//   less than one sweeping a slab — and nearest-first grows cubes.
//
// Position is voxels, not block indices
// -------------------------------------
// A task's position is the centre of its **core, in voxels**. Block indices
// would be cheaper and would be wrong: the design leaves per-phase block sizes
// open, so index `[3,0,0]` in one phase and in another name different places,
// and a distance between them would be meaningless. Voxels are the one
// coordinate system every phase shares.
//
// Cost
// ----
// A linear scan of the ready set per handout. At 256³ blocks a whole run is
// 5 597 handouts — 0.6 per second across eight nodes — so an O(ready) scan is
// free, and a policy that can be *read* is worth more here than one that is
// asymptotically better.

use crate::error::{Error, Result};
use crate::graph::TaskGraph;

use super::cache_model::{ChunkGrid, ModelledCache};

/// How the coordinator picks a task for a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandoutPolicy {
    /// The next task in plan order, ignoring who is asking.
    ///
    /// Kept, and kept selectable, because it is the baseline the others are
    /// measured against — "it is measurable, not arguable". It is also what a
    /// single global queue does if nobody thinks about it.
    Naive,
    /// Farthest-point seeding, then nearest-unclaimed to the worker's last
    /// completed block. The default.
    #[default]
    NearestFirst,
    /// Nearest-unclaimed, ranked first by how many chunks the coordinator
    /// *believes* the worker would have to fetch.
    ///
    /// The cache model is an estimate fed only by assignments; see
    /// `cache_model`. Distance is the tiebreak, so with an **empty** model this
    /// degrades exactly to `NearestFirst`.
    ///
    /// **"or useless" used to be in that sentence and is measured false.** An
    /// empty model misses every key of every candidate, so the primary sort term
    /// is constant and distance decides — that half holds. A *thrashing* model
    /// returns a miss count that **varies without carrying information**, and
    /// because `misses` outranks distance it dominates rather than defers.
    /// `super::tests::the_two_policies_are_indistinguishable_until_the_model_evicts`
    /// measures the cliff: holding two tasks' reads or more this policy is
    /// `NearestFirst` to the digit, and holding one task's or fewer it duplicates
    /// **22 fetches against 6** — a third of the way back to the naive pull the
    /// whole handout design exists to beat. It is never better at any capacity.
    ///
    /// **Not selectable today — [`HandoutPolicy::refusal`] says why and what
    /// would lift it.** The variant is kept, and kept constructible in-crate, so
    /// that the measurement which would lift the refusal can be taken: see
    /// `super::tests`. What is refused is a *caller* naming it, at
    /// [`HandoutPolicy::select`], which is the one boundary both the operator
    /// flag and a submitted `JobSpec` cross.
    CacheModelled,
}

impl HandoutPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::NearestFirst => "nearest-first",
            Self::CacheModelled => "cache-modelled",
        }
    }

    /// Which variant a name spells, whether or not a caller may select it.
    ///
    /// **Spelling, not admission.** [`Self::select`] is the admission boundary
    /// and is what anything reaching this crate from outside should call. The
    /// two are separate so that a refused variant still round-trips through its
    /// own name — a `JobSpec` recorded before a refusal landed stays readable,
    /// and a diagnostic can still print what a job asked for.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "naive" => Self::Naive,
            "nearest-first" | "nearest" => Self::NearestFirst,
            "cache-modelled" | "cache" => Self::CacheModelled,
            _ => return None,
        })
    }

    /// Why a caller may not select this policy today, or `None` if it may.
    ///
    /// A property of the **variant** rather than of the string, so that a caller
    /// holding one it obtained some other way can still ask, and so that a new
    /// entry point cannot acquire a policy without a way to find out.
    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Self::Naive | Self::NearestFirst => None,
            // The argument, in the order the evidence came in. The *set* the
            // model carries is real — chunks this worker was assigned to read is
            // derived from assignments the coordinator genuinely made, and it is
            // exactly the quantity a duplicated fetch is made of. Two things
            // about the ranking are not, and they compound:
            //
            //  * `nearest` sorts on `(misses, distance, task)`, so a modelled
            //    hit outranks distance outright, and the property `NearestFirst`
            //    was measured to buy — 62 duplicated chunks down to 6 on the
            //    harness in `super::tests` — is the thing a wrong model undoes.
            //    Unlike `placement::entitled`, which expresses the same
            //    preference as a *filter* and is therefore wrong in the cheap
            //    direction by construction, nothing bounds this one.
            //  * the LRU is sized by `WorkflowSpec::cache_bytes` and models
            //    `cache::ChunkCache`, which has **no non-test construction
            //    site**. What can physically serve a re-read on a node is the
            //    page cache, sized by free RAM — so the model understates
            //    residency most of the time, which is harmless, and *overstates*
            //    it exactly under memory pressure, which is not.
            //
            // And the measurement, which is what turns this from a caution
            // into a refusal: the two are indistinguishable until the model
            // thrashes, and then this one loses — 22 duplicated fetches against
            // 6 — while never winning at any capacity. A policy whose best case
            // is a tie with the thing it replaces has not earned a name.
            Self::CacheModelled => Some(
                "it ranks a modelled cache hit above distance while the cache it models does not exist — `cache::ChunkCache` has no non-test construction site, so no `Environment::read` is served from one. Its chunk *set* is real, and while the model holds two tasks' reads or more this policy is `nearest-first` to the digit. Below that it is measurably worse rather than merely uninformative: at a modelled capacity of one chunk it duplicates 22 fetches against `nearest-first`'s 6, a third of the way back to naive pull's 62, because a miss count that varies without carrying information still outranks distance. It is never better at any capacity. Lifted by a chunk cache on a read path whose real size is what the model is given — see `distributed::tests::the_two_policies_are_indistinguishable_until_the_model_evicts`, which is that measurement and which fails if this stops being true",
            ),
        }
    }

    /// The policy a caller **may** select, refused by name where it may not.
    ///
    /// The single admission boundary: `blockflow_local --policy` and
    /// `JobSpec::from_json` both come through here, so one refusal covers both
    /// entry points. It deliberately does **not** guard `Job::new`, because a
    /// policy set in Rust is a measurement taking it, and the measurement is
    /// what would lift the refusal.
    pub fn select(text: &str) -> Result<Self> {
        let policy = Self::parse(text).ok_or_else(|| {
            Error::invalid(format!(
                "{text:?} is not a handout policy: naive, nearest-first"
            ))
        })?;
        match policy.refusal() {
            None => Ok(policy),
            Some(why) => Err(Error::invalid(format!(
                "the handout policy {text:?} is not selectable: {why}"
            ))),
        }
    }
}

/// The centre of a task's core, in voxels.
pub fn position(graph: &TaskGraph, task: usize) -> [f64; 3] {
    let core = &graph.tasks[task].geometry.core;
    let mut out = [0.0f64; 3];
    for axis in 0..3 {
        let start = core.start.get(axis).copied().unwrap_or(0) as f64;
        let len = core.shape.get(axis).copied().unwrap_or(0) as f64;
        out[axis] = start + len / 2.0;
    }
    out
}

fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = left[axis] - right[axis];
            delta * delta
        })
        .sum()
}

/// What the coordinator knows about one worker, for the purposes of choosing.
///
/// Everything here is derived from assignments and completions the coordinator
/// already has. Nothing is asked for, and nothing waits.
pub struct WorkerView<'a> {
    /// The centre of the block this worker last completed. `None` before its
    /// first completion, which is the seeding case.
    pub anchor: Option<[f64; 3]>,
    /// The coordinator's model of this worker's cache.
    pub cache: Option<&'a ModelledCache>,
}

/// Where the other workers are, so a new one can be seeded away from them.
///
/// Their *anchors* rather than their claims: a worker's anchor is where it is
/// working, and seeding away from that is what makes two workers grow towards
/// each other from opposite edges instead of interleaving.
pub type Seeds<'a> = &'a [[f64; 3]];

/// Choose one ready task, or `None` if there are none.
///
/// `ready` is task ids, and the return value is one of them. Deterministic
/// under ties — a coordinator that chose differently on two identical states
/// would make a distributed run unreproducible for no benefit.
pub fn choose(
    policy: HandoutPolicy,
    ready: &[usize],
    graph: &TaskGraph,
    chunks: &ChunkGrid,
    worker: &WorkerView<'_>,
    seeds: Seeds<'_>,
) -> Option<usize> {
    if ready.is_empty() {
        return None;
    }
    match policy {
        HandoutPolicy::Naive => ready.iter().copied().min(),
        _ => match worker.anchor {
            None => Some(seed(ready, graph, seeds)),
            Some(anchor) => Some(nearest(policy, ready, graph, chunks, worker, anchor)),
        },
    }
}

/// A worker with no history starts as far as possible from every other worker.
///
/// Farthest-point: maximise the distance to the *nearest* existing seed. With
/// no seeds at all the choice is the lowest-positioned ready task, which is
/// arbitrary but deterministic — the first worker has to start somewhere and
/// nothing distinguishes one edge from another.
fn seed(ready: &[usize], graph: &TaskGraph, seeds: Seeds<'_>) -> usize {
    if seeds.is_empty() {
        return *ready
            .iter()
            .min_by(|&&left, &&right| {
                order(position(graph, left), left, position(graph, right), right)
            })
            .expect("ready is non-empty");
    }
    let mut best: Option<(f64, usize)> = None;
    for &task in ready {
        let here = position(graph, task);
        let nearest_seed = seeds
            .iter()
            .map(|&seed| distance_squared(here, seed))
            .fold(f64::INFINITY, f64::min);
        let better = match best {
            None => true,
            // Farther from everyone wins; a tie goes to the lower task id.
            Some((distance, id)) => {
                (nearest_seed, std::cmp::Reverse(task)) > (distance, std::cmp::Reverse(id))
            }
        };
        if better {
            best = Some((nearest_seed, task));
        }
    }
    best.expect("ready is non-empty").1
}

fn nearest(
    policy: HandoutPolicy,
    ready: &[usize],
    graph: &TaskGraph,
    chunks: &ChunkGrid,
    worker: &WorkerView<'_>,
    anchor: [f64; 3],
) -> usize {
    let mut best: Option<(usize, f64, usize)> = None;
    for &task in ready {
        let misses = match (policy, worker.cache) {
            (HandoutPolicy::CacheModelled, Some(cache)) => {
                let entry = &graph.tasks[task];
                // What the task will *fetch*, which is `source`: the two differ
                // only for a phase that reads across grids, and a cache model
                // keyed on the wrong region would be modelling chunks nobody
                // asks for.
                cache.misses(&chunks.keys(entry.phase, &entry.geometry.source))
            }
            _ => 0,
        };
        let distance = distance_squared(position(graph, task), anchor);
        let candidate = (misses, distance, task);
        let better = match best {
            None => true,
            Some(current) => less(candidate, current),
        };
        if better {
            best = Some(candidate);
        }
    }
    best.expect("ready is non-empty").2
}

/// Lexicographic on `(misses, distance, task)`, written out because `f64` is
/// not `Ord` and the tiebreak on task id is what makes the choice reproducible.
fn less(left: (usize, f64, usize), right: (usize, f64, usize)) -> bool {
    match left.0.cmp(&right.0) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => match left.1.partial_cmp(&right.1) {
            Some(std::cmp::Ordering::Less) => true,
            Some(std::cmp::Ordering::Greater) => false,
            _ => left.2 < right.2,
        },
    }
}

fn order(left: [f64; 3], left_id: usize, right: [f64; 3], right_id: usize) -> std::cmp::Ordering {
    for axis in 0..3 {
        match left[axis].partial_cmp(&right[axis]) {
            Some(std::cmp::Ordering::Equal) | None => continue,
            Some(other) => return other,
        }
    }
    left_id.cmp(&right_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposition::{Decomposition, PhaseDecomposition};
    use crate::dtype::Dtype;
    use crate::geometry::BlockGrid;

    /// A one-phase strip of 16 blocks along axis 0.
    fn strip() -> (TaskGraph, ChunkGrid) {
        let grid = BlockGrid::new([256, 16, 16], [16, 16, 16]).unwrap();
        let decomposition = Decomposition {
            volume: [256, 16, 16],
            dtype: Dtype::U16,
            phases: vec![PhaseDecomposition::derive(
                vec![0],
                vec!["op".to_string()],
                [4, 0, 0],
                [4, 0, 0],
                grid,
            )],
            chain_reach: [4, 0, 0],
        };
        (
            TaskGraph::build(&decomposition),
            ChunkGrid::new([256, 16, 16], [8, 16, 16]),
        )
    }

    fn view<'a>(anchor: Option<[f64; 3]>, cache: Option<&'a ModelledCache>) -> WorkerView<'a> {
        WorkerView { anchor, cache }
    }

    #[test]
    fn the_first_worker_starts_at_one_end_and_the_second_at_the_other() {
        let (graph, chunks) = strip();
        let ready: Vec<usize> = (0..graph.len()).collect();
        let first = choose(
            HandoutPolicy::NearestFirst,
            &ready,
            &graph,
            &chunks,
            &view(None, None),
            &[],
        )
        .unwrap();
        assert_eq!(first, 0);
        let second = choose(
            HandoutPolicy::NearestFirst,
            &ready,
            &graph,
            &chunks,
            &view(None, None),
            &[position(&graph, first)],
        )
        .unwrap();
        assert_eq!(
            second,
            graph.len() - 1,
            "the second worker must start far away"
        );
    }

    #[test]
    fn a_third_worker_lands_between_the_first_two() {
        let (graph, chunks) = strip();
        let ready: Vec<usize> = (0..graph.len()).collect();
        let seeds = [position(&graph, 0), position(&graph, graph.len() - 1)];
        let third = choose(
            HandoutPolicy::NearestFirst,
            &ready,
            &graph,
            &chunks,
            &view(None, None),
            &seeds,
        )
        .unwrap();
        assert!(
            third > 4 && third < graph.len() - 5,
            "expected a seed in the middle, got {third}"
        );
    }

    #[test]
    fn a_worker_with_history_takes_the_nearest_unclaimed_task() {
        let (graph, chunks) = strip();
        // Everything in the middle is claimed; this worker just finished 8.
        let ready = vec![0, 3, 11, 15];
        let chosen = choose(
            HandoutPolicy::NearestFirst,
            &ready,
            &graph,
            &chunks,
            &view(Some(position(&graph, 8)), None),
            &[],
        )
        .unwrap();
        assert_eq!(chosen, 11);
    }

    #[test]
    fn naive_pull_ignores_who_is_asking() {
        let (graph, chunks) = strip();
        let ready = vec![0, 3, 11, 15];
        for anchor in [None, Some(position(&graph, 12))] {
            assert_eq!(
                choose(
                    HandoutPolicy::Naive,
                    &ready,
                    &graph,
                    &chunks,
                    &view(anchor, None),
                    &[]
                ),
                Some(0)
            );
        }
    }

    #[test]
    fn the_cache_model_outranks_distance_when_it_has_something_to_say() {
        let (graph, chunks) = strip();
        let chunk_bytes = 8 * 16 * 16 * 2;
        let mut cache = ModelledCache::new(chunk_bytes * 64, chunk_bytes);
        // The worker has been given block 15's chunks but sits near block 8.
        let far = &graph.tasks[15];
        cache.note_assigned(&chunks.keys(far.phase, &far.geometry.read));
        let ready = vec![11, 15];
        let anchor = Some(position(&graph, 8));
        assert_eq!(
            choose(
                HandoutPolicy::NearestFirst,
                &ready,
                &graph,
                &chunks,
                &view(anchor, Some(&cache)),
                &[]
            ),
            Some(11),
            "distance alone prefers the near one"
        );
        assert_eq!(
            choose(
                HandoutPolicy::CacheModelled,
                &ready,
                &graph,
                &chunks,
                &view(anchor, Some(&cache)),
                &[]
            ),
            Some(15),
            "with the chunks already modelled resident, the far one is cheaper"
        );
    }

    #[test]
    fn an_empty_ready_set_chooses_nothing() {
        let (graph, chunks) = strip();
        for policy in [
            HandoutPolicy::Naive,
            HandoutPolicy::NearestFirst,
            HandoutPolicy::CacheModelled,
        ] {
            assert_eq!(
                choose(policy, &[], &graph, &chunks, &view(None, None), &[]),
                None
            );
        }
    }

    #[test]
    fn the_choice_is_the_same_every_time_it_is_asked() {
        let (graph, chunks) = strip();
        let ready: Vec<usize> = (0..graph.len()).rev().collect();
        let anchor = Some(position(&graph, 4));
        let first = choose(
            HandoutPolicy::NearestFirst,
            &ready,
            &graph,
            &chunks,
            &view(anchor, None),
            &[],
        );
        for _ in 0..10 {
            assert_eq!(
                choose(
                    HandoutPolicy::NearestFirst,
                    &ready,
                    &graph,
                    &chunks,
                    &view(anchor, None),
                    &[]
                ),
                first
            );
        }
    }

    #[test]
    fn policy_names_round_trip() {
        for policy in [
            HandoutPolicy::Naive,
            HandoutPolicy::NearestFirst,
            HandoutPolicy::CacheModelled,
        ] {
            assert_eq!(HandoutPolicy::parse(policy.as_str()), Some(policy));
        }
        assert_eq!(HandoutPolicy::parse("territories"), None);
    }

    /// **The refusal, and the two things that make it a stated state rather than
    /// a wall.**
    ///
    /// A refused policy still *spells*, so a recorded job stays readable and the
    /// error can name what was asked for; and the reason travels with the
    /// refusal, so a caller learns why here rather than in a document.
    #[test]
    fn the_cache_modelled_policy_is_refused_at_the_admission_boundary() {
        // It is still a name, and still that variant.
        assert_eq!(
            HandoutPolicy::parse("cache-modelled"),
            Some(HandoutPolicy::CacheModelled)
        );
        assert_eq!(
            HandoutPolicy::parse("cache"),
            Some(HandoutPolicy::CacheModelled)
        );

        // And it is not selectable, under either spelling.
        for name in ["cache-modelled", "cache"] {
            let error = HandoutPolicy::select(name).expect_err("refused");
            let message = error.to_string();
            assert!(
                message.contains("not selectable"),
                "the refusal must say it is a refusal: {message}"
            );
            // The reason, not merely the fact. Each clause is one of the three
            // things a reader needs: what is wrong, what is *not* wrong, and
            // what would lift it.
            for clause in ["ChunkCache", "nearest-first", "evict", "Lifted by"] {
                assert!(
                    message.contains(clause),
                    "the refusal must carry {clause:?}: {message}"
                );
            }
        }

        // The two that are selectable, are — this is the control, and without it
        // a `select` that refused everything would pass every assertion above.
        for policy in [HandoutPolicy::Naive, HandoutPolicy::NearestFirst] {
            assert_eq!(policy.refusal(), None);
            assert_eq!(
                HandoutPolicy::select(policy.as_str()).expect("selectable"),
                policy
            );
        }
        assert!(HandoutPolicy::CacheModelled.refusal().is_some());

        // An unknown name is a different error from a refused one, and says so.
        let unknown = HandoutPolicy::select("territories").expect_err("not a policy");
        assert!(
            unknown.to_string().contains("is not a handout policy"),
            "{unknown}"
        );
        assert!(
            !unknown.to_string().contains("not selectable"),
            "an unknown name is not a refusal: {unknown}"
        );
    }
}
