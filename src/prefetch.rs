// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A scheduler, not a predictor
// ============================
//
// The usual prefetcher guesses: it watches the access stream, infers a stride,
// and reads ahead. That is necessary when the future is unknown. **Here it is
// known.** The block plan is enumerated up front — every block's read extent is
// decided before the first byte moves — so a worker can *declare* its future
// reads and this becomes a scheduling problem with an exact input. There is
// nothing to infer and therefore nothing to infer wrongly.
//
// What follows from that:
//
// * **Rank, not recency.** A request carries the rank at which it will be
//   needed, usually its block's index in the plan. The queue is a min-heap on
//   `(rank, submission order)`, so the chunks nearest to being wanted are
//   fetched first, whatever order plans were submitted in.
// * **Depth is a lever with a cost, and the cost is measurable.** Depth here is
//   the number of concurrent fetches. Deeper hides more latency and, on storage
//   where concurrent readers scale, buys throughput as well — but it also
//   fetches further into a future that may not happen. When it does not, the
//   entry is evicted or the plan is cancelled, and that is emitted as
//   `PrefetchWasted`. **Waste is the cost of depth and is what tells you the
//   depth is wrong**; nothing else in the system will.
// * **Cancellation is not optional.** A slab that finishes early, or is
//   abandoned, must not leave reads in flight competing for the budget and the
//   disk with the work that replaced it. `cancel` drops a plan's queued
//   requests and, optionally, the entries it has already placed.
//
// It cannot block compute, structurally
// -------------------------------------
// Three separate mechanisms, because one would not be enough:
//
// 1. Fetching happens on this component's own threads. `submit` puts items on a
//    heap and returns; a compute thread never enters a fetch.
// 2. Retention takes an `Opportunistic` lease, which is refused outright while
//    any `Reserved` request is queueing. A prefetch that cannot be retained is
//    dropped, not waited on.
// 3. Before doing the work at all, a worker checks whether compute is waiting
//    (`MemoryBudget::revoking`) and declines if it is — so under pressure the
//    prefetcher stops spending IO and CPU too, not only memory.
//
// The first is what the test measures; the second and third are what make the
// measurement hold under load rather than only when the machine is idle.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::cache::{ArrayId, ChunkCache};
use crate::region::Region;

/// A read a worker knows it will make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionRequest {
    pub array: ArrayId,
    pub region: Region,
    /// Lower is needed sooner. Usually the block's index in the plan.
    pub rank: u32,
}

impl RegionRequest {
    pub fn new(array: ArrayId, region: Region, rank: u32) -> Self {
        Self {
            array,
            region,
            rank,
        }
    }
}

/// Something that can state its future reads.
///
/// Deliberately a trait over a `Vec` rather than a `Vec`: a plan for a large
/// run may be generated rather than stored, and the prefetcher only needs to
/// enumerate it once.
pub trait AccessPlan {
    fn requests(&self) -> Vec<RegionRequest>;
}

impl AccessPlan for Vec<RegionRequest> {
    fn requests(&self) -> Vec<RegionRequest> {
        self.clone()
    }
}

impl AccessPlan for [RegionRequest] {
    fn requests(&self) -> Vec<RegionRequest> {
        self.to_vec()
    }
}

/// A submitted plan, for cancelling it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanHandle(u64);

impl PlanHandle {
    /// A handle from a raw identifier, for a caller that stores one across a
    /// process boundary. Cancelling a handle this prefetcher never issued is
    /// harmless, which is what makes that safe.
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

/// What the prefetcher did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Requests accepted onto the queue.
    pub submitted: u64,
    /// Requests taken off the queue and acted on.
    pub started: u64,
    /// Chunks actually fetched. Lower than `started` when the cache already
    /// held them — which is a *good* outcome, not a wasted one.
    pub chunks: u64,
    /// Requests dropped because their plan was cancelled before they ran.
    pub cancelled: u64,
    /// Requests skipped because compute was queueing for the budget.
    pub declined: u64,
    /// Requests whose fetch returned an error. A prefetch failure is never
    /// propagated — the demand read will hit the same error and report it with
    /// a caller to report it to.
    pub failed: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct Item {
    rank: u32,
    seq: u64,
    plan: u64,
    array: ArrayId,
    region: Region,
}

// Ordered by `(rank, seq)` only. `Reverse<Item>` in a `BinaryHeap` then pops
// the lowest rank, and ties break in submission order so a plan's own
// enumeration order is preserved.
impl Ord for Item {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.rank, self.seq).cmp(&(other.rank, other.seq))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct Queue {
    heap: BinaryHeap<Reverse<Item>>,
    cancelled: HashSet<u64>,
    next_plan: u64,
    next_seq: u64,
    in_flight: usize,
    shutdown: bool,
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    started: AtomicU64,
    chunks: AtomicU64,
    cancelled: AtomicU64,
    declined: AtomicU64,
    failed: AtomicU64,
}

struct Shared {
    cache: Arc<ChunkCache>,
    queue: Mutex<Queue>,
    /// Signalled when there is work, or when shutting down.
    work: Condvar,
    /// Signalled when a request finishes, so `drain` can tell when the queue is
    /// genuinely quiet rather than merely empty.
    quiet: Condvar,
    counters: Counters,
}

/// Reads ahead into a [`ChunkCache`], in rank order, on its own threads.
pub struct Prefetcher {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Prefetcher {
    /// `depth` concurrent fetches, on `depth` threads.
    ///
    /// Depth is the lever. One thread hides latency against compute and nothing
    /// more; several are genuinely several concurrent requests to storage,
    /// which on a networked filesystem is a throughput gain and not only a
    /// latency one. Sweep it against `PrefetchStats` and the cache's waste
    /// counters rather than guessing — the point of emitting waste is that this
    /// number can be tuned from evidence.
    pub fn new(cache: Arc<ChunkCache>, depth: usize) -> Self {
        let depth = depth.max(1);
        let shared = Arc::new(Shared {
            cache,
            queue: Mutex::new(Queue::default()),
            work: Condvar::new(),
            quiet: Condvar::new(),
            counters: Counters::default(),
        });
        let workers = (0..depth)
            .map(|_| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || worker(shared))
            })
            .collect();
        Self { shared, workers }
    }

    /// Register a plan. Returns immediately; the reads happen elsewhere.
    pub fn submit(&self, plan: &dyn AccessPlan) -> PlanHandle {
        let requests = plan.requests();
        let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        let handle = PlanHandle(queue.next_plan);
        queue.next_plan += 1;
        for request in requests {
            let seq = queue.next_seq;
            queue.next_seq += 1;
            queue.heap.push(Reverse(Item {
                rank: request.rank,
                seq,
                plan: handle.0,
                array: request.array,
                region: request.region,
            }));
            self.shared
                .counters
                .submitted
                .fetch_add(1, Ordering::Relaxed);
        }
        drop(queue);
        self.shared.work.notify_all();
        handle
    }

    /// Drop a plan's outstanding requests.
    ///
    /// Queued items are discarded as they surface rather than searched for and
    /// removed: a heap has no cheap removal, cancellation is rare, and the
    /// check on pop is one hash lookup. Anything already in flight finishes —
    /// interrupting a storage read mid-flight is not something this can do, and
    /// pretending otherwise would be worse than letting one read complete.
    pub fn cancel(&self, handle: PlanHandle) {
        let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        queue.cancelled.insert(handle.0);
        drop(queue);
        self.shared.work.notify_all();
    }

    /// Cancel, and additionally drop the entries the plan placed but nobody
    /// read.
    ///
    /// The stronger form, for a plan that is *abandoned* rather than finished:
    /// what it speculatively loaded is now known to be useless, and holding it
    /// would evict things that are not. Counted as `PrefetchWaste::Cancelled`,
    /// which is the honest label — the depth was not wrong, the plan was.
    pub fn cancel_and_release(&self, handle: PlanHandle, arrays: &[ArrayId]) -> usize {
        self.cancel(handle);
        arrays
            .iter()
            .map(|&array| self.shared.cache.drop_unused_prefetches(array))
            .sum()
    }

    /// Block until nothing is queued or in flight.
    ///
    /// For tests and for a stage barrier. Not for the read path — a reader that
    /// waited for the prefetcher would be exactly the coupling this component
    /// is built to avoid.
    pub fn drain(&self) {
        let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        while !queue.heap.is_empty() || queue.in_flight > 0 {
            let (next, _) = self
                .shared
                .quiet
                .wait_timeout(queue, Duration::from_millis(20))
                .unwrap_or_else(|p| p.into_inner());
            queue = next;
        }
    }

    /// How many requests are queued but not yet started.
    pub fn queued(&self) -> usize {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .heap
            .len()
    }

    pub fn stats(&self) -> PrefetchStats {
        let counters = &self.shared.counters;
        PrefetchStats {
            submitted: counters.submitted.load(Ordering::Relaxed),
            started: counters.started.load(Ordering::Relaxed),
            chunks: counters.chunks.load(Ordering::Relaxed),
            cancelled: counters.cancelled.load(Ordering::Relaxed),
            declined: counters.declined.load(Ordering::Relaxed),
            failed: counters.failed.load(Ordering::Relaxed),
        }
    }

    pub fn cache(&self) -> &Arc<ChunkCache> {
        &self.shared.cache
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.shutdown = true;
            queue.heap.clear();
        }
        self.shared.work.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let item = {
            let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if queue.shutdown {
                    return;
                }
                if let Some(Reverse(item)) = queue.heap.pop() {
                    queue.in_flight += 1;
                    break item;
                }
                let (next, _) = shared
                    .work
                    // A timeout rather than a plain wait: shutdown and submit
                    // both notify, so this is not load-bearing, but a lost
                    // wakeup then costs a stall rather than a hang.
                    .wait_timeout(queue, Duration::from_millis(50))
                    .unwrap_or_else(|p| p.into_inner());
                queue = next;
            }
        };

        let cancelled = {
            let queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.cancelled.contains(&item.plan)
        };
        if cancelled {
            shared.counters.cancelled.fetch_add(1, Ordering::Relaxed);
        } else {
            shared.counters.started.fetch_add(1, Ordering::Relaxed);
            // `catch_unwind` because `in_flight` must come back down on every
            // path. A backend that panics mid-read would otherwise leave the
            // counter raised for ever, and `drain` — a stage barrier — would
            // wait for a request that no longer exists. A speculative read is
            // not worth a hang, and it is not worth taking the process down
            // either: the demand read that follows will hit the same panic
            // where there is a caller to see it.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                shared
                    .cache
                    .prefetch_region(item.array, &item.region, item.rank)
            }));
            match outcome {
                Ok(Ok(0)) => {
                    // Either everything was already resident — the good case —
                    // or the cache declined because compute is waiting. The
                    // cache counts the second; nothing is owed here.
                }
                Ok(Ok(chunks)) => {
                    shared
                        .counters
                        .chunks
                        .fetch_add(chunks as u64, Ordering::Relaxed);
                }
                Ok(Err(_)) | Err(_) => {
                    // Deliberately swallowed. A prefetch is speculative, so its
                    // failure is not the caller's business: the demand read
                    // that follows will hit the same error and report it to
                    // somebody who asked for the data.
                    shared.counters.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        queue.in_flight -= 1;
        drop(queue);
        shared.quiet.notify_all();
    }
}

/// A plan built from a block enumeration: read `region` of `array` at rank
/// `index`.
///
/// The common case, and the one that shows why this component is a scheduler:
/// the caller already has this list, because the decomposition produced it.
pub struct BlockPlan {
    requests: Vec<RegionRequest>,
}

impl BlockPlan {
    /// `regions` in the order they will be consumed; rank is the position.
    pub fn in_order(array: ArrayId, regions: impl IntoIterator<Item = Region>) -> Self {
        Self {
            requests: regions
                .into_iter()
                .enumerate()
                .map(|(rank, region)| RegionRequest::new(array, region, rank as u32))
                .collect(),
        }
    }

    /// Only the next `depth` ranks, for a caller that wants to bound how far
    /// ahead the queue reaches rather than relying on rank ordering alone.
    pub fn head(mut self, depth: usize) -> Self {
        self.requests.sort_by_key(|request| request.rank);
        self.requests.truncate(depth);
        self
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

impl AccessPlan for BlockPlan {
    fn requests(&self) -> Vec<RegionRequest> {
        self.requests.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_pop_in_rank_order_then_submission_order() {
        let mut heap: BinaryHeap<Reverse<Item>> = BinaryHeap::new();
        for (rank, seq) in [(5u32, 0u64), (1, 1), (5, 2), (0, 3)] {
            heap.push(Reverse(Item {
                rank,
                seq,
                plan: 0,
                array: ArrayId(0),
                region: Region::new(&[0], &[1]),
            }));
        }
        let order: Vec<(u32, u64)> = std::iter::from_fn(|| heap.pop())
            .map(|Reverse(item)| (item.rank, item.seq))
            .collect();
        assert_eq!(order, vec![(0, 3), (1, 1), (5, 0), (5, 2)]);
    }

    #[test]
    fn a_block_plan_ranks_regions_by_consumption_order() {
        let plan = BlockPlan::in_order(
            ArrayId(0),
            (0..4).map(|index| Region::new(&[index * 2], &[2])),
        );
        let requests = plan.requests();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[2].rank, 2);
        assert_eq!(requests[2].region, Region::new(&[4], &[2]));
    }

    #[test]
    fn head_keeps_only_the_soonest_ranks() {
        let plan =
            BlockPlan::in_order(ArrayId(0), (0..10).map(|index| Region::new(&[index], &[1])))
                .head(3);
        assert_eq!(plan.len(), 3);
        assert!(plan.requests().iter().all(|request| request.rank < 3));
    }
}
