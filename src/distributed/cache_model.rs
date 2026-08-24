// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The coordinator's picture of what each worker holds — **derived, never
// reported.**
//
// Why not simply ask
// ------------------
// The obvious implementation is for workers to report cache inserts and
// evictions. They do not have to. The coordinator assigned every task, so it
// knows the exact sequence of regions each worker has read; the cache is an LRU
// with a known byte budget; the decomposition is deterministic. Replaying the
// same eviction policy against what was handed out predicts each worker's
// contents with no report at all, and the traffic that avoids is the point —
// chunk fetches run at roughly the rate of the whole rest of the event stream,
// so reporting each insert and eviction would about double it, for information
// the coordinator could have derived.
//
// What this models, and what exists
// ---------------------------------
// **The set is a fact; the eviction is not.** Which chunks a worker has been
// assigned to read is something the coordinator genuinely knows, and it is
// exactly the quantity a *duplicated fetch* is made of — a chunk two workers
// both read costs two fetches whatever either node caches. That much needs no
// cache to be true, and it is the whole of what the handout and the placement
// filter are entitled to lean on.
//
// The LRU below is a different claim. It is sized by `WorkflowSpec::cache_bytes`
// and was written as a model of `cache::ChunkCache` — which has **no non-test
// construction site**, so no `Environment::read` is served from one. What can
// physically serve a re-read on a node is the page cache, sized by free RAM.
// So the eviction understates residency most of the time, which is harmless,
// and **overstates it exactly under memory pressure**, which is not.
//
// That is why `HandoutPolicy::CacheModelled`, which ranks on the eviction, is
// refused at `HandoutPolicy::select`, while `placement::entitled`, which uses
// the set as a strictly-better filter on scarce phases, is unchanged.
//
// The model will drift, and that is *why* this works
// --------------------------------------------------
// A worker's real cache can differ: a lease refused under memory pressure, a
// cancelled prefetch, a block short-circuited so its chunks were never read. So
// this is an estimate. But it feeds a handout *preference*, and the governing
// property of the whole design applies unchanged: **a wrong guess costs a
// fetch, never a wrong result.** The same reasoning that licenses a greedy
// scheduler licenses an approximate cache model.
//
// The constraint that keeps it honest
// -----------------------------------
// This type is fed **only** from what the coordinator already knows —
// assignments it made. It has no method that takes a report from a worker, and
// it must not grow one that a handout waits on. Prohibited, because each looks
// harmless: a worker blocking to report cache state before taking its next
// task; the coordinator querying a worker and waiting before deciding; any
// barrier or round that makes one node's progress depend on another's. If a
// summary is ever added it must be fire-and-forget, piggybacked on a message
// already being sent, and never on the handout path — a late or lost summary
// must degrade the *estimate*, not stall a worker.
//
// This is the same principle the listener trait enforces on the hot path:
// observation must never be able to slow down or block the thing it observes.
// The cache model is observation.
//
// Size
// ----
// Tens of thousands of entries — a 16 GB budget at 128³ `u16` chunks is 3 815
// per node, 30 518 across eight. That is a `HashSet` and a `VecDeque`, not a
// problem.

use std::collections::{HashSet, VecDeque};

use crate::region::Region;

/// The chunk lattice a region is read through.
///
/// Chunk identity is a flat index over this lattice, which is why the model
/// needs no chunk *contents* and no communication to compute one: the lattice
/// is a property of the array, and both ends already have it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkGrid {
    volume: [usize; 3],
    chunk: [usize; 3],
    counts: [usize; 3],
}

impl ChunkGrid {
    pub fn new(volume: [usize; 3], chunk: [usize; 3]) -> Self {
        let mut counts = [1usize; 3];
        for axis in 0..3 {
            let edge = chunk[axis].max(1);
            counts[axis] = volume[axis].div_ceil(edge).max(1);
        }
        Self {
            volume,
            chunk: [chunk[0].max(1), chunk[1].max(1), chunk[2].max(1)],
            counts,
        }
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn chunk(&self) -> [usize; 3] {
        self.chunk
    }

    pub fn chunks_per_axis(&self) -> [usize; 3] {
        self.counts
    }

    pub fn n_chunks(&self) -> u64 {
        self.counts.iter().map(|&count| count as u64).product()
    }

    /// Every chunk a read of `region` at `image` touches.
    ///
    /// `image` is in the key because a run holds several arrays at once — the
    /// input and one intermediate per phase boundary — and chunk 7 of image 0
    /// is not chunk 7 of image 1. A model keyed on the chunk alone would
    /// believe a phase-1 task's chunks were already resident because a phase-0
    /// task had read the same *coordinates* from a different array.
    pub fn keys(&self, image: usize, region: &Region) -> Vec<u64> {
        let mut out = Vec::new();
        if region.voxels() == 0 {
            return out;
        }
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for axis in 0..3 {
            let start = region.start.get(axis).copied().unwrap_or(0);
            let len = region.shape.get(axis).copied().unwrap_or(0);
            lo[axis] = start / self.chunk[axis];
            hi[axis] = ((start + len - 1) / self.chunk[axis]).min(self.counts[axis] - 1);
        }
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for k in lo[2]..=hi[2] {
                    let flat = ((i * self.counts[1]) + j) * self.counts[2] + k;
                    out.push(key(image, flat as u64));
                }
            }
        }
        out
    }
}

/// `(image, chunk)` in one integer, so the model is a set of `u64`.
///
/// Forty bits of chunk index is a trillion chunks; an image count that overflows
/// twenty-four bits would be a decomposition with sixteen million phases.
fn key(image: usize, chunk: u64) -> u64 {
    ((image as u64) << 40) | (chunk & 0xff_ffff_ffff)
}

/// What the coordinator believes one worker's cache holds.
///
/// An LRU over chunk keys with a byte budget, replayed from assignments. It
/// carries no data — only which keys would be resident — because the only
/// question asked of it is "would this task's read be served from memory?".
#[derive(Debug, Clone)]
pub struct ModelledCache {
    budget_bytes: u64,
    chunk_bytes: u64,
    capacity: usize,
    resident: HashSet<u64>,
    /// Front is least recently used. A `VecDeque` plus a set rather than a
    /// linked hash map: a touch is a removal and a push, the sequence is short,
    /// and this is a model of a cache rather than a cache.
    order: VecDeque<u64>,
    inserted: u64,
    evicted: u64,
}

impl ModelledCache {
    /// `chunk_bytes` is a decoded chunk's size — `chunk voxels x dtype width`.
    pub fn new(budget_bytes: u64, chunk_bytes: u64) -> Self {
        let chunk_bytes = chunk_bytes.max(1);
        // At least one entry, so a budget smaller than a chunk models a cache
        // that thrashes rather than one that cannot exist. The real cache
        // degrades to a pass-through in that situation, which costs a fetch —
        // and a fetch is what a modelled miss predicts anyway.
        let capacity = (budget_bytes / chunk_bytes).max(1) as usize;
        Self {
            budget_bytes,
            chunk_bytes,
            capacity,
            resident: HashSet::new(),
            order: VecDeque::new(),
            inserted: 0,
            evicted: 0,
        }
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.resident.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resident.is_empty()
    }

    pub fn inserted(&self) -> u64 {
        self.inserted
    }

    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    pub fn holds(&self, key: u64) -> bool {
        self.resident.contains(&key)
    }

    /// How many of `keys` this worker would have to fetch.
    ///
    /// Read-only: scoring a candidate must not move it up the LRU, or the act
    /// of considering a task would change the model of every worker that was
    /// considered for it.
    pub fn misses(&self, keys: &[u64]) -> usize {
        keys.iter()
            .filter(|key| !self.resident.contains(key))
            .count()
    }

    /// Record that this worker was assigned a read of these chunks.
    ///
    /// The only way state enters this type, and it is called from the handout,
    /// with the coordinator's own lock already held. Nothing waits.
    pub fn note_assigned(&mut self, keys: &[u64]) {
        for &key in keys {
            if self.resident.contains(&key) {
                if let Some(position) = self.order.iter().position(|&entry| entry == key) {
                    self.order.remove(position);
                }
            } else {
                self.resident.insert(key);
                self.inserted += 1;
            }
            self.order.push_back(key);
        }
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.resident.remove(&oldest);
                self.evicted += 1;
            }
        }
    }

    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> ChunkGrid {
        ChunkGrid::new([64, 16, 16], [16, 16, 16])
    }

    #[test]
    fn a_region_names_the_chunks_it_actually_touches() {
        let grid = grid();
        assert_eq!(grid.chunks_per_axis(), [4, 1, 1]);
        assert_eq!(grid.n_chunks(), 4);
        // wholly inside one chunk
        assert_eq!(
            grid.keys(0, &Region::new(&[0, 0, 0], &[8, 16, 16])).len(),
            1
        );
        // straddling a boundary
        assert_eq!(
            grid.keys(0, &Region::new(&[8, 0, 0], &[16, 16, 16])).len(),
            2
        );
        // the whole volume
        assert_eq!(grid.keys(0, &Region::whole(&[64, 16, 16])).len(), 4);
    }

    #[test]
    fn an_image_is_part_of_the_identity() {
        let grid = grid();
        let region = Region::new(&[0, 0, 0], &[16, 16, 16]);
        assert_ne!(grid.keys(0, &region), grid.keys(1, &region));
    }

    #[test]
    fn an_empty_region_touches_nothing() {
        assert!(grid()
            .keys(0, &Region::new(&[0, 0, 0], &[0, 0, 0]))
            .is_empty());
    }

    #[test]
    fn the_model_evicts_least_recently_used_at_its_byte_budget() {
        // three chunks' worth of budget, four chunks of demand
        let chunk_bytes = 16 * 16 * 16 * 2;
        let mut cache = ModelledCache::new(chunk_bytes * 3, chunk_bytes);
        assert_eq!(cache.capacity(), 3);
        let grid = grid();
        for block in 0..4 {
            cache.note_assigned(&grid.keys(0, &Region::new(&[block * 16, 0, 0], &[16, 16, 16])));
        }
        assert_eq!(cache.len(), 3);
        // the first is gone, the last three are held
        assert!(!cache.holds(grid.keys(0, &Region::new(&[0, 0, 0], &[16, 16, 16]))[0]));
        for block in 1..4 {
            assert!(cache.holds(grid.keys(0, &Region::new(&[block * 16, 0, 0], &[16, 16, 16]))[0]));
        }
        assert_eq!(cache.evicted(), 1);
    }

    #[test]
    fn a_re_read_moves_an_entry_up_rather_than_inserting_it_twice() {
        let mut cache = ModelledCache::new(300, 100);
        cache.note_assigned(&[1, 2, 3]);
        cache.note_assigned(&[1]);
        cache.note_assigned(&[4]);
        // 2 was the oldest once 1 was touched again, so 2 went, not 1
        assert!(cache.holds(1));
        assert!(!cache.holds(2));
        assert_eq!(cache.inserted(), 4);
    }

    #[test]
    fn scoring_a_candidate_does_not_change_the_model() {
        let mut cache = ModelledCache::new(200, 100);
        cache.note_assigned(&[1, 2]);
        let before = (cache.len(), cache.holds(1), cache.holds(2));
        assert_eq!(cache.misses(&[1, 2, 3]), 1);
        assert_eq!(cache.misses(&[9]), 1);
        assert_eq!((cache.len(), cache.holds(1), cache.holds(2)), before);
    }
}
