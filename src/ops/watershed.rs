// SPDX-License-Identifier: MIT
//
// Original work for this crate. **The kernel is not**: the flood itself lives
// in `super::scikitimage_watershed`, which is a translation of scikit-image and
// carries BSD-3-Clause. This file is the shell that adapts it, and the two are
// separate files so that the notice travels with the code it belongs to.
//
// A seeded watershed: partition a cost volume into one basin per seed.
// ====================================================================
//
// What it is, and what it is not
// ------------------------------
// Given a cost volume, a set of labelled seeds and a mask, flood outward from
// every seed at once, lowest cost first, and give every masked voxel the label
// of the seed whose flood reached it. Optionally leave a one-voxel line of
// label 0 wherever two floods met ([`Separation::Line`]).
//
// Two ops already here are near neighbours and neither can stand in:
//
// * `reconstruct` is a *reconstruction* — a marker grown under a ceiling — and
//   produces a greyscale volume, not a partition. It has no notion of which
//   marker a voxel came from, which is the entire output here.
// * `components` is an *equivalence relation* over a mask, and a basin boundary
//   is not one. Two voxels of one basin need not be connected inside it: with
//   [`Separation::Line`] the line clears voxels *between* labels, so a basin is
//   routinely cut into several connected pieces. The test
//   `a_basin_is_not_a_connected_component` measures the gap rather than asserting
//   it: on a `32 x 32 x 24` fixture, **138 basins against 383 six-connected
//   components** of the same labelled voxels. That is not a rounding error, so
//   counting components of the labelled output is not a way to count basins and
//   `ops::detect`'s `Moments::count` over them is not a basin size.
//
// The cost volume is the caller's
// -------------------------------
// This op floods *up* a cost: low first. A caller who wants basins around
// **bright** structure passes `-intensity`, which is one `voxelwise` map, and a
// caller who wants a distance-transform watershed passes the distance. Building
// the cost here would be one convention baked into an op that has no reason to
// hold one; the mask is the caller's for the same reason, and is one threshold.
//
// **`super::distance` is where that distance comes from** —
// `distance::distance_transform` resident, `distance::plan` blocked. The
// negation is the same one `voxelwise` map as above, because a
// distance-transform watershed seeds at the field's *maxima* and this op floods
// from the lowest cost. `distance::squared_distance_transform` is the cheaper
// input where only the ordering matters, which is every case where the basins
// are wanted and the distances are not: the square root is monotone, so it
// changes no comparison between two distinct values, and the squared field is
// the one whose values are exact integers rather than a libm's last bit.
//
// **A barrier, and why that is the honest declaration**
// ----------------------------------------------------
// This op declares [`Reach::all`], which makes it a planning barrier: it gets a
// phase of its own with the grid collapsed to a single block
// (`decomposition::is_planning_barrier`, `decomposition::splittable_axes`).
// That is a declared property of the op, not a special case in the planner.
//
// It is declared because **the answer is a function of one global queue's pop
// order over the whole volume**, and three separate things make that so:
//
// 1. Priority is `(cost, age)` where `age` is a global counter incremented once
//    per push. Two voxels of equal cost are ordered by which flood reached the
//    queue first *anywhere in the volume*. A block that started its own counter
//    at zero would order them differently.
// 2. Ties on both keys are resolved by the queue array's internal layout, so
//    even the seeds' relative order — they all carry `age == 0` — depends on how
//    many seeds there are and what order they were inserted in. Cutting the
//    volume changes both.
// 3. A voxel's priority is raised to its source's, so a flood carries the worst
//    cost it has crossed. That history is unbounded in length: a basin can be
//    decided by a barrier hundreds of voxels away, which is a distance no halo
//    is willing to be.
//
// (1) and (3) are properties of the algorithm; (2) is a property of an
// implementation, and it is reproduced rather than replaced because the
// reference this op must agree with bit-for-bit has it. A *different* seeded
// watershed could be decomposable. This one is not, and the test
// `a_blocked_run_is_not_the_whole_volume_answer` runs one by hand and counts the
// voxels that move rather than arguing about it.
//
// The scalable alternative has a different, stated answer
// ---------------------------------------------------------
// [`scalable_seeded_watershed`] keeps this exact flood for bounded connected
// components and sends a component whose bounding box is too large to
// [`blockwise_minimax_watershed`]. That fallback defines the watershed as a
// deterministic minimax path problem instead of reproducing the reference
// heap's global `age`. It first settles the least barrier at every voxel, then
// settles the shortest optimal-barrier path with label as the final tie. Each
// block retains its queues between activations; boundary
// improvements dirty only the receiving blocks, and one dirty wave runs in
// parallel. The exact op remains here unchanged because the two contracts are
// useful for different reasons and neither should silently impersonate the
// other.
//
// What a barrier costs, as arithmetic
// -----------------------------------
// One block spanning the volume means the phase's resident set is the volume,
// and it is worth having the number rather than the adjective. Per voxel of the
// volume, while the phase runs:
//
// | array | width |
// |---|---|
// | cost (the input) | 8 B (`f64`) |
// | seeds (a source image) | 4 B (`u32`) |
// | mask (a source image) | 1 B (`bool`) |
// | labels (the output) | 4 B (`u32`) |
// | the kernel's own copy of the mask | 1 B (`bool`) |
// | | **18 B/voxel** |
//
// Only the last row is this op's; the four above it are what the framework hands
// it, and the kernel **borrows** all four rather than copying them — the cost
// slice and the label slice go in as they are, and the seeds are written into
// the output that is about to be flooded in place. The mask is the one array
// that has to be duplicated, because carving a separating line means clearing
// voxels in it and the image it came from has other readers. A naive shell that
// materialised each of the four would cost 30 B/voxel, which on a 1024^3 volume
// is 13 GiB of pure copy.
//
// **The queue sits on top of that, and it is not the small term it looks like.**
// One queued item is 32 B (`scikitimage_watershed::HEAP_ITEM_BYTES`) and the
// queue is bounded above by six pushes per *masked* voxel, i.e. **192 B per
// masked voxel** — eleven times the dense figure, so anything above about a
// tenth of the volume being masked makes the queue the larger term. It sits well
// below its bound, because it holds a flood **front** rather than a history, and
// how far below is a fact about the data rather than about the algorithm.
// `the_queue_is_a_front_not_a_history` measures it: 20.4% of the bound on a
// volume 82% masked, 5.1% at 47%, 0.04% at 18% — which is 32, 4.6 and 0.02 bytes
// per *volume* voxel respectively.
//
// So the arithmetic a machine is sized on, taking the densely-masked case as the
// one nobody should be surprised by (18 B dense + 32 B queue = **50 B/voxel**)
// and a sparse mask as the one a caller usually has (**~18 B/voxel**):
//
// | volume | sparse mask | 80% masked |
// |---|---|---|
// | 256^3 | 288 MiB | 800 MiB |
// | 512^3 | 2.3 GiB | 6.3 GiB |
// | 1024^3 | 18 GiB | 50 GiB |
// | 2048 x 2048 x 1024 | 72 GiB | 200 GiB |
//
// So the barrier is affordable to about 1024^3 on a large node with a sparse
// mask, and not beyond it. The thing a caller does about that is cut the
// *problem* rather than the op: this op is defined on whatever volume it is
// handed, so a caller with a volume too large for it runs it on a region whose
// faces fall in background — which is a decision about the data that this crate
// has no way to make, and is why there is no parameter here for it.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Mutex;

use ndarray::{s, Array3, ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::scikitimage_watershed::watershed_raveled_reporting_peak;
use super::shapes_agree;
use super::{components::label_members_into, components::FACE_NEIGHBOURS};
use super::{LabelSeedSource, MaskSource};

/// What happens where two floods meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separation {
    /// The meeting voxels are cleared to label 0, leaving a one-voxel line
    /// between basins. Every basin is then separated from every other, and the
    /// labelled voxels are **fewer** than the masked ones.
    Line,
    /// Every masked voxel takes a label. Basins touch, and the boundary between
    /// two of them is wherever the floods happened to meet.
    Adjacent,
}

/// How the scalable watershed chooses between the exact reference flood and
/// the blockwise fallback.
///
/// The limit is on a component's **bounding box**, rather than on its foreground
/// voxel count, because the reference flood materialises that box densely. A
/// thin component winding through a large region is therefore correctly sent
/// to the bounded-work fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalableWatershedConfig {
    /// Largest component bounding box passed to [`seeded_watershed`].
    pub exact_component_max_bbox_voxels: usize,
    /// Core extent of one fallback block.
    pub block: [usize; 3],
    /// What happens where fallback basins meet.
    pub separation: Separation,
}

impl Default for ScalableWatershedConfig {
    fn default() -> Self {
        Self {
            exact_component_max_bbox_voxels: 16 * 1024 * 1024,
            block: [128, 128, 128],
            separation: Separation::Adjacent,
        }
    }
}

/// Work performed by [`scalable_seeded_watershed`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalableWatershedStats {
    pub components: usize,
    pub exact_components: usize,
    pub minimax_components: usize,
    pub seedless_components: usize,
    /// Times a fallback block was taken from the dirty workset.
    pub block_activations: usize,
    /// Successful improvements of a voxel's path key.
    pub relaxations: usize,
}

/// Work performed by one blockwise minimax flood.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimaxWatershedStats {
    /// Parallel dirty-set waves, including the initial seed wave.
    pub waves: usize,
    pub block_activations: usize,
    pub relaxations: usize,
}

/// The deterministic path order used by the scalable fallback.
///
/// A classical watershed minimises the highest cost crossed by a path. The
/// fallback first settles that scalar barrier field, then uses this key to
/// resolve equal-barrier paths by length and label. Separating those passes is
/// essential: a later ridge can make two
/// previously unequal barriers equal and reverse a secondary tie. Once the
/// barrier field is fixed, every field below travels monotonically with a
/// candidate, so block activation and worker order cannot move the answer.
#[derive(Debug, Clone, Copy)]
struct PathKey {
    steps: u64,
    label: u32,
}

impl PartialEq for PathKey {
    fn eq(&self, other: &Self) -> bool {
        self.steps == other.steps && self.label == other.label
    }
}

impl Eq for PathKey {}

impl PartialOrd for PathKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.steps
            .cmp(&other.steps)
            .then_with(|| self.label.cmp(&other.label))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct QueueEntry {
    key: PathKey,
    index: usize,
}

struct BlockSolve {
    updates: BTreeMap<usize, PathKey>,
    outgoing: Vec<QueueEntry>,
    relaxations: usize,
}

#[derive(Debug, Clone, Copy)]
struct BarrierEntry {
    barrier: f64,
    index: usize,
}

impl PartialEq for BarrierEntry {
    fn eq(&self, other: &Self) -> bool {
        self.barrier.to_bits() == other.barrier.to_bits() && self.index == other.index
    }
}

impl Eq for BarrierEntry {}

impl PartialOrd for BarrierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BarrierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.barrier
            .total_cmp(&other.barrier)
            .then_with(|| self.index.cmp(&other.index))
    }
}

struct BarrierBlockSolve {
    updates: BTreeMap<usize, f64>,
    outgoing: Vec<BarrierEntry>,
    relaxations: usize,
}

#[derive(Debug, Clone, Copy)]
struct ComponentBox {
    label: u32,
    lo: [usize; 3],
    hi: [usize; 3],
    seeds: usize,
}

impl ComponentBox {
    fn shape(self) -> [usize; 3] {
        [
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        ]
    }

    fn bbox_voxels(self) -> usize {
        let shape = self.shape();
        shape[0].saturating_mul(shape[1]).saturating_mul(shape[2])
    }
}

#[derive(Debug)]
struct ComponentResult {
    labels: Option<([usize; 3], Array3<u32>)>,
    exact: bool,
    seedless: bool,
    block_activations: usize,
    relaxations: usize,
}

impl Separation {
    fn carves_a_line(self) -> bool {
        matches!(self, Self::Line)
    }
}

/// Flood `cost` from `seeds`, within `mask`, into `out`.
///
/// * `cost` — lower floods first. A caller wanting basins around bright
///   structure passes the negation.
/// * `seeds` — 0 is "not a seed"; every other value is a label, and a label may
///   occupy more than one voxel.
/// * `mask` — `None` means the whole array. A seed outside the mask is dropped,
///   which is what `skimage`'s `markers * mask` does at the Python layer.
/// * `out` — one label per voxel, 0 where nothing reached.
///
/// The mask is consumed by value inside: [`Separation::Line`] carves the line by
/// clearing mask voxels as it goes, so the caller's copy is untouched.
pub fn seeded_watershed_into(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    flood(cost, seeds, mask, separation, out).map(|_| ())
}

/// [`seeded_watershed_into`], returning the largest number of items the queue
/// ever held.
///
/// The queue is the whole of this op's memory cost beyond the dense arrays, and
/// it can only be known by running — so it is returned rather than estimated,
/// and the module's arithmetic is checked against it.
pub fn seeded_watershed_into_reporting_peak(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    out: ArrayViewMut3<'_, u32>,
) -> Result<usize> {
    flood(cost, seeds, mask, separation, out)
}

fn flood(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<usize> {
    shapes_agree(cost.shape(), seeds.shape(), "seeded_watershed_into (seeds)")?;
    if let Some(mask) = mask.as_ref() {
        shapes_agree(cost.shape(), mask.shape(), "seeded_watershed_into (mask)")?;
    }
    shapes_agree(cost.shape(), out.shape(), "seeded_watershed_into (out)")?;

    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let line = separation.carves_a_line();

    // **Borrowed where it can be.** The kernel wants one raveled C-order slice
    // per array, which is exactly what a contiguous standard-layout view already
    // is; copying it would be a second dense array on a phase whose whole
    // difficulty is that it holds the volume. A view that is *not* contiguous —
    // a transposed or strided one — has no such slice and is materialised, which
    // is a conversion rather than an avoidable allocation.
    let copied_image;
    let image: &[f64] = match cost.as_slice() {
        Some(slice) => slice,
        None => {
            copied_image = cost.iter().copied().collect::<Vec<f64>>();
            &copied_image
        }
    };

    // `markers * mask`: a seed outside the mask is not a seed. Written straight
    // into the output, which is the array the kernel floods in place.
    match mask.as_ref() {
        Some(mask) => ndarray::Zip::from(&mut out)
            .and(seeds)
            .and(mask)
            .for_each(|slot, &label, &inside| *slot = if inside { label } else { 0 }),
        None => ndarray::Zip::from(&mut out)
            .and(seeds)
            .for_each(|slot, &label| *slot = label),
    }

    // The mask is the one array that must be owned: the kernel clears voxels in
    // it to carve the separating line, and it is handed in as a shared view of a
    // stored image that other readers must still find intact.
    let mut flags: Vec<bool> = match mask.as_ref() {
        Some(mask) => mask.iter().copied().collect(),
        None => vec![true; image.len()],
    };

    match out.as_slice_mut() {
        Some(labels) => watershed_raveled_reporting_peak(image, &shape, &mut flags, labels, line),
        None => {
            let mut labels: Vec<u32> = out.iter().copied().collect();
            let peak =
                watershed_raveled_reporting_peak(image, &shape, &mut flags, &mut labels, line)?;
            for (slot, label) in out.iter_mut().zip(labels) {
                *slot = label;
            }
            Ok(peak)
        }
    }
}

/// [`seeded_watershed_into`], allocating its own output.
pub fn seeded_watershed(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
) -> Result<Array3<u32>> {
    let mut out = Array3::zeros(cost.raw_dim());
    seeded_watershed_into(cost, seeds, mask, separation, out.view_mut())?;
    Ok(out)
}

/// A deterministic watershed for a region too large for one global queue.
///
/// Each block owns a persistent priority queue. Processing a block drains that
/// queue to a local fixed point; an improved candidate crossing a block face
/// enters the neighbour's queue and marks that neighbour dirty. The run ends
/// when no dirty block remains. A block is never recomputed merely because a
/// global round advanced.
///
/// This is a minimax watershed, not a parallel spelling of scikit-image's
/// `(value, age)` heap. Equal barriers prefer the shortest path and then the
/// lower label. Keep [`seeded_watershed`] when bit parity with scikit-image is
/// the contract.
pub fn blockwise_minimax_watershed(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    block: [usize; 3],
    separation: Separation,
) -> Result<(Array3<u32>, MinimaxWatershedStats)> {
    shapes_agree(
        cost.shape(),
        seeds.shape(),
        "blockwise_minimax_watershed (seeds)",
    )?;
    if let Some(mask) = mask.as_ref() {
        shapes_agree(
            cost.shape(),
            mask.shape(),
            "blockwise_minimax_watershed (mask)",
        )?;
    }
    if block.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "blockwise_minimax_watershed needs a positive block extent on every axis, got \
             {block:?}"
        )));
    }

    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let len = shape[0] * shape[1] * shape[2];
    let inside = |at: [usize; 3]| mask.as_ref().is_none_or(|mask| mask[at]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if inside([i, j, k]) && !cost[[i, j, k]].is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "blockwise_minimax_watershed needs finite costs inside its mask; \
                         [{i}, {j}, {k}] holds {}",
                        cost[[i, j, k]]
                    )));
                }
            }
        }
    }

    let block = [
        block[0].min(shape[0]),
        block[1].min(shape[1]),
        block[2].min(shape[2]),
    ];
    let block_counts = [
        shape[0].div_ceil(block[0]),
        shape[1].div_ceil(block[1]),
        shape[2].div_ceil(block[2]),
    ];
    let n_blocks = block_counts[0] * block_counts[1] * block_counts[2];
    let block_of = |at: [usize; 3]| {
        let index = [at[0] / block[0], at[1] / block[1], at[2] / block[2]];
        (index[0] * block_counts[1] + index[1]) * block_counts[2] + index[2]
    };

    // First solve only the minimax barrier. A secondary tie key cannot safely
    // travel in this pass: two unequal barriers may become equal when both paths
    // cross a later ridge, at which point their secondary order can reverse.
    // With the final barrier field fixed, hop distance and label form an
    // ordinary monotone shortest-path problem, solved in the second pass below.
    let (barriers, mut stats) =
        minimax_barriers(cost, seeds, mask, shape, block, block_counts, n_blocks);

    // Label 0 is already reserved for "unassigned", so a plain PathKey avoids
    // the discriminant and padding of Option<PathKey>. Together with the f64
    // infinity sentinel in the barrier pass, fallback state is 24 bytes per
    // voxel rather than about 56 bytes on 64-bit targets.
    let mut state = vec![
        PathKey {
            steps: u64::MAX,
            label: 0
        };
        len
    ];
    let mut queues: Vec<BinaryHeap<Reverse<QueueEntry>>> =
        (0..n_blocks).map(|_| BinaryHeap::new()).collect();
    let mut dirty = BTreeSet::new();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let label = seeds[at];
                if label == 0 || !inside(at) {
                    continue;
                }
                let index = ravel(at, shape);
                let key = PathKey { steps: 0, label };
                if state[index].label == 0 || key < state[index] {
                    state[index] = key;
                    let owner = block_of(at);
                    queues[owner].push(Reverse(QueueEntry { key, index }));
                    dirty.insert(owner);
                }
            }
        }
    }

    while !dirty.is_empty() {
        stats.waves += 1;
        let active = std::mem::take(&mut dirty).into_iter().collect::<Vec<_>>();
        stats.block_activations += active.len();
        let jobs = active
            .iter()
            .map(|&owner| (owner, std::mem::take(&mut queues[owner])))
            .collect::<Vec<_>>();
        let solved = jobs
            .into_par_iter()
            .map(|(owner, queue)| {
                solve_dirty_block(
                    owner,
                    queue,
                    &state,
                    &barriers,
                    cost,
                    mask,
                    shape,
                    block,
                    block_counts,
                )
            })
            .collect::<Vec<_>>();

        // Local cores are disjoint, so publish them first in block order. The
        // outgoing boundary candidates are then folded in that same order;
        // PathKey's total order makes the answer independent of the order, and
        // fixing it also makes the trace and counters reproducible.
        for result in &solved {
            stats.relaxations += result.relaxations;
            for (&index, &key) in &result.updates {
                if state[index].label == 0 || key < state[index] {
                    state[index] = key;
                }
            }
        }
        for result in solved {
            for entry in result.outgoing {
                if state[entry.index].label != 0 && entry.key >= state[entry.index] {
                    continue;
                }
                state[entry.index] = entry.key;
                stats.relaxations += 1;
                let neighbour_owner = block_of(unravel(entry.index, shape));
                queues[neighbour_owner].push(Reverse(entry));
                dirty.insert(neighbour_owner);
            }
        }
    }

    let mut out = Array3::<u32>::zeros((shape[0], shape[1], shape[2]));
    for (index, key) in state.iter().enumerate() {
        if key.label != 0 {
            out[unravel(index, shape)] = key.label;
        }
    }
    if separation == Separation::Line {
        carve_deterministic_line(&mut out, seeds, mask, &state, shape);
    }
    Ok((out, stats))
}

#[allow(clippy::too_many_arguments)]
fn minimax_barriers(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    shape: [usize; 3],
    block: [usize; 3],
    block_counts: [usize; 3],
    n_blocks: usize,
) -> (Vec<f64>, MinimaxWatershedStats) {
    let len = shape[0] * shape[1] * shape[2];
    let inside = |at: [usize; 3]| mask.as_ref().is_none_or(|mask| mask[at]);
    let block_of = |at: [usize; 3]| {
        let index = [at[0] / block[0], at[1] / block[1], at[2] / block[2]];
        (index[0] * block_counts[1] + index[1]) * block_counts[2] + index[2]
    };
    let mut state = vec![f64::INFINITY; len];
    let mut queues: Vec<BinaryHeap<Reverse<BarrierEntry>>> =
        (0..n_blocks).map(|_| BinaryHeap::new()).collect();
    let mut dirty = BTreeSet::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                if seeds[at] == 0 || !inside(at) {
                    continue;
                }
                let index = ravel(at, shape);
                let barrier = normalise_zero(cost[at]);
                if barrier.total_cmp(&state[index]).is_lt() {
                    state[index] = barrier;
                    let owner = block_of(at);
                    queues[owner].push(Reverse(BarrierEntry { barrier, index }));
                    dirty.insert(owner);
                }
            }
        }
    }

    let mut stats = MinimaxWatershedStats::default();
    while !dirty.is_empty() {
        stats.waves += 1;
        let active = std::mem::take(&mut dirty).into_iter().collect::<Vec<_>>();
        stats.block_activations += active.len();
        let jobs = active
            .iter()
            .map(|&owner| (owner, std::mem::take(&mut queues[owner])))
            .collect::<Vec<_>>();
        let solved = jobs
            .into_par_iter()
            .map(|(owner, queue)| {
                solve_barrier_block(owner, queue, &state, cost, mask, shape, block, block_counts)
            })
            .collect::<Vec<_>>();
        for result in &solved {
            stats.relaxations += result.relaxations;
            for (&index, &barrier) in &result.updates {
                if barrier.total_cmp(&state[index]).is_lt() {
                    state[index] = barrier;
                }
            }
        }
        for result in solved {
            for entry in result.outgoing {
                if entry.barrier.total_cmp(&state[entry.index]).is_ge() {
                    continue;
                }
                state[entry.index] = entry.barrier;
                stats.relaxations += 1;
                let neighbour_owner = block_of(unravel(entry.index, shape));
                queues[neighbour_owner].push(Reverse(entry));
                dirty.insert(neighbour_owner);
            }
        }
    }
    (state, stats)
}

#[allow(clippy::too_many_arguments)]
fn solve_barrier_block(
    owner: usize,
    mut queue: BinaryHeap<Reverse<BarrierEntry>>,
    state: &[f64],
    cost: ArrayView3<'_, f64>,
    mask: Option<ArrayView3<'_, bool>>,
    shape: [usize; 3],
    block: [usize; 3],
    block_counts: [usize; 3],
) -> BarrierBlockSolve {
    let block_index = [
        owner / (block_counts[1] * block_counts[2]),
        (owner / block_counts[2]) % block_counts[1],
        owner % block_counts[2],
    ];
    let lo = [
        block_index[0] * block[0],
        block_index[1] * block[1],
        block_index[2] * block[2],
    ];
    let hi = [
        (lo[0] + block[0]).min(shape[0]),
        (lo[1] + block[1]).min(shape[1]),
        (lo[2] + block[2]).min(shape[2]),
    ];
    let belongs_here =
        |at: [usize; 3]| (0..3).all(|axis| lo[axis] <= at[axis] && at[axis] < hi[axis]);
    let inside = |at: [usize; 3]| mask.as_ref().is_none_or(|mask| mask[at]);
    let mut updates = BTreeMap::new();
    let mut outgoing = Vec::new();
    let mut relaxations = 0usize;

    while let Some(Reverse(entry)) = queue.pop() {
        let held = updates
            .get(&entry.index)
            .copied()
            .unwrap_or(state[entry.index]);
        if held.to_bits() != entry.barrier.to_bits() {
            continue;
        }
        let at = unravel(entry.index, shape);
        for (axis, step) in FACE_NEIGHBOURS {
            let Some(neighbour) = stepped(at, axis, step, shape) else {
                continue;
            };
            if !inside(neighbour) {
                continue;
            }
            let neighbour_index = ravel(neighbour, shape);
            let barrier = normalise_zero(entry.barrier.max(cost[neighbour]));
            if belongs_here(neighbour) {
                let held = updates
                    .get(&neighbour_index)
                    .copied()
                    .unwrap_or(state[neighbour_index]);
                if barrier.total_cmp(&held).is_ge() {
                    continue;
                }
                updates.insert(neighbour_index, barrier);
                relaxations += 1;
                queue.push(Reverse(BarrierEntry {
                    barrier,
                    index: neighbour_index,
                }));
            } else if barrier.total_cmp(&state[neighbour_index]).is_lt() {
                outgoing.push(BarrierEntry {
                    barrier,
                    index: neighbour_index,
                });
            }
        }
    }
    BarrierBlockSolve {
        updates,
        outgoing,
        relaxations,
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_dirty_block(
    owner: usize,
    mut queue: BinaryHeap<Reverse<QueueEntry>>,
    state: &[PathKey],
    barriers: &[f64],
    cost: ArrayView3<'_, f64>,
    mask: Option<ArrayView3<'_, bool>>,
    shape: [usize; 3],
    block: [usize; 3],
    block_counts: [usize; 3],
) -> BlockSolve {
    let block_index = [
        owner / (block_counts[1] * block_counts[2]),
        (owner / block_counts[2]) % block_counts[1],
        owner % block_counts[2],
    ];
    let lo = [
        block_index[0] * block[0],
        block_index[1] * block[1],
        block_index[2] * block[2],
    ];
    let hi = [
        (lo[0] + block[0]).min(shape[0]),
        (lo[1] + block[1]).min(shape[1]),
        (lo[2] + block[2]).min(shape[2]),
    ];
    let belongs_here =
        |at: [usize; 3]| (0..3).all(|axis| lo[axis] <= at[axis] && at[axis] < hi[axis]);
    let inside = |at: [usize; 3]| mask.as_ref().is_none_or(|mask| mask[at]);
    let mut updates = BTreeMap::new();
    let mut outgoing = Vec::new();
    let mut relaxations = 0usize;

    while let Some(Reverse(entry)) = queue.pop() {
        let held = updates
            .get(&entry.index)
            .copied()
            .unwrap_or(state[entry.index]);
        if held != entry.key {
            continue;
        }
        let at = unravel(entry.index, shape);
        for (axis, step) in FACE_NEIGHBOURS {
            let Some(neighbour) = stepped(at, axis, step, shape) else {
                continue;
            };
            if !inside(neighbour) {
                continue;
            }
            let neighbour_index = ravel(neighbour, shape);
            let neighbour_barrier = barriers[neighbour_index];
            if !neighbour_barrier.is_finite() {
                continue;
            }
            let crossed = normalise_zero(barriers[entry.index].max(cost[neighbour]));
            if crossed.to_bits() != neighbour_barrier.to_bits() {
                continue;
            }
            let candidate = PathKey {
                steps: entry.key.steps.saturating_add(1),
                label: entry.key.label,
            };
            if belongs_here(neighbour) {
                let held = updates
                    .get(&neighbour_index)
                    .copied()
                    .unwrap_or(state[neighbour_index]);
                if held.label != 0 && candidate >= held {
                    continue;
                }
                updates.insert(neighbour_index, candidate);
                relaxations += 1;
                queue.push(Reverse(QueueEntry {
                    key: candidate,
                    index: neighbour_index,
                }));
            } else if state[neighbour_index].label == 0 || candidate < state[neighbour_index] {
                outgoing.push(QueueEntry {
                    key: candidate,
                    index: neighbour_index,
                });
            }
        }
    }
    BlockSolve {
        updates,
        outgoing,
        relaxations,
    }
}

/// Split a mask into independent components and choose a bounded implementation
/// for each one.
///
/// Components whose dense bounding box fits the configured limit use the exact
/// scikit-image-compatible flood. Oversized components use
/// [`blockwise_minimax_watershed`]. Independent components are evaluated in
/// parallel and assembled in component-label order, so worker count cannot
/// change the result.
///
/// "Exact" here means the reference algorithm on that masked component. It is
/// not a promise of bit identity with one reference flood over all components:
/// scikit-image's equal-valued seeds are ordered by the internal layout of one
/// global heap, and removing the other components' seeds can change that layout.
/// Use [`seeded_watershed`] when whole-volume scikit-image parity is required.
pub fn scalable_seeded_watershed(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    config: ScalableWatershedConfig,
) -> Result<(Array3<u32>, ScalableWatershedStats)> {
    shapes_agree(cost.shape(), seeds.shape(), "scalable watershed (seeds)")?;
    if let Some(mask) = mask.as_ref() {
        shapes_agree(cost.shape(), mask.shape(), "scalable watershed (mask)")?;
    }
    if config.block.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "scalable watershed needs a positive fallback block extent on every axis, got {:?}",
            config.block
        )));
    }

    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let mut component_labels = Array3::<u32>::zeros((shape[0], shape[1], shape[2]));
    let count = label_members_into(
        shape,
        |at| mask.as_ref().is_none_or(|mask| mask[at]),
        component_labels.view_mut(),
    )?;
    let mut boxes = (1..=count)
        .map(|label| ComponentBox {
            label,
            lo: [usize::MAX; 3],
            hi: [0; 3],
            seeds: 0,
        })
        .collect::<Vec<_>>();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let label = component_labels[[i, j, k]];
                if label == 0 {
                    continue;
                }
                let info = &mut boxes[(label - 1) as usize];
                for (axis, value) in [i, j, k].into_iter().enumerate() {
                    info.lo[axis] = info.lo[axis].min(value);
                    info.hi[axis] = info.hi[axis].max(value + 1);
                }
                if seeds[[i, j, k]] != 0 {
                    info.seeds += 1;
                }
            }
        }
    }

    let out = Mutex::new(Array3::<u32>::zeros((shape[0], shape[1], shape[2])));
    let results: Vec<Result<ComponentResult>> = boxes
        .par_iter()
        .map(|component| {
            let mut result =
                solve_component(cost, seeds, component_labels.view(), *component, config)?;
            if let Some((lo, labels)) = result.labels.take() {
                // Components are disjoint, but their bounding boxes may overlap.
                // Publish only labelled voxels under one short lock, then release
                // the local dense box before another component has to finish.
                let mut target = out.lock().unwrap_or_else(|poison| poison.into_inner());
                for ((i, j, k), &label) in labels.indexed_iter() {
                    if label != 0 {
                        target[[lo[0] + i, lo[1] + j, lo[2] + k]] = label;
                    }
                }
            }
            Ok(result)
        })
        .collect();

    let mut stats = ScalableWatershedStats {
        components: boxes.len(),
        ..Default::default()
    };
    for result in results {
        let result = result?;
        stats.exact_components += usize::from(result.exact);
        stats.minimax_components += usize::from(!result.exact && !result.seedless);
        stats.seedless_components += usize::from(result.seedless);
        stats.block_activations += result.block_activations;
        stats.relaxations += result.relaxations;
    }
    let out = out
        .into_inner()
        .unwrap_or_else(|poison| poison.into_inner());
    Ok((out, stats))
}

fn solve_component(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    components: ArrayView3<'_, u32>,
    component: ComponentBox,
    config: ScalableWatershedConfig,
) -> Result<ComponentResult> {
    if component.seeds == 0 {
        return Ok(ComponentResult {
            labels: None,
            exact: false,
            seedless: true,
            block_activations: 0,
            relaxations: 0,
        });
    }
    let [z0, y0, x0] = component.lo;
    let [z1, y1, x1] = component.hi;
    let local_cost = cost.slice(s![z0..z1, y0..y1, x0..x1]);
    let local_seeds = seeds.slice(s![z0..z1, y0..y1, x0..x1]);
    let local_components = components.slice(s![z0..z1, y0..y1, x0..x1]);
    let local_mask = local_components.mapv(|label| label == component.label);
    let exact = component.bbox_voxels() <= config.exact_component_max_bbox_voxels;
    let (labels, minimax) = if exact {
        (
            seeded_watershed(
                local_cost,
                local_seeds,
                Some(local_mask.view()),
                config.separation,
            )?,
            MinimaxWatershedStats::default(),
        )
    } else {
        blockwise_minimax_watershed(
            local_cost,
            local_seeds,
            Some(local_mask.view()),
            config.block,
            config.separation,
        )?
    };

    Ok(ComponentResult {
        labels: Some((component.lo, labels)),
        exact,
        seedless: false,
        block_activations: minimax.block_activations,
        relaxations: minimax.relaxations,
    })
}

fn carve_deterministic_line(
    labels: &mut Array3<u32>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    state: &[PathKey],
    shape: [usize; 3],
) {
    let mut clear = vec![false; state.len()];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let index = ravel(at, shape);
                let label = labels[at];
                if label == 0 || mask.as_ref().is_some_and(|mask| !mask[at]) {
                    continue;
                }
                for (axis, step) in [(0, 1), (1, 1), (2, 1)] {
                    let Some(other) = stepped(at, axis, step, shape) else {
                        continue;
                    };
                    let other_label = labels[other];
                    if other_label == 0 || other_label == label {
                        continue;
                    }
                    let other_index = ravel(other, shape);
                    let loser = match (seeds[at] != 0, seeds[other] != 0) {
                        (true, true) => continue,
                        (true, false) => other_index,
                        (false, true) => index,
                        (false, false) => {
                            if (state[index], index) > (state[other_index], other_index) {
                                index
                            } else {
                                other_index
                            }
                        }
                    };
                    clear[loser] = true;
                }
            }
        }
    }
    for (index, clear) in clear.into_iter().enumerate() {
        if clear {
            labels[unravel(index, shape)] = 0;
        }
    }
}

fn normalise_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn ravel(at: [usize; 3], shape: [usize; 3]) -> usize {
    (at[0] * shape[1] + at[1]) * shape[2] + at[2]
}

fn unravel(index: usize, shape: [usize; 3]) -> [usize; 3] {
    [
        index / (shape[1] * shape[2]),
        (index / shape[2]) % shape[1],
        index % shape[2],
    ]
}

fn stepped(mut at: [usize; 3], axis: usize, step: isize, shape: [usize; 3]) -> Option<[usize; 3]> {
    let value = at[axis].checked_add_signed(step)?;
    if value >= shape[axis] {
        return None;
    }
    at[axis] = value;
    Some(at)
}

// ---------------------------------------------------------------- the op --

/// [`seeded_watershed_into`] as a phase: a cost volume in, one label per voxel
/// out, seeds and mask read from stored images.
///
/// **This op is a planning barrier** and says so through [`Reach::all`]. See the
/// module documentation for the argument and for what it costs.
pub struct SeededWatershedOp {
    name: &'static str,
    seeds: LabelSeedSource,
    mask: Option<MaskSource>,
    separation: Separation,
    cost: f64,
}

impl SeededWatershedOp {
    /// `seeds` is the image holding the labels to flood from.
    pub fn new(
        name: &'static str,
        seeds: impl Into<crate::assemble::ImageId>,
        separation: Separation,
    ) -> Self {
        Self {
            name,
            seeds: LabelSeedSource::new(seeds),
            mask: None,
            separation,
            cost: match separation {
                Separation::Line => WATERSHED_LINE_COST,
                Separation::Adjacent => WATERSHED_COST,
            },
        }
    }

    /// Flood only within `mask`. Without one the whole block is floodable, and
    /// every voxel of it takes a label.
    pub fn within(mut self, mask: impl Into<crate::assemble::ImageId>) -> Self {
        self.mask = Some(MaskSource::new(mask));
        self
    }

    pub fn seed_image(&self) -> usize {
        self.seeds.image()
    }

    pub fn mask_image(&self) -> Option<usize> {
        self.mask.map(MaskSource::image)
    }

    pub fn separation(&self) -> Separation {
        self.separation
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for SeededWatershedOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The whole axis. Stated as an integer here because the trait requires a
    /// symmetric bound, and as the variant that *means* it in
    /// [`Self::reach_spec`], which is what the planner reads.
    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::all()
    }

    /// Both operands over the whole volume, which is the same statement the op
    /// makes about its own input — a flood consults every voxel of all three or
    /// it is a different flood.
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let mut declared = vec![self.seeds.source_input(Reach::all())];
        if let Some(mask) = self.mask {
            declared.push(mask.source_input(Reach::all()));
        }
        declared
    }

    /// `f64` and only `f64`.
    ///
    /// The cost is compared and never combined, so any ordered type would give
    /// the same partition — but the queue is stated in `f64` and widening at the
    /// shell would be a second place the element type is decided. A caller
    /// holding something else maps it, which is the same voxelwise pass that
    /// negates it.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    /// Refuses, and the refusal is the point: a flood with no seeds has no
    /// answer, it has an empty one. See [`BlockOp::apply_with`].
    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(self.seeds.input_only_error(self.name))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let seeds = self.seeds.u32_view(self.name, sources)?;

        let mask = match self.mask {
            Some(mask) => Some(mask.bool_view(self.name, sources)?),
            None => None,
        };

        seeded_watershed_into(
            input.view::<f64>()?,
            seeds,
            mask,
            self.separation,
            out.view_mut::<u32>()?,
        )
    }

    /// **Nothing.** A constant cost block is not a constant answer: what a voxel
    /// gets is decided by the seeds and the mask, neither of which this
    /// declaration can see, and a flat cost is precisely the case where the
    /// partition is decided by tie-breaking alone.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// [`scalable_seeded_watershed`] as a planner operation.
///
/// This first planner surface still declares [`Reach::all`]: discovering the
/// globally connected mask components currently happens inside the kernel.
/// Unlike [`SeededWatershedOp`], however, it never requires one priority queue
/// spanning every masked voxel. Independent components run concurrently and an
/// oversized component uses persistent block queues. Moving component discovery
/// into preceding fragment phases can later remove this barrier without changing
/// the operation's partition semantics.
/// It likewise does not claim whole-volume scikit-image parity; use
/// [`SeededWatershedOp`] for that contract.
pub struct ScalableWatershedOp {
    name: &'static str,
    seeds: LabelSeedSource,
    mask: Option<MaskSource>,
    config: ScalableWatershedConfig,
    cost: f64,
}

impl ScalableWatershedOp {
    pub fn new(
        name: &'static str,
        seeds: impl Into<crate::assemble::ImageId>,
        config: ScalableWatershedConfig,
    ) -> Self {
        Self {
            name,
            seeds: LabelSeedSource::new(seeds),
            mask: None,
            config,
            cost: match config.separation {
                Separation::Line => WATERSHED_LINE_COST,
                Separation::Adjacent => WATERSHED_COST,
            },
        }
    }

    pub fn within(mut self, mask: impl Into<crate::assemble::ImageId>) -> Self {
        self.mask = Some(MaskSource::new(mask));
        self
    }

    pub fn config(&self) -> ScalableWatershedConfig {
        self.config
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for ScalableWatershedOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::all()
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let mut declared = vec![self.seeds.source_input(Reach::all())];
        if let Some(mask) = self.mask {
            declared.push(mask.source_input(Reach::all()));
        }
        declared
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(self.seeds.input_only_error(self.name))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let seeds = self.seeds.u32_view(self.name, sources)?;
        let mask = match self.mask {
            Some(mask) => Some(mask.bool_view(self.name, sources)?),
            None => None,
        };
        let (labels, _) =
            scalable_seeded_watershed(input.view::<f64>()?, seeds, mask, self.config)?;
        out.view_mut::<u32>()?.assign(&labels);
        Ok(())
    }

    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured, for [`Separation::Adjacent`]; see `super::COST_MEASUREMENT`.
///
/// **Not taken through `ops::cost::measure`**, which builds one shared `f64`
/// ramp and reads the result as `f64`: this op writes `u32` and needs two
/// operands, so there is no signature to hand it through. Taken instead by
/// [`cost_report`] below, over the same 96 x 64 x 64 volume and against the same
/// `MAP_COST` unit: five runs on a busy machine gave 639, 638, 699, 700 and 1010,
/// against a voxelwise map measured at 0.63-0.75 ns/voxel in the same runs. The
/// figure stored is the middle of that, and the spread is the honest precision —
/// the crate's own note on this table applies with force here, since `statistics`
/// recalibrates from real runs anyway.
///
/// **This is two to three orders of magnitude above everything else in this
/// module's table, and the figure is real rather than a units mistake.** Every
/// voxel costs a push and a pop of a queue holding hundreds of thousands of
/// items, and a sift through that is a chain of cache misses. Two things follow
/// that a planner should have: the figure is per *masked* voxel, so a mask
/// keeping a tenth of the volume costs about a tenth of it — the constant is the
/// all-masked case, which is the one nobody should be surprised by — and no
/// block size makes it cheaper, because the op is a barrier.
pub const WATERSHED_COST: f64 = 680.0;

/// [`WATERSHED_COST`] for [`Separation::Line`], which is **4.7x** dearer and
/// measured rather than derived (2758, 2993, 3030, 3600, 3760 over the same five
/// runs).
///
/// The reason is structural, not incidental. Without a line a voxel is labelled
/// the moment it is pushed and is therefore queued exactly once; with one it is
/// not settled until it *pops*, so it is queued again by every neighbour that
/// pops before it does — up to six times. The extra queue traffic is the whole
/// difference, and it is why `Separation` is a cost input and not only a
/// behaviour switch.
pub const WATERSHED_LINE_COST: f64 = 3200.0;

/// Retake [`WATERSHED_COST`] and [`WATERSHED_LINE_COST`] on this machine. See
/// `super::COST_MEASUREMENT`.
///
/// **Measures its own denominator.** The stored table is in units of one
/// voxelwise map, and that unit is 0.71 ns/voxel on one machine and something
/// else on the next — so this prices a threshold map in the same run, with the
/// same fixture size and the same best-of rule, and divides. What comes out is a
/// ratio, which is what the planner uses and what survives a change of machine.
pub fn cost_report() -> String {
    use std::time::Instant;

    let shape = [96usize, 64, 64];
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let cost = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (((i * 7 + j * 13 + k * 29) % 251) as f64) / 251.0
    });
    let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
    let mut label = 0u32;
    for i in (4..shape[0]).step_by(16) {
        for j in (4..shape[1]).step_by(16) {
            for k in (4..shape[2]).step_by(16) {
                label += 1;
                seeds[[i, j, k]] = label;
            }
        }
    }

    let mut unit = f64::INFINITY;
    for _ in 0..40 {
        let mut out = Array3::<f64>::zeros(cost.raw_dim());
        let started = Instant::now();
        ndarray::Zip::from(&mut out)
            .and(&cost)
            .for_each(|slot, &value| *slot = if value > 0.5 { 1.0 } else { 0.0 });
        unit = unit.min(started.elapsed().as_secs_f64() / voxels * 1e9);
        std::hint::black_box(&out);
    }

    let mut report = format!("one voxelwise map  {unit:.2} ns/voxel  = MAP_COST = 1.0\n");
    for (what, separation) in [
        ("line", Separation::Line),
        ("adjacent", Separation::Adjacent),
    ] {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut out = Array3::<u32>::zeros(cost.raw_dim());
            let started = Instant::now();
            seeded_watershed_into(cost.view(), seeds.view(), None, separation, out.view_mut())
                .expect("the fixture is well formed");
            best = best.min(started.elapsed().as_secs_f64() / voxels * 1e9);
            std::hint::black_box(&out);
        }
        report.push_str(&format!(
            "seeded watershed, {what:<9} {best:.1} ns/voxel  = {:.1} x MAP_COST\n",
            best / unit
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::super::scikitimage_watershed::reference_case;
    use super::*;

    fn reference_fixture() -> (Array3<f64>, Array3<u32>, Array3<bool>) {
        let shape = (
            reference_case::SHAPE[0],
            reference_case::SHAPE[1],
            reference_case::SHAPE[2],
        );
        let source = Array3::from_shape_vec(shape, reference_case::SOURCE.to_vec())
            .expect("the fixture is 5 x 5 x 3");
        let cost = source.mapv(|value| -value);
        let mask = source.mapv(|value| value > reference_case::THRESHOLD);
        let mut seeds = Array3::<u32>::zeros(source.raw_dim());
        for (label, at) in reference_case::SEEDS {
            seeds[at] = label;
        }
        (cost, seeds, mask)
    }

    fn global_minimax_reference(
        cost: ArrayView3<'_, f64>,
        seeds: ArrayView3<'_, u32>,
        mask: ArrayView3<'_, bool>,
    ) -> Array3<u32> {
        let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
        let len = shape.into_iter().product();
        let mut barriers = vec![f64::INFINITY; len];
        let mut barrier_queue = BinaryHeap::new();
        for (at, &label) in seeds.indexed_iter() {
            if label == 0 || !mask[at] {
                continue;
            }
            let at = [at.0, at.1, at.2];
            let index = ravel(at, shape);
            barriers[index] = cost[at];
            barrier_queue.push(Reverse(BarrierEntry {
                barrier: cost[at],
                index,
            }));
        }
        while let Some(Reverse(entry)) = barrier_queue.pop() {
            if entry.barrier.to_bits() != barriers[entry.index].to_bits() {
                continue;
            }
            let at = unravel(entry.index, shape);
            for (axis, step) in FACE_NEIGHBOURS {
                let Some(next) = stepped(at, axis, step, shape) else {
                    continue;
                };
                if !mask[next] {
                    continue;
                }
                let next_index = ravel(next, shape);
                let candidate = entry.barrier.max(cost[next]);
                if candidate.total_cmp(&barriers[next_index]).is_lt() {
                    barriers[next_index] = candidate;
                    barrier_queue.push(Reverse(BarrierEntry {
                        barrier: candidate,
                        index: next_index,
                    }));
                }
            }
        }

        let unset = PathKey {
            steps: u64::MAX,
            label: 0,
        };
        let mut paths = vec![unset; len];
        let mut path_queue = BinaryHeap::new();
        for (at, &label) in seeds.indexed_iter() {
            if label == 0 || !mask[at] {
                continue;
            }
            let index = ravel([at.0, at.1, at.2], shape);
            let key = PathKey { steps: 0, label };
            if paths[index].label == 0 || key < paths[index] {
                paths[index] = key;
                path_queue.push(Reverse(QueueEntry { key, index }));
            }
        }
        while let Some(Reverse(entry)) = path_queue.pop() {
            if entry.key != paths[entry.index] {
                continue;
            }
            let at = unravel(entry.index, shape);
            for (axis, step) in FACE_NEIGHBOURS {
                let Some(next) = stepped(at, axis, step, shape) else {
                    continue;
                };
                if !mask[next] {
                    continue;
                }
                let next_index = ravel(next, shape);
                if barriers[entry.index].max(cost[next]).to_bits() != barriers[next_index].to_bits()
                {
                    continue;
                }
                let key = PathKey {
                    steps: entry.key.steps + 1,
                    label: entry.key.label,
                };
                if paths[next_index].label == 0 || key < paths[next_index] {
                    paths[next_index] = key;
                    path_queue.push(Reverse(QueueEntry {
                        key,
                        index: next_index,
                    }));
                }
            }
        }
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |at| {
            paths[ravel([at.0, at.1, at.2], shape)].label
        })
    }

    /// The whole point, through the shell rather than the kernel.
    #[test]
    fn the_shell_reproduces_the_reference_partition() {
        let (cost, seeds, mask) = reference_fixture();
        let labels = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            reference_case::EXPECTED.to_vec()
        );
    }

    #[test]
    fn a_seed_outside_the_mask_is_not_a_seed() {
        let cost = Array3::from_shape_fn((1, 1, 5), |(_, _, k)| k as f64);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 1;
        seeds[[0, 0, 4]] = 2;
        let mut mask = Array3::from_elem(cost.raw_dim(), true);
        mask[[0, 0, 4]] = false;

        let labels = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Adjacent,
        )
        .unwrap();
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn without_a_mask_every_voxel_takes_a_label() {
        let cost = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| ((i * 3 + j * 5 + k) % 7) as f64);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 1;
        seeds[[2, 3, 4]] = 2;
        let labels =
            seeded_watershed(cost.view(), seeds.view(), None, Separation::Adjacent).unwrap();
        assert!(labels.iter().all(|&label| label != 0));
    }

    #[test]
    fn minimax_fallback_is_block_size_invariant() {
        let cost = Array3::from_shape_fn((5, 7, 11), |(i, j, k)| {
            ((i * 13 + j * 7 + k * 3) % 9) as f64
        });
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 7;
        seeds[[4, 6, 10]] = 2;
        seeds[[2, 3, 5]] = 11;

        let whole = blockwise_minimax_watershed(
            cost.view(),
            seeds.view(),
            None,
            [5, 7, 11],
            Separation::Adjacent,
        )
        .expect("one block")
        .0;
        for block in [[3, 4, 5], [2, 2, 3], [1, 1, 1]] {
            let divided = blockwise_minimax_watershed(
                cost.view(),
                seeds.view(),
                None,
                block,
                Separation::Adjacent,
            )
            .expect("a divided flood")
            .0;
            assert_eq!(divided, whole, "block {block:?} changed the partition");
        }
    }

    #[test]
    fn blockwise_fallback_matches_a_single_global_minimax_queue() {
        let cost = Array3::from_shape_fn((5, 8, 13), |(i, j, k)| {
            ((i * 19 + j * 11 + k * 5) % 17) as f64
        });
        let mask = Array3::from_shape_fn(cost.raw_dim(), |(i, j, k)| (i + 2 * j + 3 * k) % 11 != 0);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 1]] = 7;
        seeds[[4, 7, 12]] = 2;
        seeds[[2, 3, 6]] = 11;
        let expected = global_minimax_reference(cost.view(), seeds.view(), mask.view());
        let actual = blockwise_minimax_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            [2, 3, 4],
            Separation::Adjacent,
        )
        .expect("a blockwise minimax flood")
        .0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn one_large_component_is_worker_count_invariant() {
        let cost = Array3::from_shape_fn((7, 13, 17), |(i, j, k)| {
            ((i * 17 + j * 5 + k * 11) % 29) as f64
        });
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 9;
        seeds[[6, 12, 16]] = 2;
        seeds[[3, 6, 8]] = 5;
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("a pool")
                .install(|| {
                    blockwise_minimax_watershed(
                        cost.view(),
                        seeds.view(),
                        None,
                        [2, 4, 5],
                        Separation::Adjacent,
                    )
                    .expect("a fallback flood")
                    .0
                })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn scalable_dispatches_by_component_bbox_and_drops_seedless_components() {
        let cost = Array3::from_shape_fn((1, 5, 12), |(_, j, k)| (j + k) as f64);
        let mut mask = Array3::<bool>::from_elem(cost.raw_dim(), false);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());

        // Three disconnected components: a small exact one, a long fallback
        // one, and one with no seed.
        for k in 0..3 {
            mask[[0, 0, k]] = true;
        }
        seeds[[0, 0, 0]] = 1;
        for k in 2..12 {
            mask[[0, 2, k]] = true;
        }
        seeds[[0, 2, 2]] = 2;
        seeds[[0, 2, 11]] = 3;
        for k in 7..10 {
            mask[[0, 4, k]] = true;
        }

        let (labels, stats) = scalable_seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            ScalableWatershedConfig {
                exact_component_max_bbox_voxels: 4,
                block: [1, 1, 3],
                separation: Separation::Adjacent,
            },
        )
        .expect("a scalable flood");
        assert_eq!(stats.components, 3);
        assert_eq!(stats.exact_components, 1);
        assert_eq!(stats.minimax_components, 1);
        assert_eq!(stats.seedless_components, 1);
        assert!(stats.block_activations > 1);
        assert!((0..3).all(|k| labels[[0, 0, k]] == 1));
        assert!((7..10).all(|k| labels[[0, 4, k]] == 0));
        assert!((2..12).all(|k| labels[[0, 2, k]] != 0));
    }

    #[test]
    fn component_parallelism_does_not_move_a_voxel() {
        let cost = Array3::from_shape_fn((3, 17, 19), |(i, j, k)| {
            ((i * 5 + j * 11 + k * 7) % 23) as f64
        });
        let mask = Array3::from_shape_fn(cost.raw_dim(), |(_, j, _)| j % 3 != 2);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        let mut label = 1u32;
        for j in (0..17).step_by(3) {
            seeds[[1, j, 1]] = label;
            label += 1;
            seeds[[1, j + 1, 17]] = label;
            label += 1;
        }
        let config = ScalableWatershedConfig {
            exact_component_max_bbox_voxels: 0,
            block: [2, 4, 5],
            separation: Separation::Adjacent,
        };
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("a pool")
                .install(|| {
                    scalable_seeded_watershed(cost.view(), seeds.view(), Some(mask.view()), config)
                        .expect("a scalable flood")
                        .0
                })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn deterministic_line_keeps_seed_voxels() {
        let cost = Array3::<f64>::zeros((1, 1, 9));
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 1;
        seeds[[0, 0, 8]] = 2;
        let labels = blockwise_minimax_watershed(
            cost.view(),
            seeds.view(),
            None,
            [1, 1, 2],
            Separation::Line,
        )
        .expect("a line flood")
        .0;
        assert_eq!(labels[[0, 0, 0]], 1);
        assert_eq!(labels[[0, 0, 8]], 2);
        assert!(labels.iter().any(|label| *label == 0));
    }

    /// The two separations are the same partition up to the line, which is what
    /// makes `Separation` a parameter rather than two ops.
    #[test]
    fn the_line_only_ever_removes_voxels() {
        let (cost, seeds, mask) = reference_fixture();
        let with_line = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();
        let without = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Adjacent,
        )
        .unwrap();
        let labelled_with = with_line.iter().filter(|&&label| label != 0).count();
        let labelled_without = without.iter().filter(|&&label| label != 0).count();
        assert!(
            labelled_with < labelled_without,
            "the line must clear something, or the parameter says nothing"
        );
        assert_eq!(
            labelled_without,
            mask.iter().filter(|&&inside| inside).count(),
            "without a line every masked voxel is labelled"
        );
    }

    /// The shell borrows a contiguous view and materialises one that is not. A
    /// transposed view is the second case, and it must give the same answer as
    /// the copy the caller could have made itself — otherwise the optimisation
    /// is a second implementation.
    #[test]
    fn a_view_that_has_no_slice_gives_the_same_answer_as_one_that_does() {
        let (cost, seeds, mask) = reference_fixture();
        let expected = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();

        // Permuted so no array involved has a standard-layout slice, then read
        // back through the inverse permutation.
        let flip = |axes: [usize; 3]| [axes[2], axes[0], axes[1]];
        let cost = cost.clone().permuted_axes(flip([0, 1, 2]));
        let seeds = seeds.clone().permuted_axes(flip([0, 1, 2]));
        let mask = mask.clone().permuted_axes(flip([0, 1, 2]));
        assert!(
            cost.as_slice().is_none(),
            "the fixture stopped exercising the copy path"
        );
        let mut out = Array3::<u32>::zeros(cost.raw_dim());
        let strided = out.slice_mut(ndarray::s![.., .., ..;1]);
        seeded_watershed_into(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
            strided,
        )
        .unwrap();
        assert_eq!(out.permuted_axes([1, 2, 0]), expected);
    }

    #[test]
    fn a_shape_mismatch_is_refused_by_name() {
        let cost = Array3::<f64>::zeros((2, 2, 2));
        let seeds = Array3::<u32>::zeros((2, 2, 3));
        let failed = seeded_watershed(cost.view(), seeds.view(), None, Separation::Line)
            .expect_err("mismatched shapes");
        assert!(failed.to_string().contains("seeds"), "{failed}");
    }

    #[test]
    fn the_shell_declares_seed_and_mask_source_dtypes() {
        let op = SeededWatershedOp::new(
            "watershed",
            crate::assemble::ImageId::supplied(0),
            Separation::Line,
        )
        .within(crate::assemble::ImageId::supplied(1));
        assert_eq!(
            op.source_inputs([5, 4, 3]),
            vec![
                SourceInput::new(crate::assemble::ImageId::supplied(0), Reach::all())
                    .holding(Dtype::U32),
                SourceInput::new(crate::assemble::ImageId::supplied(1), Reach::all())
                    .holding(Dtype::Bool),
            ]
        );
    }
}
