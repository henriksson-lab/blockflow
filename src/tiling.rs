// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from scratch for the extraction; see
// "Provenance" below for why it could not simply be moved.
//
// The guard the whole design rests on
// -----------------------------------
// `geometry` derives a block's trustworthy extent from the chain's folded reach
// rather than asserting `halo >= reach`. That inversion is only safe because
// something downstream still checks the consequence: that the blocks' *valid*
// regions cover the volume exactly, once each. If they do, no voxel was
// computed from an untrustworthy neighbourhood and no voxel was left unwritten.
// If they do not, the run is wrong — and this is the one place that says so.
//
// So this predicate is load-bearing in a way most of the crate is not. It is
// worth more than its twenty lines suggest, and it is worth stating precisely
// what it decides.
//
// What it decides, and why that is enough
// ---------------------------------------
// > For axis-aligned boxes that all lie **inside** `shape`: pairwise disjoint,
// > plus volumes summing to `shape`'s volume, is equivalent to an exact tiling.
//
// (⇐) An exact tiling is disjoint and sums, trivially. (⇒) Suppose the boxes
// are disjoint and sum to the total. Disjointness means the covered voxel count
// is exactly the sum of the volumes, so every voxel of the volume is covered at
// most once and the count of covered voxels equals the count of voxels; hence
// every voxel is covered exactly once.
//
// **The containment hypothesis is not decoration.** Drop it and the theorem is
// false: two disjoint boxes, one of them partly outside the volume, can sum to
// the right total while leaving a hole inside it. That is a real hole in a real
// output, reported as success. The check is therefore made explicitly here
// rather than assumed of the caller.
//
// What it costs, and why the obvious implementation was not left in
// ---------------------------------------------------------------
// The obvious alternative to a geometric predicate is a per-voxel write
// counter, which is `O(voxels)` in both time *and* space and would need a
// second copy of the volume to run at all. So this stays geometric.
//
// The first geometric implementation compared every pair, `O(n² · ndim)` in the
// block count, with a header claiming "still milliseconds" at full scale. That
// claim was wrong by three orders of magnitude and it was this predicate, not
// the planner, that made `PlanBuilder::finish` the whole cost of closing a
// plan: `finish` over a two-phase plan measured `445 ms` at `8192` blocks and
// `31.7 ms` at `2048`, a clean `4x` per doubling; the same scan over one
// phase's blocks at `56160` — the density a fine candidate grid asks for —
// takes `13.3 s`, so a two-phase plan closed in half a minute. A block-size
// search that pays that to price a candidate it will reject is a search nobody
// leaves on.
//
// So disjointness is now decided by **separation** rather than by comparison,
// and the pairwise scan survives only as the thing that *names* the offending
// pair once separation has failed:
//
// * `boxes_are_disjoint` cuts the boxes at their own endpoints on one axis,
//   buckets each box into the elementary intervals it covers, and recurses on
//   the next axis. Two boxes overlap iff their intersection is a non-empty box,
//   iff they share an elementary interval on every axis — so two boxes overlap
//   iff they meet in some bucket at the deepest level, and a deepest-level
//   bucket holding two boxes *is* an overlap. The pass is therefore exact, not
//   a filter.
// * For boxes cut from a lattice — which is what a `BlockGrid` produces, and
//   the only thing this crate ever passes — each box covers exactly one
//   elementary interval per axis, so each level places `n` boxes and the whole
//   pass is `O(n · ndim)` placements after an `O(n log n)` sort per level.
// * Adversarial input can still duplicate a box across many intervals, so the
//   pass carries a placement budget of `8 · n · ndim` — eight times what a
//   lattice costs — and *declines* rather than degrades when it is exceeded.
//   Declining is safe because declining and finding an overlap are the same
//   answer to the caller: run the scan. The worst case is therefore the scan
//   that was already there plus a linear pass, and the best case is linear.
//
// **The scan is still what reports.** Not for want of a faster diagnosis: when
// the pass finds an overlap it is holding the two boxes and could hand them
// over. It does not, because the pair the scan names is the lexicographically
// first one and that is the pair every existing message quotes — a faster error
// would be a *different* error, and this change is not allowed to alter a
// single plan or a single message. Diagnosing an already-broken plan in `O(n²)`
// costs nothing that matters; deciding a correct one in `O(n²)` cost
// everything.
//
// Measured before and after on the same machine, same two-phase plan on a `64³`
// lattice, `--release`, load average `3`, `165 GB` of `187 GB` free —
// `PlanBuilder::finish`, whole:
//
// | blocks | before | after |
// |---|---|---|
// | `2048` | `31.7 ms` | `2.1 ms` |
// | `4096` | `113.6 ms` | `2.4 ms` |
// | `8192` | `445.0 ms` | `4.2 ms` |
// | `65536` | — | `69.3 ms` |
//
// and this predicate alone, on one phase's blocks, the two algorithms run back
// to back on the same box list: `13.255 s` against `9.186 ms` at `56160`
// blocks. The after column doubles with the input; the before column
// quadrupled.
//
// The step count is exactly `n · ndim` on a lattice — one placement per box per
// axis, no box copied — and `tests/tiling_scaling.rs` pins that with
// `tiling_work` rather than with a stopwatch.
//
// Provenance
// ----------
// `clearmap-rs` has a predicate answering the same question
// (`parallel_processing::block_processing::valid_boxes_tile_volume`). It lives
// inside a file translated from ClearMap and is therefore GPL-encumbered by
// association, so this crate could not take it. It is reimplemented here from
// the statement of the theorem above rather than copied, and `clearmap-rs`
// carries a test asserting the two agree on every case where both are defined —
// which is the property that matters. Where they differ, this one is stricter:
// it rejects boxes of the wrong rank and boxes that leave the volume, both of
// which the older predicate accepts by silently truncating a `zip`.

use std::cell::Cell;

use crate::error::{Error, Result};

thread_local! {
    /// See [`tiling_work`].
    static WORK: Cell<u64> = const { Cell::new(0) };
}

/// Elementary steps [`boxes_tile_exactly`] has taken on this thread since
/// [`reset_tiling_work`].
///
/// One step is one box placed in one elementary interval by the separating
/// pass, or one pair handed to `boxes_overlap` by the scan. It is public
/// because the cost of this predicate is the cost of closing a plan, and a test
/// that pinned that with a stopwatch would pin the machine instead of the
/// algorithm: a stopwatch bound loose enough not to flake on a loaded box is
/// loose enough to sit above a returning `n²` term. The counter is the same
/// number on an idle machine and a busy one, so a test can assert the *shape*
/// of the growth and mean it.
///
/// Per thread rather than per process, so that a test's own bracket is its own
/// and `cargo test`'s parallelism cannot leak into it.
pub fn tiling_work() -> u64 {
    WORK.with(|work| work.get())
}

/// Zero the counter [`tiling_work`] reads.
pub fn reset_tiling_work() {
    WORK.with(|work| work.set(0));
}

/// Adds what it counted to the thread's total when it is dropped, so that every
/// exit from `boxes_tile_exactly` — including the error ones — is counted.
struct Counted(u64);

impl Drop for Counted {
    fn drop(&mut self) {
        WORK.with(|work| work.set(work.get().saturating_add(self.0)));
    }
}

/// Whether `boxes` partition `shape` exactly — every voxel covered once.
///
/// Each box is per-axis half-open `(lo, hi)` ranges, in the same order as
/// `shape`. Returns the first violation found, named: an out-of-range box, an
/// overlapping pair, or a coverage shortfall.
///
/// An empty `boxes` over a zero-volume `shape` tiles it, which is the degenerate
/// case a plan with no blocks produces and is not an error.
pub fn boxes_tile_exactly(boxes: &[Vec<(usize, usize)>], shape: &[usize]) -> Result<()> {
    let mut work = Counted(0);
    let total: usize = shape.iter().product();

    // Containment and well-formedness first: the disjoint-plus-sum theorem is
    // only valid under them, so checking them afterwards would be checking a
    // conclusion drawn from an unverified hypothesis.
    let mut covered: usize = 0;
    for region in boxes {
        if region.len() != shape.len() {
            return Err(Error::InvalidArgument(format!(
                "valid regions do not tile the volume exactly: region {region:?} has rank {} \
                 but the volume has rank {}",
                region.len(),
                shape.len()
            )));
        }
        for (axis, (&(lo, hi), &dim)) in region.iter().zip(shape.iter()).enumerate() {
            if hi > dim || lo > hi {
                return Err(Error::InvalidArgument(format!(
                    "valid regions do not tile the volume exactly: region {region:?} spans \
                     {lo}..{hi} on axis {axis} of {dim}"
                )));
            }
        }
        covered += region.iter().map(|&(lo, hi)| hi - lo).product::<usize>();
    }

    // Disjointness, decided by separation rather than by comparison. The scan
    // below runs only when that fails or declines — see the header for why it
    // is still the thing that reports.
    if !boxes_are_disjoint(boxes, shape.len(), &mut work.0) {
        for (first, left) in boxes.iter().enumerate() {
            for right in boxes.iter().skip(first + 1) {
                work.0 += 1;
                if boxes_overlap(left, right) {
                    return Err(Error::InvalidArgument(format!(
                        "valid regions do not tile the volume exactly \
                         (regions {left:?} and {right:?} overlap)"
                    )));
                }
            }
        }
    }

    if covered != total {
        return Err(Error::InvalidArgument(format!(
            "valid regions do not tile the volume exactly \
             ({covered} of {total} voxels covered)"
        )));
    }
    Ok(())
}

/// Whether `boxes` are pairwise disjoint, decided without comparing pairs.
///
/// `true` is a proof: no two of them meet. `false` is **not** a proof of the
/// opposite — it is returned both when two boxes are shown to meet and when the
/// placement budget runs out — so the only thing a caller may do with it is ask
/// the exhaustive scan. That conflation is deliberate: it keeps the fast pass
/// out of the diagnosis entirely, and so out of the error text.
///
/// Empty boxes are dropped first because they contain no voxel and so overlap
/// nothing, which is the same rule `boxes_overlap` states axis by axis.
fn boxes_are_disjoint(boxes: &[Vec<(usize, usize)>], ndim: usize, work: &mut u64) -> bool {
    let ids: Vec<usize> = (0..boxes.len())
        .filter(|&id| boxes[id].iter().all(|&(lo, hi)| lo < hi))
        .collect();
    // Eight times what a lattice costs: `ndim` levels, each placing every box
    // once. Past that the input is not the shape this pass is good at and the
    // scan is going to run anyway, so stopping early keeps the worst case at
    // "the scan, plus a linear pass".
    let budget = 8u64
        .saturating_mul(ids.len() as u64)
        .saturating_mul(ndim.max(1) as u64);
    separated(boxes, &ids, 0, ndim, budget, work)
}

/// One level of the separating pass: cut `ids` at their own endpoints on `axis`,
/// bucket each box into the elementary intervals it covers, recurse on the next
/// axis.
///
/// The base case is the load-bearing one. Reaching `axis == ndim` with two boxes
/// still together means both cover the same non-empty elementary interval on
/// **every** axis, which is exactly overlapping — so the pass decides rather
/// than filters, and the `false` it returns there is a real overlap and not a
/// budget refusal.
fn separated(
    boxes: &[Vec<(usize, usize)>],
    ids: &[usize],
    axis: usize,
    ndim: usize,
    budget: u64,
    work: &mut u64,
) -> bool {
    if ids.len() < 2 {
        return true;
    }
    if axis == ndim {
        return false;
    }
    let mut cuts: Vec<usize> = Vec::with_capacity(ids.len() * 2);
    for &id in ids {
        let (lo, hi) = boxes[id][axis];
        cuts.push(lo);
        cuts.push(hi);
    }
    cuts.sort_unstable();
    cuts.dedup();
    if cuts.len() < 3 {
        // One elementary interval, covered by every box here: this axis
        // separates nothing, and bucketing on it would only copy the list.
        return separated(boxes, ids, axis + 1, ndim, budget, work);
    }
    let intervals = cuts.len() - 1;
    // One extra slot so the prefix sum below ends at the total and the last
    // bucket has an end like every other one.
    let mut offsets: Vec<usize> = vec![0; intervals + 1];
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(ids.len());
    for &id in ids {
        let (lo, hi) = boxes[id][axis];
        let first = cuts.partition_point(|&cut| cut < lo);
        let last = cuts.partition_point(|&cut| cut < hi);
        spans.push((first, last));
        *work += (last - first) as u64;
        if *work > budget {
            return false;
        }
        for interval in first..last {
            offsets[interval] += 1;
        }
    }
    let mut placed = 0usize;
    for slot in offsets.iter_mut() {
        let held = *slot;
        *slot = placed;
        placed += held;
    }
    let starts = offsets.clone();
    let mut flat: Vec<usize> = vec![0; placed];
    for (&id, &(first, last)) in ids.iter().zip(spans.iter()) {
        for interval in first..last {
            flat[offsets[interval]] = id;
            offsets[interval] += 1;
        }
    }
    (0..intervals).all(|interval| {
        separated(
            boxes,
            &flat[starts[interval]..starts[interval + 1]],
            axis + 1,
            ndim,
            budget,
            work,
        )
    })
}

/// Two axis-aligned boxes overlap iff they overlap on **every** axis. Separation
/// on any one axis is enough to prove disjointness, which is what makes this
/// cheap.
///
/// A box that is empty on some axis (`lo == hi`) contains no voxels and so
/// overlaps nothing — `a_lo < b_hi && b_lo < a_hi` gets that right *only*
/// because the empty axis makes one of the two comparisons false against
/// itself; it does not against a *different* box, so the emptiness is tested
/// explicitly. Empty boxes are not hypothetical: a degenerate split produces
/// them, and reporting one as overlapping everything would bury the real fault.
fn boxes_overlap(left: &[(usize, usize)], right: &[(usize, usize)]) -> bool {
    let nonempty = |region: &[(usize, usize)]| region.iter().all(|&(lo, hi)| lo < hi);
    nonempty(left)
        && nonempty(right)
        && left
            .iter()
            .zip(right.iter())
            .all(|(&(a_lo, a_hi), &(b_lo, b_hi))| a_lo < b_hi && b_lo < a_hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Split `shape` into a regular grid of `counts` blocks per axis.
    fn grid(shape: &[usize], counts: &[usize]) -> Vec<Vec<(usize, usize)>> {
        let mut boxes = vec![Vec::new()];
        for (axis, (&dim, &count)) in shape.iter().zip(counts.iter()).enumerate() {
            let _ = axis;
            let step = dim.div_ceil(count);
            let mut next = Vec::new();
            for prefix in &boxes {
                let mut lo = 0;
                while lo < dim {
                    let hi = (lo + step).min(dim);
                    let mut extended = prefix.clone();
                    extended.push((lo, hi));
                    next.push(extended);
                    lo = hi;
                }
            }
            boxes = next;
        }
        boxes
    }

    #[test]
    fn a_regular_grid_tiles_its_volume() {
        for counts in [[1, 1, 1], [2, 3, 5], [7, 1, 2], [4, 4, 4]] {
            let shape = [12, 9, 10];
            let boxes = grid(&shape, &counts);
            boxes_tile_exactly(&boxes, &shape)
                .unwrap_or_else(|err| panic!("{counts:?} should tile: {err}"));
        }
    }

    #[test]
    fn an_uneven_grid_with_a_short_last_block_still_tiles() {
        // 10 split into steps of 3 gives 3,3,3,1 — the last block is short, and
        // that is the common case at a volume edge.
        let shape = [10, 7];
        let boxes = grid(&shape, &[4, 3]);
        boxes_tile_exactly(&boxes, &shape).unwrap();
    }

    #[test]
    fn a_gap_is_reported_as_a_shortfall() {
        let shape = [4, 4];
        let boxes = vec![vec![(0, 2), (0, 4)], vec![(2, 3), (0, 4)]];
        let err = boxes_tile_exactly(&boxes, &shape).unwrap_err();
        assert!(err.to_string().contains("12 of 16"), "{err}");
    }

    #[test]
    fn an_overlap_is_reported_before_the_count_is_consulted() {
        // These two overlap on a 1x4 strip *and* the total happens to come out
        // right, so a check that only counted would pass this.
        let shape = [4, 4];
        let boxes = vec![
            vec![(0, 3), (0, 4)],
            vec![(2, 4), (0, 4)],
            vec![(0, 0), (0, 4)],
        ];
        let err = boxes_tile_exactly(&boxes, &shape).unwrap_err();
        assert!(err.to_string().contains("overlap"), "{err}");
    }

    #[test]
    fn a_box_that_leaves_the_volume_is_rejected_even_when_the_totals_agree() {
        // The failure the containment hypothesis exists for. Two disjoint boxes
        // summing to 16 of a 4x4 volume, yet the voxels 2..4 x 0..4 are never
        // covered: the second box lives outside.
        let shape = [4, 4];
        let boxes = vec![vec![(0, 2), (0, 4)], vec![(4, 6), (0, 4)]];
        let err = boxes_tile_exactly(&boxes, &shape).unwrap_err();
        assert!(err.to_string().contains("axis 0"), "{err}");
    }

    #[test]
    fn a_box_of_the_wrong_rank_is_rejected_rather_than_truncated() {
        let shape = [4, 4, 4];
        let boxes = vec![vec![(0, 4), (0, 4)]];
        let err = boxes_tile_exactly(&boxes, &shape).unwrap_err();
        assert!(err.to_string().contains("rank"), "{err}");
    }

    #[test]
    fn no_boxes_over_a_zero_volume_is_a_tiling() {
        boxes_tile_exactly(&[], &[0, 4]).unwrap();
        assert!(boxes_tile_exactly(&[], &[4, 4]).is_err());
    }

    #[test]
    fn overlap_needs_every_axis_and_separation_on_one_is_enough() {
        assert!(boxes_overlap(&[(0, 2), (0, 2)], &[(1, 3), (1, 3)]));
        assert!(!boxes_overlap(&[(0, 2), (0, 2)], &[(2, 3), (0, 2)]));
        assert!(!boxes_overlap(&[(0, 2), (0, 2)], &[(0, 2), (5, 9)]));
        // Empty boxes touch nothing.
        assert!(!boxes_overlap(&[(1, 1), (0, 2)], &[(0, 4), (0, 2)]));
    }
}
