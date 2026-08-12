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
// Cost is `O(n² · ndim)` in the block count — a few hundred at tile scale, tens
// of thousands at full scale, so still milliseconds — where the obvious
// alternative, a per-voxel write counter, is `O(voxels)` in both time *and*
// space and would need a second copy of the volume to run at all.
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

use crate::error::{Error, Result};

/// Whether `boxes` partition `shape` exactly — every voxel covered once.
///
/// Each box is per-axis half-open `(lo, hi)` ranges, in the same order as
/// `shape`. Returns the first violation found, named: an out-of-range box, an
/// overlapping pair, or a coverage shortfall.
///
/// An empty `boxes` over a zero-volume `shape` tiles it, which is the degenerate
/// case a plan with no blocks produces and is not an error.
pub fn boxes_tile_exactly(boxes: &[Vec<(usize, usize)>], shape: &[usize]) -> Result<()> {
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

    for (first, left) in boxes.iter().enumerate() {
        for right in boxes.iter().skip(first + 1) {
            if boxes_overlap(left, right) {
                return Err(Error::InvalidArgument(format!(
                    "valid regions do not tile the volume exactly \
                     (regions {left:?} and {right:?} overlap)"
                )));
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
