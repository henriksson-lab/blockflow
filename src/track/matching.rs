// SPDX-License-Identifier: MIT

//! Spatial matching for comparing different segmentations of the SAME frame.
//! A match is geometric correspondence, not an accuracy claim unless one mask
//! is an appropriate independent reference. IDs remain exact uint64 values.

use super::{assign_sparse, overlap_counts, OverlapCounts};
use crate::error::{Error, Result};
use ndarray::ArrayView3;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelMatch {
    pub reference: u64,
    pub predicted: u64,
    pub shared_pixels: u64,
    pub iou: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameMatching {
    pub matches: Vec<LabelMatch>,
    pub unmatched_reference: Vec<u64>,
    pub unmatched_predicted: Vec<u64>,
}

/// Maximize match count at the specified IoU threshold, then total IoU.
/// Independent of numeric IDs, unlike fixed-mask graph comparison. Inputs must
/// address the same native pixel lattice at the same time; no drift is applied.
pub fn match_label_frame(
    reference: ArrayView3<'_, u64>,
    predicted: ArrayView3<'_, u64>,
    minimum_iou: f64,
) -> Result<FrameMatching> {
    match_overlaps(&overlap_counts(reference, predicted)?, minimum_iou)
}

/// Match merged overlap counts, allowing blockwise pixel counting upstream.
/// A private alternative per reference label makes every assignment feasible.
pub fn match_overlaps(counts: &OverlapCounts, minimum_iou: f64) -> Result<FrameMatching> {
    if !minimum_iou.is_finite() || minimum_iou <= 0.0 || minimum_iou > 1.0 {
        return Err(Error::invalid(
            "minimum matching IoU must be finite and in (0,1]",
        ));
    }
    for (id, size) in counts.sizes_a.iter().chain(&counts.sizes_b) {
        if *id == 0 || *size == 0 {
            return Err(Error::invalid(
                "overlap sizes require nonzero labels and positive pixel counts",
            ));
        }
    }
    let sources = counts.sizes_a.keys().copied().collect::<Vec<_>>();
    let targets = counts.sizes_b.keys().copied().collect::<Vec<_>>();
    let mut by_source = BTreeMap::<u64, u128>::new();
    let mut by_target = BTreeMap::<u64, u128>::new();
    let mut eligible = BTreeMap::new();
    for (&(a, b), &shared) in &counts.shared {
        let sa = *counts
            .sizes_a
            .get(&a)
            .ok_or_else(|| Error::invalid("shared pixels reference an unknown reference label"))?;
        let sb = *counts
            .sizes_b
            .get(&b)
            .ok_or_else(|| Error::invalid("shared pixels reference an unknown predicted label"))?;
        if shared == 0 || shared > sa.min(sb) {
            return Err(Error::invalid(
                "shared pixel counts must be positive and no greater than either object",
            ));
        }
        *by_source.entry(a).or_default() += u128::from(shared);
        *by_target.entry(b).or_default() += u128::from(shared);
        let union = u128::from(sa) + u128::from(sb) - u128::from(shared);
        let iou = shared as f64 / union as f64;
        if iou >= minimum_iou {
            eligible.insert((a, b), (shared, iou));
        }
    }
    if by_source
        .iter()
        .any(|(id, sum)| *sum > u128::from(counts.sizes_a[id]))
        || by_target
            .iter()
            .any(|(id, sum)| *sum > u128::from(counts.sizes_b[id]))
    {
        return Err(Error::invalid(
            "summed intersections exceed an object pixel count",
        ));
    }
    // At most k IoU terms, each <=1. A bonus >k makes one additional match
    // preferable to every possible IoU improvement among fewer matches.
    let bonus = sources.len().min(targets.len()) as f64 + 1.0;
    let mut rows = vec![Vec::new(); sources.len()];
    for (&(a, b), &(_, iou)) in &eligible {
        let row = sources.binary_search(&a).expect("known source");
        let col = targets.binary_search(&b).expect("known target");
        rows[row].push((col, -bonus - iou));
    }
    for (row, entries) in rows.iter_mut().enumerate() {
        entries.push((targets.len() + row, 0.0));
    }
    let assignment = assign_sparse(&rows, targets.len() + sources.len())?;
    let mut matches = Vec::new();
    let mut used = BTreeSet::new();
    let mut unmatched_reference = Vec::new();
    for (row, col) in assignment.into_iter().enumerate() {
        if col < targets.len() {
            let (reference, predicted) = (sources[row], targets[col]);
            let &(shared_pixels, iou) = eligible
                .get(&(reference, predicted))
                .expect("matched along an eligible pair");
            used.insert(predicted);
            matches.push(LabelMatch {
                reference,
                predicted,
                shared_pixels,
                iou,
            });
        } else {
            unmatched_reference.push(sources[row]);
        }
    }
    Ok(FrameMatching {
        matches,
        unmatched_reference,
        unmatched_predicted: targets
            .into_iter()
            .filter(|id| !used.contains(id))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn exact_pixel_match_preserves_large_unrelated_ids() {
        let big = 1u64 << 60;
        let a = Array3::from_shape_vec(
            (1, 1, 6),
            vec![0, big + 1, big + 1, big + 2, big + 2, big + 3],
        )
        .unwrap();
        let b = Array3::from_shape_vec((1, 1, 6), vec![0, 41, 41, 99, 99, 0]).unwrap();
        let result = match_label_frame(a.view(), b.view(), 0.5).unwrap();
        assert_eq!(
            result
                .matches
                .iter()
                .map(|m| (m.reference, m.predicted))
                .collect::<Vec<_>>(),
            vec![(big + 1, 41), (big + 2, 99)]
        );
        assert_eq!(result.unmatched_reference, vec![big + 3]);
        assert!(result.unmatched_predicted.is_empty());
        assert!(result.matches.iter().all(|m| m.iou == 1.0));
    }

    #[test]
    fn maximum_cardinality_precedes_greedy_iou() {
        // A overlaps X best, but only A can match Y. Greedy A->X loses B.
        let counts = OverlapCounts {
            sizes_a: BTreeMap::from([(1, 10), (2, 4)]),
            sizes_b: BTreeMap::from([(10, 10), (20, 4)]),
            shared: BTreeMap::from([((1, 10), 6), ((1, 20), 4), ((2, 10), 4)]),
        };
        let result = match_overlaps(&counts, 0.3).unwrap();
        assert_eq!(
            result
                .matches
                .iter()
                .map(|m| (m.reference, m.predicted))
                .collect::<Vec<_>>(),
            vec![(1, 20), (2, 10)]
        );
    }

    #[test]
    fn empty_frames_and_invalid_counts_are_explicit() {
        let result = match_overlaps(&OverlapCounts::default(), 0.5).unwrap();
        assert!(result.matches.is_empty());
        assert!(match_overlaps(&OverlapCounts::default(), 0.0).is_err());
        let bad = OverlapCounts {
            sizes_a: BTreeMap::from([(1, 2)]),
            sizes_b: BTreeMap::from([(2, 3)]),
            shared: BTreeMap::from([((1, 2), 3)]),
        };
        assert!(match_overlaps(&bad, 0.5).is_err());
    }
}
