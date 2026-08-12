// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Comparing a produced label volume against a known-correct one.
//
// The ids will not match. Nothing produces object 7 because the truth called it
// object 7 — labels come out of whatever order a connected-components pass, or a
// block scheduler, happened to visit things in. So the comparison is by
// **overlap**, and the only interesting question is how the two partitions
// relate:
//
// | outcome | what it means |
// |---|---|
// | matched | one truth object, one produced object, and neither has another significant partner |
// | split | one truth object shares its voxels with several produced objects |
// | merged | one produced object covers several truth objects |
// | missed | a truth object no produced object significantly overlaps |
// | spurious | a produced object that significantly overlaps no truth object |
//
// Where the honesty is
// --------------------
// "Significant" needs a definition and every definition is a judgement, so this
// one is stated rather than buried: an association counts when the shared voxels
// are at least `min_overlap` of **the smaller** of the two objects. That is
// symmetric, which matters — a rule stated only in terms of the truth object
// cannot see a merge, and a rule stated only in terms of the produced object
// cannot see a split. It also means the outcome depends on a threshold the
// caller picked, and no summary here pretends otherwise.
//
// What this does not do. It does not score. There is no F-measure, no average
// precision, no single number, because the useful failure information is in
// *which* objects split and *where*, and a scalar throws that away at exactly
// the moment somebody is trying to work out whether a halo is too small.

use std::collections::BTreeMap;

use ndarray::ArrayView3;

use crate::error::{Error, Result};

/// One truth object and the produced object that corresponds to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Matched {
    pub truth: u32,
    pub produced: u32,
    /// Voxels the two share.
    pub shared: u64,
    /// Shared over union — 1.0 when the two are the same set of voxels.
    pub iou: f64,
}

/// One truth object that several produced objects divide between them.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub truth: u32,
    /// The produced objects, ascending, with the voxels each takes.
    pub pieces: Vec<(u32, u64)>,
}

/// One produced object that covers several truth objects.
#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    pub produced: u32,
    /// The truth objects, ascending, with the voxels each contributes.
    pub parts: Vec<(u32, u64)>,
}

/// How a produced labelling relates to the truth.
///
/// Every vector is sorted by id, so two runs of the same comparison are
/// comparable line by line.
#[derive(Debug, Clone, PartialEq)]
pub struct Agreement {
    pub matched: Vec<Matched>,
    pub split: Vec<Split>,
    pub merged: Vec<Merged>,
    /// Truth objects with no significant partner.
    pub missed: Vec<u32>,
    /// Produced objects with no significant partner.
    pub spurious: Vec<u32>,
    /// Objects present in each volume, for the denominators.
    pub truth_objects: usize,
    pub produced_objects: usize,
    /// The threshold the outcomes above were decided with.
    pub min_overlap: f64,
}

impl Agreement {
    /// Truth objects that came out as exactly one produced object.
    pub fn matched_count(&self) -> usize {
        self.matched.len()
    }

    /// True when every truth object matched exactly one produced object and
    /// nothing was invented — the only outcome worth calling "correct".
    pub fn is_exact(&self) -> bool {
        self.split.is_empty()
            && self.merged.is_empty()
            && self.missed.is_empty()
            && self.spurious.is_empty()
            && self.matched.len() == self.truth_objects
    }

    /// One line, for a test failure message.
    pub fn summary(&self) -> String {
        format!(
            "{} truth, {} produced, at overlap {:.2}: {} matched, {} split, {} merged, \
             {} missed, {} spurious",
            self.truth_objects,
            self.produced_objects,
            self.min_overlap,
            self.matched.len(),
            self.split.len(),
            self.merged.len(),
            self.missed.len(),
            self.spurious.len()
        )
    }
}

impl std::fmt::Display for Agreement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.summary())
    }
}

/// Compare a produced label volume against the truth, by overlap.
///
/// `min_overlap` is a fraction in `0.0..=1.0`: an association counts when the
/// shared voxels are at least that fraction of the smaller of the two objects.
/// `0.5` is the usual choice and has the property that each object has at most
/// one partner at that level, so anything reported as a split or a merge at 0.5
/// is a genuine disagreement about *how many objects there are* rather than
/// about where their edges lie.
///
/// Label `0` is background in both volumes and is never an object.
pub fn compare_labels(
    truth: ArrayView3<u32>,
    produced: ArrayView3<u32>,
    min_overlap: f64,
) -> Result<Agreement> {
    if truth.shape() != produced.shape() {
        return Err(Error::ShapeMismatch {
            expected: truth.shape().to_vec(),
            got: produced.shape().to_vec(),
        });
    }
    if !(0.0..=1.0).contains(&min_overlap) {
        return Err(Error::invalid(format!(
            "compare_labels: min_overlap {min_overlap} is outside 0.0..=1.0"
        )));
    }

    let mut truth_size: BTreeMap<u32, u64> = BTreeMap::new();
    let mut produced_size: BTreeMap<u32, u64> = BTreeMap::new();
    let mut shared: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for (&here, &there) in truth.iter().zip(produced.iter()) {
        if here != 0 {
            *truth_size.entry(here).or_default() += 1;
        }
        if there != 0 {
            *produced_size.entry(there).or_default() += 1;
        }
        if here != 0 && there != 0 {
            *shared.entry((here, there)).or_default() += 1;
        }
    }

    // Significant associations, both ways round.
    let mut by_truth: BTreeMap<u32, Vec<(u32, u64)>> = BTreeMap::new();
    let mut by_produced: BTreeMap<u32, Vec<(u32, u64)>> = BTreeMap::new();
    for (&(here, there), &count) in &shared {
        let smaller = truth_size[&here].min(produced_size[&there]) as f64;
        if (count as f64) < min_overlap * smaller {
            continue;
        }
        by_truth.entry(here).or_default().push((there, count));
        by_produced.entry(there).or_default().push((here, count));
    }

    let mut agreement = Agreement {
        matched: Vec::new(),
        split: Vec::new(),
        merged: Vec::new(),
        missed: Vec::new(),
        spurious: Vec::new(),
        truth_objects: truth_size.len(),
        produced_objects: produced_size.len(),
        min_overlap,
    };

    for (&here, &size) in &truth_size {
        match by_truth.get(&here) {
            None => agreement.missed.push(here),
            Some(partners) if partners.len() > 1 => agreement.split.push(Split {
                truth: here,
                pieces: partners.clone(),
            }),
            Some(partners) => {
                let (there, count) = partners[0];
                // One partner from this side is not enough: if that produced
                // object also claims another truth object, this is a merge, and
                // it is reported there.
                if by_produced.get(&there).map(Vec::len).unwrap_or(0) == 1 {
                    let union = size + produced_size[&there] - count;
                    agreement.matched.push(Matched {
                        truth: here,
                        produced: there,
                        shared: count,
                        iou: count as f64 / union as f64,
                    });
                }
            }
        }
    }

    for &there in produced_size.keys() {
        match by_produced.get(&there) {
            None => agreement.spurious.push(there),
            Some(parts) if parts.len() > 1 => agreement.merged.push(Merged {
                produced: there,
                parts: parts.clone(),
            }),
            Some(_) => {}
        }
    }

    Ok(agreement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// A volume of three separated boxes, labelled `1`, `2`, `3` along x.
    fn three_objects() -> Array3<u32> {
        let mut volume = Array3::<u32>::zeros((4, 4, 12));
        for (index, range) in [(1u32, 0..3), (2, 4..7), (3, 8..11)] {
            for x in range {
                for z in 0..4 {
                    for y in 0..4 {
                        volume[[z, y, x]] = index;
                    }
                }
            }
        }
        volume
    }

    #[test]
    fn relabelled_but_identical_is_a_clean_match() {
        let truth = three_objects();
        let produced = truth.map(|&label| match label {
            1 => 30,
            2 => 10,
            3 => 20,
            other => other,
        });
        let agreement = compare_labels(truth.view(), produced.view(), 0.5).unwrap();
        assert!(agreement.is_exact(), "{}", agreement.summary());
        assert_eq!(agreement.matched.len(), 3);
        assert_eq!(agreement.matched[0].truth, 1);
        assert_eq!(agreement.matched[0].produced, 30);
        assert!((agreement.matched[0].iou - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_truth_object_cut_in_two_is_a_split() {
        let truth = three_objects();
        let mut produced = truth.clone();
        // Cut object 2 down the middle: the far half becomes its own label.
        for z in 0..4 {
            for y in 0..4 {
                produced[[z, y, 6]] = 9;
            }
        }
        let agreement = compare_labels(truth.view(), produced.view(), 0.3).unwrap();
        assert_eq!(agreement.split.len(), 1);
        assert_eq!(agreement.split[0].truth, 2);
        assert_eq!(
            agreement.split[0]
                .pieces
                .iter()
                .map(|&(id, _)| id)
                .collect::<Vec<_>>(),
            vec![2, 9]
        );
        assert!(!agreement.is_exact());
    }

    #[test]
    fn two_truth_objects_under_one_label_is_a_merge() {
        let truth = three_objects();
        let produced = truth.map(|&label| if label == 2 { 1 } else { label });
        let agreement = compare_labels(truth.view(), produced.view(), 0.5).unwrap();
        assert_eq!(agreement.merged.len(), 1);
        assert_eq!(agreement.merged[0].produced, 1);
        assert_eq!(
            agreement.merged[0]
                .parts
                .iter()
                .map(|&(id, _)| id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        // Neither of the two is also reported as matched.
        assert_eq!(
            agreement
                .matched
                .iter()
                .map(|m| m.truth)
                .collect::<Vec<_>>(),
            vec![3]
        );
    }

    #[test]
    fn an_object_nobody_produced_is_missed_and_one_nobody_asked_for_is_spurious() {
        let truth = three_objects();
        let mut produced = truth.map(|&label| if label == 3 { 0 } else { label });
        for z in 0..4 {
            for y in 0..4 {
                produced[[z, y, 11]] = 44;
            }
        }
        let agreement = compare_labels(truth.view(), produced.view(), 0.5).unwrap();
        assert_eq!(agreement.missed, vec![3]);
        assert_eq!(agreement.spurious, vec![44]);
        assert_eq!(agreement.truth_objects, 3);
        assert_eq!(agreement.produced_objects, 3);
    }

    #[test]
    fn a_sliver_of_overlap_is_not_an_association() {
        let truth = three_objects();
        let mut produced = truth.clone();
        // One voxel of object 1 gets object 2's label: far below the threshold.
        produced[[0, 0, 0]] = 2;
        let agreement = compare_labels(truth.view(), produced.view(), 0.5).unwrap();
        assert!(agreement.split.is_empty());
        assert!(agreement.merged.is_empty());
        assert_eq!(agreement.matched.len(), 3);
    }

    #[test]
    fn mismatched_shapes_are_refused() {
        let truth = three_objects();
        let produced = Array3::<u32>::zeros((4, 4, 11));
        assert!(compare_labels(truth.view(), produced.view(), 0.5).is_err());
        assert!(compare_labels(truth.view(), truth.view(), 1.5).is_err());
    }

    #[test]
    fn an_empty_produced_volume_misses_everything() {
        let truth = three_objects();
        let produced = Array3::<u32>::zeros((4, 4, 12));
        let agreement = compare_labels(truth.view(), produced.view(), 0.5).unwrap();
        assert_eq!(agreement.missed, vec![1, 2, 3]);
        assert_eq!(agreement.produced_objects, 0);
        assert!(!agreement.is_exact());
    }
}
