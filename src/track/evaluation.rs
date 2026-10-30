// SPDX-License-Identifier: MIT

//! Fixed-mask evaluation against a complete reference graph on the same
//! detection IDs and timeline. This is not CTC TRA, nor a segmentation score.
//! A partially reviewed graph must first be restricted to an explicitly
//! complete evaluation region; absent edges otherwise mean true negatives.

use super::{LinkKind, TrackGraph};
use crate::error::{Error, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AssociationMetric {
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    /// Undefined denominators are null, never a fabricated perfect score.
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

fn metric<T: Ord>(reference: &BTreeSet<T>, predicted: &BTreeSet<T>) -> AssociationMetric {
    let tp = reference.intersection(predicted).count();
    let fp = predicted.len() - tp;
    let fn_ = reference.len() - tp;
    let ratio = |a: usize, b: usize| (b > 0).then(|| a as f64 / b as f64);
    AssociationMetric {
        true_positive: tp,
        false_positive: fp,
        false_negative: fn_,
        precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + fn_),
        f1: ratio(2 * tp, 2 * tp + fp + fn_),
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrackingComparison {
    pub links: AssociationMetric,
    pub unbranched_links: AssociationMetric,
    /// An event is correct only when the mother and *both* daughters agree.
    /// A matching mother with one wrong daughter is one FP and one FN.
    pub binary_divisions: AssociationMetric,
    /// Do not hide multifurcations by treating them as binary events.
    pub reference_multifurcations: usize,
    pub predicted_multifurcations: usize,
}

fn division_sets(graph: &TrackGraph) -> (BTreeSet<(u64, u64, u64)>, usize) {
    let mut by_mother: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for link in graph.links() {
        by_mother.entry(link.from).or_default().push(link.to);
    }
    let mut events = BTreeSet::new();
    let mut multiple = 0;
    for (mother, mut daughters) in by_mother {
        daughters.sort_unstable();
        match daughters.as_slice() {
            &[a, b] => {
                events.insert((mother, a, b));
            }
            ds if ds.len() > 2 => {
                multiple += 1;
            }
            _ => {}
        }
    }
    (events, multiple)
}

pub fn compare_graphs(
    reference: &TrackGraph,
    predicted: &TrackGraph,
) -> Result<TrackingComparison> {
    if !reference
        .detection_frames()
        .eq(predicted.detection_frames())
    {
        return Err(Error::invalid(
            "fixed-mask graph comparison requires identical detection IDs and frames",
        ));
    }
    let edges = |g: &TrackGraph| {
        g.links()
            .iter()
            .map(|l| (l.from, l.to))
            .collect::<BTreeSet<_>>()
    };
    let unbranched = |g: &TrackGraph| {
        g.links()
            .iter()
            .filter(|l| matches!(l.kind, LinkKind::Continue | LinkKind::Gap))
            .map(|l| (l.from, l.to))
            .collect::<BTreeSet<_>>()
    };
    let (reference_divisions, reference_multifurcations) = division_sets(reference);
    let (predicted_divisions, predicted_multifurcations) = division_sets(predicted);
    Ok(TrackingComparison {
        links: metric(&edges(reference), &edges(predicted)),
        unbranched_links: metric(&unbranched(reference), &unbranched(predicted)),
        binary_divisions: metric(&reference_divisions, &predicted_divisions),
        reference_multifurcations,
        predicted_multifurcations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::{
        Detection, DetectionSet, Frame, LinkerCapabilities, SeriesGeometry, TrackLink,
    };

    fn graph(edges: &[(u64, u64)]) -> TrackGraph {
        let detections = DetectionSet::new(
            SeriesGeometry::new([1, 20, 20], [1.0; 3]).unwrap(),
            vec![
                Frame {
                    t_index: 0,
                    elapsed_seconds: None,
                },
                Frame {
                    t_index: 1,
                    elapsed_seconds: None,
                },
            ],
            [1, 2, 3, 4, 5]
                .into_iter()
                .map(|id| Detection {
                    id,
                    t_index: u64::from(id != 1 && id != 4),
                    native: [0.0; 3],
                    registered: [0.0; 3],
                    size: 10.0,
                })
                .collect(),
        )
        .unwrap();
        let mut links = edges
            .iter()
            .map(|&(from, to)| TrackLink {
                from,
                to,
                dt: 1,
                elapsed_seconds: None,
                cost: 0.0,
                kind: LinkKind::Continue,
            })
            .collect::<Vec<_>>();
        TrackGraph::classify(&mut links);
        TrackGraph::new(
            &detections,
            links,
            LinkerCapabilities {
                divisions: true,
                ..LinkerCapabilities::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn one_wrong_daughter_is_not_a_correct_division() {
        let reference = graph(&[(1, 2), (1, 3), (4, 5)]);
        let predicted = graph(&[(1, 2), (1, 5), (4, 3)]);
        let scores = compare_graphs(&reference, &predicted).unwrap();
        assert_eq!(scores.links.true_positive, 1);
        assert_eq!(scores.binary_divisions.true_positive, 0);
        assert_eq!(scores.binary_divisions.false_positive, 1);
        assert_eq!(scores.binary_divisions.false_negative, 1);
        assert_eq!(scores.binary_divisions.f1, Some(0.0));
    }

    #[test]
    fn absent_and_false_divisions_do_not_score_as_perfect() {
        let no_division = graph(&[(1, 2), (4, 5)]);
        let division = graph(&[(1, 2), (1, 3), (4, 5)]);
        let empty = compare_graphs(&no_division, &no_division).unwrap();
        assert_eq!(empty.binary_divisions.f1, None);
        let false_division = compare_graphs(&no_division, &division).unwrap();
        assert_eq!(false_division.binary_divisions.precision, Some(0.0));
        assert_eq!(false_division.binary_divisions.recall, None);
    }
}
