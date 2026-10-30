// SPDX-License-Identifier: MIT

use super::*;
use serde_json::json;

fn fixture() -> (DetectionSet, CandidateSet) {
    let geometry = SeriesGeometry::new([1, 100, 100], [1.0; 3]).unwrap();
    let detections = DetectionSet::new(
        geometry,
        vec![
            Frame {
                t_index: 0,
                elapsed_seconds: Some(0.0),
            },
            Frame {
                t_index: 1,
                elapsed_seconds: Some(7200.0),
            },
        ],
        [
            (1, 0, 10.0, 80.0),
            (2, 0, 12.0, 20.0),
            (3, 1, 11.0, 20.0),
            (4, 1, 13.0, 80.0),
        ]
        .into_iter()
        .map(|(id, t_index, x, size)| Detection {
            id,
            t_index,
            native: [0.0, 20.0, x],
            registered: [0.0, 20.0, x],
            size,
        })
        .collect(),
    )
    .unwrap();
    let candidates = generate_candidates(&detections, &CandidateParams::distance(10.0)).unwrap();
    (detections, candidates)
}

fn pairs(graph: &TrackGraph) -> Vec<(u64, u64)> {
    graph.links().iter().map(|e| (e.from, e.to)).collect()
}

#[test]
fn same_scores_separate_greedy_from_assignment() {
    let (detections, candidates) = fixture();
    let scores = ScoredCandidates::from_costs(
        candidates,
        vec![1.0, 2.0, 2.0, 100.0],
        json!({"fixture": "assignment conflict"}),
    )
    .unwrap();
    let lap = FrameLapLinker::new(LapParams {
        alternative: AlternativeCost::Fixed(200.0),
        ..LapParams::default()
    })
    .unwrap();
    let greedy = GreedyFrameLinker::new(AlternativeCost::Fixed(200.0)).unwrap();
    assert_eq!(
        pairs(&lap.link_scored(&detections, &scores).unwrap()),
        vec![(1, 4), (2, 3)]
    );
    assert_eq!(
        pairs(&greedy.link_scored(&detections, &scores).unwrap()),
        vec![(1, 3), (2, 4)]
    );
}

#[test]
fn size_scoring_can_change_a_crossing_assignment_without_changing_candidates() {
    let (detections, candidates) = fixture();
    let lap = FrameLapLinker::new(LapParams {
        alternative: AlternativeCost::Fixed(1000.0),
        ..LapParams::default()
    })
    .unwrap();
    let motion =
        ScoredCandidates::score(&detections, &candidates, &GeometricCost::default()).unwrap();
    let size = ScoredCandidates::score(
        &detections,
        &candidates,
        &GeometricCost::new(0.0, 400.0).unwrap(),
    )
    .unwrap();
    assert_eq!(motion.candidates().edges(), size.candidates().edges());
    assert_eq!(
        pairs(&lap.link_scored(&detections, &motion).unwrap()),
        vec![(1, 3), (2, 4)]
    );
    assert_eq!(
        pairs(&lap.link_scored(&detections, &size).unwrap()),
        vec![(1, 4), (2, 3)]
    );
    // The adapter is usable through the existing public Linker contract.
    let adapter = CostedLinker {
        solver: lap,
        scorer: GeometricCost::new(0.0, 400.0).unwrap(),
    };
    assert_eq!(
        pairs(&adapter.link(&detections, &candidates).unwrap()),
        vec![(1, 4), (2, 3)]
    );
    assert_eq!(
        adapter.parameters()["scorer"]["parameters"]["size_weight"],
        400.0
    );
}

#[test]
fn extracted_geometric_scoring_preserves_legacy_frame_and_segment_outputs() {
    let (detections, candidates) = fixture();
    let frame = LapParams {
        size_weight: 400.0,
        ..LapParams::default()
    };
    let scores = ScoredCandidates::score(
        &detections,
        &candidates,
        &GeometricCost::new(0.0, 400.0).unwrap(),
    )
    .unwrap();
    let lap = FrameLapLinker::new(frame).unwrap();
    assert_eq!(
        lap.link(&detections, &candidates).unwrap().links(),
        lap.link_scored(&detections, &scores).unwrap().links()
    );
    let segment = SegmentLapLinker::new(SegmentParams {
        frame,
        splitting: Some(5.0),
        gap_closing: Some(GapClosing {
            max_distance: 5.0,
            max_frame_gap: 2,
        }),
        ..SegmentParams::default()
    })
    .unwrap();
    assert_eq!(
        segment.link(&detections, &candidates).unwrap().links(),
        segment.link_scored(&detections, &scores).unwrap().links()
    );
}

#[test]
fn imported_scores_are_finite_signed_and_match_the_timeline() {
    let (detections, candidates) = fixture();
    assert!(ScoredCandidates::from_costs(candidates.clone(), vec![0.0], json!({})).is_err());
    assert!(
        ScoredCandidates::from_costs(candidates.clone(), vec![f64::NAN; 4], json!({})).is_err()
    );
    let scores =
        ScoredCandidates::from_costs(candidates.clone(), vec![-3.0, -2.0, -2.0, 10.0], json!({}))
            .unwrap();
    assert_eq!(
        pairs(
            &FrameLapLinker::default()
                .link_scored(&detections, &scores)
                .unwrap()
        ),
        vec![(1, 4), (2, 3)]
    );
    let mut wrong_edges = candidates.edges().to_vec();
    wrong_edges[0].elapsed_seconds = Some(1.0);
    let wrong = CandidateSet::from_edges(wrong_edges).unwrap();
    let scores = ScoredCandidates::from_costs(wrong, vec![1.0; 4], json!({})).unwrap();
    assert!(FrameLapLinker::default()
        .link_scored(&detections, &scores)
        .is_err());
}

#[test]
fn greedy_respects_birth_and_death_alternatives() {
    let (detections, candidates) = fixture();
    let scores = ScoredCandidates::from_costs(candidates, vec![5.0; 4], json!({})).unwrap();
    let greedy = GreedyFrameLinker::new(AlternativeCost::Fixed(2.0)).unwrap();
    assert!(greedy
        .link_scored(&detections, &scores)
        .unwrap()
        .links()
        .is_empty());
    assert!(FrameLapLinker::new(LapParams {
        alternative: AlternativeCost::Fixed(2.0),
        ..LapParams::default()
    })
    .unwrap()
    .link_scored(&detections, &scores)
    .unwrap()
    .links()
    .is_empty());
}
