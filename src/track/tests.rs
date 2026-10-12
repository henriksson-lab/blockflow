// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;

use ndarray::Array3;

use super::lap::{assign_sparse, hungarian};
use super::*;

/// SplitMix64: a deterministic generator, so fixtures need no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

const EXTENT: [usize; 3] = [1, 512, 512];

fn geometry() -> SeriesGeometry {
    SeriesGeometry::new(EXTENT, [1.0, 1.0, 1.0]).unwrap()
}

/// A ground-truth movie: cells on well-separated lanes performing small
/// random steps, some born late and some ending early. Returns the detections
/// and the true links.
struct Movie {
    frames: Vec<Frame>,
    detections: Vec<Detection>,
    truth: Vec<(u64, u64)>,
}

fn id_of(t: u64, y: f64, x: f64) -> u64 {
    // The same scheme the example uses: frame offset plus a per-frame local id.
    t * (EXTENT[1] * EXTENT[2]) as u64 + 1 + (y as u64) * EXTENT[2] as u64 + x as u64
}

fn movie(seed: u64, cells: usize, frames: u64) -> Movie {
    let mut rng = Rng(seed);
    let mut detections = Vec::new();
    let mut truth = Vec::new();
    for cell in 0..cells {
        // Lanes 40 px apart on a grid, steps of at most 3 px: the nearest
        // wrong neighbour is always far further than the true one.
        let mut y = 30.0 + 40.0 * (cell / 11) as f64 + rng.range(-2.0, 2.0);
        let mut x = 30.0 + 40.0 * (cell % 11) as f64 + rng.range(-2.0, 2.0);
        let start = if cell % 5 == 1 { 3 } else { 0 };
        let end = if cell % 7 == 2 {
            frames - 4
        } else {
            frames - 1
        };
        let mut previous = None;
        for t in start..=end {
            let id = id_of(t, y.round(), x.round());
            detections.push(Detection {
                id,
                t_index: t,
                native: [0.0, y, x],
                registered: [0.0, y, x],
                size: 50.0 + rng.range(-5.0, 5.0),
            });
            if let Some(p) = previous {
                truth.push((p, id));
            }
            previous = Some(id);
            y += rng.range(-3.0, 3.0);
            x += rng.range(-3.0, 3.0);
        }
    }
    truth.sort_unstable();
    Movie {
        frames: (0..frames)
            .map(|t| Frame {
                t_index: t,
                elapsed_seconds: None,
            })
            .collect(),
        detections,
        truth,
    }
}

fn link(set: &DetectionSet, max_distance: f64) -> TrackGraph {
    let candidates = generate_candidates(set, &CandidateParams::distance(max_distance)).unwrap();
    FrameLapLinker::default().link(set, &candidates).unwrap()
}

fn pairs(graph: &TrackGraph) -> Vec<(u64, u64)> {
    graph.links().iter().map(|l| (l.from, l.to)).collect()
}

#[test]
fn hungarian_matches_brute_force() {
    let mut rng = Rng(7);
    for n in 1..=6usize {
        for _ in 0..40 {
            let matrix: Vec<f64> = (0..n * n)
                .map(|_| {
                    if rng.unit() < 0.25 {
                        f64::INFINITY
                    } else {
                        (rng.range(0.0, 10.0) * 4.0).round() / 4.0
                    }
                })
                .collect();
            // Brute force over permutations.
            let mut perm: Vec<usize> = (0..n).collect();
            let mut best = f64::INFINITY;
            permute(&mut perm, 0, &mut |p| {
                let cost: f64 = p.iter().enumerate().map(|(r, &c)| matrix[r * n + c]).sum();
                if cost < best {
                    best = cost;
                }
            });
            match hungarian(&matrix, n) {
                Ok(assignment) => {
                    let cost: f64 = assignment
                        .iter()
                        .enumerate()
                        .map(|(r, &c)| matrix[r * n + c])
                        .sum();
                    assert!((cost - best).abs() < 1e-9, "n={n} cost={cost} best={best}");
                    let mut cols = assignment.clone();
                    cols.sort_unstable();
                    assert_eq!(cols, (0..n).collect::<Vec<_>>());
                }
                Err(_) => assert!(best.is_infinite()),
            }
        }
    }
}

fn permute(p: &mut Vec<usize>, k: usize, f: &mut dyn FnMut(&[usize])) {
    if k == p.len() {
        f(p);
        return;
    }
    for i in k..p.len() {
        p.swap(k, i);
        permute(p, k + 1, f);
        p.swap(k, i);
    }
}

#[test]
fn recovers_ground_truth_links_exactly() {
    let m = movie(1, 99, 20);
    let set = DetectionSet::new(geometry(), m.frames, m.detections).unwrap();
    let graph = link(&set, 12.0);
    assert_eq!(pairs(&graph), m.truth);
}

#[test]
fn tracks_and_events_follow_the_timeline() {
    let m = movie(2, 33, 12);
    let set = DetectionSet::new(geometry(), m.frames, m.detections).unwrap();
    let graph = link(&set, 12.0);
    let tracks = graph.tracks(&set, 10.0).unwrap();
    assert_eq!(tracks.len(), 33);
    for (cell, track) in tracks.iter().enumerate() {
        assert_eq!(track.track_id, track.first_id());
        assert!(track.parents.is_empty() && track.children.is_empty());
        let _ = cell;
    }
    let late: Vec<_> = tracks.iter().filter(|t| t.t_start == 3).collect();
    assert!(!late.is_empty());
    for track in &late {
        assert_eq!(track.events.start_reason, StartReason::Unmatched);
    }
    let early: Vec<_> = tracks.iter().filter(|t| t.t_end == 8).collect();
    assert!(!early.is_empty());
    for track in &early {
        assert_eq!(track.events.end_reason, EndReason::Unmatched);
    }
    for track in tracks.iter().filter(|t| t.t_start == 0) {
        assert_eq!(track.events.start_reason, StartReason::FirstFrame);
    }
    for track in tracks.iter().filter(|t| t.t_end == 11) {
        assert_eq!(track.events.end_reason, EndReason::LastFrame);
    }
    let ids = TrackGraph::track_ids(&tracks);
    assert_eq!(ids.len(), set.detections().len());
}

#[test]
fn output_is_independent_of_input_order() {
    let m = movie(3, 60, 10);
    let a = DetectionSet::new(geometry(), m.frames.clone(), m.detections.clone()).unwrap();
    let mut shuffled = m.detections.clone();
    let mut rng = Rng(99);
    for i in (1..shuffled.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        shuffled.swap(i, j);
    }
    let b = DetectionSet::new(geometry(), m.frames, shuffled).unwrap();
    let ga = link(&a, 12.0);
    let gb = link(&b, 12.0);
    assert_eq!(ga.links(), gb.links());
}

#[test]
fn components_give_the_same_optimum_as_one_matrix() {
    // Dense, ambiguous scene: many cells within each other's gate.
    let mut rng = Rng(11);
    let mut detections = Vec::new();
    for t in 0..2u64 {
        for i in 0..40u64 {
            let y = rng.range(10.0, 120.0);
            let x = rng.range(10.0, 120.0);
            detections.push(Detection {
                id: t * 1000 + i + 1,
                t_index: t,
                native: [0.0, y, x],
                registered: [0.0, y, x],
                size: 10.0,
            });
        }
    }
    let frames = (0..2)
        .map(|t| Frame {
            t_index: t,
            elapsed_seconds: None,
        })
        .collect();
    let set = DetectionSet::new(geometry(), frames, detections).unwrap();
    let candidates = generate_candidates(&set, &CandidateParams::distance(15.0)).unwrap();
    let alternative = 100.0;
    let linker = FrameLapLinker::new(LapParams {
        alternative: AlternativeCost::Fixed(alternative),
        ..LapParams::default()
    })
    .unwrap();
    let graph = linker.link(&set, &candidates).unwrap();
    let ours: f64 = graph.links().iter().map(|l| l.cost).sum::<f64>()
        + alternative * 2.0 * (40.0 - graph.links().len() as f64);

    // One augmented matrix over the whole pair.
    let sources: Vec<u64> = set.in_frame(0).iter().map(|d| d.id).collect();
    let targets: Vec<u64> = set.in_frame(1).iter().map(|d| d.id).collect();
    let (n, m) = (sources.len(), targets.len());
    let size = n + m;
    let mut matrix = vec![f64::INFINITY; size * size];
    for e in candidates.edges() {
        let r = sources.binary_search(&e.from).unwrap();
        let c = targets.binary_search(&e.to).unwrap();
        matrix[r * size + c] = e.distance * e.distance;
    }
    for i in 0..n {
        matrix[i * size + m + i] = alternative;
    }
    for j in 0..m {
        matrix[(n + j) * size + j] = alternative;
        for k in 0..n {
            matrix[(n + j) * size + m + k] = 0.0;
        }
    }
    let assignment = hungarian(&matrix, size).unwrap();
    let whole: f64 = assignment
        .iter()
        .enumerate()
        .map(|(r, &c)| matrix[r * size + c])
        .sum();
    assert!(
        (ours - whole).abs() < 1e-6,
        "components {ours} whole {whole}"
    );
}

#[test]
fn a_missing_frame_splits_tracks_with_frame_gap() {
    let m = movie(4, 10, 8);
    let frames: Vec<Frame> = m.frames.into_iter().filter(|f| f.t_index != 4).collect();
    let detections: Vec<Detection> = m
        .detections
        .into_iter()
        .filter(|d| d.t_index != 4)
        .collect();
    let set = DetectionSet::new(geometry(), frames, detections).unwrap();
    assert!(set.gap_after(3) && set.gap_before(5));
    let graph = link(&set, 12.0);
    assert!(graph.links().iter().all(|l| l.dt == 1));
    let tracks = graph.tracks(&set, 0.0).unwrap();
    for track in &tracks {
        if track.t_end == 3 {
            assert_eq!(track.events.end_reason, EndReason::FrameGap);
        }
        if track.t_start == 5 {
            assert_eq!(track.events.start_reason, StartReason::FrameGap);
        }
    }
    assert!(tracks.iter().any(|t| t.t_end == 3) && tracks.iter().any(|t| t.t_start == 5));
}

#[test]
fn border_state_uses_physical_threshold() {
    let detections = vec![
        Detection {
            id: 1,
            t_index: 0,
            native: [0.0, 3.0, 200.0],
            registered: [0.0, 3.0, 200.0],
            size: 1.0,
        },
        Detection {
            id: 2,
            t_index: 0,
            native: [0.0, 200.0, 200.0],
            registered: [0.0, 200.0, 200.0],
            size: 1.0,
        },
    ];
    let frames = vec![
        Frame {
            t_index: 0,
            elapsed_seconds: None,
        },
        Frame {
            t_index: 1,
            elapsed_seconds: None,
        },
    ];
    let set = DetectionSet::new(geometry(), frames, detections).unwrap();
    let graph = link(&set, 5.0);
    let tracks = graph.tracks(&set, 10.0).unwrap();
    assert_eq!(tracks[0].events.end_border, BorderState::NearBorder);
    assert!(tracks[0].events.likely_left_fov());
    assert_eq!(tracks[1].events.end_border, BorderState::Interior);
    assert!(!tracks[1].events.likely_left_fov());
}

#[test]
fn speed_gating_scales_with_elapsed_time() {
    let at = |id, t, x: f64| Detection {
        id,
        t_index: t,
        native: [0.0, 100.0, x],
        registered: [0.0, 100.0, x],
        size: 1.0,
    };
    let detections = vec![at(1, 0, 100.0), at(2, 1, 108.0), at(3, 2, 116.0)];
    let frames = vec![
        Frame {
            t_index: 0,
            elapsed_seconds: Some(0.0),
        },
        Frame {
            t_index: 1,
            elapsed_seconds: Some(10.0),
        },
        Frame {
            t_index: 2,
            elapsed_seconds: Some(12.0),
        },
    ];
    let set = DetectionSet::new(geometry(), frames, detections).unwrap();
    let params = CandidateParams {
        max_distance: None,
        max_speed: Some(1.0),
    };
    let candidates = generate_candidates(&set, &params).unwrap();
    // 8 px in 10 s passes; 8 px in 2 s does not.
    assert_eq!(
        candidates
            .edges()
            .iter()
            .map(|e| (e.from, e.to))
            .collect::<Vec<_>>(),
        vec![(1, 2)]
    );
    assert_eq!(candidates.edges()[0].elapsed_seconds, Some(10.0));
}

#[test]
fn overlap_features_feed_the_cost() {
    let mut a = Array3::<u64>::zeros((1, 8, 8));
    let mut b = Array3::<u64>::zeros((1, 8, 8));
    for y in 0..4 {
        for x in 0..4 {
            a[[0, y, x]] = 1;
            b[[0, y, x + 1]] = 10;
        }
    }
    let counts = overlap_counts(a.view(), b.view()).unwrap();
    assert_eq!(counts.shared[&(1, 10)], 12);
    let detections = vec![
        Detection {
            id: 1,
            t_index: 0,
            native: [0.0, 1.5, 1.5],
            registered: [0.0, 1.5, 1.5],
            size: 16.0,
        },
        Detection {
            id: 10,
            t_index: 1,
            native: [0.0, 1.5, 2.5],
            registered: [0.0, 1.5, 2.5],
            size: 16.0,
        },
    ];
    let frames = vec![
        Frame {
            t_index: 0,
            elapsed_seconds: None,
        },
        Frame {
            t_index: 1,
            elapsed_seconds: None,
        },
    ];
    let set = DetectionSet::new(
        SeriesGeometry::new([1, 8, 8], [1.0; 3]).unwrap(),
        frames,
        detections,
    )
    .unwrap();
    let mut candidates = generate_candidates(&set, &CandidateParams::distance(3.0)).unwrap();
    let linker = FrameLapLinker::new(LapParams {
        iou_weight: 1.0,
        ..LapParams::default()
    })
    .unwrap();
    assert!(
        linker.link(&set, &candidates).is_err(),
        "iou cost without overlap must be refused"
    );
    candidates.attach_overlap(0, &counts).unwrap();
    let iou = candidates.edges()[0].overlap.unwrap().iou;
    assert!((iou - 12.0 / 20.0).abs() < 1e-12);
    let graph = linker.link(&set, &candidates).unwrap();
    assert!((graph.links()[0].cost - (1.0 + (1.0 - 0.6))).abs() < 1e-12);
}

#[test]
fn validation_holds_linkers_to_their_capabilities() {
    let at = |id, t| Detection {
        id,
        t_index: t,
        native: [0.0, 50.0, 50.0],
        registered: [0.0, 50.0, 50.0],
        size: 1.0,
    };
    let frames = vec![
        Frame {
            t_index: 0,
            elapsed_seconds: None,
        },
        Frame {
            t_index: 1,
            elapsed_seconds: None,
        },
    ];
    let set = DetectionSet::new(geometry(), frames, vec![at(1, 0), at(2, 1), at(3, 1)]).unwrap();
    let divide = |to| graph::TrackLink {
        from: 1,
        to,
        dt: 1,
        elapsed_seconds: None,
        cost: 0.0,
        kind: LinkKind::Divide,
    };
    let none = LinkerCapabilities::default();
    assert!(TrackGraph::new(&set, vec![divide(2), divide(3)], none).is_err());
    let with = LinkerCapabilities {
        divisions: true,
        ..none
    };
    let graph = TrackGraph::new(&set, vec![divide(2), divide(3)], with).unwrap();
    let tracks = graph.tracks(&set, 0.0).unwrap();
    assert_eq!(tracks.len(), 3);
    assert_eq!(tracks[0].children, vec![2, 3]);
    assert_eq!(tracks[0].events.end_reason, EndReason::Division);
    assert_eq!(tracks[1].parents, vec![1]);
    assert_eq!(tracks[1].events.start_reason, StartReason::Division);
    // A divide link whose source does not branch is inconsistent.
    assert!(TrackGraph::new(&set, vec![divide(2)], with).is_err());
    // Backward in time is refused.
    let back = graph::TrackLink {
        from: 2,
        to: 1,
        dt: 1,
        elapsed_seconds: None,
        cost: 0.0,
        kind: LinkKind::Continue,
    };
    assert!(TrackGraph::new(&set, vec![back], none).is_err());
    let (ctc, labels) = export::ctc_tracks(&tracks).unwrap();
    assert_eq!(ctc, "1 0 0 0\n2 1 1 1\n3 1 1 1\n");
    assert_eq!(labels, BTreeMap::from([(1, 1), (2, 2), (3, 3)]));
}

#[test]
fn detection_set_refuses_bad_timelines() {
    let f = |t, s| Frame {
        t_index: t,
        elapsed_seconds: s,
    };
    assert!(DetectionSet::new(geometry(), vec![], vec![]).is_err());
    assert!(DetectionSet::new(geometry(), vec![f(1, None), f(1, None)], vec![]).is_err());
    assert!(DetectionSet::new(geometry(), vec![f(0, Some(0.0)), f(1, None)], vec![]).is_err());
    assert!(DetectionSet::new(geometry(), vec![f(0, Some(5.0)), f(1, Some(5.0))], vec![]).is_err());
    let stray = Detection {
        id: 1,
        t_index: 9,
        native: [0.0; 3],
        registered: [0.0; 3],
        size: 1.0,
    };
    assert!(DetectionSet::new(geometry(), vec![f(0, None)], vec![stray]).is_err());
}

#[test]
fn sparse_assignment_matches_brute_force() {
    let mut rng = Rng(23);
    for n in 1..=5usize {
        for m in 0..=5usize {
            for _ in 0..30 {
                // Each row: random candidate columns among 0..m with costs that
                // may be negative, plus its own private column m + row.
                let rows: Vec<Vec<(usize, f64)>> = (0..n)
                    .map(|row| {
                        let mut edges: Vec<(usize, f64)> = Vec::new();
                        for col in 0..m {
                            if rng.unit() < 0.6 {
                                edges.push((col, (rng.range(-5.0, 5.0) * 4.0).round() / 4.0));
                            }
                        }
                        edges.push((m + row, (rng.range(0.0, 3.0) * 4.0).round() / 4.0));
                        edges
                    })
                    .collect();
                let assignment = assign_sparse(&rows, m + n).unwrap();
                let cost_of = |row: usize, col: usize| {
                    rows[row]
                        .iter()
                        .find(|&&(c, _)| c == col)
                        .map(|&(_, cost)| cost)
                };
                let ours: f64 = assignment
                    .iter()
                    .enumerate()
                    .map(|(row, &col)| cost_of(row, col).expect("assigned along an edge"))
                    .sum();
                let mut used = assignment.clone();
                used.sort_unstable();
                used.dedup();
                assert_eq!(used.len(), n, "columns must be distinct");
                // Brute force: every injective choice of one edge per row.
                fn best(rows: &[Vec<(usize, f64)>], row: usize, used: &mut Vec<usize>) -> f64 {
                    if row == rows.len() {
                        return 0.0;
                    }
                    let mut min = f64::INFINITY;
                    for &(col, cost) in &rows[row] {
                        if used.contains(&col) {
                            continue;
                        }
                        used.push(col);
                        min = min.min(cost + best(rows, row + 1, used));
                        used.pop();
                    }
                    min
                }
                let expected = best(&rows, 0, &mut Vec::new());
                assert!(
                    (ours - expected).abs() < 1e-9,
                    "n={n} m={m} ours={ours} best={expected}"
                );
            }
        }
    }
}

#[test]
fn a_large_dense_cluster_links_quickly() {
    // 4,000 detections per frame on a 160 x 160 px patch with a 6 px gate:
    // one connected component of thousands of nodes — the shape that made the
    // dense per-component solve take minutes on a real field.
    let mut rng = Rng(5);
    let mut detections = Vec::new();
    for t in 0..2u64 {
        for i in 0..4000u64 {
            let y = rng.range(0.0, 160.0);
            let x = rng.range(0.0, 160.0);
            detections.push(Detection {
                id: t * 10_000 + i + 1,
                t_index: t,
                native: [0.0, y, x],
                registered: [0.0, y, x],
                size: 1.0,
            });
        }
    }
    let frames = (0..2)
        .map(|t| Frame {
            t_index: t,
            elapsed_seconds: None,
        })
        .collect();
    let set = DetectionSet::new(geometry(), frames, detections).unwrap();
    let candidates = generate_candidates(&set, &CandidateParams::distance(6.0)).unwrap();
    assert!(candidates.len() > 50_000);
    let started = std::time::Instant::now();
    let graph = FrameLapLinker::default().link(&set, &candidates).unwrap();
    assert!(graph.links().len() > 3000);
    assert!(
        started.elapsed().as_secs_f64() < 10.0,
        "took {:?}",
        started.elapsed()
    );
}

fn det(id: u64, t: u64, y: f64, x: f64) -> Detection {
    Detection {
        id,
        t_index: t,
        native: [0.0, y, x],
        registered: [0.0, y, x],
        size: 10.0,
    }
}

fn frames(n: u64) -> Vec<Frame> {
    (0..n)
        .map(|t| Frame {
            t_index: t,
            elapsed_seconds: None,
        })
        .collect()
}

fn segment_linker(gap: Option<u64>, split: bool, merge: bool) -> SegmentLapLinker {
    SegmentLapLinker::new(SegmentParams {
        frame: LapParams {
            alternative: AlternativeCost::Fixed(30.0),
            ..LapParams::default()
        },
        gap_closing: gap.map(|g| GapClosing {
            max_distance: 8.0,
            max_frame_gap: g,
        }),
        splitting: split.then_some(8.0),
        merging: merge.then_some(8.0),
        alternative: SegmentAlternative::Fixed {
            track_start: 30.0,
            track_end: 30.0,
            no_splitting: 30.0,
            no_merging: 30.0,
        },
    })
    .unwrap()
}

#[test]
fn a_division_becomes_a_parent_with_two_children() {
    // Mother at x=100 for t=0..2; at t=3 two daughters at x=96 and x=104,
    // which continue apart. A far-away bystander keeps the frame step honest.
    let mut d = vec![];
    for t in 0..3 {
        d.push(det(1 + t, t, 100.0, 100.0));
    }
    for t in 3..6 {
        let spread = 4.0 + 2.0 * (t - 3) as f64;
        d.push(det(10 + t, t, 100.0, 100.0 - spread));
        d.push(det(20 + t, t, 100.0, 100.0 + spread));
    }
    for t in 0..6 {
        d.push(det(100 + t, t, 300.0, 300.0));
    }
    let set = DetectionSet::new(geometry(), frames(6), d).unwrap();
    let candidates = generate_candidates(&set, &CandidateParams::distance(6.0)).unwrap();

    // Frame linking alone: the mother continues into one daughter, the other
    // daughter starts unmatched.
    let plain = FrameLapLinker::new(LapParams {
        alternative: AlternativeCost::Fixed(30.0),
        ..LapParams::default()
    })
    .unwrap()
    .link(&set, &candidates)
    .unwrap();
    assert!(plain.links().iter().all(|l| l.kind == LinkKind::Continue));

    let graph = segment_linker(None, true, false)
        .link(&set, &candidates)
        .unwrap();
    assert!(graph.capabilities().divisions);
    assert_eq!(graph.successors(3), &[13, 23]);
    let divides: Vec<_> = graph
        .links()
        .iter()
        .filter(|l| l.kind == LinkKind::Divide)
        .collect();
    assert_eq!(divides.len(), 2);
    let tracks = graph.tracks(&set, 0.0).unwrap();
    let mother = tracks.iter().find(|t| t.track_id == 1).unwrap();
    assert_eq!(mother.children, vec![13, 23]);
    assert_eq!(mother.events.end_reason, EndReason::Division);
    for daughter in [13, 23] {
        let track = tracks.iter().find(|t| t.track_id == daughter).unwrap();
        assert_eq!(track.parents, vec![1]);
        assert_eq!(track.events.start_reason, StartReason::Division);
    }
}

#[test]
fn a_one_frame_segmentation_split_is_a_split_merge_bubble() {
    // One cell at x=100; at t=2 segmentation cuts it into two pieces at
    // x=97 and x=103; at t=3 it is one cell again.
    let d = vec![
        det(1, 0, 100.0, 100.0),
        det(2, 1, 100.0, 100.0),
        det(31, 2, 100.0, 97.0),
        det(32, 2, 100.0, 103.0),
        det(4, 3, 100.0, 100.0),
        det(5, 4, 100.0, 100.0),
    ];
    let set = DetectionSet::new(geometry(), frames(5), d).unwrap();
    let candidates = generate_candidates(&set, &CandidateParams::distance(6.0)).unwrap();
    let graph = segment_linker(None, true, true)
        .link(&set, &candidates)
        .unwrap();
    assert_eq!(graph.successors(2), &[31, 32]);
    assert_eq!(graph.predecessors(4), &[31, 32]);
    let kinds: BTreeMap<(u64, u64), LinkKind> = graph
        .links()
        .iter()
        .map(|l| ((l.from, l.to), l.kind))
        .collect();
    assert_eq!(kinds[&(2, 31)], LinkKind::Divide);
    assert_eq!(kinds[&(2, 32)], LinkKind::Divide);
    assert_eq!(kinds[&(31, 4)], LinkKind::Merge);
    assert_eq!(kinds[&(32, 4)], LinkKind::Merge);
    let tracks = graph.tracks(&set, 0.0).unwrap();
    // Before the bubble, the two pieces, after the bubble: four tracks.
    assert_eq!(tracks.len(), 4);
    let after = tracks.iter().find(|t| t.track_id == 4).unwrap();
    assert_eq!(after.parents, vec![31, 32]);
    assert_eq!(after.events.start_reason, StartReason::Merge);
    // CTC cannot hold a merge, and says so.
    assert!(export::ctc_tracks(&tracks).is_err());
    // Merges disabled: the same input has no node with two predecessors.
    let no_merge = segment_linker(None, true, false)
        .link(&set, &candidates)
        .unwrap();
    assert!(no_merge
        .links()
        .iter()
        .all(|l| no_merge.predecessors(l.to).len() == 1));
}

#[test]
fn gap_closing_bridges_a_missed_detection() {
    // The cell is not detected at t=2.
    let d = vec![
        det(1, 0, 50.0, 50.0),
        det(2, 1, 50.0, 51.0),
        det(4, 3, 50.0, 53.0),
        det(5, 4, 50.0, 54.0),
    ];
    let set = DetectionSet::new(geometry(), frames(5), d).unwrap();
    let candidates = generate_candidates(&set, &CandidateParams::distance(6.0)).unwrap();
    let without = segment_linker(Some(1), false, false)
        .link(&set, &candidates)
        .unwrap();
    assert_eq!(without.tracks(&set, 0.0).unwrap().len(), 2);
    let graph = segment_linker(Some(2), false, false)
        .link(&set, &candidates)
        .unwrap();
    assert!(graph.capabilities().gaps);
    let gap = graph.links().iter().find(|l| l.from == 2).unwrap();
    assert_eq!((gap.to, gap.dt, gap.kind), (4, 2, LinkKind::Gap));
    let tracks = graph.tracks(&set, 0.0).unwrap();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].detections, vec![1, 2, 4, 5]);
}

#[test]
fn an_edge_chosen_both_as_merge_and_as_split_is_one_link() {
    // Segment A ends at t=1 (x=100). Segment B starts at t=2 next to it, and
    // a far bystander pair keeps alternatives in play. A's end is both a merge
    // row (to B's first detection) and a split point (for B's start); either
    // choice is the edge 2 -> 3.
    let d = vec![
        det(1, 0, 100.0, 100.0),
        det(2, 1, 100.0, 100.0),
        det(3, 2, 100.0, 107.0),
        det(4, 3, 100.0, 108.0),
    ];
    let set = DetectionSet::new(geometry(), frames(4), d).unwrap();
    // A 6 px frame gate leaves 2 -> 3 (7 px) unlinked by the frame step.
    let candidates = generate_candidates(&set, &CandidateParams::distance(6.0)).unwrap();
    // Alternatives of 100 make each row prefer its 49-cost link on its own:
    // the end row merges into 3, the split row splits to 3's start.
    let linker = SegmentLapLinker::new(SegmentParams {
        frame: LapParams {
            alternative: AlternativeCost::Fixed(30.0),
            ..LapParams::default()
        },
        gap_closing: None,
        splitting: Some(8.0),
        merging: Some(8.0),
        alternative: SegmentAlternative::Fixed {
            track_start: 100.0,
            track_end: 100.0,
            no_splitting: 100.0,
            no_merging: 100.0,
        },
    })
    .unwrap();
    let graph = linker.link(&set, &candidates).unwrap();
    let pairs: Vec<_> = graph.links().iter().map(|l| (l.from, l.to)).collect();
    let mut unique = pairs.clone();
    unique.dedup();
    assert_eq!(pairs, unique);
    assert!(pairs.contains(&(2, 3)), "{pairs:?}");
}
