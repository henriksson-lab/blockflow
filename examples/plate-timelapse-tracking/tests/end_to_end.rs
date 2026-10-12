// SPDX-License-Identifier: MIT
//
// The plate pipeline end to end on synthetic plates with known answers:
// correctness against truth, the published layout, resume, the stale-chunk
// rule, invariance to blocks and concurrency, and timeline gaps.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::collection::SeriesKey;
use blockflow::object_table::TableReader;
use blockflow::time_series::LabelStack;
use ndarray::Array3;
use plate_timelapse_tracking::params::{Normalization, Params, Segmenter, Stage};
use plate_timelapse_tracking::run::{run, Input};
use plate_timelapse_tracking::score::score;
use plate_timelapse_tracking::synth::{read_truth, write_image, write_plate, SynthSpec};
use serde_json::json;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "plate-timelapse-tracking-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn params() -> Params {
    Params {
        segmenter: Segmenter::Threshold { threshold: 100.0 },
        normalization: Normalization::None,
        segment_channel: 0,
        measure_channel: 1,
        layer: "cells".into(),
        halo: 16,
        blocks: vec![64, 256],
        label_chunk: 64,
        block_workers: 2,
        frame_jobs: 3,
        series_jobs: 2,
        max_distance: Some(6.0),
        max_speed: None,
        iou_weight: 0.0,
        size_weight: 0.0,
        alternative_cost: None,
        gap_closing: None,
        max_gap: 2,
        splitting: None,
        merging: None,
        segment_alternative_cost: None,
        border_distance: 10.0,
        drift_correction: false,
        drift_level: None,
        track_labels: true,
        overwrite: false,
        verify_planes: true,
        output: None,
        stop_after: None,
    }
}

fn spec() -> SynthSpec {
    SynthSpec {
        frames: 7,
        size: [150, 150],
        cells: 10,
        ..SynthSpec::default()
    }
}

fn field(plate: &Path, column: &str) -> PathBuf {
    plate.join("A").join(column).join("0")
}

fn planes(root: &Path, frames: usize) -> Vec<Array3<u64>> {
    let stack = LabelStack::open(root).unwrap();
    (0..frames)
        .map(|t| stack.read_plane(0, t).unwrap())
        .collect()
}

#[test]
fn a_plate_is_segmented_measured_and_tracked_exactly() {
    let dir = scratch("exact");
    let plate = dir.join("plate.zarr");
    let truth = write_plate(&plate, &spec()).unwrap();
    let report = run(&Input::Plate(plate.clone()), &params()).unwrap();
    assert_eq!(report.failed(), 0, "{:?}", report.series);
    assert_eq!(report.frames_ran, 14);
    for (column, truth) in ["1", "2"].iter().zip(&truth) {
        let tracking = field(&plate, column).join("tracking/cells");
        let scored = score(
            &tracking.join("detections.csv"),
            &read_truth(truth).unwrap(),
            4.0,
        )
        .unwrap();
        assert_eq!(scored["detection_recall"], json!(1.0), "{scored}");
        assert_eq!(scored["detection_precision"], json!(1.0), "{scored}");
        assert_eq!(scored["link_recall"], json!(1.0), "{scored}");
        assert_eq!(scored["link_precision"], json!(1.0), "{scored}");
        assert_eq!(
            scored["median_relative_marker_error"],
            json!(0.0),
            "{scored}"
        );

        // Published layout.
        let root = field(&plate, column);
        let detections = TableReader::open(root.join("tables/cells")).unwrap();
        let rows = detections.spec().row_count;
        assert!(rows > 0);
        // The index keeps every frame at every occupancy level.
        for shape in detections.occupancy_shapes() {
            assert_eq!(shape[0], 7);
        }
        let ids = detections.read_u64("detection_id", 0..rows).unwrap();
        let tracks = detections.read_u64("track_id", 0..rows).unwrap();
        let t = detections.read_u64("t_index", 0..rows).unwrap();
        let stack = LabelStack::open(root.join("labels/cells")).unwrap();
        let track_stack = LabelStack::open(root.join("labels/cells-tracks")).unwrap();
        for frame in 0..7 {
            let labels = stack.read_plane(0, frame).unwrap();
            let by_track = track_stack.read_plane(0, frame).unwrap();
            for ((&id, &track), &tt) in ids.iter().zip(&tracks).zip(&t) {
                if tt as usize != frame {
                    continue;
                }
                // Every detection's id is in its own frame's plane, and the
                // track-label plane shows its track there.
                let pixels: Vec<_> = labels
                    .iter()
                    .zip(by_track.iter())
                    .filter(|(&l, _)| l == id)
                    .collect();
                assert!(!pixels.is_empty());
                assert!(pixels.iter().all(|(_, &k)| k == track));
            }
            // A pyramid level exists for every frame.
            assert_eq!(stack.read_plane(1, frame).unwrap().shape(), [1, 75, 75]);
        }
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(tracking.join("tracking-summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["tracking"]["linker"], json!("frame-lap"));
        assert_eq!(
            summary["tracking"]["capabilities"]["divisions"],
            json!(false)
        );
        assert!(tracking.join("ctc/res_track.txt").is_file());
        assert!(!root.join("labels/.cells-blockflow-work").exists());
    }
    let index = fs::read_to_string(plate.join("tracking/cells-index.csv")).unwrap();
    assert_eq!(index.lines().count(), 3);
    assert!(index.lines().skip(1).all(|l| l.ends_with(",ok")));

    // A second run refuses to overwrite without --overwrite.
    let refused = run(&Input::Plate(plate.clone()), &params()).unwrap();
    assert_eq!(refused.failed(), 2);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn resume_skips_completed_frames_and_reruns_damaged_ones() {
    let dir = scratch("resume");
    let clean_plate = dir.join("clean.zarr");
    let plate = dir.join("plate.zarr");
    write_plate(&clean_plate, &spec()).unwrap();
    write_plate(&plate, &spec()).unwrap();
    run(&Input::Plate(clean_plate.clone()), &params()).unwrap();

    // Interrupted after the frame stage.
    let mut stop = params();
    stop.stop_after = Some(Stage::Frames);
    let first = run(&Input::Plate(plate.clone()), &stop).unwrap();
    assert_eq!(first.frames_ran, 14);
    let work = field(&plate, "1").join("labels/.cells-blockflow-work");
    let stack = LabelStack::open(work.join("stack")).unwrap();
    // Completed frames have omitted background chunks: presence of chunks is
    // not what makes them complete.
    let stored: usize = fs::read_dir(work.join("stack/0/c/2"))
        .unwrap()
        .map(|row| fs::read_dir(row.unwrap().path()).unwrap().count())
        .sum();
    assert!(
        stored < 9,
        "expected omitted fill-value chunks of 9, found {stored}"
    );

    // Frame 3 is damaged: its manifest is gone and its plane holds stale ids
    // from "an earlier attempt". (Each frame plan writes its whole plane, so
    // the rerun would overwrite them anyway; clearing the plane first is the
    // guard for any future plan that does not — `tests/time_series.rs` tests
    // the clear itself.)
    fs::remove_file(work.join("frames/000003/manifest.json")).unwrap();
    let mut junk = stack.read_plane(0, 3).unwrap();
    for y in 0..64 {
        for x in 0..64 {
            if junk[[0, y, x]] == 0 {
                junk[[0, y, x]] = 999_999;
            }
        }
    }
    stack.write_plane(0, 3, &junk).unwrap();

    // Frame 5 is damaged differently: its rows no longer match the manifest.
    let rows = work.join("frames/000005/rows.json");
    let mut bytes = fs::read(&rows).unwrap();
    bytes.push(b' ');
    fs::write(&rows, bytes).unwrap();

    let second = run(&Input::Plate(plate.clone()), &params()).unwrap();
    assert_eq!(second.failed(), 0, "{:?}", second.series);
    assert_eq!(second.frames_ran, 2);
    assert_eq!(second.frames_skipped, 12);
    for column in ["1", "2"] {
        assert_eq!(
            planes(&field(&plate, column).join("labels/cells"), 7),
            planes(&field(&clean_plate, column).join("labels/cells"), 7)
        );
        assert_eq!(
            fs::read_to_string(field(&plate, column).join("tracking/cells/detections.csv"))
                .unwrap(),
            fs::read_to_string(field(&clean_plate, column).join("tracking/cells/detections.csv"))
                .unwrap()
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn results_do_not_depend_on_blocks_or_concurrency() {
    let dir = scratch("invariance");
    let mut outputs = Vec::new();
    for (index, (blocks, frame_jobs, block_workers)) in
        [(vec![64], 1, 1), (vec![256], 4, 3), (vec![128, 256], 2, 2)]
            .into_iter()
            .enumerate()
    {
        let plate = dir.join(format!("plate{index}.zarr"));
        write_plate(&plate, &spec()).unwrap();
        let mut p = params();
        p.blocks = blocks;
        p.frame_jobs = frame_jobs;
        p.block_workers = block_workers;
        let report = run(&Input::Plate(plate.clone()), &p).unwrap();
        assert_eq!(report.failed(), 0, "{:?}", report.series);
        outputs.push((
            planes(&field(&plate, "1").join("labels/cells"), 7),
            fs::read_to_string(field(&plate, "1").join("tracking/cells/detections.csv")).unwrap(),
            fs::read_to_string(field(&plate, "1").join("tracking/cells/tracks.csv")).unwrap(),
        ));
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_manifest_timeline_gap_splits_tracks_and_results_go_to_an_output_root() {
    let dir = scratch("manifest");
    let image = dir.join("images").join("well-B3.zarr");
    write_image(&image, &spec(), 99).unwrap();
    let source_listing = fs::read_dir(&image).unwrap().count();
    // Stored frames 0..7 are acquisition indices 0,1,2,3,5,6,7: index 4 was
    // never acquired.
    let timeline: Vec<_> = [0, 1, 2, 3, 5, 6, 7]
        .iter()
        .map(|&t| json!({"t_index": t, "elapsed_seconds": t as f64 * 300.0}))
        .collect();
    let manifest = dir.join("collection.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "version": 1,
            "plate_id": "P7",
            "series": [{
                "path": "images/well-B3.zarr", "well_row": "B", "well_column": "3", "field": "0",
                "channels": {"segment": 0, "measure": 1}, "timeline": timeline
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let mut p = params();
    p.output = Some(dir.join("results"));
    let report = run(&Input::Manifest(manifest), &p).unwrap();
    assert_eq!(report.failed(), 0, "{:?}", report.series);
    // The source image is untouched.
    assert_eq!(fs::read_dir(&image).unwrap().count(), source_listing);
    let result = dir.join("results").join("B-3-f0");
    let summary: serde_json::Value = serde_json::from_slice(
        &fs::read(result.join("tracking/cells/tracking-summary.json")).unwrap(),
    )
    .unwrap();
    assert!(
        summary["events"]["end_reason"]["frame_gap"]
            .as_u64()
            .unwrap()
            > 0,
        "{summary}"
    );
    assert!(
        summary["events"]["start_reason"]["frame_gap"]
            .as_u64()
            .unwrap()
            > 0,
        "{summary}"
    );
    let detections = TableReader::open(result.join("tables/cells")).unwrap();
    let rows = detections.spec().row_count;
    let t = detections.read_u64("t_index", 0..rows).unwrap();
    assert!(!t.contains(&4) && t.contains(&7));
    let links = TableReader::open(result.join("tables/cells-links")).unwrap();
    let from = links
        .read_u64("from_id", 0..links.spec().row_count)
        .unwrap();
    let to = links.read_u64("to_id", 0..links.spec().row_count).unwrap();
    let frame_of = |id: u64| (id - 1) / (150 * 150);
    assert!(from
        .iter()
        .zip(&to)
        .all(|(&a, &b)| frame_of(b) == frame_of(a) + 1));
    assert!(dir.join("results/tracking/cells-index.json").is_file());
    // The result root is an overlay of the source image, and the tables declare their roles.
    let group: serde_json::Value =
        serde_json::from_slice(&fs::read(result.join("zarr.json")).unwrap()).unwrap();
    assert_eq!(group["attributes"]["overlay"]["version"], json!(1));
    let source = result.join(group["attributes"]["overlay"]["source"].as_str().unwrap());
    assert_eq!(
        source.canonicalize().unwrap(),
        image.canonicalize().unwrap()
    );
    assert_eq!(
        detections.spec().extensions["tracking"]["role"],
        json!("detections")
    );
    assert_eq!(
        detections.spec().extensions["tracking"]["tracks"],
        json!("../cells-tracks")
    );
    assert_eq!(links.spec().extensions["tracking"]["role"], json!("links"));
    let tracks = TableReader::open(result.join("tables/cells-tracks")).unwrap();
    assert_eq!(
        tracks.spec().extensions["tracking"]["role"],
        json!("tracks")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_single_image_runs_without_a_collection() {
    let dir = scratch("single");
    let image = dir.join("field.zarr");
    write_image(&image, &spec(), 5).unwrap();
    let mut p = params();
    p.normalization = Normalization::MoviePercentile {
        low: 1.0,
        high: 99.8,
    };
    p.segmenter = Segmenter::Threshold { threshold: 0.5 };
    let key = SeriesKey {
        plate_id: "single".into(),
        acquisition_id: None,
        well_row: "C".into(),
        well_column: "4".into(),
        field_id: "0".into(),
    };
    let report = run(
        &Input::Image {
            path: image.clone(),
            key,
        },
        &p,
    )
    .unwrap();
    assert_eq!(report.failed(), 0, "{:?}", report.series);
    assert!(report.index.is_none());
    assert!(report.series[0].tracks > 0);
    let summary: serde_json::Value = serde_json::from_slice(
        &fs::read(image.join("tracking/cells/tracking-summary.json")).unwrap(),
    )
    .unwrap();
    assert!(summary["segmentation"]["movie_normalization"].is_array());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn drift_correction_recovers_links_a_global_stage_shift_breaks() {
    let dir = scratch("drift");
    // 12 px of stage drift per frame against a 6 µm (~9 px) gate.
    let spec = SynthSpec {
        frames: 6,
        size: [200, 200],
        cells: 14,
        drift: [-3.0, 12.0],
        ..SynthSpec::default()
    };
    let mut results = Vec::new();
    for correct in [false, true] {
        let plate = dir.join(format!("plate-{correct}.zarr"));
        let truth = write_plate(&plate, &spec).unwrap();
        let mut p = params();
        p.drift_correction = correct;
        p.drift_level = Some(0);
        let report = run(&Input::Plate(plate.clone()), &p).unwrap();
        assert_eq!(report.failed(), 0, "{:?}", report.series);
        let scored = score(
            &field(&plate, "1").join("tracking/cells/detections.csv"),
            &read_truth(&truth[0]).unwrap(),
            4.0,
        )
        .unwrap();
        results.push(scored["link_recall"].as_f64().unwrap());
        if correct {
            assert!(
                scored["link_precision"].as_f64().unwrap() >= 0.99,
                "{scored}"
            );
            let summary: serde_json::Value = serde_json::from_slice(
                &fs::read(field(&plate, "1").join("tracking/cells/tracking-summary.json")).unwrap(),
            )
            .unwrap();
            let offsets = summary["tracking"]["drift_offsets_physical"]
                .as_array()
                .unwrap();
            let last = offsets.last().unwrap();
            // 5 frames x 12 px x 0.65 µm = 39 µm in x.
            assert!((last[1].as_f64().unwrap() - 39.0).abs() < 1.5, "{last}");
        }
    }
    assert!(
        results[0] < 0.2,
        "without correction links should mostly fail: {results:?}"
    );
    assert!(
        results[1] >= 0.99,
        "with correction links should be recovered: {results:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn segment_linking_publishes_divisions_as_lineage() {
    let dir = scratch("lineage");
    let plate = dir.join("plate.zarr");
    let spec = SynthSpec {
        frames: 12,
        size: [200, 200],
        cells: 16,
        divide: 0.05,
        seed: 21,
        ..SynthSpec::default()
    };
    let truth = write_plate(&plate, &spec).unwrap();
    let mut p = params();
    p.max_distance = Some(8.0);
    p.splitting = Some(10.0);
    p.merging = Some(10.0);
    p.gap_closing = Some(10.0);
    let report = run(&Input::Plate(plate.clone()), &p).unwrap();
    assert_eq!(report.failed(), 0, "{:?}", report.series);
    let root = field(&plate, "1");
    let tracking = root.join("tracking/cells");
    let truth = read_truth(&truth[0]).unwrap();
    assert!(
        truth.iter().any(|r| r.parent != 0),
        "fixture must contain divisions"
    );
    let scored = score(&tracking.join("detections.csv"), &truth, 4.0).unwrap();
    assert!(
        scored["division_recall"].as_f64().unwrap() >= 0.5,
        "{scored}"
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&fs::read(tracking.join("tracking-summary.json")).unwrap()).unwrap();
    assert_eq!(summary["tracking"]["linker"], json!("segment-lap"));
    assert_eq!(
        summary["tracking"]["capabilities"]["divisions"],
        json!(true)
    );
    assert!(
        summary["events"]["end_reason"]["division"]
            .as_u64()
            .unwrap()
            > 0,
        "{summary}"
    );
    // Lineage is in the tables: tracks with a parent, links of kind divide.
    let tracks = TableReader::open(root.join("tables/cells-tracks")).unwrap();
    let n = tracks.spec().row_count;
    assert!(tracks
        .read_u64("n_parents", 0..n)
        .unwrap()
        .iter()
        .any(|&p| p > 0));
    let links = TableReader::open(root.join("tables/cells-links")).unwrap();
    let kinds = links.read_u8("kind", 0..links.spec().row_count).unwrap();
    assert!(kinds.contains(&1), "a divide link");
    let graph: serde_json::Value =
        serde_json::from_slice(&fs::read(tracking.join("graph.json")).unwrap()).unwrap();
    assert!(!graph.as_object().unwrap().is_empty());
    let _ = fs::remove_dir_all(&dir);
}
