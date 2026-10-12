// SPDX-License-Identifier: MIT

//! Orchestration: enumerate series, validate them all, run every frame of
//! every series on one bounded pool, finalise each series, write the
//! collection index.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use blockflow::collection::{open_manifest, open_plate, SeriesKey, SeriesRef};
use blockflow::{Error, Result};
use serde_json::{json, Value};

use crate::frame::{run_frame, FrameOutcome, FrameRow};
use crate::params::{hash_json, Params, Stage};
use crate::pool::run as pool_run;
use crate::segment::load_backend;
use crate::series::{finalize, prepare, SeriesJob};

/// What to process.
#[derive(Clone, Debug)]
pub enum Input {
    /// One field image, with an explicit key for the outputs.
    Image { path: PathBuf, key: SeriesKey },
    /// An OME-NGFF HCS plate.
    Plate(PathBuf),
    /// A collection manifest for separately stored images.
    Manifest(PathBuf),
}

#[derive(Clone, Debug)]
pub struct SeriesStatus {
    pub key: SeriesKey,
    pub source_uri: String,
    pub result_path: Option<PathBuf>,
    pub frames: usize,
    pub detections: usize,
    pub tracks: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RunReport {
    pub series: Vec<SeriesStatus>,
    pub frames_ran: usize,
    pub frames_skipped: usize,
    pub index: Option<PathBuf>,
}

impl RunReport {
    pub fn failed(&self) -> usize {
        self.series.iter().filter(|s| s.error.is_some()).count()
    }
}

pub fn run(input: &Input, params: &Params) -> Result<RunReport> {
    params.validate()?;
    let (series, collection_root): (Vec<SeriesRef>, Option<PathBuf>) = match input {
        Input::Image { path, key } => (
            vec![SeriesRef {
                key: key.clone(),
                image: path.clone(),
                source_uri: path.display().to_string(),
                well_indices: None,
                channels: None,
                timeline: None,
            }],
            None,
        ),
        Input::Plate(root) => (open_plate(root)?, Some(root.clone())),
        Input::Manifest(path) => (
            open_manifest(path)?,
            Some(path.parent().unwrap_or(Path::new(".")).to_path_buf()),
        ),
    };
    if series.is_empty() {
        return Err(Error::invalid("the collection contains no series"));
    }

    // Validate and stage every series before any expensive work.
    let mut statuses: Vec<SeriesStatus> = series
        .iter()
        .map(|s| SeriesStatus {
            key: s.key.clone(),
            source_uri: s.source_uri.clone(),
            result_path: None,
            frames: 0,
            detections: 0,
            tracks: 0,
            error: None,
        })
        .collect();
    let mut jobs: Vec<Option<SeriesJob>> = Vec::with_capacity(series.len());
    for (status, s) in statuses.iter_mut().zip(&series) {
        match prepare(s, params) {
            Ok(job) => {
                status.frames = job.n_frames();
                status.result_path = Some(job.result_root.clone());
                jobs.push(Some(job));
            }
            Err(error) => {
                status.error = Some(format!("prepare: {error}"));
                jobs.push(None);
            }
        }
    }

    // Every frame of every series, on one bounded pool.
    let backend = load_backend(&params.segmenter)?;
    let tasks: Vec<(usize, usize)> = jobs
        .iter()
        .enumerate()
        .filter_map(|(j, job)| job.as_ref().map(|job| (j, job.n_frames())))
        .flat_map(|(j, n)| (0..n).map(move |t| (j, t)))
        .collect();
    let frame_seconds: Mutex<Vec<f64>> = Mutex::new(vec![0.0; jobs.len()]);
    let outcomes = pool_run(tasks.clone(), params.frame_jobs, |(j, t)| {
        let started = Instant::now();
        let job = jobs[j].as_ref().expect("task for a prepared series");
        let outcome = run_frame(job, t, &backend, params);
        frame_seconds.lock().unwrap_or_else(|p| p.into_inner())[j] +=
            started.elapsed().as_secs_f64();
        outcome
    });
    let frame_seconds = frame_seconds
        .into_inner()
        .unwrap_or_else(|p| p.into_inner());
    let mut rows: Vec<Vec<FrameRow>> = vec![Vec::new(); jobs.len()];
    let mut report = RunReport::default();
    for ((j, t), outcome) in tasks.into_iter().zip(outcomes) {
        match outcome {
            Ok(outcome) => {
                match &outcome {
                    FrameOutcome::Ran(_) => report.frames_ran += 1,
                    FrameOutcome::Skipped(_) => report.frames_skipped += 1,
                }
                rows[j].extend(outcome.rows().iter().cloned());
            }
            Err(error) => {
                if statuses[j].error.is_none() {
                    statuses[j].error = Some(format!("frame {t}: {error}"));
                }
            }
        }
    }
    if params.stop_after == Some(Stage::Frames) {
        report.series = statuses;
        return Ok(report);
    }

    // Finalise each series independently.
    let finals: Vec<usize> = (0..jobs.len())
        .filter(|&j| jobs[j].is_some() && statuses[j].error.is_none())
        .collect();
    let rows = Mutex::new(rows);
    let results = pool_run(finals.clone(), params.series_jobs, |j| {
        let job = jobs[j].as_ref().expect("prepared");
        let series_rows = std::mem::take(&mut rows.lock().unwrap_or_else(|p| p.into_inner())[j]);
        finalize(job, params, series_rows, json!(frame_seconds[j]))
    });
    for (j, result) in finals.into_iter().zip(results) {
        match result {
            Ok(result) => {
                statuses[j].detections = result.detections;
                statuses[j].tracks = result.tracks;
            }
            Err(error) => statuses[j].error = Some(format!("finalize: {error}")),
        }
    }

    if let Some(root) = collection_root {
        let index_root = params.output.clone().unwrap_or(root);
        report.index = Some(write_index(&index_root, params, &statuses)?);
    }
    report.series = statuses;
    Ok(report)
}

/// `tracking/<layer>-index.{json,csv}` at the collection root: one row per
/// series, joinable to its per-field tables through the series key.
fn write_index(root: &Path, params: &Params, statuses: &[SeriesStatus]) -> Result<PathBuf> {
    let dir = root.join("tracking");
    fs::create_dir_all(&dir).map_err(Error::backend)?;
    let params_hash = hash_json(&json!({
        "frame": params.frame_json(params.segment_channel, params.measure_channel),
        "tracking": params.tracking_json(),
    }));
    let relative = |path: &Option<PathBuf>| -> Value {
        path.as_ref()
            .map(|p| {
                Value::String(
                    p.strip_prefix(root)
                        .map(|r| r.display().to_string())
                        .unwrap_or_else(|_| p.display().to_string()),
                )
            })
            .unwrap_or(Value::Null)
    };
    let rows: Vec<Value> = statuses
        .iter()
        .map(|s| {
            json!({
                "plate_id": s.key.plate_id,
                "acquisition_id": s.key.acquisition_id,
                "well_row": s.key.well_row,
                "well_column": s.key.well_column,
                "field_id": s.key.field_id,
                "source_uri": s.source_uri,
                "result_path": relative(&s.result_path),
                "n_frames": s.frames,
                "n_tracks": s.tracks,
                "n_detections": s.detections,
                "params_hash": params_hash,
                "status": if s.error.is_some() { "failed" } else { "ok" },
                "error": s.error,
            })
        })
        .collect();
    let json_path = dir.join(format!("{}-index.json", params.layer));
    fs::write(
        &json_path,
        serde_json::to_vec_pretty(&json!({"layer": params.layer, "series": rows}))
            .map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    let mut csv = String::from(
        "plate_id,acquisition_id,well_row,well_column,field_id,source_uri,result_path,n_frames,n_tracks,n_detections,params_hash,status\n",
    );
    for row in &rows {
        let cell = |key: &str| match &row[key] {
            Value::Null => String::new(),
            Value::String(s) => s.replace(',', ";"),
            other => other.to_string(),
        };
        csv.push_str(
            &[
                "plate_id",
                "acquisition_id",
                "well_row",
                "well_column",
                "field_id",
                "source_uri",
                "result_path",
                "n_frames",
                "n_tracks",
                "n_detections",
                "params_hash",
                "status",
            ]
            .iter()
            .map(|k| cell(k))
            .collect::<Vec<_>>()
            .join(","),
        );
        csv.push('\n');
    }
    fs::write(dir.join(format!("{}-index.csv", params.layer)), csv).map_err(Error::backend)?;
    Ok(json_path)
}
