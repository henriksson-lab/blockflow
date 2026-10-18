// SPDX-License-Identifier: MIT

//! **The series level**: everything that crosses frames.
//!
//! [`prepare`] validates one series before any expensive work — axes,
//! channels, timeline, identity range, destinations — creates (or reopens)
//! its hidden work directory and staged label stack, and computes any
//! movie-wide normalisation once. [`finalize`] runs after every frame of the
//! series has a valid artifact: it links detections into a [`TrackGraph`],
//! builds the label pyramids frame by frame, writes the tables and exports,
//! and publishes everything by rename, with the summary written last.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::collection::{ChannelMapping, SeriesRef};
use blockflow::label_pyramid::{downsample_label_plane, refresh_label_registry};
use blockflow::ome_zarr::ensure_zarr_group;
use blockflow::time_series::{LabelStack, TimeSeriesImage};
use blockflow::track::{
    export, generate_candidates, overlap_counts, Detection, DetectionSet, Frame, SeriesGeometry,
    TrackGraph,
};
use blockflow::{Error, Result};
use ngff_label_map::{ScopedId, TrackingIdentityState};
use serde_json::{json, Value};

use crate::frame::{write_atomic, FrameRow};
use crate::params::{fnv64, hash_json, Normalization, Params};
use crate::pool::parallel;
use crate::segment::{default_drift_level, estimate_drift, movie_sample, percentiles};
use crate::tables::{identity_graph, write_detections, write_links, write_tracks, TableContext};

fn durable_lineages(tracks: &[blockflow::track::Track]) -> BTreeMap<u64, u64> {
    let by_id = tracks
        .iter()
        .map(|track| (track.track_id, track))
        .collect::<BTreeMap<_, _>>();
    let mut result = BTreeMap::new();
    for &seed in by_id.keys() {
        if result.contains_key(&seed) {
            continue;
        }
        let mut component = std::collections::BTreeSet::new();
        let mut pending = vec![seed];
        while let Some(id) = pending.pop() {
            if !component.insert(id) {
                continue;
            }
            if let Some(track) = by_id.get(&id) {
                pending.extend(track.parents.iter().copied());
                pending.extend(track.children.iter().copied());
            }
        }
        // Initial output allocates a stable lineage from its oldest existing track. The explicit
        // map in identity-state.json owns this decision during later relinking.
        let lineage = *component
            .iter()
            .next()
            .expect("component contains its seed");
        result.extend(component.into_iter().map(|track| (track, lineage)));
    }
    result
}

/// One series, validated and staged.
pub struct SeriesJob {
    pub series: SeriesRef,
    pub image: TimeSeriesImage,
    pub segment_channel: usize,
    pub measure_channel: usize,
    /// One entry per stored time index.
    pub timeline: Vec<Frame>,
    pub frame_shape: [usize; 3],
    pub spacing: [f64; 3],
    /// Where `labels/`, `tables/` and `tracking/` are published.
    pub result_root: PathBuf,
    pub work: PathBuf,
    pub stack: LabelStack,
    pub label_source: String,
    pub params_hash: String,
    pub input_hash: String,
    pub movie_range: Option<(f32, f32)>,
    /// Image-based stage drift between consecutive stored frames, `[dy, dx]`
    /// in level-0 pixels, when drift correction is on. Refined with the
    /// detections and accumulated in [`finalize`].
    pub drift: Option<Vec<[f64; 2]>>,
}

impl SeriesJob {
    pub fn n_frames(&self) -> usize {
        self.timeline.len()
    }
}

/// Declare `result_root` an overlay dataset of `image`, so a viewer shows its labels and tables
/// over the source pixels without copying them: `{"overlay": {"version": 1, "source": ...}}`
/// in the group's attributes, with `source` relative to `result_root` when the two share a
/// filesystem prefix and absolute otherwise.
pub fn mark_overlay(result_root: &Path, image: &Path) -> Result<()> {
    let image = fs::canonicalize(image).map_err(Error::backend)?;
    let root = fs::canonicalize(result_root).map_err(Error::backend)?;
    let source = relative_path(&root, &image)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| image.display().to_string());
    let path = result_root.join("zarr.json");
    let mut group: Value = serde_json::from_slice(&fs::read(&path).map_err(Error::backend)?)
        .map_err(Error::backend)?;
    group["attributes"]["overlay"] = json!({"version": 1, "source": source});
    write_atomic(
        &path,
        &serde_json::to_vec_pretty(&group).map_err(Error::backend)?,
    )
}

/// `to` relative to directory `from`, both absolute; `None` if they share no prefix.
fn relative_path(from: &Path, to: &Path) -> Option<PathBuf> {
    let from: Vec<_> = from.components().collect();
    let to: Vec<_> = to.components().collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    if common <= 1 {
        return None;
    }
    let mut out = PathBuf::new();
    for _ in common..from.len() {
        out.push("..");
    }
    for component in &to[common..] {
        out.push(component.as_os_str());
    }
    Some(out)
}

/// The published destinations of one series.
pub fn destinations(result_root: &Path, layer: &str) -> Vec<PathBuf> {
    vec![
        result_root.join("labels").join(layer),
        result_root.join("labels").join(format!("{layer}-tracks")),
        result_root.join("tables").join(layer),
        result_root.join("tables").join(format!("{layer}-tracks")),
        result_root.join("tables").join(format!("{layer}-links")),
        result_root.join("tracking").join(layer),
    ]
}

pub fn result_root(series: &SeriesRef, output: Option<&Path>) -> PathBuf {
    match output {
        Some(output) => output.join(series.key.slug()),
        None => series.image.clone(),
    }
}

/// Validate and stage one series.
pub fn prepare(series: &SeriesRef, params: &Params) -> Result<SeriesJob> {
    let image = TimeSeriesImage::open(&series.image)?;
    let channels = series.channels.unwrap_or(ChannelMapping {
        segment: params.segment_channel,
        measure: params.measure_channel,
    });
    for (what, channel) in [("segment", channels.segment), ("measure", channels.measure)] {
        if channel >= image.n_channels() {
            return Err(Error::invalid(format!(
                "{what} channel {channel} is outside the {} channels of {}",
                image.n_channels(),
                series.image.display()
            )));
        }
    }
    let timeline = match &series.timeline {
        Some(timeline) => timeline.clone(),
        None => image.timeline()?,
    };
    if timeline.len() != image.n_frames() {
        return Err(Error::invalid(format!(
            "the timeline lists {} frames; {} stores {}",
            timeline.len(),
            series.image.display(),
            image.n_frames()
        )));
    }
    for pair in timeline.windows(2) {
        if pair[1].t_index <= pair[0].t_index {
            return Err(Error::invalid(
                "timeline frame indices must strictly increase",
            ));
        }
    }
    let frame_shape = image.frame_shape(0)?;
    let spacing = image.spacing(0)?;
    let voxels = frame_shape.iter().map(|&n| n as u64).product::<u64>();
    let last = timeline.last().expect("validated non-empty").t_index;
    last.checked_add(1)
        .and_then(|n| n.checked_mul(voxels))
        .ok_or_else(|| Error::invalid("detection ids for this series would overflow u64"))?;

    let result_root = result_root(series, params.output.as_deref());
    for destination in destinations(&result_root, &params.layer) {
        if destination.exists() && !params.overwrite {
            return Err(Error::invalid(format!(
                "{} exists; pass --overwrite to replace it",
                destination.display()
            )));
        }
    }
    let label_source = if params.output.is_some() {
        fs::canonicalize(&series.image)
            .map_err(Error::backend)?
            .display()
            .to_string()
    } else {
        "../../".to_owned()
    };
    let params_hash = hash_json(&params.frame_json(channels.segment, channels.measure));
    let level0 = image.levels()[0].path.clone();
    let level0_metadata =
        fs::read(series.image.join(&level0).join("zarr.json")).map_err(Error::backend)?;
    let input_hash = hash_json(&json!({
        "image": fs::canonicalize(&series.image).map_err(Error::backend)?,
        "level0": format!("{:016x}", fnv64(&level0_metadata)),
        "frame_shape": frame_shape,
        "timeline": timeline.iter().map(|f| json!([f.t_index, f.elapsed_seconds])).collect::<Vec<_>>(),
    }));

    if params.output.is_some() {
        ensure_zarr_group(&result_root)?;
        mark_overlay(&result_root, &series.image)?;
    }
    let work = result_root
        .join("labels")
        .join(format!(".{}-blockflow-work", params.layer));
    let job_record = json!({"params_hash": params_hash, "input_hash": input_hash});
    let record_path = work.join("job.json");
    let reusable = fs::read(&record_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .is_some_and(|stored| stored == job_record);
    if work.exists() && !reusable {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    fs::create_dir_all(&work).map_err(Error::backend)?;
    write_atomic(&record_path, job_record.to_string().as_bytes())?;
    let stack_root = work.join("stack");
    let stack = if stack_root.join("zarr.json").is_file() {
        let stack = LabelStack::open(&stack_root)?;
        if stack.n_frames() != image.n_frames() || stack.frame_shape(0)? != frame_shape {
            return Err(Error::invalid(format!(
                "staged label stack {} does not match the image; delete the work directory",
                stack_root.display()
            )));
        }
        stack
    } else {
        if stack_root.exists() {
            fs::remove_dir_all(&stack_root).map_err(Error::backend)?;
        }
        LabelStack::create(&stack_root, &image, [params.label_chunk; 2], &label_source)?
    };

    let movie_range = match params.normalization {
        Normalization::MoviePercentile { low, high } => {
            let path = work.join("normalization.json");
            let stored = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .filter(|v| v["params_hash"] == json!(params_hash))
                .and_then(|v| Some((v["low"].as_f64()? as f32, v["high"].as_f64()? as f32)));
            match stored {
                Some(range) => Some(range),
                None => {
                    let sample = movie_sample(
                        &image,
                        channels.segment,
                        &work.join("tmp").join("normalize"),
                    )?;
                    let range = percentiles(sample, low, high)?;
                    write_atomic(
                        &path,
                        json!({"params_hash": params_hash, "low": range.0, "high": range.1})
                            .to_string()
                            .as_bytes(),
                    )?;
                    Some(range)
                }
            }
        }
        _ => None,
    };

    let drift = if params.drift_correction {
        let level = match params.drift_level {
            Some(level) if level < image.levels().len() => level,
            Some(level) => {
                return Err(Error::invalid(format!(
                    "--drift-level {level} is outside the image's {} levels",
                    image.levels().len()
                )))
            }
            None => default_drift_level(&image)?,
        };
        let path = work.join("drift.json");
        let key = json!({"version": 2, "input_hash": input_hash, "channel": channels.segment, "level": level});
        let stored = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .filter(|v| v["key"] == key)
            .and_then(|v| serde_json::from_value::<Vec<[f64; 2]>>(v["offsets"].clone()).ok());
        Some(match stored {
            Some(offsets) => offsets,
            None => {
                let offsets = estimate_drift(
                    &image,
                    channels.segment,
                    level,
                    &work.join("tmp").join("drift"),
                )?;
                write_atomic(
                    &path,
                    json!({"key": key, "offsets": offsets})
                        .to_string()
                        .as_bytes(),
                )?;
                offsets
            }
        })
    } else {
        None
    };

    Ok(SeriesJob {
        series: series.clone(),
        image,
        segment_channel: channels.segment,
        measure_channel: channels.measure,
        timeline,
        frame_shape,
        spacing,
        result_root,
        work,
        stack,
        label_source,
        params_hash,
        input_hash,
        movie_range,
        drift,
    })
}

/// What a finished series reports to the collection index.
#[derive(Debug, Clone)]
pub struct SeriesResult {
    pub frames: usize,
    pub detections: usize,
    pub tracks: usize,
    pub links: usize,
    pub summary: Value,
}

/// Link, build pyramids, write tables and exports, publish.
pub fn finalize(
    job: &SeriesJob,
    params: &Params,
    mut rows: Vec<FrameRow>,
    frame_stats: Value,
) -> Result<SeriesResult> {
    let started = Instant::now();
    rows.sort_by_key(|r| (r.t_index, r.detection_id));
    let has_z = job.frame_shape[0] > 1;
    let geometry = SeriesGeometry::new(job.frame_shape, job.spacing)?;
    // Stage drift: refine each image-based pair shift with the detections
    // (every cell votes for its displacement), then accumulate. Registered =
    // native, minus the frame's drift, times spacing.
    let offsets: Option<Vec<[f64; 2]>> = job.drift.as_ref().map(|shifts| {
        let points = |stored: usize| -> Vec<[f64; 2]> {
            rows.iter()
                .filter(|r| r.stored_t == stored)
                .map(|r| [r.centroid[1], r.centroid[2]])
                .collect()
        };
        // Search radius: twice the gate, in pixels, but at least 20 px.
        let gate = params.max_distance.unwrap_or(f64::INFINITY);
        let pixel = job.spacing[1].min(job.spacing[2]);
        let radius = (2.0 * gate / pixel).clamp(20.0, 200.0);
        let refined: Vec<[f64; 2]> = shifts
            .iter()
            .enumerate()
            .map(|(pair, &prior)| {
                blockflow::track::drift::refine_with_points(
                    &points(pair),
                    &points(pair + 1),
                    prior,
                    radius,
                    1.0,
                )
            })
            .collect();
        blockflow::track::drift::cumulative(&refined)
    });
    let detections: Vec<Detection> = rows
        .iter()
        .map(|row| {
            let [oy, ox] = offsets
                .as_ref()
                .map_or([0.0, 0.0], |offsets| offsets[row.stored_t]);
            let corrected = [row.centroid[0], row.centroid[1] - oy, row.centroid[2] - ox];
            Detection {
                id: row.detection_id,
                t_index: row.t_index,
                native: row.centroid,
                registered: geometry.physical(corrected),
                size: row.area as f64,
            }
        })
        .collect();
    let set = DetectionSet::new(geometry, job.timeline.clone(), detections)?;

    // Candidates, with overlap features when the cost uses them.
    let mut candidates = generate_candidates(&set, &params.candidate_params()?)?;
    if params.iou_weight > 0.0 {
        let stored_of: BTreeMap<u64, usize> = job
            .timeline
            .iter()
            .enumerate()
            .map(|(stored, f)| (f.t_index, stored))
            .collect();
        for (from, to) in set.adjacent_pairs() {
            let counts = overlap_counts(
                job.stack.read_plane(0, stored_of[&from.t_index])?.view(),
                job.stack.read_plane(0, stored_of[&to.t_index])?.view(),
            )?;
            candidates.attach_overlap(from.t_index, &counts)?;
        }
    }
    let linker = params.linker()?;
    let graph = linker.link(&set, &candidates)?;
    let tracks = graph.tracks(&set, params.border_distance)?;
    let track_of = TrackGraph::track_ids(&tracks);
    let lineage_of = durable_lineages(&tracks);
    let linked = Instant::now();

    // Pyramids, frame by frame, space only.
    build_pyramid(&job.stack, &job.work.join("tmp").join("pyramid"), params)?;
    let tracks_root = job.work.join("tracks-stack");
    if tracks_root.exists() {
        fs::remove_dir_all(&tracks_root).map_err(Error::backend)?;
    }
    if params.track_labels {
        let track_stack = LabelStack::create(
            &tracks_root,
            &job.image,
            [params.label_chunk; 2],
            &job.label_source,
        )?;
        parallel((0..job.n_frames()).collect(), params.frame_jobs, |t| {
            let plane = job.stack.read_plane(0, t)?;
            let relabelled = plane.mapv(|id| {
                if id == 0 {
                    0
                } else {
                    track_of.get(&id).copied().unwrap_or(0)
                }
            });
            track_stack.write_plane(0, t, &relabelled)
        })?;
        build_pyramid(
            &track_stack,
            &job.work.join("tmp").join("track-pyramid"),
            params,
        )?;
    }
    let pyramids = Instant::now();

    // Tables.
    let tables = job.work.join("tables");
    if tables.exists() {
        fs::remove_dir_all(&tables).map_err(Error::backend)?;
    }
    fs::create_dir_all(&tables).map_err(Error::backend)?;
    let elapsed: BTreeMap<u64, Option<f64>> = job
        .timeline
        .iter()
        .map(|f| (f.t_index, f.elapsed_seconds))
        .collect();
    let context = TableContext {
        has_z,
        extent: job.frame_shape,
        spacing: job.spacing,
        frames: job.timeline.last().map_or(1, |f| f.t_index + 1),
        layer: &params.layer,
        elapsed: &elapsed,
        track_labels: params.track_labels,
    };
    let layer = &params.layer;
    write_detections(&tables.join(layer), &context, &rows, &track_of, &lineage_of)?;
    write_tracks(
        &tables.join(format!("{layer}-tracks")),
        &context,
        &set,
        &tracks,
        &lineage_of,
    )?;
    write_links(
        &tables.join(format!("{layer}-links")),
        &context,
        &set,
        &graph,
    )?;

    // Exports and summary.
    let tracking = job.work.join("tracking");
    if tracking.exists() {
        fs::remove_dir_all(&tracking).map_err(Error::backend)?;
    }
    fs::create_dir_all(tracking.join("ctc")).map_err(Error::backend)?;
    let detection_time = rows
        .iter()
        .map(|row| (row.detection_id, row.t_index as u32))
        .collect::<BTreeMap<_, _>>();
    let track_state = track_of
        .iter()
        .map(|(&detection, &track)| {
            (
                ScopedId::at_time(detection_time[&detection], detection),
                track,
            )
        })
        .collect();
    let links = graph
        .links()
        .iter()
        .map(|link| {
            (
                (
                    ScopedId::at_time(detection_time[&link.from], link.from),
                    ScopedId::at_time(detection_time[&link.to], link.to),
                ),
                link.kind.code(),
            )
        })
        .collect();
    let identity_state = TrackingIdentityState::new(track_state, lineage_of.clone(), links)
        .map_err(Error::backend)?;
    fs::write(
        tracking.join("identity-state.json"),
        serde_json::to_vec_pretty(&identity_state).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    match export::ctc_tracks(&tracks) {
        Ok((text, labels)) => {
            fs::write(tracking.join("ctc").join("res_track.txt"), text).map_err(Error::backend)?;
            let mut map = String::from("ctc_label,track_id\n");
            for (track, label) in &labels {
                map.push_str(&format!("{label},{track}\n"));
            }
            fs::write(tracking.join("ctc").join("labels.csv"), map).map_err(Error::backend)?;
        }
        Err(error) => {
            fs::write(tracking.join("ctc").join("UNAVAILABLE"), error.to_string())
                .map_err(Error::backend)?;
        }
    }
    fs::write(
        tracking.join("tracks.csv"),
        export::napari_tracks_csv(&set, &tracks)?,
    )
    .map_err(Error::backend)?;
    fs::write(
        tracking.join("graph.json"),
        serde_json::to_vec_pretty(&export::napari_graph(&tracks)).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    fs::write(
        tracking.join("detections.csv"),
        detections_csv(&rows, &track_of, &elapsed),
    )
    .map_err(Error::backend)?;

    let mut start_reasons: BTreeMap<&str, usize> = BTreeMap::new();
    let mut end_reasons: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut entered, mut left) = (0usize, 0usize);
    for track in &tracks {
        *start_reasons
            .entry(track.events.start_reason.as_str())
            .or_default() += 1;
        *end_reasons
            .entry(track.events.end_reason.as_str())
            .or_default() += 1;
        entered += usize::from(track.events.likely_entered_fov());
        left += usize::from(track.events.likely_left_fov());
    }
    let key = &job.series.key;
    let summary = json!({
        "schema": "blockflow.plate-timelapse-tracking/v1",
        "series": {
            "plate_id": key.plate_id, "acquisition_id": key.acquisition_id,
            "well_row": key.well_row, "well_column": key.well_column, "field_id": key.field_id,
            "source": job.series.source_uri,
            "image": job.series.image,
        },
        "layer": params.layer,
        "outputs": {
            "labels": format!("labels/{layer}"),
            "track_labels": params.track_labels.then(|| format!("labels/{layer}-tracks")),
            "detections": format!("tables/{layer}"),
            "tracks": format!("tables/{layer}-tracks"),
            "links": format!("tables/{layer}-links"),
            "identity_state": format!("tracking/{layer}/identity-state.json"),
            "exports": ["tracks.csv", "graph.json", "detections.csv", "ctc/res_track.txt"],
        },
        "frame_shape": job.frame_shape,
        "spacing": job.spacing,
        "timeline": job.timeline.iter().map(|f| json!({"t_index": f.t_index, "elapsed_seconds": f.elapsed_seconds})).collect::<Vec<_>>(),
        "counts": {
            "frames": job.n_frames(), "detections": rows.len(), "tracks": tracks.len(),
            "links": graph.links().len(),
        },
        "events": {
            "start_reason": start_reasons, "end_reason": end_reasons,
            "likely_entered_fov": entered, "likely_left_fov": left,
            "note": "observations only: death, true appearance and segmentation misses are not inferred",
        },
        "segmentation": {
            "parameters": params.frame_json(job.segment_channel, job.measure_channel),
            "params_hash": job.params_hash,
            "input_hash": job.input_hash,
            "movie_normalization": job.movie_range,
        },
        "tracking": {
            "linker": linker.name(),
            "capabilities": {
                "gaps": graph.capabilities().gaps,
                "divisions": graph.capabilities().divisions,
                "merges": graph.capabilities().merges,
            },
            "linker_parameters": linker.parameters(),
            "parameters": params.tracking_json(),
            "drift_image_shifts_pixels": job.drift,
            "drift_offsets_physical": offsets.as_ref().map(|offsets| offsets
                .iter()
                .map(|o| [o[0] * job.spacing[1], o[1] * job.spacing[2]])
                .collect::<Vec<_>>()),
            "coordinates": "native voxel [z, y, x] in tables; registered = (native - drift) x spacing for gating",
        },
        "codes": {
            "start_reason": {"0": "first_frame", "1": "unmatched", "2": "frame_gap", "3": "division", "4": "merge"},
            "end_reason": {"0": "last_frame", "1": "unmatched", "2": "frame_gap", "3": "division", "4": "merge"},
            "link_kind": {"0": "continue", "1": "divide", "2": "merge", "3": "gap"},
        },
        "timings_seconds": {
            "frames": frame_stats,
            "link": (linked - started).as_secs_f64(),
            "pyramids": (pyramids - linked).as_secs_f64(),
        },
    });
    write_atomic(
        &tracking.join("tracking-summary.json"),
        &serde_json::to_vec_pretty(&summary).map_err(Error::backend)?,
    )?;

    publish(job, params)?;
    Ok(SeriesResult {
        frames: job.n_frames(),
        detections: rows.len(),
        tracks: tracks.len(),
        links: graph.links().len(),
        summary,
    })
}

/// Levels `1..` of `stack`, each frame's plane from the level above it.
fn build_pyramid(stack: &LabelStack, scratch: &Path, params: &Params) -> Result<()> {
    for level in 1..stack.n_levels() {
        let blocks: Vec<[usize; 3]> = params
            .blocks
            .iter()
            .map(|&edge| {
                let [z, _, _] = stack.frame_shape(level).unwrap_or([1, 1, 1]);
                [z, edge, edge]
            })
            .collect();
        parallel((0..stack.n_frames()).collect(), params.frame_jobs, |t| {
            stack.clear_level_plane(level, t)?;
            let work = scratch.join(format!("{level}-{t}"));
            let outcome = downsample_label_plane(
                stack.plane(level - 1, t)?,
                stack.plane(level, t)?,
                &work,
                &blocks,
                params.block_workers,
            );
            let _ = fs::remove_dir_all(&work);
            outcome
        })?;
    }
    Ok(())
}

/// Move the staged outputs into place. The tracking directory, holding the
/// summary, goes last: its presence means the series is complete.
fn publish(job: &SeriesJob, params: &Params) -> Result<()> {
    let layer = &params.layer;
    let root = &job.result_root;
    for destination in destinations(root, layer) {
        if destination.exists() {
            fs::remove_dir_all(&destination).map_err(Error::backend)?;
        }
    }
    ensure_zarr_group(&root.join("labels"))?;
    ensure_zarr_group(&root.join("tables"))?;
    let moves = [
        (
            job.work.join("tables").join(layer),
            root.join("tables").join(layer),
        ),
        (
            job.work.join("tables").join(format!("{layer}-tracks")),
            root.join("tables").join(format!("{layer}-tracks")),
        ),
        (
            job.work.join("tables").join(format!("{layer}-links")),
            root.join("tables").join(format!("{layer}-links")),
        ),
        (job.work.join("stack"), root.join("labels").join(layer)),
    ];
    for (from, to) in moves {
        fs::rename(&from, &to).map_err(|e| {
            Error::backend(format!(
                "publish {} -> {}: {e}",
                from.display(),
                to.display()
            ))
        })?;
    }
    let label_metadata = root.join("labels").join(layer).join("zarr.json");
    let mut label_json: Value =
        serde_json::from_slice(&fs::read(&label_metadata).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    label_json["attributes"]["identity_graph"] =
        serde_json::to_value(identity_graph(layer)).map_err(Error::backend)?;
    write_atomic(
        &label_metadata,
        &serde_json::to_vec_pretty(&label_json).map_err(Error::backend)?,
    )?;
    if params.track_labels {
        fs::rename(
            job.work.join("tracks-stack"),
            root.join("labels").join(format!("{layer}-tracks")),
        )
        .map_err(Error::backend)?;
    }
    refresh_label_registry(&root.join("labels"))?;
    fs::create_dir_all(root.join("tracking")).map_err(Error::backend)?;
    fs::rename(job.work.join("tracking"), root.join("tracking").join(layer))
        .map_err(Error::backend)?;
    fs::remove_dir_all(&job.work).map_err(Error::backend)?;
    Ok(())
}

fn detections_csv(
    rows: &[FrameRow],
    track_of: &BTreeMap<u64, u64>,
    elapsed: &BTreeMap<u64, Option<f64>>,
) -> String {
    let opt = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    let mut text = String::from(
        "detection_id,track_id,t_index,elapsed_seconds,z,y,x,area_pixels,equivalent_diameter,\
         eccentricity,segment_mean,measure_sum,measure_mean,measure_min,measure_max\n",
    );
    for r in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.detection_id,
            track_of.get(&r.detection_id).copied().unwrap_or(0),
            r.t_index,
            opt(elapsed.get(&r.t_index).copied().flatten()),
            r.centroid[0],
            r.centroid[1],
            r.centroid[2],
            r.area,
            r.equivalent_diameter,
            opt(r.eccentricity),
            r.segment_mean,
            r.measure_sum,
            opt(r.measure_mean),
            opt(r.measure_min),
            opt(r.measure_max),
        ));
    }
    text
}
