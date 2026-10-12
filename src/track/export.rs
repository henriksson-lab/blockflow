// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Interchange formats for a track graph.
//!
//! * **Cell Tracking Challenge** `res_track.txt` — `L B E P` per track — so
//!   the official TRA/DET tools can score a result. CTC masks carry one label
//!   per *track* across frames; [`ctc_tracks`] returns the mapping a caller
//!   uses to relabel its planes. CTC has one parent per track, so merges are
//!   refused rather than written wrongly.
//! * **napari-style CSV** — `track_id,t,z,y,x` — plus the parent map napari's
//!   `Tracks` layer takes as `graph`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Map, Value};

use super::{DetectionSet, Track};
use crate::error::{Error, Result};

/// CTC `res_track.txt` and the `track_id -> CTC label` map. Labels are
/// `1..=N` in `track_id` order.
pub fn ctc_tracks(tracks: &[Track]) -> Result<(String, BTreeMap<u64, u64>)> {
    let labels: BTreeMap<u64, u64> = tracks
        .iter()
        .enumerate()
        .map(|(index, track)| (track.track_id, index as u64 + 1))
        .collect();
    let mut text = String::new();
    for track in tracks {
        let parent = match track.parents.as_slice() {
            [] => 0,
            [parent] => labels[parent],
            _ => {
                return Err(Error::invalid(format!(
                    "track {} has {} parents; the CTC format allows one",
                    track.track_id,
                    track.parents.len()
                )))
            }
        };
        writeln!(
            text,
            "{} {} {} {}",
            labels[&track.track_id], track.t_start, track.t_end, parent
        )
        .expect("writing to a String");
    }
    Ok((text, labels))
}

/// `track_id,t,z,y,x` rows in `(track_id, t)` order, native coordinates.
pub fn napari_tracks_csv(detections: &DetectionSet, tracks: &[Track]) -> Result<String> {
    let mut text = String::from("track_id,t,z,y,x\n");
    for track in tracks {
        for id in &track.detections {
            let d = detections
                .get(*id)
                .ok_or_else(|| Error::invalid(format!("track names unknown detection {id}")))?;
            writeln!(
                text,
                "{},{},{},{},{}",
                track.track_id, d.t_index, d.native[0], d.native[1], d.native[2]
            )
            .expect("writing to a String");
        }
    }
    Ok(text)
}

/// napari's `graph`: `{child track_id: [parent track_ids]}` for tracks that
/// have parents.
pub fn napari_graph(tracks: &[Track]) -> Value {
    let mut graph = Map::new();
    for track in tracks.iter().filter(|t| !t.parents.is_empty()) {
        graph.insert(
            track.track_id.to_string(),
            Value::from(track.parents.clone()),
        );
    }
    Value::Object(graph)
}
