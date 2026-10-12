// SPDX-License-Identifier: MIT

//! Score a tracking result against synthetic truth.
//!
//! Detections are matched to truth cells per frame by nearest centroid within
//! a radius, one-to-one, closest pairs first. Predicted links are read from
//! `detections.csv` (consecutive detections of one track) and, when present
//! beside it, `graph.json` (the last detection of each parent track to the
//! first of its child). A truth link (cell `c` in consecutive frames) is
//! recovered when that predicted link exists between the matched detections;
//! a predicted link is correct when both ends matched the same truth cell. A
//! truth division is recovered when the mother's last detection is linked to
//! both daughters' first detections.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use blockflow::{Error, Result};
use serde_json::{json, Value};

use crate::synth::TruthRow;

#[derive(Clone, Debug)]
struct Predicted {
    id: u64,
    track: u64,
    t: u64,
    y: f64,
    x: f64,
    measure_mean: Option<f64>,
}

fn read_detections(path: &Path) -> Result<Vec<Predicted>> {
    let text = fs::read_to_string(path).map_err(Error::backend)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| Error::invalid(format!("detections.csv has no {name} column")))
    };
    let (id, track, t, y, x, mean) = (
        col("detection_id")?,
        col("track_id")?,
        col("t_index")?,
        col("y")?,
        col("x")?,
        col("measure_mean")?,
    );
    lines
        .filter(|l| !l.is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let parse = |i: usize| -> Result<f64> {
                f[i].parse::<f64>()
                    .map_err(|_| Error::invalid(format!("bad detections line {line:?}")))
            };
            Ok(Predicted {
                id: parse(id)? as u64,
                track: parse(track)? as u64,
                t: parse(t)? as u64,
                y: parse(y)?,
                x: parse(x)?,
                measure_mean: f[mean].parse::<f64>().ok(),
            })
        })
        .collect()
}

/// Score `detections.csv` of one series against its truth rows.
pub fn score(detections_csv: &Path, truth: &[TruthRow], radius: f64) -> Result<Value> {
    let predicted = read_detections(detections_csv)?;
    let frames: BTreeSet<u64> = truth
        .iter()
        .map(|r| r.t)
        .chain(predicted.iter().map(|p| p.t))
        .collect();
    // detection id -> truth cell
    let mut matched: BTreeMap<u64, u64> = BTreeMap::new();
    let mut marker_error = Vec::new();
    for &t in &frames {
        let cells: Vec<&TruthRow> = truth.iter().filter(|r| r.t == t).collect();
        let dets: Vec<&Predicted> = predicted.iter().filter(|p| p.t == t).collect();
        let mut pairs = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            for (j, d) in dets.iter().enumerate() {
                let distance = ((c.y - d.y).powi(2) + (c.x - d.x).powi(2)).sqrt();
                if distance <= radius {
                    pairs.push((distance, i, j));
                }
            }
        }
        pairs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let (mut used_c, mut used_d) = (BTreeSet::new(), BTreeSet::new());
        for (_, i, j) in pairs {
            if used_c.contains(&i) || used_d.contains(&j) {
                continue;
            }
            used_c.insert(i);
            used_d.insert(j);
            matched.insert(dets[j].id, cells[i].cell);
            if let Some(mean) = dets[j].measure_mean {
                marker_error.push((mean - cells[i].marker).abs() / cells[i].marker.max(1.0));
            }
        }
    }
    let truth_links: BTreeSet<(u64, u64)> = {
        let present: BTreeSet<(u64, u64)> = truth.iter().map(|r| (r.cell, r.t)).collect();
        present
            .iter()
            .filter(|&&(cell, t)| present.contains(&(cell, t + 1)))
            .map(|&(cell, t)| (cell, t))
            .collect()
    };
    let mut by_track: BTreeMap<u64, Vec<&Predicted>> = BTreeMap::new();
    for p in &predicted {
        by_track.entry(p.track).or_default().push(p);
    }
    for track in by_track.values_mut() {
        track.sort_by_key(|p| p.t);
    }
    let mut links: BTreeSet<(u64, u64)> = BTreeSet::new();
    for track in by_track.values() {
        for pair in track.windows(2) {
            links.insert((pair[0].id, pair[1].id));
        }
    }
    let graph_path = detections_csv.with_file_name("graph.json");
    if let Ok(bytes) = fs::read(&graph_path) {
        let graph: Value = serde_json::from_slice(&bytes).map_err(Error::backend)?;
        for (child, parents) in graph.as_object().into_iter().flatten() {
            let child: u64 = child
                .parse()
                .map_err(|_| Error::invalid("bad graph.json key"))?;
            let Some(first) = by_track.get(&child).and_then(|t| t.first()) else {
                continue;
            };
            for parent in parents
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
            {
                if let Some(last) = by_track.get(&parent).and_then(|t| t.last()) {
                    links.insert((last.id, first.id));
                }
            }
        }
    }
    let t_of: BTreeMap<u64, u64> = predicted.iter().map(|p| (p.id, p.t)).collect();
    let predicted_links = links.len();
    let mut correct_links = 0usize;
    let mut evaluable_links = 0usize;
    let mut recovered: BTreeSet<(u64, u64)> = BTreeSet::new();
    for &(a_id, b_id) in &links {
        if let (Some(&a), Some(&b)) = (matched.get(&a_id), matched.get(&b_id)) {
            evaluable_links += 1;
            if a == b && t_of[&b_id] == t_of[&a_id] + 1 {
                correct_links += 1;
                recovered.insert((a, t_of[&a_id]));
            }
        }
    }
    // Divisions: mother's last row and the daughters' first rows.
    let detection_of: BTreeMap<(u64, u64), u64> =
        matched.iter().map(|(&d, &c)| ((c, t_of[&d]), d)).collect();
    let mut daughters: BTreeMap<u64, Vec<&TruthRow>> = BTreeMap::new();
    for row in truth.iter().filter(|r| r.parent != 0) {
        daughters.entry(row.parent).or_default().push(row);
    }
    let (mut divisions, mut divisions_found) = (0usize, 0usize);
    for (mother, kids) in &daughters {
        let t = kids[0].t - 1;
        if !truth.iter().any(|r| r.cell == *mother && r.t == t) {
            continue;
        }
        divisions += 1;
        let found = detection_of.get(&(*mother, t)).is_some_and(|&m| {
            kids.iter().all(|k| {
                detection_of
                    .get(&(k.cell, k.t))
                    .is_some_and(|&d| links.contains(&(m, d)))
            })
        });
        divisions_found += usize::from(found);
    }
    let ratio = |a: usize, b: usize| if b == 0 { 1.0 } else { a as f64 / b as f64 };
    marker_error.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Ok(json!({
        "truth_detections": truth.len(),
        "predicted_detections": predicted.len(),
        "matched_detections": matched.len(),
        "detection_recall": ratio(matched.len(), truth.len()),
        "detection_precision": ratio(matched.len(), predicted.len()),
        "truth_links": truth_links.len(),
        "predicted_links": predicted_links,
        "link_recall": ratio(recovered.len(), truth_links.len()),
        "link_precision": ratio(correct_links, evaluable_links),
        "tracks": by_track.len(),
        "truth_divisions": divisions,
        "division_recall": ratio(divisions_found, divisions),
        "truth_cells": truth.iter().map(|r| r.cell).collect::<BTreeSet<_>>().len(),
        "median_relative_marker_error": marker_error.get(marker_error.len() / 2).copied(),
    }))
}
