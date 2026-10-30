// SPDX-License-Identifier: MIT

//! Spatial correspondence for alternative segmentations, without assuming ID equality.
use blockflow::track::{
    matching::{match_overlaps, FrameMatching},
    OverlapCounts,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, fs, io::Write};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    minimum_iou: f64,
    frames: Vec<Frame>,
    provenance: serde_json::Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    t_index: u64,
    reference_sizes: Vec<Size>,
    predicted_sizes: Vec<Size>,
    shared_pixels: Vec<Shared>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Size {
    id: u64,
    pixels: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Shared {
    reference: u64,
    predicted: u64,
    pixels: u64,
}
#[derive(Serialize)]
struct OutputFrame {
    t_index: u64,
    matching: FrameMatching,
}

fn run() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("match-overlaps requires --release".into());
    }
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: match-overlaps COUNTS.json NEW_MATCHING.json".into());
    }
    let input: Input = serde_json::from_slice(&fs::read(&args[0])?)?;
    if input.schema != "blockflow.frame-overlaps/v1" {
        return Err("unsupported overlap input schema".into());
    }
    let mut frames = BTreeMap::new();
    for frame in input.frames {
        let mut counts = OverlapCounts::default();
        for row in frame.reference_sizes {
            if counts.sizes_a.insert(row.id, row.pixels).is_some() {
                return Err("duplicate reference size row".into());
            }
        }
        for row in frame.predicted_sizes {
            if counts.sizes_b.insert(row.id, row.pixels).is_some() {
                return Err("duplicate predicted size row".into());
            }
        }
        for row in frame.shared_pixels {
            if counts
                .shared
                .insert((row.reference, row.predicted), row.pixels)
                .is_some()
            {
                return Err("duplicate intersection row".into());
            }
        }
        let matching = match_overlaps(&counts, input.minimum_iou)?;
        if frames
            .insert(
                frame.t_index,
                OutputFrame {
                    t_index: frame.t_index,
                    matching,
                },
            )
            .is_some()
        {
            return Err("duplicate frame index".into());
        }
    }
    let result = json!({"schema":"blockflow.spatial-matching/v1", "minimum_iou":input.minimum_iou,
        "provenance":input.provenance, "frames": frames.into_values().collect::<Vec<_>>(),
        "interpretation":"spatial correspondence only; not accuracy without an appropriate independent reference"});
    let bytes = serde_json::to_vec_pretty(&result)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("match-overlaps: {error}");
        std::process::exit(1);
    }
}
