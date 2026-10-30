// SPDX-License-Identifier: MIT
//! Apply caller-scored endpoint joins to an existing fixed-mask graph.
use blockflow::track::{
    refine::join_endpoints, Candidate, CandidateSet, Detection, DetectionSet, Frame, LinkKind,
    LinkerCapabilities, ScoredCandidates, SeriesGeometry, TrackGraph, TrackLink,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Input {
    schema: String,
    extent: [usize; 3],
    spacing: [f64; 3],
    frames: Vec<InputFrame>,
    detections: Vec<InputDetection>,
}
#[derive(Deserialize)]
struct InputFrame {
    t_index: u64,
    elapsed_seconds: Option<f64>,
}
#[derive(Deserialize)]
struct InputDetection {
    id: u64,
    t_index: u64,
    native: [f64; 3],
    registered: [f64; 3],
    size: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scores {
    schema: String,
    scores: Vec<Score>,
    provenance: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Score {
    from: u64,
    to: u64,
    cost: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: join-endpoints SUITE.json LINKS.json SCORES.json NEW_OUTDIR".into());
    }
    let input_bytes = fs::read(&args[0])?;
    let link_bytes = fs::read(&args[1])?;
    let score_bytes = fs::read(&args[2])?;
    let input: Input = serde_json::from_slice(&input_bytes)?;
    if input.schema != "blockflow.tracking-suite-input/v1" {
        return Err("unsupported suite schema".into());
    }
    let detections = DetectionSet::new(
        SeriesGeometry::new(input.extent, input.spacing)?,
        input
            .frames
            .into_iter()
            .map(|f| Frame {
                t_index: f.t_index,
                elapsed_seconds: f.elapsed_seconds,
            })
            .collect(),
        input
            .detections
            .into_iter()
            .map(|d| Detection {
                id: d.id,
                t_index: d.t_index,
                native: d.native,
                registered: d.registered,
                size: d.size,
            })
            .collect(),
    )?;
    let mut raw_links: Vec<Value> = serde_json::from_slice(&link_bytes)?;
    let endpoint = |row: &Value, key: &str| -> Result<u64, Box<dyn std::error::Error>> {
        row.get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("invalid uint64 {key}").into())
    };
    let pair = |a, b| -> Result<Candidate, Box<dyn std::error::Error>> {
        let from = detections.get(a).ok_or("unknown source identity")?;
        let to = detections.get(b).ok_or("unknown target identity")?;
        let dt = to
            .t_index
            .checked_sub(from.t_index)
            .filter(|&x| x > 0)
            .ok_or("non-forward link")?;
        let elapsed_seconds = match (
            detections
                .frame(from.t_index)
                .and_then(|f| f.elapsed_seconds),
            detections.frame(to.t_index).and_then(|f| f.elapsed_seconds),
        ) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let distance = from
            .registered
            .iter()
            .zip(to.registered)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(Candidate {
            from: a,
            to: b,
            from_t: from.t_index,
            dt,
            elapsed_seconds,
            distance,
            relative_size_change: if to.size.max(from.size) > 0.0 {
                (to.size - from.size).abs() / to.size.max(from.size)
            } else {
                0.0
            },
            overlap: None,
        })
    };
    let mut base = Vec::new();
    for row in &raw_links {
        let e = pair(endpoint(row, "from")?, endpoint(row, "to")?)?;
        if row.get("dt").is_some_and(|v| v.as_u64() != Some(e.dt)) {
            return Err("input link dt disagrees with timeline".into());
        }
        base.push(TrackLink {
            from: e.from,
            to: e.to,
            dt: e.dt,
            elapsed_seconds: e.elapsed_seconds,
            cost: row.get("cost").and_then(Value::as_f64).unwrap_or(0.),
            kind: LinkKind::Continue,
        });
    }
    TrackGraph::classify(&mut base);
    let graph = TrackGraph::new(
        &detections,
        base,
        LinkerCapabilities {
            gaps: true,
            divisions: true,
            merges: false,
        },
    )?;
    if graph
        .detection_frames()
        .any(|(id, _)| graph.successors(id).len() > 2)
    {
        return Err("binary lineage input required".into());
    }
    let scores: Scores = serde_json::from_slice(&score_bytes)?;
    if scores.schema != "blockflow.endpoint-scores/v1" {
        return Err("unsupported endpoint score schema".into());
    }
    let mut sorted = scores
        .scores
        .iter()
        .map(|s| Ok((pair(s.from, s.to)?, s.cost)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    sorted.sort_by_key(|(e, _)| (e.from, e.to));
    let candidates = CandidateSet::from_edges(sorted.iter().map(|(e, _)| e.clone()).collect())?;
    let scored = ScoredCandidates::from_costs(
        candidates,
        sorted.iter().map(|(_, c)| *c).collect(),
        scores.provenance.clone(),
    )?;
    let start = Instant::now();
    let refined = join_endpoints(&detections, &graph, &scored)?;
    let solve_seconds = start.elapsed().as_secs_f64();
    let old = graph
        .links()
        .iter()
        .map(|e| (e.from, e.to))
        .collect::<BTreeSet<_>>();
    let added = refined
        .links()
        .iter()
        .filter(|e| !old.contains(&(e.from, e.to)))
        .collect::<Vec<_>>();
    for e in &added {
        raw_links.push(json!({"from":e.from,"to":e.to,"dt":e.dt,"elapsed_seconds":e.elapsed_seconds,"cost":e.cost,"kind":e.kind.as_str(),"stage":"endpoint_refinement"}));
    }
    raw_links.sort_by_key(|v| (v["from"].as_u64(), v["to"].as_u64()));
    fs::create_dir(&args[3])?;
    fs::write(args[3].join("suite-input.json"), input_bytes)?;
    fs::write(args[3].join("base-links.json"), link_bytes)?;
    fs::write(args[3].join("endpoint-scores.json"), score_bytes)?;
    fs::write(
        args[3].join("join-endpoints.rs"),
        include_bytes!("join-endpoints.rs"),
    )?;
    fs::write(
        args[3].join("links.json"),
        serde_json::to_vec_pretty(&raw_links)?,
    )?;
    let summary = json!({"schema":"blockflow.endpoint-refinement-result/v1","complete":true,"source_paths":args[..3],
        "fixed_detections":detections.detections().len(),"base_links":graph.links().len(),"links":refined.links().len(),
        "added_continuations":added.iter().filter(|e|e.dt==1).count(),"added_gaps":added.iter().filter(|e|e.dt>1).count(),
        "added_cost_sum":added.iter().map(|e|e.cost).sum::<f64>(),"solve_seconds":solve_seconds,"scorer":scores.provenance,
        "reference_evaluation":null,"interpretation":"Preserves all existing edges and divisions; graph continuity is not tracking accuracy. Only added costs share one incremental energy scale; do not sum costs across stages."});
    fs::write(
        args[3].join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{} continuations, {} gaps in {:.6}s",
        summary["added_continuations"], summary["added_gaps"], solve_seconds
    );
    Ok(())
}
