// SPDX-License-Identifier: MIT

//! Row-only, fixed-mask comparisons. Never opens image channels or writes masks.

use blockflow::track::{
    self, AlternativeCost, CandidateParams, Detection, DetectionSet, Frame, FrameLapLinker,
    GapClosing, GeometricCost, GreedyFrameLinker, LapParams, LinkKind, LinkerCapabilities,
    ScoredCandidates, ScoredLinker, SegmentLapLinker, SegmentParams, SeriesGeometry, TrackGraph,
    TrackLink,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    extent: [usize; 3],
    spacing: [f64; 3],
    frames: Vec<InputFrame>,
    detections: Vec<InputDetection>,
    max_distance: f64,
    runs: Vec<Run>,
    provenance: Value,
    #[serde(default)]
    baseline_pairs: Option<Vec<[u64; 2]>>,
    #[serde(default)]
    reference: Option<Reference>,
    #[serde(default)]
    score_tables: BTreeMap<String, ImportedScores>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedScores {
    costs: Vec<ImportedCost>,
    provenance: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedCost {
    from: u64,
    to: u64,
    cost: f64,
}

fn imported_scores(
    name: &str,
    candidates: &track::CandidateSet,
    table: &ImportedScores,
) -> Result<ScoredCandidates> {
    let candidate_of = candidates
        .edges()
        .iter()
        .map(|c| ((c.from, c.to), c))
        .collect::<BTreeMap<_, _>>();
    let mut costs = BTreeMap::new();
    for row in &table.costs {
        let key = (row.from, row.to);
        if !candidate_of.contains_key(&key) {
            return Err(format!(
                "score table {name} contains an edge outside the shared candidate gate: {} -> {}",
                row.from, row.to
            )
            .into());
        }
        if costs.insert(key, row.cost).is_some() {
            return Err(format!(
                "score table {name} duplicates edge {} -> {}",
                row.from, row.to
            )
            .into());
        }
    }
    let edges = costs.keys().map(|key| candidate_of[key].clone()).collect();
    Ok(ScoredCandidates::from_costs(
        track::CandidateSet::from_edges(edges)?,
        costs.into_values().collect(),
        json!({"name": "imported", "table": name, "parameters": table.provenance}),
    )?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    /// Reference covers every input detection, including isolated objects.
    complete: bool,
    kind: String,
    pairs: Vec<[u64; 2]>,
    provenance: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputFrame {
    t_index: u64,
    elapsed_seconds: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputDetection {
    id: u64,
    t_index: u64,
    native: [f64; 3],
    registered: [f64; 3],
    size: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Solver {
    FrameLap,
    FrameGreedy,
    SegmentLap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    name: String,
    solver: Solver,
    size_weight: f64,
    #[serde(default)]
    scorer: Option<String>,
    #[serde(default)]
    alternative_cost: Option<f64>,
    #[serde(default)]
    gap_closing: Option<f64>,
    #[serde(default)]
    max_gap: Option<u64>,
    #[serde(default)]
    splitting: Option<f64>,
}

fn imported_graph(set: &DetectionSet, pairs: &[[u64; 2]]) -> Result<TrackGraph> {
    let links = pairs
        .iter()
        .map(|&[from, to]| {
            let a = set.get(from).ok_or("unknown imported graph source")?;
            let b = set.get(to).ok_or("unknown imported graph target")?;
            let dt = b
                .t_index
                .checked_sub(a.t_index)
                .filter(|&dt| dt > 0)
                .ok_or("imported graph is not forward in time")?;
            let elapsed_seconds = match (
                set.frame(a.t_index).unwrap().elapsed_seconds,
                set.frame(b.t_index).unwrap().elapsed_seconds,
            ) {
                (Some(a), Some(b)) => Some(b - a),
                _ => None,
            };
            Ok(TrackLink {
                from,
                to,
                dt,
                elapsed_seconds,
                cost: 0.0,
                kind: LinkKind::Continue,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut links = links;
    TrackGraph::classify(&mut links);
    Ok(TrackGraph::new(
        set,
        links,
        LinkerCapabilities {
            gaps: true,
            divisions: true,
            merges: true,
        },
    )?)
}

fn run() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("tracking-suite must be built with --release".into());
    }
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: tracking-suite INPUT.json NEW_OUTPUT_DIRECTORY [REPEATS=5]".into());
    }
    let input_path = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let repeats = args
        .get(2)
        .map(|a| a.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let bytes = fs::read(&input_path)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    if input.schema != "blockflow.tracking-suite-input/v1" {
        return Err("unsupported input schema".into());
    }
    let geometry = SeriesGeometry::new(input.extent, input.spacing)?;
    let set = DetectionSet::new(
        geometry,
        input
            .frames
            .iter()
            .map(|f| Frame {
                t_index: f.t_index,
                elapsed_seconds: f.elapsed_seconds,
            })
            .collect(),
        input
            .detections
            .iter()
            .map(|d| Detection {
                id: d.id,
                t_index: d.t_index,
                native: d.native,
                registered: d.registered,
                size: d.size,
            })
            .collect(),
    )?;
    let mut names = BTreeSet::new();
    for run in &input.runs {
        if run.name.is_empty()
            || run.name.len() > 80
            || run.name == "."
            || run.name == ".."
            || !run
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
            || !names.insert(&run.name)
        {
            return Err(
                "run names must be unique, safe directory names of 1..80 ASCII characters".into(),
            );
        }
        if !matches!(run.solver, Solver::SegmentLap)
            && (run.gap_closing.is_some() || run.splitting.is_some() || run.max_gap.is_some())
        {
            return Err("gap/split options require segment_lap".into());
        }
        if let Some(name) = &run.scorer {
            if !input.score_tables.contains_key(name) {
                return Err(format!("unknown score table {name}").into());
            }
            if run.size_weight != 0.0 {
                return Err("imported scoring requires size_weight=0; combine additional terms explicitly upstream".into());
            }
            if matches!(run.solver, Solver::SegmentLap) {
                return Err("imported scores currently support frame solvers only; legacy segment event costs use different units".into());
            }
        }
        if run.gap_closing.is_none() && run.max_gap.is_some() {
            return Err("max_gap requires gap_closing".into());
        }
    }
    if input.runs.is_empty() {
        return Err("at least one comparison run is required".into());
    }
    let reference = input.reference.as_ref().map(|r| {
        if !r.complete || !["synthetic", "human_reviewed_complete"].contains(&r.kind.as_str()) {
            return Err("reference must explicitly cover the complete input graph and be synthetic or human_reviewed_complete".into());
        }
        imported_graph(&set, &r.pairs)
    }).transpose()?;
    let baseline = input
        .baseline_pairs
        .as_ref()
        .map(|pairs| imported_graph(&set, pairs))
        .transpose()?;
    let baseline_edges = baseline.as_ref().map(|graph| {
        graph
            .links()
            .iter()
            .map(|l| (l.from, l.to))
            .collect::<BTreeSet<_>>()
    });
    let now = Instant::now();
    let candidates =
        track::generate_candidates(&set, &CandidateParams::distance(input.max_distance))?;
    let candidate_seconds = now.elapsed().as_secs_f64();
    let score_tables = input
        .score_tables
        .iter()
        .map(|(name, table)| Ok((name.clone(), imported_scores(name, &candidates, table)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    fs::create_dir(&output)?; // Refuse any existing run; no source files are modified.
    fs::write(output.join("input.json"), &bytes)?;
    let mut reports = Vec::new();
    for config in &input.runs {
        let cost = GeometricCost::new(0.0, config.size_weight)?;
        let start = Instant::now();
        let scores = if let Some(name) = &config.scorer {
            score_tables[name].clone()
        } else {
            ScoredCandidates::score(&set, &candidates, &cost)?
        };
        let scoring_seconds = start.elapsed().as_secs_f64();
        let alternative = config
            .alternative_cost
            .map(AlternativeCost::Fixed)
            .unwrap_or_default();
        let frame = LapParams {
            alternative,
            ..LapParams::default()
        };
        let solver: Box<dyn ScoredLinker> = match config.solver {
            Solver::FrameLap => Box::new(FrameLapLinker::new(frame)?),
            Solver::FrameGreedy => Box::new(GreedyFrameLinker::new(alternative)?),
            Solver::SegmentLap => Box::new(SegmentLapLinker::new(SegmentParams {
                frame,
                gap_closing: config.gap_closing.map(|max_distance| GapClosing {
                    max_distance,
                    max_frame_gap: config.max_gap.unwrap_or(2),
                }),
                splitting: config.splitting,
                ..SegmentParams::default()
            })?),
        };
        let mut timings = Vec::new();
        let mut graph: Option<TrackGraph> = None;
        for _ in 0..repeats {
            let start = Instant::now();
            let next = solver.link_scored(&set, &scores)?;
            timings.push(start.elapsed().as_secs_f64());
            if graph
                .as_ref()
                .is_some_and(|previous| previous.links() != next.links())
            {
                return Err(
                    format!("{} changed output across repeated solves", config.name).into(),
                );
            }
            graph = Some(next);
        }
        let graph = graph.unwrap();
        let tracks = graph.tracks(&set, 25.0)?;
        let mut outdegree: BTreeMap<u64, usize> = BTreeMap::new();
        for link in graph.links() {
            *outdegree.entry(link.from).or_default() += 1;
        }
        let edges = graph
            .links()
            .iter()
            .map(|l| (l.from, l.to))
            .collect::<BTreeSet<_>>();
        let difference = baseline_edges.as_ref().map(|baseline| json!({
            "equal": baseline == &edges, "added": edges.difference(baseline).count(), "removed": baseline.difference(&edges).count(),
            "interpretation": "agreement with previous output, not tracking accuracy",
        }));
        let evaluation = reference
            .as_ref()
            .map(|reference| track::evaluation::compare_graphs(reference, &graph))
            .transpose()?;
        timings.sort_by(f64::total_cmp);
        let links = graph
            .links()
            .iter()
            .map(|l| {
                json!({"from": l.from, "to": l.to, "dt": l.dt,
            "elapsed_seconds": l.elapsed_seconds, "cost": l.cost, "kind": l.kind.as_str()})
            })
            .collect::<Vec<_>>();
        let report = json!({ "name": config.name, "solver": {"name": solver.name(), "parameters": solver.parameters()},
            "scorer": scores.provenance(), "candidates": scores.candidates().len(), "scope": "scorer controls frame association only; segment-LAP retains geometric event costs",
            "counts": {"links": graph.links().len(), "tracks": tracks.len(), "binary_divisions": outdegree.values().filter(|&&n| n == 2).count(),
                "multifurcations": outdegree.values().filter(|&&n| n > 2).count()},
            "scoring_seconds": scoring_seconds, "solver_seconds": {"repeats": repeats, "sorted_samples": timings, "median": timings[timings.len()/2]},
            "baseline_edge_difference": difference, "reference_evaluation": evaluation });
        fs::create_dir(output.join(&config.name))?;
        fs::write(
            output.join(&config.name).join("links.json"),
            serde_json::to_vec(&links)?,
        )?;
        fs::write(
            output.join(&config.name).join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!(
            "{}: {} links, {} tracks; median {:.6}s",
            config.name,
            graph.links().len(),
            tracks.len(),
            timings[timings.len() / 2]
        );
        reports.push(report);
    }
    let summary = json!({"schema": "blockflow.tracking-suite-result/v1", "complete": true,
        "input_provenance": input.provenance, "reference_provenance": input.reference.as_ref().map(|r| &r.provenance),
        "frames": set.frames().len(), "detections": set.detections().len(), "candidates": candidates.len(),
        "candidate_generation_seconds": candidate_seconds, "runs": reports,
        "validation": if reference.is_some() {"reference-based fixed-mask graph comparison"} else {"diagnostics only; no reference accuracy asserted"} });
    // Completion is published last; an interrupted directory has no summary.
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tracking-suite: {error}");
        std::process::exit(1);
    }
}
