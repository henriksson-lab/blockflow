// SPDX-License-Identifier: MIT

//! Score a tracking against a hand-annotated lineage (detection, division,
//! link and naming metrics), or with `--sanity`, the lineage against
//! perturbed copies of itself. See the example's `README.md` and
//! `celegans_4d::metrics`.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use blockflow::{Error, Result};
use celegans_4d::metrics::{score, Coverage, ScoreParams, Tracking};
use clap::{Parser, ValueEnum};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CoverageChoice {
    /// Every unmatched prediction is a false positive.
    Full,
    /// Unmatched predictions count only within --partial-radius of a truth detection.
    Partial,
    /// No false positives; precision is not reported.
    RecallOnly,
}

#[derive(Debug, Parser)]
#[command(
    name = "score_tracking",
    about = "Tracking metrics against an imported Endrov lineage"
)]
struct Args {
    /// Ground-truth detections table, e.g. `rec.ome.zarr/DIC/tables/endrov`.
    #[arg(long)]
    truth: PathBuf,
    /// Predicted detections table (same format). Required unless --sanity.
    #[arg(long)]
    pred: Option<PathBuf>,
    /// The image whose level-0 transform maps voxels to µm. Default: each table's own image.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Matching gate, µm (3-D).
    #[arg(long, default_value_t = 3.0)]
    gate: f64,
    #[arg(long, default_value_t = 1)]
    division_tolerance: u64,
    #[arg(long, value_enum, default_value_t = CoverageChoice::Full)]
    coverage: CoverageChoice,
    /// For --coverage partial, µm. Default: the gate.
    #[arg(long)]
    partial_radius: Option<f64>,
    /// Inclusive time-index range, `a:b`.
    #[arg(long)]
    t_range: Option<String>,
    /// Score the truth against itself and perturbed copies.
    #[arg(long)]
    sanity: bool,
    /// Also write the JSON here.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("score_tracking: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main_inner() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("this example requires cargo run --release"));
    }
    let args = Args::parse();
    let t_range = args
        .t_range
        .as_deref()
        .map(|r| {
            let (a, b) = r
                .split_once(':')
                .ok_or_else(|| Error::invalid("--t-range is a:b"))?;
            let p = |s: &str| {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| Error::invalid(format!("--t-range: {s:?}")))
            };
            Ok::<_, Error>((p(a)?, p(b)?))
        })
        .transpose()?;
    let params = ScoreParams {
        gate: args.gate,
        division_tolerance: args.division_tolerance,
        coverage: match args.coverage {
            CoverageChoice::Full => Coverage::Full,
            CoverageChoice::Partial => Coverage::Partial {
                radius: args.partial_radius.unwrap_or(args.gate),
            },
            CoverageChoice::RecallOnly => Coverage::RecallOnly,
        },
        t_range,
    };
    let truth = Tracking::read(&args.truth, args.image.as_deref())?;
    let result = if args.sanity {
        sanity(&truth, &params)?
    } else {
        let pred_path = args
            .pred
            .as_ref()
            .ok_or_else(|| Error::invalid("pass --pred or --sanity"))?;
        let pred = Tracking::read(pred_path, args.image.as_deref())?;
        json!({"truth": args.truth, "pred": pred_path, "scores": score(&truth, &pred, &params)?})
    };
    let text = serde_json::to_string_pretty(&result).map_err(Error::backend)?;
    if let Some(out) = &args.out {
        fs::write(out, &text)
            .map_err(|e| Error::backend(format!("write {}: {e}", out.display())))?;
    }
    println!("{text}");
    Ok(())
}

/// The truth against itself and against known perturbations; each case
/// states what a correct implementation must report.
fn sanity(truth: &Tracking, params: &ScoreParams) -> Result<Value> {
    let mut cases = Vec::new();
    let mut case = |name: &str,
                    expect: &str,
                    t: &Tracking,
                    pred: &Tracking,
                    p: &ScoreParams|
     -> Result<()> {
        let s = score(t, pred, p)?;
        cases.push(json!({
            "case": name,
            "expect": expect,
            "detection_f1": s.detection.f1, "detection_precision": s.detection.precision, "detection_recall": s.detection.recall,
            "detection_unscored": s.detection.unscored,
            "matched_distance_mean": s.matched_distance_mean,
            "division_f1": s.division.f1, "division_recall": s.division.recall, "division_precision": s.division.precision,
            "link_recall": s.links.recall, "link_precision": s.links.precision,
            "naming_of_matched": s.naming.as_ref().and_then(|n| n.of_matched),
            "naming_end_to_end": s.naming.as_ref().and_then(|n| n.end_to_end),
        }));
        Ok(())
    };
    case("self", "all 1", truth, truth, params)?;
    for (i, sigma) in [0.25, 0.5, 1.0, 2.0, 3.0].into_iter().enumerate() {
        case(
            &format!("jitter_sigma_{sigma}um"),
            "F1 falls as 3σ√3 nears the gate; matched distance ≈ σ·√(8/π) (Maxwell mean)",
            truth,
            &truth.jittered(sigma, 1 + i as u64),
            params,
        )?;
    }
    case(
        "divisions_removed_all",
        "division recall 0, detection F1 1, link recall < 1",
        truth,
        &truth.without_divisions(1),
        params,
    )?;
    case(
        "divisions_removed_every_2nd",
        "division recall ≈ 0.5, precision 1",
        truth,
        &truth.without_divisions(2),
        params,
    )?;
    case(
        "detections_dropped_every_10th",
        "detection recall ≈ 0.9, precision 1",
        truth,
        &truth.thinned(10),
        params,
    )?;
    case(
        "sibling_names_swapped",
        "geometry 1; naming < 1",
        truth,
        &truth.siblings_swapped(),
        params,
    )?;
    // Partial truth: keep every other lineage tree (by root track id), as an
    // annotator following some founders; the full lineage is the prediction.
    let roots = truth.lineage_roots();
    let order: Vec<u64> = roots
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let kept: std::collections::BTreeSet<u64> = order.iter().step_by(2).copied().collect();
    let partial = truth.restricted(|track| roots.get(&track).is_some_and(|r| kept.contains(r)));
    for (label, coverage, expect) in [
        ("partial_truth_full_mode", Coverage::Full, "precision well below 1: the unannotated trees count as FP"),
        ("partial_truth_partial_mode", Coverage::Partial { radius: params.gate }, "precision ≈ 1, recall 1; FP only for unannotated nuclei within the radius of an annotated one"),
        ("partial_truth_recall_only", Coverage::RecallOnly, "precision null, recall 1"),
    ] {
        case(label, expect, &partial, truth, &ScoreParams { coverage, ..*params })?;
    }
    Ok(json!({"params": params, "cases": cases}))
}
