#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_cellprofiler_sweep.sh IMAGE OUTPUT_DIR [REFERENCE_OBJECT_CSV]

Runs a small worker/chunk/cache sweep around the CellProfiler-style benchmark.
The current output-generating benchmark path is resident-only, so
worker/chunk/cache values are recorded as requested execution metadata for that
path. The planned simulator probe applies those knobs to the segmentation, seed
suppression, final-filtering and shape/intensity measurement skeleton. By
default, the benchmark also materializes planned object rows and compares them
against resident and reference tables when those inputs are available.

Environment variables:
  BF_SWEEP_WORKERS      Space-separated worker counts, default "1 4".
  BF_SWEEP_CHUNKS       Space-separated chunk shapes, default "1x256x256 1x512x512".
  BF_SWEEP_CACHE_BYTES  Space-separated cache budgets, default "0 67108864".
  BF_MERGE_LINE_BASIN_PIXELS Optional watershed-line merge threshold passed through to each run.
  BF_MERGE_LINE_MAX_SADDLE_DROP Optional merge saddle guard passed through to each run.
  BF_PLAN_MATERIALIZATION_REPEATS Repeat planned materialization timing for each run.
  BF_REFERENCE_SUMMARY  Optional CellProfiler reference-summary.json.
  BF_COMPARE_ARGS       Extra arguments passed to cellprofiler-compare.
  CARGO_BIN_FLAGS       Extra cargo flags before "--", for example "--release".
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

image="$1"
output_dir="$2"
reference_csv="${3:-}"

workers="${BF_SWEEP_WORKERS:-1 4}"
chunks="${BF_SWEEP_CHUNKS:-1x256x256 1x512x512}"
caches="${BF_SWEEP_CACHE_BYTES:-0 67108864}"

mkdir -p "$output_dir/runs"

run_index=0
for worker in $workers; do
  for chunk in $chunks; do
    for cache in $caches; do
      run_dir="$output_dir/runs/run-${run_index}"
      BF_WORKERS="$worker" \
      BF_CHUNK_SHAPE="$chunk" \
      BF_CACHE_BYTES="$cache" \
      scripts/run_cellprofiler_benchmark.sh "$image" "$run_dir" "$reference_csv"
      run_index=$((run_index + 1))
    done
  done
done

python3 - "$output_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
runs = []
for report_path in sorted((out_dir / "runs").glob("run-*/benchmark-report.json")):
    with report_path.open() as handle:
        report = json.load(handle)
    requested = report["execution_config"]["requested"]
    comparison = report.get("comparison") or {}
    metrics = comparison.get("metrics") or {}
    blockflow = report.get("blockflow") or {}
    wall = report.get("wall_time") or {}
    planner = report.get("planner_simulator") or {}
    plan_probe = planner.get("plan_probe") or {}
    plan_requested = plan_probe.get("requested") or {}
    simulator = plan_probe.get("simulator") or {}
    plan = plan_probe.get("plan") or {}
    materialized = planner.get("materialized_outputs") or plan_probe.get("materialized_outputs") or {}
    planned_comparison = (
        planner.get("planned_vs_reference")
        or report.get("planned_comparison")
        or {}
    )
    planned_comparison_metrics = planned_comparison.get("metrics") or {}
    planned_resident = (
        planner.get("planned_vs_resident")
        or report.get("planned_resident_comparison")
        or {}
    )
    planned_resident_metrics = planned_resident.get("metrics") or {}
    runs.append(
        {
            "run": report_path.parent.name,
            "workers": requested["workers"],
            "chunk_shape": requested["chunk_shape"],
            "cache_bytes": requested["cache_bytes"],
            "applied_requested_execution_config": report["execution_config"][
                "applies_requested_execution_config"
            ],
            "planned_probe_applied_requested_execution_config": report["execution_config"].get(
                "planned_probe_applies_requested_execution_config",
                report["execution_config"].get("planned_prefix_applies_requested_execution_config"),
            ),
            "objects": blockflow.get("objects"),
            "maxima_downsample": blockflow.get(
                "maxima_downsample", plan_requested.get("maxima_downsample")
            ),
            "merge_line_basin_pixels": blockflow.get(
                "merge_line_basin_pixels",
                plan_requested.get("merge_line_basin_pixels"),
            ),
            "merge_line_max_saddle_drop": blockflow.get(
                "merge_line_max_saddle_drop",
                plan_requested.get("merge_line_max_saddle_drop"),
            ),
            "pipeline_seconds": wall.get("blockflow_pipeline_seconds"),
            "planned_probe_status": planner.get("status"),
            "planned_probe_estimated_seconds": planner.get("estimated_pipeline_seconds"),
            "planned_probe_estimate_over_resident": planner.get("estimate_error_ratio"),
            "planned_probe_phases": plan.get("phases"),
            "planned_probe_skeleton_phases": plan.get("skeleton_phases"),
            "planned_probe_measurement_rows_phase": plan.get("measurement_rows_phase"),
            "planned_probe_tasks": simulator.get("tasks_run"),
            "planned_probe_cache_hits": simulator.get("cache_hits"),
            "planned_probe_cache_misses": simulator.get("cache_misses"),
            "planned_probe_peak_bytes": simulator.get("peak_bytes"),
            "planned_objects": materialized.get("objects"),
            "planned_materialization_seconds": materialized.get("seconds"),
            "planned_materialization_seconds_min": materialized.get("seconds_min", materialized.get("seconds")),
            "planned_materialization_seconds_mean": materialized.get("seconds_mean"),
            "planned_materialization_seconds_median": materialized.get("seconds_median"),
            "planned_materialization_repeats": materialized.get("repeats", 1 if materialized else None),
            "planned_label_nonzero_voxels": (materialized.get("label_image") or {}).get(
                "nonzero_voxels"
            ),
            "planned_vs_reference_passed": planned_comparison.get("passed"),
            "planned_vs_reference_matched": planned_comparison.get("matched_objects"),
            "planned_vs_reference_mean_centroid_distance": planned_comparison_metrics.get(
                "mean_centroid_distance"
            ),
            "planned_vs_reference_mean_area_relative_error": planned_comparison_metrics.get(
                "mean_area_relative_error"
            ),
            "planned_vs_resident_passed": planned_resident.get("passed"),
            "planned_vs_resident_matched": planned_resident.get("matched_objects"),
            "planned_vs_resident_mean_centroid_distance": planned_resident_metrics.get(
                "mean_centroid_distance"
            ),
            "planned_vs_resident_mean_area_relative_error": planned_resident_metrics.get(
                "mean_area_relative_error"
            ),
            "reference_objects": comparison.get("reference_objects"),
            "mean_centroid_distance": metrics.get("mean_centroid_distance"),
            "mean_area_relative_error": metrics.get("mean_area_relative_error"),
            "mean_mean_intensity_relative_error": metrics.get(
                "mean_mean_intensity_relative_error"
            ),
            "comparison_passed": comparison.get("passed"),
        }
    )

def numeric(row, key):
    value = row.get(key)
    return float("inf") if value is None else value

valid_estimates = [row for row in runs if row.get("planned_probe_estimated_seconds") is not None]
best_estimated = min(valid_estimates, key=lambda row: numeric(row, "planned_probe_estimated_seconds"), default=None)
valid_materialized = [row for row in runs if row.get("planned_materialization_seconds") is not None]
best_materialized = min(valid_materialized, key=lambda row: numeric(row, "planned_materialization_seconds"), default=None)
valid_materialized_min = [row for row in runs if row.get("planned_materialization_seconds_min") is not None]
best_materialized_min = min(
    valid_materialized_min,
    key=lambda row: numeric(row, "planned_materialization_seconds_min"),
    default=None,
)
resident_matched = [
    row
    for row in runs
    if row.get("planned_vs_resident_passed") is True
    and row.get("planned_vs_resident_mean_area_relative_error") == 0
]
best_resident_matched = min(
    resident_matched,
    key=lambda row: numeric(row, "planned_probe_estimated_seconds"),
    default=None,
)

summary = {
    "status": "planned_segmentation_with_measurements_sweep",
    "runs": runs,
    "regret_summary": {
        "best_by_simulated_seconds": best_estimated,
        "best_by_planned_materialization_seconds": best_materialized,
        "best_by_planned_materialization_seconds_min": best_materialized_min,
        "best_planned_resident_exact_by_simulated_seconds": best_resident_matched,
        "simulated_seconds_spread": (
            None
            if not valid_estimates
            else {
                "min": min(numeric(row, "planned_probe_estimated_seconds") for row in valid_estimates),
                "max": max(numeric(row, "planned_probe_estimated_seconds") for row in valid_estimates),
            }
        ),
        "materialization_seconds_spread": (
            None
            if not valid_materialized
            else {
                "min": min(numeric(row, "planned_materialization_seconds") for row in valid_materialized),
                "max": max(numeric(row, "planned_materialization_seconds") for row in valid_materialized),
            }
        ),
        "note": "materialization seconds are measured planned-executor diagnostic time for this benchmark run, not a stable microbenchmark; use repeated runs before treating small differences as planner regret",
    },
    "planner_guidance": {
        "usable_for_planned_measurement_skeleton_choice": True,
        "usable_for_full_pipeline_choice": True,
        "reason": "worker/chunk/cache settings are applied to the planned segmentation, seed suppression, optional watershed-line basin merging, final-filtering and shape/intensity measurement simulator probe; planned object-table materialization is now recorded and compared when enabled",
        "next_step": "use planned-vs-resident/reference outliers plus observed executor runtime to decide whether planner changes or segmentation semantic fixes matter more",
    },
}

with (out_dir / "sweep-report.json").open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

fields = [
    "run",
    "workers",
    "chunk_shape",
    "cache_bytes",
    "applied_requested_execution_config",
    "planned_probe_applied_requested_execution_config",
    "objects",
    "maxima_downsample",
    "merge_line_basin_pixels",
    "merge_line_max_saddle_drop",
    "pipeline_seconds",
    "planned_probe_status",
    "planned_probe_estimated_seconds",
    "planned_probe_estimate_over_resident",
    "planned_probe_phases",
    "planned_probe_skeleton_phases",
    "planned_probe_measurement_rows_phase",
    "planned_probe_tasks",
    "planned_probe_cache_hits",
    "planned_probe_cache_misses",
    "planned_probe_peak_bytes",
    "planned_objects",
    "planned_materialization_seconds",
    "planned_materialization_seconds_min",
    "planned_materialization_seconds_mean",
    "planned_materialization_seconds_median",
    "planned_materialization_repeats",
    "planned_label_nonzero_voxels",
    "planned_vs_reference_passed",
    "planned_vs_reference_matched",
    "planned_vs_reference_mean_centroid_distance",
    "planned_vs_reference_mean_area_relative_error",
    "planned_vs_resident_passed",
    "planned_vs_resident_matched",
    "planned_vs_resident_mean_centroid_distance",
    "planned_vs_resident_mean_area_relative_error",
    "reference_objects",
    "mean_centroid_distance",
    "mean_area_relative_error",
    "mean_mean_intensity_relative_error",
    "comparison_passed",
]
with (out_dir / "sweep-report.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(runs)
PY
