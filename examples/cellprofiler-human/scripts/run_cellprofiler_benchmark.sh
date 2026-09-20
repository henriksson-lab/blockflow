#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  examples/cellprofiler-human/scripts/run_cellprofiler_benchmark.sh IMAGE OUTPUT_DIR [REFERENCE_OBJECT_CSV]

Runs the planned Blockflow CellProfiler-style benchmark path and, when a
reference object CSV is supplied, runs semantic table comparison against it.

Environment variables:
  CARGO_BIN_FLAGS   Extra cargo flags before "--", for example "--release".
  BF_MIN_SIZE       Minimum object size, default 50.
  BF_MAX_SIZE       Maximum object size, default 5027. Set empty with BF_NO_MAX_SIZE=1.
  BF_SIGMA          XY Gaussian sigma, default 1.0.
  BF_DECLUMP_SIGMA  XY Gaussian sigma for intensity declumping, default 1.3488.
  BF_THRESHOLD_METHOD Global threshold method, default li.
  BF_THRESHOLD_BINS Otsu bin count when used, default 256.
  BF_SEED_MIN_DISTANCE Watershed seed suppression distance, default 6.
  BF_MAXIMA_DOWNSAMPLE Lower-resolution XY seed-maxima block factor, default 3.
  BF_DECLUMP_METHOD Declump watershed source, intensity or distance, default intensity.
  BF_ADJACENT_BASINS Set to 1 to let watershed basins touch instead of carving lines.
  BF_NO_FILL_HOLES_AFTER_DECLUMPING Set to 1 to skip post-declump hole filling.
  BF_MERGE_LINE_BASIN_PIXELS Merge planned labels separated by at least this many watershed-line pixels, default 0.
  BF_MERGE_LINE_MAX_SADDLE_DROP Optional maximum weak-boundary-minus-line mean for line merges.
  BF_WORKERS       Planned worker count, default 1.
  BF_CHUNK_SHAPE   Planned chunk shape, for example 1x256x256.
  BF_CACHE_BYTES   Planned cache budget in bytes.
  BF_DISTANCE_BLOCK Distance-transform block edge for the planned simulator probe, default 256.
  BF_INPUT_ZARR    Prepared rank-3 Zarr input array or store. Defaults to OUTPUT_DIR/input.zarr.
  BF_ENABLE_RESIDENT_REFERENCE Set to 1 to also run the old resident reference binary.
  BF_PLAN_MATERIALIZATION_REPEATS Repeat planned materialization timing, default 1.
  BF_REFERENCE_SUMMARY Optional CellProfiler reference-summary.json.
  BF_REFERENCE_LABELS Optional grayscale/integer reference label image.
  BF_COMPARE_ARGS   Extra arguments passed to cellprofiler-compare.
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
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$image" ]]; then
  echo "Input image not found: $image" >&2
  exit 1
fi

if [[ -n "$reference_csv" && ! -f "$reference_csv" ]]; then
  echo "Reference CSV not found: $reference_csv" >&2
  exit 1
fi

mkdir -p "$output_dir"

min_size="${BF_MIN_SIZE:-50}"
max_size="${BF_MAX_SIZE:-5027}"
sigma="${BF_SIGMA:-1.0}"
declump_sigma="${BF_DECLUMP_SIGMA:-1.3488}"
threshold_method="${BF_THRESHOLD_METHOD:-li}"
threshold_bins="${BF_THRESHOLD_BINS:-256}"
seed_min_distance="${BF_SEED_MIN_DISTANCE:-6}"
maxima_downsample="${BF_MAXIMA_DOWNSAMPLE:-3}"
declump_method="${BF_DECLUMP_METHOD:-intensity}"
basin_args=()
if [[ "${BF_ADJACENT_BASINS:-}" == "1" ]]; then
  basin_args+=(--adjacent-basins)
fi
hole_args=()
if [[ "${BF_NO_FILL_HOLES_AFTER_DECLUMPING:-}" == "1" ]]; then
  hole_args+=(--no-fill-holes-after-declumping)
fi
merge_line_basin_pixels="${BF_MERGE_LINE_BASIN_PIXELS:-0}"
merge_line_max_saddle_drop="${BF_MERGE_LINE_MAX_SADDLE_DROP:-}"
merge_args=()
if [[ "$merge_line_basin_pixels" != "0" ]]; then
  merge_args+=(--merge-line-basin-pixels "$merge_line_basin_pixels")
fi
if [[ -n "$merge_line_max_saddle_drop" ]]; then
  merge_args+=(--merge-line-max-saddle-drop "$merge_line_max_saddle_drop")
fi
workers="${BF_WORKERS:-1}"
chunk_shape="${BF_CHUNK_SHAPE:-1x256x256}"
cache_bytes="${BF_CACHE_BYTES:-0}"
distance_block="${BF_DISTANCE_BLOCK:-256}"
enable_resident_reference="${BF_ENABLE_RESIDENT_REFERENCE:-0}"
plan_materialization_repeats="${BF_PLAN_MATERIALIZATION_REPEATS:-1}"
reference_summary="${BF_REFERENCE_SUMMARY:-}"
reference_labels="${BF_REFERENCE_LABELS:-}"
planned_objects_csv="$output_dir/blockflow/planned_objects.csv"
input_zarr_store="${BF_INPUT_ZARR:-$output_dir/input.zarr}"

# Prepare the source once. The timed Blockflow path reads the same Zarr array
# that a normal invocation of cellprofiler-human reads.
if [[ ! -f "$input_zarr_store/zarr.json" && ! -f "$input_zarr_store/level0/zarr.json" ]]; then
  "$script_dir/prepare_input_zarr.sh" "$image" "$input_zarr_store"
fi
input_zarr_array="$input_zarr_store"
if [[ -f "$input_zarr_store/level0/zarr.json" ]]; then
  input_zarr_array="$input_zarr_store/level0"
fi

size_args=(--min-size "$min_size")
if [[ "${BF_NO_MAX_SIZE:-}" == "1" ]]; then
  size_args+=(--no-max-size)
else
  size_args+=(--max-size "$max_size")
fi
plan_size_args=(--min-size "$min_size")
if [[ "${BF_NO_MAX_SIZE:-}" == "1" ]]; then
  plan_size_args+=(--no-max-size)
else
  plan_size_args+=(--max-size "$max_size")
fi

read -r -a cargo_bin_flags <<< "${CARGO_BIN_FLAGS:-}"
read -r -a compare_args <<< "${BF_COMPARE_ARGS:-}"
label_args=()
if [[ -n "$reference_labels" ]]; then
  if [[ ! -f "$reference_labels" ]]; then
    echo "Reference label image not found: $reference_labels" >&2
    exit 1
  fi
  label_args=(
    --blockflow-labels "$output_dir/blockflow/labels.png"
    --reference-labels "$reference_labels"
  )
fi

if [[ -n "$reference_summary" && ! -f "$reference_summary" ]]; then
  echo "Reference summary JSON not found: $reference_summary" >&2
  exit 1
fi

{
  printf 'blockflow_command='
  printf '%q ' cargo run -p blockflow-cellprofiler-human --bin cellprofiler-human "${cargo_bin_flags[@]}" -- \
    --input-zarr "$input_zarr_array" \
    --out "$output_dir/plan-probe.json" --chunk "$chunk_shape" \
    --workers "$workers" --cache-bytes "$cache_bytes" --sigma "$sigma" \
    --threshold-method "$threshold_method" --threshold-bins "$threshold_bins" \
    "${plan_size_args[@]}" --seed-min-distance "$seed_min_distance" \
    --maxima-downsample "$maxima_downsample" \
    --declump-method "$declump_method" \
    "${merge_args[@]}" \
    --distance-block "$distance_block" \
    --materialize-repeats "$plan_materialization_repeats" \
    --materialize-objects "$output_dir/blockflow"
  printf '\n'
} > "$output_dir/benchmark-command.txt"

cargo run -p blockflow-cellprofiler-human --bin cellprofiler-human "${cargo_bin_flags[@]}" -- \
  --input-zarr "$input_zarr_array" \
  --out "$output_dir/plan-probe.json" \
  --chunk "$chunk_shape" \
  --workers "$workers" \
  --cache-bytes "$cache_bytes" \
  --sigma "$sigma" \
  --threshold-method "$threshold_method" \
  --threshold-bins "$threshold_bins" \
  "${plan_size_args[@]}" \
  --seed-min-distance "$seed_min_distance" \
  --maxima-downsample "$maxima_downsample" \
  --declump-method "$declump_method" \
  "${merge_args[@]}" \
  --distance-block "$distance_block" \
  --materialize-repeats "$plan_materialization_repeats" \
  --materialize-objects "$output_dir/blockflow"

if [[ "$enable_resident_reference" == "1" ]]; then
  {
    printf 'resident_reference_command='
    printf '%q ' cargo run -p blockflow-cellprofiler-human --bin cellprofiler-resident-reference "${cargo_bin_flags[@]}" -- \
      --input "$image" --out "$output_dir/resident" "${size_args[@]}" --sigma "$sigma" \
      --declump-sigma "$declump_sigma" --threshold-method "$threshold_method" --threshold-bins "$threshold_bins" \
      --seed-min-distance "$seed_min_distance" --maxima-downsample "$maxima_downsample" \
      --declump-method "$declump_method" \
      "${basin_args[@]}" "${hole_args[@]}" "${merge_args[@]}"
    printf '\n'
  } >> "$output_dir/benchmark-command.txt"

  cargo run -p blockflow-cellprofiler-human --bin cellprofiler-resident-reference "${cargo_bin_flags[@]}" -- \
    --input "$image" \
    --out "$output_dir/resident" \
    "${size_args[@]}" \
    --sigma "$sigma" \
    --declump-sigma "$declump_sigma" \
    --threshold-method "$threshold_method" \
    --threshold-bins "$threshold_bins" \
    --seed-min-distance "$seed_min_distance" \
    --maxima-downsample "$maxima_downsample" \
    --declump-method "$declump_method" \
    "${basin_args[@]}" \
    "${hole_args[@]}" \
    "${merge_args[@]}"
fi

if [[ -n "$reference_csv" ]]; then
  {
    printf 'compare_command='
    printf '%q ' cargo run -p blockflow-cellprofiler-human --bin cellprofiler-compare "${cargo_bin_flags[@]}" -- \
      --blockflow "$planned_objects_csv" --reference "$reference_csv" \
      --out "$output_dir/comparison.json" "${label_args[@]}" "${compare_args[@]}"
    printf '\n'
  } >> "$output_dir/benchmark-command.txt"

  cargo run -p blockflow-cellprofiler-human --bin cellprofiler-compare "${cargo_bin_flags[@]}" -- \
    --blockflow "$planned_objects_csv" \
    --reference "$reference_csv" \
    --out "$output_dir/comparison.json" \
    "${label_args[@]}" \
    "${compare_args[@]}"
fi

if [[ "$enable_resident_reference" == "1" && -f "$planned_objects_csv" && -f "$output_dir/resident/objects.csv" ]]; then
  {
    printf 'planned_resident_compare_command='
    printf '%q ' cargo run -p blockflow-cellprofiler-human --bin cellprofiler-compare "${cargo_bin_flags[@]}" -- \
      --blockflow "$planned_objects_csv" --reference "$output_dir/resident/objects.csv" \
      --out "$output_dir/planned-resident-comparison.json" \
      --reference-area count --reference-centroid-z centroid_z --reference-centroid-y centroid_y \
      --reference-centroid-x centroid_x --reference-mean-intensity intensity_mean \
      --reference-integrated-intensity intensity_sum --reference-bbox-min-z bbox_min_z \
      --reference-bbox-min-y bbox_min_y --reference-bbox-min-x bbox_min_x \
      --reference-bbox-max-z bbox_max_z --reference-bbox-max-y bbox_max_y \
      --reference-bbox-max-x bbox_max_x
    printf '\n'
  } >> "$output_dir/benchmark-command.txt"

  cargo run -p blockflow-cellprofiler-human --bin cellprofiler-compare "${cargo_bin_flags[@]}" -- \
    --blockflow "$planned_objects_csv" \
    --reference "$output_dir/resident/objects.csv" \
    --out "$output_dir/planned-resident-comparison.json" \
    --reference-area count \
    --reference-centroid-z centroid_z \
    --reference-centroid-y centroid_y \
    --reference-centroid-x centroid_x \
    --reference-mean-intensity intensity_mean \
    --reference-integrated-intensity intensity_sum \
    --reference-bbox-min-z bbox_min_z \
    --reference-bbox-min-y bbox_min_y \
    --reference-bbox-min-x bbox_min_x \
    --reference-bbox-max-z bbox_max_z \
    --reference-bbox-max-y bbox_max_y \
    --reference-bbox-max-x bbox_max_x
fi

python3 - "$output_dir" "$reference_summary" "$workers" "$chunk_shape" "$cache_bytes" "$maxima_downsample" "$merge_line_basin_pixels" "$merge_line_max_saddle_drop" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
reference_summary_path = Path(sys.argv[2]) if sys.argv[2] else None
requested_workers = sys.argv[3]
requested_chunk_shape = sys.argv[4]
requested_cache_bytes = sys.argv[5]
maxima_downsample = int(sys.argv[6])
merge_line_basin_pixels = int(sys.argv[7])
merge_line_max_saddle_drop = None if not sys.argv[8] else float(sys.argv[8])

def read_json(path):
    if path and path.exists():
        with path.open() as handle:
            return json.load(handle)
    return None

blockflow = read_json(out_dir / "blockflow" / "planned-summary.json")
comparison = read_json(out_dir / "comparison.json")
plan_probe = read_json(out_dir / "plan-probe.json")
planned_resident_comparison = read_json(out_dir / "planned-resident-comparison.json")
reference = read_json(reference_summary_path) if reference_summary_path else None
plan_materialized = None if plan_probe is None else plan_probe.get("materialized_outputs")
input_zarr = None if plan_probe is None else plan_probe.get("input_zarr")
planner_missing = []
if plan_probe is not None and not plan_materialized:
    planner_missing.append("object CSV/table materialization")
elif (
    plan_materialized is not None
    and (plan_materialized.get("objects") or 0) == 0
):
    planner_missing.append("validated non-empty object CSV/table materialization")
requested_execution = {
    "workers": requested_workers,
    "chunk_shape": requested_chunk_shape,
    "cache_bytes": requested_cache_bytes,
}

report = {
    "blockflow": blockflow,
    "input_zarr": input_zarr,
    "cellprofiler_reference": reference,
    "comparison": comparison,
    "planned_resident_comparison": planned_resident_comparison,
    "execution_config": {
        "requested": requested_execution,
        "maxima_downsample": maxima_downsample,
        "merge_line_basin_pixels": merge_line_basin_pixels,
        "merge_line_max_saddle_drop": merge_line_max_saddle_drop,
        "applied": {
            "workers": requested_workers,
            "chunk_shape": requested_chunk_shape,
            "cache_bytes": requested_cache_bytes,
        },
        "applies_requested_execution_config": plan_probe is not None,
        "reason": "the Blockflow example path is planned execution; chunk, worker and cache settings are consumed by cellprofiler-human and its materialization path",
    },
    "wall_time": {
        "blockflow_pipeline_seconds": None if blockflow is None else blockflow.get("seconds"),
        "cellprofiler_wall_seconds": None if reference is None else reference.get("wall_seconds"),
    },
    "planner_simulator": {
        "status": None if plan_probe is None else plan_probe.get("status", "planned_segmentation_with_measurements_simulated"),
        "observed_pipeline_seconds": None if blockflow is None else blockflow.get("seconds"),
        "estimated_pipeline_seconds": None if plan_probe is None else plan_probe.get("simulator", {}).get("estimated_pipeline_seconds"),
        "estimate_error_ratio": None,
        "scope": None if plan_probe is None else plan_probe.get("scope"),
        "not_included": planner_missing if plan_probe is not None else None,
        "materialized_outputs": plan_materialized,
        "planned_vs_reference": comparison,
        "planned_vs_resident": planned_resident_comparison,
        "plan_probe": plan_probe,
        "reason": "the simulator covers the planned segmentation skeleton, min-distance seed suppression, optional watershed-line basin merging, final object-size filtering and shape/intensity measurement phases; the same planned path materializes the primary Blockflow object table",
    },
}

bf = report["wall_time"]["blockflow_pipeline_seconds"]
cp = report["wall_time"]["cellprofiler_wall_seconds"]
if bf is not None and cp is not None and cp != 0:
    report["wall_time"]["blockflow_over_cellprofiler"] = bf / cp
estimate = report["planner_simulator"]["estimated_pipeline_seconds"]
if bf is not None and estimate is not None and bf != 0:
    report["planner_simulator"]["estimate_error_ratio"] = estimate / bf

with (out_dir / "benchmark-report.json").open("w") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
