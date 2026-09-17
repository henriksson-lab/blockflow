#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_cellprofiler_semantic_sweep.sh IMAGE REFERENCE_OBJECT_CSV OUTPUT_DIR

Runs a small semantic parameter sweep for the CellProfiler-style benchmark.
Each run executes cellprofiler-human, compares the object table against a
CellProfiler-style reference CSV, and writes sweep.csv plus best.json.

Environment variables:
  CARGO_PROFILE_FLAGS Extra cargo build flags, for example "--release".
  BF_SWEEP_SIGMAS     Space-separated sigma values, default "1.0 1.5 2.0".
  BF_SWEEP_DECLUMP_SIGMAS Space-separated declump sigma values, default "1.3488".
  BF_SWEEP_MIN_SIZES  Space-separated min-size values, default "30 50 70".
  BF_SWEEP_SEEDS      Space-separated seed distances, default "5 6 7 8 9".
  BF_SWEEP_MAXIMA_DOWNSAMPLES Space-separated lower-resolution maxima factors, default "3".
  BF_SWEEP_BASINS     Space-separated watershed modes, default "line adjacent".
  BF_SWEEP_DECLUMP    Space-separated declump methods, default "intensity distance".
  BF_SWEEP_MERGE_LINE_BASIN_PIXELS Space-separated resident watershed-line merge thresholds, default "0".
  BF_SWEEP_MERGE_LINE_MAX_SADDLE_DROPS Space-separated saddle-drop guards; use none for unguarded, default "none".
  BF_MAX_SIZE         Maximum object size, default 5027. Set BF_NO_MAX_SIZE=1 to disable.
  BF_THRESHOLD_METHOD Global threshold method, default li.
  BF_THRESHOLD_BINS   Otsu bin count when used, default 256.
  BF_COMPARE_ARGS     Extra arguments passed to cellprofiler-compare.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 2
fi

image="$1"
reference_csv="$2"
output_dir="$3"

if [[ ! -f "$image" ]]; then
  echo "Input image not found: $image" >&2
  exit 1
fi
if [[ ! -f "$reference_csv" ]]; then
  echo "Reference CSV not found: $reference_csv" >&2
  exit 1
fi

mkdir -p "$output_dir"

read -r -a cargo_profile_flags <<< "${CARGO_PROFILE_FLAGS:-}"
read -r -a sigmas <<< "${BF_SWEEP_SIGMAS:-1.0 1.5 2.0}"
read -r -a declump_sigmas <<< "${BF_SWEEP_DECLUMP_SIGMAS:-1.3488}"
read -r -a min_sizes <<< "${BF_SWEEP_MIN_SIZES:-30 50 70}"
read -r -a seeds <<< "${BF_SWEEP_SEEDS:-5 6 7 8 9}"
read -r -a maxima_downsamples <<< "${BF_SWEEP_MAXIMA_DOWNSAMPLES:-3}"
read -r -a basins <<< "${BF_SWEEP_BASINS:-line adjacent}"
read -r -a declump_methods <<< "${BF_SWEEP_DECLUMP:-intensity distance}"
read -r -a merge_line_basin_pixels <<< "${BF_SWEEP_MERGE_LINE_BASIN_PIXELS:-0}"
read -r -a merge_line_max_saddle_drops <<< "${BF_SWEEP_MERGE_LINE_MAX_SADDLE_DROPS:-none}"
read -r -a compare_args <<< "${BF_COMPARE_ARGS:-}"

max_size="${BF_MAX_SIZE:-5027}"
threshold_method="${BF_THRESHOLD_METHOD:-li}"
threshold_bins="${BF_THRESHOLD_BINS:-256}"

cargo build --features cellprofiler-benchmark \
  --bin cellprofiler-human \
  --bin cellprofiler-compare \
  "${cargo_profile_flags[@]}"

profile_dir="debug"
if [[ " ${cargo_profile_flags[*]} " == *" --release "* ]]; then
  profile_dir="release"
fi

size_args=()
if [[ "${BF_NO_MAX_SIZE:-}" == "1" ]]; then
  size_args+=(--no-max-size)
else
  size_args+=(--max-size "$max_size")
fi

python3 - "$image" "$reference_csv" "$output_dir" "$profile_dir" \
  "$threshold_method" "$threshold_bins" \
  "${sigmas[*]}" -- "${declump_sigmas[*]}" -- "${min_sizes[*]}" -- "${seeds[*]}" -- "${maxima_downsamples[*]}" -- "${basins[*]}" -- "${declump_methods[*]}" -- "${merge_line_basin_pixels[*]}" -- "${merge_line_max_saddle_drops[*]}" -- \
  "${size_args[@]}" -- "${compare_args[@]}" <<'PY'
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

image = sys.argv[1]
reference_csv = sys.argv[2]
output_dir = Path(sys.argv[3])
profile_dir = sys.argv[4]
threshold_method = sys.argv[5]
threshold_bins = sys.argv[6]

first_sep = sys.argv.index("--", 7)
second_sep = sys.argv.index("--", first_sep + 1)
third_sep = sys.argv.index("--", second_sep + 1)
fourth_sep = sys.argv.index("--", third_sep + 1)
fifth_sep = sys.argv.index("--", fourth_sep + 1)
sixth_sep = sys.argv.index("--", fifth_sep + 1)
seventh_sep = sys.argv.index("--", sixth_sep + 1)
eighth_sep = sys.argv.index("--", seventh_sep + 1)
ninth_sep = sys.argv.index("--", eighth_sep + 1)
tenth_sep = sys.argv.index("--", ninth_sep + 1)
sigmas = sys.argv[7:first_sep][0].split()
declump_sigmas = sys.argv[first_sep + 1:second_sep][0].split()
min_sizes = sys.argv[second_sep + 1:third_sep][0].split()
seeds = sys.argv[third_sep + 1:fourth_sep][0].split()
maxima_downsamples = sys.argv[fourth_sep + 1:fifth_sep][0].split()
basins = sys.argv[fifth_sep + 1:sixth_sep][0].split()
declump_methods = sys.argv[sixth_sep + 1:seventh_sep][0].split()
merge_line_thresholds = sys.argv[seventh_sep + 1:eighth_sep][0].split()
merge_line_saddle_drops = sys.argv[eighth_sep + 1:ninth_sep][0].split()
size_args = sys.argv[ninth_sep + 1:tenth_sep]
compare_args = sys.argv[tenth_sep + 1:]

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

human_bin = Path("target") / profile_dir / "cellprofiler-human"
compare_bin = Path("target") / profile_dir / "cellprofiler-compare"
rows = []
run_index = 0

for sigma in sigmas:
    for declump_sigma in declump_sigmas:
        for min_size in min_sizes:
            for seed in seeds:
                for maxima_downsample in maxima_downsamples:
                    if int(maxima_downsample) < 1:
                        raise SystemExit(f"invalid maxima downsample factor: {maxima_downsample}")
                    for basin in basins:
                        if basin not in {"line", "adjacent"}:
                            raise SystemExit(f"unknown watershed basin mode: {basin}")
                        for declump_method in declump_methods:
                            if declump_method not in {"intensity", "distance"}:
                                raise SystemExit(f"unknown declump method: {declump_method}")
                            for merge_line_threshold in merge_line_thresholds:
                                if int(merge_line_threshold) < 0:
                                    raise SystemExit(
                                        f"invalid watershed-line merge threshold: {merge_line_threshold}"
                                    )
                                for merge_saddle_drop in merge_line_saddle_drops:
                                    saddle_args = []
                                    if merge_saddle_drop != "none":
                                        float(merge_saddle_drop)
                                        saddle_args = [
                                            "--merge-line-max-saddle-drop",
                                            merge_saddle_drop,
                                        ]
                                    basin_args = ["--adjacent-basins"] if basin == "adjacent" else []
                                    merge_args = (
                                        []
                                        if merge_line_threshold == "0"
                                        else ["--merge-line-basin-pixels", merge_line_threshold]
                                    )
                                    run_dir = output_dir / f"run-{run_index:03d}"
                                    run_dir.mkdir()
                                    blockflow_dir = run_dir / "blockflow"
                                    comparison_path = run_dir / "comparison.json"
                                    subprocess.run(
                                        [
                                            str(human_bin),
                                            "--input",
                                            image,
                                            "--out",
                                            str(blockflow_dir),
                                            "--min-size",
                                            min_size,
                                            *size_args,
                                            "--sigma",
                                            sigma,
                                            "--declump-sigma",
                                            declump_sigma,
                                            "--threshold-method",
                                            threshold_method,
                                            "--threshold-bins",
                                            threshold_bins,
                                            "--seed-min-distance",
                                            seed,
                                            "--maxima-downsample",
                                            maxima_downsample,
                                            "--declump-method",
                                            declump_method,
                                            *basin_args,
                                            *merge_args,
                                            *saddle_args,
                                        ],
                                        check=True,
                                        stdout=subprocess.DEVNULL,
                                    )
                                    subprocess.run(
                                        [
                                            str(compare_bin),
                                            "--blockflow",
                                            str(blockflow_dir / "objects.csv"),
                                            "--reference",
                                            reference_csv,
                                            "--out",
                                            str(comparison_path),
                                            *compare_args,
                                        ],
                                        check=True,
                                        stdout=subprocess.DEVNULL,
                                    )
                                    comparison = json.loads(comparison_path.read_text())
                                    metrics = comparison["metrics"]
                                    failures = comparison.get("failure_summary", {})
                                    row = {
                                        "run": run_index,
                                        "sigma": sigma,
                                        "declump_sigma": declump_sigma,
                                        "min_size": min_size,
                                        "seed_min_distance": seed,
                                        "maxima_downsample": maxima_downsample,
                                        "watershed_basins": basin,
                                        "declump_method": declump_method,
                                        "merge_line_basin_pixels": merge_line_threshold,
                                        "merge_line_max_saddle_drop": merge_saddle_drop,
                                        "objects": comparison["blockflow_objects"],
                                        "reference_objects": comparison["reference_objects"],
                                        "count_delta": comparison["blockflow_objects"] - comparison["reference_objects"],
                                        "matched": comparison["matched_objects"],
                                        "passed": comparison["passed"],
                                        "mean_centroid_distance": metrics["mean_centroid_distance"],
                                        "max_centroid_distance": metrics["max_centroid_distance"],
                                        "mean_area_relative_error": metrics["mean_area_relative_error"],
                                        "max_area_relative_error": metrics["max_area_relative_error"],
                                        "mean_mean_intensity_relative_error": metrics[
                                            "mean_mean_intensity_relative_error"
                                        ],
                                        "max_mean_intensity_relative_error": metrics[
                                            "max_mean_intensity_relative_error"
                                        ],
                                        "unmatched_blockflow": failures.get("unmatched_blockflow"),
                                        "unmatched_reference": failures.get("unmatched_reference"),
                                        "area_threshold_failures": failures.get("area_threshold_failures"),
                                        "mean_intensity_threshold_failures": failures.get(
                                            "mean_intensity_threshold_failures"
                                        ),
                                    }
                                    row["_score"] = (
                                        row["unmatched_blockflow"] + row["unmatched_reference"],
                                        row["area_threshold_failures"],
                                        row["mean_area_relative_error"],
                                        row["mean_centroid_distance"],
                                    )
                                    rows.append(row)
                                    run_index += 1

rows.sort(key=lambda row: row["_score"])
fieldnames = [key for key in rows[0] if key != "_score"]
with (output_dir / "sweep.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({key: value for key, value in row.items() if key != "_score"} for row in rows)
with (output_dir / "best.json").open("w") as handle:
    json.dump(
        [{key: value for key, value in row.items() if key != "_score"} for row in rows[:10]],
        handle,
        indent=2,
    )
    handle.write("\n")
print(json.dumps([{key: value for key, value in row.items() if key != "_score"} for row in rows[:10]], indent=2))
PY
