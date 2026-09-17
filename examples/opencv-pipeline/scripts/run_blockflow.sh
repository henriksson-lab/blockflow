#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  examples/opencv-pipeline/scripts/run_blockflow.sh INPUT_DIR OUTPUT_DIR

Runs the Blockflow side of the OpenCV comparison pipeline over every .bmp image
in INPUT_DIR.

Environment variables:
  CARGO_PROFILE_FLAGS Extra cargo build flags, for example "--release".
  BF_SIGMA        Gaussian sigma, default 1.5.
  BF_MIN_SIZE     Minimum component size, default 20.
  BF_MODE         Pipeline mode, segment or transform, default segment.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

input_dir="$1"
output_dir="$2"
sigma="${BF_SIGMA:-1.5}"
min_size="${BF_MIN_SIZE:-20}"
mode="${BF_MODE:-segment}"
read -r -a cargo_profile_flags <<< "${CARGO_PROFILE_FLAGS:-${CARGO_BIN_FLAGS:-}}"

cargo build -p blockflow-opencv-pipeline --bin opencv-pipeline "${cargo_profile_flags[@]}"
profile_dir="debug"
if [[ " ${cargo_profile_flags[*]} " == *" --release "* ]]; then
  profile_dir="release"
fi
blockflow_bin="target/$profile_dir/opencv-pipeline"

mkdir -p "$output_dir/runs"

start="${EPOCHREALTIME:-$(date +%s)}"
i=0
for image in "$input_dir"/*.bmp; do
  run_dir="$output_dir/runs/run-$(printf "%03d" "$i")"
  "$blockflow_bin" \
    --input "$image" \
    --out "$run_dir" \
    --sigma "$sigma" \
    --min-size "$min_size" \
    --mode "$mode" >/dev/null
  i=$((i + 1))
done
end="${EPOCHREALTIME:-$(date +%s)}"

python3 - "$output_dir" "$i" "$start" "$end" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
count = int(sys.argv[2])
start = float(sys.argv[3])
end = float(sys.argv[4])
objects = 0
area = 0
pipeline = 0.0
for path in sorted((out / "runs").glob("run-*/summary.json")):
    summary = json.loads(path.read_text())
    objects += summary["objects"]
    area += summary["total_foreground_area"]
    pipeline += summary["pipeline_seconds"]
report = {
    "runner": "blockflow",
    "images": count,
    "mode": json.loads((out / "runs" / "run-000" / "summary.json").read_text()).get("mode", "segment") if count else "segment",
    "wall_seconds": end - start,
    "pipeline_seconds_sum": pipeline,
    "objects": objects,
    "total_foreground_area": area,
}
(out / "batch-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
