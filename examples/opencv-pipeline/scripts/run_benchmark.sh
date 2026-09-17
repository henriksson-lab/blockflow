#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/opencv-pipeline/bench-${count}}"
mode="${3:-${BF_MODE:-segment}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/fetch_fixture.sh" "$count" "$bench/images"
CARGO_PROFILE_FLAGS="${CARGO_PROFILE_FLAGS:---release}" \
  BF_MODE="$mode" \
  "$script_dir/run_blockflow.sh" "$bench/images" "$bench/blockflow"
BF_MODE="$mode" "$script_dir/run_opencv.sh" "$bench/images" "$bench/opencv"

cargo run -p blockflow-opencv-pipeline --bin opencv-compare --release -- \
  --blockflow "$bench/blockflow/batch-summary.json" \
  --opencv "$bench/opencv/batch-summary.json" \
  --out "$bench/comparison.json" \
  --max-object-delta 0 \
  --max-area-relative-error 0.02
