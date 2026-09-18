#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/skimage-pipeline/bench-${count}}"
mode="${3:-${BF_MODE:-segment}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/fetch_fixture.sh" "$count" "$bench/images"
CARGO_PROFILE_FLAGS="${CARGO_PROFILE_FLAGS:---release}" \
  BF_MODE="$mode" \
  "$script_dir/run_blockflow.sh" "$bench/images" "$bench/blockflow"
BF_MODE="$mode" "$script_dir/run_skimage.sh" "$bench/images" "$bench/skimage"

cargo run -p blockflow-skimage-pipeline --bin skimage-compare --release -- \
  --blockflow "$bench/blockflow/batch-summary.json" \
  --skimage "$bench/skimage/batch-summary.json" \
  --out "$bench/comparison.json" \
  --max-object-delta 0 \
  --max-area-relative-error 0.02
