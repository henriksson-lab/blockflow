#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/percent-positive/blockflow}"
count="${2:-10}"
threshold="${BF_THRESHOLD:-110}"
zarr_dir="${3:-.tmp/percent-positive/input.zarr}"

cargo run -p blockflow-percent-positive --bin percent-positive --release -- \
  --out "$out" \
  --images "$count" \
  --threshold "$threshold" \
  --zarr-dir "$zarr_dir"
