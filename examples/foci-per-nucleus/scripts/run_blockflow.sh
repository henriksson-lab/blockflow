#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/foci-per-nucleus/blockflow}"
count="${2:-10}"
zarr_dir="${3:-.tmp/foci-per-nucleus/input.zarr}"

cargo run -p blockflow-foci-per-nucleus --bin foci-per-nucleus --release -- \
  --out "$out" \
  --images "$count" \
  --zarr-dir "$zarr_dir"
