#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/foci-per-nucleus/blockflow}"
count="${2:-10}"
zarr_dir="${3:-.tmp/foci-per-nucleus/input.zarr}"

prepare_args=()
if [[ "${BF_PREPARE_ONLY:-0}" == "1" ]]; then
  prepare_args=(--prepare-only)
fi

cargo run -p blockflow-foci-per-nucleus --bin foci-per-nucleus --release -- \
  --out "$out" \
  --images "$count" \
  --zarr-dir "$zarr_dir" \
  "${prepare_args[@]}"
