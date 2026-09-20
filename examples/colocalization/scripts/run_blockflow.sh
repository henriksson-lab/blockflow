#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/colocalization/blockflow}"
count="${2:-10}"
fixture_dir="${3:-}"
zarr_dir="${4:-.tmp/colocalization/input.zarr}"

fixture_args=()
if [[ -n "$fixture_dir" ]]; then
  fixture_args=(--fixture-dir "$fixture_dir")
fi

if [[ -x target/release/colocalization && "${BF_USE_CARGO:-0}" != "1" ]]; then
  target/release/colocalization \
    --out "$out" \
    --images "$count" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}"
else
  cargo run -p blockflow-colocalization --bin colocalization --release -- \
    --out "$out" \
    --images "$count" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}"
fi
