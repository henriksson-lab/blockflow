#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/object-3d-measurement/blockflow}"
count="${2:-10}"
fixture_dir="${3:-}"
zarr_dir="${4:-.tmp/object-3d-measurement/input.zarr}"

fixture_args=()
if [[ -n "$fixture_dir" ]]; then
  fixture_args=(--fixture-dir "$fixture_dir")
fi

if [[ -x target/release/object-3d-measurement && "${BF_USE_CARGO:-0}" != "1" ]]; then
  target/release/object-3d-measurement \
    --out "$out" \
    --images "$count" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}"
else
  cargo run -p blockflow-object-3d-measurement --bin object-3d-measurement --release -- \
    --out "$out" \
    --images "$count" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}"
fi
