#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/wound-assay/blockflow}"
count="${2:-10}"
threshold="${BF_THRESHOLD:-100}"
fixture_dir="${3:-}"
zarr_dir="${4:-.tmp/wound-assay/input.zarr}"

prepare_args=()
if [[ "${BF_PREPARE_ONLY:-0}" == "1" ]]; then
  prepare_args=(--prepare-only)
fi

fixture_args=()
if [[ -n "$fixture_dir" ]]; then
  fixture_args=(--fixture-dir "$fixture_dir")
fi

if [[ -x target/release/wound-assay && "${BF_USE_CARGO:-0}" != "1" ]]; then
  target/release/wound-assay \
    --out "$out" \
    --images "$count" \
    --threshold "$threshold" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}" \
    "${prepare_args[@]}"
else
  cargo run -p blockflow-wound-assay --bin wound-assay --release -- \
    --out "$out" \
    --images "$count" \
    --threshold "$threshold" \
    --zarr-dir "$zarr_dir" \
    "${fixture_args[@]}" \
    "${prepare_args[@]}"
fi
