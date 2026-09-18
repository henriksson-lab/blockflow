#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/dask-image-pipeline/bench-${count}}"
mode="${3:-${BF_MODE:-segment}}"
chunk="${4:-${BF_CHUNK:-256x256}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/fetch_fixture.sh" "$count" "$bench/images"
CARGO_PROFILE_FLAGS="${CARGO_PROFILE_FLAGS:---release}" \
  BF_MODE="$mode" \
  "$script_dir/run_blockflow.sh" "$bench/images" "$bench/blockflow"
BF_MODE="$mode" BF_CHUNK="$chunk" "$script_dir/run_dask_image.sh" "$bench/images" "$bench/dask-image"

cargo run -p blockflow-dask-image-pipeline --bin dask-image-compare --release -- \
  --blockflow "$bench/blockflow/batch-summary.json" \
  --dask-image "$bench/dask-image/batch-summary.json" \
  --out "$bench/comparison.json" \
  --max-object-delta 0 \
  --max-area-relative-error 0.02
