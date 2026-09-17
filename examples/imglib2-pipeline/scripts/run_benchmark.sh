#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/imglib2-pipeline/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/fetch_fixture.sh" "$count" "$bench/images"
CARGO_PROFILE_FLAGS="${CARGO_PROFILE_FLAGS:---release}" \
  "$script_dir/run_blockflow.sh" "$bench/images" "$bench/blockflow"
"$script_dir/run_imglib2.sh" "$bench/images" "$bench/imglib2"

cargo run -p blockflow-imglib2-pipeline --bin imglib2-compare --release -- \
  --blockflow "$bench/blockflow/batch-summary.json" \
  --imglib2 "$bench/imglib2/batch-summary.json" \
  --out "$bench/comparison.json" \
  --max-object-delta 0 \
  --max-area-relative-error 0.01
