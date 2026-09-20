#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/wound-assay/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_timed() {
  local label="$1"
  shift
  /usr/bin/time -v -o "$bench/${label}-time.txt" "$@"
}

"$script_dir/fetch_fixture.sh" "$count" "$bench/fixtures"
cargo build -p blockflow-wound-assay --release
BF_PREPARE_ONLY=1 "$script_dir/run_blockflow.sh" "$bench/preparation" "$count" "$bench/fixtures" "$bench/input.zarr"
run_timed blockflow "$script_dir/run_blockflow.sh" "$bench/blockflow" "$count" "$bench/fixtures" "$bench/input.zarr"
run_timed skimage env BF_REFERENCE=skimage "$script_dir/run_reference.sh" "$bench/skimage" "$count" "$bench/fixtures"
run_timed opencv env BF_REFERENCE=opencv "$script_dir/run_reference.sh" "$bench/opencv" "$count" "$bench/fixtures"

for reference in skimage opencv; do
  python3 - "$bench/blockflow/summary.json" "$bench/$reference/summary.json" <<'PY'
import json
import sys
left = json.load(open(sys.argv[1]))
right = json.load(open(sys.argv[2]))
for key in ["covered_area", "height", "images", "open_area", "open_fraction", "threshold", "width"]:
    if left[key] != right[key]:
        raise SystemExit(f"summary mismatch for {key}: {left[key]!r} != {right[key]!r}")
PY
  diff -u "$bench/$reference/images.csv" "$bench/blockflow/images.csv"
  diff -u "$bench/$reference/profile.csv" "$bench/blockflow/profile.csv"
done

printf 'wound-assay skimage/opencv benchmarks matched for %s image(s): %s\n' "$count" "$bench"
