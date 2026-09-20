#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/colocalization/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_timed() {
  local label="$1"
  shift
  /usr/bin/time -v -o "$bench/${label}-time.txt" "$@"
}

"$script_dir/fetch_fixture.sh" "$count" "$bench/fixtures"
cargo build -p blockflow-colocalization --release
run_timed blockflow "$script_dir/run_blockflow.sh" "$bench/blockflow" "$count" "$bench/fixtures" "$bench/input.zarr"
run_timed reference "$script_dir/run_reference.sh" "$bench/reference" "$count" "$bench/fixtures"

python3 - "$bench/blockflow/summary.json" "$bench/reference/summary.json" <<'PY'
import json
import sys
left = json.load(open(sys.argv[1]))
right = json.load(open(sys.argv[2]))
for key in ["finite_pairs", "images", "objects", "pairs"]:
    if left[key] != right[key]:
        raise SystemExit(f"summary mismatch for {key}: {left[key]!r} != {right[key]!r}")
PY
diff -u "$bench/reference/objects.csv" "$bench/blockflow/objects.csv"

printf 'colocalization benchmark matched for %s image(s): %s\n' "$count" "$bench"
