#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/percent-positive/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/run_blockflow.sh" "$bench/blockflow" "$count" "$bench/input.zarr"
"$script_dir/run_reference.sh" "$bench/reference" "$count"

python3 - "$bench/reference/summary.json" "$bench/blockflow/summary.json" <<'PY'
import json
import sys

left = json.load(open(sys.argv[1]))
right = json.load(open(sys.argv[2]))
for key in [
    "images",
    "marker_sum",
    "negative",
    "objects",
    "percent_positive",
    "positive",
    "threshold",
    "total_area",
]:
    if left[key] != right[key]:
        raise SystemExit(f"summary mismatch for {key}: {left[key]!r} != {right[key]!r}")
PY
diff -u "$bench/reference/objects.csv" "$bench/blockflow/objects.csv"

printf 'percent-positive benchmark matched for %s image(s): %s\n' "$count" "$bench"
