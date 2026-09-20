#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/foci-per-nucleus/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/run_blockflow.sh" "$bench/blockflow" "$count" "$bench/input.zarr"
"$script_dir/run_reference.sh" "$bench/reference" "$count"

python3 - "$bench/reference/summary.json" "$bench/blockflow/summary.json" <<'PY'
import json
import sys

left = json.load(open(sys.argv[1]))
right = json.load(open(sys.argv[2]))
for key in ["assigned_foci", "images", "nuclei", "total_foci", "unassigned_foci"]:
    if left[key] != right[key]:
        raise SystemExit(f"summary mismatch for {key}: {left[key]!r} != {right[key]!r}")
PY
diff -u "$bench/reference/nuclei.csv" "$bench/blockflow/nuclei.csv"
diff -u "$bench/reference/foci.csv" "$bench/blockflow/foci.csv"

printf 'foci-per-nucleus benchmark matched for %s image(s): %s\n' "$count" "$bench"
