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
BF_PREPARE_ONLY=1 "$script_dir/run_blockflow.sh" "$bench/preparation" "$count" "$bench/fixtures" "$bench/input.zarr"
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
python3 - "$bench/reference/objects.csv" "$bench/blockflow/objects.csv" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="") as handle:
    reference = list(csv.DictReader(handle))
with open(sys.argv[2], newline="") as handle:
    blockflow = list(csv.DictReader(handle))
if len(reference) != len(blockflow):
    raise SystemExit(f"object row count mismatch: {len(reference)} != {len(blockflow)}")
for index, (left, right) in enumerate(zip(reference, blockflow)):
    if left.keys() != right.keys():
        raise SystemExit(f"object row {index} has different columns")
    for column in left:
        if column in {"image", "label", "count", "finite_count"}:
            same = left[column] == right[column]
        else:
            same = math.isclose(float(left[column]), float(right[column]), rel_tol=0.0, abs_tol=2e-5)
        if not same:
            raise SystemExit(f"object row {index} {column}: {left[column]} != {right[column]}")
PY

printf 'colocalization benchmark matched for %s image(s): %s\n' "$count" "$bench"
