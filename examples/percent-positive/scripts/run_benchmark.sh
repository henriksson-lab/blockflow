#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/percent-positive/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/run_blockflow.sh" "$bench/blockflow" "$count"
"$script_dir/run_reference.sh" "$bench/reference" "$count"

diff -u "$bench/reference/summary.json" "$bench/blockflow/summary.json"
diff -u "$bench/reference/objects.csv" "$bench/blockflow/objects.csv"

printf 'percent-positive benchmark matched for %s image(s): %s\n' "$count" "$bench"
