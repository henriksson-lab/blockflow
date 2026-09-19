#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
bench="${2:-.tmp/foci-per-nucleus/bench-${count}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/run_blockflow.sh" "$bench/blockflow" "$count"
"$script_dir/run_reference.sh" "$bench/reference" "$count"

diff -u "$bench/reference/summary.json" "$bench/blockflow/summary.json"
diff -u "$bench/reference/nuclei.csv" "$bench/blockflow/nuclei.csv"
diff -u "$bench/reference/foci.csv" "$bench/blockflow/foci.csv"

printf 'foci-per-nucleus benchmark matched for %s image(s): %s\n' "$count" "$bench"
