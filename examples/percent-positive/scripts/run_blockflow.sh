#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/percent-positive/blockflow}"
count="${2:-10}"
threshold="${BF_THRESHOLD:-110}"

cargo run -p blockflow-percent-positive --bin percent-positive --release -- \
  --out "$out" \
  --images "$count" \
  --threshold "$threshold"
