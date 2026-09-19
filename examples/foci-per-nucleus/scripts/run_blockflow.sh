#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/foci-per-nucleus/blockflow}"
count="${2:-10}"

cargo run -p blockflow-foci-per-nucleus --bin foci-per-nucleus --release -- \
  --out "$out" \
  --images "$count"
