#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/percent-positive/reference}"
count="${2:-10}"
threshold="${BF_THRESHOLD:-110}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$script_dir/../reference-python/percent_positive_reference.py" \
  --out "$out" \
  --images "$count" \
  --threshold "$threshold"
