#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/foci-per-nucleus/reference}"
count="${2:-10}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$script_dir/../reference-python/foci_per_nucleus_reference.py" \
  --out "$out" \
  --images "$count"
