#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/colocalization/reference}"
count="${2:-10}"
fixture_dir="${3:-.tmp/colocalization/fixtures}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$script_dir/../reference-python/colocalization_reference.py" \
  --out "$out" \
  --images "$count" \
  --fixture-dir "$fixture_dir"
