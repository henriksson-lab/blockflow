#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/object-3d-measurement/reference}"
count="${2:-10}"
fixture_dir="${3:-.tmp/object-3d-measurement/fixtures}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "$script_dir/../reference-skimage/object_3d_measurement_skimage.py" \
  --out "$out" \
  --images "$count" \
  --fixture-dir "$fixture_dir"
