#!/usr/bin/env bash
set -euo pipefail

out="${1:-.tmp/wound-assay/reference}"
count="${2:-10}"
threshold="${BF_THRESHOLD:-100}"
fixture_dir="${3:-.tmp/wound-assay/fixtures}"
framework="${BF_REFERENCE:-skimage}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$framework" in
  skimage)
    reference="$script_dir/../reference-skimage/wound_assay_skimage.py"
    ;;
  opencv)
    reference="$script_dir/../reference-opencv/wound_assay_opencv.py"
    ;;
  *)
    echo "unknown BF_REFERENCE=$framework; expected skimage or opencv" >&2
    exit 2
    ;;
esac

python3 "$reference" \
  --out "$out" \
  --images "$count" \
  --threshold "$threshold" \
  --fixture-dir "$fixture_dir"
