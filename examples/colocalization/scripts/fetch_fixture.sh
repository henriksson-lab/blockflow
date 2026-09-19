#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
out_dir="${2:-.tmp/colocalization/fixtures}"

mkdir -p "$out_dir"

python3 - "$count" "$out_dir" <<'PY'
import csv
import sys
from pathlib import Path

count = int(sys.argv[1])
out_dir = Path(sys.argv[2])
objects_per_image = 4


def object_rect(image, local):
    row = local // 2
    col = local % 2
    y0 = 9 + row * 31 + (image % 4)
    x0 = 11 + col * 42 + ((image + local) % 5)
    height = 19 + ((image + local) % 5)
    width = 22 + ((2 * image + local) % 6)
    return y0, x0, height, width


for image in range(count):
    with (out_dir / f"objects-{image:03}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "local", "y0", "x0", "height", "width"])
        for local in range(objects_per_image):
            label = image * 100 + local + 1
            writer.writerow([label, local, *object_rect(image, local)])

print(f"wrote {count} colocalization fixture files to {out_dir}")
PY
