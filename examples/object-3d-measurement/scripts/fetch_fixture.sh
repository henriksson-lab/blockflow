#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
out_dir="${2:-.tmp/object-3d-measurement/fixtures}"

mkdir -p "$out_dir"

python3 - "$count" "$out_dir" <<'PY'
import csv
import sys
from pathlib import Path

count = int(sys.argv[1])
out_dir = Path(sys.argv[2])
objects_per_image = 4


def object_box(image, local):
    start = [
        2 + (local % 2) * 13 + (image % 2),
        4 + (local // 2) * 20 + (image % 3),
        5 + (local % 2) * 25 + ((image + local) % 4),
    ]
    extent = [
        5 + ((image + local) % 4),
        8 + ((2 * image + local) % 5),
        9 + ((image + 2 * local) % 6),
    ]
    return start, extent


for image in range(count):
    with (out_dir / f"boxes-{image:03}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "z0", "y0", "x0", "dz", "dy", "dx"])
        for local in range(objects_per_image):
            label = image * 100 + local + 1
            start, extent = object_box(image, local)
            writer.writerow([label, *start, *extent])

print(f"wrote {count} 3-D fixture files to {out_dir}")
PY
