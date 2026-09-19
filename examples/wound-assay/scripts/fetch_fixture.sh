#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
out_dir="${2:-.tmp/wound-assay/fixtures}"

mkdir -p "$out_dir"

python3 - "$count" "$out_dir" <<'PY'
import sys
from pathlib import Path

count = int(sys.argv[1])
out_dir = Path(sys.argv[2])
height = 96
width = 144


def wound_bounds(image, y):
    center = width // 2 + (image % 5) - 2
    half = 13 + (image % 4) + ((y + image) % 9) // 3
    drift = ((y * 7 + image * 3) % 11) - 5
    left = center - half + int(drift / 2)
    right = center + half + int(drift / 3)
    return max(0, left), min(width, right)


def intensity(image, y, x):
    left, right = wound_bounds(image, y)
    if left <= x < right:
        return 42 + ((x + 3 * y + image) % 17)
    return 166 + ((2 * x + y + image) % 29)


for image in range(count):
    path = out_dir / f"wound-{image:03}.pgm"
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(bytes(intensity(image, y, x) for y in range(height) for x in range(width)))

print(f"wrote {count} wound fixture images to {out_dir}")
PY
