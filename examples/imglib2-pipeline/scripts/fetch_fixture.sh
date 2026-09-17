#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
out_dir="${2:-.tmp/imglib2-pipeline/images}"

mkdir -p "$out_dir"

python3 - "$count" "$out_dir" <<'PY'
import math
import struct
import sys
from pathlib import Path

count = int(sys.argv[1])
out_dir = Path(sys.argv[2])
width = 256
height = 256

def pixel(x, y, n):
    background = 18 + ((x * 7 + y * 11 + n * 13) % 19)
    value = background
    centers = [
        (54 + (n * 3) % 17, 58 + (n * 5) % 19, 15, 205),
        (139 + (n * 7) % 23, 78 + (n * 2) % 13, 19, 230),
        (88 + (n * 5) % 29, 169 + (n * 3) % 17, 13, 190),
        (181 + (n * 2) % 11, 177 + (n * 7) % 23, 21, 220),
        (205 - (n * 3) % 31, 45 + (n * 11) % 37, 10, 180),
    ]
    for cx, cy, radius, intensity in centers:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        if r2 <= radius * radius:
            value += int(intensity * math.exp(-r2 / (2.0 * (radius / 2.1) ** 2)))
    return max(0, min(255, value))

for n in range(count):
    path = out_dir / f"synthetic-{n:03d}.bmp"
    row_stride = ((width + 3) // 4) * 4
    pixel_bytes = row_stride * height
    palette_bytes = 256 * 4
    pixel_offset = 14 + 40 + palette_bytes
    file_size = pixel_offset + pixel_bytes
    with path.open("wb") as handle:
        handle.write(b"BM")
        handle.write(struct.pack("<IHHI", file_size, 0, 0, pixel_offset))
        handle.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 8, 0, pixel_bytes,
                                 2835, 2835, 256, 256))
        for value in range(256):
            handle.write(bytes((value, value, value, 0)))
        for y in range(height - 1, -1, -1):
            row = bytes(pixel(x, y, n) for x in range(width))
            handle.write(row)
            handle.write(b"\0" * (row_stride - width))
print(f"wrote {count} images to {out_dir}")
PY
