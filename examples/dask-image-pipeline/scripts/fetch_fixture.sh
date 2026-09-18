#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
out_dir="${2:-.tmp/dask-image-pipeline/images}"
shape="${BF_FIXTURE_SHAPE:-1024x1024}"

mkdir -p "$out_dir"

python3 - "$count" "$out_dir" "$shape" <<'PY'
import math
import struct
import sys
from pathlib import Path

count = int(sys.argv[1])
out_dir = Path(sys.argv[2])
height, width = (int(part) for part in sys.argv[3].lower().split("x"))

def pixel(x, y, n):
    background = 18 + ((x * 7 + y * 11 + n * 13) % 19)
    value = background
    tile = 256
    centers = []
    for ty in range(max(1, height // tile)):
        for tx in range(max(1, width // tile)):
            base_x = tx * tile
            base_y = ty * tile
            centers.extend([
                (base_x + 54 + (n * 3 + tx * 5) % 17, base_y + 58 + (n * 5 + ty * 3) % 19, 15, 205),
                (base_x + 139 + (n * 7 + ty * 2) % 23, base_y + 78 + (n * 2 + tx * 7) % 13, 19, 230),
                (base_x + 88 + (n * 5 + tx * 3) % 29, base_y + 169 + (n * 3 + ty * 5) % 17, 13, 190),
                (base_x + 181 + (n * 2 + ty * 11) % 11, base_y + 177 + (n * 7 + tx * 2) % 23, 21, 220),
                (base_x + 205 - (n * 3 + tx * 2) % 31, base_y + 45 + (n * 11 + ty * 3) % 37, 10, 180),
            ])
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
