#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Small helpers for `run.sh` (standard library only).

    pipeline.py region POSITIONS --tile NAME --box Z0:Z1,Y0:Y1,X0:X1
        A box given in one tile's own level-0 voxels, as a box of the fused
        volume at level 0: the tile's anchor (plane 0 along z, its median
        per-slice position in-plane, minus the origin), in-plane times the
        ratio of the positions' voxel size to the tile's level 0 (2 for level
        1), plus the box. Prints `Z0:Z1,Y0:Y1,X0:X1`. Uses the same anchor rule as
        tile-stitching's `relayout::anchor`, so the box holds the tile's
        voxels up to its per-slice wobble about the median.

    pipeline.py stage TIME_FILE DIR
        One stage's wall time (s), peak resident memory (GB) and the disk use
        of DIR (GB), from `/usr/bin/time -v` output, as JSON.
"""
import json
import os
import sys


def anchor(slices):
    ys = sorted(p[1] for p in slices)
    xs = sorted(p[2] for p in slices)
    return [slices[0][0], ys[(len(ys) - 1) // 2], xs[(len(xs) - 1) // 2]]


def region(argv):
    positions = json.load(open(argv[0]))
    opts = dict(zip(argv[1::2], argv[2::2]))
    name, box = opts["--tile"], opts["--box"]
    matches = [t for t in positions["tiles"] if os.path.basename(t["path"].rstrip("/")).startswith(f"tile_{name}")
               or f"[{name.replace('x', ' x ')}]" in t["path"]]
    if len(matches) != 1:
        sys.exit(f"pipeline.py: {len(matches)} tiles match {name!r}")
    tile = matches[0]
    slices = tile.get("slices") or [[tile["rigid"][0] + k, tile["rigid"][1], tile["rigid"][2]]
                                    for k in range(tile["shape"][0])]
    group = json.load(open(os.path.join(tile["path"], "zarr.json")))
    multiscale = (group["attributes"].get("ome") or group["attributes"])["multiscales"][0]
    scale0 = next(t["scale"] for t in multiscale["datasets"][0]["coordinateTransformations"]
                  if t["type"] == "scale")
    factor = positions["voxel_size"][2] / scale0[-1]
    a = anchor(slices)
    origin = positions["origin"]
    start = [a[0] - origin[0],
             round(a[1] * factor) - round(origin[1] * factor),
             round(a[2] * factor) - round(origin[2] * factor)]
    ranges = [tuple(int(v) for v in r.split(":")) for r in box.split(",")]
    print(",".join(f"{s + lo}:{s + hi}" for s, (lo, hi) in zip(start, ranges)))


def stage(argv):
    time_file, directory = argv
    wall = rss = None
    for line in open(time_file):
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            parts = [float(p) for p in line.rsplit(" ", 1)[1].split(":")]
            wall = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        elif line.startswith("Maximum resident set size"):
            rss = int(line.rsplit(" ", 1)[1]) / 1e6
    size = 0
    for root, _, files in os.walk(directory):
        for f in files:
            try:
                size += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    print(json.dumps({"wall_s": wall, "peak_rss_gb": rss, "disk_gb": size / 1e9}))


if __name__ == "__main__":
    {"region": region, "stage": stage}[sys.argv[1]](sys.argv[2:])
