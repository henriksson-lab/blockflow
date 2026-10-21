#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Untimed input preparation for the reference side of the benchmark.

The reference reads `.npy` (or TIFF) stacks in its `(x, y, z)` order, which
its own pipeline writes when it converts the microscope's files; both sides
start from the same OME-Zarr voxels, so this extracts them. Every target is
written once and kept (a `.json` beside it records what it holds); a target
whose record differs is rewritten.

    prepare.py tiles --tile-dir D --level L [--z A:B] [--rows A:B] [--columns A:B]
                     [--channel C] --out DIR
        `DIR/npy/tile_YYxXX.npy` per tile (grid indices renumbered from 0,
        as the reference's tile expression wants) and `DIR/setup.json`.
    prepare.py box --image I [--channel C] --level L --start Z,Y,X --shape Z,Y,X --out FILE.npy
        One box (a non-positive extent runs to the end of the axis).
    prepare.py channels --image I --out DIR
        `DIR/source_c<k>.npy` for every channel of level 0.

Arrays are written Fortran-ordered `(x, y, z)`, so the bytes are in the
OME-Zarr's own `[z, y, x]` C order and the reference reads them as it reads
its own conversions. Needs numpy and zarr 3 (not the reference's environment).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import zarr


def span(text):
    if not text:
        return None
    a, b = text.split(":")
    return int(a), int(b)


def triplet(text):
    return [int(v) for v in text.split(",")]


def level_array(image, level):
    return zarr.open_array(str(Path(image) / str(level)), mode="r")


def read_zyx(array, channel, window):
    """`window` over the last three axes; a leading channel axis takes `channel`."""
    if array.ndim == 4:
        return np.asarray(array[(channel,) + window])
    if array.ndim == 3:
        return np.asarray(array[window])
    raise SystemExit(f"expected [c,] z, y, x, got shape {array.shape}")


def save_xyz(path, zyx, record):
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists() and json.loads(meta.read_text()) == record:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asfortranarray(zyx.transpose(2, 1, 0)))
    meta.write_text(json.dumps(record, indent=1))
    return True


def tiles(args):
    out = Path(args.out)
    pattern = re.compile(r"tile_(\d+)x(\d+)")
    found = {}
    for path in sorted(Path(args.tile_dir).iterdir()):
        m = pattern.match(path.name)
        if m:
            found[(int(m.group(1)), int(m.group(2)))] = path
    rows = sorted({r for r, _ in found})
    cols = sorted({c for _, c in found})
    if args.rows:
        rows = [r for r in rows if args.rows[0] <= r < args.rows[1]]
    if args.columns:
        cols = [c for c in cols if args.columns[0] <= c < args.columns[1]]
    entries = []
    for iy, row in enumerate(rows):
        for ix, col in enumerate(cols):
            source = found[(row, col)]
            array = level_array(source, args.level)
            nz = array.shape[-3]
            z = args.z or (0, nz)
            window = (slice(z[0], min(z[1], nz)), slice(None), slice(None))
            record = {"source": str(source), "level": args.level, "channel": args.channel,
                      "z": list(z), "shape_zyx": [min(z[1], nz) - z[0]] + list(array.shape[-2:])}
            dst = out / "npy" / f"tile_{iy:02d}x{ix:02d}.npy"
            meta = dst.with_suffix(".json")
            if not (dst.exists() and meta.exists() and json.loads(meta.read_text()) == record):
                save_xyz(dst, read_zyx(array, args.channel, window), record)
                print("wrote", dst, record["shape_zyx"], flush=True)
            entries.append({"name": source.name, "grid_yx": [iy, ix], "row": row, "column": col,
                            "shape_zyx": record["shape_zyx"]})
    (out / "setup.json").write_text(json.dumps(
        {"tile_dir": args.tile_dir, "level": args.level, "channel": args.channel, "z": args.z,
         "rows": rows, "columns": cols, "tiles": entries}, indent=1))


def box(args):
    array = level_array(args.image, args.level)
    start, shape = triplet(args.start), triplet(args.shape)
    dims = array.shape[-3:]
    shape = [s if s > 0 else dims[d] - start[d] for d, s in enumerate(shape)]
    window = tuple(slice(a, a + s) for a, s in zip(start, shape))
    record = {"image": args.image, "level": args.level, "channel": args.channel,
              "start_zyx": start, "shape_zyx": shape, "voxels": int(np.prod(shape))}
    dst = Path(args.out)
    meta = dst.with_suffix(".json")
    if dst.exists() and meta.exists() and json.loads(meta.read_text()) == record:
        return
    save_xyz(dst, read_zyx(array, args.channel, window), record)
    print("wrote", dst, shape, flush=True)


def channels(args):
    array = level_array(args.image, 0)
    if array.ndim != 4:
        raise SystemExit(f"expected c, z, y, x, got {array.shape}")
    out = Path(args.out)
    for c in range(array.shape[0]):
        record = {"image": args.image, "level": 0, "channel": c, "shape_zyx": list(array.shape[1:])}
        dst = out / f"source_c{c}.npy"
        meta = dst.with_suffix(".json")
        if dst.exists() and meta.exists() and json.loads(meta.read_text()) == record:
            continue
        save_xyz(dst, np.asarray(array[c]), record)
        print("wrote", dst, record["shape_zyx"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="what", required=True)
    p = sub.add_parser("tiles")
    p.add_argument("--tile-dir", required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--z", type=span)
    p.add_argument("--rows", type=span)
    p.add_argument("--columns", type=span)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("box")
    p.add_argument("--image", required=True)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--start", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("channels")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"tiles": tiles, "box": box, "channels": channels}[args.what](args)


if __name__ == "__main__":
    main()
