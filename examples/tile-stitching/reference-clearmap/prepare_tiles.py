#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Extract the same tiles and z-range for both sides of the stitching comparison.

Reads channel 0 of each OME-Zarr tile (the UltraII `Filter0000` channel; its
planes are identical to the original TIFF pages) and writes
- `npy/tile_YYxXX.npy`: the z-range in the reference's `(x, y, z)` order, grid
  indices renumbered from 0, which is what its pipeline stitches from
  (`use_npy: true`);
- `tiles.json`: the tile set for `tile-stitching`, pointing at the OME-Zarr
  tiles themselves.

Needs a Python with `zarr` (3.x) and numpy; not the reference's environment.
"""
import json
import os
from pathlib import Path

import numpy as np
import zarr

TILES = Path(os.environ.get("STITCH_TILES", "/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles"))
OUT = Path(os.environ.get("STITCH_OUT", "/husky/henriksson/lightsheet-reference/stitching"))
Z0, Z1 = (int(v) for v in os.environ.get("STITCH_Z", "1200:1440").split(":"))
ROWS = [int(v) for v in os.environ.get("STITCH_ROWS", "3,4,5").split(",")]   # first index: y
COLS = [int(v) for v in os.environ.get("STITCH_COLS", "4,5,6").split(",")]   # second index: x
# The nominal overlaps (x, y) in voxels, as the reference parses them from the
# original OME-TIFF metadata of Brain-39L (`parse_overlaps` -> (66, 212)).
OVERLAP = [int(v) for v in os.environ.get("STITCH_OVERLAP", "66,212").split(",")]


def main():
    npy = OUT / "npy"
    npy.mkdir(parents=True, exist_ok=True)
    entries = []
    for iy, row in enumerate(ROWS):
        for ix, col in enumerate(COLS):
            name = f"tile_{row:02d}x{col:02d}.zarr"
            entries.append({"path": str(TILES / name), "grid": [ix, iy]})
            dst = npy / f"tile_{iy:02d}x{ix:02d}.npy"
            if dst.exists():
                continue
            array = zarr.open_array(str(TILES / name / "0"), mode="r")
            block = np.asarray(array[0, Z0:Z1])            # (z, y, x)
            np.save(dst, np.asfortranarray(block.transpose(2, 1, 0)))
            print("wrote", dst, block.shape, flush=True)
    tile_set = {"grid_axes": ["x", "y"], "overlap": OVERLAP, "tiles": entries}
    (OUT / "tiles.json").write_text(json.dumps(tile_set, indent=1))
    (OUT / "setup.json").write_text(json.dumps(
        {"tiles": str(TILES), "z_range": [Z0, Z1], "rows": ROWS, "cols": COLS,
         "overlap_xy": OVERLAP}, indent=1))


if __name__ == "__main__":
    main()
