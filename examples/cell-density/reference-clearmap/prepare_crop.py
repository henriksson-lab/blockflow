#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Extract the crop both sides of the cell comparison read.

Writes `$CELLS_OUT/input.npy`: level 0 of `$CELLS_TILE` over the crop, in the
reference's `(x, y, z)` order (what its pipeline reads from a stitched
`.npy`), and `setup.json`. `cell-density` reads the same voxels from the
OME-Zarr tile itself with `--crop-start/--crop-shape`.

Needs a Python with `zarr` (3.x) and numpy; not the reference's environment.
"""
import json
import os
from pathlib import Path

import numpy as np
import zarr

TILE = os.environ.get("CELLS_TILE", "/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_03x02.zarr")
OUT = Path(os.environ.get("CELLS_OUT", "/husky/henriksson/lightsheet-reference/cells/crop"))


def triplet(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",")]


def main():
    array = zarr.open_array(os.path.join(TILE, "0"), mode="r")
    if array.ndim != 3:
        raise SystemExit(f"expected a (z, y, x) level 0, got shape {array.shape}")
    start = triplet("CELLS_CROP_START", "400,248,246")
    shape = triplet("CELLS_CROP_SHAPE", "256,512,512")
    # A non-positive shape means "to the end of the axis".
    shape = [s if s > 0 else array.shape[d] - start[d] for d, s in enumerate(shape)]
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "input.npy"
    window = tuple(slice(a, a + s) for a, s in zip(start, shape))
    if not dst.exists():
        block = np.asarray(array[window])  # (z, y, x)
        np.save(dst, np.asfortranarray(block.transpose(2, 1, 0)))
        print("wrote", dst, block.shape, flush=True)
    (OUT / "setup.json").write_text(json.dumps(
        {"tile": TILE, "crop_start_zyx": start, "crop_shape_zyx": shape,
         "voxels": int(np.prod(shape))}, indent=1))


if __name__ == "__main__":
    main()
