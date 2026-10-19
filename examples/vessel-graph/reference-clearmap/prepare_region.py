#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Cut a region of both channels of one OME-Zarr tile into `.npy` files.

Writes `<out>/source_c<k>.npy` (`uint16`, `[z, y, x]`) and `region.json`. Needs
numpy and zarr (any Python but the reference's own, which has no zarr).

    prepare_region.py --tile T.zarr --out DIR --z 1600:1664 --y 400:656 --x 74:330
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def span(text):
    start, stop = (int(part) for part in text.split(":"))
    return slice(start, stop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--z", required=True, type=span)
    parser.add_argument("--y", required=True, type=span)
    parser.add_argument("--x", required=True, type=span)
    args = parser.parse_args()
    array = zarr.open_array(str(args.tile / "0"), mode="r")
    if array.ndim != 4:
        raise SystemExit(f"expected c, z, y, x; the tile has shape {array.shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    for channel in range(array.shape[0]):
        region = np.ascontiguousarray(array[channel, args.z, args.y, args.x])
        np.save(args.out / f"source_c{channel}.npy", region)
        print(f"channel {channel}: {region.shape} {region.dtype}, "
              f"p50 {np.percentile(region, 50):.0f}, p99 {np.percentile(region, 99):.0f}")
    (args.out / "region.json").write_text(json.dumps({
        "tile": str(args.tile),
        "zyx_start": [args.z.start, args.y.start, args.x.start],
        "zyx_stop": [args.z.stop, args.y.stop, args.x.stop],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
