#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Export level 0 of label images of an OME-Zarr image as `.npy` masks.

Writes `<out>/<label>.npy` (`bool`, `[z, y, x]`, label value != 0) per label.
Needs numpy and zarr (any Python but the reference's own, which has no zarr).

    export_labels.py --zarr crop.ome.zarr --out DIR vessel-vessel-mask vessel-artery-mask
"""

import argparse
from pathlib import Path

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        array = zarr.open_array(str(args.zarr / "labels" / label / "0"), mode="r")
        mask = np.ascontiguousarray(np.asarray(array[...]) != 0)
        np.save(args.out / f"{label}.npy", mask)
        print(f"{label}: {mask.shape}, {int(mask.sum())} voxels")


if __name__ == "__main__":
    main()
