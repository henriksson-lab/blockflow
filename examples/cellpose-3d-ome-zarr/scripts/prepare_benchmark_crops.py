#!/usr/bin/env python3
"""Extract the fixed real-data crops used by the 3D CUDA benchmarks."""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr


CROPS = {
    "dense-bench": ((0, 16), (560, 624), (960, 1024)),
    "sparse-bench": ((0, 16), (1064, 1128), (2416, 2480)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series", default="0")
    parser.add_argument("--level", default="0")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--time", type=int, default=0)
    args = parser.parse_args()

    array = zarr.open_array(str(args.zarr / args.series / args.level), mode="r")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (z, y, x) in CROPS.items():
        volume = np.asarray(
            array[args.time, args.channel, slice(*z), slice(*y), slice(*x)]
        )
        path = args.output / f"{name}.tif"
        tifffile.imwrite(path, volume, photometric="minisblack")
        report[name] = {
            "path": str(path),
            "zyx": [list(z), list(y), list(x)],
            "shape": list(volume.shape),
        }
    (args.output / "benchmark-crops.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
