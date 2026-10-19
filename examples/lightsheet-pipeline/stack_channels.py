#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Stack single-channel OME-Zarr images (level 0) into one `(c, z, y, x)`
OME-Zarr 0.5 image, keeping the first image's level-0 scale and translation.

    stack_channels.py OUT.ome.zarr IN0.ome.zarr [IN1.ome.zarr ...]

`tile-stitching` fuses one channel per run; `vessel-graph segment` reads the
vessel and artery channels from one image. Needs `zarr` (v3) and `numpy`.
"""
import json
import sys

import numpy as np
import zarr


def main(out, inputs):
    frames, arrays = [], []
    for path in inputs:
        group = json.load(open(f"{path}/zarr.json"))
        multiscale = group["attributes"]["ome"]["multiscales"][0]
        frames.append((multiscale["axes"], multiscale["datasets"][0]["coordinateTransformations"]))
        arrays.append(zarr.open_array(f"{path}/0", mode="r"))
    shape, dtype = arrays[0].shape, arrays[0].dtype
    if any(a.shape != shape or a.dtype != dtype for a in arrays):
        sys.exit("stack_channels.py: the images differ in shape or type")
    if any(f != frames[0] for f in frames):
        sys.exit("stack_channels.py: the images differ in frame")
    axes, transforms = frames[0]
    root = zarr.open_group(out, mode="w-", zarr_format=3)
    stacked = root.create_array(
        "0", shape=(len(arrays),) + shape, dtype=dtype, chunks=(1, 32, 128, 128),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
        fill_value=0)
    for c, array in enumerate(arrays):
        for z in range(0, shape[0], 32):
            stacked[c, z:z + 32] = np.asarray(array[z:z + 32])
    widen = []
    for t in transforms:
        key = t["type"]
        widen.append({"type": key, key: [1.0 if key == "scale" else 0.0] + list(t[key])})
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": out.rstrip("/").split("/")[-1],
            "axes": [{"name": "c", "type": "channel"}] + axes,
            "datasets": [{"path": "0", "coordinateTransformations": widen}],
        }],
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
