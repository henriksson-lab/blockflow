#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference side of the end-to-end workload on osf-cztq2, as its CellMap
pipeline runs a sample, at the scope the blockflow chain runs it
(`lightsheet-pipeline/run.sh` stages align, fuse, register, cells):

1. stitch the cFos tiles of one pyramid level, rigid and wobbly, and fuse
   them (`stitch.py`, the same search ranges);
2. resample the fused signal and the autofluorescence to 25 µm
   (`Resampling.resample`);
3. register: resampled signal -> resampled autofluorescence (fixed = signal,
   `align_affine.txt`), autofluorescence -> atlas reference (fixed =
   autofluorescence, the given files), with elastix (`Elastix.align`);
4. detect cells on a level-0 box of one tile (`detect_cells`, the 3.1
   `cell_map.yml` parameters, as `cell-density/reference-clearmap/
   clearmap_cells.py`) and keep the size band [50, 900);
5. carry the cells into the atlas as its `CellDetector.transform_coordinates`
   does (`resample_points` from the stitched level-0 shape to the resampled
   shape, then `transform_points` through both registrations) and label them
   with the annotation (`astype(int)`, 0 outside), counted per region.

    e2e.py --npy-dir DIR --overlap X,Y --rigid-shift X,Y,Z --wobbly-shift X,Y --level-factor X,Y,Z
           --auto-npy F --atlas-dir DIR --auto-parameters F [...] --box-npy F --box-tile ROWxCOL
           --box-start Z,Y,X --processes N --out DIR [--variant corrected]

The box is detected on the tile's own level-0 voxels (prepared from the
OME-Zarr, untimed) and placed in the stitched frame at the tile's median
wobbly position: the reference fuses the whole sample at one level and has no
way to fuse a level-0 box from coarser positions, so this skips the level-0
fusion it would need (which only makes it faster). Writes `timing.json`
(seconds per step), `cells_atlas.csv` and `regions.csv`. Run with the
reference's Python (env `clearmap-gt2`).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

import stitch as stitch_step  # noqa: E402
import register as register_step  # noqa: E402


def triple(text, kind=int):
    return [kind(v) for v in text.split(",")]


def detect(box, out, processes):
    """`clearmap_cells.py`'s detection and size filter."""
    import ClearMap.ImageProcessing.Experts.Cells as cells
    import pandas as pd
    try:
        import pyarrow  # noqa: F401
    except ImportError:  # as clearmap_cells.py: Feather swapped for pickle
        pd.DataFrame.to_feather = lambda self, path, *a, **k: self.to_pickle(path)
        pd.read_feather = lambda path, *a, **k: pd.read_pickle(path)
    parameter = copy.deepcopy(cells.default_cell_detection_parameter)
    parameter["illumination_correction"] = None
    parameter["background_correction"]["shape"] = (10, 10)
    parameter["maxima_detection"]["shape"] = 5
    parameter["maxima_detection"]["h_max"] = None
    parameter["intensity_detection"]["measure"] = ["source"]
    parameter["shape_detection"]["threshold"] = 500
    processing = copy.deepcopy(cells.default_cell_detection_processing_parameter)
    processing.update(processes=processes, size_min=10, size_max=16, overlap=5, verbose=False)
    sink = out / "cells_raw.feather"
    cells.detect_cells(str(box), str(sink), cell_detection_parameter=parameter,
                       processing_parameter=processing)
    table = pd.read_feather(str(sink))
    keep = (table["size"] >= 50) & (table["size"] < 900)
    return table, table[keep].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--rigid-shift", required=True)
    parser.add_argument("--wobbly-shift", required=True)
    parser.add_argument("--level-factor", required=True, help="x,y,z: level-0 voxels per stitched voxel")
    parser.add_argument("--tile-um", default="1.625,1.625,6", help="x,y,z µm of a level-0 tile voxel")
    parser.add_argument("--auto-npy", required=True)
    parser.add_argument("--auto-um", default="5,5,6")
    parser.add_argument("--atlas-dir", required=True, type=Path,
                        help="the registration harness's atlas/ (reference.tif, annotation.tif)")
    parser.add_argument("--signal-parameters", action="append", default=None)
    parser.add_argument("--auto-parameters", action="append", required=True)
    parser.add_argument("--box-npy", required=True)
    parser.add_argument("--box-tile", required=True, help="grid index ROWxCOL of the box's tile")
    parser.add_argument("--box-start", required=True, help="z,y,x in the tile's level-0 voxels")
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--variant", default="corrected")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = args.processes
    seconds = {}
    t_all = time.time()

    # 1. stitch
    t0 = time.time()
    layout, stitch_timing = stitch_step.stitch(args.npy_dir, out / "stitch", triple(args.overlap),
                                               triple(args.rigid_shift), triple(args.wobbly_shift), n,
                                               args.variant)
    seconds["stitch"] = time.time() - t0
    factor = np.array(triple(args.level_factor, float))
    stitched_um = np.array(triple(args.tile_um, float)) * factor

    # 2. resample
    import ClearMap.Alignment.Resampling as resampling
    import ClearMap.Alignment.Elastix as elastix
    import ClearMap.IO.IO as clearmap_io
    t0 = time.time()
    signal = out / "resampled_signal.tif"
    resampling.resample(str(out / "stitch" / "stitched.npy"), resampled=str(signal),
                        original_resolution=tuple(stitched_um), resampled_resolution=(25, 25, 25),
                        processes=n, verbose=False)
    seconds["resample_signal"] = time.time() - t0
    t0 = time.time()
    auto = out / "resampled_autofluorescence.tif"
    resampling.resample(args.auto_npy, resampled=str(auto),
                        original_resolution=tuple(triple(args.auto_um, float)),
                        resampled_resolution=(25, 25, 25), processes=n, verbose=False)
    seconds["resample_autofluorescence"] = time.time() - t0

    # 3. register
    alignment = Path(CLEARMAP) / "ClearMap" / "Resources" / "Alignment"
    files = lambda names: [Path(p) if os.sep in p else alignment / p for p in names]  # noqa: E731
    seconds["align_signal_to_auto"] = register_step.align(
        signal, auto, files(args.signal_parameters or ["align_affine.txt"]), out / "signal_to_auto", n)
    seconds["align_auto_to_reference"] = register_step.align(
        auto, args.atlas_dir / "reference.tif", files(args.auto_parameters), out / "auto_to_reference", n)

    # 4. cells
    t0 = time.time()
    raw, filtered = detect(args.box_npy, out, n)
    seconds["detect_cells"] = time.time() - t0

    # 5. into the atlas
    t0 = time.time()
    row, col = (int(v) for v in args.box_tile.split("x"))
    index = next(i for i, s in enumerate(layout.sources) if tuple(s.tile_position) == (col, row))
    source = layout.sources[index]
    positions = np.asarray(source.positions, dtype=float)       # (slices, (x, y, z)) at the level
    valid = np.asarray(source.status) >= 0
    in_plane = np.median(positions[valid] if valid.any() else positions, axis=0)[:2]
    origin = np.asarray(getattr(layout, "origin_wobbly", (0, 0, 0)), dtype=float)
    tile_origin = np.array([in_plane[0], in_plane[1], source.position[2]], dtype=float) - origin
    bz, by, bx = triple(args.box_start)
    coordinates = filtered[["x", "y", "z"]].to_numpy(dtype=float)
    stitched0 = coordinates + np.array([bx, by, bz]) + tile_origin * factor
    stitched_shape = np.load(out / "stitch" / "stitched.npy", mmap_mode="r").shape
    resampled_shape = clearmap_io.shape(str(signal))
    points = resampling.resample_points(stitched0, original_shape=tuple(np.array(stitched_shape) * factor),
                                        resampled_shape=resampled_shape)
    # For checks.py: the stitched level-0 points and the per-axis factor
    # resample_points applied (`points * f`), so the check can also carry
    # them with the voxel-centre convention the image resampling uses.
    np.save(out / "points_stitched0_xyz.npy", stitched0)
    np.save(out / "resample_factor_xyz.npy", np.asarray(resampled_shape) / (np.array(stitched_shape) * factor))
    for i, step in enumerate(("signal_to_auto", "auto_to_reference")):
        points = elastix.transform_points(points, transform_directory=str(out / step), binary=True,
                                          temp_file=str(out / f"points_in.{i}.bin"),
                                          result_directory=str(out / f"points.{i}"), n_processes=n)
    annotation = np.asarray(clearmap_io.read(str(args.atlas_dir / "annotation.tif")))  # (x, y, z)
    voxel = np.asarray(points).astype(int)
    inside = np.all((voxel >= 0) & (voxel < np.array(annotation.shape)), axis=1)
    region = np.zeros(len(voxel), dtype=np.int64)
    region[inside] = annotation[tuple(voxel[inside].T)]
    seconds["map_and_label"] = time.time() - t0

    filtered = filtered.copy()
    filtered[["xt", "yt", "zt"]] = np.asarray(points)
    filtered["region"] = region
    filtered.to_csv(out / "cells_atlas.csv", index=False)
    ids, counts = np.unique(region, return_counts=True)
    with open(out / "regions.csv", "w") as f:
        f.write("region,count\n")
        f.writelines(f"{i},{c}\n" for i, c in zip(ids, counts))
    record = {"processes": n, "variant": args.variant, "steps_s": {k: round(v, 3) for k, v in seconds.items()},
              "stitch": stitch_timing, "total_s": round(time.time() - t_all, 3),
              "stitched_shape_xyz": list(stitched_shape), "resampled_signal_shape_xyz": list(resampled_shape),
              "raw_cells": int(len(raw)), "filtered_cells": int(len(filtered)),
              "inside_atlas": int(inside.sum()), "labelled": int((region > 0).sum()),
              "box_tile_origin_level0_xyz": (tile_origin * factor).tolist()}
    (out / "timing.json").write_text(json.dumps(record, indent=1))
    print("E2E", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
