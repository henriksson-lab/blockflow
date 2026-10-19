#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference run: the reference's elastix registration on the prepared inputs.

`Elastix.align` with its own shipped parameter files, unmodified, and the
fixed/moving roles of its 3.1 registration configuration
(`config/defaults/v3.1/registration.yml`):

- `auto_to_reference`: fixed = resampled autofluorescence, moving = atlas
  reference, `align_affine.txt` then `align_bspline.txt`;
- `signal_to_auto` (with `--signal`): fixed = resampled signal, moving =
  resampled autofluorescence, `align_affine.txt`.

Then, with transformix on the result:
- the atlas annotation warped onto the fixed grid with nearest-neighbour
  interpolation (`FinalBSplineInterpolationOrder 0` in a copy of the last
  transform file), for the label-overlap comparison;
- a grid of fixed-image voxel points through the transform (`-def`), for the
  point comparison (`points.txt` written by `compare.py --write-points`).

Writes under `$REG_OUT/clearmap/` and records wall times in `timing.json`.
Run with the reference's Python (env `clearmap-gt2`).
"""
import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

import ClearMap.Alignment.Elastix as elx  # noqa: E402

OUT = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration"))
ALIGNMENT = Path(CLEARMAP) / "ClearMap" / "Resources" / "Alignment"


def align(fixed, moving, parameter_files, result, processes):
    """Wall time, load average before the run, and the peak resident memory of
    any child so far (elastix is the first large one)."""
    if result.exists():
        shutil.rmtree(result)
    result.mkdir(parents=True)
    load = os.getloadavg()
    t0 = time.time()
    elx.align(fixed_image=str(fixed), moving_image=str(moving),
              parameter_files=[str(p) for p in parameter_files],
              result_directory=str(result), processes=processes,
              check_alignment_success=True)
    wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    (result / "stats.json").write_text(json.dumps(
        {"wall_s": wall, "load_average": load, "peak_rss_mb": peak, "threads": processes}))
    return wall


def nearest_copy(transform: Path, target: Path) -> Path:
    """The transform files with the last one's final interpolation set to
    nearest neighbour, for labels. Earlier links are copied unchanged and the
    initial-transform pointers are rewritten to the copies."""
    target.mkdir(parents=True, exist_ok=True)
    chain = []
    current = transform
    while True:
        chain.append(current)
        text = current.read_text()
        initial = re.search(r'\(InitialTransformParametersFileName "([^"]+)"\)', text).group(1)
        if initial == "NoInitialTransform":
            break
        current = Path(initial)
    for path in chain:
        text = path.read_text()
        text = re.sub(r'\(InitialTransformParametersFileName "([^"]+)"\)',
                      lambda m: m.group(0) if m.group(1) == "NoInitialTransform"
                      else f'(InitialTransformParametersFileName "{target / Path(m.group(1)).name}")',
                      text)
        text = re.sub(r"\(FinalBSplineInterpolationOrder \d+\)", "(FinalBSplineInterpolationOrder 0)", text)
        text = re.sub(r'\(ResultImagePixelType "[^"]+"\)', '(ResultImagePixelType "int")', text)
        text = re.sub(r'\(ResultImageFormat "[^"]+"\)', '(ResultImageFormat "mhd")', text)
        (target / path.name).write_text(text)
    return target / transform.name


def annotation_mhd() -> Path:
    """The annotation as a MetaImage of 32-bit integers: transformix reads the
    prepared annotation TIFF (32-bit unsigned, written by the reference's IO)
    as garbage, so it is handed the same voxels in a format it reads
    unambiguously. Axis order as elastix sees the TIFF: x fastest."""
    import numpy as np
    import tifffile

    target = OUT / "atlas" / "annotation_int32.mhd"
    if not target.exists():
        data = tifffile.imread(OUT / "atlas" / "annotation.tif").astype(np.int32)  # (z, y, x)
        data.tofile(target.with_suffix(".raw"))
        target.write_text(
            "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
            f"DimSize = {data.shape[2]} {data.shape[1]} {data.shape[0]}\n"
            "ElementSpacing = 1 1 1\nOffset = 0 0 0\nElementType = MET_INT\n"
            f"ElementDataFile = {target.with_suffix('.raw').name}\n")
    return target


def transformix(args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [elx.transformix_binary] + args + ["-out", str(out)]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true",
                        help="do not (re)run the autofluorescence -> atlas registration")
    args = parser.parse_args()
    out = OUT / "clearmap"
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing["processes"] = args.processes
    fixed = out / "resampled_autofluorescence.tif"
    moving = OUT / "atlas" / "reference.tif"
    result = out / "auto_to_reference"
    if not args.skip_align:
        if not args.skip_atlas:
            timing["auto_to_reference_s"] = align(
                fixed, moving, [ALIGNMENT / "align_affine.txt", ALIGNMENT / "align_bspline.txt"],
                result, args.processes)
            print("auto_to_reference", timing["auto_to_reference_s"], flush=True)
        if args.signal:
            timing["signal_to_auto_s"] = align(
                out / "resampled_signal.tif", fixed, [ALIGNMENT / "align_affine.txt"],
                out / "signal_to_auto", args.processes)
            print("signal_to_auto", timing["signal_to_auto_s"], flush=True)
    # Labels onto the fixed grid, after each stage.
    for index in (0, 1):
        nearest = nearest_copy(result / f"TransformParameters.{index}.txt", out / f"nearest.{index}")
        timing[f"annotation_warp_{index}_s"] = transformix(
            ["-in", str(annotation_mhd()), "-tp", str(nearest)],
            out / f"annotation_in_auto.{index}")
    points = OUT / "points" / "auto_points_elastix.txt"
    if points.exists():
        for index in (0, 1):
            timing[f"points_{index}_s"] = transformix(
                ["-def", str(points), "-tp", str(result / f"TransformParameters.{index}.txt")],
                out / f"points.{index}")
        signal_points = OUT / "points" / "signal_points_elastix.txt"
        if signal_points.exists() and (out / "signal_to_auto").exists():
            # The reference's cell pipeline: transformix through the first
            # registration, its output points through the second.
            t0 = time.time()
            transformix(["-def", str(signal_points), "-tp",
                         str(out / "signal_to_auto" / "TransformParameters.0.txt")],
                        out / "signal_points.0")
            moved = [re.search(r"OutputPoint = \[ ([^\]]+) \]", line).group(1)
                     for line in (out / "signal_points.0" / "outputpoints.txt").read_text().splitlines()]
            relay = out / "signal_points.0" / "relay.txt"
            relay.write_text(f"point\n{len(moved)}\n" + "\n".join(moved) + "\n")
            transformix(["-def", str(relay), "-tp", str(result / "TransformParameters.1.txt")],
                        out / "signal_points.1")
            timing["signal_points_chain_s"] = time.time() - t0
    timing_path.write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
