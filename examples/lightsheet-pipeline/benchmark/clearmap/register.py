#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference side of the registration workload: `Elastix.align` with the
reference's shipped parameter files, unmodified, on the prepared 25 µm
inputs of `atlas-registration/reference-clearmap/` (fixed = resampled
autofluorescence, moving = atlas reference, the roles of its 3.1
`registration.yml`).

    register.py --fixed F.tif --moving M.tif --parameters align_affine.txt [--parameters ...]
                --out DIR --processes N

elastix runs with `-threads N`; it also writes its result images, as the
reference's pipeline has it do. Writes `DIR/timing.json`. Run with the
reference's Python (env `clearmap-gt2`).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)
ALIGNMENT = Path(CLEARMAP) / "ClearMap" / "Resources" / "Alignment"


def align(fixed, moving, parameter_files, result, processes):
    import ClearMap.Alignment.Elastix as elx
    if result.exists():
        shutil.rmtree(result)
    result.mkdir(parents=True)
    t0 = time.time()
    elx.align(fixed_image=str(fixed), moving_image=str(moving),
              parameter_files=[str(p) for p in parameter_files],
              result_directory=str(result), processes=processes, check_alignment_success=True)
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--parameters", action="append", required=True,
                        help="a file name in the reference's Resources/Alignment, or a path")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args()
    files = [Path(p) if os.sep in p else ALIGNMENT / p for p in args.parameters]
    seconds = align(args.fixed, args.moving, files, args.out, args.processes)
    record = {"align_s": seconds, "processes": args.processes, "parameters": [str(f) for f in files]}
    (args.out / "timing.json").write_text(json.dumps(record, indent=1))
    print("REGISTERED", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
