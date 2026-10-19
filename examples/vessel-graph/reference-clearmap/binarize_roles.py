#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Binarise a `.npy` volume with the reference's vasculature binarisation at
one channel role's configuration defaults.

The roles are the reference's `config/defaults/v3.1/vasculature.yml`
templates: `vessel` (`vessels_template`, clip `[350, 1000]`, deconvolve
threshold left at 750) and `artery` (`large_vessels_template`, clip
`[1000, 8000]`, deconvolve threshold 450, and equalize and vesselize off, as
the reference's `_binarize` does for every channel but the all-vessel one).
Only the binarisation runs: no smoothing or filling.

    binarize_roles.py --source S.npy --layout zyx --role vessel --out B.npy

`--layout` is the source's axis order; the output is always `[z, y, x]`
`bool`, and a `<out>.json` beside it records the parameters and wall time.
Run with the reference's own Python (`clearmap-gt2`).
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

CLEARMAP = Path("/data/henriksson/github/claude/clearmap-rs/ClearMap-py")

ROLES = {
    "vessel": {"clip_range": (350, 1000), "threshold": None, "large": False},
    "artery": {"clip_range": (1000, 8000), "threshold": 450, "large": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--layout", choices=["zyx", "xyz"], required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clearmap", type=Path, default=CLEARMAP)
    args = parser.parse_args()

    sys.path.insert(0, str(args.clearmap))
    import ClearMap.IO.IO as io
    import ClearMap.ImageProcessing.Experts.Vasculature as vasc
    import ClearMap.ImageProcessing.Filter.Rank.Rank as rank_mod

    # numpy 2: the rank filters' Cython takes `const char` masks and element
    # arrays, which the Python side hands over as uint8 / bool.
    original_selem = rank_mod._initialize_selem
    rank_mod._initialize_selem = lambda selem, ndim: np.asarray(
        original_selem(selem, ndim), dtype=np.int8)
    original_apply = rank_mod._apply_code

    def apply_with_int8_mask(*a, **kw):
        mask = kw.get("mask")
        if mask is not None and mask.dtype != np.int8:
            kw["mask"] = np.ascontiguousarray(mask).view(np.int8)
        return original_apply(*a, **kw)

    rank_mod._apply_code = apply_with_int8_mask

    source = np.load(args.source)
    if args.layout == "zyx":
        source = np.ascontiguousarray(source.transpose(2, 1, 0))  # to x, y, z

    role = ROLES[args.role]
    parameter = copy.deepcopy(vasc.default_binarization_parameter)
    parameter["clip"]["clip_range"] = role["clip_range"]
    if role["threshold"] is not None:
        parameter["deconvolve"]["threshold"] = role["threshold"]
    if role["large"]:
        parameter.update(equalize=None, vesselize=None)
    processing = copy.deepcopy(vasc.default_binarization_processing_parameter)
    # "serial", not an integer: a process pool writes the blocks into forked
    # children's copies of an in-memory sink, which come back all zero.
    processing["processes"] = "serial"

    started = time.perf_counter()
    binary = vasc.binarize(io.as_source(source), sink=None,
                           binarization_parameter=parameter,
                           processing_parameter=processing)
    seconds = time.perf_counter() - started
    binary = np.ascontiguousarray(np.asarray(binary, dtype=bool).transpose(2, 1, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, binary)
    record = {
        "source": str(args.source),
        "role": args.role,
        "clip_range": list(role["clip_range"]),
        "deconvolve_threshold": parameter["deconvolve"]["threshold"],
        "equalize": parameter["equalize"] is not None,
        "vesselize": parameter["vesselize"] is not None,
        "shape_zyx": list(binary.shape),
        "voxels": int(binary.sum()),
        "seconds": seconds,
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
