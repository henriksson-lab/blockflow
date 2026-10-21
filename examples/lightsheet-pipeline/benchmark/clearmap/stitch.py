#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference side of the stitching workloads: rigid and wobbly alignment,
placement and fusion of `.npy` tiles, as `tile-stitching/reference-clearmap/
clearmap_stitch.py` runs them, with the search ranges given (they are in
voxels of the level stitched) and the process count given.

    stitch.py --npy-dir DIR --out DIR --overlap X,Y --rigid-shift X,Y,Z --wobbly-shift X,Y
              --processes N [--variant shipped|patched|corrected]

`--variant` picks which of the reference's defects are corrected first, by
the textual patches of `clearmap_stitch.py`: `shipped` none (its wobbly
placement is then a no-op and the result is the rigid layout), `patched` the
placement defect, `corrected` (default) the placement and the tracing
tie-break (the harness's main comparison). Writes into `--out` the same files
as `clearmap_stitch.py` (`rigid_positions.json`, `wobbly_positions.npz`,
`stitched.npy`, `timing.json`), so its `compare.py` reads them.

Run with the reference's Python (env `clearmap-gt2`).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[3] / "tile-stitching" / "reference-clearmap" / "clearmap_stitch.py"
VARIANTS = {"shipped": ("0", "0"), "patched": ("1", "0"), "corrected": ("1", "1")}


def load_harness(variant):
    """Import the harness module with the variant's patches applied (it
    patches the reference at import, from its environment variables)."""
    placement, tracing = VARIANTS[variant]
    os.environ["STITCH_PATCH_PLACEMENT"] = placement
    os.environ["STITCH_PATCH_TRACING"] = tracing
    os.environ["STITCH_PATCH_SMOOTHING"] = "0"
    spec = importlib.util.spec_from_file_location("clearmap_stitch", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def triple(text):
    return [int(v) for v in text.split(",")]


def stitch(npy_dir, out, overlap, rigid_shift, wobbly_shift, processes, variant):
    """Rigid + wobbly stitching; returns the timing record."""
    harness = load_harness(variant)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compat  # noqa: F401  (worker sinks reopened r+; see compat.py)
    srig, swob, np = harness.srig, harness.swob, harness.np
    out.mkdir(parents=True, exist_ok=True)
    timing = {"processes": processes, "variant": variant, "overlap_xy": overlap,
              "rigid_shift_xyz": rigid_shift, "wobbly_shift_xy": wobbly_shift}
    t0 = time.time()
    expression = str(Path(npy_dir) / "tile_<Y,2>x<X,2>.npy")
    layout = swob.WobblyLayout(expression=expression, tile_axes=("X", "Y"), overlaps=overlap)
    print("sources", len(layout.sources), "alignments", len(layout.alignments), flush=True)
    rx, ry, rz = rigid_shift
    srig.align_layout_rigid_mip(layout, depth=[overlap[0], overlap[1], None],
                                max_shifts=[(-rx, rx), (-ry, ry), (-rz, rz)],
                                ranges=[None, None, None], background=(0, 100),
                                clip=25000, processes=processes, verbose=False)
    layout.place(method="optimization", min_quality=-np.inf, lower_to_origin=True, verbose=False)
    timing["rigid_s"] = time.time() - t0
    rigid = {
        "axes": ["x", "y", "z"],
        "tiles": [{"name": Path(s.location).name if getattr(s, "location", None) else str(i),
                   "tile_position": [int(v) for v in s.tile_position],
                   "position": [int(v) for v in s.position]} for i, s in enumerate(layout.sources)],
        "alignments": [{"pre": layout.sources.index(a.pre), "post": layout.sources.index(a.post),
                        "displacement": [int(v) for v in a.displacement],
                        "quality": float(a.quality)} for a in layout.alignments],
    }
    (out / "rigid_positions.json").write_text(json.dumps(rigid, indent=1))

    t1 = time.time()
    wx, wy = wobbly_shift
    swob.align_layout(layout, axis_range=(None, None, 3),
                      max_shifts=[(-wx, wx), (-wy, wy), (0, 0)], axis_mip=None,
                      stack_validation_params=dict(method="foreground", valid_range=(200, None), size=None),
                      prepare=dict(method="normalization", clip=None, normalize=True),
                      slice_validation_params=dict(method="foreground", valid_range=(200, 20000), size=1500),
                      prepare_slice=None,
                      find_shifts=dict(method="tracing", cutoff=3 * np.sqrt(2)),
                      processes=processes, verbose=False)
    timing["wobbly_align_s"] = time.time() - t1
    t2 = time.time()
    swob.place_layout(layout, min_quality=-np.inf, method="optimization",
                      smooth=dict(method="window", window="bartlett", window_length=100, binary=None),
                      smooth_optimized=dict(method="window", window="bartlett", window_length=20, binary=10),
                      fix_isolated=False, lower_to_origin=True, processes=processes, verbose=False)
    timing["wobbly_place_s"] = time.time() - t2
    np.savez(out / "wobbly_positions.npz",
             **{f"positions_{i}": np.asarray(s.positions) for i, s in enumerate(layout.sources)},
             **{f"status_{i}": np.asarray(s.status) for i, s in enumerate(layout.sources)},
             tile_positions=np.asarray([s.tile_position for s in layout.sources]),
             **{f"measured_{k}": np.asarray(a.displacements, dtype=float) for k, a in enumerate(layout.alignments)},
             **{f"measured_status_{k}": np.asarray(a.status) for k, a in enumerate(layout.alignments)},
             **{f"measured_lower_{k}": np.asarray(a.lower_coordinate) for k, a in enumerate(layout.alignments)},
             lower=np.asarray(layout.lower_wobbly))
    t3 = time.time()
    sink = out / "stitched.npy"
    if sink.exists():
        sink.unlink()
    swob.stitch_layout(layout, sink=str(sink), method="interpolation", processes=processes, verbose=False)
    timing["stitch_s"] = time.time() - t3
    timing["total_s"] = time.time() - t0
    stitched = np.load(sink, mmap_mode="r")
    timing["stitched_shape_xyz"] = list(stitched.shape)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    return layout, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overlap", required=True, help="X,Y voxels of the level")
    parser.add_argument("--rigid-shift", default="30,30,20")
    parser.add_argument("--wobbly-shift", default="30,15")
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="corrected")
    args = parser.parse_args()
    _, timing = stitch(args.npy_dir, args.out, triple(args.overlap), triple(args.rigid_shift),
                       triple(args.wobbly_shift), args.processes, args.variant)
    print("STITCHED", json.dumps(timing), flush=True)


if __name__ == "__main__":
    sys.exit(main())
