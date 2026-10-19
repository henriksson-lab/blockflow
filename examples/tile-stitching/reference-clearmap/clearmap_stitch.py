#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference run: rigid and wobbly stitching with the reference implementation.

Follows its stitching orchestrator (`align_channel_rigid`,
`stitch_channel_wobbly`) with the 3.1 default parameters of
`config/defaults/v3.1/stitching.yml` and `n_processes: -1`, on the `.npy`
tiles written by `prepare_tiles.py`. Writes to `$STITCH_OUT/clearmap/`:
`rigid_positions.json`, `wobbly_positions.npz` (per tile the `(x, y, z)`
lower corner of every slice, and its status), `stitched.npy` (`(x, y, z)`),
and `timing.json`.

Two defects of the reference can be corrected first, each by a textual
patch of the one function's source (asserted to match exactly once):

- `STITCH_PATCH_PLACEMENT=1`: in 3.1, `stitching_wobbly._connected_components`
  rebinds its `displacements` argument to an empty list before reading it, so
  every slice's component has no alignments, `_place_slice` returns the rigid
  positions unchanged and the "wobbly" result is the rigid layout. The patch
  only renames that list; the measured per-slice displacements are the same
  either way.
- `STITCH_PATCH_TRACING=1`: in `stitching_wobbly.shifts_from_tracing`, two
  equally long trajectories are told apart by their summed error, but each
  point's error is read from `errors[l]`, the landscape of slice `l` of the
  overlap, where `l` is the point's index among the *measured* slices of the
  stretch. With `axis_range` step 3 that is mostly the landscape of another
  (often unmeasured, all-zero) slice, so the tie is broken by chance. The
  patch reads `errors[s + measured_se[l]]`, the slice the point is on.
- `STITCH_PATCH_SMOOTHING=1` (a choice more than a defect): `place_layout`
  smooths each tile's optimised positions over every slice of the layout,
  including the slices outside the tile and its isolated slices, which hold
  its rigid position rather than an optimised one, and the binary pass sets
  the first and last 11 to their median. The patch smooths each tile over
  its own slices only, as `tile-stitching` does.

`STITCH_REF_DIR` names the output directory (default `clearmap-patched` with
the placement patch, else `clearmap`).

Run with the reference's Python (env `clearmap-gt2`).
"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

import ClearMap.Alignment.Stitching.stitching_rigid as srig  # noqa: E402
import ClearMap.Alignment.Stitching.stitching_wobbly as swob  # noqa: E402
from ClearMap.Utils.utilities import sanitize_n_processes  # noqa: E402

OUT = Path(os.environ.get("STITCH_OUT", "/husky/henriksson/lightsheet-reference/stitching"))
PROCESSES = sanitize_n_processes(int(os.environ.get("CLEARMAP_PROCESSES", "-1")))
PATCH = os.environ.get("STITCH_PATCH_PLACEMENT", "0") == "1"
PATCH_TRACING = os.environ.get("STITCH_PATCH_TRACING", "0") == "1"
PATCH_SMOOTHING = os.environ.get("STITCH_PATCH_SMOOTHING", "0") == "1"
# The wobbly search along y (default 15, the 3.1 default, as `tile-stitching`'s
# `--max-shift 15,30`).
SHIFT_Y = int(os.environ.get("STITCH_WOBBLY_SHIFT_Y", "15"))


def _patch(module, name, edits):
    """Redefine `module.name` from its own source with `edits` applied, each
    `(old, new)` occurring exactly once."""
    source = inspect.getsource(getattr(module, name))
    for old, new in edits:
        assert source.count(old) == 1, f"{name}: {old!r} occurs {source.count(old)} times"
        source = source.replace(old, new)
    exec(compile(source, f"<patched {name}>", "exec"), module.__dict__)


if PATCH:
    _patch(swob, "_connected_components", [
        ("        displacements = []\n", "        shifts = []\n"),
        ("displacements.append(d)", "shifts.append(d)"),
        ("component_displacements.append(displacements)", "component_displacements.append(shifts)"),
    ])
if PATCH_TRACING:
    _patch(swob, "shifts_from_tracing", [
        ("errors[t[0]][tuple(positions[t[0]][t[1]])]",
         "errors[s + measured_se[t[0]]][tuple(positions[t[0]][t[1]])]"),
    ])
if PATCH_SMOOTHING:
    _patch(swob, "place_layout", [
        ("        for p in positions_optimized:\n",
         "        for source, p in zip(sources, positions_optimized):\n"
         "            p = p[source.coordinate:source.coordinate + source.height]\n"),
    ])


def main():
    setup = json.loads((OUT / "setup.json").read_text())
    overlaps = setup["overlap_xy"]
    out = OUT / os.environ.get("STITCH_REF_DIR", "clearmap-patched" if PATCH else "clearmap")
    out.mkdir(parents=True, exist_ok=True)
    timing = {"processes": PROCESSES, "placement_patched": PATCH, "tracing_patched": PATCH_TRACING,
              "smoothing_patched": PATCH_SMOOTHING,
              "wobbly_shift_y": SHIFT_Y}
    t0 = time.time()

    # get_wobbly_layout: the tile expression, axes sorted alphabetically.
    expression = str(OUT / "npy" / "tile_<Y,2>x<X,2>.npy")
    layout = swob.WobblyLayout(expression=expression, tile_axes=("X", "Y"), overlaps=overlaps)
    print("sources", len(layout.sources), "alignments", len(layout.alignments), flush=True)

    # align_channel_rigid, stitching.yml rigid defaults.
    projection_thickness = [overlaps[0], overlaps[1], None]
    max_shifts = [(-30, 30), (-30, 30), (-20, 20)]
    srig.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                ranges=[None, None, None], background=(0, 100),
                                clip=25000, processes=PROCESSES, verbose=True)
    layout.place(method="optimization", min_quality=-np.inf, lower_to_origin=True, verbose=True)
    timing["rigid_s"] = time.time() - t0
    rigid = {
        "axes": ["x", "y", "z"],
        "tiles": [{"name": Path(s.location).name if hasattr(s, "location") and s.location else str(i),
                   "tile_position": [int(v) for v in s.tile_position],
                   "position": [int(v) for v in s.position]}
                  for i, s in enumerate(layout.sources)],
        "alignments": [{"pre": layout.sources.index(a.pre), "post": layout.sources.index(a.post),
                        "displacement": [int(v) for v in a.displacement],
                        "quality": float(a.quality)} for a in layout.alignments],
    }
    (out / "rigid_positions.json").write_text(json.dumps(rigid, indent=1))
    print("RIGID", [t["position"] for t in rigid["tiles"]], flush=True)

    # stitch_channel_wobbly: align, place, stitch, stitching.yml wobbly defaults.
    t1 = time.time()
    swob.align_layout(layout, axis_range=(None, None, 3),
                      max_shifts=[(-30, 30), (-SHIFT_Y, SHIFT_Y), (0, 0)], axis_mip=None,
                      stack_validation_params=dict(method="foreground", valid_range=(200, None), size=None),
                      prepare=dict(method="normalization", clip=None, normalize=True),
                      slice_validation_params=dict(method="foreground", valid_range=(200, 20000), size=1500),
                      prepare_slice=None,
                      find_shifts=dict(method="tracing", cutoff=3 * np.sqrt(2)),
                      processes=PROCESSES, verbose=True)
    timing["wobbly_align_s"] = time.time() - t1
    t2 = time.time()
    swob.place_layout(layout, min_quality=-np.inf, method="optimization",
                      smooth=dict(method="window", window="bartlett", window_length=100, binary=None),
                      smooth_optimized=dict(method="window", window="bartlett", window_length=20, binary=10),
                      fix_isolated=False, lower_to_origin=True, processes=PROCESSES, verbose=True)
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
    swob.stitch_layout(layout, sink=str(sink), method="interpolation", processes=PROCESSES, verbose=True)
    timing["stitch_s"] = time.time() - t3
    timing["total_s"] = time.time() - t0
    stitched = np.load(sink, mmap_mode="r")
    timing["stitched_shape_xyz"] = list(stitched.shape)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print("STITCHED", stitched.shape, json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
