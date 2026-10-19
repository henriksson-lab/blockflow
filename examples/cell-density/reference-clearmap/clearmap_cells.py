#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference run: cell detection with the reference implementation.

Follows its cell-map orchestrator (`run_cell_detection`, `filter_cells`)
with the 3.1 defaults of `config/defaults/v3.1/cell_map.yml`: background
disk (10, 10), maxima shape 5, no h-max, shape threshold 500, intensity
measure `source`, size band [50, 900), blocks along z of 10..16 planes with
overlap 5 on 10 processes. Reads `$CELLS_OUT/input.npy` from
`prepare_crop.py`; writes `$CELLS_OUT/clearmap/{cells_raw,cells_filtered}.csv`
(columns `x, y, z, size, source`, in the crop's voxels) and `timing.json`.

Run with the reference's Python (env `clearmap-gt2`).
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

import ClearMap.ImageProcessing.Experts.Cells as cells  # noqa: E402
import pandas as pd  # noqa: E402

# The reference passes per-block tables through Feather files, which needs
# pyarrow; the environment lacks it, so Feather is swapped for pickle (same
# tables, other container). Workers are forked and inherit the swap.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pd.DataFrame.to_feather = lambda self, path, *a, **k: self.to_pickle(path)
    pd.read_feather = lambda path, *a, **k: pd.read_pickle(path)

OUT = Path(os.environ.get("CELLS_OUT", "/husky/henriksson/lightsheet-reference/cells/crop"))
PROCESSES = int(os.environ.get("CLEARMAP_PROCESSES", "10"))
# Block processing along z: size_min, size_max, overlap (3.1 defaults 10, 16, 5).
BLOCK = [int(v) for v in os.environ.get("CLEARMAP_BLOCK", "10,16,5").split(",")]


def main():
    out = OUT / os.environ.get("CELLS_REF_DIR", "clearmap")
    out.mkdir(parents=True, exist_ok=True)
    parameter = copy.deepcopy(cells.default_cell_detection_parameter)
    parameter["illumination_correction"] = None
    parameter["background_correction"]["shape"] = (10, 10)
    parameter["maxima_detection"]["shape"] = 5
    parameter["maxima_detection"]["h_max"] = None
    parameter["intensity_detection"]["measure"] = ["source"]
    parameter["shape_detection"]["threshold"] = 500
    processing = copy.deepcopy(cells.default_cell_detection_processing_parameter)
    processing.update(processes=PROCESSES, size_min=BLOCK[0], size_max=BLOCK[1], overlap=BLOCK[2],
                      verbose=True)

    sink = out / "cells_raw.feather"
    if sink.exists():
        sink.unlink()
    started = time.time()
    cells.detect_cells(str(OUT / "input.npy"), str(sink), cell_detection_parameter=parameter,
                       processing_parameter=processing)
    detect_s = time.time() - started
    table = pd.read_feather(str(sink))
    table.to_csv(out / "cells_raw.csv", index=False)
    # filter_cells: thresholds[0] <= value < thresholds[1].
    keep = (table["size"] >= 50) & (table["size"] < 900)
    table[keep].to_csv(out / "cells_filtered.csv", index=False)
    timing = {"processes": PROCESSES, "block_min_max_overlap": BLOCK, "detect_s": detect_s, "raw": int(len(table)),
              "filtered": int(keep.sum()), "columns": list(table.columns)}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print("CELLS", json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
