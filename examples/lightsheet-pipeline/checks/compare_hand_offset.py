#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""The pipeline's cells against the earlier single-tile, hand-offset run.

    compare_hand_offset.py PIPELINE_SAMPLE_DIR [OLD_RUN_DIR] > comparison.json

The earlier run (`cells-atlas/`, cell-density's README "Atlas space, real")
detected cells in a window of tile 01x02 of osf-cztq2 and placed the tile in a
nominal-grid mosaic by `--sample-offset-um`. The pipeline fused the same window
from the stitched layout and registered the stitched signal. Here:

1. old cells (tile voxels) are carried to the pipeline's level-0 box through
   the tile's per-slice stitched positions (level 1, doubled in-plane), and
   matched one to one with the new cells within 2 voxels in every axis
   (nearest first);
2. for the matched pairs: the atlas-position difference (µm, 25 µm voxels),
   its mean vector (a systematic shift) and spread; same finest region and
   hemisphere; and the same against the reference's own elastix chain and
   labelling of the old cells (`clearmap/tile_01x02_crop/cells_reference.csv`).

Needs numpy and scipy.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

OLD = "/husky/henriksson/lightsheet-reference/cells-atlas"


def table(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def main(sample, old=OLD):
    positions = json.load(open(os.path.join(sample, "stitch/align.positions.json")))
    region = open(os.path.join(sample, "cells/region.txt")).read().strip()
    box_start = np.array([int(r.split(":")[0]) for r in region.split(",")])
    tile = next(t for t in positions["tiles"] if t["path"].rstrip("/").endswith("tile_01x02.zarr"))
    slices = np.array(tile["slices"])
    origin = np.array(positions["origin"])
    factor = positions["voxel_size"][2] / 1.625

    old_rows = table(os.path.join(old, "ours/tile_01x02_crop/tables/cell-density-filtered/table.csv"))
    new_rows = table(os.path.join(sample, "cells/detect/tables/cell-density-filtered/table.csv"))
    ref_rows = {r["label_id"]: r for r in table(os.path.join(old, "clearmap/tile_01x02_crop/cells_reference.csv"))}

    def xyz(rows, prefix=""):
        return np.array([[float(r[prefix + a]) for a in "zyx"] for r in rows])

    old_tile = xyz(old_rows)
    plane = np.clip(np.round(old_tile[:, 0]).astype(int), 0, len(slices) - 1)
    at = slices[plane]
    old_box = np.stack([
        at[:, 0] - origin[0] + (old_tile[:, 0] - plane),
        np.round(at[:, 1] * factor) - round(origin[1] * factor) + old_tile[:, 1],
        np.round(at[:, 2] * factor) - round(origin[2] * factor) + old_tile[:, 2],
    ], axis=1) - box_start
    new_box = xyz(new_rows)

    # One-to-one, nearest first, within 2 voxels per axis (Chebyshev).
    tree = cKDTree(new_box)
    dist, index = tree.query(old_box, k=4, p=np.inf, distance_upper_bound=2.0)
    candidates = sorted((d, i, j) for i in range(len(old_box)) for d, j in zip(dist[i], index[i]) if np.isfinite(d))
    used_old, used_new, pairs = set(), set(), []
    for d, i, j in candidates:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        pairs.append((i, j, d))
    pairs.sort()
    i_old = np.array([p[0] for p in pairs])
    i_new = np.array([p[1] for p in pairs])
    offsets = new_box[i_new] - old_box[i_old]

    def atlas(rows, idx):
        return np.array([[float(rows[k]["atlas_" + a]) for a in "zyx"] for k in idx])

    def summary(a, b):
        diff = (b - a) * 25.0
        norm = np.linalg.norm(diff, axis=1)
        return {
            "median_um": float(np.median(norm)), "p95_um": float(np.percentile(norm, 95)),
            "max_um": float(norm.max()),
            "mean_shift_um_zyx": diff.mean(axis=0).round(2).tolist(),
            "median_after_removing_mean_shift_um": float(np.median(np.linalg.norm(diff - diff.mean(axis=0), axis=1))),
        }

    new_atlas = atlas(new_rows, i_new)
    old_atlas = atlas(old_rows, i_old)
    matched_ref = [k for k in i_old if old_rows[k]["label_id"] in ref_rows]
    ref_atlas = np.array([[float(ref_rows[old_rows[k]["label_id"]]["atlas_" + a]) for a in "zyx"] for k in matched_ref])
    new_for_ref = np.array([new_atlas[list(i_old).index(k)] for k in matched_ref])

    same_region = float(np.mean([new_rows[j]["region"] == old_rows[i]["region"] for i, j in zip(i_old, i_new)]))
    same_hemisphere = float(np.mean([new_rows[j]["hemisphere"] == old_rows[i]["hemisphere"] for i, j in zip(i_old, i_new)]))
    pos = {k: n for n, k in enumerate(i_old)}
    same_region_ref = float(np.mean([new_rows[i_new[pos[k]]]["region"] == ref_rows[old_rows[k]["label_id"]]["region"]
                                     for k in matched_ref]))
    out = {
        "old_cells": len(old_rows), "new_cells": len(new_rows), "matched": len(pairs),
        "matched_fraction_of_old": len(pairs) / len(old_rows),
        "box_start_level0": box_start.tolist(),
        "detection_offset_voxels": {
            "median_abs_zyx": np.median(np.abs(offsets), axis=0).tolist(),
            "mean_zyx": offsets.mean(axis=0).round(3).tolist(),
        },
        "atlas_new_vs_hand_offset": summary(old_atlas, new_atlas),
        "atlas_new_vs_reference_elastix": summary(ref_atlas, new_for_ref),
        "atlas_hand_offset_vs_reference_elastix": summary(
            ref_atlas, np.array([old_atlas[pos[k]] for k in matched_ref])),
        "same_region_new_vs_hand_offset": same_region,
        "same_hemisphere_new_vs_hand_offset": same_hemisphere,
        "same_region_new_vs_reference": same_region_ref,
        "same_region_hand_offset_vs_reference": float(np.mean(
            [old_rows[k]["region"] == ref_rows[old_rows[k]["label_id"]]["region"] for k in matched_ref])),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
