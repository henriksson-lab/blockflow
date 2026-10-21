#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Output agreement for one benchmark repeat, so a fast but wrong run shows.

    checks.py vessels RUN_DIR            (numpy, zarr)
    checks.py registration RUN_DIR       (the reference's Python: transformix)
    checks.py e2e RUN_DIR                (the reference's Python: pandas, scipy, transformix)

`RUN_DIR` holds `clearmap/` and `ours/` as `bench.py` writes them. Prints
one JSON object. The stitching and cell workloads use the per-stage
harnesses' own `compare.py` instead (`bench.py` runs them).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
POINTS = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration")) / "points"


def label_zyx(image, name):
    import zarr
    array = zarr.open_array(str(Path(image) / "labels" / name / "0"), mode="r")
    data = np.asarray(array[...])
    fill = array.fill_value
    return (data != 0) & (data != fill) if fill not in (None, 0) else data != 0


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / total if total else 1.0


def vessels(run):
    cm_dir, ours_dir = run / "clearmap" / "work", run / "ours"
    cm_final = np.load(cm_dir / "binary_final.npy").transpose(2, 1, 0)       # (x, y, z) -> (z, y, x)
    cm_skeleton = np.load(cm_dir / "skeleton.npy").transpose(2, 1, 0) != 0
    image = ours_dir / "image.ome.zarr"
    mask = label_zyx(image, "vessel-mask")
    skeleton = label_zyx(image, "vessel-skeleton")
    cm = json.loads((cm_dir / "timing.json").read_text())
    graph = json.loads((ours_dir / "graph.json").read_text())
    length = graph["edge"].get("length_units")
    if isinstance(length, dict):          # {"width": 1, "values": [...]}
        length = length["values"]
    return {
        "mask_voxels": {"clearmap": int(cm_final.sum()), "ours": int(mask.sum())},
        "mask_dice": round(dice(cm_final != 0, mask), 4),
        "skeleton_voxels": {"clearmap": int(cm_skeleton.sum()), "ours": int(skeleton.sum())},
        "graph_vertices": {"clearmap": cm["graph_reduced"]["vertices"], "ours": graph["n_vertices"]},
        "graph_edges": {"clearmap": cm["graph_reduced"]["edges"], "ours": len(graph["edges"])},
        "total_length_um": {"clearmap": cm.get("total_edge_length_units", cm.get("total_edge_length")),
                            "ours": float(np.sum(length)) if length is not None else None},
    }


def registration(run):
    """The registration harness's point grid through both sides' last
    transform: distance between the two atlas positions, in 25 µm voxels."""
    import csv
    import re
    cm_tp = sorted((run / "clearmap" / "reg").glob("TransformParameters.*.txt"))[-1]
    out = run / "check-points"
    out.mkdir(exist_ok=True)
    binary = Path(CLEARMAP) / "ClearMap" / "External" / "elastix" / "build"
    env = dict(os.environ, LD_LIBRARY_PATH=f"{binary / 'bin'}:{binary / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    subprocess.run([str(binary / "bin" / "transformix"), "-def", str(POINTS / "auto_points_elastix.txt"),
                    "-tp", str(cm_tp), "-out", str(out)], check=True, stdout=subprocess.DEVNULL, env=env)
    theirs = np.array([[float(v) for v in re.search(r"OutputPoint = \[ ([^\]]+) \]", line).group(1).split()][::-1]
                       for line in (out / "outputpoints.txt").read_text().splitlines()])
    bin_dir = Path(os.environ.get("BENCH_BIN", "/home/mahogny/github/claude/blockflow/target/release"))
    ours_csv = out / "ours_points.csv"
    subprocess.run([str(bin_dir / "atlas-registration"), "transform-points", "--registration",
                    str(run / "ours" / "reg"), "--input", str(POINTS / "auto_points.csv"),
                    "--output", str(ours_csv)], check=True, capture_output=True)
    with open(ours_csv) as handle:
        ours = np.array([[float(r[f"moving_{a}"]) for a in "zyx"] for r in csv.DictReader(handle)])
    d = np.linalg.norm(ours - theirs, axis=1) * 25.0
    return {"points": int(len(d)), "transform": cm_tp.name, "median_um": round(float(np.median(d)), 2),
            "p95_um": round(float(np.percentile(d, 95)), 2), "max_um": round(float(d.max()), 2),
            "note": "seed-to-seed noise of the native side alone is ~16 µm median (registration harness)"}


def e2e(run):
    """Cells matched between the two sides, then their atlas positions and
    regions compared twice: as the reference ships, and with the reference's
    points carried in the voxel-centre convention (see the README,
    "Atlas positions and the voxel convention")."""
    import pandas as pd
    from scipy.spatial import cKDTree
    cm = pd.read_csv(run / "clearmap" / "cells_atlas.csv")
    ours_dir = next((run / "ours").glob("*/cells/detect/tables/cell-density-filtered"))
    ours = pd.read_csv(ours_dir / "table.csv")
    record = {"filtered_cells": {"clearmap": int(len(cm)), "ours": int(len(ours))}}
    # Both are in box voxels: the reference's of the tile's own box, ours of
    # the fused box, which holds the same tile voxels shifted by the tile's
    # anchor (z) and per-slice wobble: the best integer shift within 8 voxels
    # is found first.
    a = cm[["z", "y", "x"]].to_numpy(float)
    b = ours[["z", "y", "x"]].to_numpy(float)
    if not (len(a) and len(b)):
        return record
    tree = cKDTree(b)
    best = max(((dz, dy, dx) for dz in range(-8, 9) for dy in range(-8, 9) for dx in range(-8, 9)),
               key=lambda t: int(np.isfinite(tree.query(a + np.array(t), distance_upper_bound=1.5)[0]).sum()))
    dist, j = tree.query(a + np.array(best), distance_upper_bound=2.0)
    hit = np.isfinite(dist)
    record["box_shift_zyx"] = list(best)
    record["matched_within_2_voxels"] = int(hit.sum())
    if not hit.any():
        return record
    our_region = ours["region"].to_numpy()[j[hit]]
    mine = ours[["atlas_z", "atlas_y", "atlas_x"]].to_numpy(float)[j[hit]]

    def compare(theirs_zyx, region):
        diff = mine - theirs_zyx
        d = np.linalg.norm(diff, axis=1) * 25.0
        return {"same_finest_region": round(float(np.mean(region == our_region)), 4),
                "atlas_distance_um": {"median": round(float(np.median(d)), 1),
                                      "p95": round(float(np.percentile(d, 95)), 1)},
                "ours_minus_theirs_zyx_voxels": [round(float(v), 3) for v in np.median(diff, axis=0)]}

    record["as_shipped"] = compare(cm[["zt", "yt", "xt"]].to_numpy(float)[hit], cm["region"].to_numpy()[hit])
    stitched = run / "clearmap" / "points_stitched0_xyz.npy"
    if stitched.exists():
        record["voxel_centres"] = compare(*_reference_centred(run, np.load(stitched)[hit]))
    return record


def _reference_centred(run, stitched0):
    """The reference's chain with `(p + 0.5) * f - 0.5` in place of its
    `resample_points`' `p * f`, and its annotation read at the nearest voxel
    centre in place of `astype(int)`: atlas positions `[z, y, x]` and regions."""
    sys.path.insert(0, CLEARMAP)
    import ClearMap.Alignment.Elastix as elastix
    import ClearMap.IO.IO as clearmap_io
    out = run / "check-centred"
    out.mkdir(exist_ok=True)
    f = np.load(run / "clearmap" / "resample_factor_xyz.npy")
    points = (stitched0 + 0.5) * f - 0.5
    for i, step in enumerate(("signal_to_auto", "auto_to_reference")):
        points = elastix.transform_points(points, transform_directory=str(run / "clearmap" / step), binary=True,
                                          temp_file=str(out / f"points.{i}.bin"),
                                          result_directory=str(out / f"points.{i}"), n_processes=4)
    points = np.asarray(points)
    annotation = np.asarray(clearmap_io.read(str(POINTS.parent / "atlas" / "annotation.tif")))  # (x, y, z)
    voxel = np.rint(points).astype(int)
    inside = np.all((voxel >= 0) & (voxel < np.array(annotation.shape)), axis=1)
    region = np.zeros(len(voxel), dtype=np.int64)
    region[inside] = annotation[tuple(voxel[inside].T)]
    return points[:, ::-1], region


def main():
    kind, run = sys.argv[1], Path(sys.argv[2])
    result = {"vessels": vessels, "registration": registration, "e2e": e2e}[kind](run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
