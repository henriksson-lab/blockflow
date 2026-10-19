#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Compare `cell-density` with the reference run on the same crop.

Reads `$CELLS_OUT/clearmap/cells_{raw,filtered}.csv` and our
`$CELLS_OUT/$CELLS_OURS_DIR/out.zarr/tables/cell-density{,-filtered}/table.csv`
(default `ours`), and writes `$CELLS_OUT/comparison.{json,md}` (or
`comparison-$CELLS_OURS_DIR.*`):
- one-to-one matching of detections within `$CELLS_MATCH_RADIUS` voxels
  (default 2, Euclidean in voxel units), closest pairs first; precision is
  the matched fraction of ours, recall of the reference's;
- on matched cells: the fraction at the identical voxel, and the Pearson
  and Spearman correlation of size and of `source` intensity;
- counts before and after the size band, and filtered cells of one side
  that the other side detected but filtered out (so where the band, not
  the detection, disagrees);
- wall time and peak memory from each side's `stats.json`, and throughput.

Needs numpy, scipy and pandas.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

OUT = Path(os.environ.get("CELLS_OUT", "/husky/henriksson/lightsheet-reference/cells/crop"))
RADIUS = float(os.environ.get("CELLS_MATCH_RADIUS", "2"))
OURS = os.environ.get("CELLS_OURS_DIR", "ours")
REF = os.environ.get("CELLS_REF_DIR", "clearmap")
NAME = "comparison" + ("" if OURS == "ours" else f"-{OURS}") + ("" if REF == "clearmap" else f"-vs-{REF}")


def load():
    setup = json.loads((OUT / "setup.json").read_text())
    start = np.asarray(setup["crop_start_zyx"], dtype=float)
    ref = {k: pd.read_csv(OUT / REF / f"cells_{k}.csv") for k in ("raw", "filtered")}
    tables = OUT / OURS / "out.zarr" / "tables"
    ours = {"raw": pd.read_csv(tables / "cell-density" / "table.csv"),
            "filtered": pd.read_csv(tables / "cell-density-filtered" / "table.csv")}
    for t in ref.values():
        t["pz"], t["py"], t["px"] = t["z"], t["y"], t["x"]
    for t in ours.values():
        t["pz"], t["py"], t["px"] = t["z"] - start[0], t["y"] - start[1], t["x"] - start[2]
    return setup, ref, ours


def points(t):
    return t[["pz", "py", "px"]].to_numpy(dtype=float)


def match(a, b, radius):
    """One-to-one pairs (i in a, j in b) within `radius`, closest first."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0)
    pairs = cKDTree(a).sparse_distance_matrix(cKDTree(b), radius, output_type="ndarray")
    order = np.lexsort((pairs["j"], pairs["i"], pairs["v"]))
    used_a, used_b, out, dist = set(), set(), [], []
    for i, j, v in zip(pairs["i"][order], pairs["j"][order], pairs["v"][order]):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
        dist.append(v)
    # sparse_distance_matrix leaves out zero distances; add identical points.
    exact = cKDTree(a).query_ball_tree(cKDTree(b), 0.0)
    for i, js in enumerate(exact):
        for j in js:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                out.append((i, j))
                dist.append(0.0)
    return np.asarray(out, dtype=int).reshape(-1, 2), np.asarray(dist)


def correlations(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return {"pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
            "median_ratio_ours_over_reference": float(np.median(y / np.maximum(x, 1e-9)))}


def compare(ref, ours, label):
    pairs, dist = match(points(ours), points(ref), RADIUS)
    o, r = ours.iloc[pairs[:, 0]], ref.iloc[pairs[:, 1]]
    return {
        "label": label, "ours": int(len(ours)), "reference": int(len(ref)), "matched": int(len(pairs)),
        "precision": len(pairs) / max(len(ours), 1), "recall": len(pairs) / max(len(ref), 1),
        "same_voxel": float(np.mean(dist == 0)) if len(dist) else None,
        "within_1": float(np.mean(dist <= 1)) if len(dist) else None,
        "size": correlations(r["size"].to_numpy(float), o["size"].to_numpy(float)),
        "size_equal": float(np.mean(r["size"].to_numpy() == o["size"].to_numpy())) if len(pairs) else None,
        "source": correlations(r["source"].to_numpy(float), o["source"].to_numpy(float)),
        "source_equal": float(np.mean(r["source"].to_numpy() == o["source"].to_numpy())) if len(pairs) else None,
    }


def main():
    setup, ref, ours = load()
    report = {"setup": setup, "radius": RADIUS,
              "raw": compare(ref["raw"], ours["raw"], "all detections"),
              "filtered": compare(ref["filtered"], ours["filtered"], "size band [50, 900)")}
    # Filtered cells of one side found by the other side's detection at all.
    pairs, _ = match(points(ours["filtered"]), points(ref["raw"]), RADIUS)
    report["ours_filtered_found_in_reference_raw"] = len(pairs) / max(len(ours["filtered"]), 1)
    pairs, _ = match(points(ref["filtered"]), points(ours["raw"]), RADIUS)
    report["reference_filtered_found_in_ours_raw"] = len(pairs) / max(len(ref["filtered"]), 1)
    voxels = setup["voxels"]
    for side, directory in (("clearmap", REF), ("ours", OURS)):
        path = OUT / directory / "stats.json"
        if path.exists():
            stats = json.loads(path.read_text())
            stats["megavoxels_per_s"] = voxels / stats["wall_s"] / 1e6
            report[f"stats_{side}"] = stats
    (OUT / f"{NAME}.json").write_text(json.dumps(report, indent=1))
    lines = [f"crop {setup['crop_start_zyx']} + {setup['crop_shape_zyx']} of `{setup['tile']}` "
             f"({voxels / 1e6:.1f} M voxels), match radius {RADIUS:g} voxels", "",
             "| set | ours | reference | matched | precision | recall | same voxel | size r (Pearson / Spearman), equal | source r, equal |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in ("raw", "filtered"):
        c = report[key]
        size = c["size"] or {"pearson": float("nan"), "spearman": float("nan")}
        src = c["source"] or {"pearson": float("nan"), "spearman": float("nan")}
        lines.append(f"| {c['label']} | {c['ours']} | {c['reference']} | {c['matched']} | "
                     f"{100 * c['precision']:.1f}% | {100 * c['recall']:.1f}% | {100 * (c['same_voxel'] or 0):.1f}% | "
                     f"{size['pearson']:.3f} / {size['spearman']:.3f}, {100 * (c['size_equal'] or 0):.1f}% | "
                     f"{src['pearson']:.3f} / {src['spearman']:.3f}, {100 * (c['source_equal'] or 0):.1f}% |")
    lines += ["", f"Filtered cells of ours detected (any size) by the reference: "
                  f"{100 * report['ours_filtered_found_in_reference_raw']:.1f}%; of the reference's, by ours: "
                  f"{100 * report['reference_filtered_found_in_ours_raw']:.1f}%.", "",
              "| side | wall | user+sys | peak RSS (largest process) | tree PSS | M voxels/s (wall) |",
              "|---|---|---|---|---|---|"]
    for side in ("clearmap", "ours"):
        s = report.get(f"stats_{side}")
        if s:
            lines.append(f"| {side} | {s['wall_s']:.1f} s | {s.get('user_s', 0) + s.get('system_s', 0):.0f} s | "
                         f"{s.get('max_rss_gb', 0):.2f} GB | {s['tree_peak_pss_gb']:.2f} GB | {s['megavoxels_per_s']:.2f} |")
    (OUT / f"{NAME}.md").write_text("\n".join(lines) + "\n")
    print((OUT / f"{NAME}.md").read_text())


if __name__ == "__main__":
    main()
