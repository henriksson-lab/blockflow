#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Compare `tile-stitching` with the reference run on the same tiles.

Reads `$STITCH_OUT/$REF` (default `clearmap-patched`) and `$STITCH_OUT/$OURS`
(default `ours`) and writes `$STITCH_OUT/comparison-$OURS-vs-$REF.{json,md}`:
- rigid: per-tile position relative to tile 0, per axis, ours minus reference;
- per slice: for every neighbouring pair and every plane of the first tile,
  the in-plane displacement of the second tile relative to the first (which
  is what the fusion sees, and is free of each side's per-slice gauge), ours
  minus reference, over planes valid on both sides; also the per-tile
  positions relative to tile 0's rigid position;
- fused volume: Pearson correlation in the voxels two or more tiles cover,
  in the voxels one tile covers, and over everything both cover, with the
  reference mapped onto ours by one offset (tile 0's first plane) and by a
  per-plane offset (the median over tiles), see below;
- wall time and peak memory from each side's `stats.json`.

Needs a Python with `zarr` (3.x) and numpy.
"""
import json
import os
from pathlib import Path

import numpy as np
import zarr

OUT = Path(os.environ.get("STITCH_OUT", "/husky/henriksson/lightsheet-reference/stitching"))
REF = os.environ.get("REF", "clearmap-patched")
OURS = os.environ.get("OURS", "ours")
NAME = f"comparison-{OURS}-vs-{REF}"


def load_reference():
    rigid = json.loads((OUT / REF / "rigid_positions.json").read_text())
    wob = np.load(OUT / REF / "wobbly_positions.npz")
    n = len(rigid["tiles"])
    tiles = []
    for i in range(n):
        tile_position = [int(v) for v in wob["tile_positions"][i]]  # (X, Y)
        tiles.append({
            "grid": tile_position,
            # (x, y, z) -> (z, y, x)
            "rigid": np.asarray(rigid["tiles"][i]["position"])[::-1],
            "slices": np.asarray(wob[f"positions_{i}"])[:, ::-1].astype(float),
            "valid": np.asarray(wob[f"status_{i}"]) >= 0,
        })
    # The reference's stitched array starts at its `origin_wobbly`, which is
    # `min(lower, 0)` per axis, not at `lower`.
    return tiles, np.minimum(np.asarray(wob["lower"]), 0)[::-1]


def load_ours():
    data = json.loads((OUT / OURS / "fused.zarr.positions.json").read_text())
    tiles = []
    for t in data["tiles"]:
        rigid = np.asarray(t["rigid"])
        if "slices" in t:
            slices = np.asarray(t["slices"], dtype=float)
            valid = np.asarray([s != "invalid" for s in t["status"]])
        else:  # --rigid-only: every plane at the rigid position
            planes = np.arange(t["shape"][0])
            slices = np.stack([rigid[0] + planes, np.full_like(planes, rigid[1]),
                               np.full_like(planes, rigid[2])], axis=1).astype(float)
            valid = np.ones(len(planes), dtype=bool)
        tiles.append({"grid": t["grid"], "rigid": rigid, "slices": slices, "valid": valid,
                      "shape": t["shape"]})
    return tiles, np.asarray(data["origin"]), data


def summary(values):
    values = np.abs(np.asarray(values, dtype=float))
    if values.size == 0:
        return None
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max()), "within_1": float(np.mean(values <= 1)),
            "exact": float(np.mean(values == 0)), "n": int(values.size)}


def main():
    ref, ref_lower = load_reference()
    ours, our_origin, our_json = load_ours()
    # Match tiles by grid index.
    by_grid = {tuple(t["grid"]): t for t in ref}
    pairs = [(o, by_grid[tuple(o["grid"])]) for o in ours]
    report = {"tiles": len(pairs)}

    # Rigid positions relative to tile 0.
    rigid_rows = []
    for o, r in pairs:
        d = (o["rigid"] - pairs[0][0]["rigid"]) - (r["rigid"] - pairs[0][1]["rigid"])
        rigid_rows.append({"grid": o["grid"], "ours": (o["rigid"] - pairs[0][0]["rigid"]).tolist(),
                           "reference": (r["rigid"] - pairs[0][1]["rigid"]).tolist(),
                           "difference_zyx": d.tolist()})
    report["rigid"] = rigid_rows
    report["rigid_max_abs_difference_zyx"] = np.max(
        np.abs([row["difference_zyx"] for row in rigid_rows]), axis=0).tolist()

    # Per-slice pair displacements.
    index = {tuple(o["grid"]): i for i, (o, _) in enumerate(pairs)}
    neighbours = []
    for (gx, gy), i in index.items():
        for step in ((1, 0), (0, 1)):
            j = index.get((gx + step[0], gy + step[1]))
            if j is not None:
                neighbours.append((i, j))
    diff_y, diff_x, per_pair = [], [], []
    for i, j in neighbours:
        dy, dx = [], []
        o_pre, o_post = pairs[i][0], pairs[j][0]
        r_pre, r_post = pairs[i][1], pairs[j][1]
        for plane in range(len(o_pre["slices"])):
            rel = []
            for pre, post in ((o_pre, o_post), (r_pre, r_post)):
                z = pre["slices"][plane][0]
                k = int(round(z - post["slices"][0][0]))
                if not (0 <= k < len(post["slices"])) or not (pre["valid"][plane] and post["valid"][k]):
                    rel = None
                    break
                rel.append(post["slices"][k][1:] - pre["slices"][plane][1:])
            if rel is None:
                continue
            d = rel[0] - rel[1]
            dy.append(d[0])
            dx.append(d[1])
        diff_y += dy
        diff_x += dx
        per_pair.append({"pre": o_pre["grid"], "post": o_post["grid"],
                         "planes": len(dy), "y": summary(dy), "x": summary(dx)})
    report["wobbly_pair_displacement_difference"] = {"y": summary(diff_y), "x": summary(diff_x),
                                                     "pairs": per_pair}
    # Per-tile positions relative to tile 0's rigid position, plane by plane.
    abs_y, abs_x = [], []
    for o, r in pairs:
        n = min(len(o["slices"]), len(r["slices"]))
        ok = o["valid"][:n] & r["valid"][:n]
        d = ((o["slices"][:n] - pairs[0][0]["rigid"]) - (r["slices"][:n] - pairs[0][1]["rigid"]))[ok]
        abs_y += list(d[:, 1])
        abs_x += list(d[:, 2])
    report["wobbly_tile_position_difference"] = {"y": summary(abs_y), "x": summary(abs_x)}

    # Fused volumes, the reference's frame mapped into ours.
    # Both fit in memory at this size; whole reads avoid strided plane reads.
    cm = np.load(OUT / REF / "stitched.npy")  # (x, y, z)
    our = zarr.open_array(str(OUT / OURS / "fused.zarr" / "0"), mode="r")[...]
    p_ours = pairs[0][0]["slices"][0] - our_origin
    p_ref = pairs[0][1]["slices"][0] - ref_lower
    offset = (p_ours - p_ref).astype(int)  # ours[v + offset] ~ reference[v]
    report["fused_shape_ours_zyx"] = list(our.shape)
    report["fused_shape_reference_zyx"] = list(cm.shape[::-1])
    report["frame_offset_zyx"] = offset.tolist()
    # Two ways to map a reference plane into ours: one offset for the volume
    # (tile 0's first plane), and per plane the median over tiles of the
    # offset between the two placements, which removes the per-slice gauge
    # (a common in-plane translation of the whole plane that each side picks
    # its own way) and leaves only the seams' relative placement.
    tile_yx = np.asarray(pairs[0][0]["shape"][1:])
    frames = ("volume", "per_plane")
    sums = {(f, k): np.zeros(6) for f in frames for k in ("overlap", "single", "all")}
    equal = {key: 0 for key in sums}
    for zc in range(cm.shape[2]):
        count = np.zeros(cm.shape[1::-1], dtype=np.uint8)  # (y, x), in the reference frame
        deltas = []
        for o, r in pairs:
            k = int(round(zc + ref_lower[0] - r["slices"][0][0]))
            if 0 <= k < len(r["slices"]):
                y0, x0 = (r["slices"][k][1:] - ref_lower[1:]).astype(int)
                count[max(y0, 0):y0 + tile_yx[0], max(x0, 0):x0 + tile_yx[1]] += 1
                if k < len(o["slices"]):
                    deltas.append((o["slices"][k] - our_origin) - (r["slices"][k] - ref_lower))
        if not deltas:
            continue
        a = np.asarray(cm[:, :, zc]).T.astype(np.float64)
        for frame in frames:
            shift = offset if frame == "volume" else np.round(np.median(deltas, axis=0)).astype(int)
            zo, ys, xs = zc + shift[0], shift[1], shift[2]
            if not 0 <= zo < our.shape[0]:
                continue
            b = np.zeros_like(a)
            src = np.asarray(our[zo, max(ys, 0):ys + a.shape[0], max(xs, 0):xs + a.shape[1]], dtype=np.float64)
            b[max(-ys, 0):max(-ys, 0) + src.shape[0], max(-xs, 0):max(-xs, 0) + src.shape[1]] = src
            for key, mask in (("overlap", count >= 2), ("single", count == 1), ("all", count >= 1)):
                x, y = a[mask], b[mask]
                sums[frame, key] += [x.size, x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()]
                equal[frame, key] += int(np.sum(x == y))
    corr = {}
    for (frame, key), (n, sx, sy, sxy, sxx, syy) in sums.items():
        cov = sxy / n - sx * sy / n**2
        corr.setdefault(frame, {})[key] = {
            "voxels": int(n),
            "pearson": float(cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))),
            "equal_fraction": equal[frame, key] / n, "mean_reference": sx / n, "mean_ours": sy / n}
    report["fused_correlation"] = corr

    for side, directory in (("clearmap", REF), ("ours", OURS)):
        path = OUT / directory / "stats.json"
        if path.exists():
            report[f"stats_{side}"] = json.loads(path.read_text())
    timing = OUT / REF / "timing.json"
    if timing.exists():
        report["clearmap_stage_timing"] = json.loads(timing.read_text())
    report["reference"], report["ours"] = REF, OURS
    (OUT / f"{NAME}.json").write_text(json.dumps(report, indent=1))
    write_markdown(report)
    print((OUT / f"{NAME}.md").read_text())


def write_markdown(r):
    lines = ["| measure | value |", "|---|---|"]
    lines.append(f"| tiles | {r['tiles']} |")
    lines.append(f"| rigid, max abs difference (z, y, x) | {r['rigid_max_abs_difference_zyx']} |")
    for name, key in (("per-slice pair displacement", "wobbly_pair_displacement_difference"),
                      ("per-slice tile position", "wobbly_tile_position_difference")):
        for axis in ("y", "x"):
            s = r[key][axis]
            if s:
                lines.append(f"| {name} {axis}: median / p95 / max abs, within 1 voxel | "
                             f"{s['median']:.1f} / {s['p95']:.1f} / {s['max']:.0f}, {100 * s['within_1']:.1f}% (n={s['n']}) |")
    for frame, by_key in r["fused_correlation"].items():
        for key, c in by_key.items():
            lines.append(f"| fused Pearson r, {frame} frame, {key} ({c['voxels']} voxels) | {c['pearson']:.4f} "
                         f"(equal {100 * c['equal_fraction']:.1f}%) |")
    for side, label in (("clearmap", r["reference"]), ("ours", r["ours"])):
        s = r.get(f"stats_{side}")
        if s:
            lines.append(f"| `{label}`: wall / user+sys / peak RSS (largest process) / tree PSS | "
                         f"{s['wall_s']:.0f} s / {s.get('user_s', 0) + s.get('system_s', 0):.0f} s / "
                         f"{s.get('max_rss_gb', 0):.1f} GB / {s['tree_peak_pss_gb']:.1f} GB |")
    lines.append("")
    lines.append("| tile (x, y) | ours rigid (z, y, x) | reference rigid | difference |")
    lines.append("|---|---|---|---|")
    for row in r["rigid"]:
        lines.append(f"| {row['grid']} | {row['ours']} | {row['reference']} | {row['difference_zyx']} |")
    (OUT / f"{NAME}.md").write_text(f"`{OURS}` against `{REF}`\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
