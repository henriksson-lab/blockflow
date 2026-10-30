#!/usr/bin/env python3
"""Visual check of the DIC maps written by `dic_maps --write`.

For a few annotated keyframe planes: raw DIC, phase-like reconstruction,
smoothness and nucleus-likeness, with the annotated nuclei (keyframes in that
plane) drawn as circles of their radius. Reads the zarr directly.

    python3 scripts/dic_panels.py <rec.ome.zarr/DIC> <out.png> [n_panels]
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    image, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    group = image.rstrip("/").split("/")[-1]
    parent = image.rstrip("/")[: -len(group)]
    cal = json.load(open(f"{image}/calibration/dic/summary.json"))
    px = cal["pixel_um"]
    det = [d for d in csv.DictReader(open(f"{image}/tracking/endrov/detections.csv")) if d["source"] == "keyframe"]
    planes = defaultdict(list)
    for d in det:
        planes[(int(d["t_index"]), int(round(float(d["z"]))))].append(d)
    # the planes with most annotated nuclei, spread over time
    ranked = sorted(planes, key=lambda k: -len(planes[k]))
    times = sorted({k[0] for k in planes})
    gap = max(1, (times[-1] - times[0]) // (2 * n)) if times else 1
    chosen = []
    for k in ranked:
        if all(abs(k[0] - c[0]) >= gap for c in chosen):
            chosen.append(k)
        if len(chosen) == n:
            break
    chosen.sort()
    arrays = {name: zarr.open(f"{parent}{group}{suffix}/0", mode="r") for name, suffix in
              [("DIC", ""), ("phase", "_phase"), ("smoothness", "_smooth"), ("nucleus-likeness", "_nucleus")]}
    sh = cal["shell"]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(24, 5.2 * len(chosen)), squeeze=False)
    for row, (t, z) in enumerate(chosen):
        for col, (name, arr) in enumerate(arrays.items()):
            plane = np.asarray(arr[t, 0, z]).astype(float)
            ax = axes[row, col]
            cmap = "gray" if col < 2 else ("viridis" if col == 2 else "magma")
            ax.imshow(plane, cmap=cmap, vmin=np.percentile(plane, 1), vmax=np.percentile(plane, 99.5))
            for d in planes[(t, z)]:
                ax.add_patch(plt.Circle((float(d["x"]), float(d["y"])), float(d["radius"]) / px,
                                        fill=False, color="red" if col < 2 else "white", lw=1.2))
            tt = np.linspace(0, 2 * np.pi, 200)
            a, b, ang = sh["a"] / px, sh["b"] / px, np.radians(sh["angle_deg"])
            f = 1 - ((z * cal["z_step_um"] - sh["cz"]) / sh["c"]) ** 2 if np.isfinite(sh["c"]) else 1
            if f > 0:
                a, b = a * np.sqrt(f), b * np.sqrt(f)
                cx, cy = sh["cx"] / px, sh["cy"] / px
                ax.plot(cx + a * np.cos(tt) * np.cos(ang) - b * np.sin(tt) * np.sin(ang),
                        cy + a * np.cos(tt) * np.sin(ang) + b * np.sin(tt) * np.cos(ang), "y--", lw=0.7)
            m = 0.15 * sh["a"] / px
            x0, x1 = max(0, (sh["cx"] - sh["a"]) / px - m), min(plane.shape[1], (sh["cx"] + sh["a"]) / px + m)
            y0, y1 = max(0, (sh["cy"] - sh["a"]) / px - m), min(plane.shape[0], (sh["cy"] + sh["a"]) / px + m)
            if col == 0:
                th = np.radians(cal["theta_deg"])
                ax.arrow(x0 + 50, y0 + 50, 35 * np.cos(th), 35 * np.sin(th), color="yellow", width=2)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_title(f"{group} t{t} z{z}: {name} ({len(planes[(t, z)])} annotated)")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=60)


if __name__ == "__main__":
    main()
