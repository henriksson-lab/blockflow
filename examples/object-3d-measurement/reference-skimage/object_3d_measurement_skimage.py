#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from skimage import measure


SHAPE = (32, 48, 56)
SPACING = (1.5, 0.75, 0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--out", default=".tmp/object-3d-measurement/skimage")
    parser.add_argument("--images", type=int, default=10)
    return parser.parse_args()


def load_labels(path):
    labels = np.zeros(SHAPE, dtype=np.int64)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(row["label"])
            z0 = int(row["z0"])
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            dz = int(row["dz"])
            dy = int(row["dy"])
            dx = int(row["dx"])
            labels[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] = label
    return labels


def rows_for_image(image, fixture_dir):
    labels = load_labels(fixture_dir / f"boxes-{image:03}.csv")
    rows = []
    for prop in measure.regionprops(labels, spacing=SPACING):
        bbox = prop.bbox
        extent = [(bbox[axis + 3] - bbox[axis]) * SPACING[axis] for axis in range(3)]
        rows.append(
            {
                "image": image,
                "label": int(prop.label),
                "count": int(prop.num_pixels),
                "bbox_min_z": int(bbox[0]),
                "bbox_min_y": int(bbox[1]),
                "bbox_min_x": int(bbox[2]),
                "bbox_max_z": int(bbox[3]),
                "bbox_max_y": int(bbox[4]),
                "bbox_max_x": int(bbox[5]),
                "physical_bbox_extent_z": extent[0],
                "physical_bbox_extent_y": extent[1],
                "physical_bbox_extent_x": extent[2],
            }
        )
    return rows


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    fixture_dir = Path(args.fixture_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows = []
    for image in range(args.images):
        rows.extend(rows_for_image(image, fixture_dir))
    pipeline_seconds = time.perf_counter() - started

    with (out / "objects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "count",
                "bbox_min_z",
                "bbox_min_y",
                "bbox_min_x",
                "bbox_max_z",
                "bbox_max_y",
                "bbox_max_x",
                "physical_bbox_extent_z",
                "physical_bbox_extent_y",
                "physical_bbox_extent_x",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "physical_bbox_extent_z": f"{row['physical_bbox_extent_z']:.6f}",
                    "physical_bbox_extent_y": f"{row['physical_bbox_extent_y']:.6f}",
                    "physical_bbox_extent_x": f"{row['physical_bbox_extent_x']:.6f}",
                }
            )

    summary = {
        "images": args.images,
        "objects": len(rows),
        "pipeline_seconds": pipeline_seconds,
        "spacing_x": SPACING[2],
        "spacing_y": SPACING[1],
        "spacing_z": SPACING[0],
        "total_voxels": sum(row["count"] for row in rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"objects={len(rows)} output={out}")


if __name__ == "__main__":
    main()
