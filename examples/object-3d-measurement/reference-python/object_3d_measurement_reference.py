#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SHAPE = (32, 48, 56)
SPACING = (1.5, 0.75, 0.5)
OBJECTS_PER_IMAGE = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".tmp/object-3d-measurement/reference")
    parser.add_argument("--images", type=int, default=10)
    return parser.parse_args()


def object_box(image, local):
    z0 = 2 + (local % 2) * 13 + (image % 2)
    y0 = 4 + (local // 2) * 20 + (image % 3)
    x0 = 5 + (local % 2) * 25 + ((image + local) % 4)
    dz = 5 + ((image + local) % 4)
    dy = 8 + ((2 * image + local) % 5)
    dx = 9 + ((image + 2 * local) % 6)
    return (z0, y0, x0), (dz, dy, dx)


def row_for(image, local):
    label = image * 100 + local + 1
    start, extent = object_box(image, local)
    bbox_min = start
    bbox_max = tuple(start[axis] + extent[axis] for axis in range(3))
    count = extent[0] * extent[1] * extent[2]
    physical_extent = tuple(extent[axis] * SPACING[axis] for axis in range(3))
    return {
        "image": image,
        "label": label,
        "count": count,
        "bbox_min_z": bbox_min[0],
        "bbox_min_y": bbox_min[1],
        "bbox_min_x": bbox_min[2],
        "bbox_max_z": bbox_max[0],
        "bbox_max_y": bbox_max[1],
        "bbox_max_x": bbox_max[2],
        "physical_bbox_extent_z": physical_extent[0],
        "physical_bbox_extent_y": physical_extent[1],
        "physical_bbox_extent_x": physical_extent[2],
    }


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        row_for(image, local)
        for image in range(args.images)
        for local in range(OBJECTS_PER_IMAGE)
    ]

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
        "spacing_x": SPACING[2],
        "spacing_y": SPACING[1],
        "spacing_z": SPACING[0],
        "total_voxels": sum(row["count"] for row in rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
