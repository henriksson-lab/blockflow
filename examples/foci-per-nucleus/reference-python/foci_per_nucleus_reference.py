#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEIGHT = 104
WIDTH = 136
NUCLEI_PER_IMAGE = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".tmp/foci-per-nucleus/reference")
    parser.add_argument("--images", type=int, default=10)
    return parser.parse_args()


def nucleus_rect(image, local):
    y0 = 12 + (local // 3) * 42 + (image % 4)
    x0 = 10 + (local % 3) * 40 + ((image + local) % 6)
    height = 24 + ((image + 2 * local) % 5)
    width = 26 + ((2 * image + local) % 7)
    return y0, x0, height, width


def foci_specs(image):
    specs = []
    for local in range(NUCLEI_PER_IMAGE):
        y0, x0, height, width = nucleus_rect(image, local)
        count = 1 + ((image + local) % 3)
        for index in range(count):
            y = y0 + 3 + ((image + 5 * index + local) % max(1, height - 6))
            x = x0 + 4 + ((2 * image + 7 * index + local) % max(1, width - 8))
            intensity = 180.0 + 11.0 * index + 3.0 * local + (image % 5)
            specs.append((y, x, intensity))
    specs.append((2 + image % 5, 3 + image % 7, 99.0))
    y0, x0, height, width = nucleus_rect(image, 0)
    specs.append((y0, x0 + width, 120.0))
    return specs


def fixture_rows(images):
    nuclei = []
    foci = []
    total_assigned = 0
    unassigned = 0
    for image in range(images):
        label_at = {}
        for local in range(NUCLEI_PER_IMAGE):
            label = image * 100 + local + 1
            y0, x0, height, width = nucleus_rect(image, local)
            for y in range(y0, y0 + height):
                for x in range(x0, x0 + width):
                    label_at[(y, x)] = label
            nuclei.append(
                {
                    "image": image,
                    "label": label,
                    "area": height * width,
                    "foci_count": 0,
                    "foci_intensity_sum": 0.0,
                }
            )
        by_label = {row["label"]: row for row in nuclei if row["image"] == image}
        for focus_index, (y, x, intensity) in enumerate(foci_specs(image), start=1):
            label = label_at.get((y, x), 0)
            if label:
                by_label[label]["foci_count"] += 1
                by_label[label]["foci_intensity_sum"] += intensity
                total_assigned += 1
            else:
                unassigned += 1
            foci.append(
                {
                    "image": image,
                    "focus": image * 1000 + focus_index,
                    "y": y,
                    "x": x,
                    "intensity": intensity,
                    "nucleus_label": label,
                }
            )
    return nuclei, foci, total_assigned, unassigned


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nuclei, foci, assigned, unassigned = fixture_rows(args.images)

    with (out / "nuclei.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "area",
                "foci_count",
                "foci_intensity_sum",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in nuclei:
            writer.writerow(
                {
                    **row,
                    "foci_intensity_sum": f"{row['foci_intensity_sum']:.6f}",
                }
            )

    with (out / "foci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "focus", "y", "x", "intensity", "nucleus_label"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in foci:
            writer.writerow({**row, "intensity": f"{row['intensity']:.6f}"})

    summary = {
        "assigned_foci": assigned,
        "images": args.images,
        "nuclei": len(nuclei),
        "total_foci": len(foci),
        "unassigned_foci": unassigned,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
