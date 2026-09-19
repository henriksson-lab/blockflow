#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEIGHT = 96
WIDTH = 128
OBJECTS_PER_IMAGE = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".tmp/percent-positive/reference")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=110.0)
    return parser.parse_args()


def object_rect(image_index, local):
    row = local // 3
    col = local % 3
    y0 = 10 + row * 36 + (image_index % 3)
    x0 = 12 + col * 36 + ((image_index + local) % 5)
    height = 16 + ((image_index + local) % 4)
    width = 18 + ((2 * image_index + local) % 5)
    return y0, x0, height, width


def base_intensity(image_index, local):
    return 58.0 + 9.0 * local + 4.0 * (image_index % 7)


def pixel_intensity(image_index, local, y, x):
    return base_intensity(image_index, local) + ((y + 2 * x + image_index) % 11)


def fixture_rows(images, threshold):
    rows = []
    positive = 0
    negative = 0
    total_area = 0
    marker_sum = 0.0
    for image_index in range(images):
        for local in range(OBJECTS_PER_IMAGE):
            label = image_index * 100 + local + 1
            y0, x0, height, width = object_rect(image_index, local)
            count = height * width
            values = [
                pixel_intensity(image_index, local, y, x)
                for y in range(y0, y0 + height)
                for x in range(x0, x0 + width)
            ]
            intensity_sum = sum(values)
            mean = intensity_sum / count
            is_positive = mean >= threshold
            positive += int(is_positive)
            negative += int(not is_positive)
            total_area += count
            marker_sum += intensity_sum
            rows.append(
                {
                    "image": image_index,
                    "label": label,
                    "count": count,
                    "centroid_y": y0 + (height - 1) / 2.0,
                    "centroid_x": x0 + (width - 1) / 2.0,
                    "mean_intensity": mean,
                    "sum_intensity": intensity_sum,
                    "positive": is_positive,
                }
            )
    return rows, positive, negative, total_area, marker_sum


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, positive, negative, total_area, marker_sum = fixture_rows(
        args.images, args.threshold
    )

    with (out / "objects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "count",
                "centroid_y",
                "centroid_x",
                "mean_intensity",
                "sum_intensity",
                "positive",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "centroid_y": f"{row['centroid_y']:.6f}",
                    "centroid_x": f"{row['centroid_x']:.6f}",
                    "mean_intensity": f"{row['mean_intensity']:.6f}",
                    "sum_intensity": f"{row['sum_intensity']:.6f}",
                    "positive": "true" if row["positive"] else "false",
                }
            )

    summary = {
        "images": args.images,
        "objects": len(rows),
        "threshold": args.threshold,
        "positive": positive,
        "negative": negative,
        "percent_positive": positive / len(rows),
        "total_area": total_area,
        "marker_sum": marker_sum,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
