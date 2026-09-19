#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path


HEIGHT = 72
WIDTH = 96
OBJECTS_PER_IMAGE = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".tmp/colocalization/reference")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--fixture-dir", required=True)
    return parser.parse_args()


def channels(image, local, y, x):
    base_a = 15.0 + 5.0 * local + 2.0 * (image % 6)
    a = base_a + ((x + 2 * y + image) % 23)
    if local % 2 == 0:
        b = 4.0 + 1.7 * a + ((3 * x + y + image) % 7)
    else:
        b = 140.0 - 1.2 * a + ((x + 5 * y + image) % 9)
    return a, b


def derived(row):
    n = row["finite_count"]
    cov = row["sum_ab"] - row["sum_a"] * row["sum_b"] / n
    var_a = row["sum_a2"] - row["sum_a"] * row["sum_a"] / n
    var_b = row["sum_b2"] - row["sum_b"] * row["sum_b"] / n
    denom = math.sqrt(var_a * var_b)
    pearson = cov / denom if denom > 0 else math.nan
    slope = cov / var_a if var_a > 0 else math.nan
    overlap_denom = math.sqrt(row["sum_a2"] * row["sum_b2"])
    overlap = row["sum_ab"] / overlap_denom if overlap_denom > 0 else math.nan
    manders_m1 = row["positive_a_where_b"] / row["positive_a"]
    manders_m2 = row["positive_b_where_a"] / row["positive_b"]
    return pearson, slope, overlap, manders_m1, manders_m2


def measurements(images, fixture_dir):
    rows = []
    for image in range(images):
        with (fixture_dir / f"objects-{image:03}.csv").open(newline="") as handle:
            objects = list(csv.DictReader(handle))
        for obj in objects:
            label = int(obj["label"])
            local = int(obj["local"])
            y0 = int(obj["y0"])
            x0 = int(obj["x0"])
            height = int(obj["height"])
            width = int(obj["width"])
            row = {
                "image": image,
                "label": label,
                "count": height * width,
                "finite_count": height * width,
                "sum_a": 0.0,
                "sum_b": 0.0,
                "sum_a2": 0.0,
                "sum_b2": 0.0,
                "sum_ab": 0.0,
                "positive_a": 0.0,
                "positive_b": 0.0,
                "positive_a_where_b": 0.0,
                "positive_b_where_a": 0.0,
            }
            for y in range(y0, y0 + height):
                for x in range(x0, x0 + width):
                    a, b = channels(image, local, y, x)
                    row["sum_a"] += a
                    row["sum_b"] += b
                    row["sum_a2"] += a * a
                    row["sum_b2"] += b * b
                    row["sum_ab"] += a * b
                    if a > 0:
                        row["positive_a"] += a
                        if b > 0:
                            row["positive_a_where_b"] += a
                    if b > 0:
                        row["positive_b"] += b
                        if a > 0:
                            row["positive_b_where_a"] += b
            rows.append(row)
    return rows


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(args.fixture_dir)
    started = time.perf_counter()
    rows = measurements(args.images, fixture_dir)
    pipeline_seconds = time.perf_counter() - started

    with (out / "objects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "count",
                "finite_count",
                "pearson",
                "slope_b_on_a",
                "overlap_coefficient",
                "manders_m1",
                "manders_m2",
                "sum_a",
                "sum_b",
                "sum_ab",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            pearson, slope, overlap, m1, m2 = derived(row)
            writer.writerow(
                {
                    "image": row["image"],
                    "label": row["label"],
                    "count": row["count"],
                    "finite_count": row["finite_count"],
                    "pearson": f"{pearson:.6f}",
                    "slope_b_on_a": f"{slope:.6f}",
                    "overlap_coefficient": f"{overlap:.6f}",
                    "manders_m1": f"{m1:.6f}",
                    "manders_m2": f"{m2:.6f}",
                    "sum_a": f"{row['sum_a']:.6f}",
                    "sum_b": f"{row['sum_b']:.6f}",
                    "sum_ab": f"{row['sum_ab']:.6f}",
                }
            )

    summary = {
        "finite_pairs": sum(row["finite_count"] for row in rows),
        "images": args.images,
        "objects": len(rows),
        "pairs": sum(row["count"] for row in rows),
        "pipeline_seconds": pipeline_seconds,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
