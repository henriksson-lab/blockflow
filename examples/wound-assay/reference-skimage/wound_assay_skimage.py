#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np


HEIGHT = 96
WIDTH = 144


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--out", default=".tmp/wound-assay/skimage")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=100.0)
    return parser.parse_args()


def measure_image(path, threshold):
    image = iio.imread(path).astype(np.float64, copy=False)
    mask = image < threshold
    open_area = int(mask.sum())
    profile = mask.sum(axis=0).astype(np.int64)
    return open_area, profile


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    fixture_dir = Path(args.fixture_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows = []
    profile = np.zeros(WIDTH, dtype=np.int64)
    total_open = 0
    for image in range(args.images):
        open_area, image_profile = measure_image(
            fixture_dir / f"wound-{image:03}.pgm", args.threshold
        )
        total_open += open_area
        profile += image_profile
        rows.append(
            {
                "image": image,
                "open_area": open_area,
                "covered_area": HEIGHT * WIDTH - open_area,
                "open_fraction": open_area / (HEIGHT * WIDTH),
            }
        )
    pipeline_seconds = time.perf_counter() - started

    with (out / "images.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "open_area", "covered_area", "open_fraction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "open_fraction": f"{row['open_fraction']:.6f}"})

    with (out / "profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["x", "open_count", "open_fraction"],
            lineterminator="\n",
        )
        writer.writeheader()
        denom = args.images * HEIGHT
        for x, count in enumerate(profile):
            writer.writerow(
                {"x": x, "open_count": int(count), "open_fraction": f"{count / denom:.6f}"}
            )

    total = args.images * HEIGHT * WIDTH
    summary = {
        "covered_area": total - total_open,
        "height": HEIGHT,
        "images": args.images,
        "open_area": total_open,
        "open_fraction": total_open / total,
        "pipeline_seconds": pipeline_seconds,
        "threshold": args.threshold,
        "width": WIDTH,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"open_area={total_open} output={out}")


if __name__ == "__main__":
    main()
