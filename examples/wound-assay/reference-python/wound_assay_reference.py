#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEIGHT = 96
WIDTH = 144


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".tmp/wound-assay/reference")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=100.0)
    return parser.parse_args()


def wound_bounds(image, y):
    center = WIDTH // 2 + ((image % 5) - 2)
    half = 13 + (image % 4) + ((y + image) % 9) // 3
    drift = ((y * 7 + image * 3) % 11) - 5
    left = center - half + int(drift / 2)
    right = center + half + int(drift / 3)
    return max(0, left), min(WIDTH, right)


def intensity(image, y, x):
    left, right = wound_bounds(image, y)
    if left <= x < right:
        return 42.0 + ((x + 3 * y + image) % 17)
    return 166.0 + ((2 * x + y + image) % 29)


def measure(images, threshold):
    image_rows = []
    profile = [0 for _ in range(WIDTH)]
    total_open = 0
    total_pixels = images * HEIGHT * WIDTH
    for image in range(images):
        open_area = 0
        for y in range(HEIGHT):
            for x in range(WIDTH):
                is_open = intensity(image, y, x) < threshold
                if is_open:
                    open_area += 1
                    profile[x] += 1
        total_open += open_area
        image_rows.append(
            {
                "image": image,
                "open_area": open_area,
                "covered_area": HEIGHT * WIDTH - open_area,
                "open_fraction": open_area / (HEIGHT * WIDTH),
            }
        )
    return image_rows, profile, total_open, total_pixels


def main():
    args = parse_args()
    if args.images < 1:
        raise SystemExit("--images must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image_rows, profile, total_open, total_pixels = measure(args.images, args.threshold)

    with (out / "images.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "open_area", "covered_area", "open_fraction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in image_rows:
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
                {
                    "x": x,
                    "open_count": count,
                    "open_fraction": f"{count / denom:.6f}",
                }
            )

    summary = {
        "covered_area": total_pixels - total_open,
        "height": HEIGHT,
        "images": args.images,
        "open_area": total_open,
        "open_fraction": total_open / total_pixels,
        "threshold": args.threshold,
        "width": WIDTH,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
