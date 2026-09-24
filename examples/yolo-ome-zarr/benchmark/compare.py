#!/usr/bin/env python3
"""Compare Blockflow mosaic detections with per-image original detections."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blockflow", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--block", type=int, default=640)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blockflow = defaultdict(list)
    with args.blockflow.open(newline="") as handle:
        for row in csv.DictReader(handle):
            global_x = float(row["x"])
            image = int(global_x // args.block)
            blockflow[image].append(
                (global_x - image * args.block, float(row["y"]), int(row["class"]), float(row["confidence"]))
            )
    original = defaultdict(list)
    with args.original.open(newline="") as handle:
        for row in csv.DictReader(handle):
            original[int(row["image"])].append(
                (float(row["x"]), float(row["y"]), int(row["class"]), float(row["confidence"]))
            )

    matched = []
    missing = 0
    extra = 0
    for image in sorted(set(blockflow) | set(original)):
        available = set(range(len(original[image])))
        for rust in blockflow[image]:
            candidates = [
                (math.hypot(rust[0] - original[image][index][0], rust[1] - original[image][index][1]), index)
                for index in available
                if rust[2] == original[image][index][2]
            ]
            if not candidates:
                extra += 1
                continue
            distance, index = min(candidates)
            if distance > args.radius:
                extra += 1
                continue
            available.remove(index)
            matched.append((distance, abs(rust[3] - original[image][index][3])))
        missing += len(available)
    result = {
        "blockflow": sum(map(len, blockflow.values())),
        "original": sum(map(len, original.values())),
        "matched": len(matched),
        "blockflow_only": extra,
        "original_only": missing,
        "max_center_distance_pixels": max((item[0] for item in matched), default=None),
        "mean_center_distance_pixels": sum(item[0] for item in matched) / len(matched) if matched else None,
        "max_confidence_difference": max((item[1] for item in matched), default=None),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
