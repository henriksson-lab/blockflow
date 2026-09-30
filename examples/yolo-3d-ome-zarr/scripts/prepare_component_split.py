#!/usr/bin/env python3
"""Build a leakage-safe split with connected positives and empty background."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import zarr


def axis_tiles(length: int, patch: int, step: int):
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    rows = []
    for index, start in enumerate(starts):
        low = 0 if index == 0 else (starts[index - 1] + patch + start) // 2
        high = (
            length
            if index + 1 == len(starts)
            else (start + patch + starts[index + 1]) // 2
        )
        rows.append((start, low, high))
    return rows


def overlaps(left, right, patch):
    return all(
        left[axis] < right[axis] + patch[axis]
        and right[axis] < left[axis] + patch[axis]
        for axis in range(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--teacher", default="cellpose3d-cpsam")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-z", type=int, default=32)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--halo-z", type=int, default=4)
    parser.add_argument("--halo", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2079)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=2.0,
        help="empty input windows added per positive ownership tile",
    )
    args = parser.parse_args()

    image = zarr.open_array(str(args.zarr / "0" / "0"), mode="r")
    volume = tuple(int(value) for value in image.shape[-3:])
    patch = (args.patch_z, args.patch, args.patch)
    halo = (args.halo_z, args.halo, args.halo)
    columns = args.zarr / "tables" / args.teacher / "columns"
    points = np.stack(
        [
            np.asarray(zarr.open_array(str(columns / f"centroid_{axis}"), mode="r")[:])
            for axis in "zyx"
        ],
        axis=1,
    )
    print(f"loaded {len(points)} teacher centroids for volume {volume}", flush=True)

    axes = [
        axis_tiles(volume[i], patch[i], patch[i] - 2 * halo[i])
        for i in range(3)
    ]
    positive = []
    negative = []
    owners = np.zeros(len(points), dtype=np.uint8)
    for z, z0, z1 in axes[0]:
        for y, y0, y1 in axes[1]:
            for x, x0, x1 in axes[2]:
                mask = np.all(
                    (points >= (z0, y0, x0)) & (points < (z1, y1, x1)), axis=1
                )
                indices = np.flatnonzero(mask)
                owners[indices] += 1
                if len(indices):
                    positive.append({"start": (z, y, x), "objects": len(indices)})
                start = np.asarray((z, y, x))
                if not np.any(np.all((points >= start) & (points < start + patch), axis=1)):
                    negative.append((z, y, x))
    if not np.all(owners == 1):
        raise SystemExit("tile cores do not assign every teacher object exactly once")
    print(
        f"classified {len(positive)} positive and {len(negative)} empty input windows",
        flush=True,
    )

    parent = list(range(len(positive)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(len(positive)):
        for right in range(left):
            if overlaps(positive[left]["start"], positive[right]["start"], patch):
                union(left, right)

    grouped = {}
    for index, tile in enumerate(positive):
        grouped.setdefault(find(index), []).append(tile)
    components = sorted(
        grouped.values(),
        key=lambda rows: sum(row["objects"] for row in rows),
        reverse=True,
    )
    if len(components) < 3:
        raise SystemExit("at least three non-overlapping positive-tile groups are required")
    print(f"formed {len(components)} positive overlap components", flush=True)

    # The largest connected component contains the crowded region and stays
    # intact in training. Randomize the independent exterior components, then
    # balance them between validation and frozen test by teacher-object count.
    train_components = [components[0]]
    exterior = components[1:]
    random.Random(args.seed).shuffle(exterior)
    validation_components, test_components = [], []
    validation_objects = test_objects = 0
    for component in exterior:
        count = sum(row["objects"] for row in component)
        if validation_objects <= test_objects:
            validation_components.append(component)
            validation_objects += count
        else:
            test_components.append(component)
            test_objects += count

    def starts(components):
        return [list(row["start"]) for component in components for row in component]

    def object_count(components):
        return sum(row["objects"] for component in components for row in component)

    partitions = {
        "train": starts(train_components),
        "validation": starts(validation_components),
        "test": starts(test_components),
    }
    positive_counts = {name: len(rows) for name, rows in partitions.items()}
    targets = {
        name: round(count * args.negative_ratio) for name, count in positive_counts.items()
    }
    rng = random.Random(args.seed ^ 0x6E65676174697665)
    rng.shuffle(negative)
    unused = list(negative)
    added = {name: 0 for name in partitions}
    names = list(partitions)
    while any(added[name] < targets[name] for name in names):
        progressed = False
        for name in names:
            if added[name] >= targets[name]:
                continue
            for index, candidate in enumerate(unused):
                if all(
                    not overlaps(candidate, other, patch)
                    for other_name, rows in partitions.items()
                    if other_name != name
                    for other in rows
                ):
                    partitions[name].append(list(candidate))
                    unused.pop(index)
                    added[name] += 1
                    progressed = True
                    break
        if not progressed:
            missing = {
                name: targets[name] - added[name]
                for name in names
                if added[name] < targets[name]
            }
            raise SystemExit(f"could not add leakage-safe empty windows: {missing}")

    for left_index, left_name in enumerate(names):
        for right_name in names[:left_index]:
            if any(
                overlaps(left, right, patch)
                for left in partitions[left_name]
                for right in partitions[right_name]
            ):
                raise SystemExit(f"{left_name} and {right_name} windows overlap")

    result = {
        "volume": list(volume),
        "patch": list(patch),
        "ownership_halo": list(halo),
        "augmentation": "fluorescence",
        "seed": args.seed,
        "strategy": "largest overlapping positive-tile component in training; randomized independent exterior components balanced between validation and test; deterministic empty input windows added without cross-partition overlap",
        "train": partitions["train"],
        "validation": partitions["validation"],
        "test": partitions["test"],
        "summary": {
            "teacher_objects": len(points),
            "positive_tiles": len(positive),
            "components": len(components),
            "train_objects": object_count(train_components),
            "validation_objects": object_count(validation_components),
            "test_objects": object_count(test_components),
            "negative_ratio": args.negative_ratio,
            "train_empty_windows": added["train"],
            "validation_empty_windows": added["validation"],
            "test_empty_windows": added["test"],
            "train_windows": len(partitions["train"]),
            "validation_windows": len(partitions["validation"]),
            "test_windows": len(partitions["test"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
