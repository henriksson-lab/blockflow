#!/usr/bin/env python3
"""Prepare identical 640 px images as PNGs and a chunk-aligned OME-Zarr strip."""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width, 1.0)
    resized = (round(width * ratio), round(height * ratio))
    if (width, height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized[0]) / 2
    pad_y = (size - resized[1]) / 2
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    candidates = sorted(
        path
        for path in args.source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.images]
    if len(candidates) != args.images:
        raise SystemExit(f"wanted {args.images} images, found {len(candidates)}")
    args.output.mkdir(parents=True, exist_ok=False)
    image_dir = args.output / "images"
    image_dir.mkdir()
    prepared = []
    for index, source in enumerate(candidates):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read {source}")
        image = letterbox(image, args.size)
        destination = image_dir / f"{index:04}.png"
        if not cv2.imwrite(str(destination), image):
            raise SystemExit(f"could not write {destination}")
        prepared.append((destination, image))

    sequence = prepared * args.repeats
    (args.output / "images.txt").write_text(
        "".join(f"{path}\n" for path, _ in sequence), encoding="utf-8"
    )
    root = args.output / "input.zarr"
    level = root / "0"
    level.mkdir(parents=True)
    (root / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "multiscales": [{"version": "0.5", "datasets": [{"path": "0"}]}]
                },
            },
            indent=2,
        )
        + "\n"
    )
    (level / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3, args.size, args.size * len(sequence)],
                "data_type": "uint8",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, args.size, args.size]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": 1}},
                ],
                "attributes": {},
            },
            indent=2,
        )
        + "\n"
    )
    for index, (_, bgr) in enumerate(sequence):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            chunk = level / "c" / str(channel) / "0" / str(index)
            chunk.parent.mkdir(parents=True, exist_ok=True)
            chunk.write_bytes(gzip.compress(rgb[:, :, channel].tobytes(), compresslevel=1))
    print(f"prepared {len(sequence)} image blocks at {args.output}")


if __name__ == "__main__":
    main()
