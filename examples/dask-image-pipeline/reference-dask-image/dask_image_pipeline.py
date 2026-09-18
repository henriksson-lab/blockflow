#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import time
from pathlib import Path

import dask.array as da
import dask_image.ndfilters
import dask_image.ndmorph
import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=".tmp/dask-image-pipeline/dask-image")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--mode", choices=["segment", "transform"], default="segment")
    parser.add_argument("--chunk", default="256x256")
    return parser.parse_args()


def parse_chunk(raw):
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--chunk must be HxW, for example 256x256")
    y, x = (int(part) for part in parts)
    if y < 1 or x < 1:
        raise ValueError("--chunk dimensions must be positive")
    return (y, x)


def transform_if_requested(image, mode):
    if mode == "segment":
        return image
    return da.pad(image[:-5, :-7], ((5, 0), (7, 0)), mode="constant")[: image.shape[0], : image.shape[1]]


def main():
    args = parse_args()
    chunk = parse_chunk(args.chunk)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    image = iio.imread(args.input)
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.float64, copy=False)
    image_array = da.from_array(image, chunks=chunk)
    load_seconds = time.perf_counter() - started

    pipeline_started = time.perf_counter()
    prepared = transform_if_requested(image_array, args.mode)
    if args.sigma == 0.0:
        smoothed = prepared
    else:
        smoothed = dask_image.ndfilters.gaussian_filter(
            prepared,
            sigma=args.sigma,
            truncate=3.0,
            mode="reflect",
        )
    threshold = float(filters.threshold_otsu(smoothed.compute()))
    mask = smoothed > threshold
    footprint = np.ones((3, 3), dtype=bool)
    mask = dask_image.ndmorph.binary_opening(mask, structure=footprint)
    mask = dask_image.ndmorph.binary_closing(mask, structure=footprint)
    materialized_mask = mask.compute()
    labels, _ = ndi.label(
        materialized_mask,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    )
    counts = np.bincount(labels.ravel())
    keep = counts >= args.min_size
    keep[0] = False
    labels = labels * keep[labels]
    labels, _ = ndi.label(
        labels > 0,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    )
    props = measure.regionprops(labels)
    pipeline_seconds = time.perf_counter() - pipeline_started

    with (out_dir / "objects.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count", "centroid_y", "centroid_x"])
        for index, prop in enumerate(props, start=1):
            writer.writerow([index, int(prop.area), prop.centroid[0], prop.centroid[1]])

    total_area = int(sum(prop.area for prop in props))
    summary = {
        "input": args.input,
        "mode": args.mode,
        "chunk": args.chunk,
        "objects": len(props),
        "total_foreground_area": total_area,
        "threshold": threshold,
        "sigma": args.sigma,
        "min_size": args.min_size,
        "load_seconds": load_seconds,
        "pipeline_seconds": pipeline_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"objects={len(props)} threshold={threshold:.6f} output={out_dir}")


if __name__ == "__main__":
    main()
