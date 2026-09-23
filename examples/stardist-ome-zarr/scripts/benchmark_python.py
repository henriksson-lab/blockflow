#!/usr/bin/env python3
"""Time the official Python StarDist implementation on a prepared OME-Zarr plane."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import zarr
from stardist.models import StarDist2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=70.0)
    parser.add_argument("--n-tiles", help="Comma-separated Y,X tile counts")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    read_started = time.perf_counter()
    root = zarr.open_group(args.zarr, mode="r")
    image = np.asarray(root["0"][args.channel])
    image = np.clip((image.astype(np.float32) - args.low) / (args.high - args.low), 0, 1)
    read_seconds = time.perf_counter() - read_started

    load_started = time.perf_counter()
    model = StarDist2D(None, name=args.model.name, basedir=str(args.model.parent))
    model_load_seconds = time.perf_counter() - load_started

    n_tiles = None
    if args.n_tiles:
        n_tiles = tuple(int(value) for value in args.n_tiles.split(","))

    predict_started = time.perf_counter()
    labels, details = model.predict_instances(
        image,
        axes="YX",
        sparse=True,
        n_tiles=n_tiles,
        show_tile_progress=False,
    )
    predict_seconds = time.perf_counter() - predict_started

    result = {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "low": args.low,
        "high": args.high,
        "n_tiles": n_tiles,
        "cells": int(len(details["points"])),
        "max_label": int(labels.max(initial=0)),
        "read_and_normalize_seconds": read_seconds,
        "model_load_seconds": model_load_seconds,
        "predict_instances_seconds": predict_seconds,
        "script_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
