#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Count exact same-frame mask overlaps for Blockflow's spatial matcher.

The reference may be another prediction. This program does not label either
segmentation as truth and never reads fluorescence or changes source arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def frame(array, t):
    if array.ndim == 3:
        return np.asarray(array[t])
    if array.ndim == 4 and array.shape[1] == 1:
        return np.asarray(array[t, 0])
    raise ValueError("expected tyx or singleton-z tzyx labels")


def counts(a, b):
    if a.shape != b.shape or a.dtype.kind not in "iu" or b.dtype.kind not in "iu" or a.min() < 0 or b.min() < 0:
        raise ValueError("labels must be non-negative integer arrays on the same lattice")
    ids_a, index_a, sizes_a = np.unique(a, return_inverse=True, return_counts=True)
    ids_b, index_b, sizes_b = np.unique(b, return_inverse=True, return_counts=True)
    flat = index_a.ravel() * len(ids_b) + index_b.ravel()
    intersections = np.bincount(flat, minlength=len(ids_a)*len(ids_b)).reshape(len(ids_a), len(ids_b))
    shared = [{"reference": int(ids_a[i]), "predicted": int(ids_b[j]), "pixels": int(intersections[i, j])}
              for i, j in zip(*np.nonzero(intersections)) if ids_a[i] != 0 and ids_b[j] != 0]
    return {"reference_sizes": [{"id": int(i), "pixels": int(n)} for i, n in zip(ids_a, sizes_a) if i != 0],
            "predicted_sizes": [{"id": int(i), "pixels": int(n)} for i, n in zip(ids_b, sizes_b) if i != 0],
            "shared_pixels": shared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--reference-role", choices=["previous_prediction", "independent_reviewed_reference"], required=True)
    parser.add_argument("--minimum-iou", type=float, default=.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = zarr.open_array(str(args.reference), mode="r")
    b = zarr.open_array(str(args.predicted), mode="r")
    if a.shape[0] != b.shape[0]:
        raise ValueError("frame counts differ; explicitly align the temporal scope first")
    rows, hashes = [], []
    for t in range(a.shape[0]):
        x, y = frame(a, t), frame(b, t)
        rows.append({"t_index": t, **counts(x, y)})
        hashes.append({"t_index": t, "reference_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                       "predicted_sha256": hashlib.sha256(y.tobytes()).hexdigest()})
        print(f"t={t}: {len(rows[-1]['reference_sizes'])} reference, {len(rows[-1]['predicted_sizes'])} predicted objects", flush=True)
    result = {"schema": "blockflow.frame-overlaps/v1", "minimum_iou": args.minimum_iou,
              "frames": rows, "provenance": {"reference": str(args.reference.resolve()),
                  "predicted": str(args.predicted.resolve()), "reference_role": args.reference_role,
                  "pixel_hashes": hashes, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    with args.output.open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
