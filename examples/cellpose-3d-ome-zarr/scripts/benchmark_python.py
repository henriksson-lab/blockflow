#!/usr/bin/env python3
"""Benchmark original Python Cellpose3D on the same TIFF crops as Rust."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--anisotropy", type=float, default=1.98)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    args = parser.parse_args()

    import torch
    from cellpose import io, models

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    load_started = time.perf_counter()
    model = models.CellposeModel(
        gpu=True, pretrained_model=args.model, use_bfloat16=False
    )
    load_seconds = time.perf_counter() - load_started
    inputs = [(path, io.imread(str(path))) for path in args.image]

    kwargs = dict(
        do_3D=True,
        z_axis=0,
        channel_axis=None,
        anisotropy=args.anisotropy,
        batch_size=args.batch_size,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
    )
    for _ in range(args.warmup):
        model.eval(inputs[0][1], **kwargs)
        torch.cuda.synchronize()

    measurements = []
    for run in range(args.runs):
        for path, image in inputs:
            torch.cuda.synchronize()
            started = time.perf_counter()
            masks, _flows, _styles = model.eval(image, **kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            mask_path = None
            if args.mask_dir is not None:
                args.mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = args.mask_dir / f"{path.stem}-run{run}-python-masks.npy"
                np.save(mask_path, np.asarray(masks, dtype=np.int32))
            masks = np.asarray(masks)
            measurements.append(
                dict(
                    image=str(path),
                    run=run,
                    seconds=seconds,
                    cells=int(masks.max(initial=0)),
                    foreground_voxels=int(np.count_nonzero(masks)),
                    mask=str(mask_path) if mask_path else None,
                )
            )

    report = dict(
        implementation="cellpose-python",
        scope="model.eval(do_3D=True)",
        model=args.model,
        device=torch.cuda.get_device_name(0),
        anisotropy=args.anisotropy,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        model_load_seconds=load_seconds,
        measurements=measurements,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
