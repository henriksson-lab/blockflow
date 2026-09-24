#!/usr/bin/env python3
"""Timed GPU inference through the translated model's original PyTorch source."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.original))
    from utils import util
    from utils.dataset import resize

    checkpoint = torch.load(args.weights, map_location="cuda", weights_only=False)
    model = checkpoint["model"].float().fuse().cuda().eval()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    model = model.to(dtype=dtype)
    paths = [Path(line) for line in args.images.read_text().splitlines() if line]

    warmup = torch.zeros((args.batch_size, 3, 640, 640), device="cuda", dtype=dtype)
    for _ in range(3):
        model(warmup)
    torch.cuda.synchronize()

    detections = 0
    detection_rows = []
    profile = {name: 0.0 for name in ("decode", "upload", "forward", "nms", "download")}
    started = time.perf_counter()
    for offset in range(0, len(paths), args.batch_size):
        stage_started = time.perf_counter()
        images = []
        for path in paths[offset : offset + args.batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (640, 640):
                raise RuntimeError(f"invalid prepared image {path}")
            image, _, _ = resize(image, 640, False)
            images.append(np.ascontiguousarray(image.transpose(2, 0, 1)[::-1]))
        profile["decode"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        samples = torch.from_numpy(np.stack(images)).to(device="cuda", dtype=dtype) / 255.0
        if args.profile:
            torch.cuda.synchronize()
        profile["upload"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        outputs = model(samples)
        if args.profile:
            torch.cuda.synchronize()
        profile["forward"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        outputs = util.non_max_suppression(outputs, confidence_threshold=0.25, iou_threshold=0.45)
        if args.profile:
            torch.cuda.synchronize()
        profile["nms"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        for batch_index, output in enumerate(outputs):
            image_index = offset + batch_index
            detections += len(output)
            for detection in output.detach().float().cpu().tolist():
                detection_rows.append(
                    (
                        image_index,
                        (detection[0] + detection[2]) * 0.5,
                        (detection[1] + detection[3]) * 0.5,
                        detection[4],
                        int(detection[5]),
                    )
                )
        profile["download"] += time.perf_counter() - stage_started
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "implementation": "jahongir7174/YOLOv11-pt",
        "device": torch.cuda.get_device_name(),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "images": len(paths),
        "detections": detections,
        "seconds": seconds,
        "images_per_second": len(paths) / seconds,
    }
    if args.profile:
        result["profile_seconds"] = profile
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.detections:
        with args.detections.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image", "x", "y", "confidence", "class"))
            writer.writerows(detection_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
