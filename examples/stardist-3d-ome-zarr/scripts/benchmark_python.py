#!/usr/bin/env python3
"""Run original Python StarDist3D and emit the artifact consumed by Rust."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stardist-repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stardist_repo / "stardist"))

    import tensorflow as tf
    from stardist.models import StarDist3D

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise SystemExit("TensorFlow GPU is unavailable")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    model_root = args.stardist_repo / "stardist" / "models" / "examples"
    model = StarDist3D(None, name="3D_demo", basedir=str(model_root))
    image = tifffile.imread(args.image).astype(np.float32) / 255.0
    network_input = image[None, :, :, :, None]
    for _ in range(args.warmup):
        model.keras_model.predict(network_input, verbose=0)

    started = time.perf_counter()
    for _ in range(args.repeats):
        raw_prob, raw_dist = model.keras_model.predict(network_input, verbose=0)
    raw_seconds = (time.perf_counter() - started) / args.repeats

    started = time.perf_counter()
    sparse_prob, sparse_dist, sparse_points = model.predict_sparse(
        image,
        axes="ZYX",
        normalizer=None,
        n_tiles=None,
        show_tile_progress=False,
    )
    sparse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    labels, instances = model._instances_from_prediction(
        image.shape,
        sparse_prob,
        sparse_dist,
        points=sparse_points,
        prob_thresh=None,
        nms_thresh=None,
        return_labels=True,
    )
    postprocess_seconds = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        input_ndhwc=network_input,
        input_ncdhw=np.transpose(network_input, (0, 4, 1, 2, 3)),
        raw_prob_ndhwc=raw_prob.astype(np.float32),
        raw_prob_ncdhw=np.transpose(raw_prob, (0, 4, 1, 2, 3)).astype(np.float32),
        raw_dist_ndhwc=raw_dist.astype(np.float32),
        raw_dist_ncdhw=np.transpose(raw_dist, (0, 4, 1, 2, 3)).astype(np.float32),
        sparse_prob=sparse_prob.astype(np.float32),
        sparse_dist=sparse_dist.astype(np.float32),
        sparse_points=sparse_points.astype(np.float32),
        labels=labels.astype(np.uint32),
        points=instances["points"].astype(np.float32),
        prob=instances["prob"].astype(np.float32),
        dist=instances["dist"].astype(np.float32),
    )
    report = {
        "implementation": "python-stardist",
        "image": str(args.image),
        "shape_zyx": list(image.shape),
        "device": devices[0].name,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_inference_seconds": raw_seconds,
        "predict_sparse_seconds": sparse_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_sparse_instances_seconds": sparse_seconds + postprocess_seconds,
        "objects": int(labels.max(initial=0)),
        "foreground_voxels": int(np.count_nonzero(labels)),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "artifact": str(args.output),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
