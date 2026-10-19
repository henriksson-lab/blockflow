#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Run the reference's deep vessel filling on a binary `[z, y, x]` mask.

Two things are written, both `[z, y, x]`:

* `<out>/filled.npy` (`bool`): the reference's `fill_vessels` as its TubeMap
  runs it (`pipeline_orchestrators/tube_map.py`, `_deep_fill_channel`) at one
  channel role's shipped configuration (`config/defaults/v3.1/vasculature.yml`):
  `vessel` resamples by 1 in blocks of at most 500 voxels with an overlap of
  50, `artery` by 2 in blocks of 1000 with an overlap of 100; threshold 0.5,
  block axes `all`, `size_min` `fixed`. The network is used as the reference
  uses it, which is in training mode: it never calls `eval()`, so its dropout
  layers (p 0.15) are active and the result is not deterministic. `--eval`
  switches them off.
* `<out>/probability.npy` (`float32`, with `--probability`): the network's
  foreground probability over the whole mask as one block, in evaluation mode
  (no dropout), resampled as the role says, without the reference's merge with
  the mask. The network's output is shorter than its input by up to 7 voxels
  per axis (`8 * floor((n + 2) / 8) - 2` for an input of `n`); the missing
  planes are NaN.

A `<out>/run.json` records the parameters, wall time and device. The
reference's arrays are `[x, y, z]`; the mask is transposed to that order, so
the network sees the axes the reference's own pipeline hands it.

    fill_vessels.py --mask M.npy --role vessel --out DIR [--device cpu|cuda] [--eval] [--probability]

Run with the reference's own Python (`clearmap-gt2`, which needs torch).
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

CLEARMAP = Path("/data/henriksson/github/claude/clearmap-rs/ClearMap-py")

ROLES = {
    "vessel": {"resample": 1, "size_max": 500, "overlap": 50},
    "artery": {"resample": 2, "size_max": 1000, "overlap": 100},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval", action="store_true",
                        help="switch the network's dropout off (the reference leaves it on)")
    parser.add_argument("--probability", action="store_true",
                        help="also write the whole-volume probability (evaluation mode)")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads (default: torch's own choice)")
    parser.add_argument("--clearmap", type=Path, default=CLEARMAP)
    args = parser.parse_args()

    sys.path.insert(0, str(args.clearmap))
    import torch
    import ClearMap.IO.IO as io
    import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vf

    if args.threads:
        torch.set_num_threads(args.threads)
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch sees no GPU")
    role = ROLES[args.role]
    mask = np.load(args.mask)
    if mask.dtype != bool:
        mask = mask != 0
    source = np.ascontiguousarray(mask.transpose(2, 1, 0))  # to x, y, z
    args.out.mkdir(parents=True, exist_ok=True)

    network = vf.vessel_filling_network(cuda=cuda)
    if not cuda:
        network = network.float()
    if args.eval:
        network.eval()

    processing = dict(vf.default_fill_vessels_processing_parameter)
    processing.update(size_max=role["size_max"], size_min="fixed", axes="all",
                      overlap=role["overlap"])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sink = vf.fill_vessels(io.as_source(source), None, resample=role["resample"],
                           threshold=0.5, network=network, cuda=cuda,
                           processing_parameter=processing, verbose=False)
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    filled = np.ascontiguousarray(np.asarray(sink, dtype=bool).transpose(2, 1, 0))
    np.save(args.out / "filled.npy", filled)
    record = {
        "mask": str(args.mask),
        "role": args.role,
        **role,
        "threshold": 0.5,
        "device": args.device,
        "dtype": "float16" if cuda else "float32",
        "dropout": not args.eval,
        "torch_threads": torch.get_num_threads(),
        "shape_zyx": list(filled.shape),
        "input_voxels": int(mask.sum()),
        "filled_voxels": int(filled.sum()),
        "seconds": seconds,
    }
    if cuda:
        record["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())

    if args.probability:
        net = vf.vessel_filling_network(cuda=False).float().eval()
        started = time.perf_counter()
        with torch.no_grad():
            data = torch.from_numpy(source.astype(np.float32))[None, None]
            r = role["resample"]
            if r > 1:
                data = torch.nn.MaxPool3d(kernel_size=r)(data)
            out = net(data)
            if r > 1:
                out = torch.nn.Upsample(mode="trilinear", scale_factor=r,
                                        align_corners=False)(out)
        record["probability_seconds"] = time.perf_counter() - started
        out = out[0, 0].numpy()
        probability = np.full(source.shape, np.nan, dtype=np.float32)
        cut = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, out.shape))
        probability[cut] = out[cut]
        np.save(args.out / "probability.npy",
                np.ascontiguousarray(probability.transpose(2, 1, 0)))
        record["probability_shape_zyx"] = list(out.shape[::-1])

    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
