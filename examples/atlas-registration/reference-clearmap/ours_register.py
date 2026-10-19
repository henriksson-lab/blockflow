#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Native run: `atlas-registration` on the same inputs as the reference run.

- `ours/auto_to_reference/`: fixed = autofluorescence, moving = atlas
  reference, the reference's `align_affine.txt` then `align_bspline.txt`,
  unmodified; the annotation warped onto the fixed lattice (nearest), the
  reference warped after each stage, and the comparison's point grid carried
  through;
- `ours/auto_to_reference_affine/`: the affine file alone (the same affine,
  same seed), for the affine-stage comparison;
- `ours/chain/` (with `--signal`): the two-step chain signal → autofluorescence
  → atlas, and the point grid of the signal image through it.

Each writes `stats.json` with wall time and peak memory. Any Python 3.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

OUT = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration"))
BINARY = os.environ.get("ATLAS_REGISTRATION",
                        "/home/mahogny/github/claude/blockflow/target/release/atlas-registration")
ALIGNMENT = Path(os.environ.get(
    "CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")) / "ClearMap" / \
    "Resources" / "Alignment"


def timed(arguments, stats: Path, threads):
    """Run under `/usr/bin/time`, and record wall time and peak resident memory."""
    environment = dict(os.environ)
    if threads:
        environment["RAYON_NUM_THREADS"] = str(threads)
    report = stats.with_name("time.txt")
    stats.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    completed = subprocess.run(["/usr/bin/time", "-f", "%e %M", "-o", str(report), BINARY]
                               + [str(a) for a in arguments], check=True, capture_output=True,
                               text=True, env=environment)
    wall = time.time() - t0
    elapsed, peak_kb = report.read_text().split()[-2:]
    record = {"wall_s": wall, "time_elapsed_s": float(elapsed), "peak_rss_mb": int(peak_kb) / 1024.0,
              "threads": threads or os.cpu_count(), "load_average": os.getloadavg(),
              "stdout": completed.stdout.splitlines()}
    stats.write_text(json.dumps(record, indent=1))
    print(stats, f"{wall:.1f} s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--seed", default="121212")
    parser.add_argument("--threads", type=int, default=0, help="0: all cores")
    parser.add_argument("--skip-atlas", action="store_true",
                        help="do not (re)run the autofluorescence -> atlas registrations")
    args = parser.parse_args()
    zarr = OUT / "zarr"
    ours = OUT / "ours"
    points = OUT / "points" / "auto_points.csv"
    common = ["--fixed", zarr / "autofluorescence_25um.ome.zarr",
              "--moving", zarr / "reference_25um.ome.zarr", "--resolution", "25",
              "--seed", args.seed, "--warp-source", zarr / "annotation_25um.ome.zarr", "--nearest"]
    runs = [("auto_to_reference", ["align_affine.txt", "align_bspline.txt"]),
            ("auto_to_reference_affine", ["align_affine.txt"])]
    for name, files in [] if args.skip_atlas else runs:
        out = ours / name
        parameters = []
        for file in files:
            parameters += ["--parameters", ALIGNMENT / file]
        timed(["register"] + common + parameters +
              ["--out", out, "--warped", out / "annotation_in_auto.ome.zarr", "--warped-stages"],
              out / "stats.json", args.threads)
        if points.exists():
            subprocess.run([BINARY, "transform-points", "--registration", out, "--input", points,
                            "--output", out / "points.csv"], check=True, capture_output=True)
    if args.signal:
        out = ours / "chain"
        timed(["chain", "--image", zarr / "signal_25um.ome.zarr",
               "--image", zarr / "autofluorescence_25um.ome.zarr",
               "--image", zarr / "reference_25um.ome.zarr", "--resolution", "25",
               "--seed", args.seed,
               "--parameters", f"0={ALIGNMENT / 'align_affine.txt'}",
               "--parameters", f"1={ALIGNMENT / 'align_affine.txt'}",
               "--parameters", f"1={ALIGNMENT / 'align_bspline.txt'}",
               "--out", out], out / "stats.json", args.threads)
        signal_points = OUT / "points" / "signal_points.csv"
        if signal_points.exists():
            for registration, csv in [(out, "points.csv"), (out / "step0", "points_step0.csv")]:
                subprocess.run([BINARY, "transform-points", "--registration", registration,
                                "--input", signal_points, "--output", out / csv], check=True,
                               capture_output=True)


if __name__ == "__main__":
    main()
