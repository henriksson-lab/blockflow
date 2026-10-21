#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Speed and memory of the light-sheet pipeline, blockflow against the
reference (ClearMap 3.1), on identical inputs with matched parameters.

    bench.py list
    bench.py prepare  [WORKLOAD[:SIZE] ...] [--smoke]
    bench.py run      [WORKLOAD[:SIZE] ...] [--smoke] [--threads N] [--repeats R] [options]
    bench.py summarize [--results DIR] [--max-load L] [--exclude-loaded]

A workload is one pipeline stage (or the whole chain) at a named size; see
`list` and the README. `run` prepares what is missing, then for each repeat
runs both sides back to back, alternating which goes first (ABAB), each into
a fresh output directory, under `measure.py`, and appends one row per run to
`results/runs.jsonl` and `results/runs.csv`. With `--check` (default) each
repeat's outputs are compared afterwards (outside the measurement) and the
result is stored in `results/checks.jsonl`. `summarize` prints medians and
spread per workload and the ratios, flagging runs above the load threshold.

Standard library only; it runs the per-side commands with the interpreters
and binaries given (see `--help`).
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
EXAMPLES = PIPELINE.parent
ROOT = EXAMPLES.parent
DATA = Path("/husky/otherdataset/teresa/clearmap_em")
REFERENCE = Path("/husky/henriksson/lightsheet-reference")
REG = REFERENCE / "registration"
CLEARMAP_ROOT = Path(os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py"))
ALIGNMENT = CLEARMAP_ROOT / "ClearMap" / "Resources" / "Alignment"
SIDES = ("clearmap", "ours")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")
CSV_COLUMNS = ["run_id", "workload", "size", "side", "repeat", "position", "threads", "variant", "exit",
               "wall_s", "user_s", "sys_s", "cpu_s", "tree_rss_peak_gb", "tree_pss_peak_gb",
               "tree_anon_peak_gb", "max_rss_single_gb", "processes_peak", "load1_before", "load1_after",
               "load1_max", "other_cores_mean", "over_max_load", "cache", "started",
               "blockflow_commit", "blockflow_dirty", "clearmap_commit", "out"]


# ---------------------------------------------------------------------------
# helpers

def ceil_div(value, factor):
    return max(1, math.ceil(value / factor))


def zarr_scales(image):
    """Per-level `[z, y, x]` scale of an OME-Zarr image (the last three axes)."""
    meta = json.loads((Path(image) / "zarr.json").read_text())
    attributes = meta.get("attributes", {})
    multiscale = (attributes.get("ome") or attributes)["multiscales"][0]
    out = []
    for dataset in multiscale["datasets"]:
        scale = next(t["scale"] for t in dataset["coordinateTransformations"] if t["type"] == "scale")
        out.append(scale[-3:])
    return out


def level_factors(image, level):
    """Level-0 voxels per voxel of `level`, `[z, y, x]`, rounded."""
    scales = zarr_scales(image)
    return [round(a / b) for a, b in zip(scales[level], scales[0])]


def run_quiet(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def files_under(paths):
    for path in paths:
        path = Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for root, _, names in os.walk(path):
                for name in names:
                    yield Path(root) / name


def chunk_paths(array_dir, z_range=None, channel=None):
    """The chunk files (and metadata) of one Zarr v3 array that a read of
    `channel` (for a leading channel axis) over planes `z_range` touches;
    everything when both are None. For warming only what a run reads."""
    array_dir = Path(array_dir)
    meta = json.loads((array_dir / "zarr.json").read_text())
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    root = array_dir / "c"
    if len(shape) == 4:
        root = root / str(channel or 0)
    zc = chunk[-3]
    z0, z1 = z_range or (0, shape[-3])
    out = [array_dir / "zarr.json"]
    out += [root / str(k) for k in range(z0 // zc, -(-min(z1, shape[-3]) // zc))]
    return out


def warm_read(paths):
    """Read every file once into the page cache; returns bytes and seconds."""
    t0 = time.time()
    total = 0
    for path in files_under(paths):
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
        except OSError:
            pass
    return total, time.time() - t0


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            out[key] = int(value.split()[0]) * 1024
    return out


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(v) for v in f.read().split()[:3]]


def filesystem(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    result = run_quiet(["findmnt", "-T", str(path), "-n", "-o", "SOURCE,FSTYPE,TARGET"])
    return result.stdout.strip()


def rmtree(path):
    if Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# provenance, gathered once per invocation

def host_info(args):
    lscpu = run_quiet(["lscpu"]).stdout
    model = next((line.split(":", 1)[1].strip() for line in lscpu.splitlines()
                  if line.startswith("Model name")), platform.processor())
    mem = meminfo()
    return {"hostname": socket.gethostname(), "kernel": platform.release(), "cpu_model": model,
            "nproc": os.cpu_count(), "sockets": next((line.split(":", 1)[1].strip() for line in lscpu.splitlines()
                                                      if line.startswith("Socket(s)")), None),
            "ram_gb": round(mem["MemTotal"] / 2**30, 1),
            "mem_available_gb_at_start": round(mem["MemAvailable"] / 2**30, 1),
            "fs_inputs": filesystem(DATA), "fs_bench_root": filesystem(args.root)}


def git_state(path, scope=None):
    head = run_quiet(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip() or None
    status = run_quiet(["git", "-C", str(path), "status", "--porcelain", "--"] + (scope or [])).stdout
    dirty = [line for line in status.splitlines() if line.strip()]
    diff = run_quiet(["git", "-C", str(path), "diff", "HEAD", "--"] + (scope or [])).stdout
    return {"commit": head, "dirty_paths": len(dirty),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None}


def python_info(python):
    code = ("import json, sys, platform\nout = {'python': sys.version.split()[0], 'executable': sys.executable}\n"
            "for m in ('numpy', 'scipy', 'torch', 'graph_tool', 'zarr', 'pandas'):\n"
            "    try:\n        out[m] = __import__(m).__version__\n    except Exception:\n        pass\n"
            "print(json.dumps(out))")
    result = run_quiet([python, "-I", "-c", code])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": result.stderr[-500:]}


def provenance(args):
    scope = ["src", "Cargo.toml", "Cargo.lock", "examples/tile-stitching", "examples/atlas-registration",
             "examples/cell-density", "examples/vessel-graph", "examples/atlas-regions",
             "examples/lightsheet-pipeline"]
    binaries = {}
    for name in ("tile-stitching", "atlas-registration", "cell-density", "vessel-graph"):
        path = Path(args.bin) / name
        if path.exists():
            binaries[name] = {"sha256": sha256(path), "mtime": datetime.datetime.fromtimestamp(
                path.stat().st_mtime).isoformat(timespec="seconds")}
    elastix = CLEARMAP_ROOT / "ClearMap" / "External" / "elastix" / "build"
    env = dict(os.environ, LD_LIBRARY_PATH=f"{elastix / 'bin'}:{elastix / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    version = run_quiet([str(elastix / "bin" / "elastix"), "--version"], env=env).stdout.strip()
    return {"host": host_info(args), "blockflow": {**git_state(ROOT, scope), "binaries": binaries},
            "clearmap": {**git_state(CLEARMAP_ROOT), "root": str(CLEARMAP_ROOT), "elastix": version,
                         "python": python_info(args.clearmap_py)},
            "python_ours": python_info(args.py)}


# ---------------------------------------------------------------------------
# workloads

class Workload:
    """One stage at named sizes; subclasses fill in the two sides."""
    name = ""
    description = ""
    sizes = {}            # size -> parameters
    expected = {}         # size -> rough duration per side, for the README and `list`
    differences = []      # what a reader of the numbers must know
    check_kind = None

    def prepare(self, ctx, size):
        pass

    def inputs(self, ctx, size, side):
        return []

    def setup(self, ctx, size, side, out):
        pass

    def command(self, ctx, size, side, out):
        raise NotImplementedError

    def environment(self, ctx, size, side, out):
        return {}

    def check(self, ctx, size, run_dir):
        return None

    def cleanup(self, ctx, size, side, out):
        pass

    def side_timing(self, out, side):
        for candidate in (out / "timing.json", out / "work" / "timing.json", out / "stitch" / "timing.json"):
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text())
                except json.JSONDecodeError:
                    return None
        return None


def stitch_check(ctx, run_dir):
    result = run_quiet([ctx.py, "-I", str(EXAMPLES / "tile-stitching" / "reference-clearmap" / "compare.py")],
                       env=dict(os.environ, STITCH_OUT=str(run_dir), REF="clearmap", OURS="ours"))
    path = run_dir / "comparison-ours-vs-clearmap.json"
    if result.returncode != 0 or not path.exists():
        return {"error": result.stderr[-2000:]}
    report = json.loads(path.read_text())
    return {k: report[k] for k in report if k not in ("timing", "stats")}


class StitchWorkload(Workload):
    """Rigid + wobbly stitching and fusion of a tile set at one level."""
    tile_dir = None
    overlap_yx = None     # level-0 voxels
    check_kind = "stitch"

    def params(self, size):
        return self.sizes[size]

    def factors(self, size):
        level = self.params(size)["level"]
        first = sorted(p for p in Path(self.tile_dir).iterdir() if p.name.startswith("tile_"))[0]
        return level_factors(first, level)       # z, y, x

    def input_dir(self, ctx, size):
        p = self.params(size)
        key = f"{Path(self.tile_dir).parent.name}-l{p['level']}-z{p.get('z') or 'all'}" \
              f"-r{p.get('rows') or 'all'}-c{p.get('columns') or 'all'}".replace(":", "_")
        return ctx.root / "inputs" / "tiles" / key

    def prepare(self, ctx, size):
        p = self.params(size)
        command = [ctx.py, "-I", str(HERE / "prepare.py"), "tiles", "--tile-dir", str(self.tile_dir),
                   "--level", str(p["level"]), "--out", str(self.input_dir(ctx, size))]
        for key in ("z", "rows", "columns"):
            if p.get(key):
                command += [f"--{key}", p[key]]
        subprocess.run(command, check=True)

    def search(self, size):
        """Search ranges at the level: the reference's level-0 defaults
        (stitching.yml v3.1: rigid x, y 30, z 20; per slice x 30, y 15)
        divided by the level's factors, rounded up."""
        fz, fy, fx = self.factors(size)
        return {"rigid": (ceil_div(30, fx), ceil_div(30, fy), ceil_div(20, fz)),
                "wobbly": (ceil_div(30, fx), ceil_div(15, fy))}

    def selected_tiles(self, size):
        p = self.params(size)
        tiles = sorted(t for t in Path(self.tile_dir).iterdir() if t.name.startswith("tile_"))

        def inside(index, span):
            if not span:
                return True
            a, b = (int(v) for v in span.split(":"))
            return a <= index < b
        return [t for t in tiles if inside(int(t.name[5:7]), p.get("rows")) and inside(int(t.name[8:10]), p.get("columns"))]

    def inputs(self, ctx, size, side):
        if side == "clearmap":
            return [self.input_dir(ctx, size) / "npy"]
        p = self.params(size)
        z = tuple(int(v) for v in p["z"].split(":")) if p.get("z") else None
        out = []
        for tile in self.selected_tiles(size):
            out += [tile / "zarr.json"] + chunk_paths(tile / str(p["level"]), z, channel=0)
        return out

    def command(self, ctx, size, side, out):
        p = self.params(size)
        fz, fy, fx = self.factors(size)
        s = self.search(size)
        oy, ox = self.overlap_yx
        if side == "clearmap":
            return [ctx.clearmap_py, "-I", str(HERE / "clearmap" / "stitch.py"),
                    "--npy-dir", str(self.input_dir(ctx, size) / "npy"), "--out", str(out),
                    "--overlap", f"{round(ox / fx)},{round(oy / fy)}",
                    "--rigid-shift", ",".join(map(str, s["rigid"])),
                    "--wobbly-shift", ",".join(map(str, s["wobbly"])),
                    "--processes", str(ctx.threads), "--variant", ctx.variant]
        command = [str(ctx.bin / "tile-stitching"), "--tile-dir", str(self.tile_dir),
                   "--overlap", f"{oy},{ox}", "--level", str(p["level"]),
                   "--max-shift", f"{s['wobbly'][1]},{s['wobbly'][0]}",
                   "--rigid-max-shift", f"{s['rigid'][1]},{s['rigid'][0]}",
                   "--max-shift-z", str(s["rigid"][2]),
                   "--out", str(out / "fused.zarr"), "--out-levels", "1"]
        if p.get("z"):
            command += ["--z-range", p["z"]]
        if p.get("rows"):
            command += ["--rows", p["rows"]]
        if p.get("columns"):
            command += ["--columns", p["columns"]]
        return command

    def check(self, ctx, size, run_dir):
        return stitch_check(ctx, run_dir)

    def cleanup(self, ctx, size, side, out):
        (out / "stitched.npy").unlink(missing_ok=True)
        rmtree(out / "fused.zarr")


class StitchSa3x8(StitchWorkload):
    name = "stitch-sa3x8"
    description = "osf-sa3x8 (vasculature), 3 x 3 tiles of 3369 x 1304 x 404, channel 0"
    tile_dir = DATA / "osf-sa3x8" / "tiles"
    overlap_yx = (212, 66)     # what the reference's parse_overlaps reads from Brain-39L (x 66, y 212)
    sizes = {"smoke": {"level": 0, "z": "1200:1260"},
             "harness": {"level": 0, "z": "1200:1440"},
             "l1": {"level": 1}}
    expected = {"smoke": "~0.5-1 min per side", "harness": "~1-2.5 min per side (earlier: 35-145 s / 40-45 s)",
                "l1": "~5-15 min per side (not measured yet; prepare writes 4 GB of .npy)"}
    differences = [
        "The reference reads .npy tiles in its (x, y, z) order (prepare.py, untimed: its own pipeline's "
        "file conversion is not part of the measurement); blockflow reads the OME-Zarr tiles (zstd chunks) "
        "inside its measurement.",
        "Reference variant `corrected` by default (its wobbly-placement and tracing-tie-break defects "
        "patched, the harness's main comparison); `--variant shipped` runs it as shipped, whose wobbly "
        "placement is a no-op (the result is the rigid layout).",
        "Search ranges are the reference's stitching.yml v3.1 level-0 values divided by the level's factors "
        "(rounded up) on both sides.",
        "blockflow fuses at --out-levels 1 (no pyramid), as the reference writes one array.",
        "The reference smooths each tile's positions over the whole stack, blockflow over the tile's own "
        "slices (harness README).",
    ]


class StitchCztq2(StitchWorkload):
    name = "stitch-cztq2"
    description = "osf-cztq2 (cFos), 8 x 5 tiles of 1030 x 1007 x 1004, whole mosaic at one level"
    tile_dir = DATA / "osf-cztq2" / "cFos"
    overlap_yx = (120, 120)    # measured from the whole-sample stitch; the tiles carry no translations
    sizes = {"smoke": {"level": 3, "rows": "0:2", "columns": "0:2"},
             "l3": {"level": 3},
             "l2": {"level": 2},
             "l1": {"level": 1}}
    expected = {"smoke": "~0.5-1 min per side", "l3": "~2-5 min per side",
                "l2": "~5-20 min per side (blockflow earlier: about 4 min align at level 1 + 1 min fuse)",
                "l1": "~15-60 min per side; prepare writes 21 GB of .npy"}
    differences = StitchSa3x8.differences[:2] + [
        "The overlap (120 level-0 voxels in y and x) is given to both; the conversion carries no tile "
        "translations. Levels 1 and 2 keep z (1030 planes), level 3 halves it; search ranges are scaled "
        "per axis accordingly.",
        "blockflow aligns and fuses in one command here (without --positions-only), as the reference does; "
        "the pipeline itself aligns at level 1 and fuses separately (see the e2e workload).",
    ]


class Register(Workload):
    name = "register"
    description = "osf-cztq2 autofluorescence (25 µm) -> ABA 2017 25 µm reference, elastix parameter files"
    sizes = {"smoke": {"files": ["align_affine.txt"]},
             "full": {"files": ["align_affine.txt", "align_bspline.txt"]}}
    expected = {"smoke": "elastix ~1.5 min, blockflow ~10 s", "full": "elastix ~9 min, blockflow ~1.5 min"}
    differences = [
        "Inputs are the registration harness's: the reference's own 25 µm resample of the autofluorescence "
        "and its prepared atlas (orientation (1, 2, 3), z 0:256), as TIFF for elastix and as OME-Zarr "
        "(the same voxels through img2omezarr) for blockflow. Resampling is not timed on either side.",
        "Both are stochastic (elastix's random sampler; blockflow's with seed 121212); agreement is at the "
        "level of seed-to-seed noise (~16 µm median), checked on the harness's point grid.",
        "elastix writes its warped result image after each stage (the reference's configuration); "
        "blockflow warps the reference after each stage to measure similarity but writes only the "
        "transform files and registration.json.",
        "blockflow's own parallelism is rayon (RAYON_NUM_THREADS); elastix gets -threads N.",
    ]
    check_kind = "registration"

    def inputs(self, ctx, size, side):
        if side == "clearmap":
            return [REG / "clearmap" / "resampled_autofluorescence.tif", REG / "atlas" / "reference.tif"]
        return [REG / "zarr" / "autofluorescence_25um.ome.zarr", REG / "zarr" / "reference_25um.ome.zarr"]

    def command(self, ctx, size, side, out):
        files = self.sizes[size]["files"]
        if side == "clearmap":
            command = [ctx.clearmap_py, "-I", str(HERE / "clearmap" / "register.py"),
                       "--fixed", str(REG / "clearmap" / "resampled_autofluorescence.tif"),
                       "--moving", str(REG / "atlas" / "reference.tif"), "--out", str(out / "reg"),
                       "--processes", str(ctx.threads)]
            for f in files:
                command += ["--parameters", f]
            return command
        command = [str(ctx.bin / "atlas-registration"), "register",
                   "--fixed", str(REG / "zarr" / "autofluorescence_25um.ome.zarr"),
                   "--moving", str(REG / "zarr" / "reference_25um.ome.zarr"),
                   "--resolution", "25", "--seed", "121212", "--out", str(out / "reg")]
        for f in files:
            command += ["--parameters", str(ALIGNMENT / f)]
        return command

    def check(self, ctx, size, run_dir):
        return run_check(ctx, "registration", run_dir, ctx.clearmap_py)

    def side_timing(self, out, side):
        path = out / "reg" / ("timing.json" if side == "clearmap" else "registration.json")
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data if side == "clearmap" else {"timings": data.get("timings")}


class Cells(Workload):
    name = "cells"
    description = "osf-cztq2 cFos tile 03x02, level 0, cell detection and size filter"
    tile = DATA / "osf-cztq2" / "cFos" / "tile_03x02.zarr"
    sizes = {"smoke": {"start": "400,248,246", "shape": "256,512,512"},
             "tile": {"start": "0,0,0", "shape": "0,0,0"}}
    expected = {"smoke": "~10-40 s per side", "tile": "~1.5-15 min per side (earlier at load 5-11: 81 s / 99 s)"}
    differences = [
        "The reference reads the crop as .npy (prepare.py, untimed); blockflow reads the OME-Zarr tile "
        "with --crop-start/--crop-shape inside its measurement.",
        "Parameters: the reference's cell_map.yml v3.1 (background disk 10 x 10, maxima shape 5, no h-max, "
        "shape threshold 500, size band [50, 900)) and cell-density --threshold 500 with its defaults, "
        "which match.",
        "The reference processes z blocks of 10..16 planes with overlap 5 (its shipped block sizes), "
        "which cut cells along z: most of the filtered-count gap (cell-density harness README); "
        "blockflow's tiled watershed is block-grid invariant.",
        "The reference's environment lacks pyarrow, so its per-block Feather tables are pickles "
        "(clearmap_cells.py).",
        "The reference's shipped process count is 10; here both sides get --threads.",
    ]
    check_kind = "cells"

    def input_dir(self, ctx, size):
        return ctx.root / "inputs" / f"cells-{size}"

    def prepare(self, ctx, size):
        p = self.sizes[size]
        d = self.input_dir(ctx, size)
        subprocess.run([ctx.py, "-I", str(HERE / "prepare.py"), "box", "--image", str(self.tile),
                        "--start", p["start"], "--shape", p["shape"], "--out", str(d / "input.npy")], check=True)
        record = json.loads((d / "input.json").read_text())
        (d / "setup.json").write_text(json.dumps(
            {"tile": str(self.tile), "crop_start_zyx": record["start_zyx"], "crop_shape_zyx": record["shape_zyx"],
             "voxels": record["voxels"]}, indent=1))

    def inputs(self, ctx, size, side):
        if side == "clearmap":
            return [self.input_dir(ctx, size) / "input.npy"]
        setup = json.loads((self.input_dir(ctx, size) / "setup.json").read_text())
        z0, nz = setup["crop_start_zyx"][0], setup["crop_shape_zyx"][0]
        return [self.tile / "zarr.json"] + chunk_paths(self.tile / "0", (z0, z0 + nz))

    def command(self, ctx, size, side, out):
        setup = json.loads((self.input_dir(ctx, size) / "setup.json").read_text())
        if side == "clearmap":
            return [ctx.clearmap_py, "-u", "-I", str(EXAMPLES / "cell-density" / "reference-clearmap" / "clearmap_cells.py")]
        return [str(ctx.bin / "cell-density"), "--input", str(self.tile), "--output", str(out / "out.zarr"),
                "--crop-start", ",".join(map(str, setup["crop_start_zyx"])),
                "--crop-shape", ",".join(map(str, setup["crop_shape_zyx"])),
                "--threshold", "500", "--workers", str(ctx.threads)]

    def environment(self, ctx, size, side, out):
        if side == "clearmap":
            return {"CELLS_OUT": str(self.input_dir(ctx, size)), "CELLS_REF_DIR": str(out),
                    "CLEARMAP_PROCESSES": str(ctx.threads), "CLEARMAP_BLOCK": "10,16,5"}
        return {}

    def check(self, ctx, size, run_dir):
        shutil.copy(self.input_dir(ctx, size) / "setup.json", run_dir / "setup.json")
        result = run_quiet([ctx.py, "-I", str(EXAMPLES / "cell-density" / "reference-clearmap" / "compare.py")],
                           env=dict(os.environ, CELLS_OUT=str(run_dir), CELLS_REF_DIR="clearmap",
                                    CELLS_OURS_DIR="ours"))
        found = sorted(run_dir.glob("comparison*.json"))
        if result.returncode != 0 or not found:
            return {"error": result.stderr[-2000:]}
        report = json.loads(found[-1].read_text())
        keep = ("ours", "reference", "matched", "precision", "recall", "same_voxel")
        return {k: {f: report[k].get(f) for f in keep} for k in ("raw", "filtered") if k in report}

    def cleanup(self, ctx, size, side, out):
        rmtree(out / "out.zarr")
        for name in ("cells_raw.feather", "cells_raw.csv"):
            (out / name).unlink(missing_ok=True)


class Vessels(Workload):
    name = "vessels"
    description = "osf-sa3x8, both channels (vessel, artery): binarize, smooth, fill, union, skeleton, graph"
    deep_fill = False
    sizes = {"smoke": {"image": REFERENCE / "vessel-planned" / "data" / "crop-src.ome.zarr"},
             "region": {"image": REFERENCE / "vessel-planned" / "data" / "region-src.ome.zarr"}}
    expected = {"smoke": "~1-2.5 min per side (crop 128 x 512 x 404 of tile 05x05)",
                "region": "~3-6 min per side (fused box 512 x 512 x 256)"}
    differences = [
        "Reference: its TubeMap steps called one by one with vasculature.yml v3.1 (clearmap/vessels.py), "
        "every n_processes set to --threads; blockflow: vessel-graph segment then graph --all-components "
        "(ours/vessels.sh). No atlas step on either side.",
        "The reference reads .npy channels (prepare.py, untimed) and writes every intermediate as .npy, as "
        "its workspace does; blockflow reads the OME-Zarr copy (copied before the measurement) and writes "
        "its labels into it.",
        "Two adaptations make the reference run in parallel at all (clearmap/compat.py): its block workers "
        "reopen memory-mapped sinks in mode w+ (truncating them; SIGBUS with more than two processes), "
        "patched to r+; and the rank filters' const-char masks.",
        "The reference's binarization depends on its process count through block optimisation (Dice 0.995 "
        "between 1 and 8 processes on the crop's artery channel).",
        "blockflow's graph keeps every component (--all-components) as the reference's clean + reduce do; "
        "blockflow additionally classifies arteries and computes edge lengths in the same command.",
        "Known algorithmic differences (vessel-graph README, Limitations): rounding of the clip, ceiling "
        "window percentiles, the tubeness filter; binarization Dice 0.98 per channel against the reference.",
    ]
    check_kind = "vessels"

    def input_dir(self, ctx, size):
        return ctx.root / "inputs" / f"vessels-{size}"

    def prepare(self, ctx, size):
        d = self.input_dir(ctx, size)
        source = self.sizes[size]["image"]
        if not (d / "src.ome.zarr").exists():
            d.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, d / "src.ome.zarr.partial")
            os.rename(d / "src.ome.zarr.partial", d / "src.ome.zarr")
        subprocess.run([ctx.py, "-I", str(HERE / "prepare.py"), "channels", "--image", str(d / "src.ome.zarr"),
                        "--out", str(d)], check=True)

    def inputs(self, ctx, size, side):
        d = self.input_dir(ctx, size)
        return [d / "source_c0.npy", d / "source_c1.npy"] if side == "clearmap" else [d / "src.ome.zarr"]

    def setup(self, ctx, size, side, out):
        if side == "ours":
            shutil.copytree(self.input_dir(ctx, size) / "src.ome.zarr", out / "image.ome.zarr")

    def spacing_xyz(self, ctx, size):
        z, y, x = zarr_scales(self.input_dir(ctx, size) / "src.ome.zarr")[0]
        return f"{x},{y},{z}"

    def command(self, ctx, size, side, out):
        if side == "clearmap":
            command = [ctx.clearmap_py, "-u", "-I", str(HERE / "clearmap" / "vessels.py"),
                       "--inputs", str(self.input_dir(ctx, size)), "--out", str(out / "work"),
                       "--processes", str(ctx.threads), "--spacing", self.spacing_xyz(ctx, size)]
            if self.deep_fill:
                command.append("--deep-fill")
                if ctx.deep_fill_eval:
                    command.append("--eval")
            return command
        command = ["bash", str(HERE / "ours" / "vessels.sh"), str(ctx.bin), str(out / "image.ome.zarr"),
                   str(out), str(ctx.threads)]
        if self.deep_fill:
            command.append(str(ctx.weights))
        return command

    def check(self, ctx, size, run_dir):
        return run_check(ctx, "vessels", run_dir, ctx.py)

    def cleanup(self, ctx, size, side, out):
        if side == "clearmap":
            for path in (out / "work").glob("*.npy"):
                path.unlink()
            (out / "work" / "graph_reduced.gt").unlink(missing_ok=True)
        else:
            rmtree(out / "image.ome.zarr")
            (out / "graph.json").unlink(missing_ok=True)


class VesselsDeepFill(Vessels):
    name = "vessels-deepfill"
    description = Vessels.description + ", with the deep vessel filling (CNN) per channel, on the CPU"
    deep_fill = True
    expected = {"smoke": "~1.5-2.5 min per side", "region": "~4-10 min per side"}
    differences = Vessels.differences + [
        "Deep filling on the CPU on both sides (blockflow has no GPU path yet; the reference's orchestrator "
        "would use CUDA). torch gets --threads via torch.set_num_threads.",
        "The reference runs its network with dropout active (it never calls eval(); nondeterministic, about "
        "50% more filled voxels); --deep-fill-eval switches it off. blockflow is the network in evaluation "
        "mode.",
        "Reference blocks of 500 (1000 at resample 2) with overlap 50 (100), less than the network's reach; "
        "blockflow plans 256 x 512 x 512 blocks with a 64 (128) voxel halo.",
    ]


class EndToEnd(Workload):
    name = "e2e"
    description = ("osf-cztq2: stitch all 40 tiles at one level, register signal -> autofluorescence -> atlas, "
                   "cells on a level-0 box of tile 01x02, into the atlas and labelled")
    tile_dir = DATA / "osf-cztq2" / "cFos"
    sizes = {"smoke": {"level": 3, "auto_files": ["align_affine.txt"], "box": "320:448,384:640,384:640"},
             "full": {"level": 2, "auto_files": ["align_affine.txt", "align_bspline.txt"],
                      "box": "256:768,192:832,192:832"}}
    expected = {"smoke": "reference ~5 min, blockflow ~1 min", "full": "reference ~20-40 min, blockflow ~5-10 min"}
    differences = [
        "blockflow runs lightsheet-pipeline/run.sh (stages align, fuse, register, cells) with "
        "STITCH_LEVEL = FUSE_LEVEL = the size's level; the reference stitches and fuses at that level "
        "(clearmap/e2e.py). The fuse stage writes its default 4-level pyramid; the reference one array.",
        "The reference resamples the fused signal and the full-resolution autofluorescence (.npy, prepared "
        "untimed) to 25 µm with Resampling.resample; blockflow reads a pyramid level of the autofluorescence "
        "OME-Zarr and resamples inside the registration.",
        "Cells: blockflow fuses the level-0 box from the level-L positions and detects on it; the reference "
        "cannot fuse a level-0 box from coarser positions, so it detects on the same box of tile 01x02's own "
        "voxels (prepared untimed) and places it at the tile's median wobbly position. Its real pipeline "
        "would fuse the whole sample at level 0 first, which is not measured (so the reference side is "
        "faster here than in practice).",
        "Atlas labelling: the reference's transform_points twice (transformix) and annotation lookup with "
        "astype(int); blockflow's cell-density --registration (nearest-centre rounding by default).",
        "smoke uses only the affine for autofluorescence -> atlas; full adds the B-spline.",
    ] + StitchSa3x8.differences[1:3]
    check_kind = "e2e"

    def level(self, size):
        return self.sizes[size]["level"]

    def stitch(self):
        return StitchCztq2()

    def stitch_size(self, size):
        return {3: "l3", 2: "l2", 1: "l1"}[self.level(size)]

    def box(self, size):
        spans = [tuple(int(v) for v in s.split(":")) for s in self.sizes[size]["box"].split(",")]
        return [a for a, _ in spans], [b - a for a, b in spans]

    def prepare(self, ctx, size):
        self.stitch().prepare(ctx, self.stitch_size(size))
        d = ctx.root / "inputs" / f"e2e-{size}"
        start, shape = self.box(size)
        subprocess.run([ctx.py, "-I", str(HERE / "prepare.py"), "box",
                        "--image", str(self.tile_dir / "tile_01x02.zarr"), "--start", ",".join(map(str, start)),
                        "--shape", ",".join(map(str, shape)), "--out", str(d / "box.npy")], check=True)
        subprocess.run([ctx.py, "-I", str(HERE / "prepare.py"), "box",
                        "--image", str(DATA / "osf-cztq2" / "autofluorescence.zarr"), "--start", "0,0,0",
                        "--shape", "0,0,0", "--out", str(ctx.root / "inputs" / "cztq2-autofluorescence.npy")],
                       check=True)

    def inputs(self, ctx, size, side):
        stitch = self.stitch()
        tiles = stitch.inputs(ctx, self.stitch_size(size), side)
        if side == "clearmap":
            return tiles + [ctx.root / "inputs" / f"e2e-{size}" / "box.npy",
                            ctx.root / "inputs" / "cztq2-autofluorescence.npy",
                            REG / "atlas" / "reference.tif", REG / "atlas" / "annotation.tif"]
        start, shape = self.box(size)
        level0 = chunk_paths(self.tile_dir / "tile_01x02.zarr" / "0", (start[0], start[0] + shape[0]))
        return tiles + level0 + [DATA / "osf-cztq2" / "autofluorescence.zarr",
                                 REFERENCE / "atlas" / "ABA_25um_2017_1_2_3_z0-256.ome.zarr"]

    def setup(self, ctx, size, side, out):
        if side != "ours":
            return
        stitch = self.stitch()
        s = stitch.search(self.stitch_size(size))
        p = self.sizes[size]
        params = out / "params.env"
        params.write_text("\n".join([
            f"source {PIPELINE / 'params' / 'osf-cztq2.env'}",
            f"OUT_ROOT={out}",
            f"BIN={ctx.bin}",
            f"PY={ctx.py}",
            f"STITCH_LEVEL={p['level']}",
            f"FUSE_LEVEL={p['level']}",
            f"STITCH_MAX_SHIFT={s['wobbly'][1]},{s['wobbly'][0]}",
            f"STITCH_RIGID_MAX_SHIFT={s['rigid'][1]},{s['rigid'][0]}",
            f"STITCH_MAX_SHIFT_Z={s['rigid'][2]}",
            f"AUTO_TO_ATLAS=\"{' '.join(p['auto_files'])}\"",
            f"CELL_TILE_BOX={p['box']}",
            f"CELL_ARGS=\"--workers {ctx.threads}\"",
            ""]))

    def command(self, ctx, size, side, out):
        p = self.sizes[size]
        stitch = self.stitch()
        ss = self.stitch_size(size)
        if side == "ours":
            return ["bash", str(PIPELINE / "run.sh"), str(DATA / "osf-cztq2"), str(out / "params.env"),
                    "align", "fuse", "register", "cells"]
        fz, fy, fx = stitch.factors(ss)
        s = stitch.search(ss)
        start, _ = self.box(size)
        command = [ctx.clearmap_py, "-u", "-I", str(HERE / "clearmap" / "e2e.py"),
                   "--npy-dir", str(stitch.input_dir(ctx, ss) / "npy"),
                   "--overlap", f"{round(120 / fx)},{round(120 / fy)}",
                   "--rigid-shift", ",".join(map(str, s["rigid"])),
                   "--wobbly-shift", ",".join(map(str, s["wobbly"])),
                   "--level-factor", f"{fx},{fy},{fz}",
                   "--auto-npy", str(ctx.root / "inputs" / "cztq2-autofluorescence.npy"),
                   "--atlas-dir", str(REG / "atlas"),
                   "--box-npy", str(ctx.root / "inputs" / f"e2e-{size}" / "box.npy"),
                   "--box-tile", "01x02", "--box-start", ",".join(map(str, start)),
                   "--processes", str(ctx.threads), "--variant", ctx.variant, "--out", str(out)]
        for f in p["auto_files"]:
            command += ["--auto-parameters", f]
        return command

    def check(self, ctx, size, run_dir):
        return run_check(ctx, "e2e", run_dir, ctx.clearmap_py)

    def side_timing(self, out, side):
        if side == "clearmap":
            path = out / "timing.json"
            return json.loads(path.read_text()) if path.exists() else None
        manifest = next(out.glob("*/manifest.json"), None)
        if not manifest:
            return None
        stages = json.loads(manifest.read_text()).get("stages", {})
        return {"steps_s": {k: v.get("wall_s") for k, v in stages.items()},
                "peak_rss_gb": {k: v.get("peak_rss_gb") for k, v in stages.items()}}

    def cleanup(self, ctx, size, side, out):
        if side == "clearmap":
            (out / "stitch" / "stitched.npy").unlink(missing_ok=True)
            for path in out.glob("*.feather"):
                path.unlink()
        else:
            for sample in out.glob("osf-*"):
                rmtree(sample / "stitch" / "fused.ome.zarr")
                rmtree(sample / "cells" / "region.ome.zarr")


def run_check(ctx, kind, run_dir, python):
    result = run_quiet([python, "-I", str(HERE / "checks.py"), kind, str(run_dir)],
                       env=dict(os.environ, BENCH_BIN=str(ctx.bin), CLEARMAP_ROOT=str(CLEARMAP_ROOT)))
    if result.returncode != 0:
        return {"error": result.stderr[-2000:]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": result.stdout[-2000:]}


WORKLOADS = {w.name: w for w in (StitchSa3x8(), StitchCztq2(), Register(), Cells(), Vessels(),
                                 VesselsDeepFill(), EndToEnd())}


# ---------------------------------------------------------------------------
# running

class Context:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.threads = args.threads
        self.py = args.py
        self.clearmap_py = args.clearmap_py
        self.bin = Path(args.bin)
        self.variant = getattr(args, "variant", "corrected")
        self.weights = Path(getattr(args, "weights", "") or "")
        self.deep_fill_eval = getattr(args, "deep_fill_eval", False)


def selection(args):
    chosen = []
    names = args.workloads or list(WORKLOADS)
    for item in names:
        name, _, size = item.partition(":")
        if name not in WORKLOADS:
            sys.exit(f"bench.py: unknown workload {name!r}; `bench.py list` shows them")
        workload = WORKLOADS[name]
        if not size:
            size = "smoke" if args.smoke else next(iter(s for s in workload.sizes if s != "smoke"))
        if size not in workload.sizes:
            sys.exit(f"bench.py: {name} has no size {size!r} (sizes: {', '.join(workload.sizes)})")
        chosen.append((workload, size))
    return chosen


def thread_environment(n):
    return {v: str(n) for v in THREAD_VARIABLES}


def wait_for_load(args):
    """Wait (up to --wait-max-s) for the 1-minute load to drop to --max-load."""
    waited = 0.0
    if not args.max_load:
        return waited
    while loadavg()[0] > args.max_load and waited < args.wait_max_s:
        time.sleep(10)
        waited += 10
    if loadavg()[0] > args.max_load:
        print(f"   warning: load {loadavg()[0]:.1f} > --max-load {args.max_load}; running anyway "
              f"(the row is flagged)", flush=True)
    return waited


def append_row(results, row):
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    path = results / "runs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_COLUMNS})


def run_side(ctx, args, prov, workload, size, side, run_dir, repeat, position, cache_note, measured=True):
    out = run_dir / side
    if out.exists():
        sys.exit(f"bench.py: {out} exists; outputs go to fresh directories")
    out.mkdir(parents=True)
    workload.setup(ctx, size, side, out)
    warm = {"mode": args.warm}
    inputs = workload.inputs(ctx, size, side)
    input_bytes = sum(p.stat().st_size for p in files_under(inputs))
    if args.warm == "read":
        total, seconds = warm_read(inputs)
        warm.update(bytes=total, seconds=round(seconds, 1))
    warm["input_bytes"] = input_bytes
    available = meminfo()["MemAvailable"]
    warm["fits_in_available_memory"] = input_bytes < 0.5 * available
    if args.warm == "none":
        warm["state"] = "unknown (no warm-up)"
    elif not warm["fits_in_available_memory"]:
        warm["state"] = "partial (inputs exceed half the available memory)"
    else:
        warm["state"] = "warm" if args.warm == "read" else cache_note
    waited = wait_for_load(args)
    command = workload.command(ctx, size, side, out)
    if args.cpus:
        command = ["taskset", "-c", args.cpus] + command
    environment = dict(os.environ, **thread_environment(ctx.threads), **workload.environment(ctx, size, side, out))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    measure = [ctx.py, "-I", str(HERE / "measure.py"), "--out", str(out / "measure.json"),
               "--interval", str(args.interval), "--pss-interval", str(args.pss_interval)]
    if args.kill_gb:
        measure += ["--kill-gb", str(args.kill_gb)]
    label = f"{workload.name}:{size} {side} r{repeat}"
    print(f"== {label}: {' '.join(command)}", flush=True)
    with open(out / "log.txt", "w") as log:
        status = subprocess.run(measure + ["--"] + command, stdout=log, stderr=subprocess.STDOUT,
                                env=environment, cwd=out).returncode
    m = json.loads((out / "measure.json").read_text()) if (out / "measure.json").exists() else {}
    print(f"   {label}: exit {m.get('exit')} wall {m.get('wall_s')} s, cpu {m.get('cpu_s')} s, "
          f"tree RSS {m.get('tree_rss_peak_gb')} GB, PSS {m.get('tree_pss_peak_gb')} GB, "
          f"max single {m.get('max_rss_single_gb')} GB, load {m.get('load_before', [None])[0]}", flush=True)
    if status != 0:
        print(f"   {label} FAILED, see {out / 'log.txt'}", flush=True)
    if not measured:
        return None
    row = {
        "run_id": f"{run_dir.name}-{side}", "workload": workload.name, "size": size, "side": side,
        "repeat": repeat, "position": position, "threads": ctx.threads,
        "variant": ctx.variant if side == "clearmap" else None,
        "exit": m.get("exit"), **{k: m.get(k) for k in ("wall_s", "user_s", "sys_s", "cpu_s", "tree_rss_peak_gb",
                                                       "tree_pss_peak_gb", "tree_anon_peak_gb",
                                                       "max_rss_single_gb", "processes_peak", "threads_peak",
                                                       "io", "load1_max", "load1_mean", "other_cpu_s",
                                                       "other_cores_mean", "started")},
        "load1_before": (m.get("load_before") or [None])[0], "load1_after": (m.get("load_after") or [None])[0],
        "load_before": m.get("load_before"), "load_after": m.get("load_after"),
        "max_load": args.max_load, "waited_for_load_s": waited,
        "over_max_load": bool(args.max_load and (m.get("load_before") or [0])[0] > args.max_load),
        "cache": warm["state"], "warm_up": warm, "pinning": args.cpus,
        "thread_environment": thread_environment(ctx.threads), "extra_environment": workload.environment(ctx, size, side, out),
        "command": command, "out": str(out), "side_timing": workload.side_timing(out, side),
        "blockflow_commit": prov["blockflow"]["commit"], "blockflow_dirty": prov["blockflow"]["dirty_paths"],
        "clearmap_commit": prov["clearmap"]["commit"], "provenance": prov,
    }
    return row


def cmd_run(args):
    ctx = Context(args)
    chosen = selection(args)
    if args.build:
        for package, extra in (("blockflow-tile-stitching", []), ("blockflow-atlas-registration", []),
                               ("blockflow-cell-density", []), ("blockflow-vessel-graph", ["--features", "deep-fill"])):
            subprocess.run(["cargo", "build", "--release", "-p", package] + extra, cwd=ROOT, check=True)
    results = Path(args.results or (ctx.root / "results"))
    prov = provenance(args)
    (results).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    sides = [s for s in SIDES if s in args.sides.split(",")]
    for workload, size in chosen:
        print(f"## {workload.name}:{size} — {workload.description}", flush=True)
        workload.prepare(ctx, size)
        base = ctx.root / "work" / workload.name / size
        cache_note = "page cache as left by earlier runs"
        if args.warm == "run":
            for side in sides:
                run_side(ctx, args, prov, workload, size, side, base / f"{stamp}-warmup", -1, -1, "", measured=False)
                workload.cleanup(ctx, size, side, base / f"{stamp}-warmup" / side)
            cache_note = "warm (discarded warm-up run)"
        for repeat in range(args.repeats):
            order = sides if repeat % 2 == 0 else sides[::-1]
            run_dir = base / f"{stamp}-r{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for position, side in enumerate(order):
                row = run_side(ctx, args, prov, workload, size, side, run_dir, repeat, position, cache_note)
                rows.append(row)
                append_row(results, row)
            if args.check and len(sides) == 2 and all(r["exit"] == 0 for r in rows):
                t0 = time.time()
                check = workload.check(ctx, size, run_dir)
                record = {"run_ids": [r["run_id"] for r in rows], "workload": workload.name, "size": size,
                          "repeat": repeat, "seconds": round(time.time() - t0, 1), "result": check}
                (run_dir / "check.json").write_text(json.dumps(record, indent=1))
                with open(results / "checks.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"   check: {json.dumps(check)[:600]}", flush=True)
            if not args.keep and repeat < args.repeats - 1:
                for side in sides:
                    workload.cleanup(ctx, size, side, run_dir / side)
    cmd_summarize(args)


def cmd_prepare(args):
    ctx = Context(args)
    for workload, size in selection(args):
        print(f"## prepare {workload.name}:{size}", flush=True)
        workload.prepare(ctx, size)


def cmd_list(args):
    for workload in WORKLOADS.values():
        print(f"{workload.name}: {workload.description}")
        for size, params in workload.sizes.items():
            print(f"  {size:8s} {json.dumps(params, default=str)}")
            print(f"           expected: {workload.expected.get(size, '?')}")
        print("  differences:")
        for d in workload.differences:
            print(f"   - {d}")
        print()


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": round(statistics.median(values), 3), "min": round(min(values), 3),
            "max": round(max(values), 3), "n": len(values)}


def cmd_summarize(args):
    results = Path(args.results or (Path(args.root) / "results"))
    path = results / "runs.jsonl"
    if not path.exists():
        sys.exit(f"bench.py: no runs in {results}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    threshold = args.max_load
    groups = {}
    for row in rows:
        if args.workloads and row["workload"] not in [w.partition(":")[0] for w in args.workloads]:
            continue
        loaded = bool(threshold and (row.get("load1_before") or 0) > threshold)
        if loaded and args.exclude_loaded:
            continue
        key = (row["workload"], row["size"], row["threads"])
        groups.setdefault(key, {}).setdefault(row["side"], []).append({**row, "_loaded": loaded})
    lines = [f"Runs in {path}; load threshold {threshold or 'none'}"
             + (" (runs above it excluded)" if args.exclude_loaded else " (* = runs above it)"), "",
             "| workload | size | threads | side | n ok | wall s median [min-max] | CPU s median | "
             "tree RSS GB | tree PSS GB | max single RSS GB | load1 before | loaded |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for (workload, size, threads), sides in sorted(groups.items()):
        entry = {"workload": workload, "size": size, "threads": threads, "sides": {}}
        for side in SIDES:
            runs = sides.get(side, [])
            ok = [r for r in runs if r.get("exit") == 0]
            stats = {k: spread([r.get(k) for r in ok]) for k in
                     ("wall_s", "cpu_s", "tree_rss_peak_gb", "tree_pss_peak_gb", "max_rss_single_gb", "load1_before")}
            stats["failed"] = len(runs) - len(ok)
            stats["loaded"] = sum(r["_loaded"] for r in runs)
            entry["sides"][side] = stats
            if not runs:
                continue
            w = stats["wall_s"]

            def med(k):
                return stats[k]["median"] if stats[k] else None
            lines.append(f"| {workload} | {size} | {threads} | {side} | {len(ok)}"
                         f"{' (+%d failed)' % stats['failed'] if stats['failed'] else ''} | "
                         f"{w['median'] if w else '-'} [{w['min'] if w else '-'}-{w['max'] if w else '-'}] | "
                         f"{med('cpu_s')} | {med('tree_rss_peak_gb')} | {med('tree_pss_peak_gb')} | "
                         f"{med('max_rss_single_gb')} | {med('load1_before')} | "
                         f"{'*' * stats['loaded'] if stats['loaded'] else ''} |")
        a, b = entry["sides"].get("clearmap"), entry["sides"].get("ours")
        ratios = {}
        if a and b:
            for k, label in (("wall_s", "wall"), ("cpu_s", "cpu"), ("tree_pss_peak_gb", "tree_pss"),
                             ("tree_rss_peak_gb", "tree_rss"), ("max_rss_single_gb", "max_single_rss")):
                if a.get(k) and b.get(k) and b[k]["median"]:
                    ratios[label] = round(a[k]["median"] / b[k]["median"], 2)
        entry["clearmap_over_ours"] = ratios
        summary.append(entry)
    lines += ["", "Ratios, reference / blockflow, of the medians (above 1: blockflow faster or smaller):", "",
              "| workload | size | threads | wall | CPU | tree PSS | tree RSS | max single RSS |",
              "|---|---|---|---|---|---|---|---|"]
    for entry in summary:
        r = entry["clearmap_over_ours"]
        lines.append(f"| {entry['workload']} | {entry['size']} | {entry['threads']} | {r.get('wall', '-')} | "
                     f"{r.get('cpu', '-')} | {r.get('tree_pss', '-')} | {r.get('tree_rss', '-')} | "
                     f"{r.get('max_single_rss', '-')} |")
    checks = results / "checks.jsonl"
    if checks.exists():
        lines += ["", "Latest output check per workload:", ""]
        latest = {}
        for line in checks.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                latest[(record["workload"], record["size"])] = record
        for (workload, size), record in sorted(latest.items()):
            lines.append(f"- {workload}:{size}: {json.dumps(record['result'])[:700]}")
    text = "\n".join(lines) + "\n"
    (results / "summary.md").write_text(text)
    (results / "summary.json").write_text(json.dumps(summary, indent=1))
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("command", choices=["list", "prepare", "run", "summarize"])
    parser.add_argument("workloads", nargs="*", help="WORKLOAD or WORKLOAD:SIZE (default: all)")
    parser.add_argument("--smoke", action="store_true", help="the smoke size of every workload named (or all)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="threads/processes for both sides (default: all cores)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sides", default="clearmap,ours")
    parser.add_argument("--variant", default="corrected", choices=["shipped", "patched", "corrected"],
                        help="the reference's stitching defects corrected (see clearmap/stitch.py)")
    parser.add_argument("--deep-fill-eval", action="store_true",
                        help="run the reference's vessel-filling network in eval mode (dropout off)")
    parser.add_argument("--max-load", type=float, default=0.0,
                        help="flag runs whose 1-minute load before the start exceeds this (0: no threshold)")
    parser.add_argument("--wait-max-s", type=float, default=0.0,
                        help="wait up to this long for the load to drop below --max-load before each run")
    parser.add_argument("--exclude-loaded", action="store_true", help="summarize: leave flagged runs out")
    parser.add_argument("--warm", choices=["read", "run", "none"], default="read",
                        help="page cache: read each side's inputs before its run (default), one discarded "
                             "warm-up run of each side, or nothing")
    parser.add_argument("--cpus", default="", help="pin both sides to these CPUs (taskset -c LIST)")
    parser.add_argument("--interval", type=float, default=0.05, help="memory sampling interval, s")
    parser.add_argument("--pss-interval", type=float, default=1.0, help="PSS sampling interval, s (0: off)")
    parser.add_argument("--kill-gb", type=float, default=0.0, help="kill a run whose tree RSS exceeds this")
    parser.add_argument("--no-check", dest="check", action="store_false", help="skip the output checks")
    parser.add_argument("--keep", action="store_true", help="keep every repeat's outputs (default: the last)")
    parser.add_argument("--build", action="store_true", help="cargo build --release the four packages first")
    parser.add_argument("--root", default=str(REFERENCE / "benchmark"),
                        help="inputs/, work/ and results/ go here (default %(default)s)")
    parser.add_argument("--results", default=None, help="results directory (default ROOT/results)")
    parser.add_argument("--bin", default=str(ROOT / "target" / "release"))
    parser.add_argument("--py", default=shutil.which("python3") or "python3",
                        help="a Python with numpy, zarr 3, pandas, scipy")
    parser.add_argument("--clearmap-py", default="/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python")
    parser.add_argument("--weights", default=str(REFERENCE / "vessel-filling" / "cnn_binary_vessel_filling.safetensors"))
    args = parser.parse_args()
    {"list": cmd_list, "prepare": cmd_prepare, "run": cmd_run, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
