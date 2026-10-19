#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Compare `atlas-registration` with the reference's elastix run.

Both registered the same inputs (the autofluorescence resampled to 25 µm,
fixed; the prepared atlas reference, moving) with the same parameter files.

`--write-points` writes a grid of fixed-image points inside the sample (every
8th voxel where the fixed intensity is above its median) for both sides:
`points/auto_points_elastix.txt` (elastix `point` file, `x y z` in the
reference's axis order) and `points/auto_points.csv` (`z,y,x`).

Without it, reads under `$REG_OUT` (default
`/husky/henriksson/lightsheet-reference/registration`):
- `clearmap/auto_to_reference/result.{0,1}.mhd`: the reference warped by
  elastix after the affine and after the B-spline;
- `clearmap/annotation_in_auto.{0,1}/result.mhd`: the annotation warped by
  transformix (nearest neighbour) after each;
- `clearmap/points.{0,1}/outputpoints.txt`: the grid through each;
- `ours/<run>/stage{0,1}_warped.ome.zarr`, `ours/<run>/annotation_in_auto.ome.zarr`,
  `ours/<run>_affine/annotation_in_auto.ome.zarr` and `ours/<run>/points.csv`
  (`ours/<run>_affine/points.csv` for the affine alone);
- `clearmap/timing.json` and `ours/<run>/stats.json`,

and writes `comparison.json` and prints a summary. The numbers:
- correlation and mutual information (64 × 64 bins, natural log) of the fixed
  image against each warped reference, over every voxel (the definition of
  `similarity.rs`);
- Dice of the two warped annotations per major region (the reference's
  ontology, `ABA_annotation.json`), after the affine and after the B-spline;
- the distance between the two sides' atlas positions of each grid point, in
  µm (25 µm voxels): median, 95th percentile, maximum.

Needs numpy, tifffile, imagecodecs (for the zstd-compressed OME-Zarr chunks).
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration"))
ONTOLOGY = Path(os.environ.get(
    "ONTOLOGY", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py/ClearMap/Resources/Atlas/"
                "ABA_annotation.json"))
VOXEL_UM = 25.0
BINS = 64
MAJOR = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
         "fiber tracts", "VS"]


def read_mhd(path: Path) -> np.ndarray:
    header = dict(line.split(" = ", 1) for line in path.read_text().splitlines() if " = " in line)
    shape = [int(v) for v in header["DimSize"].split()][::-1]
    dtype = {"MET_SHORT": "<i2", "MET_USHORT": "<u2", "MET_INT": "<i4", "MET_UINT": "<u4",
             "MET_FLOAT": "<f4", "MET_UCHAR": "u1", "MET_CHAR": "i1"}[header["ElementType"]]
    data = np.fromfile(path.parent / header["ElementDataFile"], dtype=dtype)
    return data.reshape(shape)  # (z, y, x)


def read_zarr(root: Path, level: int = 0) -> np.ndarray:
    """Level `level` of an OME-Zarr v3 image with regular chunks and the
    bytes/zstd/blosc/gzip codecs; leading singleton axes are dropped."""
    import gzip

    from imagecodecs import blosc_decode, zstd_decode
    meta = json.loads((root / str(level) / "zarr.json").read_text())
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    dtype = np.dtype({"uint8": "u1", "uint16": "<u2", "uint32": "<u4", "uint64": "<u8",
                      "int32": "<i4", "float32": "<f4", "float64": "<f8"}[meta["data_type"]])
    codecs = [c["name"] for c in meta["codecs"]]
    separator = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    fill = meta.get("fill_value", 0) or 0
    out = np.full(shape, fill, dtype)
    grid = [-(-s // c) for s, c in zip(shape, chunk)]
    for index in np.ndindex(*grid):
        path = root / str(level) / separator.join(["c"] + [str(i) for i in index])
        if not path.exists():
            continue
        data = path.read_bytes()
        if "zstd" in codecs:
            data = zstd_decode(data)
        elif "blosc" in codecs:
            data = blosc_decode(data)
        elif "gzip" in codecs:
            data = gzip.decompress(data)
        block = np.frombuffer(data, dtype).reshape(chunk)
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunk, shape))
        out[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]
    return out


def similarity(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    correlation = float(np.corrcoef(a, b)[0, 1])

    def bins(v):
        lo, hi = v.min(), v.max()
        return np.minimum(((v - lo) / (hi - lo) * BINS).astype(int), BINS - 1)

    joint = np.bincount(bins(a) * BINS + bins(b), minlength=BINS * BINS).reshape(BINS, BINS)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return {"correlation": correlation,
            "mutual_information": float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())}


def major_regions():
    """Annotation id -> index into MAJOR (or -1), from the ontology tree."""
    tree = json.loads(ONTOLOGY.read_text())
    roots = tree["msg"] if isinstance(tree, dict) and "msg" in tree else tree
    mapping = {}

    def walk(node, major):
        acronym = node.get("acronym")
        if acronym in MAJOR:
            major = MAJOR.index(acronym)
        mapping[int(node["id"])] = major
        for child in node.get("children", []):
            walk(child, major)

    for root in roots if isinstance(roots, list) else [roots]:
        walk(root, -1)
    return mapping


def dice_by_region(a: np.ndarray, b: np.ndarray, mapping) -> dict:
    ids = np.union1d(np.unique(a), np.unique(b))
    lut = {int(i): mapping.get(int(i), -1) for i in ids}
    lookup = np.vectorize(lambda v: lut[int(v)], otypes=[np.int16])
    ra, rb = lookup(a), lookup(b)
    out = {}
    for index, name in enumerate(MAJOR):
        x, y = ra == index, rb == index
        total = x.sum() + y.sum()
        out[name] = float(2 * (x & y).sum() / total) if total else None
    x, y = a > 0, b > 0
    out["brain"] = float(2 * (x & y).sum() / (x.sum() + y.sum()))
    return out


def read_elastix_points(path: Path) -> np.ndarray:
    """`OutputPoint = [ x y z ]` -> (z, y, x)."""
    rows = []
    for line in path.read_text().splitlines():
        match = re.search(r"OutputPoint = \[ ([^\]]+) \]", line)
        rows.append([float(v) for v in match.group(1).split()][::-1])
    return np.array(rows)


def read_our_points(path: Path) -> np.ndarray:
    import csv
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[float(r[f"moving_{a}"]) for a in "zyx"] for r in rows])


def distances(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=1) * VOXEL_UM
    return {"points": int(len(d)), "median_um": float(np.median(d)),
            "p95_um": float(np.percentile(d, 95)), "max_um": float(d.max())}


def write_points():
    import tifffile
    target = OUT / "points"
    target.mkdir(parents=True, exist_ok=True)
    for image, name in [("resampled_autofluorescence.tif", "auto_points"),
                        ("resampled_signal.tif", "signal_points")]:
        if not (OUT / "clearmap" / image).exists():
            continue
        fixed = tifffile.imread(OUT / "clearmap" / image)  # (z, y, x)
        threshold = np.median(fixed)
        grid = np.stack(np.meshgrid(*[np.arange(4, n - 4, 8) for n in fixed.shape],
                                    indexing="ij"), -1).reshape(-1, 3)
        grid = grid[fixed[tuple(grid.T)] > threshold]
        with open(target / f"{name}_elastix.txt", "w") as handle:
            handle.write(f"point\n{len(grid)}\n")
            for z, y, x in grid:
                handle.write(f"{x} {y} {z}\n")
        with open(target / f"{name}.csv", "w") as handle:
            handle.write("z,y,x\n")
            for z, y, x in grid:
                handle.write(f"{z},{y},{x}\n")
        print(f"{name}: {len(grid)} points")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-points", action="store_true")
    parser.add_argument("--run", default="auto_to_reference")
    args = parser.parse_args()
    if args.write_points:
        write_points()
        return
    import tifffile
    cm = OUT / "clearmap"
    ours = OUT / "ours" / args.run
    affine_only = OUT / "ours" / f"{args.run}_affine"
    fixed = tifffile.imread(cm / "resampled_autofluorescence.tif")
    result = {"inputs": {"fixed_shape_zyx": list(fixed.shape)}}

    stages = {}
    for index, name in enumerate(["affine", "bspline"]):
        theirs = read_mhd(cm / "auto_to_reference" / f"result.{index}.mhd")
        mine = read_zarr(ours / f"stage{index}_warped.ome.zarr")
        stages[name] = {
            "reference": similarity(fixed, theirs),
            "ours": similarity(fixed, mine),
            "ours_vs_reference": similarity(theirs, mine),
        }
    result["similarity"] = stages

    mapping = major_regions()
    dice = {}
    final_ours = read_zarr(ours / "annotation_in_auto.ome.zarr")
    final_theirs = read_mhd(cm / "annotation_in_auto.1" / "result.mhd")
    dice["bspline_ours_vs_reference"] = dice_by_region(final_ours, final_theirs, mapping)
    affine_theirs = read_mhd(cm / "annotation_in_auto.0" / "result.mhd")
    dice["reference_affine_vs_reference_bspline"] = dice_by_region(affine_theirs, final_theirs, mapping)
    if (affine_only / "annotation_in_auto.ome.zarr").exists():
        affine_ours = read_zarr(affine_only / "annotation_in_auto.ome.zarr")
        dice["affine_ours_vs_reference"] = dice_by_region(affine_ours, affine_theirs, mapping)
    result["dice"] = dice

    points = {}
    for index, name, ours_csv in [(0, "affine", affine_only / "points.csv"),
                                  (1, "bspline", ours / "points.csv")]:
        theirs_path = cm / f"points.{index}" / "outputpoints.txt"
        if theirs_path.exists() and ours_csv.exists():
            points[name] = distances(read_our_points(ours_csv), read_elastix_points(theirs_path))
    if (cm / "points.0" / "outputpoints.txt").exists() and (cm / "points.1" / "outputpoints.txt").exists():
        points["reference_affine_vs_reference_bspline"] = distances(
            read_elastix_points(cm / "points.0" / "outputpoints.txt"),
            read_elastix_points(cm / "points.1" / "outputpoints.txt"))
    chain_theirs = cm / "signal_points.1" / "outputpoints.txt"
    chain_ours = OUT / "ours" / "chain" / "points.csv"
    if chain_theirs.exists() and chain_ours.exists():
        points["chain_signal_to_atlas"] = distances(read_our_points(chain_ours),
                                                    read_elastix_points(chain_theirs))
        step_theirs = cm / "signal_points.0" / "outputpoints.txt"
        step_ours = OUT / "ours" / "chain" / "points_step0.csv"
        if step_ours.exists():
            points["chain_step0_signal_to_auto"] = distances(read_our_points(step_ours),
                                                             read_elastix_points(step_theirs))
    result["points"] = points

    timing = {}
    if (cm / "timing.json").exists():
        timing["reference"] = json.loads((cm / "timing.json").read_text())
    for path in [ours / "stats.json", affine_only / "stats.json",
                 OUT / "ours" / "chain" / "stats.json"]:
        if path.exists():
            timing[path.parent.name] = json.loads(path.read_text())
    result["timing"] = timing
    (OUT / "comparison.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
