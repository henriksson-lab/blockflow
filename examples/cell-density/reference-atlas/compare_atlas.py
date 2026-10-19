#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Atlas-space cells, `cell-density --registration` against the reference.

Takes the cells `cell-density` detected and mapped (its filtered
`table.csv`), carries **the same cells** through the reference's elastix
transforms with the reference's own `Elastix.transform_points` (signal ->
autofluorescence, then autofluorescence -> atlas, as its CellMap pipeline
does), labels them the reference's way, and compares:

- the per-cell region (finest annotation id, and major region), hemisphere
  and depth;
- per-region counts (and the crust at 3 atlas voxels, the reference's
  `remove_crust` default, applied to both sides from their depths);
- region volumes (the reference's `get_lateralised_volume_map`, in µm³);
- the density map, against the reference's `Voxelization.voxelize` (sphere,
  radius 5 atlas voxels, its CellMap default) of its coordinates.

The reference's labelling (`Annotation.label_points`) truncates coordinates
(`astype(int)`), takes 0 outside the atlas and -1 for the hemisphere there;
its crust (`remove_crust`) floors and keeps `distance > threshold` with the
distance file in atlas voxels. Those are restated below on the same prepared
TIFFs its registration used (the `Annotation` module imports GUI code missing
from the environment).

Both sides start from the same sample-grid coordinates: detection voxel `i`
goes to the signal mosaic's voxel `i * scale + offset`, with `scale` and
`offset` read from the table's provenance (`cell_density.registration.
detection_to_sample`), so the comparison isolates registration and labelling.

Run with the reference's Python (env `clearmap-gt2`):

    $PY -I compare_atlas.py --table OURS/tables/cell-density-filtered \
        --density OURS_density.ome.zarr --out /husky/henriksson/lightsheet-reference/cells-atlas/clearmap

Writes `cells_reference.csv` (label_id, reference atlas z, y, x, region,
hemisphere, distance in voxels), `comparison.json`, and prints a summary.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

REG = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration"))
ATLAS = REG / "atlas"
SUFFIX = "1_2_3__slice_None_None_None__slice_None_None_None__slice_0_256_None__"
ONTOLOGY = Path(CLEARMAP) / "ClearMap" / "Resources" / "Atlas" / "ABA_annotation.json"
MAJOR = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
         "fiber tracts", "VS"]
VOXEL_UM = 25.0
CRUST_VOXELS = 3


def read_table(path: Path):
    import pandas as pd
    return pd.read_csv(path / "table.csv")


def provenance(path: Path) -> dict:
    meta = json.loads((path / "zarr.json").read_text())
    return meta["attributes"]["cell_density"]


def read_zarr(root: Path, level: int = 0) -> np.ndarray:
    """Level `level` of an OME-Zarr v3 image (zstd, blosc or gzip codecs)."""
    import gzip

    from imagecodecs import blosc_decode, zstd_decode
    meta = json.loads((root / str(level) / "zarr.json").read_text())
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    dtype = np.dtype({"uint8": "u1", "uint16": "<u2", "uint32": "<u4", "uint64": "<u8",
                      "float32": "<f4", "float64": "<f8"}[meta["data_type"]])
    codecs = [c["name"] for c in meta["codecs"]]
    separator = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    out = np.zeros(shape, dtype)
    for index in np.ndindex(*[-(-s // c) for s, c in zip(shape, chunk)]):
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


def reference_transform(sample_zyx: np.ndarray, work: Path) -> np.ndarray:
    """The reference's chain: `transform_points` through signal -> auto, then
    auto -> atlas. Its points are `(x, y, z)` of its arrays, our `z, y, x`
    reversed (the TIFFs its registration read are our arrays' pages)."""
    import ClearMap.Alignment.Elastix as elx
    work.mkdir(parents=True, exist_ok=True)
    points = sample_zyx[:, ::-1].astype(float)
    points = elx.transform_points(
        points, transform_parameter_file=str(REG / "clearmap" / "signal_to_auto" / "TransformParameters.0.txt"),
        indices=False, binary=True, result_directory=str(work / "step0"),
        temp_file=str(work / "step0.bin"))
    points = elx.transform_points(
        points, transform_parameter_file=str(REG / "clearmap" / "auto_to_reference" / "TransformParameters.1.txt"),
        indices=False, binary=True, result_directory=str(work / "step1"),
        temp_file=str(work / "step1.bin"))
    return np.asarray(points)[:, ::-1]


def label(points_zyx: np.ndarray, volume: np.ndarray, invalid, rounding: str) -> np.ndarray:
    """`label_points`: truncate (`astype(int)`) or round to the nearest voxel,
    `invalid` outside the volume."""
    if rounding == "truncate":
        index = points_zyx.astype(int)
    else:
        index = np.rint(points_zyx).astype(int)
    inside = np.all((index >= 0) & (index < np.array(volume.shape)), axis=1)
    out = np.full(len(points_zyx), invalid, dtype=np.float64)
    out[inside] = volume[tuple(index[inside].T)]
    return out


def major_map():
    tree = json.loads(ONTOLOGY.read_text())
    roots = tree["msg"] if isinstance(tree, dict) and "msg" in tree else tree
    mapping = {}

    def walk(node, major):
        if node.get("acronym") in MAJOR:
            major = node["acronym"]
        mapping[int(node["id"])] = major
        for child in node.get("children", []):
            walk(child, major)

    for root in roots if isinstance(roots, list) else [roots]:
        walk(root, None)
    return mapping


def counts(regions, hemispheres, keep):
    out = {}
    for r, h in zip(regions[keep], hemispheres[keep]):
        out[(r, h)] = out.get((r, h), 0) + 1
    return out


def count_agreement(ours: dict, theirs: dict, top: int = 15):
    keys = sorted(set(ours) | set(theirs), key=lambda k: -(ours.get(k, 0) + theirs.get(k, 0)))
    total = sum(ours.values())
    l1 = sum(abs(ours.get(k, 0) - theirs.get(k, 0)) for k in keys)
    rows = [{"region": k[0], "hemisphere": k[1], "ours": ours.get(k, 0), "reference": theirs.get(k, 0),
             "relative_difference": (ours.get(k, 0) - theirs.get(k, 0)) / max(theirs.get(k, 0), 1)}
            for k in keys[:top]]
    big = [k for k in keys if theirs.get(k, 0) >= 100]
    within5 = sum(abs(ours.get(k, 0) - theirs[k]) <= 0.05 * theirs[k] for k in big)
    return {"cells": total, "regions": len(keys),
            "sum_abs_count_difference_over_cells": l1 / max(total, 1),
            "regions_with_100_plus_reference_cells": len(big),
            "of_those_within_5_percent": int(within5), "largest": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True,
                        help="cell-density's filtered table directory")
    parser.add_argument("--density", type=Path, help="cell-density's atlas density image")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import tifffile

    args.out.mkdir(parents=True, exist_ok=True)
    table = read_table(args.table)
    record = provenance(args.table)
    mapping = record["registration"]["detection_to_sample"]
    scale, offset = np.array(mapping["scale_zyx"]), np.array(mapping["offset_zyx"])
    detection = table[["z", "y", "x"]].to_numpy(float)
    sample = detection * scale + offset
    ours_atlas = table[["atlas_z", "atlas_y", "atlas_x"]].to_numpy(float)

    t0 = time.time()
    theirs_atlas = reference_transform(sample, args.out / "transformix")
    transform_s = time.time() - t0

    annotation = tifffile.imread(ATLAS / f"ABA_25um_2017_annotation_{SUFFIX}.tif").astype(np.int64)
    hemispheres = tifffile.imread(ATLAS / f"ABA_25um_2017_hemispheres_{SUFFIX}.tif").astype(np.int64)
    distance = tifffile.imread(ATLAS / f"ABA_25um_2017_distance_to_surface_{SUFFIX}.tif")
    t0 = time.time()
    theirs_region = label(theirs_atlas, annotation, 0, "truncate").astype(np.int64)
    theirs_hemi = label(theirs_atlas, hemispheres, -1, "truncate").astype(np.int64)
    floor = np.floor(theirs_atlas)
    theirs_depth = label(floor, distance, np.nan, "truncate")
    label_s = time.time() - t0
    np.savetxt(args.out / "cells_reference.csv",
               np.column_stack([table["label_id"], theirs_atlas, theirs_region, theirs_hemi, theirs_depth]),
               delimiter=",", header="label_id,atlas_z,atlas_y,atlas_x,region,hemisphere,distance_voxels",
               comments="", fmt=["%d", "%.6f", "%.6f", "%.6f", "%d", "%d", "%.6f"])

    ours_region = table["region"].to_numpy(np.int64)
    ours_hemi = table["hemisphere"].to_numpy(np.int64)
    # Atlas voxels; tables written before the shared atlas-space helper had
    # `distance_to_surface` in µm.
    if "distance_to_surface_voxels" in table:
        ours_depth = table["distance_to_surface_voxels"].to_numpy(float)
    else:
        ours_depth = table["distance_to_surface"].to_numpy(float) / VOXEL_UM
    # Isolating the two causes: our coordinates labelled the reference's way,
    # and the reference's coordinates labelled to the nearest voxel.
    ours_truncated = label(ours_atlas, annotation, 0, "truncate").astype(np.int64)
    theirs_nearest = label(theirs_atlas, annotation, 0, "nearest").astype(np.int64)
    # Our own nearest-voxel lookup against these TIFFs: a check that the
    # OME-Zarr atlas is the TIFF atlas.
    ours_check = label(ours_atlas, annotation, 0, "nearest").astype(np.int64)

    majors = major_map()
    major = np.vectorize(lambda r: majors.get(int(r)) or "none", otypes=[object])
    distance_um = np.linalg.norm(ours_atlas - theirs_atlas, axis=1) * VOXEL_UM
    n = len(table)
    keep_ours = ours_depth > CRUST_VOXELS
    keep_theirs = theirs_depth > CRUST_VOXELS
    result = {
        "cells": n,
        "atlas_position_difference_um": {"median": float(np.median(distance_um)),
                                         "p95": float(np.percentile(distance_um, 95)),
                                         "max": float(distance_um.max())},
        "same_region": float(np.mean(ours_region == theirs_region)),
        "same_major_region": float(np.mean(major(ours_region) == major(theirs_region))),
        "same_hemisphere": float(np.mean(ours_hemi == theirs_hemi)),
        "same_region_both_truncated": float(np.mean(ours_truncated == theirs_region)),
        "same_region_ours_truncated_vs_ours_nearest": float(np.mean(ours_truncated == ours_region)),
        "same_region_reference_transform_both_nearest": float(np.mean(theirs_nearest == ours_region)),
        "same_region_same_transform_truncate_vs_nearest": float(np.mean(theirs_nearest == theirs_region)),
        "our_labels_match_tiff_atlas": float(np.mean(ours_check == ours_region)),
        "outside_atlas": {"ours": int(np.sum(ours_hemi < 0)), "reference": int(np.sum(theirs_hemi < 0))},
        "counts_finest": count_agreement(counts(ours_region, ours_hemi, np.ones(n, bool)),
                                         counts(theirs_region, theirs_hemi, np.ones(n, bool))),
        "counts_major": count_agreement(counts(major(ours_region), ours_hemi, np.ones(n, bool)),
                                        counts(major(theirs_region), theirs_hemi, np.ones(n, bool))),
        "counts_finest_both_truncated": count_agreement(
            counts(ours_truncated, ours_hemi, np.ones(n, bool)),
            counts(theirs_region, theirs_hemi, np.ones(n, bool))),
        "crust_3_voxels": {
            "kept_ours": int(keep_ours.sum()), "kept_reference": int(keep_theirs.sum()),
            "same_decision": float(np.mean(keep_ours == keep_theirs)),
            "counts_finest": count_agreement(counts(ours_region, ours_hemi, keep_ours),
                                             counts(theirs_region, theirs_hemi, keep_theirs)),
        },
        "seconds_reference": {"transform_points_two_steps": transform_s, "label": label_s},
    }

    # Region volumes, the reference's `get_lateralised_volume_map` (µm³).
    regions_csv = args.table.parent / args.table.name.replace("-filtered", "") / "regions.csv"
    if regions_csv.exists():
        import pandas as pd
        ours_volumes = pd.read_csv(regions_csv)
        mismatched = 0
        for row in ours_volumes.itertuples():
            if row.hemisphere < 0:
                continue
            truth = int(np.sum((annotation == row.region) & (hemispheres == row.hemisphere)))
            mismatched += int(truth != row.volume_voxels)
        result["region_volumes"] = {"rows": int((ours_volumes.hemisphere >= 0).sum()),
                                    "voxel_counts_differing_from_reference": mismatched}

    if args.density is not None and args.density.exists():
        import ClearMap.Analysis.Measurements.Voxelization as vox
        t0 = time.time()
        theirs_density = vox.voxelize(theirs_atlas[:, ::-1].copy(), sink=None, shape=annotation.shape[::-1],
                                      weights=None, method="sphere", radius=(5, 5, 5), verbose=False)
        theirs_density = np.asarray(theirs_density).transpose(2, 1, 0)
        result["seconds_reference"]["voxelize"] = time.time() - t0
        ours_density = read_zarr(args.density).astype(np.float64)
        result["density"] = {
            "total_ours": float(ours_density.sum()), "total_reference": float(theirs_density.sum()),
            "correlation": float(np.corrcoef(ours_density.ravel(), theirs_density.ravel())[0, 1]),
        }

    (args.out / "comparison.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict) or k in
                      ("atlas_position_difference_um", "outside_atlas", "seconds_reference",
                       "density", "region_volumes")}, indent=1, default=str))
    for key in ("counts_finest", "counts_finest_both_truncated", "counts_major"):
        print(key, {k: v for k, v in result[key].items() if k != "largest"})
    print("crust", {k: v for k, v in result["crust_3_voxels"].items() if k != "counts_finest"})


if __name__ == "__main__":
    main()
