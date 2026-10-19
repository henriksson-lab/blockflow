#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Inputs for the registration comparison, made the reference's way.

Writes under `$REG_OUT` (default `/husky/henriksson/lightsheet-reference/registration`):

- `clearmap/resampled_autofluorescence.tif`: the osf-cztq2 autofluorescence
  stack (5, 5, 6 µm in x, y, z) resampled to 25 µm with the reference's
  `Resampling.resample`, as its CellMap script does;
- `atlas/ABA_25um_2017_{annotation,reference,hemispheres,distance_to_surface}_*.tif`:
  the 2017 ABA 25 µm atlas prepared with `Annotation.prepare_annotation_files`,
  with orientation `(1, 2, 3)` and slicing `(:, :, 0:256)` (one hemisphere and
  the midline, along z). The reference's CellMap script uses `(1, -2, 3)` for
  its own example data; for this sample that puts the cerebellum at the wrong
  end, as mid-sagittal planes of the two show (cerebellum at high y and
  dorsal at low x in both, lateral at low z), so the y axis is kept;
- `clearmap/resampled_signal.tif` (optional, `--signal`): a rough mosaic of
  the cFos tiles at 25 µm, placed at their nominal grid positions, for the
  signal -> autofluorescence step. It is a test input, not a stitched sample.

TIFF axis order is the reference's: pages are z, so the arrays written are
`(x, y, z)` in its convention and read back by `img2omezarr` as `(z, y, x)`.

Run with the reference's Python (env `clearmap-gt2`).
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

OUT = Path(os.environ.get("REG_OUT", "/husky/henriksson/lightsheet-reference/registration"))
AUTO_TIFFS = Path("/husky/otherdataset/teresa/clearmap_em/downloads/osf-cztq2/autofluorescence")
CFOS = Path("/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos")
ATLAS = Path("/husky/otherdataset/teresa/clearmap_em/downloads/atlas")
ORIENTATION = (1, 2, 3)
SLICING = (slice(None), slice(None), slice(0, 256))


def resample_autofluorescence(processes: int) -> dict:
    import ClearMap.Alignment.Resampling as res

    import tifffile

    out = OUT / "clearmap"
    out.mkdir(parents=True, exist_ok=True)
    # The reference's CellMap script first converts the plane list to one
    # `.npy` (`io.convert`), and resamples that. Its plane-list reader trips
    # over the first plane's OME series (which spans the whole stack), so the
    # conversion is done here, plane by plane, in its `(x, y, z)` order.
    stack = OUT / "autofluorescence_xyz.npy"
    t0 = time.time()
    if not stack.exists():
        files = sorted(AUTO_TIFFS.glob("*.tif"))
        first = tifffile.TiffFile(files[0]).pages[0].asarray()
        array = np.lib.format.open_memmap(stack, mode="w+", dtype=first.dtype,
                                          shape=(first.shape[1], first.shape[0], len(files)))
        for z, path in enumerate(files):
            array[:, :, z] = tifffile.TiffFile(path).pages[0].asarray().T
        array.flush()
        del array
    convert_s = time.time() - t0
    sink = out / "resampled_autofluorescence.tif"
    t0 = time.time()
    res.resample(str(stack), resampled=str(sink), original_resolution=(5, 5, 6),
                 resampled_resolution=(25, 25, 25), processes=processes, verbose=False)
    return {"convert_autofluorescence_s": convert_s,
            "resample_autofluorescence_s": time.time() - t0}


def prepare_atlas() -> dict:
    """`Annotation.prepare_annotation_files`, restated: that module imports the
    reference's GUI colour code (`vispy`), which this environment lacks. The
    steps are its own: read with its IO, transpose and reverse by
    `orientation_to_transposition`, slice, write with its IO, named by its
    `get_atlas_filepath` rule."""
    import ClearMap.IO.IO as clearmap_io
    from ClearMap.Alignment.Resampling import orientation_to_transposition

    # The 2017 files: the default atlas of the 3.1 configuration.
    names = {
        "annotation": ATLAS / "ABA_25um_2017_annotation.tif",
        "reference": ATLAS / "ABA_25um_2017_reference.tif",
        "hemispheres": ATLAS / "ABA_25um_2017_hemispheres.tif",
        "distance_to_surface": ATLAS / "clearmap-work" / "ABA_25um_2017_distance_to_surface.tif",
    }
    directory = OUT / "atlas"
    directory.mkdir(parents=True, exist_ok=True)
    postfix = f"{ORIENTATION[0]}_{ORIENTATION[1]}_{ORIENTATION[2]}_{SLICING}"
    postfix = postfix.replace(" ", "")
    for c in "(,)":
        postfix = postfix.replace(c, "_")
    t0 = time.time()
    files = {}
    for key, path in names.items():
        target = directory / f"ABA_25um_2017_{key}_{postfix}.tif"
        if not target.exists():
            data = np.array(clearmap_io.read(str(path)))
            data = data.transpose(orientation_to_transposition(ORIENTATION))
            flips = tuple(slice(None, None, -1) if o < 0 else slice(None) for o in ORIENTATION)
            data = data[flips][SLICING]
            clearmap_io.write(str(target), np.ascontiguousarray(data))
        files[key] = str(target)
    return {"prepare_atlas_s": time.time() - t0, "atlas_files": files}


def mosaic_signal() -> dict:
    """cFos tiles at their nominal grid positions, 25 µm, mean in overlaps.

    The tiles' pitch is taken so that the mosaic spans the autofluorescence
    stack's physical extent (both were acquired over the same field)."""
    import tifffile
    from imagecodecs import blosc_decode, zstd_decode

    def read_level(root: Path, level: int):
        meta = json.loads((root / str(level) / "zarr.json").read_text())
        shape, chunk = meta["shape"], meta["chunk_grid"]["configuration"]["chunk_shape"]
        dtype = np.dtype({"uint16": "<u2", "uint8": "u1", "float32": "<f4"}[meta["data_type"]])
        codecs = [c["name"] for c in meta["codecs"]]
        out = np.zeros(shape, dtype)
        grid = [-(-s // c) for s, c in zip(shape, chunk)]
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    path = root / str(level) / "c" / str(i) / str(j) / str(k)
                    if not path.exists():
                        continue
                    data = path.read_bytes()
                    if "blosc" in codecs:
                        data = blosc_decode(data)
                    elif "zstd" in codecs:
                        data = zstd_decode(data)
                    block = np.frombuffer(data, dtype).reshape(chunk)
                    sl = tuple(slice(a * c, min((a + 1) * c, s)) for a, c, s in zip((i, j, k), chunk, shape))
                    out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
        scale = json.loads((root / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0][
            "datasets"][level]["coordinateTransformations"][0]["scale"]
        return out, scale

    from scipy.ndimage import zoom

    tiles = sorted(CFOS.glob("tile_*x*.zarr"))
    rows = 1 + max(int(t.name[5:7]) for t in tiles)
    cols = 1 + max(int(t.name[8:10]) for t in tiles)
    auto_extent = np.array([1030 * 6.0, 2387 * 5.0, 1422 * 5.0])  # z, y, x in µm
    level = 4
    first, scale = read_level(tiles[0], level)
    tile_extent = np.array(first.shape) * np.array(scale)
    pitch_y = (auto_extent[1] - tile_extent[1]) / (rows - 1)
    pitch_x = (auto_extent[2] - tile_extent[2]) / (cols - 1)
    shape = np.round(auto_extent / 25.0).astype(int)
    total = np.zeros(shape, np.float64)
    weight = np.zeros(shape, np.float64)
    for tile in tiles:
        r, c = int(tile.name[5:7]), int(tile.name[8:10])
        data, scale = read_level(tile, level)
        factor = np.array(scale) / 25.0
        small = zoom(data.astype(np.float32), factor, order=1)
        start = np.round(np.array([0.0, r * pitch_y, c * pitch_x]) / 25.0).astype(int)
        stop = np.minimum(start + small.shape, shape)
        sl = tuple(slice(a, b) for a, b in zip(start, stop))
        total[sl] += small[tuple(slice(0, b - a) for a, b in zip(start, stop))]
        weight[sl] += 1.0
    mosaic = (total / np.maximum(weight, 1.0)).astype(np.float32)
    out = OUT / "clearmap"
    out.mkdir(parents=True, exist_ok=True)
    # (z, y, x) -> pages z, rows y, cols x: the reference's (x, y, z) on reading.
    tifffile.imwrite(out / "resampled_signal.tif", mosaic)
    return {"signal_rows": rows, "signal_cols": cols, "signal_pitch_um": [pitch_y, pitch_x],
            "signal_shape_zyx": list(mosaic.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--signal", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    if not (OUT / "clearmap" / "resampled_autofluorescence.tif").exists():
        record.update(resample_autofluorescence(args.processes))
    record.update(prepare_atlas())
    if args.signal:
        record.update(mosaic_signal())
    (OUT / "clearmap").mkdir(parents=True, exist_ok=True)
    (OUT / "clearmap" / "prepare.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
