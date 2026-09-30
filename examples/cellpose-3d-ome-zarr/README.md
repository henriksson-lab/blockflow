# Cellpose 3D over OME-Zarr

This is the normal Blockflow path for volumetric Cellpose: it reads a selected
`[z,y,x]` channel from an OME-Zarr pyramid, plans overlapping blocks, runs
Cellpose's orthogonal-view 3D inference, writes a label pyramid, and finalizes a
spatially indexed object table for newvolim.

Always use a release build. A CPU first run is:

```sh
cargo run --release -p blockflow-cellpose-3d-ome-zarr -- \
  --zarr /path/to/image.zarr \
  --model /path/to/cpsam.safetensors \
  --channel 0 --time 0 \
  --anisotropy 4.0 \
  --device cpu
```

For CUDA, add `--features cuda --device cuda`. `--anisotropy` is physical Z
spacing divided by XY spacing. Set `--halo-z` and `--halo` large enough to hold
the largest cell in each direction. The output is written below
`labels/<layer>` and `tables/<layer>` and registered in `labels/zarr.json`.

The first incoming volume still needs to establish useful block, halo,
normalization, and memory settings. The command prints total elapsed seconds so
that the end-to-end run can be recorded without a separate timing wrapper.

For a volume that fits comfortably in host memory, pass `--whole-volume`.
Cellpose3D already tiles the three sets of orthogonal 2D planes internally;
using one outer Blockflow block avoids repeating complete planes in overlapping
3D blocks. Keep planned outer blocks for volumes whose input, normalized copy,
flows, and labels do not fit together in host memory.

## Direct CUDA benchmark

`scripts/prepare_benchmark_crops.py` extracts the fixed crowded and isolated
16 x 64 x 64 crops recorded in `BENCHMARKS.md`. Compare normal Python Cellpose
with the native Rust inference path using:

```sh
python3 scripts/benchmark_python.py \
  --model ~/.cellpose/models/cpsam \
  --image /path/to/sparse-bench.tif --image /path/to/dense-bench.tif \
  --output /tmp/cellpose-python.json

cargo run --release -p blockflow-cellpose-3d-ome-zarr --features cuda \
  --bin cellpose-3d-benchmark -- \
  --model ~/.cellpose/models/cpsam_global.safetensors \
  --image /path/to/sparse-bench.tif --image /path/to/dense-bench.tif \
  --output /tmp/cellpose-rust.json
```

The Rust checkpoint must be the globally converted CP-SAM checkpoint. A
checkpoint converted with a smaller relative-position table is incompatible
with the orthogonal 3D views.

## Clustered PBMC production run

The validated full-volume command was:

```sh
cargo run --release -p blockflow-cellpose-3d-ome-zarr --features cuda \
  --bin cellpose-3d-ome-zarr -- \
  --zarr /husky/otherdataset/teresa/single/clustered-pbmcs.ome.zarr \
  --model ~/.cellpose/models/cpsam_global.safetensors \
  --device cuda --anisotropy 1.98 --whole-volume --workers 1 \
  --layer cellpose3d-cpsam
```

It completed in 9,738.827 seconds, found 115 objects, and wrote a five-level
label pyramid plus an indexed object table. Peak host RSS was 58.3 GiB. The
whole-volume setting is recommended for this 318-million-voxel image on a host
with sufficient memory.
