# StarDist 3D over OME-Zarr

This example reads an axis-aware OME-Zarr `[z,y,x]` volume through Blockflow,
runs the native StarDist3D Candle CUDA network and 3D polyhedron
postprocessing, and writes the same label-pyramid and object-table layout as the
Cellpose 3D example.

Always build and run it in release mode:

```sh
cargo run --release -p blockflow-stardist-3d-ome-zarr --features cuda -- \
  --zarr /path/to/image.zarr \
  --model /path/to/stardist3d-model \
  --channel 0 --time 0
```

The model directory must contain `config.json`, `thresholds.json`, and normally
`weights_best.h5`. There is no general official StarDist3D model, so model
choice is part of dataset validation. Measure one representative crop first:
3D polyhedron rendering may dominate the network time.

Outputs are registered under `labels/<layer>` and `tables/<layer>` for
newvolim. Set `--halo-z` and `--halo` from the largest expected object extent.

## Direct CUDA benchmark

The Python benchmark script times raw prediction and instance construction
separately and writes an NPZ artifact. The upstream Rust benchmark consumes
that artifact, checks the tensors and labels, and reports the matching native
timings:

```sh
python3 scripts/benchmark_python.py \
  --stardist-repo /path/to/stardist-rs \
  --image /path/to/dense-bench.tif \
  --output /tmp/stardist-dense.npz

cd /path/to/stardist-rs
cargo run --release --features candle-cuda,hdf5 \
  --example bench_candle_real_data -- 3d /tmp/stardist-dense.npz cuda
```

Use the crop preparation script in the Cellpose 3D example to reproduce the
real-data crops.
