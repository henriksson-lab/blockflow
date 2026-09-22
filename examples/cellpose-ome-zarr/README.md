# Cellpose over an OME-Zarr channel

This example segments a fluorescence channel without loading the slide into memory. It uses the
normal Blockflow path: attach a plane from the OME-Zarr pyramid, ask the planner to choose a block
size for the real Cellpose operation, execute the plan, and write results that newvolim discovers:

- `labels/<layer>/`: a tiled, multiscale `uint64` label image with stable cell IDs;
- `tables/<layer>/table.csv`: area, centroid, and source-channel intensity keyed by those IDs.

The label pyramid is the annotation. newvolim can render it filled or as outlines and can inspect an
exact label ID without loading one whole-slide vector file.

## Model

The example accepts a Cellpose checkpoint. Cellpose's normal model downloader puts the default model
under `~/.cellpose/models`:

```bash
cellpose --download_model
```

Pass the resulting `cpsam_v2`, `cpsam`, or compatible safetensors checkpoint to `--model`.

## Run the 2079 slide

Channel 0 is `DAPI`; channels 4, 6, and 8 are DAPI from later acquisition rounds. A CPU run is:

```bash
cargo run --release -p blockflow-cellpose-ome-zarr -- \
  --zarr /husky/otherdataset/teresa/2079_merged_registered.zarr \
  --model ~/.cellpose/models/cpsam \
  --channel 0 \
  --layer cellpose-dapi \
  --device cpu \
  --empty-below 4 \
  --halo 64 \
  --blocks 512,1024,2048
```

Cellpose applies its standard percentile normalization independently inside each outer block. The
remaining Cellpose controls have their library defaults and can be changed with `--diameter`,
`--cellprob-threshold`, `--flow-threshold`, `--min-size`, and `--batch-size`.

For CUDA, enable the package feature and select a device:

```bash
cargo run --release -p blockflow-cellpose-ome-zarr --features cuda -- \
  --zarr /path/to/image.zarr \
  --model ~/.cellpose/models/cpsam \
  --channel 0 --layer cellpose-dapi \
  --device cuda --cuda-device 0
```

CUDA requires an NVIDIA device and a working driver. Model inference is serialized because one
model owns one device. Use the default `--workers 1`: two workers did not improve the measured CUDA
workflow and increased host memory. Batch sizes 16 and 32 also did not improve over the default
`--batch-size 8` on the 16 GiB Quadro RTX 5000 used for the benchmark.

The backend uses Cellpose's masks-only API. It produces the same masks while avoiding flow-color
rendering and auxiliary outputs that the annotation workflow does not consume. To diagnose a new
device, collect synchronized per-block stage timings from a release build:

```bash
cargo run --release -p blockflow-cellpose-ome-zarr --features cuda -- \
  --zarr /path/to/image.zarr \
  --model ~/.cellpose/models/cpsam \
  --channel 0 --layer cellpose-profile \
  --device cuda --profile-json cellpose-profile.json
```

Profiling adds device synchronizations and its wall time is not a normal performance measurement.

If CUDA reports `CUDA_ERROR_NO_DEVICE` even though `nvidia-smi` works in a normal terminal, check
whether the command is running in a container or filesystem sandbox that does not expose
`/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm`. CUDA libraries and a loaded kernel driver
are insufficient when those device nodes are hidden.

## View in newvolim

Point newvolim at the original dataset after the run:

```bash
/home/mahogny/github/claude/newvolim/target/release/newvolim-server \
  --bind 127.0.0.1:9876 \
  --page-dir /home/mahogny/github/claude/newvolim/crates/newvolim-ui/dist \
  --allow-root /husky/otherdataset/teresa \
  --dataset 2079=/husky/otherdataset/teresa/2079_merged_registered.zarr
```

Enable `cellpose-dapi` under **Labels** and select **Outlines**. The table can color labels by area
or mean DAPI intensity. Its `label_id` is also the join key for measuring other channels later with
`Measurements::for_labels`, without running Cellpose again.
