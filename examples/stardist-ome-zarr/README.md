# StarDist over an OME-Zarr channel

This example segments fluorescent nuclei without loading the slide into memory. It attaches one
channel of an existing `[c, y, x]` OME-Zarr pyramid, offers the actual StarDist fragment phase to the
block planner, executes the chosen plan, and writes results back in forms that newvolim discovers:

- `labels/<layer>/`: a multiscale `uint64` label image with stable cell IDs;
- `tables/<layer>/table.csv`: area, centroid, and DAPI intensity keyed by the same IDs.

The label pyramid is the annotation. In newvolim it can be shown filled or as outlines, clicked to
inspect an exact ID, and colored by a measurement-table column. A vector GeoJSON file is a poor fit
for a whole slide because it must be loaded as one document; the label layer stays tiled.

## Model

Use the official `2D_versatile_fluo` model, which is trained for fluorescent nuclei. Download and
unpack it:

```bash
mkdir -p .tmp/stardist-models
curl -L \
  https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip \
  -o .tmp/stardist-models/2D_versatile_fluo.zip
unzip .tmp/stardist-models/2D_versatile_fluo.zip -d .tmp/stardist-models
find .tmp/stardist-models -name config.json -print
```

Pass the directory printed by the last command as `--model`. It must contain `config.json`,
`thresholds.json`, and `weights_best.h5`.

## Run the 2079 slide

Channel 0 is named `DAPI` in this dataset. The later acquisition rounds also contain DAPI channels
at indices 4, 6, and 8.

```bash
cargo run --release -p blockflow-stardist-ome-zarr -- \
  --zarr /husky/otherdataset/teresa/2079_merged_registered.zarr \
  --model .tmp/stardist-models/2D_versatile_fluo \
  --channel 0 \
  --layer stardist-dapi \
  --low 1 --high 70 \
  --empty-below 4 \
  --halo 64 \
  --blocks 512,1024,2048
```

Build and run with `--features cuda` on a host with a working CUDA driver. The CPU backend is useful
for a crop or validation run, but a 66,048 × 157,440 full-resolution channel is a GPU-scale job.

The intensity range is fixed across all blocks so the same pixel always has the same normalized
value. In a random sample of 24 stored level-0 DAPI chunks from this slide, the 99.8th percentile was
69 and most background was 3–4, which motivates the values above. Inspect representative regions
before the full run and adjust `--low`, `--high`, the model thresholds, or the input scale if nuclei
are systematically missed or merged. `--halo` must be at least the largest expected nucleus diameter
in pixels.

## View in newvolim

Point newvolim at the original dataset after the run:

```bash
/home/mahogny/github/claude/newvolim/target/release/newvolim-server \
  --bind 127.0.0.1:9876 \
  --page-dir /home/mahogny/github/claude/newvolim/crates/newvolim-ui/dist \
  --allow-root /husky/otherdataset/teresa \
  --dataset 2079=/husky/otherdataset/teresa/2079_merged_registered.zarr
```

Enable `stardist-dapi` under **Labels** and select **Outlines**. The matching measurement table can
paint labels by area or mean DAPI intensity.

The table already establishes the later quantification contract: `label_id` is the join key. Further
channels can be measured over `labels/stardist-dapi/0` with `Measurements::for_labels`, without
running StarDist again.
