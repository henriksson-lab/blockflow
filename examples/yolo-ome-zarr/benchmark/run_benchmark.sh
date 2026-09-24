#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 COCO_VAL_IMAGE_DIR YOLOV11_RS_CHECKOUT OUTPUT_DIR" >&2
  exit 2
fi

repo=$(cd "$(dirname "$0")/../../.." && pwd)
source_dir=$1
translation=$2
output=$3
original="$translation/YOLOv11-pt"
original_weights="$original/weights/best.pt"
rust_weights="$translation/weights/model.safetensors"
config="$translation/default_args.yaml"

if [[ -e "$output" ]]; then
  echo "output already exists: $output" >&2
  exit 2
fi

cd "$repo"
python3 examples/yolo-ome-zarr/benchmark/prepare.py \
  --source "$source_dir" --output "$output" --images 16 --repeats 2
torch_lib=$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cargo build --release -p blockflow-yolo-ome-zarr \
  --no-default-features --features libtorch

for run in 1 2 3; do
  target/release/yolo-ome-zarr \
    --zarr "$output/input.zarr" \
    --weights "$rust_weights" --config "$config" \
    --block 640 --halo 0 --workers 4 \
    --out "$output/blockflow-$run.csv" \
    --summary "$output/blockflow-$run-summary.csv" \
    --work "$output/work-$run" >"$output/blockflow-$run.log"
  cat "$output/blockflow-$run.log"

  detection_args=()
  if [[ $run -eq 1 ]]; then
    detection_args=(--detections "$output/original.csv")
  fi
  python3 examples/yolo-ome-zarr/benchmark/original.py \
    --original "$original" --weights "$original_weights" \
    --images "$output/images.txt" --precision fp32 --batch-size 1 \
    --output "$output/original-$run.json" "${detection_args[@]}"
done

python3 examples/yolo-ome-zarr/benchmark/compare.py \
  --blockflow "$output/blockflow-1.csv" --original "$output/original.csv" \
  --output "$output/agreement.json"
python3 examples/yolo-ome-zarr/benchmark/summarize.py "$output"
