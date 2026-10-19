#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# Deep vessel filling, ours against the reference's, on a crop of a vasculature
# tile: both fill the same per-channel masks (our segmentation's, before deep
# filling), and the graphs of both filled masks are compared.
#
#   examples/vessel-graph/reference-clearmap/run_deep_fill.sh [weights|prepare|clearmap|ours|compare|all]
#
# Environment (defaults in brackets):
#   VESSEL_FILL_REFERENCE  outputs        [/husky/henriksson/lightsheet-reference/vessel-filling]
#   VESSEL_TILE            OME-Zarr tile  [/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x05.zarr]
#   VESSEL_CROP            z,y,x spans    [1600:1728,400:912,:]
#   CLEARMAP_PY            the reference's Python, with torch [/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python]
#   CLEARMAP_WEIGHTS       the reference's .pth [<ClearMap>/ImageProcessing/machine_learning/vessel_filling/resources/cnn_binary_vessel_filling.pth]
#   PY                     a Python with numpy and zarr [python3]
#   BIN                    vessel-graph built with `--features deep-fill` [target/release/vessel-graph]
#   GPU                    run the reference on the GPU too (1/0) [1 when nvidia-smi finds one]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${VESSEL_FILL_REFERENCE:-/husky/henriksson/lightsheet-reference/vessel-filling}"
TILE="${VESSEL_TILE:-/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x05.zarr}"
CROP="${VESSEL_CROP:-1600:1728,400:912,:}"
CLEARMAP_PY="${CLEARMAP_PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}"
CLEARMAP_WEIGHTS="${CLEARMAP_WEIGHTS:-/data/henriksson/github/claude/clearmap-rs/ClearMap-py/ClearMap/ImageProcessing/machine_learning/vessel_filling/resources/cnn_binary_vessel_filling.pth}"
PY="${PY:-python3}"
BIN="${BIN:-$ROOT/target/release/vessel-graph}"
if [ -z "${GPU:-}" ]; then
    GPU=0
    nvidia-smi -L >/dev/null 2>&1 && GPU=1
fi
WEIGHTS="$OUT/cnn_binary_vessel_filling.safetensors"
DIR="$OUT/crop"
STEP="${1:-all}"
timed() { local log="$1"; shift; /usr/bin/time -v "$@" 2> >(tee "$log" >&2); }

weights() {
    mkdir -p "$OUT"
    "$CLEARMAP_PY" -I "$HERE/convert_weights.py" "$CLEARMAP_WEIGHTS" "$WEIGHTS"
}

prepare() {
    mkdir -p "$DIR"
    # Our segmentation of both channels without deep filling: its per-channel
    # masks are what both sides fill.
    timed "$DIR/segment-nofill.log" "$BIN" segment --zarr "$TILE" --channel 0,1 \
        --role vessel,artery --crop "$CROP" --crop-out "$DIR/nofill.ome.zarr" \
        --no-deep-fill --work "$DIR/work-nofill"
    "$PY" -I "$HERE/export_labels.py" --zarr "$DIR/nofill.ome.zarr" --out "$DIR/input" \
        vessel-vessel-mask vessel-artery-mask
}

clearmap() {
    for role in vessel artery; do
        local mask="$DIR/input/vessel-$role-mask.npy" out="$DIR/clearmap/$role"
        # Evaluation mode (no dropout) with the whole-crop probability, then
        # twice as the reference ships it (dropout on), on the CPU; then on the
        # GPU (float16), as the reference's TubeMap runs it.
        timed "$out-cpu-eval.log" "$CLEARMAP_PY" -I "$HERE/fill_vessels.py" --mask "$mask" \
            --role "$role" --out "$out-cpu-eval" --eval --probability
        timed "$out-cpu.log" "$CLEARMAP_PY" -I "$HERE/fill_vessels.py" --mask "$mask" \
            --role "$role" --out "$out-cpu"
        timed "$out-cpu-2.log" "$CLEARMAP_PY" -I "$HERE/fill_vessels.py" --mask "$mask" \
            --role "$role" --out "$out-cpu-2"
        if [ "$GPU" = 1 ]; then
            timed "$out-gpu.log" "$CLEARMAP_PY" -I "$HERE/fill_vessels.py" --mask "$mask" \
                --role "$role" --out "$out-gpu" --device cuda
        fi
    done
}

ours() {
    timed "$DIR/segment-deepfill.log" "$BIN" segment --zarr "$TILE" --channel 0,1 \
        --role vessel,artery --crop "$CROP" --crop-out "$DIR/deepfill.ome.zarr" \
        --deep-fill-weights "$WEIGHTS" --work "$DIR/work-deepfill"
    for run in nofill deepfill; do
        timed "$DIR/graph-$run.log" "$BIN" graph --zarr "$DIR/$run.ome.zarr" \
            --artery-channel 1 --out "$DIR/graph-$run.json"
    done
}

compare() {
    (cd "$ROOT" && VESSEL_FILL_WEIGHTS="$WEIGHTS" VESSEL_FILL_REFERENCE="$OUT" \
        cargo test --release -p blockflow-vessel-graph --features deep-fill \
        --test deep_fill_reference -- --ignored --nocapture --test-threads 1)
}

case "$STEP" in
    weights) weights ;;
    prepare) prepare ;;
    clearmap) clearmap ;;
    ours) ours ;;
    compare) compare ;;
    all) weights; prepare; clearmap; ours; compare ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
