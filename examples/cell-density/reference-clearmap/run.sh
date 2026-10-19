#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# Detect cells on the same crop with the reference implementation and with
# `cell-density`, measure both, and compare.
#
#   examples/cell-density/reference-clearmap/run.sh [prepare|clearmap|ours|compare|all]
#
# Environment (defaults in brackets):
#   CELLS_TILE        OME-Zarr image [/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_03x02.zarr]
#   CELLS_CROP_START  z,y,x [400,248,246]
#   CELLS_CROP_SHAPE  z,y,x [256,512,512]; 0 runs to the end of an axis
#   CELLS_OUT         outputs [/husky/henriksson/lightsheet-reference/cells/crop]
#   CELLS_THRESHOLD   our --threshold [500, the reference's 3.1 shape threshold]
#   CELLS_OURS_ARGS   extra cell-density flags []
#   CELLS_OURS_DIR    our output directory under CELLS_OUT [ours]
#   CLEARMAP_PROCESSES  reference worker processes [10, its 3.1 default]
#   CLEARMAP_PY       the reference's Python [/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python]
#   PY                a Python with zarr, numpy, scipy [python3]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export CELLS_TILE="${CELLS_TILE:-/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_03x02.zarr}"
export CELLS_CROP_START="${CELLS_CROP_START:-400,248,246}"
export CELLS_CROP_SHAPE="${CELLS_CROP_SHAPE:-256,512,512}"
export CELLS_OUT="${CELLS_OUT:-/husky/henriksson/lightsheet-reference/cells/crop}"
CELLS_THRESHOLD="${CELLS_THRESHOLD:-500}"
CLEARMAP_PY="${CLEARMAP_PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}"
PY="${PY:-python3}"
STEP="${1:-all}"

prepare() { "$PY" "$HERE/prepare_crop.py"; }

clearmap() {
    mkdir -p "$CELLS_OUT/clearmap"
    "$PY" "$HERE/measure.py" "$CELLS_OUT/clearmap/stats.json" -- \
        "$CLEARMAP_PY" -u "$HERE/clearmap_cells.py" > "$CELLS_OUT/clearmap/run.log" 2>&1
    tail -1 "$CELLS_OUT/clearmap/run.log"
}

ours() {
    (cd "$ROOT" && cargo build --release -p blockflow-cell-density)
    local dir="$CELLS_OUT/${CELLS_OURS_DIR:-ours}"
    mkdir -p "$dir"
    # The crop as cell-density takes it; a non-positive extent runs to the end.
    local shape
    shape="$("$PY" -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["crop_shape_zyx"])))' "$CELLS_OUT/setup.json")"
    {
        date -Is
        git -C "$ROOT" log -1 --format='HEAD %h %cI'
        git -C "$ROOT" status --short examples/cell-density src | head -50
        find "$ROOT/examples/cell-density/src" -name '*.rs' -printf '%TY-%Tm-%Td %TH:%TM:%.2TS %p\n' | sort
        sha256sum "$ROOT/target/release/cell-density"
    } > "$dir/source-state.txt"
    rm -rf "$dir/out.zarr"
    # shellcheck disable=SC2086
    "$PY" "$HERE/measure.py" "$dir/stats.json" -- \
        "$ROOT/target/release/cell-density" --input "$CELLS_TILE" --output "$dir/out.zarr" \
        --crop-start "$CELLS_CROP_START" --crop-shape "$shape" \
        --threshold "$CELLS_THRESHOLD" ${CELLS_OURS_ARGS:-} > "$dir/report.json" 2> "$dir/run.log"
    tail -1 "$dir/run.log"
}

compare() { "$PY" "$HERE/compare.py"; }

case "$STEP" in
    prepare) prepare ;;
    clearmap) clearmap ;;
    ours) ours ;;
    compare) compare ;;
    all) prepare; clearmap; ours; compare ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
