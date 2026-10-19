#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# Stitch the same tiles with the reference implementation and with
# `tile-stitching`, measure both, and compare.
#
#   examples/tile-stitching/reference-clearmap/run.sh [prepare|clearmap|ours|compare|all]
#
# Environment (defaults in brackets):
#   STITCH_TILES   OME-Zarr tiles   [/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles]
#   STITCH_OUT     outputs          [/husky/henriksson/lightsheet-reference/stitching]
#   STITCH_Z       z-range          [1200:1440]
#   STITCH_ROWS / STITCH_COLS  tile indices [3,4,5 / 4,5,6]
#   CLEARMAP_PY    the reference's Python [/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python]
#   PY             a Python with zarr, numpy, scipy [python3]
#   STITCH_VARIANTS  reference variants to run, by directory name [all five]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export STITCH_OUT="${STITCH_OUT:-/husky/henriksson/lightsheet-reference/stitching}"
export STITCH_Z="${STITCH_Z:-1200:1440}"
CLEARMAP_PY="${CLEARMAP_PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}"
PY="${PY:-python3}"
STEP="${1:-all}"

prepare() { "$PY" "$HERE/prepare_tiles.py"; }

# As shipped (clearmap/); with its per-slice placement defect corrected
# (clearmap-patched/, see clearmap_stitch.py); that with the wobbly search
# along y widened from 15 to 30 (clearmap-patched-y30/); and with both its
# placement and its tracing tie-break corrected (clearmap-corrected/), the
# main comparison; and that with each tile's positions smoothed over its own
# slices only (clearmap-corrected-span/). STITCH_VARIANTS limits the run to
# some of them by name.
clearmap() {
    for variant in "clearmap 0 0 0 15" "clearmap-patched 1 0 0 15" "clearmap-patched-y30 1 0 0 30" \
                   "clearmap-corrected 1 1 0 15" "clearmap-corrected-span 1 1 1 15"; do
        set -- $variant
        if [ -n "${STITCH_VARIANTS:-}" ] && [[ " $STITCH_VARIANTS " != *" $1 "* ]]; then continue; fi
        dir="$STITCH_OUT/$1"
        mkdir -p "$dir"
        STITCH_REF_DIR=$1 STITCH_PATCH_PLACEMENT=$2 STITCH_PATCH_TRACING=$3 STITCH_PATCH_SMOOTHING=$4 \
            STITCH_WOBBLY_SHIFT_Y=$5 \
            "$PY" "$HERE/measure.py" "$dir/stats.json" -- \
            "$CLEARMAP_PY" -u "$HERE/clearmap_stitch.py" > "$dir/run.log" 2>&1
        tail -1 "$dir/run.log"
    done
}

# Defaults (wobbly search y +-15, x +-30, as the reference's), the search
# along y widened to 30 (ours-y30/, beside clearmap-patched-y30/), and
# --rigid-only (ours-rigid/) to set beside the reference as shipped.
ours() {
    (cd "$ROOT" && cargo build --release -p blockflow-tile-stitching)
    for variant in ours ours-y30 ours-rigid; do
        extra=()
        [ "$variant" = ours-y30 ] && extra=(--max-shift 30)
        [ "$variant" = ours-rigid ] && extra=(--rigid-only)
        mkdir -p "$STITCH_OUT/$variant"
        rm -rf "$STITCH_OUT/$variant/fused.zarr"
        "$PY" "$HERE/measure.py" "$STITCH_OUT/$variant/stats.json" -- \
            "$ROOT/target/release/tile-stitching" --tiles "$STITCH_OUT/tiles.json" \
            --out "$STITCH_OUT/$variant/fused.zarr" --z-range "$STITCH_Z" --out-levels 1 \
            "${extra[@]}" > "$STITCH_OUT/$variant/run.log" 2>&1
        tail -1 "$STITCH_OUT/$variant/run.log"
    done
}

compare() {
    REF=clearmap-corrected OURS=ours "$PY" "$HERE/compare.py"
    REF=clearmap-corrected-span OURS=ours "$PY" "$HERE/compare.py"
    REF=clearmap-patched OURS=ours "$PY" "$HERE/compare.py"
    REF=clearmap-patched-y30 OURS=ours-y30 "$PY" "$HERE/compare.py"
    REF=clearmap OURS=ours-rigid "$PY" "$HERE/compare.py"
    REF=clearmap OURS=ours "$PY" "$HERE/compare.py"
}

case "$STEP" in
    prepare) prepare ;;
    clearmap) clearmap ;;
    ours) ours ;;
    compare) compare ;;
    all) prepare; clearmap; ours; compare ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
