#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# Binarise the same volumes per channel role with the reference and with
# `vessel-graph`, and report the Dice coefficient per role.
#
#   examples/vessel-graph/reference-clearmap/run.sh [prepare|clearmap|compare|all]
#
# Environment (defaults in brackets):
#   VESSEL_REFERENCE  outputs        [/husky/henriksson/lightsheet-reference/vessels]
#   VESSEL_FIXTURES   recorded crop  [/husky/henriksson/clearmap-img/outputs]
#   VESSEL_TILE       OME-Zarr tile  [/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x05.zarr]
#   VESSEL_REGION     z,y,x spans    [1600:1664 400:656 74:330]
#   CLEARMAP_PY       the reference's Python [/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python]
#   PY                a Python with numpy and zarr [python3]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export VESSEL_REFERENCE="${VESSEL_REFERENCE:-/husky/henriksson/lightsheet-reference/vessels}"
FIXTURES="${VESSEL_FIXTURES:-/husky/henriksson/clearmap-img/outputs}"
TILE="${VESSEL_TILE:-/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x05.zarr}"
read -r Z Y X <<< "${VESSEL_REGION:-1600:1664 400:656 74:330}"
CLEARMAP_PY="${CLEARMAP_PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}"
PY="${PY:-python3}"
STEP="${1:-all}"
CROP="$VESSEL_REFERENCE/fixture"
REGION="$VESSEL_REFERENCE/tile05x05-z1600"

prepare() {
    mkdir -p "$CROP"
    cp "$FIXTURES/vasc-binarize/source.npy" "$CROP/source_xyz.npy"
    "$PY" "$HERE/prepare_region.py" --tile "$TILE" --out "$REGION" --z "$Z" --y "$Y" --x "$X"
}

clearmap() {
    for role in vessel artery; do
        "$CLEARMAP_PY" -u "$HERE/binarize_roles.py" --source "$CROP/source_xyz.npy" \
            --layout xyz --role "$role" --out "$CROP/clearmap_$role.npy"
    done
    # Channel 0 is the all-vessel stain, channel 1 the artery stain.
    "$CLEARMAP_PY" -u "$HERE/binarize_roles.py" --source "$REGION/source_c0.npy" \
        --layout zyx --role vessel --out "$REGION/clearmap_vessel.npy"
    "$CLEARMAP_PY" -u "$HERE/binarize_roles.py" --source "$REGION/source_c1.npy" \
        --layout zyx --role artery --out "$REGION/clearmap_artery.npy"
}

compare() {
    (cd "$ROOT" && cargo test --release -p blockflow-vessel-graph --test segment_roles \
        -- --ignored --nocapture)
}

case "$STEP" in
    prepare) prepare ;;
    clearmap) clearmap ;;
    compare) compare ;;
    all) prepare; clearmap; compare ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
