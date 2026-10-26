#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# Graph-level comparison of `vessel-graph` against the reference's TubeMap on
# one volume, with each side's graph half also run on the other side's
# skeleton and mask, so segmentation and graph building can be told apart.
#
#   examples/vessel-graph/reference-clearmap/compare_graph.sh WORK [all|cross|pairs|atlas|table]
#
# WORK must hold (the benchmark's `vessels` workload writes the same files):
#   inputs/src.ome.zarr               the two-channel image (vessel, artery)
#   clearmap/                         `benchmark/clearmap/vessels.py --out` (its .npy files and graph_reduced.gt)
#   ours/image.ome.zarr, ours/graph.json
#                                     `vessel-graph segment` + `graph --all-components --out` on a copy
# and, for `atlas`, ours/graph-atlas.json (`graph --registration --atlas`).
#
# Environment (defaults in brackets):
#   BIN           blockflow binaries   [target/release of this checkout]
#   CLEARMAP_PY   the reference's Python [/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python]
#   PY            numpy + scipy + zarr  [python3]
#   PROCESSES     reference processes and our workers [12]
#   REGISTRATION, ATLAS                 for `atlas` (sample -> atlas, and the prepared atlas)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
W="$(cd "$1" && pwd)"
STEP="${2:-all}"
BIN="${BIN:-$ROOT/target/release}"
CLEARMAP_PY="${CLEARMAP_PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}"
PY="${PY:-python3}"
N="${PROCESSES:-12}"
SPACING_XYZ="$("$PY" -I -c "
import json, sys
g = json.load(open('$W/inputs/src.ome.zarr/zarr.json'))['attributes']['ome']['multiscales'][0]
s = [t for t in g['datasets'][0]['coordinateTransformations'] if t['type'] == 'scale'][0]['scale'][-3:]
print(','.join(str(v) for v in s[::-1]))")"
SPACING_ZYX="$(echo "$SPACING_XYZ" | awk -F, '{print $3","$2","$1}')"
CM="$W/clearmap"
cm_graph() { "$CLEARMAP_PY" -u -I "$HERE/clearmap_graph.py" "$@" 2> >(grep -v -e pkg_resources -e declare_namespace >&2); }
compare() { "$PY" -I "$HERE/compare_graph.py" "$@"; }

cross() {
    mkdir -p "$W/cross"
    cm_graph export --graph "$CM/graph_reduced.gt" --out "$CM/graph.npz"
    # Our labels and the artery channel as .npy for the reference.
    L="$W/cross/ours-labels"
    "$PY" -I "$HERE/export_labels.py" --zarr "$W/ours/image.ome.zarr" --out "$L" \
        vessel-skeleton vessel-mask vessel-artery-mask
    "$PY" -I -c "
import numpy as np, zarr
np.save('$L/artery-raw.npy', np.asarray(zarr.open_array('$W/ours/image.ome.zarr/0', mode='r')[1]))"
    # The reference's graph half on our skeleton and mask.
    cm_graph build --skeleton "$L/vessel-skeleton.npy" --mask "$L/vessel-mask.npy" \
        --artery-mask "$L/vessel-artery-mask.npy" --artery-raw "$L/artery-raw.npy" --layout zyx \
        --spacing "$SPACING_XYZ" --processes "$N" --out "$W/cross/ref-graph-on-ours"
    # The reference's skeletonisation of our mask, then its graph half.
    cm_graph skeletonize --mask "$L/vessel-mask.npy" --layout zyx --processes "$N" \
        --out "$W/cross/ref-skeleton-of-our-mask.npy"
    cm_graph build --skeleton "$W/cross/ref-skeleton-of-our-mask.npy:xyz" --mask "$L/vessel-mask.npy" \
        --artery-mask "$L/vessel-artery-mask.npy" --artery-raw "$L/artery-raw.npy" --layout zyx \
        --spacing "$SPACING_XYZ" --processes "$N" --out "$W/cross/ref-graph-on-ref-skeleton-of-our-mask"
    # The same with our mask's axes handed over reversed (our [z, y, x] as its
    # [x, y, z]): the reference's thinning in our axis frame.
    "$PY" -I -c "
import numpy as np
np.save('$W/cross/our-mask-zyx-as-xyz.npy', np.asfortranarray(np.load('$L/vessel-mask.npy')))"
    cm_graph skeletonize --mask "$W/cross/our-mask-zyx-as-xyz.npy" --layout xyz --processes "$N" \
        --out "$W/cross/ref-skeleton-of-our-mask-in-our-frame.npy"
    rm "$W/cross/our-mask-zyx-as-xyz.npy"
    # Our graph half on the reference's skeleton and masks (its artery mask is
    # the artery channel's filled mask, as `_set_artery_binary` reads).
    rm -rf "$W/cross/ours-on-ref"
    mkdir -p "$W/cross/ours-on-ref"
    cp -r "$W/ours/image.ome.zarr" "$W/cross/ours-on-ref/image.ome.zarr"
    rm -rf "$W/cross/ours-on-ref/image.ome.zarr/tables"
    compare import-labels --zarr "$W/cross/ours-on-ref/image.ome.zarr" \
        vessel-skeleton="$CM/skeleton.npy:xyz" vessel-mask="$CM/binary_final.npy:xyz" \
        vessel-artery-mask="$CM/filled_c1.npy:xyz"
    "$BIN/vessel-graph" graph --zarr "$W/cross/ours-on-ref/image.ome.zarr" --artery-channel 1 \
        --all-components --workers "$N" --work "$W/cross/ours-on-ref/work" --no-tables \
        --out "$W/cross/ours-on-ref/graph.json"
    # The reference's classifier (no atlas: the distance test passes, as ours).
    for g in "$CM" "$W/cross/ref-graph-on-ours" "$W/cross/ref-graph-on-ref-skeleton-of-our-mask"; do
        cm_graph classify --graph "$g/graph_reduced.gt" --out "$g/classify.npz"
    done
}

pairs() {
    mkdir -p "$W/compare"
    local our_skel="$W/ours/image.ome.zarr#vessel-skeleton" our_mask="$W/ours/image.ome.zarr#vessel-mask"
    local ref_skel="$CM/skeleton.npy:xyz" ref_mask="$CM/binary_final.npy:xyz"
    # 1. Everything: ours against the reference.
    compare pair --a ours="$W/ours/graph.json" --b ref="$CM/graph.npz" --classes-b "$CM/classify.npz" \
        --skeleton-a "$our_skel" --skeleton-b "$ref_skel" --mask-a "$our_mask" --mask-b "$ref_mask" \
        --spacing "$SPACING_ZYX" --out "$W/compare/1-full.json" > /dev/null
    # 2. Graph half only, on our skeleton and mask.
    compare pair --a ours="$W/ours/graph.json" --b ref-graph="$W/cross/ref-graph-on-ours/graph.npz" \
        --classes-b "$W/cross/ref-graph-on-ours/classify.npz" --shape "$(shape)" \
        --spacing "$SPACING_ZYX" --out "$W/compare/2-graph-on-our-skeleton.json" > /dev/null
    # 3. Graph half only, on the reference's skeleton and mask.
    compare pair --a ours-graph="$W/cross/ours-on-ref/graph.json" --b ref="$CM/graph.npz" \
        --classes-b "$CM/classify.npz" --shape "$(shape)" \
        --spacing "$SPACING_ZYX" --out "$W/compare/3-graph-on-ref-skeleton.json" > /dev/null
    # 4. Skeletonisation only, on our mask (and the graphs built on each).
    compare pair --a ours="$W/ours/graph.json" \
        --b ref-skeleton="$W/cross/ref-graph-on-ref-skeleton-of-our-mask/graph.npz" \
        --classes-b "$W/cross/ref-graph-on-ref-skeleton-of-our-mask/classify.npz" \
        --skeleton-a "$our_skel" --skeleton-b "$W/cross/ref-skeleton-of-our-mask.npy:xyz" \
        --spacing "$SPACING_ZYX" --out "$W/compare/4-skeletonize-our-mask.json" > /dev/null
    # 4b. The reference's thinning in our axis frame against ours.
    compare skeletons --a "$our_skel" --b "$W/cross/ref-skeleton-of-our-mask-in-our-frame.npy:zyx" \
        --spacing "$SPACING_ZYX" --out "$W/compare/skeleton-in-our-frame.json" > /dev/null
    # 5. Segmentation only: the reference's skeleton and graph half on our mask against on its own.
    compare pair --a ref-on-our-mask="$W/cross/ref-graph-on-ref-skeleton-of-our-mask/graph.npz" \
        --b ref="$CM/graph.npz" \
        --classes-a "$W/cross/ref-graph-on-ref-skeleton-of-our-mask/classify.npz" --classes-b "$CM/classify.npz" \
        --skeleton-a "$W/cross/ref-skeleton-of-our-mask.npy:xyz" --skeleton-b "$ref_skel" \
        --mask-a "$our_mask" --mask-b "$ref_mask" \
        --spacing "$SPACING_ZYX" --out "$W/compare/5-segmentation-only.json" > /dev/null
}

shape() {
    "$PY" -I -c "
import zarr
print(','.join(str(v) for v in zarr.open_array('$W/inputs/src.ome.zarr/0', mode='r').shape[-3:]))"
}

atlas() {
    : "${REGISTRATION:?set REGISTRATION}" "${ATLAS:?set ATLAS}"
    mkdir -p "$W/compare" "$W/atlas-work"
    compare edge-distance --graph ref="$CM/graph.npz" --image "$W/inputs/src.ome.zarr" \
        --registration "$REGISTRATION" --atlas "$ATLAS" --bin "$BIN" --work "$W/atlas-work" \
        --spacing "$SPACING_ZYX" --out "$CM/edge_distance_to_surface.npy"
    cm_graph classify --graph "$CM/graph_reduced.gt" --distance "$CM/edge_distance_to_surface.npy" \
        --out "$CM/classify-atlas.npz"
    compare pair --a ours="$W/ours/graph-atlas.json" --b ref="$CM/graph.npz" \
        --classes-b "$CM/classify-atlas.npz" --distance-b "$CM/edge_distance_to_surface.npy" --shape "$(shape)" \
        --spacing "$SPACING_ZYX" --out "$W/compare/6-full-atlas-classes.json" > /dev/null
    # Our graph half on the reference's skeleton, in atlas space: the
    # classifier with the distance to the surface on identical edges.
    "$BIN/vessel-graph" graph --zarr "$W/cross/ours-on-ref/image.ome.zarr" --artery-channel 1 \
        --all-components --workers "$N" --work "$W/cross/ours-on-ref/work-atlas" --no-tables \
        --registration "$REGISTRATION" --atlas "$ATLAS" --out "$W/cross/ours-on-ref/graph-atlas.json"
    compare pair --a ours-graph="$W/cross/ours-on-ref/graph-atlas.json" --b ref="$CM/graph.npz" \
        --classes-b "$CM/classify-atlas.npz" --distance-b "$CM/edge_distance_to_surface.npy" --shape "$(shape)" \
        --spacing "$SPACING_ZYX" --out "$W/compare/7-graph-on-ref-skeleton-atlas-classes.json" > /dev/null
    compare atlas --a ours="$W/ours/graph-atlas.json" --b ref="$CM/graph.npz" --image "$W/inputs/src.ome.zarr" \
        --registration "$REGISTRATION" --atlas "$ATLAS" --bin "$BIN" --work "$W/atlas-work" \
        --spacing "$SPACING_ZYX" --out "$W/compare/atlas.json" > /dev/null
}

table() {
    compare table "$W"/compare/[0-9]-*.json
}

case "$STEP" in
    cross) cross ;;
    pairs) pairs ;;
    atlas) atlas ;;
    table) table ;;
    all) cross; pairs; [ -n "${REGISTRATION:-}" ] && atlas; table ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
