#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# One light-sheet sample, end to end in atlas space:
#   stitch the signal tiles -> register fused signal -> autofluorescence -> atlas
#   -> cells and/or vessels on level-0 boxes of the fused volume, in atlas space.
#
#   examples/lightsheet-pipeline/run.sh SAMPLE_DIR PARAMS [STAGE...]
#
# SAMPLE_DIR holds the converted inputs (OME-Zarr only): a directory of signal
# tiles and the autofluorescence image, named in PARAMS. PARAMS is a bash file
# of KEY=value lines (see params/*.env and the README). STAGEs are any of
#   align fuse register cells vessels
# (default: all, with cells/vessels as PARAMS enables them). A stage whose
# output exists is skipped unless FORCE=1; outputs go to $OUT_ROOT/$SAMPLE_ID
# in the fixed layout the README documents.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[ $# -ge 2 ] || { sed -n '5,17p' "$0"; exit 2; }
SAMPLE="$(cd "$1" && pwd)"
PARAMS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
shift 2

# ---- parameters (defaults; PARAMS overrides) -------------------------------
SAMPLE_ID="$(basename "$SAMPLE")"
OUT_ROOT=/husky/henriksson/lightsheet-reference/pipeline
BIN="$ROOT/target/release"
BUILD=0
PY="${PY:-python3}"
SIGNAL_TILES=""                     # directory of tiles, relative to SAMPLE_DIR
SIGNAL_CHANNEL=0
AUTOFLUORESCENCE=""                 # relative to SAMPLE_DIR
ATLAS=""                            # prepared atlas (atlas-prepare), sample orientation
ALIGNMENT_DIR=/data/henriksson/github/claude/clearmap-rs/ClearMap-py/ClearMap/Resources/Alignment
# stitching
STITCH_LEVEL=1                      # level the alignment runs on
STITCH_OVERLAP=""                   # Y,X level-0 voxels; empty: the tiles' NGFF translations
STITCH_ROWS=""                      # START:STOP grid rows, empty: all
STITCH_COLUMNS=""
STITCH_MAX_SHIFT=8,15               # per slice, Y,X, voxels of STITCH_LEVEL
STITCH_RIGID_MAX_SHIFT=15,15
STITCH_MAX_SHIFT_Z=20
STITCH_RIGID_ONLY=0
STITCH_ARGS=""
FUSE_LEVEL=2                        # level of the whole fused volume (registration input)
# registration
REGISTRATION_RESOLUTION=25
SIGNAL_TO_AUTO="align_affine.txt"
AUTO_TO_ATLAS="align_affine.txt align_bspline.txt"
SIGNAL_TO_AUTO_INITIAL=""           # how signal -> auto starts: empty (the file's: centres),
                                    #   physical, centre, or Z,Y,X (an initial translation, um)
REGISTRATION_ARGS=""
# cells
CELLS=0
CELL_REGION=""                      # Z0:Z1,Y0:Y1,X0:X1 of the fused volume at level 0, or
CELL_TILE=""                        #   a tile name (RRxCC) and
CELL_TILE_BOX=""                    #   a box in that tile's own level-0 voxels
CELL_ARGS=""
# vessels
VESSELS=0
VESSEL_CHANNELS="0"                 # tile channels to fuse; the first is the vessel channel
VESSEL_ROLES="vessel"
VESSEL_REGION=""
VESSEL_TILE=""
VESSEL_TILE_BOX=""
VESSEL_SEGMENT_ARGS=""
VESSEL_GRAPH_ARGS=""
# shellcheck disable=SC1090
source "$PARAMS"

OUT="$OUT_ROOT/$SAMPLE_ID"
STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(align fuse register cells vessels)
want() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
need() { [ -n "${!1}" ] || { echo "run.sh: $1 is not set in $PARAMS" >&2; exit 2; }; }
need SIGNAL_TILES; need AUTOFLUORESCENCE; need ATLAS
mkdir -p "$OUT"/{stitch,registration,cells,vessels,logs}
cp "$PARAMS" "$OUT/params.env"

if [ "$BUILD" = 1 ]; then
    (cd "$ROOT" && for p in blockflow-tile-stitching blockflow-atlas-registration \
        blockflow-cell-density blockflow-vessel-graph; do
        cargo build --release -p "$p"; done)
fi

# Run one stage under /usr/bin/time; record its wall time, peak memory and the
# disk use of its output directory in logs/<name>.json.
stage() {
    local name="$1" dir="$2"; shift 2
    echo "== $name: $*" | tee "$OUT/logs/$name.log"
    local started; started=$(date -Is)
    /usr/bin/time -v -o "$OUT/logs/$name.time" "$@" >> "$OUT/logs/$name.log" 2>&1 || {
        echo "run.sh: stage $name failed, see $OUT/logs/$name.log" >&2; tail -5 "$OUT/logs/$name.log" >&2; exit 1; }
    "$PY" "$HERE/pipeline.py" stage "$OUT/logs/$name.time" "$dir" \
        | jq --arg started "$started" --arg command "$*" '. + {started: $started, command: $command}' \
        > "$OUT/logs/$name.json"
    echo "   $(jq -c '{wall_s, peak_rss_gb, disk_gb}' "$OUT/logs/$name.json")"
}
done_or_force() { [ "${FORCE:-0}" != 1 ] && [ -e "$1" ]; }

POSITIONS="$OUT/stitch/align.positions.json"
FUSED="$OUT/stitch/fused.ome.zarr"
CHAIN="$OUT/registration"

# ---- 1. align: tile positions at STITCH_LEVEL ---------------------------------
if want align && ! done_or_force "$POSITIONS"; then
    args=(--tile-dir "$SAMPLE/$SIGNAL_TILES" --channel "$SIGNAL_CHANNEL" --level "$STITCH_LEVEL"
          --max-shift "$STITCH_MAX_SHIFT" --rigid-max-shift "$STITCH_RIGID_MAX_SHIFT"
          --max-shift-z "$STITCH_MAX_SHIFT_Z" --positions-only --out "$OUT/stitch/align")
    [ -n "$STITCH_OVERLAP" ] && args+=(--overlap "$STITCH_OVERLAP")
    [ -n "$STITCH_ROWS" ] && args+=(--rows "$STITCH_ROWS")
    [ -n "$STITCH_COLUMNS" ] && args+=(--columns "$STITCH_COLUMNS")
    [ "$STITCH_RIGID_ONLY" = 1 ] && args+=(--rigid-only)
    # shellcheck disable=SC2206
    args+=($STITCH_ARGS)
    stage align "$OUT/stitch" "$BIN/tile-stitching" "${args[@]}"
fi

# ---- 2. fuse: the whole sample at FUSE_LEVEL (the registration's signal) ------
if want fuse && ! done_or_force "$FUSED"; then
    rm -rf "$FUSED"
    stage fuse "$FUSED" "$BIN/tile-stitching" --layout-from "$POSITIONS" \
        --channel "$SIGNAL_CHANNEL" --level "$FUSE_LEVEL" --out "$FUSED"
fi

# ---- 3. register: fused signal -> autofluorescence -> atlas -------------------
if want register && ! done_or_force "$CHAIN/chain.json"; then
    rm -rf "$CHAIN"
    params=()
    for f in $SIGNAL_TO_AUTO; do params+=(--parameters "0=$ALIGNMENT_DIR/$f"); done
    for f in $AUTO_TO_ATLAS; do params+=(--parameters "1=$ALIGNMENT_DIR/$f"); done
    case "$SIGNAL_TO_AUTO_INITIAL" in
        "") ;;
        parameters|physical|centre|center) params+=(--initialise "0=$SIGNAL_TO_AUTO_INITIAL") ;;
        *) params+=(--initial-translation "0=$SIGNAL_TO_AUTO_INITIAL") ;;
    esac
    # shellcheck disable=SC2086
    stage register "$CHAIN" "$BIN/atlas-registration" chain \
        --image "$FUSED" --image "$SAMPLE/$AUTOFLUORESCENCE" --image "$ATLAS" \
        --resolution "$REGISTRATION_RESOLUTION" "${params[@]}" --out "$CHAIN" $REGISTRATION_ARGS
fi

# A level-0 box of the fused volume: REGION as given, or TILE's own BOX.
box_of() {
    local region="$1" tile="$2" box="$3"
    if [ -n "$region" ]; then echo "$region"; return; fi
    [ -n "$tile" ] && [ -n "$box" ] || { echo "run.sh: give a region or a tile and box" >&2; exit 2; }
    "$PY" "$HERE/pipeline.py" region "$POSITIONS" --tile "$tile" --box "$box"
}

# ---- 4. cells: a level-0 box, detected and mapped into the atlas -------------
if want cells && [ "$CELLS" = 1 ] && ! done_or_force "$OUT/cells/report.json"; then
    region=$(box_of "$CELL_REGION" "$CELL_TILE" "$CELL_TILE_BOX")
    echo "$region" > "$OUT/cells/region.txt"
    rm -rf "$OUT/cells/region.ome.zarr" "$OUT/cells/detect" "$OUT/cells/density.ome.zarr"
    stage cells-fuse "$OUT/cells/region.ome.zarr" "$BIN/tile-stitching" --layout-from "$POSITIONS" \
        --channel "$SIGNAL_CHANNEL" --level 0 --region "$region" --out "$OUT/cells/region.ome.zarr"
    # shellcheck disable=SC2086
    stage cells "$OUT/cells" "$BIN/cell-density" --input "$OUT/cells/region.ome.zarr" \
        --output "$OUT/cells/detect" --registration "$CHAIN" --atlas "$ATLAS" \
        --density "$OUT/cells/density.ome.zarr" $CELL_ARGS
    # cell-density's JSON report is the last stdout object in its log.
    sed -n '/^{/,/^}/p' "$OUT/logs/cells.log" > "$OUT/cells/report.json"
fi

# ---- 5. vessels: a level-0 box, segmented, graphed and mapped ----------------
if want vessels && [ "$VESSELS" = 1 ] && ! done_or_force "$OUT/vessels/graph.json"; then
    region=$(box_of "$VESSEL_REGION" "$VESSEL_TILE" "$VESSEL_TILE_BOX")
    echo "$region" > "$OUT/vessels/region.txt"
    rm -rf "$OUT/vessels"/channel-*.ome.zarr "$OUT/vessels/region.ome.zarr"
    channels=()
    for c in ${VESSEL_CHANNELS//,/ }; do
        stage "vessels-fuse-$c" "$OUT/vessels/channel-$c.ome.zarr" "$BIN/tile-stitching" \
            --layout-from "$POSITIONS" --channel "$c" --level 0 --region "$region" \
            --out "$OUT/vessels/channel-$c.ome.zarr" --out-levels 1
        channels+=("$OUT/vessels/channel-$c.ome.zarr")
    done
    stage vessels-stack "$OUT/vessels/region.ome.zarr" "$PY" "$HERE/stack_channels.py" \
        "$OUT/vessels/region.ome.zarr" "${channels[@]}"
    n=${#channels[@]}
    channel_list=$(seq -s, 0 $((n - 1)))
    # shellcheck disable=SC2086
    stage vessels-segment "$OUT/vessels/region.ome.zarr" "$BIN/vessel-graph" segment \
        --zarr "$OUT/vessels/region.ome.zarr" --channel "$channel_list" --role "$VESSEL_ROLES" \
        --work "$OUT/vessels/work-segment" $VESSEL_SEGMENT_ARGS
    artery=()
    [ "$n" -gt 1 ] && artery=(--artery-channel 1)
    # shellcheck disable=SC2086
    stage vessels-graph "$OUT/vessels" "$BIN/vessel-graph" graph --zarr "$OUT/vessels/region.ome.zarr" \
        "${artery[@]}" --registration "$CHAIN" --atlas "$ATLAS" \
        --branch-density "$OUT/vessels/branch-density.ome.zarr" --out "$OUT/vessels/graph.json" \
        $VESSEL_GRAPH_ARGS
fi

# ---- manifest -----------------------------------------------------------------
commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git -C "$ROOT" status --porcelain -- examples/tile-stitching examples/atlas-registration \
    examples/cell-density examples/vessel-graph src 2>/dev/null | wc -l)
jq -n --arg sample "$SAMPLE" --arg id "$SAMPLE_ID" --arg commit "$commit" --argjson dirty "$dirty" \
    --arg finished "$(date -Is)" \
    --argjson stages "$(for f in "$OUT"/logs/*.json; do [ -e "$f" ] && jq -c --arg n "$(basename "$f" .json)" '{($n): .}' "$f"; done | jq -s 'add // {}')" \
    --argjson binaries "$(for b in tile-stitching atlas-registration cell-density vessel-graph; do
        [ -e "$BIN/$b" ] && jq -n --arg b "$b" --arg t "$(date -r "$BIN/$b" -Is)" '{($b): $t}'; done | jq -s 'add // {}')" \
    '{schema: 1, sample: $sample, sample_id: $id, layout: "examples/lightsheet-pipeline/README.md",
      blockflow_commit: $commit, uncommitted_paths: $dirty, binaries_built: $binaries,
      finished: $finished, stages: $stages}' > "$OUT/manifest.json"
echo "manifest: $OUT/manifest.json"
