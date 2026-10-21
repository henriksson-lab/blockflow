#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# blockflow side of the vessel workloads: segment both channels (vessel,
# artery) of an OME-Zarr copy, then build the vessel graph over its labels.
#
#   vessels.sh BIN_DIR ZARR OUT_DIR WORKERS [WEIGHTS]
#
# ZARR is a fresh copy (segment writes its labels into it). With WEIGHTS (a
# safetensors file) the deep filling runs, otherwise --no-deep-fill. Writes
# OUT_DIR/{graph.json,stages.json}; the scratch directories are removed.
set -euo pipefail
BIN="$1"; ZARR="$2"; OUT="$3"; WORKERS="$4"; WEIGHTS="${5:-}"
mkdir -p "$OUT"
fill=(--no-deep-fill)
[ -n "$WEIGHTS" ] && fill=(--deep-fill-weights "$WEIGHTS")
t0=$(date +%s.%N)
"$BIN/vessel-graph" segment --zarr "$ZARR" --channel 0,1 --role vessel,artery \
    --workers "$WORKERS" --work "$OUT/work-segment" "${fill[@]}"
t1=$(date +%s.%N)
# Every component, as the reference's clean + reduce keeps them.
"$BIN/vessel-graph" graph --zarr "$ZARR" --artery-channel 1 --all-components \
    --workers "$WORKERS" --work "$OUT/work-graph" --out "$OUT/graph.json"
t2=$(date +%s.%N)
printf '{"steps_s": {"segment": %s, "graph": %s}, "deep_fill": %s}\n' \
    "$(awk "BEGIN {print $t1 - $t0}")" "$(awk "BEGIN {print $t2 - $t1}")" \
    "$([ -n "$WEIGHTS" ] && echo true || echo false)" > "$OUT/timing.json"
