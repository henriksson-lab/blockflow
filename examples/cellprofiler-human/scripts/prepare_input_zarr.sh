#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  examples/cellprofiler-human/scripts/prepare_input_zarr.sh IMAGE ZARR_STORE

Prepares or validates the rank-3 Zarr input store used by the planned
CellProfiler-style benchmark. The executable array is written at
ZARR_STORE/level0 and can be reused with BF_INPUT_ZARR=ZARR_STORE.

Environment variables:
  CARGO_BIN_FLAGS Extra cargo flags before "--", for example "--release".
  BF_CHUNK_SHAPE  Input chunk shape, default 1x256x256.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

image="$1"
store="$2"

if [[ ! -f "$image" ]]; then
  echo "Input image not found: $image" >&2
  exit 1
fi

chunk_shape="${BF_CHUNK_SHAPE:-1x256x256}"
read -r -a cargo_bin_flags <<< "${CARGO_BIN_FLAGS:-}"

cargo run -p blockflow-cellprofiler-human --bin cellprofiler-plan-probe "${cargo_bin_flags[@]}" -- \
  --input "$image" \
  --ensure-input-zarr "$store" \
  --chunk "$chunk_shape" \
  --out "$store/prepare-plan.json"
