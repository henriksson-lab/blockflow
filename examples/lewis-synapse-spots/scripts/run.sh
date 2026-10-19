#!/usr/bin/env bash
# Usage: scripts/run.sh OME_ZARR_DIR OUT_DIR [extra lewis-synapse-spots args]
# Builds and runs the example with this crate's own lockfile (which pins the
# workspace's zarrs_filesystem: 0.3.14 fsyncs every write, ~15x slower here).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
zarr="$1"; out="$2"; shift 2
cargo build --release --manifest-path "$here/Cargo.toml"
target="${CARGO_TARGET_DIR:-$here/target}"
"$target/release/blockflow-lewis-synapse-spots" --out "$out" "$@" "$zarr"
