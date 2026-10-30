#!/usr/bin/env bash
# ROZT (newvolim) on the C. elegans 4-D OME-Zarrs, so imported lineages can be viewed.
# Port and roots can be overridden: PORT=9877 ZARR_ROOT=... NEWVOLIM=... serve_rozt.sh
set -euo pipefail
port=${PORT:-9877}
zarr_root=${ZARR_ROOT:-/husky/otherdataset/teresa/4d_elegans/zarr}
newvolim=${NEWVOLIM:-$HOME/github/claude/newvolim}
state=${STATE_DIR:-/tmp/rozt-elegans-state}
cd "$newvolim"
exec target/release/newvolim-server --bind "0.0.0.0:$port" \
  --allow-root "$zarr_root" \
  --catalog-source "elegans=$zarr_root" \
  --page-dir crates/newvolim-ui/dist \
  --state-dir "$state" --job-dir "$state/jobs"
