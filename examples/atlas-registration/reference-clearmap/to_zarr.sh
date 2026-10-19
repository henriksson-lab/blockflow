#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Original work for this crate.
#
# The prepared inputs as OME-Zarr, through OME-TIFF (which carries the 25 µm
# voxel size img2omezarr reads) and img2omezarr. Labels are downsampled by
# nearest neighbour; only level 0 is used by the comparison.
set -euo pipefail
REG_OUT=${REG_OUT:-/husky/henriksson/lightsheet-reference/registration}
IMG2OMEZARR=${IMG2OMEZARR:-/home/mahogny/github/claude/img2omezarr/target/release/img2omezarr}
PY=${PY:-/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python}
mkdir -p "$REG_OUT/tif25" "$REG_OUT/zarr"
"$PY" -I - "$REG_OUT" <<'PYEOF'
import os
import sys

import tifffile

out = sys.argv[1]
meta = {'axes': 'ZYX', 'PhysicalSizeX': 25.0, 'PhysicalSizeY': 25.0, 'PhysicalSizeZ': 25.0,
        'PhysicalSizeXUnit': 'µm', 'PhysicalSizeYUnit': 'µm', 'PhysicalSizeZUnit': 'µm'}
pairs = [('clearmap/resampled_autofluorescence.tif', 'autofluorescence_25um'),
         ('atlas/reference.tif', 'reference_25um'),
         ('atlas/annotation.tif', 'annotation_25um'),
         ('atlas/hemispheres.tif', 'hemispheres_25um')]
if os.path.exists(f'{out}/clearmap/resampled_signal.tif'):
    pairs.append(('clearmap/resampled_signal.tif', 'signal_25um'))
for source, name in pairs:
    target = f'{out}/tif25/{name}.ome.tif'
    if not os.path.exists(target):
        tifffile.imwrite(target, tifffile.imread(f'{out}/{source}'), ome=True, metadata=meta)
    print(target, flush=True)
PYEOF
for name in autofluorescence_25um reference_25um signal_25um; do
  [ -f "$REG_OUT/tif25/$name.ome.tif" ] || continue
  "$IMG2OMEZARR" convert "$REG_OUT/tif25/$name.ome.tif" "$REG_OUT/zarr/$name.ome.zarr" --overwrite > /dev/null
done
for name in annotation_25um hemispheres_25um; do
  "$IMG2OMEZARR" convert "$REG_OUT/tif25/$name.ome.tif" "$REG_OUT/zarr/$name.ome.zarr" \
    --overwrite --downsampling nearest > /dev/null
done
ls "$REG_OUT/zarr"
