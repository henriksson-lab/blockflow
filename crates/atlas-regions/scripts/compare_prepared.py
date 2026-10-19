# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Compare an atlas written by atlas-prepare with prepare_reference.py's output.

    python compare_prepared.py OURS.ome.zarr 'OUT_DIR/ABA_25um_2017_%s_1_-2_3_z0-256.zyx.npy'
"""
import sys

import numpy as np
import zarr

ours, ref = sys.argv[1], sys.argv[2]
paths = {'annotation': 'labels/annotation/0', 'hemispheres': 'labels/hemispheres/0',
         'reference': '0', 'distance_to_surface': 'distance_to_surface/0'}
for name, path in paths.items():
    a = np.asarray(zarr.open_array(f'{ours}/{path}', mode='r'))
    b = np.load(ref % name)
    if a.dtype.kind == 'f':
        print(name, a.shape, b.shape, 'max |diff|', float(np.abs(a.astype(np.float64) - b).max()))
    else:
        print(name, a.shape, b.shape, 'mismatches',
              int((a.astype(np.int64) != b.astype(np.int64)).sum()))
