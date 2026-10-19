# SPDX-License-Identifier: MIT
# Original work for this crate.
"""The reference's atlas preparation (the body of its
Annotation.prepare_annotation_files: transpose, flip, slice), on its own TIFF
reader, saved as [z, y, x] .npy files for compare_prepared.py. Its Annotation
module imports vispy, so the body is repeated here rather than called.

    python prepare_reference.py REFERENCE_ROOT SOURCE_DIR OUT_DIR
"""
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import ClearMap.IO.IO as clearmap_io  # noqa: E402
from ClearMap.Alignment.orientation import orientation_to_transposition  # noqa: E402

source, out = sys.argv[2], sys.argv[3]
slicing = (slice(None), slice(None), slice(0, 256))
orientation = (1, -2, 3)
for name in ['annotation', 'hemispheres', 'reference', 'distance_to_surface']:
    data = np.array(clearmap_io.read(f'{source}/ABA_25um_2017_{name}.tif'))
    data = data.transpose(orientation_to_transposition(orientation))
    slices = [slice(None, None, -1) if o < 0 else slice(None) for o in orientation]
    data = data[tuple(slices)][slicing]
    np.save(f'{out}/ABA_25um_2017_{name}_1_-2_3_z0-256.zyx.npy',
            np.ascontiguousarray(data.transpose(2, 1, 0)))
