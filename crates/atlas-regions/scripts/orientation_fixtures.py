# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Expected outputs of the reference orientation functions on small arrays,
written as JSON in this crate's [z, y, x] order.

The reference holds arrays as (x, y, z): its array is the transpose of ours.
Run with the reference's Python environment:

    python scripts/orientation_fixtures.py REFERENCE_ROOT tests/fixtures/orientation.json
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from ClearMap.Alignment import orientation as ori  # noqa: E402

shape_zyx = (4, 3, 2)
data_zyx = np.arange(np.prod(shape_zyx)).reshape(shape_zyx)
data_xyz = data_zyx.transpose(2, 1, 0)
rng = np.random.default_rng(7)
points_zyx = np.concatenate([rng.uniform(-1, 5, (6, 3)).round(3), [[0, 0, 0], [3, 2, 1]]])
points_xyz = points_zyx[:, ::-1].copy()
resolution_zyx = (5.0, 3.0, 2.0)

cases = []
for orientation in [(1, 2, 3), (-1, 2, 3), (1, -2, 3), (1, 2, -3), (2, -1, 3), (2, 1, 3),
                    (3, 1, -2), (-3, -2, -1), (2, 3, 1), (3, 2, 1), (-1, -2, -3), (1, 3, 2)]:
    case = {"orientation_xyz": list(orientation)}
    for inverse in (False, True):
        tag = "inverse" if inverse else "forward"
        # The array the forward case starts from is the original data; the
        # inverse case starts from the oriented one, as a round trip would.
        source = data_xyz if not inverse else ori.orient(data_xyz, orientation)
        out = np.asarray(ori.orient(source, orientation, inverse=inverse))
        out_zyx = out.transpose(2, 1, 0)
        shape_in_xyz = data_xyz.shape
        pts = ori.orient_points(points_xyz, orientation, shape=shape_in_xyz, inverse=inverse)
        case[tag] = {
            "input_shape_zyx": list(source.transpose(2, 1, 0).shape),
            "input_zyx": np.ascontiguousarray(source.transpose(2, 1, 0)).ravel().tolist(),
            "output_shape_zyx": list(out_zyx.shape),
            "output_zyx": np.ascontiguousarray(out_zyx).ravel().tolist(),
            # Points are oriented with the shape of the *unoriented* data, as the
            # reference documents, in both directions.
            "points_zyx": np.asarray(pts)[:, ::-1].tolist(),
            "shape_zyx": list(ori.orient_shape(shape_in_xyz, orientation, inverse=inverse))[::-1],
            "resolution_zyx": list(ori.orient_resolution(resolution_zyx[::-1], orientation,
                                                         inverse=inverse))[::-1],
        }
    cases.append(case)

with open(sys.argv[2], "w") as out:
    json.dump({"data_shape_zyx": list(shape_zyx), "points_zyx": points_zyx.tolist(),
               "resolution_zyx": list(resolution_zyx), "cases": cases}, out, indent=1)
