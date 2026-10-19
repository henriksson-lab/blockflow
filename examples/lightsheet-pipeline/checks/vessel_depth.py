# SPDX-License-Identifier: MIT
# Original work for this crate.
"""A check of a vessel graph's atlas labels that does not use the atlas:
each vertex's depth below the tissue surface measured in the sample image,
per atlas label.

    python3 vessel_depth.py GRAPH_JSON REGION_TXT FUSED_ZARR ONTOLOGY_JSON [OUT_JSON]

The surface is where the fused image (its level with a voxel of about 25 um)
first rises above the midpoint of its background and tissue levels along z,
per (y, x) column. A vertex's depth is its z (level 0 of the fused frame, the
region's start added) less the surface's there, in micrometres along z.
Labels that are layers of the cortex should come in order of depth, layer 1
at the surface.
"""
import collections
import json
import sys

import numpy as np
import zarr
from scipy import ndimage

graph = json.load(open(sys.argv[1]))
start = [int(r.split(":")[0]) for r in open(sys.argv[2]).read().strip().split(",")]
group = zarr.open_group(sys.argv[3], mode="r")
multiscale = group.attrs["ome"]["multiscales"][0]
onto = json.load(open(sys.argv[4]))
names = {}


def walk(node):
    names[node["id"]] = node["acronym"]
    for child in node.get("children", []):
        walk(child)


for root in onto["msg"]:
    walk(root)


def scale_of(dataset):
    return np.array(next(t["scale"] for t in dataset["coordinateTransformations"] if t["type"] == "scale"))[-3:]


level0 = scale_of(multiscale["datasets"][0])
coarse = next(d for d in multiscale["datasets"] if scale_of(d).max() >= 20)
factor = scale_of(coarse) / level0
image = np.asarray(group[coarse["path"]][...]).astype(np.float32)
image = ndimage.uniform_filter(image.reshape(image.shape[-3:]), 3)
background, tissue = np.percentile(image, [5, 60])
inside = image > (background + tissue) / 2
surface = np.where(inside.any(0), inside.argmax(0), -1)  # coarse z per (y, x)

# The box is fused at level 0 of the tiles, which is finer than the fused
# volume's level 0 when that was fused at a coarser level: read its scale.
box = json.load(open(sys.argv[1].replace("graph.json", "region.ome.zarr/zarr.json")))
box_scale = scale_of(box["attributes"]["ome"]["multiscales"][0]["datasets"][0])
coordinates = np.array(graph["vertex"]["coordinates"]["values"]).reshape(-1, 3)
labels = np.array(graph["vertex"]["annotation"]["values"]).astype(int)
# box voxel -> fused level-0 position (in box voxels) -> coarse index
physical = (coordinates + np.array(start)) * box_scale
cells = np.floor(physical / scale_of(coarse)).astype(int)
cells[:, 1:] = np.clip(cells[:, 1:], 0, np.array(surface.shape) - 1)
surface_z = surface[cells[:, 1], cells[:, 2]] * scale_of(coarse)[0]
depth = physical[:, 0] - surface_z
per = collections.defaultdict(list)
for label, d in zip(labels, depth):
    per[names.get(label, str(label))].append(d)
rows = sorted(
    ([k, len(v), float(np.percentile(v, 10)), float(np.median(v)), float(np.percentile(v, 90))]
     for k, v in per.items() if len(v) >= 20),
    key=lambda r: r[3],
)
out = {
    "vertices": int(len(labels)),
    "above_surface_share": float((depth < -25).mean()),
    "depth_um_by_label": [{"label": r[0], "vertices": r[1], "p10": round(r[2]), "median": round(r[3]), "p90": round(r[4])} for r in rows],
}
print(json.dumps(out, indent=1))
if len(sys.argv) > 5:
    json.dump(out, open(sys.argv[5], "w"), indent=1)
