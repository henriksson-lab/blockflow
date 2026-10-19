# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Summarise where a vessel graph's vertices land in the atlas.

    python3 vessel_labels.py GRAPH_JSON ONTOLOGY_JSON [OUT_JSON]

GRAPH_JSON is `vessel-graph graph --registration ... --out` output; ONTOLOGY
the Allen-style JSON (ABA_annotation.json). Prints the share of vertices per
finest region and per major division, the share labelled 0, and per depth
band below the atlas surface (distance_to_surface, atlas voxels) the leading
regions, so the result can be checked for anatomical sense (cortex near the
surface, its layers in order).
"""
import collections
import json
import sys

graph = json.load(open(sys.argv[1]))
onto = json.load(open(sys.argv[2]))
records = {}


def walk(node, parent):
    records[node["id"]] = (node["acronym"], node["name"], parent)
    for child in node.get("children", []):
        walk(child, node["id"])


for root in onto.get("msg", onto if isinstance(onto, list) else [onto]):
    walk(root, None)


def ancestors(label):
    out = []
    while label in records:
        out.append(label)
        label = records[label][2]
    return out


MAJOR = {"Isocortex": 315, "OLF": 698, "HPF": 1089, "CTXsp": 703, "CNU": 623,
         "TH": 549, "HY": 1097, "MB": 313, "P": 771, "MY": 354, "CB": 512,
         "fiber tracts": 1009, "VS": 73}
major_of = {v: k for k, v in MAJOR.items()}

labels = [int(v) for v in graph["vertex"]["annotation"]["values"]]
depth = graph["vertex"]["distance_to_surface"]["values"]
n = len(labels)
finest = collections.Counter(labels)
major = collections.Counter()
for label in labels:
    found = next((major_of[a] for a in ancestors(label) if a in major_of), "0/other")
    major[found] += 1


def name(label):
    return records[label][0] if label in records else ("0 (outside)" if label == 0 else str(label))


bands = collections.defaultdict(collections.Counter)
for label, d in zip(labels, depth):
    bands[min(int(d // 2) * 2, 20)][name(label)] += 1

summary = {
    "vertices": n,
    "label_0_share": finest[0] / n,
    "unknown_labels": sorted({l for l in labels if l and l not in records}),
    "major": {k: round(v / n, 4) for k, v in major.most_common()},
    "finest": [[name(l), round(c / n, 4)] for l, c in finest.most_common(15)],
    "depth_bands_atlas_voxels": {
        f"{b}-{b + 2}": [[k, c] for k, c in bands[b].most_common(4)] for b in sorted(bands)
    },
}
print(json.dumps(summary, indent=1))
if len(sys.argv) > 3:
    json.dump(summary, open(sys.argv[3], "w"), indent=1)
