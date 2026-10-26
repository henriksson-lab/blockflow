#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Graph-level comparison of two vessel graphs: `vessel-graph`'s against the
reference's (or either against itself on another skeleton).

A graph is a `.json` that `vessel-graph graph --out` wrote, or a `.npz` that
`clearmap_graph.py` wrote. Coordinates are level-0 voxels `[z, y, x]`;
distances are in µm (`--spacing`, z,y,x). Needs numpy, scipy and zarr (any
Python but the reference's own).

    compare_graph.py pair --a ours=graph.json --b ref=graph.npz \
        [--skeleton-a IMAGE.ome.zarr#vessel-skeleton] [--skeleton-b skeleton.npy:xyz] \
        [--mask-a ...] [--mask-b ...] [--classes-b classify.npz] --out pair.json
    compare_graph.py atlas --a ... --b ... --image IMAGE.ome.zarr --registration DIR \
        --atlas ATLAS.ome.zarr --bin BIN_DIR --work DIR --out atlas.json
    compare_graph.py edge-distance --graph graph.npz --image ... --registration ... \
        --atlas ... --bin ... --work DIR --out distance.npy
    compare_graph.py skeletons --a IMAGE.ome.zarr#vessel-skeleton --b skeleton.npy:xyz --out s.json
    compare_graph.py import-labels --zarr IMAGE.ome.zarr vessel-skeleton=skeleton.npy:xyz ...
    compare_graph.py table pair1.json pair2.json ...

`pair` reports, per graph, the graph-level statistics (vertices, edges,
degree histogram, total length and length density, branch and end points
and their densities, components, self-loops, multi-edges, polyline points),
and between them:

- **skeleton** (with `--skeleton-*`): voxels in common, and one-to-one
  matched (greedy, nearest first) within 3 µm (a 26-neighbour) and 5 µm;
- **vertices**: branch points (degree >= 3) and end points (degree 1)
  matched one-to-one within `--vertex-tol` (5 µm) and twice that, with
  precision, recall and, for matched branch points, the degree confusion;
- **edges**: an edge of `a` matches one of `b` when both its ends matched
  vertices that `b`'s edge joins (all reduced vertices, matched within
  `--vertex-tol`); for matched edges, the polylines' symmetric mean distance
  and Hausdorff distance, length and tortuosity (length over chord) side by
  side, the edge radius (`radius_units`, the maximum along the edge) and the
  per-point radius at the nearest point of the other polyline; and the
  artery mask vote, artery and vein classes on matched edges (with
  `--classes-*` for a reference graph classified by `clearmap_graph.py`);
- **coverage**: the share of each graph's polyline length within 3 µm of the
  other's polylines.

`atlas` carries both graphs and a grid over the image (every 8th voxel)
through the registration with `atlas-registration transform-points`, labels
them in the prepared atlas at the nearest voxel, and reports per region (the
top ones by volume in the image) and per depth (distance to the surface, in
atlas voxels, inside the brain) the length density and branch density of
each graph. Both
sides go through the same registration, so only the graphs differ.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SKELETON_TOLS_UM = (3.0, 5.0)
# A cortical layer's acronym: `SSs2/3`, `SSp-n1`, `MOp6a`, ...
LAYER = re.compile(r".*[A-Za-z-](1|2/3|4|5|6a|6b)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(spec):
    """`name=path` (or a bare path) to a graph dict with `[z, y, x]` arrays."""
    name, _, path = spec.rpartition("=")
    path = Path(path)
    name = name or path.parent.name
    if path.suffix == ".npz":
        data = dict(np.load(path))
        data["source"] = "reference"
    else:
        data = ours_from_json(path)
        data["source"] = "vessel-graph"
    data["name"] = name
    data["path"] = str(path)
    return data


def ours_from_json(path):
    graph = json.loads(Path(path).read_text())

    def column(level, name, width=1):
        entry = graph[level].get(name)
        if entry is None:
            return None
        values = np.asarray(entry["values"] if isinstance(entry, dict) else entry, dtype=float)
        return values.reshape(-1, width) if width > 1 else values

    polylines = np.asarray(graph["polylines"], dtype=np.int64).reshape(-1, 2)  # [start, end) points
    data = {"vertex_zyx": column("vertex", "coordinates", 3),
            "edge_vertices": np.asarray(graph["edges"], dtype=np.int64).reshape(-1, 2),
            "vertex_radius_vx": column("vertex", "radii"), "vertex_radius_um": column("vertex", "radius_units"),
            "edge_length_um": column("edge", "length_units"), "edge_radius_vx": column("edge", "radii"),
            "edge_radius_um": column("edge", "radius_units"),
            "edge_artery_binary": column("edge", "artery_binary"), "edge_artery_raw": column("edge", "artery_raw"),
            "edge_artery": column("edge", "artery"), "edge_vein": column("edge", "vein"),
            "point_zyx": column("along", "coordinates", 3), "point_radius_um": column("along", "radius_units"),
            "point_annotation": column("along", "annotation"),
            "edge_distance_to_surface": column("edge", "distance_to_surface"),
            "edge_point_start": polylines[:, 0], "edge_point_count": polylines[:, 1] - polylines[:, 0],
            "volume": graph.get("volume")}
    for level, key in (("vertex", "vertex_atlas_zyx"), ("along", "point_atlas_zyx")):
        atlas = column(level, "coordinates_atlas", 3)
        if atlas is not None:
            data[key] = atlas
    return {k: v for k, v in data.items() if v is not None}


def load_volume(spec):
    """`IMAGE.ome.zarr#label` (level 0) or `path.npy[:xyz|:zyx]` as a `[z, y, x]` bool array."""
    if "#" in spec:
        import zarr
        image, label = spec.split("#", 1)
        array = zarr.open_array(str(Path(image) / "labels" / label / "0"), mode="r")
        return np.asarray(array[...]) != 0
    path, _, layout = spec.partition(":")
    array = np.load(path, mmap_mode="r")
    array = np.asarray(array) != 0
    return np.ascontiguousarray(array.transpose(2, 1, 0)) if layout == "xyz" else array


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def degrees(graph):
    edges = graph["edge_vertices"]
    n = len(graph["vertex_zyx"])
    return np.bincount(edges.ravel(), minlength=n)  # a self-loop counts twice


def polylines(graph):
    start, count = graph["edge_point_start"], graph["edge_point_count"]
    points = graph["point_zyx"]
    return [points[s:s + c] for s, c in zip(start, count)]


def polyline_lengths(graph, spacing):
    points = graph["point_zyx"] * spacing
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    start, count = graph["edge_point_start"], graph["edge_point_count"]
    end = start + count - 1
    lengths = cumulative[end] - cumulative[start]
    return lengths


def stats(graph, spacing, volume_shape):
    deg = degrees(graph)
    edges = graph["edge_vertices"]
    n_v, n_e = len(deg), len(edges)
    length = graph["edge_length_um"]
    volume_mm3 = float(np.prod(volume_shape) * np.prod(spacing)) * 1e-9
    adjacency = coo_matrix((np.ones(n_e), (edges[:, 0], edges[:, 1])), shape=(n_v, n_v))
    n_comp, label = connected_components(adjacency, directed=False)
    comp_length = np.bincount(label[edges[:, 0]], weights=length, minlength=n_comp)
    comp_vertices = np.bincount(label, minlength=n_comp)
    pairs = np.sort(edges, axis=1)
    unique_pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    hist = np.bincount(np.minimum(deg, 7), minlength=8)
    chord = np.linalg.norm((graph["vertex_zyx"][edges[:, 1]] - graph["vertex_zyx"][edges[:, 0]]) * spacing, axis=1)
    ok = chord > 0
    tort = length[ok] / chord[ok]
    total = float(length.sum())
    record = {
        "vertices": int(n_v), "edges": int(n_e),
        "degree_histogram": {("7+" if d == 7 else str(d)): int(c) for d, c in enumerate(hist)},
        "branch_points": int((deg >= 3).sum()), "end_points": int((deg == 1).sum()),
        "total_length_mm": round(total * 1e-3, 3),
        "volume_mm3": round(volume_mm3, 5),
        "length_density_mm_per_mm3": round(total * 1e-3 / volume_mm3, 1),
        "branch_density_per_mm3": round(float((deg >= 3).sum()) / volume_mm3, 0),
        "end_density_per_mm3": round(float((deg == 1).sum()) / volume_mm3, 0),
        "components": int(n_comp),
        "largest_component_length_share": round(float(comp_length.max() / total), 4) if total else None,
        "largest_component_vertex_share": round(float(comp_vertices.max() / n_v), 4) if n_v else None,
        "single_edge_components": int(np.sum(np.bincount(label[edges[:, 0]], minlength=n_comp) == 1)),
        "self_loops": int(np.sum(edges[:, 0] == edges[:, 1])),
        "multi_edges": int(len(pairs[pairs[:, 0] != pairs[:, 1]]) - len(unique_pairs)),
        "polyline_points": int(len(graph["point_zyx"])),
        "polyline_unique_voxels": int(len(np.unique(graph["point_zyx"].astype(np.int64), axis=0))),
        "edge_length_um": quantiles(length),
        "tortuosity": quantiles(tort),
        "polyline_length_matches_edge_length": bool(np.allclose(polyline_lengths(graph, spacing), length,
                                                                rtol=1e-6, atol=1e-6)),
    }
    for key, name in (("edge_radius_um", "edge_radius_um"), ("vertex_radius_um", "vertex_radius_um")):
        if key in graph:
            values = graph[key]
            record[name] = quantiles(values[values >= 0])
            record[name + "_not_found"] = int(np.sum(values < 0))
    for key in ("edge_artery_binary", "edge_artery", "edge_vein"):
        if key in graph:
            flag = graph[key] > 0
            record[key.replace("edge_", "") + "_edges"] = int(flag.sum())
            record[key.replace("edge_", "") + "_length_share"] = round(float(length[flag].sum() / total), 4) if total else None
    return record


def quantiles(values, qs=(5, 25, 50, 75, 95)):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    out = {f"p{q}": round(float(np.percentile(values, q)), 3) for q in qs}
    out["mean"] = round(float(values.mean()), 3)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def greedy_match(a, b, tol):
    """One-to-one matching of point sets `a` and `b` (µm), nearest pairs
    first, within `tol`: (index in a, index in b, distance)."""
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    # Coincident points first (each set holds a point once), then the rest
    # nearest first.
    d0, j0 = cKDTree(b).query(a, distance_upper_bound=1e-9)
    exact = np.isfinite(d0)
    ia0, jb0 = np.nonzero(exact)[0], j0[exact]
    rest_a = np.nonzero(~exact)[0]
    free_b = np.ones(len(b), bool)
    free_b[jb0] = False
    rest_b = np.nonzero(free_b)[0]
    ia, jb, dist = [ia0], [jb0], [np.zeros(len(ia0))]
    if len(rest_a) and len(rest_b):
        pairs = cKDTree(a[rest_a]).sparse_distance_matrix(cKDTree(b[rest_b]), tol, output_type="ndarray")
        order = np.lexsort((pairs["j"], pairs["i"], pairs["v"]))
        used_a, used_b = set(), set()
        pi, pj, pv = [], [], []
        for i, j, v in zip(pairs["i"][order].tolist(), pairs["j"][order].tolist(), pairs["v"][order].tolist()):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pi.append(i)
            pj.append(j)
            pv.append(v)
        ia.append(rest_a[np.asarray(pi, int)])
        jb.append(rest_b[np.asarray(pj, int)])
        dist.append(np.asarray(pv, float))
    return np.concatenate(ia).astype(int), np.concatenate(jb).astype(int), np.concatenate(dist)


def prf(n_matched, n_a, n_b):
    """`a` is the graph under test, `b` the reference: precision is the
    share of `a` matched, recall the share of `b`."""
    precision = n_matched / n_a if n_a else 1.0
    recall = n_matched / n_b if n_b else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"matched": int(n_matched), "a": int(n_a), "b": int(n_b), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def compare_skeletons(sa, sb, spacing):
    pa = np.argwhere(sa) * spacing
    pb = np.argwhere(sb) * spacing
    record = {"exact": prf(int(np.logical_and(sa, sb).sum()), int(sa.sum()), int(sb.sum()))}
    for tol in SKELETON_TOLS_UM:
        ia, _, _ = greedy_match(pa, pb, tol)
        record[f"within_{tol:g}um"] = prf(len(ia), len(pa), len(pb))
    return record


def compare_vertices(ga, gb, spacing, tol):
    da, db = degrees(ga), degrees(gb)
    va, vb = ga["vertex_zyx"] * spacing, gb["vertex_zyx"] * spacing
    record = {}
    for name, sel_a, sel_b in (("branch_points", da >= 3, db >= 3), ("end_points", da == 1, db == 1)):
        ia_idx, ib_idx = np.nonzero(sel_a)[0], np.nonzero(sel_b)[0]
        for t in (tol, 2 * tol):
            ia, jb, dist = greedy_match(va[ia_idx], vb[ib_idx], t)
            entry = prf(len(ia), len(ia_idx), len(ib_idx))
            entry["median_distance_um"] = round(float(np.median(dist)), 3) if len(dist) else None
            if name == "branch_points":
                deg_a, deg_b = da[ia_idx[ia]], db[ib_idx[jb]]
                entry["degree_equal"] = round(float(np.mean(deg_a == deg_b)), 4) if len(ia) else None
                confusion = {}
                for x, y in zip(np.minimum(deg_a, 6), np.minimum(deg_b, 6)):
                    key = f"{x if x < 6 else '6+'}->{y if y < 6 else '6+'}"
                    confusion[key] = confusion.get(key, 0) + 1
                entry["degree_confusion_a_to_b"] = dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:12])
            record[f"{name}_within_{t:g}um"] = entry
    return record


def path_distances(pa, pb):
    """Symmetric mean and Hausdorff distance between two polylines (µm)."""
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2))
    ab, ba = d.min(axis=1), d.min(axis=0)
    return 0.5 * (ab.mean() + ba.mean()), max(ab.max(), ba.max()), d.argmin(axis=1)


def compare_edges(ga, gb, spacing, tol):
    va, vb = ga["vertex_zyx"] * spacing, gb["vertex_zyx"] * spacing
    ia, jb, _ = greedy_match(va, vb, tol)
    a_to_b = np.full(len(va), -1)
    a_to_b[ia] = jb
    lookup = {}
    for e, (u, v) in enumerate(gb["edge_vertices"]):
        lookup.setdefault((min(u, v), max(u, v)), []).append(e)
    lines_a, lines_b = polylines(ga), polylines(gb)
    used_b = np.zeros(len(gb["edge_vertices"]), bool)
    matches = []   # (edge a, edge b, mean, hausdorff, nearest index per point of a)
    for e, (u, v) in enumerate(ga["edge_vertices"]):
        mu, mv = a_to_b[u], a_to_b[v]
        if mu < 0 or mv < 0:
            continue
        candidates = [c for c in lookup.get((min(mu, mv), max(mu, mv)), []) if not used_b[c]]
        if not candidates:
            continue
        pa = lines_a[e] * spacing
        best = min(((c,) + path_distances(pa, lines_b[c] * spacing) for c in candidates), key=lambda t: t[1])
        used_b[best[0]] = True
        matches.append((e, best[0], best[1], best[2], best[3]))
    n_a, n_b = len(ga["edge_vertices"]), len(gb["edge_vertices"])
    record = {"matched_by_endpoints": prf(len(matches), n_a, n_b)}
    if not matches:
        return record, matches
    ea = np.array([m[0] for m in matches])
    eb = np.array([m[1] for m in matches])
    mean_d = np.array([m[2] for m in matches])
    haus = np.array([m[3] for m in matches])
    la, lb = ga["edge_length_um"][ea], gb["edge_length_um"][eb]
    length_a, length_b = ga["edge_length_um"], gb["edge_length_um"]
    record["matched_length_share"] = {"a": round(float(la.sum() / length_a.sum()), 4),
                                      "b": round(float(lb.sum() / length_b.sum()), 4)}
    record["path_mean_distance_um"] = quantiles(mean_d)
    record["path_hausdorff_um"] = quantiles(haus)
    record["path_within_1_voxel"] = round(float(np.mean(haus <= 1.01 * float(np.max(spacing)) * np.sqrt(3))), 4)
    record["length_um_matched"] = {"a": quantiles(la), "b": quantiles(lb),
                                   "a_minus_b": quantiles(la - lb),
                                   "relative_within_10pc": round(float(np.mean(np.abs(la - lb) <= 0.1 * lb)), 4),
                                   "pearson": round(float(np.corrcoef(la, lb)[0, 1]), 4)}

    def tortuosity(g, edges):
        ends = g["vertex_zyx"][g["edge_vertices"][edges]]
        chord = np.linalg.norm((ends[:, 1] - ends[:, 0]) * spacing, axis=1)
        return g["edge_length_um"][edges] / np.where(chord > 0, chord, np.nan)
    ta, tb = tortuosity(ga, ea), tortuosity(gb, eb)
    ok = np.isfinite(ta) & np.isfinite(tb)
    record["tortuosity_matched"] = {"a": quantiles(ta[ok]), "b": quantiles(tb[ok]),
                                    "pearson": round(float(np.corrcoef(ta[ok], tb[ok])[0, 1]), 4)}
    if "edge_radius_um" in ga and "edge_radius_um" in gb:
        ra, rb = ga["edge_radius_um"][ea], gb["edge_radius_um"][eb]
        ok = (ra >= 0) & (rb >= 0)
        record["edge_radius_um_matched"] = {
            "a": quantiles(ra[ok]), "b": quantiles(rb[ok]), "equal": round(float(np.mean(ra[ok] == rb[ok])), 4),
            "a_minus_b": quantiles(ra[ok] - rb[ok]), "pearson": round(float(np.corrcoef(ra[ok], rb[ok])[0, 1]), 4)}
    if "edge_radius_vx" in ga and "edge_radius_vx" in gb:
        # In voxels the walk's stopping distance has no tie: the µm radius
        # of a tie group depends on which member each side's order names.
        ra, rb = ga["edge_radius_vx"][ea], gb["edge_radius_vx"][eb]
        ok = (ra >= 0) & (rb >= 0)
        record["edge_radius_vx_matched"] = {
            "equal": round(float(np.mean(np.isclose(ra[ok], rb[ok]))), 4),
            "a_minus_b": quantiles(ra[ok] - rb[ok])}
    if "point_radius_um" in ga and "point_radius_um" in gb:
        pra, prb = [], []
        for e, f, _, _, nearest in matches:
            sa, sb = ga["edge_point_start"][e], gb["edge_point_start"][f]
            ca = ga["edge_point_count"][e]
            pra.append(ga["point_radius_um"][sa:sa + ca])
            prb.append(gb["point_radius_um"][sb + nearest])
        pra, prb = np.concatenate(pra), np.concatenate(prb)
        ok = (pra >= 0) & (prb >= 0)
        record["point_radius_um_matched"] = {
            "points": int(ok.sum()), "equal": round(float(np.mean(pra[ok] == prb[ok])), 4),
            "a_minus_b": quantiles(pra[ok] - prb[ok]),
            "pearson": round(float(np.corrcoef(pra[ok], prb[ok])[0, 1]), 4)}
    if "edge_distance_to_surface" in ga and "edge_distance_to_surface" in gb:
        da, db = ga["edge_distance_to_surface"][ea], gb["edge_distance_to_surface"][eb]
        record["edge_distance_to_surface_matched"] = {"equal": round(float(np.mean(da == db)), 4),
                                                      "a_minus_b": quantiles(da - db)}
    for key in ("edge_artery_binary", "edge_artery", "edge_vein"):
        if key in ga and key in gb:
            fa, fb = ga[key][ea] > 0, gb[key][eb] > 0
            both, either = int((fa & fb).sum()), int((fa | fb).sum())
            record[key.replace("edge_", "") + "_matched"] = {
                "a_only": int((fa & ~fb).sum()), "b_only": int((~fa & fb).sum()), "both": both,
                "neither": int((~fa & ~fb).sum()), "agreement": round(float(np.mean(fa == fb)), 4),
                "jaccard": round(both / either, 4) if either else None, "kappa": kappa(fa, fb)}
    return record, matches


def kappa(fa, fb):
    po = np.mean(fa == fb)
    pe = fa.mean() * fb.mean() + (1 - fa.mean()) * (1 - fb.mean())
    return round(float((po - pe) / (1 - pe)), 4) if pe < 1 else None


def coverage(ga, gb, spacing, tol=3.0):
    """Share of each graph's polyline length within `tol` of the other's
    polyline points (each point weighted by half its two segments)."""
    def weights(g):
        pts = g["point_zyx"] * spacing
        w = np.zeros(len(pts))
        for s, c in zip(g["edge_point_start"], g["edge_point_count"]):
            seg = np.linalg.norm(np.diff(pts[s:s + c], axis=0), axis=1)
            w[s:s + c - 1] += seg / 2
            w[s + 1:s + c] += seg / 2
        return pts, w
    pa, wa = weights(ga)
    pb, wb = weights(gb)
    da, _ = cKDTree(pb).query(pa, distance_upper_bound=tol)
    db, _ = cKDTree(pa).query(pb, distance_upper_bound=tol)
    return {"a_length_near_b": round(float(wa[np.isfinite(da)].sum() / wa.sum()), 4),
            "b_length_near_a": round(float(wb[np.isfinite(db)].sum() / wb.sum()), 4), "tol_um": tol}


def pair(args):
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    ga, gb = load_graph(args.a), load_graph(args.b)
    for spec, g in ((args.classes_a, ga), (args.classes_b, gb)):
        if spec:
            classes = np.load(spec)
            g["edge_artery"] = classes["edge_artery"].astype(float)
            g["edge_vein"] = classes["edge_vein"].astype(float)
    if args.distance_b:
        gb["edge_distance_to_surface"] = np.load(args.distance_b)
    shape = args.shape and [int(v) for v in args.shape.split(",")]
    record = {"a": {"name": ga["name"], "path": ga["path"]}, "b": {"name": gb["name"], "path": gb["path"]},
              "spacing_zyx_um": spacing.tolist(), "vertex_tol_um": args.vertex_tol}
    sk = {}
    for side, spec in (("a", args.skeleton_a), ("b", args.skeleton_b)):
        if spec:
            sk[side] = load_volume(spec)
            shape = shape or list(sk[side].shape)
    shape = shape or ga.get("volume") or gb.get("volume")
    record["volume_zyx"] = shape
    record["stats"] = {"a": stats(ga, spacing, shape), "b": stats(gb, spacing, shape)}
    if args.mask_a and args.mask_b:
        ma, mb = load_volume(args.mask_a), load_volume(args.mask_b)
        both = int(np.logical_and(ma, mb).sum())
        record["mask"] = {"a": int(ma.sum()), "b": int(mb.sum()),
                          "dice": round(2 * both / (int(ma.sum()) + int(mb.sum())), 4)}
    if len(sk) == 2:
        record["skeleton"] = compare_skeletons(sk["a"], sk["b"], spacing)
    record["vertices"] = compare_vertices(ga, gb, spacing, args.vertex_tol)
    record["edges"], _ = compare_edges(ga, gb, spacing, args.vertex_tol)
    record["coverage"] = coverage(ga, gb, spacing)
    text = json.dumps(record, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def skeletons(args):
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    sa, sb = load_volume(args.a), load_volume(args.b)
    record = {"a": args.a, "b": args.b, **compare_skeletons(sa, sb, spacing),
              "identical": bool(np.array_equal(sa, sb))}
    text = json.dumps(record, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------------------
# Atlas space
# ---------------------------------------------------------------------------

def frame(image):
    import zarr
    group = zarr.open_group(str(image), mode="r")
    multiscales = group.attrs["ome"]["multiscales"][0] if "ome" in group.attrs else group.attrs["multiscales"][0]
    dataset = multiscales["datasets"][0]
    scale = translation = None
    for t in dataset["coordinateTransformations"]:
        if t["type"] == "scale":
            scale = np.asarray(t["scale"][-3:], float)
        elif t["type"] == "translation":
            translation = np.asarray(t["translation"][-3:], float)
    shape = group[dataset["path"]].shape[-3:]
    return scale, translation if translation is not None else np.zeros(3), shape


def registration_fixed_frame(registration):
    registration = Path(registration)
    chain = registration / "chain.json"
    first = registration / json.loads(chain.read_text())["steps"][0] if chain.exists() else registration
    fixed = json.loads((first / "registration.json").read_text())["fixed"]["level0_frame"]
    return np.asarray(fixed["scale"], float), np.asarray(fixed["translation"], float)


def to_atlas(points_zyx, image, registration, bin_dir, work, tag):
    """Image level-0 voxels through the registration to atlas level-0 voxels."""
    scale, translation, _ = frame(image)
    fixed_scale, fixed_translation = registration_fixed_frame(registration)
    fixed = (translation + points_zyx * scale - fixed_translation) / fixed_scale
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    source, result = work / f"{tag}.csv", work / f"{tag}.atlas.csv"
    np.savetxt(source, fixed, delimiter=",", header="z,y,x", comments="", fmt="%.6f")
    subprocess.run([str(Path(bin_dir) / "atlas-registration"), "transform-points", "--registration",
                    str(registration), "--input", str(source), "--output", str(result)], check=True,
                   capture_output=True)
    with open(result) as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[float(r[f"moving_{a}"]) for a in "zyx"] for r in rows])


class Atlas:
    def __init__(self, path):
        import zarr
        path = Path(path)
        self.annotation = np.asarray(zarr.open_array(str(path / "labels" / "annotation" / "0"), mode="r")[...])
        self.distance = np.asarray(zarr.open_array(str(path / "distance_to_surface" / "0"), mode="r")[...])
        props = zarr.open_group(str(path / "labels" / "annotation"), mode="r").attrs["image-label"]["properties"]
        self.acronym = {p["label-value"]: p["acronym"] for p in props}
        self.parent = {p["label-value"]: p["parent"] for p in props}

    def voxel(self, atlas_zyx):
        v = np.rint(atlas_zyx).astype(np.int64)
        inside = np.all((v >= 0) & (v < np.array(self.annotation.shape)), axis=1)
        return v, inside

    def region(self, atlas_zyx):
        v, inside = self.voxel(atlas_zyx)
        out = np.zeros(len(v), np.int64)
        out[inside] = self.annotation[tuple(v[inside].T)]
        return out

    def depth(self, atlas_zyx):
        """Distance to the surface at the nearest voxel, clamped into the atlas."""
        v = np.clip(np.rint(atlas_zyx).astype(np.int64), 0, np.array(self.distance.shape) - 1)
        return self.distance[tuple(v.T)].astype(float)

    def area(self, label):
        """The region a layer belongs to (a name ending in a layer number or
        `1`, `2/3`, `6a`, `6b` collapses to its parent)."""
        name = self.acronym.get(label, str(label))
        if self.parent.get(label) is not None and LAYER.match(name):
            return self.parent[label]
        return label


def point_weights(graph, spacing):
    pts = graph["point_zyx"] * spacing
    w = np.zeros(len(pts))
    for s, c in zip(graph["edge_point_start"], graph["edge_point_count"]):
        seg = np.linalg.norm(np.diff(pts[s:s + c], axis=0), axis=1)
        w[s:s + c - 1] += seg / 2
        w[s + 1:s + c] += seg / 2
    return w


def atlas(args):
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    graphs = [load_graph(args.a), load_graph(args.b)]
    atlas_volume = Atlas(args.atlas)
    _, _, shape = frame(args.image)
    stride = args.grid
    grid = np.stack(np.meshgrid(*[np.arange(stride // 2, n, stride) for n in shape], indexing="ij"),
                    axis=-1).reshape(-1, 3).astype(float)
    grid_atlas = to_atlas(grid, args.image, args.registration, args.bin, args.work, "grid")
    cell_mm3 = float(stride ** 3 * np.prod(spacing)) * 1e-9
    grid_region = atlas_volume.region(grid_atlas)
    grid_area = np.array([atlas_volume.area(r) for r in grid_region])
    grid_depth = atlas_volume.depth(grid_atlas)
    record = {"grid_stride": stride, "grid_points": int(len(grid)), "atlas": str(args.atlas),
              "registration": str(args.registration), "sides": {}}
    per_side = {}
    for g in graphs:
        tag = g["name"]
        points_atlas = to_atlas(g["point_zyx"], args.image, args.registration, args.bin, args.work, f"{tag}-points")
        vertex_atlas = to_atlas(g["vertex_zyx"], args.image, args.registration, args.bin, args.work, f"{tag}-vertices")
        check = {}
        if "point_atlas_zyx" in g:
            check["max_abs_vs_own_coordinates_atlas"] = round(float(np.abs(points_atlas - g["point_atlas_zyx"]).max()), 6)
        point_region = atlas_volume.region(points_atlas)
        if "point_annotation" in g:
            check["own_annotation_equal"] = round(float(np.mean(point_region == g["point_annotation"])), 6)
        w = point_weights(g, spacing)
        deg = degrees(g)
        branch = deg >= 3
        per_side[tag] = {"point_region": point_region, "w": w,
                         "point_depth": atlas_volume.depth(points_atlas),
                         "vertex_region": atlas_volume.region(vertex_atlas)[branch],
                         "vertex_depth": atlas_volume.depth(vertex_atlas)[branch]}
        per_side[tag]["point_area"] = np.array([atlas_volume.area(r) for r in per_side[tag]["point_region"]])
        per_side[tag]["vertex_area"] = np.array([atlas_volume.area(r) for r in per_side[tag]["vertex_region"]])
        record["sides"][tag] = {"path": g["path"], **check,
                                "length_outside_atlas_share": round(float(w[per_side[tag]["point_region"] == 0].sum()
                                                                          / w.sum()), 4)}
    names = [g["name"] for g in graphs]

    def table(grid_key, point_key, vertex_key, top):
        labels, counts = np.unique(grid_key, return_counts=True)
        order = np.argsort(-counts)
        rows = []
        for label, count in zip(labels[order][:top], counts[order][:top]):
            volume = count * cell_mm3
            row = {"region": atlas_volume.acronym.get(int(label), str(label)), "id": int(label),
                   "volume_mm3": round(volume, 5)}
            for tag in names:
                s = per_side[tag]
                row[f"length_density_{tag}"] = round(float(s["w"][s[point_key] == label].sum()) * 1e-3 / volume, 1)
                row[f"branch_density_{tag}"] = round(float(np.sum(s[vertex_key] == label)) / volume, 0)
            a, b = names
            row["length_ratio"] = round(row[f"length_density_{a}"] / row[f"length_density_{b}"], 4) \
                if row[f"length_density_{b}"] else None
            row["branch_ratio"] = round(row[f"branch_density_{a}"] / row[f"branch_density_{b}"], 4) \
                if row[f"branch_density_{b}"] else None
            rows.append(row)
        return rows
    record["regions"] = table(grid_region, "point_region", "vertex_region", args.top)
    record["areas"] = table(grid_area, "point_area", "vertex_area", args.top)
    edges = np.arange(0, float(np.ceil(max(grid_depth.max(), 1))) + args.depth_bin, args.depth_bin)
    depth_rows = []
    # Inside the brain only: outside the annotation the distance is 0 too.
    grid_bin = np.where(grid_region != 0, np.digitize(grid_depth, edges) - 1, -1)
    for k in range(len(edges) - 1):
        count = int(np.sum(grid_bin == k))
        if count < args.min_cells:
            continue
        volume = count * cell_mm3
        row = {"depth_atlas_voxels": [float(edges[k]), float(edges[k + 1])], "volume_mm3": round(volume, 5)}
        for tag in names:
            s = per_side[tag]
            pb = np.where(s["point_region"] != 0, np.digitize(s["point_depth"], edges) - 1, -1)
            vb = np.where(s["vertex_region"] != 0, np.digitize(s["vertex_depth"], edges) - 1, -1)
            row[f"length_density_{tag}"] = round(float(s["w"][pb == k].sum()) * 1e-3 / volume, 1)
            row[f"branch_density_{tag}"] = round(float(np.sum(vb == k)) / volume, 0)
        depth_rows.append(row)
    record["depth"] = depth_rows
    text = json.dumps(record, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def edge_distance(args):
    """Per edge, the minimum distance to the surface along its polyline
    (atlas voxels, nearest voxel, clamped): the reference's
    `_compute_distance_to_surface`, through this registration."""
    g = load_graph(args.graph)
    points_atlas = to_atlas(g["point_zyx"], args.image, args.registration, args.bin, args.work, f"{g['name']}-points")
    depth = Atlas(args.atlas).depth(points_atlas)
    out = np.array([depth[s:s + c].min() for s, c in zip(g["edge_point_start"], g["edge_point_count"])])
    np.save(args.out, out)
    print(f"{len(out)} edges, distance to surface p50 {np.median(out):.1f} atlas voxels")


def import_labels(args):
    """Overwrite level 0 of label images with `.npy` volumes (same shape)."""
    import zarr
    for item in args.labels:
        label, _, spec = item.partition("=")
        array = zarr.open_array(str(Path(args.zarr) / "labels" / label / "0"), mode="r+")
        volume = load_volume(spec)
        if tuple(volume.shape) != tuple(array.shape[-3:]):
            raise SystemExit(f"{label}: {volume.shape} against {array.shape}")
        array[...] = volume.astype(array.dtype).reshape(array.shape)
        print(f"{label}: {int(volume.sum())} voxels from {spec}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def table(args):
    rows = []
    for path in args.pairs:
        r = json.loads(Path(path).read_text())
        sa, sb = r["stats"]["a"], r["stats"]["b"]
        v = r["vertices"]
        e = r["edges"]
        tol = r["vertex_tol_um"]
        bp = v[f"branch_points_within_{tol:g}um"]
        bp2 = v[f"branch_points_within_{2 * tol:g}um"]
        row = {"pair": Path(path).stem, "a": r["a"]["name"], "b": r["b"]["name"],
               "vertices": f"{sa['vertices']} / {sb['vertices']}", "edges": f"{sa['edges']} / {sb['edges']}",
               "length_mm": f"{sa['total_length_mm']:.1f} / {sb['total_length_mm']:.1f}",
               "branch_points": f"{sa['branch_points']} / {sb['branch_points']}",
               "end_points": f"{sa['end_points']} / {sb['end_points']}",
               "components": f"{sa['components']} / {sb['components']}",
               "skeleton_3um_f1": r.get("skeleton", {}).get("within_3um", {}).get("f1"),
               "skeleton_exact_f1": r.get("skeleton", {}).get("exact", {}).get("f1"),
               f"branch_f1_{tol:g}um": bp["f1"], f"branch_f1_{2 * tol:g}um": bp2["f1"],
               "branch_degree_equal": bp["degree_equal"],
               "edges_f1": e["matched_by_endpoints"]["f1"],
               "path_mean_p50_um": e.get("path_mean_distance_um", {}).get("p50"),
               "length_within_10pc": e.get("length_um_matched", {}).get("relative_within_10pc"),
               "edge_radius_equal": e.get("edge_radius_um_matched", {}).get("equal"),
               "coverage": f"{r['coverage']['a_length_near_b']} / {r['coverage']['b_length_near_a']}"}
        for key in ("artery_binary", "artery", "vein"):
            m = e.get(f"{key}_matched")
            if m:
                row[f"{key}_kappa"] = m["kappa"]
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        print("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--a", required=True, help="name=graph under test (.json or .npz)")
    p.add_argument("--b", required=True, help="name=reference graph")
    p.add_argument("--skeleton-a")
    p.add_argument("--skeleton-b")
    p.add_argument("--mask-a")
    p.add_argument("--mask-b")
    p.add_argument("--classes-a")
    p.add_argument("--classes-b")
    p.add_argument("--distance-b", help="per-edge distance to the surface of b (edge-distance)")
    p.add_argument("--shape", help="z,y,x volume shape for densities (default: the skeleton's)")
    p.add_argument("--spacing", default="1.6,1.625,1.625", help="z,y,x µm per voxel")
    p.add_argument("--vertex-tol", type=float, default=5.0, help="vertex matching distance, µm")
    p.add_argument("--out")
    for name in ("atlas", "edge-distance"):
        p = sub.add_parser(name)
        if name == "atlas":
            p.add_argument("--a", required=True)
            p.add_argument("--b", required=True)
            p.add_argument("--grid", type=int, default=8, help="voxel stride of the volume grid")
            p.add_argument("--top", type=int, default=12)
            p.add_argument("--depth-bin", type=float, default=2.0, help="atlas voxels")
            p.add_argument("--min-cells", type=int, default=20)
        else:
            p.add_argument("--graph", required=True)
        p.add_argument("--image", required=True)
        p.add_argument("--registration", required=True)
        p.add_argument("--atlas", required=True)
        p.add_argument("--bin", required=True, help="directory with atlas-registration")
        p.add_argument("--work", required=True)
        p.add_argument("--spacing", default="1.6,1.625,1.625")
        p.add_argument("--out")
    p = sub.add_parser("skeletons")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--spacing", default="1.6,1.625,1.625")
    p.add_argument("--out")
    p = sub.add_parser("import-labels")
    p.add_argument("--zarr", required=True)
    p.add_argument("labels", nargs="+", help="label=volume.npy[:xyz]")
    p = sub.add_parser("table")
    p.add_argument("pairs", nargs="+")
    args = parser.parse_args()
    {"pair": pair, "skeletons": skeletons, "atlas": atlas, "edge-distance": edge_distance, "import-labels": import_labels,
     "table": table}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
