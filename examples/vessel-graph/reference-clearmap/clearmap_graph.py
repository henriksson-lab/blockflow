#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""The reference's graph half on any skeleton and mask, and its graphs as
plain arrays, so `compare_graph.py` can compare them with `vessel-graph`'s.

Run with the reference's Python (env `clearmap-gt2`). Volumes are `.npy`
files; `--layout` gives their axis order, and a `:xyz` or `:zyx` suffix on
a path that one's (the reference's own intermediates are `xyz`,
`export_labels.py` writes `zyx`).

    clearmap_graph.py skeletonize --mask M.npy --layout zyx --out SKEL.npy
    clearmap_graph.py build --skeleton S.npy --mask M.npy --artery-mask A.npy \
        --artery-raw R.npy --layout zyx --spacing 1.625,1.625,1.6 --out DIR
    clearmap_graph.py export --graph G.gt --out G.npz
    clearmap_graph.py classify --graph G.gt [--distance D.npy] --out C.npz

`skeletonize` is the reference's `skeletonize(delete_border=True)`.
`build` is the graph half of the benchmark's `clearmap/vessels.py`, call for
call (`graph_from_skeleton`, `measure_radius` with max radius 150 per axis
in µm, `measure_expression` max of the artery mask at the radius and of the
artery intensity 3 voxels beyond it, `clean_graph` and `reduce_graph` with
the orchestrator's mappings); it writes `DIR/graph_reduced.gt`,
`DIR/graph.npz` and `DIR/build.json`. `classify` runs the reference's
`VesselClassifier` with the `vasculature.yml` v3.1 thresholds in µm, on
every component; without `--distance` (per edge, atlas voxels) the
distance-to-surface test passes everywhere, as `vessel-graph` does when the
graph has no atlas.

The `.npz` holds `[z, y, x]` level-0 voxel coordinates (the reference's are
`[x, y, z]`, reversed here): `vertex_zyx`, `vertex_degree`,
`vertex_radius_vx`, `vertex_radius_um`, `edge_vertices`, `edge_length_um`,
`edge_radius_vx`, `edge_radius_um`, `edge_artery_binary`, `edge_artery_raw`,
`point_zyx`, `point_radius_um`, `edge_point_start`, `edge_point_count`
(each edge's polyline, source end first).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CLEARMAP = os.environ.get("CLEARMAP_ROOT", "/data/henriksson/github/claude/clearmap-rs/ClearMap-py")
sys.path.insert(0, CLEARMAP)

# vasculature.yml v3.1, vessel_type_postprocessing.
CLASSIFY = {
    "vein_intensity_range": (0.0, 2500.0), "restrictive_vein_radius": 15.0,
    "permissive_vein_radius": 16.25, "final_vein_radius": 12.5, "arteries_min_noise_edges": 2,
    "artery_intensity_min": 200.0, "artery_trace_radius": 6.4, "vein_trace_radius": 8.0,
    "vein_intensity_min": 150.0, "distance_to_surface_min": 15.0, "max_artery_trace_iter": 5,
    "max_vein_trace_iter": 5, "min_artery_component_edges": 30, "min_vein_component_edges": 30,
}
ARTERINESS_SHIFT_VX = 3.0


def load_xyz(spec, layout):
    """A `.npy` volume (`path[:xyz|:zyx]`, else in `layout`) as the
    reference's `(x, y, z)` array, saved beside it if it had to be
    transposed (the reference's functions take paths)."""
    path, _, own = str(spec).partition(":")
    path, layout = Path(path), own or layout
    if layout == "xyz":
        return path
    out = path.with_name(path.stem + ".xyz.npy")
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        array = np.load(path, mmap_mode="r")
        # Fortran order, as the reference's own `(x, y, z)` intermediates are.
        np.save(out, np.ascontiguousarray(np.asarray(array)).transpose(2, 1, 0))
    return out


def skeletonize(args):
    import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization
    mask = load_xyz(args.mask, args.layout)
    binary = np.load(mask, mmap_mode="r")
    if binary.dtype != bool:
        mask = mask.with_name(mask.stem + ".bool.npy")
        np.save(mask, np.asarray(binary) != 0)
    out = Path(args.out)
    t0 = time.time()
    skeletonization.skeletonize(str(mask), sink=str(out), delete_border=True, n_processes=args.processes,
                                verbose=False)
    print(f"skeletonize: {time.time() - t0:.1f} s, {int(np.count_nonzero(np.load(out)))} voxels")


def build(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lightsheet-pipeline" / "benchmark" / "clearmap"))
    import compat  # noqa: F401  (rank filter masks, worker sinks; see benchmark/clearmap/compat.py)
    import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression
    import ClearMap.Analysis.Measurements.radius_measurements as measure_radius
    from ClearMap.Analysis.graphs import graph_processing
    from ClearMap.Analysis.graphs.graph_processing import Percentile

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.processes
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    seconds = {}
    skeleton = load_xyz(args.skeleton, args.layout)
    if np.load(skeleton, mmap_mode="r").dtype != bool:
        converted = skeleton.with_name(skeleton.stem + ".bool.npy")
        np.save(converted, np.load(skeleton) != 0)
        skeleton = converted
    final = load_xyz(args.mask, args.layout)
    if np.load(final, mmap_mode="r").shape != np.load(skeleton, mmap_mode="r").shape:
        raise SystemExit("the skeleton and the mask differ in shape: check their layouts")

    t0 = time.time()
    graph = graph_processing.graph_from_skeleton(str(skeleton), spacing=tuple(spacing), physical_units="µm",
                                                 check_border=False, n_processes=n, verbose=False)
    seconds["graph_from_skeleton"] = time.time() - t0

    t0 = time.time()
    coordinates = graph.vertex_coordinates()
    axial_um = measure_radius.measure_radius(str(final), coordinates, value=0, fraction=None, max_radius=150,
                                             return_indices=False, default=-1, return_radii_as_scalar=False,
                                             scale=spacing)
    axial_vx = axial_um / spacing[None, :]
    graph.define_vertex_property("radius_units_axial", axial_um)
    graph.define_vertex_property("radii_axial", axial_vx)
    graph.set_vertex_radii(np.linalg.norm(axial_vx, axis=1))
    graph.define_vertex_property("radius_units", np.linalg.norm(axial_um, axis=1))
    if not graph.has_graph_property("spacing"):
        graph.add_graph_property("spacing", spacing)
    seconds["radii"] = time.time() - t0

    def expression(source, shift_um, name, as_float):
        radii_um = graph.vertex_property("radius_units_axial") + shift_um
        search = np.floor(np.max(radii_um / spacing, axis=1)).astype(np.int32)
        values = measure_expression.measure_expression(str(source), coordinates, search, method="max",
                                                       n_processes=n)
        graph.define_vertex_property(name, np.asarray(values, dtype=float) if as_float else values)

    t0 = time.time()
    signals = []
    if args.artery_mask:
        expression(load_xyz(args.artery_mask, args.layout), 0.0, "artery_binary", False)
        signals.append("artery_binary")
    if args.artery_raw:
        expression(load_xyz(args.artery_raw, args.layout), ARTERINESS_SHIFT_VX * float(np.mean(spacing)),
                   "artery_raw", True)
        signals.append("artery_raw")
    seconds["signals"] = time.time() - t0
    raw = (int(graph.n_vertices), int(graph.n_edges))

    mappings = copy.copy(graph_processing.DEFAULT_VERTEX_TO_VERTEX)
    mappings.update({name: np.max for name in signals})
    t0 = time.time()
    cleaned = graph_processing.clean_graph(graph, vertex_mappings=mappings, verbose=False)
    seconds["clean_graph"] = time.time() - t0
    to_edge = copy.copy(graph_processing.DEFAULT_VERTEX_TO_EDGE)
    if "artery_binary" in signals:
        to_edge["artery_binary"] = Percentile(100 / 3)
    if "artery_raw" in signals:
        to_edge["artery_raw"] = np.max
    geometry = ["coordinates", "coordinates_units", "radii", "radius_units", "length", "chain_id", "_vertex_id_",
                "radii_axial", "radius_units_axial"] + signals
    t0 = time.time()
    reduced = graph_processing.reduce_graph(cleaned, vertex_to_edge_mappings=to_edge, edge_to_edge_mappings=None,
                                            compute_edge_length=True, edge_geometry_vertex_properties=geometry,
                                            return_maps=False, verbose=False)
    seconds["reduce_graph"] = time.time() - t0
    reduced.save(str(out / "graph_reduced.gt"))
    np.savez_compressed(out / "graph.npz", **arrays(reduced))
    record = {"skeleton": str(args.skeleton), "mask": str(args.mask), "spacing_xyz": spacing.tolist(),
              "processes": n, "seconds": {k: round(v, 3) for k, v in seconds.items()},
              "skeleton_voxels": int(np.count_nonzero(np.load(skeleton, mmap_mode="r"))),
              "volume_xyz": list(np.load(skeleton, mmap_mode="r").shape),
              "graph_raw": {"vertices": raw[0], "edges": raw[1]},
              "graph_cleaned": {"vertices": int(cleaned.n_vertices), "edges": int(cleaned.n_edges)},
              "graph_reduced": {"vertices": int(reduced.n_vertices), "edges": int(reduced.n_edges)}}
    (out / "build.json").write_text(json.dumps(record, indent=1) + "\n")
    print("BUILD", json.dumps(record), flush=True)


def arrays(graph):
    """A reduced graph as plain `[z, y, x]` arrays (see the module doc)."""
    rev = [2, 1, 0]
    result = {"vertex_zyx": np.asarray(graph.vertex_coordinates(), dtype=float)[:, rev],
              "vertex_degree": np.asarray(graph.vertex_degrees(), dtype=np.int64),
              "edge_vertices": np.asarray(graph.edge_connectivity(), dtype=np.int64)}
    vprops, eprops = graph.vertex_properties, graph.edge_properties
    if "radii" in vprops:
        result["vertex_radius_vx"] = np.asarray(graph.vertex_property("radii"), dtype=float)
    if "radius_units" in vprops:
        result["vertex_radius_um"] = np.asarray(graph.vertex_property("radius_units"), dtype=float)
    for name, key in (("length", "edge_length_um"), ("radii", "edge_radius_vx"), ("radius_units", "edge_radius_um"),
                      ("artery_binary", "edge_artery_binary"), ("artery_raw", "edge_artery_raw"),
                      ("artery", "edge_artery"), ("vein", "edge_vein")):
        if name in eprops:
            result[key] = np.asarray(graph.edge_property(name), dtype=float)
    points, ranges = graph.edge_geometry("coordinates", as_list=False, return_indices=True)
    result["point_zyx"] = np.asarray(points, dtype=float)[:, rev]
    ranges = np.asarray(ranges, dtype=np.int64)
    result["edge_point_start"] = ranges[:, 0]
    result["edge_point_count"] = ranges[:, 1] - ranges[:, 0]
    if "radius_units" in graph.edge_geometry_property_names or \
            graph.edge_geometry_property_name("radius_units") in graph.edge_geometry_property_names:
        result["point_radius_um"] = np.asarray(graph.edge_geometry("radius_units", as_list=False), dtype=float)
    # Orient each polyline from the edge's source, as `vessel-graph`'s are.
    vertex = result["vertex_zyx"]
    first = result["point_zyx"][result["edge_point_start"]]
    flipped = np.any(first != vertex[result["edge_vertices"][:, 0]], axis=1)
    result["edge_vertices"][flipped] = result["edge_vertices"][flipped][:, ::-1]
    return result


def export(args):
    from ClearMap.Analysis.graphs import graph_gt
    graph = graph_gt.load(str(args.graph))
    np.savez_compressed(args.out, **arrays(graph))
    print(f"exported {graph.n_vertices} vertices, {graph.n_edges} edges to {args.out}")


def classify(args):
    from ClearMap.Analysis.graphs import graph_gt
    from ClearMap.Analysis.graphs.vessel_classifier import ClassificationConfig, VesselClassifier, VesselSignals
    graph = graph_gt.load(str(args.graph))
    distance = np.load(args.distance) if args.distance else np.full(graph.n_edges, np.inf)
    eprops = graph.edge_properties
    signals = VesselSignals(radii=np.asarray(graph.edge_property("radius_units"), dtype=float),
                            artery_binary=np.asarray(graph.edge_property("artery_binary")).astype(bool),
                            artery_intensity=np.asarray(graph.edge_property("artery_raw"), dtype=float)
                            if "artery_raw" in eprops else None,
                            distance_to_surface=distance)
    t0 = time.time()
    VesselClassifier(graph, ClassificationConfig(**CLASSIFY), signals).classify()
    artery = np.asarray(graph.edge_property("artery")).astype(bool)
    vein = np.asarray(graph.edge_property("vein")).astype(bool)
    np.savez_compressed(args.out, edge_artery=artery, edge_vein=vein)
    print(f"classified in {time.time() - t0:.1f} s: {int(artery.sum())} artery, {int(vein.sum())} vein "
          f"of {graph.n_edges} edges")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("skeletonize")
    p.add_argument("--mask", required=True)
    p.add_argument("--layout", choices=("xyz", "zyx"), default="zyx")
    p.add_argument("--out", required=True)
    p.add_argument("--processes", type=int, default=8)
    p = sub.add_parser("build")
    p.add_argument("--skeleton", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--artery-mask")
    p.add_argument("--artery-raw")
    p.add_argument("--layout", choices=("xyz", "zyx"), default="zyx")
    p.add_argument("--spacing", default="1.625,1.625,1.6", help="x,y,z µm per voxel")
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export")
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("classify")
    p.add_argument("--graph", required=True)
    p.add_argument("--distance", help="per-edge distance to the surface, atlas voxels (.npy)")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"skeletonize": skeletonize, "build": build, "export": export, "classify": classify}[args.command](args)


if __name__ == "__main__":
    main()
