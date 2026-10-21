#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Reference side of the vessel workloads: the reference's TubeMap from two
raw channels to a reduced vessel graph, step by step as its orchestrator
(`pipeline_orchestrators/tube_map.py`) runs them with the 3.1
`vasculature.yml` defaults, every process count set to `--processes`:

per channel (0 the all-vessel stain as `vessel`, 1 the artery stain as
`artery`): binarize (clip range and deconvolve threshold of the role;
equalize and vesselize off for `artery`; z blocks of 10..16 planes with
overlap 5), smooth (6 iterations of the configuration rule, blocks of 50),
fill; with `--deep-fill`, `fill_vessels` (resample 1 / 2, blocks of 500 /
1000 with overlap 50 / 100, threshold 0.5) on the CPU, with the network as
the reference uses it (dropout active; `--eval` switches it off);
then the OR of the channels, its smooth + fill (`post_process_binary_combined`),
skeletonize (`delete_border=True`), `graph_from_skeleton`, the radii
(`measure_radius`, max radius 150, per axis in µm), the artery binary and
artery intensity per vertex (`measure_expression`, max, the orchestrator's
search radii), `clean_graph` and `reduce_graph` with the orchestrator's
mappings. No atlas step (the blockflow side runs `graph` without
`--registration` too).

    vessels.py --inputs DIR --out DIR --processes N [--spacing X,Y,Z] [--deep-fill [--eval]]

`--inputs` holds `source_c0.npy` and `source_c1.npy` from `prepare.py
channels` (`(x, y, z)`). Every intermediate is a `.npy` file in `--out`, as
the reference's workspace has them. Writes `timing.json` (seconds per step,
voxel and graph counts) and `graph_reduced.gt`. Run with the reference's
Python (env `clearmap-gt2`).
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

import ClearMap.IO.IO as clearmap_io  # noqa: E402
import ClearMap.ImageProcessing.Experts.Vasculature as vasculature  # noqa: E402
import ClearMap.ImageProcessing.Binary.Filling as binary_filling  # noqa: E402
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization  # noqa: E402
import ClearMap.ParallelProcessing.BlockProcessing as block_processing  # noqa: E402
from ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing import initialize_sink  # noqa: E402
import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression  # noqa: E402
import ClearMap.Analysis.Measurements.radius_measurements as measure_radius  # noqa: E402
from ClearMap.Analysis.graphs import graph_processing  # noqa: E402
from ClearMap.Analysis.graphs.graph_processing import Percentile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compat  # noqa: E402,F401  (worker sinks reopened r+, rank filter masks; see compat.py)

# vasculature.yml v3.1: vessels_template and large_vessels_template.
ROLES = {
    "vessel": {"clip_range": (350, 1000), "threshold": None, "large": False,
               "deep_fill": {"resample": 1, "size_max": 500, "overlap": 50}},
    "artery": {"clip_range": (1000, 8000), "threshold": 450, "large": True,
               "deep_fill": {"resample": 2, "size_max": 1000, "overlap": 100}},
}
CHANNELS = [(0, "vessel"), (1, "artery")]
# tube_map.VesselGraphProcessor._LEGACY_THRESHOLDS: the arteriness search shift.
ARTERINESS_SHIFT_VX = 3.0


class Steps:
    def __init__(self):
        self.seconds = {}

    def run(self, name, function, *args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        self.seconds[name] = round(time.time() - t0, 3)
        print(f"{name}: {self.seconds[name]:.1f} s", flush=True)
        return result


def binarize(source, sink, role, processes):
    parameter = copy.deepcopy(vasculature.default_binarization_parameter)
    parameter["clip"]["clip_range"] = ROLES[role]["clip_range"]
    if ROLES[role]["threshold"] is not None:
        parameter["deconvolve"]["threshold"] = ROLES[role]["threshold"]
    if ROLES[role]["large"]:
        parameter.update(equalize=None, vesselize=None)
    processing = copy.deepcopy(vasculature.default_binarization_processing_parameter)
    processing.update(processes=processes, size_min=10, size_max=16, overlap=5, as_memory=True,
                      verbose=False)
    vasculature.binarize(str(source), str(sink), binarization_parameter=parameter,
                         processing_parameter=processing)
    return sink


def smooth(source, processes):
    """`_smooth`: with 6 iterations the result goes to `<stem>_smooth_tmp.npy`."""
    parameter = copy.deepcopy(vasculature.default_postprocessing_parameter["smooth"])
    perf = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
    perf.update(size_max=50)
    unused = source.with_name(source.stem + "_smoothed.npy")
    _, path = vasculature.apply_smoothing(clearmap_io.open_ro(str(source)), str(unused), parameter, perf,
                                          processes=processes, verbose=False)
    return Path(path) if path else unused


def fill(source, sink, processes):
    source = clearmap_io.open_ro(str(source))
    target = initialize_sink(str(sink), shape=source.shape, dtype=source.dtype, order=source.order,
                             return_buffer=False)
    binary_filling.fill(source, sink=target, processes=processes, verbose=False)
    return sink


def deep_fill(source, sink, role, evaluation, threads):
    import torch
    import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vessel_filling
    torch.set_num_threads(threads)
    network = vessel_filling.vessel_filling_network(cuda=False).float()
    if evaluation:
        network.eval()
    settings = ROLES[role]["deep_fill"]
    processing = copy.deepcopy(vessel_filling.default_fill_vessels_processing_parameter)
    processing.update(size_max=settings["size_max"], size_min="fixed", axes="all",
                      overlap=settings["overlap"])
    vessel_filling.fill_vessels(str(source), str(sink), resample=settings["resample"], threshold=0.5,
                                network=network, cuda=False, processing_parameter=processing,
                                verbose=False)
    return sink


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--spacing", default="1.625,1.625,1.6", help="x,y,z µm per voxel")
    parser.add_argument("--deep-fill", action="store_true")
    parser.add_argument("--eval", action="store_true", help="deep fill with dropout off")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = args.processes
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    steps = Steps()
    t0 = time.time()

    last = {}
    for channel, role in CHANNELS:
        source = args.inputs / f"source_c{channel}.npy"
        binary = steps.run(f"binarize_c{channel}", binarize, source, out / f"binary_c{channel}.npy", role, n)
        smoothed = steps.run(f"smooth_c{channel}", smooth, binary, n)
        filled = steps.run(f"fill_c{channel}", fill, smoothed, out / f"filled_c{channel}.npy", n)
        last[channel] = filled
        if args.deep_fill:
            last[channel] = steps.run(f"deep_fill_c{channel}", deep_fill, filled,
                                      out / f"deep_filled_c{channel}.npy", role, args.eval, n)

    combined = out / "binary_combined.npy"
    steps.run("combine", block_processing.process, np.logical_or, [str(last[c]) for c, _ in CHANNELS],
              str(combined), size_max=500, overlap=0, processes=n, verbose=False)
    final = out / "binary_final.npy"
    perf = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
    perf["size_max"] = 50
    steps.run("postprocess_combined", vasculature.postprocess, str(combined), str(final),
              postprocessing_parameter=copy.deepcopy(vasculature.default_postprocessing_parameter),
              processing_parameter=perf, processes=n, verbose=False)
    skeleton = out / "skeleton.npy"
    steps.run("skeletonize", skeletonization.skeletonize, str(final), sink=str(skeleton),
              delete_border=True, n_processes=n, verbose=False)

    graph = steps.run("graph_from_skeleton", graph_processing.graph_from_skeleton, str(skeleton),
                      spacing=tuple(spacing), physical_units="µm", check_border=False, n_processes=n,
                      verbose=False)

    def radii():
        coordinates = graph.vertex_coordinates()
        axial_um = measure_radius.measure_radius(str(final), coordinates, value=0, fraction=None,
                                                 max_radius=150, return_indices=False, default=-1,
                                                 return_radii_as_scalar=False, scale=spacing)
        axial_vx = axial_um / spacing[None, :]
        graph.define_vertex_property("radius_units_axial", axial_um)
        graph.define_vertex_property("radii_axial", axial_vx)
        graph.set_vertex_radii(np.linalg.norm(axial_vx, axis=1))
        graph.define_vertex_property("radius_units", np.linalg.norm(axial_um, axis=1))
        if not graph.has_graph_property("spacing"):
            graph.add_graph_property("spacing", spacing)
    steps.run("radii", radii)

    def artery_expression(source, shift_um, name, as_float):
        coordinates = graph.vertex_coordinates()
        radii_um = graph.vertex_property("radius_units_axial") + shift_um
        search = np.floor(np.max(radii_um / spacing, axis=1)).astype(np.int32)
        values = measure_expression.measure_expression(str(source), coordinates, search, method="max",
                                                       n_processes=n)
        graph.define_vertex_property(name, np.asarray(values, dtype=float) if as_float else values)
    # `_set_artery_binary` reads the artery channel's filled (not deep-filled) mask.
    steps.run("artery_binary", artery_expression, out / "filled_c1.npy", 0.0, "artery_binary", False)
    steps.run("artery_raw", artery_expression, args.inputs / "source_c1.npy",
              ARTERINESS_SHIFT_VX * float(np.mean(spacing)), "artery_raw", True)
    raw_counts = (int(graph.n_vertices), int(graph.n_edges))

    mappings = copy.copy(graph_processing.DEFAULT_VERTEX_TO_VERTEX)
    mappings.update({"artery_binary": np.max, "artery_raw": np.max})
    cleaned = steps.run("clean_graph", graph_processing.clean_graph, graph, vertex_mappings=mappings,
                        verbose=False)
    to_edge = copy.copy(graph_processing.DEFAULT_VERTEX_TO_EDGE)
    to_edge.update({"artery_binary": Percentile(100 / 3), "artery_raw": np.max})
    geometry = ["coordinates", "coordinates_units", "radii", "radius_units", "length", "chain_id",
                "_vertex_id_"]
    for name in ("radii_axial", "radius_units_axial"):
        if name in cleaned.vertex_properties:
            geometry.append(name)
    geometry.extend(["artery_binary", "artery_raw"])
    reduced = steps.run("reduce_graph", graph_processing.reduce_graph, cleaned,
                        vertex_to_edge_mappings=to_edge, edge_to_edge_mappings=None,
                        compute_edge_length=True, edge_geometry_vertex_properties=geometry,
                        return_maps=False, verbose=False)
    steps.run("save_graph", reduced.save, str(out / "graph_reduced.gt"))
    total = time.time() - t0

    def count(path):
        return int(np.count_nonzero(np.load(path, mmap_mode="r")))
    record = {"processes": n, "deep_fill": args.deep_fill, "dropout": args.deep_fill and not args.eval,
              "spacing_xyz": spacing.tolist(), "steps_s": steps.seconds, "total_s": round(total, 3),
              "final_voxels": count(final), "skeleton_voxels": count(skeleton),
              "graph_raw": {"vertices": raw_counts[0], "edges": raw_counts[1]},
              "graph_cleaned": {"vertices": int(cleaned.n_vertices), "edges": int(cleaned.n_edges)},
              "graph_reduced": {"vertices": int(reduced.n_vertices), "edges": int(reduced.n_edges)}}
    # `length` (with `spacing` and physical units given, in µm).
    for prop in ("length", "length_units"):
        try:
            record[f"total_edge_{prop}"] = float(np.sum(reduced.edge_property(prop)))
        except Exception:  # noqa: BLE001 - the property is optional
            pass
    (out / "timing.json").write_text(json.dumps(record, indent=1))
    print("VESSELS", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
