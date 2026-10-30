#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Decoder configuration/API sequence adapted from Trackastra 0.5.6.
# Copyright (c) 2024, Benjamin Gallusser, Martin Weigert.
# Additional bridge implementation: Blockflow contributors, 2026.
# See TRACKASTRA-LICENSE.txt for the retained notice and terms.
"""Version-pinned Trackastra bridge for Blockflow's fixed-mask tracking suite.

Predict once from Phase pixels and native masks, then decode the immutable score
cache separately. Inputs are opened read-only; no mask replacements are written.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time


VERSION = "0.5.6"
ADAPTER_BYTES = Path(__file__).read_bytes()
ADAPTER_SHA256 = hashlib.sha256(ADAPTER_BYTES).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def require_version():
    actual = importlib.metadata.version("trackastra")
    if actual != VERSION:
        raise ValueError(f"adapter requires trackastra=={VERSION}, found {actual}")


def make_graph(cache):
    """Use the shared physical gate, no extra top-K pruning, one-frame links."""
    import networkx as nx
    graph = nx.DiGraph()
    for node in cache["nodes"]:
        graph.add_node(node["id"], time=node["t_index"], label=node["id"],
                       coords=node["registered"][-2:], weight=1,
                       fixed_mask_selected=True)
    if len(graph) != len(cache["nodes"]):
        raise ValueError("duplicate native node IDs in prediction cache")
    for edge in cache["edges"]:
        a, b = edge["from"], edge["to"]
        if a not in graph or b not in graph:
            raise ValueError("prediction references an unknown native node")
        if graph.nodes[b]["time"] != graph.nodes[a]["time"] + 1:
            raise ValueError("predictions must link consecutive frame indices")
        weight = edge["weight"]
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Trackastra association weight must be finite in [0,1]")
        if graph.has_edge(a, b):
            raise ValueError("duplicate prediction edge")
        distance = math.dist(graph.nodes[a]["coords"], graph.nodes[b]["coords"])
        if distance <= cache["max_distance"]:
            graph.add_edge(a, b, weight=weight, distance=distance)
    return graph


def validate_solution(candidate, solution, allow_divisions):
    import networkx as nx
    if set(solution) != set(candidate):
        raise ValueError("fixed-mask decoder must preserve all detection IDs")
    if not nx.is_directed_acyclic_graph(solution):
        raise ValueError("decoder produced a cycle")
    for node in solution:
        if solution.in_degree(node) > 1 or solution.out_degree(node) > (2 if allow_divisions else 1):
            raise ValueError("decoder violates lineage degree constraints")
    if any(not candidate.has_edge(a, b) for a, b in solution.edges):
        raise ValueError("decoder invented a link outside the shared candidates")


def predict(args):
    import numpy as np
    import torch
    import zarr
    from trackastra.data import build_windows, get_features
    from trackastra.model import Trackastra
    from trackastra.model.predict import predict_windows
    from trackastra.utils import normalize
    require_version()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU was requested for this experiment but CUDA is unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
    spec_bytes = args.suite_input.read_bytes()
    spec = json.loads(spec_bytes)
    if spec["schema"] != "blockflow.tracking-suite-input/v1":
        raise ValueError("unsupported suite input")
    phase_meta = json.loads((args.phase / "zarr.json").read_text())
    mask_meta = json.loads((args.masks / "zarr.json").read_text())
    phase_group = json.loads((args.phase.parent / "zarr.json").read_text())["attributes"]
    ome = phase_group.get("ome", phase_group)
    name = ome["multiscales"][0].get("name", "")
    if "phase" not in name.lower():
        raise ValueError("source image metadata must explicitly identify Phase")
    if phase_meta["dimension_names"] != ["t", "c", "z", "y", "x"] or mask_meta["dimension_names"] != ["t", "z", "y", "x"]:
        raise ValueError("this first adapter requires tczyx Phase and tzyx masks")
    if phase_meta["shape"][1:3] != [1, 1] or mask_meta["shape"][1] != 1:
        raise ValueError("this first adapter supports one Phase channel and 2D masks")
    frames = spec["frames"]
    if [f["t_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Trackastra bridge currently requires a dense zero-based timeline")
    expected_shape = [len(frames), *spec["extent"]]
    if mask_meta["shape"] != expected_shape or phase_meta["shape"] != [len(frames), 1, *spec["extent"]]:
        raise ValueError("source arrays and detection geometry have different shapes")
    args.output.mkdir()  # Existing prediction directories are refused.
    (args.output / "adapter.py").write_bytes(ADAPTER_BYTES)
    (args.output / "suite-input.json").write_bytes(spec_bytes)
    phase = zarr.open_array(str(args.phase), mode="r")
    masks = zarr.open_array(str(args.masks), mode="r")
    by_frame = {t: [] for t in range(len(frames))}
    for detection in spec["detections"]:
        by_frame[detection["t_index"]].append(detection)
    imgs = np.empty((len(frames), *spec["extent"][-2:]), dtype=phase.dtype)
    dense_masks = np.empty(imgs.shape, dtype=np.uint32)
    mappings = []
    pixel_hashes = []
    for t in range(len(frames)):
        image = np.asarray(phase[t, 0, 0])
        native = np.asarray(masks[t, 0])
        ids, inverse = np.unique(native, return_inverse=True)
        if ids[0] != 0:
            raise ValueError("expected background label zero")
        if [int(x) for x in ids[1:]] != sorted(d["id"] for d in by_frame[t]):
            raise ValueError(f"native masks and input detections differ in frame {t}")
        imgs[t] = image
        dense_masks[t] = inverse.reshape(native.shape).astype(np.uint32)
        mappings.append([int(x) for x in ids])
        counts = np.bincount(dense_masks[t].ravel())
        for dense, detection_id in enumerate(mappings[-1][1:], start=1):
            record = next(d for d in by_frame[t] if d["id"] == detection_id)
            if counts[dense] != record["size"]:
                raise ValueError(f"mask area mismatch for detection {detection_id}")
        pixel_hashes.append({"t_index": t, "phase_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                             "native_mask_sha256": hashlib.sha256(native.tobytes()).hexdigest()})
    print("Read Phase and masks; exact native IDs and areas verified", flush=True)
    model = Trackastra.from_folder(args.model_dir, device="cuda", batch_size=args.batch_size)
    if model.device != "cuda" or model.train_args["features"] != "wrfeat":
        raise ValueError("expected CUDA general_2d wrfeat model")
    model.transformer.eval()
    start = time.monotonic()
    features = get_features(dense_masks, imgs=normalize(imgs), features="wrfeat", ndim=2, n_workers=args.workers)
    # Extract morphology/intensity from unresampled pixels; give the model the
    # SAME registered centroids as classical tracking, converted to pixel units.
    # No fluorescence or interpolated image content enters these features.
    for t, feature in enumerate(features):
        native_lookup = {d["id"]: d for d in by_frame[t]}
        for i, label in enumerate(feature.labels):
            detection = native_lookup[mappings[t][int(label)]]
            if not np.allclose(feature.coords[i], detection["native"][-2:], atol=0.002, rtol=0):
                raise ValueError("measured native centroids disagree with input detection geometry")
            feature.coords[i] = np.asarray(detection["registered"][-2:]) / np.asarray(spec["spacing"][-2:])
    feature_seconds = time.monotonic() - start
    windows = build_windows(features, window_size=model.transformer.config["window"], as_torch=True)
    start = time.monotonic()
    torch.cuda.synchronize()
    predictions = predict_windows(windows=windows, features=features, model=model.transformer,
                                  edge_threshold=0.05, spatial_dim=2, batch_size=args.batch_size)
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - start
    node_map = {int(node["id"]): mappings[int(node["time"])][int(node["label"])] for node in predictions["nodes"]}
    if set(node_map.values()) != {d["id"] for d in spec["detections"]} or len(node_map) != len(spec["detections"]):
        raise ValueError("prediction nodes do not preserve the complete detection set")
    weights = [{"from": node_map[int(pair[0])], "to": node_map[int(pair[1])], "weight": float(weight)}
               for pair, weight in predictions["weights"]]
    cache = {"schema": "blockflow.trackastra-predictions/v1", "nodes": spec["detections"],
             "edges": weights, "max_distance": spec["max_distance"], "input_sha256": hashlib.sha256(spec_bytes).hexdigest(),
             "baseline_pairs": spec.get("baseline_pairs"), "reference": spec.get("reference"),
             "provenance": {"trackastra": VERSION, "model": str(args.model_dir.resolve()),
                            "model_files_sha256": {p.name: digest(p) for p in args.model_dir.iterdir() if p.is_file()},
                            "adapter_sha256": ADAPTER_SHA256, "source_arrays": {"phase": str(args.phase.resolve()), "masks": str(args.masks.resolve())},
                            "source_pixels": pixel_hashes, "features": "wrfeat", "window": model.transformer.config["window"],
                            "edge_threshold": 0.05, "coordinates": "registered centroids / pixel spacing, no image resampling",
                            "normalization": "upstream normalize over full Phase movie", "device": model.device,
                            "gpu": torch.cuda.get_device_name(), "feature_seconds": feature_seconds, "inference_seconds": inference_seconds,
                            "batch_size": args.batch_size, "workers": args.workers,
                            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "torch", "zarr", "motile", "ilpy", "pyscipopt"]}}}
    graph = make_graph(cache)
    save(args.output / "predictions.json", cache)
    save(args.output / "complete.json", {"complete": True, "predictions_sha256": digest(args.output / "predictions.json"),
                                         "nodes": len(graph), "gated_edges": graph.number_of_edges(), "raw_predicted_edges": len(weights)})
    print(f"Cached {len(graph)} nodes, {len(weights)} scores, {graph.number_of_edges()} gated edges; GPU inference {inference_seconds:.3f}s", flush=True)


def ilp_decode(graph, allow_divisions, timeout, threads, split_cost=0.25):
    import ilpy
    import motile
    from trackastra.tracking.ilp import ILP_CONFIGS, solution_to_graph
    from types import SimpleNamespace
    parameters = vars(ILP_CONFIGS["gt"]).copy()
    parameters["splitC"] = split_cost
    p = SimpleNamespace(**parameters)
    track_graph = motile.TrackGraph(graph, frame_attribute="time")
    solver = motile.Solver(track_graph)
    solver.add_cost(motile.costs.NodeSelectedCost(weight=p.nodeW, constant=p.nodeC, attribute="weight"))
    solver.add_cost(motile.costs.EdgeSelectedCost(weight=p.edgeW, constant=p.edgeC, attribute="weight"))
    solver.add_cost(motile.costs.NodeAppearCost(constant=p.appearC))
    solver.add_cost(motile.costs.NodeDisappearCost(constant=p.disappearC))
    if allow_divisions:
        solver.add_cost(motile.costs.NodeSplitCost(constant=p.splitC))
    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2 if allow_divisions else 1))
    # Explicitly preserve all masks, rather than relying only on nodeC=-10.
    solver.add_constraint(motile.constraints.Pin("fixed_mask_selected"))
    solution = solver.solve(timeout=timeout, num_threads=threads, backend=ilpy.Preference.Scip)
    selected = solver.get_variables(motile.variables.NodeSelected)
    if any(solution[index] < 0.5 for index in selected.values()):
        raise ValueError("ILP did not return a feasible fixed-mask incumbent")
    result = solution_to_graph(solver, track_graph)
    objective = sum(float(solver.costs[i]) * float(solution[i]) for i in range(solver.num_variables))
    return result, {"backend": "SCIP", "status": str(solution.get_status()),
                    "native_status": str(solution.native_status), "objective": objective,
                    "timeout_seconds": timeout, "threads": threads, "costs": vars(p),
                    "fixed_masks_pinned": True}


def decode(args):
    from trackastra.tracking import track_greedy
    require_version()
    completion = json.loads((args.predictions / "complete.json").read_text())
    prediction_file = args.predictions / "predictions.json"
    if completion["predictions_sha256"] != digest(prediction_file):
        raise ValueError("prediction cache changed after completion")
    cache = json.loads(prediction_file.read_text())
    if cache["schema"] != "blockflow.trackastra-predictions/v1":
        raise ValueError("unsupported prediction schema")
    graph = make_graph(cache)
    args.output.mkdir()
    (args.output / "adapter.py").write_bytes(ADAPTER_BYTES)
    reports = []
    baseline = None if cache.get("baseline_pairs") is None else {tuple(pair) for pair in cache["baseline_pairs"]}
    for name, allow_divisions, use_ilp in [("greedy_nodiv", False, False), ("greedy", True, False),
                                            ("ilp_nodiv", False, True), ("ilp", True, True)]:
        start = time.monotonic()
        if use_ilp:
            solution, solver_info = ilp_decode(graph, allow_divisions, args.timeout, args.threads, split_cost=args.split_cost)
        else:
            solution = track_greedy(graph, allow_divisions=allow_divisions)
            omitted = len(graph) - len(solution)
            solution.add_nodes_from((node, dict(data)) for node, data in graph.nodes(data=True) if node not in solution)
            solver_info = {"threshold": 0.5, "restored_isolated_nodes": omitted, "optimality": "greedy heuristic"}
        elapsed = time.monotonic() - start
        validate_solution(graph, solution, allow_divisions)
        edges = set(solution.edges)
        links = [{"from": a, "to": b, "dt": 1, "weight": graph.edges[a, b]["weight"],
                  "distance": graph.edges[a, b]["distance"],
                  "kind": "divide" if solution.out_degree(a) == 2 else "continue"} for a, b in sorted(edges)]
        report = {"name": name, "nodes": len(solution), "links": len(links),
                  "binary_divisions": sum(solution.out_degree(node) == 2 for node in solution),
                  "decode_seconds": elapsed, "solver": solver_info,
                  "baseline_edge_difference": None if baseline is None else {"added": len(edges - baseline), "removed": len(baseline - edges),
                                               "interpretation": "agreement, not accuracy"},
                  "reference_evaluation": None}
        save(args.output / f"{name}-links.json", links)
        save(args.output / f"{name}-report.json", report)
        reports.append(report)
        print(f"{name}: {len(links)} links, {report['binary_divisions']} divisions in {elapsed:.3f}s", flush=True)
    save(args.output / "summary.json", {"schema": "blockflow.trackastra-decode/v1", "complete": True,
                                        "predictions_sha256": digest(prediction_file), "adapter_sha256": ADAPTER_SHA256,
                                        "candidate_nodes": len(graph), "candidate_edges": graph.number_of_edges(),
                                        "gate": {"registered_physical_max_distance": cache["max_distance"], "max_neighbors": None,
                                                 "max_frame_difference": 1, "score_floor": 0.05},
                                        "validation": "diagnostics only; no reference accuracy asserted", "runs": reports})


def export_scores(args):
    prediction_file = args.predictions / "predictions.json"
    completion = json.loads((args.predictions / "complete.json").read_text())
    if completion["predictions_sha256"] != digest(prediction_file):
        raise ValueError("prediction cache changed after completion")
    cache = json.loads(prediction_file.read_text())
    input_bytes = (args.predictions / "suite-input.json").read_bytes()
    if hashlib.sha256(input_bytes).hexdigest() != cache["input_sha256"]:
        raise ValueError("copied suite input changed")
    spec = json.loads(input_bytes)
    if spec["reference"] is not None:
        raise ValueError("this first diagnostic exporter does not alter reference-bearing inputs")
    graph = make_graph(cache)
    node_of = {d["id"]: d for d in spec["detections"]}
    tables = {name: {"costs": [], "provenance": {"source_predictions_sha256": digest(prediction_file)}}
              for name in ["trackastra", "motion_matched", "motion_size400_matched"]}
    for a, b, data in sorted(graph.edges(data=True)):
        x, y = node_of[a], node_of[b]
        distance_squared = sum((xx - yy)**2 for xx, yy in zip(x["registered"], y["registered"]))
        largest = max(x["size"], y["size"])
        size_change = 0 if largest == 0 else abs(x["size"] - y["size"]) / largest
        for name, cost in [("trackastra", 0.5 - data["weight"]), ("motion_matched", distance_squared),
                           ("motion_size400_matched", distance_squared + 400 * size_change)]:
            tables[name]["costs"].append({"from": a, "to": b, "cost": cost})
    tables["trackastra"]["provenance"].update(formula="0.5 - association_weight", units="dimensionless energy",
                                              interpretation="greedy threshold 0.5 with zero birth/death costs; not calibrated log probability")
    for name in ["motion_matched", "motion_size400_matched"]:
        tables[name]["provenance"].update(units="squared physical distance", candidate_subset="identical learned-score candidates above floor 0.05",
                                         size_weight=400 if name == "motion_size400_matched" else 0)
    spec["score_tables"] = tables
    spec["runs"] = [{"name": f"{name}-{solver}", "solver": f"frame_{solver}", "scorer": name,
                     "size_weight": 0, "alternative_cost": 0 if name == "trackastra" else None}
                    for name in tables for solver in ["greedy", "lap"]]
    spec["provenance"] = {"source_input_provenance": spec["provenance"],
                           "exported_predictions_sha256": digest(prediction_file), "adapter_sha256": ADAPTER_SHA256,
                           "purpose": "matched-candidate score/solver diagnostics, not reference accuracy"}
    with args.output.open("x") as handle:
        json.dump(spec, handle, allow_nan=False)
        handle.write("\n")
    print(f"Exported {graph.number_of_edges()} scored pairs per table to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    for flag in ["suite-input", "phase", "masks", "model-dir", "output"]:
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    d = sub.add_parser("decode")
    d.add_argument("--predictions", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--timeout", type=float, default=120)
    d.add_argument("--split-cost", type=float, default=0.25)
    d.add_argument("--threads", type=int, default=min(8, len(os.sched_getaffinity(0))))
    e = sub.add_parser("export-scores")
    e.add_argument("--predictions", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args)
    elif args.command == "export-scores":
        export_scores(args)
    else:
        if args.timeout <= 0 or args.threads < 1 or not math.isfinite(args.split_cost) or args.split_cost < 0:
            parser.error("timeout/threads must be positive and split-cost finite/non-negative")
        decode(args)


if __name__ == "__main__":
    main()
