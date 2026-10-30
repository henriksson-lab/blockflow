#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Optional Ultrack 0.8.0 segmentation-hypothesis experiment from Phase masks.

Outputs use their own segment/track identities. They must not be joined to
original fluorescence tables by numeric ID. Source arrays are read-only.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/blockflow-ultrack-matplotlib")
SOURCE_BYTES = Path(__file__).read_bytes()


class LabelMovie:
    def __init__(self, path, frames=None, crop=None):
        import zarr
        self.path = Path(path)
        self.array = zarr.open_array(str(path), mode="r")
        if self.array.ndim not in [3, 4] or (self.array.ndim == 4 and self.array.shape[1] != 1):
            raise ValueError("label input must be tyx or tzyx with singleton z")
        self.frames = frames or self.array.shape[0]
        if not 0 < self.frames <= self.array.shape[0]:
            raise ValueError("invalid frame prefix")
        self.crop = crop or [0, 0, *self.array.shape[-2:]]
        y, x, h, w = self.crop
        if min(y, x) < 0 or min(h, w) <= 0 or y + h > self.array.shape[-2] or x + w > self.array.shape[-1]:
            raise ValueError("invalid crop")
        self.shape = (self.frames, h, w)
        self.ndim = 3
        self.dtype = self.array.dtype
        self.pixel_hashes = {}

    def __getitem__(self, t):
        import numpy as np
        if not 0 <= t < self.frames:
            raise IndexError(t)
        y, x, h, w = self.crop
        values = np.asarray(self.array[t, 0, y:y+h, x:x+w] if self.array.ndim == 4 else self.array[t, y:y+h, x:x+w])
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if t in self.pixel_hashes and self.pixel_hashes[t] != digest:
            raise ValueError("source labels changed during the experiment")
        self.pixel_hashes[t] = digest
        return values


def drift_field(spec, shape):
    """Ultrack shifts targets back to the preceding frame; normalized by Y/X."""
    import numpy as np
    offsets = {}
    for row in spec["detections"]:
        offset = np.asarray(row["native"][-2:]) - np.asarray(row["registered"][-2:]) / np.asarray(spec["spacing"][-2:])
        if row["t_index"] in offsets and not np.allclose(offsets[row["t_index"]], offset, rtol=0, atol=1e-7):
            raise ValueError("input registration is not a common translation per frame")
        offsets[row["t_index"]] = offset
    flow = np.zeros((shape[0], 2, 1, 1), dtype="f4")
    for t in range(1, shape[0]):
        if t not in offsets or t-1 not in offsets:
            raise ValueError("missing translation for a frame")
        flow[t, :, 0, 0] = -(offsets[t] - offsets[t-1]) / np.asarray(shape[-2:])
    if abs(flow).max() > 1:
        raise ValueError("translation exceeds normalized flow range")
    return flow


def load_run_config(path):
    from ultrack import MainConfig
    raw = json.loads(Path(path).read_text())
    names = {"data_config": "data", "segmentation_config": "segmentation", "linking_config": "linking", "tracking_config": "tracking"}
    for attribute, alias in names.items():
        if attribute in raw:
            if alias in raw:
                raise ValueError("ambiguous configuration aliases")
            raw[alias] = raw.pop(attribute)
    if set(raw) != set(names.values()):
        raise ValueError("configuration does not contain the four expected sections")
    return MainConfig.model_validate(raw)


def main():
    import numpy as np
    from ultrack import MainConfig, segment, link, to_tracks_layer, tracks_to_zarr
    from ultrack.imgproc.flow import add_flow
    from ultrack.core.solve.sqltracking import SQLTracking
    from ultrack.utils.edge import labels_to_contours
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-input", type=Path, required=True)
    parser.add_argument("--masks", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("Y", "X", "H", "W"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--min-frontier", type=float, default=0.0)
    parser.add_argument("--backend", choices=["cbc", "scip"], default="cbc")
    parser.add_argument("--seed-leaves", action="store_true")
    parser.add_argument("--solver-progress", action="store_true")
    parser.add_argument("--reuse-work", type=Path, help="fork hypotheses/links from an existing completed or terminal-failed run")
    parser.add_argument("--resume-solve", action="store_true", help="reuse this run database after a terminal export failure; rerun only the solve/export")
    args = parser.parse_args()
    if importlib.metadata.version("ultrack") != "0.8.0":
        raise ValueError("this adapter requires ultrack==0.8.0")
    if args.resume_solve and args.reuse_work:
        parser.error("resume-solve and reuse-work are mutually exclusive")
    if args.workers < 1 or args.timeout <= 0 or args.sigma < 0 or not math.isfinite(args.sigma):
        parser.error("invalid worker count, timeout or sigma")
    data = args.suite_input.read_bytes()
    spec = json.loads(data)
    if spec["schema"] != "blockflow.tracking-suite-input/v1" or [f["t_index"] for f in spec["frames"]] != list(range(len(spec["frames"]))):
        raise ValueError("expected suite input with a dense zero-based timeline")
    movies = [LabelMovie(path, args.frames, args.crop) for path in args.masks]
    if any(m.shape != movies[0].shape for m in movies):
        raise ValueError("hypothesis sources have different shapes")
    if any(m.array.shape[0] != len(spec["frames"]) or list(m.array.shape[-2:]) != spec["extent"][-2:] for m in movies):
        raise ValueError("source label arrays and suite geometry differ")
    if args.resume_solve:
        if (args.output / "suite-input.json").read_bytes() != data or not (args.output / "config.json").is_file():
            raise ValueError("cannot resume: prior source/configuration do not match")
        (args.output / ("resumed-adapter-" + hashlib.sha256(SOURCE_BYTES).hexdigest()[:12] + ".py")).write_bytes(SOURCE_BYTES)
    else:
        args.output.mkdir()
        (args.output / "adapter.py").write_bytes(SOURCE_BYTES)
        (args.output / "suite-input.json").write_bytes(data)
    work = args.output / "work"
    work.mkdir(exist_ok=args.resume_solve)
    if args.reuse_work:
        import sqlite3
        import shutil
        if (args.reuse_work / "suite-input.json").read_bytes() != data:
            raise ValueError("reused hypotheses came from a different suite input")
        source_config = load_run_config(args.reuse_work / "config.json")
        source_work = source_config.data_config.working_dir
        with sqlite3.connect("file:" + str(source_work / "data.db") + "?mode=ro", uri=True) as source_db:
            with sqlite3.connect(str(work / "data.db")) as target_db:
                source_db.backup(target_db)
        shutil.copyfile(source_work / "metadata.toml", work / "metadata.toml")
    config = MainConfig()
    config.data_config.working_dir = work.resolve()
    config.data_config.n_workers = args.workers
    config.segmentation_config.n_workers = args.workers
    config.segmentation_config.min_area = 50
    config.segmentation_config.max_area = 50000
    config.segmentation_config.min_frontier = args.min_frontier
    config.linking_config.n_workers = args.workers
    config.linking_config.max_distance = spec["max_distance"]
    config.linking_config.max_neighbors = 10
    config.tracking_config.solver_name = "CBC"
    config.tracking_config.n_threads = args.workers
    config.tracking_config.time_limit = args.timeout
    config.tracking_config.solution_gap = .01
    if args.resume_solve or args.reuse_work:
        previous = load_run_config((args.reuse_work or args.output) / "config.json")
        previous.data_config.working_dir = work.resolve()
        previous.tracking_config.time_limit = args.timeout
        if previous.model_dump() != config.model_dump():
            raise ValueError("cannot resume: configuration changed")
        segmentation_seconds = linking_seconds = None
        (args.output / "config.json").write_text(config.model_dump_json(indent=2, by_alias=True))
        print("Reusing a stored hypothesis/link database", flush=True)
    else:
        (args.output / "config.json").write_text(config.model_dump_json(indent=2, by_alias=True))
        start = time.monotonic()
        foreground, contours = labels_to_contours(movies, sigma=args.sigma or None,
            foreground_store_or_path=str(args.output / "foreground.zarr"),
            contours_store_or_path=str(args.output / "contours.zarr"))
        print("Converted source labels to foreground/contours", flush=True)
        segment(config=config, foreground=foreground, contours=contours)
        segmentation_seconds = time.monotonic() - start
        print("Segmentation hypotheses complete", flush=True)
        add_flow(config=config, vector_field=drift_field(spec, movies[0].shape))
        start = time.monotonic()
        link(config=config, scale=spec["spacing"][-2:])
        linking_seconds = time.monotonic() - start
        print("Drift-aware linking complete", flush=True)
    start = time.monotonic()
    tracker = SQLTracking(config)
    if tracker.num_batches != 1:
        raise ValueError("this first adapter expects a single whole-movie solve")
    if args.resume_solve or args.reuse_work:
        tracker.reset_solution()
    if args.backend == "scip":
        from mip_scip_bridge import solve_with_scip, ultrack_selected_dataframe, ultrack_leaf_seed
        bridge_path = Path(__file__).with_name("mip_scip_bridge.py")
        (args.output / "mip_scip_bridge.py").write_bytes(bridge_path.read_bytes())
        tracker.construct_model(index=0)
        warm_start = seed_info = None
        if args.seed_leaves:
            warm_start, seed_info = ultrack_leaf_seed(tracker.solver, work / "data.db")
            print(f"Verified greedy seed: {seed_info}", flush=True)
        values, solver = solve_with_scip(tracker.solver._model, args.output / "tracking.mps",
                                         timeout=args.timeout, relative_gap=config.tracking_config.solution_gap,
                                         threads=args.workers, warm_start=warm_start, verbose=args.solver_progress)
        solver["primal_seed"] = seed_info
        (args.output / "solver-attempt.json").write_text(json.dumps(solver, indent=2, allow_nan=False) + "\n")
        selection = ultrack_selected_dataframe(tracker.solver, values)
        tracker.solver.solution = lambda: selection.copy()
        tracker.add_solution(index=0)
    else:
        tracker(index=0)
        model = tracker.solver._model  # Version-pinned diagnostic access.
        solver = {"backend": "CBC", "status": str(model.status), "objective": model.objective_value,
                  "bound": model.objective_bound, "gap": model.gap, "solutions": model.num_solutions,
                  "timeout_seconds": args.timeout}
        if model.num_solutions < 1:
            raise RuntimeError("Ultrack returned no feasible hypothesis selection")
    solve_seconds = time.monotonic() - start
    (args.output / "solver.json").write_text(json.dumps(solver, indent=2, allow_nan=False) + "\n")
    tracks, graph = to_tracks_layer(config)
    tracks.to_csv(args.output / "tracks.csv", index=False)
    (args.output / "track-parents.json").write_text(json.dumps({str(k): [int(parent)] for k, parent in graph.items()}, indent=2) + "\n")
    array = tracks_to_zarr(config=config, tracks_df=tracks, store_or_path=str(args.output / "segments.zarr"),
                           chunks=(1, 256, 256))
    array.attrs.update({"tracking_suite_role": "new segmentation/track identities; do not join old feature rows by ID",
                        "crop_yxhw": movies[0].crop, "source_channel": "Phase-derived masks only"})
    spec_by_t = defaultdict_ids(spec)
    output_counts = []
    for t in range(movies[0].shape[0]):
        labels = np.asarray(array[t])
        source = movies[0][t]
        output_counts.append({"t": t, "segments": int(np.unique(labels).size - int((labels == 0).any())),
                              "foreground_pixel_difference": int(np.count_nonzero((labels > 0) != (source > 0)))})
        if not args.crop and set(int(x) for x in np.unique(source) if x) != spec_by_t[t]:
            raise ValueError("original source mask identities disagree with suite input")
    summary = {"schema": "blockflow.ultrack-experiment/v1", "complete": True, "ultrack": "0.8.0",
        "source_sha256": hashlib.sha256(SOURCE_BYTES).hexdigest(), "input_sha256": hashlib.sha256(data).hexdigest(),
        "sources": [{"array": str(m.path.resolve()), "pixel_sha256_by_t": m.pixel_hashes} for m in movies],
        "shape": movies[0].shape, "crop_yxhw": movies[0].crop, "sigma": args.sigma,
        "hypothesis_sources": len(movies), "resumed_solve": args.resume_solve,
        "reused_work_from": str(args.reuse_work) if args.reuse_work else None, "solver": solver, "selected_detections": len(tracks),
        "tracks": int(tracks["track_id"].nunique()), "division_parents": len({int(parent) for parent in graph.values()}),
        "frames": output_counts, "segmentation_seconds": segmentation_seconds,
        "linking_seconds": linking_seconds, "solve_seconds": solve_seconds,
        "reference_evaluation": None, "validation": "diagnostic only; changed masks require spatial matching to a reviewed reference"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{solver['status']}: {len(tracks)} detections, {summary['tracks']} tracks", flush=True)


def defaultdict_ids(spec):
    from collections import defaultdict
    result = defaultdict(set)
    for row in spec["detections"]:
        result[row["t_index"]].add(row["id"])
    return result


if __name__ == "__main__":
    main()
