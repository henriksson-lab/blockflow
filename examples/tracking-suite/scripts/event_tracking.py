#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed-mask event assignment with Brownian motion and joint daughter scoring.

An experimental row-only model. No fluorescence or source image is opened.
Scoring, event generation and optimization are separate callable components.
Shape factors are provisional energies; posterior calibration is not claimed.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Protocol


SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


@dataclass(frozen=True)
class Parameters:
    diffusion_um2_per_hour: float = 8.0
    localization_sigma_um: float = 1.5
    mean_cycle_hours: float = 24.0
    observation_probability: float = 0.95
    new_detection_probability: float = 0.05
    border_probability: float = 0.2
    border_um: float = 25.0
    continuation_log_size_sigma: float = 0.65
    continuation_size_weight: float = 1.0
    division_log_sum_sigma: float = 0.7
    division_log_balance_sigma: float = 1.2
    division_log_separation_sigma: float = 0.8
    division_separation_scale: float = 0.75
    division_max_separation_um: float = 60.0
    division_feature_weight: float = 1.0

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"non-finite model parameter {name}")
            if name.endswith("probability"):
                if not 0 < value < 1:
                    raise ValueError(f"{name} must be in (0,1)")
            elif name in {"diffusion_um2_per_hour", "continuation_size_weight", "division_feature_weight", "border_um"}:
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Event:
    key: str
    kind: str
    parents: tuple
    children: tuple
    cost: float
    cost_terms: dict
    features: dict


class EventScorer(Protocol):
    def score(self, kind, parents, children): ...
    def parameters(self): ...


def validate_input(spec):
    if spec["schema"] != "blockflow.tracking-suite-input/v1":
        raise ValueError("unsupported suite input")
    frames = spec["frames"]
    if not frames or any(b["t_index"] <= a["t_index"] for a, b in zip(frames, frames[1:])):
        raise ValueError("acquired frame indices must be strictly increasing")
    if any(f["elapsed_seconds"] is None or not math.isfinite(f["elapsed_seconds"]) or f["elapsed_seconds"] < 0 for f in frames):
        raise ValueError("Brownian model requires finite timestamps on every frame")
    if any(b["elapsed_seconds"] <= a["elapsed_seconds"] for a, b in zip(frames, frames[1:])):
        raise ValueError("timestamps must strictly increase")
    if spec["extent"][0] != 1 or len(spec["extent"]) != 3:
        raise ValueError("this first event model supports 2D Phase geometry")
    if any(n <= 0 for n in spec["extent"]) or any(not math.isfinite(x) or x <= 0 for x in spec["spacing"]):
        raise ValueError("invalid series geometry")
    if not math.isfinite(spec["max_distance"]) or spec["max_distance"] <= 0:
        raise ValueError("invalid candidate gate")
    times = {f["t_index"] for f in frames}
    ids = set()
    for row in spec["detections"]:
        identity = row["id"]
        if type(identity) is not int or not 0 < identity < 2**64 or identity in ids:
            raise ValueError("detection IDs must be distinct positive uint64 integers")
        ids.add(identity)
        if row["t_index"] not in times or not math.isfinite(row["size"]) or row["size"] <= 0:
            raise ValueError("invalid detection time or mask size")
        if len(row["native"]) != 3 or len(row["registered"]) != 3 or row["native"][0] != 0 or row["registered"][0] != 0:
            raise ValueError("this event model requires 2D [0,y,x] coordinates")
        if any(not math.isfinite(x) for x in row["native"] + row["registered"]):
            raise ValueError("non-finite detection position")


class BrownianJointScorer:
    """Analytic Brownian density ratio plus explicit provisional shape factors.

    The lifetime prior is exponential (memoryless), NOT a fitted age-dependent
    hazard. New tracks have unknown birth age. Temporal age state is deferred.
    """
    def __init__(self, spec, params):
        params.validate()
        self.spec, self.p = spec, params
        self.frames = {f["t_index"]: f for f in spec["frames"]}
        self.first, self.last = spec["frames"][0]["t_index"], spec["frames"][-1]["t_index"]
        area = math.prod(n * scale for n, scale in zip(spec["extent"][-2:], spec["spacing"][-2:]))
        count = Counter(d["t_index"] for d in spec["detections"])
        self.density = {t: max(count[t], 1) / area for t in self.frames}

    def parameters(self):
        return {"name": "brownian-joint-division-v1", **asdict(self.p),
                "density": "target detections / physical field area",
                "age_prior": "constant exponential hazard; no age inference",
                "shape_factors": "provisional quadratic energies, not calibrated likelihood ratios"}

    def near_border(self, row):
        distances = [min(x + .5, n - .5 - x) * scale for x, n, scale in
                     zip(row["native"][-2:], self.spec["extent"][-2:], self.spec["spacing"][-2:])]
        return min(distances) <= self.p.border_um

    def motion(self, source, target_coords, target_t):
        dt = (self.frames[target_t]["elapsed_seconds"] - self.frames[source["t_index"]]["elapsed_seconds"]) / 3600
        variance = 2 * self.p.diffusion_um2_per_hour * dt + 2 * self.p.localization_sigma_um**2
        distance_squared = sum((a - b)**2 for a, b in zip(source["registered"][-2:], target_coords))
        # -log Gaussian displacement density / uniform clutter density, in 2D.
        energy = distance_squared / (2 * variance) + math.log(2 * math.pi * variance * self.density[target_t])
        return energy, dt, {"squared_displacement_um2": distance_squared, "variance_per_axis_um2": variance}

    def score(self, kind, parents, children):
        features = {}
        if kind == "birth":
            child = children[0]
            probability = self.p.border_probability if self.near_border(child) else self.p.new_detection_probability
            terms = {"birth": 0.0 if child["t_index"] == self.first else -math.log(probability)}
        elif kind == "end":
            parent = parents[0]
            probability = self.p.border_probability if self.near_border(parent) else 1 - self.p.observation_probability
            terms = {"end": 0.0 if parent["t_index"] == self.last else -math.log(probability)}
        elif kind == "continue":
            parent, child = parents[0], children[0]
            energy, dt, features = self.motion(parent, child["registered"][-2:], child["t_index"])
            log_growth = math.log(2) * dt / self.p.mean_cycle_hours
            log_ratio = math.log(child["size"] / parent["size"])
            terms = {"motion_density_ratio": energy,
                     "continuation_prior": dt / self.p.mean_cycle_hours - math.log(self.p.observation_probability),
                     "log_size": self.p.continuation_size_weight * .5 * ((log_ratio - log_growth) / self.p.continuation_log_size_sigma)**2}
            features.update(log_size_ratio=log_ratio, elapsed_hours=dt)
        elif kind == "divide":
            parent, a, b = parents[0], children[0], children[1]
            center = [(x + y) / 2 for x, y in zip(a["registered"][-2:], b["registered"][-2:])]
            energy, dt, features = self.motion(parent, center, a["t_index"])
            probability = -math.expm1(-dt / self.p.mean_cycle_hours)
            log_sum = math.log((a["size"] + b["size"]) / parent["size"])
            log_balance = math.log(a["size"] / b["size"])
            separation = math.dist(a["registered"][-2:], b["registered"][-2:])
            diameter = 2 * math.sqrt(parent["size"] * math.prod(self.spec["spacing"][-2:]) / math.pi)
            expected = self.p.division_separation_scale * diameter
            log_separation = math.log(max(separation, 1e-6) / expected)
            weight = self.p.division_feature_weight
            terms = {"joint_center_motion": energy,
                     "division_prior": -math.log(probability) - 2 * math.log(self.p.observation_probability),
                     "sum_area": weight * .5 * (log_sum / self.p.division_log_sum_sigma)**2,
                     "daughter_balance": weight * .5 * (log_balance / self.p.division_log_balance_sigma)**2,
                     "daughter_separation": weight * .5 * (log_separation / self.p.division_log_separation_sigma)**2}
            features.update(log_sum_area_ratio=log_sum, log_daughter_area_ratio=log_balance,
                            daughter_separation_um=separation, expected_separation_um=expected,
                            elapsed_hours=dt, center="unweighted daughter midpoint")
        else:
            raise ValueError(f"unknown event kind {kind}")
        cost = sum(terms.values())
        if not math.isfinite(cost):
            raise ValueError("event scorer returned non-finite energy")
        return cost, terms, features


def generate_events(spec, scorer: EventScorer, allow_divisions=True, max_separation=60):
    from scipy.spatial import cKDTree
    validate_input(spec)
    by_frame = defaultdict(list)
    for row in sorted(spec["detections"], key=lambda row: (row["t_index"], row["id"])):
        by_frame[row["t_index"]].append(row)
    events = []
    def append(kind, parents, children):
        cost, terms, features = scorer.score(kind, parents, children)
        key = f"{kind}:" + ",".join(str(p["id"]) for p in parents) + ":" + ",".join(str(c["id"]) for c in children)
        events.append(Event(key, kind, tuple(p["id"] for p in parents), tuple(c["id"] for c in children), cost, terms, features))
    for row in sorted(spec["detections"], key=lambda row: row["id"]):
        append("birth", [], [row])
        append("end", [row], [])
    for first, second in zip(spec["frames"], spec["frames"][1:]):
        if second["t_index"] != first["t_index"] + 1:
            continue
        targets = by_frame[second["t_index"]]
        if not targets:
            continue
        tree = cKDTree([r["registered"][-2:] for r in targets])
        for source in by_frame[first["t_index"]]:
            neighbors = [targets[index] for index in sorted(tree.query_ball_point(source["registered"][-2:], spec["max_distance"]))]
            for target in neighbors:
                append("continue", [source], [target])
            if allow_divisions:
                for a, b in itertools.combinations(neighbors, 2):
                    if math.dist(a["registered"][-2:], b["registered"][-2:]) <= max_separation:
                        append("divide", [source], sorted([a, b], key=lambda row: row["id"]))
    return events


def validate_event_graph(spec, events):
    validate_input(spec)
    rows = {r["id"]: r for r in spec["detections"]}
    keys = set()
    for event in events:
        expected = {"birth": (0, 1), "end": (1, 0), "continue": (1, 1), "divide": (1, 2)}
        if event.kind not in expected or (len(event.parents), len(event.children)) != expected[event.kind]:
            raise ValueError("event has invalid kind/cardinality")
        if not event.key or event.key in keys or not math.isfinite(event.cost):
            raise ValueError("event keys must be unique/nonempty and energies finite")
        keys.add(event.key)
        if len(set(event.parents + event.children)) != len(event.parents + event.children):
            raise ValueError("event repeats a detection")
        if any(n not in rows for n in event.parents + event.children):
            raise ValueError("event references unknown detection")
        if event.parents and event.children:
            t = rows[event.parents[0]]["t_index"]
            if any(rows[n]["t_index"] != t + 1 for n in event.children):
                raise ValueError("event must link to the next frame index")
        if any(not math.isfinite(v) for v in event.cost_terms.values()) or not math.isclose(sum(event.cost_terms.values()), event.cost, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("event energy disagrees with its recorded terms")


def validate_selected(spec, events):
    ids = {r["id"] for r in spec["detections"]}
    incoming, outgoing = Counter(), Counter()
    for event in events:
        incoming.update(event.children)
        outgoing.update(event.parents)
    if set(incoming) != ids or set(outgoing) != ids or any(n != 1 for n in incoming.values()) or any(n != 1 for n in outgoing.values()):
        raise ValueError("selected events do not give each fixed detection one incoming/outgoing event")


def solve_events(spec, events, timeout=120, threads=1):
    from pyscipopt import Model, quicksum
    validate_event_graph(spec, events)
    ids = {r["id"] for r in spec["detections"]}
    model = Model("blockflow-joint-events")
    model.hideOutput()
    model.setRealParam("limits/time", timeout)
    model.setIntParam("parallel/maxnthreads", threads)
    variables = [model.addVar(vtype="B", name=e.key, obj=e.cost) for e in events]
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for event, variable in zip(events, variables):
        if any(n not in ids for n in event.parents + event.children):
            raise ValueError("event references unknown detection")
        for node in event.parents:
            outgoing[node].append(variable)
        for node in event.children:
            incoming[node].append(variable)
    for node in sorted(ids):
        model.addCons(quicksum(incoming[node]) == 1)
        model.addCons(quicksum(outgoing[node]) == 1)
    model.setMinimize()
    start = time.monotonic()
    model.optimize()
    optimize_seconds = time.monotonic() - start
    solution = model.getBestSol()
    if solution is None:
        raise RuntimeError(f"SCIP returned no feasible solution: {model.getStatus()}")
    selected = [event for event, variable in zip(events, variables) if model.getSolVal(solution, variable) > .5]
    validate_selected(spec, selected)
    objective = sum(e.cost for e in selected)
    if not math.isclose(objective, model.getSolObjVal(solution), rel_tol=1e-8, abs_tol=1e-5):
        raise ValueError("selected event objective disagrees with solver")
    return selected, {"backend": "SCIP/PySCIPOpt", "status": str(model.getStatus()),
                      "objective": objective, "primal_bound": model.getPrimalbound(), "dual_bound": model.getDualbound(),
                      "gap": model.getGap(), "optimize_seconds": optimize_seconds, "timeout_seconds": timeout,
                      "requested_threads": threads, "note": "without history-dependent factors this model factorizes over frame pairs"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diffusion", type=float, default=8)
    parser.add_argument("--no-divisions", action="store_true")
    parser.add_argument("--size-weight", type=float, default=1)
    parser.add_argument("--division-feature-weight", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--shape-features", type=Path)
    parser.add_argument("--mother-elongation-weight", type=float, default=0)
    parser.add_argument("--relax-projected-area", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout) or args.threads < 1:
        parser.error("finite positive timeout and positive threads required")
    data = args.suite_input.read_bytes()
    spec = json.loads(data)
    validate_input(spec)
    parameters = Parameters(diffusion_um2_per_hour=args.diffusion, continuation_size_weight=args.size_weight,
                            division_feature_weight=args.division_feature_weight)
    scorer = BrownianJointScorer(spec, parameters)
    shape_source = None
    if args.shape_features is not None:
        from shape_event_tracking import ShapeEventScorer, SOURCE_BYTES as SHAPE_SOURCE_BYTES
        scorer = ShapeEventScorer(scorer, spec, json.loads(args.shape_features.read_bytes()), hashlib.sha256(data).hexdigest(), args.mother_elongation_weight, args.relax_projected_area)
        shape_source = SHAPE_SOURCE_BYTES
    elif args.mother_elongation_weight != 0 or args.relax_projected_area:
        parser.error("shape policies require --shape-features")
    start = time.monotonic()
    events = generate_events(spec, scorer, allow_divisions=not args.no_divisions,
                             max_separation=parameters.division_max_separation_um)
    generation_seconds = time.monotonic() - start
    args.output.mkdir()
    (args.output / "event_tracking.py").write_bytes(SOURCE_BYTES)
    if shape_source is not None:
        (args.output / "shape_event_tracking.py").write_bytes(shape_source)
        (args.output / "shape-features.json").write_bytes(args.shape_features.read_bytes())
    (args.output / "suite-input.json").write_bytes(data)
    write_json(args.output / "event-graph.json", {"schema": "blockflow.tracking-events/v1",
               "input_sha256": hashlib.sha256(data).hexdigest(),
               "nodes": [{"id": r["id"], "t_index": r["t_index"]} for r in spec["detections"]],
               "model": scorer.parameters(), "events": [asdict(e) for e in events]})
    print(f"Generated {len(events)} events: {dict(Counter(e.kind for e in events))}", flush=True)
    start = time.monotonic()
    selected, solver = solve_events(spec, events, args.timeout, args.threads)
    decode_seconds = time.monotonic() - start
    links = []
    for event in selected:
        if event.kind in {"continue", "divide"}:
            links.extend({"from": event.parents[0], "to": child, "dt": 1, "kind": event.kind,
                          "event_key": event.key, "event_cost": event.cost} for child in event.children)
    links.sort(key=lambda link: (link["from"], link["to"]))
    baseline = {tuple(pair) for pair in spec.get("baseline_pairs") or []}
    predicted = {(r["from"], r["to"]) for r in links}
    write_json(args.output / "selected-events.json", [asdict(e) for e in selected])
    write_json(args.output / "links.json", links)
    write_json(args.output / "summary.json", {"schema": "blockflow.event-tracking-result/v1", "complete": True,
        "input_sha256": hashlib.sha256(data).hexdigest(), "source_sha256": SOURCE_SHA256,
        "model": scorer.parameters(), "fixed_detections": len(spec["detections"]),
        "candidate_events": dict(Counter(e.kind for e in events)), "selected_events": dict(Counter(e.kind for e in selected)),
        "links": len(links), "generation_seconds": generation_seconds, "decode_seconds": decode_seconds, "solver": solver,
        "baseline_edge_difference": {"added": len(predicted - baseline), "removed": len(baseline - predicted)},
        "reference_evaluation": None, "validation": "diagnostics only; provisional energies, no calibrated accuracy asserted"})
    print(f"{solver['status']}: {len(links)} links, {Counter(e.kind for e in selected)['divide']} divisions; {decode_seconds:.3f}s total decode", flush=True)


if __name__ == "__main__":
    main()
