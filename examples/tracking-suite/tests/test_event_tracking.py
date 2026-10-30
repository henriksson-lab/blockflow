import dataclasses
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/event_tracking.py"
spec = importlib.util.spec_from_file_location("event_tracking", path)
events_api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = events_api
spec.loader.exec_module(events_api)


def fixture(two_sources=False):
    ids = [2**60 + n for n in range(1, 5)]
    source = dict(id=ids[0], t_index=0, native=[0, 50, 50], registered=[0, 50, 50], size=100)
    daughters = [dict(id=ids[1], t_index=1, native=[0, 50, 48], registered=[0, 50, 48], size=50),
                 dict(id=ids[2], t_index=1, native=[0, 50, 52], registered=[0, 50, 52], size=50)]
    if two_sources:
        detections = [source, dict(id=ids[3], t_index=0, native=[0, 52, 50], registered=[0, 52, 50], size=100), *daughters]
    else:
        detections = [source, *daughters, dict(id=ids[3], t_index=1, native=[0, 50, 49], registered=[0, 50, 49], size=5)]
    return dict(schema="blockflow.tracking-suite-input/v1", extent=[1, 100, 100], spacing=[1, 1, 1],
                frames=[dict(t_index=0, elapsed_seconds=0), dict(t_index=1, elapsed_seconds=7200)],
                detections=detections, max_distance=10)


class EventTrackingTests(unittest.TestCase):
    def test_brownian_variance_uses_elapsed_time_and_localization(self):
        spec = fixture()
        model = events_api.BrownianJointScorer(spec, events_api.Parameters())
        source = spec["detections"][0]
        energy, dt, features = model.motion(source, [50, 50], 1)
        self.assertEqual(dt, 2)
        self.assertEqual(features["variance_per_axis_um2"], 36.5)
        self.assertAlmostEqual(energy, math.log(2 * math.pi * 36.5 * (3 / 10000)))
        spec["frames"][1]["elapsed_seconds"] = 14400
        model = events_api.BrownianJointScorer(spec, events_api.Parameters())
        self.assertEqual(model.motion(source, [50, 50], 1)[2]["variance_per_axis_um2"], 68.5)

    def test_joint_event_prefers_two_daughters_over_a_tiny_neighbor(self):
        spec = fixture()
        parameters = events_api.Parameters(border_um=0)
        model = events_api.BrownianJointScorer(spec, parameters)
        events = events_api.generate_events(spec, model)
        selected, result = events_api.solve_events(spec, events, timeout=5)
        divisions = [e for e in selected if e.kind == "divide"]
        self.assertEqual(len(divisions), 1)
        self.assertEqual(divisions[0].children, (2**60 + 2, 2**60 + 3))
        self.assertEqual(result["status"], "optimal")
        events_api.validate_selected(spec, selected)

    def test_sparse_ilp_matches_independent_exhaustive_event_selection(self):
        spec = fixture(two_sources=True)
        model = events_api.BrownianJointScorer(spec, events_api.Parameters())
        random_source = random.Random(37)
        events = [dataclasses.replace(event, cost=(cost := random_source.uniform(-2, 3)), cost_terms={"fixture": cost})
                  for event in events_api.generate_events(spec, model)]
        nodes = {r["id"] for r in spec["detections"]}
        best = float("inf")
        for choices in itertools.product([False, True], repeat=len(events)):
            selected = [event for event, chosen in zip(events, choices) if chosen]
            if all(sum(node in e.parents for e in selected) == 1 and sum(node in e.children for e in selected) == 1 for node in nodes):
                best = min(best, sum(e.cost for e in selected))
        _, result = events_api.solve_events(spec, events, timeout=5)
        self.assertAlmostEqual(result["objective"], best)

    def test_invalid_joint_events_are_rejected(self):
        spec = fixture()
        model = events_api.BrownianJointScorer(spec, events_api.Parameters())
        events = events_api.generate_events(spec, model)
        division = next(e for e in events if e.kind == "divide")
        repeated = dataclasses.replace(division, children=(division.children[0], division.children[0]))
        with self.assertRaises(ValueError):
            events_api.solve_events(spec, [repeated])
        inconsistent = dataclasses.replace(division, cost=division.cost + 1)
        with self.assertRaises(ValueError):
            events_api.solve_events(spec, [inconsistent])
        with self.assertRaises(ValueError):
            events_api.Parameters(diffusion_um2_per_hour=-1).validate()


if __name__ == "__main__":
    unittest.main()
