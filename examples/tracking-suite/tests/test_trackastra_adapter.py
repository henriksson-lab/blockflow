import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from types import SimpleNamespace

import networkx as nx

path = Path(__file__).resolve().parents[1] / "scripts/trackastra_adapter.py"
spec = importlib.util.spec_from_file_location("trackastra_adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def fixture(self):
        base = 2**60
        return {
            "max_distance": 10,
            "nodes": [
                {"id": base + 1, "t_index": 0, "registered": [0, 10, 10]},
                {"id": base + 2, "t_index": 1, "registered": [0, 10, 9]},
                {"id": base + 3, "t_index": 1, "registered": [0, 10, 11]},
                {"id": base + 4, "t_index": 1, "registered": [0, 10, 40]},
            ],
            "edges": [
                {"from": base + 1, "to": base + 2, "weight": .9},
                {"from": base + 1, "to": base + 3, "weight": .9},
                {"from": base + 1, "to": base + 4, "weight": .99},
            ],
        }

    def test_physical_gate_and_large_native_identities(self):
        cache = self.fixture()
        graph = adapter.make_graph(cache)
        self.assertEqual(set(graph), {d["id"] for d in cache["nodes"]})
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(graph.out_degree(2**60 + 1), 2)
        self.assertEqual(graph.in_degree(2**60 + 4), 0)

    def test_invalid_scores_and_graphs_are_rejected(self):
        cache = self.fixture()
        cache["edges"][0]["weight"] = float("nan")
        with self.assertRaises(ValueError):
            adapter.make_graph(cache)
        graph = adapter.make_graph(self.fixture())
        missing = graph.copy()
        missing.remove_node(2**60 + 4)
        with self.assertRaises(ValueError):
            adapter.validate_solution(graph, missing, True)
        with self.assertRaises(ValueError):
            adapter.validate_solution(graph, graph, False)

    def test_absent_baseline_is_not_an_empty_reference(self):
        cache = self.fixture()
        cache["schema"] = "blockflow.trackastra-predictions/v1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions"
            predictions.mkdir()
            adapter.save(predictions / "predictions.json", cache)
            adapter.save(predictions / "complete.json", {"predictions_sha256": adapter.digest(predictions / "predictions.json")})
            output = root / "decoded"
            adapter.decode(SimpleNamespace(predictions=predictions, output=output, timeout=5, threads=1, split_cost=.25))
            summary = json.loads((output / "summary.json").read_text())
            for run in summary["runs"]:
                self.assertIsNone(run["baseline_edge_difference"])
                self.assertIsNone(run["reference_evaluation"])

    def test_scip_joint_daughters_and_fixed_isolated_nodes(self):
        graph = adapter.make_graph(self.fixture())
        solution, info = adapter.ilp_decode(graph, True, timeout=5, threads=1)
        adapter.validate_solution(graph, solution, True)
        self.assertEqual(solution.out_degree(2**60 + 1), 2)
        self.assertIn(2**60 + 4, solution)
        self.assertEqual(info["status"], "OPTIMAL")
        no_divisions, _ = adapter.ilp_decode(graph, False, timeout=5, threads=1)
        adapter.validate_solution(graph, no_divisions, False)
        self.assertEqual(no_divisions.out_degree(2**60 + 1), 1)


if __name__ == "__main__":
    unittest.main()
