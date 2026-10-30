import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

path = Path(__file__).resolve().parents[1] / "scripts/ultrack_adapter.py"
spec = importlib.util.spec_from_file_location("ultrack_adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class UltrackAdapterTests(unittest.TestCase):
    def test_solver_bridge_preserves_max_and_min_objectives(self):
        import mip
        bridge_path = path.with_name("mip_scip_bridge.py")
        bridge_spec = importlib.util.spec_from_file_location("mip_scip_bridge", bridge_path)
        bridge = importlib.util.module_from_spec(bridge_spec)
        bridge_spec.loader.exec_module(bridge)
        with tempfile.TemporaryDirectory() as directory:
            for maximize in [True, False]:
                model = mip.Model(solver_name=mip.CBC)
                model.verbose = 0
                x = model.add_var(name="node(0)", var_type=mip.BINARY)
                y = model.add_var(name="node(1)", var_type=mip.BINARY)
                model += x + y <= 1
                model += x - y >= 0
                model.objective = mip.maximize(3*x - 2*y + 2) if maximize else mip.minimize(-3*x + 2*y + 2)
                values, report = bridge.solve_with_scip(model, Path(directory) / f"problem-{maximize}.mps", timeout=5, relative_gap=0)
                self.assertEqual(values[x.name], 1)
                self.assertEqual(values[y.name], 0)
                self.assertEqual(report["objective"], 5 if maximize else -1)
                self.assertEqual(report["original_model_max_violation"], 0)

    def test_target_shift_has_correct_sign_and_normalization(self):
        spec = {"spacing": [1, 2, 2], "detections": [
            {"t_index": 0, "native": [0, 10, 10], "registered": [0, 20, 20]},
            {"t_index": 1, "native": [0, 20, 7], "registered": [0, 20, 20]},
        ]}
        field = adapter.drift_field(spec, (2, 100, 200))
        np.testing.assert_allclose(field[1, :, 0, 0], [-.1, .015])
        shifted = np.asarray([20, 7]) + field[1, :, 0, 0] * [100, 200]
        np.testing.assert_allclose(shifted, [10, 10], atol=1e-6)

    def test_label_proxy_preserves_exact_ids_and_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.zarr"
            data = zarr.open_array(str(path), mode="w", shape=(2, 1, 12, 12), dtype="u8")
            data[0, 0, 4, 5] = 2**60 + 3
            movie = adapter.LabelMovie(path, frames=1, crop=[3, 4, 5, 6])
            self.assertEqual(movie.shape, (1, 5, 6))
            self.assertEqual(int(movie[0][1, 1]), 2**60 + 3)
            self.assertEqual(int(data[0, 0, 4, 5]), 2**60 + 3)
            data[0, 0, 4, 5] = 2**60 + 4
            with self.assertRaises(ValueError):
                movie[0]

    def test_configuration_aliases_round_trip_without_falling_back_to_defaults(self):
        from ultrack import MainConfig
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = MainConfig()
            config.data_config.working_dir = Path(directory)
            config.data_config.n_workers = 2
            config.linking_config.max_distance = 37.0
            config.tracking_config.solver_name = "CBC"
            for aliases in [True, False]:
                path.write_text(config.model_dump_json(by_alias=aliases))
                loaded = adapter.load_run_config(path)
                self.assertEqual(loaded.model_dump(), config.model_dump())
                self.assertEqual(loaded.data_config.working_dir, Path(directory))
                self.assertEqual(loaded.linking_config.max_distance, 37.0)
            values = json.loads(path.read_text())
            values["data"] = values["data_config"]
            path.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                adapter.load_run_config(path)


if __name__ == "__main__":
    unittest.main()
