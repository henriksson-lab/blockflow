"""Synthetic integrity checks; no biological validation or held-out inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import importlib.util

path=Path(__file__).resolve().parents[1]/"scripts/run_frozen_comparison.py"
spec=importlib.util.spec_from_file_location("frozen_comparison",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
preflight=module.preflight


class FreezeTests(unittest.TestCase):
    def fixture(self,root):
        capsule=root/'capsule';capsule.mkdir()
        code=capsule/'method.py';code.write_text('frozen test method\n')
        spec=dict(runs=[],max_distance=40.,reference=None,detections=[])
        input_path=root/'input.json';input_path.write_text(json.dumps(spec))
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        case=dict(id='synthetic-development',input_sha256=digest(input_path),source_pixels=[])
        plan=dict(schema='blockflow.frozen-comparison/v1',files={'method.py':digest(code)},packages={},
                  cases=[case],suite_runs=[],max_distance=40.,model_files_sha256={})
        return capsule,code,input_path,plan,digest

    def test_unlisted_input_and_changed_manifest_or_code_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);capsule,code,inp,plan,digest=self.fixture(root)
            manifest=capsule/'manifest.json';manifest.write_text(json.dumps(plan))
            _,case,actual=preflight(capsule,inp,expected_manifest=digest(manifest))
            self.assertEqual(case['id'],'synthetic-development')
            self.assertEqual(actual,digest(manifest))
            other=root/'heldout.json';other.write_text(json.dumps(dict(runs=[],max_distance=40.,reference=None,detections=[],site='unlisted')))
            with self.assertRaisesRegex(ValueError,'not listed'):
                preflight(capsule,other)
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                preflight(capsule,inp,expected_manifest='wrong checksum')
            code.write_text('changed after freeze\n')
            with self.assertRaisesRegex(ValueError,'changed or missing'):
                preflight(capsule,inp)

    def test_reused_cache_must_match_frozen_cache_model_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);capsule,_,inp,plan,digest=self.fixture(root)
            cache=root/'predictions';cache.mkdir()
            payload=dict(schema='blockflow.trackastra-predictions/v1',input_sha256=digest(inp),nodes=[],max_distance=40.,
                         provenance=dict(model_files_sha256={},source_pixels=[]))
            (cache/'predictions.json').write_text(json.dumps(payload))
            (cache/'complete.json').write_text(json.dumps(dict(predictions_sha256=digest(cache/'predictions.json'))))
            plan['cases'][0]['predictions_sha256']=digest(cache/'predictions.json')
            (capsule/'manifest.json').write_text(json.dumps(plan))
            preflight(capsule,inp,cache)
            payload['provenance']['source_pixels']=[{'different':'source'}]
            (cache/'predictions.json').write_text(json.dumps(payload))
            (cache/'complete.json').write_text(json.dumps(dict(predictions_sha256=digest(cache/'predictions.json'))))
            with self.assertRaisesRegex(ValueError,'differs from the frozen'):
                preflight(capsule,inp,cache)


if __name__=='__main__':
    unittest.main()
