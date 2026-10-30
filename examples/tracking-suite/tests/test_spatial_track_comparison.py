import copy
import unittest

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/compare_spatial_tracks.py"
spec = importlib.util.spec_from_file_location("spatial_track_comparison", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compare = module.compare


class SpatialGraphTests(unittest.TestCase):
    def fixture(self):
        def spec(ids):
            return dict(frames=[dict(t_index=0),dict(t_index=1)],extent=[1,20,20],spacing=[1.,1.,1.],
                        detections=[dict(id=x,t_index=int(i>0)) for i,x in enumerate(ids)])
        # Two geometrically matched daughters; third candidate is unmatched.
        a,b=spec([1,2,3]),spec([2**60+1,2**60+2,2**60+3,2**60+4])
        m=dict(minimum_iou=.5,frames=[dict(t_index=0,matching=dict(matches=[dict(reference=1,predicted=2**60+1)],unmatched_reference=[],unmatched_predicted=[])),
            dict(t_index=1,matching=dict(matches=[dict(reference=2,predicted=2**60+2),dict(reference=3,predicted=2**60+3)],unmatched_reference=[],unmatched_predicted=[2**60+4]))])
        return a,b,m

    def test_wrong_unmatched_daughter_is_not_hidden(self):
        a,b,m=self.fixture()
        first=[dict(**{'from':1,'to':2}),dict(**{'from':1,'to':3})]
        second=[dict(**{'from':2**60+1,'to':2**60+2}),dict(**{'from':2**60+1,'to':2**60+4})]
        result=compare(a,first,b,second,m,[1])
        self.assertEqual(result['links']['common'],1)
        self.assertEqual(result['binary_divisions']['common'],0)
        self.assertEqual(result['binary_divisions']['second'],1)
        self.assertEqual(result['binary_divisions']['second_with_unmatched_node'],1)
        self.assertIsNone(result['reference_evaluation'])
        second[1]['to']=2**60+3
        self.assertEqual(compare(a,first,b,second,m)['binary_divisions']['common'],1)

    def test_partial_or_reused_matching_is_rejected(self):
        a,b,m=self.fixture()
        incomplete=copy.deepcopy(m)
        incomplete['frames'][1]['matching']['unmatched_predicted']=[]
        with self.assertRaises(ValueError):
            compare(a,[],b,[],incomplete)
        m['frames'][1]['matching']['matches'][1]['predicted']=2**60+2
        with self.assertRaises(ValueError):
            compare(a,[],b,[],m)


if __name__=='__main__':
    unittest.main()
