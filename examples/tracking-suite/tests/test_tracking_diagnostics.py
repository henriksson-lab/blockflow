import copy
import unittest

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/tracking_diagnostics.py"
spec = importlib.util.spec_from_file_location("tracking_diagnostics",path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
diagnose = module.diagnose


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        base=2**60
        def node(identity,t,x):
            return dict(id=base+identity,t_index=t,native=[0.,50.,x],registered=[0.,50.,x],size=10.)
        rows=[node(i,0,x) for i,x in enumerate((5.,30.,60.,90.,120.,91.),start=1)]
        rows += [node(i,1,x) for i,x in zip((101,102,103,104),(30.,60.,90.,120.))]
        spec=dict(schema='blockflow.tracking-suite-input/v1',frames=[dict(t_index=0,elapsed_seconds=0.),dict(t_index=1,elapsed_seconds=7200.)],
                  extent=[1,100,160],spacing=[1.,1.,1.],max_distance=3.,detections=rows)
        def edge(a,b,w):return {'from':base+a,'to':base+b,'weight':w}
        cache=dict(schema='blockflow.trackastra-predictions/v1',max_distance=3.,nodes=copy.deepcopy(rows),
                   edges=[edge(3,102,.4),edge(4,103,.9),edge(5,104,.8),edge(6,103,.95)])
        return base,spec,cache,[edge(6,103,.95)]

    def test_distinguishes_missing_candidates_low_scores_and_competition(self):
        base,spec,cache,links=self.fixture()
        result=diagnose(spec,links,cache,border_um=0.)
        self.assertEqual(result['nonterminal_end_support'],{
            'no_next_frame_physical_candidate':1,
            'physical_candidate_exists_none_returned_by_learned_model':1,
            'returned_scores_below_threshold':1,
            'threshold_eligible_targets_assigned_to_other_sources':1,
            'threshold_eligible_target_unclaimed':1})
        self.assertEqual(result['counts']['nonterminal_ends'],5)
        self.assertEqual(result['counts']['branches'],9)
        self.assertEqual(sum(x['observations'] for x in result['branches']),10)
        self.assertIn(base+5,[x['id'] for x in result['ends']])
        self.assertIsNone(result['reference_evaluation'])
        cache['edges'][0]['weight']=.5
        inclusive=diagnose(spec,links,cache,border_um=0.)
        exact=next(e for e in inclusive['ends'] if e['id']==base+3)
        self.assertEqual(exact['category'],'threshold_eligible_target_unclaimed')

    def test_division_branches_and_acquired_gap_accounting(self):
        _,spec,_,_=self.fixture()
        rows=spec['detections'][:3]
        rows[1]['t_index']=2;rows[2]['t_index']=2
        spec['detections']=rows
        spec['frames']=[dict(t_index=t,elapsed_seconds=t*10.) for t in (0,1,2)]
        links=[{'from':rows[0]['id'],'to':r['id']} for r in rows[1:]]
        result=diagnose(spec,links)
        self.assertEqual(result['counts']['binary_divisions'],1)
        self.assertEqual(result['counts']['lineage_roots'],1)
        self.assertEqual(result['counts']['branches'],3)
        self.assertEqual(result['counts']['gap_links'],2)
        self.assertEqual(result['counts']['nonterminal_ends'],0)
        # Empty acquired t1 remains an explicit frame.
        self.assertEqual(result['per_frame'][1]['detections'],0)

    def test_stale_cache_and_merged_or_duplicate_links_are_rejected(self):
        base,spec,cache,links=self.fixture()
        stale=copy.deepcopy(cache)
        stale['nodes'][0]['registered'][1]+=1
        with self.assertRaises(ValueError):diagnose(spec,links,stale)
        with self.assertRaises(ValueError):diagnose(spec,links+links,cache)
        with self.assertRaises(ValueError):diagnose(spec,links+[{'from':base+4,'to':base+103}],cache)


if __name__=='__main__':
    unittest.main()
