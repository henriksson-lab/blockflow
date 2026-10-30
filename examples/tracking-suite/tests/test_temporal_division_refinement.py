import copy
from pathlib import Path
import sys
import unittest

import numpy as np

here=Path(__file__).resolve()
scripts=here.parents[1]/'scripts'
if not scripts.is_dir():scripts=here.parent.parent/'blockflow/examples/tracking-suite/scripts'
sys.path.insert(0,str(scripts))
from temporal_mask_features import intersections,frame_offsets,TemporalEvidence,TemporalEventScorer
from refine_divisions import reconsider


class FakeScorer:
    def score(self,kind,parents,children):
        cost={'divide':5.,'continue':1.,'birth':1.}[kind]
        if kind=='continue' and children[0]['id']==3:cost=8.
        return cost,{'synthetic':cost},{}
    def parameters(self):return {'name':'synthetic'}


def fixture():
    spec=dict(schema='blockflow.tracking-suite-input/v1',extent=[1,30,30],spacing=[1.,1.,1.],max_distance=40.,
        frames=[dict(t_index=i,elapsed_seconds=i*3600) for i in range(3)],
        detections=[dict(id=i,t_index=t,native=[0.,10.,float(i)],registered=[0.,10.,float(i)],size=100.)
                    for i,t in ((1,0),(2,1),(3,1),(4,2),(5,2))])
    links=[{'from':1,'to':2,'dt':1,'kind':'divide','weight':.8},
           {'from':1,'to':3,'dt':1,'kind':'divide','weight':.6},
           {'from':2,'to':4,'dt':1,'kind':'continue','weight':.95},
           {'from':3,'to':5,'dt':1,'kind':'continue','weight':.9}]
    return spec,links


class TemporalDivisionTests(unittest.TestCase):
    def test_integer_drift_sign_no_wrap_and_exact_large_ids(self):
        big=2**60+1
        a=np.zeros((8,8),dtype='u8');b=a.copy();a[2:4,3:5]=big;b[3:5,5:7]=big+1
        self.assertEqual(intersections(a,b,(1,2)),[{'from':big,'to':big+1,'intersection_pixels':4}])
        self.assertEqual(intersections(b,a,(-1,-2)),[{'from':big+1,'to':big,'intersection_pixels':4}])
        self.assertEqual(intersections(a,b,(8,0)),[])
        spec,_=fixture();spec['detections'][1]['registered'][-2:]=[8.,1.]
        # Same target frame needs exactly the same stored drift for all cells.
        with self.assertRaises(ValueError):frame_offsets(spec)

    def test_local_optimum_preserves_ordinary_links_and_exact_input(self):
        spec,links=fixture();original=copy.deepcopy(links)
        result,report=reconsider(spec,links,FakeScorer())
        self.assertEqual(links,original)
        self.assertEqual([(v['from'],v['to']) for v in result],[(1,2),(2,4),(3,5)])
        self.assertEqual(result[1:],links[2:])
        self.assertEqual(result[0]['kind'],'continue')
        self.assertEqual(report['decisions'][0]['new_roots'],[3])
        # Exhaustive enumeration of the allowed local outcomes:5,2,9.
        self.assertEqual(min(a['cost'] for a in report['decisions'][0]['alternatives']),2.)
        self.assertEqual(report['decisions'][0]['energy_improvement'],3.)
        unchanged,_=reconsider(spec,links,FakeScorer(),margin=3.)
        self.assertEqual(unchanged,links)
        bad=links+[dict(links[0])]
        with self.assertRaises(ValueError):reconsider(spec,bad,FakeScorer())

    def test_positive_future_merge_evidence_and_input_binding(self):
        spec,_=fixture()
        data=dict(schema='blockflow.temporal-mask-overlaps/v1',input_sha256='fixture',
            frames=[dict(t_index=i) for i in range(3)],provenance={},
            detections=[dict(id=r['id'],t_index=r['t_index'],area_pixels=r['size']) for r in spec['detections']],
            overlaps=[{'from':1,'to':2,'intersection_pixels':40},
                      {'from':1,'to':3,'intersection_pixels':40},
                      {'from':2,'to':4,'intersection_pixels':50},
                      {'from':3,'to':4,'intersection_pixels':50}])
        evidence=TemporalEvidence(spec,data,'fixture');value=evidence.division(1,2,3)
        self.assertEqual(value['transient_split'],.5)
        self.assertEqual(value['preexisting_neighbor'],0.)
        self.assertEqual(value['common_future_mask'],4)
        self.assertFalse(value['history_context_available'])
        wrapper=TemporalEventScorer(FakeScorer(),evidence,2.)
        self.assertEqual(wrapper.score('divide',[spec['detections'][0]],spec['detections'][1:3])[0],6.)
        with self.assertRaises(ValueError):TemporalEvidence(spec,data,'changed')
        data['overlaps'][-1]['intersection_pixels']=51
        with self.assertRaises(ValueError):TemporalEvidence(spec,data,'fixture')

    def features(self,spec,edges):
        return dict(schema='blockflow.temporal-mask-overlaps/v1',input_sha256='fixture',
            frames=[dict(t_index=f['t_index']) for f in spec['frames']],provenance={},
            detections=[dict(id=r['id'],t_index=r['t_index'],area_pixels=r['size']) for r in spec['detections']],
            overlaps=[{'from':a,'to':b,'intersection_pixels':n} for a,b,n in edges])

    def test_preexisting_neighbor_and_distinct_future_daughters(self):
        spec,_=fixture();other=copy.deepcopy(spec['detections'][0]);other['id']=6;spec['detections'].append(other)
        data=self.features(spec,[(1,2,40),(1,3,40),(6,3,60),(2,4,80),(3,5,80)])
        evidence=TemporalEvidence(spec,data,'fixture');value=evidence.division(1,2,3)
        self.assertAlmostEqual(value['preexisting_neighbor'],.2)
        self.assertEqual(value['transient_split'],0.)
        self.assertEqual(value['merged_history'],0.)
        self.assertEqual(value['neighbor_evidence'][1]['competitor'],6)

    def test_previous_separate_bodies_are_positive_unmerge_evidence(self):
        spec,_=fixture();spec['frames'].append(dict(t_index=3,elapsed_seconds=10800))
        spec['detections']=[dict(id=i,t_index=t,native=[0.,5.,5.],registered=[0.,5.,5.],size=area)
            for i,t,area in [(10,0,100),(11,0,100),(20,1,200),(30,2,100),(31,2,100),(40,3,100),(41,3,100)]]
        data=self.features(spec,[(10,20,80),(11,20,80),(10,30,70),(11,31,70),
                                 (20,30,60),(20,31,60),(30,40,80),(31,41,80)])
        value=TemporalEvidence(spec,data,'fixture').division(20,30,31)
        self.assertAlmostEqual(value['merged_history'],.7)
        self.assertEqual(value['transient_split'],0.)
        self.assertTrue(value['history_context_available'])

    def test_supported_division_retains_every_original_link_record(self):
        class PairScorer(FakeScorer):
            def score(self,kind,parents,children):
                cost={'divide':1.,'continue':4.,'birth':4.}[kind]
                return cost,{'synthetic':cost},{}
        spec,links=fixture();result,report=reconsider(spec,links,PairScorer())
        self.assertEqual(result,links)
        self.assertEqual(report['divisions_changed'],0)
        self.assertEqual(report['divisions_retained'],1)

    def test_boolean_feature_identity_cannot_alias_an_integer(self):
        spec,_=fixture();data=self.features(spec,[])
        data['detections'][0]['id']=True
        with self.assertRaises(ValueError):TemporalEvidence(spec,data,'fixture')
        data=self.features(spec,[]);data['frames'][0]['t_index']=False
        with self.assertRaises(ValueError):TemporalEvidence(spec,data,'fixture')

    def test_nonadjacent_division_preserved_and_counted(self):
        spec,_=fixture()
        spec['detections']=spec['detections'][:3]
        for row in spec['detections'][1:]:row['t_index']=2
        links=[{'from':1,'to':2,'kind':'divide','dt':2},
               {'from':1,'to':3,'kind':'divide','dt':2}]
        result,report=reconsider(spec,links,FakeScorer())
        self.assertEqual(result,links)
        self.assertEqual(report['divisions_reconsidered'],0)
        self.assertEqual(report['divisions_retained'],1)
        self.assertEqual(report['divisions_changed'],0)


if __name__=='__main__':unittest.main()
