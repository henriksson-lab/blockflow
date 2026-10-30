import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from measure_tracking_shapes import descriptors
from shape_event_tracking import ShapeEventScorer


class Base:
    def parameters(self):return {'name':'synthetic_base'}
    def score(self,kind,parents,children):
        terms={'motion':1.,'sum_area':10.} if kind=='divide' else {'motion':1.}
        return sum(terms.values()),terms,{}


class ShapeTests(unittest.TestCase):
    def test_exact_large_ids_and_rotation_invariant_elongation(self):
        big=2**60+1
        a=np.zeros((40,80),dtype='u8');a[10:14,5:65]=big
        first=descriptors(a)[0];second=descriptors(a.T)[0]
        self.assertEqual(first['id'],big)
        self.assertEqual(first['area_pixels'],240)
        self.assertLess(first['axis_ratio'],.1)
        self.assertAlmostEqual(first['axis_ratio'],second['axis_ratio'])
        a[:]=0;a[10:20,10:20]=big
        self.assertAlmostEqual(descriptors(a)[0]['axis_ratio'],1.)

    def fixture(self,ratio):
        node=dict(id=1,t_index=0,native=[0.,5.,5.],size=10.)
        spec=dict(spacing=[1.,1.,1.],detections=[node])
        row=dict(id=1,t_index=0,centroid_native_yx=[5.,5.],area_pixels=10,axis_ratio=ratio)
        features=dict(schema='blockflow.tracking-shape-features/v1',input_sha256='synthetic',detections=[row],provenance={})
        return node,spec,features

    def test_only_division_energy_changes_and_area_relaxation_is_explicit(self):
        node,spec,features=self.fixture(.2)
        scorer=ShapeEventScorer(Base(),spec,features,'synthetic',2.,False)
        cost,terms,_=scorer.score('divide',(node,),())
        self.assertAlmostEqual(cost,11.-2*math.log(.2))
        self.assertEqual(scorer.score('continue',(node,),())[0],1.)
        node,spec,features=self.fixture(1.)
        relaxed=ShapeEventScorer(Base(),spec,features,'synthetic',2.,True)
        self.assertEqual(relaxed.score('divide',(node,),())[0],1.)
        self.assertTrue(relaxed.parameters()['projected_area_relaxation'])
        features['input_sha256']='different'
        with self.assertRaises(ValueError):ShapeEventScorer(Base(),spec,features,'synthetic')

    def test_malformed_geometry_and_anisotropic_pixels_rejected(self):
        for centroid in ([], [5.], [5., 5., 5.], [float('nan'), 5.]):
            node,spec,features=self.fixture(.5)
            features['detections'][0]['centroid_native_yx']=centroid
            with self.assertRaises(ValueError):
                ShapeEventScorer(Base(),spec,features,'synthetic')
        node,spec,features=self.fixture(.5)
        spec['spacing']=[1.,1.,2.]
        with self.assertRaises(ValueError):
            ShapeEventScorer(Base(),spec,features,'synthetic')


if __name__=='__main__':unittest.main()
