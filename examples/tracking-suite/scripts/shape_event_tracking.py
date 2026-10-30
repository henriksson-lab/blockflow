#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Optional additive event scorer using cached geometric descriptors."""
import hashlib
import math
from pathlib import Path


SOURCE_BYTES=Path(__file__).read_bytes()


class ShapeEventScorer:
    def __init__(self,base,spec,features,input_sha256,mother_weight=2.,relax_projected_area=False):
        if not math.isclose(spec['spacing'][1],spec['spacing'][2],rel_tol=1e-12,abs_tol=0):raise ValueError('Initial pixel-shape scorer requires isotropic YX spacing')
        if not math.isfinite(mother_weight) or mother_weight<0:raise ValueError('Invalid mother elongation weight')
        if features.get('schema')!='blockflow.tracking-shape-features/v1' or features['input_sha256']!=input_sha256:
            raise ValueError('Shape features belong to a different input')
        self.rows={r['id']:r for r in features['detections']}
        if len(self.rows)!=len(features['detections']) or set(self.rows)!={d['id'] for d in spec['detections']}:
            raise ValueError('Shape feature identities differ')
        for d in spec['detections']:
            r=self.rows[d['id']]
            centroid = r['centroid_native_yx']
            if len(centroid) != 2 or any(not math.isfinite(v) for v in centroid):
                raise ValueError('Shape centroid must contain two finite coordinates')
            if r['t_index']!=d['t_index'] or r['area_pixels']!=d['size'] or any(not math.isclose(a,b,abs_tol=1e-7,rel_tol=0) for a,b in zip(centroid,d['native'][-2:])):
                raise ValueError('Shape geometry differs')
            ratio=r['axis_ratio']
            if not math.isfinite(ratio) or not 0<ratio<=1:raise ValueError('Invalid mask axis ratio')
        self.base=base;self.weight=mother_weight;self.relax=relax_projected_area
        self.feature_provenance=features['provenance']

    def parameters(self):
        return dict(name='shape-aware-event-wrapper-v1',base=self.base.parameters(),mother_elongation_weight=self.weight,
            projected_area_relaxation=self.relax,feature_provenance=self.feature_provenance,
            source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),
            interpretation='Soft geometric energies, not calibrated division probabilities')

    def score(self,kind,parents,children):
        _,terms,values=self.base.score(kind,parents,children)
        terms=dict(terms);values=dict(values)
        if kind=='divide':
            ratio=self.rows[parents[0]['id']]['axis_ratio']
            terms['mother_elongation']=-self.weight*math.log(ratio)
            values['mother_axis_ratio']=ratio
            if self.relax:
                # Projected2Darea can rise strongly after a rounded mother
                # spreads into daughters. Retain the existing area factor for
                # elongated mothers, relax it smoothly for rounded ones.
                factor=(1-ratio)**2
                values['base_sum_area_energy']=terms['sum_area']
                values['projected_area_factor']=factor
                terms['sum_area']*=factor
        cost=sum(terms.values())
        if not math.isfinite(cost):raise ValueError('Non-finite shape event energy')
        return cost,terms,values
