#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cache drift-aligned mask intersections for adjacent/three-frame context.

Only labels are read. Raster alignment rounds stored global drift differences
to integer pixels; it does not estimate individual-cell motion or optical flow.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def intersections(first, second, shift_yx=(0, 0)):
    if first.ndim != 2 or first.shape != second.shape or first.dtype.kind not in 'ui' or second.dtype.kind not in 'ui':
        raise ValueError('Two equally shaped integer label planes required')
    if (first.dtype.kind=='i' and np.any(first<0)) or (second.dtype.kind=='i' and np.any(second<0)):
        raise ValueError('Negative labels are invalid')
    dy, dx = shift_yx
    if type(dy) is not int or type(dx) is not int:
        raise ValueError('Raster shifts must be integers')
    h,w=first.shape
    if abs(dy)>=h or abs(dx)>=w:
        return []
    a=first[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
    b=second[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)]
    valid=(a>0)&(b>0)
    if not np.any(valid):
        return []
    ids_a,index_a=np.unique(a[valid],return_inverse=True)
    ids_b,index_b=np.unique(b[valid],return_inverse=True)
    packed=index_a.astype('u8')*len(ids_b)+index_b.astype('u8')
    pairs,counts=np.unique(packed,return_counts=True)
    return [dict(**{'from':int(ids_a[int(pair)//len(ids_b)]),'to':int(ids_b[int(pair)%len(ids_b)])},
                 intersection_pixels=int(count)) for pair,count in zip(pairs,counts)]


def frame_offsets(spec):
    offsets={};spacing=spec['spacing'][-2:]
    for row in spec['detections']:
        value=tuple(n-r/s for n,r,s in zip(row['native'][-2:],row['registered'][-2:],spacing))
        t=row['t_index']
        if t in offsets and any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-7) for a,b in zip(value,offsets[t])):
            raise ValueError('Expected a common stored translation per frame')
        offsets[t]=value
    return offsets


class TemporalEvidence:
    """Positive counterevidence to mitosis, never absence-of-overlap rejection."""
    def __init__(self,spec,features,input_sha256):
        if features.get('schema')!='blockflow.temporal-mask-overlaps/v1' or features.get('input_sha256')!=input_sha256:
            raise ValueError('Temporal features belong to a different input')
        self.rows={d['id']:d for d in spec['detections']}
        expected={t['t_index'] for t in spec['frames']}
        if any(type(f.get('t_index')) is not int for f in features['frames']):
            raise ValueError('Temporal frame indices must be exact integers')
        supplied={t['t_index'] for t in features['frames']}
        if supplied!=expected or len(supplied)!=len(features['frames']):
            raise ValueError('Temporal features must cover all acquired frames')
        if any(type(r.get('id')) is not int or not 0<r['id']<2**64 or
               type(r.get('t_index')) is not int or type(r.get('area_pixels')) not in (int,float)
               for r in features['detections']):
            raise ValueError('Invalid exact feature identity, time or area')
        declared={r['id']:r for r in features['detections']}
        if len(declared)!=len(features['detections']) or set(declared)!=set(self.rows):
            raise ValueError('Temporal feature identities differ')
        for identity,row in self.rows.items():
            if declared[identity]!={'id':identity,'t_index':row['t_index'],'area_pixels':row['size']}:
                raise ValueError('Temporal feature geometry differs')
        self.incoming=defaultdict(dict);self.outgoing=defaultdict(dict)
        for edge in features['overlaps']:
            a,b,n=edge['from'],edge['to'],edge['intersection_pixels']
            if type(a) is not int or type(b) is not int or a not in self.rows or b not in self.rows:
                raise ValueError('Invalid exact temporal overlap identity')
            dt=self.rows[b]['t_index']-self.rows[a]['t_index']
            if dt not in (1,2) or type(n) is not int or not 0<n<=min(self.rows[a]['size'],self.rows[b]['size']):
                raise ValueError('Invalid temporal intersection/time')
            if a in self.incoming[b]:
                raise ValueError('Duplicate temporal intersection')
            self.incoming[b][a]=n;self.outgoing[a][b]=n
        for a,targets in self.outgoing.items():
            for dt in (1,2):
                if sum(n for b,n in targets.items() if self.rows[b]['t_index']-self.rows[a]['t_index']==dt)>self.rows[a]['size']:
                    raise ValueError('Overlaps exceed source mask area')
        for b,sources in self.incoming.items():
            for dt in (1,2):
                if sum(n for a,n in sources.items() if self.rows[b]['t_index']-self.rows[a]['t_index']==dt)>self.rows[b]['size']:
                    raise ValueError('Overlaps exceed target mask area')
        self.frames=expected
        self.provenance=features['provenance']

    def division(self,mother,a,b):
        parent=self.rows[mother];t=parent['t_index'];children=[a,b]
        if any(self.rows[c]['t_index']!=t+1 for c in children):
            raise ValueError('Division context requires adjacent daughters')
        neighbors=[];neighbor_penalty=0.
        for child in children:
            own=self.incoming[child].get(mother,0)/self.rows[child]['size']
            other=[(n/self.rows[child]['size'],p) for p,n in self.incoming[child].items()
                   if p!=mother and self.rows[p]['t_index']==t]
            fraction,identity=max(other,default=(0.,None))
            excess=max(0.,fraction-own);neighbor_penalty+=excess
            neighbors.append(dict(child=child,mother_fraction=own,competitor=identity,
                                  competitor_fraction=fraction,excess=excess))
        # Two children strongly occupying one subsequent mask: temporary split
        # or occlusion/merge, not proof of biological nondivision.
        future=[]
        for target,n in self.outgoing[a].items():
            if self.rows[target]['t_index']==t+2 and target in self.outgoing[b]:
                strength=min(n/self.rows[a]['size'],self.outgoing[b][target]/self.rows[b]['size'])
                future.append((strength,target))
        remerge,target=max(future,default=(0.,None))
        # Two separate bodies before the mother was one mask, with daughters
        # returning to those footprints: merge/unmerge counterevidence.
        histories=[]
        previous={p:n/self.rows[p]['size'] for p,n in self.incoming[mother].items() if self.rows[p]['t_index']==t-1}
        for p,fp in previous.items():
            ap=self.incoming[a].get(p,0)/self.rows[a]['size']
            if not ap:continue
            for q,fq in previous.items():
                if p==q:continue
                bq=self.incoming[b].get(q,0)/self.rows[b]['size']
                if bq:histories.append((min(fp,fq,ap,bq),p,q))
        unmerge,p,q=max(histories,default=(0.,None,None))
        return dict(preexisting_neighbor=neighbor_penalty,transient_split=remerge,
                    merged_history=unmerge,neighbor_evidence=neighbors,
                    common_future_mask=target,previous_distinct_masks=[p,q],
                    future_context_available=t+2 in self.frames,
                    history_context_available=t-1 in self.frames)


class TemporalEventScorer:
    def __init__(self,base,evidence,weight=2.):
        if not math.isfinite(weight) or weight<0:
            raise ValueError('Temporal weight must be finite and nonnegative')
        self.base,self.evidence,self.weight=base,evidence,weight

    def parameters(self):
        return dict(name='temporal-mask-counterevidence-v1',base=self.base.parameters(),weight=self.weight,
                    provenance=self.evidence.provenance,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    interpretation='Positive mask counterevidence; soft energies, not mitosis probabilities. Future frames used offline.')

    def score(self,kind,parents,children):
        _,terms,values=self.base.score(kind,parents,children)
        terms,values=dict(terms),dict(values)
        if kind=='divide':
            context=self.evidence.division(parents[0]['id'],children[0]['id'],children[1]['id'])
            for name in ('preexisting_neighbor','transient_split','merged_history'):
                terms['temporal_'+name]=self.weight*context[name]
            values['temporal_mask_context']=context
        return sum(terms.values()),terms,values


def main():
    import zarr
    from event_tracking import validate_input
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('suite-input','masks','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier temporal features')
    raw=args.suite_input.read_bytes();spec=json.loads(raw);validate_input(spec);offsets=frame_offsets(spec)
    source=zarr.open_array(str(args.masks),mode='r')
    if source.ndim!=4 or source.shape[1]!=1 or list(source.shape[-2:])!=spec['extent'][-2:]:
        raise ValueError('Expected native tzyx labels with singleton z')
    rows=defaultdict(dict)
    for row in spec['detections']:rows[row['t_index']][row['id']]=row
    records=[];overlaps=[];shifts=[];recent=deque(maxlen=2)
    for f in spec['frames']:
        t=f['t_index'];frame=np.asarray(source[t,0]);ids,counts=np.unique(frame,return_counts=True)
        actual={int(i):int(n) for i,n in zip(ids,counts) if i}
        if actual!={identity:row['size'] for identity,row in rows[t].items()}:
            raise ValueError('Native labels differ from suite identities/areas')
        records.append(dict(t_index=t,native_labels_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
        for previous_t,previous in recent:
            if t-previous_t not in (1,2):continue
            if t not in offsets or previous_t not in offsets:
                # Empty frames have no foreground intersection or useful drift
                # estimate; no fabricated positive context is added.
                continue
            exact=[b-a for a,b in zip(offsets[previous_t],offsets[t])]
            rounded=[int(round(v)) for v in exact]
            shifts.append(dict(from_t=previous_t,to_t=t,target_sample_shift_yx=rounded,
                               rounding_error_yx=[r-v for r,v in zip(rounded,exact)]))
            overlaps.extend(intersections(previous,frame,tuple(rounded)))
        recent.append((t,frame))
    result=dict(schema='blockflow.temporal-mask-overlaps/v1',input_sha256=hashlib.sha256(raw).hexdigest(),
        detections=[dict(id=r['id'],t_index=r['t_index'],area_pixels=r['size']) for r in spec['detections']],
        frames=records,overlaps=overlaps,provenance=dict(source_masks=str(args.masks.resolve()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shifts=shifts,
            intensity_features_measured=False,alignment='Stored global drift only, rounded to nearest native pixel; no cell-motion alignment'))
    TemporalEvidence(spec,result,result['input_sha256'])
    with args.output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print('Temporal overlap cache:',len(records),'frames,',len(overlaps),'nonzero intersections',flush=True)


if __name__=='__main__':main()
