#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only continuity and candidate-support diagnostics, without accuracy claims.

Inputs are one fixed-mask suite input, decoded link arrays and optionally its
immutable Trackastra prediction cache. No image or fluorescence is opened.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path


SOURCE_BYTES = Path(__file__).read_bytes()


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, minimum=None, median=None, p90=None, maximum=None)
    def q(p):
        x = (len(values)-1)*p
        a,b = math.floor(x),math.ceil(x)
        return values[a]*(b-x)+values[b]*(x-a) if a!=b else values[a]
    return dict(n=len(values),minimum=values[0],median=q(.5),p90=q(.9),maximum=values[-1])


class FrameIndex:
    def __init__(self, rows, gate):
        self.gate=gate
        self.cells=defaultdict(list)
        for row in rows:
            y,x=row['registered'][-2:]
            self.cells[math.floor(y/gate),math.floor(x/gate)].append(row)

    def neighbors(self,row):
        y,x=row['registered'][-2:]
        cy,cx=math.floor(y/self.gate),math.floor(x/self.gate)
        return [other for dy,dx in itertools.product((-1,0,1),repeat=2)
                for other in self.cells.get((cy+dy,cx+dx),())
                if math.dist(row['registered'][-2:],other['registered'][-2:])<=self.gate]


def diagnose(spec, links, cache=None, threshold=.5, border_um=25.):
    if spec.get('schema')!='blockflow.tracking-suite-input/v1':
        raise ValueError('unsupported suite input schema')
    if not math.isfinite(threshold) or not 0<=threshold<=1 or not math.isfinite(border_um) or border_um<0:
        raise ValueError('invalid diagnostic threshold or border margin')
    frames=spec['frames']
    times=[f['t_index'] for f in frames]
    if not times or times!=sorted(set(times)):
        raise ValueError('timeline must be nonempty, strictly ordered and unique')
    known_times=[f['elapsed_seconds'] for f in frames if f.get('elapsed_seconds') is not None]
    if any(not math.isfinite(t) or t<0 for t in known_times) or any(b<=a for a,b in zip(known_times,known_times[1:])):
        raise ValueError('recorded timestamps must be finite, nonnegative and strictly increasing')
    if len(spec['extent'])!=3 or spec['extent'][0]!=1 or len(spec['spacing'])!=3:
        raise ValueError('this diagnostic supports 2D [0,y,x] geometry')
    if any(n<=0 for n in spec['extent']) or any(not math.isfinite(x) or x<=0 for x in spec['spacing']):
        raise ValueError('invalid physical geometry')
    gate=spec['max_distance']
    if not math.isfinite(gate) or gate<=0:
        raise ValueError('positive physical gate required')
    rows={}
    by_time=defaultdict(list)
    for row in spec['detections']:
        identity=row['id']
        if type(identity) is not int or not 0<identity<2**64 or identity in rows or row['t_index'] not in times:
            raise ValueError('detection identity or frame invalid')
        for key in ('native','registered'):
            if len(row[key])!=3 or row[key][0]!=0 or not all(math.isfinite(x) for x in row[key]):
                raise ValueError('finite 2D coordinates required')
        if any(not -.5<=x<n-.5 for x,n in zip(row['native'][-2:],spec['extent'][-2:])):
            raise ValueError('native centroid outside field')
        rows[identity]=row
        by_time[row['t_index']].append(row)
    incoming={}
    outgoing=defaultdict(list)
    edges=set()
    rank={t:i for i,t in enumerate(times)}
    gap_links=[]
    for link in links:
        a,b=link['from'],link['to']
        if a not in rows or b not in rows or rows[a]['t_index']>=rows[b]['t_index'] or (a,b) in edges or b in incoming:
            raise ValueError('unknown, backward, duplicate or merging graph link')
        edges.add((a,b));incoming[b]=a;outgoing[a].append(b)
        if len(outgoing[a])>2:
            raise ValueError('binary lineage graph required')
        delta=rank[rows[b]['t_index']]-rank[rows[a]['t_index']]
        if delta>1:
            gap_links.append(dict(source=a,target=b,skipped_acquired_frames=delta-1))
    def border(row):
        return min(min(v+.5,n-.5-v)*s for v,n,s in zip(row['native'][-2:],spec['extent'][-2:],spec['spacing'][-2:]))<=border_um
    indexes={t:FrameIndex(by_time[t],gate) for t in times}
    score_by_source=defaultdict(dict)
    if cache is not None:
        if cache.get('schema')!='blockflow.trackastra-predictions/v1' or cache['max_distance']!=gate:
            raise ValueError('prediction cache schema or gate differs')
        cached={d['id']:d for d in cache['nodes']}
        if len(cached)!=len(cache['nodes']) or set(cached)!=set(rows):
            raise ValueError('prediction cache must preserve exact fixed mask identities')
        for identity,row in rows.items():
            other=cached[identity]
            if any(other[k]!=row[k] for k in ('t_index','registered','native','size')):
                raise ValueError('prediction coordinates or times differ')
        seen=set()
        for edge in cache['edges']:
            a,b,w=edge['from'],edge['to'],edge['weight']
            if a not in rows or b not in rows or (a,b) in seen or rows[b]['t_index']!=rows[a]['t_index']+1 or not math.isfinite(w) or not 0<=w<=1:
                raise ValueError('invalid prediction edge')
            seen.add((a,b))
            if math.dist(rows[a]['registered'][-2:],rows[b]['registered'][-2:])<=gate:
                score_by_source[a][b]=w
    ends=[]
    counts=Counter()
    for identity,row in sorted(rows.items()):
        if outgoing[identity] or row['t_index']==times[-1]:
            continue
        t=row['t_index']; next_t=times[rank[t]+1]
        nearby=indexes[next_t].neighbors(row)
        scores=score_by_source[identity]
        # Upstream greedy includes equality: it discards weights < threshold.
        eligible={x for x,w in scores.items() if w>=threshold}
        if not nearby:
            category='no_next_frame_physical_candidate'
        elif cache is None:
            category='physical_candidate_exists_no_score_cache'
        elif not scores:
            category='physical_candidate_exists_none_returned_by_learned_model'
        elif not eligible:
            category='returned_scores_below_threshold'
        elif all(x in incoming for x in eligible):
            category='threshold_eligible_targets_assigned_to_other_sources'
        else:
            category='threshold_eligible_target_unclaimed'
        counts[category]+=1
        ends.append(dict(id=identity,t_index=t,next_acquired_t=next_t,border=border(row),category=category,
                         physical_candidates=len(nearby),returned_gated_scores=len(scores),
                         maximum_returned_score=max(scores.values(),default=None),
                         threshold_eligible_targets=sorted(eligible),
                         assigned_sources={str(x):incoming.get(x) for x in sorted(eligible)}))
    branches=[]
    visited=set()
    stamps={f['t_index']:f.get('elapsed_seconds') for f in frames}
    for start in sorted(rows):
        if start in incoming and len(outgoing[incoming[start]])==1:
            continue
        path=[start]
        while len(outgoing[path[-1]])==1:
            path.append(outgoing[path[-1]][0])
        if visited.intersection(path):
            raise ValueError('branch accounting reused a detection')
        visited.update(path)
        first,last=rows[path[0]]['t_index'],rows[path[-1]]['t_index']
        duration=None if stamps[first] is None or stamps[last] is None else stamps[last]-stamps[first]
        branches.append(dict(first_id=start,last_id=path[-1],observations=len(path),first_t=first,last_t=last,duration_seconds=duration))
    if visited!=set(rows):
        raise ValueError('branch accounting omitted detections')
    per_frame=[]
    for t in times:
        ids=[r['id'] for r in by_time[t]]
        per_frame.append(dict(t_index=t,detections=len(ids),starts=sum(x not in incoming for x in ids),
                              ends=sum(not outgoing[x] for x in ids),divisions=sum(len(outgoing[x])==2 for x in ids)))
    return dict(schema='blockflow.tracking-diagnostics/v1',interpretation='graph continuity and candidate support, not reference accuracy or biological fate',
                reference_evaluation=None,parameters=dict(max_distance=gate,diagnostic_score_threshold=threshold,border_um=border_um),
                counts=dict(detections=len(rows),links=len(edges),lineage_roots=len(rows)-len(incoming),branches=len(branches),
                            binary_divisions=sum(len(v)==2 for v in outgoing.values()),isolated_detections=sum(x not in incoming and not outgoing[x] for x in rows),
                            interior_starts=sum(x not in incoming and rows[x]['t_index']!=times[0] for x in rows),
                            nonterminal_ends=len(ends),nonterminal_border_ends=sum(e['border'] for e in ends),gap_links=len(gap_links)),
                branch_observations=distribution([b['observations'] for b in branches]),
                branch_duration_seconds=distribution([b['duration_seconds'] for b in branches if b['duration_seconds'] is not None]),
                nonterminal_end_support=dict(sorted(counts.items())),per_frame=per_frame,branches=branches,ends=ends,gap_links=gap_links,
                limitations=['An end is not cell death; a start is not cell birth.',
                             'Short branches can result from true division, fragmentation or segmentation errors.',
                             'Physical candidate proximity is not identity evidence.',
                             'Missing returned model scores may be filtered scores; this does not prove missing physical candidates.',
                             'Threshold categories describe cached support, not the objective of a global or geometric solver.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('suite-input','links','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--predictions',type=Path,help='directory containing predictions.json and complete.json')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--border-um',type=float,default=25.)
    args=p.parse_args()
    source_bytes=args.suite_input.read_bytes();link_bytes=args.links.read_bytes()
    cache=None
    provenance={'suite_input':dict(path=str(args.suite_input.resolve()),sha256=hashlib.sha256(source_bytes).hexdigest()),
                'links':dict(path=str(args.links.resolve()),sha256=hashlib.sha256(link_bytes).hexdigest()),
                'generator_sha256':hashlib.sha256(SOURCE_BYTES).hexdigest()}
    if args.predictions:
        data=(args.predictions/'predictions.json').read_bytes()
        completion=json.loads((args.predictions/'complete.json').read_bytes())
        digest=hashlib.sha256(data).hexdigest()
        if completion['predictions_sha256']!=digest:
            raise ValueError('prediction cache changed after completion')
        cache=json.loads(data)
        if cache.get('input_sha256')!=hashlib.sha256(source_bytes).hexdigest():
            raise ValueError('prediction cache belongs to a different suite input')
        provenance['predictions']=dict(path=str(args.predictions.resolve()),sha256=digest)
    result=diagnose(json.loads(source_bytes),json.loads(link_bytes),cache,args.threshold,args.border_um)
    result['provenance']=provenance
    with args.output.open('x') as handle:
        json.dump(result,handle,allow_nan=False,indent=2);handle.write('\n')
    print(json.dumps(result['counts']))


if __name__=='__main__':
    main()
