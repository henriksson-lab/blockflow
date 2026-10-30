#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reconsider proposed binary divisions while preserving ordinary base links.

Each local component chooses the original split, continuation to daughter A
plus birth of B, or continuation to B plus birth of A. Other links are fixed.
This restricted event problem factorizes exactly; no general MIP is needed.
It does not repair masks, introduce new edges, or infer calibrated confidence.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

from event_tracking import BrownianJointScorer, Parameters, validate_input


def validate_links(spec,links):
    rows={r['id']:r for r in spec['detections']};incoming=set();pairs=set();outgoing=defaultdict(list)
    for link in links:
        a,b=link['from'],link['to']
        if type(a) is not int or type(b) is not int or a not in rows or b not in rows:
            raise ValueError('Base links need exact existing uint64 detection IDs')
        dt=rows[b]['t_index']-rows[a]['t_index']
        if dt<=0 or ('dt' in link and link['dt']!=dt) or b in incoming or (a,b) in pairs:
            raise ValueError('Base graph has invalid time, duplicate edge or merge')
        incoming.add(b);pairs.add((a,b));outgoing[a].append(link)
    for a,edges in outgoing.items():
        if len(edges)>2:raise ValueError('Only binary divisions supported')
        for edge in edges:
            expected='divide' if len(edges)==2 else 'gap' if rows[edge['to']]['t_index']-rows[a]['t_index']>1 else 'continue'
            if edge.get('kind',expected)!=expected:
                raise ValueError('Declared link kind differs from graph topology')
    return rows,outgoing


def reconsider(spec,links,scorer,margin=0.):
    validate_input(spec)
    if not math.isfinite(margin) or margin<0:raise ValueError('Nonnegative finite improvement margin required')
    rows,outgoing=validate_links(spec,links);replacement={};decisions=[];skipped=[]
    for mother,edges in sorted(outgoing.items()):
        if len(edges)!=2:continue
        edges=sorted(edges,key=lambda r:r['to']);a,b=[row['to'] for row in edges];t=rows[mother]['t_index']
        if any(rows[c]['t_index']!=t+1 for c in (a,b)):
            skipped.append(dict(mother=mother,reason='nonadjacent daughters; base edges preserved'))
            continue
        split,terms,features=scorer.score('divide',[rows[mother]],[rows[a],rows[b]])
        alternatives=[dict(kind='divide',keep=[a,b],new_roots=[],cost=split,terms=terms,features=features)]
        for keep,other in ((a,b),(b,a)):
            continuation,ct,cv=scorer.score('continue',[rows[mother]],[rows[keep]])
            birth,bt,bv=scorer.score('birth',[],[rows[other]])
            alternatives.append(dict(kind='continue',keep=[keep],new_roots=[other],cost=continuation+birth,
                terms={**{'continuation_'+k:v for k,v in ct.items()},**{'other_birth_'+k:v for k,v in bt.items()}},
                features=dict(continuation=cv,other_birth=bv)))
        if any(not math.isfinite(option['cost']) or not math.isclose(sum(option['terms'].values()),option['cost'],rel_tol=1e-9,abs_tol=1e-9) for option in alternatives):
            raise ValueError('Invalid local event energy or recorded terms')
        best=min(alternatives,key=lambda r:(r['cost'],r['kind']!='divide',r['keep']))
        chosen=best if best['cost']+margin<split else alternatives[0]
        changed=chosen['kind']!='divide'
        if changed:
            keep=chosen['keep'][0];edge=copy.deepcopy(next(e for e in edges if e['to']==keep))
            # Old objective metadata belongs to the prior graph, not to the new
            # continuation. Complete prior records remain in decisions.json.
            for key in ('event_key','event_cost'):
                edge.pop(key,None)
            edge.update(kind='continue',dt=1,division_refinement_mother=mother)
            replacement[mother]=[edge]
        decisions.append(dict(mother=mother,children=[a,b],changed=changed,selected_kind=chosen['kind'],
            selected_children=chosen['keep'],new_roots=chosen['new_roots'],energy_improvement=split-chosen['cost'],
            margin_required=margin,alternatives=alternatives,prior_links=edges))
    result=[]
    for link in links:
        if link['from'] not in replacement:result.append(copy.deepcopy(link))
    for edges in replacement.values():result.extend(edges)
    result.sort(key=lambda e:(e['from'],e['to']))
    validate_links(spec,result)
    original={(e['from'],e['to']) for e in links};new={(e['from'],e['to']) for e in result}
    if not new<=original:raise AssertionError('Refinement introduced an unproposed association')
    preserved=[e for e in links if len(outgoing[e['from']])!=2]
    by_pair={(e['from'],e['to']):e for e in result}
    if any(by_pair.get((e['from'],e['to']))!=e for e in preserved):
        raise AssertionError('Ordinary base links changed')
    return result,dict(decisions=decisions,skipped=skipped,base_divisions=sum(len(es)==2 for es in outgoing.values()),
        divisions_reconsidered=len(decisions),divisions_retained=sum(not d['changed'] for d in decisions)+len(skipped),
        divisions_changed=len(replacement),base_links=len(links),links=len(result),ordinary_links_preserved=len(preserved),
        added_edges=0,removed_edges=len(original-new),
        scope='Restricted local event optimum; at least one proposed daughter retained. New roots are not biological births.',
        cost_fields='Unchanged link fields retain their base-model meaning; local energy terms are separate and must not be summed with old objectives.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('suite-input','base-links','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--diffusion',type=float,default=8.)
    parser.add_argument('--shape-features',type=Path)
    parser.add_argument('--mother-elongation-weight',type=float,default=0.)
    parser.add_argument('--relax-projected-area',action='store_true')
    parser.add_argument('--temporal-overlaps',type=Path)
    parser.add_argument('--context-weight',type=float,default=2.)
    parser.add_argument('--margin',type=float,default=0.)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous refinements')
    raw=args.suite_input.read_bytes();spec=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    base_raw=args.base_links.read_bytes();base=json.loads(base_raw)
    scorer=BrownianJointScorer(spec,Parameters(diffusion_um2_per_hour=args.diffusion))
    snapshots=[Path(__file__),Path(__file__).with_name('event_tracking.py')]
    if args.shape_features:
        from shape_event_tracking import ShapeEventScorer
        scorer=ShapeEventScorer(scorer,spec,json.loads(args.shape_features.read_bytes()),digest,args.mother_elongation_weight,args.relax_projected_area)
        snapshots.append(Path(__file__).with_name('shape_event_tracking.py'))
    elif args.mother_elongation_weight!=0 or args.relax_projected_area:
        parser.error('Shape policies require --shape-features')
    if args.temporal_overlaps:
        from temporal_mask_features import TemporalEvidence,TemporalEventScorer
        scorer=TemporalEventScorer(scorer,TemporalEvidence(spec,json.loads(args.temporal_overlaps.read_bytes()),digest),args.context_weight)
        snapshots.append(Path(__file__).with_name('temporal_mask_features.py'))
    links,report=reconsider(spec,base,scorer,args.margin)
    report.update(schema='blockflow.division-refinement/v1',input_sha256=digest,
        base_links_sha256=hashlib.sha256(base_raw).hexdigest(),model=scorer.parameters(),reference_evaluation=None)
    args.output.mkdir(parents=True)
    for source in snapshots:(args.output/source.name).write_bytes(source.read_bytes())
    (args.output/'suite-input.json').write_bytes(raw);(args.output/'base-links.json').write_bytes(base_raw)
    for source,name in ((args.shape_features,'shape-features.json'),(args.temporal_overlaps,'temporal-overlaps.json')):
        if source:(args.output/name).write_bytes(source.read_bytes())
    for name,value in (('links.json',links),('decisions.json',report)):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print('Division refinement:',report['divisions_changed'],'changed of',report['divisions_reconsidered'],
          '; ordinary links preserved:',report['ordinary_links_preserved'],flush=True)


if __name__=='__main__':main()
