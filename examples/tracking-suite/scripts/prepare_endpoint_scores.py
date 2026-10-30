#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare auditable incremental scores for the endpoint refinement stage.

learned_continuation uses ONLY adjacent-frame returned model scores; no new
divisions are possible. brownian_gap uses ONLY dt=2 physical-gated pairs and
the separately documented provisional event model. They are separate stages,
never silently combined into a single energy scale.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from tracking_diagnostics import diagnose, FrameIndex
from event_tracking import BrownianJointScorer, Parameters, validate_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('suite-input','links','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['learned_continuation','brownian_gap'],required=True)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--threshold',type=float,default=.3)
    args=p.parse_args()
    spec=json.loads(args.suite_input.read_bytes())
    links=json.loads(args.links.read_bytes())
    validate_input(spec)
    if [f['t_index'] for f in spec['frames']]!=list(range(len(spec['frames']))):
        raise ValueError('these initial score policies require a dense zero-based acquired timeline')
    # Reuse independent graph/accounting validation, not a bare link-degree guess.
    diagnose(spec,links)
    rows={r['id']:r for r in spec['detections']}
    out=Counter(e['from'] for e in links);incoming=Counter(e['to'] for e in links)
    ends=[r for r in rows.values() if not out[r['id']]]
    starts=[r for r in rows.values() if not incoming[r['id']]]
    provenance={'suite_input_sha256':hashlib.sha256(args.suite_input.read_bytes()).hexdigest(),
                'base_links_sha256':hashlib.sha256(args.links.read_bytes()).hexdigest(),
                'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mode':args.mode,'source_channel':'Phase only','old_divisions_immutable':True,
                'reference_evaluation':None,'energy':'incremental join cost relative to leaving end and start unmatched (zero)'}
    scored=[]
    if args.mode=='learned_continuation':
        if args.predictions is None or not math.isfinite(args.threshold) or not 0<args.threshold<1:
            raise ValueError('learned mode requires prediction cache and threshold in (0,1)')
        raw=(args.predictions/'predictions.json').read_bytes();cache=json.loads(raw)
        complete=json.loads((args.predictions/'complete.json').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=complete['predictions_sha256'] or cache['input_sha256']!=provenance['suite_input_sha256']:
            raise ValueError('cache/source provenance differs')
        diagnose(spec,links,cache)
        for e in cache['edges']:
            a,b=e['from'],e['to']
            if not out[a] and not incoming[b] and math.dist(rows[a]['registered'],rows[b]['registered'])<=spec['max_distance']:
                scored.append(dict(**{'from':a,'to':b},cost=args.threshold-e['weight']))
        provenance.update(predictions_sha256=hashlib.sha256(raw).hexdigest(),threshold=args.threshold,
                          parameters='cost=threshold-weight; only strict improvements selected; adjacent frames only')
    else:
        params=Parameters()
        scorer=BrownianJointScorer(spec,params)
        index={f['t_index']:FrameIndex([r for r in starts if r['t_index']==f['t_index']],spec['max_distance']) for f in spec['frames']}
        for a in ends:
            # Explicitly one skipped frame, retaining the original physical gate.
            if a['t_index']+2 not in index:
                continue
            for b in index[a['t_index']+2].neighbors(a):
                continue_cost,_,_=scorer.score('continue',(a,),(b,))
                end_cost,_,_=scorer.score('end',(a,),())
                birth_cost,_,_=scorer.score('birth',(),(b,))
                # Penalize failure to observe the intervening detection under
                # the SAME provisional observation model. Uncalibrated.
                missed=-math.log(1-params.observation_probability)
                scored.append(dict(**{'from':a['id'],'to':b['id']},cost=continue_cost+missed-end_cost-birth_cost))
        provenance.update(parameters=scorer.parameters(),max_dt=2,min_dt=2,max_distance=spec['max_distance'],
                          event_model_sha256=hashlib.sha256(Path(__import__('event_tracking').__file__).read_bytes()).hexdigest(),
                          gap_cost='continue + -log(1-observation_probability) - end - birth; one missing detection',
                          limitation='Provisional gap energies, no calibrated posterior or proof of correct identity')
    with args.output.open('x') as f:
        json.dump(dict(schema='blockflow.endpoint-scores/v1',scores=sorted(scored,key=lambda e:(e['from'],e['to'])),provenance=provenance),f,allow_nan=False,indent=2);f.write('\n')
    print(f'{args.mode}: {len(scored)} endpoint candidates; {sum(e["cost"]<0 for e in scored)} improving energies')


if __name__=='__main__':
    main()
