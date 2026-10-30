#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay a content-hashed, fixed-mask comparison without tuning its methods.

The freeze lists admitted inputs and immutable code/model/binary hashes. A new
held-out input belongs in a later freeze after the Phase reference assessment;
it cannot be silently substituted into this development comparison package.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def preflight(capsule,input_path,predictions=None,expected_manifest=None):
    manifest_path=capsule/'manifest.json'
    digest=sha(manifest_path)
    if expected_manifest is not None and digest!=expected_manifest:
        raise ValueError('Comparison manifest checksum differs')
    plan=json.loads(manifest_path.read_bytes())
    if plan.get('schema')!='blockflow.frozen-comparison/v1':
        raise ValueError('Unsupported comparison freeze schema')
    for relative,checksum in plan['files'].items():
        path=capsule/relative
        if path.resolve().is_relative_to(capsule.resolve()) is False or not path.is_file() or sha(path)!=checksum:
            raise ValueError('Frozen comparison file changed or missing: '+relative)
    for package,version in plan['packages'].items():
        if importlib.metadata.version(package)!=version:
            raise ValueError('Runtime package version differs: '+package)
    input_hash=sha(input_path)
    matches=[c for c in plan['cases'] if c['input_sha256']==input_hash]
    if len(matches)!=1:
        raise ValueError('Input is not listed in this comparison freeze; held-out execution remains reference-gated')
    case=matches[0]
    spec=json.loads(input_path.read_bytes())
    if spec.get('reference') is not None or spec['runs']!=plan['suite_runs'] or spec['max_distance']!=plan['max_distance']:
        raise ValueError('Comparison inputs/settings differ or embed a reference')
    if predictions is not None:
        raw=predictions/'predictions.json'
        completion=json.loads((predictions/'complete.json').read_bytes())
        if sha(raw)!=completion['predictions_sha256'] or sha(raw)!=case['predictions_sha256']:
            raise ValueError('Reuse cache differs from the frozen completed cache')
        cache=json.loads(raw.read_bytes())
        verify_cache(plan,case,spec,input_hash,cache)
    return plan,case,digest


def verify_cache(plan,case,spec,input_hash,cache):
    if cache.get('schema')!='blockflow.trackastra-predictions/v1' or cache['input_sha256']!=input_hash:
        raise ValueError('Prediction cache input identity differs')
    if cache['nodes']!=spec['detections'] or cache['max_distance']!=plan['max_distance']:
        raise ValueError('Prediction cache geometry differs')
    provenance=cache['provenance']
    if provenance['model_files_sha256']!=plan['model_files_sha256'] or provenance['source_pixels']!=case['source_pixels']:
        raise ValueError('Prediction cache model or source pixels differ')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',type=Path,required=True)
    p.add_argument('--suite-input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-predictions',type=Path)
    p.add_argument('--manifest-sha256')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous comparison runs')
    capsule=args.capsule.resolve()
    plan,case,manifest_sha=preflight(capsule,args.suite_input,args.reuse_predictions,args.manifest_sha256)
    scripts=capsule/'code/blockflow/examples/tracking-suite/scripts'
    args.output.mkdir()
    out=args.output.resolve()
    shutil.copy2(args.suite_input,out/'suite-input.json')
    input_path=out/'suite-input.json'
    journal=dict(schema='blockflow.frozen-comparison-run/v1',status='running',case=case['id'],
                 manifest_sha256=manifest_sha,input_sha256=sha(input_path),reference_evaluation=None,
                 methods=plan['primary_methods'],stages=[])
    write(out/'run.json',journal)
    def run(name,command):
        record=dict(name=name,command=list(map(str,command)),status='running')
        journal['stages'].append(record);write(out/'run.json',journal)
        start=time.monotonic()
        try:
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,check=True)
        except BaseException:
            record.update(status='failed',seconds=time.monotonic()-start)
            journal['status']='failed';write(out/'run.json',journal)
            raise
        record.update(status='complete',seconds=time.monotonic()-start)
        write(out/'run.json',journal)
        print(name+' complete',flush=True)
    try:
        run('classical',[capsule/'bin/tracking-suite',input_path,out/'classical',str(plan['repeats'])])
        predictions=out/'predictions'
        if args.reuse_predictions:
            shutil.copytree(args.reuse_predictions,predictions)
            journal['prediction_reuse']=dict(path=str(args.reuse_predictions.resolve()),predictions_sha256=sha(predictions/'predictions.json'))
        else:
            run('predict',[sys.executable,scripts/'trackastra_adapter.py','predict','--suite-input',input_path,
                '--phase',case['phase'],'--masks',case['masks'],'--model-dir',capsule/'model/general_2d',
                '--batch-size','1','--workers','0','--output',predictions])
            verify_cache(plan,case,json.loads(input_path.read_bytes()),sha(input_path),json.loads((predictions/'predictions.json').read_bytes()))
        run('decode',[sys.executable,scripts/'trackastra_adapter.py','decode','--predictions',predictions,
                      '--timeout',str(plan['solver']['timeout']),'--threads',str(plan['solver']['threads']),
                      '--split-cost',str(plan['solver']['split_cost']),'--output',out/'learned'])
        run('joint_events',[sys.executable,scripts/'event_tracking.py','--suite-input',input_path,
                            '--timeout',str(plan['solver']['timeout']),'--threads','1','--output',out/'joint'])
        base=out/'learned/greedy-links.json'
        run('adjacent_scores',[sys.executable,scripts/'prepare_endpoint_scores.py','--suite-input',input_path,
            '--links',base,'--mode','learned_continuation','--predictions',predictions,
            '--threshold',str(plan['refinement']['threshold']),'--output',out/'adjacent-scores.json'])
        run('adjacent_join',[capsule/'bin/join-endpoints',input_path,base,out/'adjacent-scores.json',out/'adjacent'])
        run('gap_scores',[sys.executable,scripts/'prepare_endpoint_scores.py','--suite-input',input_path,
            '--links',out/'adjacent/links.json','--mode','brownian_gap','--output',out/'gap-scores.json'])
        run('gap_join',[capsule/'bin/join-endpoints',input_path,out/'adjacent/links.json',out/'gap-scores.json',out/'refined'])
        journal.update(status='complete',reference_evaluation=None,
            comparison_outputs=dict(lap='classical/segment_lap-size400/links.json',learned_greedy='learned/greedy-links.json',
                learned_global='learned/ilp-links.json',learned_refined='refined/links.json',joint='joint/links.json'))
        write(out/'run.json',journal)
    except BaseException:
        journal['status']='failed';write(out/'run.json',journal)
        raise


if __name__=='__main__':
    main()
