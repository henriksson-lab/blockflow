#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare predicted graphs after spatial mask matching; never report accuracy.

Unmatched cells remain in graph totals. An event agrees only when its mother
and BOTH daughters match. Labels are never assumed to share an identity.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read_graph(spec, links):
    nodes = {d['id']: d['t_index'] for d in spec['detections']}
    if len(nodes) != len(spec['detections']):
        raise ValueError('duplicate detection identity')
    edges = set()
    children = defaultdict(set)
    incoming = set()
    for row in links:
        a, b = row['from'], row['to']
        if a not in nodes or b not in nodes or nodes[a] >= nodes[b]:
            raise ValueError('link endpoints or time are invalid')
        if (a,b) in edges or b in incoming:
            raise ValueError('duplicate link or merged ancestry')
        edges.add((a,b))
        incoming.add(b)
        children[a].add(b)
    if any(len(ds)>2 for ds in children.values()):
        raise ValueError('binary lineage comparison rejects multifurcations')
    return nodes, edges, children


def compare(first_spec, first_links, second_spec, second_links, matching, mothers=()):
    a, ea, ca = read_graph(first_spec, first_links)
    b, eb, cb = read_graph(second_spec, second_links)
    if first_spec['frames'] != second_spec['frames'] or first_spec['extent'] != second_spec['extent'] or first_spec['spacing'] != second_spec['spacing']:
        raise ValueError('align physical geometry and timeline before comparison')
    forward, reverse, seen_a, seen_b = {}, {}, set(), set()
    times = set()
    for frame in matching['frames']:
        t, m = frame['t_index'], frame['matching']
        if t in times:
            raise ValueError('duplicate matching frame')
        times.add(t)
        for row in m['matches']:
            x,y = row['reference'], row['predicted']
            if x in seen_a or y in seen_b or a.get(x)!=t or b.get(y)!=t:
                raise ValueError('matching is not one-to-one on the graph frames')
            forward[x], reverse[y] = y,x
            seen_a.add(x); seen_b.add(y)
        for ids,seen,nodes in ((m['unmatched_reference'],seen_a,a),(m['unmatched_predicted'],seen_b,b)):
            for identity in ids:
                if identity in seen or nodes.get(identity)!=t:
                    raise ValueError('invalid unmatched identity or frame')
                seen.add(identity)
    if seen_a != set(a) or seen_b != set(b) or times != {f['t_index'] for f in first_spec['frames']}:
        raise ValueError('matching must explicitly cover every detection and frame')
    common_edges = sum((forward[x],forward[y]) in eb for x,y in ea if x in forward and y in forward)
    da = {x: ds for x,ds in ca.items() if len(ds)==2}
    db = {x: ds for x,ds in cb.items() if len(ds)==2}
    common_divisions = sum(forward[x] in db and {forward[y] for y in ds} == db[forward[x]]
                           for x,ds in da.items() if x in forward and all(y in forward for y in ds))
    cases=[]
    for mother in mothers:
        if mother not in a:
            raise ValueError('requested mother is absent')
        other = forward.get(mother)
        daughters = cb.get(other,set()) if other is not None else set()
        cases.append(dict(first_mother=mother, first_t=a[mother], first_children=sorted(ca.get(mother,set())),
                          second_mother=other, second_children=sorted(daughters),
                          second_children_mapped_to_first=[reverse.get(y) for y in sorted(daughters)],
                          unmapped_second_children=sorted(y for y in daughters if y not in reverse),
                          exact_binary_event_agreement=mother in da and other in db and
                            all(y in forward for y in da[mother]) and {forward[y] for y in da[mother]}==db[other]))
    return dict(schema='blockflow.spatial-track-agreement/v1', interpretation='prediction agreement, not reference accuracy',
                reference_evaluation=None, minimum_iou=matching['minimum_iou'],
                objects=dict(first=len(a),second=len(b),matched=len(forward)),
                links=dict(first=len(ea),second=len(eb),common=common_edges,
                           first_with_unmatched_endpoint=sum(x not in forward or y not in forward for x,y in ea),
                           second_with_unmatched_endpoint=sum(x not in reverse or y not in reverse for x,y in eb)),
                binary_divisions=dict(first=len(da),second=len(db),common=common_divisions,
                                      first_with_unmatched_node=sum(x not in forward or any(y not in forward for y in ds) for x,ds in da.items()),
                                      second_with_unmatched_node=sum(x not in reverse or any(y not in reverse for y in ds) for x,ds in db.items())),cases=cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('first-input','first-links','second-input','second-links','matching','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mothers',type=int,nargs='*',default=[])
    args=p.parse_args()
    keys=('first_input','first_links','second_input','second_links','matching')
    data=[json.loads(getattr(args,k).read_bytes()) for k in keys]
    report=compare(*data,mothers=args.mothers)
    report['provenance']={k:dict(path=str(getattr(args,k).resolve()),sha256=hashlib.sha256(getattr(args,k).read_bytes()).hexdigest()) for k in keys}
    report['provenance']['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2); handle.write('\n')


if __name__=='__main__':
    main()
