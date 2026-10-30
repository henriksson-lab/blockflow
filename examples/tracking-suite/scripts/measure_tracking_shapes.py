#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure geometric mask descriptors once for pluggable tracking scorers.

Only integer label arrays are opened. No Phase intensity, GFP or Red features
are measured. Covariance includes the uniform pixel's variance1/12.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def descriptors(labels):
    if labels.ndim!=2 or labels.dtype.kind not in 'ui':raise ValueError('Expected2D integer labels')
    ids,inverse,counts=np.unique(labels,return_inverse=True,return_counts=True)
    index=inverse.ravel();y,x=np.indices(labels.shape,dtype=float)
    ys=np.bincount(index,weights=y.ravel());xs=np.bincount(index,weights=x.ravel())
    yy=np.bincount(index,weights=(y*y).ravel());xx=np.bincount(index,weights=(x*x).ravel())
    xy=np.bincount(index,weights=(x*y).ravel())
    result=[]
    for i,identity in enumerate(ids):
        if identity==0:continue
        n=int(counts[i]);cy,cx=ys[i]/n,xs[i]/n
        vy=max(0.,yy[i]/n-cy*cy)+1/12;vx=max(0.,xx[i]/n-cx*cx)+1/12;cov=xy[i]/n-cx*cy
        spread=np.sqrt(max(0.,(vy-vx)**2+4*cov*cov))
        major=max(1/12,(vy+vx+spread)/2);minor=max(1/12,(vy+vx-spread)/2)
        result.append(dict(id=int(identity),area_pixels=n,centroid_native_yx=[float(cy),float(cx)],
                           axis_ratio=float(np.sqrt(minor/major)),
                           major_axis_length_pixels=float(4*np.sqrt(major)),
                           minor_axis_length_pixels=float(4*np.sqrt(minor))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('suite-input','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();raw=args.suite_input.read_bytes();spec=json.loads(raw)
    if args.output.exists():raise FileExistsError('Preserve existing shape descriptors')
    source=zarr.open_array(str(args.masks),mode='r')
    expected={f['t_index']:{d['id']:d for d in spec['detections'] if d['t_index']==f['t_index']} for f in spec['frames']}
    records=[];hashes=[]
    for t,rows in expected.items():
        frame=np.asarray(source[t,0] if source.ndim==4 and source.shape[1]==1 else source[t])
        if list(frame.shape)!=spec['extent'][-2:]:raise ValueError('Mask geometry differs')
        features=descriptors(frame)
        if {f['id'] for f in features}!=set(rows):raise ValueError('Shape source identities differ from input')
        for f in features:
            d=rows[f['id']]
            if f['area_pixels']!=d['size'] or not np.allclose(f['centroid_native_yx'],d['native'][-2:],atol=1e-7,rtol=0):
                raise ValueError('Shape source pixels/geometry differ from detections')
            f['t_index']=t
        records.extend(features);hashes.append(dict(t_index=t,native_labels_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
    result=dict(schema='blockflow.tracking-shape-features/v1',input_sha256=hashlib.sha256(raw).hexdigest(),
        features=['axis_ratio','major_axis_length_pixels','minor_axis_length_pixels'],detections=records,
        provenance=dict(source_masks=str(args.masks.resolve()),native_pixels=hashes,
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            geometry='Eigenvalues of2D mask pixel covariance, each pixel contributes variance1/12',
            intensity_features_measured=False))
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(f'Measured geometry for{len(records)} detections',flush=True)


if __name__=='__main__':main()
