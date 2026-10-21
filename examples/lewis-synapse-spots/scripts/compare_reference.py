"""Compare a lewis-synapse-spots run with scripts/spots_reference.py output.
Usage: python3 -I compare_reference.py REFERENCE_DIR RUN_DIR OME_ZARR_DIR [LAYER]"""
import sys,csv,numpy as np,zarr
from scipy.spatial import cKDTree
pd,bd,zd=sys.argv[1:4]; layer=sys.argv[4] if len(sys.argv)>4 else 'lewis'
names=[r['image'] for r in csv.DictReader(open(f'{bd}/images.csv'))]
bsp={}
for r in csv.DictReader(open(f'{bd}/spots.csv')): bsp.setdefault(r['image'],[]).append(r)
for n in names:
    d=np.load(f'{pd}/{n}.npz'); pc=d['cell']; ps=d['spots']
    lab=lambda k: zarr.open_array(f'{zd}/{n}.ome.zarr/labels/{layer}-{k}/0',mode='r')[0]
    cl=lab('synapse'); bs=lab('spots')
    img=[r for r in csv.DictReader(open(f'{bd}/images.csv')) if r['image']==n][0]
    bc=cl==int(img['synapse_label'])
    iou=(pc&bc).sum()/(pc|bc).sum()
    pp=np.array([[float(r['y']),float(r['x'])] for r in csv.DictReader(open(f'{pd}/{n}.spots.csv'))])
    bp=np.array([[float(r['y']),float(r['x'])] for r in bsp[n]])
    # proto y,x are peak pixels; blockflow y,x are label centroids -> match within 2 px
    t=cKDTree(bp); dd,_=t.query(pp); tp=cKDTree(pp); db,_=tp.query(bp)
    sp_agree=((ps>0)==(bs>0)).mean()
    print(f'{n[6:40]:36s} maskIoU={iou:.4f} diff_px={int((pc^bc).sum())} spots p={len(pp)} b={len(bp)} p_unmatched={int((dd>2).sum())} b_unmatched={int((db>2).sum())} spotpx_agree={sp_agree:.4f}')
