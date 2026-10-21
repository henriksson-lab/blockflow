"""QC montage of a lewis-synapse-spots run: synapse outline (cyan), spot
outlines (amber), spots outside the synapse (magenta).
Labels are read from the images (labels/<layer>-synapse, labels/<layer>-spots).
Usage: python3 -I qc_overlay.py OME_ZARR_DIR RUN_DIR OUT.png [PER_CONDITION] [OFFSET] [LAYER]"""
import sys, csv, os
import numpy as np, zarr
from PIL import Image
from skimage import segmentation

CONDITIONS = ['icam1slb', 'icam1ucht1slb', 'icam1ucht1cd80slb', 'icam1ucht1cd80cd58slb']

def main(zdir, run, out, per=4, offset=0, layer='lewis'):
    images = {r['image']: r for r in csv.DictReader(open(f'{run}/images.csv'))}
    spots = {}
    for r in csv.DictReader(open(f'{run}/spots.csv')):
        spots.setdefault(r['image'], []).append(r)
    rows = []
    for cond in CONDITIONS:
        names = sorted(n for n in images if n.split('_')[1] == cond)[offset:offset + per]
        tiles = []
        for n in names:
            root = f'{zdir}/{n}.ome.zarr'
            a = zarr.open_array(f'{root}/0', mode='r')[0, 0, 0].astype(float)
            lo, hi = np.percentile(a, [1, 99.8])
            g = (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([g] * 3, -1)
            cell = zarr.open_array(f'{root}/labels/{layer}-synapse/0', mode='r')[0] == int(images[n]['synapse_label'])
            lab = zarr.open_array(f'{root}/labels/{layer}-spots/0', mode='r')[0].astype(np.int64)
            outside = {int(s['spot']) for s in spots.get(n, []) if s['in_synapse'] == '0'}
            sb = segmentation.find_boundaries(lab, mode='inner')
            rgb[sb] = [255, 190, 0]
            rgb[sb & np.isin(lab, list(outside))] = [255, 0, 255]
            rgb[segmentation.find_boundaries(cell)] = [0, 255, 255]
            tiles.append(rgb)
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:6]), *sys.argv[6:7])
