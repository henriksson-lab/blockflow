# SPDX-License-Identifier: MIT
# Original work for this crate.
# Exhaustive translation search: where the fixed image (a partial mosaic)
# lies in the moving one (its autofluorescence), by normalised cross-
# correlation on a common coarse grid. Prints t = moving_physical -
# fixed_physical for the best match, z,y,x in the images' unit.
#   python3 find_translation.py FIXED.ome.zarr MOVING.ome.zarr [SPACING [OUT_JSON]]
# (gives SIGNAL_TO_AUTO_INITIAL for a partial mosaic; see the README)
import json, sys
import numpy as np
import zarr
from scipy import ndimage
from scipy.signal import fftconvolve

def open_level(path, target):
    root = zarr.open_group(path, mode='r')
    ms = root.attrs['ome']['multiscales'][0] if 'ome' in root.attrs else root.attrs['multiscales'][0]
    axes = [a['name'] for a in ms['axes']]
    best = None
    for ds in ms['datasets']:
        ct = ds['coordinateTransformations']
        sc = next(c['scale'] for c in ct if c['type'] == 'scale')
        tr = next((c['translation'] for c in ct if c['type'] == 'translation'), [0.0] * len(sc))
        sp = [sc[axes.index(a)] for a in 'zyx']
        if best is None or max(sp) <= target:
            best = (ds['path'], sp, [tr[axes.index(a)] for a in 'zyx'])
    arr = root[best[0]]
    data = np.asarray(arr[...] if arr.ndim == 3 else arr[(0,) * (arr.ndim - 3)], dtype=np.float32)
    return data, np.array(best[1]), np.array(best[2])

spacing = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
fixed, fs, ft = open_level(sys.argv[1], spacing)
moving, ms_, mt = open_level(sys.argv[2], spacing)
print('fixed', fixed.shape, fs, ft, ' moving', moving.shape, ms_, mt)
def regrid(img, sc):
    return ndimage.zoom(img, sc / spacing, order=1)
f = np.log1p(regrid(fixed, fs)); m = np.log1p(regrid(moving, ms_))
print('regridded', f.shape, m.shape)
# normalised cross-correlation of the template f over m (valid positions)
f0 = f - f.mean(); n = f.size
ones = np.ones_like(f)
num = fftconvolve(m, f0[::-1, ::-1, ::-1], mode='valid')
s1 = fftconvolve(m, ones, mode='valid'); s2 = fftconvolve(m * m, ones, mode='valid')
var = np.maximum(s2 - s1 * s1 / n, 1e-9)
ncc = num / np.sqrt(var * (f0 * f0).sum())
best = np.unravel_index(np.argmax(ncc), ncc.shape)
order = np.argsort(ncc, axis=None)[::-1][:5]
print('best offset (grid voxels)', best, 'ncc', ncc[best].round(4))
for o in order:
    print('  ', np.unravel_index(o, ncc.shape), ncc.flat[o].round(4))
# fixed regrid voxel 0 sits at fixed physical ft (voxel-centre conventions
# shift both by the same half-voxel to first order); in moving it is at
# moving physical mt + best * spacing
t = mt + np.array(best) * spacing - ft
print('translation z,y,x', ','.join(f'{v:.1f}' for v in t))
json.dump({'spacing': spacing, 'offset': [int(b) for b in best], 'ncc': float(ncc[best]), 'translation': t.tolist()}, open(sys.argv[4] if len(sys.argv) > 4 else '/dev/stdout', 'w'))
