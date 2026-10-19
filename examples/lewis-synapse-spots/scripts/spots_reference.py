"""Reference prototype: LFA-1 spot detection + synapse mask on the Lewis SLB images.
Usage: python3 -I spots_reference.py RAW_TIFF_DIR OUT_DIR"""
import sys, glob, os, csv
import numpy as np, tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, feature

S1, S2 = 1.5, 2.4      # DoG sigmas (px); spot sigma ~1.6 px at 65 nm
K_SPOT = 5.0           # peak threshold, robust sigmas of DoG response
MIN_SEP = 2.5          # peaks closer than this to a stronger kept one are dropped
K_CELL = 5.0           # cell mask threshold, robust sigmas above background
CELL_FRAC = 0.15       # ...and at least this fraction of (p99.5 - background)
CELL_SIGMA = 6.0
CLOSE_R = 10

def peaks(dog, thr):
    """8-connected regional maxima at/above thr (one per plateau, its first
    pixel), then greedy suppression strongest first (ties: coordinate) of
    peaks closer than MIN_SEP to a kept one. Same rule as the blockflow example."""
    keep = np.where(dog >= thr, dog, -np.inf)
    lm = morphology.local_maxima(keep, connectivity=2, allow_borders=True) & np.isfinite(keep)
    lab = measure.label(lm, connectivity=2)
    cand = []
    for r in measure.regionprops(lab):
        y, x = min(map(tuple, r.coords))
        cand.append((-dog[y, x], y, x))
    cand.sort()
    kept = []
    for _, y, x in cand:
        if all((y - ky) ** 2 + (x - kx) ** 2 >= MIN_SEP ** 2 for ky, kx in kept):
            kept.append((y, x))
    return np.array(kept, dtype=int).reshape(-1, 2)

def background_mode(v):
    """Background = histogram mode; sigma from the half of the peak below it."""
    lo, hi = np.percentile(v, [0.1, 99.9])
    h, e = np.histogram(v, bins=512, range=(lo, hi))
    h = ndi.gaussian_filter1d(h.astype(float), 2)
    mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    below = v[v <= mode]
    return mode, 1.2533 * np.mean(mode - below)   # half-normal mean -> sigma

def robust(v):
    med = np.median(v); return med, 1.4826 * np.median(np.abs(v - med))

def process(path):
    a = tifffile.imread(path).astype(np.float64)
    dog = ndi.gaussian_filter(a, S1, mode='reflect') - ndi.gaussian_filter(a, S2, mode='reflect')
    dmed, dsig = robust(dog)
    thr = dmed + K_SPOT * dsig
    # cell (synapse) mask
    sm = ndi.gaussian_filter(a, CELL_SIGMA, mode='reflect')
    bmed, bsig = background_mode(sm)
    cthr = bmed + max(K_CELL * bsig, CELL_FRAC * (np.percentile(sm, 99.5) - bmed))
    cell = sm > cthr
    cell = morphology.closing(cell, morphology.disk(CLOSE_R))  # keeps pixels at the frame edge, as blockflow does
    cell = ndi.binary_fill_holes(cell)
    lab = measure.label(cell)
    if lab.max():
        props = measure.regionprops(lab)
        c = np.array(a.shape) / 2
        best = max(props, key=lambda p: p.area / (1 + np.hypot(*(np.array(p.centroid) - c)) / 50))
        cell = lab == best.label
    # peaks
    pk = peaks(dog, thr)
    markers = np.zeros(a.shape, np.int32)
    for i, (y, x) in enumerate(pk, 1): markers[y, x] = i
    spots = segmentation.watershed(-dog, markers, mask=dog > dmed + 0.5 * K_SPOT * dsig)
    rows = []
    for p in measure.regionprops(spots, intensity_image=a):
        y, x = pk[p.label - 1]
        rows.append(dict(spot=p.label, y=y, x=x, cy=p.centroid_weighted[0], cx=p.centroid_weighted[1],
                         area=p.area, peak=p.image_intensity[p.image].max(), dog=dog[y, x], sum=p.image_intensity[p.image].sum(),
                         ecc=p.eccentricity, in_cell=int(cell[tuple(p.coords.T)].any())))  # any overlap, as blockflow
    return a, dog, cell, spots, rows, dict(bg=bmed, bg_sigma=bsig, dog_sigma=dsig, thr=thr, cell_area=int(cell.sum()))

if __name__ == '__main__':
    raw, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    summ = []
    for f in sorted(glob.glob(raw + '/*.tif')):
        name = os.path.basename(f)[:-4]
        a, dog, cell, spots, rows, info = process(f)
        np.savez_compressed(f'{out}/{name}.npz', cell=cell, spots=spots)
        with open(f'{out}/{name}.spots.csv', 'w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]) if rows else ['spot']); w.writeheader(); w.writerows(rows)
        n_in = sum(r['in_cell'] for r in rows)
        summ.append(dict(image=name, spots=len(rows), spots_in_cell=n_in, **info))
    with open(f'{out}/summary.csv', 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
