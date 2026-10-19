# Lewis synapse spots

Detects LFA-1 puncta and the synapse outline in T-cell immune synapses formed
on supported lipid bilayers (SLBs), one TIRF plane per image. This is the first
step of an analysis of whether LFA-1 becomes spatially heterogeneous across
four bilayer compositions independently of synapse size; it produces the
points, the outline and per-spot measurements that the spatial statistics will
use.

| condition (file prefix) | bilayer |
|---|---|
| `20min_icam1slb` | ICAM-1 |
| `20min_icam1ucht1slb` | + anti-CD3 (UCHT1) |
| `20min_icam1ucht1cd80slb` | + CD80 |
| `20min_icam1ucht1cd80cd58slb` | + CD58 |

Input: OME-Zarr images, `[t, c, z, y, x]` = `[1, 1, 1, 400, 400]`, uint16,
65 nm pixels (converted from the TIFFs with `img2omezarr`). Intensities are not
comparable between images (laser power and exposure differ), so every level
here is set per image.

## Pipeline

The step table is at the top of [`src/main.rs`](src/main.rs). In short:

* **spots:** difference of Gaussians (`1.5` / `2.4` px), peaks = 8-connected
  regional maxima above `median + 5 * 1.4826 MAD` of the response, greedy
  suppression of peaks closer than `2.5` px to a stronger one, seeded watershed
  of the negated response inside `median + 2.5 * 1.4826 MAD`;
* **synapse:** Gaussian `6` px, threshold
  `mode + max(5 sigma_bg, 0.15 (P99.5 - mode))` (`GlobalThreshold::BackgroundMode`),
  closing with a radius-10 disk, hole filling, components; the synapse is the
  largest component discounted by distance from the field centre.

## Run

```sh
examples/lewis-synapse-spots/scripts/run.sh /husky/otherdataset/teresa/lewis/omezarr .tmp/lewis
```

Writes `spots.csv` (one row per spot: centroid, intensity-weighted centroid,
area, raw sum and max, response max, eccentricity, principal axes, the synapse
component under it, `in_synapse`), `images.csv` (one row per image: synapse
area, centroid, eccentricity, mean intensity, spot counts), `params.json`, and
16-bit label PNGs under `labels/`, and `synapse_features.csv` (below). All
216 images take ~4 min, most of it the Monte Carlo point-pattern reference.

## Per-synapse features

`synapse_features.csv` has one row per image, for analysis across conditions
(`R/synapse_umap.R` in the analysis repo makes UMAPs and per-feature plots from
it). Intensity features are ratios within the synapse after subtracting the
image's own background (the mode outside the synapse), since intensities are
not comparable between images. Spatial features are normalised to the
synapse: radial position is distance-to-edge over the inradius (`ops::distance`,
measured per spot in the plan), and point patterns are compared with 99 sets of
random points in the same mask. The table at the top of
[`src/features.rs`](src/features.rs) lists them.

The crate is standalone (its own `[workspace]` and lockfile) so it builds
without touching the root workspace; its `Cargo.lock` is the workspace's.

## Reference and QC

[`scripts/spots_reference.py`](scripts/spots_reference.py) is the same
pipeline in SciPy/scikit-image, written first to choose parameters.
[`scripts/compare_reference.py`](scripts/compare_reference.py) compares the two
runs. On the 216 images: all synapse masks identical, all spot counts
identical, spot label images identical except 6 pixels (DoG values on the
extent level, rounding), and 6 of 39 913 spot rows differing by that pixel.

[`scripts/qc_overlay.py`](scripts/qc_overlay.py) draws the synapse (cyan),
spots (amber) and spots outside the synapse (magenta).

## Known limits

* One noise level per image: in bright, dense regions shot noise is larger
  and the DoG fragments clusters into many small spots. A variance-stabilising
  transform (Anscombe) before the DoG is the next thing to try.
* A C-shaped synapse whose cleared centre opens to the outside is not filled.
* Touching neighbour cells merge with the synapse.
