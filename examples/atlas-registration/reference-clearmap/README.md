# Registration against the reference (ClearMap 3.1, elastix 4.9)

The same inputs, the same parameter files, both sides; what comes out compared
voxel by voxel and point by point.

## Inputs

- **Sample**: osf-cztq2 (the reference's CellMap example data), its
  autofluorescence stack (1422 × 2387 × 1030 at 5 × 5 × 6 µm, x × y × z),
  resampled to 25 µm by the reference's `Resampling.resample` →
  248 × 478 × 285 (z, y, x).
- **Atlas**: ABA 2017 25 µm reference and annotation (the 3.1 default atlas),
  prepared by the reference's `prepare_annotation_files` logic with
  orientation `(1, 2, 3)` and slicing `(:, :, 0:256)` → 256 × 528 × 320.
  **Orientation was determined by inspection**: the reference's CellMap script
  uses `(1, -2, 3)` for its example, but on these data that puts the cerebellum
  at the wrong end — mid-sagittal planes show the cerebellum at high y and
  dorsal at low x in both the sample and the unflipped atlas, with lateral at
  low z, and the sample spans the atlas's first ~240 planes (one hemisphere and
  the midline). The orientation module was not used; the atlas was prepared
  with the reference's own code path (restated in `prepare.py`, because its
  `Annotation` module imports GUI code missing from the environment).
- **Signal** (for the two-step chain): the 40 cFos tiles (8 × 5, 1.625 µm)
  placed at nominal grid positions and averaged in the overlaps at 25 µm
  (247 × 477 × 284). A test input for the chain, not a stitched sample.
- Native side: the same TIFFs as OME-Zarr, through OME-TIFF (which carries the
  25 µm voxel size) and `img2omezarr` (`to_zarr.sh`), under
  `/husky/henriksson/lightsheet-reference/registration/zarr/`.

## Runs

| step | reference | native |
|---|---|---|
| autofluorescence → atlas | `Elastix.align`, fixed = autofluorescence, moving = reference, `align_affine.txt` + `align_bspline.txt` as shipped, 40 threads | `atlas-registration register`, the same two files, 40 threads, seed 121212 |
| affine alone | elastix's `TransformParameters.0.txt` | `register` with `align_affine.txt` only (the same affine: same seed) |
| signal → autofluorescence → atlas | `Elastix.align` (affine) for the first step; points through `transformix -def` twice, as the reference's cell pipeline does | `atlas-registration chain` |
| annotation onto the sample | `transformix`, nearest neighbour | `--warp-source annotation --nearest` |

```text
PY=/home/mahogny/miniconda3/envs/clearmap-gt2/bin/python
$PY -I prepare.py --signal           # resample, prepare the atlas, signal mosaic
./to_zarr.sh                         # OME-Zarr copies for the native side
$PY -I compare.py --write-points     # the point grids
$PY -I clearmap_register.py --signal --processes 40
python3 -I ours_register.py --signal --threads 40
$PY -I compare.py                    # comparison.json
```

The ignored test `tests/register_reference_run.rs` repeats the native
autofluorescence → atlas run and checks the point agreement.

## Results

All numbers from `comparison.json` (`/husky/henriksson/lightsheet-reference/registration/`),
seed 121212 on the native side. Both sides are deterministic: a repeated
elastix run and a repeated native run gave identical transforms.

### Image similarity (fixed autofluorescence against the warped atlas reference, every voxel)

| after | elastix correlation | native correlation | elastix MI | native MI | elastix vs native warped images (correlation) |
|---|---|---|---|---|---|
| affine | 0.6588 | 0.6626 | 0.4900 | 0.4930 | 0.9956 |
| affine + B-spline | 0.6551 | 0.6574 | 0.5284 | 0.5373 | 0.9941 |

MI: mutual information in nats, 64 × 64 equal-width bins over each image's
range. The per-level mutual information elastix logs for the affine (its
4000-sample estimate at the last iteration of each level: 0.666, 0.593, 0.549,
0.513) matches the native last-tenth means (0.661, 0.591, 0.545, 0.526), and
the automatically estimated gains are of the same size (elastix `SP_a` 1049,
544, 220, 263; native 1167, 794, 312, 152).

### Warped annotation: Dice, native against elastix, per major region

| region | after affine | after B-spline | elastix's own affine vs its B-spline |
|---|---|---|---|
| Isocortex | 0.971 | 0.969 | 0.900 |
| OLF | 0.973 | 0.967 | 0.806 |
| HPF | 0.957 | 0.957 | 0.816 |
| CTXsp | 0.968 | 0.949 | 0.692 |
| STR | 0.976 | 0.981 | 0.861 |
| PAL | 0.939 | 0.959 | 0.731 |
| TH | 0.963 | 0.970 | 0.875 |
| HY | 0.968 | 0.975 | 0.859 |
| MB | 0.983 | 0.984 | 0.854 |
| P | 0.973 | 0.968 | 0.746 |
| MY | 0.984 | 0.969 | 0.825 |
| CB | 0.979 | 0.981 | 0.854 |
| fiber tracts | 0.921 | 0.902 | 0.440 |
| VS | 0.935 | 0.944 | 0.631 |
| whole brain | 0.995 | 0.994 | 0.952 |

The last column is what the B-spline stage changes, for scale. Two native runs
with seeds 121212 and 2 agree with each other to 0.915 (fiber tracts) – 0.988
(Isocortex), whole brain 0.993: the same as native against elastix.

### Points

A grid of 31 497 autofluorescence voxels (every 8th voxel above the median
intensity) through both transforms, distance between the two atlas positions:

| transform | median | 95th percentile | max |
|---|---|---|---|
| affine | 15.2 µm | 39.9 µm | 52.6 µm |
| affine + B-spline | 16.0 µm | 51.0 µm | 115.7 µm |
| chain, signal → autofluorescence (affine) | 3.7 µm | 6.8 µm | 9.0 µm |
| chain, signal → autofluorescence → atlas (31 375 signal voxels) | 16.6 µm | 50.2 µm | 106.5 µm |
| *native seed 121212 vs native seed 2* | *16.1 µm* | *49.4 µm* | *123.2 µm* |
| *native seed 2 vs elastix* | *14.3 µm* | *35.6 µm* | *78.8 µm* |
| *elastix affine vs elastix affine + B-spline* | *172.8 µm* | *339.3 µm* | *517.7 µm* |

Native from the **original** autofluorescence OME-Zarr (6 × 5 × 5 µm, sampled
onto the 25 µm lattice from its 24 × 20 × 20 µm level instead of the
reference's resampling), points mapped from resampled to original voxels by the
reference's `index × original / resampled` rule: 24.4 µm median, 65.5 µm 95th
percentile against elastix; 20.0 / 29.3 µm against the native run on the
reference's resampled image — the half-voxel conventions of the two
resamplings, not the registration.

### Wall time and memory

Same machine (40 cores), back to back, with other jobs running (load average
91–125 at the starts).

| | elastix 4.9 (40 threads) | native (40 threads) | ratio |
|---|---|---|---|
| affine, 4 levels × 500 iterations | 75.0 s | 6.6 s | 11× |
| B-spline, 3 levels × 5000 iterations | 436.0 s | 53.3 s | 8× |
| autofluorescence → atlas, whole command | 531.6 s | 76.0 s | 7× |
| signal → autofluorescence (affine), whole command | 81.8 s | — | |
| chain signal → autofluorescence → atlas, whole command | 613.4 s (two runs) | 79.6 s | 7.7× |
| peak resident memory | 2.6 GB | 2.5 GB | |

The native whole command includes reading and resampling both images, warping
the reference after each stage and measuring it, warping the annotation and
writing all outputs (`registration.json` → `timings`); elastix's includes
writing its two result images. A first elastix run under different load took
666 s for autofluorescence → atlas (native 116–131 s then).

## Reading the numbers

- Both optimisers are stochastic. Two native runs with different seeds differ
  from each other as much as either differs from elastix, so the agreement is
  at the level of the method's own sampling noise; elastix's B-spline moves
  points by about ten times that.
- The two sides warp with different interpolators (elastix: cubic B-spline,
  16-bit result; native: trilinear, float), which enters the image numbers but
  not the label or point numbers.
- Timings were taken on a shared 40-core machine with other jobs running; the
  load average at each start is recorded in `stats.json`.
