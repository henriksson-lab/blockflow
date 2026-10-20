# Bead to layer distance over OME-Zarr

Fluorescent beads (about 1 µm, PSF-blurred) are laid on a mucus layer and sink
towards an epithelium. This example finds the top of the mucus and the apical
surface of the epithelium, detects the beads, and measures each bead's
vertical and nearest-point (Euclidean) distance to the epithelium. It also
flags large bright AF488 objects in the mucus — probably dying cells — whose
beads should be ignored. Everything runs as **one planned blockflow run** over
level 0; nothing loads a whole volume.

## Pipeline

| step | how | blockflow pieces |
|---|---|---|
| cost images | upper-surface edges as a relative boundary difference: `−(S(z) − S(z−1)) / (B̄ − 30)`, `S` smoothed at σ 1,2,2 voxels, `B` at 5,4,4 (`ops::RelativeEdgeOp`, here). Mucus top: AF594 edge. Epithelium: AF488 edge + 0.5 × C-Blue edge + 0.5 × "AF594 just below" (on `ln(v − 30)`) | `SmoothOp`, `VoxelwiseMapOp`, `ops::ZTapsOp` / `RelativeEdgeOp` (here) |
| surfaces | mucus top and epithelium found jointly as one exact minimum cut over columns pooled 4 × 4: step ≤ 2 slices between neighbours, mucus ≥ 2 slices thick. Past `--max-surface-nodes`, **multigrid**: exact on coarser columns, then exact cuts on 32 × 32-column tiles with shifting edges at full resolution | `ops::surfaces` (`append_layered_surface_phases`, `SurfaceMethod`), `graph_cut` |
| large AF488 objects | inside the mucus band, smoothed AF488 above median + 15 MAD-σ and C-Blue not above its own such level (beads are bright in AF488 too); 26-connected components ≥ 4000 µm³; IDs from each object's first voxel | `SurfaceBandOp`, `SampledValuesOp`, `LabelComponentsOp` / `RelabelComponentsOp`, `cells` (here) |
| bead candidates | DoG on C-Blue; keep the response at or above median + 12 MAD-σ; exact regional maxima; strongest first, at least 6 µm apart | `BlobResponseOp`, `GlobalThreshold::RobustSigma`, `KeepAtOrAbove`, `append_response_peak_table_phases`, `suppress_within_distance` |
| stained epithelium | smoothed AF488 at or above a log-Otsu level sampled within 10 slices of the fitted epithelium, closed (radius 1,2,2), pieces under 10⁶ µm³ dropped. Inside also: fitted tissue 2+ slices down, and all these seal off from the lumen (`z = 0` face), so beneath the epithelium is inside and a fold's pocket open to the mucus is not | `ApplyGlobalThresholdOp` (`Mask`), `MorphologyOp`, `append_remove_small_objects_phases`, `fill::append_open_on`, `MaskUnionOp` (here) |
| distances | each candidate's signed closest distance (negative inside) and nearest point, to the fitted tissue region and to the stained epithelium; exact up to `--distance-cap-um` (600) | `ops::point_distance` (boundary voxels per block, searched from each point) |
| classes | `in_tissue` if more than 5 µm below the epithelium; `spike` if not PSF-shaped (neighbours below 35 % / 15 % of the peak in z / xy); `dim` below a log-Otsu split of the rest; `in_large_object` if touching one | `beads` (here) |
| output | label images and paired object tables published into the image; CSVs; QC images from windowed reads | `ome_zarr::publish_label`, `object_table::finalize_object_table`, `WidenLabelsOp`, `SurfaceLinesOp` |

### Why the edge is not a log

The first version took a central difference of `ln(smoothed − 30)`. The log
stretches the dark side of an edge, so its steepest point — and the surface —
landed **two slices above** the first stained slice on the synthetic test
(1.3–1.5 slices, about 7 µm, on the real samples), making bead distances short
by that much. The relative edge straddles the slice boundary and divides by a
brightness that barely changes across it: no offset, and nearly as insensitive
to depth fading (a 10× dimmer edge scores 0.95× as strongly). `--edge-recipe log`
keeps the old cost.

## Input

An OME-Zarr image with axes `t, c, z, y, x` and channels C-Blue (0), AF488 (1),
AF594 (2); AF647 (3) is ignored. Block edges in y and x must be multiples of
the surface stride (default 4), and of stride × factor when multigrid is used.

## Run

```sh
cargo run --release -p blockflow-bead-layer-distance -- \
  --zarr /path/to/image.ome.zarr --out /path/to/report
```

`--help` lists the parameters (surface step limits and weights, edge recipe,
bead scale, thresholds, `--surface-method`, `--max-samples`,
`--max-surface-nodes`, block edge, workers). Set
`BLOCKFLOW_MEMORY_BUDGET_BYTES` to bound memory: by default blockflow's chunk
cache takes a quarter of half the machine's RAM.

## Output

Into the image, replacing any previous run (review state in the tables
included):

| name | label image | table |
|---|---|---|
| `bjorn-layers` | 1 mucus, 2 tissue (at and below the epithelium top), with NGFF colours | — |
| `bjorn-surfaces` | one-voxel mucus top (1) and epithelium top (2) | — |
| `bjorn-beads` | every candidate as a small ellipsoid, labelled by ID (peak voxel + 1) | one row per candidate: `deleted` (= not counted), `class`, distances, relative depth, response, PSF scores, `large_object` |
| `bjorn-large-af488` | large AF488 objects, labelled by ID | volume, mean AF488 and C-Blue, `n_beads` |
| `bjorn-epithelium` | 1 = stained epithelium and what lies beneath it | — |

Bead table columns per mask (`surface_` = fitted tissue region, `epithelium_`
= stained): `*_dist_um` (signed, negative inside), `*_nearest_{z,y,x}_um`,
`*_status` (exact / capped = an upper bound / none within reach). The existing
`vertical_dist_um` and `euclidean_dist_um` (to the fitted height map) stay;
`surface_dist_um` equals `euclidean_dist_um` wherever the height map has no
overhang. Distances do not change classes or counts.

Into `--out`: `beads.csv`, `distance_summary.csv` (per mask, over counted
beads: negatives, inexact rows, quantiles), `large_af488_objects.csv`, `height_{epi,mucus}.{npy,png}`,
`qc_xz.png`, `qc_xz_objects.png` and crop galleries.

## Validation

Three samples (512 × 512 × 92–123, 2.16 × 2.16 × 5 µm). With
`--edge-recipe log` the output — heights, both CSVs, all four label images — is
identical to the earlier resident implementation this example replaces. With
the default recipe:

| sample | candidates | beads | in tissue | large objects | median vertical / 3D distance | wall |
|---|---|---|---|---|---|---|
| 2776_center1 (good) | 9452 | 335 | 1111 | 219 | 190 / 170 µm | 42 s |
| 2466-center (average) | 9779 | 996 | 1264 | 300 | 195 / 171 µm | 162 s |
| 2481-rightbis (poor, folded) | 10276 | 1903 | 1584 | 36 | 95 / 81 µm | 49 s |

(The log recipe gave 335 / 981 / 1848 beads at 185 / 185 / 90 µm median
vertical distance. Fewer candidates are `in_tissue` now because the nuclear
peaks in the top epithelial slices are within 5 µm of the corrected surface;
the brightness split removes them instead.) 8 workers on a shared host under
load (load average 10–25), so times are indicative.

**Distances** (counted beads; µm; median, 5th–95th percentile, negatives):

| sample | fitted surface | stained epithelium | stain ≥ 100 µm above the fit (columns) |
|---|---|---|---|
| 2776_center1 | 170 (83–350), 0 | 149 (−9–330), 28 | 0.5 % |
| 2466-center | 171 (117–259), 0 | 122 (29–210), 13 | 3.8 % |
| 2481-rightbis | 81 (29–153), 12 | 10 (−7–80), 498 | 16.8 % |

The fitted-surface distance equals the height-map `euclidean_dist_um` on
every candidate of all three samples. The stained mask is **not yet
trustworthy**: AF488 in the mucus (lumps, large objects) joins the
epithelium and pulls the nearest point up, most of all in 2481, where 10 % of
columns have stain above the fitted mucus top. Wall time grew by 0–14 s per
sample.

**Multigrid against the exact cut** (`--surface-method compare:4:10:32`,
coarse columns 4× wider, band 10 slices, 32-column tiles): energy within
0.000 / 0.046 / 0.053 % of the exact optimum; epithelium columns more than a
slice off 0 / 1.3 / 6.0 %, the last where two nearly equally good answers
exist (2481's mucus lump). Refining by single-column moves instead stalled at
0.6–1.9 %.

`tests/end_to_end.rs` runs the whole pipeline on a synthetic image with two
block grids, checks the outputs are byte-identical, that the surfaces sit on
the drawn ones, and that 9/9 beads and the one large object are found.

## Scaling

Synthetic images made by mirror-tiling the good sample, run with
`BLOCKFLOW_MEMORY_BUDGET_BYTES=4000000000`:

| image | voxels | surfaces | wall | peak RSS |
|---|---|---|---|---|
| 512 × 512 × 115 | 30 M | exact | 42 s | 3.7 GB (default budget) |
| 2048 × 2048 × 115 | 482 M | exact at stride 8 | 5:05 | 5.9 GB |
| 2048 × 2048 × 115 | 482 M | multigrid ×2 at stride 4 | 8:45 | 5.6 GB |
| 4096 × 4096 × 115 | 1.93 G | multigrid ×4 at stride 4 | 57:01 | 6.6 GB |

What memory still depends on: the coarse surface graph (bounded by
`--max-surface-nodes`, ~200 B per node), the threshold sample sets (bounded by
`--max-samples`), the 2D surface heights at column resolution, and the bead
candidates held by the selection step (about 30 B each).

## Limitations

- Height resolution is whole slices, and columns are 4 × 4 voxels.
- Large objects that touch each other are one object; beads above the fitted
  mucus top are kept and measured (they may be in suspension).
- The 2D height maps in the bead step and the coarse graph grow with the
  image's area; in practice they fit to well beyond 10k × 10k.
