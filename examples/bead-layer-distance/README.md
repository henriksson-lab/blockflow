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
| cost images | per channel: Gaussian → `ln(v − 30)` → a z filter: AF594 top edge (mucus), AF488 top edge + 0.5 × C-Blue top edge + 0.5 × "AF594 just below" (epithelium) | `SmoothOp`, `VoxelwiseMapOp`, `ops::ZTapsOp` (here) |
| surfaces | mucus top and epithelium found jointly as one exact minimum cut over columns pooled 4 × 4: step ≤ 2 slices between neighbours, mucus ≥ 2 slices thick | `ops::surfaces::append_layered_surface_phases`, `graph_cut` |
| large AF488 objects | inside the mucus band, smoothed AF488 above median + 15 MAD-σ and C-Blue not above its own such level (beads are bright in AF488 too); 26-connected components ≥ 4000 µm³; IDs from each object's first voxel | `SurfaceBandOp`, `SampledValuesOp`, `LabelComponentsOp` / `RelabelComponentsOp`, `cells` (here) |
| bead candidates | DoG on C-Blue; keep the response at or above median + 12 MAD-σ; exact regional maxima; strongest first, at least 6 µm apart | `BlobResponseOp`, `GlobalThreshold::RobustSigma`, `KeepAtOrAbove`, `append_response_peak_table_phases`, `suppress_within_distance` |
| classes | `in_tissue` if more than 5 µm below the epithelium; `spike` if not PSF-shaped (neighbours below 35 % / 15 % of the peak in z / xy); `dim` below a log-Otsu split of the rest; `in_large_object` if touching one | `beads` (here) |
| output | label images and paired object tables published into the image; CSVs; QC images from windowed reads | `ome_zarr::publish_label`, `object_table::finalize_object_table`, `WidenLabelsOp`, `SurfaceLinesOp` |

## Input

An OME-Zarr image with axes `t, c, z, y, x` and channels C-Blue (0), AF488 (1),
AF594 (2); AF647 (3) is ignored. Block edges in y and x must be multiples of
the surface stride (default 4).

## Run

```sh
cargo run --release -p blockflow-bead-layer-distance -- \
  --zarr /path/to/image.ome.zarr --out /path/to/report
```

`--help` lists the parameters (surface step limits and weights, bead scale,
thresholds, block edge, workers).

## Output

Into the image, replacing any previous run (review state in the tables
included):

| name | label image | table |
|---|---|---|
| `bjorn-layers` | 1 mucus, 2 tissue (at and below the epithelium top), with NGFF colours | — |
| `bjorn-surfaces` | one-voxel mucus top (1) and epithelium top (2) | — |
| `bjorn-beads` | every candidate as a small ellipsoid, labelled by ID (peak voxel + 1) | one row per candidate: `deleted` (= not counted), `class`, distances, relative depth, response, PSF scores, `large_object` |
| `bjorn-large-af488` | large AF488 objects, labelled by ID | volume, mean AF488 and C-Blue, `n_beads` |

Into `--out`: `beads.csv`, `large_af488_objects.csv`, `height_{epi,mucus}.{npy,png}`,
`qc_xz.png`, `qc_xz_objects.png` and crop galleries.

## Validation

Three samples (512 × 512 × 92–123, 2.16 × 2.16 × 5 µm), compared with the
earlier resident implementation this example replaces: heights, both CSVs and
all four label images are identical.

| sample | candidates | beads | in tissue | large objects | wall | peak RSS |
|---|---|---|---|---|---|---|
| 2776_center1 (good) | 9452 | 335 | 6484 | 198 | 30 s | 3.7 GB |
| 2466-center (average) | 9779 | 981 | 5199 | 299 | 86 s | 4.2 GB |
| 2481-rightbis (poor, folded) | 10276 | 1848 | 3966 | 36 | 42 s | 3.5 GB |

8 workers on a shared, loaded host (load average about 12), so times are
indicative. `tests/end_to_end.rs` runs the whole pipeline on a synthetic image
with two block grids and checks the results are byte-identical.

## Limitations

- **Surface bias.** The edge cost is taken after smoothing (σz = 1 slice)
  and a log, which is steepest on the dark side of an edge: on the synthetic
  test both surfaces sit **two slices (10 µm here) above** the first stained
  slice. Distances to the epithelium are therefore short by up to that much.
  Pinned by the test; not yet corrected.
- **Height resolution** is whole slices, and columns are 4 × 4 voxels.
- **Memory still grows with the image** in three places: the threshold
  samples (one value per 3–7 voxels), the pooled surface graph
  (`K × nz × ny/4 × nx/4` nodes) and the 2D height maps handed to the bead
  step. Histogram-based thresholds and a multigrid surface fit are the planned
  fixes.
- Large objects that touch each other are one object; beads above the fitted
  mucus top are kept and measured (they may be in suspension).
