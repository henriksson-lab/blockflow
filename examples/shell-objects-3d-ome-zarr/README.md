# shell-objects-3d-ome-zarr

Finds round, thick-walled shells (cysts: a thin bright wall over a darker band)
in a 3D OME-Zarr image without learning, planned and out of core, and publishes
them into the image as `labels/<layer>` + `tables/<layer>`, replacing an earlier
layer of the same name. Design and rationale: `SHELL_OBJECTS_3D_PLAN.md`.

```
cargo run --release -p blockflow-shell-objects-3d-ome-zarr -- \
  --zarr /husky/otherdataset/teresa/nedal_data/ND_WT_5_s1.ome.zarr \
  --exemplar 316,142,408 --workers 32 --out results
```

Another image can reuse the saved template instead of an exemplar:
`--template results/ND_WT_5_s1__shell-objects.template.json`.

## Pipeline

| run | phases |
|---|---|
| coarse grid | stored level + `append_block_mean_phases` to an isotropic `--scan-factor` grid (kept) |
| scan, per template | `multi_radius_ncc_chain` (one `TransformBankOp`: shell band of a radial template, 6 radii), `append_response_peak_table_phases`, `SuppressRowsOp` |
| level 0, per pass | `RowSourceOp` (candidates), `ShellFitOp` with a Gaussian prefilter (smooths only the tiles its fits sample), `FilterRowsOp`, `SuppressRowsOp` (centre inside a better object), `StarConvexLabelsOp` (last pass only) |
| publish | `finalize_object_table`, `publish_label`, `POST /v1/datasets/rescan` to the viewer |

The template starts from the exemplar's radial profile on the coarse grid and,
with `--bootstrap N`, is rebuilt from the N best fits before a second scan;
with `--bootstrap 0` there is one pass, and it paints.

## Output

- `labels/<layer>`: `u64`, label = linear index of the candidate voxel + 1.
- `tables/<layer>`: `label_id`, `centroid_*` (fitted centre clamped into the
  volume), `center_*` (unclamped), `diameter_um`, and the fit features.
- unless `--no-features`, 63 rotation-invariant features per kept object
  (`src/features.rs`), appended to `tables/<layer>` and written to
  `<image>__<layer>.features.csv`: radial profile in shape-adapted radius,
  wall step and thickness, radialness (gradient energy and structure-tensor
  alignment along the radius), radial/tangential variograms, per-shell
  statistics, tensor anisotropies, and spherical-harmonic band power of three
  shells with the coherence of neighbouring shells. Measured host-side on a
  level-0 window per object, in parallel; 1–1.5 s per object and thread. Unit
  tests check invariance under an exact grid rotation and that spoke and grain
  phantoms separate as expected.
- unless `--no-interior`, the bright sub-objects inside each published object
  (`ops::interior`, blockflow `src/`): `labels/<layer>-inside` and
  `tables/<layer>-inside` (one row per sub-object: `parent` = the object's
  label, `diameter_um`, `voxels`, mean `level`, radial position `s`,
  `aspect`, `siblings` = sub-objects in its connected bright region, `peak`),
  and per object the `inside_*` summary columns in `tables/<layer>` (count,
  connected regions, sub-objects per region, largest share, bright fraction,
  median diameter and its CV, median aspect, mean `s` of the bright phase,
  spread of the dark phase). Also `<image>__<layer>.interior.csv`,
  `.inside.csv` and the `.inside.png` contact sheet (sub-object outlines).
- `--out`: `<image>__<layer>.csv` (every candidate with its status),
  `.template.json`, `.accepted.png` / `.rejected.png` contact sheets, and with
  `--keep-rays` the per-ray radii of the published objects.

## Recall tuning (ND_WT_5_s1, 2026-10-19)

The defaults favour recall: `--min-strong 0.35 --min-strong-large 0.28`
(were 0.45 / 0.3) and `--duplicate-priority strong_margin` (was the scan
`score`, which let an undersized fit inside a cyst displace the whole one).
That keeps 54 objects instead of 44. By eye, 5 of the 10 added are real
(mostly small cysts, r ≈ 19–24 voxels) and 5 are false positives (tissue
folds and one holder-edge artifact), and none of the 44 is lost. The
validation below used the old defaults.

## All nedal images (2026-10-19)

All 25 volumes, `--template` from ND_WT_5_s1 (the bootstrap rebuilds it per
image), no candidate cap: 1 400–5 300 candidates and 19–424 kept objects per
image, 3 781 in all; 40–120 s and 7–16 GB per image at 32 workers. The cap
mattered: at 600 candidates ND_WT_5_s1 kept 54, uncapped 72. False positives
(tissue blobs, dark cavities) are left in on purpose, for curation.

## Sub-objects (2026-10-20)

Inside each published object, per object on a window around it, in the same
level-0 run as the painting (`InteriorOp`, then `RunLabelsOp` paints the
sub-object labels from x-runs): normalise by the object's own radial profile
(dark band 0, wall peak 1), take the interior `s < 0.85` at level
`--interior-level` (0.65) after a sigma-1 smoothing, seed at the regional
maxima of a sigma-2 smoothing, flood with a seeded watershed, merge
neighbouring basins whose lower peak rises less than `--interior-prominence`
(0.1) above their saddle, and drop sub-objects under `--interior-min-diameter`
(4 µm equivalent). Each object is segmented once, by the block owning its row;
where objects overlap, a voxel belongs to the one it is relatively deepest in,
as in `labels/<layer>` (`NearbyShapesOp` hands each block its neighbours'
shapes), so every sub-object lies inside its parent's label. The answer does
not depend on the block grid (unit test).

Tuned by eye on ND_WT_5_s1's contact sheet (72 objects, 596 sub-objects at
level 0.5, 519 at 0.65): packed cells, lobes and single bright bodies are
outlined well. Neither a sub-object's mean nor its peak level separates
cells from the grain of a grey interior (they overlap); size does, hence the
diameter floor. Limitations: in cysts whose grey interior radiates from a
bright centre the centre keeps some grey tails; a lumpy mass with shallow
lobes is one sub-object (by the prominence rule, on purpose). Cost on
ND_WT_5_s1: about +8 s in the painting pass and +4 s publishing (91 s in all).

## Validation (ND_WT_5_s1, 2026-10-18)

Against the `cystfind` prototype's published result (46 objects, ~40 real by
eye): 44 kept, 43 matched within 10 voxels, centre difference median 0.08 /
max 3.2 voxels, radius difference median 0.05 / max 2.5 voxels. The 3
prototype-only objects were borderline accepts there; one object is new.

Wall time with `--exemplar` and the default bootstrap, 32 workers, release
build, on a shared 40-thread machine (load ~9 from other jobs): **36 s, peak
RSS 7.9 GB**, against 83 s for the prototype. Per stage: coarse grid 1.6 s, each
scan 3.2–3.4 s (prototype 1.2 s), first fit pass 9 s, painting pass 12 s
(prototype 21 s per pass, no painting), contact sheets 1.2 s, publish 3.8 s.
The first port took 3:11 at 16 workers and 11.9 GB; the speed-ups changed no
output byte (CSV, contact sheets, and the label pyramid voxel for voxel).

## Limitations

- Spacing: lengths are level-0 voxels and the grid is assumed isotropic; the
  stored spacing only scales `diameter_um` (`--voxel-size` overrides it).
- A fit block reads its whole raw window (block plus fit reach) whether or not
  it holds candidates; it smooths only around them.
- The fit's window reads and its compute alternate rather than overlap (the
  executor reads, then computes, per worker).
- The bootstrap's second pass refits every candidate, including those the first
  pass already fitted at the same voxel.
