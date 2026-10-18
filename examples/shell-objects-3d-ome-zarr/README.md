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
- `--out`: `<image>__<layer>.csv` (every candidate with its status),
  `.template.json`, `.accepted.png` / `.rejected.png` contact sheets, and with
  `--keep-rays` the per-ray radii of the published objects.

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
