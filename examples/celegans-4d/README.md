# C. elegans 4-D DIC recordings (Endrov / S-BIAD191)

Hand-annotated Endrov cell lineages as Blockflow tracking tables, inside the
OME-Zarr recordings that `img2omezarr` converts. You can then view the lineage
over the DIC images in ROZT (newvolim): trails, the lineage tree with cell names,
and track selection. It also holds the benchmark of milestone M2 of the project's `GOAL.md`: a
held-out-keyframe evaluation of position refiners, and tracking metrics
(detection, division, link, naming) against the imported lineage. For that reason
the lineage model keeps keyframes and interpolated positions apart.

| module | what |
|---|---|
| `src/endrov.rs` | `<nuclineage>` parser (`rmd.ostxml`, `.ost` dir or `.ost.zip`) and Endrov's interpolation and lifetime rules |
| `src/import.rs` | lineage → detections / tracks / links tables in the `plate-timelapse-tracking` format |
| `src/labels.rs` | optional nucleus spheres as a `uint64` label stack |
| `src/benchmark.rs` | held-out-keyframe benchmark: the `Refiner` trait, the `linear` baseline, scenarios, statistics, report |
| `src/metrics.rs` | tracking metrics against a lineage: detection / division / link P-R-F1, naming accuracy, partial-truth modes, sanity perturbations |
| `src/bin/import_endrov_lineage.rs` | the import CLI |
| `src/bin/holdout_benchmark.rs` | the benchmark CLI |
| `src/bin/score_tracking.rs` | the metrics CLI |
| `scripts/import_all.sh` | batch import for every recording under a zarr root |
| `scripts/serve_rozt.sh` | ROZT server over the zarr root |

## Run

```bash
cargo build --release -p blockflow-celegans-4d
target/release/import_endrov_lineage \
  --image /husky/otherdataset/teresa/4d_elegans/zarr/ceh37_030306.ome.zarr/DIC --labels
# every recording under a root; one line per recording in <root>/endrov-import-index.tsv
examples/celegans-4d/scripts/import_all.sh /husky/otherdataset/teresa/4d_elegans/zarr --labels
```

| flag | default | meaning |
|---|---|---|
| `--image` | required | the image group the lineage annotates. Its stored frames are the detection timeline. |
| `--ostxml` | `<image>/../OST/rmd.ostxml` | the annotation source. It can also be an `.ost.zip`, e.g. `*_dataonly.ost.zip`. |
| `--lineage` | auto | the `<nuclineage id>`. Auto picks the only lineage with keyframes that is not `estcell`. |
| `--layer` | `endrov` | the output name |
| `--leaf-end` | `last-keyframe` | `hold` applies Endrov's own rule: a childless nucleus is held at its last position until the recording ends |
| `--pixel-origin` | `corner` | `centre` turns off the half-pixel shift (see Coordinates) |
| `--keep-markers` | off | keeps single-keyframe nuclei that have no parent or children (`gast`, `venc`, `:anterior`, …) as tracks. Otherwise they are listed in the summary. |
| `--labels` | off | writes `labels/<layer>`. Each detection is a sphere of its radius, at every level, with value = detection id. |
| `--overwrite` | off | replaces this layer's existing outputs |

## Output (inside `<image>/`)

The tables are native NGFF object tables. Their identity graph, spatial index and
`tracking` extensions are the ones `plate-timelapse-tracking` writes; the helpers
and `write_links` are called from that crate. newvolim discovers them as a
tracking set on the image's dataset.

**`tables/<layer>`**: one row per (nucleus, stored frame) while the nucleus exists.

| column | |
|---|---|
| `detection_id` | `t_index·10000 + cell_number`. Unique, stable across reruns, and readable. |
| `t, z, y, x` | coordinates: time index and level-0 voxel. Clamped into the stack; `outside_stack` says when clamping happened. |
| `track_id`, `lineage_id`, `t_index` | as in `plate-timelapse-tracking`. Track = cell; its id is the cell's first detection. |
| `elapsed_seconds` | Endrov frame id, in seconds |
| `cell_number`, `named` | index of the nucleus in name order; `named = 0` for Endrov's unidentified `:N` nuclei |
| `source`, `keyframe` | 0 = keyframe, 1 = linear between keyframes, 2 = held after the last keyframe |
| `key_before`, `key_after` | frame ids of the bracketing keyframes. `key_after` is NaN while held. |
| `world_x, world_y, world_z, radius` | Endrov µm, as interpolated |
| `expression_<name>`, `…_measured` | Endrov `<exp>` level (linear, held outside its range), and whether it was stored at this frame |

**`tables/<layer>-tracks`**: one row per cell. It has the plate columns
(`t_start`, `t_end`, `parent_track_id`, `n_children`, start/end reasons, …),
plus `cell_number`, `named`, `n_keyframes`, `first_keyframe`, `last_keyframe`,
`override_start` and `override_end`. Cell names go in the extension
`tracking.names = {track_id: name}`, because native tables hold numbers only.

**`tables/<layer>-links`**: `continue` within a cell, and `divide` from a
parent's last detection to each child's first. A nucleus with a single child
gets an inter-track `continue`/`gap` link, so both nuclei stay named tracks.

**`tracking/<layer>/`** holds:
- `tracking-summary.json`, written last, with counts, markers and semantics;
- `identity-state.json`;
- `cells.csv`: one row per cell, with name, parent and children;
- `detections.csv`;
- the napari exports `tracks.csv` and `graph.json`.

## Endrov semantics

The rules come from `github.com/mahogny/Endrov`, `ev/endrov/typeLineage/Lineage.java`. The class was called `NucLineage` before the 2011 rename.

- `<pos f x y z r>`: keyframes in world µm. `f` is the frame id in seconds (`EvDecimal`, so it can be fractional).
- `interpolatePos` (l. 1033–1107) gives the position:
  - linear in x, y, z and r between keyframes;
  - held at the last keyframe after it;
  - absent outside `[start=, end=]`, both inclusive;
  - absent at and after the earliest child's first frame. A parent and its children never share a frame, and nothing interpolates from the parent toward the child.
- Before the first keyframe, Endrov reports the first position marked invisible. Here the nucleus is absent.
- A parent keyframe at or after its child's first frame is shadowed, as it is in Endrov.
- `getFirstFrame` is `start=`, otherwise the first keyframe. `getLastFrame` is `end=`, otherwise the later of the last keyframe and the first child frame.
- A childless nucleus without `end=` is held for ever in Endrov. Here it ends at its last keyframe unless `--leaf-end hold` is given.
- `<exp><v f l/>` expression levels are interpolated linearly, with the first and last levels held outside their range (`LineageExp.interpolateLevel`, l. 48–65).

## Coordinates

Endrov maps world to image with `pixel = (world − disp)/res`, with `res` in µm per stored pixel (`EvStack.transformWorldImage`).
- Integer x/y pixel coordinates are pixel corners.
- z plane `k` lies at `k·resZ + dispZ`.
- No axis is flipped.
- Binning plays no part, because `resX` is already per stored pixel.

OME voxel coordinates are voxel centres. So the table position is
`(world − translation)/scale − ½` in x and y, and `(world − translation)/scale`
in z, using the image's own level-0 transform. Both DIC channels here have
translation 0.

The mapping was checked on the DIC. Inside a nucleus DIC is smooth; around it,
the cytoplasm is granular. The test measures the contrast of local high-pass
texture: the ring at 1.1–1.6 r minus the disc at 0.6 r, normalised.

| ceh37, 354 keyframes (69 frames) | mean contrast | fraction > 0 |
|---|---|---|
| identity mapping | **0.27** | **1.00** |
| y flipped / x flipped | 0.02 / 0.01 | 0.52 / 0.53 |
| scale ×0.95 / ×1.05 / ×2 (binning) | 0.04 / 0.00 / 0.01 | 0.65 / 0.47 / 0.53 |
| shift 20 px | ≤ 0.02 | ≈ 0.55 |
| z ±1 plane / ±2 planes | 0.28 / 0.20–0.23 | 0.99 / 0.94 |
| time ±3 / ±10 frames | 0.22 / 0.10–0.11 | 0.96 / 0.74 |

The best xy shift is within about 1 px of identity. For FR317 (43 keyframes),
identity gives 0.38 with a fraction of 0.98, and the flips give 0.01.

Endrov's model fit `estcell`, which is the Ce2008 model fitted to each recording,
scores 0.004 with a fraction of 0.50 under every tested mapping. That is chance
level: its positions do not land on DIC nuclei, so it is not imported by default.

## Counts

| recording | cells (named) | markers | keyframes | detections (key / interp / held) | links | divisions | frames |
|---|---|---|---|---|---|---|---|
| ceh37_030306 | 147 (66) | 2 | 1118 | 5057 (1118 / 3647 / 292) | 5051 | 70 | 906 |
| FR317_070308 | 8 (8) | 9 | 40 | 45 (38 / 7 / 0) | 41 | 2 | 600 |

- **ceh37 source:** `ceh37_030306_dataonly.ost.zip` holds the same `<nuclineage id="1">` as the recording's own `rmd.ostxml`. They have the same `ostdatemodify`, and every `<nuc>` is semantically identical; only the whitespace differs. So the default source is used.
- **FR317 lineage:** only the 4- to 6-cell stage is annotated (ABa, ABp and their children, EMS and P2'). It also has landmarks (`:anterior`, `:posterior`, `:mid1`, `:mid2`, `:shell_up`, `:shell_down`) and event markers (`gast`, `venc`, `2ftail`).

## View in ROZT

```bash
examples/celegans-4d/scripts/serve_rozt.sh       # PORT=9877 by default
# http://<host>:9877/?dataset=ceh37_030306-ome-zarr-DIC  → ribbon: Tracking tab
curl -X POST localhost:9877/v1/datasets/rescan   # after re-importing into a running server
```

## Benchmark: held-out keyframes

A hand-placed keyframe is the only position truth. The benchmark removes
keyframes, interpolates across the hole with Endrov's rule (the importer's code,
`Nucleus::interpolate`), asks a **refiner** for the position and radius at the
removed frame, and compares the answer with the annotator's.

```bash
target/release/holdout_benchmark \
  --image /husky/otherdataset/teresa/4d_elegans/zarr/ceh37_030306.ome.zarr/DIC \
  --refiner linear --out bench/ceh37_030306/linear
```

| flag | default | meaning |
|---|---|---|
| `--image`, `--ostxml`, `--lineage`, `--pixel-origin` | as for the import | the lineage is parsed from the ostxml, not from the tables |
| `--refiner` | `linear` | `linear` (the baseline), `linear-plane-z` (z rounded to a stack plane), `nearest` (the nearer remaining keyframe, held) |
| `--strides` | `2,3,4,6,8` | stride scenarios, see below; `--no-single` skips the single holdout |
| `--channel` | 0 | the channel `Recording::read_box` reads |
| `--out` | required | `keyframes.csv`, `summary.json`, `report.md` |

**Scenarios.** A cell's *visible* keyframes are those inside `[start=, end=]`
and before its first child. *Interior* means neither the first nor the last
visible one. The first visible keyframe is the division frame, and the last
comes before the children appear, so division frames are never held out and
nothing is extrapolated.
- `single`: each interior keyframe is removed in turn, with the rest of the
  lineage intact. The gap is the spacing between its two neighbours.
- `strideN`: in every cell at once, the visible keyframes at index `j` with
  `j mod N ≠ 0` are removed (the first and last stay), and every removed one
  is scored. Neighbour cells are thinned too, so a refiner sees what a
  real sparse annotation would give it. Larger `N` gives longer gaps.

**Refiners.** `celegans_4d::benchmark::Refiner` has one method,
`refine(&Recording, &Query) -> Refined`.
- `Recording` holds the image (blockflow `TimeSeriesImage`), the importer's
  world↔voxel mapping, the frame ids and `read_box(t, centre, half_um)`, which
  reads level-0 voxels around a world position.
- `Query` has the cell, frame and `t_index`, the linear interpolation, the
  cell's remaining keyframes, the lineage as the refiner may see it (held-out
  keyframes removed), and the other cells present at that frame, each with its
  source (keyframe, interpolated or held).
- `Refined` is a sphere plus an optional confidence.

M4 (classic DIC refinement) and M5 (a learned detector) plug in as further
implementations and a `--refiner` choice.

**Errors** are in Endrov world µm:
- `3D` is the Euclidean distance, `XY` the in-plane distance, `z` is `|dz|`
  and `r` is `|dr|`;
- `d3_over_radius` is the 3D error over the true radius;
- `mean_dz` and `mean_dr` are the signed biases.

Every row also carries the linear interpolation's error. `ratio` (in JSON,
`median_ratio_to_linear`) is the refiner's median 3D error over the linear
median on the same rows. The GOAL target for M4/M5 is `ratio ≤ 0.5`.

**Summaries** give n, mean, median, p90 and max for each scenario: overall, for
named cells, and grouped by gap in stored frames, by annotated cells present
(the full lineage), by z plane of the truth, and by 100-frame time bins. There
is also a table of all scenarios pooled by gap, which shows error against gap
length. Every computation runs in a fixed order and floats are rounded in the
JSON, so a rerun gives byte-identical files.

### Baseline (linear), 2026-10-28

Errors in µm. Single holdout:

| recording | n | 3D median / p90 | XY median / p90 | z median / p90 | r median / p90 |
|---|---|---|---|---|---|
| ceh37_030306 | 836 | 0.96 / 1.89 | 0.62 / 1.43 | 0.60 / 1.50 | 0.00 / 0.22 |
| FR317_070308 | 22 | 1.76 / 2.65 | 1.19 / 2.21 | 1.00 / 1.74 | 0.00 / 0.26 |

ceh37, by gap (single holdout, then all scenarios pooled):

| gap (frames) | single n | single 3D median / p90 | pooled n | pooled 3D median / p90 | pooled XY median | pooled z median |
|---|---|---|---|---|---|---|
| 2 | 10 | 0.66 / 1.11 | 15 | 0.76 / 0.92 | 0.26 | 0.00 |
| 3–4 | 70 | 0.83 / 1.72 | 135 | 0.81 / 1.62 | 0.42 | 0.50 |
| 5–8 | 302 | 0.91 / 1.75 | 655 | 0.94 / 1.94 | 0.57 | 0.56 |
| 9–16 | 388 | 1.01 / 1.99 | 1331 | 1.08 / 2.21 | 0.69 | 0.58 |
| 17–32 | 66 | 1.16 / 2.01 | 1370 | 1.53 / 3.06 | 1.00 | 0.75 |
| 33–64 | — | — | 577 | 1.93 / 3.66 | 1.33 | 0.95 |

ceh37, overall 3D median / p90 by stride:

| stride | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 3D | 0.98 / 1.87 | 1.15 / 2.34 | 1.31 / 2.56 | 1.53 / 3.21 | 1.84 / 3.65 |

Reference points:
- `nearest` gives a 3D median / p90 of 1.53 / 2.90 on the ceh37 single holdout (ratio 1.60).
- `linear-plane-z` gives 1.10 / 1.86 (ratio 1.15). Rounding z to a plane makes the median worse, even though the truth lies on planes.

**Annotation facts that bound the benchmark:**
- **z is quantised.** Every keyframe z lies exactly on a stack plane, in both
  recordings (1.5 µm planes; xy is 0.129 µm per pixel). The truth therefore
  carries up to ±0.75 µm of rounding in z. The interpolated z is continuous,
  which accounts for the 0.6 µm median `|dz|`; p90 is one plane. A z error
  below about 0.4 µm cannot be measured.
- **The radius is rarely changed.** Only 28% of consecutive keyframe pairs of
  a cell differ in radius, so the median `|dr|` is 0. The radius error says
  little until a refiner estimates radii itself.
- **Keyframe spacing.** In ceh37 the median is 4 frames (4 min). The quantiles
  10/25/75/90 % are 2/3/6/8 frames, and the maximum is 20. 60 of 971 pairs are
  on consecutive frames. FR317 is annotated densely (spacing 1–4 frames) but
  only at the 4- to 6-cell stage: 8 cells and 22 interior keyframes, so its
  numbers are noisy. They are larger in µm because its nuclei are larger
  (median radius 3.6 µm against 1.9 µm).
- **Annotation range.** ceh37 is annotated only over t = 0–229 (about 3.8 h,
  starting from 6 cells), with at most about 64 annotated cells present at one
  time. Per-time errors are largest early (t < 100: 3D median 1.17) and in the
  5–16-cell range, where cells move fastest.

## Benchmark: tracking metrics

For full-tracking runs (M6), `score_tracking` compares a predicted tracking
with the imported lineage. Both are tracking tables in the
`plate-timelapse-tracking` format. The tracks and links tables are found
through the detections table's `tracking` extension, and names through
`tracking.names` on the tracks table.

```bash
target/release/score_tracking --truth <rec>/DIC/tables/endrov --pred <rec>/DIC/tables/<tracker-layer> \
  [--gate 3] [--division-tolerance 1] [--coverage full|partial|recall-only] [--partial-radius µm] \
  [--t-range a:b] [--image <img>] [--out scores.json]
target/release/score_tracking --truth <rec>/DIC/tables/endrov --sanity     # the truth against perturbed copies
```

Positions are physical, `voxel · scale + translation`, using level-0 of the
table's own image (`--image` overrides it). Truth rows with `outside_stack = 1`
are dropped. Only frames where the truth has detections are scored, optionally
narrowed by `--t-range`.

| metric | definition |
|---|---|
| detection P / R / F1 | Per frame, a one-to-one matching of truth to predictions that minimises total 3D distance, using pairs closer than `--gate` µm (default 3). Each truth detection can also stay unmatched at cost = gate. The solver is blockflow's `track::assign_sparse` (the LAP linker's solver, now public). `matched_distance_mean` is also reported. |
| division P / R / F1 | A division is a detection with ≥ 2 outgoing links. Its time is the children's first frame and its position is the mother's. Truth and predicted divisions are matched one-to-one when `|Δt| ≤ --division-tolerance` (default 1) frames and the mothers are within the gate (cost = distance + |Δt|). |
| link recall / precision | A truth link is recovered when both of its ends are matched and the prediction links the two matched detections directly. A predicted link is correct when both of its ends are matched and the truth links them. |
| naming accuracy | Counted over truth detections of named cells (not `:N`). A detection is correct when its match carries the same track name. `of_matched` divides by the matched detections, `end_to_end` by all of them. It is `null` when the prediction has no names. |

**Partial ground truth.** ceh37 names 66 of 147 nuclei and does not annotate
every cell of the embryo. A prediction on an unannotated cell is then not an
error. `--coverage` handles this:
- `full`: every unmatched prediction is a false positive.
- `partial`: an unmatched prediction (or predicted division) is a false
  positive only within `--partial-radius` µm (default: the gate) of a truth
  detection in the same frame (for a division, that frame or the one before).
  That is, it duplicates or displaces an annotated nucleus. Other predictions
  are counted as `unscored`. Precision then means *precision around annotated
  cells*.
- `recall-only`: there are no false positives, and precision and F1 are `null`.

Recall, link recall and naming are the same in all three modes.

**Sanity check** (`--sanity`, ceh37, gate 3 µm):

| case | detection F1 | division F1 (recall) | link recall | naming (of matched) |
|---|---|---|---|---|
| truth vs itself | 1 | 1 | 1 | 1 |
| jitter σ = 0.5 µm per axis | 1 (mean distance 0.80) | 1 | 1 | 1 |
| jitter σ = 1 µm | 0.974 | 0.971 | 0.948 | 1 |
| jitter σ = 2 µm | 0.486 | 0.457 | 0.225 | 0.992 |
| all divisions removed | 1 | null (0) | 0.972 | 1 |
| every 2nd division removed | 1 | 0.667 (0.5) | 0.986 | 1 |
| every 10th detection dropped | 0.947 (R 0.90) | 0.803 | 0.807 | 1 |
| sibling names swapped | 1 | 1 | 1 | 0.048 |
| every other lineage tree as truth, `full` | 0.649 (P 0.48) | 0.692 | 1 | 1 |
| the same, `partial` | 1 (2625 unscored) | 1 | 1 | 1 |
| the same, `recall-only` | null (R 1) | null (R 1) | 1 | 1 |

At σ = 2 µm per axis, the mean 3D displacement (3.2 µm) is past the 3 µm gate,
hence the drop in F1. The jitter mean distances follow the Maxwell mean σ·√(8/π).

## DIC calibration and maps (milestone M3)

The ops live in blockflow: `src/ops/eggshell.rs` (shell geometry) and
`src/ops/dic.rs` (normalisation, phase reconstruction, smoothness,
nucleus-likeness, shell-free shear estimate). Two binaries here drive them:

```bash
cargo build --release -p blockflow-celegans-4d --bin dic_calibrate --bin dic_maps
Z=/husky/otherdataset/teresa/4d_elegans/zarr
# every recording under the root; one line per recording in the TSV
target/release/dic_calibrate --root $Z --mask-labels --overwrite --summary $Z/dic-calibration-index.tsv
# validation against the hand annotations (keyframes), then the maps as ROZT channels
target/release/dic_maps --image $Z/ceh37_030306.ome.zarr/DIC --validate
target/release/dic_maps --image $Z/ceh37_030306.ome.zarr/DIC --write --overwrite   # --frames all for every frame
python3 examples/celegans-4d/scripts/dic_panels.py $Z/ceh37_030306.ome.zarr/DIC panels.png 3
```

**Angle convention.** Image coordinates: x = column (right), y = row (down),
angles from +x toward +y. θ is the **bright-edge angle**: the shell (and any
optically dense body) is bright on the edges whose outward normal points along
θ. The phase-derivative direction of `I ≈ I0 + κ ∂φ/∂s` is `s = θ + 180°`.

### `dic_calibrate`

On `--frames` (default 8) frames spread over the movie, for every plane:

1. **Initial ellipse**: rays from a seed (the annotation centroid, else the
   deepest textured point) stop where the 1 µm texture RMS stays low for 5 µm;
   a RANSAC ellipse ignores rays that ran into a touching neighbour (FR317 has
   one).
2. **Shell = outermost line**: a thin-line DoG (1 / 2.5 px) in elliptical polar
   coordinates, smoothed along the shell by 1 µm of arc (long lines survive,
   granules do not); each ray keeps the last line before 8 µm of background.
3. **Ridge**: the extremum of the signed line response within −1/+0.6 µm,
   RANSAC at 0.33 µm. Conic fit: Halir & Flusser (1998) direct least squares.
4. **Ellipsoid per frame**: `ρ(z)² = R0²(1 − ((z−z0)/c)²)` over the planes
   (`ρ = √(ab)`), centre/aspect/angle as weighted medians near the equator.
   Long in-plane axis = AP axis (sign not determined).
5. **Calibration**: the signed line response at the inlier ridge points of the
   planes within 0.45·c of the equator, divided by the plane median (the
   medium level), pooled over frames, robust (Tukey) fit of
   `I_edge(φ) = a + b·cos(φ − θ)`, and again with `+ b2·cos 2(φ − θ2)`.
6. **Shell-free cross-check**: second moments of the image gradient (σ 0.2 µm)
   inside the 0.9-scaled shell give the shear axis (mod 180°); the sign of the
   third moment along it gives θ (positive along the bright-edge direction for
   phase-dense bodies).

Outputs: `<image>/calibration/dic/{summary.json, frames.csv, planes.csv}`,
`rozt:imageMetadata` properties `dic.theta_deg`, `dic.a`, `dic.b`, `dic.b2`,
`dic.theta_gradient_deg`, `dic.estimator_difference_deg`, `dic.relief_scale`,
`eggshell.{cx,cy,cz,a,b,c}_um`, `eggshell.ap_angle_deg`; with `--mask-labels`,
`labels/eggshell` (1 inside the shell, every frame and level, nearest calibrated
frame's ellipsoid).

| recording | θ shell | spread (per-frame MAD) | a | b | b2 (θ2) | θ gradient | Δ | shell semi-axes a×b×c µm, AP angle |
|---|---|---|---|---|---|---|---|---|
| ceh37_030306 | −131.9° | 1.7° | 0.0005 | 0.039 | 0.009 (2-harmonic θ −132°) | −133.5° | 1.6° | 26.0 × 19.0 × 23.3, 3° |
| FR317_070308 | −163.6° | 5.6° | 0.013 | 0.038 | 0.015 (θ2 13°) | −179.2° | 15.6° | 27.0 × 15.7 × 19.0, −72° |
| TB2142_071129 (1 of 8 frames hatched, left out) | −40.9° | 7.9° | −0.000 | 0.026 | 0.026 (θ2 −31°) | −63.8° | 23.0° | 30.1 × 15.6 × 17.8, 0° |
| TB2164_080118 (3 of 8 left out; no annotations) | −121.5° | 11.9° | 0.000 | 0.021 | 0.008 (θ2 70°) | −106.2° | −15.3° | 26.8 × 15.2 × 16.6, −4° |

`a`, `b`, `b2` are relative to the medium level. The second harmonic's axis θ2
is aligned with θ (mod 180°) in all three, which is what a quadratic term of
the DIC transfer (`I = I0 + k1·x + k2·x²` with `x ∝ cos(φ − θ)` gives
`cos²`, i.e. `cos 2(φ − θ)`), so **b2/b is a non-linearity index of the bias
setting**: 0.23 (ceh37), 0.39 (FR317), 1.0 (TB2142); TB2164: 0.4, with θ2 12° off the θ axis. The estimators agree to
2° on ceh37; on FR317 and TB2142 the gradient estimate is 15–23° off and moves
with its derivative scale (ceh37: −133.5/−129/−122° at σ 0.2/0.4/0.8 µm;
FR317: −179/−177/−172°). Sources: JPEG 8×8 blocks in FR317 pull the gradient
axis towards 0/90°, and the elongated late-stage anatomy of TB2142 is not
isotropic. **Use the shell θ**; the gradient θ is a sanity check of the axis and
of the sign (the sign agreed on all four). The xy shell centre is stable
(drift ≤ 0.6 µm on ceh37/FR317); the z centre and `c` are poorly constrained
when the stack (ceh37: 30 µm) is thinner than the egg, and drift between
frames (ceh37 z0 8 → 25 µm), so `labels/eggshell` is reliable in xy, not at
the z poles. Frames without a shell (hatched larva, collapsed shell) have b ≈ 0; frames whose b is under `--min-relative-b` (0.35) of the median are left out (`accepted = 0` in `frames.csv`).

Batch: `--root` walks `<root>/*/DIC`, one TSV line per recording (status
`error` with the reason when no shell is found). ~25 s per 70-plane recording
on 8 threads.

### `dic_maps`

Per frame, per plane, with the calibration:

- **normalise**: `(I − plane median)/relief_scale` (`DicNormaliseOp`;
  `relief_scale` = 1.4826·MAD inside the shell);
- **phase**: `Ψ = −i k_s D/(k_s² + 1/ℓ²)` with ℓ = 10 µm, zero-padded by 3ℓ
  (`DicIntegrateOp`). Tikhonov-regularised version of the Fourier-space
  inverse of the shear derivative of Arnison et al. 2004 (J. Microsc. 214:7),
  equivalently Kam's (1998) line integration along the shear with an
  exponential leak. The embryo reconstructs positive (correct sign), cell
  outlines appear, nuclei are faint phase dips, and streaks run along the shear;
- **smoothness** `−ln E` (`DicTextureOp`): `E` = 2·λ_min of the structure
  tensor of the granule band (DoG 0.13/0.5 µm, window 1 µm) — the isotropic
  part of the texture, which drops the straight shear relief at nucleus
  borders; `--measure energy` keeps plain band energy;
- **nucleus-likeness** (`CentreSurroundOp`): disc-minus-surround of the
  smoothness at r = 1.5, 2.5, 3.5 µm (surround 1.6 r), max over r.

`--write` adds `<rec>/DIC_phase`, `DIC_smooth`, `DIC_nucleus` (uint8,
`value = offset + stored·scale` in `blockflow:dicMap`, 0 = not computed),
same grid as DIC, keyframe frames by default; it sets
`attributes.rozt.channelGroup = <recording>` on DIC and the three groups, which
is how newvolim groups siblings as channels of one view (`image_group.rs`,
same grid required). ceh37 keyframes: 0.6–0.9 GB per group.

**Validation** (`--validate`, `calibration/dic/validation/{summary.json,points.csv}`):
every annotated keyframe nucleus in its plane against 200 random points per
annotated plane inside the 0.85-scaled shell. Random points include
unannotated nuclei, so AUCs are lower bounds.

| | ceh37 | FR317 | TB2142 |
|---|---|---|---|
| annotated centres (frames) | 1118 (207) | 38 (12) | 8682 (501) |
| AUC smoothness raw/energy | 0.946 | 0.949 | 0.686 |
| AUC smoothness raw/isotropic (default) | **0.951** | 0.946 | 0.708 |
| AUC smoothness phase/energy | 0.713 | 0.727 | 0.571 |
| AUC smoothness phase/isotropic | 0.932 | 0.944 | 0.666 |
| AUC nucleus-likeness | **0.953** | **0.976** | 0.810 |
| disc − ring smoothness > 0 | 100 % | 100 % | 96 % |
| GT → nearest local max of nucleus-likeness, median | **0.50 µm** (random points 2.88) | **0.49 µm** (3.10) | 0.93 µm (2.80) |
| … within 1 µm | 78 % (random 7 %) | 87 % (6 %) | 53 % (7 %) |
| … within 2 µm | 85 % (26 %) | 97 % (22 %) | 74 % (27 %) |
| … within the nucleus radius | 83 % | 97 % | 67 % |

TB2142 is annotated far past the ceh37 range, and the maps degrade with the
stage because the nuclei shrink below the fixed scales (1 µm texture window,
radii 1.5–3.5 µm):

| TB2142 frames | centres | median radius | nucleus-likeness AUC | median distance to max |
|---|---|---|---|---|
| 0–149 | 1350 | 2.3 µm | 0.94 | — |
| 150–299 | 5482 | 1.6 µm | 0.84 | 0.86 µm |
| 300–449 | 1740 | 1.3 µm | 0.64 | 1.68 µm |
| 450–699 | 110 | 1.2 µm | 0.47 | 2.93 µm |

So the scales must follow the stage (or be chosen per nucleus from the
interpolated radius) beyond ~150 cells.

`--z-sigma 2` (energy smoothed over ±1 µm in z) on FR317: nucleus-likeness AUC
0.982, 92 % within 1 µm — a small gain from the cheap 3-D step.

What did not help: plain band energy on the reconstruction (AUC 0.71: the
reconstruction's streaks and cell outlines are band energy too); the
isotropic measure repairs it (0.93) but does not beat the raw image.
Nucleus-likeness is also high in the smooth medium and perivitelline space, so
restrict detection to the shrunk shell interior (`labels/eggshell`).

## Limitations

- **GFP/disp sign:** the sign of `dispXum` is ambiguous between Endrov versions (Sept 2010 – May 2011 files). img2omezarr writes translation = −disp. The DIC channels here have disp 0, so this is untested. It matters for importing onto GFP grids.
- **Cell names in ROZT:** they appear in the lineage tree and the track card, through the `tracking.names` extension. They are not yet searchable.
