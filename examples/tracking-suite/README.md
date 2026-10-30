# Fixed-mask tracking suite

Compare association models and solvers on exactly the same detections. The
runner reads JSON rows, never image channels, and writes independent graph
results. Segmentation, fluorescence measurements and source datasets are not
modified. Run this stage before introducing segmentation alternatives.

```sh
cargo build --release -p blockflow-tracking-suite
target/release/tracking-suite INPUT.json NEW_OUTPUT_DIRECTORY 5
```

The third argument is the number of repeated solves (default 5). Release builds
are required. An existing output directory is refused. `summary.json` is written
last and marks a completed run; interrupted directories must not be treated as
completed results. Each named run writes its exact `links.json` and `report.json`.
The input is copied verbatim so every result retains the geometry, gating and
configuration that produced it. Attach source/checkpoint/code hashes in input
provenance when preparing a scientific comparison.

## Scoring and solvers

`blockflow::track::cost::LinkCost` scores a gated candidate. `ScoredCandidates`
keeps the features and costs in one canonical order, with scorer provenance.
`ScoredLinker` consumes those scores, allowing comparisons without recomputing
them. `CostedLinker { solver, scorer }` implements the original `Linker` trait,
so existing workflows can use a combination without a new pipeline abstraction.

Built-in pieces in this first increment:

| Piece | Behavior |
|---|---|
| `GeometricCost` | Registered physical distance squared plus optional relative-size and IoU costs. The legacy formula is preserved. |
| `GreedyFrameLinker` | Cheapest eligible links first; one predecessor/successor, with explicit birth/death alternatives. Simple control, not a global optimizer. |
| `FrameLapLinker` | Existing sparse assignment; scored API bypasses its built-in feature weights, retaining its birth/death configuration. |
| `SegmentLapLinker` | Existing two-stage assignment. Imported scores change the **frame stage only**; gap/split/merge costs remain the original squared-distance model. |

The runner currently exposes motion/size scoring and these three solvers. IoU
and custom descriptors are library APIs, not runner options yet. Add a scorer
that captures a descriptor table keyed by `detection_id` to test shape or learned
features. Compute expensive descriptors once upstream. Different segmentations
need a separate matching/evaluation stage; they must not be compared by assuming
equal numeric IDs mean equal cells.

Finite signed energies are supported for imported model scores. They are not
automatically calibrated probabilities. To forbid an association, omit its
candidate rather than using infinity. Birth/death costs must be in the same
units as link costs. The relative-to-max alternative policy is retained for
baseline compatibility; use explicit alternatives when interpreting a
probabilistic model. Pair costs alone cannot express a joint mother/daughter
event or a history-dependent cell-age prior.

## Input

Schema `blockflow.tracking-suite-input/v1`:

```json
{
  "schema": "blockflow.tracking-suite-input/v1",
  "extent": [1, 100, 100],
  "spacing": [1, 1.244, 1.244],
  "frames": [
    {"t_index": 0, "elapsed_seconds": 0},
    {"t_index": 1, "elapsed_seconds": 7200}
  ],
  "detections": [
    {"id": 1, "t_index": 0, "native": [0, 20, 20], "registered": [0, 24.88, 24.88], "size": 100},
    {"id": 2, "t_index": 1, "native": [0, 21, 20], "registered": [0, 26.124, 24.88], "size": 100}
  ],
  "max_distance": 40,
  "runs": [
    {"name": "motion-lap", "solver": "frame_lap", "size_weight": 0},
    {"name": "size-greedy", "solver": "frame_greedy", "size_weight": 400},
    {"name": "legacy-segments", "solver": "segment_lap", "size_weight": 400,
     "gap_closing": 60, "max_gap": 2, "splitting": 30}
  ],
  "provenance": {"source": "describe source and its hashes here"},
  "baseline_pairs": [[1, 2]],
  "reference": null
}
```

Coordinates: native voxels `[z,y,x]`, registered physical coordinates in the
shared drift-corrected frame. Sizes are pixels/voxels. IDs are exact uint64
integers, not floating-point numbers. Timeline and candidate timestamps must
agree. Gate distances use the registered coordinate units. `alternative_cost`
optionally sets a fixed per-birth/per-death cost; omitted means the existing
relative-to-max policy. `max_gap` is a frame-index difference, not a count of
missing intermediate frames. Merging is disabled in this runner.

`baseline_pairs` only measures agreement with an older result. It never creates
accuracy scores. To evaluate a complete synthetic or independently reviewed
reference on the same detections, supply:

```json
{"complete": true, "kind": "human_reviewed_complete",
 "pairs": [[1, 2]], "provenance": {"review_revision": "..."}}
```

as `reference`. `kind` can also be `synthetic`. An incomplete review is rejected;
first restrict the input to an explicitly complete reviewed region and handle
its temporal/spatial boundaries. A declaration alone does not establish review
quality. A scientifically valid benchmark still requires representative review,
separate tuning/test sets and uncertainty about ambiguous cases.

## Evaluation and interpretation

`track::evaluation::compare_graphs` requires identical detection IDs and frames.
It reports TP/FP/FN and precision/recall/F1 for links, unbranched links, and exact
binary-division triplets. One wrong daughter makes a division incorrect even
when the mother and other daughter match. Undefined denominators are null,
not perfect scores. Multifurcations are counted separately.

These are fixed-mask graph scores, **not** CTC TRA or segmentation accuracy.
Identity-switch, complete-lineage, mask-quality and correction-effort metrics
remain future additions. With no reference the runner produces diagnostics only.
Divisions reported by a tracker can represent segmentation splits, not confirmed
biological mitoses. A smaller count of such events does not establish improvement.

## Verified initial pilot

On 2026-10-29 the Incucyte B2 pilot (27 frames, 15,800 detections, 25,884 gated
candidates) was replayed through the runner. `segment_lap` with size weight 400,
40-unit frame gate, 60-unit/2-frame gap gate and 30-unit split gate reproduced
**all 15,029 original links, 1,975 tracks and 602 candidate binary divisions**.
Seven motion/size/solver variants were solved five times each and were
deterministic within each configuration. Approximate release solve medians on
that host were 0.02 seconds for frame solvers and 0.055 seconds for segment LAP.
These are diagnostics on one field, not accuracy results or universal runtime
guarantees. The model weights and source masks were not changed.

Tests:

```sh
cargo test --release -p blockflow --lib track::
cargo test --release -p blockflow-tracking-suite
```

The initial verification passed 27 tracking unit tests and 3 CLI integration
tests, including a complete synthetic division, fixed-score greedy/LAP
disagreement, size-dependent crossing assignments, signed costs, birth/death
alternatives, baseline parity and rejection of incomplete reference declarations.


## Learned-score bridge (Trackastra 0.5.6)

`scripts/trackastra_adapter.py` is a version-pinned experimental bridge. It
requires `trackastra==0.5.6`, Zarr, and a CUDA PyTorch installation for prediction.
It currently supports the pilot's 2D `tczyx` Phase and `tzyx` native-mask arrays,
with metadata explicitly identifying Phase. It verifies every mask ID and area
against the suite input. Features come from unresampled Phase pixels and masks;
registered centroid coordinates are supplied separately in pixel units.
The released `general_2d` checkpoint uses `wrfeat` and four-frame windows.

```sh
python scripts/trackastra_adapter.py predict \
  --suite-input INPUT.json --phase PHASE_GROUP/0 --masks MASK_GROUP/0 \
  --model-dir GENERAL_2D_MODEL --output NEW_PREDICTION_DIRECTORY
python scripts/trackastra_adapter.py decode \
  --predictions NEW_PREDICTION_DIRECTORY --output NEW_DECODE_DIRECTORY \
  --timeout 120 --threads 8
python scripts/trackastra_adapter.py export-scores \
  --predictions NEW_PREDICTION_DIRECTORY --output NEW_MATCHED_INPUT.json
target/release/tracking-suite NEW_MATCHED_INPUT.json NEW_RESULT_DIRECTORY 5
```

The prediction cache freezes the executed adapter and records checkpoint,
input, pixel and software identities. Prediction is run once; each decoder
consumes exactly the same cache. Greedy with/without divisions and SCIP global
optimization with/without divisions are compared. Global decoding uses the
upstream `gt` costs, explicitly pins every fixed-mask node, and reports solver
status. The timeout limits optimization, not graph construction. A solution
without a feasible fixed-mask incumbent is rejected. A proven optimum applies
to this objective, not biological truth. `--split-cost` exposes a recorded
sensitivity parameter (default 0.25); it is not automatically fitted.

All decoding uses the same physical distance gate as the classical suite,
without extra top-K pruning, and only one-frame edges. Predictions are
thresholded by upstream score floor 0.05, so the learned candidate subset can
be smaller than the original geometric graph. `export-scores` puts learned,
motion-only and motion-plus-size costs on that identical subset, allowing
separate scorer and solver comparisons. The learned frame cost is
`0.5 - association_weight` with zero birth/death costs, reproducing the greedy
0.5 threshold apart from possible exact threshold ties; it is not a calibrated
negative log likelihood. Geometric costs retain their original units and
alternative policy. These remain configured model comparisons, not claims of
equivalent probabilistic assumptions.

Imported tables are keyed by exact `(from,to)` detection IDs, supplied in
`score_tables`, and selected by each run's optional `scorer` name. Each table
contains `costs: [{from, to, cost}, ...]` and `provenance`. Row order is irrelevant;
duplicates and edges outside the shared gate are rejected. Imported runs require
`size_weight: 0` because table costs already contain their feature combination.
The runner presently restricts imported scores to frame solvers: the legacy
segment solver's event costs use geometric units, so mixing them with arbitrary
learned energies would be misleading.

Tests for the optional bridge:

```sh
python tests/test_trackastra_adapter.py
```

On the initial B2 cache, GPU inference took about 5.3 seconds after feature
extraction, with 15,800 nodes, 18,681 predicted edges and 16,351 physically gated
edges. Both SCIP runs returned OPTIMAL within ten seconds of decoding. With
divisions, greedy produced 687 events and default global decoding 873; a split
cost of 1 reduced the latter to 425. No accuracy follows from these counts.
Blockflow's imported-score greedy exactly reproduced all 13,811 native
Trackastra greedy-no-division edges. Switching only to LAP removed one edge and
added two on these scores. This isolates the optimizer comparison without
repeating inference or changing masks.

Trackastra attribution and retained license are in
`scripts/TRACKASTRA-LICENSE.txt`. See its [official repository](https://github.com/weigertlab/trackastra)
and [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf).


## Explicit association events and a provisional motion model

`scripts/event_tracking.py` separates an `EventScorer` protocol, candidate-event
construction and a SCIP event solver. It is an optional Python prototype, not
an additional dependency of the Rust library. Inputs are the same immutable
Phase geometry rows used by the other fixed-mask comparisons.

An event has a stable key, kind, parent/daughter IDs, a scalar energy, named
energy terms and measured features. Birth/end events involve one detection;
continuations involve a parent and successor; a binary division involves one
mother and **both daughters in a single variable**. Generation uses the same
physical gate and no top-K pruning. Validation checks identities, temporal
ordering, cardinality, finite scores and the sum of recorded energy terms.

The optimizer chooses binary event variables. For every detection it enforces
exactly one incoming event and one outgoing event. A division is charged once
and supplies incoming events to both daughters, so the solver cannot select
only half of it. It minimizes the sum of selected event energies, including
births/ends. Returned assignments are checked independently against the event
constraints and solver objective; status, primal/dual bounds and gap are saved.
Unlike the two-stage LAP heuristic, all proposed continuation and paired
division events compete in the same optimization. Without age/history factors,
this version still factorizes over frame pairs; calling a whole-movie solver
does not itself introduce temporal evidence.

Brownian displacement variance per spatial axis is
`v = 2*D*elapsed_hours + 2*localization_sigma^2`. In two dimensions the motion
energy is `r^2/(2*v) + log(2*pi*v*rho)`, a negative Gaussian-density ratio to
uniform clutter with target-frame density `rho = detections / physical area`.
Thus the dimensional density is explicitly normalized relative to background.
Pilot spatial units must be micrometers. The default diffusion coefficient is
an **initial assumption**, not an inferred biological measurement.

The current division model uses motion of the daughter midpoint, sum-area,
daughter-area-balance and separation factors. Shape factors are provisional
quadratic energies, not calibrated likelihood ratios. They can be replaced
through the scorer protocol. Cell rounding changes apparent Phase area greatly,
so tight area-conservation assumptions are inappropriate. Birth/end penalties
are lower near image boundaries; the first/last observations are censored with
zero corresponding boundary cost. Starts are not assumed to be newborn cells.
The current division prior is an exponential lifetime with constant hazard;
**history-dependent cell age is not yet inferred**. Segmentation misses can
still create births/ends because gap closing and detection rejection are not
implemented in this first event prototype.

```sh
python scripts/event_tracking.py --suite-input INPUT.json --output NEW_DIRECTORY \
  --diffusion 8 --timeout 120 --threads 1
python tests/test_event_tracking.py
```

Use `--no-divisions --size-weight 0` for motion-only continuation, and
`--no-divisions --size-weight 1` for the same model with a weak size factor.
`--division-feature-weight` controls the provisional joint geometry/size
factors. Outputs include the complete proposed event graph, selected events,
ordinary source/target links and the exact executed script/input. Two division
links share one event energy; do not double-count their `event_cost` fields.

The first B2 runs reached a zero solver gap: motion-only generated13,296 links,
motion+size13,177, and the joint model13,957 with765 division events. These
settings are more fragmented than the older baseline; no superiority is
claimed. All15,800 detections are retained. Tests compare a tiny joint problem
with independent exhaustive event-subset enumeration, check elapsed-time
variance, verify large uint64 identities and reject invalid daughter events.
The prototype supplies a reproducible alternative model; parameter tuning must
use representative Phase reference data, not the selected GFP stress cohort.


## Segmentation-hypothesis stage (Ultrack 0.8.0)

`scripts/ultrack_adapter.py` is an optional, version-pinned experiment adapter.
It accepts one or more Phase-derived mask arrays (`--masks` can be repeated),
constructs foreground/contour representations and alternative hierarchical
segments, links them with the existing drift translations, and selects a
nonoverlapping tracking solution. It does not read fluorescence. Its output
`segments.zarr`, `tracks.csv` and `track-parents.json` have **new identities**;
never join the old feature tables to these labels by numeric ID.

```sh
python scripts/ultrack_adapter.py --suite-input INPUT.json \
  --masks ORIGINAL_MASK_ARRAY --output NEW_EXPERIMENT \
  --sigma 0 --min-frontier 0.5 --backend scip --seed-leaves \
  --timeout 120 --workers 4
python tests/test_ultrack_adapter.py
```

The source label proxy opens arrays read-only and supports optional crops and
frame prefixes for integration probes. Spatial scale and the physical link gate
come from the suite input. A constant vector field shifts each target back to
the preceding frame by the negative registration increment, normalized by the
image Y/X extent as required by Ultrack. Regression tests verify its sign and
units. This does not estimate optical flow or interpolate source masks.

Native CBC decoding and an experimental SCIP bridge are supported. The bridge
exports the SAME Python-MIP model to MPS, checks variable identities, reinstates
the original objective including its constant/sense, solves with SCIP and
checks the returned values against **every original constraint**, integrality
and bounds before persisting a selection. Tests cover maximization and
minimization. `--seed-leaves` supplies a nonoverlapping-leaf/greedy-link primal
seed whose feasibility is independently checked; it changes no constraints or
objective coefficients. An accepted seed is not evidence that optimization has
improved it. Reports record its objective and how many variables subsequently
changed.

`--reuse-work PREVIOUS_EXPERIMENT` forks the prior SQLite hypothesis/link
database using a consistent backup and re-solves in a NEW result directory.
This avoids repeating segmentation, preserves the original result, and permits
a runtime-limit change while retaining the other model configuration.
`--resume-solve` is reserved for recovery inside the same experimental run after
a terminal export failure. Configuration serialization uses Ultrack's aliases;
regression tests protect against silently loading default settings instead.

Initial evidence: the six-frame crop completed with both CBC and SCIP. On the
first full-field configuration the problem had429,270 variables; CBC found no
feasible solution in120 seconds and cold SCIP supplied only an empty solution.
A feasible warm start enabled a full mask export, but the optimizer did not
improve its objective within that budget (relative gap about201%). Label
pruning without contour smoothing reduced the problem to221,608 variables and
about83% gap, still without objective improvement in120 seconds. These are
**integration/feasibility outputs, not optimized production tracking results**.
A separate600-second budget reached the1% objective-gap target after469seconds
(primal3036.55342, dual3066.54465, gap0.0098767), but changed none of the seeded
variables and still selected zero divisions. This is not a useful lineage
candidate for the current experiment. A small aggregate objective gap does not
validate rare division events; further solver tuning is deferred.

See the [Ultrack documentation](https://royerlab.github.io/ultrack/) for its
hierarchical segmentation model and configuration. This adapter currently uses
only a single input-mask source in its full-field trials; the multiple-source
interface is prepared but ensemble performance is not yet established.

## Comparing different masks spatially

`track::matching::{match_label_frame, match_overlaps}` provides exact uint64
same-frame spatial correspondence. It maximizes the number of one-to-one
matches above an IoU threshold, then total IoU. Inputs must share the same native
pixel lattice and time; this is distinct from motion registration.
`match_overlaps` accepts merged overlap counts, allowing blockwise counting.
It validates object sizes and intersection totals, including zero/unknown IDs.

```sh
python scripts/prepare_mask_overlaps.py --reference PREVIOUS_MASK_ARRAY \
  --predicted NEW_MASK_ARRAY --reference-role previous_prediction \
  --output COUNTS.json
target/release/match-overlaps COUNTS.json NEW_MATCHING.json
```

The counting helper uses dense *indices*, not raw IDs, to avoid allocating
arrays of size `max(label)` when native uint64 IDs are large. The result
preserves its declared reference role. A match to an older prediction is
agreement, **not accuracy**. With a reviewed reference, boundary conventions
and temporal/spatial evaluation scope must be established separately.

In the first full-field leaf-seed experiment, IoU>=0.5 yielded13,916 matches,
1,884 unmatched original objects and4,326 unmatched new objects. These totals
balance to15,800 original and18,242 new detections; they do not classify errors
without an independent reference. Do not present mean matched IoU alone or drop
unmatched objects when evaluating a tracking method after segmentation changes.


### Comparing lineages across segmentation identities

`scripts/compare_spatial_tracks.py` accepts two suite inputs, their link JSON
arrays and the `match-overlaps` result. It verifies complete one-to-one spatial
matching coverage, separate identities and equal geometry/timeline. Unmatched
objects and links remain explicit in totals. An agreeing binary division must
match the mother and **both** daughters; matching one daughter is insufficient.
This reports prediction agreement, never reference accuracy. Example:

```sh
python scripts/compare_spatial_tracks.py \
  --first-input original.json --first-links original-links.json \
  --second-input alternative.json --second-links alternative-links.json \
  --matching spatial-matching.json --output NEW-agreement.json
```

In the B2 diameter35/50 experiment, 14,979 objects spatially matched at IoU0.5
(out of15,800/15,279). The learned greedy graphs shared13,077 links and386
exact division triplets (of687/649). A smaller division count did not establish
improvement: the diameter50 run reintroduced a previously removed neighboring
cell capture in a selected diagnostic case. These observations are development
diagnostics, not a held-out accuracy comparison.


## Continuity diagnostics and scored endpoint refinement

`scripts/tracking_diagnostics.py` reads a suite input, a link array and an
optional completed prediction-cache directory. It reports maximal unbranched
chain lengths, roots, isolated detections, starts/ends by acquired frame,
gap links, and the physical/learned support at nonterminal ends. Its score
threshold is inclusive, matching upstream greedy decoding. An end is not cell
death; nearby candidates are not proof of identity, and these are not accuracy
metrics. The retained three-movie diagnostic reports used a strict comparison
at0.5 before this boundary convention was corrected; all three caches were
checked to contain zero scores exactly0.5, so their results are unchanged.

```sh
python scripts/tracking_diagnostics.py --suite-input INPUT.json \
  --links LINKS.json --predictions PREDICTION_DIRECTORY --output NEW-diagnostics.json
```

`blockflow::track::refine::join_endpoints` accepts an existing graph and
`ScoredCandidates`. Its global sparse assignment can add continuations or
gaps, preserving all original links and costs. Candidates must join current
ends to current starts. Scores are **incremental costs relative to leaving
both unmatched**, whose cost is zero. Only negative-cost joins are eligible.
There are no implicit distance costs, temporal gates or birth/death penalties:
these belong to the caller's candidate/scoring policy and provenance. Existing
and added costs need not belong to a single posterior and must not be summed
as if they did. Existing wrong divisions cannot be repaired by this stage.

The release CLI keeps original link JSON records (including model weights),
archives inputs, and refuses existing output directories:

```sh
cargo build --release -p blockflow-tracking-suite --bin join-endpoints
./target/release/join-endpoints INPUT.json BASE-LINKS.json SCORES.json NEW_OUTDIR
```

Score schema is `blockflow.endpoint-scores/v1` with `scores` rows containing
exact uint64 `from`, `to`, finite `cost`, plus arbitrary `provenance`.
`scripts/prepare_endpoint_scores.py` supplies two explicit experimental policies:

* `learned_continuation`: adjacent-frame returned association scores,
  `threshold - weight`; default exploratory threshold0.3. No new divisions.
* `brownian_gap`: exactly one skipped frame (dt2), the original physical gate,
  and the provisional elapsed-time Brownian/size model from `event_tracking.py`.
  Cost is continuation plus missed-observation penalty minus separate end/birth
  costs. Its priors have not been independently calibrated.

These policies run as separate stages; their energy scales are never mixed in
one assignment. The missing-observation penalty is
`-log(1-observation_probability)`. For example:

```sh
python scripts/prepare_endpoint_scores.py --suite-input INPUT.json \
  --links BASE-LINKS.json --mode learned_continuation \
  --predictions PREDICTION_DIRECTORY --threshold 0.3 --output NEW-SCORES.json
```

Continuity improvement must still be assessed for false joins and missed divisions on
reviewed Phase references; division count preservation is not biological
validation.


## Frozen development comparison

`scripts/run_frozen_comparison.py` replays a content-hashed comparison package:
release binaries, Python methods, model weights, fixed control settings and
admitted suite inputs. It can reuse the exact completed prediction cache or
rerun Phase prediction for a listed development case. The runner checks file,
model, source-pixel, input and cache hashes plus runtime package versions before
replay. Stages keep separate logs and a completion journal; existing output
directories are refused. Unlisted inputs cannot silently become held-out tests.

The Incucyte package `comparison-freeze-v1` lists five candidates: the original
segment LAP, learned greedy, learned global, refined learned greedy, and joint
Brownian/division assignment. A full B2 cached replay reproduced all five link
records exactly. Its reference evaluation is still null: reviewed development
Phase references and acceptance criteria must precede new held-out execution.
This is a comparison freeze, not a production selection.

```sh
python FROZEN/run_comparison.py --capsule FROZEN \
  --suite-input FROZEN/inputs/B2.json --reuse-predictions EXISTING_CACHE \
  --manifest-sha256 EXPECTED_SHA256 --output NEW_COMPARISON
```


## Cached shape features and replaceable event factors

`scripts/measure_tracking_shapes.py` measures label geometry once: covariance
axis ratio and major/minor axis lengths. It compresses exact uint64 identities,
checks every mask area/centroid against the fixed input, and records source
pixel hashes. It does not measure any intensity channel. Axis lengths are in
native pixels; this initial scorer requires isotropic Y/X spacing.

```sh
python scripts/measure_tracking_shapes.py --suite-input INPUT.json \
  --masks NATIVE_MASK_ARRAY --output NEW_SHAPES.json
python scripts/event_tracking.py --suite-input INPUT.json --output NEW_RUN \
  --shape-features NEW_SHAPES.json --mother-elongation-weight 2 \
  --relax-projected-area --diffusion 16.752324891196903
python tests/test_shape_tracking.py
```

`ShapeEventScorer` wraps the existing `EventScorer`; candidate generation and
assignment remain unchanged. Only division energies change. An elongated
mother receives the soft penalty `-weight * log(axis_ratio)`. Optional projected
area relaxation multiplies the existing `sum_area` term by
`(1-axis_ratio)^2`: a rounded mother may have much less projected area than its
spread daughters. Continuation, birth and end terms retain the wrapped scorer's
values. Other shape factors can implement the same protocol with premeasured
features. The wrapper validates the exact input/geometry and archives its
executed code and feature sidecar with the run. Defaults retain the old model.
These factors are provisional energies, not calibrated mitosis probabilities.

The example above is a selected Incucyte development configuration, not a
universal recommendation. The diffusion scale was fitted to 31 assistant
reviewed C2 centroid displacements with localization sigma fixed at 1.5um.
One observation cadence cannot identify diffusion and localization separately;
mask/morphology jitter also contributes. The combined variant preserved 29/31
selected C2 continuations versus 28/31 for LAP and the original joint model,
and restored one exposed B2 paired-division control. In the separate C6 sample
it still broke a clear continuation preserved by LAP. These small assistant
reviews do not justify replacing the baseline or claim population accuracy.
The original frozen comparison remains reproducible with its original code.


## Conservative division refinement and temporal mask context

`scripts/refine_divisions.py` reconsiders binary divisions from an existing
fixed-mask graph while preserving every ordinary continuation/gap link record.
Each component compares three alternatives: keep the mother/both-daughter
event, continue to A and leave B as a new root, or continue to B and leave A as
a new root. Exactly one proposed successor is retained after removing a split;
no new association is introduced. These local constrained problems factorize,
so direct enumeration gives their optimum without a general MIP. This is not
an unconstrained global tracking optimum. New roots are unmatched observations,
not asserted biological births. Wrong ordinary links and masks cannot be fixed.

```sh
python scripts/refine_divisions.py --suite-input INPUT.json \
  --base-links BASE_LINKS.json --shape-features SHAPES.json \
  --diffusion 16.752324891196903 --mother-elongation-weight 2 \
  --relax-projected-area --margin 1 --output NEW_REFINEMENT
python tests/test_temporal_division_refinement.py
```

The improvement margin leaves a split unchanged unless an alternative improves
its explicit energy by more than that value (CLI default0). It is an abstention
policy in model-energy units, not calibrated probability/confidence. In the
selected Incucyte development study, margin0 removed a source-supported late
B2 division by only0.063energy units; a single declared margin1 sensitivity
retained it while preserving the observed continuation gains. This selection
used exposed development data. The values above are experimental study settings,
not defaults validated across datasets.

`decisions.json` records every alternative, named energy term, feature,
selected children, new roots, margin, prior link record and input/model/code
provenance. All ordinary links are checked unchanged and all output edges must
be a subset of the input edges. Legacy link cost/weight fields retain their
original meaning; do not sum them with the separate new local event energies.
Outputs and source masks/graphs remain separate; existing directories refused.

Optional label-only context is measured once and can wrap any event scorer:

```sh
python scripts/temporal_mask_features.py --suite-input INPUT.json \
  --masks NATIVE_MASK_ARRAY --output NEW_OVERLAPS.json
# Add these flags to refine_divisions.py to test the contextual wrapper:
# --temporal-overlaps NEW_OVERLAPS.json --context-weight 2
```

Adjacent and lag2mask intersections use stored global drift, rounded to native
integer pixels with no wrap. Source hashes, shifts/rounding errors, exact uint64
IDs, areas and overlap caps are retained/checked. This does not estimate
individual-cell motion or optical flow and reads no intensity channel.
`TemporalEvidence` measures positive counterevidence: a daughter overlaps another
preexisting parent more than its proposed mother; two children largely rejoin
one future mask; or they recover distinct bodies that preceded a merged mother
mask. No overlap is not evidence against mitosis. Temporal factors are soft,
provisional energies; the future term uses offline observations and is censored
at the movie end. True daughters can touch or become merged by segmentation,
so these signals must not be treated as certain biological nondivision.

In the selected assistant-reviewed development sets, shape/motion refinement
preserved30/31C2continuations versus28/31LAP, and11/13B6versus10/13LAP, with two
B6unmatched-mask intervals retained in denominators. Margin1 retained the two
reviewed exposed B2paired-division controls and all seven older selected B2
triplets. A separate, explicitly exposed C6regression retained24/24reviewed
continuations and its one paired division. This is not a fresh held-out test.
No population division recall/precision or production superiority is inferred.
Additional temporal penalties removed more divisions without an extra gain on
these selected continuation tests, and some sampled removals remained ambiguous.
Keep the contextual wrapper experimental; preserve the original LAP baseline.
