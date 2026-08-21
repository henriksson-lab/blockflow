# Segmentation, morphology and object measurement

*An ops survey for `blockflow`. Family B of four: this document covers turning
intensities into objects and measuring the objects. Point operations, filtering
and intensity transforms are family A (`filtering-and-transforms.md`); geometry,
registration and composition are family C; the channel axis and the time axis —
unmixing, channel arithmetic, colocalisation, tracking, temporal filtering — are
family D (`channels-and-time.md`).*

---

## 0. What this document is for

The uninteresting question is "does ImageJ have it". The useful one is:

> **Can `blockflow`'s decomposition model express this operation, and at what
> cost?**

Every row below is classified against that question, using six categories the
crate has already earned by building things that fell into them. A feature
checklist ages out in a release; the classification is the part still worth
reading in a year, because it says what the *framework* would have to grow, not
what a caller would have to type.

Sources weighted by what is actually used rather than by what exists:
scikit-image (`segmentation`, `morphology`, `measure`, `feature`), OpenCV,
ImageJ/Fiji as it is actually driven (`Analyze Particles`, the `Auto Threshold`
and `Auto Local Threshold` sets, MorphoLibJ, the 3D suite,
`Skeletonize3D`/`AnalyzeSkeleton`, TrackMate, trainable segmentation), ITK, VTK,
and the operation set that commercial analysis packages converge on.
Where scikit-image and OpenCV agree on a primitive that Fiji offers only as a
plugin, that agreement is treated as evidence the primitive belongs in a general
library.

### The six categories

| # | Category | What it means | The crate's own evidence |
|---|---|---|---|
| 1 | **Bounded reach** | A halo the op can declare, derived from its parameters. Cheap and blockable. | `morphology`, `rank`, `local`, `adjacency` (reach 1), `walk` (reach = the offset list) |
| 2 | **Whole-axis reach** | `AxisReach::All` on one axis, free on the others. Separable sweeps; the swept axis drops out of `splittable_axes` and the other two stay cuttable. | The exact Euclidean distance transform, §5 — and see §5's correction: it declares in the **phase** frame, which is the only frame in which this category is plannable |
| 3 | **Whole-volume, resident** | No halo bounds it. The op declares `Reach::all` and becomes a planning barrier with the grid collapsed to one block. | `watershed` — declared, with three reasons written down |
| 4 | **Iterative to a fixed point** | A loop of rounds whose stopping rule is a whole-volume reduction. The phase's external reach is *one substage's* reach however many substages run, because blocks exchange outputs between substages rather than re-deriving them from an ever-wider read. | `reconstruct` (grey reconstruction, h-extrema), `configuration_to_fixed_point` |
| 5 | **Fragment-and-join** | A per-block partial result merged globally. Anything with a global connectivity requirement lands here. | `fill`, `regional`, `detect` — one program in `components` |
| 6 | **Not expressible today** | And why, specifically. | See §12 |

**One note on numbering, so the four surveys compose.** Family A's survey uses
categories 1–4 with the same meanings and then numbers *not expressible today*
as **5**, because family A has no operation with a global connectivity
requirement and so never needed the fragment-and-join row. Here fragment-and-join
is **5** and *not expressible* is **6**. Where this document says "category 5" it
means the `fill`/`regional`/`detect` shape.

Family A also numbered the crate's four structural gaps **G1–G4**, and this
document points at them by number rather than restating them:

| | |
|---|---|
| **G1** | *(Renamed in the index's §8.1; identifier unchanged. This document restated family A's name, "no rank-reducing phase", and that name is wrong — see the correction below.)* **The geometry cannot declare a collapsed or a broadcast axis.** `Reach` is per-axis halo widths on a same-rank output and there is no way to say that an axis is consumed whole and written at extent 1, or held at 1 while the output grows. A side output may have any rank, but side outputs are **terminal** — nothing in the chain reads one back |
| **G2** | No phase whose inputs sit at different offsets or resolutions. A `Decomposition` reads one image and maps a block to one region of it; extra images are still read at the block's own extent |
| **G3** | No complex element variant, so a spectrum cannot be a plan image |
| **G4** | The cost model cannot price `log n` — `cost_per_voxel()` takes no volume argument |

> **Corrected — measured** (`tests/collapsing_phase.rs`; consolidated in the
> index's §8). *What this document said,* following family A: a phase cannot
> reduce rank, because `output_shape` is `[usize; 3] -> [usize; 3]` and an image
> is rank 3. *What was measured:* the rank cap is a **deliberate floor**, not a
> limitation — `src/voxels.rs` argues for rank 3 against `ArrayD` on measured
> grounds, and one and two dimensions are modelled as three with degenerate axes
> precisely so that lower rank adds no element types. So `[X, Y, 1]` is an
> ordinary `Voxels`, and a phase that collapses an axis **plans, runs and is
> decomposition-invariant today** — through the `†` cross-grid escape, that is,
> by *not* declaring the dependency. Family C classified this `2†` and was
> right. What G1 names, correctly, is the missing **declaration**, and it has a
> second half this document never saw: the **broadcast**, `[X, Y, 1] -> [X, Y, Z]`,
> which is what applies a global threshold level back over a volume (§1) and
> which also runs today by the same escape.
>
> One consequence lands squarely in this family and is new: **nothing checks a
> stated fetch against the dependency it stands in for.**
> `Decomposition::check` verifies that a block's `source` lies *inside* the
> image it reads and never that it *covers* what the reach claimed, so a
> collapsing phase that reads one plane instead of the whole axis is accepted by
> every guard and is wrong at every position. That is the index's **G14** and it
> is a correctness gap, unlike everything else in §12.
>
> **Landed, and one clause of that prediction was wrong.** The geometry change
> shipped. A collapsing phase can now declare `AxisReach::All` in
> `Space::source_voxels()` and the declaration is **checked against the fetch**,
> so the two traps above are refused by name where the op declares. Two things
> did *not* happen. The `†` route did not become an ordinary one — the fetch is
> still stated per block, so the builder is still hand-written. And **G14 did
> not close**: the check holds an op to what it *said* and cannot make an op
> speak, so a phase declaring `Reach::none()` that reads one plane of an axis it
> means to consume still plans, still runs, and is still wrong at every voxel.
> G14 is partially closed, and §12 carries the residue.
>
> One consequence lands in this family in particular and is in §5: the same
> declaration, on an axis a phase keeps and consumes, is now a **whole-axis
> mandate** — which is a free partial answer to the constraint §5 asks for.

**Two facts about categories 4 and 5 that shape most of what follows.**

*Category 5 costs N+1 passes for N blocks.* Phase 0 labels each block locally at
reach zero and emits six face planes. Phase 1 has a whole-lattice fragment
reach — and `fragment_phase` sets `halo = max(reach, fragment reach × block
edge)`, so phase 1's halo is the whole volume, so **every block of phase 1 reads
the entire intermediate image**. The coupling is load-bearing, not a bug: phases
are pipelined rather than barriered, and the halo is what makes block *b* of
phase 1 wait for the phase-0 blocks whose fragments it reads. `fill.rs` states
this outright. A hole fill at 168 blocks therefore reads the volume 169 times.
The named way out is a **barrier phase** — a phase declared to start only when
the previous one finished needs no halo to express the same dependency — and
that is the single largest cost lever in this family.

*Category 4's stopping rule must be global, and the crate has measured why.* A
per-block "this block changed nothing" test is unsafe for thinning: a region
buried inside structure thicker than itself has no border voxels at all until
the erosion front arrives, so it removes nothing in one round and something
several rounds later. `directional.rs` refuses to offer a per-block convergence
entry point on exactly that ground. The same file records the second half:
`clear_faces` and `faces_are_clear` are **volume-level** preconditions, and
applying either per block would zero a face of every interior block — carving a
hole along every seam. So for thinning, both the stopping rule and the border
handling are forced to be whole-volume, and the shape is not a choice.

---

## 1. Thresholding

| Operation | Category | Have it? |
|---|---|---|
| Fixed global threshold, `>` and `>=` | 1 (reach 0) | **Yes** — `voxelwise::Threshold`, both tests, because they disagree on every voxel of a background that is exactly at the level |
| Two-sided window (band) threshold | 1 (reach 0) | Expressible as two thresholds and an `and`; not shipped as one op |
| Global auto-threshold from the histogram: Otsu, Li, triangle, max-entropy, IsoData, moments, Yen, minimum, percentile | **5, and unnecessarily so** — see below | **No.** None of the standard set exists anywhere in the crate |
| A volume histogram as a first-class artefact | 5 (the fragment is a bin vector, merged by `+`) | **No** |
| Local/adaptive threshold against a windowed statistic | 1 (two-term reach: window + lattice spacing) | **Yes** — `local::AdaptiveThresholdOp`, `value` vs `scale × statistic + offset` |
| The local statistic itself: mean, standard deviation, arbitrary rank/percentile, IsoData over the window's histogram, caller-supplied reducer | 1 | **Yes** — `local::Statistic` |
| The mean+k·σ and median forms of local thresholding | 1 | **Yes**, by composition — the affine `scale`/`offset` is the parameterisation |
| Local threshold over a masked population | 1 | **Yes** — `masked_local_statistic_into`, population read from a `Bool` image |
| Statistic on a sample lattice with interpolation between samples | 1 | **Yes** — `SampleLattice`, globally anchored so it is decomposition-invariant |
| Hysteresis (two levels, keep the low-level components that touch a high-level seed) | 5 | **No** |
| Multi-level / multi-Otsu | 5 for the levels, 1 for the application | **No** |
| Background-subtract-then-threshold (low-pass, subtract, threshold the difference) | 1 | The pieces are there (`smooth`, `voxelwise`, `normalise`); not one op. Family A owns the low-pass |

**The local threshold set here is good and the global one is empty.** That is an
odd shape: an adaptive threshold is the harder operation and the crate has a
strong version of it, including a windowed IsoData, which is unusual — it is
explicitly not an approximation of the window's mean, and on a bimodal window the
two are far apart. Meanwhile the single most-used operation in the whole of image
analysis, "threshold this volume at Otsu's level", cannot be done.

**Why it is category 5 and why that is worse than it needs to be.** A global
auto-threshold is a two-stage reduction: a whole-volume histogram, then a
pointwise comparison. The histogram merges by `+` per bin — associative,
commutative, exact in integers, and the fragment is a few kilobytes rather than
six face planes. But phase 1 needs *every* block's histogram, so it declares a
whole-lattice fragment reach, so it inherits a whole-volume halo, so it reads the
whole volume per block. **A reduction whose answer is one scalar pays the same
N+1 read amplification as a hole fill**, and there is no reason in the
mathematics for it — only in the pipelining. This is the clearest single argument
for the barrier phase `fill.rs` names.

**One thing this section left out, and it is G1's other half.** A global
auto-threshold is stated above as "a whole-volume histogram, then a pointwise
comparison". The second step reads *one number* and applies it at every voxel —
a source held at extent 1 while the output spans the volume, which is a
**broadcast**, and it is as undeclarable as the reduction is. It runs today by
the same `†` escape (`BlockOp::takes_extent_from_placement` is the waiver, and
`ops::lattice`'s interpolate half already declares it), so nothing here is
unbuildable; both halves are hand-written and unchecked. The two are one
statement seen from two sides and the index records the unification once, in
its §9.

Two smaller notes. Commercial suites have converged on a similar small method set
(Otsu, IsoData, triangle, a peak split, a σ-based rule) plus a band rather than a
cut, which is worth knowing when choosing which methods to ship: five well-chosen
ones cover essentially all practical use. And a threshold picked *interactively*
by extending a window around clicked values is application logic, not a library
op — but it needs the histogram, which is not.

---

## 2. Binary morphology

| Operation | Category | Have it? |
|---|---|---|
| Erosion, dilation | 1, reach = the element | **Yes** — `morphology::erode_into`/`dilate_into` |
| Opening, closing | 1, reach = **twice** the element | **Yes**, written *once* as compositions, so the reach falls out of the composition rather than being asserted. A hand-written opening is the classic op whose author writes `radius` and is wrong by a factor of two with no symptom except at block seams |
| Structuring elements: box, inscribed ellipsoid, an open ellipsoid matching another convention, arbitrary offset list, stepped/decimated, asymmetric (even extents, two-sided reach) | — | **Yes** — `element::StructuringElement`, including `from_offsets`. The element set is more general than most libraries' (OpenCV: rect/cross/ellipse; scikit-image: a footprint array, which `from_offsets` matches) |
| White and black top-hat | 1, reach = twice the element | **No op**, but a `Chain::Parallel` diamond (open, source) into a voxelwise difference is exactly it. Worth shipping because the reach is easy to get wrong |
| Morphological gradient (dilation − erosion) | 1 | Same: expressible, not shipped |
| Hit-or-miss | 1, reach 1 | **Yes, generically** — `configuration` rewrites a mask by a caller-supplied table over the 27-bit 3×3×3 neighbourhood, and its own header names hit-or-miss as one of the things such a table is. `ConfigurationTable::assign_matching` builds one from templates without evaluating a rule 134 million times |
| Binary reconstruction from a seed under a mask | 4 | **Effectively yes** via `reconstruct` — but see the caveat: `HExtremaOp` and the reconstruction shell `accepts` **`f64` only**, so a `bool` mask has to be widened. The kernel is generic over `Copy + PartialOrd`; the shell is not |
| Hole filling | **5**, N+1 passes | **Yes** — `fill`. Connectivity is the *background's* and is a parameter |
| Border clearing (drop components touching the volume face) | **5**, and it is `fill`'s exact program | **No** — and this is the cheapest missing op in the family. `fill` already computes, per component, "does this reach the outside"; border clearing keeps the components for which that is *true* on the foreground instead of filling the ones for which it is false on the background |
| Area/size opening (drop components below a voxel count) | 5 | **No** — same program, per-component fact is a count rather than a boolean |
| Keep components hit by a marker (binary reconstruction by markers) | 5 | **No** as an op; it is the same program with the per-component fact "does any marker fall in me". Commercial suites expose this with a keep/drop switch and it is heavily used |
| Ultimate erosion (stop each component one step before it vanishes) | 4 + 5 | **No** |
| Thinning / curve skeletonisation, stated pass count | 1, reach `8n` (subfield) or `12n` (directional) | **Yes, twice** — `skeleton` (8 sub-iterations per pass, subfield-restricted, each a `BlockOp`, the pass a `Chain::Sequence` so the reach is the fold) and `directional` (the published 12-sub-iteration template algorithm, where the *pass* must be the op because a sub-iteration reads a second array — the border set, taken once per pass and deliberately stale — that a `Sequence` has nowhere to thread) |
| Thinning to a fixed point | **4**, and the stopping rule *must* be a whole-volume reduction | Whole-volume free functions only (`thin_to_fixed_point`, `directional_to_fixed_point`). Not dressed as ops, deliberately |
| A table-driven rule run to a fixed point | **4**, external reach 1 whatever the substage count | **Yes** — `ConfigurationFixedPointOp`. So a thinning rule *expressed as a 3×3×3 table* can already run to a fixed point inside the framework; the two published thinning schemes cannot, because their sub-iterations are not one table each |
| Pruning (remove skeleton branches shorter than *k*) | 1 for *k* rounds of end-point deletion; 4 for "prune to stability"; 6 for "prune branches shorter than a length" | **No**. The first two forms are a `configuration` table away. The third needs the skeleton graph — §10 |
| Medial axis with a radius per voxel | 2 (once a distance transform exists) | **No** |
| Skeleton by influence zones of the background | 3 | **No** |

**A note on which skeleton.** `skeleton` and `directional` are not variants of one
thing. The subfield rule anchors on a voxel's coordinate parity — a fact about
*where the block is*, which is why `ThinningOp` reads `Anchor` — while the
template rule is translation-invariant and needs no anchor at all. Both produce
**curve** skeletons; neither offers a surface-preserving end-point rule, and
`skeleton.rs` argues that is a second algorithm rather than a parameter.

**What is missing at the top of this section is small and cheap.** Top-hat and
morphological gradient are compositions the crate can already express and should
ship for the same reason opening and closing are written as compositions: so the
reach is derived rather than declared. Border clearing, area opening and
marker-based selection are one program — `components`' program — with a different
per-component fact, which is precisely the refactor `fill` and `regional`
already went through.

---

## 3. Grey morphology

| Operation | Category | Have it? |
|---|---|---|
| Grey erosion / dilation | 1 | **Yes, but not in `morphology`** — `morphology` `accepts` `Bool` and `F64` and is a set operation. Grey erosion is `rank_filter_into` with `Rank::lowest()`, dilation with `Rank::highest()`, over every type but `f16` |
| Grey opening / closing | 1, twice the element | Compositions of the above; not shipped as ops |
| Grey top-hat | 1 | **No** |
| Morphological gradient on intensities | 1 | **No** |
| Rank/percentile filter over an element, including a masked population and an excluded centre | 1 | **Yes** — `rank`, with `ExcludedCentre` for the case where the centre must not vote |
| Sliding-histogram rank over a scan line | 1, with state between voxels | **Yes** — `sliding`, the one op whose kernel carries state, with a stated bounded-integer element-type constraint rather than silent binning |
| Grey reconstruction by dilation and by erosion | **4** | **Yes** — `reconstruct::Reconstruction`, one loop for both polarities |
| h-maxima / h-minima transform | **4** | **Yes** — `HExtremaOp`, with `h` asserted to be a prominence threshold in intensity units against peaks of known prominence |
| Regional **maxima** | **5** | **Yes** — `regional`, as `fill`'s two phases with a different per-label fact |
| Regional **minima** | 5 | **No** — see below |
| Opening/closing by reconstruction, top-hat by reconstruction | 4 + 1 | **No**, but compositions of what exists |
| Extended maxima/minima (h-extrema followed by regional extrema) | 4 then 5 | Composable from what exists |

**`regional` is narrower than its name.** It computes regional *maxima* only.
There is no polarity parameter and no minima entry point, and the module's
argument for taking one `Connectivity` rather than two does not extend to
polarity. This matters more than it looks, because **regional minima of a
gradient are the standard seed source for an unseeded watershed**, and that is
the recipe the crate cannot complete (§7).

Also worth stating: the reconstruction shell accepts `f64` only. A binary
reconstruction — which is what "keep the components a marker touches" is — is
therefore an `f64` volume's worth of memory for a one-bit question.

> **That observation has since been measured from another direction, and it is
> now a register entry.** The same eight-bytes-for-one-bit cost turns up wherever
> a mask is an *image* rather than a buffer inside an op: a verdict is an
> `f64 → f64` `MapFn` (`voxelwise::Threshold` is one), and
> `voxelwise::LogicCombine::accepts` takes only `Bool | F64` **and requires every
> branch of a fan-in to agree**, so one arm that cannot narrow binds the rest and
> the image the phase writes is `f64`. On a binarising stage at tile scale that
> is **57.853 → 46.28 GiB** at the peak — the difference between a stage that
> runs at a size and one that does not. Registered as **G15** in the index, with
> this paragraph cited as the independent sighting. `NarrowOp::to_mask` is
> already the crate's way into `Bool` and already carries the mask convention, so
> what is missing is a verdict that lands on `Bool` without a round trip through
> `f64`, and an `accepts` that does not force the widest arm on the rest.

---

## 4. Labelling and connectivity

| Operation | Category | Have it? |
|---|---|---|
| `Connectivity` as a stated parameter: 6, 18, 26 | — | **Yes** — `components::Connectivity`, defaulting to faces, taken separately by `fill` (the background's), `regional` and `detect` (the foreground's), because the complementary-pair convention deliberately pairs a narrow one with a wide one |
| The union-find, the six-face geometry, the seam walk, the seeded flood over a per-voxel membership test | 5 | **Yes** — `components`, extracted rather than copied. It is machinery, not surface |
| **Connected components as a globally consistent label volume** | **5** — the program exists, the op does not | **No.** This is the largest single gap in family B |
| One point (or one measured row) per connected region | 5 | **Yes** — `detect`, a `fragments -> fragments` phase pair that writes no image at all |
| Label filtering: drop labels by size, by shape, by intensity | 5 | **No** as a volume op. `detect`'s `Emission::Measured` rows can be filtered by `rows::FilterRowsOp`, but there is no way to project a filtered row set back onto the label volume — `label::LabelPointsOp` stamps *points*, which loses the object's shape |
| Relabelling / renumbering | 1 given a map | **No** |
| Object splitting (distance transform → watershed) | 2 + 3 | Blocked on the distance transform, §5 |
| Object merging by a criterion | 5 | **No** |

**Why the missing label volume is the big one.** `fill` phase 0 already writes
block-local labels as a `u32` image, and phase 1 already closes them into global
components with a union-find. What it then does is *rewrite them into a mask*.
An op that rewrote them into **global label numbers** instead would be the same
program with the same fragments and a different final map — and `fill.rs` itself
names the shape: "the three-phase shape — label, merge, relabel". Without it:

- `tabulate` takes a label volume and there is **no op in the crate that produces
  one**. The most complete per-object measurement the crate has cannot be driven
  by the crate's own segmentation.
- The whole "segment, label, filter labels, measure, render" pipeline that
  `Analyze Particles`, `connectedComponentsWithStats` and `regionprops` all
  assume is broken at its second step.

The only global numbering that does exist is deliberate and instructive:
`adjacency` refuses to number the set voxels *because* a number in that scheme is
a fact about the whole volume, and emits coordinate pairs instead. That is the
right instinct, and it is exactly why a *label* — which genuinely must be a global
fact — needs the third phase rather than a trick.

---

## 5. Distance and geodesic

| Operation | Category | Have it? |
|---|---|---|
| **Exact Euclidean distance transform** | **2** — three separable sweeps, each `AxisReach::All` on the axis it sweeps and reach zero on the other two, plus a pointwise finish | **Specified, and not present here.** Built in a sibling application crate as four phases; its author states the op bodies are ordinary `BlockOp`s with nothing application-specific in them and belong in `blockflow::ops::distance`, and that `ops/watershed.rs` already says it wants one. Verified there against a brute-force nearest-background search and, bit for bit, against a reference field over 77 045 760 voxels. **Read the category with the precision below** |
| Approximate / chamfer distance (OpenCV's 3×3 and 5×5 masks) | 1 per pass, but 4 as a parallel propagation | **No**, and there is evidence against wanting it: a chamfer propagates a fixed neighbourhood's weights and is wrong by a few percent in directions the mask does not represent — it looks right on a sphere and fails on a thin diagonal sheet. The separable exact form is `O(n)` per lane anyway, so the usual reason for a chamfer does not apply |
| Signed distance | 2 + 1 | Free once the transform exists: two transforms and one voxelwise combine |
| Distance to the nearest **labelled** object (feature transform / nearest-label field) | 2, with a second output | **No.** The mechanism is there — a `BlockOp` may declare side outputs with their own dtype — so this is a second array, not a framework question |
| Geodesic distance / distance under a mask | **4, and the exactness is lost** | **No** |
| A bounded outward walk that reports the distance at which a test first holds | **1** — reach = the offset list, exactly | **Yes** — `walk::OffsetWalkOp`, and it is the honest bounded form of the unbounded question. A walk that reaches the end of the list reports *that* rather than a distance, and a target inside the volume but outside the fetched window is **refused** rather than truncated |

> **Precision added after measurement**, so that this row is not cited for more
> than it supports. The sweep declares `AxisReach::All` on the swept axis with
> **no `.in_space(..)`** — so it lands in the default `Space::phase_voxels()`,
> the phase's own frame — and it does **not** change shape. Its lattice cuts
> only the two free axes, so the swept axis is never cut, and it is planned by
> an ordinary `PlanBuilder::pixels` phase. **A same-rank whole-axis stencil,
> category 2, taking neither escape.** That matters because the *other* frame
> behaves differently: `AxisReach::All` in `Frame::Source` — which is what a
> shape-changing sweep would need — was refused unconditionally at the time this
> was measured, and since the geometry change landed is granted on an axis the
> op consumes and **checked against the fetch** (index §8.1, §8.5). So this op
> is evidence that category 2 works in the **phase** frame, and is **not**
> evidence about `Frame::Source`, about a collapsed axis, or about the `†`
> route. The claim in this row was already the narrow one; the distinction it
> did not draw is now drawn — and it is now a distinction between two things
> that both work rather than between one that works and one that cannot.

**The one thing an exact transform wanted from the framework and did not get.**
*(Read with the update at the end of this note: half of it has since arrived.)*
`AxisReach::All` on one axis correctly drops that axis from `splittable_axes` and
leaves the other two cuttable, and a lattice that grants a short halo against a
whole-axis reach is refused by name. A lattice that *cuts* the swept axis with
the full halo is accepted and is right — every block re-reads the whole lane, so
every core is trustworthy and the regions tile. It is redundant, not wrong. But
`BlockConstraint::Extent` mandates all three block extents or none, so an op
declaring whole-axis reach on one axis can only *permit* the lattice to leave
that axis whole, never *require* it. **A one-axis extent constraint** —
`FullExtent(axis)`, or an `Extent` of `Option<usize>` — is the whole of the gap,
and it is a cost gap rather than a correctness one.

> **Half of this arrived, and by refusal rather than by constraint.** Since the
> geometry change landed, an op declaring `AxisReach::All` in **`Frame::Source`**
> mandates that the axis is left whole *or* given a whole-axis halo: the block's
> read has to span the axis for the clamp exception to be granted, so a cut axis
> under a finite halo leaves every block degenerate and the tiling check refuses
> the plan. Pinned on `[11, 4, 4]` cut `[4, 4, 4]`: halos `none()` and
> `[3, 0, 0]` are refused, `Reach::all()` makes every block read `0..11` and the
> plan checks. **No `BlockConstraint::FullExtent(axis)` was added and none is
> needed for correctness** — the op declares what it consumes and the guard that
> already ran enforces it.
>
> Two qualifications this family should hold onto. **The frame matters:** the
> distance transform declares in the *phase* frame (see the precision above),
> where the axis is at full extent and `is_whole` already drops it from
> `splittable_axes`; the source frame is what a phase that consumes an axis *it
> does not reproduce* uses, and is the one that now carries the mandate. **And
> the planner-facing half is still open:** `Constraints`/`BlockConstraint` still
> cannot say "do not cut axis *k*", so the enumerator can propose a lattice that
> will be refused rather than avoiding it. G9 in the index's register has been
> re-scoped to that half — still a cost gap, now a *planning-quality* one rather
> than a missing capability.

**Why geodesic distance drops two categories.** A separable sweep is exact
because a 1-D lower envelope over a lane is a function of that lane alone. Put a
mask in the way and it is not: the shortest path under a mask leaves the lane.
So a masked or geodesic distance falls back to iterated propagation to a fixed
point — category 4, which the framework handles well — and loses exactness with
whatever neighbourhood the propagation uses. That trade is worth writing into
the op's documentation rather than discovering.

---

## 6. Region-based segmentation

| Operation | Category | Have it? |
|---|---|---|
| **Seeded watershed** over a caller-supplied cost volume, with a caller-supplied mask | **3** — declared, with reasons | **Yes** — `watershed`, an MIT shell over a vendored BSD-3 translation of a reference implementation kept in its own file so the notice travels with it |
| Watershed lines vs. touching basins | — | **Yes** — `Separation::Line` / `Separation::Adjacent` |
| Marker-controlled watershed | 3 | **Yes** — that is what "seeded" means here |
| Distance-based watershed (the standard splitter for touching convex objects) | 2 then 3 | Blocked on the distance transform. The design is right: the cost volume is the caller's, so all three classical variants are one op with a different cost |
| Gradient-based watershed | 3, with family A producing the gradient | Same |
| Unseeded watershed (regional minima of the gradient as seeds) | 5 then 5 then 3 | **No** — broken at two links: no regional *minima*, and no connected-component label volume to turn them into seeds |
| Watershed with a depth/merge threshold | 4 then 3 | **No**, but h-minima before the flood is the standard way to get it and `HExtremaOp` exists |
| Region growing from seeds under a predicate | 4 or 5 | **No** |
| Level sets, active contours (Chan–Vese, geodesic active contours) | **4 in shape, 6 in fact** — see below | **No** |
| Graph cuts / max-flow | **6** | **No** |
| Random walker (a global sparse solve) | **6** | **No** |
| Superpixels: SLIC, Felzenszwalb, quickshift | **6** for SLIC and 6 for Felzenszwalb | **No** |
| Contour extraction / marching cubes from a label volume | rank-changing output | **No**. VTK's emphasis — contouring and surface extraction from labels — has no analogue here, and the output is a mesh rather than a volume |

**The watershed barrier is honest and should stay.** Three separate things make
the answer a function of one global queue's pop order: priority is `(cost, age)`
with `age` a global push counter, ties on both keys resolve by the queue array's
internal layout, and a voxel's priority is raised to its source's, so a flood
carries the worst cost it has crossed — a basin can be decided by a barrier
hundreds of voxels away. The first and third are properties of the algorithm; the
second is a property of the implementation and is reproduced because the
reference this op must agree with bit-for-bit has it. A *different* seeded
watershed could be decomposable; this one is not, and a test runs a blocked
version and counts the voxels that move rather than arguing.

One consequence worth carrying into measurement: **a basin is not a connected
component.** With `Separation::Line` the line clears voxels between labels, so a
basin is routinely cut into several connected pieces — measured at 138 basins
against 383 six-connected components on one fixture. Counting components of a
watershed output is not a way to count basins.

**Why level sets are category 6 rather than 4.** The per-iteration update is a
local stencil, which `iterate` handles exactly right: the phase's external reach
is one substage's however many substages run. But the Chan–Vese update needs the
two *region means* — a whole-volume reduction — recomputed every iteration and
broadcast to every block. There is no way today for a substage to consume a
scalar reduced over the previous substage's whole output. **That single missing
mechanism — a scalar broadcast inside an iterative phase — is what stands
between this framework and the entire variational-segmentation family**, and it
is the same mechanism a global auto-threshold wants (§1) with the loop wrapped
round it. It is the highest-leverage item in this document.

SLIC has the same shape for the same reason: bounded reach per iteration (each
cluster searches a `2S` window), a global cluster-centre update between
iterations. Felzenszwalb builds a global minimum spanning tree over the whole
adjacency graph and is category 6 outright, as is max-flow and as is the random
walker's sparse solve — all three are global optimisations whose working data
structure is not a volume at all.

---

## 7. Object measurement

This is the section with the most concrete, checkable gap, because
`measure.regionprops` is effectively the canonical list of what a general library
is expected to produce, and OpenCV's `connectedComponentsWithStats` is the
minimum-viable subset that everyone actually uses.

**The crate has two measurement ops and they do not overlap the way you would
expect.**

- **`detect`** measures *connected components of a mask*. Its `Emission::Measured`
  form emits ten `u64` columns per component: `count`, three coordinate sums,
  three per-axis minima and three per-axis maxima. Accumulators are **integers**,
  which is a decision rather than an accident — a `u64` first moment carries a
  volume up to about 65 536 on a side exactly, where an `f64` stops being exact
  at about 11 500, a factor of nearly two hundred in the volume that can be
  answered. Past the bound it **refuses** rather than wraps. Note this set is
  precisely `connectedComponentsWithStats`: area, bounding box, centroid.
- **`tabulate`** reduces a *second array* over the regions of a **label volume**:
  `count`, `nonfinite`, a fixed-point value sum, min, max as read, three
  coordinate sums, and three per-axis cross moments `Σ(v·x)`. The cross moment is
  there because it is the one per-region quantity a consumer cannot derive from
  the columns beside it — two regions with the same count, the same total value
  and the same coordinate totals can hold their value differently and have
  different first moments. The **weighted centroid** is its quotient, taken once,
  at the end, with the fixed-point scale cancelling exactly.

### Against `regionprops`

| Measurement | Merges exactly across a seam? | In `blockflow`? |
|---|---|---|
| `label` | key | `tabulate` |
| `area` / `num_pixels` (volume in 3-D) | `+` | **Both** |
| `bbox`, `bbox_area`, `extent`, `slice` | `min`/`max` | **`detect` only.** `tabulate` has no bounding box, so the op that reads intensities cannot report one |
| `centroid`, `centroid_local` | `+` then one division | **Both**, exactly: accumulate in integers, divide once, round once, half up, no floating point at any step |
| `centroid_weighted` | `+` on `Σ(v·x)` and `Σv` | **`tabulate` only** |
| `intensity_min`, `intensity_max`, `intensity_mean`, integrated density | `min`/`max`/`+` | **`tabulate`**, with non-finite voxels counted and excluded rather than poisoning the answer |
| `intensity_std` | `+` on `Σv²` | **No** — merges perfectly well; simply not carried. Cheap to add |
| Per-object histogram | `+` per bin | **No.** Needs a fixed bin count in the schema or a variable-width row |
| `moments`, `moments_central`, `inertia_tensor`, `inertia_tensor_eigvals` | `+` on `Σx_ax_b` | **No**, and `detect.rs` says why: `Σx²` is exactly as associative as `Σx`, but its range is about `L⁵/3`, so a `u64` carries it only to `L ≈ 1800` where the first moment survives to 65 536. Six columns that stop working at a tenth of the volume the other ten survive was judged not worth it before a consumer asked |
| `axis_major_length`, `axis_minor_length`, `orientation`, `eccentricity` | derived from the above | **No** — all blocked on the second moments |
| `moments_hu`, `moments_normalized` | derived | **No** |
| `perimeter`, `perimeter_crofton`, surface area | `+` **with a face-ownership rule** | **No**, and `detect.rs` is precise about the obstruction: phase 0 is halo-free, so a voxel on a block boundary cannot tell a neighbour outside the *volume* from one outside the *block*, and would count a face that is not there. Making it right needs a halo of one and a rule about which side of a seam owns the face — a design, not a field |
| `euler_number` per object | `+` on cell counts, same ownership rule | **No.** `skeleton::euler_characteristic` and `betti_numbers` exist but are whole-array free functions used to check the simple-point predicate, not ops and not per-object |
| `area_convex`, `solidity`, `image_convex` | a hull merge — associative and commutative but **not a fixed-width accumulator** | **No** |
| `feret_diameter_max`, min Feret, Feret angles | see below | **No** |
| `equivalent_diameter_area`, sphericity, circularity, roundness, compactness, convexity | derived from volume, surface area and hull | **No**, and each is blocked on one of the three rows above |
| `coords`, `image` (the object's own voxel list or crop) | — | **Partly**: `coordinates` emits one row per set voxel and `adjacency` one row per adjacent pair; grouping by label is a table operation |

### Three specific things worth acting on

1. **Second moments are recoverable by changing the origin.** The stated
   obstruction is range: `Σx²` about the volume origin is `O(L⁵)`. About the
   object's **own bounding-box minimum** it is `O(L_object⁵)`, which for the
   objects anyone measures is nothing. The bounding box is already a column in
   `detect`'s row, it is a function of the component alone, and it is therefore a
   decomposition-invariant origin — so a block can accumulate about the *volume*
   origin in `i128` and the merge can re-centre once, at the end, or blocks can
   agree on the origin in a second pass. Either way it is arithmetic rather than
   a framework question, and it unlocks orientation, principal axes, axis lengths
   and eccentricity in one move — four of the most-used `regionprops` fields.
2. **Feret diameters have a fixed-width form.** The exact maximum Feret needs a
   convex hull, which is not a fixed-width accumulator. But the support function
   sampled on a fixed set of *K* directions is: per direction, `max(x·d)` and
   `min(x·d)`, merged by `max` and `min`, exactly. That gives max and min Feret
   and their angles to the angular resolution chosen, which is how commercial
   packages compute them anyway — one uses 128 fixed angular positions and calls
   the result the Feret diameter. **2K more columns, all associative, no framework
   change.**
3. **`tabulate` should carry the bounding box and `Σv²`.** Both merge trivially,
   both are asked for constantly, and their absence is what forces a caller to
   run `detect` and `tabulate` over the same objects and join two tables on a
   label the crate cannot produce in the first place.

### One structural note on data model

ITK draws a distinction worth a sentence: a `LabelMap` stores objects as run-length
encoded regions with attributes attached, rather than as a dense label image. It
makes label filtering, relabelling and per-object attribute queries cheap and
makes per-voxel access dear. `blockflow`'s equivalent already exists in a different
form — `crate::table`, one row per object, merged by column — and the crate's
`detect` deliberately writes **no image at all** because the answer is a handful
of rows rather than a volume. That is the right analogue and it is worth naming:
the missing piece is not a `LabelMap` type, it is the round trip back from a
filtered row set to a label volume.

---

## 8. Topology and skeleton analysis

| Operation | Category | Have it? |
|---|---|---|
| Simple-point, border-point and end-point predicates over a 3×3×3 neighbourhood | 1 | **Yes** — `skeleton`, checked against Betti numbers computed by a completely different route over thousands of neighbourhoods |
| Euler characteristic of a volume, under the stated (26, 6) convention | 3 as written | **Yes, but not as an op** — a whole-array free function, computed from the cubical-complex definition rather than from a table, existing to check the simple-point predicate |
| Betti numbers (components, tunnels, cavities) | 3 as written | **Yes**, same status. `b1` follows from `χ = b0 − b1 + b2`, which is the only way to see a tunnel |
| Per-object Euler number | 5 with a face-ownership rule | **No** |
| End-point and branch-point masks from a skeleton | **1**, reach 1 | **No as an op**, and it is one `ConfigurationTable` away — the predicate is a function of the 27-bit neighbourhood and `is_end_point` already exists |
| Skeleton → graph (nodes and edges) | **5, tipping into 6** | **No.** `adjacency` gives the edge list of the voxel graph — one row per adjacent pair of set voxels, at reach 1, in canonical order, with no global numbering needed because a pair is emitted as two coordinates rather than two indices into a global list. What is missing is the **contraction** of runs into node-to-node edges, and `adjacency.rs` flags the exact difficulty: at that stage a coordinate pair stops identifying anything, because two distinct runs can join the same two positions and a run can return to where it started |
| Branch length along a run | 5 | **No** |
| Tortuosity (path length over endpoint distance) | derived | **No** |
| Radius at a skeleton point | 1 if bounded, 2 if from a distance transform | **Bounded form yes** — `walk` reports the distance at which a test first holds along a fixed offset list. The exact form wants the distance transform |

**Why chain-following is the sharpest instance of category 6.** A graph walk's
halo is a *graph distance*: "how far along the object do I have to travel to
reach the other end of this run". The reach algebra states halo widths per axis
in voxels. There is no per-axis number that means "however far this chain goes",
and a chain of *n* voxels can be contained in a bounding box of side 2. So the
framework cannot state the dependency, and following chains was made resident.
That is not a defect in the reach algebra — it is a genuine mismatch between a
lattice-shaped dependency language and a graph-shaped dependency, and the honest
fixes are the fragment-and-join shape (contract runs per block, merge the
dangling ends across seams — which *is* expressible, and is `components`'
program again with runs instead of components) or a barrier.

This is the one place where a general library and this framework genuinely pull
apart. `AnalyzeSkeleton` is one of the most-used things in Fiji; its output —
branches, junctions, end points, branch lengths, euclidean distances — is
squarely what a caller wants. The path to it here is the per-block run
contraction, and the seam merge is where the design effort goes.

---

## 9. Tracking and correspondence

**Handed to family D.** Linking objects across a series is the time axis, and
family D owns it — see `channels-and-time.md`. One sentence is owed from here,
because the hand-off has a shape: what family B produces for such a caller is a
**per-frame object table with exact, decomposition-invariant measurements**
(`detect` and `tabulate`, whose integer accumulators are precisely the property a
tracker needs, since an object's centroid must not change with the tiling), and
what a linking phase then needs from the framework is **G2** — two inputs at
different offsets along the series axis.

**Colocalisation coefficients** (Pearson, Manders, Costes) are family D's for the
same reason: they are global reductions over a *pair* of volumes, which is D's
central problem rather than this family's, even though they are measurements.

---

## 10. Machine-learning segmentation, and where the boundary is

This is the direction both academic and commercial tools have gone, so dodging it
would date the document. The honest position is that **most of what trainable
segmentation needs is family-B and family-A infrastructure, and the one part that
is genuinely out of scope is the model runtime.**

The shape is always the same three stages:

**(a) A feature stack.** A filter bank at several scales — Gaussians, gradient
magnitudes, Hessian components, structure-tensor quantities, difference-of-
Gaussians, box means, and in some products the intermediate activations of a
frozen pretrained network. Fiji's trainable segmentation, commercial products and
every homegrown pipeline agree on roughly this list; the products that expose it
tend to ship two or three **fixed, non-editable** banks of 25–33 features rather
than letting the user compose one.

*Where it belongs:* the filters are family A, and stacking *channels* into the
feature vector is family D's; the segmentation boundary — what the library owes a
classifier — is this family's. What family B owes is the **shape**:
one input, *K* outputs. `BlockOp::side_outputs` already declares arrays beside the
primary result, each with its own name, dtype and **rank**, and `side_region` maps
a block's slice into each output's own coordinate space — checked, because the
regions a phase produces must tile the declared output exactly. So a *K*-channel
feature stack is already expressible and already in the byte accounting, which is
not a small thing: the change that introduced side outputs was motivated by a run
that wrote 158.6 MB while the framework counted 95.2 MB, short by a factor of
1.67, because extra results had nowhere to be declared. *(Unverified: whether a
later phase can read a side output as its input image, or whether side outputs
are terminal. If they are terminal, a feature stack must be K images rather than
one phase's K outputs, which is more plumbing and the same cost.)*

> **Settled: they are terminal**, so the second horn is the one that holds — a
> feature stack a later phase reads is K images, more plumbing, the same cost.
> Verified in the index's §6 and now pinned by a test (`tests/tuple_map.rs`): a
> side output lands in a `String`-keyed map on the environment and never in
> `images`, and `Chain::source` takes a number. **A run being able to be handed
> images does not change this**, and it is worth being explicit because the two
> arrived together: a supplied input existed before the run, a side output is
> written during it, and the two are addressed differently.

*What must not happen:* the crate must not ship a feature list. A bank of
"the 33 standard features" is a domain default wearing a general name, and it
violates the crate's first rule — everything is a parameter — even though it
would pass every vocabulary check.

**(b) Per-voxel classification.** Reach 0, *K* input operands, one or *C* outputs.
This is the one place the op algebra is genuinely short: `voxelwise::CombineOp`
and `LogicCombine` take **two** operands, and a `Chain::Parallel` diamond joins
branches pairwise. A **reach-0 map over *K* operands** is a small, well-shaped
addition and it is the whole of what per-voxel inference needs from the op layer.
Everything else about the stage is trivially blockable: no halo, no seam, no
global fact.

> **Built — `ops::mixing`.** `TupleOp` is the shell, `TupleKernel` the kernel
> trait, and a caller-supplied reducer over *K* operands is exactly the boundary
> (c) below argues for; `LinearMap` is the first kernel and is the matrix case.
> The extra operands are images, which is why this waited on G5, and they may now
> be arrays the run was **handed** rather than ones it computed — which is what a
> feature stack built by an earlier, separate run is.
>
> **One clause of the shape was wrong, and the distinction it forced is still
> the right one to know — but the case that bit no longer bites.**
> `BlockOp::apply_side` was not handed the `SourceInputs`, so an op whose extra
> outputs are a function of its *source inputs* could not compute them there;
> `ops::mixing` shipped carrying them across from `apply_with` in a per-block
> map. **That has since been fixed** — the argument is threaded from the executor
> and the map is deleted (index §11.3) — so the clause is now true as written.
>
> The two cases are still worth separating, because they are still different
> operations and one of them still pays. **A feature stack is *one* input and
> *K* outputs** — the shape §10(a) asks for — so its side outputs were always
> functions of the op's own input and `apply_side` always had what it needed:
> nothing about this changed for it. **A classifier over *K* channels writing
> *C* class maps** is the shape that hit the defect, and it is the shape that is
> now buildable without a workaround; what it pays instead is that the inputs are
> streamed twice — `apply_with` computes output 0 and `apply_side` the rest —
> which is **1.08–1.10×**, with the flop count unchanged because the kernel takes
> a window rather than recomputing the first row.
>
> Measured, since B's whole argument here is that the stage is cheap and the
> model runtime is not: at `K = K′ = 16` in `f32` over a `[128, 128, 32]` block,
> ~40 ns per position and 4.0 flops per byte of image traffic — streaming-bound,
> and tiling the output loop is worth 2.5–2.8× of it, which the split into two
> passes keeps entire.
>
> **And one thing to read past, because it is easy to file here and does not
> belong.** `ops::ridge`'s scale map runs its multi-scale pass twice, and that is
> *not* an instance of this defect and never was: the winning scale is an
> intermediate of `apply`, made and discarded inside the evaluation, while
> `SourceInputs` carries **stored images**. No argument to `apply_side` could
> have retired that second pass. It is a priced design choice — opt-in, and free
> to a caller who does not ask — rather than something a signature was hiding.

**(c) Model execution — out of scope, as a dependency.** The classifier itself,
its training, its serialised form and its runtime are the application's. A
general image-processing library that links a tensor runtime has acquired a
dependency graph that has nothing to do with images. The right boundary is a
**caller-supplied per-voxel reducer over K operands**, exactly as `local::
Statistic::Custom` already takes a caller-supplied `Reducer` behind an `Arc` with
a stated key — the crate has already drawn this line once, for windowed
statistics, and the same line works here.

**What the crate additionally owes, and mostly already has:**

- **Training-set extraction.** Training needs feature values at a sparse set of
  labelled positions. That is `rows::GatherRowsOp` exactly — rows in, the same rows
  with one more column read at the row's own coordinate, declared as a second
  array, reach zero, decomposed by row range with no overlap because an overlap
  here is a *correctness* failure rather than a cost. Run it once per feature
  channel and the training table falls out. This is worth naming because nobody
  looking for machine-learning support would think to look at a table op.
- **A probability stack to a label volume.** Argmax over *C* channels is again a
  reach-0 map over *K* operands. Note the data model question: images are rank 3,
  so a *C*-class output is *C* images rather than one rank-4 image, and the argmax
  reads all of them.
- **Semantic classes are not instances.** Every product that ships trainable
  segmentation hands off to the conventional chain afterwards — the classifier
  replaces the *threshold* step, and hole filling, splitting and connected-
  component labelling still run before measurement. That is the strongest possible
  argument that the operations in §2, §4 and §6 do not become less important when
  a classifier arrives; they become the consumer of its output. In particular the
  missing label-volume op (§4) blocks the machine-learning path just as
  thoroughly as it blocks the classical one.

---

## 11. Present, but narrower than the name suggests

| Module | What the name suggests | What it actually is |
|---|---|---|
| `morphology` | morphology | **Binary only** — `accepts` `Bool` and `F64`, and it is a set operation. Grey erosion/dilation live in `rank` as `Rank::lowest()`/`highest()` |
| `regional` | regional extrema | **Regional maxima only.** No polarity parameter, no minima entry point |
| `detect` | detection | **One point, or one measured row, per connected region of a mask.** Ten geometric columns; no intensity, no second input array |
| `rows` | rows of a volume | **Rows of a `Table`** — scale, gather, filter. Three ops rather than one because a single `Fn(Row) -> Option<Row>` would have to make the weakest declaration of each of three properties (does it move a row, is the output a subsequence, does it read pixels) |
| `components` | connected components | **The shared machinery** — union-find, six-face geometry, seam walk, flood fill — used by `fill`, `regional` and `detect`. It exports one thing to callers: `Connectivity`. It does **not** produce a label volume |
| `skeleton` | skeletonisation | One thinning sub-iteration as a `BlockOp`, the pass as a `Chain`, and a whole-volume fixed point that is deliberately not an op. Also houses `euler_characteristic` and `betti_numbers` as whole-array free functions that exist to check the simple-point predicate |
| `configuration` | configuration | A **fully general** 3×3×3 table rewrite — a majority vote, a boundary-preserving smoothing, a hit-or-miss, or a cellular automaton, depending entirely on the caller's 2²⁷-entry table. Wider than the name suggests, and the crate ships no table but the identity |
| `walk` | a walk | A **bounded** walk along a fixed offset list, reporting the distance attached to the offset it stopped at. Not a flood, not a search |
| `label` | labelling | **Stamps scattered points into a volume as names**, lowest label wins on collision. Not connected-component labelling |
| `adjacency` | adjacency | Every adjacent pair of set voxels, as rows carrying two coordinates. The edge list of the voxel graph, not a region adjacency graph |
| `fill` | filling | Hole filling specifically, on the **background's** connectivity, which is fixed and not a parameter |

---

## 12. Not expressible today, and what each would need

| Operation | Why not | What the framework would need |
|---|---|---|
| **Global auto-threshold** (Otsu and the rest); any global scalar reduction — also **G1** | Expressible but at N+1 passes: phase 1's whole-lattice fragment reach is also its halo, so a reduction whose answer is one scalar reads the whole volume per block | A **barrier phase** — declared to start when the previous one finished, so the dependency is stated without a halo. `fill.rs` already names this as the way out and as the open architectural question |
| **Level sets, active contours, Chan–Vese; SLIC** | The per-iteration update is a local stencil, which `iterate` handles at reach one substage — but the update consumes a whole-volume reduction (region means, cluster centres) recomputed every iteration | A **scalar broadcast inside an iterative phase**: a substage able to read a value reduced over the previous substage's whole output. Same mechanism as the row above, with a loop round it. **Highest-leverage item in this document** |
| **Skeleton-to-graph, chain following, branch length, tortuosity, run-based pruning** | A graph walk's halo is a *graph distance*, and the reach algebra states per-axis voxel widths. A chain of *n* voxels fits in a bounding box of side 2, so no per-axis number bounds it | Either a fragment-and-join formulation (contract runs per block, merge dangling ends at seams — `components`' program with runs instead of components), or a barrier. Not a change to `Reach` |
| **Object linking across a series** — **G2**, and family D's | A linking phase reads frame *t* and frame *t+1* | A phase whose **inputs sit at different offsets**. Family A wants the same thing for pyramids; see `channels-and-time.md` |
| **Per-object second moments, orientation, principal axes, eccentricity** | Not a framework gap — a range gap. `Σx²` about the volume origin is `O(L⁵)` and overflows `u64` at `L ≈ 1800` | `i128` accumulation, or re-centring on the object's own bounding-box minimum (already a column, already decomposition-invariant). Arithmetic, not architecture |
| **Surface area, perimeter, per-object Euler number** | Phase 0 is halo-free, so a boundary voxel cannot distinguish a neighbour outside the volume from one outside the block, and counts a face that is not there | A halo of one on the labelling phase plus a stated **face-ownership rule** at the seam. A design, and a small one |
| **Convex hull, solidity, exact Feret diameters** | The merge is a hull merge — associative and commutative, but the fragment is a variable-size point set rather than a fixed-width accumulator | Either variable-size fragments with a hull merge, or accept the *K*-direction support-function approximation, which needs nothing |
| **A collapsed or a broadcast axis, declared** — **G1** *(renamed; the row read "a rank-reducing phase")* | Not the rank: `[X, Y, 1]` is a legal `Array3` and a degenerate axis is how this crate models lower rank, on purpose. `Reach` is per-axis halo widths on a same-rank output and cannot say "consumed whole" or "held at 1" | **Corrected.** A collapsing phase — a projection, a reduction to `[1, 1, 1]` a later phase reads, the map that broadcasts a threshold level back — **runs today**, verified byte-identical to a whole-volume reference across 25 cuts of the two free axes (`tests/collapsing_phase.rs`), through the `†` cross-grid escape: `Reach::none()` in `Space::source_index()` plus a fetch stated per block. So this row is not about expressibility. The terminal side-output route is still there and still terminal, and is cheaper where the result is terminal. **Update: the collapse can now be declared.** `AxisReach::All` in `Space::source_voxels()` plus `with_sources` states the dependency and is checked against the fetch, so the collapsing half of this row is answered and the escape is strictly weaker for it. The **broadcast** — the map that applies a threshold level back over the volume — still has no declaration, because a pinned axis cannot be expressed; that is what is left of G1, and the per-axis extent rule (`Whole` and `Fixed(1)`) is still what would state both with one mechanism |
| **A stated fetch that is checked against the declaration** — **G14**, minted in the index by a later correction pass, **now partially closed** | `Decomposition::check` verified only that a block's `source` lay **inside** the image it reads, never that it *covered* what the reach claimed. It now also refuses a fetch that fails to cover a **declared** whole axis, by name. What is *not* checked is **silence**: a phase that declares nothing and reads one plane of an axis it means to consume is self-consistent, passes every guard, and is wrong at every position | Closed for the declared case. The residue — "the check holds an op to what it *said*; it cannot make an op speak" — wants a **total** per-axis rule in which no axis can be silent, which is G1's extent rule with the totality as the load-bearing part. **The one entry in this table that is a correctness gap**; everything else here is capability or cost |
| ~~**A reach-0 map over K operands**~~ → **built** | `CombineOp`/`LogicCombine` take two; `Chain::Parallel` joins pairwise — and `ops::mixing` is now the K-ary shell that neither was | **`ops::mixing::{TupleOp, TupleKernel, LinearMap}`**, exactly the "K-ary voxelwise shell" this row asked for: one `BlockOp` with K−1 `source_inputs` at reach 0 and K′−1 `side_outputs`, no new axis, and not a `Combine` because that trait has no side outputs. The extra operands are images, so this needed **G5**, which is also closed. **One clause of the shape was wrong and has since been fixed**: `BlockOp::apply_side` was not handed the `SourceInputs`, and now is — see §10(b), where the case that bit and the case that never did are separated, and where what the fix costs (1.08–1.10×, the inputs streamed twice) is recorded |
| **An op that requires one axis be left whole** | `BlockConstraint::Extent` mandates all three block extents or none | `FullExtent(axis)`, or an `Extent` of `Option<usize>`. A cost gap, not a correctness one — a lattice that cuts a whole-axis-reach axis with the full halo is redundant, not wrong |
| **Graph cuts, random walker, Felzenszwalb superpixels** | Global optimisations whose working data structure is not a volume — a max-flow, a sparse linear solve, a minimum spanning tree | Nothing in the reach algebra helps. These are resident algorithms over a derived graph, and the right answer is probably that they are out of scope for the block layer and belong on top of a region-adjacency graph the crate could produce |

---

## 13. Assigned to other families

- **Gradient magnitude, Hessian and structure-tensor responses, difference-of-Gaussians, ridge/tubeness enhancement.** Family A. They matter here as watershed cost volumes and as feature-stack channels, but they are filters.
- **Background estimation and illumination/stripe correction.** Family A — `background` and `normalise`.
- **Deconvolution.** Family A.
- **Resampling, interpolation and coordinate transforms** (`resample`, `coordinates`, `lattice`). Family C, except where a lattice appears as the sample grid of a local statistic, which is family B's use of family C's machinery.
- **Registration and drift correction**, including the FFT-based squared-difference landscape over integer lags. Family C for the spatial transform; family D for drift along the time axis.
- **Object linking across a series, optical flow, temporal filtering, kymographs.** Family D (§9).
- **Colocalisation coefficients, spectral/stain unmixing, crosstalk correction, channel arithmetic.** Family D — global reductions over a pair of volumes, or operations along the channel axis.
- **Rendering objects back to an image** — filled regions, contours, ID-encoded colours. This straddles: the object table is family B's, the rasterisation is family C's. `voxelize` and `label` are the crate's existing half of it.
- **Mesh and surface extraction from a label volume** (marching cubes, contouring). Output is not a volume; family C if anyone, and arguably out of scope.

---

## 14. If only three things were built

1. **Connected components to a globally consistent label volume** — the `label,
   merge, relabel` third phase. It unblocks `tabulate`, object filtering, seeded
   watershed from detected markers, and the hand-off from any classifier. Nothing
   else in this document unblocks as much.
2. **The exact Euclidean distance transform**, moved in from where it already
   works. Category 2, already measured bit-identical to a reference over 77
   million voxels, and `watershed` already says it wants one. Brings distance-
   based object splitting, signed distance and medial axis with it.
3. **A barrier phase**, which turns every global scalar reduction from N+1 passes
   into two — and, with a scalar broadcast inside `iterate`, opens the whole
   variational-segmentation family that is currently category 6.

The rest of the gaps in §12 are columns, tables and small shells.
