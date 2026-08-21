# Geometry, registration, and multi-image composition

*Ops survey, family C: moving pixels in space, and combining images that are
not aligned.*

This is a survey, not a plan and not a specification. Nothing in it is
implemented; nothing in it is promised. Its purpose is to say, for each
operation a general image-processing library is expected to support, **whether
`blockflow`'s decomposition model can express it and at what cost** — and where
it cannot, what the framework would have to grow. That last column is the part
still worth reading in a year.

The reference points behind the "what a library is expected to have" column are
the geometric transforms and stitching pipeline of a large computer-vision
library, a general scientific imaging library's `transform` / `registration` /
`marching_cubes` modules, the registration framework of a medical imaging
toolkit (transforms, metrics and optimisers as *separate* things), the imaging
transform and reslice classes of a visualisation toolkit, the plugins people
actually use in a general biological image analysis application, and the
processing catalogue of a commercial acquisition-and-analysis package. Where
those references disagree, the disagreement is the interesting part and is
noted.

---

## 0. Scope, and what belongs to someone else

Three sibling surveys cover the rest: **family A** takes point operations,
filtering and intensity transforms (`docs/ops-survey/filtering-and-transforms.md`);
**family B** takes segmentation, morphology and measurement
(`docs/ops-survey/segmentation-and-measurement.md`); **family D** takes the
*non-spatial* axes — channel and time
(`docs/ops-survey/channels-and-time.md`).

Family D is the important recent split, because A, B and C are all spatial:
between them they cover the three axes of a volume and nothing else. Two things
that read like family C are D's:

* **Channel merge and split, blending modes, composite construction and
  colour-space conversion.** These are arithmetic along the channel axis at
  reach 0; nothing in them moves a pixel in space.
* **Alignment across a time series (drift correction), and chromatic
  aberration.** The split worth keeping clean: **family C owns the transform
  machinery** — estimating a displacement, evaluating a warp, resampling
  through it — and **family D owns the observation** that the transform is
  estimated frame to frame and applied per channel. §9 says the same thing in
  one row each and points at D.

Six things straddle a boundary and are named here once so nobody has to guess:

| straddles | owner | why |
|---|---|---|
| the anti-alias prefilter before a downsample | **A** | it is a low-pass filter with a reach of its own. `ops::resample`'s header argues at length that it must stay a separate op precisely so both terms are priced; family C only insists that the composition exists |
| `ops::fft` as a *transform* | **A** | as a frequency-domain filter it is A's. What family C owns is the **correlation and squared-difference landscape** built on it, which is a registration primitive |
| the threshold that produces the mask an isosurface is extracted from | **B** | marching cubes takes a level and a scalar field; where the level came from is a segmentation question |
| connected components, object counts and measurements over a stitched mosaic | **B** | the mosaic is family C's; everything measured on it afterwards is not |
| estimating drift between consecutive frames | **C's primitive, D's use** | the landscape is `ops::fft` and is family C's. That the frames are consecutive in *time*, and that the correction is applied to every channel, is D's |
| spectral unmixing | **D** | it was in this survey's first draft as a many-input reach-0 op. It is arithmetic along the channel axis and moves nothing in space |

---

## 1. The library's outer boundary

This family is where the boundary of a block-processing library actually sits,
so it is worth drawing plainly rather than surveying past it.

### In

* Anything that reads voxels and writes voxels, including where the output grid
  is not the input grid.
* Anything that reads a coordinate list and writes a coordinate list.
* **Evaluating** a transform — at an image (resample through it) and at a point
  set. Two different operations; see §4.
* **Computing** the quantity an optimiser searches: a similarity metric, a
  correlation surface, a squared-difference landscape over a lag window.
* Extracting a triangle list from a scalar field.

### Out, and why

| out of scope | the line |
|---|---|
| **Rendering and interactive viewing** — orthogonal-view widgets, 3-D volume rendering, blend-and-look overlays, zoom interpolation, display curves, colour maps applied for the eye | A projection that becomes a **stored image** is an op. A projection that exists to be looked at is a viewer. The commercial reference draws this line explicitly and correctly: display state becomes pixel data only at an explicit "make an image of this" step, at export or through a "new image from current view" command. `blockflow` has no display state to bake, and should not acquire one |
| **Optimisers** | The library owes an *evaluable* transform and a *computable* metric. The descent — pattern search, gradient, Levenberg–Marquardt, and the multi-resolution schedule wrapped around it — is the caller's. This is the medical-imaging toolkit's separation and it is the single most transferable idea in that whole framework: transform, metric, interpolator and optimiser are four independent things, and a library that fuses them ships one registration instead of a registration framework |
| **Global solvers** | A least-squares or bundle-adjustment solve over pairwise tile offsets is dense linear algebra on `O(tiles)` unknowns and reads no voxels. It is in scope as the *consumer* of a `fragment` fold — the README already names "per-block displacement estimates a least-squares solve consumes" as a sidecar use case — and out of scope as an op |
| **General mesh processing** — decimation, remeshing, Laplacian smoothing, boolean operations, parameterisation, mesh file formats | The line is the triangle list. Isosurface extraction *reads voxels*; everything downstream of the triangle list does not, and a voxel library that grows a half-hearted mesh library serves neither. See §8 |
| **Acquisition geometry** — stage coordinates, carrier calibration, travel order, tile-region rasterisation from a drawn contour, focus surfaces fitted over a tile grid, autofocus search | The commercial reference has a great deal of this and none of it belongs here. It is application geometry: it decides *where images came from*, and by the time a volume reaches this crate that question is answered. What does cross the boundary is the **result**: a nominal per-tile origin, which §6 treats as an input |
| **4-D and time as an axis** | The README already settled this: volumes are 3-D, channels are separate arrays combined by a two-source read, and time is not in scope. §9 says what that costs a drift-correction operation |

---

## 2. The categories

Families A and B use a shared scale, and this document keeps it. **1, 2 and 3
mean the same thing in all three documents.** The differences are stated here so
the four surveys compose:

* A's and B's **4** is *iterative to a fixed point*. Family C has **no instance
  of it** — a registration search is iterative, but it is an optimisation run
  and §1 puts it outside the library.
* B's **5** is *fragment-and-join*. Family C uses it in exactly two places, both
  below: the pairwise-estimate fold of §6 and the vertex weld of §8.
* **Not expressible today** is A's 5 and B's 6. This document calls it **5**,
  matching A, and every instance points at a numbered gap.
* Family C adds one category the spatial-filtering families had no need for:
  **P**, below. It is not a rung on the same ladder — it is a different kind of
  work — so it gets a letter rather than a number.

| # | category | what it means | the crate's own example |
|---|---|---|---|
| **1** | **Bounded reach** | a halo the op declares, `AxisReach::Bounded { lo, hi }`, possibly `PerBlock`. Fuses with neighbours, cuts anywhere, prices cleanly | `ops::resample` under linear interpolation reaches `ceil((up - down) / (2·down))` and nothing more |
| **2** | **Whole-axis reach** | `AxisReach::All` on one axis, free on the others. A **planning barrier by type** rather than by comparison. Blocks must span that axis | **none in `src/ops/`.** Family A verified by grep that nothing in `ops/` declares `All` on a single axis. A separable per-axis resampling or warping sweep is exactly this shape — but see the correction below, because *which space* the reach is stated in decides whether it can be declared at all |
| **3** | **Whole-volume, resident** | no halo bounds it and the shape does not fit `BlockOp` either. Free functions and a plan the caller holds | `ops::fft` — and its header states the three independent reasons it cannot be an op |
| **P** | **Resident but cheap** | point arithmetic over a coordinate list. Decomposes by *row range* with reach exactly zero, and a halo there would be a **defect**, not a cost — a row duplicated into two ranges is emitted twice | `ops::rows`, whose header makes this argument. **A point transform is this and nothing more**; building a lattice for one is building a mechanism that is not needed |
| **5** | **Not expressible today** | the framework has no declaration for it. Every instance names its gap | see below |

> **Corrected — measured**, by the experiment §12 called for. "The variant
> exists with no user" was this document's reading too, and it conflates two
> declarations one word apart. `AxisReach::All` in the **phase** frame — the
> default `Space::phase_voxels()`, which is what a same-rank separable sweep
> wants — plans, works, and has a user outside this crate. `AxisReach::All` in
> `Frame::Source` was, when the experiment was run, **refused
> unconditionally**: no halo satisfied it, at any extent, on any grid. So the
> source-frame pairing had no user because it could not have had one — a
> different statement from "nobody has written one", and the one this family
> needs, since a shape-changing sweep is where the two frames come apart.
>
> **Overtaken: the geometry change landed.** The exception is now granted per
> axis when the reach on that axis is `AxisReach::All` **and** the block's read
> spans the whole of it (`geometry.rs:319-322`), and a declared whole axis is
> checked against `BlockGeometry::source` (`decomposition.rs:763-791`). So
> source-frame `All` is plannable, is **the recommended way** to state a
> whole-axis dependency, and is the only one of the three declarations that is
> checked. Phase-frame `All` still plans and is still vacuous on a collapsed
> axis — **two of the three are wrong in different ways, so this family must
> always name the frame.** The measurement is `tests/collapsing_phase.rs` (15
> tests); the consolidated account is the index's §8 and §8.5.

### The gaps, by family A's numbering

Family A numbered the crate's four structural gaps and this document points at
them rather than inventing a second set:

* **G1 — the geometry cannot declare a pinned (broadcast) axis.** *(Renamed
  twice in the index's §8.1, identifier unchanged both times. Family A's
  original name was "no rank-reducing phase"; that name describes a problem the
  crate does not have, since `[1, Y, X]` is a legal `Array3` — which is the
  point this section and §7 were already making. The first rename read "a
  collapsed **or** a broadcast axis"; the collapsed half has since landed.)*
  **Projections are the family C instance** (§7): expressible at `2†` — verified
  — and, since the geometry change landed, **declarable and enforced** as well.
  What is left of G1 for this family is the projection's inverse, the
  **broadcast**, whose pinned axis still cannot be stated.
* **G2 — no phase whose inputs sit at different offsets.** A `Decomposition`
  reads one image and maps a block to one region of it. **Mosaicking is the
  family C instance** (§6), and it is the gap this family cares about most.
* **G3 — no complex element variant.** A spectrum cannot be a plan image.
  Family C reaches it only through phase correlation (§5).
* **G4 — the cost model cannot price `log n`.** A resident transform prices as
  linear. Family C's registration landscape is exactly such a transform, so
  every plan containing one is mispriced. *(And since the partition search's
  objective changed, the misprice chooses a grid rather than only reporting a
  wrong number — §10.)*

A fifth, which this document did not name and needed: **G5 — there is no second
input volume**, family D's, and the index's §6 adjudicated that family C needs it
too, since N tiles are N acquired arrays. **It is closed** — a run can be handed
`k` arrays — and §6 and §10 record what that does and does not unblock here. The
short answer is that it unblocks getting the tiles in and nothing else; G2 and C2
are untouched.

Three gaps are family C's own and are numbered **C1–C3** in §10, to avoid
colliding with A's scale.

### The cost modifier

One thing that is not a category, because three ops already pay it:

> **† cross-grid, hand-built phase.** An op whose output grid is not its input
> grid states its fetch per block in `BlockGeometry::source`, declares
> `Reach::none()` in `Space::source_index` — a *marked* zero, with the real
> dependency carried by the fetch region — and ships its own plan builder beside
> itself. `ops::resample` (`resample_phase`), `ops::lattice`
> (`lattice_statistic_phase`, `lattice_interpolate_phase`) and
> `ops::adjacency` do exactly this.
>
> The price is real and should be stated every time it is invoked: **the
> automatic planners cannot produce such a phase.** `Trivial`, `Enumerating`,
> `Greedy` and `Materialising` all plan against one volume per group. A
> shape-changing op is planned by a function written next to it, which is why
> a category marked `1†` is genuinely more expensive than a bare `1`, and why
> several operations below that *look* like category 5 are really `2†` with a
> hand-written builder.

---

## 3. Geometric transforms

| operation | category | have it? | notes |
|---|---|---|---|
| **crop** to an axis-aligned box | 1† | **partially** — the *plan shape* exists, no op does | `src/tests.rs`'s `crop_plan` builds a two-phase plan whose second phase is cut from a smaller volume and reads a window of the image below. It is a test, not an op. Its own doc comment records the limit: a cross-grid fetch may **move** an extent, never resize it, and a non-zero reach across a crop seam needs the reach stated in the source space |
| **pad** / grow the canvas | 5 (C2) | no | The inverse of a crop and *not* symmetric with it. A `Region` has `start: Vec<usize>` and cannot begin below zero, so "pad 10 voxels below the origin" has no representation; padding is a re-origining of the whole volume. See §10 |
| **flip / mirror** an axis | 1 | no | Reach 0, one output voxel per input voxel — but the *output index* is `n - 1 - i`, which is not an offset. `InputMap` has `Stencil` (grow the output region) and `Affine` (scale it by a rational). Neither reverses an axis. Cheap to add as a third `InputMap` arm; nothing about it is hard |
| **transpose / permute axes** | 1 | **declared, not acted on** | `reach::Space` already carries `axes: [usize; 3]`, and its doc comment is explicit that this is deliberate rather than unfinished: a permuted branch needs the lattice, the read extent, the valid region and the anchor permuted *together*, and a permuted reach is the cheapest of the five. The tag exists and is hashed into the fingerprint; `BlockGrid` and `Anchor` are the work |
| **rotate by a multiple of 90°** | 1 | no | Composition of a permute and a flip. Falls out of the two rows above, and should be spelled as that composition rather than as its own op |
| **rotate by an arbitrary angle** | 1† | no | Bounded reach: the halo is the interpolation kernel's radius plus the displacement the rotation induces across a block, which is bounded by the block diagonal times `sin θ`. That last term is the problem in practice — it grows with block size, so a large block pays a large halo, and the reach is honest but expensive. Wants either an output-grid-cut phase (`†`) or an acceptance that a rotating phase blocks badly |
| **rigid / similarity / affine warp** | 1† | no | Same shape as rotation with a general linear part. The halo per axis is derivable from the matrix and the block extent, exactly and in closed form, which makes this the *best-behaved* member of the family: the reach is a function of the parameters and nothing else, which is the rule `ops/` holds itself to |
| **projective warp** | 1† | no | The reach is still bounded over a bounded block, but it is no longer uniform — it varies with position, which is what `AxisReach::PerBlock` is for. This is the case that motivates per-block asymmetric halos on an axis that was not previously cut |
| **arbitrary displacement-field warp** | **5 (C1)** | no | **The reach is data-dependent**: the halo is whatever the displacement field says. `Reach` is required to be a function of the block index and nothing else — the doc comment says so, and the reason is that a `Decomposition` is parity-visible and must be reproducible from what it records. A field-driven halo would make a plan unreproducible. See §10 for the two ways out |
| **interpolation kernels** | — | **nearest and linear only** | `ops::resample::Interpolation` has exactly two variants, and the module header states why higher order is absent, distinguishing two different absences: a **cubic convolution** is four taps per axis, would fit the existing reach derivation, and is missing only because *which* cubic is a parameter nobody has asked for; a **spline of order ≥ 3** is not a wider window at all but a global IIR prefilter whose dependency is `AxisReach::All` — a separate op with a separate reach, i.e. category 2, not a variant of category 1. Lanczos is a wider convolution and is category 1 like the cubic. Recording which kind of absence a missing feature is, is worth more than the feature |
| **boundary conventions** | — | **clamp, and only clamp** | Every op in `ops/` clamps at a real volume boundary, and `resample`'s header states the rule the whole module shares: at a real boundary the clamp is the whole story, and at a block seam it is wrong on purpose, because the halo guard exists to turn a silent wrong answer into a loud one. There is no reflect, no wrap, no constant-fill, no "transparent" mode. The large computer-vision library offers five; the commercial reference offers essentially none and the survey of it found no wrap/reflect/clamp selector anywhere. **A boundary mode is a per-op parameter, not a framework feature** — the reach does not change — so this is a gap in `ops/`, not in the model |

### What the references agree on that is missing entirely

Every reference exposes flip, 90° rotation, arbitrary rotation with a stated
sign convention and a stated rotation centre, and an "adapt the output canvas
versus keep the input size" switch. That last switch is the same choice
`ops::resample::OutputExtent` already makes for scaling and would want making
once, for all of them.

---

## 4. Resampling, and the two coordinate spaces

### What the crate has

`ops::resample` resizes by an **exact rational per axis**, `up / down`, reduced,
with the centred half-voxel sample map evaluated in integer arithmetic. It is
decomposable, its reach is derived and shown tight to one voxel, and its
alignment cost is measured (1.00× for every downsampling and every integer
growth; 1.52× to 2.01× for a rational growth at a small block edge, shrinking
with the edge).

**Its most transferable finding is not about interpolation.** It gained an
explicit `OutputExtent` because two references disagreed about what a resampling
factor *means*:

| reference | output samples | scale used |
|---|---|---|
| one | `ceil(n · r_src / r_dst)` | `n / out` |
| the other | `floor(n · up / down)` | `down / up` |

Both halves of a convention are one choice. **Mixing them matches neither** —
you get the first library's sample count evaluated on the second library's
grid, which is a well-formed volume that agrees with nothing. The interpolation
was never the disagreement. `OutputExtent::Factor` is the second convention
stated honestly; `OutputExtent::Stated { input, output }` carries *both* extents
so the factor is derived from them rather than checked against them, which is
what keeps the two from drifting.

The cost of a stated extent is recorded and is a genuine decomposition fact: an
extent-derived ratio is essentially always in lowest terms (`348/2137` and
`249/3823` both have `gcd = 1`), so the sampling lattice's period is the whole
axis and the only legal cut is no cut. Under a stated extent the op therefore
takes its shape from the plan (`takes_extent_from_placement`) and states its
fetch per block — the `†` escape, declared and paid for.

### The table

| operation | category | have it? | notes |
|---|---|---|---|
| up/downsample by a rational factor | 1† | **yes** | nearest and linear; `Ratio` per axis; anisotropic by construction |
| resample to a stated output extent | 1† | **yes** | with the cut restriction above, declared |
| **box / mean downsample (binning)** | 1 | **no, and it is a composition** | Every reference has this as its own operation, because it is the *correct* downsample for continuous-valued data. In this crate it is `Chain::sequence([box mean over an n-wide element, nearest resample by 1/n])`, and `resample`'s header argues that keeping it a composition is right: the plan then prices both terms, haloes both terms and can cut between them. Worth stating in the docs as **the** way to bin, because a caller who reaches for `resample` alone gets aliasing and the module says so |
| **aliasing on downsample** | — | **stated, not fixed** | Deliberate. Point-sampling aliases; a test demonstrates a pattern at the sampling frequency surviving into the output as a constant. The three reasons are in the header and the strongest is that a low-pass before a *nearest* resample would corrupt exactly the case nearest exists for — a mean of labels is not a label |
| **multi-resolution pyramid** | 1† ×levels | **no** | A pyramid is `n` chained downsample phases and every level is an image. Nothing in the model resists it; what is missing is a builder that produces the chain and an environment convention for addressing "level `k` of the pyramid" as distinct from "image `k` of the plan". Both references that have it treat it as storage, not as an operation |
| **anisotropic voxel spacing** | — | **absent entirely** | There is no voxel size, no unit, no physical-versus-index distinction anywhere in the crate. A grep for spacing, voxel size, physical or micron finds only `ops::local`'s *lattice* spacing, which is a count of voxels. Everything is index space |
| **physical versus index coordinates** | — | **absent, and this is a real decision to make** | See below |

### On physical coordinates

This is the largest single gap in family C and it is a design decision rather
than a missing op.

Every reference that does registration well carries a physical frame: an origin,
a per-axis spacing, and (in the medical toolkit and the visualisation toolkit) a
direction matrix. Registration between two volumes acquired at different
spacings is *defined* in that frame and is meaningless without it. The
commercial reference carries spacing as metadata in eight length units, has a
batch operation that rewrites it without touching a pixel, and — tellingly — its
own specification cannot say whether resample, downsample or rotate update it.

`blockflow`'s position today is the opposite and is internally consistent: a
volume is an index lattice, a `Region` is a box of `usize`, and every reach is
counted in voxels or in lattice steps. That consistency is worth something. But
it means **the crate cannot express "these two volumes are the same object at
different resolutions"**, which is the premise of half of §5.

There are two honest positions and the crate should pick one deliberately:

1. **Stay in index space, and say so.** Physical spacing is the application's;
   it converts to a rational `Ratio` at the boundary and hands the crate index
   arithmetic. This is cheap, matches "everything is a parameter", and is
   probably right — `Ratio` is already exact, which is the property a spacing
   conversion needs most.
2. **Carry a spacing per image**, as metadata that no op reads and no reach
   counts, so that a transform evaluator can be handed both frames. This is
   what an elastix-format parameter file needs to be evaluated at all.

Position 1 with a documented conversion recipe is the smaller change and it
should be written down, because right now a caller has to infer it.

---

## 5. Applying a transform: to an image, and to a point set

**These are two different operations with different conventions, and both are
needed.** They differ in three ways that a library must not conflate:

* **Direction.** Resampling an image through a transform `T` iterates over
  *output* voxels and evaluates `T⁻¹` to find where to read. Transforming a
  point set iterates over *input* points and evaluates `T` forward. A library
  that ships only one of the two directions has shipped half a transform, and
  the caller who needs the other has to invert — which for a B-spline is not
  available in closed form at all.
* **Representation.** An image transform lands on a lattice and interpolates.
  A point transform lands wherever it lands and must keep sub-voxel precision,
  because rounding at every step of a chain accumulates.
* **What "the same transform" means.** Applying `T` to an image and to the
  points measured in it must move them consistently. That is a testable
  property and it is the one that catches a sign or an origin error.

### What the crate has

| operation | category | have it? | notes |
|---|---|---|---|
| resample an image through an axis-aligned rational scale | 1† | **yes** | `ops::resample` |
| resample an image through a general transform | 1† / 5 | **no** | §3 |
| scale a point set | **P** | **yes, narrowly** | `ops::rows::ScaleRowsOp` — one `f64` factor per axis, applied to each coordinate independently, rounded **ties-to-even** before the cast. The rounding rule is stated and matched to a reference implementation, which is exactly the kind of thing that is invisible until it disagrees |
| transform a point set by a general transform | **P** | **no** | Category P, and worth saying plainly: this is point arithmetic over a coordinate list. It wants no lattice, no halo and no block geometry. The right shape is a free function over a coordinate list plus a thin `FragmentOp` shell, exactly what `ops::rows` already is |
| **sub-voxel point coordinates** | — | **no** | `points::Point` holds `at: [usize; 3]`, and `table` holds `usize` coordinates. `ops::rows::scaled_index` refuses a negative factor because "a table holds `usize` coordinates". So a transformed point cannot be negative, cannot be fractional, and is rounded at every step. For a *rendering* consumer this is fine; for a registration chain it is not |

### Evaluating versus producing — the spine

The distinction that should organise everything below comes from a working
implementation rather than from a reference: an **elastix-format transform
evaluator** — parameter-file reading plus affine, Euler, translation and
B-spline point evaluation — was written in a sibling application crate rather
than in a library. Its author separated two things:

| | what it is | where it belongs |
|---|---|---|
| **evaluating** a transform | given parameters and a point (or an output lattice), produce a coordinate (or a resampled volume). Deterministic, cheap, testable against a reference to the last decimal | **a library.** Category P for points, 1† for images |
| **producing** a transform | given two images and a metric, search a parameter space until the metric stops improving. Stochastic, expensive, schedule-dependent, and reproducible only if the whole schedule is pinned | **an optimisation run.** Not a library concern, and not a `Decomposition` — a plan that peeked at data would seam differently on two datasets |

That split is not a stylistic preference. A `Decomposition` is binding,
parity-visible and decided from shape and dtype only; a registration *search* is
by definition data-dependent. **A registration cannot be a plan. A transform
evaluation can be, and should be.** Everything in §5.1 that this crate could
sensibly grow is on the evaluation side of that line.

**What that evaluator actually covers**, checked rather than assumed, because it
is the closest thing to a specification family C has for the evaluation side:

* **Transforms:** translation, Euler (with the `Rz·Rx·Ry` versus `Rz·Ry·Rx`
  ordering switch handled and defaulted), affine (row-major matrix), and
  B-spline — all collapsing to `y = A(x − c) + t + c` for the linear ones.
  Every other keyword is **refused by name** rather than silently treated as
  identity, which is the same discipline `ops/` holds itself to.
* **B-spline:** cubic basis only, order refused by value if not 3; grid origin,
  spacing, index and a **direction matrix** all read; four control points per
  axis; and — the detail worth stealing — **outside the valid region the
  displacement is exactly zero, explicitly not clamped**. That is a boundary
  convention stated rather than inherited, and it is the opposite of §3's
  clamp-everywhere rule, so a library growing both owes a decision about which
  applies where.
* **Composition:** a full initial-transform chain, with two documented
  combination modes (`compose` and `add`), a cycle guard, and a
  filesystem-free variant beside the on-disk one.
* **Image metadata:** origin, spacing and direction, with `index_to_physical`
  and `physical_to_continuous_index` both provided, an explicit
  index-rounding-convention switch, and a named identity geometry for callers
  who only have voxel indices. **This is the physical-coordinate frame §4 says
  `blockflow` does not have**, built outside the library because the library
  had nowhere to put it.
* **Direction:** it evaluates at **points only**. There is no image warp
  through it anywhere.
* **Production:** none. No optimiser, no metric, no subprocess. It reads
  parameter files some other registration produced.

Two conclusions follow. First, **the evaluation side is a real, finite,
already-written body of work** — it is not speculative, and a library version of
it would be mostly a relocation. Second, the thing that kept it outside the
library is not the transform arithmetic but the **physical frame** it needs to
evaluate in, which is §4's open decision.

### 5.1 Registration, as a table

| operation | category | have it? | notes |
|---|---|---|---|
| **cross-correlation over a lag window** | **3** | **yes** | `ops::fft::Correlation2`, through the correlation theorem |
| **sum-of-squared-difference landscape** | **3** | **yes** | `ops::fft::SquaredDifference` produces a mean squared difference at every integer lag, normalised by the **exact overlap count**, with the two energy terms computed as exact rectangle sums rather than as two more transforms — cheaper (two forward transforms and one inverse instead of four and one) and more accurate (three of four terms exact to one rounding). An empty overlap reports `INFINITY` rather than dividing by an epsilon |
| **the padding rule** | — | **yes, and sharper than the usual one** | `N >= max(A, B, A − lo, B + hi)` rather than "pad to the sum of the extents". Two 1304-long extents with lags in `[−30, 30]` need 1334, not 2607. The two constraints are **one-sided**, so where the window sits matters as much as how wide it is: two width-61 windows over 96-long extents need 126 and 156 respectively. This exploits an off-centre window, which is the normal case for a mosaic whose nominal offsets are known. Rounding up to a 5-smooth length is worth 5.9× and is the default |
| **phase correlation** | **3**, or 5 (G3) as a *phase* | **no, but it is one step away** | The transform, the plans, the padding rule and the lag window all exist. Phase correlation is the same pipeline with the cross-power spectrum normalised by its magnitude before the inverse. What is missing is a normalisation step on a `Spectrum`, and the argument for adding it is that it is far more robust to illumination differences across a seam than a raw correlation |
| **normalised cross-correlation** | **3** | **no** | The three-term expansion `SquaredDifference` already computes is exactly what NCC needs (`Ea`, `Eb`, `C` over the overlap). NCC is a different combination of the same four quantities. This is a small addition to an existing file, not a new capability |
| **mutual information** | **3** | **no, and it does not fit the same machinery** | MI is a joint-histogram functional, not a convolution, so no transform accelerates it and it must be evaluated per candidate transform rather than over a whole lag window at once. It is the metric that works across *modalities*, which is why the medical toolkit centres on it. Shape: a resident two-input reduction producing a scalar — closest to `ops::tabulate` in shape, and definitely not a `BlockOp` |
| **sub-voxel peak fit** | **P** | **no** | `Landscape::argmin` returns an integer lag. Every reference refines it — a parabola or a Gaussian through the peak and its neighbours. Cheap, entirely local to `ops::fft`, and the difference between voxel-accurate and useful. The commercial reference's autofocus does the same thing to its sharpness curve, which is the identical operation in one dimension |
| **3-D landscape** | **3** | **no — the existing one is 2-D only** | `RealTransform2`, `Correlation2`, `ShiftWindow` and `SquaredDifference` are all `[usize; 2]` and `[isize; 2]`. The consumer's parallelism is across *planes* of a stack, one landscape per plane, which is the right shape for a plane-wise mosaic and the wrong shape for a volumetric registration. Extending to three axes is mechanical but is not free |
| **block matching** | 1 + 3 | **no** | A local correlation per patch on a grid, producing a sparse displacement field. Structurally this is `ops::lattice`'s pattern — evaluate something on a coarse lattice, interpolate back — with `ops::fft` as the per-lattice-point kernel and a **vector-valued** output. Which is the problem: an image holds one scalar per voxel, so a three-component field is three images |
| **multi-resolution pyramid as a registration strategy** | — | **no** | And it is a *schedule*, not an operation: register at the coarsest level, use the result to initialise the next. Every reference exposes it as a single "quality" number — the commercial one literally maps `Low/Medium/High/Highest` onto 2/3/4/maximum pyramid levels. It belongs to the optimiser, and the library owes it only the downsampled levels (§4) and a transform that can be scaled between them |
| **feature / keypoint registration** | 3 + P | **no** | Detect, describe, match, then fit a model under a robust estimator. The detection half is family A/B (a corner or blob response is a filter); the matching and model-fitting half is family C and is category P — arithmetic over two coordinate lists. `ops::detect` already produces one point per connected region, which is a fiducial detector in all but name, and `alignment/linear_sum_assignment` in the sibling application crate is the matching half. This is the most nearly-assembled missing capability in the survey |
| **rigid / affine transform model fitting** | **P** | **no** | Given corresponding point pairs, solve for the transform. Small dense linear algebra; reads no voxels; category P and arguably out of the library entirely, in the same bucket as the global solver of §1 |
| **B-spline / deformable registration** | **5 (C1)** | **no** | Producing one is an optimisation run and is out. **Evaluating** one is in, and is category P for points and category 5 for images (§3, displacement fields) |
| **drift correction over a time series** | — | see §9 | |

---

## 6. Composition and mosaicking

The commercial reference decomposes stitching into four ordered, independently
controlled stages, and that decomposition is worth adopting wholesale because
the large computer-vision library's stitching module and the general biological
application's grid-stitching plugin both make the same cuts:

| stage | what it does | category | have it? |
|---|---|---|---|
| **1. layout** | each tile carries a nominal origin; the mosaic extent is the union of tile footprints. No image content is examined | **5 (G2)** | no |
| **2. per-tile radiometric correction** | a multiplicative flat-field correction applied to each tile *before* geometry is computed | 1 (family A) | `ops::background` and `ops::normalise` are the shape of it; per-tile application is blocked by stage 1 |
| **3. pairwise alignment** | estimate the offset between overlapping tiles from content in the overlap band, refining the nominal placement | **3** | **the primitive, yes** — this is precisely what `ops::fft` was built for, off-centre lag window and all. The orchestration, no |
| **4a. global placement** | a least-squares fit over all pairwise estimates, producing one consistent origin per tile | out (§1) | no — but the `fragment` + sidecar mechanism is the right carrier for the pairwise estimates, and the README names this exact case |
| **4b. fusion / blending** | average grey values across tile edges so no seam remains | 1 | no |

The reference's own four end-to-end variants are a useful statement of what each
stage buys, and worth repeating because they say why 4b alone is not enough:
alignment only leaves shading roll-off and visible edges; alignment plus shading
correction leaves edges; alignment plus fusion is insufficient when roll-off is
strong; all three leave no visible transition. **Stage 2 is not optional and it
is not cosmetic.**

### Why stage 1 is category 5 — this is G2

> **No phase has inputs at different offsets.** A `Decomposition` hands a phase
> one image and maps each block to one region of it. `PhaseDecomposition`
> carries `source_images`, so a phase *can* read a second array — but
> `strategy.rs` reads every source image at the **same** `fetch` region as the
> input, and `check_source_images` refuses an image "on a different lattice".
> N tiles at N origins is N translations, and no `Reach` expresses a
> translation.

Two further things block it independently, and both are worth naming so the
fix is not mistaken for one change:

* **A `Region` cannot begin below zero** — this is **C2**. `start: Vec<usize>`. A tile whose
  refined origin moves it left of the mosaic origin has no region. Every
  mosaicking implementation deals with this by re-origining the whole layout
  after the global solve, which is fine — but it means the layout is not
  expressible until after the solve, and the solve needs the layout.
* **`Placement::sources` already exists** — `Vec<(usize, Anchor)>`, "per source
  image, where that buffer sits in its own image" — and is the beginning of the
  answer. What is missing is the *plan* side: `BlockGeometry::source` is one
  region, not one per source image.

### Mosaicking in the intended shape

Assume the gaps above are closed. The pipeline decomposes cleanly and it is
worth writing down because it shows the framework already has most of the
machinery:

1. A phase per overlapping tile pair, reading two images at two origins,
   computing a landscape with `ops::fft`, and writing **one fragment per pair**:
   the argmin lag and a confidence. Category 3 per pair, embarrassingly
   parallel across pairs.
2. A fold over those fragments — `fragment::fold_fragments` streams them one at
   a time — feeding a least-squares solve. Out of the library (§1); in the
   library's *storage*, which is the point of the sidecar mechanism.
3. A final phase reading N images at N solved origins and writing the fused
   mosaic. This is the stage-1 gap again, now with the origins known.

Steps 1 and 3 are the same missing declaration. That is the useful finding: **one
framework change unblocks both ends of mosaicking**, and it is a change to how a
plan states a fetch, not a change to `Reach`.

> **One thing this pipeline assumed without naming it has since been supplied.**
> Steps 1 and 3 both say "reading N images", and when this was written a run had
> exactly one input array and every other image was written by a phase of that
> run — so N tiles could not be images at all, whatever the fetch said. That was
> **G5**, which family D minted and this document did not name; the index's §6
> adjudicated that family C needs it. **It is closed**: a run can be handed N
> arrays, addressed in a disjoint high range. The pipeline above is unchanged and
> so is its blocker — the finding that steps 1 and 3 are one declaration stands,
> and it is now the *only* thing between this pipeline and a run.

---

## 7. Projections and reslicing

| operation | category | have it? | notes |
|---|---|---|---|
| max / min / mean / sum projection along an axis | **2†, verified — and now declarable** | no op, but the route works | see below. Measured in `tests/collapsing_phase.rs`: plans, runs, byte-identical to a whole-volume reference across 25 cuts of the two free axes. Since the geometry change landed the dependency can also be *stated* — `AxisReach::All` in `Space::source_voxels()` plus `with_sources` — and is checked against the fetch. `2†` is unchanged: the fetch is still per block |
| standard-deviation and weighted-mean projection | 2†, verified | no | the commercial reference offers all five; they are the same shape and the same route |
| **broadcasting a collapsed axis back**, `[1, Y, X] -> [N, Y, X]` | **2†** | no | The projection's inverse, and the same escape plus `BlockOp::takes_extent_from_placement` — a waiver `ops::lattice`'s interpolate half already declares. It belongs in this table because it is the *other* half of one statement (index §9), and because the map half of every reduce-then-broadcast workflow in families A, B and D is this row. Its own missing declaration is a **pinned axis**, and `InputMap::Affine` provably cannot express one: the source extent is the block extent times a rational, and `up = 0` gives extent 0, not 1 |
| depth-coded projection (source plane encoded by colour) | — | no | it produces three images from one and encodes a *display* choice. Composition of a projection and an argmax-index projection; the colouring is the viewer's |
| slab projection (start + thickness) | 2† | no | the same op with a bounded axis range — which makes it **category 1**, not 2, and is the cheap and more useful form |
| arbitrary-plane reslice | **5 (C1)** | no | a general oblique resample; see §3's displacement-field row, of which this is a well-behaved special case (affine), but with an output grid whose axes are not the input's |
| orthogonal views | out (§1) | — | a viewer arrangement of three reslices |
| montage / tiling of a series | **5 (G2)** | no | N sub-volumes written to N offsets in one output — the §6 gap again, in its simplest form. Notably, the commercial reference has no montage operation either; the nearest thing is re-tiling at export |
| kymograph (reslice along a drawn path) | 5 | no | **family D's**, since its second axis is time — noted here because the *reslice* half is family C's shape: a path-driven gather, not a lattice op at all. Closest to `ops::rows::gather`, which reads a volume at a row's own coordinate — a genuinely plausible route for both families |

### On the projection gap, stated carefully

G1's framing is that a projection is not expressible because `Reach` states
per-axis halo widths on a **same-rank** output, while a projection's reach is
"the whole of one axis, collapsed" — a different output geometry, not a halo
width. That is correct about `Reach`, and it is not the whole story, so here is
the more precise version, because family C is where projections live and the
difference decides whether one can be built today.

A projection along axis 0 of an `[N, Y, X]` volume produces `[1, Y, X]`, which
**is** a legal 3-D volume — the README is explicit that a 2-D problem is a
volume of depth 1. So the output shape is representable and `output_shape` can
return it. What actually breaks is the **declaration**:

* The honest reach is `AxisReach::All` on axis 0 in `Space::source_voxels()`.
  Converted into the phase's own voxels against an axis of extent 1, `All`
  resolves to `(1, 1)`; `Frame::Source` denies the clamp exception at the
  phase's own boundary — correctly, since that boundary is an interior position
  of the array being read — so the trustworthy extent is empty, the valid region
  collapses, and the tiling check fires. **The truthful reach is not a reach the
  guard accepts.**
* The available route is the `†` escape: cut the grid on the output volume,
  state the fetch per block in `BlockGeometry::source`, declare `Reach::none()`
  in `Space::source_index` — the *marked* zero — and hand-write a phase builder.
  Structurally this is exactly `ops::lattice`'s statistic half, which already
  reduces a window of a fine image to one coarse sample; a projection is the
  same op with the window equal to the whole axis.

  **Superseded as a recommendation.** Since the geometry change landed, the
  route to use is `AxisReach::All` in `Space::source_voxels()` **plus**
  `with_sources`: the same grid, the same per-block fetch, the same hand-written
  builder, and the dependency *stated* instead of hidden — which is the only
  version of it that anything checks. The escape still plans; it is now strictly
  weaker. Keep the paragraph because it is the shape `ops::lattice` still has.

So the correct classification is **2† — expressible only through the cross-grid
escape**, and the costs are: a hand-written plan builder per op, no automatic
planner can produce the phase, and the real dependency is recorded as a fetch
region rather than as a reach, so nothing about it reads as "this op depends on
the whole axis" at a glance.

> **Verified — measured.** *What this section said:* the paragraph above,
> argued from reading `reach.rs`, `geometry.rs`, `decomposition.rs` and
> `ops/lattice.rs`, with no op built and no plan run, and flagged unverified in
> §12. *What was measured:* the fifty-line experiment §12 asked for, now
> `tests/collapsing_phase.rs`, twelve tests. **The argument holds to the
> arithmetic.** `[N, Y, X] -> [1, Y, X]` plans, runs, and is byte-identical to a
> whole-volume reference across 25 cuts of the two free axes. `2†` stands and is
> no longer unverified; families A and D, which classified a projection as
> `X (G1)`, are wrong and have withdrawn it.
>
> Three things the read could not have shown, and they sharpen rather than
> soften the finding:
>
> 1. **The refusal is unconditional, not a guard a halo appeases.** Proven over
>    extents 1, 2, 5 and 32 and every block edge, including `Reach::all()`
>    offered as the halo: in `Frame::Source`, `trust_lo` and `trust_hi` cross
>    for every possible read.
> 2. **The same words in the phase's own frame plan, and say nothing.**
>    `Space::phase_voxels()` gives a phase that runs and is correct — but
>    `is_whole` requires `extent > 1` (`reach.rs:322`), so against the collapsed
>    axis `AxisReach::All` is not even a planning barrier. It is accepted
>    because it is vacuous.
> 3. **Nothing checks the fetch against the declaration.** A projection that
>    reads only its own block passes every guard and is wrong at every position;
>    a fetch covering half the axis is accepted too. That is the index's **G14**,
>    and it is what makes the `†` route in this table a correctness risk and not
>    only an inconvenience — every `†` op in this family, `ops::resample` and
>    `ops::lattice` and `ops::adjacency` included, is correct because its author
>    wrote the fetch out right.

> **Then the geometry change landed, and items 1 and 3 above are superseded.**
> Both are kept, because between them they are why the fix has the shape it has.
> `tests/collapsing_phase.rs` is now 15 tests; the index's §8.5 is the full
> account. For this family the four things that matter:
>
> 1. **The refusal is now conditional and the condition is exactly right.** The
>    clamp exception is denied in `Frame::Source` because a *cropping* phase's
>    edge is an interior position of the array it reads — a neighbour exists
>    there and a halo could have reached it. **That does not hold for an axis the
>    op consumes entirely:** there is no beyond, so no such neighbour. The grant
>    is restored per axis when the reach there is `AxisReach::All` *and* the
>    block's read spans the whole of that axis (`geometry.rs:319-322`). Family C
>    should note which half of its own §3 argument this is: the frame still
>    protects the crop, and stops charging the projection for it.
> 2. **The declaration is checked against the fetch** (`decomposition.rs:763-791`),
>    by name, reporting the axis, the block, the fetched range, the required
>    range and the phase's own extent — the last being the number that explains
>    why no halo helps. Item 3 above is closed *for a declared whole axis* and
>    open everywhere else; see §10.
> 3. **So the escape is now strictly weaker, and this table's route changes.**
>    Use `AxisReach::All` in `Space::source_voxels()` **plus** `with_sources`.
>    The escape records *that* a dependency exists; the declaration records
>    *what would satisfy it*. Measured: a half-axis fetch is accepted under the
>    escape and refused under the declaration — and under the escape it then
>    runs and returns wrong numbers.
> 4. **`2†` stands.** The fetch is still stated per block, so a hand-written
>    builder is still required and no automatic planner produces the phase. What
>    this document predicted would become a plain `2` did not.

What would make projections first-class rather than escaped is **not** a
rank-reducing phase — the rank was never the problem, and this section is where
that was first said. It is a declaration that says "axis `k` of the input is
consumed entirely and appears in the output at extent 1", from which the fetch
region, the block grid and the barrier all follow. Its mirror image, "axis `k`
of the input is pinned at extent 1 while the output grows", is the broadcast row
above, and the index's §9 records the proposal that says both at once. See §10.

---

## 8. Surface and mesh

Isosurface extraction is the one mesh operation that belongs in a voxel library,
and the boundary is worth stating plainly rather than surveying past.

| operation | category | have it? | notes |
|---|---|---|---|
| **marching cubes / isosurface extraction** | **1, with a non-voxel output** | no | The *reach* is trivial: one voxel on each side, a 2×2×2 cell. What does not fit is the output — a triangle list is not an image. It is exactly the shape `ops::detect` already has (a `fragments -> fragments` phase writing a variable-length payload per block), and the seam problem is the one `ops::components` already solves (vertices on a block face must be shared, not duplicated). **This is the most nearly-expressible of the missing operations in family C**, and the only interesting question is whether de-duplicating vertices across a seam is worth a merge phase or whether a caller can weld afterwards |
| **contouring** (the 2-D case) | 1 | no | marching squares; same shape, one dimension fewer |
| mesh decimation, remeshing, smoothing | **out** | — | reads no voxels. §1 |
| boolean operations on meshes, parameterisation, mesh IO | **out** | — | as above |
| surface fitting to scattered points (a polynomial or spline surface through `(x, y, z)` samples) | P | no | The commercial reference uses this heavily for focus surfaces, with a robust pre-fit that rejects outliers beyond `n·σ`. It is small dense linear algebra over a coordinate list — category P, and probably out of the library for the same reason as the global solver. Worth noting only because it is *not* a mesh operation and people file it under one |

**The line:** a triangle list produced from a scalar field is the last thing a
voxel library owes. Everything downstream of it reads no voxels, has its own
mature libraries, and its own data structures with none of `blockflow`'s
concerns. Extracting the surface is in; owning the surface is not.

---

## 9. Series composition, and what went to family D

Most of what this section originally held is family D's: channel merge and
split, blending modes, composite construction, colour-space conversion and
spectral unmixing are all arithmetic along the channel axis at reach 0, and
`docs/ops-survey/channels-and-time.md` is where they are surveyed.

What stays here is the machinery those operations *apply*, plus the one row
whose difficulty is genuinely spatial.

| operation | category | have it? | notes |
|---|---|---|---|
| **alignment across a time series (drift correction)** | **3 + 5 (G2)** | **primitive yes, application no** | **Family C owns the transform machinery, family D owns the use.** Estimating the drift between two frames is exactly `ops::fft` — category 3, and the primitive exists. Applying it is a translation per frame: N volumes at N offsets, which is G2 in its one-dimensional form. Because time is not an axis here, each frame is its own image, so a drift correction is a chain of images rather than an op — arguably the honest shape. See family D for the estimation schedule and the per-channel application |
| **chromatic / channel alignment** | 3 + 1† | primitive yes, warp no | The same estimate, then a **sub-voxel translation** of one channel, which is §3's affine warp at a fractional offset. Family C owes the warp; family D owes the observation that one transform is estimated per channel pair and applied to whole channels. Note it is a *sub-voxel* translation and therefore needs interpolation — an integer-lag argmin is not enough, which is the §5 sub-voxel-peak-fit row again |
| **extended depth of focus** | **2†, verified** | no | pick, per lateral position, the axially sharpest sample. A projection (§7) whose reducer is "argmax of a local sharpness measure" rather than "max", so it is a composition: a family A filter, then a collapsing phase. Every reference has it and it decomposes the same way. *Corrected:* this row read `2† / 5 (G1)`, offering "not expressible" as the honest alternative. It is not an alternative — the `†` route is verified to work (§7), so the only thing G1 costs this row is the declaration |
| **multi-view fusion** | 5 (G2) | no | N volumes at N solved poses, blended. Structurally identical to mosaicking (§6) with rotation in the poses rather than translation alone, and blocked by the same one thing |

The model behind all of this is already fixed by the README and is worth
repeating because it is what makes this section short: volumes are 3-D,
**channels are separate arrays** combined by a step that reads two sources, and
time is not in scope. Every "N frames" operation above is therefore N images,
not one volume with an extra axis — which is why they all reduce to G2 rather
than to a missing axis.

---

## 10. What the framework would need

Consolidated, in rough order of how much each unblocks. Family A's numbering is
used for the four shared structural gaps; the three that are family C's own are
numbered **C1–C3**.

### G2 — a phase whose inputs sit at different offsets *(family C's most-wanted)*

**Unblocks, in family C:** mosaicking at both ends (§6), montage, drift
correction (§9), multi-view fusion, and any two-image operation where the images
are not already registered. Family A wants the same gap for coarse-to-fine
pyramids; the two asks are compatible and family A's framing — a `SourceInput`
carrying a rational **scale and offset** rather than only a `Reach` — is the
more general one and subsumes what mosaicking needs (offset with scale 1).

`BlockGeometry::source` is one `Region`; make it one region **per source
image**. `Placement::sources` already carries `Vec<(usize, Anchor)>` on the
execution side, so the shape is agreed at one end; `strategy.rs`'s
`env.read(image, fetch)` loop — which today passes the *same* `fetch` to every
source image — is where it lands, and `check_source_images`'s "different
lattice" refusal is what would have to become "different lattice, and here is
where each block reads it".

Two things travel with it: **C2** below, and an accounting term, since
`exact_read_voxels` currently sums one fetch per block.

**Three, now — and the third arrived from family D's gap closing.** `blockflow`
can be handed more than one array: **G5** is closed
(`ArrayEnvironment::with_inputs`, `ZarrEnvironment::create_with_inputs`, and a
supplied array addressed as `ImageId::supplied(i)`). The index adjudicated in its
§6 that family C needs G5 too and that this document, predating it, did not name
it — N tiles are N acquired arrays, and getting them into one run at all was G5.
**That half is done and mosaicking is exactly as blocked as it was**, by this row
and by C2, which is the useful confirmation: getting the tiles in was never the
hard part, it was the part nobody could do.

The third thing that travels with G2 is what the closure exposed. A supplied
array is required to be in **image 0's coordinate space** — stated as a rule
rather than recorded per input, and enforced: `check_source_images` compares it
against the volume of every phase that reads it and refuses the pair **by name**.
So a phase downstream of *any* reshape cannot read a supplied array, which for
this family means a tile cannot be read after a rescale, and multi-resolution
anything is refused at plan time rather than mis-fetched. That refusal is the
right failure and it is this row's territory: the general form —
`BlockGeometry::source` as one region per source image, `SourceInput` carrying a
rational scale and offset — is what would replace it with an answer.

### G1 — the geometry cannot declare a collapsed or a broadcast axis

*(Renamed in the index's §8.1; identifier unchanged so citations survive. Family
A's original name, "a rank-reducing phase", named the wrong thing: nothing needs
a rank change, because `[1, Y, X]` is a legal `Array3` and a degenerate axis is
how this crate models lower rank on purpose. §7 had already established the rank
was not the obstruction; the rename makes the register agree with it.)*

**Unblocks, in family C:** the *declaration* for projections, slab projections,
extended depth of focus and orthogonal-view images — none of which it blocks the
*building* of, because §7's `†` route is now verified to work. And it makes
`ops::lattice`'s statistic half a special case of a general thing rather than a
bespoke arrangement.

Family A states two candidate routes (a lower-rank **side output**, which works
today but is terminal; and a rank-reducing phase with a `Collapse` statement).
Family C's addition, now measured, is that **a projection is reachable today
through the `†` escape** — see §7. So the ordering is: the escape works now,
is expensive to write, and is unchecked (G14); the side output does not help,
because a projection's whole point is to be read by later ops; and the real fix
is a per-axis extent rule on the input map rather than a `Collapse` on `Reach`,
because the same rule states the **broadcast** — the projection's inverse, which
runs today by the same escape and whose own declaration `InputMap::Affine`
provably cannot express. The index's §9 records that proposal. Whatever shape it
takes, it should be shaped so that `ops::lattice` can be rewritten onto it as
the proof it is general.

**Landed, with one half of what this section wanted.** The geometry change
shipped. A **collapsed** axis can now be declared — `AxisReach::All` in
`Space::source_voxels()` — and the declaration is enforced against the fetch, so
for projections, slab projections, extended depth of focus and orthogonal-view
images the missing declaration is no longer missing. `2†` still stands, because
the fetch is still stated per block.

What is left of G1 for this family is the **broadcast**: the pinned axis of §7's
broadcast row, which `InputMap::Affine` provably cannot express and which is
still reachable only through the escape. The index narrowed G1's name to that
half and kept the identifier; the per-axis extent rule of the index's §9 remains
the proposal that would state both sides with one mechanism.

**And one thing this section should not be read as claiming.** The check that
landed holds an op to what it *said*; it cannot make an op speak. A phase that
declares nothing and fetches one plane of an axis it means to consume still
plans, still runs, and is still wrong at every position — the index's **G14**,
partially closed and with that residue named.

### C1 — bounding a data-dependent reach

**Unblocks:** displacement-field warping, arbitrary-plane reslicing, and with
them the *apply* side of deformable registration.

A reach must be a function of the block index and nothing else, because a
`Decomposition` is parity-visible and reproducible; a displacement field's halo
is whatever the field says. Three ways out, in increasing order of honesty:

* **A declared bound.** The caller states the maximum displacement; the op
  declares that as its reach and **refuses at the block** any displacement
  exceeding it, by name. This is the shape `ops::voxelize` already takes with
  its structuring element — "boundedness is the plannability condition" — and
  it is almost certainly the right answer. It costs the caller a number they
  usually know.
* **A precomputed per-block table.** Scan the field once, before planning, and
  emit `AxisReach::PerBlock`. Exact and tight, but it makes the plan a function
  of data — forbidden — unless the scan is a separate, earlier *run* whose
  output is a plan input, which is legitimate.
* **A B-spline grid's own bound.** For a spline transform the displacement is
  bounded by the control-point coefficients, which are in the parameter file
  and are read before anything is planned. This is the declared bound with the
  number *derived* instead of stated, it satisfies "everything is a parameter"
  without asking the caller to guess, and it is the case worth building first
  because it is exactly what an elastix-format evaluator is already handed
  (§5).

### C2 — a region that can be positioned relative to a non-zero origin

**Unblocks:** padding (§3), tile placement before the global solve (§6), and
any operation whose output canvas is larger than its input.

`Region { start: Vec<usize>, shape: Vec<usize> }` cannot begin below zero.
Mosaicking implementations deal with this by re-origining the whole layout after
the global solve — which is fine, and may be all that is needed — but it means
the layout is not expressible until after the solve, and the solve needs the
layout. Either a signed origin, or a documented convention that re-origining
happens before planning and the crate never sees a negative coordinate. **The
convention is the cheaper answer and it should be written down**, because right
now a caller has to infer it.

### C3 — axis permutation acted on, not merely carried

**Unblocks:** transpose, 90° rotations, reslicing along a non-native axis.

Already half-designed: `reach::Space` carries `axes: [usize; 3]` and it is
fingerprinted. Its own doc comment says what remains — a permuted branch needs
the lattice, the read extent, the valid region and the anchor permuted
*together*, and the permuted reach is the cheapest of the five. `BlockGrid` and
`Anchor` are the work.

### G3 — a complex element variant

**Unblocks, in family C:** phase correlation as a *phase* rather than as a
resident kernel. That is the only thing family C wants from it, and it is not
urgent, because phase correlation as a resident kernel (category 3) is perfectly
usable for a mosaic. **Family A's call** — it owns frequency-domain filtering,
where the gap actually bites — and A's survey names a band-pass filter as the
natural forcing case. Noted here only because family C's registration primitive
is what first exposed the observation.

### G4 — pricing `log n`

**Unblocks, in family C:** honest pricing of any plan containing a registration
landscape. `ops::fft` is a resident transform and the cost model prices it as
linear, so a plan that runs one per tile pair is mispriced by a factor that
grows with tile size. This does not *block* anything in family C; it makes
every mosaicking cost estimate wrong in a known direction, which is worth
recording before somebody trusts one.

**And the misprice now reaches further than a wrong estimate.** The partition
search's *objective* has changed, and this was checked rather than assumed. It
always swept block candidates per phase; what was uniform was the **answer**,
because the old objective — the phase's serial work, `cost_per_block × n_blocks`
— is `volume × redundancy × per-voxel`, where `n_blocks` cancels and redundancy
falls monotonically, so the sweep answered "the largest candidate that fits"
every time. It is now `max(pool, channel)`: the pool bound
`cost_per_block × ceil(n_blocks / workers)` and the channel bound
`read × read_cost + core × write`, measured at **2.6×** on a mixed plan against a
control with identical reads and identical serial work.

So a mispriced phase no longer only reports a wrong number — **it chooses a
grid**, through the pool bound, which is the half that decides how finely to cut.
For this family that lands on exactly the plan §6 describes: a landscape per tile
pair, priced linear, sitting in a chain whose other phases are being sized
against it. The index's §11.2 carries the account once for all four families.

### Smaller, and worth doing regardless

* **Sub-voxel point coordinates.** `points::Point` holds `[usize; 3]` and
  `ops::rows::scaled_index` refuses a negative factor because "a table holds
  `usize` coordinates". A chain of point transforms therefore rounds at every
  step, and a transformed point cannot be negative. `[f64; 3]` in the table, or
  a documented fixed-point convention, is what an evaluate-a-transform-at-a-
  point-set operation needs to be worth having.
* **A third `InputMap` arm for an axis reversal**, which gets flip and 90°
  rotation between them.
* **A stated boundary-mode parameter convention** for ops that want more than
  clamp, so each op does not invent one.
* **A `Combine` implementation or two that are not boolean**, since the trait
  already permits them — though the arithmetic ones are family D's to specify.

---

## 11. Present, but narrower than the name suggests

Checked rather than assumed.

| name | what a reader expects | what it is |
|---|---|---|
| `op::InputMap::Affine` | a general affine coordinate map | an **axis-aligned rational rescale** — `up: [usize; 3]`, `down: [usize; 3]` — plus an interpolation window. No rotation, no shear, no translation. The name is the strongest single mis-signal in family C. **And no pin:** the source extent it implies is the block extent times a rational, so holding an axis at 1 for every block extent needs a factor tending to zero and `up = 0` gives extent **0**. That is why the broadcast row of §7 has no declaration rather than an unwritten one |
| `ops::rows::ScaleRowsOp` | a transform applied to a point set | a **per-axis scalar multiply** of `usize` coordinates with ties-to-even rounding. No offset, no rotation, no sub-voxel result |
| `ops::fft` | a Fourier transform | a **2-D real-plane** transform. Every shape in the module is `[usize; 2]`. Correct and fast for plane-wise work; not a volumetric transform |
| `ops::resample::Interpolation` | a kernel family | **two** variants, nearest and linear. Linear refuses `bool` at plan time rather than silently dilating a mask, which is right and worth knowing |
| `ops::coordinates` | coordinate handling | **mask in, list of set-voxel coordinates out**, and it is nearly all about *ordering*. Not a coordinate-space facility |
| `voxelwise::CombineOp` / `LogicCombine` | image arithmetic between two images | **boolean connectives only** — `And`, `Or`, `Xor`. Arithmetic combination of two images needs a caller-supplied `Combine`, which the trait permits and the crate does not ship |
| `Geometry` / `InputMap` (as a whole) | the declaration ops use | **nothing consumes it yet.** Its own doc comment says so. It is the landed-with-a-default first step of a migration, and it is where two of §10's asks belong |
| `src/tests.rs`'s `crop_plan` | a crop | a **test fixture** demonstrating a shape-changing plan. There is no crop op |

---

## 12. Unverified, and what could not be established

* **The projection argument in §7** was derived by reading `reach.rs`,
  `geometry.rs`, `decomposition.rs` and `ops/lattice.rs`. No projection op was
  built and no plan was run to confirm that the `†` route actually plans, that
  the tiling check passes, or that `AxisReach::All` in `Space::source_voxels()`
  collapses the valid region as argued. It was **unverified**, and named as the
  one claim here that a fifty-line experiment would settle.
  **Settled: the experiment was run and the argument holds** —
  `tests/collapsing_phase.rs`, twelve tests, and the correction is in §7. This
  entry is kept rather than deleted because what it asked for is what happened,
  and a survey that removed its own open question would lose the fact that
  asking it was right. **And then it was acted on:** the geometry change landed
  (the file is now 15 tests), so the "truthful reach" this bullet doubted could
  even be stated is now the recommended way to state it. Two rounds, and the
  order was the right one — the experiment first, the change second.
* **The rotation halo estimate** in §3 (block diagonal times `sin θ`) is stated
  from geometry, not measured. The interaction between that halo and block size
  is the thing that decides whether a rotating phase is practical, and it has
  not been measured. **Unverified.**
* **Whether the automatic planners could be taught shape-changing phases at all,
  or whether hand-written builders are inherent**, was not established. What was
  established is that today they plan one volume per group and that the three
  shape-changing ops each ship their own builder.
* **The elastix-format evaluator's coverage** (§5) was read and is reported as
  found. What was *not* checked is whether its conventions agree with any
  particular reference implementation beyond what its own header claims — the
  header states they were pinned by recorded comparison against a reference
  binary rather than read from documentation, which is the stronger claim, but
  this survey did not re-run that comparison. **Unverified at second hand.**
* **`docs/design/BLOCK_OPS.md` is quoted by many module headers throughout
  `src/` and does not exist in the repository.** Every design argument attributed
  to it in this survey is therefore cited from the *header that quotes it* —
  `reach.rs`, `geometry.rs`, `decomposition.rs`, `ops/resample.rs`,
  `ops/lattice.rs`, `ops/element.rs` — and not from the document, which could
  not be read. Where a header paraphrases rather than quotes, this survey may be
  one restatement further from the original than it appears.
* **No numbers in this document were measured for it.** Every figure quoted —
  3.27×, 5.9×, 38×, 1.52–2.01×, the padding table — is taken from the module
  header or test that measured it, and those are runnable. Trust the ratios.

---

## Appendix: the shortest summary

Family C's primitives are in better shape than its orchestration.

* **Resampling is solid**, and its hardest-won finding is about *conventions*,
  not interpolation: an output extent is half of what "resample by a factor"
  means, and mixing two references' halves matches neither.
* **The registration kernel exists and is good** — an SSD and correlation
  landscape with a padding rule sharper than the textbook one, exploiting an
  off-centre lag window, measured at 38× the direct computation. It is 2-D and
  it has no sub-voxel refinement.
* **Everything that combines images at different offsets is blocked by one
  thing — G2**: a phase reads all its images at one region. Mosaicking, montage,
  drift correction and multi-view fusion are the same missing declaration seen
  four times. Family A wants the same gap for coarse-to-fine pyramids, and its
  more general framing (a scale *and* an offset per source) subsumes what family
  C needs. **And it is now the only thing**: the second gap under all of them —
  that N acquired tiles could not be images at all — was **G5**, which this
  document never named, and it has closed. A supplied array must be in image 0's
  coordinate space, so what used to be "no way in" is now "in, at one region",
  which is this bullet exactly.
* **Projections are reachable, and can now say so.** They were reachable only
  through the cross-grid escape — verified, `tests/collapsing_phase.rs` — which
  works, is expensive to write, and was **unchecked**: nothing compared a stated
  fetch against the dependency it stood in for, so a projection reading one
  plane instead of the axis was accepted and wrong at every position. Since the
  geometry change landed, `AxisReach::All` in `Space::source_voxels()` plus
  `with_sources` states the dependency and is **checked against the fetch**, and
  the escape is strictly weaker for this purpose. Two things did not change:
  `2†` — the fetch is still per block, so the builder is still hand-written —
  and the residue of the index's **G14**, since a phase that declares nothing is
  still checked against nothing. What is left of **G1** here is the projection's
  inverse, the **broadcast**, whose pinned axis still cannot be stated and which
  would retroactively simplify `ops::lattice` alongside it.
* **The boundary is where it should be.** Rendering, viewers, optimisers, global
  solvers and mesh processing are outside, and the crate has not drifted toward
  any of them.
