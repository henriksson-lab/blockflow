<!-- SPDX-License-Identifier: MIT -->

# Operations over the channel axis and the time axis

*A survey of family D: what a general image-processing library is expected to
offer over the two axes that are not spatial, what `blockflow` has today, and —
the part worth keeping — **whether this crate's decomposition model can express
each thing, and at what cost**.*

Three sibling documents cover the spatial families: **A** takes point
operations, filtering and transforms; **B** takes segmentation, morphology and
measurement; **C** takes geometry, registration and composition. This one exists
because those three, between them, cover the three axes of a volume and nothing
else. The classification below is family A's, reused rather than reinvented, so
that the four documents compose.

---

## 1. The fact at the centre of this document

**`Voxels` is `Array3` for every one of its eleven element variants.** Checked
in `src/voxels.rs`, which does not merely happen to be rank 3 — its header
argues for it, on measured grounds, against `ArrayD`: "an *image* is a volume",
two dimensions are the degenerate case of three, and a dynamic rank "bought
nothing and cost an indirection per index."

Everything else in the crate agrees with it, and each of these was read rather
than assumed:

* `Reach` is `{ axes: [AxisReach; 3] }`. A reach is three per-axis halo widths.
* `Workflow` is `{ chain, input: ArrayRef, output: ArrayRef, shape: [usize; 3],
  dtype }` — **one chain, one input, one output, one shape, one element type.**
* `BlockOp::output_shape` is `[usize; 3] -> [usize; 3]`.
* A `Table` row's position is `POSITION_WORDS = 3` words, always and always
  first; payload columns are `U64` or `F64`.
* `ArrayEnvironment::new(input, n_phases, chunk)` and
  `ZarrEnvironment::create(root, input, chunk)` both seed **image 0 from one
  `Voxels`** and create every other image as `pending`, to be written by a phase.
  Neither has a method that adopts a second existing array as an image.

So there is no channel axis and no time axis in the data model, and — the part
that matters more — **there is no second input volume either.**

`BlockOp::source_inputs` lets a phase name additional stored images with a
per-input `Reach`, and `Chain::Source` lets a leaf read one. That is real and it
is used (`ops::tabulate` reads a labels image and a values image;
`ops::walk` reads a second array over its offset sequence). But those images are
**intermediates of the same plan**, in the same coordinate frame, read at the
block's own fetch region — `check_source_images` refuses by name any source
input whose reach exceeds the phase's own halo, on the stated ground that "the
executor reads a source image at the block's own fetch region." A second
*acquired* volume has no way in.

A multi-channel operation is therefore N separate runs composed by the
application. A time series is the same story, with the added problem that its
natural operations are not voxel-local at all.

> **Corrected — G5 has closed, and the two sentences above are the ones that
> moved.** `ArrayEnvironment::with_inputs` and
> `ZarrEnvironment::create_with_inputs` seed a run with image 0 **and** a list of
> supplied arrays, so *"neither has a method that adopts a second existing array
> as an image"* is no longer true and *"there is no second input volume either"*
> is closed. A supplied array is an image in every sense this document uses the
> word: read through `source_images`, fetched through `Environment::read`, priced
> by the same byte accounting, named by `Chain::Source` and by
> `BlockOp::source_inputs`.
>
> **The rest of the paragraph above stands and is now the load-bearing part.**
> `check_source_images` still refuses a source input whose reach exceeds the
> phase's own halo, on the same ground — the executor reads a source image at the
> block's own fetch region — and a supplied input is additionally required to be
> in **image 0's coordinate space**, so a phase downstream of a reshape cannot
> read one and is refused *by name* at plan time. That residue is **G2**, which
> is unchanged and is now this family's remaining structural blocker for
> everything in §5.3.
>
> **What this means for the tables below.** Every row in this document reading
> "blocked by G5", "once G5 lands", "needs G5" or "modulo G5" is **unblocked**,
> and the rows that matter most are corrected where they stand. A
> multi-channel operation is no longer N runs composed by the application *by
> necessity*; N runs remains the right shape for per-channel *parameters* (§4.5),
> which was always a separate argument and is untouched.

> **Corrected — the framing, not the conclusion.** *What this document did:*
> opened with `Voxels` being `Array3`-only as "the fact at the centre", quoted
> `voxels.rs`'s argument for it, and then reasoned about the rest — §2's G1
> bullet, §5.5's rank-reducing rows, §6's table — as though rank 3 were a cap to
> work around. *What is actually the case:* **3-D is a deliberate floor, not a
> ceiling.** One and two dimensions are *modelled* as three with degenerate
> axes, and the reason is stated in the header this section already quotes:
> doing it that way means lower rank costs **no new element types**. A `[Z, 1, 1]`
> per-frame statistic and a `[1, Y, X]` projection are ordinary `Voxels`. That
> is why a collapsing phase plans, runs and is decomposition-invariant today
> (`tests/collapsing_phase.rs`), which this document's `5 (G1)` rows deny —
> those rows are corrected in place below.
>
> **The conclusion of §3 is untouched and stands.** N volumes rather than a
> fourth axis rests on the reach-0 sort of §3.1, on the cost of §3.2, and on the
> storage contract of §3.4, and none of the three depends on how the rank floor
> is read. If anything the floor strengthens it: an axis added to a type whose
> whole design is "an image is a volume, and lower rank is a degenerate volume"
> is an axis fighting the type, not completing it.

---

## 2. The categories, and the gap that is mine

Family A's five categories, unchanged, so the tables below read the same way:

| # | Category | What it means here |
|---|---|---|
| **1** | **Bounded reach** | A halo the op declares. Cheap, blockable, the ordinary case. |
| **2** | **Whole-axis reach** | `AxisReach::All` on one axis, bounded on the others. Still no user in `ops/` — and, in `Frame::Source`, never a possible one; see the note under the gaps below. |
| **3** | **Whole-volume, resident** | `Reach::all()` — the phase gets a single block. |
| **4** | **Iterative to a fixed point** | `IterativeOp` + `iterative_phase`. |
| **5** | **Not expressible today** | And why, specifically. §6 consolidates mine. |

And family A's four structural gaps, referred to below by number:

* **G1** — *(renamed in the index's §8.1; identifier unchanged)* the geometry
  cannot declare a collapsed or a broadcast axis. This document, following
  family A, called it "no rank-reducing phase"; that name describes a problem
  the crate does not have.
* **G2** — no phase whose inputs sit at different offsets (or resolutions).
* **G3** — no complex element variant.
* **G4** — the cost model cannot price `log n`.

> **Corrected — measured** (`tests/collapsing_phase.rs`; index §8). Two things
> this table and this list get wrong, and they are the same thing twice.
>
> **G1 does not block a collapse; it blocks saying one.** `[X, Y, Z] -> [X, Y, 1]`
> plans, runs and is byte-identical to a whole-volume reference across 25 cuts
> of the two free axes — through the `†` cross-grid escape, that is, by *not*
> declaring the dependency. Family C classified this `2†` and was right. Every
> **`5 (G1)`** in this document that names a *collapse of a spatial axis* — the
> projection rows of §5.5, the per-frame statistic rows of §5.2, the
> colocalisation accumulators of §4.3 — is therefore wrong about
> expressibility, and each is corrected where it stands. The **broadcast**,
> `[X, Y, 1] -> [X, Y, Z]`, is G1's other half and also runs today by the same
> escape; it matters here because the rescale in every decay- and
> flicker-correction row is one.
>
> **Category 2 is two declarations.** In the phase frame `AxisReach::All` plans
> and works; in `Frame::Source` it was refused unconditionally, at every extent,
> under every halo. "Still no user in `ops/`" is true, and for the source-frame
> pairing there could not have been one.
>
> **Landed, and two of the three things predicted here did not happen.** The
> geometry change shipped. Source-frame `All` is now granted on an axis the op
> **consumes** and is **checked against the block's fetch**, so a collapse can
> be declared and held to it. But `2†` did **not** become a plain `2` — the
> fetch is still stated per block, so the builder is still hand-written — and
> **G14 did not close**: the check holds an op to what it *said* and cannot make
> an op speak, so a phase declaring nothing that reads one plane of an axis it
> means to consume still runs and is still wrong at every voxel.
>
> **So: name the frame.** For a collapse, `Space::source_voxels()`, which is
> checked. In the phase's own frame the same words against an extent-1 axis are
> *vacuous* — `is_whole` requires `extent > 1` — accepted without claiming
> anything. Two of the three declarations that plan are wrong in different ways.
> The escape still plans and is strictly weaker than the declaration.
> The **broadcast** half of G1 did not land at all.

To which this family adds two, and it is worth being precise about which is
which, because they are usually confused:

* **G5 — there is no second input volume.** A run is seeded with one array.
  Every other image is written by a phase of that run. This is the gap that
  blocks the whole channel family, and it is *not* a missing axis.
  *(**Closed.** See §1's correction and the index's §10.)*
* **G6 — there is no axis to sweep.** An operation along channel or time has no
  axis in the type to declare a reach on. This is the gap that would be closed
  by a fourth axis, and it turns out to block far less than G5 does.

The distinction between G5 and G6 is the subject of the next section.

> **And the distinction is what the closure vindicates.** G5 closed; G6 did not,
> and nothing in this document now wants it to. The whole argument of §3 is that
> the channel family needs **arity** and not an axis, and arity is what G5's
> closure supplies: a supplied array is where the extra operands come from, and
> `ops::mixing` is the shell that consumes them. This document's central claim —
> *"the strongest argument for a channel axis argues against it"* — is now
> argued by a working op rather than by a sort of the reference set.

---

## 3. The question: a fourth axis, or N volumes?

Both answers are defensible. **The answer argued here is N volumes** — the
crate should stay rank 3 and gain the ability to be handed more than one of
them, plus a K-ary reach-0 op shell. The argument is not a preference; it is a
sort of the operations by what they actually need.

### 3.1 The test that decides it

For any operation over a non-spatial axis, ask: **does it need a *window* along
that axis, or only the *tuple of values at one voxel*?**

A window needs an axis, because a window is what `Reach` describes. A tuple
needs *arity* — K operands at reach 0 — which is a different mechanism
entirely and is nearly present already.

Sort the reference set by that test and the result is lopsided:

**Channel: every operation in the set is reach 0 along C.**

* Linear spectral unmixing, stain or dye separation (the Ruifrok–Johnston
  family, as `skimage.color.rgb2hed` and Fiji's `Colour Deconvolution`),
  crosstalk and bleed-through correction — all are **one small matrix applied
  per voxel to the vector of channel values**. These are the operations that
  most clearly need simultaneous access to N channels at one voxel, and they
  are exactly the ones that would declare `AxisReach::none()` on a channel axis
  if it existed. A commercial acquisition-and-analysis package documents its
  unmixing as "strictly pixel-by-pixel least-squares fit against up to ten
  reference spectra", with the component count required to be at most the
  acquired channel count. Reach 0.
* Channel arithmetic, ratio and index images: reach 0, arity 2.
* Colocalisation coefficients: reach 0 in space, arity 2, then a **global
  reduction** — G1, not an axis.
* Merge, split, composite construction, colour-space conversion: reach 0.
* A per-voxel feature stack fed to a classifier: reach 0 in the channel
  direction by construction (the *spatial* reach belongs to the filters that
  built the stack, and those are family A's).
* Per-channel geometric correction: N *separate* transforms by definition — the
  whole point is that each channel gets a different one.

**So the strongest argument for a channel axis argues against it.** The
operations that need all N channels at once need them at *one voxel*. A channel
axis would give them an index and charge them a rank.

**Time is where the case is real, and it is narrower than it looks.**

* Temporal or running median as a background estimate, temporal smoothing, a
  gliding average, a first- or second-order temporal derivative: these *are*
  bounded reach along T. This is the one family that genuinely wants an axis.
* But every one of them is also a **K-ary reach-0 map over W consecutive
  frames**, and that is the same mechanism the channel family wants. A gliding
  average of window `W` is documented in the commercial package with the output
  extent stated exactly: `T' = T − W + 1`. Read as N volumes, that is `T − W + 1`
  runs, each naming `W` images, each with reach 0.
* And the *expensive* temporal operations — flow estimation, drift, linking —
  are not voxel-local at all. They estimate a transform or a correspondence
  between two frames. A fourth axis does nothing for them.

### 3.2 What a fourth axis would cost, stated

`Voxels` goes to rank 4 or to `ArrayD`, against an argument already written down
and already measured. `Reach` goes to `[AxisReach; 4]`. So do `Anchor`,
`BlockGeometry`, `BlockGrid`, `Region` in its executor use, `output_shape`,
`placed_output_shape`, `side_region`, the tiling check, the budget arithmetic,
the cache keys and the distributed placement — all of them `[usize; 3]` today.

Every op pays that, including the many that will never see a fourth extent above
one. A degenerate axis is not free: it is a dimension in every loop and every
check.

And then there is the tell. A channel axis is an axis the planner **must never
cut** — a block holding half the channels of a voxel cannot unmix it. So the
first thing a fourth axis would need is `BlockConstraint::FullExtent(axis)`,
which family B already lists as missing. You would add an axis and immediately
have to forbid the planner from using it. An axis the planner may not cut is not
an axis; it is arity with extra steps.

*Updated, and the tell survives intact.* Since the geometry change landed, an op
can mandate a whole axis **by declaring it** — `AxisReach::All` in the source
frame, enforced by refusal — so the constraint this paragraph says is missing is
now half-present. That does not weaken the argument; it sharpens it. The mandate
is a *refusal*, not a hint, so a fourth axis would still be an axis every plan
had to be forbidden from cutting, and the enumerator would still propose cuts to
be refused. "Arity with extra steps" was never about which mechanism does the
forbidding.

### 3.3 What the neighbours pay

**ITK is the instructive case, because it *does* have the concept.**
`itk::VectorImage` and the multi-component pixel types (`Vector`,
`CovariantVector`, `RGBPixel`, `SymmetricSecondRankTensor`) give ITK a real
channel model. The price is not the axis — it is the **fork in every generic
filter** between "operates on a scalar" and "operates on a tuple", carried by
`NumericTraits`, by pixel accessors, by `ImageAdaptor`, and by
component-selection filters for the many operations that are meaningful only per
component. It also shows up as two ways of saying the same thing that are not
interchangeable: a component count fixed at compile time versus one fixed at run
time. This crate's entire answer to the same question is `voxels.rs`'s: element
type as a run-time tag, and **one `match` per op shell**, at the seam where the
shell adapts to a kernel. A tuple element type doubles that match, permanently,
for every op. *(This description of ITK is from knowledge of the library, not
from a read of its source in this session — **unverified** in the sense the
other three documents use the word.)*

**VTK** carries multi-component scalars on the point-data array of a
`vtkImageData`, and its imaging filters mostly loop components; extracting one
is a filter (`vtkImageExtractComponents`). It also marks the boundary this
document has to respect: the moment channels are composed into a display
colour, you are in a mapper and out of the block-processing library.

**A commercial package's pixel type is a tuple too** — its format list includes
three-component colour types at two integer widths, a float colour type, and
complex types at one and three components. That is the same decision as ITK's,
made by a product rather than a toolkit, and it comes with the same consequence:
a large part of its function list is documented as applying "per colour
channel", and several functions exist purely to move between the tuple form and
the separate-image form (build a colour image from three extraction images and
split one back; build and split a hue/lightness/saturation triple; convert a
classification result between one-channel-per-class and a single label channel).
**Half the utility menu is conversions between the two representations**, which
is what a tuple pixel type costs you at the boundaries.

### 3.4 The evidence from the layer below

This is the strongest single piece of evidence and it comes from the storage
side rather than the algorithm side.

A sibling application and its container writer agreed a contract in which the
channel axis **exists in the store**: an NGFF-style axis order with a `c` axis,
a per-channel provenance list, and an NGFF axes list that must agree with it or
the read is an error. It carries one hard requirement: **the chunk and shard
extent along `c` must be 1.** The reasoning is recorded and is exactly right —
that is what makes the semantically correct layout free, because no chunk ever
holds two channels, so reading one channel touches the bytes a per-channel array
would, in the same number of requests. "Keep channels apart for speed" and "use
a channel axis for correctness" are not in tension; chunk shape settles it.

And the reader, correspondingly, maps the spatial axes out of the declared axis
order and treats every other axis as **pinned** — selectable by index,
defaulting to 0, with an out-of-range selection a named error rather than a
clamp.

So: **the channel axis lives exactly where it is cheap — in the metadata and the
addressing — and is projected away at the read boundary at no cost.** The block
framework sits downstream of that projection. Giving it a fourth axis would
re-import a concept the layer below has already priced and disposed of, and
would then have to forbid the planner from cutting it, restoring by hand the
property the chunk shape already guarantees.

The same package's *processing* granularity agrees independently: its
acquisition-side unit of work for pipelined processing is a full stack **per
channel**, per time point. The natural unit is a rank-3 volume, in the product
as much as in this crate.

### 3.5 The honest cost of the N-volume answer

Four costs, none fatal, all worth writing down.

1. **The set is untyped.** Nothing would know that images *k*..*k+C* are one
   channel set. `Combine::output_shape` catches an extent mismatch at run time
   inside one phase and `check_source_images` catches a missing image, but "these
   are the N channels of one acquisition, in this order, at this voxel size" is
   a fact only the caller holds. The sibling contract had to write it out by
   hand — a per-channel provenance list, a hard error on shape or element-type
   mismatch, a rule for which axis becomes `z` when two inputs disagree about
   their own axes, and a recorded gap where two inputs' physical voxel sizes
   could disagree and resolve silently. Every one of those is a check a channel
   axis would have got for free.
2. **N plans instead of one.** A per-channel filter is N workflows and the
   planner prices each alone. The apparent loss — shared reads — is illusory
   under the storage contract above, because no chunk holds two channels anyway.
   *Unverified:* whether N single-channel plans lose anything measurable against
   one C-fused plan has not been measured here.
3. **A K-out operation needs K side outputs or K passes.** Unmixing produces C′
   unmixed channels and optionally a residual channel from one input. A
   `BlockOp` can declare K side outputs today, each with its own name, element
   type and rank, checked to tile the declared output exactly. Whether a *later
   phase can read one back* is **unverified** (family B flags the same
   question). If they are terminal, an unmixing whose result must be filtered
   afterwards costs a second pass.

   > **Settled: they are terminal, and the second pass is the price.** Verified
   > in the index's §6 and now pinned by a test
   > (`tests/tuple_map.rs`): a side output lands in a `String`-keyed map on the
   > environment and never in `images`, so nothing can name one where an image
   > goes. **G5 does not bear on this**, and it is worth saying why, because the
   > two look adjacent: a supplied input is an array that existed before the run
   > and a side output is written during it, and the two are addressed
   > differently — an image by number, a side output by name — while
   > `Chain::source` and `SourceInput::image` take a number. Giving a side output
   > an image address would be a *third* thing, and that third thing is what an
   > ordinary image already is.
   >
   > **And there is a second cost this entry did not see**, measured since — and
   > it is now a *price* rather than a defect. The outputs beside the primary are
   > a function of all K inputs, and `BlockOp::apply_side` was not handed the
   > source inputs; `ops::mixing` shipped carrying them across from `apply_with`
   > in a per-block map. **That has since been fixed**: the argument is threaded
   > from the executor, the map is deleted, and `apply_side` is now a **total
   > function of its operands** rather than of its operands plus what an earlier
   > call left behind — which for this family matters because the arrays that map
   > held were whole `f64` blocks that no counter knew about, in the one part of
   > the crate that exists to make extra results countable.
   >
   > What is left is the cost this entry was right to anticipate, in a smaller
   > form than "a second pass over the volume": `apply_with` computes output 0
   > and `apply_side` outputs `1..K′`, so the **inputs are streamed twice** —
   > **1.08–1.10×** at K = K′ = 16, with the flop count unchanged because the
   > kernel takes a window (`from`) rather than recomputing row 0. The index's
   > §11.1 and §11.3.
4. **`Combine` cannot write more than one array.** Verified: the trait has
   `name`, `reach`, `reach_spec`, `accepts`, `produces`, `output_shape`, `apply`,
   `constant_maps_to`, `cost_per_voxel` and no side outputs. So the K-in/K-out
   shape has to be a `BlockOp` with source inputs and side outputs, not a fan-in.

### 3.6 The answer

**N volumes, plus an application that composes them** — with three additions,
none of which is a fourth axis and two of which the other families are already
asking for:

* **G5**: an environment that can be seeded with more than one input array, so
  that a second acquired volume can be an image. Everything downstream —
  `Chain::Source`, `SourceInput`, `check_source_images`, the byte accounting,
  image lifetimes, `Hints::keep_images` — already handles images above zero. What
  is missing is the numbering (images `0..k` are inputs, phase `p` writes image
  `k + p`) and the two constructors. **This single change unlocks more of this
  document than anything else in it.**

  > **Built — and the numbering this bullet asked for could not be built.** The
  > two constructors are `ArrayEnvironment::with_inputs` and
  > `ZarrEnvironment::create_with_inputs`, and the sentence about everything
  > downstream already working was right and is why it was cheap. The numbering
  > was not: `0..k` fails **twice**, and both failures are recorded in the
  > index's §10 rather than smoothed away. *One* — the executor addresses images
  > positionally (`env.read(task.phase, …)`, `env.write(task.phase + 1, …)`, at
  > some fifteen sites), so `k + p` needs the executor rewritten. *Two* — and
  > independently — a caller needs an input's address **before** it builds the
  > ops that read it, while the phase count is not known until `finish`, so
  > `0..k` cannot be reached from the builder either. What shipped is a
  > **disjoint high range**: `ImageId::SUPPLIED_BASE = usize::MAX / 2 + 1`,
  > `ImageId::supplied(i)`, with images the run writes numbered exactly as before.
  > Adding an input renumbers nothing, which is the property the numbering this
  > bullet proposed would have destroyed.
  >
  > Two rules travel with a supplied image and both are this family's business.
  > Its **element type has no fold** — no phase writes it — so the readers
  > declare it (`SourceInput::holding`, `PhaseDecomposition::supplied_dtypes`)
  > and disagreement is refused by name. Its **shape** is image 0's coordinate
  > space, so a per-channel volume that has been rebinned cannot be read
  > directly: that is **G2**, refused by name rather than mis-fetched.
* **A K-ary reach-0 op shell**, which family B asks for from the classifier side
  and which is what unmixing, stain separation, crosstalk correction, ratio
  images, channel argmax, colour-space conversion and a windowed temporal filter
  all are.

  > **Built, as `ops::mixing`** — `TupleOp` the shell, `TupleKernel` the kernel
  > trait, `LinearMap` the per-voxel matrix, which is the unmixing case exactly.
  > The shape is the one this document specified: one `BlockOp`, `K − 1`
  > `source_inputs` at reach 0, `K′ − 1` `side_outputs`, no new trait beyond the
  > kernel and **no new axis**. One clause was wrong — `apply_side` was not
  > handed the source inputs — **and it has since been fixed**, so the shape this
  > document specified is now true as written; see §3.5's entry 3 and the index's
  > §11.3. Measured at `K = K′ = 16` in `f32`: ~40 ns per position and **4.0**
  > flops per byte of image traffic, not the 2 that is usually quoted for this
  > shape, which is the `K′ = 1` case; the fix costs a further **1.08–1.10×** for
  > streaming the inputs twice, and keeps the tiling's 2.5–2.8× entire.
* **G1 and G2**, shared with A, B and C: a declaration for a collapsed or a
  broadcast axis *(corrected — this read "a rank-reducing phase", and there is
  no rank to reduce; §2)*, and a phase whose inputs sit at different offsets.

The operations that would change this answer, if the crate's users turned out to
need them, are the ones with a genuine *window* along a non-spatial axis and a
long one: a temporal median over a window of dozens of frames, or a smoothing
along a spectral axis of a many-band acquisition. At `W` in the low tens, K-ary
arity is fine. At `W` in the hundreds, K operands stops being a sensible way to
say "a window", and that is the point at which the fourth axis earns its keep.
Nothing in the reference set is there.

---

## 4. The channel axis

### 4.1 Per-voxel operations across channels

The core of the family: a small matrix, or a small solve, applied per voxel to
the vector of channel values.

| Operation | Where it is standard | Category | Have it? |
|---|---|---|---|
| Linear spectral unmixing (least squares against N reference spectra) | commercial packages ship it as a licensed module; Fiji's `Spectral Unmixing`; the shape is a per-voxel linear solve | **1** (reach 0), arity C in, C′ out | ~~**No.**~~ → **the shape is shipped.** This row read "expressible as a `BlockOp` reading one channel with the other C−1 as `source_inputs` and writing C′−1 side outputs — once G5 lands", and that is exactly what `ops::mixing::TupleOp` is, with `LinearMap` as the matrix kernel. What is not shipped is the *fit*: `LinearMap` applies a caller-supplied matrix, and deriving one from reference spectra is the caller's least squares, which reads no voxels |
| Stain or dye separation (Ruifrok–Johnston) | `skimage.color.rgb2hed` and its inverse; Fiji's `Colour Deconvolution`, one of its most-used plugins | **1** | **No.** Same shape as unmixing with a fixed 3×3 matrix; two independent libraries agreeing on it is good evidence it belongs |
| Crosstalk / bleed-through correction | every acquisition package; a subtractive or matrix form of the same operation | **1** | **No.** Same shape again |
| Noise-weighted unmixing (weight channels by their signal-dependent noise) | documented as a slower variant in a commercial package | **1** | **No.** A parameter on the same op, not a different one |
| Residual / goodness-of-fit output alongside the unmixed channels | a commercial package emits the largest remaining least-squares residual per voxel as an extra channel; high value = poor fit | **1**, second output | **Mechanism yes** — `side_outputs` is exactly this, declared per array with its own name, element type and rank. Worth copying: a per-voxel diagnostic beside the answer is a good habit |
| Autoscaling unmixed channels to comparable levels | same package, as a checkbox | **2† + 2†** — a whole-volume statistic, then a broadcast rescale | **No**, and it queues behind family A's global reduction. *Corrected:* this row read `5 (G1)`. Both halves run today through the `†` escape — the statistic collapses, the rescale broadcasts — so neither is blocked; both are undeclarable and unchecked (index §8) |
| Per-voxel argmax over C class-probability channels | a commercial trainable-segmentation module ships this as a named conversion in both directions | **1**, arity C | **No.** Family B names the same op from the classifier side |

**The structural note.** A matrix over channels is the strongest case in this
document for simultaneous access to N channels at one voxel, and it is
*expressible with no new axis at all* — one `BlockOp`, C source inputs at reach
0, C′ side outputs. The only thing standing in front of it is G5: the other C−1
channels cannot currently be images.

### 4.2 Channel arithmetic and ratiometric measures

| Operation | Category | Have it? |
|---|---|---|
| Add, subtract, multiply, divide, min, max, weighted linear combination of two channels | **1**, arity 2 | **No** — this is family A's largest cheap gap, seen from the channel side. `ops::background::DifferenceCombine` is the only arithmetic combine that exists |
| Ratio and index images (`(A − B) / (A + B)` and relatives) | **1** | **No.** One combine, plus a divide-by-zero policy. `ops::normalise::Removal::Divide { floor }` already shows the crate's convention for the latter |
| Free-form expression over one or two channels | **1** | **No.** A commercial package exposes exactly this — a formula field over two inputs with a per-input **channel selector**, plus a normalisation choice for when the two inputs have different element types. The channel selector is the interesting part: it is the product's admission that its channel axis has to be indexed away before arithmetic |
| Boolean union / intersection over N channels' masks | **1**, arity N | **Yes, today, and this is the useful surprise.** `LogicCombine` accepts `inputs.len() >= 2` and folds left over all of them, and `Chain::parallel` takes a `Vec<Chain>` of two or more branches. A fan-in of N `Chain::Source` leaves under one `LogicCombine` is a legal, blockable, K-ary channel union — verified by reading `Chain::apply_placed`. Blocked only by G5 |
| Masking one channel by another | **1** | **Effectively yes** via `And`, same caveat |

**The worked example, and it is a cautionary one.** A sibling application unions
the binary results of two channels. It does it by reading both whole volumes and
folding one into the other with a resident element-wise OR, in an orchestrator,
**driven beside the block grid rather than as a phase** — and its own upstream
did that step *blocked*, over a list of sources. So the port lost a
decomposition that already existed, precisely at the step where the plan's
single input became binding. The shape that would restore it — a K-ary fan-in of
source leaves — is in the crate already and unusable only because the second
channel cannot be an image.

> **Unblocked, and the example has since produced a new finding of its own.**
> The second channel can be an image, so the fan-in of `Chain::Source` leaves
> under one `LogicCombine` is buildable and the lost decomposition is
> recoverable. What the same work then measured is **what a mask costs to hold**:
> a verdict is an `f64 → f64` `MapFn` (`voxelwise::Threshold` is one), so a
> thresholded arm produces `f64`; `LogicCombine::accepts` takes only
> `Bool | F64` **and requires every branch to agree**, so one arm that cannot
> narrow binds the rest; and the image the phase writes is `f64` —
> **eight bytes a voxel for a one-bit fact.** At tile scale that is the
> difference between a stage that runs and one that does not: **57.853 → 46.28
> GiB** at the peak if the mask images were `Bool`. Registered as **G15** in the
> index, where family B's independent version of the same complaint — the
> reconstruction shell accepting `f64` only — is recorded beside it.

### 4.3 Colocalisation

The important thing about this group is not the coefficients; it is that they
are **global reductions over a pair of volumes**, not per-voxel maps. That makes
them G1 and G5 together, and it makes them a good forcing case for both.

*Corrected:* the pairing is right and the weights are not. **G5 is what blocks
them** — there is no second acquired volume — while G1 costs them a declaration
they can do without today, since the collapse to a handful of scalars runs
through the `†` escape. As a forcing case they force G5.

| Operation | Category | Have it? |
|---|---|---|
| Pearson correlation coefficient over a pair | **2† + G5** — one scalar from two volumes | **No.** The accumulators are `Σa`, `Σb`, `Σa²`, `Σb²`, `Σab` and a count: all sums, therefore associative, therefore mergeable across blocks in exactly the way `ops::tabulate`'s fixed-point columns are. *Corrected:* the row read `5 (G1)`; a `[1, 1, 1]` result is a legal `Voxels` and the collapse is expressible through the escape. **G5 is the part that actually blocks it** — the *pair* of volumes — and the decomposed form's cost is G7's barrier, not G1's declaration |
| Manders' M1/M2 overlap coefficients | **2† + G5** | **No.** Same accumulator shape, with a threshold per channel. *Corrected* as the row above |
| Costes automatic threshold (regress one channel on the other, walk the threshold down until correlation vanishes) | **5**, and worse: an *iterative* global reduction | **No.** This is family B's "scalar broadcast inside an iterative phase" with a pair of volumes instead of one — its highest-leverage missing mechanism, met from another direction |
| Costes randomisation significance test | **5**, plus randomised resampling | **No**, and probably should not be: block-shuffling a volume is not a block operation |
| Li's intensity correlation quotient, Spearman rank variants | **2† + G5** | **No.** The rank variants need a global sort, which is a different and harder reduction — and *that*, not the collapse, is what makes them hard. *Corrected* as the rows above |
| Per-voxel colocalisation *map* (a product or minimum image) | **1**, arity 2 | **No**, but this is just §4.2's arithmetic |
| Object-level colocalisation (which object in channel A overlaps which in channel B) | **1** for the overlap, then a table join | **Partly.** A sibling application implements exactly this over per-channel binary volumes and representative coordinates, resident. In framework terms it is `ops::tabulate` with a label volume from one channel and a label volume from the other — two images, reach 0 — and then a join outside. It is the *nearest* thing to expressible in the whole section |

**Weighting note.** Fiji's `Coloc 2` is the most-used implementation of the
first four rows and computes all of them in one pass with a shared statistics
gather. That is the right shape here too: one reduction phase producing a
handful of scalars, not one phase per coefficient.

### 4.4 Merge, split, composites and colour space

| Operation | Category | Have it? |
|---|---|---|
| Split a composite into per-channel volumes | trivially **1** | **Not applicable** — under the N-volume model this is the identity, and under the storage contract it is a read-time axis selection. **This is the model paying off** |
| Merge N volumes into a composite | **1**, arity N | **No**, and it is the one operation in this family that a rank-3 image cannot hold the *output* of. The honest answer is that a composite is a display artefact and belongs at the boundary |
| Build a colour image from three extraction volumes, and split it back | OpenCV `mixChannels`/`split`/`merge`; a commercial package ships both directions as named functions | **1** | **No.** Under N volumes, "merge" is the caller's tuple and "split" is free |
| RGB ↔ HSV / HLS / Lab / other colour spaces | `skimage.color`, OpenCV `cvtColor`, and a commercial package's build-and-split pair for hue/lightness/saturation | **1**, arity 3 in, 3 out | **No.** Same K-in/K-out shell as unmixing, and a good second customer for it |
| False-colour and lookup-table mapping for display | **1** | **No**, and **out of scope.** Family A already lists a LUT remap as a missing *value* operation; a LUT applied to produce display colour is a mapper's job, and VTK draws the line in the same place |
| Depth- or time-coded colour projection | **2†** for the projection *(corrected from `5 (G1)`: the spatial half is a collapse and runs today through the `†` escape)*, then display | **No.** A commercial package ships it with a revealing detail: one mode projects first and then colours the winning plane, another colours every plane and projects each colour component separately, and the two disagree. That disagreement is a display decision, not an image-processing one |

**Where the boundary sits.** A block-processing library owes the *values*; it
does not owe the *colours*. Everything above the horizontal rule of "produces a
volume of numbers" is in scope; everything that produces a picture is not. The
crate has already drawn this line once — it has a GUI and an animation module
for observing a run, and no rendering anywhere in `ops/`.

### 4.5 Per-channel everything else

Three short entries, each mostly somebody else's.

* **Per-channel parameters.** A commercial package's deconvolution exposes a
  checkbox that splits its parameter page into one tab per channel, with a
  separate point-spread function per channel validated against the channel's own
  wavelength, and its super-resolution reconstruction has the same
  adjust-per-channel switch. Under N volumes this is free and needs no
  mechanism at all: N runs, N parameter sets. Under a channel axis it is a
  parameter *vector* on every op that has a parameter. **This is a second
  independent argument for N volumes** and it is worth more than it looks —
  per-channel parameterisation is the common case, not the exception.
* **Chromatic aberration and per-channel geometric correction.** A commercial
  package ships a dedicated channel-alignment function with a transformation
  model (translation, rotation, isotropic and skew scaling, affine), a
  quality/speed setting, an interpolation rule, and — in the extended form — the
  ability to save and reload the estimated transform. Fiji's stack-alignment
  plugins are the same shape. **The transform machinery is family C's; the
  observation that is mine is that it is applied *per channel*, and that the
  estimated transform is a small object that outlives the run.** A framework that
  cannot hand a phase a caller-supplied transform per channel will end up with
  the application driving N runs, which is exactly what the N-volume model
  expects anyway.
* **Per-voxel feature stacks across channels.** A commercial trainable
  segmentation documents it plainly: all channel intensities of a voxel feed one
  feature vector, and the model records which mode it was trained in
  (single-channel with an index, or all-channel) as part of its identity.
  **Family B owns the machine-learning boundary**; the channel-axis part is that
  a K-channel feature stack is `side_outputs` today and a per-voxel classifier is
  the same K-ary reach-0 shell as unmixing. Handed over.

---

## 5. The time axis

### 5.1 Temporal filtering

| Operation | Where it is standard | Category | Have it? |
|---|---|---|---|
| Running / temporal median over a window of frames, as a background estimate | the standard background estimator for a series; ubiquitous in practice | **1** along T, or **K-ary reach 0** over W frames | **No.** As K operands it needs G5 and a median combine — and note the combine already exists in spirit: `ops::element::Rank`/`Percentile`/`select_nth` is the order-statistic machinery, currently applied over a spatial element |
| Gliding / running mean over W frames | a commercial package ships it with the output extent stated exactly: `T′ = T − W + 1` | same | **No.** The shrinking extent is the interesting part: a windowed operation along a non-spatial axis *loses* frames rather than padding, which is a different boundary convention from any in family A's table |
| Temporal derivative, first and second order (central differences along T) | same package: `out[t] = in[t+1] − in[t−1]` and `out[t] = in[t−1] + in[t+1] − 2·in[t]`, with an iterative binomial smoothing parameter and a clip-or-absolute rule for negatives | **1** along T, reach 1; **K-ary reach 0** with K = 3 | **No.** The smallest possible instance of the whole family, and the clearest one to build first |
| Temporal smoothing / low-pass along the series | everywhere | **1** along T | **No** |
| Event and transient detection over a trace | the consumer of the above | **5 (G1)** — it operates on a trace, not a volume | **No**, and see §5.5 |
| Temporal correlation, and cross-correlation between two series | a commercial package exposes correlation along X, Y, Z **and time**, plus a cross-correlation mode requiring the second input to match in dimensionality and size | **3** / **5 (G3, G5)** | **No.** `ops::fft::Correlation2` is the primitive and is 2-D and not a phase |

**The observation that ties this section to the rest of the crate.** Every row
above is a *window along an axis that does not exist*, and every row above is
also *K operands at reach zero*. The second reading is available with G5 and a
K-ary shell; the first needs G6. Since the second reading also serves the entire
channel family and the first serves nothing else, the choice is not close.

### 5.2 Correction along the series

| Operation | Category | Have it? |
|---|---|---|
| Per-channel signal decay correction over a series (fit a decay curve to a per-frame statistic, then rescale) | **2†** for the per-frame statistic, then **2†** for the map *(corrected from `5 (G1)` then `1`: the statistic collapses and the rescale broadcasts, and both run today through the `†` escape — see §2. Neither is blocked; neither can be declared)* | **No.** Fiji's decay-correction plugin offers the three usual variants — simple ratio, exponential fit, and histogram matching — and all three are "reduce per frame, fit, then map", the same two-phase shape family A names for contrast stretching |
| Illumination flicker correction frame to frame | same shape | **No.** A commercial package groups it with decay correction, bad-pixel replacement and background removal as *acquisition corrections* applied before anything else |
| Normalising a series to its own first frames | **1**, arity 2 | **No.** The same package's formula calculator has an explicit option to use only the first *n* time points of the second input for exactly this. As N volumes it is a two-operand map, blocked by G5 |
| Per-plane equalisation along an acquisition-order axis | **2** on that axis, or **2†** for the statistic *(corrected from `5 (G1)`)* | **No** — family A has the same row for the spatial case, and the same package's decay correction is applied along the *depth* axis rather than time, which is the same operation on whichever axis was acquired sequentially |
| Filling a missing frame from its predecessor, or with zero | **1**, arity 2 | **No.** Trivial, but worth listing because a real series has gaps and a library that assumes it does not will be worked around |

**The generalisation worth extracting.** Every row here is *a reduction to one
number per frame, a fit over those numbers, and a per-voxel rescale.* Only the
middle step is unusual, and it is unusual in a good way: it operates on `T`
scalars, which is not a volume at all, and which the caller can perfectly well
do. So this family needs **G1 and nothing else** — the fit belongs outside.

### 5.3 Motion estimation

| Operation | Category | Have it? |
|---|---|---|
| Dense optical flow (Farnebäck, Horn–Schunck, TV-L1) | OpenCV `calcOpticalFlowFarneback`; `skimage.registration.optical_flow_tvl1` and `optical_flow_ilk` | **4** (iterative) over **G2** (two frames) and **G2** again (coarse-to-fine over a pyramid) | **No**, and it needs both halves of G2 plus G5. The *output* is fine: a flow field is two or three volumes, one per component, which is the N-volume model again |
| Sparse feature-tracking flow (Lucas–Kanade pyramidal) | OpenCV `calcOpticalFlowPyrLK` | **1** per point, over **G2** | **No.** The output is a point table, which the crate can hold — `ops::rows` and `ops::coordinates` are the right shape |
| Frame-to-frame translation estimation for drift or jitter correction | Fiji's `Correct 3D drift` and `Linear Stack Alignment with SIFT`; a commercial package's time-alignment function, with a model chosen from translation / rotation / isotropic scaling / skew / affine | **3** / **5 (G3, G5)** for the estimate; the *transform* is family C's | **Primitives yes, phase no.** `ops::fft::Correlation2` and `SquaredDifference` compute exactly the landscape this needs and neither implements `BlockOp` — family A's §8 explains why, and every one of its three reasons applies here |
| Accumulating per-frame shifts into an absolute correction | not a volume operation at all | **Not applicable** — a prefix sum over `T` small vectors, associative, and the caller's |
| Applying the estimated correction | **family C** | — |

**The detail worth stealing from the commercial package.** Its time-alignment
function has a "third dimension" selector, and the manual warns that it **must**
be set to the depth axis when aligning stacks over time, because otherwise each
plane is aligned independently and the stack tears. That is the same class of
error the crate's `BlockConstraint` exists to prevent: an operation whose
correctness depends on an axis not being cut independently. If a temporal
alignment is ever built here, "the depth axis may not be cut" is a
`FullExtent(axis)` constraint, which family B has already asked for and which
nothing yet provides.

*Updated: half of it now exists, by refusal rather than by constraint.* An op
that declares `AxisReach::All` on that axis in **`Space::source_voxels()`**
mandates it: a block's read has to span the axis for the plan to check, so a cut
axis under a finite halo is refused. The op says what it consumes and the guard
enforces it — no constraint type was added and none is needed for correctness.
What is still missing is the planner-facing half: `Constraints` cannot be told
"do not cut axis *k*", so an enumerator proposes lattices that will be refused
rather than avoiding them. The register's G9 has been re-scoped to that.

### 5.4 Tracking — where the boundary is, and the answer

Family B left this here, and the question is real rather than rhetorical,
because the crate already has table-shaped ops (`ops::tabulate`, `ops::rows`,
`ops::adjacency`) and so cannot dismiss tracking as "not our data model".

The honest answer has three layers, and the boundary falls between the second
and the third.

**Layer 1 — the per-frame object table. The library owes this, and has it.**
`ops::detect` and `ops::tabulate` produce measured rows per connected region,
with integer and fixed-point accumulators chosen precisely so that a region cut
across blocks merges exactly. That property is not a nicety for a tracker: if a
centroid depended on how the frame was tiled, every link cost would depend on
the tiling too. Present, and better than it needs to be.

**Layer 2 — candidate link generation. The library owes this, and it is a new
op shape.** Emitting the pairs of rows within a spatial radius across one step
of the series is `ops::adjacency`'s pattern — emit the pair by its two
coordinates, never by two indices into a global list, and the merge is a sort.
But there is a wrinkle that neither family B nor I expected, and it is worth
recording: **its decomposition is spatial while its data is tabular.**
`ops::rows`'s header is emphatic that a row op decomposes by *row range with no
overlap*, because an overlap there duplicates a row and no downstream check can
tell the duplicate from a real one. A pair generator needs a *spatial*
neighbourhood, which a row range does not give it. So this is not "a row op with
a reach"; it is a genuinely new shape — rows decomposed by region — and that is
the finding, not the missing feature.

**Layer 3 — the assignment. The library does not owe this.** A bipartite
matching, a linear assignment, gap closing across skipped frames, and track
splitting and merging are a global optimisation over a graph whose vertices are
rows. The working structure is a cost matrix or a min-cost flow, and no `Reach`
describes it — it is family B's "graph cuts and random walker" row with time in
place of space. Three independent pieces of evidence say the same thing:
TrackMate's own architecture separates a detector (an image operation, with a
plugin interface) from a tracker (not one); commercial measurement subsystems
ship per-time-point index columns and a synchronised chart but **no linking, no
track identity and no lineage**; and the shape simply does not decompose.

**But the *result* is expressible today, and that is worth saying.** A track is
a `U64` track-id column and a `U64` frame column on the existing row schema,
whose position words are already the three spatial coordinates. The crate can
*hold* a track table now. It should not compute one.

So: **the library owes the tables and the candidate pairs; it does not owe the
assignment.** That is the same line family B drew, reached independently, with
one correction — layer 2 is harder than "a phase whose inputs sit at different
offsets", because its decomposition and its payload disagree about what a block
is.

### 5.5 Kymographs and per-object traces

Both **collapse the time axis**, and both were written here as instances of a
gap the crate already has. *Corrected:* the collapse is not the gap — it runs
today (§2) — and what each of these two actually needs is different. The
kymograph needs an output space that is not the input's; the per-object trace
needs G5. The rows below say which.

| Operation | Category | Have it? |
|---|---|---|
| Kymograph along a caller-supplied path | Fiji's `KymographBuilder` and the multi-kymograph family; a commercial package ships it with a width parameter averaging across the path | **X**, *and* a coordinate-space change — *and the coordinate-space change is the whole of it.* **Corrected:** this row read `5 (G1)`; a collapse is not what stops a kymograph, since a collapse runs today. Its output axes are (distance along a path, time), neither of which is an axis of the input, and no per-axis extent rule expresses that | **No.** The output axes are (distance along the path, time) — neither is a spatial axis of the input, which is family A's §8 objection to a correlation landscape applied to time. **But:** if the path is fixed before the run, a rank-2 **side output** with a `side_region` naming each block's slice of the path is buildable today, terminal, with no framework change. That is a real, small, useful thing |
| Per-object intensity-over-time trace | the standard readout of a series | **2†** per frame *(corrected from `5 (G1)`)*, then a join | **Partly.** `ops::tabulate` over the same label volume, run per frame, produces `T` tables; the trace is their join on label id, which is the caller's. `T` passes, expressible now modulo G5 |
| Projection along time (maximum, minimum, mean, standard deviation) | a commercial package projects over *any* of depth, channel, time or acquisition axis with one function, "that dimension collapsed to 1" | **2† + G5** | **No.** *Corrected from `5 (G1)`.* The spatial case is a collapse and **runs today** through the `†` escape, verified byte-identical to a whole-volume reference (`tests/collapsing_phase.rs`); family C classified it `2†` and was right, and this document's `5 (G1)` is withdrawn. Along *time* under the N-volume model it is a fold over `T` images, so what blocks it is **G5** and not G1. The generality of the commercial version is still the point — one projection operator over a named axis, not one per axis — and it is now also the shape of the fix: a per-axis extent rule naming the collapsed axis (index §9) |
| Slab projection with a start and a thickness | same package, with the reduction chosen from maximum, minimum, average, weighted average and standard deviation | **2†** with a bounded extent *(corrected from `5 (G1)`)* | **No** |
| Intensity-over-time as an input to event detection | not a volume operation | — | Out of scope once the trace exists |

**Why this section matters out of proportion to its size.** A commercial package
implements *one* projection function parameterised by which axis collapses, and
that is the shape the missing declaration should copy: not "project along Z" but
"collapse axis *a* by reduction *r*". Under the N-volume model, collapsing the
time axis is a fold over `T` images rather than over an axis.

*Corrected, and the correction sharpens the point.* The last sentence read "**G1
and G5 between them give you the time projection for free, and G1 alone gives you
the spatial one**". **G1 gives you neither, because the spatial one is already
there**: a collapsing phase plans and runs today through the `†` escape. What
G5 gives you is the time projection, by making `T` acquired volumes into images.
What G1 gives you is what the commercial package's parameterisation is made of —
a *named collapsed axis*, stated once, from which the per-block fetch follows.
That is worth having for the reason this section already gives, and it is a
declaration rather than a capability.

---

## 6. Not expressible today, and the smallest change each needs

Consolidated. Each entry names the gap and the smallest change that closes it.

| Operation | Why not | Smallest framework change |
|---|---|---|
| **Anything reading two acquired volumes** — unmixing, stain separation, crosstalk correction, channel arithmetic, ratio images, colocalisation, two-frame temporal anything, a supplied shading reference, a per-channel mask | ~~**G5**~~ → **nothing. G5 has closed.** *(The row read: a run is seeded with one array, `ArrayEnvironment::new` / `ZarrEnvironment::create`, and every other image is written by a phase.)* What is left is a residue and it is **G2's**: a supplied array must be in image 0's coordinate space, so one that has been rebinned or resampled cannot be read directly — refused by name at plan time | ~~Image numbering in which images `0..k` are inputs and phase `p` writes image `k + p`~~ — **that half was impossible, twice over, and the index's §10 keeps both objections.** What shipped is `ImageId::supplied(i)` in a disjoint high range plus `ArrayEnvironment::with_inputs` and `ZarrEnvironment::create_with_inputs`. The rest of this cell was right and is why it was cheap: `Chain::Source`, `SourceInput`, `check_source_images`, image lifetimes and the byte accounting already worked for images above zero. **It was the highest-leverage item in this document and it is done** |
| **Per-voxel matrix over C channels producing C′ channels** (unmixing, stain separation, colour-space conversion, per-voxel classification, argmax) | ~~Needs G5 for the inputs; the outputs are fine~~ → **nothing.** Both halves are in the tree | **Built: `ops::mixing`.** The shell (`TupleOp`), the kernel trait (`TupleKernel`) and the matrix kernel (`LinearMap`), in exactly the shape this row specified — one `BlockOp`, C−1 `source_inputs`, C′−1 `side_outputs`, no new axis, and not a `Combine` for the reason given. **One clause of the row was wrong and has since been fixed:** `BlockOp::apply_side` was not handed the `SourceInputs`, so the outputs beside the primary were computed in `apply_with` and carried across in a per-block map. The argument is now threaded from the executor and the map is deleted, so the row's shape is true as written; the price is that the inputs are streamed twice, **1.08–1.10×** at C = C′ = 16 |
| **N-channel Boolean union / intersection** | ~~Only G5~~ → **nothing structural.** `LogicCombine` is already K-ary, `Chain::parallel` already takes N branches, and the branches can now be supplied images | The cost that remains is **G15**, not a gap in the shape: the fan-in is bound to `f64` because a verdict is an `f64 → f64` `MapFn` and `accepts` requires every branch to agree, so a mask image costs eight bytes a voxel — measured at **57.853 → 46.28 GiB** of peak on the stage that does it. See §4.2 |
| **Colocalisation coefficients** (Pearson, Manders, Costes) | **G5.** A handful of scalars from a **pair** of volumes. *Corrected from `G1 + G5`:* the collapse to `[1, 1, 1]` runs today through the `†` escape, so G1 costs the declaration and not the result; the pair is what blocks it | G5. Two further routes for the result itself, both available now: the `†` escape as an image a later phase can read, or a low-rank **side output**, terminal. The accumulators are all sums and therefore associative; follow `ops::tabulate`'s fixed-point convention so a region cut across blocks merges exactly rather than approximately |
| **Costes automatic threshold** | G1 + G5 *inside a loop* | Family B's "scalar broadcast inside an iterative phase", with a pair of volumes. Nothing smaller works |
| **Windowed temporal filters** (running median, gliding average, temporal derivative, temporal smoothing) | **G5**, then arity | The K-ary shell again, with `W` operands. Note the boundary rule these need and no existing op has: a window along a series **shrinks the extent** (`T′ = T − W + 1`) rather than padding |
| **Per-frame or per-plane statistic for decay and flicker correction** | **Nothing — it works today at `2†`.** *Corrected from `G1`.* The statistic collapses two axes and the rescale that reads it back broadcasts one; both run through the `†` escape, both are hand-written, and neither is checked (index G14) | Nothing, to build it. To *declare* it: G1's per-axis extent rule, `Whole` for the statistic and `Fixed(1)` for the rescale. A low-rank side output is still the cheaper route where the result is terminal. The *fit* over the `T` scalars is the caller's and should stay there |
| **Time projection over `T` volumes** | **G5.** *Corrected from `G1 + G5`:* the spatial projection runs today (`tests/collapsing_phase.rs`), so under N volumes what is missing is the `T` images | G5, and then it is a fold. A projection operator should still be parameterised by *which axis collapses*, not written once per axis — that parameterisation is what G1's declaration would give it |
| **Kymograph** | **An output space that is not the input space.** *Corrected:* the row read "**G1**, plus an output space that is not the input space", and G1 is not the part that bites — a collapse runs today. Its axes are (distance along a path, time), neither of which is an axis of the input | For a **fixed** path: nothing — a rank-2 side output with a `side_region` per block works today, terminal. For a path chosen from the data: a coordinate mapping, which is a bigger thing than a collapsed axis and is not G1 |
| **Frame-to-frame drift estimation as a phase** | **G2 + G3 + G5** — two inputs at different offsets, a spectrum with nowhere to live, and a second volume | All three. Family A's `ops::fft` header already states the first two precisely. The crate has the primitives and no way to make them a phase |
| **Dense optical flow** | **G2** twice — two frames, and coarse-to-fine across resolutions — plus **G5** and category 4 | G2 in its general form (a `SourceInput` carrying a rational scale and offset per axis, in `ops::resample`'s spirit). This is the most expensive entry here and the least urgent |
| **Candidate link generation across frames** | Its decomposition is spatial and its payload is tabular; `ops::rows` decomposes by row range with **no overlap**, on correctness grounds | A fragment op reading two row streams with a *spatial* neighbourhood — a new op shape, not a parameter. See §5.4 |
| **Track linking, gap closing, splits and merges** | A global optimisation over a graph. No `Reach` describes it | **Nothing.** Out of scope, argued in §5.4. The crate should produce the tables and stop |
| **A composite / colour image as a plan image** | A plan image is one element type; a composite is a tuple | **Nothing.** Out of scope — it is a display artefact. VTK and a commercial package both put it on the far side of the same line |
| **An axis the planner may not cut** | `BlockConstraint::Extent` mandates all three block extents or none | `FullExtent(axis)`, which family B already lists. Needed by any temporal alignment that must not tear a stack, and needed *immediately* by any fourth axis — which is the argument against the fourth axis |

---

## 7. Present, and more useful than the name suggests

Checked against the code, in the spirit of the other three documents' "narrower
than the name suggests" tables — except that in this family the surprises run
the other way.

| Item | What a reader assumes | What it actually is |
|---|---|---|
| `ops::voxelwise::LogicCombine` | A binary connective | **K-ary.** `accepts` requires `inputs.len() >= 2` and `apply` folds left over all of them, allocating nothing in the two-branch case. A three- or five-channel union is one node |
| `Chain::parallel` | A two-branch diamond | **N branches**, `Vec<Chain>`, minimum two and checked. With `Chain::Source` leaves as branches — legal, verified in `apply_placed` — this is a K-ary multi-image combine |
| `BlockOp::side_outputs` | A debug hook | **The K-out half of a K-in/K-out operation.** Each array has its own name, element type and **rank**, and `side_region` maps a block's slice into that array's own coordinate space, checked to tile exactly. This is where unmixed channels, a residual map and a kymograph all go |
| `BlockOp::source_inputs` | More inputs | **More images**, in the same coordinate frame, read at the block's own fetch region — `check_source_images` refuses a source reach wider than the phase's halo, by name. *Corrected: it read "more images **of the same plan** … not a second acquisition".* Since G5 landed a named image may be one the run was **handed**, addressed by `ImageId::supplied(i)`; what has not changed is the coordinate frame, which is still image 0's and still checked. It carries a `dtype` too, and for a supplied image that declaration is **required** — no phase writes one, so the readers are the only statement of what it holds |
| `ops::element::Rank` / `Percentile` / `select_nth` | Spatial rank filtering | The order-statistic machinery, independent of what the population is. A temporal median over `W` operands is the same `select_nth` over a different population |
| `crate::table::Table` | Region measurements | Rows whose **position is three words** and whose payload is `U64`/`F64` columns. A frame index and a track id are both `U64`. The crate can hold a track table today |
| `ops::adjacency` | A region adjacency graph | Every adjacent pair of set voxels as rows carrying **two coordinates**. The pattern a cross-frame candidate generator should copy — the pair, not two indices |
| `ops::tabulate` | Per-label statistics | A reduction over **two images** — labels and values — with associative fixed-point accumulators. The nearest thing in the crate to a per-object measurement across a channel boundary, and the template for a colocalisation reduction |

---

## 8. Handed to other families

One sentence each, so nobody goes looking.

* **Arithmetic combines** between two volumes — **family A**, §2. I need them for
  ratios and channel arithmetic; they are the same op.
* **Global reductions and the two-phase reduce-then-map shape** — **family A**,
  §3 and G1. Decay correction, autoscaling and colocalisation all queue behind
  the same mechanism.
* **All spatial filtering**, including the filter banks that make a feature
  stack's channels — **family A**.
* **The frequency-domain primitives** and the complex element type — **family
  A**, §8 and G3. Temporal and cross-correlation need both.
* **Segmentation, morphology, per-object measurement, and the machine-learning
  boundary** — **family B**. A per-voxel classifier over a channel stack is the
  K-ary shell (mine) feeding B's chain (theirs).
* **The transforms themselves** — resampling, interpolation conventions,
  registration, the estimated transform applied — **family C**. Mine is only the
  observation that they are estimated *frame to frame* and applied *per channel*.
* **Stitching, tiling and multi-view fusion** — **family C**, even where a
  commercial package folds view and illumination axes into channels, which is a
  container operation and not an image one.
* **Rendering, colour and display** — nobody's. Out of scope, argued in §4.4.

---

## 9. If only three things were built

Weighted by how many entries above they unblock.

1. **More than one input image (G5).** Two constructors and an image-numbering
   convention. It unblocks the entire channel family — unmixing, stain
   separation, crosstalk, arithmetic, ratios, colocalisation, masking against a
   supplied reference — plus every two-frame temporal operation, plus the N-way
   Boolean union that already has all its other parts. Nothing else in this
   document unblocks a fraction as much, and it is not a fourth axis.

   > **Done.** The two constructors are `ArrayEnvironment::with_inputs` and
   > `ZarrEnvironment::create_with_inputs`. The "image-numbering convention" was
   > the wrong half of the ask and **could not have been built** — §3.6 and the
   > index's §10 keep both objections to it, because either one would have been
   > found the hard way.
2. **A K-ary reach-0 op shell**, as one `BlockOp` with `source_inputs` and
   `side_outputs`. It is unmixing, stain separation, crosstalk correction,
   colour-space conversion, per-voxel classification, channel argmax, and every
   windowed temporal filter, all at once. Family B asks for it independently,
   which is the usual sign.

   > **Done, as `ops::mixing`**, in the shape asked for. One clause of that shape
   > was wrong — `apply_side` was not handed the source inputs — and **it has
   > since been fixed too** (§3.5's entry 3), so nothing this item asked for is
   > outstanding. **The ranking was right and is the thing worth keeping from
   > this list:** the two items this document put first are the two that landed,
   > in this order, and the second needed the first.
3. **A rank-reducing result (G1)**, starting with the terminal side-output form
   that already works. It is every colocalisation coefficient, every per-frame
   statistic for decay and flicker correction, every projection along any axis,
   and — for a fixed path — the kymograph. Start with the accumulator-shaped
   reductions, because they are associative and `ops::tabulate` has already
   shown how to merge them exactly.

   *Corrected, and it demotes this item.* There is no rank to reduce: a
   `[1, Y, X]` projection and a `[Z, 1, 1]` per-frame statistic are ordinary
   `Voxels` and both **run today**, through the `†` escape, hand-written
   (`tests/collapsing_phase.rs`). So none of the four things listed is waiting
   on a framework change to be *built*. What G1 — under its corrected name, "the
   geometry cannot declare a collapsed or a broadcast axis" — would buy is that
   they can be **stated**, planned automatically, and checked, the last of which
   matters most: nothing today compares a stated fetch against the dependency it
   stands in for, so a projection reading one plane instead of the axis is
   accepted and wrong at every position (index G14). Items 1 and 2 of this list
   are unaffected and remain in that order.

The fourth axis is not on this list, and after writing the rest of the document
that is the conclusion I am most confident of. Every operation this family
surveys is either reach zero along the non-spatial axis — in which case it wants
*arity*, not an axis — or is not a voxel operation at all. The axis would be
paid for by every op in the crate and used by one small family of temporal
filters that K operands already serve.

---

*Where this document says "no", it means "not found in `src/` on a read of the
module in question". Where it says **unverified**, it means exactly that.
Nothing here was inferred from a module name.*
