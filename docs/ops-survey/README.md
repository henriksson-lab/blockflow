<!-- SPDX-License-Identifier: MIT -->

# Ops survey — index and consolidated gap register

Four documents, written in parallel from a shared brief, survey what a general
image-processing library is expected to support and measure `blockflow` against
it. This file is the map over them and the one place their findings are
reconciled. It restates none of their content; it makes them navigable, fixes
the numbering they diverge on, and consolidates every structural gap into one
register.

| | document | covers |
|---|---|---|
| **A** | [`filtering-and-transforms.md`](filtering-and-transforms.md) | point operations, histogram and intensity statistics, linear and non-linear filtering, multi-scale and derivatives, frequency domain, restoration |
| **B** | [`segmentation-and-measurement.md`](segmentation-and-measurement.md) | thresholding, binary and grey morphology, labelling and connectivity, distance, region segmentation, object measurement, topology and skeletons, the machine-learning boundary |
| **C** | [`geometry-and-registration.md`](geometry-and-registration.md) | geometric transforms, resampling and coordinate conventions, registration primitives, mosaicking, projections and reslicing, surfaces |
| **D** | [`channels-and-time.md`](channels-and-time.md) | the channel axis, the time axis, tracking, and whether the crate needs a fourth axis at all |

---

## 1. What this survey is, and what it is not

It **is** a map of what a general image-processing library is expected to
support, measured against what this crate has, with every row classified by
**whether the decomposition model can express it and at what cost**. That last
column is the part with the long life: a feature list ages out in a release,
but what `Reach`, `Decomposition` and `CostModel` can and cannot say does not.

It is **not** a roadmap, **not** a commitment, and **not** a feature checklist.
Nothing surveyed here is promised and most of it should never be built. Where a
document says "if only three things were built", it is ranking by how much each
unblocks, not asking for three things.

**Sources.** ImageJ/Fiji weighted by what is actually driven rather than by
what exists in the menus, ITK, VTK, OpenCV and scikit-image. Where two
independent libraries expose the same primitive, that agreement was treated as
evidence the primitive belongs in a general library — which is the only defence
against surveying one library's habits and calling them a field.

**Where "no" and "unverified" mean what they say.** All four documents end with
the same declaration: "no" means not found on a read of the module in question,
"unverified" means exactly that, and nothing was inferred from a module name.
Claims this index could not check are marked unverified below rather than
smoothed away.

---

## 2. The shared classification, stated once

The four documents use compatible but not identical numbering. This is where
that is fixed. **The scheme below is the one to read them in**; the documents
themselves were not edited, because they are other workers' files.

> **Corrected — this sentence stopped being true and two later passes made it
> so.** The four *are* edited now, in place, by every correction pass since:
> §5's and §6's "corrected in place" lines mean exactly that, and this pass has
> touched all four again for G5 and G10. The convention that replaced the
> original one is the one these documents actually hold to — **a correction is
> written beside what it corrects and neither is deleted** — and it is what
> makes editing another worker's file safe. The numbering scheme below is
> unchanged either way.

### The scheme

Categories **1–5** are the shapes the framework can express, each earned by
something the crate has actually built. **P** is a different kind of work, not a
rung on the same ladder, so it keeps C's letter. **X** is the absence of a shape
rather than a shape, so it is *not* a number — which is precisely the collision
A and B ran into, and removing the number removes it permanently.

| | category | what it means | the crate's worked example |
|---|---|---|---|
| **1** | **Bounded reach** | a halo the op declares — `AxisReach::Bounded { lo, hi }`, possibly asymmetric, possibly `PerBlock`. Cheap, blockable, the ordinary case | `ops::rank`, `ops::smooth`, `ops::morphology`, `ops::resample` |
| **2** | **Whole-axis reach** | `AxisReach::All` on one axis, bounded on the others. A separable sweep; the other two axes stay cuttable | **none in `src/ops/`** — and §8 says why the four documents' reading of that emptiness was wrong |
| **3** | **Whole-volume, resident** | no halo bounds it. `Reach::all()` makes the phase a planning barrier with one block | `ops::watershed`; `ops::fft` is the pure case and declines to be an op at all |
| **4** | **Iterative to a fixed point** | a loop of substages whose stopping rule is a whole-volume reduction. The phase's *external* reach is one substage's, however many run | `crate::iterate`, `ops::reconstruct`, `ops::configuration`'s fixed point |
| **5** | **Fragment-and-join** | a per-block partial result merged globally. Anything with a global connectivity requirement | `ops::fill`, `ops::regional`, `ops::detect` — one program in `ops::components` |
| **P** | **Resident but cheap** | point arithmetic over a coordinate list. Decomposes by *row range* at reach exactly zero, where a halo would be a **defect** rather than a cost — a duplicated row is emitted twice | `ops::rows` |
| **X** | **Not expressible today** | not a category; the absence of one. Every instance names a gap from §3 | — |

> **Corrected — measured, then overtaken.** *What all four documents said:* the
> category-2 variant "exists with no user", which each read as an unwritten op.
> *What the experiment measured:* `AxisReach::All` **in `Frame::Source`** was
> refused unconditionally — no halo satisfied it at any extent on any grid — so
> that pairing had no user because it could not have had one. *What §8.5 then
> changed:* it is now granted on an axis the op **consumes**, and refused only
> where the axis is cut without a whole-axis halo. So the source-frame pairing is
> plannable today, is the **recommended** way to state a whole-axis dependency,
> and is the only one of the three that is checked.
>
> `AxisReach::All` in the **phase** frame remains a different thing: it plans,
> it has a user outside this crate (§8.4), and on an axis of extent 1 it is
> accepted without being a statement of anything (`reach.rs:322`). **Two of the
> three declarations that plan are wrong in different ways, so a document that
> says "declare `All`" must say in which frame.** Category 2 is real; the
> emptiness was never evidence of neglect. See §8.

> **Corrected — one worked example moved a category, and the cell it moved out of
> is kept.** *Category 3's example reads "`ops::watershed`; `ops::fft` is the pure
> case and declines to be an op at all."* That is still true **of `ops::fft`**,
> whose three types really cannot be a phase — but it was read as saying the
> *frequency domain* is category 3, and that reading is now false in the tree.
> `ops::convolve::TransformConvolveOp` computes a linear filter through the same
> transform and is **category 1**: an `AxisReach::Bounded` halo the op declares,
> cheap, blockable, and byte-identical across genuinely distinct lattices. The
> move is not a re-classification of one op, it is the general shape — an
> operation whose *answer* is real does not inherit its intermediate's reach, so a
> transform used **inside** a bounded stencil is bounded. What remains category 3
> is a transform whose *output* is the spectrum, and no operation this survey
> lists wants one. §12.4's G3 entry carries the argument.
>
> The price of the move is not zero and it is not paid where a reader would
> guess: the halo is the kernel's **plus the tile's alignment slack**, because the
> op's tile grid is anchored to the volume and a block cannot know where its tiles
> begin. That is G9, and it is the one row this work leaves larger than it found
> it.

### The cost modifier, carried unchanged from C

> **†  cross-grid, hand-built phase.** An op whose output grid is not its input
> grid states its fetch per block in `BlockGeometry::source`, declares
> `Reach::none()` in `Space::source_index` — a *marked* zero, with the real
> dependency carried by the fetch region — and ships its own plan builder beside
> itself. `ops::resample`, `ops::lattice` and `ops::adjacency` all do this.
>
> The price should be stated every time the mark is used: **no automatic
> planner can produce such a phase.** `Trivial`, `Enumerating`, `Greedy` and
> `Materialising` all plan one volume per group. `1†` is genuinely dearer than
> a bare `1`.
>
> **Updated after §8.5, and it changes what to write, not what to pay.** For a
> **whole-axis** dependency the `Reach::none()` escape is now *strictly weaker*
> than saying so: `AxisReach::All` in `Space::source_voxels()` plus
> `with_sources` states the same fetch and is **checked against it**, where the
> escape's marked zero is a claim nothing can fail to meet. Measured: a
> half-axis fetch is accepted under the escape and refused under the
> declaration. So the mark now covers two different things — a cross-grid fetch
> that is an *affine map* (`ops::resample`, `ops::lattice`, `ops::adjacency`),
> where the escape remains the only way to say it and remains unchecked, and a
> cross-grid fetch that is a *whole axis*, where the escape is now the wrong
> choice. **The `†` price is unchanged in both**: the fetch is still per block,
> so the builder is still hand-written.

### Reading each document's local numbering

| in document | local | read as | note |
|---|---|---|---|
| A, B, C, D | 1, 2, 3 | **1, 2, 3** | identical in all four |
| A, B, D | 4 | **4** | identical |
| C | 4 | — | **no instance in family C.** A registration search is iterative, but C puts optimisation outside the library, so the row is empty rather than absent |
| B | 5 | **5** | fragment-and-join. B is the only document with instances of it as a *category*; C uses the shape twice (the pairwise-estimate fold, the vertex weld) without numbering it |
| A, C, D | — | **5** | none of the three numbered fragment-and-join. Where they describe a per-block partial merged globally, it is category 5 |
| A, C, D | 5 | **X** | "not expressible today" |
| B | 6 | **X** | same |
| C | P | **P** | kept as a letter |
| C | † | **†** | kept, and applicable across the scheme |

Two consequences worth carrying: **A's "5" and B's "5" are different things**,
and C's "5" follows A. Anywhere the four are read together, `X` disambiguates.

---

## 3. The consolidated gap register

Every structural gap the four found, in one table. **G1–G6 and C1–C3 are the
identifiers the documents already use and are preserved unchanged.** G7–G21 are
allocated here, in the shared `G` series, for gaps the documents named in prose
but did not number — this index is the register of record for that series.
**G1 keeps its identifier and loses its name**: §8.1 says why the old one was
wrong. **G14 was minted here**, by the correction pass of §8, and **G15 and G16
are minted here** by this one; those three are the members of the series no
document named at all. **G17, G18 and G19 have a different provenance again, and
it is worth distinguishing**: they were minted by *work* rather than by a
reading — G17 when a consumer's third copy of a missing producer turned up in
this crate's own test suite, G18 and G19 by the sweep that observation prompted,
and **G21 by the sweep that closed that sweep's own stated blind spot** (§3.2).
No amount of re-reading `src/ops/` would have produced any of the four, which is
the finding §3.1 is about.

**Two rows are now closed and are kept in the table rather than struck.** G5 is
closed outright and G10 is closed but for one clause; both rows carry the
prescription they shipped *against*, because in each case the prescription was
wrong in a way worth more than the outcome. A register that recorded only what
landed would be a register that could make the same mistake twice.

> **Updated by §12's sweep — four rows now, and one of them is a third state.**
> G15 and G16 are closed too, and each carries a clause the shipping mechanism
> contradicted, so the sentence above holds for all four. **G7 is neither open
> nor closed and the table now says so**: the barrier and the hoisted reduction
> are built, and *no op in `src/ops/` declares either*, so the price the row
> measures is still being paid in full. The word for that is **buildable**, and
> it is deliberately not "built" — see §12.1. A register that marks a row closed
> on the arrival of a mechanism, rather than on its adoption, reads as an
> inventory of what is present, which is the one thing §1 says this is not.

**What is minted and what is a note on an existing row.** The line this pass
drew, since it is the register of record and the choice is otherwise arbitrary:
a **new identifier** for a subject no row covers and nothing is currently
changing — G15 and G16 are both — and a **note on the row** for a defect in a
clause of a row whose shape has already shipped and whose fix is in flight. The
`apply_side` defect under G10 was the second kind: an identifier minted for it
would have been closed before any document could cite it, and the five documents
cite this series by number. **That is no longer a prediction** — the fix landed
inside the same pass that recorded it, so a `G17` would have been born and closed
without a single citation. §11.3 states the rule in the place it was applied, and
now also what it caught.

**G17 is minted here and closed here, and the precedent is G16.** The line
above says a new identifier is for a subject no row covers *and nothing is
currently changing*, and §11.3 records why: an identifier born and closed inside
one pass is one no document can cite. G16 was minted and closed in one pass
anyway, on the argument that a closed row carrying a wrong prescription is worth
more than no row, and G17 is the same case for a different reason — the gap was
**old and unnumbered**, paid three times over three passes before anyone closed
it, and what it leaves behind is a fact about this survey rather than about the
crate.

**That fact: the survey could not see this gap, because the crate's own test
suite was paying it.** Every other row here was found by reading `src/ops/`.
This one's three copies were in two consumer stages and in
`blockflow/tests/rows_group.rs` — the acceptance suite for the very ops the
missing producer feeds — and a survey of operations does not read a test
directory. The copy inside this crate is the sharpest evidence in the row and it
is the one no pass of this document was positioned to find. **A test that has to
supply the missing half of the module it is testing is a gap report**, and
nothing here was reading them.

"Blocks" counts **how many of the four families the gap blocks**, which is the
ranking no single document can see.

| id | gap | what it blocks | smallest framework change | blocks | raised by |
|---|---|---|---|---|---|
| **G1** | **the geometry cannot declare a pinned (broadcast) axis** *(renamed twice, both kept: "no rank-reducing phase" originally, then "a collapsed **or** a broadcast axis" by this survey's correction pass, then narrowed to this because the **collapsed half landed** — §8.1, §8.5)* | the **declaration**, not the result — and now one side of it only. **Closed:** an op that consumes an axis whole and writes it at extent 1 can declare `AxisReach::All` in `Space::source_voxels()`, and the declaration is enforced against its fetch. That covers projections, slab projections, extended depth of focus, per-plane and per-frame statistics, and the collapsing half of every reduce-then-map workflow. **Open:** every op that holds a *source* axis at extent 1 while its output grows — the map half of contrast stretching, equalisation, decay and flicker correction, autoscaling. Those run today through the `†` escape and cannot be stated | a per-axis extent rule on the input map, `Fixed(1)` being the pin. §9. Not `output_shape` and not `Voxels`: `[X, Y, 1]` is already a legal `Array3` and needs nothing. `InputMap::Affine` provably cannot express the pin (§8.2) | **4** (A B C D) for the open half — re-derived in §8.2, and note the *kind* changed twice: from "cannot be built" to "cannot be said" to "half of it can now be said" | A |
| **G2** | no phase whose inputs sit at different offsets or resolutions | mosaicking at both ends, montage, multi-view fusion, drift correction, coarse-to-fine pyramids, Laplacian-pyramid reconstruction, dense optical flow. **And, since G5 landed, one more instance that is this row's and not G5's:** a supplied input is in **image 0's coordinate space**, by a stated rule rather than a recorded one, so a phase downstream of a reshape cannot read one. `check_source_images` compares the rule against the reading phase's own volume and refuses the pair **by name** at plan time, which is the right failure — the alternative is a block fetching the wrong region of the right array and producing a well-formed wrong volume | `BlockGeometry::source` becomes one region **per source image**; `SourceInput` carries a rational scale and offset per axis rather than only a `Reach`. `Placement::sources` already carries `Vec<(usize, Anchor)>` on the execution side | **3** (A C D) — **re-derived after G5, and unchanged.** The new instance is reachable only by a family that both reshapes and reads a supplied array: **A** (a supplied flat-field reference read after a resample), **C** (tiles read after any rescale — §6's stage 3 at two resolutions), **D** (a per-channel reference after a rebin). **B** names no supplied-array operation at all, before or after a reshape, so it is not added | A |
| **G3** | ~~no complex element variant~~ → **still open, and now demonstrated from inside the crate.** `ops::fft` ships a real 2-D transform whose spectrum is an `Array2<Complex<f64>>`, and it is a **library and not an op** — there is no `Dtype::Complex*` and no `Voxels` variant for it to write, so it cannot be a phase. §2's table already called it "the pure case"; what is new is that the case is now in-tree rather than hypothetical. §12.4 → **refused — the prescription was tried against the code and is not the change; the capability it was raised for is built without it.** The word is deliberately not *closed* and not *buildable*: nothing complex shipped, and a row marked closed would read as though something had. The absence is still real; what is new is that it is now **deliberate and argued** rather than pending. `ops::convolve::TransformConvolveOp` ships — a linear filter computed through the Fourier transform, an ordinary `BlockOp` with an ordinary **bounded** reach, writing `f64` voxels, byte-identical against a whole-volume reference across block extents asserted to be genuinely distinct — and it needs no `Dtype::Complex*` because the spectrum never leaves the inside of one `apply`. `ops::fft` gained `RealTransform3`, the third axis `ops::deconvolve`'s table records as missing, so the transform is no longer two-dimensional either. **The element type was never what stood in the way.** `ops::fft`'s own header gives three reasons it declines to be an op and the dtype is the *third*; the first two — two inputs of different extents, and an output indexed by **lag** rather than by voxel — are G2's and a coordinate-system change, and neither moves if a complex variant is added. **Three findings against building it, each read off the code rather than argued.** *(i) `Voxels` is `f64`-scalar-shaped at the root, not merely enum-shaped.* `VoxelElement` requires `into_f64` and `from_f64`; `Voxels::filled` takes an `f64`; `uniform() -> Option<f64>` is what feeds `BlockOp::constant_maps_to`, a **short circuit**; `widened() -> Array3<f64>` is what every shell that accumulates calls; `unwritten` is a `NaN` or a `MAX`. A complex element satisfies none of them — it can only project to a real and lie to a short circuit about what the block holds. What *is* buildable is a `Dtype::Complex*` with **no** `Voxels` variant, which is exactly what `Dtype::F16` already is and what `voxels.rs` refuses by name: a byte-width tag for storage that no block can hold. That is small and real and it is **not** what this row asks for. *(ii) The blast radius is silent rather than loud, and it was measured rather than counted by eye:* a `Dtype::Complex128` was added to a copy of the tree and `cargo check --all-features --lib --tests` run against it. **23 distinct exhaustive `match` sites in 14 files** — 29 compiler errors in the lib and 30 with its own tests, because the macro-expanded ones report more than once — `npy.rs` 5, `zarr_env.rs` 4, `ops/convolve.rs` and `ops/rank.rs` 2 each, and one apiece in `voxels.rs`, `strategy.rs`, `cache_tests.rs` and `ops/{deconvolve,label,local,resample,ridge,smooth,voxelize,voxelwise}.rs`. Those are the loud half and they are the *cheap* half. The hazard is `accepts`, because most of them are written as **denials** — `dtype != Dtype::F16`, or a `!matches!` of a short list — at twelve sites in nine files: `ops/resample.rs` and `ops/voxelwise.rs` and `ops/rank.rs` twice each, and `probes.rs`, `ops/regional.rs`, `ops/smooth.rs`, `ops/convolve.rs`, `ops/deconvolve.rs` and `ops/ridge.rs` once each. Every one of those would say **yes** to a complex block at plan time and fail in the executor, which is the exact failure `accepts` exists to remove. `TransformConvolveOp` writes its own as a list for that reason and says so. *(iii) It unblocks nothing on its own list.* A frequency-domain filter, a band-pass, a stripe removal, a Wiener or regularised inverse and a transform convolution are each **one** op that transforms, modifies and inverse-transforms internally — `ops::deconvolve`'s table said so before this pass and it was right. Exposing the spectrum as a plan image is strictly *worse*: its reach is `All` on every axis in both directions, so the phases either side are single-block whatever the dtype, and the image costs sixteen bytes a voxel to materialise a value only the next phase can read. **§12.4's G3 paragraph predates this** and its closing advice — "take it when a concrete op forces it, and nothing yet has" — is superseded here: a concrete op came, and it did not force it. That paragraph is not this pass's to rewrite. `src/ops/convolve.rs`, `src/ops/fft.rs`, `tests/transform_convolution.rs` | a spectrum as a plan image, and therefore every frequency-domain filter, band-pass and stripe removal, Wiener and regularised-inverse deconvolution, phase correlation *as a phase*, transform-based convolution, temporal cross-correlation → **re-derived, and it is nothing this row can close.** Frequency-domain filtering, band-pass, stripe removal, Wiener and regularised-inverse deconvolution and transform-based convolution are **buildable now**, each as one op, and one of them is built. Phase correlation *as a phase* and temporal cross-correlation are the two entries a complex dtype was never going to reach: both need two inputs of different extents and an output indexed by **lag** rather than by voxel, which is G2 plus a coordinate-system change. That residue is recorded here rather than added to G2's row, which is not this pass's to edit | `Dtype::Complex64`/`Complex128` with matching `Voxels` variants, plus a decision about the half-spectrum layout — which is `[rows, cols/2 + 1]`, an extent A read as touching G1's territory and which, after §8, is an ordinary shape change and touches nothing. The largest of the shared four; take it when a concrete op forces it → **withdrawn, and what replaces it belongs to another row.** The `Voxels` variants and the half-spectrum layout buy nothing above. What an FFT-family op actually pays is a **halo**, and the framework has nowhere to discount it: overlap-save anchors its tile grid to the volume — which is what makes it byte-identical across lattices, and the tile is asserted to be *in* the arithmetic so the invariance is not vacuous — so a block cannot know where its tiles begin and must declare `tile - 1 + lo` below and `tile - 1 + hi` above, **paid even when the plan's blocks are a whole number of tiles**, which is the common case. `BlockConstraint::Extent` mandates all three extents and gives up the search; `Constraints` has no per-axis rule at all. **That is G9, reached from a new direction**, and it is the change worth making. **It was made, in the same pass and by the same worker: `AxisReach::Aligned` ships, and G9's row carries the measurement, the blast radius and the reason it is a reach rather than a constraint.** The halo this row calls the real price is now discounted on any lattice whose block edge is a whole number of tiles, which is three of the coarse ladder's four rungs. Two adjacent rows are *not* binding here and this pass says so rather than leaving it to be assumed: **G4** is not, because the `n log n` is priceable when `n` is the **tile** rather than the volume — `TransformConvolveOp::cost_per_voxel` is a constant like every other coefficient, and what is missing is a search that would *compare* it against the direct op, not a coefficient that could express it; and **G1** is not, because no spectrum is ever an image | **3** (A C D) → **0**, with the residue re-derived and **moved rather than dropped**: **A**'s frequency-domain family is buildable and one member is built; **C**'s phase correlation and **D**'s temporal cross-correlation are still blocked, by **G2**, and belong in that count and not this one | A |
| **G4** | ~~the cost model cannot price `log n`~~ → **still open, and the mispriced operation is now in-tree.** `PhaseTraffic` landed and made the price see a *second array*; it did not make it see a *superlinear* one, and `ops::fft` is now the concrete `n log n` the row was written about. §12.4 | nothing outright — it **misprices**. The planner cannot choose between a direct convolution and a transform-based one, and every plan containing a registration landscape is wrong in a known direction. **And the misprice now moves plans, where it used to be nearly inert.** Under the old objective — the phase's serial work, `cost_per_block × n_blocks` — a wrong per-voxel cost changed no decision: `n_blocks` cancels, redundancy falls monotonically, and the sweep answered "the largest candidate that fits" whatever the coefficient was. The objective is now `max(pool, channel)` and the block edge is a real per-phase choice, so a phase priced linear that is really `n log n` is priced wrong **in the pool bound**, which is the half that decides how finely to cut. §11.2 | a `cost_per_voxel_in_volume(volume)` beside the existing pair, additive with a default that preserves every current declaration. *Unverified:* whether `crate::statistics`' one-denominator-per-op calibration tolerates a second | **2** (A C) — unchanged; the objective change alters what the misprice *costs*, not who has an operation that is mispriced | A |
| **G5** | ~~there is no second input volume~~ → **closed.** A run can be handed `k` arrays beside image 0, and they are images in every sense the plan already had: read through `source_images`, fetched through `Environment::read`, priced by the same byte accounting *(§10 — and the mechanism this row prescribed turned out to be unbuildable, which is recorded there rather than quietly replaced)* | the entire channel family — unmixing, stain separation, crosstalk correction, channel arithmetic, ratio images, colocalisation, N-channel Boolean union; every two-frame temporal operation; a **supplied** shading or flat-field reference; and, though C did not name it, getting N mosaic tiles into one run at all. **All of it is now reachable**, and `ops::mixing` (G10) is the first consumer. What is *not* reached by it is anything needing those arrays at a different offset or resolution — that is G2, and G5 does not touch it | **The row's prescription was impossible, and both objections are kept because a register that recorded only the outcome would lose them.** *What this row said:* "image numbering in which images `0..k` are inputs and phase `p` writes image `k + p`, plus constructors taking a list." *Objection 1 — the executor.* `strategy.rs` addresses images positionally, as `env.read(task.phase, …)` and `env.write(task.phase + 1, …)`, at ~15 sites. Solving that for `k` inputs forces image 0 to be an input and puts the rest above `n_phases`, which is not the stated numbering at all; getting the stated one means rewriting the executor. *Objection 2 — the builder, and independent of the first.* A caller needs an input's address **before** it constructs the ops that read it (`Chain::source` takes the number; a `BlockOp` stores it), and the phase count is not known until `finish`, because `PlanBuilder::partition` lets a strategy choose. So `0..k` is unreachable from the builder even if the executor were rewritten. *What shipped instead:* a **disjoint high address range** — `ImageId::SUPPLIED_BASE = usize::MAX / 2 + 1`, `ImageId::supplied(i)` — with images the run writes numbered exactly as they were. The two properties `0..k` lacks are the whole reason: the address is knowable before a single phase exists, and **adding an input renumbers nothing**. The rest of the row was right and is what made it cheap: `Chain::Source`, `SourceInput`, `check_source_images`, image lifetimes and the byte accounting already worked above image 0 | **3** (A C D) — the count while it was open. C never named G5; §6 adjudicated that C needs it, which is why C is in the count and not in the "raised by" column | D |
| **G6** | there is no non-spatial axis to sweep | a genuine *window* along channel or time — a temporal median over dozens of frames, smoothing along a many-band spectral axis | a fourth axis on `Voxels`, `Reach`, `Anchor`, `BlockGeometry`, `BlockGrid`, `output_shape`, `side_region`, the tiling check, the budget arithmetic, the cache keys and the distributed placement. **D argues against closing it** — see §6 | **1** (D) | D |
| **C1** | a data-dependent reach cannot be bounded | displacement-field warping, arbitrary-plane reslicing, and with them the *apply* side of deformable registration | a **declared bound**: the caller states a maximum displacement, the op declares it as its reach and refuses at the block, by name, anything exceeding it — the shape `ops::voxelize` already takes. For a spline transform the bound is derivable from the control points, so nobody has to guess | **1** (C) | C |
| **C2** | a `Region` cannot begin below zero | padding, growing a canvas, tile placement before the global solve — the layout is not expressible until after the solve, and the solve needs the layout | a signed origin, **or** a written-down convention that re-origining happens before planning and the crate never sees a negative coordinate. C judges the convention the cheaper answer and the one to write | **1** (C) | C |
| **C3** | axis permutation is carried but not acted on | transpose, 90° rotation, reslicing along a non-native axis | `reach::Space` already carries `axes: [usize; 3]` and it is fingerprinted; a permuted branch needs lattice, read extent, valid region and anchor permuted *together*. `BlockGrid` and `Anchor` are the work | **1** (C) | C |
| **G7** | ~~no barrier phase~~ → ~~built, and not yet collected~~ → **built and collected.** `FragmentOp::barrier()` and `FragmentOp::reduce()` ship, the plan records and refuses on them, both schedulers gate, and **all four ops of this shape now override both** — `ops::fill`, `ops::regional`, `ops::detect`, `ops::label`. On the recorded volume the shipped path went **106.07 GiB → 4.84, a factor of 21.9**, with 23 627 components agreeing in every arm at every lattice. §12.1 | nothing outright — it **costs**. A fragment-and-join reduction takes N+1 passes for N blocks, because phase 1's whole-lattice fragment reach is also its halo. A reduction whose answer is one scalar reads the whole volume per block, exactly as a hole fill does | a phase declared to start when the previous one finished, stating the dependency without a halo. `fill.rs` names this as the way out and as the open architectural question. **The single largest cost lever in family B** — and it now has a **measured price**, which it did not before. §4's globally consistent label volume was built twice, once as a phase (the merge inside the plan, so the halo is the whole volume) and once as a table applied at read time (the merge outside any plan, so there is no halo because there is no phase). Same volume, same recorded data, same downstream consumer, four lattices: the in-plan arm's reads are exactly `mask + blocks x (u32 volume) + consumer`, so its **read amplification is the block count, counted and not modelled** — 0.655, 1.440, 8.772 and **67.427 GiB** at 1, 4, 32 and 256 blocks, against a flat **0.393 GiB** for the out-of-plan merge. At 256 blocks that is **171x the pixel traffic** to produce an answer that is 148.7 KB. **And the pixels are only two thirds of it**: the same clause makes each block gather every fragment as well, so the fragment traffic goes `(1 + blocks) x` all the fragments — 0.067, 0.170, 2.241, **34.864 GiB** against a decorated 0.067, 0.068, 0.136, **0.271** — and the *total* extra at 256 blocks is **101.9 GiB against a decorated 4.2**, or **25.4x**. A barrier addresses the halo and therefore the pixels; it does not address the gather, which needs the reduction to run once rather than per block — **and it now does**, `FragmentOp::reduce`. See `docs/design/barriers.md`, whose §10 is the collection of this row. **Two figures in this row are lattice-dependent and must not be quoted bare:** the `25.4x` is this volume's, and the *share* of it a barrier alone recovers is a ratio of the fragment set to the volume rather than a property of the barrier — 2.7x here, 1.56x on `ops::fill` over `[16, 32, 32]` at 256 blocks where the fragment set is 175% of the label image, and exactly 1.00x on `ops::detect`, which fetches no pixels at any halo. The barrier's absence is not a coefficient here; it is the whole cost. Two things this pins that the row previously only asserted: the toll is **linear in the block count**, so it is worst exactly where cutting finely is most wanted; and and the way out, which when this row was written was reachable only by moving the reduction out of the plan and applying it at read time — at the price of the caller splitting its own pipeline into two `execute_phases` calls, because there was no barrier to express it with — is now expressible in one plan, and the caller's split is no longer the affordable shape | **1** (B) named; A's and D's global reductions pay the same toll by the same route | B |
| **G8** | no scalar broadcast inside an iterative phase → **open, and confirmed open by the thing that looked most likely to close it.** `docs/design/barriers.md` §5 guessed a barrier's declaration might be the substage declaration one level down; §8.10 records that nothing tests it and `src/iterate.rs` is unchanged. §12.4 | level sets, active contours, Chan–Vese, SLIC, Costes automatic thresholding — every scheme whose per-iteration local stencil consumes a whole-volume reduction recomputed each round | a substage able to read a value reduced over the previous substage's whole output. Same mechanism as G7 with a loop around it. **B calls it the highest-leverage item in its document** | **2** (B D) | B |
| **G9** | ~~an op cannot require one axis be left whole~~ → **the planner cannot be told not to cut an axis** *(partially answered by §8.5; re-scoped, identifier unchanged)* → **still open, and the same complaint was answered in the other dimension.** `PartitionSearch::SingleGroup` lets a caller say "do not cut *between slots*" for a reason the cost model cannot see; nothing lets it say "do not cut *axis k*". → **and now open in two distinguishable ways.** `decomposition::refined_ladder` (opt-in, `Constraints::with_refined_ladder`) answers the *granularity* half; the *per-axis* half has nothing to reach for, because the search's own constructor is `BlockGrid::along` and it takes one integer. §12.4 → **and now open in a third way, which has been measured and half-answered in the same pass.** The first way is an axis left whole; the second is granularity; the third is a **divisibility** — an op that needs the block *edge* to be a whole number of something of its own. `ops::convolve::TransformConvolveOp` is the first payer: its transform tile grid is anchored to the volume, so a core that starts mid-tile must reach back to that tile's start, and it therefore declared `tile - 1` of slack per side **on every lattice**, including the ones where the slack buys nothing. `BlockGrid::cores` builds `start = index * block`, so an edge that is a multiple of the tile makes every core start aligned and the true halo exactly the kernel's — and the planner's ladder is powers of two, so a power-of-two tile is aligned on **three of the coarse ladder's four rungs**. The waste was the ordinary case, not a corner | the exact Euclidean distance transform (cost, not correctness — a lattice that cuts a whole-axis-reach axis with the full halo is redundant, not wrong; and see §8.4 for what that op is and is not evidence about); any temporal alignment that must not tear a stack. **The correctness half is answered.** An op declaring `AxisReach::All` in `Frame::Source` now mandates that the axis is left whole **or** given a whole-axis halo — declared by the op, enforced by the tiling check that already existed, with no constraint type added and none needed → **and one thing it cost, counted rather than modelled, before this pass closed the cost half.** Read amplification on `1024^3`, 32-voxel tile, radius-4 kernel, all three axes cuttable, through `BlockGrid::mean_read_voxels`: at the coarse ladder's rungs that are whole tiles — 32, 64, 128 — the worst-case halo reads **30.176x, 8.309x and 3.232x** where an aligned lattice needs **1.917x, 1.394x and 1.173x**, a waste of **15.7x, 6.0x and 2.75x**; at radius 8, **11.5x / 5.2x / 2.6x**. In residency, at edge 128, **62.1 MB against 20.1 MB per block** and **2.48 GB against 0.80 GB** at 40-way concurrency. `tests/transform_convolution.rs` carries the table, asserts the gap at the three aligned rungs, and carries the liveness partner that would catch a ladder chosen so far out that the gap had closed. **And amplification is the optimistic half, which the planner-level test found and this row did not predict.** `cuttable_axes` drops an axis whose `edge + lo + hi` is not less than the extent, and it runs *before* anything is priced — so on a volume that is not large against the slack the axis is not amplified, it is **dropped**, and the phase degenerates to one block reading the whole volume. That is G7's cost reached from the opposite end. Measured on `96^3` with a 32-voxel tile and a radius-2 kernel: `Greedy` at candidate edge 32 plans **27 blocks** with the discount and **one** without. Both regimes are real — amplification is what a volume large against the slack pays, degeneration is what a volume that is not pays. **→ Corrected, and it weakens the first half of this row: amplification is not time.** Every figure above is in **voxels**, and G20 measures what a voxel of halo actually costs. On a cold sequential read the extra bytes ride along on readahead — `3.48x` the bytes for `1.32x` the time — so **the `30.176x`, `8.309x` and `3.232x` overstate the time cost of the slack by roughly `2.6x` when the data is cold**. Warm they are close to right (`3.48x` bytes, `2.98x` time), and on a chunked store they *understate* it. So the amplification column is a byte figure and must be quoted as one. **What does not weaken is the half this row rests on**: the degeneration to one block is a count and not a ratio — 27 blocks against one — and no cache discounts a phase that reads the whole volume per block; and the residency figure that dominates everything, `42x` cold against warm, is *worse* for a plan that fetches a wide halo it will not reuse. The stride discount is worth less than the byte figures imply and is not worth nothing, and the difference between those two statements is the reason G20 exists | **Re-scoped to the planner-facing half only.** `Constraints`/`BlockConstraint` still cannot express "do not cut axis *k*", so the enumerator proposes a lattice that will be **refused** rather than avoiding it — a search that wastes candidates and a caller who sees an error instead of a plan. `BlockConstraint::FullExtent(axis)`, or an `Extent` of `Option<usize>`, is still the shape; it is now a **planning-quality** change and no longer a correctness one → **built, and not as the constraint this row kept asking for.** The cost half of the divisibility case is **closed**: `AxisReach::Aligned { stride, lo, hi }` ships in `src/reach.rs`, and `Reach::in_voxels` — which is handed the lattice, and which `decomposition::price_phase` **already calls once per candidate grid** — is where it is discounted. **One more line was needed and it was found by the test rather than by reading:** `decomposition::cuttable_axes` decides which axes may be cut *before* anything is priced, and it was asking the unresolved reach, so an aligned op had every axis dropped and its phase collapsed to one block before the discount could be taken. It now resolves against the candidate edge, which is the identity for every other reach and therefore changes no plan that existed. Everything that cannot see a lattice still answers the worst case, which is the only safe direction. **Three reasons it is a reach and not a `BlockConstraint`, and each is a fact about the code rather than a preference.** *(i)* `BlockConstraint` has exactly one operation, `lattice()`, which produces **the** grid, and `phase_for_group`'s own comment says the candidate list is *replaced* rather than filtered — a divisibility admits many grids and fits neither. *(ii)* `Chain::block_constraint` folds by **equality**, so two ops with different strides could never share a phase; a reach folds by taking the worst case, which costs the discount and not the plan. *(iii)* A constraint turns a **cost** into an **error** — it removes candidates and refuses callers — where a reach turns it into a *price*, so the planner prefers an aligned edge on its own and nobody has to be refused. **The blast radius was measured, not estimated — and the first measurement was wrong in the direction a truncated probe is always wrong:** an `AxisReach` variant was added to a copy of the tree and `cargo check --all-features --lib --tests` run against it, which reported **5 sites, all in `src/reach.rs`**. That number is an **under-count**, and the reason is the same class of error as this register's `head`-truncated grep: the check **aborted at the lib-test stage**, so the integration tests were never compiled. The real count is **6**, the sixth being a hand-written `BlockOp` in `tests/partition_search.rs` that matches `AxisReach` to derive its own symmetric bound. It was found by the full `cargo test` and not by the probe. *A build that stops early hides every site after the one it stopped at*, and a probe is only a measurement of what it got to. Six sites, five of them in `src/reach.rs`, is still small. The *silent* half was the wire, as it was for G3: `AxisReach::from_json`'s last arm accepts any object and reads `lo`/`hi` from it, so an aligned reach would have decoded as a `Bounded` carrying its **discounted** sides with the stride dropped — a halo narrower than the op needs, on a lattice nothing would then check, in another process. It is handled before that arm and pinned by a test that names the wrong answer. **What is still open is the rest of this row:** an op still cannot *demand* an aligned lattice and `Constraints` still has no per-axis rule. **The third thing this sentence used to list was measured and turned out to be a defect rather than a limitation, so it is fixed rather than carried:** *"the discount is lost when the aligned op shares a phase — the fold flattens rather than inventing a lattice that satisfies two strides."* A phase's reach is its ops' reaches **added**, so flattening meant adding a reach of *nothing* was not the identity, and fusing the op with a voxelwise map lost the whole discount — **27 blocks against one on `96^3`**, which is a phase reading the whole volume per block and not merely a dearer halo. `AxisReach::Aligned` now carries **both** of its answers rather than one plus the rule `stride - 1`, and folds each componentwise: adding nothing is the identity, adding a bounded reach is exact, `max` is exact on a common multiple and generous off it, and two strides take their least common multiple. **The one-plus-a-rule form could not have been fixed in place**: two stride-32 reaches added would have claimed `31 + lo` off-alignment where the truth is `31 + 31 + lo`, an under-halo of 31 a side. And a common multiple past every candidate edge degrades to exactly what flattening gave, so the fold is never dearer than the rule it replaced — asserted against coprime strides rather than argued. **And the do-nothing baseline is priced, because it weakens the case and had to be:** a caller could always shrink the tile instead, and a 16-voxel tile at radius 4 costs `130.22 ns/voxel` against a 32-voxel tile's `94.86` (1.37x compute) while reading `3.772x` against `8.309x` at edge 64 (2.2x traffic). That workaround is real. The variant beats it by **2.7x of traffic at 0.73x of the compute**, which is what made it worth building **And the interaction with this row's *other* half inverts a complaint, which is why it belongs here and not only in §12.4.** The per-axis half and `refined_ladder` are both about the **menu** — which edges the search may propose, a *caller*-to-planner question answered in `Constraints`. A divisibility is an *op*-to-planner question about which of those edges are cheap for one phase, and it is answered in the reach, the one quantity the pricer already re-resolves per candidate. They are different questions, and they interact **favourably**: because the planner can only express a **scalar** ladder and that ladder is powers of two, a power-of-two stride is already aligned on most rungs — three of the coarse four — so the discount rarely changes *which* lattice is chosen and mostly stops the plan over-fetching on the one it was going to choose anyway. Had the planner been able to offer arbitrary per-axis edges, alignment would have been rarer and this would have had to remove candidates. **The limitation this row calls a defect is what made the fix cheap.** | **2** (B D) named, C implied by any separable sweep — unchanged, and now counting who wants the *planner-facing* half → unchanged by the divisibility case, which is **A**'s and is a cost rather than a block: the transform convolution planned correctly at every lattice before the discount existed and still does. Recorded here because the *kind* of a gap and its count are different facts | B |
| **G10** | ~~no K-ary reach-0 op shell~~ → **closed.** `ops::mixing` ships the shell (`TupleOp`), the kernel trait (`TupleKernel`) and the first kernel (`LinearMap`, the per-voxel matrix); `tests/tuple_map.rs` pins the map, the mixing, decomposition invariance, that every input is really read, and that side outputs are still terminal | per-voxel classification, argmax over C probability channels, linear unmixing, stain separation, crosstalk correction, colour-space conversion, and every windowed temporal filter — **all now buildable**, the extra inputs being supplied images (G5) and the extra outputs side outputs | **The clause was right about the shape, contained one error, and the error has since been fixed — all three are kept, because the middle one is the useful part.** *What this row said:* "one `BlockOp` with C−1 `source_inputs` and C′−1 `side_outputs`. No new trait and no new axis." The shape is exactly that, no axis was added, and "not a `Combine`" holds for the reason given — the trait has no side outputs. *What it did not see:* **`BlockOp::apply_side` was not handed the `SourceInputs`**, so a side output that is a function of the op's source inputs could not be computed there; `ops::mixing` shipped with a per-block map keyed by the buffer offset, filled by `apply_with` and drained by `apply_side`. *What has since landed:* the argument is threaded from the executor's call site (`strategy.rs:1032`) through `Chain::apply_side` and `Environment::apply_side`, and the per-block map is **deleted, field and all**. **So this clause is now true exactly as written.** What it bought is worth more than the tidiness: `apply_side` is a **total function of its operands** rather than of its operands plus what an earlier call left behind — the old shape was correct only while two calls stayed paired, and it held whole `f64` blocks resident that no counter knew about, in a crate whose side outputs exist *because* 95.2 MB was once counted against 158.6 MB written. The price is that the inputs are streamed twice: **1.08–1.10×** at K = K′ = 16, with the tiling's 2.5–2.8× fully retained and the flop count unchanged. Blast radius **4 overriding implementors out of ~30**, a default absorbing the rest, and nine implementors in the sibling application crate untouched; and the documented "side outputs and source leaves do not compose yet" limitation is **deleted** as a side effect. §11.3 | **2** (B D) when open; **0** now, and this time with **no residue either** — re-derived rather than carried. The residue this row briefly held costed **B** (a classifier over K channels writing C class maps) and **D** (unmixing writing C′ channels beside a residual) and blocked neither; both now pay 1.08–1.10× of streaming instead, which is a price and not a gap. A's arity-2 arithmetic combines are the adjacent, smaller case and are still unwritten | B and D independently |
| **G11** | rows cannot be decomposed by spatial region | candidate link generation across a series — the pairs of rows within a spatial radius across one step | a fragment op reading two row streams under a *spatial* neighbourhood. `ops::rows` decomposes by row range with **no overlap**, on correctness grounds — an overlap duplicates a row and no downstream check can tell it from a real one — so this is a new shape, not a parameter on an existing one | **1** (D) | D, correcting B |
| **G12** | `iterate::Operand::Running` is singular | blind deconvolution and any alternating-minimisation scheme | a second pair of alternating private buffers and a convergence rule over both. The restriction is deliberate and documented; worth recording, not worth building until something asks | **1** (A) | A |
| **G13** | point coordinates cannot be sub-voxel or negative | a chain of point transforms rounds at every step, so evaluating a transform at a point set is not worth having; sparse feature-tracking output has nowhere exact to land | `[f64; 3]` in the table, or a documented fixed-point convention. `points::Point` holds `[usize; 3]` and `ops::rows::scaled_index` refuses a negative factor because "a table holds `usize` coordinates" | **1** (C) | C |
| **G14** | **no declaration anywhere is checked against the fetch** *(partially closed by §8.5: source-frame `AxisReach::All` is now checked; **silence is not**)* | nothing outright — it admits a **wrong answer**. A projection that reads only its own block is accepted by every guard, has exactly the right shape, and is wrong at every position; a fetch covering *half* the collapsed axis is accepted too. `Decomposition::check` verifies that a block's `source` lies **inside** the image it reads (`decomposition.rs:713-728`), never that it *covers* what the reach claimed. The one guard that does look at a declaration (`decomposition.rs:702-711`) checks only that a source-lattice reach is accompanied by *some* per-block fetch, not by a sufficient one. **The only gap in this register that is about correctness rather than capability or cost** | **Landed for the declared case** (`decomposition.rs:763-791`), which refuses a fetch that does not cover a declared whole axis and names the axis, the block, the fetched range, the required range and the phase's own extent. **The residue is silence** — "Part 2 holds an op to what it *said*; it cannot make an op speak" — and every fetch that is not a whole-axis claim, an affine map among them, has nothing to be checked against. Closing the residue wants a **total** per-axis rule where no axis can be silent (§9's proposal, and its totality is the load-bearing part) | **4** (A B C D) — re-derived in §8.3 after the landing; the number holds, the reachable failures are fewer, and the residue is sharper | this correction pass |
| **G15** | ~~**a mask cannot be held as one bit** — there is no `Bool`-producing threshold, and the fan-in that would join one is bound to `f64`~~ → **closed, and the second half closed in a shape this row did not propose.** `VoxelwiseMaskOp<ThresholdMask>` produces `Dtype::Bool`, and `LogicCombine` joins arms of *different* carriers — but only when the caller states the output with `producing()`, which is not the "`accepts` that does not force the widest arm" this row asked for. §12.2 | nothing outright — it **costs, in the one currency a tile-scale stage runs out of**. A verdict is an `f64 → f64` `MapFn` (`voxelwise::Threshold` is one), so a thresholded arm produces `f64`; `LogicCombine::accepts` takes only `Bool \| F64` **and requires every branch of a fan-in to agree**, so one arm that cannot narrow binds all of them; and the image the phase writes is then `f64` — **eight bytes a voxel for a one-bit fact**. Measured on the binarize stage at tile scale: the peak falls **57.853 → 46.28 GiB** if the mask images are `Bool`, which is the difference between a stage that runs at a size and one that does not. Family B reached the same cost from the other side and wrote it down — the reconstruction shell `accepts` `f64` only, so "a binary reconstruction is an `f64` volume's worth of memory for a one-bit question" (B §3) | a threshold op that **produces** `Bool` rather than a mapped `f64`, and a `Logic` sink that joins `Bool` arms. `NarrowOp::to_mask` is already the crate's way into `Bool` and already carries the mask convention (non-zero is true, and `Narrowing::new` refuses `Dtype::Bool` by name because a two-valued target is a comparison and not a rounding), so the convention is not the missing part — the missing part is a verdict that lands on it without a round trip through `f64`, and an `accepts` that does not force the widest arm on the rest. *Unverified here:* whether narrowing **every** arm of a real fan-in is a general answer; the binarize work reports it is not, because agreement binds each arm to the weakest, and this pass did not build the counter-example | **3** (A B D) — derived, not assigned: **A** (Boolean logic between two volumes, §2's row, over thresholded branches), **B** (every binary-morphology and reconstruction consumer, and B's own note above), **D** (§4.2's N-channel Boolean union of masks, the one row D calls a useful surprise). **C** holds no mask image: its isosurface takes a scalar field and the threshold that produced it is B's by §0 | the binarize work, and B independently |
| **G16** | ~~`Decomposition` cannot say what a plan's **peak** costs~~ → **closed, with one argument this row did not foresee and one prediction that ran backwards.** `Decomposition::peak_image_bytes(work)` ships; the three image kinds do matter to it, exactly as prescribed. What the row did not see is that the plan alone cannot answer — whether a phase writes an image is the *op's* answer — and that the disagreement it feared ran the other way. §12.3 | nothing, and it blocks nobody — but it is the single number that decides whether a stage can run at a size, and it is **open-coded in three consumer test files** — *reported by the work that asked for it and not re-counted here; there is no such walk under `blockflow`'s own `tests/`, which is consistent with the consumers being outside this crate.* The walk is the same one every time: for each phase, the images alive across it, priced by `volume_at × dtype`, maximised over phases. `readers_of_image`, `images_dead_after`, `image_visibility` and now `image_kind` are all the pieces, all public, all already agreeing with each other; nothing folds them. Three hand-rolled copies of a lifetime walk is three chances to disagree with the executor about when an image dies, and the executor's answer is the binding one | `Decomposition::peak_image_bytes()`, derived on exactly the argument `image_visibility` is derived on — a field that could disagree with the arithmetic is a field that eventually does. **The three kinds matter to it**: a supplied input is alive for the whole run and is not the run's to free, an intermediate may be dropped and rebuilt, and an output has a materialisation obligation, so a peak that treats all three alike is not the number a caller wants. `docs/design/images-and-phases.md` prices one such peak by hand — 67.8 GiB of images alive at one phase, against a measured `VmHWM` of 124.9 GiB — and records that the figure is a property of the *linearisation* and not of the DAG, which is the second reason to have it in the crate rather than in three tests | **0** — it blocks no operation in any family. Recorded with a zero rather than left out, on §8.2's precedent: the count and the *kind* of a gap are different facts, and a row that carried only the count would read as unimportant | this correction pass |
| **G17** | ~~**nothing in the crate produces a fragment stream from a table or a point set already in hand**~~ → **closed, and the row it looked like turned out to be two.** `ops::rows::RowSourceOp` and `points::PointSourceOp` ship. Every consumer op in `ops::rows` and every consumer of points follows a phase that *emitted* some — `ops::coordinates` and `ops::detect` derive theirs from an image — so a plan whose **input is a table** had nothing to start it and each caller wrote the producer itself. Fifteen lines: reach 0, one `Coverage::EveryBlock` output, and a filter that keeps the rows this block's core holds | nothing outright, and it blocks no operation in any family — what it costs is a **keying rule restated per caller**, which is the one thing here that cannot be restated safely. A producer and its consumer must agree on which block owns a coordinate, and that agreement is the precondition `GroupFold::merge`'s duplication refusal and `ops::voxelize`'s core check both rest on from the other side. Written out five times in tree — twice in the row form (a consumer stage, and **`tests/rows_group.rs` in this crate**), three times in the point form (a consumer stage, `tests/voxelize.rs`, `tests/point_labels.rs`, the last two character for character) | **two ops, not one, and that is the correction.** The two look like one shape over two payloads and are not: a point blob is **headerless** where a table blob carries its schema in front, deliberately, because any four-word blob is a valid point set and the header would be a constant the reader already knows — so a general row producer over `Schema::points` would write the right words behind a header `ops::voxelize` does not read. `points.rs` had already declined that trade in prose. The **keying differs too**: the row producer uses `ops::detect::owner_of`, which clamps, because a table refuses an out-of-volume row later and by name; the point producer divides **unclamped**, because a point the lattice cannot place must be written *nowhere* rather than into the last block, which its consumer would then have to refuse. Two producers, two rules, neither a case of the other | **0** — it blocks no operation in any family, on §8.2's and G16's precedent for recording a zero rather than leaving the row out | this pass, and see the note above on why no earlier one could have |
| **G18** | ~~**the probe set has holes, and the test suite fills them by copying**~~ → **closed, and the row overcounted its own fix.** `probes::CountingIdentityOp`, `probes::NullFragmentOp` and `BlockSummaryOp::with_pixels(bool)` ship; five copies are gone. **The row claimed the knob would delete four on its own and it deletes three**, and the two it does not delete are the correction worth more than the outcome: `barrier_phase.rs`'s `BlockSumOp` and `WideningOp` declare `writes_pixels() == true` and carry a **caller-chosen payload** that the next phase decodes, neither of which `BlockSummaryOp` can express at all. They are not knob cases and never were; the row read `reads_pixels` off them and stopped looking | what it cost was **an identity that counts** and **a fragment op that does nothing**, each written twice and each verified byte-for-byte identical below its doc comment, plus **a pixel-free block summary** written twice more. The sharpest of the five is the pair in `image_lifetime.rs`: a tally and its own negative control, *"the same op with **one thing changed**"* — a promise two hand-written structs could only make and now a bool, so the premise of the residency measurement is structural | landed as three additions to `probes` and **not** to `ops`, on this row's one real claim: none of them makes anything expressible. `CountingIdentityOp` is an identity plus a counter; `NullFragmentOp` fixes `Coverage::Sparse` because an op declaring `EveryBlock` and returning `BlockOutput::nothing` would be refused by the coverage guard, correctly; `with_pixels(false)` **refuses from inside** if the executor hands it pixels anyway, which is the assertion `fragment_ops.rs` used to make in a `SeedOp::apply` of its own and is now made everywhere the knob is used | **0** — it blocks no operation in any family. Recorded on G16's precedent for a zero | the test-suite sweep, §3.1 |
| **G19** | ~~**no op emits one row per set voxel *carrying a value read at that voxel***~~ → **closed, exactly as prescribed: a column, not a mechanism.** `ops::coordinates::SetVoxelsOp::with_values(column)` ships, with `valued_coordinate_schema`, `set_voxel_values_into` and `encode_set_voxel_values` beside their unvalued twins. Everything the op already decided is untouched — reach 0, `Coverage::EveryBlock`, and the rule that a voxel is in exactly one core and its coordinate is a volume coordinate. `tests/row_table_ops.rs` is on it and its fourteen assertions are unchanged | the producer half of every plan that filters or groups rows by a quantity the image holds. `SetVoxelsOp`'s rows had **no payload column** and `ops::rows::RowSourceOp` takes a table the caller already has, so a suite testing a row filter had nothing to filter **on** and wrote its own producer to get one. **The test reported the gap in its own words** — *"that is what needs the shell that does not exist yet"* — and that sentence has been **inverted rather than deleted**: the file now says the shell exists and what it is | the value is read through `Voxels::widened`, so one `f64` column serves every input dtype and a consumer's schema does not change when the input's does — a column that followed the input would make a downstream `Grouping` depend on a decision taken upstream of it. **Still not a substitute for `GatherRowsOp`**, and the op's own header says so: this reads the value out of *the image the rows came from*, at the voxel the row is about; a gather samples a **second** array at rows that already exist, which is the case the consumers have | **0** — it blocks no operation in any family, and it is a producer rather than a capability. Recorded on G16's precedent | the test-suite sweep, §3.1 |
| **G20** | **a halo voxel is priced as a core voxel, and the crate has no evidence that it costs one.** `CostModel::read_cost_per_voxel` is a single coefficient and every site multiplies it by the **read extent** — `decomposition.rs:2367`, `strategy.rs:3102`, `assemble.rs:987` — where the write terms all use `mean_core_voxels`. Nothing anywhere distinguishes a voxel fetched for the first time from one fetched again because it fell in a neighbour's halo: the counters (`EnvCounters::read_voxels`, `chunks_read`) are monotone `fetch_add` per read with no residency set, so a chunk straddling two blocks is counted twice by design. The **fraction** of a block's read that is halo is `1 - core/read`, which is a property of the **candidate**, so one weight over both is a candidate-dependent error — the family this batch has now found several times. *Raised by the project owner: "halo cost should also not be overstated given io cache."* | nothing outright — it **misprices**, and the measurement says the misprice is neither small nor one-signed. **Three paths, three answers.** *(i) In memory there is no IO to cache.* `ArrayEnvironment::read` is a slice copy out of a resident `Array3` plus a fresh allocation — no file, no mmap; `env.rs` calls its own chunk grid "an accounting fiction, used by `chunks_touched` to price IO that is not happening". Nearly every figure this project states in voxels was taken here, so "given io cache" has nothing to apply to. What does apply is the **CPU** cache: on `512^3` `u16`, holding the region size and the allocation fixed and varying only source residency, a warm source costs **0.309x** a cold one at `32^3` regions and **0.687x** at `64^3`. Real, bounded below by the allocation and the destination write, and **itself a function of the block shape** — which is the disease, not the cure. The halo sweep agrees from the other side: at edge 128, where it is monotone, `2.065x` the voxels cost `1.764x` the time, an effective halo weight of `0.72`. *(ii) On a chunked store the halo is **dearer** than its voxels, not cheaper.* `ZarrEnvironment`, `256^3` `u16`, chunk `64^3`, block edge 64, gzip(1) — **the crate's default for every integer dtype**: a halo of 4 is `1.308x` the voxels and **`5.33x` the time cold, `4.42x` warm**. *(iii) And the voxel count is not even the right variable, which a control settles.* A halo of **64** — a whole chunk — fetches `15.625x` the voxels, twelve times the halo of 4, and costs **`3.6x` less** than it uncompressed and about the same under gzip. The column that tracks the cost is `ZarrEnvironment::unaligned_reads`: 64 of 64 reads partial at halo 4, 8 and 16, and **0** at halo 0 and halo 64. A halo's first effect on a chunked store is not that it fetches more, it is that it **stops the read landing on whole chunks**. *(iv) And the page cache buys little of it.* `warm/cold` under gzip is `0.78`-`1.00` — a second pass over data entirely in the page cache is barely faster, because the inflate is paid again and there is nothing to skip it: `src/cache.rs`'s `ChunkCache`, with its tiers and coalescing, has **no non-test construction site anywhere in the crate**, so no `Environment::read` is cached. Uncompressed, where there is nothing to re-decode, the page cache is worth about `2x`, and that is the one cell in the whole table where the original hypothesis holds. `tests/halo_cost.rs`, `tests/halo_io_cost.rs` **→ and a second worker measured the *other* storage layout, with the opposite sign. Both results stand; together they are the answer.** On a flat 1 GiB volume of 512 planes at halo 5, with `posix_fadvise(DONTNEED)` for the cold arm, that work reports **`3.48x` the bytes for `1.32x` the time cold** — non-monotone, four blocks *faster* than one, because a cold read is latency- and readahead-bound and the extra bytes ride along nearly free — and **`2.98x` warm**, tracking bytes because a warm read is a memcpy. So on a sequential layout halo bytes **overstate** halo time by about `2.6x` cold and `1.17x` warm. On the chunked store measured here they **understate** it by about `4x`, because there the halo's first effect is to break chunk alignment rather than to add bytes. **Two paths, two signs, and neither is a timing artefact** — the chunked result is confirmed by a counter, not a clock. *That worker's measurement is not re-taken here and this row does not duplicate it; the two cover different layouts on purpose.* **And the figure that dwarfs both is theirs: cold is `42x` warm at one block.** Residency beats amplification by an order of magnitude. | **a second weight for halo voxels — and it is refused. The refusal is the finding, and the `42x` is the argument.** `price_phase` already holds both quantities (`mean_core_voxels()` and `read_per_block`), so the model *could* say it, and it would be **DP-safe**: `strategy.rs`'s "what would break it" list names *cross-phase* coupling — a materialisation charge that knows its consumer's halo, a cache that survives a phase boundary — and a discount on a phase's **own** halo is a function of its own grid and reach. So the search is not the obstacle. **Three reasons not to, in increasing order of force.** *(i) No constant is right.* The measured weight is `0.31`-`0.72` in memory, about `0.38` on a cold sequential file (`1.32/3.48`), about `0.86` warm on the same file, and above `4` on the default chunked codec. It is not one-signed, so a constant is wrong by a factor of ten on some path. *(ii) The planner cannot choose by path.* `Strategy::decompose(workflow, constraints)` takes **no `Environment`** and is documented `BINDING: deterministic, hashable, data-blind`; the plan is made before the storage path exists. That is exactly what `src/npy.rs`'s coalescing measurement refused — `1.36x` cold, `0.70x` warm, "a direction that depends on the page cache, which this cannot know" — and the same refusal is owed here. *(iii) And this is the one that decides it: **the coefficient this model is missing is not "halo against core", it is "cold against warm", and that one is `42x`.*** A halo weight would refine a term that is dominated by a factor twenty times larger, supplied by nothing the planner can see. Building the smaller correction while the larger one is unmodelled would make the price *look* better calibrated without being so, which is worse than a coefficient that is honestly coarse. **What is recorded instead, so today's behaviour stops being accidental.** A single weight charges a halo voxel at core price. That **over**-charges cold sequential reads by about `2.6x` on the halo term — the *safe* direction, since it biases against exactly the fine cuts with wide halos that `docs/design/barriers.md` exists to prevent — is about right warm, and **under**-charges a chunked store by about `4x`, which is not safe and is now written down. **Two routes stay open and neither is this pass's.** `crate::statistics` already fits `Term::Read` from real runs, and the halo fraction varies across the block sizes a calibration set contains, so a **second fitted** coefficient is identifiable where a guessed one is not — a calibration change with its own measurements. And the lever the control actually points at is **chunk alignment**, which unlike residency *is* knowable at plan time: the planner holds the block grid and the chunk grid both. The crate counts it (`ZarrEnvironment::unaligned_reads`) and never prices it. That is where a read-side pricing change should go **→ Corrected by measuring it, and the sentence that was wrong is "the planner holds the block grid and the chunk grid both". It does not, and when the lever was priced it turned out not to belong to the planner at all.** *(i) The planner is not told the chunk shape and there is no way to tell it.* The shape enters the crate only at **execute** time — `Environment::chunk_shape()`, used by `strategy.rs`'s per-block accounting, and `check_chunk_exclusive_writes`, called from `zarr_env.rs`. `Constraints` carries a budget, a concurrency, a cost model, a scalar ladder and a split-axis list, and nothing about layout below the volume. *(ii) The op's stride and the store's chunk constrain **two different lattices**, so they do not simply compose.* `AxisReach::Aligned` constrains the **core** lattice — `BlockGrid::cores` builds `start = index * edge`, so an edge that divides the stride puts every core start on a tile boundary. Chunk alignment constrains the **read** lattice, and a read starts at `core - halo`. An aligned edge is therefore *not enough*: the **halo** must also be a whole number of chunks, and a halo is the op's reach, chosen for a kernel and never for a store. Asserted on counters rather than a clock, with every row on an edge of exactly two chunks so only the halo varies: halo 0 and halo 32 give **zero** partial reads, halo 4 and halo 8 give **every** read partial. The lcm fold that made two op strides compose cannot help, because the binding term is not on the edge. *(iii) Priced, the lever is real, conditional, and smaller than it looked.* The only way to buy alignment for a halo the op did not choose is to **over-read to the boundary**, and on `256^3` with a `32^3` chunk and a halo of 4 that pays only when a block is at least **two chunks wide**: at one chunk per edge the over-read is `11.5x` the voxels and costs `1.6x`-`2.1x` **more**; at two and four chunks it wins `1.15x` and `1.55x` under the default codec, `5.6x` and `4.1x` uncompressed. **And it only pays cold** — warm it is a small loss at every width, because it is strictly more bytes and there is no IO left to save. *(iv) So the sign depends on residency after all, which is what this row said nobody can know — and that relocates the lever rather than killing it.* A fetch that rounds itself out to chunk boundaries needs no `Constraints` field, no plan change and no exception to data-blindness: `ZarrEnvironment::read` already holds the requested region **and** `array.chunk`, and the condition — is this block two chunks wide — is visible there too. **It is an environment decision and not a cost-model one**, which is why nothing about it is built into the model here. Built as a default it would be wrong in one direction or the other; the shape it should take, if anyone takes it, is an opt-in on the environment carrying the table above, exactly as `Compression` is. `tests/halo_io_cost.rs` | **0** — it blocks no operation in any family. Recorded with a zero on §8.2's precedent and G16's, because the count and the *kind* of a gap are different facts, and a row that carried only the count would read as unimportant | the project owner |
| **G21** | **`differing_voxels` is private, and thirteen test files have re-derived it — eight of them without an extent check** → **closed, and the part the copies got wrong was also wrong inside the crate.** `strategy::differing_voxels` is now `pub`, takes two `Voxels` rather than the executor's `BlockBuf`, and **refuses a mismatch of extent or of element type by name** instead of folding it into an absence. That last is the load-bearing change and the private version had the same defect one level in: it *did* compare shapes, but it answered `None`, and the single call site wrote `.unwrap_or(0)` — so "these are different shapes" and "there is nothing to compare" both became **zero differing voxels**. The two are now distinguished: a new `differing_block_voxels` returns `Ok(None)` for an accounting environment, which is the one honest absence, and propagates a mismatch as an error the executor can no longer swallow | the count *"how many voxels do these two volumes disagree on"*, which is the number every parity and every decomposition-invariance argument is finally made of. `strategy.rs` has it, dtype-generic over `VoxelElement`, at **one call site**, and its own doc argues for exactly the shape a caller wants: *"A free function rather than a method … an environment has nothing to say about it that it does not already say through `same`."* It is `fn`, not `pub fn`. `Voxels` has no `differing` and no `same`. So the consumer wrote it thirteen times, in three incompatible shapes — over `Array3` and over `Voxels`, `bool`-only and `u16`-only — and **eight of the thirteen omit any extent check**. `zip` truncates, so those eight answer a *small* difference count for two volumes of different extents, which in a parity suite is the failure reading as a pass. *(The first count of this was **ten**, and it was wrong: a single-line grep for `assert_eq!(…shape` missed two multi-line guards. Corrected before it was acted on, and recorded because the same grep will be written again.)* | `pub fn differing_voxels(left: &Voxels, right: &Voxels) -> Result<u64>` in `voxels.rs`, beside the element accessors — **and it must refuse rather than truncate on an extent mismatch**, which is the half the copies got wrong and the half a private function was never asked for. It deletes thirteen copies and adds the missing guard to eight. **The prescription is not hypothetical — it has been built already, twice, and neither copy is reachable.** `clearmap_ng::stages::skeletonize::removed(before: &Voxels, after: &Voxels) -> Result<usize>` is this function exactly: public, over `Voxels`, and it **refuses** on an extent mismatch with `Error::ShapeMismatch` before it counts — the half the eight got wrong, arrived at independently. Its doc says *"a named public function rather than a comparison inlined into the loop … because a test can then ask it directly"*, and `tests/skeletonize_oracle.rs` is the one file whose `differing` is a one-line delegation to it. So the shape is settled and the argument for it is already written down; what is wrong is only **where it lives**: one copy private in a framework module, the other public but named for thinning inside a domain stage, and a test comparing two smoothed volumes can reach neither → **built, with the fixture chosen so that the wrong answer is a *plausible* one.** The test compares a `[2, 3, 4]` volume against a `[2, 3, 8]` one that agrees everywhere the smaller reaches, and asserts that both extents are named in the refusal in either argument order, that both element types are named when those differ instead, and — the liveness partner — that the truncating `zip` this replaces answers **zero** on that fixture, run inside the test so the number it would have produced is on the record rather than described. **The method half was not built and was not reached for**: `Voxels::differing` would belong in `src/voxels.rs`, which is another worker's file this pass, so the free function shipped and the method is routed rather than taken. `strategy.rs` | **0** — it blocks no operation in any family, and it is a measurement rather than a capability. Recorded on G16's precedent, and it is the row in this table with the highest ratio of *wrong copies* to *cost of the fix* | the free-function sweep, §3.2 |

### 3.1 The test-suite sweep, and the three verdicts it needs

G17 was found by accident: its third copy sat in this crate's own
`tests/rows_group.rs`, and the observation that fell out of it is the reason
this section exists.

> **A test that has to supply the missing half of the module it is testing is a
> gap report**, and nothing was reading them. Every other row above was found by
> reading `src/ops/`. A survey of operations does not open a test directory.

So the sweep was run deliberately: **every `impl …Op for` in both crates' test
suites** — **61 across 32 files in `blockflow/tests/` as the corpus stood when
the sweep ran**, one in `clearmap-ng/tests/` — read with its doc comment and
classified. *That figure is the corpus before this pass's own migrations, and it
is stated with its condition because the pass then removed nine of them: a
reader grepping today finds **52 across 28 files**, and the difference is G17's
and G18's closures, not a miscount.* What makes the
result trustworthy is that **three verdicts were available, not one**, and the
last two are the ones that keep it honest:

| verdict | what it means | may it become a row? |
|---|---|---|
| **supplies the missing half** | the suite must build a producer to exercise a consumer, or the reverse, because the library ships one side | **yes** — this is the gap |
| **deliberately wrong** | the fixture exists to break a rule so a guard can be shown to fire | **never**, and it must be named so a later sweep does not re-open it |
| **stale hand-roll** | the library op now exists and the test predates it | **no** — it is adoption, not absence, and mistaking one for the other would put a closed gap back in the register |

**What it found.** Two rows, both above: **G18** (the probe set, four shapes,
two of them verified byte-for-byte identical) and **G19** (a valued coordinate
producer, reported by `tests/row_table_ops.rs` in its own words). Everything
else resolved to one of the other two verdicts.

**Both are now built, and building them corrected the sweep three times.** That
is recorded here rather than smoothed away, because a sweep that could not be
wrong would not have been worth running:

* **G18's fix was overcounted.** `with_pixels` deletes **three** copies, not
  four. `BlockSumOp` and `WideningOp` declare `writes_pixels() == true` and
  carry a payload the next phase decodes; the sweep read `reads_pixels` off them
  and stopped. They are ordinary per-test ops and stay.
* **`MarkRowsOp` is deliberate, not stale.** The sweep filed it as adoption —
  `SetVoxelsOp` would serve it — and that is wrong for a reason the file states:
  its payload is *"uninformative … the point"*, because the rows come from one
  image and the gather samples another, and *"with a producer that already
  carried the value, a gather that returned its own input would look right"*.
  `with_values` on a `Bool` image does emit `1.0` at every set voxel, so the
  numbers would match — but only by coincidence of the input dtype, and the
  test's argument would become circular. **A migration that keeps every number
  and destroys the reason is a migration that changes what the test asserts.**
  It stays hand-written and belongs in the deliberate list below.
* **One `Binarize` had a stated reason, and the reason had expired.** Its doc
  said it was *"kept here rather than taken from `src/ops` because what is under
  test is the consumer"* — true when written, since `src/ops` then had a
  threshold producing `1.0`/`0.0` in an `f64` buffer and none producing `Bool`.
  So the sentence was **inverted rather than deleted**, and the op migrated.

#### A note on G1 rather than a new identifier

`tests/collapsing_phase.rs` hand-writes `MaxAlongFirstAxis` — a maximum along
axis 0 writing extent 1 — and **no shipped op and no shipped probe does that**:
`BlockOp::output_shape` is overridden in exactly three places under `src/ops/`
— `lattice`'s two and `resample`'s — and none of them pins an axis; of the
probes, only `DecimateOp` overrides it, and it halves rather than collapses.
(A grep for `fn output_shape` finds three more in `voxelwise` and `background`.
Those are `Combine::output_shape`, a different method with a different
signature — fallible, over a list of branch shapes — and they are not ops
choosing an output extent.)

That is **not** a new gap, and the distinction is the register's own rule. G1's
subject is the *declaration*, and the declaration landed: an op that consumes an
axis whole and writes it at extent 1 can say so, and §8.1 records that this very
file is what settled it. What is missing is an op that **uses** it — which is
precisely the state G7 has a word for. **The collapsed half of G1 is
`buildable`, not built**, and the price is still being paid in full, by a test.
A new identifier here would cover a subject a row already covers, which §3's
line forbids; a row marked closed on the arrival of a mechanism rather than its
adoption is the thing §12.1 refused to do.

#### Stale hand-rolls — adoption, not absence

Recorded because a later sweep will find them and must not re-file them as gaps:

* **`Binarize`, three times — all three now migrated** —
  `masked_local_statistic.rs`, `masked_rank_filter.rs`, `seeded_watershed.rs`,
  all `f64 > level -> Bool` at reach 0. `ops::voxelwise::VoxelwiseMaskOp::threshold` does exactly this and is
  **G15 closed**; it post-dates all three test files by days, and only
  `mask_carrier*.rs` has adopted it. The `seeded_watershed.rs` copy is the one
  with no doc comment at all.
* **`Binarize` a fourth time**, in `sliding_histogram.rs`, is **not** stale and
  **not** a gap: it thresholds `u16` in the source's own units, and
  `VoxelwiseMaskOp`'s own doc names that case and sanctions the hand-roll —
  *"writes its own `BlockOp`, which is what `MaskFn` being a trait rather than a
  closed enum is for."*
* **`Negate`** (`seeded_watershed.rs`) — **migrated**; `VoxelwiseMapOp::new` serves it. Its
  doc argues that the cost volume is the caller's, which is true and is
  unaffected by whether the caller writes the op or names it.
* **`PointsOp`** (`point_offset_walk.rs`) — **migrated**. It **is
  `ops::rows::RowSourceOp` exactly**, the op G17 closed. `rows_group.rs`, `voxelize.rs` and
  `point_labels.rs` were migrated when it landed; this one was not, because the
  sweep that found it ran after.
* ~~**`MarkRowsOp`** (`row_table_ops.rs`)~~ — **withdrawn**: it is deliberate,
  not stale, for the reason given above. Listed here struck through rather than
  removed, because the next sweep will reach the same wrong conclusion from the
  same evidence unless it can see that this one was already tried.

#### Deliberately wrong, and therefore not gaps

Named so the next sweep stops here rather than re-deriving it. Each exists to
make a guard fire, and each would be *worse* as a library op, because a library
op that can be wrong on purpose is a library op somebody will be wrong with by
accident: `HoleOp` and `SparseOnly` (coverage), `UnderdeclaredWalkOp` and
`ShortReach` (a reach short by one — the second pins a wrong answer **no guard
sees**), `ForgetfulOp`, `ForgetfulOperand` and `CountOp` (a declaration made and
not honoured), `SilentMergeOp` and `ContradictoryOp` and `DriftingOp` and
`DriftingMergeOp` (a seam claim the fold contradicts), `MisplacingOp` (side
regions that do not tile), `WindowedOperandOp` (an operand wider than its halo),
`EmptyTable` (a map that gives a block nothing), `Silent` (a supplied image with
no element type), and `clearmap-ng`'s `BlockLocalTallyOp`, whose `name()` is
literally *"block-local tally (negative control)"*.

`MarkRowsOp` in `tests/row_table_ops.rs` joins this list by the correction
above: a producer whose payload must carry **no** trace of the voxel it came
from, so that the only voxel identity in the answer is the gathered column.

`UnkeyedSourceOp` in `tests/point_labels.rs` belongs to this list and is the one
that set the rule: when `PointSourceOp` moved into `points`, that one stayed
hand-written, **because its purpose is to key wrongly** and no library op should
be able to.

### 3.2 The free-function sweep — §3.1's stated blind spot, closed

§3.1 matched `impl …Op for`, and said so as a limit: *it would miss a
hand-written **free function** standing in for a missing library function.
Whether that class exists is unmeasured.* It is measured now.

**The corpus, and the asymmetry inverts.** §3.1 found 61 hand-written ops in
`blockflow`'s own tests against **one** in the consumer's. For free functions it
runs the other way: **when this sweep ran** `blockflow/tests/` held **1860** top-level free functions
and `clearmap-ng/tests/` **1438** — both are snapshots of a corpus six workers
are editing, and the first has already drifted upward since — and it is the *consumer* where the copies
pile up — `fixture_root` in 26 separate files, `read_npy` in 20, `differing` in
13. **`blockflow` hand-writes ops; its consumer hand-writes helpers**, and the
two sweeps needed different discriminators for that reason.

**A repeated helper is weaker evidence than a repeated op**, so the census was
not taken as the finding. Hashing each helper's *body* is: across `blockflow`'s
most-repeated names, `run` is 44 copies and **41 distinct bodies**, `scene`,
`chain` and `cases` are 100% distinct, and `plan` is 30 copies over 22 bodies.
Those names are a **convention** — every file has a `run`, a `plan`, a
`reference` — which is the opposite of duplication. The most-duplicated helper
in all 1860 is

```rust
fn workflow(chain: Chain) -> Workflow { Workflow::new(chain, VOLUME, Dtype::F64) }
```

at ten copies: a one-line constructor binding a **per-file constant**, which no
shared version could serve. Category D.

**The one structural cause, and it is the consumer's own.** Each `tests/*.rs` is
its own crate, so sharing needs a `tests/common/mod.rs`. `blockflow` has the
mechanism and uses it (`tests/patch_grid/mod.rs`). `clearmap-ng` has **no**
`tests/` subdirectory, **no** `mod` declaration in any of its 65 test files, and
no helper dev-dependency. That alone explains `fixture_root`×26, `read_npy`×20
and the `as_*` accessors **without implying anything about this library**. They
are D — the library owes callers ops and types, not a path convention.

**What survived as a real gap: one row, and it is a good one.** **G21**,
`differing_voxels`. It is not setup — it is the count every parity argument is
made of — and a `tests/common/mod.rs` would only relocate it, not resolve it.
**Eight** of its thirteen copies omit an extent check and `zip` truncates, so
they answer a *small* difference for two volumes of different extents: in a
parity suite, the failure reading as a pass. That is the only finding in this
sweep where the duplication is also **wrong**. Whether it has *bitten* — whether
any of the eight compares mismatched extents at the parameter points it runs
at — is §3.3, **and the row is closed there**: it was minted open by this sweep
and shipped inside the same batch, so "survived as a real gap" is what this
sweep found, not the row's current state.

**Two more, recorded and not minted:**

* **`intensities`, ten copies of a redundant loop.** They copy
  `Rendered.intensity` — already an `Array3<f64>` of exactly the wanted shape —
  into an identically-shaped `Array3<f64>`, voxel by voxel. `tests/ridge_filter.rs`
  already writes the one-liner `scene().render().intensity`. Duplication with the
  library shipping precisely what is wanted, so **D**, not a gap; the cleanup is
  in ten test files and belongs to whoever owns them.
* **`sha256`, four independent FIPS 180-4 transcriptions** in `clearmap-ng`,
  three returning `String` and one `[u8; 32]`, of which **only `fill_oracle.rs`
  carries NIST vectors** — the other three are validated against recorded
  manifests alone. `blockflow` has no SHA-256 and should not: `digest() -> u64`
  is a non-cryptographic fingerprint and a general image-processing framework is
  not where a hash belongs. The observation that is worth routing is in the
  consumer, not here — `clearmap_ng::oracle::registry::ClearMapDigest::sha256_prefix()`
  is a **public accessor for a SHA-256 prefix in a crate graph that ships no way
  to compute one** — and the four copies' comments state the omission as a
  deliberate dependency-budget choice. Not a row in *this* register.

**And the verdict distribution is the result.** Of the twelve repeated helpers
examined, **one** was a gap, three were stale hand-rolls against `ndarray` or
`npy` conveniences the library already ships, and **eight were D — neither**.
§3.1's ratio was the reverse. A sweep that returned "gap" as often for helpers
as it did for ops would have been a sweep measuring its own enthusiasm.

### 3.3 Has G21 bitten? No — and the distribution of *why* is the finding

G21 is not merely duplication: eight copies of `differing` omit the extent check
and `zip` truncates, so a mismatch answers a *small* difference count — in a
parity suite, **the failure reading as a pass**. Whether any of the eight is
comparing mismatched extents *today* was unestablished, and it is the urgent
half. Every call site of all eight was traced to where its arguments' extents
are fixed.

**The answer is no.** No comparison in the suite is currently over two volumes
of different extents. The recorded surfaces were read directly rather than
assumed: `vasc-tile-py/tile-p40` and `vasc-tile-col-on` are `(404, 1304, 3369)`
on every surface; `vasc-pipeline-axial` is `(96, 96, 32)` on all seven;
`cellmap-parity/stageprobe` is `(128, 128, 64)` and `stageprobe384` is
`(384, 384, 256)`, each consistent across `source`, `background` and
`maxima_raw`. Nothing is silently passing.

**But the eight do not divide evenly, and that is what to act on.**

| file | sites | why it is safe |
|---|---|---|
| `histogram_percentile_oracle.rs` | 6 | **structural.** Reads no `.npy` at all; every array is constant-shape or shape-preserving from one source |
| `planner_machine.rs` | 4 | **structural.** No recorded array; every arm is one plan over one `fixture(volume)` under different *machine* settings, and worker count and block edge cannot reach an extent |
| `smooth_oracle.rs` | 6 | **structural.** No `.npy`; all from one `source()`, and the one cross-implementation site has its right operand re-pinned by `Array3::from_shape_vec` |
| `planner_win.rs` | 3 | **structural.** One recorded array, asserted `shape() == TILE`, from which both arms of every comparison descend |
| `binarize_gap.rs` | 3 | 1 structural; **2 by fixture** — computed-vs-`binary.npy`, loader checks rank only |
| `cells_gap.rs` | 4 | **by fixture** — vs `maxima_raw.npy`. The `theirs` arms are hard-pinned by `from_shape_vec`, so only the `ours` arms are exposed |
| `workflow.rs` | 7 | 2 structural; **5 by fixture**. The one real shape assertion in the file pins the *source*, not the comparands |
| `composition_tile.rs` | 8 | **by fixture, and nothing downstream would catch it** — see below |

Four files cannot differ **under any fixture**. Four are safe because the
recorded directory happens to be internally consistent — *"the guard is data,
not code"* — and `fixture_root()` is overridable by `CLEARMAP_NG_FIXTURES`, so
the property is environmental.

**`composition_tile.rs` is the one to name.** Each of its links runs a
shape-preserving stage over one recorded surface and compares the result against
a **second, independently loaded** recorded surface; `recorded` validates rank
and nothing else; and the number that could have caught a truncation —
`link.produced` beside `link.recorded_count` — is **printed and never
asserted** (only `produced > 0`). So a truncated `zip` yielding `differing == 0`
would pass the link, pass the report, and pass `assert!(parted.is_empty())`.
That is a parity claim resting on a comparison that cannot fail, and it is the
same shape as a fixture-mode defect **arriving in the comparator instead of in
the fixture**.

**Two things already in the tree that argue the fix.** `cells_gap.rs` carries an
implicit extent check for a *different* array — it demands a changed-voxel count
equal `source.len()`, which fails if `background.npy` is short — and has no
equivalent for the array it actually compares. And `oracle/mod.rs`'s
`Comparison::run` **does** compare shapes before counting. So the consumer has
got this right twice (there, and in `skeletonize::removed`) and unguarded eight
times, which is what a private function in the framework buys.

**What this changes about G21: nothing about the prescription, everything about
the urgency.** It is a hygiene fix in four files and a real exposure in four
more, none of the *test* copies currently wrong.

**And the row closed while this was being traced, with the sharper half of the
finding.** `pub fn differing_voxels` now exists — so the migrations are
unblocked and land with each file's owner — but what closing it turned up is
that **the framework's own copy had a worse variant of the same defect**. It
*did* compare shapes; it answered `None`, and the single call site wrote
`.unwrap_or(0)`, so *"these are different shapes"* and *"there is nothing to
compare"* both became **zero differing voxels**. Read together, the comparator
was wrong at three levels and right at two:

| where | guard | what a mismatch produced |
|---|---|---|
| eight consumer test copies | none | a small count, silently |
| `strategy::differing_voxels` (was) | present, then **swallowed** by `.unwrap_or(0)` | **zero** — indistinguishable from agreement |
| `skeletonize::removed`, `Comparison::run` | present and propagated | an error |

The tests were the *safer* of the unguarded two, because their mismatch would at
least have produced a non-zero count on differing data; the framework's collapsed
to the one value that reads as success. That is why the sweep's answer here —
**no, it has not bitten** — is about the eight copies and must not be read as
covering the crate: **the guard that was already there had been turned into its
own negation one line away, and only closing the row found it.**

The one repair that should not wait is still `composition_tile.rs`, and it is
independent of G21: `assert_eq!(link.produced, link.recorded_count)`, a number
the test already computes and already prints.

#### What was repaired, and what the public function could not reach

**`composition_tile.rs` got both nets.** Its local `differing` now asserts the
extents — one guard covering all eight of its call sites — and both tile tests
now assert `produced == recorded_count` per link, *after* the parity assertion
so a real divergence names itself first. Those two counts are taken over the
**whole** of each array while `differing` walks them zipped, so with parity
holding they are equal by construction and can part only if the extents differed
and the comparison truncated. That is the one failure `differing` cannot report.

**Of the eight, three could call the new function and five could not**, which is
the part of G21's prescription that did not survive contact:
`differing_voxels` takes two `Voxels`, and only `planner_machine`,
`planner_win` and `smooth_oracle` compare `Voxels`. They are migrated and now
inherit the refusal. The other five compare `Array3<bool>` or
`ArrayView3<f64>` read straight off recordings, and wrapping those to borrow one
assertion would clone a pipeline volume — so they get the assertion locally.
**A `Voxels`-shaped signature serves the framework's own callers and about half
of the consumer's**, and an `Array`-shaped sibling is the missing half; that is
routed rather than built, since `voxels.rs` is not this pass's to take.

**One had a reason and keeps it.** `histogram_percentile_oracle.rs` compares
`to_bits()`, under which two `NaN`s of identical bits agree; `differing_voxels`
compares with `!=`, under which they differ. For an oracle asserting that two
implementations produced *the same `f64`*, bit identity is the claim and `!=`
would be a weaker one. It keeps its local comparison and gains only the extent
check — the same discipline that kept `MarkRowsOp` hand-written in §3.1.

**Two were not touched and are reported instead**: `cells_gap.rs` belongs to the
oracle worker by the `tests/cells_*` rule, and `binarize_gap.rs` was being
edited by another hand minutes before this pass reached it. Both are
`Array3<bool>` comparisons, so both want the local assertion rather than the
migration.

**What is verified, and what is not.** All six edited files **compile** —
`cargo check --release` over the six targets, clean. **The four that may be run
were run and pass**: `planner_machine` 4, `smooth_oracle` 11,
`histogram_percentile_oracle` 8, `workflow` 21 (7 ignored) — **44 passed, 0
failed**, so neither the migrations nor the new extent assertions fire on any
fixture in the suite, which is the same answer §3.3 reached statically. The
remaining two are both on the do-not-run list: `planner_win`, and
`composition_tile.rs`, whose two new assertions are **unverified by execution
and deliberately so**: its fixture is 45.1 GiB and it is on the
do-not-run list, so the change was made and left for a quiet box rather than
silently skipped. What makes that safe to land unverified is the shape of its
failure — a one-line equality between two counts the test already computes,
whose only way to fire is the mismatch it was added to catch. A computation
would not have been safe to land that way. The command is

```text
cargo test --release --test composition_tile -- --ignored --nocapture
```

### Raised, and explicitly *not* framework gaps

B in particular was careful to separate these, and the distinction is the most
perishable thing in the four documents — a reader skimming for gaps will
mis-file them within a year. They are listed here so that cannot happen.

| finding | why it is not a framework gap | what it actually needs |
|---|---|---|
| **second moments, orientation, principal axes, eccentricity** | a **range** problem, not an expressiveness one. `Σx²` about the volume origin is `O(L⁵)` and overflows `u64` at `L ≈ 1800`, where the first moment survives to about 65 536 | re-centre on the object's **own bounding-box minimum**, which is already a `detect` column, is a function of the component alone and is therefore decomposition-invariant; or accumulate in `i128`. Arithmetic, not architecture — and it unlocks four of the most-used `regionprops` fields in one move |
| **surface area, perimeter, per-object Euler number** | phase 0 is halo-free, so a boundary voxel cannot tell a neighbour outside the *volume* from one outside the *block* | a halo of one on the labelling phase plus a stated **face-ownership rule** at the seam. A design, and a small one |
| **convex hull, solidity, exact Feret diameters** | the merge is associative and commutative but the fragment is a variable-size point set rather than a fixed-width accumulator | either variable-size fragments with a hull merge, or the support function on a fixed set of *K* directions — `max(x·d)` and `min(x·d)` per direction, merged by `max` and `min`, exactly. **2K columns and no framework change** |
| **`intensity_std`, per-object histogram, a bounding box on `tabulate`** | all merge trivially; simply not carried | columns |
| **arithmetic combines, arbitrary-kernel convolution, top-hat, morphological gradient, border clearing, area opening, marker-based selection** | every one is either a composition the crate can already express or `ops::components`' program with a different per-component fact | ops nobody has written. See §5 |
| **`wrap` and `constant` boundary conventions** | a boundary convention changes **no reach** — a fold brings an offset that left the array back toward the edge it left by. Stated in `ops::smooth`'s header and tested | a per-op parameter, and a shared convention for spelling it |
| **track linking, gap closing, splits and merges; graph cuts; random walker; Felzenszwalb superpixels** | global optimisations whose working structure is a cost matrix, a max-flow or a spanning tree — not a volume. No `Reach` describes them | nothing. Out of scope for the block layer; the crate should produce the tables and stop |
| **a composite or colour image as a plan image; false-colour and display LUTs** | a plan image is one element type and a composite is a tuple; and a colour is a display artefact | nothing. Out of scope, and both VTK and this crate already draw the line in the same place |
| **splitting a composite into per-channel volumes** | under the N-volume model it is the identity, and under the storage contract below it is a read-time axis selection | nothing. **This is the model paying off** |

---

## 4. Where the four agree

Convergent recommendations are worth more than any single document's, because
two families reaching the same conclusion from different operations is evidence
about the framework rather than about one survey's taste. Each of these was
checked against the sources and, where cheap, against the code.

**1. No globally consistent label volume.** *(B §4, §14; verified against
`src/`.)* `ops::fill` phase 0 already writes block-local labels as a `u32` image
and phase 1 already closes them into global components with a union-find — and
then rewrites them into a *mask*. The missing third phase is the shape
`fill.rs:91` names in its own header: "the three-phase shape — label, merge,
relabel". Verified: no op under `src/ops/` produces a label volume, while
`ops::tabulate`'s header opens "One row per region of a **label volume**", so
the crate's most complete per-object measurement cannot be driven by the crate's
own segmentation. `ops::label` stamps scattered points and is not this. D
depends on the same missing op for per-object traces over a series. Nothing else
in B unblocks as much.

> **Built — and the convergence was right about what was missing and wrong about
> its shape.** `ops::label` now carries both. The paragraph above is kept
> **unaltered**, including the sentence that is now false — *"no op under
> `src/ops/` produces a label volume"* — because what it got wrong is the part
> worth having on the record, and an absence that is edited out of existence
> when it lands takes its own history with it.
>
> *What the convergence said:* the missing third phase is `fill.rs:91`'s "label,
> merge, relabel". **That phase cannot exist.**
> `fragment::check_phase_work` refuses a pixel phase after a fragment-only one —
> image `p+1` would go unwritten and phase `p+1` would read an image nobody
> produced — so the three-phase shape is unplannable, and `fill.rs`'s own header
> already said so in a paragraph the convergence did not reach. The merge folds
> into the relabelling phase exactly as it does in `fill`, which means every
> block re-runs the whole union-find *and*, because a whole-lattice fragment
> reach is also the halo, every block re-reads the whole label image. That is
> not a detail of the implementation; it is most of what the measurement below
> is about.
>
> *What shipped, and it is two things rather than one, on purpose.* The merge's
> answer is a **table** — one `u32` per `(block, local label)`, 92-149 KB against
> a 268 MB label volume on the volume measured — and there are two things to do
> with it. `RelabelComponentsOp` **materialises**: a `fragments -> volume` phase
> writing a second `u32` image, the shape the framework admits.
> `RelabelledEnvironment` **decorates**: an `Environment` that applies the table
> to reads of the first image as they are served, correct at any read extent, so
> a consumer's lattice need not be the labelling's. The second subsumes the
> first — a trivial identity op over a decorated environment writes the
> materialised volume, which is one mechanism and not two, and
> `tests/global_labels.rs` asserts the two produce the same bytes.
>
> *What the convergence could not have seen, and it is the load-bearing part:* a
> union-find root is a correct **partition** with a **decomposition-dependent
> name**. Writing `find(node) + 1` into the volume gives a label volume whose
> labels change when the block size changes, and every consumer that *stores* a
> label — a table of regions, a graph whose vertices are labels, anything written
> to disk beside another run — is then wrong in a way no per-voxel comparison
> catches. So the numbering is a rule about the volume: components are numbered
> in the order their lowest voxel is met in a row-major scan of the **whole
> volume**, which is `label_members_into_with`'s own within-block rule lifted.
> It costs one `u64` per block-local label in the fragment and `Union::fold_min`
> to fold it, and it is what makes the blocked answer **byte-identical** to the
> whole-volume reference rather than a relabelling of it.
>
> *And the consumer needed a change to be drivable at all.* `ops::tabulate` is
> the op this row exists for, and `TabulateValuesOp` declared no element type on
> its operands — so a **supplied** label volume, which is what a consumer of an
> earlier run is handed, was refused by name at plan time because nothing in the
> plan could say what it held. `TabulateValuesOp::holding` closes that. The
> convergence said the crate's measurement could not be driven by the crate's own
> segmentation; it turned out there were *two* reasons and only one of them was
> the missing op.
>
> *Measured, on a recorded volume, and the answer has a condition.* See G7's row
> for the numbers and `tests/label_materialisation_cost.rs` for the run. The
> short form: **decorate**, and materialise only through the identity op over
> the decorator — but the reason is not the one the framing suggests. Avoiding
> the write is worth 8 bytes a voxel, about `1.05-1.45x`; everything beyond that
> is the halo. **This is a price for G7, not a property of laziness.**
>
> *And the expiry condition needed correcting, which is recorded rather than
> quietly fixed.* The first version of this note said that if G7 closes the
> recommendation goes to about `1.2x` and becomes marginal. **That counted
> pixels only.** There are two amplifications and they are in different
> currencies: the relabelling phase's whole-lattice fragment reach becomes a
> whole-volume *halo* (pixels, `blocks x` the label image) **and** a per-block
> *gather of every fragment* (`(1 + blocks) x` all the fragments, the second
> factor growing too because the fragments are faces and cutting finely makes
> more face — `blocks^(4/3)` on a cubically-cut lattice, and how much depends on
> *which axis* is cut, which `docs/design/barriers.md` §7.6 sweeps). At 256 blocks those are 67.4 GiB and 34.9 GiB against a decorated total
> of 4.2 GiB. **A barrier removes the first and leaves the second**, so closing
> G7 alone takes the gap from `25.4x` of total traffic to about `9.4x` — better,
> and not marginal. Marginal needs the merge to run **once** as well, which is
> the three-phase shape, which is blocked by the image-numbering rule and is a
> different gap. `docs/design/barriers.md` specifies both and says which is
> which.

**2. A K-ary reach-0 op shell — G10.** *(B §10(b) and §12; D §3.6, §6, §9,
independently.)* One shape unlocks linear unmixing, stain separation, crosstalk
correction, colour-space conversion, per-voxel classification, channel argmax
and every windowed temporal filter at once. Verified in D's document and again
here that it must be a `BlockOp` with `source_inputs` and `side_outputs` rather
than a `Combine`: the `Combine` trait has no side outputs and so cannot write
more than one array. Two families asking for the same shell from opposite ends
is the usual sign.

> **Built — and the convergence was right about the shape.** `ops::mixing` is
> the shell, `TupleKernel` the kernel trait, `LinearMap` the per-voxel matrix.
> No new axis; not a `Combine`, for the reason both documents gave. The extra
> inputs are supplied images, which is why this had to wait for G5 and why the
> two rows closed together. **One clause of the shape was wrong and has since
> been fixed:** `BlockOp::apply_side` was not handed the `SourceInputs`, so the
> outputs beside the primary could not be computed where the trait says they are
> computed. The argument is now threaded from the executor and the workaround is
> deleted, so the convergence's own sentence — *one `BlockOp` with `source_inputs`
> and `side_outputs`* — is true as written. **Both documents asked for the shape
> and neither could have seen the clause**, which is the usual shape of a
> convergence being right: it fixes what to build and not what the trait has to
> be handed to build it.
>
> Two smaller things the convergence did not say, both measured and both in
> §11.1: the shape is **streaming-bound, not compute-bound** — 4.0 flops per
> byte of image traffic at `K = K′ = 16` in `f32`, not the 2 that gets quoted,
> which holds only at `K′ = 1` — and **tiling the output loop is worth 2.5–2.8×**,
> which is larger than any coefficient anyone would argue about.

**3. Two-volume arithmetic.** *(A §2, §13; echoed by B for top-hat and
morphological gradient, by C, and by D §4.2 as "family A's largest cheap gap,
seen from the channel side".)* Verified in `src/ops/voxelwise.rs`: `Logic` has
exactly `And`, `Or`, `Xor`, and `ops::background::DifferenceCombine` is the only
arithmetic combine in the crate. Add/sub/mul/div/min/max as `Combine` sinks on
`ops::background`'s proven diamond, plus convolution with an arbitrary (and
separately, an arbitrary separable) kernel, unblocks difference-of-Gaussians,
unsharp mask, Laplacian, gradient and Sobel, highpass-by-subtraction and the
whole image-calculator family. **One precision the four documents leave
implicit:** arithmetic between two *branches of one plan* is bounded reach and
zero framework change, and that is what A is claiming; arithmetic against a
**supplied second acquired volume** additionally needs G5. Both are wanted; only
the first is free.

> **Updated: the second is now free too, and neither has been written.** G5 has
> landed, so a supplied second acquired volume is an image and the distinction
> above no longer separates a cheap case from a blocked one — it separates two
> cases that both want the same unwritten `Combine`s. The convergence is
> unchanged and so is its ranking; what changed is that nothing in the framework
> stands behind either half of it any more.

**4. Category 2 is empty.** *(A §1 by grep; C §2 citing A; verified again
here.)* No op declares `AxisReach::All` on a single axis — the only occurrences
of the variant under `src/ops/` are prose in module headers, and the only
`Reach::all()` is `ops::watershed`'s whole-volume declaration, which is
category 3. So **the framework can already express a shape nobody has used.**
A wants it for a per-axis transform pass and an integral image; B's exact
Euclidean distance transform is category 2 and lives in a sibling crate; C notes
that a separable per-axis resampling or warping sweep is exactly this shape.
Four families, one unused variant.

> **Corrected — measured, then overtaken.** The convergence is real and the
> *reason* the four gave for it was not. "Nobody has written one" is true of
> `src/ops/`; at the time of the experiment **`AxisReach::All` in the source
> frame could not have been written**, being refused at every extent, on every
> grid, under every halo. §8.5 changed that: it is now granted on a consumed
> axis and is the right way to declare one. The variant the sibling crate's
> distance transform actually uses is the **phase**-frame one, whose numbers are
> measured against the array the phase itself writes, and it works — §8.4 states
> what that op is and is not evidence for. The two are one word apart in a call,
> they are not the same claim, they now behave differently rather than one being
> impossible, and no document separated them.

**5. A scalar broadcast inside an iterative phase — G8.** *(B §6 and §12; D
§4.3 reaching it from Costes automatic thresholding.)* The per-iteration update
of a level set or of SLIC is a local stencil, which `iterate` already handles at
one substage's reach however many run. What is missing is a substage able to
consume a scalar reduced over the previous substage's whole output. Verified:
`src/iterate.rs` has no broadcast or scalar mechanism. It is the global-threshold
gap with a loop around it, and it stands between this framework and the entire
variational-segmentation family.

---

## 5. Corrections the four made to the brief they were given

A survey that hides its own errors is worth less than one that records them.

* **C: a projection's problem is the declaration, not the rank.** The brief and
  A's G1 framing both hold that a projection is not expressible because `Reach`
  states per-axis halo widths on a same-rank output. C established that a
  projection along axis 0 of an `[N, Y, X]` volume produces `[1, Y, X]`, which
  **is** a legal 3-D volume — the crate is explicit that a 2-D problem is a
  volume of depth 1 — so `output_shape` can return it. What breaks is the
  **truthful declaration**: `AxisReach::All` on axis 0 in `Space::source_voxels()`
  resolves against an extent-1 axis to `(1, 1)`, `Frame::Source` correctly denies
  the clamp exception at what is an interior position of the array being read,
  the trustworthy extent is empty, the valid region collapses and the tiling
  check fires. The available route is the `†` cross-grid escape, making the
  correct classification **2†** rather than X. **Flagged unverified by C
  itself**: the argument was from reading `reach.rs`, `geometry.rs`,
  `decomposition.rs` and `ops/lattice.rs`, with no projection op built and no
  plan run. C named a ~50-line experiment as what would settle it, and said it
  should be settled before anything is built on it.

  > **Settled — measured.** The experiment was run
  > (`tests/collapsing_phase.rs`, twelve tests). **C is right, to the
  > arithmetic.** `[X, Y, Z] -> [X, Y, 1]` plans, runs, and is byte-identical to
  > a whole-volume reference across 25 cuts of the two free axes. The `2†`
  > classification stands and is **no longer unverified**; A's and D's `X (G1)`
  > for projections is withdrawn, in the direction this index already suspected.
  > C's mechanism is confirmed line for line, and the experiment found three
  > things C could not have seen from a read — the refusal is unconditional, the
  > phase-frame declaration is vacuous rather than wrong, and nothing checks the
  > fetch. All three are in §8.
  >
  > **And then the first and third changed.** §8.5 landed: the refusal is now
  > conditional — granted on an axis the op consumes, kept where the axis is cut
  > without a whole-axis halo — and a declared whole-axis reach **is** checked
  > against the fetch. C's `2†` is unaffected, because the fetch is still stated
  > per block; what changed is that C's "truthful declaration" is now available
  > and is the one to use. The vacuity of the phase-frame form is unchanged and
  > is now a named trap.

* **D: the strongest argument for a channel axis argues against it.** Linear
  spectral unmixing was carried into the brief as the case that most obviously
  needs simultaneous access to N channels — and D established that it is **reach
  0 along the channel axis**. So are stain separation, crosstalk correction,
  channel arithmetic, colour-space conversion and per-voxel classification. The
  operations that need all N channels at once need them *at one voxel*, which
  wants **arity**, not an axis; an axis would give them an index and charge every
  op in the crate a rank. D's tell: a channel axis is one the planner must never
  cut, so a fourth axis would need G9 (`FullExtent(axis)`) immediately —
  "an axis the planner may not cut is not an axis; it is arity with extra
  steps." The layer below agrees independently: the storage contract already
  puts the channel axis in the metadata and the addressing with a chunk extent
  of 1 along it, and projects it away at the read boundary at no cost.

* **All four: the 3-D floor is a decision, not a cap.** Every one of the four
  documents reads `Voxels` being `Array3`-only as a limitation to work around —
  A's G1 bullet ("`Voxels` is eleven `Array3` variants. A phase cannot produce
  an (n−1)-D output"), B's restatement of it, C's §7 conceding the point before
  arguing past it, and D's §1, which puts the fact at the centre of its document
  and then reasons from it. **It is the opposite of a cap.** `src/voxels.rs`
  argues for rank 3 on measured grounds against `ArrayD`: an image *is* a volume,
  one and two dimensions are modelled as three with degenerate axes, and a
  dynamic rank "bought nothing and cost an indirection per index." The floor
  exists so that lower rank costs **no new element types** — which is exactly
  the property that makes a projection's output legal (§8.1). D quotes that
  header correctly and then builds part of its channel-axis argument on the
  other reading; **D's conclusion — N volumes, not a fourth axis — is untouched
  by this and stands**, because it rests on the reach-0 sort of **D's §3.1** and
  on the storage contract of **D's §3.4**, neither of which depends on how the
  rank cap is read. *(Both were bare `§3.1` and `§3.4` when this document had no
  §3 subsections of its own. It now has §3.1–§3.3 and no §3.4, so the bare form
  pointed a reader at the test-suite sweep and at nothing.)* What is corrected
  is the reasoning, not the answer.

* **D, correcting B: candidate-pair generation is not an instance of G2.** B
  handed tracking to D with the reading that a linking phase needs two inputs at
  different offsets along the series axis. D established that the hard half is
  elsewhere: **its decomposition is spatial while its payload is tabular.**
  `ops::rows`' header is emphatic that a row op decomposes by row range with *no
  overlap*, because an overlap duplicates a row and no downstream check can tell
  the duplicate from a real one — and a pair generator needs a *spatial*
  neighbourhood, which a row range does not give it. That is a genuinely new op
  shape, registered here as **G11**, and it is the finding rather than the
  missing feature. B's own G2 instance was this hand-off, so with the correction
  applied G2 blocks A, C and D rather than all four.

---

## 6. What this index had to adjudicate

Places where two documents say different things about the same thing, and the
reader would otherwise have to guess.

| the question | A | B | C | D | adjudication |
|---|---|---|---|---|---|
| what number is "not expressible" | 5 | 6 | 5 | 5 | **neither.** `X`, per §2 — it is the absence of a shape, and giving it a number is what made the collision possible |
| what number is fragment-and-join | unnumbered | 5 | unnumbered (used twice) | unnumbered | **5**, per B, the only document with it as a category |
| is a projection expressible today | X (G1) | — | **2†**, unverified | X (G1) | **C, and now measured.** `tests/collapsing_phase.rs` settles it: `2†`, verified, decomposition-invariant. A's and D's `X (G1)` is withdrawn. A and D remain right about the *desirable* form and wrong that nothing can be built today — and the desirable form is not the one any of the three named; see §8.1 and §9. **Since §8.5 landed it is also *declarable*, and `2†` still stands** because the fetch is stated per block |
| can a projection state the dependency it has | no | — | no, only the marked zero | no | **yes, since §8.5** — `AxisReach::All` in `Space::source_voxels()` plus `with_sources`, checked against the fetch. All four documents predate it and all four have been corrected in place. This is the only row where the answer changed because the crate changed rather than because a document was wrong |
| is `Voxels` being `Array3`-only a limitation | treated as one | treated as one | treated as one | treated as one, and reasons from it | **none of the four.** It is a deliberate floor: 1-D and 2-D are 3-D with degenerate axes, specifically so that lower rank adds no element types (`src/voxels.rs`). D's conclusion is unaffected; the reasoning is corrected in §5 |
| is the sibling distance transform evidence for the collapsed-axis route | — | cites it as the category-2 instance | — | — | **B's citation is correct and narrower than it reads.** It is a same-rank whole-axis stencil in the **phase** frame taking neither escape. It is not evidence about `Frame::Source` and not evidence about a collapsed axis; §8.4 |
| is object linking G2 | — | yes | — | **no: a new shape** | **D**, per §5. Recorded as G11 |
| is a **supplied** shading or flat-field reference blocked | "G2 does not block it — `SourceInput` handles same-extent second images" | — | — | blocked by **G5** | **D**, and verified here: `ArrayEnvironment::new(input, n_phases, chunk)` seeds image 0 from one `Voxels` and creates every other image `pending`, to be written by a phase. A *supplied* array cannot be an image. A's row is the one factual error the four documents contain. **Since G5 landed the answer is "no longer blocked", and D is still the one who was right** — A reached the right end state by the wrong route, and `ArrayEnvironment::with_inputs` is the method whose absence made A's row wrong. One residue is now genuinely A's: the reference has to be in **image 0's coordinate space**, so a phase reading one after a resample is refused by name. That is G2's row, not G5's |
| are side outputs terminal | asserted terminal | **unverified** in §10, "family A established … terminal" in §12 — B contradicts itself | — | unverified, citing B | **terminal, and now verified here.** `SourceInput.image` is an *image* index "in the same numbering `Chain::Source` and `PhaseDecomposition::source_images` use", and side outputs land in a separate named map on the environment, not in `images`. No later phase can name one. So the cheap G1 route works only for terminal results, exactly as A said. **Unchanged by G5, and now pinned by a test** (`tests/tuple_map.rs`, `a_side_output_is_not_an_image_and_cannot_be_named_as_one`): a supplied input is an array that existed before the run and a side output is written during it; the two are addressed differently — an image by number, a side output by name — and `Chain::source` and `SourceInput::image` take a number. Making a side output readable would be a *third* thing, an image written by a phase that a later phase reads, which is what an ordinary image already is |
| is a whole-volume histogram X or category 5 | X (G1) | 5, at N+1 passes | — | X (G1) | **both, about different halves.** The *fragment* route exists and is expensive — that is G7. The *image* route, where a later phase reads the reduction back, does not exist — that is G1. Neither document is wrong; a reader comparing them would think one was |
| does family C need G5 | — | — | frames mosaicking as G2 + C2 only | mints G5 | **C needs it too.** N tiles are N acquired arrays; getting them into one run is G5, placing them is G2, and re-origining after the solve is C2. C's document predates G5 and so does not name it. **With G5 closed the split is sharper than it was, and it favours C's original reading of its own problem:** the tiles can now be handed to one run, and mosaicking is exactly as blocked as it was, by G2 and C2. Getting them in was never the hard part; it was the part nobody could do |
| who owns arithmetic combines | claims them (§2, §13) | — | "family D's to specify" | "family A's, §2" | **A**, two to one, and A is where the diamond pattern they build on already lives |
| does family C have a category 4 | — | — | **no instance** | — | recorded as an empty cell, not a divergence. A registration search is iterative; C puts optimisation outside the library, so the row is empty by design |

---

## 7. Two facts about the repository, stated plainly

**`docs/design/BLOCK_OPS.md` is cited from many module headers and does not
exist in this repository.** Verified **when the four documents were written**:
45 citations across 28 files under `src/` and `tests/`, and no such file
anywhere in the tree. *(Both halves have been re-checked since and only one
moved. The file is **still absent** — that is the fact, and it has not changed.
The count has: other workers have removed citations while editing the headers
that carried them, and a re-count today gives **36 across 20 files**. The
figure is kept with its date rather than chased, because it is the *absence*
that is the finding and a count that drifts is not evidence about it.
`docs/` has also grown: when this was written it held only `ops-survey/`, and
it now holds six documents under `docs/design/`.)* Every design argument attributed to it
in the four documents is
therefore cited from the *header that quotes it*, not from the document. Where a
header paraphrases rather than quotes, a survey may be one restatement further
from the original than it appears. **Do not go looking for it.**

**`tests/no_domain_vocabulary.rs` cannot see these documents.** It walks
`src/` and `tests/`, keeping paths whose `extension == "rs"`. Nothing under
`docs/` is scanned, by any of its three tests. All four workers applied the
vocabulary rule anyway and verified with their own case-insensitive substring
scans, as did this index, as did the correction pass that added §8 and §9 —
which had particular reason to, since the sibling module it read for §8.4 is
full of the vocabulary and none of it travelled. **That is a convention these documents hold themselves
to, not something enforced.** If it matters that it stay true, the scan needs a
second root and a second extension — which is a change to a test, and no
document may make it.

---

## 8. The collapse experiment, and what it settled

Written after the four documents, from `tests/collapsing_phase.rs` — twelve
tests around one op, a maximum along axis 0 of an `[11, 13, 17]` volume into
`[1, 13, 17]`. Everything in this section was measured against the tree as it
stands. The section exists because three of the five documents state something
this settles against them, and a survey that quietly rewrote itself would be
worth less than one that says what it got wrong.

### 8.1 `[X, Y, Z] -> [X, Y, 1]` works today, and G1 is misnamed

**What A and D said:** a projection is `X (G1)` — "not expressible today",
because `Voxels` is eleven `Array3` variants and a phase cannot reduce rank.
**What C said:** `2†`, expressible through the cross-grid escape, and flagged
unverified. **What was measured:** C.

* A collapsing phase **plans, runs and is decomposition-invariant.** Byte
  identical to a whole-volume reference across 25 cuts of the two free axes —
  edges 1, 2, 5, 6, 13 on one and 1, 3, 4, 8, 17 on the other, so that some
  divide the extent, some leave a short last block, and some divide neither.
* It works **only through the escape**: `PhaseDecomposition::with_sources`
  stating the fetch per block, plus `Reach::none()` in `Space::source_index()`
  — the *marked* zero. That is, it works by **not declaring the dependency**.
  C's mechanism is confirmed to the arithmetic, and C's price is confirmed with
  it: no automatic planner can produce such a phase.

  > **Superseded by §8.5.** It now also works **with** the dependency declared,
  > and that is the route to use: `AxisReach::All` in `Space::source_voxels()`
  > **plus** `with_sources`. The escape still plans and is now *strictly
  > weaker* — it records **that** a dependency exists, where the declaration
  > records **what would satisfy it** and is checked against the fetch. The
  > price C named is unchanged: a per-block fetch still means a hand-written
  > builder, so `†` still applies.
* **The truthful declaration is refused, and the refusal is unconditional.**
  `AxisReach::All` in `Frame::Source` is refused with the tiling message; the
  cause is upstream, in `BlockGeometry::derive_with`, where `Frame::Source`
  denies the clamp exception at both ends so that `trust_lo = read_lo + extent`
  and `trust_hi = read_hi - extent` cross for **every possible read**. Proven
  over extents 1, 2, 5 and 32 and every block edge, including `Reach::all()`
  offered as the halo. **No halo satisfies it at any extent on any grid.** This
  is not a guard that a wider halo appeases; it is a statement the framework
  cannot make.

  > **Superseded — this bullet described the tree before §8.5 landed, and is
  > kept because it is what the refusal was and because the shape of the fix
  > follows from it.** The exception is now granted *per axis*, on two
  > conditions that make it mean something: the reach on that axis is
  > `AxisReach::All`, and this block's read spans the whole of that axis
  > (`geometry.rs:319-322`). So on an axis the op **consumes**, every halo
  > satisfies it — including none — because there is no beyond and therefore no
  > neighbour a halo could have reached. On an axis that is **cut** with a
  > finite halo, no block spans it, every block stays degenerate and the tiling
  > check fires exactly as this bullet describes. The old sentence is true of
  > the cut case and was over-general.
* **The phase-frame `All` plans and is vacuous.** The same words in
  `Space::phase_voxels()` are accepted and the phase runs correctly — but on a
  collapsed axis `is_whole` requires `extent > 1` (`reach.rs:322`), so an
  extent-1 axis is not a barrier and the declaration states nothing. It is
  accepted *because* it says nothing, which is a worse position to be in than
  being refused.
* **The op's own declaration was never the obstruction.** `Chain::reach_spec`
  accepts `AxisReach::All` on one axis against its symmetric bound in every
  space. What refuses is the plan's geometry, and only in the source frame.

**So G1's name was wrong.** "No rank-reducing phase" describes a problem the
crate does not have: `[X, Y, 1]` is a legal `Array3`, nothing needs a rank
change, `output_shape` returns it, `Voxels` holds it and the tiling check
accepts it. A degenerate axis *is* how this crate models lower rank, on
purpose (§5, the 3-D floor). The precise gap is one level down and is about the
**geometry**, not the type: *the geometry cannot declare a collapsed or a
broadcast axis*. The identifier stays `G1` so every citation in the four
documents survives; only the name and the framing change.

> **Narrowed again by §8.5, and both earlier names kept.** The collapsed half of
> that name **closed**: a collapsed axis can now be declared truthfully, and the
> declaration is enforced. What is left is the broadcast half, so G1's name is
> narrowed a second time — *the geometry cannot declare a pinned (broadcast)
> axis*. The identifier is unchanged for the second time and for the same
> reason.
>
> **And one thing this section over-claimed, which §8.5's landing makes worth
> separating.** "Nothing needs a rank change" is true of every operation the
> four documents survey, because every one of them is served by a degenerate
> axis. It is not the same as saying G1's *original* subject does not exist: an
> output of genuinely different rank, and the budget arithmetic that would go
> with it, is untouched by anything measured here. What this survey established
> is that **it has no named consumer** among the surveyed operations — not that
> it is a non-problem. If one ever appears, it is still G1's, under the first
> name.

### 8.2 Broadcast belongs in the register beside collapse, at `2†`

`[X, Y, 1] -> [X, Y, Z]` is the same statement from the other side, and it is
**runnable today by the same route**: the escape, plus
`BlockOp::takes_extent_from_placement` — a waiver that is not hypothetical,
since `ops::lattice`'s `LatticeInterpolateOp` already declares it
(`ops/lattice.rs:1079`) and `ops::resample` does too.

Its missing half is a **pinned-axis declaration**, and this one is provable
rather than merely absent. `InputMap::Affine` maps an output index through
`up`/`down` (`op.rs:433-437`), so the source extent on an axis is the block
extent times a rational. Holding it at 1 for every block extent needs a factor
tending to zero, and `up = 0` gives extent **0**, not 1. *There is no
`Affine` that pins an axis*, so the broadcast side of G1 is not waiting for
someone to write a ratio down.

**G1's blocked-family count, re-derived rather than adjusted.** With both halves
known, the count has to be taken again from the instance lists, because *what*
G1 blocks has changed shape even where the number has not.

| | what G1 was read as blocking | status now | still blocked by G1? |
|---|---|---|---|
| **A** | whole-volume histogram, global statistics, contrast stretching, equalisation, histogram matching, noise-σ, per-plane statistics, projections | the collapses run at `2†`; so does the broadcast map half. **Nothing here is unbuildable** | **yes — the declaration.** Every one of them must either consume an axis whole or hold one at 1, and none can say so |
| **B** | a rank-reducing phase a later phase reads; the global auto-threshold's level | reduction to `[1, 1, 1]` and the broadcast back both run at `2†`. The *histogram's* own cost is G7's, not G1's, and always was — this index adjudicated that split in §6 and it is unchanged | **yes — the declaration** |
| **C** | projections, slab projections, extended depth of focus, orthogonal-view images | all `2†`, verified. C said so and was right | **yes — the declaration** |
| **D** | time projection, per-frame statistics, kymograph, colocalisation coefficients, autoscaling | the collapse half is `2†`; the pair-of-volumes half is G5's and unchanged | **yes — the declaration** |

**G1 blocks 4 (A B C D)** — the same number the four documents reached, for a
different reason, and it is worth being explicit about the change of kind
because the number hides it. **Before:** G1 blocked *building*, and every
instance above was unbuildable. **After:** G1 blocks *saying*, and every
instance above is buildable by not saying it. The count of families that cannot
build their instance is now **0**. The count of families that cannot declare it
is **4**. A register row that only carried the first number would now read as
closed, which is why the row carries the second.

### 8.3 The gap none of the five named — G14

**No declaration anywhere is checked against the fetch.** This is the finding
worth the most, because it is the only one in the register that is about a
wrong answer rather than a missing capability or an avoidable cost.

* A projection that reads **only its own block** on the collapsed axis is
  accepted by every guard in `Decomposition::check`. The phase's volume has
  extent 1 there, so the default fetch is one plane, the declared output shape
  matches, the regions tile, and nothing objects. The result is a complete,
  well-formed volume of exactly the right shape and is wrong at every one of
  its positions.
* On a fixture whose answer happens to be uniform along the collapsed axis, the
  same short read is **indistinguishable from correct** — which is why the
  experiment's fixture is built so that the maximum is attained on a different
  plane at every position and never on plane 0.
* A fetch covering **half** the axis is accepted too, with the region *stated*.
  `check` verifies that a block's `source` lies **inside** the image it reads
  (`decomposition.rs:713-728`), never that it covers what the reach claimed.
* The one guard that does compare a declaration against a fetch
  (`decomposition.rs:702-711`) only refuses a source-lattice reach with *no*
  per-block fetch at all. It is a presence check, not a coverage check.

> **Partially closed by §8.5, and the residue is the interesting half.** Part 2
> of the landed change compares a source-frame `AxisReach::All` declaration
> against `BlockGeometry::source` and refuses a fetch that does not cover the
> axis, naming the axis, the block, the fetched range, the required range and
> the phase's own extent (`decomposition.rs:763-791`). Both traps above are now
> **refused by name** where the op declares. What is *not* closed is **silence**:
> a phase that declares `Reach::none()` and reads one plane of an axis it means
> to consume is self-consistent, passes every guard, and is wrong at every voxel
> — pinned by `an_undeclared_short_read_runs_and_is_wrong_at_every_position`.
> The lander's phrasing is the one to keep: **"Part 2 holds an op to what it
> *said*; it cannot make an op speak."** So G14 is closed for source-frame `All`
> and open everywhere else, which includes every `†` op whose fetch is an affine
> map rather than a whole-axis claim — there is nothing for those to be checked
> against.

Registered as **G14**, blocking **4** families. The number is derived, not
assigned: a gap that admits a wrong answer blocks a family if that family has
an operation whose only route is a stated fetch. **A** — per-plane statistics,
the broadcast map half of contrast stretching and equalisation, and
`ops::lattice`'s statistic half under every local statistic it owns. **B** —
the same lattice phases under `ops::local`'s adaptive threshold, and the
broadcast of a global threshold level back over the volume. **C** — every op it
already marks `†`: `ops::resample`, `ops::lattice`, `ops::adjacency`, plus
projections, slab projections and extended depth of focus. **D** — time
projection, per-frame statistics for decay and flicker correction, and the
fixed-path kymograph. All four, and in every one of them the failure mode is
the same: a right-shaped volume full of wrong numbers.

**The count after §8.5: still 4, re-derived rather than carried.** The half that
closed is the half where an op *declares* a whole-axis dependency in the source
frame — which, of the instances above, covers projections, slab projections,
extended depth of focus, per-plane and per-frame statistics, and the collapsing
half of every reduce-then-map workflow, in A, B, C and D alike. What remains
exposed in all four is the same list read the other way: **A** and **B** through
`ops::lattice`'s statistic and interpolate halves, whose fetches are affine maps
with no whole-axis claim to check; **C** through `ops::resample`,
`ops::lattice` and `ops::adjacency`, for the same reason; **D** through the
fixed-path kymograph and any of the above driven per frame. And every family is
exposed to silence, which is a property of the guard rather than of an op.
So the number is unchanged, the *reachable* failures are fewer, and the residue
is sharper: it is now "an op that says nothing" rather than "an op nobody
checks".

### 8.4 What the sibling distance transform is, and is not, evidence for

Family B classifies the exact Euclidean distance transform built in a sibling
application crate as the category-2 instance, and the index repeats it in §4.
**That classification is correct and it is narrower than it reads**, so it is
worth pinning down before anyone cites it for something it does not support.

Its `SweepOp` declares `AxisReach::All` on the swept axis and nothing on the
other two — with **no `.in_space(..)`**, so it lands in the default
`Space::phase_voxels()`. It does **not** change shape. Its lattice is built by
a helper that cuts only the two free axes, so the swept axis is never cut, and
it is planned by an ordinary `PlanBuilder::pixels` phase. **A same-rank
whole-axis stencil, category 2, taking neither escape.**

It is therefore *not* evidence that `AxisReach::All` is plannable in
`Frame::Source` (it is not, §8.1), *not* evidence that a collapsed axis can be
declared, and *not* a user of the `†` route. **Any line in any of the five
documents citing it as a consumer of the collapsed-axis or `Frame::Source`
route must be withdrawn.** Checked across all five: B's §5 and §0 rows and this
index's §4 name it only as a category-2 instance, which stands, and each has
been given the precision above; no document made the stronger claim, so nothing
had to be struck. What each did lack was the distinction between the two spaces,
and that is now stated where the claim is made.

### 8.5 The change that was in flight has landed

*This subsection said "**Not done, and not to be read as done**" and listed what
the change would make true. It has landed. The prediction is kept below the
account, because two of its four clauses were wrong and the difference is the
most useful thing in this section.*

**State, re-measured rather than trusted:** `blockflow` **1555 / 0 / 22**
default and **1637 / 0 / 25** under `--all-features`, from 1552 and 1634; the
`+3` is `tests/collapsing_phase.rs` going 12 tests to 15. A sorted diff of every
test *name* confirms the change is confined to that one file. The sibling
application crate is unmoved at **625 / 0 / 101**.

**What landed, in two parts.**

* **Part 1 — the exception, in `BlockGeometry::derive_with`
  (`geometry.rs:319-322`).** `Frame::Source` is still denied the clamp
  exception in general, and for the reason it always was: a cropping phase's
  edge is an interior position of the array it reads, so a neighbour exists
  there and a halo could have reached it. **That reasoning does not hold for an
  axis the op consumes entirely** — there is no beyond, and so no neighbour.
  `All` is not a distance that ran off the end; it is the statement that the end
  is where the op stops. So the grant is restored **per axis**, on two
  conditions: the reach on that axis is `AxisReach::All`, and this block's read
  spans the whole of that axis. `Frame::Phase` is untouched, so no plan that
  checked before this existed moves.
* **Part 2 — the check, in `Decomposition::check`
  (`decomposition.rs:763-791`).** A whole-axis reach in the source frame is a
  *claim*, and only the fetch can meet it, so the plan is refused when
  `BlockGeometry::source` does not cover the axis. The message names the axis,
  the block, the fetched range, the required range, and — the number that
  explains why no halo helps — the phase's own extent:
  > declares reach `[All, Bounded{0,0}, Bounded{0,0}]` in `source/voxels[0,1,2]`
  > — the whole of axis 0 of image 0 — and block `[0,0,0]` fetches `0..1` of
  > that axis, where the whole of it is `0..11`. … the halo is measured in this
  > phase's own volume, which is 1 voxel(s) on axis 0, so no halo widens the
  > fetch.

**Five consequences, and the second is a correction to this document.**

1. **A collapsing phase can now say what it means, and is held to it.** All
   three declarations that plan — truthful, phase-frame, escape — still plan and
   still give the right answer on an honest fetch. Only the truthful one is
   *checked*.
2. **G14 is only partially closed, and this section predicted otherwise.**
   *What this subsection said:* "G14 closes as a consequence rather than needing
   a guard of its own, because a derived fetch cannot disagree with the
   declaration it was derived from." *What landed:* nothing derives a fetch.
   Part 2 checks a declaration against a fetch **only where a declaration is
   made**, and silence is still unchecked — a phase that declares `Reach::none()`
   and reads one plane of an axis it means to consume passes every guard and is
   wrong at every voxel. **"Part 2 holds an op to what it *said*; it cannot make
   an op speak."** §8.3 carries the residue and §9 carries what would close it.
3. **G9 is partially answered — by refusal, not by constraint.** An op declaring
   source-frame `All` now mandates that the axis is left whole **or** given a
   whole-axis halo, enforced by the tiling check that already existed. Pinned on
   `[11, 4, 4]` cut `[4, 4, 4]`: halos `none()` and `[3, 0, 0]` leave every
   block degenerate and are refused; `Reach::all()` makes every block read
   `0..11` and the plan checks. **No `BlockConstraint::FullExtent(axis)` was
   added and none is needed for correctness** — the op declares and the guard
   enforces, rather than the planner being configured. What stays open is the
   planner-facing half; see the register's G9 row.
4. **The `ops::lattice` escape is now strictly weaker than the truthful form.**
   The escape records *that* a dependency exists; source-frame `All` records
   *what would satisfy it*, and `check` compares it against the fetch. Measured:
   a half-axis or block-local fetch is **accepted under the escape and refused
   under the declaration** — and the tests assert the contrast, because under
   both the escape and the phase-frame form the identical half-axis plan still
   checks, runs, and returns wrong numbers. **Wherever a document recommends the
   escape for a whole-axis dependency, the recommendation is now
   `AxisReach::All` in `Space::source_voxels()` plus `with_sources`.**
5. **The vacuity is now a trap with a name, and the name is the frame.**
   `AxisReach::All` in the phase's *own* frame plans on a collapsed axis and
   means nothing: `is_whole` requires `extent > 1`, so against an extent-1 axis
   it is not a barrier, not a claim, and nothing checks it. Two of the three
   declarations that plan are wrong in different ways, so **any document that
   lists "declare `All`" as the answer has to say which frame.** Every such line
   in these five documents has been given one.

**What did *not* land.** `2†` does not become a plain `2` for a projection: the
fetch is still stated per block, so a hand-written builder is still required and
the `†` price stands. G1 proper — an output of genuinely different rank, and the
budget arithmetic — is untouched. And nothing derives a fetch from a
declaration, which is §9's proposal and remains a proposal.

---

## 9. A proposal, recorded once: collapse and broadcast are one statement

Recorded here rather than in a family document because it belongs to no family,
and recorded as **a proposal, not a decision**.

Collapse and broadcast look like opposites and are the same statement seen from
two sides: *the source extent on this axis does not depend on the output block.*
A projection fixes it at the whole axis; a broadcast fixes it at 1. Everything
else about the two is symmetric.

The shape that says it is a **per-axis extent rule on the input map**, beside
the per-axis coordinate map that is already there:

```rust
enum AxisExtent {
    Scaled { up: usize, down: usize },  // today's `InputMap::Affine`
    Fixed(usize),                       // the broadcast; `Fixed(1)` is the pin
    Whole,                              // the collapse
}
```

Three properties are worth stating, because they are what make it worth
recording at all:

1. **From either arm the per-block fetch is derivable — without a table.**
   `Whole` gives `0..extent` on that axis; `Fixed(n)` gives `0..n`; `Scaled`
   gives what `Affine` gives now. That is precisely what `with_sources`
   computes by hand today, in both cases, in every op that takes the escape.
   `InputMap::Table` — one region per block, resolved when the plan is built —
   stops being the only way to say either of them.
2. **It closes G14's residue — but only if the rule is total.** A fetch derived
   from a declaration cannot fail to cover it, so there is nothing left to
   check. *Corrected after §8.5:* the reason this matters is not the derivation
   but the **totality**. What §8.5 left open is *silence* — an op that declares
   nothing is checked against nothing — and a per-axis rule in which every axis
   carries an arm has no silence in it: `Scaled { up: 1, down: 1 }` is itself a
   claim that this block reads its own extent, and a fetch can be held to it.
   **That totality is the property to preserve if anyone builds this**, and it
   is the part that does the work, not the derivation.
3. **It needs neither a rank change nor a new element type**, which is the
   whole point of §5's 3-D floor: `[X, Y, 1]` was always legal and the gap was
   never in `Voxels`.

`Geometry` and `InputMap` already exist and, as C's §11 records, nothing
consumes them yet — they landed with a default so that the step which changes
no behaviour is separate from the step that moves a declaration onto them. This
proposal is a candidate for that second step. It is not a commitment, nothing
here is promised, and it should be measured against whatever the in-flight
change of §8.5 actually turns out to be.

---

## 10. G5 has closed, and the mechanism this register prescribed was impossible

The register said what would close G5. It was wrong twice, for two unrelated
reasons, and this section exists because **the two objections are worth more
than the outcome** — either one of them would have been found the hard way by
whoever tried it, and the second one would have been found after the first was
fixed.

### 10.1 What this register prescribed

> *"image numbering in which images `0..k` are inputs and phase `p` writes image
> `k + p`, plus constructors taking a list."*

### 10.2 Objection 1 — the executor addresses images positionally

`strategy.rs` does not look images up; it *computes* them from the phase index,
as `env.read(task.phase, …)` and `env.write(task.phase + 1, …)`, at roughly
fifteen sites. There is no seam at which "image `p`" could be reinterpreted as
"image `k + p`" without touching every one of them. Solving the arithmetic for
`k` inputs the other way — keeping the executor and moving the inputs — forces
image 0 to be an input and puts the remaining `k − 1` **above `n_phases`**,
which is not the prescribed numbering at all; it is the shipped one, arrived at
by elimination. **The prescription cannot be built without rewriting the
executor**, and rewriting the executor was not what the row was proposing.

### 10.3 Objection 2 — the builder cannot reach `0..k` either

This one is independent of the first and survives any amount of executor work.
A caller needs a supplied input's **address before it constructs the ops that
read it**: `Chain::source` takes the number, and a `BlockOp` that reads a second
array stores the number inside itself. The number of phases is not known until
`finish`, because `PlanBuilder::partition` lets a strategy choose it. So under
`0..k` the address of every phase-written image is unknown at the moment the ops
are built — and worse, **adding one input renumbers every image in the plan**:
an op holding `Chain::source(3)` because it meant "phase 2's output" would go on
compiling and silently mean something else.

That is the sharper of the two objections, because it is about what a caller can
*say* rather than about what the crate does internally — the same distinction
§8.2 had to draw for G1.

### 10.4 What shipped, and the two properties it has

A **disjoint high address range**:

```rust
ImageId::SUPPLIED_BASE = usize::MAX / 2 + 1;   // the first supplied address
ImageId::supplied(i)                          // the ith array handed to the run
```

Images the run writes keep today's numbering **exactly**: image 0 is the input,
image `p + 1` is what phase `p` wrote. Against the two objections:

* **The address is knowable before any phase exists.** `ImageId::supplied(2)` is a
  constant. The ops that read it are built first, which is the order a builder
  actually runs in.
* **Adding an input renumbers nothing.** The range is disjoint from anything a
  plan can reach by counting phases — it is the high bit of a `usize`, so the
  test is one instruction — and it is stable under a partition that adds phases.

It is not free of cost, and the cost is stated where it lands: a raw address
prints as a nineteen-digit number nobody typed, so `describe_image` exists and
every diagnostic about a supplied input says "supplied input 2".

### 10.5 The API to record

| | |
|---|---|
| `ArrayEnvironment::with_inputs` | seed a run with image 0 **and** the supplied arrays |
| `ZarrEnvironment::create_with_inputs` | the same, on the stored side |
| `Decomposition::supplied_input_images` | every supplied input any phase reads, ascending, **derived from `source_images`** — an array nothing reads is not an image of this plan, whatever the environment was handed |
| `Decomposition::n_supplied_inputs` | how many arrays this plan expects to be handed |
| `Decomposition::image_kind` | `Input` / `Intermediate` / `Output`; see §10.7 |
| `PhaseDecomposition::supplied_dtypes` | what each supplied image the phase reads holds |
| `SourceInput::dtype`, `SourceInput::holding` | a reader saying what the image it names holds |

### 10.6 Two rules a supplied input obeys, and they are different kinds of rule

**Its element type has no fold.** For an image the run writes, the type is the
fold of the chain up to it and `Decomposition::dtype_at` is the answer. No phase
writes a supplied input, so there is no chain to fold and **the readers are the
only declaration there is**: an op naming one without saying what it holds is
refused by name at plan time, and two readers that disagree are refused by name
too. That is worth noticing beside **G14**, which is about declarations nothing
checks: this is a declaration whose only possible validation is *agreement
between the things that declare it*, and it is checked. It is not a
counter-example to G14 — nothing here is compared against a fetch — but it is the
first declaration in the crate that is required rather than optional.

**Its shape is a stated rule: image 0's coordinate space.** A supplied input is
read at the reading block's own fetch region, so it has to be in the space that
fetch is stated in, and that is image 0's. Stated as a rule rather than recorded
per input — and it is not an assumption, because `check_source_images` compares
it against the volume of every phase that reads it and refuses the pair by name.
**So a phase downstream of a reshape cannot read a supplied input**, and that is
**G2's territory**, recorded in G2's row. The failure is a named refusal at plan
time and not a block quietly fetching the wrong region of the right array.

### 10.7 The three kinds landed, and `discard_image` is why

`decomposition::ImageKind { Input, Intermediate, Output }` exists and
`Visibility` is now **derived** from it (`Input | Output => Published`). It did
not land as a rename; it landed because `Environment::discard_image` had to
refuse a supplied input, for exactly the reason `docs/design/images-and-phases.md`
gives for wanting the split at all: **an intermediate may be dropped and
recomputed, and an input cannot be recomputed at any price, because no phase
produces it.** `Published` covered both ends of the run on the shared ground
that somebody outside reads them, which gets image 0's behaviour right for a
reason that is not the real one — and a scheduler that trades residency for
recomputation needs the real one.

The design note's framing — *"there is no privileged index range, input is a
kind"* — is the framing the code now has `ImageKind` to say. The address half
went the other way, and the two are not in conflict: **the range says where an
array lives and the kind says what it is**, and it is the kind that answers
"may I free this".

### 10.8 `n_images() == n_phases() + 1` did **not** become false

The design note flags that it would the moment G5 landed. It did not.
`n_images()` now means *the images the run writes into* — image 0 plus one per
phase — and `n_supplied_inputs()` is the other half. They are deliberately not
added by anything: every caller of `n_images` wants **what the plan fills in** —
a chunk list, an image table, a bound on an image a phase may name. The identity
holds, the count that would have broken it was never the count anybody wanted,
and the note has been corrected in place.

### 10.9 What G5 did not close

* **Side outputs are still terminal.** They land in a separate `String`-keyed map
  on the environment, never in `images`. A supplied input is an array that
  existed before the run; a side output is written during it; the two are
  addressed differently, and `Chain::source` takes a number. Pinned by
  `tests/tuple_map.rs`. **G5 does not bear on this and never did.**
* **G2 is untouched**, and now has one more instance (§10.6).
* **G6 is untouched**, and D's argument against it is unaffected: G5 gives arity
  its operands, which is what D said the channel family wanted instead of an
  axis.

---

## 11. Three measured facts, and what each bears on

### 11.1 The K-ary reach-0 shape costs what a streaming kernel costs

Measured at `K = K′ = 16` in `f32` over a `[128, 128, 32]` block —
the size a plan really uses, not a micro-benchmark's:

* **~40 ns per position**, for `2KK′` flops against `(K + K′) · 4` bytes of image
  traffic — **4.0 flops per byte**. The figure that gets quoted for this shape is
  2 flops per byte, and that is the `K′ = 1` case; a K-in/K-out map is twice it,
  which moves it from "hopeless" to "streaming-bound but worth doing".
* **Tiling the output loop is worth 2.5–2.8×**, and it is **entirely traffic**.
  The kernel is output-major, so one accumulator is live at a time; done over a
  whole block that reads every input once *per output*, and done over a tile it
  reads them once per output *from cache*. Untiled, each of the 16 output passes
  re-reads all 16 inputs from memory. The curve is flat from a few hundred
  positions upward — 64 / 256 / 1024 / 4096 are the same answer and "no tiling"
  is the different one.

**That factor is larger than any coefficient choice anyone would argue about**,
which is the transferable part: for a K-ary shape the question is not the
arithmetic, it is how many concurrent streams the loop keeps open — 32 at
`K = K′ = 16`, which is at the edge of what a hardware prefetcher tracks.

**A third figure, added when the `apply_side` fix landed (§11.3).** Output 0 is
now computed in `apply_with` and outputs `1..K′` in `apply_side`, so the inputs
are streamed **twice**: `2K` reads and `K′` writes where there was `K` and `K′`.
That is **1.08–1.10×** — 22.3–22.6 ms fused against 24.3–24.6 ms split, same
block, same `K`.

The two things to keep from it are the ones that stop a later reader drawing the
wrong conclusion from "streamed twice":

* **The tiling's 2.5–2.8× is fully retained**, and it is arithmetic rather than
  luck. That factor is a pass *per output* — **272** units of traffic against 32
  — and this is a second *pass*, which is **48** against 32. Both passes are
  tiled. A split into two tiled passes cannot cost what an untiled loop costs,
  and the two numbers are not on the same scale.
* **The flop count did not change.** The kernel trait gained a `from` window, so
  each call computes the window of outputs it was asked for and no other.
  Without it the split would have recomputed row 0 and cost **1.14–1.15×**.
  A window on the kernel is what turns "compute the rest" into a statement the
  kernel can act on rather than a comment.

Two incidentals, recorded because both are the kind of thing that gets
attributed to the wrong cause later:

* **The default target is SSE2.** `-C target-cpu=native` was worth ~30% on a box
  with AVX-512. Any figure in these documents measured without it is a floor.
* `dst[i] += c * src[i]` beat `iter_mut().zip()` by **20%** in this kernel.

### 11.2 The partition search's *objective* changed, not its shape

This one is easy to misread, so it is stated as a correction to a thing nobody
wrote down: **the search always swept block candidates per phase.** What was
uniform was the answer, and the reason is arithmetic. The old objective was the
phase's serial work, `cost_per_block × n_blocks`, which is
`volume × redundancy × per-voxel`: `n_blocks` cancels, redundancy falls
monotonically as the block grows, and so the sweep answered **"the largest
candidate that fits"** in every phase of every chain. The per-phase freedom was
freedom on paper.

The objective is now the phase's predicted wall clock:

```text
makespan(phase) = max( cost_per_block x ceil(n_blocks / workers) ,  read x read_cost + core x write )
                       \------------- the pool -------------/       \-------- the channel --------/
```

The pool bound is why a cut is worth taking — the unit of parallelism is the
block, and a reach-0 op pays nothing for the cut. The channel bound is why it is
not always worth taking — the pool bound divides *everything* by the pool, reads
included, and workers do not multiply bandwidth. **Measured at 2.6×** on a mixed
plan, against a control with **identical reads and identical serial work**: the
two plans differ only in how many tasks there are to run, which is exactly what
the old objective could not see. At `workers == 1` the two objectives are the
same expression and the same bits, so no plan built before it moves, and the old
objective stays reachable as the negative control.

**Where this bears on the register.**

* **G4.** Recorded in the row. The short version: the misprice used to be nearly
  inert, because a monotone objective takes the largest candidate whatever the
  coefficient is. Now the coefficient chooses a block edge, and a phase priced
  linear that is really `n log n` is wrong in the half of the `max` that decides
  how finely to cut. G4 got worse without changing.
* **G9.** The planner-facing half — "the enumerator proposes a lattice that will
  be refused rather than avoiding it" — costs more than it did, for the same
  reason: candidates the search wastes are candidates taken out of a choice that
  now matters.
* **The `†` note in §2 is unchanged** and was checked rather than assumed:
  `Trivial`, `Enumerating`, `Greedy` and `Materialising` still plan one volume
  per group, so a cross-grid phase still needs a hand-written builder. The
  objective decides *which* grid, not *whether* a shape-changing phase can be
  produced.
* **Nothing in the four documents asserted the old behaviour**, so nothing had to
  be struck. What they lacked was any statement of how a block edge is chosen at
  all, and it is now here.

### 11.3 The `apply_side` defect — a note on G10, and it has since landed

*This subsection was written as "a worker is fixing this now — it has not
landed". It landed while this pass was running. The account is rewritten and the
decision it recorded is kept below it, because the decision is the part with the
longer life.*

**What the defect was.** `BlockOp::apply_side` was handed the buffer the op read
and the primary result, and **not** the `SourceInputs`. For a K-ary op that is
not enough: output `o` is a function of all `K` inputs and the call could see
one of them. `ops::mixing` worked around it by computing outputs `1..K′` in
`apply_with` and carrying them across in a per-block map keyed by the buffer
offset.

**What landed.** The argument is threaded from the executor's call site
(`strategy.rs:1032`, which was holding the source buffers already) through
`Chain::apply_side` and `Environment::apply_side`. The per-block map in
`ops::mixing` is **deleted, field and all**. State, re-measured rather than
trusted: `blockflow` **1593 / 0 / 22** default and **1678 / 0 / 25** under
`--all-features`; the sibling application crate is unmoved at **641 / 0 / 103**.
**G10's original clause is now true exactly as written** — *one `BlockOp` with
C−1 `source_inputs` and C′−1 `side_outputs`, no new trait and no new axis.*

**The property bought, which is more than "the workaround is gone".**
`apply_side` is now a **total function of its operands**, rather than a function
of its operands plus whatever an earlier call left behind. That is the sentence
worth keeping. The old shape was correct *only while two calls stayed paired* —
correctness resting on a protocol between two trait methods, enforced by a
refusal by name when the pair came apart, which is a runtime check standing in
for a signature. And it held whole `f64` blocks resident **that no counter knew
about**, in a crate whose side outputs exist *precisely because* a run once wrote
158.6 MB while the framework counted 95.2 MB, short by a factor of 1.67. A
mechanism built to close an accounting shortfall had opened a smaller one of its
own.

**The cost, measured.** `apply_with` now asks the kernel for output 0 and
`apply_side` for outputs `1..K′`, so the inputs are streamed twice: a block's
traffic goes from `K` reads and `K′` writes to `2K` reads and `K′` writes.
At `K = K′ = 16` in `f32` over a `[128, 128, 32]` block, **1.08–1.10×**
(22.3–22.6 ms fused against 24.3–24.6 ms split).

Two things about that number are worth more than the number:

* **The tiling's 2.5–2.8× is fully retained**, and the reason is arithmetic
  rather than luck. That figure comes from a pass **per output** — 272 units of
  traffic against 32 — not from a second pass, which is 48 against 32. Both
  passes are tiled; a split into two tiled passes cannot cost what an untiled
  loop costs. Anyone reading "the inputs are streamed twice" and expecting the
  tiling win back is reading the wrong comparison.
* **The flop count did not change**, because the kernel trait gained a `from`
  window: each call computes the window of outputs it was asked for and no
  other. Without it the split would have recomputed row 0 in the second pass and
  cost **1.14–1.15×** instead.

**The blast radius, since it sizes the change for anyone reading the row later:**
**4 overriding implementors out of ~30**. A default implementation absorbed the
rest, and all nine `BlockOp` implementors in the sibling application crate
compiled untouched.

**And a bonus that fell out, which is a limitation deleted rather than a cost
paid.** `Chain::apply_side`'s `Parallel` and `Sequence` arms now re-derive their
intermediates through `apply_with` with `sources` threaded down unchanged and
addressed by image rather than by position. So a subtree that both declares a
side output **and** contains a `Chain::Source` is handed the buffer the executor
read instead of failing by naming the image: the documented *"side outputs and
source leaves do not compose yet"* limitation is gone, and `op.rs` now says the
opposite in its place.

### One correction that is not about `apply_side` at all

**`ops::ridge`'s second pass was never an instance of this residue**, and the
sentence this subsection used to carry — that "for every op that shipped before
`ops::mixing` that was enough, `ops::ridge`'s scale map recomputes from the op's
own input" — reads as though the source inputs would have retired it. They
would not, and cannot. The scale map is the **argmax over scales**: an
*intermediate of `apply`*, made and discarded inside the multi-scale evaluation.
`SourceInputs` carries **stored images**, and the winning scale is stored
nowhere, so no argument to `apply_side` can retire that second pass. It is why
the map is opt-in and why a caller who does not ask for it pays nothing. The
three places in `ridge.rs` itself were corrected by the lander; this is the same
correction in the register, and it is worth making because "an op that recomputes
in `apply_side`" and "an op that could not compute in `apply_side`" look alike
from a distance and are different problems — the second one is now closed and the
first one is a design choice with its price written down.

**Why a note and not a `G17` — the decision, kept, and now demonstrated.** The
rule this pass applied, stated once so the next pass can apply the same one:

* **A new identifier** is for a subject no row covers, where nothing is
  currently changing — G15 and G16 both qualify, and G14 did.
* **A note on the row** is for a defect in a *clause* of a row whose shape has
  already shipped and whose fix is in flight. An identifier minted here would
  have been closed before any document could cite it, and five documents cite
  this series by number; a series with a stillborn member in it is worse than a
  row with a correction in it. The correction is visible either way, which is the
  property that actually matters.

**That second bullet is no longer a prediction.** The fix landed inside the same
pass that recorded it, so a `G17` minted for it would have been born and closed
without a single citation. The rule is kept as written; this is what it is for.

---

## 12. The sweep after the barrier batch

A pass over the register against the tree, not against a summary of it. Each
row below was checked by reading the code the row is about; where the answer is
"landed", the mechanism is named and the file is named. **One of them was
checked badly and is corrected below**: G9's entry asserted an absence on a
`grep | head` whose output was truncated, and the absence was not there. The
correction is in §12.4 with the wrong sentence kept above it, because a pass that
claims this method owes the reader its own failure of it. Two rows changed from
open to closed, one changed from open to **built but not collected** — which is
a state this register did not previously have a word for — and four stayed open
with something now known about them that was not known before. The rest were
checked and are unchanged, which is said rather than left to inference.

**The convention is unchanged and is the reason this section exists.** A row's
original wording stays; the inversion is appended to it and the argument lives
here. What the register is for is the history of what was missing and why — a
row rewritten to describe the present would lose the part worth keeping, which
is usually the sentence that turned out to be wrong.

### 12.1 G7 — the barrier is **available** and is **not collected**

> **Inverted, and by construction rather than by revision.** This entry's whole
> method was to record a grep instead of a report, *"a claim that can be
> re-checked in one command survives the pass that made it"* — and the command
> now returns the opposite. `grep -rn "fn barrier(&self)" src/` returns
> `ops::fill`, `ops::regional`, `ops::detect` and `ops::label` beside the trait
> default and the distributed probe. All four declare a hoisted `reduce` as well.
> **G7 is collected**, at 106.07 GiB → 4.84 on the recorded volume, a factor of
> 21.9. The entry is kept whole below because two of its claims were wrong in
> ways worth keeping visible, and both are named at the end of it.

`FragmentOp::barrier()` and `FragmentOp::reduce()` ship. The plan records the
declaration on `PhaseDecomposition::barrier` and hashes it into the fingerprint;
`TaskGraph::barriers` is one bool per phase; both schedulers gate on it, and a
`reduce` without a `barrier` is refused at plan time. `docs/design/barriers.md`
§8 is the account of what was built and what its specification got wrong.

**And no shipped op declares one.** Checked here rather than taken from the
note: `grep -rn "fn barrier(&self)" src/` returns exactly one hit, the default
in `src/fragment.rs` that answers `false`; `fn reduce(&self` returns the default
beside it and one unrelated method on a local kernel trait. `barriers.md` §8.10
says the same thing in words — `ops::fill`, `ops::regional`, `ops::detect` and
`ops::label` "are all still the in-plan shape; migrating them is where the 25.4x
is actually collected".

So **the row's measured price is still being paid, in full, by every op that
paid it before.** That is the whole of this entry, because a register that
recorded "built" here would be read as "the toll is gone", and the toll is
exactly where it was. The right word is *buildable*: the declaration exists and
every op that would benefit has still to make it.

**And when they do make it, the row does not close either.** §4's item 1 already
records the arithmetic and it is worth pointing at from here rather than
restating: a barrier removes the pixel amplification and leaves the fragment
gather, taking the measured gap from `25.4x` of total traffic to about `9.4x`.
Getting the rest needs the merge to run **once** rather than per block, and that
is blocked by a *second* rule — `Decomposition::n_images() == n_phases() + 1`,
which makes a fragment-only phase terminal and the three-phase "label, merge,
relabel" shape unplannable. `docs/design/barriers.md` §1 separates the two and
says the conflation is "the easy error and the reason the first attempt at this
analysis reached the wrong conclusion". **G7 names one of the two.** Adopting
the barrier will move it to *built*; it will not empty it.

> **This paragraph is false and it is the more useful of the two errors, because
> it is the exact conflation the sentence it quotes is warning about.** Hoisting
> the merge is **not** blocked by Rule A. `barriers.md` §7.5's whole-phase blob
> is specifically the version that does not need it — a result that belongs to
> the phase rather than to a block needs somewhere to *stand*, not an extra
> image — and §7.5 point 3 says so in as many words: *"This is the version that
> does not need Rule A, and that is why it is the one specified."* The four
> migrated ops are the proof: each runs its merge once inside a **two-phase**
> plan with `n_images() == n_phases() + 1` untouched. Rule A is exactly where it
> was and blocks exactly what §1.1 says it blocks, which is the *three*-phase
> shape and not the reduction.
>
> So G7 does close, and it did not need the second rule moved.

> **The second error is smaller and is a quantity, not a claim.** The
> arithmetic above — a barrier taking `25.4x` to about `9.4x` — is right for the
> recorded volume and is quoted here as though it described the barrier. It
> describes the ratio of the fragment set to the volume: a barrier removes the
> pixel half, so its share is whatever fraction the pixels were. Measured
> per op: `1.56x` of `91.3x` on `ops::fill` over `[16, 32, 32]` at 256 blocks,
> where the fragment set is 175% of the label image; and **exactly `1.00x`** on
> `ops::detect`, asserted as an equality, because that op declares
> `reads_pixels() == false` and fetches no pixels at any halo. `barriers.md`
> §10.5 records it.

**One correction to the census, and it is smaller than it first looked.**
`barriers.md` §5 counts "four shipped ops" of this shape — `ops::fill`,
`ops::regional`, `ops::detect`, `ops::label` — and names `ops::detect`'s phase 1
as "the one existing escape" from the pixel half of the toll, on the grounds that
it declares `reads_pixels() == false` and the executor then performs no pixel IO
for it. **`ops::tabulate`'s merge is a fifth op of the shape and a second user of
that escape**, so the count is five and the escape is not one op's.

The first draft of this entry claimed more than that and was wrong, which is
worth keeping: it read `MergeTabulationOp::reach == 0` and `gathers() == false`
as something the op had *done*. `reads_pixels() == false` is the **default** on
`FragmentOp` — "an op that says nothing should cost nothing" — so the merge did
not elect the escape, it inherited it, and every fragment-only phase in the crate
has always had it. What that changes about §5's sentence is only the arithmetic:
the escape is "available only to a reduction whose answer is not a volume", which
is a precondition met by more of the five than the "one existing escape" phrasing
suggests, and by none of the three that write a volume. **The row is unchanged.**
For all five, the fragment gather is the half that remains, a barrier does not
address it, and the hoisted reduction is what does — `barriers.md` §7's
separation, reached here from a second direction.

### 12.2 G15 — closed, and the join closed in a shape the row did not propose

Both halves landed.

* **The verdict lands on `Bool` without a round trip.** `VoxelwiseMaskOp` over a
  `MaskFn` `accepts` `f64` and `produces(_) -> Dtype::Bool`, and
  `ThresholdMask` is the `MaskFn` — so a thresholded arm writes a one-bit image
  and the image is allocated from that declaration.
* **A fan-in joins arms that disagree.** `LogicCombine::accepts` still requires
  every branch to be a mask carrier, but the clause forcing them to *agree* is
  now conditional: `self.output.is_some() || inputs.iter().all(|&d| d == inputs[0])`.

**The row asked for the second half in a different shape and should be read as
having guessed wrong about it.** What it asked for was "an `accepts` that does
not force the widest arm on the rest" — an inference. What shipped refuses to
infer: a fan-in whose branches disagree has a well-defined answer and no obvious
width to write it in, so the caller states the width with
`LogicCombine::producing(dtype)` and gets a refusal at plan time otherwise. The
type's own header gives the reason, and it is the crate's standing one: *this
crate does not let an image's width be an inference.*

The row's own **unverified** clause — "whether narrowing **every** arm of a real
fan-in is a general answer; the binarize work reports it is not" — is therefore
answered by not being asked. Nothing narrows every arm. The arms keep their
widths and the sink states its own, which is why `producing` also applies where
the branches *do* agree: it is how a chain's sink narrows at the first join and
stays narrow without any arm changing width in step with it.

**Not re-measured.** The row's 57.853 → 46.28 GiB is a consumer's figure on a
consumer's stage; the mechanism it needed is present and whether that consumer
has collected it is not this document's to claim.

### 12.3 G16 — closed, with one argument unforeseen and one prediction backwards

`Decomposition::peak_image_bytes(work) -> Result<u64>` ships, with
`tests/peak_image_bytes.rs` beside it. **Whether the three hand-rolled copies
have gone is not this document's to say** — the row was careful that they are in
*consumer* test files and that there is no such walk under this crate's own
`tests/`, and that is still true: the new suite tests the function, it does not
replace them. What closed is the gap, which was that the crate could not answer
the question at all.

**The three kinds do matter to it, exactly as prescribed.** A supplied input is
seeded into the live set before the walk starts and is never retained out of it —
"a supplied input has no producer to die after" — while an internal image is
dropped after its last reader, and the zero-reader rule is the same rule at zero
readers.

**What the row did not foresee: the plan alone cannot answer.** The prescription
was `Decomposition::peak_image_bytes()`, no argument. It takes one, and the
reason is the same reason `predicted_cost` takes it: phase `p` writes image
`p + 1` *unless it does not*, a fragment phase that writes no pixels writes no
image, and whether a phase writes is the **op's** answer and not the plan's
(`PhaseWork::writes_an_image`). `&[]` is accepted for an all-pixel plan; a
slotless phase with no entry is refused rather than assumed to write. A register
row cannot be blamed for missing that, but it is exactly the kind of thing this
column is for: the shape of the answer was one argument wider than the shape of
the question.

**And the prediction that ran backwards.** The row's argument for building it in
the crate was "derived on exactly the argument `image_visibility` is derived on —
a field that could disagree with the arithmetic is a field that eventually does."
The disagreement was real and it was **the other way round**: the walk was exact
and the two *published* helpers were not. `readers_of_image` counted phase `p` as
a reader of image `p` unconditionally, which a fragment phase that reads no
pixels is not, and `images_dead_after` did not apply the zero-reader rule at all.
Both have since moved to where they belong, so the two agree and the executor
frees what this predicts. The row was right that a second copy of a lifetime walk
is a liability; it had the direction of the liability inverted.

**One caveat the row did not have, and it is measured.** The figure is
*directionally right and quantitatively not*: at one block a run saved 25% more
than it predicts, because a `Chain::Parallel` branch buffer is not an image and
no `Decomposition` can see it; at `32^3` a run saved half what it predicts,
because `ImageStore` allocates lazily and frees eagerly so the peak is a moment
the run does not have. A caller sizing a machine still wants a measured `VmHWM`.
That does not reopen the row — it blocked nothing when open and folds three
copies into one now — but a number quoted from it should be quoted with this.

### 12.4 Four rows still open, with what is now known

**G3 — no complex element variant.** `ops::fft` landed: a real 2-D transform
with `Spectrum = Array2<Complex<f64>>`, a backend enum, and a smooth-length
helper. It is a **library, not an op**, and it is now the in-tree demonstration
of the row rather than an argument for it — there is no `Dtype::Complex64`, no
`Voxels` variant, and therefore nothing for a phase to write. §2's table already
recorded that it "declines to be an op at all"; what changed is that the sentence
now describes code. **The row's size estimate is unaffected** and so is its
advice: take it when a concrete op forces it, and nothing yet has.

> **Corrected, and the sentence that was wrong is the last one.** *"take it when
> a concrete op forces it, and nothing yet has."* A concrete op came — and it did
> not force it. `ops::convolve::TransformConvolveOp` is a frequency-domain
> operation that is an ordinary `BlockOp` with an ordinary **bounded** reach,
> byte-identical across genuinely distinct lattices against a whole-volume
> reference, and it needs no complex element type at all, because its spectrum
> lives inside one `apply` and dies there. So the row's advice was the right
> *test* and it has now been run; the answer it returned is the opposite of the
> one the sentence expected.
>
> **The paragraph above is also wrong in its middle, and in the more useful
> place.** *"there is no `Dtype::Complex64`, no `Voxels` variant, and therefore
> nothing for a phase to write."* The **therefore** does not follow. What a phase
> writes is its *output*, and every operation on the row's own list — a
> frequency-domain filter, a band-pass, a stripe removal, a Wiener or
> regularised-inverse deconvolution, a transform convolution — has a **real**
> output. The spectrum in each is an internal intermediate, which
> `ops::deconvolve`'s table had already reasoned to before the question was
> asked: such an op "would therefore have to be *one* op that transforms, divides
> and inverse-transforms internally ... and never expose one". The decisive clue
> was in-tree and predated the row's re-reading of it.
>
> **And the row is now refused rather than open**, on three findings the G3 row
> carries in full: `Voxels` is `f64`-scalar-shaped at the root, so a complex
> element can only project to a real and lie to a short circuit; the blast radius
> was **measured** by adding `Dtype::Complex128` to a copy of the tree and reading
> the compiler, and the loud half (23 exhaustive `match` sites) is the *cheap*
> half against twelve `accepts` predicates written as denials, every one of which
> would say **yes** to a complex block at plan time and fail in the executor; and
> it unblocks nothing on its own list. What the family actually pays is a
> **halo**, which is G9's, and G9's entry below is where that now leads.

**G4 — the cost model cannot price `log n`.** Two things moved and neither is
this. `PhaseTraffic` landed — `images_read` and `writes_an_image` per phase — so
the price now sees that a `Chain::Source` arm traverses two arrays, which was an
under-charge "in the direction the model is not allowed to be wrong in". That is
the *arity* of the traffic, not its growth. Meanwhile `ops::fft` put the actual
`n log n` operation in the tree. So the row's example is no longer hypothetical
and the row is no closer to closed: `cost_per_voxel` and `cost_per_voxel_in` are
still the only two coefficients, and neither takes the volume.

**G8 — no scalar broadcast inside an iterative phase.** Confirmed open by the
thing most likely to have closed it. `barriers.md` §5 guessed that a barrier's
declaration might be the same declaration a *substage* reduction needs one level
down; §8.10 records the outcome flatly — "**G8 is untouched.** Nothing here tests
that and `src/iterate.rs` is unchanged; the guess is still a guess." Verified
here: `iterate::Operand` still has `Running` and `Fixed` and no way to name a
value reduced over the previous substage's whole output. This is the row B calls
the highest-leverage item in its document, and a batch that built the reduction
machinery one level up did not reach it.

**G9 — the planner cannot be told not to cut an axis.** `PartitionSearch::SingleGroup`
landed, and it is the same complaint answered in a different dimension: *a caller
that has already decided its chain is one phase, for a reason the cost model
cannot see, does not want a search that may cut it.* That is "do not cut between
slots". Nothing answers "do not cut axis `k`". `Constraints` still carries a
budget, a list of scalar block edges, a concurrency and a cost model;
`BlockConstraint` still has `Extent([usize; 3])` and `Regions(Vec<Region>)`, and
the anisotropic `Extent` can pin an axis only by mandating all three — giving up
the search entirely rather than constraining it. `BlockGrid::along(volume, axes,
edge)` builds exactly the lattice the row wants and **no strategy consults it**:
every use in the tree is a test hand-building a grid. So the row is unchanged and
its shape is confirmed rather than superseded — the missing thing is a constraint
the *search* reads, and two adjacent ways to say it now exist for callers who
build their own plan.

> **Corrected — and this time by a third instance of the complaint, which
> answered half of itself.** *"the missing thing is a constraint the search
> reads."* For the **divisibility** case that is now false, and the reason is
> worth more than the correction. An op can need the block *edge* to be a whole
> number of something of its own — `ops::convolve::TransformConvolveOp` needs it
> to be a whole number of its transform tile — and there are two places to say
> so: a constraint the search reads, which is what this row has always asked
> for, or a **reach that resolves against the lattice**, which is what shipped.
> `AxisReach::Aligned { stride, lo, hi }` answers the worst case to everything
> that cannot see a grid and is discounted in `Reach::in_voxels`, which
> `decomposition::price_phase` already calls once per candidate grid. So the
> planner *prices* an aligned edge cheaper and prefers it on its own.
>
> **Two call sites, not one, and the second was found by a test.** `cuttable_axes`
> decides which axes may be cut *before* anything is priced, and it was asking the
> unresolved reach — so an aligned op had every axis dropped and its phase
> collapsed to a single block before the discount could be taken. Measured on
> `96^3` with a 32-voxel tile: `Greedy` plans **27 blocks** with the resolution
> and **one** without. That is this register's G7 cost — a phase reading the whole
> volume per block — arrived at from the opposite end, and it means the read
> amplification figures are the *optimistic* half of what the slack costs.
>
> **The reach is the better shape and the code says why, at three places.**
> `BlockConstraint` has one operation, `lattice()`, which produces *the* grid,
> and `phase_for_group`'s comment says the candidate list is "replaced rather
> than filtered" — a divisibility admits many grids and fits neither.
> `Chain::block_constraint` folds by **equality**, so two ops with different
> strides could never share a phase. And a constraint turns a **cost** into an
> **error**, which is the wrong currency for a gap this row itself calls
> "planning-quality ... no longer a correctness one".
>
> **Measured, not estimated, on both halves.** The loud half: an `AxisReach`
> variant added to a copy of the tree and `cargo check --all-features --lib
> --tests` run against it gives **5 sites, all in `src/reach.rs`** — **and that
> is an under-count, for the same reason a `head`-truncated grep was**: the check
> aborted at the lib-test stage and never compiled the integration tests. The
> real count is **6**; the sixth is a hand-written `BlockOp` in
> `tests/partition_search.rs`, found by the full run. A probe measures what it
> got to, and one that stops early hides every site after the stop. The silent
> half is the wire, exactly as it was for G3 — `AxisReach::from_json`'s last arm
> accepts any object and reads `lo`/`hi` out of it, so an aligned reach would
> have come back as a `Bounded` carrying its *discounted* sides with the stride
> dropped. **What a compiler finds is the cheap half twice running**, and the
> lesson generalises past both variants: a hand-enumerated round-trip case list
> is precisely the assertion a new variant slips past, and `the_rich_forms_survive_the_wire`
> would have let this one through.
>
> **What is still open here is the rest of the row.** An op cannot *demand* an
> aligned lattice and `Constraints` still has no per-axis rule; the per-axis half
> is untouched.
>
> **A third item stood here and it was a defect, not a limitation.** *"the
> discount is lost when the aligned op shares a phase, because the fold flattens
> rather than inventing a lattice that satisfies two strides."* A phase's reach
> is its ops' reaches **added**, so flattening meant adding a reach of *nothing*
> was not the identity: fusing the op with a voxelwise map cost **27 blocks
> against one** on `96^3`. The variant now carries both of its answers and folds
> each componentwise. **The one-answer-plus-a-rule form could not have been fixed
> in place** — two stride-32 reaches added would have claimed `31 + lo`
> off-alignment against a truth of `31 + 31 + lo`, an under-halo of 31 a side —
> which is why the shape changed rather than the arithmetic.
>
> **And this is not the scalar-ladder finding wearing a different hat**, which is
> the question worth answering explicitly because the two look alike. The per-axis
> half of this row and `refined_ladder` are both about the **menu** — which edges
> the search may propose, a *caller*-to-planner question answered in
> `Constraints`. A divisibility is an *op*-to-planner question about which of
> those edges are cheap for one phase, and it is answered in the reach, the one
> quantity the pricer already re-resolves per candidate. The two do interact, and
> favourably: **because the planner can only express a scalar ladder, and that
> ladder is powers of two, a power-of-two tile is already aligned on most rungs**
> — three of the coarse four — so the discount rarely changes *which* lattice is
> chosen and mostly stops the plan over-fetching on the one it was going to
> choose anyway. Had the planner been able to offer arbitrary per-axis edges,
> alignment would have been rarer and this would have had to remove candidates.
> The limitation the row below calls a defect is what made this fix cheap.

> **Corrected by measurement, and the sentence that was wrong is kept.** *"`BlockGrid::along(volume,
> axes, edge)` builds exactly the lattice the row wants and **no strategy consults
> it**: every use in the tree is a test hand-building a grid."* **The second half
> is false.** The planner consults it at five live sites — `strategy.rs:2182`,
> `:3186` and `:3869`, `assemble.rs:1001` and `distributed/spec.rs:962` — and
> three of those are the core of a candidate loop, `cuttable_axes` then
> `BlockGrid::along` per rung. The error was mine and it was mechanical: I ran
> `grep -rn "BlockGrid::along" src/ tests/ | head -8`, and `head` cut the source
> hits off below the test hits. A truncated grep is not a check, and a row
> asserting an absence is exactly where that costs something — someone acting on
> the sentence would have gone looking for a caller that has been there all along.
> (`assemble.rs:812` is a doc comment rather than a call; the fifth live site is
> `distributed/spec.rs`'s `split_every_op`.)
>
> **The accurate version is sharper, not weaker: the planner can only express a
> *scalar* ladder.** The function no planner path consults is `BlockGrid::new`,
> the per-axis constructor — its only live callers are `BlockConstraint::grid`
> and `op.rs:263`, which is a caller *mandating* an extent, and
> `distributed/wire.rs`, which is reconstructing one. So every grid the search
> itself proposes comes from one integer, and a per-axis candidate cannot be
> offered to it at all. That is the same complaint the row makes, one level more
> precise, and it is why the missing thing is still a constraint the search reads.
>
> **And a scalar edge is not a cubic block, which is the finding that redirected
> the row.** `BlockGrid::along` clamps each axis at the volume and `cuttable_axes`
> drops an axis a cut would not narrow, so edge 512 on a `404 x 1304 x 3369`
> volume gives `[404, 512, 512]`. The diagonal is projected onto the volume's own
> box before it is priced and the reachable shapes are already anisotropic —
> which is why a sweep found the lever to be **granularity, not anisotropy**:
> full per-axis freedom won a minority of cells and by at most `1.4x`, where a
> finer scalar ladder won by up to `2.7x`. `Constraints::block_candidates`' own
> header carries that sweep.
>
> **So G9's spatial half is now partly answered in the scalar dimension and
> untouched in the per-axis one.** `decomposition::refined_ladder` has landed,
> opt-in through `Constraints::with_refined_ladder()`, taking `[16, 32, 64, 128]`
> to `[16, 24, 32, 48, 64, 96, 128]` — one rung at `3/4` of each power of two,
> with the floor never lowered. Re-run here rather than quoted: on `1024^3` at
> 40-way concurrency over nine budgets and three admission charges,
> `tests/block_ladder.rs` reports **a larger admitted block in 15 of 27 cells,
> best gain 3.38x in volume, and never a smaller one**.
>
> **The default is deliberately unmoved**, and both reasons are the kind this
> register exists to keep: the search is `partitions x candidates^phases`, so
> refining squares the per-phase factor — 81 combinations become 625 at four
> phases — and every recorded parity figure was measured under the coarse ladder.
> A finer ladder is a caller's statement, made in one place a grep can find.
>
> This is a correction of the register **by measurement**, not a row closing. G9
> stays open, and it is now open in two distinguishable ways rather than one: the
> *granularity* half is reachable today by opting in, and the *per-axis* half has
> nothing to reach for.

**G12, G13, G6, G2, C1, C2, C3 — checked, unchanged.** `iterate::Operand::Running`
is still singular and still documented as deliberately so; `points::Point::at` is
still `[usize; 3]`; `Region::start` is still `Vec<usize>`; `SourceInput` carries
an image, a dtype and a `Reach` and still no scale or offset, so G2 is untouched
by the second-input work exactly as G5's row said it would be.

### 12.5 What landed and is not in this register, deliberately

The `.npy` reader and writer, `Elements`, and the exact integer widenings on
`Elements` and `Voxels` all landed in this batch and **close no row here**, which
is worth one sentence so that a later sweep does not go looking for the row they
should have closed. This register is about what the **decomposition model** can
and cannot express — reach, geometry, cost, images, fragments. A file format is
none of those: `NpySource` and `NpySink` implement `RegionSource`/`RegionSink`,
which is an interface the model already had and which the register never named a
gap in. The one place they touch a row is that `Elements::widened_i64` and
`Voxels::widened_i64` refuse the silent-rounding failure that a `f64`-only
widening admits, and that failure has no identifier here because it was never a
framework gap either — it was a defect in a consumer's private reader, found by
migrating it onto the crate's.

*This index restates no finding from the four documents; where it says
**unverified**, it means exactly that; and where it says "verified", it means a
file under `src/` or `tests/` was read for this document and is named above.
Sections 8 and 9 were added by a later correction pass; everything in §8 was
measured against `tests/collapsing_phase.rs` and the files it names, and §9 is a
proposal and is marked as one. §8.5 was rewritten a second time when the change
it described as in flight landed — its prediction is kept below its account,
because two of that prediction's four clauses were wrong. §§10–11 were added by a
later pass again, for the closure of **G5** and **G10** and for three measured
facts; the same convention holds throughout — **the prescription that turned out
to be wrong is kept beside what shipped**, twice in §10 and once in the G10 row.
§11.3 was written as an account of a change in flight and **rewritten when it
landed inside the same pass**; its prediction that an identifier minted there
would be closed before anyone could cite it is kept, because it came true. As of
this pass **no change described here is in flight**: everything §§10–11 record is
in the tree. The phrase survives in four places and none of them is a pending
change — §8.5's account of what it replaced and §9's closing sentence are about
the geometry change, which landed earlier, and §3's preamble and §11.3 use it to
state the *rule* for when a defect gets a note instead of an identifier.*

> **Corrected by §12's sweep — the sentence above stopped being true and is kept
> anyway.** "No change described here is in flight" held when it was written and
> does not now. The barrier migration of the four ops named in §12.1 is in flight
> in another worker's files as this is written, so **G7's row will move on its
> own** — from *buildable* to *built* — without a word of this register
> changing. That is the first time a row's state has depended on a file this
> document does not own, and it is why §12.1 records the grep rather than the
> report: a claim that can be re-checked in one command survives the pass that
> made it.
>
> > **It moved, and the method worked exactly as designed.** All four ops
> > declare both halves; the grep returns the opposite of what it returned; G7 is
> > *built and collected*. What the method did **not** protect was the prose
> > beside the grep: §12.1's argument that hoisting is blocked by a second rule
> > was wrong, and no command could have re-checked it, because it was a claim
> > about what is possible rather than about what is present. Both are recorded
> > in place. The lesson is not that the grep was the wrong instrument — it is
> > that a re-checkable claim and an argument were sitting in one paragraph
> > wearing the same confidence.
