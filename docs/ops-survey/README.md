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
identifiers the documents already use and are preserved unchanged.** G7–G16 are
allocated here, in the shared `G` series, for gaps the documents named in prose
but did not number — this index is the register of record for that series.
**G1 keeps its identifier and loses its name**: §8.1 says why the old one was
wrong. **G14 was minted here**, by the correction pass of §8, and **G15 and G16
are minted here** by this one; those three are the members of the series no
document named at all.

**Two rows are now closed and are kept in the table rather than struck.** G5 is
closed outright and G10 is closed but for one clause; both rows carry the
prescription they shipped *against*, because in each case the prescription was
wrong in a way worth more than the outcome. A register that recorded only what
landed would be a register that could make the same mistake twice.

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

"Blocks" counts **how many of the four families the gap blocks**, which is the
ranking no single document can see.

| id | gap | what it blocks | smallest framework change | blocks | raised by |
|---|---|---|---|---|---|
| **G1** | **the geometry cannot declare a pinned (broadcast) axis** *(renamed twice, both kept: "no rank-reducing phase" originally, then "a collapsed **or** a broadcast axis" by this survey's correction pass, then narrowed to this because the **collapsed half landed** — §8.1, §8.5)* | the **declaration**, not the result — and now one side of it only. **Closed:** an op that consumes an axis whole and writes it at extent 1 can declare `AxisReach::All` in `Space::source_voxels()`, and the declaration is enforced against its fetch. That covers projections, slab projections, extended depth of focus, per-plane and per-frame statistics, and the collapsing half of every reduce-then-map workflow. **Open:** every op that holds a *source* axis at extent 1 while its output grows — the map half of contrast stretching, equalisation, decay and flicker correction, autoscaling. Those run today through the `†` escape and cannot be stated | a per-axis extent rule on the input map, `Fixed(1)` being the pin. §9. Not `output_shape` and not `Voxels`: `[X, Y, 1]` is already a legal `Array3` and needs nothing. `InputMap::Affine` provably cannot express the pin (§8.2) | **4** (A B C D) for the open half — re-derived in §8.2, and note the *kind* changed twice: from "cannot be built" to "cannot be said" to "half of it can now be said" | A |
| **G2** | no phase whose inputs sit at different offsets or resolutions | mosaicking at both ends, montage, multi-view fusion, drift correction, coarse-to-fine pyramids, Laplacian-pyramid reconstruction, dense optical flow. **And, since G5 landed, one more instance that is this row's and not G5's:** a supplied input is in **image 0's coordinate space**, by a stated rule rather than a recorded one, so a phase downstream of a reshape cannot read one. `check_source_images` compares the rule against the reading phase's own volume and refuses the pair **by name** at plan time, which is the right failure — the alternative is a block fetching the wrong region of the right array and producing a well-formed wrong volume | `BlockGeometry::source` becomes one region **per source image**; `SourceInput` carries a rational scale and offset per axis rather than only a `Reach`. `Placement::sources` already carries `Vec<(usize, Anchor)>` on the execution side | **3** (A C D) — **re-derived after G5, and unchanged.** The new instance is reachable only by a family that both reshapes and reads a supplied array: **A** (a supplied flat-field reference read after a resample), **C** (tiles read after any rescale — §6's stage 3 at two resolutions), **D** (a per-channel reference after a rebin). **B** names no supplied-array operation at all, before or after a reshape, so it is not added | A |
| **G3** | no complex element variant | a spectrum as a plan image, and therefore every frequency-domain filter, band-pass and stripe removal, Wiener and regularised-inverse deconvolution, phase correlation *as a phase*, transform-based convolution, temporal cross-correlation | `Dtype::Complex64`/`Complex128` with matching `Voxels` variants, plus a decision about the half-spectrum layout — which is `[rows, cols/2 + 1]`, an extent A read as touching G1's territory and which, after §8, is an ordinary shape change and touches nothing. The largest of the shared four; take it when a concrete op forces it | **3** (A C D) | A |
| **G4** | the cost model cannot price `log n` | nothing outright — it **misprices**. The planner cannot choose between a direct convolution and a transform-based one, and every plan containing a registration landscape is wrong in a known direction. **And the misprice now moves plans, where it used to be nearly inert.** Under the old objective — the phase's serial work, `cost_per_block × n_blocks` — a wrong per-voxel cost changed no decision: `n_blocks` cancels, redundancy falls monotonically, and the sweep answered "the largest candidate that fits" whatever the coefficient was. The objective is now `max(pool, channel)` and the block edge is a real per-phase choice, so a phase priced linear that is really `n log n` is priced wrong **in the pool bound**, which is the half that decides how finely to cut. §11.2 | a `cost_per_voxel_in_volume(volume)` beside the existing pair, additive with a default that preserves every current declaration. *Unverified:* whether `crate::statistics`' one-denominator-per-op calibration tolerates a second | **2** (A C) — unchanged; the objective change alters what the misprice *costs*, not who has an operation that is mispriced | A |
| **G5** | ~~there is no second input volume~~ → **closed.** A run can be handed `k` arrays beside image 0, and they are images in every sense the plan already had: read through `source_images`, fetched through `Environment::read`, priced by the same byte accounting *(§10 — and the mechanism this row prescribed turned out to be unbuildable, which is recorded there rather than quietly replaced)* | the entire channel family — unmixing, stain separation, crosstalk correction, channel arithmetic, ratio images, colocalisation, N-channel Boolean union; every two-frame temporal operation; a **supplied** shading or flat-field reference; and, though C did not name it, getting N mosaic tiles into one run at all. **All of it is now reachable**, and `ops::mixing` (G10) is the first consumer. What is *not* reached by it is anything needing those arrays at a different offset or resolution — that is G2, and G5 does not touch it | **The row's prescription was impossible, and both objections are kept because a register that recorded only the outcome would lose them.** *What this row said:* "image numbering in which images `0..k` are inputs and phase `p` writes image `k + p`, plus constructors taking a list." *Objection 1 — the executor.* `strategy.rs` addresses images positionally, as `env.read(task.phase, …)` and `env.write(task.phase + 1, …)`, at ~15 sites. Solving that for `k` inputs forces image 0 to be an input and puts the rest above `n_phases`, which is not the stated numbering at all; getting the stated one means rewriting the executor. *Objection 2 — the builder, and independent of the first.* A caller needs an input's address **before** it constructs the ops that read it (`Chain::source` takes the number; a `BlockOp` stores it), and the phase count is not known until `finish`, because `PlanBuilder::partition` lets a strategy choose. So `0..k` is unreachable from the builder even if the executor were rewritten. *What shipped instead:* a **disjoint high address range** — `ImageId::SUPPLIED_BASE = usize::MAX / 2 + 1`, `ImageId::supplied(i)` — with images the run writes numbered exactly as they were. The two properties `0..k` lacks are the whole reason: the address is knowable before a single phase exists, and **adding an input renumbers nothing**. The rest of the row was right and is what made it cheap: `Chain::Source`, `SourceInput`, `check_source_images`, image lifetimes and the byte accounting already worked above image 0 | **3** (A C D) — the count while it was open. C never named G5; §6 adjudicated that C needs it, which is why C is in the count and not in the "raised by" column | D |
| **G6** | there is no non-spatial axis to sweep | a genuine *window* along channel or time — a temporal median over dozens of frames, smoothing along a many-band spectral axis | a fourth axis on `Voxels`, `Reach`, `Anchor`, `BlockGeometry`, `BlockGrid`, `output_shape`, `side_region`, the tiling check, the budget arithmetic, the cache keys and the distributed placement. **D argues against closing it** — see §6 | **1** (D) | D |
| **C1** | a data-dependent reach cannot be bounded | displacement-field warping, arbitrary-plane reslicing, and with them the *apply* side of deformable registration | a **declared bound**: the caller states a maximum displacement, the op declares it as its reach and refuses at the block, by name, anything exceeding it — the shape `ops::voxelize` already takes. For a spline transform the bound is derivable from the control points, so nobody has to guess | **1** (C) | C |
| **C2** | a `Region` cannot begin below zero | padding, growing a canvas, tile placement before the global solve — the layout is not expressible until after the solve, and the solve needs the layout | a signed origin, **or** a written-down convention that re-origining happens before planning and the crate never sees a negative coordinate. C judges the convention the cheaper answer and the one to write | **1** (C) | C |
| **C3** | axis permutation is carried but not acted on | transpose, 90° rotation, reslicing along a non-native axis | `reach::Space` already carries `axes: [usize; 3]` and it is fingerprinted; a permuted branch needs lattice, read extent, valid region and anchor permuted *together*. `BlockGrid` and `Anchor` are the work | **1** (C) | C |
| **G7** | no barrier phase | nothing outright — it **costs**. A fragment-and-join reduction takes N+1 passes for N blocks, because phase 1's whole-lattice fragment reach is also its halo. A reduction whose answer is one scalar reads the whole volume per block, exactly as a hole fill does | a phase declared to start when the previous one finished, stating the dependency without a halo. `fill.rs` names this as the way out and as the open architectural question. **The single largest cost lever in family B** | **1** (B) named; A's and D's global reductions pay the same toll by the same route | B |
| **G8** | no scalar broadcast inside an iterative phase | level sets, active contours, Chan–Vese, SLIC, Costes automatic thresholding — every scheme whose per-iteration local stencil consumes a whole-volume reduction recomputed each round | a substage able to read a value reduced over the previous substage's whole output. Same mechanism as G7 with a loop around it. **B calls it the highest-leverage item in its document** | **2** (B D) | B |
| **G9** | ~~an op cannot require one axis be left whole~~ → **the planner cannot be told not to cut an axis** *(partially answered by §8.5; re-scoped, identifier unchanged)* | the exact Euclidean distance transform (cost, not correctness — a lattice that cuts a whole-axis-reach axis with the full halo is redundant, not wrong; and see §8.4 for what that op is and is not evidence about); any temporal alignment that must not tear a stack. **The correctness half is answered.** An op declaring `AxisReach::All` in `Frame::Source` now mandates that the axis is left whole **or** given a whole-axis halo — declared by the op, enforced by the tiling check that already existed, with no constraint type added and none needed | **Re-scoped to the planner-facing half only.** `Constraints`/`BlockConstraint` still cannot express "do not cut axis *k*", so the enumerator proposes a lattice that will be **refused** rather than avoiding it — a search that wastes candidates and a caller who sees an error instead of a plan. `BlockConstraint::FullExtent(axis)`, or an `Extent` of `Option<usize>`, is still the shape; it is now a **planning-quality** change and no longer a correctness one | **2** (B D) named, C implied by any separable sweep — unchanged, and now counting who wants the *planner-facing* half | B |
| **G10** | ~~no K-ary reach-0 op shell~~ → **closed.** `ops::mixing` ships the shell (`TupleOp`), the kernel trait (`TupleKernel`) and the first kernel (`LinearMap`, the per-voxel matrix); `tests/tuple_map.rs` pins the map, the mixing, decomposition invariance, that every input is really read, and that side outputs are still terminal | per-voxel classification, argmax over C probability channels, linear unmixing, stain separation, crosstalk correction, colour-space conversion, and every windowed temporal filter — **all now buildable**, the extra inputs being supplied images (G5) and the extra outputs side outputs | **The clause was right about the shape, contained one error, and the error has since been fixed — all three are kept, because the middle one is the useful part.** *What this row said:* "one `BlockOp` with C−1 `source_inputs` and C′−1 `side_outputs`. No new trait and no new axis." The shape is exactly that, no axis was added, and "not a `Combine`" holds for the reason given — the trait has no side outputs. *What it did not see:* **`BlockOp::apply_side` was not handed the `SourceInputs`**, so a side output that is a function of the op's source inputs could not be computed there; `ops::mixing` shipped with a per-block map keyed by the buffer offset, filled by `apply_with` and drained by `apply_side`. *What has since landed:* the argument is threaded from the executor's call site (`strategy.rs:1032`) through `Chain::apply_side` and `Environment::apply_side`, and the per-block map is **deleted, field and all**. **So this clause is now true exactly as written.** What it bought is worth more than the tidiness: `apply_side` is a **total function of its operands** rather than of its operands plus what an earlier call left behind — the old shape was correct only while two calls stayed paired, and it held whole `f64` blocks resident that no counter knew about, in a crate whose side outputs exist *because* 95.2 MB was once counted against 158.6 MB written. The price is that the inputs are streamed twice: **1.08–1.10×** at K = K′ = 16, with the tiling's 2.5–2.8× fully retained and the flop count unchanged. Blast radius **4 overriding implementors out of ~30**, a default absorbing the rest, and nine implementors in the sibling application crate untouched; and the documented "side outputs and source leaves do not compose yet" limitation is **deleted** as a side effect. §11.3 | **2** (B D) when open; **0** now, and this time with **no residue either** — re-derived rather than carried. The residue this row briefly held costed **B** (a classifier over K channels writing C class maps) and **D** (unmixing writing C′ channels beside a residual) and blocked neither; both now pay 1.08–1.10× of streaming instead, which is a price and not a gap. A's arity-2 arithmetic combines are the adjacent, smaller case and are still unwritten | B and D independently |
| **G11** | rows cannot be decomposed by spatial region | candidate link generation across a series — the pairs of rows within a spatial radius across one step | a fragment op reading two row streams under a *spatial* neighbourhood. `ops::rows` decomposes by row range with **no overlap**, on correctness grounds — an overlap duplicates a row and no downstream check can tell it from a real one — so this is a new shape, not a parameter on an existing one | **1** (D) | D, correcting B |
| **G12** | `iterate::Operand::Running` is singular | blind deconvolution and any alternating-minimisation scheme | a second pair of alternating private buffers and a convergence rule over both. The restriction is deliberate and documented; worth recording, not worth building until something asks | **1** (A) | A |
| **G13** | point coordinates cannot be sub-voxel or negative | a chain of point transforms rounds at every step, so evaluating a transform at a point set is not worth having; sparse feature-tracking output has nowhere exact to land | `[f64; 3]` in the table, or a documented fixed-point convention. `points::Point` holds `[usize; 3]` and `ops::rows::scaled_index` refuses a negative factor because "a table holds `usize` coordinates" | **1** (C) | C |
| **G14** | **no declaration anywhere is checked against the fetch** *(partially closed by §8.5: source-frame `AxisReach::All` is now checked; **silence is not**)* | nothing outright — it admits a **wrong answer**. A projection that reads only its own block is accepted by every guard, has exactly the right shape, and is wrong at every position; a fetch covering *half* the collapsed axis is accepted too. `Decomposition::check` verifies that a block's `source` lies **inside** the image it reads (`decomposition.rs:713-728`), never that it *covers* what the reach claimed. The one guard that does look at a declaration (`decomposition.rs:702-711`) checks only that a source-lattice reach is accompanied by *some* per-block fetch, not by a sufficient one. **The only gap in this register that is about correctness rather than capability or cost** | **Landed for the declared case** (`decomposition.rs:763-791`), which refuses a fetch that does not cover a declared whole axis and names the axis, the block, the fetched range, the required range and the phase's own extent. **The residue is silence** — "Part 2 holds an op to what it *said*; it cannot make an op speak" — and every fetch that is not a whole-axis claim, an affine map among them, has nothing to be checked against. Closing the residue wants a **total** per-axis rule where no axis can be silent (§9's proposal, and its totality is the load-bearing part) | **4** (A B C D) — re-derived in §8.3 after the landing; the number holds, the reachable failures are fewer, and the residue is sharper | this correction pass |
| **G15** | **a mask cannot be held as one bit** — there is no `Bool`-producing threshold, and the fan-in that would join one is bound to `f64` | nothing outright — it **costs, in the one currency a tile-scale stage runs out of**. A verdict is an `f64 → f64` `MapFn` (`voxelwise::Threshold` is one), so a thresholded arm produces `f64`; `LogicCombine::accepts` takes only `Bool \| F64` **and requires every branch of a fan-in to agree**, so one arm that cannot narrow binds all of them; and the image the phase writes is then `f64` — **eight bytes a voxel for a one-bit fact**. Measured on the binarize stage at tile scale: the peak falls **57.853 → 46.28 GiB** if the mask images are `Bool`, which is the difference between a stage that runs at a size and one that does not. Family B reached the same cost from the other side and wrote it down — the reconstruction shell `accepts` `f64` only, so "a binary reconstruction is an `f64` volume's worth of memory for a one-bit question" (B §3) | a threshold op that **produces** `Bool` rather than a mapped `f64`, and a `Logic` sink that joins `Bool` arms. `NarrowOp::to_mask` is already the crate's way into `Bool` and already carries the mask convention (non-zero is true, and `Narrowing::new` refuses `Dtype::Bool` by name because a two-valued target is a comparison and not a rounding), so the convention is not the missing part — the missing part is a verdict that lands on it without a round trip through `f64`, and an `accepts` that does not force the widest arm on the rest. *Unverified here:* whether narrowing **every** arm of a real fan-in is a general answer; the binarize work reports it is not, because agreement binds each arm to the weakest, and this pass did not build the counter-example | **3** (A B D) — derived, not assigned: **A** (Boolean logic between two volumes, §2's row, over thresholded branches), **B** (every binary-morphology and reconstruction consumer, and B's own note above), **D** (§4.2's N-channel Boolean union of masks, the one row D calls a useful surprise). **C** holds no mask image: its isosurface takes a scalar field and the threshold that produced it is B's by §0 | the binarize work, and B independently |
| **G16** | `Decomposition` cannot say what a plan's **peak** costs | nothing, and it blocks nobody — but it is the single number that decides whether a stage can run at a size, and it is **open-coded in three consumer test files** — *reported by the work that asked for it and not re-counted here; there is no such walk under `blockflow`'s own `tests/`, which is consistent with the consumers being outside this crate.* The walk is the same one every time: for each phase, the images alive across it, priced by `volume_at × dtype`, maximised over phases. `readers_of_image`, `images_dead_after`, `image_visibility` and now `image_kind` are all the pieces, all public, all already agreeing with each other; nothing folds them. Three hand-rolled copies of a lifetime walk is three chances to disagree with the executor about when an image dies, and the executor's answer is the binding one | `Decomposition::peak_image_bytes()`, derived on exactly the argument `image_visibility` is derived on — a field that could disagree with the arithmetic is a field that eventually does. **The three kinds matter to it**: a supplied input is alive for the whole run and is not the run's to free, an intermediate may be dropped and rebuilt, and an output has a materialisation obligation, so a peak that treats all three alike is not the number a caller wants. `docs/design/images-and-phases.md` prices one such peak by hand — 67.8 GiB of images alive at one phase, against a measured `VmHWM` of 124.9 GiB — and records that the figure is a property of the *linearisation* and not of the DAG, which is the second reason to have it in the crate rather than in three tests | **0** — it blocks no operation in any family. Recorded with a zero rather than left out, on §8.2's precedent: the count and the *kind* of a gap are different facts, and a row that carried only the count would read as unimportant | this correction pass |

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
  by this and stands**, because it rests on the reach-0 sort of §3.1 and on the
  storage contract of §3.4, neither of which depends on how the rank cap is
  read. What is corrected is the reasoning, not the answer.

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
exist in this repository.** Verified: 45 citations across 28 files under `src/`
and `tests/`, and no such file anywhere in the tree. *(When the four documents
were written, `docs/` contained only `ops-survey/`. It now also contains
`docs/design/images-and-phases.md`, landed by another worker, which records the
same dangling-citation count as a known state and does not repair it.
`BLOCK_OPS.md` itself is still absent.)* Every design argument attributed to it
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
