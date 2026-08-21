<!-- SPDX-License-Identifier: MIT -->

# Point operations, filtering and transforms

*A survey of family A: what a general image-processing library is expected to
offer in this area, what `blockflow` has today, and — the part worth keeping —
**whether this crate's decomposition model can express each thing, and at what
cost**.*

Two sibling documents cover the rest: **B** takes segmentation, morphology and
measurement; **C** takes geometry, registration and composition. Where an
operation straddles a boundary it is named here in one sentence and handed
over.

---

## 1. Why the classification, and what the columns mean

A feature checklist against ImageJ would be easy to write and useless to read.
Every library in the reference set has a Gaussian blur; the question that
decides what this crate can become is not *does it exist elsewhere* but **what
shape does it have when the volume does not fit in memory**. That shape is what
`Reach`, `Decomposition` and `CostModel` are made of, and it is the column that
will still be true in a year when the op list has doubled.

Five categories, all of which this crate has already earned by building
something in them (or by writing down, in a module header, exactly why it did
not):

| # | Category | What it means here | The crate's worked example |
|---|---|---|---|
| **1** | **Bounded reach** | A halo the op declares: `AxisReach::Bounded { lo, hi }`, possibly asymmetric, possibly `PerBlock`. Cheap, blockable, the ordinary case. | `ops::rank`, `ops::smooth`, `ops::ridge` |
| **2** | **Whole-axis reach** | `AxisReach::All` on one axis, bounded on the others. A separable sweep along one axis; the planner may still cut the other two. | Expressible (`Reach::per_axis`); **nothing in `ops/` uses it today** — verified by grep. See the correction below: *which space* the reach is stated in decides whether it is expressible at all, and this row conflates two declarations that behave oppositely |
| **3** | **Whole-volume, resident** | No halo bounds it — every output element depends on every input element. `Reach::all()` makes the op a planning barrier: its phase gets a single block. | `ops::watershed` declares `Reach::all()`; `ops::fft` is the pure case and declines even that (§9) |
| **4** | **Iterative to a fixed point** | A loop of substages whose stopping rule is a whole-volume reduction. `IterativeOp` + `iterative_phase` exist for exactly this. **The phase's external reach is the *per-substage* reach**, however many substages run — depth is paid in private round trips, not in halo. | `crate::iterate`, `tests/iterative_phase.rs`; `ops::reconstruct` and `ops::directional`'s fixed-point loop |
| **5** | **Not expressible today** | And *why*, specifically. §11 consolidates these. | |

> **Corrected — measured** (`tests/collapsing_phase.rs`; consolidated in the
> index's §8). Two corrections to the table above, and they are connected.
>
> **Category 2 is two declarations, not one, and they behave differently.**
> `AxisReach::All` in the **phase** frame — the default `Space::phase_voxels()`
> — plans and works; that is what a same-rank separable sweep wants, and it has
> a user outside this crate. `AxisReach::All` in `Frame::Source` was, when this
> was measured, refused unconditionally.
>
> **That second half has since changed:** the geometry change landed, and
> source-frame `All` is now granted on an axis the op **consumes** and checked
> against the block's fetch. So both frames plan, and **any line in this
> document that says "declare `All`" has to say which one** — §13's item 3 and
> §8's VTK paragraph both do, now. The short version: **phase frame for a
> same-rank sweep, source frame for an axis the phase consumes and does not
> reproduce.** Against an axis of extent 1 the phase-frame form is *vacuous*
> — `is_whole` requires `extent > 1` — so it is accepted without claiming
> anything, which is the trap worth knowing.
>
> **Category 5 was the wrong cell for a projection.** See the G1 correction
> immediately below.

The four known structural gaps, restated once so the later tables can point at
them by number:

* **G1 — the geometry cannot declare a collapsed or a broadcast axis.**
  *(Renamed in the index's §8.1. Identifier unchanged so every citation in the
  four documents survives.)* `Reach` states per-axis halo widths on a same-rank
  output, and there is no way to say either "this axis is consumed whole and
  written at extent 1" or "this axis is held at 1 while the output grows".
  A *side output* may have any rank — `Output::shape` is a `Vec<usize>` — but
  side outputs are terminal: nothing in the chain reads them back.

  > **Corrected — measured.** *What this document said:* "no rank-reducing
  > phase — `BlockOp::output_shape` is `[usize; 3] -> [usize; 3]` and `Voxels`
  > is eleven `Array3` variants, so a phase cannot produce an (n−1)-D output."
  > *What was measured:* the rank was never the obstruction and the `Array3`
  > floor is not a cap to work around — it is a **decision**. `src/voxels.rs`
  > argues for rank 3 on measured grounds against `ArrayD`: an image *is* a
  > volume, one and two dimensions are modelled as three with degenerate axes,
  > and a dynamic rank "bought nothing and cost an indirection per index." So
  > `[X, Y, 1]` is a legal `Voxels`, `output_shape` returns it, and a collapsing
  > phase **plans, runs and is decomposition-invariant today** — byte-identical
  > to a whole-volume reference across 25 cuts of the two free axes. It worked
  > only through the `†` cross-grid escape, that is, by *not* declaring the
  > dependency, which is family C's `2†` and is now verified. **This document's
  > `X (G1)` for a projection is withdrawn.** What G1 names, correctly, is the
  > missing *declaration* — and its other half, the **broadcast**
  > `[X, Y, 1] -> [X, Y, Z]`, which family A needs for the map half of every
  > reduce-then-map workflow in §3.
  >
  > **Updated again: the collapse half landed.** The dependency can now be
  > stated — `AxisReach::All` in `Space::source_voxels()` plus `with_sources` —
  > and is checked against the fetch, so the escape is strictly weaker for a
  > whole-axis dependency and should not be recommended for one. **The broadcast
  > half did not land**, so every "map" half in §3 is still escape-only and
  > still unchecked, and G1's name has been narrowed to that half. `2†` stands
  > for both, since the fetch is still stated per block.
* **G2 — no phase whose inputs sit at different offsets.** A `Decomposition`
  reads one image and maps a block to one region of it. `Chain::Source` and
  `SourceInput` add *more* images, each still read at the block's own extent.
* **G3 — no complex element variant.** `Dtype` has twelve real variants and
  `dtype.rs` has a test asserting `from_numpy_name("complex64") == None`. A
  spectrum cannot be a plan image.
* **G4 — the cost model cannot price `log n`.** `cost_per_voxel()` takes no
  volume argument, so an op's declared unit cost cannot grow with the size of
  the thing it transforms. `cost_per_voxel_in(block)` exists and lets a cost
  depend on the *block*, and its contract requires it to agree with
  `cost_per_voxel()` in the limit where the block spans the volume — which is
  exactly the limit a resident transform lives in. So a transform prices as
  linear.

  > **Unchanged as a statement, and it now costs more than it did.** The
  > partition search's **objective** has changed — see §10's G4 entry, which is
  > where the arithmetic is. The short form: the old objective was monotone in
  > the block edge, so a wrong per-voxel cost changed no decision; the new one is
  > `max(pool, channel)` and the block edge is a real per-phase choice, so a
  > coefficient that is wrong by `log n` now chooses a grid.

**Sources**, weighted by what is actually used rather than by what exists:
ImageJ/Fiji's `Process` menu and its widely-used plugins; ITK's filtering,
denoising and frequency modules; VTK's imaging filters where their emphasis
differs; OpenCV's `imgproc` filtering and `photo` restoration; scikit-image's
`filters`, `restoration`, `exposure` and `feature`. Where OpenCV and
scikit-image both expose a primitive that ImageJ offers only as a plugin, that
agreement is treated as evidence the primitive belongs in a general library.

---

## 2. Point and voxelwise operations

`ops::voxelwise` is the whole of this area today, and it is **a mechanism
rather than a catalogue**. `VoxelwiseMapOp<M: MapFn>` applies any pure
`f64 -> f64`, and there is a blanket `MapFn` impl for closures, so any unary
point operation is one line at the call site. What is *named and shipped* is
three maps: `Identity`, `Not`, `Threshold`. `WidenOp` widens any dtype to
`f64`; `NarrowOp` casts back down with a stated `Narrowing` (rounding rule,
saturating clamp, NaN→0) or to `Bool` under the mask convention.

`CombineOp` and `LogicCombine` are the two-input side, and they are **Boolean
only** — `And`, `Or`, `Xor`. There is no arithmetic between two volumes
anywhere in the crate.

| Operation | Where it is standard | Category | Have it? |
|---|---|---|---|
| Unary arithmetic: add/multiply constant, invert, square, sqrt, reciprocal, abs | everywhere | 1 (reach 0) | **Mechanism yes, names no** — one closure each through `VoxelwiseMapOp` |
| Gamma, log, exp, sigmoid | ImageJ `Math`, ITK unary functors, skimage `exposure.adjust_*` | 1 | Same: expressible, not shipped as named ops |
| Clamp / intensity windowing | ITK `ClampImageFilter`, `IntensityWindowingImageFilter`, OpenCV `inRange` | 1 | Via closure or `Narrowing`'s saturating clamp; no named op |
| Lookup table (LUT) remap | OpenCV `LUT`, VTK `vtkImageMapToColors`, ImageJ `Apply LUT` | 1 | **No.** A table-driven map is a different cost and a different `constant_maps_to` story from a closure; worth its own op |
| Type conversion + rescale (`img_as_float`, `RescaleIntensity`, `convertScaleAbs`) | all five | 1 | **Partly** — `WidenOp`/`NarrowOp` convert; **rescaling by the source range is a whole-volume reduction and is missing** (see §3) |
| Binary arithmetic between two volumes: add, subtract, multiply, divide, min, max, absdiff, weighted sum | all five; ImageJ `Image Calculator` is one of its most-used commands | 1 (reach 0, two operands) | **No — and this is the largest cheap gap in the family.** `ops::background::DifferenceCombine` is a two-branch subtract and is the only arithmetic combine that exists; it is a `Combine` sink in a `Chain::parallel` diamond, so the pattern is proven and generalising it is mechanical |
| Boolean logic between two volumes | all five | 1 | **Yes** — `CombineOp` (against a resident operand) and `LogicCombine` (fan-in over branches) |
| Masking (`MaskImageFilter`, `vtkImageMask`, `Apply Mask`) | all five | 1 | **Effectively yes** via `And` on a mask, but no named masked-copy op with a fill value |
| Free-form expression over one or two volumes | ImageJ macro `Math`, `Image Calculator` | 1 | Expressible per-op; nothing composes an expression tree |

**Category note.** Everything in this section is reach 0. The dividing line is
not decomposability, it is *arity*: a one-input point op is a `BlockOp`; a
two-input one is either a `Combine` at the foot of a diamond or an op with a
resident `Arc<Voxels>` operand. Both work today. Nothing here needs framework
work — it needs ops written.

> **A third arrangement exists now, and it is the general one.** Since G5 and
> G10 closed, a K-input reach-0 op is `ops::mixing::TupleOp`: the other operands
> are images — computed by this run or **handed** to it — read at the block's own
> fetch region, with K′ results written as side outputs. That is the shape this
> section's two-operand cases are the small end of, and it removes the last
> reason for a resident `Arc<Voxels>` operand where the operand is a whole
> acquired volume. The category note is unchanged: still reach 0, still arity,
> still ops nobody has written.

---

## 3. Histogram and intensity statistics

This is the thinnest area relative to what the reference libraries offer, and
the reason is structural rather than accidental.

A histogram of a volume is an **(n−1)-D-or-less output from an n-D input** —
one bin array from a volume. That is **G1** exactly. VTK exposes it as
`vtkImageAccumulate`, ITK as `Statistics::ImageToHistogramFilter`, OpenCV as
`calcHist`, skimage as `exposure.histogram`, and ImageJ puts it on `Ctrl+H`;
all five treat it as a primitive. In this crate a histogram exists **only
inside a window**: `ops::sliding` maintains one incrementally along a scan line
and `ops::local::Statistic::Isodata` bins one per lattice sample. Neither
produces a histogram anyone can read.

| Operation | Category | Have it? |
|---|---|---|
| Whole-volume histogram | **5 (G1)** — the output is a bin array, not a volume. See §11 for the two ways out | **No** |
| Whole-volume min/max/mean/percentile statistics | **5 (G1)**, or 3 as a degenerate resident reduction | **No** as a phase. `crate::statistics` is planner telemetry, not image statistics; `ops::tabulate` reduces *per label region* into a `Table` and is the nearest thing — that is family B's op, and its shape (a `Table` side output, associative partial merges) is the template a global reduction should copy |
| Contrast stretching / percentile stretch (`RescaleIntensity`, `enhance_contrast` with saturation) | **5 (G1)** for the statistics pass, then **1** for the map | **No** — and this is the clean two-phase example: a reduction phase producing two numbers, then a reach-0 map that reads them |
| Global histogram equalisation | same shape: reduce, then map through the CDF | **No** |
| Local / adaptive histogram equalisation (CLAHE) | **1** — it is a tiled statistic plus an interpolation, which is precisely `ops::local`'s and `ops::lattice`'s mechanism | **No**, despite `ops::local` having every part of the machinery. Widely used in all five sources; **the strongest single candidate for the next op in this family** |
| Histogram matching to a reference | **5 (G1)** twice over — two CDFs, then a map | **No** |
| Threshold *statistics* (Otsu, isodata, Li, triangle, Yen…) | 1 windowed / **5 (G1)** global | **Partly** — `ops::local::Isodata` computes a Ridler–Calvard threshold per lattice window, one method of the dozen ITK and ImageJ ship. Global variants need G1. *The thresholding **decision** is family B; only the statistic is mine.* |
| Percentile of a window | **1** | **Yes** — `element::Rank`, `Percentile`, and two explicit truncation conventions for clipped windows |
| Local mean / standard deviation on a coarse lattice, interpolated back | **1** (reach = lattice distance + window radius) | **Yes** — `ops::local`, and split into two priced phases by `ops::lattice` |

**One correction to the three rows above** (contrast stretching, equalisation,
histogram matching). Each is stated as "a reduction phase, then a reach-0 map
that reads the numbers back", and each carries `5 (G1)` for the reduction half
only. The *map* half is a G1 instance too, and of the other kind: a scalar or a
per-plane statistic applied over a volume is a **broadcast** — a source axis
held at extent 1 while the output grows — which is the same missing declaration
seen from the other side (index §9). It **runs today** through the `†` escape
plus `BlockOp::takes_extent_from_placement`, so neither half of a reduce-then-map
workflow is unbuildable.

*Updated after the geometry change landed.* The **reduction** half can now be
declared and is checked: `AxisReach::All` in `Space::source_voxels()` plus
`with_sources`, and a fetch that does not cover the axis is refused by name. The
**map** half cannot — a pinned axis has no declaration — so it is still
escape-only and still unchecked. Of the pair, one is now honest and one is not,
which is worth knowing before writing either.

**Why the split matters more than the histogram.** `ops::lattice`'s header
makes the general argument: at a spacing of 25, fusing the statistic and its
interpolation costs 25 voxels of *fine-resolution* halo per side per block that
neither phase needs alone. Any adaptive-histogram op should be built as two
phases from the start for the same reason.

---

## 4. Linear filtering

| Operation | Category | Have it? |
|---|---|---|
| Gaussian blur, separable, per-axis sigma | **1** | **Yes** — `ops::smooth::Gaussian`. Sigma 0 on an axis gives a plane-wise blur, so 2-D is the special case as the crate intends. Cost is charged per *tap* (sum of kernel lengths, not product) — the planner knows it is separable |
| Convolution with an arbitrary kernel | **1**, or **3** for a large kernel done by transform | **No.** Every one of the five sources has it (`filter2D`, `vtkImageConvolve`, `ConvolutionImageFilter`, ImageJ `Convolve`). This is a real gap and an easy one |
| Separable convolution with caller-supplied 1-D kernels | **1** | **No** as a general op — the separable machinery exists but is private to the Gaussian path |
| Difference of Gaussians | **1** | **No.** Two `SmoothOp`s in a `Chain::parallel` joined by a subtract `Combine` is exactly `ops::background`'s diamond, so the shape is proven — it wants the subtract combine from §2 |
| Laplacian / Laplacian-of-Gaussian | **1** | **No** as a named op. `ops::ridge` computes the full Hessian (all six second derivatives) internally, so the trace is one fold away |
| Gradient magnitude, Sobel, Scharr, Roberts, Prewitt | **1** | **No.** ImageJ's `Find Edges`, `vtkImageGradient`, `GradientMagnitudeRecursiveGaussian`, `cv::Sobel`, `filters.sobel` — this is unanimous across the sources |
| Unsharp mask | **1** | **No.** Blur, subtract, scale, add — a diamond, again wanting the arithmetic combine |
| Box / mean filter | **1** | **No** as a named op; `ops::local::Statistic::Mean` computes a windowed mean but on a **sample lattice with interpolation**, which is a different (and, at spacing 1, more expensive) thing |
| Integral image / summed-area table | **2 or 3** — a prefix sum is `AxisReach::All` per axis, and a 3-D one is three such sweeps | **No.** OpenCV `integral` and every constant-time box filter rest on it. *This is the crate's most natural category-2 op and there is currently no category-2 op at all* |
| Binomial filter | **1** | **No** (a fixed separable kernel; falls out of arbitrary separable convolution) |
| Highpass by subtraction of a lowpass | **1** | **No** as an op; two ops and a subtract |

**Reach is derived, not written down.** `ops::smooth` declares
`gaussian_radius(sigma, truncate)` and no more, and its header spends a
paragraph explaining why it does *not* copy `ops::ridge`'s `+ 1` — that `+ 1`
is the second-difference stencil composing with the smoothing, and a smoothing
has no stencil on top of its convolution. The difference is one voxel and it is
argued rather than guessed. Any new linear filter should be held to the same
standard: an over-declared reach is a real fetch at every block.

---

## 5. Boundary conventions — who means what by "reflect"

Worth its own section because this crate has already been bitten by it, and
because a decomposed run makes the question *harder*: a convention is a rule
about the **array's own edge**, which under decomposition is the volume's face
where the volume has one, and is otherwise an interior position the halo keeps
every core voxel's taps away from. `ops::smooth`'s acceptance suite asserts
that a decomposed run reflects about the **volume** and not about the block.

`ops::ridge::Boundary` — the crate's only boundary type, reused by `smooth` —
has exactly two variants:

* **`Clamp`** (default): `a a a a | a b c d | d d d d`.
* **`Reflect`**: period `2 * extent`, mirror **half a sample outside** the edge,
  so `-1` reads `0`. It folds *repeatedly* by the periodicity, so a radius-40
  kernel on a 3-voxel axis is an ordinary call rather than a corner case.

The distinction that matters is whether the mirror lands *between* `-1` and `0`
or *on* index `0`. The two differ in every boundary voxel:

| Convention | Sequence at the low edge | This crate | OpenCV | SciPy / scikit-image | ITK | VTK / ImageJ |
|---|---|---|---|---|---|---|
| Replicate nearest sample | `a a a \| a b c d` | **`Clamp`** (default) | `BORDER_REPLICATE` | `nearest` | `ZeroFluxNeumannBoundaryCondition` (the default for most neighbourhood filters) | the usual behaviour for ImageJ's convolutions |
| Mirror, half-sample, period `2n` | `c b a \| a b c d` | **`Reflect`** | `BORDER_REFLECT` | `reflect` (a.k.a. `grid-mirror`) | — | — |
| Mirror, whole-sample, period `2(n-1)` | `d c b \| a b c d` | **absent** | `BORDER_REFLECT_101` (= `BORDER_DEFAULT`) | `mirror` | — | — |
| Constant fill | `k k k \| a b c d` | **absent** | `BORDER_CONSTANT` | `constant` / `grid-constant` | `ConstantBoundaryCondition` | — |
| Periodic / wrap | `b c d \| a b c d` | **absent** | `BORDER_WRAP` | `wrap` / `grid-wrap` | `PeriodicBoundaryCondition` | implicit in every frequency-domain filter |

Three things follow.

1. **The two names collide across libraries.** SciPy's `reflect` is OpenCV's
   `BORDER_REFLECT`; SciPy's `mirror` is OpenCV's `BORDER_REFLECT_101` — and
   OpenCV's *default* is the one SciPy calls `mirror`, while SciPy's default
   for `gaussian_filter` is the one OpenCV calls `BORDER_REFLECT`. Anyone
   porting a pipeline between them changes every edge voxel by accident. This
   crate's `Reflect` is the SciPy/OpenCV `reflect` sense, and its doc comment
   says so explicitly, which is the right amount of paranoia.
2. **The gaps are `wrap` and `constant`.** `wrap` is not a nicety: a
   frequency-domain filter is *inherently* periodic, so any future
   transform-based op is implicitly asserting a convention this crate cannot
   currently name. `constant` is what several published algorithm definitions
   assume.
3. **A boundary convention changes no reach.** Whichever rule resolves a tap,
   the position it lands on is inside `[v - r, v + r]`, because a fold brings an
   offset that left the array back toward the edge it left by. This is stated in
   `ops::smooth`'s header and tested. It is the property that makes adding
   `Wrap` and `Constant` a local change.

---

## 6. Non-linear filtering

| Operation | Category | Have it? |
|---|---|---|
| Rank / order statistics: median, min, max, arbitrary percentile | **1** | **Yes, and this is the crate's strongest area.** `ops::rank` over an arbitrary `StructuringElement` (box, two ellipsoid conventions, per-axis asymmetric extents, optional decimation), with a *masked* variant where a second `bool` image supplies population membership at every offset |
| Sliding-window histogram acceleration for rank filters | **1** (same reach; different cost) | **Yes** — `ops::sliding`, with `ScanPlan` decomposing the element into leavers and joiners along the cheapest axis. It is the crate's only user of `cost_per_voxel_in(block)`, because priming is `O(window)` per line and therefore `O(window / line)` per voxel |
| Other windowed-histogram statistics: entropy, majority/modal, pop, autolevel, local Otsu | **1** | **No.** `sliding::HistogramQuery` is an object-safe trait built for exactly this and `RankQuery` is its only implementor. skimage's `filters.rank` module ships fifteen of these on the same machinery — **the closest analogue to `ops::rank` in any of the sources, and the clearest map of where this one could go** |
| Local variance / standard deviation | **1** | **Yes** — `ops::local::Statistic::Deviation`, though on a sample lattice rather than densely |
| Bilateral filter | **1** | **No.** In OpenCV, ITK, skimage and as a much-used ImageJ plugin — four-way agreement |
| Non-local means | **1** with a large halo (search window + patch radius) | **No.** OpenCV `photo`, ITK `PatchBasedDenoising`, skimage `restoration.denoise_nl_means` |
| Anisotropic diffusion (Perona–Malik, curvature flow) | **4** — a fixed iteration count is category 1 with reach `iterations x stencil`; a convergence criterion is category 4 | **No.** ITK devotes a module to it; VTK ships 2-D and 3-D versions; OpenCV has it in `ximgproc`. `ops::deconvolve` already demonstrates the fixed-count form (reach `2 * radius * iterations`) and `crate::iterate` the convergent one, so **both shapes are available** |
| Total-variation denoising | **4** | **No** |
| Wavelet denoising | **3** — a wavelet transform is a whole-volume basis change | **No** |
| Kuwahara / edge-preserving mean, sigma filter | **1** | **No** |
| Despeckle / single-outlier removal, remove-outliers | **1** | **No** as named ops (a 3×3 median is `RankFilterOp::median` with a small element; the *conditional* replacement is not) |
| Guided filter, domain-transform, rolling-guidance | **1**–**2** | **No** (OpenCV `ximgproc` only; below the "actually used" bar for a first pass) |

**The one structural note.** `ops::sliding` accepts only `Bool`, `U8`, `U16`,
`U32` — a value must index a bin. Float data goes through `ops::rank`'s dense
path instead. That is the same restriction skimage's `filters.rank` carries
(uint8/uint16 only), arrived at independently, which is a good sign the
restriction is inherent rather than a shortcut.

---

## 7. Multi-scale and shape from derivatives

| Operation | Category | Have it? |
|---|---|---|
| Hessian at a scale, closed-form symmetric eigenvalues | **1** (reach = Gaussian radius + 1 for the stencil) | **Yes** — `ops::ridge::{hessian_at, symmetric_eigenvalues}`, non-iterative |
| Ridge / tube enhancement from Hessian eigenvalues | **1** | **Yes** — `RidgeResponse` (bounded, saturating line/blob/strength terms) and `RatioResponse` (unbounded power-law), selectable via `Response`, with `Polarity::{Ridge, Valley}` |
| Blob and plate/sheet enhancement | **1** | **Partly.** The same eigenvalue triple determines all three geometries and the two `Response` folds discriminate line-like from blob-like, but there is **no separately named blob or plate measure** the way ITK exposes `HessianToObjectnessMeasure`'s dimension parameter and a sheetness measure |
| Scale space: evaluate over a list of sigmas, keep the voxelwise best | **1** (reach = max over scales) | **Yes** — `ops::ridge::ScaleSpace`, with γ-normalisation and an optional `U32` side output holding the winning scale index. Every scale is evaluated at **full resolution**; this is a scale-space search, not a pyramid |
| Image pyramid (Gaussian, Laplacian) | **1** per level, but the *chain* wants **G2** | **No.** Each level is a blur plus a decimation, and `ops::resample` can already change extent — so one level is buildable today. What is not, is an op that reads two levels *at different resolutions*, which is what a Laplacian pyramid's reconstruction and every coarse-to-fine algorithm needs. *Resampling itself is family C; the multi-scale use of it is mine.* |
| Structure tensor (gradient outer product), cornerness, coherence | **1** | **No.** Distinct from the Hessian and present in all five sources (`cornerHarris`, `cornerEigenValsAndVecs`, `feature.structure_tensor`, ITK's structure-tensor filter, VTK's gradient stack). `ops::ridge` is Hessian-only |
| Corner detection (Harris, Shi–Tomasi, FAST) | **1** | **No**. *Where a corner detector is used to seed a registration, that use is family C; the response image is mine.* |
| Blob detection by DoG / LoG / determinant-of-Hessian across scales | **1** for the response; the maxima are family B | **Partly** — the determinant is available from `ops::ridge`'s eigenvalues; DoG and LoG responses are not (§4) |
| Gabor / oriented filter banks | **1** | **No.** Note that `ops::directional` is **not** this — see §12 |

---

## 8. Frequency domain — the pure category-3 case

`ops::fft` is the most useful module in this survey, because it is the one
place where the crate has already written down, in detail, why an operation
does not fit.

What is there: `RealTransform2` (a 2-D real→half-complex forward transform and
its inverse, with plans built once and reused, two selectable backends);
`Correlation2` (linear cross-correlation via the correlation theorem, with a
`Padding` policy that guarantees no wrap contamination); `SquaredDifference`
(a normalised mean-squared-difference landscape over a window of integer lags);
and direct O(lags × overlap) oracles for both. The forward/inverse pair is
real, tested to `~1.5e-15` on a round trip, and `f64` rather than `f32` for a
stated correctness reason.

**Nothing in `ops::fft` implements `BlockOp`, and the absence is the
statement.** Its header gives three *independent* reasons, and it is worth
keeping them separate because they fail differently:

1. **Two inputs of different extents.** A `BlockOp` is handed one `Voxels` and
   writes one; a correlation landscape is a function of a *pair* of planes
   whose shapes need not agree. (**G2**.)
2. **An output space that is not the input space.** The landscape is indexed by
   **lag**. `output_shape` can rescale an axis; it cannot replace the coordinate
   system with a different one.
3. **No element type to hold a spectrum.** (**G3**.) The intermediate the whole
   method rests on cannot be named in the pipeline's own algebra.

And underneath all three, the reason a transform is category 3 at all: **a
Fourier coefficient is a sum over every element of its input.** There is no
halo that makes one.

| Operation | Category | Have it? |
|---|---|---|
| Forward / inverse transform of a plane | **3**, and today **5** as a *phase* — see G3 | **Yes as a library type**, not as an op. 2-D only; there is no 3-D transform |
| Linear cross-correlation | **3** (resident), **5** as a phase | **Yes as a library type** — `Correlation2` |
| Normalised squared-difference landscape | **3** / **5** | **Yes** — `SquaredDifference`, with the energy terms computed as exact compensated rectangle sums rather than by extra transforms |
| Phase correlation | **3** / **5** | **No** — the conjugate product is there; normalising it by magnitude is not. *Its main consumer, translation registration, is family C; the transform primitive is mine.* |
| Frequency-domain filtering: ideal / Butterworth / Gaussian low- and high-pass | **3** / **5 (G3)** | **No.** VTK ships these as named objects (`vtkImageIdealLowPass`, `vtkImageButterworthHighPass`) where ITK puts them in a remote module; skimage has `filters.butterworth`; ImageJ's `Bandpass Filter` is among the most-used items on its `FFT` menu |
| Band-pass and stripe / periodic-artefact removal | **3** / **5 (G3)** | **No.** Present in every source in some form, usually either as a directional band-stop in the spectrum or as a wavelet–transform hybrid. `README.md` already names "stripe / illumination correction" as the general name for one such op, so the crate expects to grow one |
| Spectrum display, quadrant swap, magnitude/phase extraction | **3**, and all need **G3** | **No** |
| `mulSpectrums`-style filtering of a stored spectrum | **5 (G3)** | **No**, and cannot be until a spectrum is an image |
| Convolution of a large kernel by transform | **3**, or **1** if the kernel is small enough to halo | **No.** OpenCV's `filter2D` switches to a DFT above a size threshold and ITK has `FFTConvolutionImageFilter`; the *choice between two ways of computing the same thing* is exactly what `cost_per_voxel_in` was built to let the planner make — but **G4** means it cannot price the transform side |

**VTK is the useful comparison here**, because its execution model is the
closest of the five to this crate's. VTK filters answer a
`RequestUpdateExtent` by declaring what input extent they need for a requested
output extent — the same question `Reach` answers — and `vtkImageFFT` answers
it by demanding the whole extent along the transformed axis, which is
`AxisReach::All` on that axis and bounded on the others. **That is a category-2
answer to a problem this crate currently treats as category 5**, and it is the
strongest argument in this document that a per-axis 1-D transform pass is worth
building before a full 3-D one: the crate's `Reach` can already say what VTK
says.

**With one precision, added after measurement, and updated once.** It can say it
*in the phase frame*, which is the frame a same-rank pass wants and the one
whose numbers are measured against the array the phase itself writes. A pass
that also changed shape would want `Frame::Source` — where `AxisReach::All` was
refused unconditionally when this was measured, and where, since the geometry
change landed, it is **granted on an axis the phase consumes and checked against
the fetch**. So the claim above stands for a 1-D transform pass that writes the
same extent it read, and a shape-changing one now has its own frame to say it
in. **The distinction is one argument in a call, the two behave differently, and
a document recommending `All` has to name the frame** — index §8.5.

---

## 9. Restoration

| Operation | Category | Have it? |
|---|---|---|
| Iterative deconvolution, Richardson–Lucy family, spatial domain | **1** — reach `2 x radius x iterations`, declared and argued tight | **Yes** — `ops::deconvolve`. The PSF is a caller-supplied separable non-negative normalised kernel (`PointSpread`), which is rule 1 of `README.md` applied properly |
| Iterative deconvolution with a convergence criterion rather than a count | **4** | **No** — but `crate::iterate` exists for it, and `iterate.rs`'s header cites `ops::deconvolve` by name as the case that motivated it. The gain is real: today the reach grows with the iteration count, and as a substage loop it would not |
| Inverse / regularised-inverse / Wiener / Tikhonov deconvolution | **3**, needing **G3** and **G4** | **No.** `ops::deconvolve`'s header specifies precisely what building it would take — a third-axis complex transform pass, somewhere to store a spectrum, an `n log n` cost term, and `Reach::all` — and states that none of it exists. This is the single best-documented category-5 entry in the crate |
| Total-variation-regularised deconvolution | **4** | **No** |
| Blind deconvolution / PSF estimation | **4**, plus a second running estimate | **No.** `Operand::Running` is singular by construction — "an op declaring two would need two ping-pongs, which is a different shape and is not this one" — so blind deconvolution needs a framework change, not just an op |
| Background estimation by morphological opening; top-hat correction | **1** — reach `2 x element reach` | **Yes** — `ops::background`, built as a `Chain::parallel` diamond of `[identity, open]` joined by `DifferenceCombine`. It writes no new kernel; it is pure composition, which is the pattern §2's missing arithmetic combines would unlock elsewhere |
| Rolling-ball background subtraction | **1** | **No.** ImageJ's `Subtract Background` and skimage's `restoration.rolling_ball` are the same algorithm and it is very widely used. It is a grey opening by a ball-shaped element in an intensity-augmented space; `ops::element::ElementShape::Ellipsoid` is most of the geometry already |
| Divisive shading / illumination correction against a reference or an estimate | **1** | **Partly.** `ops::normalise::Removal::Divide { floor }` divides by a *locally estimated* level, which is the estimate-from-the-image case. ~~**Correcting against a supplied reference volume is a second input at the same offset — G2 does not block it (`SourceInput` handles same-extent second images), but no op does it**~~ — *corrected twice.* **First:** the index adjudicated this against family D and D was right — `SourceInput` handled second *images of the same plan*, and a supplied array could not be an image at all, so what blocked it was **G5** and this sentence is the one factual error the four documents contain. **Second: G5 has since closed**, so the end state this row asserts is now true, by a route it did not name — `ArrayEnvironment::with_inputs`, and the reference addressed as `ImageId::supplied(i)`. Two residues, and both are G2's rather than this row's: the reference must be in **image 0's coordinate space** (a reference at a different binning is refused by name), and it must be read at the block's own fetch region. Still no op does it |
| Polynomial / spline surface fit for shading | **3** — the fit is a global least squares | **No** |
| Bias-field correction (the N4/N3 family) | **4** | **No** |
| Noise estimation (σ estimate, `estimate_sigma`) | **5 (G1)** — a scalar out of a volume | **No** |
| Noise generation (Gaussian, Poisson/shot, salt-and-pepper) for testing | **1** | **No** as ops, though `crate::synthetic` generates test volumes. All of ITK, skimage and this crate's peers ship these; they are cheap and they make restoration testable |
| Hot / stuck pixel detection and replacement | **1** | **No** |
| Per-plane intensity equalisation along an axis (decay, flicker, plane-to-plane drift) | **2** on the sweep axis; the per-plane statistic is **2†** and the rescale that reads it back is **2†** | **No.** Common in stack processing. *Corrected:* this row read `5 (G1)` for the statistic, on the reasoning that "one scalar per plane is a rank-1 output from a rank-3 input". It is a `[Z, 1, 1]` output from a `[Z, Y, X]` input — two axes collapsed, the third still cuttable — and that is a legal `Array3` and a phase that runs today through the `†` escape. The rescale is the broadcast, same route. Neither is blocked; neither can be declared |

---

## 10. Operations that are not expressible today, and what each would need

The consolidated list. Each entry names the gap and the smallest framework
change that would close it.

### G1 — the geometry cannot declare a collapsed or a broadcast axis

*(Renamed in the index's §8.1; identifier unchanged. The old name, "a
rank-reducing phase", is kept visible here because it is what this document
argued and what three other documents cite.)*

**Blocks the declaration for:** whole-volume histogram; global
min/max/mean/percentile; contrast stretching; global histogram equalisation;
histogram matching; global threshold statistics; noise-σ estimation; per-plane
statistics for axis-wise equalisation; any projection along an axis; **and the
map half of every one of the two-phase entries above**, which is a broadcast.

> **Corrected — measured.** *What this document said:* "`BlockOp::output_shape`
> is `[usize; 3] -> [usize; 3]` and `Voxels` is eleven `Array3` variants.
> `Reach` is per-axis halo widths on a same-rank output — there is no way to say
> *this axis collapses*." *What was measured* (`tests/collapsing_phase.rs`):
> the second sentence is right and the first is the wrong reason. `[X, Y, 1]` is
> a legal `Array3` — the 3-D floor is a deliberate decision, not a cap, and
> lower rank is modelled by degenerate axes precisely so that no element type
> has to be added — so `output_shape` returns it and a collapsing phase plans,
> runs and is decomposition-invariant today. **A projection is not blocked.** It
> is expressible at family C's `2†`, through the `†` escape, by declaring
> `Reach::none()` in `Space::source_index()` and stating the fetch per block —
> that is, by not declaring the dependency at all. The truthful declaration,
> `AxisReach::All` in `Frame::Source`, was refused unconditionally and no halo
> rescued it — **until the geometry change landed**, since when it is granted on
> an axis the op consumes and is the route to use; see the update at the end of
> this block.
>
> Three consequences for the list above. **First,** the entries divide: those
> whose output is a volume with a degenerate axis (per-plane statistics,
> projections, a scalar as `[1, 1, 1]`) are buildable now; those whose output is
> in a coordinate space that is not the input's at all (a histogram's *bin*
> axis) are a different problem, and the *cost* of a decomposed histogram is
> G7's, which the index adjudicated separately. **Second,** the broadcast half
> joins the list, and its declaration is provably absent rather than merely
> unwritten: `InputMap::Affine` makes the source extent the block extent times a
> rational, so pinning an axis at 1 needs a factor tending to zero and `up = 0`
> gives extent 0. **Third,** nothing checks a stated fetch against the
> dependency it stands in for, so every one of these built through the escape is
> correct only by its author's care — the index's **G14**, and the only
> correctness gap in the register.
>
> **Landed, and it split this list in two.** The geometry change shipped. The
> **collapse** can now be declared truthfully — `AxisReach::All` in
> `Space::source_voxels()` plus `with_sources` — and the declaration is checked
> against the fetch, so per-plane statistics, projections and every scalar
> reduction consumed as an image have a way to say what they read. The
> **broadcast** did not: a pinned axis still has no declaration, so every map
> half above is escape-only and unchecked. That asymmetry is why G1's name has
> been narrowed to the broadcast, and it is the state of the tree now.
>
> **And what the check does not do**, because it is easy to over-read: it holds
> an op to what it *said*, and cannot make an op speak. A phase that declares
> nothing and fetches one plane of an axis it means to consume still plans,
> still runs, and is still wrong at every position. That residue is G14's, and
> it is why "declare it" is now advice worth giving rather than a formality.

**What it would need — two candidate routes, and they are not equivalent:**

* **The cheap one, available today.** A side output of lower rank
  (`Output::shape` is a `Vec<usize>`, `side_region` maps a block's slice into
  it, and the executor already checks that the regions tile the declared output
  exactly). This works *now* for anything terminal. Its limit is that side
  outputs are terminal by construction — "nothing reads them back inside the
  chain, because there is no fan-in" — so a histogram computed this way cannot
  feed the map that uses it **within one workflow**. Two workflows, with the
  scalar passed between them by the caller, is the honest shape and it is
  available immediately.
* **The real one.** *Corrected:* this read "a phase whose output rank differs
  from its input rank, and a `Reach` variant that can say `Collapse` on an axis
  — touching `output_shape`, `Voxels`, the tiling check and the budget
  arithmetic." It touches none of those: the rank never changes, `[X, Y, 1]` is
  already legal and already tiles. What is needed is a **per-axis extent rule on
  the input map** — `Whole` for the collapse, `Fixed(1)` for the broadcast,
  `Scaled` for what `InputMap::Affine` already does — from which the per-block
  fetch is derivable without a table. That is the larger change and it is what a
  fused *reduce-then-map* workflow needs, both halves of it. The index's §9
  records it as a proposal.

**Note the associativity requirement**, which `ops::tabulate` already
demonstrates and any global reduction must meet: every column it accumulates
combines by an operation that is associative and commutative **in the type it
is performed in** — sums in fixed point, `min`/`max` as read — so that a region
cut across blocks merges exactly rather than approximately. A histogram is a
sum of counts and passes trivially; a mean is a `(sum, count)` pair; a
percentile is not associative and needs the histogram first. That ordering is
the design, not an implementation detail.

### G2 — inputs at different offsets or resolutions

**Blocks:** correlation between two differently-shaped planes; image pyramids
used coarse-to-fine; Laplacian-pyramid reconstruction; any op reading a
downsampled level beside a full-resolution one.

**Why:** a `Decomposition` reads one image and maps a block to one region of
it. `Chain::Source` and `SourceInput` already add further images — so *more
than one input* is solved — but each is read at the block's own extent, in the
block's own space.

**What it would need:** a `SourceInput` carrying a coordinate mapping (a
rational scale and offset per axis, in the spirit of `ops::resample`'s exact
`up/down` and its centred half-voxel convention) rather than only a `Reach`.
`ops::lattice` is the existing proof that a cross-grid phase can be built and
checked — both of its ops are cross-grid already — so the machinery is not
foreign to the crate; it is the *general* form that is missing.

### G3 — no complex element variant

**Blocks:** a spectrum as a plan image; therefore every frequency-domain
filter, band-pass and stripe-removal op, Wiener and regularised-inverse
deconvolution, phase correlation as a phase, and transform-based convolution.

**Why:** `Dtype` has twelve real variants, `Voxels` eleven, and `Complex<f64>`
appears in exactly one file. `dtype.rs` carries a test pinning
`from_numpy_name("complex64") == None`, so this is a decision rather than an
oversight.

**What it would need:** `Dtype::Complex64`/`Complex128` with a `size_of`,
matching `Voxels` variants, and a decision about the half-spectrum layout — a
real-input transform produces `[rows, cols/2 + 1]`, which is *not* the input
shape, so a spectrum image also touches G1's territory. This is the largest of
the four and it should be taken only when a concrete op needs it; a band-pass
filter is the natural forcing case.

### G4 — the cost model cannot price `log n`

**Blocks:** the planner choosing correctly between a direct convolution and a
transform-based one; pricing any resident transform at all.

**Why:** `cost_per_voxel()` takes no argument, so an op's declared unit cost
cannot grow with the size of what it transforms. `cost_per_voxel_in(block)`
does take the block — and `ops::sliding` uses it for a genuinely block-dependent
term — but its contract requires agreement with `cost_per_voxel()` in the limit
where the block spans the volume, which is exactly where a resident transform
operates.

**What it would need:** the smallest honest change is a
`cost_per_voxel_in_volume(volume)` alongside the existing pair, or a
`cost_per_voxel` that takes the volume. Both are additive with a default that
preserves every current op's declaration. *Unverified:* whether the calibration
in `crate::statistics`, which measures nanoseconds per unit of declared cost
and keeps one denominator per op, tolerates a second denominator — that needs a
look at `Snapshot::calibrate` before anyone commits to a signature.

> **What changed underneath this entry, and it is not the gap.** The partition
> search's **objective** has changed, and it changes what the misprice does. This
> was checked rather than assumed, because the natural reading of "the search now
> chooses a block size per phase" is that it did not before, and that is wrong.
>
> *What was always true:* the search has always swept block candidates per phase.
> *What was uniform was the answer.* The old objective was the phase's serial
> work, `cost_per_block × n_blocks`, which is
> `volume × redundancy × per-voxel`: `n_blocks` cancels, redundancy falls
> monotonically as the block grows, and the sweep therefore answered **"the
> largest candidate that fits"** in every phase of every chain. Under a monotone
> objective a wrong coefficient changes no decision, which is why G4 has been
> nearly inert since it was written.
>
> *What it is now:* `max(pool, channel)`, the phase's predicted wall clock, where
> the pool bound is `cost_per_block × ceil(n_blocks / workers)` and the channel
> bound is `read × read_cost + core × write`. Measured at **2.6×** on a mixed
> plan against a control with **identical reads and identical serial work** — the
> two plans differ only in how many tasks there are, which is exactly what the
> old objective could not see. At `workers == 1` the two are the same expression
> and the same bits, so nothing built before it moves.
>
> **So G4 got worse without changing.** A resident transform priced as linear is
> now priced wrong in the *pool* bound, which is the half that decides how finely
> to cut; and §8's row on choosing between a direct convolution and a
> transform-based one — "exactly what `cost_per_voxel_in` was built to let the
> planner make" — is now a choice the planner is really making, with one of the
> two alternatives priced wrong. The register's §11.2 carries the same account
> once for all four families.

### A fifth, smaller one — a second running estimate

**Blocks:** blind deconvolution, and any alternating-minimisation scheme.

**Why:** `iterate::Operand::Running` must appear exactly once — "an op
declaring two would need two ping-pongs, which is a different shape and is not
this one." The restriction is deliberate and documented.

**What it would need:** a second pair of alternating private buffers, and a
convergence rule over both. Worth recording, not worth building until something
asks.

---

## 11. Present, but narrower than the name suggests

Checked against the code rather than the module list.

| Module | What a reader assumes | What it actually is |
|---|---|---|
| `ops::smooth` | Smoothing filters, plural | **Gaussian only.** One separable, per-axis-sigma Gaussian and nothing else. No box filter, no median, no arbitrary kernel. Always produces `f64` whatever it read — a deliberate refusal to pick a rounding rule silently, at the cost of a `u16` image becoming four times its size |
| `ops::normalise` | Intensity rescaling, min-max normalisation, equalisation | **None of those.** Three *local* ops: subtract-or-divide by a locally estimated level; a local z-score (`(v − centre) / spread`); and a bounded local gain from two order statistics. Every one is windowed. There is no global normalisation of any kind |
| `ops::element` | An element *type* abstraction | **Structuring elements** — neighbourhood geometry (box, two ellipsoid conventions, asymmetric per-axis extents, decimation with two step-origin conventions) plus the order-statistic machinery (`Rank`, `Percentile`, `Total`, `select_nth`). Nothing to do with `Dtype` |
| `ops::local` | Local filters generally, perhaps CLAHE | **Coarse-lattice statistic plus interpolation.** Mean, deviation, rank, isodata and a `Custom` reducer, evaluated on a globally-anchored `SampleLattice` and interpolated back separably. No equalisation, no bilateral, no non-local means, no diffusion, no Kuwahara. Its reach is *lattice distance + window radius*, which is volume-dependent and can far exceed the window |
| `ops::sliding` | A family of sliding-window statistics | **One statistic.** `HistogramQuery` is a general trait; `RankQuery` is its only implementor, so today this is an *acceleration of rank filtering*, not a statistic family. `Bool`/`U8`/`U16`/`U32` only |
| `ops::rank` | Median filter | **Any order statistic**, over any structuring element, with a masked variant — broader than the name, for once |
| `ops::background` | Background estimation methods | **One method**: morphological opening, corrected by subtraction. No rolling ball, no surface fit, no divisive correction, no median background |
| `ops::deconvolve` | Deconvolution | **One algorithm**: spatial-domain iterative Richardson–Lucy-style, fixed iteration count, caller-supplied separable PSF. No Wiener, no inverse, no blind |
| `ops::fft` | A transform op | **Not a `BlockOp` at all**, and deliberately. Free functions and plain structs. 2-D planes only; no 3-D transform; no filtering; no phase correlation |
| `ops::ridge` | Ridge filtering | Accurate, and broader than it reads: it owns the Gaussian kernel builder, the boundary convention, the Hessian, the closed-form symmetric eigenvalue solver and the scale space that the rest of the family depends on |
| `ops::directional` | Oriented or directional filtering | **Not a filter at all** — a 12-subiteration template-driven binary thinning. See §12 |

---

## 12. Handed to another family

Named here in one sentence each, so a reader does not go looking.

* **`ops::directional`** — despite the name, this is Pálagyi–Kuba parallel
  binary thinning (12 subiterations, 14 deletion templates, reach 12 per axis).
  **Family B**, with `ops::skeleton` and `ops::morphology`.
* **The thresholding *decision*** — Otsu, isodata, triangle and friends as
  *binarisation*. **Family B.** Only the statistic that feeds the decision is
  mine, and §3 covers it.
* **Maxima/minima detection over a filter response** — blob detection ends in a
  local-maxima search over the multi-scale response; the response is mine, the
  search is **family B** (`ops::regional`, `ops::detect`).
* **Resampling, and the interpolation convention that goes with it** —
  `ops::resample`'s exact rational factors and centred half-voxel map are
  **family C**. Pyramids use them; the multi-scale *use* is discussed in §7.
* **Registration** — translation estimation from a correlation landscape, and
  everything built on it, is **family C**. `ops::fft`'s `SquaredDifference` is
  the primitive underneath it and is surveyed here as a transform.
* **Phase correlation's consumer** — same split: the operation is a transform
  (mine, §8), the alignment it computes is **family C**.
* **Corner detection as a registration seed** — **family C**; the cornerness
  response image is mine (§7).
* **Compositing, blending, channel merge/split, projections along an axis** —
  **family C**, except that a projection is also a G1 instance and §3 counts it
  as evidence for the same gap. *Corrected:* it is evidence for G1's missing
  **declaration** and not for anything being unbuildable — family C classified a
  projection `2†` and has been proved right (`tests/collapsing_phase.rs`). Read
  the G1 rows in §3 and §9 with §10's correction beside them.
* **Per-label reduction (`ops::tabulate`)** — **family B**, but its associative
  partial-merge design is the template any global reduction in this family
  should follow (§10, G1).

---

## 13. If only three things were built

Weighted by how many entries above they unblock, not by difficulty.

1. **Arithmetic `Combine`s and a general convolution op.** Add/subtract/
   multiply/divide/min/max as `Combine` sinks, on `ops::background`'s existing
   diamond pattern, plus convolution with an arbitrary (and separately, an
   arbitrary separable) kernel. Between them these unblock difference-of-
   Gaussians, unsharp mask, Laplacian, gradient and Sobel, highpass-by-
   subtraction, and the image-calculator operations — all category 1, no
   framework change, and every one of them is in all five sources.
2. **A rank-reducing side output, then a rank-reducing phase (G1).** Start with
   the histogram, because it is a sum of counts and therefore trivially
   associative, and because contrast stretching, equalisation, matching, global
   threshold statistics and noise estimation all queue behind it.

   *Corrected, and it moves this item.* The second half is not "a rank-reducing
   phase" — nothing needs a rank change — it is **a declaration for a collapsed
   or a pinned axis**, and the thing it buys is not the ability to build these
   but the ability to state them, to have a planner produce them, and to have
   the fetch checked. Both halves of a reduce-then-map workflow run today
   through the `†` escape; both are unchecked. What still queues behind a
   framework change is the *cost* of a decomposed histogram, and that is G7's
   barrier phase rather than this item.
3. **A one-axis transform pass declared as `AxisReach::All` on that axis.** The
   crate has a `Reach` variant with no user under `src/ops/`. VTK's imaging
   pipeline answers the same question the same way. It would also give the crate
   its first category-2 op, alongside the other obvious one — an integral image
   — and it is the step that makes a 3-D transform a composition rather than a
   rewrite.

   *One precision, added after measurement, and it is now the whole of the
   advice: **name the frame**.* Declare it in the **phase** frame, which is
   where a same-rank sweep belongs and where the variant has always worked. A
   same-rank whole-axis stencil in the phase frame is a shape that already
   works — the exact Euclidean distance transform in a sibling application crate
   is one, planned by an ordinary `PlanBuilder::pixels` phase on a lattice that
   never cuts the swept axis — so this item is genuinely available and not
   blocked on anything.

   The **source** frame is for a phase that consumes an axis it does not
   reproduce. It was refused unconditionally when this was measured; since the
   geometry change landed it is granted there and, uniquely among the three
   declarations, **checked against the fetch**. And against an axis of extent 1
   the phase-frame form is *vacuous* — `is_whole` requires `extent > 1`, so it
   is accepted without claiming anything. Two of the three declarations that
   plan are wrong in different ways, so "declare `All`" is not an instruction
   until the frame is named.

---

*Where this document says "no", it means "not found in `src/ops/` on a read of
the module in question". Where it says **unverified**, it means exactly that.
Nothing here was inferred from a module name.*
