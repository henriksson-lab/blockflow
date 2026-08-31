# Pixel classification: a Labkit- and ilastik-shaped workflow, in this crate

A plan, written before the work. The goal is to build the *workflow* those two
tools implement — a stack of image features, a random forest over it, trained
from sparse labels and applied to a whole volume — out of this crate's ops, with
its blocking, its planner and its decomposition invariance.

> **Provenance.** Everything in "What the two tools actually do" was read off
> their own documentation on **2026-08-31** and is cited to the page it came
> from. Everything in "What we will build" is a decision with its reason, not a
> measurement; where a number appears that has not been measured, it says so.

## What the two tools actually do

**Labkit** is random-forest pixel classification. The classifier is WEKA's; the
image plumbing is ImgLib2 and BigDataViewer, which is what lets it stream
terabyte images; GPU is optional and it can distribute over an HPC cluster. Its
feature stack is documented exactly, and this is the list we are matching. Given
sigmas `σ₁ … σₙ` and a 2-D-plane-wise or 3-D mode, per scale:

| feature | outputs per scale, 3-D |
|---|---|
| the original image | 1, once |
| Gaussian blur | 1 |
| difference of Gaussians, pairwise between the blurred versions | `C(n,2)` total |
| Gaussian gradient magnitude | 1 |
| Laplacian of Gaussian | 1 |
| Hessian eigenvalues | 3 |
| structure tensor eigenvalues, at integration scales `γ = 1, 3` | 6 |
| morphological min, max, mean, variance over `⌊1 + 2σᵢ⌋` | 4 |

**ilastik** is five workflows, and only the first is the same algorithm:

* **pixel classification** — a random forest, features grouped as
  colour/intensity, edge and texture, each at several sigmas. The page does not
  enumerate the filters; the implementation is vigra's and the family matches
  Labkit's closely;
* **carving** — a **seeded watershed** on a boundary map, flooded from
  inside/outside seeds. This crate already has that kernel;
* **object classification**, **tracking**, **counting** — different problems,
  out of scope here.

## Three findings that shaped the plan

**1. There is nothing worth loading in the trainable tools' own formats.**
Labkit writes a `.classifier`; ilastik writes a `.ilp`, an HDF5 project holding
the forest, the feature selection and the labels, applied with `--headless`.
Neither is documented as an interchange format, and — the reason that does not
matter — a forest over fixed-sigma filters is fitted to one microscope, one
stain, one magnification. Both tools are *built* on retraining in seconds from
brush strokes. So nobody publishes these, and a loader would buy compatibility
with a user's own previous run at the price of bit-reproducing someone else's
feature stack in perpetuity. **Declined.**

**2. The models that *are* published are a different thing entirely.** The 29
entries at `bioimage.io/#/models?partner=ilastik` are deep-learning U-Nets —
`NucleiSegmentationBoundaryModel`, `MitochondriaEMSegmentationBoundaryModel`,
`Neuron Segmentation in EM (Membrane Prediction)` and so on, mostly boundary
predictors. They feed ilastik's *Neural Network Classification* workflow, which
runs pre-trained CNNs by DOI through TikTorch, not the random-forest one. No
random forest appears among them. Consuming those is the `model_segment` path
this crate already has for stardist, cellpose and yolo, and it is **a separate
piece of work** — though worth noting that a boundary probability map is exactly
what `ops::scikitimage_watershed` wants as a cost volume, so the two meet.

**3. The Rust ecosystem does not have the forest we need.** Surveyed
2026-08-31: `linfa-trees` 0.8.1 fits **single trees only** and linfa has no
forest crate; `smartcore` 0.6.14 has `RandomForestClassifier`, is pure Rust and
is maintained, but **its fitted trees are private with no access to the node
structure**, so a model cannot be converted into a layout we control, and
prediction goes through its generic matrix trait.

## The decision: we own the forest

The asymmetry in this workload is the opposite of ordinary tabular ML, and it is
what makes a general-purpose crate the wrong tool:

* **training is tiny.** Brush strokes label a few thousand voxels. Training cost
  is irrelevant;
* **inference is enormous.** Every voxel, through every tree. A hundred trees at
  depth ten is on the order of a thousand dependent load-compare-branch steps
  *per voxel*, against the 10⁸–10⁹ voxels this crate exists for.

So the library question is almost entirely an inference question, and inference
is the part a tabular ML crate does worst — shaped for a matrix of samples,
opaque where it needs to be flat. We own a `Forest`: a flat node array with a
documented layout, evaluated batched over voxels. That is roughly a hundred lines
and it is the hot path.

We own the trainer too. Bagging, `mtry` feature subsampling and Gini over a few
thousand samples is not a large amount of code, it does not have to be
world-class to match a tool that retrains per dataset, and it removes the
dependency. **`smartcore` earns a place as a dev-dependency oracle instead**:
train both on the same labels, assert the predictions agree. That is the witness
shape `ops::scikitimage_watershed` already uses against skimage.

`Forest` is then the interchange point — trained here, or converted from
elsewhere later if that ever becomes worth it.

## What already exists

Nearly all of the feature stack:

| Labkit feature | this crate |
|---|---|
| Gaussian blur, difference of Gaussians | `ops::smooth` |
| gradient magnitude, Laplacian | `ops::convolve` |
| Hessian eigenvalues | `ops::ridge` |
| min, max over an element | `ops::rank` |
| mean, variance over an element | `ops::local` |
| structure tensor eigenvalues | **missing** |
| seeded watershed, for a carving-shaped workflow | `ops::scikitimage_watershed` |

## Two constraints that shape everything

**The predictor must be single-threaded per call.** The block executor already
parallelises, and `simulate::Machine::contention` exists because nested
parallelism is what makes forty workers behave like 2.41. A predictor spawning
its own threads would fight the machinery this crate spent its measurements on.

**The feature stack must never be materialised whole**, and the arithmetic says
why. At five sigmas in 3-D the stack is

```
1 original + 5 Gaussian + 10 DoG + 5 gradient + 5 Laplacian
  + 15 Hessian + 30 structure tensor + 20 morphological  =  91 images
```

At `1024³` and `f32` one image is **4 GiB** and the stack is **0.4 TiB**. It has
to be computed per block and fused, which is precisely what this crate is for and
is the strongest argument the workload belongs here at all. It also means a
**91-arm fan-in**, which is far past anything `Chain::Parallel` and the
peak-image accounting have been measured against — see the risks.

The predictor itself is voxelwise: **reach zero, one pass**, so it is
decomposition-invariant by construction and the existing invariance suites cover
it for free.

## Build order, and what to measure at each step

1. **`ops::structure_tensor`.** The one missing feature. Eigenvalues of the
   tensor at integration scales `γ = 1, 3`, 6 outputs per scale in 3-D and 4 in
   2-D. Witness it the way `ops::ridge` is witnessed.
2. **The feature-stack chain builder.** Sigmas and a 2-D/3-D flag in, a `Chain`
   out. **Measure first**: what a 91-arm `Chain::Parallel` does to
   `Decomposition::peak_image_bytes`, to the partition search's `O(n²)` priced
   runs, and to `working_set_bytes_per_block`. This is the step most likely to
   find a real defect, and it should be run through the arena and the `costs/`
   scenarios before any classifier exists.
3. **The predictor op**, over a `Forest`. **Measure `cost_per_voxel`** — ns per
   voxel at a few tree counts and depths — before fixing the design. If it lands
   where the arithmetic above suggests, this is the most expensive op in the
   crate by a wide margin and the planner's whole treatment of the chain follows
   from that number rather than from a declared constant.
4. **The trainer**, last, because it is the cheap end. Sparse labels in, a
   `Forest` out, with `smartcore` as the agreement oracle in tests.
5. **The workflow wrappers**: train and predict, assembled from the above, in the
   shape `ops::background::remove_background` uses — a function returning a
   `Chain`, so the planner sees the whole thing and can cut it where it likes.

## Steps 1 and 2, done — and what the measurement found

> Measured **2026-08-31**, on the machine this crate was developed on. The
> tables are reproducible: `cargo test --release --test feature_stack --
> --ignored --nocapture` and `cargo test --release --lib -- --ignored
> --nocapture ops::cost`.

### Step 1 needed three ops, not one

The table above says the structure tensor was the only missing feature. It was
wrong by two. `RidgeFilterOp` folds the Hessian's three eigenvalues through a
`Response` and emits one number; a classifier wants the three separately, and
`Response` is a closed enum on purpose. And there was no gradient magnitude
anywhere — it is a square root of a sum of squares of three linear filters, so
`ops::convolve`, which holds one linear filter, could not express it.

So step 1 shipped:

* `ops::structure_tensor::StructureTensorOp` — one eigenvalue per op;
* `ops::structure_tensor::GradientMagnitudeOp` — placed there rather than in
  `convolve` because it is the one non-trivial invariant of the structure tensor
  at `rho = 0`, which is the case `StructureTensor::new` refuses;
* `ops::ridge::HessianEigenvalueOp` — one eigenvalue, one scale, not a
  `ScaleSpace`.

Two things worth carrying forward. **The structure tensor's reach adds where
`ops::ridge`'s takes a maximum** — `radius(sigma) + 1 + radius(rho)`, because
the three stages apply in turn where ridge's scales are alternatives — and
`tests/pixel_classification_features.rs` runs the maximum-fold to show it gives
a different volume. And **its cost is asymmetric in its two scales by a factor
of six**: the derivative smoothing runs once, the integration smoothing runs
once per tensor component. Measured 6.59, 5.00 and 4.95 across three runs, with
the model fitting the three timing rows to 0.15%.

### Step 2 found the risk list was pointing at the wrong thing

| sigmas | arms | slots | reach | search |
|---|---|---|---|---|
| 1 | 17 | 1 | 13 | 0.2 ms |
| 3 | 52 | 1 | 49 | 0.4 ms |
| 5 | 91 | 1 | 193 | 0.5 ms |

**The 91-arm fan-in is one slot.** `Chain::Parallel` is a single slot however
many branches it has, so the partition search never sees the arms: it prices one
contiguous run in under a millisecond at every size. The document's stated risk
— "whether the partition search survives two orders of magnitude more" — does
not arise. The same fact is a constraint: the planner **cannot** put a phase
boundary inside the stack, so all 91 channels are computed and discarded within
one block. That is the fusion the memory arithmetic needs, and it is not a
choice.

**The halo is what binds, and it grows fourfold with the last sigma.** 13, 25,
49, 97, 193 — each sigma doubles the scale and the structure tensor at
`gamma = 3` multiplies by four on top of it, so the widest arm is that one and
not the morphological box a reader would guess. Consequences, measured:

* cut on every axis with 128-voxel blocks, the read amplification is 1.7, 2.7,
  5.5, **15.9** at one to four sigmas;
* at five sigmas the planner **declines to block at all** — one block, the whole
  volume — which is the right answer and is only available while the volume
  fits;
* under a 256 MB budget it plans at three sigmas with 67x amplification and
  **refuses outright at four and five**. That is not a tuning failure. A halo of
  `r` puts a floor of `(2r)^3` bytes-per-voxel under a block's working set that
  no block size escapes — shrinking the block leaves the footprint almost
  unchanged and makes the amplification worse — and at `r = 193` that floor is
  460 MB in `f64` before any concurrency.

`tests/feature_stack.rs` pins all of this.

### What it changes about step 5

The wrappers cannot be one fused phase at Labkit's default sigmas. The fix is
already in the crate and is the thing to build step 5 around: **materialise the
wide-sigma channels in their own phases and have the predictor read them as
`Chain::Source` leaves**, which are reach 0. Then

* each feature phase carries the halo of the few arms in it, rather than every
  arm carrying the widest;
* the predictor phase has reach **zero**, so its blocks can be as small as the
  budget likes, and its 91 arms are 91 *block-sized* buffers with no halo at all.

That last point is the other open number, and it is deliberately not measured
yet. `budget.rs` charges `working_set_bytes_per_block` as `resident_voxels x
bytes_per_voxel x 2.0` — one buffer in, one out — and a fan-in escapes the arity
term only when its combine declares a `Combine::fold_carrier`, which makes it
hold three buffers whatever the arm count. **A forest predictor cannot declare
one**: it needs all 91 channels at a voxel simultaneously to walk a tree, which
is the definition of not being a left fold over pairs. So the predictor's fan-in
holds one buffer per arm, and every residency figure measured in step 2 — taken
with a folding placeholder combine — is a floor for the real chain rather than
an estimate of it. The allocator measurement belongs with the predictor, in
step 3, and `tests/working_set_residency.rs` is the harness for it.

### Steps 3, 4 and 5, done — and the cost expectation was inverted, then restored

**The predictor is what the plan said it was.** Measured at 91 channels over
`64x64x32`, one thread:

```text
  trees   depth   nodes  mean path visits/voxel     ns/voxel     ns/visit
     10       8     820       6.87         68.7        447.3         6.51
    100      20   30594      10.35       1035.1       6527.5         6.31
    200      20   57582      10.22       2044.0      13196.8         6.46
```

`ns/visit` is flat across a twenty-fold range of node counts, so the cost model
is `trees x mean path x 6.56` — not the node count, and not the maximum depth,
which would misprice an unbalanced forest by the ratio of its deepest path to its
average. At Labkit's 100 trees the predictor is **6,528 ns per voxel**.

**Then the stack turned out to cost 4.1 million.** Priced through the crate's own
`cost_per_voxel`, the 91 arms declared 4,117,711 against the predictor's 6,298 —
the predictor was 0.2% of the chain, not the dominant term this document
predicted. One arm was 83% of it: `morphology/16/min`, a rank filter over a
`67^3` box, which is 300,763 voxels read per voxel written. And the declaration
was **honest** — measured at 1,005,839 ns per voxel, 0.86 ns per declared unit,
right in line with every other arm's 0.28 to 6.4.

**A box is separable, and that recovered a factor of 240.**

* **min and max** are three one-dimensional passes, exactly — `min over A x B =
  min over A of (min over B)`, which holds for any total order and survives
  truncation at the volume boundary, because a clipped box is still a product of
  clipped intervals. `3(2r+1)` reads against `(2r+1)^3`: **1,496x** at `r = 33`.
  Byte-identical to the direct box, checked.
* **the mean** the same way, exact in the mathematics and differing only in
  summation order.
* **the deviation**, which is not separable, through `var = E[x^2] - E[x]^2` —
  two separable means and a fan-in. This one is a real trade and is the default
  with an escape: the identity is the textbook cancellation case, losing digits
  as `(mean / sd)^2`, so a pedestal of `3e4` with a modulation of `1e-3` breaks
  it badly. Both halves are asserted — agreement to `1e-12` on ordinary data and
  visible failure on a pedestal — and `FeatureStack::with_exact_deviation`
  restores the two-pass window for a caller in that corner.

After it, the stack declares **17,182** and no arm exceeds 4.7%. The predictor is
**27% of the chain** at 100 trees, which is the balance this document expected,
reached by fixing the filters rather than by assuming it.

**What shipped for steps 4 and 5.** `Forest::train` — bagging, `mtry`, Gini,
deterministic from its seed; `ops::classify::train_workflow` and
`predict_workflow`, in the shape `ops::background::remove_background` uses. The
end-to-end test trains on two brush strokes of 576 voxels and recovers 95% of a
65,536-voxel volume whose two classes have the *same mean intensity*, so nothing
but the neighbourhood features can separate them.

### Still open

* **The sparse training path.** `gather_samples` computes the whole stack over
  the whole volume it is handed and then keeps the labelled rows. That is right
  for a crop, which is what an annotator draws on, and wrong for a whole volume.
  The shape it wants is a sampler as a phase **side output**: labels are a
  `Chain::source` arm away, `BlockOp::apply_side` already emits per-block
  fragments and `ops::rows` already collects them, so a block holding no labelled
  voxel could be skipped — which is nearly every block.
* **The predictor's residency**, still unmeasured through the allocator. It
  cannot declare a `fold_carrier`, so its fan-in holds one buffer per arm where
  `budget.rs` charges for two in total. `tests/working_set_residency.rs` is the
  harness.
* ~~**`smartcore` as an agreement oracle.**~~ **Done**, as a dev-dependency, and
  the comparison is on held-out accuracy rather than on predictions. Voxel-for-
  voxel agreement was the wrong thing to ask for and it is worth saying why: two
  independently implemented forests bag from different generators, draw `mtry`
  differently — with replacement here, without there — and break Gini ties
  differently, and any one of those changes which rows land in which tree. What
  is comparable is what a forest is *for*, and the failures worth catching (a
  threshold on the wrong side, an ignored `mtry`, an unrecovered informative
  column) all show as a systematic gap in accuracy rather than as rounding. The
  fixture is deliberately non-separable, because on a separable one both score
  1.000 and the test would pass for anything.
* **`f32` for the feature stack**, still unmeasured, and the reason to answer it
  has changed shape. It was proposed as a memory saving, and as a memory saving
  it is now the *more* attractive of the two halves: the 91 channels are 91 live
  buffers in the predictor's fan-in — the residency item above — and halving each
  is halving that. What it is not is a speed question. The stack declares 17,182
  against the predictor's 6,298, so it is about **2.7 times** the predictor's
  cost and the two are within an order of magnitude of each other; narrowing the
  channels does not change the arithmetic in either op, and the precision
  question — whether a forest's splits are stable at `f32` — is a separate
  measurement that has to be taken before the memory saving can be banked.

## Risks, named

* ~~**The 91-arm fan-in** is the big unknown.~~ **Settled, and it was two of
  three.** The partition search is untroubled — a `Parallel` node is one slot —
  and the reach fold is correct. The residency accounting is *not* settled and
  is now the sharper risk: see "What it changes about step 5" above.
* ~~**The predictor's cost may dominate everything.**~~ **Measured: it is 27% of
  the chain at 100 trees**, and only after the morphological family was made
  separable — before that it was 0.2%, and the risk that mattered was one nobody
  had written down. The planner's treatment of this chain follows from the two
  terms being within a factor of three of each other rather than from either
  swamping the other.
* **`f32` for the feature stack** halves the memory against `f64` and is what
  both reference tools use; whether the forest's splits are stable at that
  precision is unmeasured, and the answer should be a measurement, not a
  default.
* ~~**The 2-D-plane-wise mode**~~ **Done, and it did fall out of the
  parameterisation**: `ops::features::Geometry::PlaneWise` sets the scale to zero
  on the normal and the element to one voxel thick, and nothing else changes. One
  thing had to be relaxed to allow it — `HessianEigenvalueOp` validates against
  `gaussian_weights`, which documents a zero scale as "this axis is not blurred",
  rather than against `ScaleSpace`, which refuses one for a reason that belongs
  to a list of alternatives and not to a single scale.

## Not doing

* `.classifier` and `.ilp` loaders — see finding 1.
* A bioimage.io RDF adapter — see finding 2. Worth doing, separately, and it is
  the `model_segment` path rather than this one.
* GPU. Labkit's is optional and this crate has no GPU placement at all
  (`docs/design/planner-gaps.md` scopes it out and lists what is missing).
