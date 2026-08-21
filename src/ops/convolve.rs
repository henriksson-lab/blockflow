// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of a discrete
// convolution, not adapted from any implementation of one.
//
// A linear filter with a **caller-supplied** kernel.
//
// `ops::smooth` is the crate's other linear filter and it is a Gaussian: the
// weights are derived from a sigma, the kernel is separable by construction, and
// nothing about it can be asked for. This one takes the weights. That is the
// difference and it is the whole reason the file exists — a difference of
// Gaussians, a Laplacian, a gradient, a Sobel, a binomial, an unsharp mask and
// every hand-rolled stencil are one kernel each and were unreachable while the
// only convolution in the crate was one particular kernel.
//
// Correlation or convolution — this file implements **both, by name**
// ------------------------------------------------------------------
// The two differ by a reflection of the kernel, and the names libraries give
// them do not line up. OpenCV's `filter2D` computes a **correlation** and its
// own documentation says so — a caller who wants a convolution is told to flip
// the kernel and move the anchor. `scipy.ndimage` declines to choose and ships
// `convolve` and `correlate` as two functions. `numpy.convolve` convolves.
// Several others are named for one and compute the other. A crate that picked
// one silently would be wrong for half its readers in a way that is invisible on
// the symmetric kernels everybody tests with, so [`Sense`] is a **parameter with
// no default**:
//
// ```text
// Sense::Correlate:  out[v] = sum_k  w_k * in[v + o_k]
// Sense::Convolve:   out[v] = sum_k  w_k * in[v - o_k]
// ```
//
// The two agree for every kernel symmetric under negation and differ for every
// kernel that is not, which is why the fixtures in the acceptance suite are
// asymmetric and why there is a test asserting that a symmetric one *cannot*
// tell them apart. `Kernel::reflected` is the other route to the same place, and
// the identity `correlate(k) == convolve(k.reflected())` is checked rather than
// asserted in prose.
//
// **The accumulation order is the element's own and does not depend on the
// sense.** Both walk the offsets in the element's ascending lexicographic order,
// negating the *index* under `Convolve` rather than reordering the sum. That is
// deliberate twice over: floating-point addition does not associate, so a
// re-sorted sum would be a different number and the flip would no longer isolate
// the geometry; and the order is a property of the kernel rather than of the
// block, which is what makes the answer decomposition-invariant to the bit.
//
// The boundary is a parameter, and it is resolved at the buffer's edge
// -------------------------------------------------------------------
// [`Boundary`] is `ops::ridge`'s, shared with `ops::smooth`, and it is a
// constructor argument rather than a default: `super`'s header records that the
// separable convolution is "the one place where that rule is a parameter rather
// than a constant", and this is the second op with a neighbourhood wide enough
// for the invented samples to dominate the answer at a face.
//
// It is applied at the edge of the **buffer**, exactly as `ops::smooth` applies
// it, and `apply` therefore ignores its [`Anchor`]. The argument is `super`'s:
// at a real volume face the buffer's edge *is* the volume's face, because the
// fetch is clamped to the volume; in the interior the buffer's edge is a halo
// edge, the invented samples feed only halo voxels, and those are discarded — so
// a sufficient halo makes the two readings the same function, and an
// insufficient one is caught loudly by the halo guard rather than quietly by a
// wrong edge.
//
// A `ClippedStart` element is refused, and the reason is not `ops::sliding`'s
// ----------------------------------------------------------------------------
// `StructuringElement::offsets_at` exists because an element whose decimation
// counts from [`StepOrigin::ClippedStart`] re-phases at a low face of the
// volume: the offsets it reads there are a *different residue class* from the
// ones it reads in the interior. `ops::rank` honours that by asking per voxel;
// `ops::sliding` refuses it because a carried histogram has no decomposition
// into leavers and joiners when consecutive windows read different classes.
//
// This op refuses it too, and for a third reason that is worth stating because
// it is the one a general kernel introduces. A rank filter's per-offset fact is
// *membership*, and a re-phased set answers that for itself. A convolution's
// per-offset fact is **a number the caller supplied for that member** — the
// weights are a list parallel to `StructuringElement::offsets` — and the
// re-phased set contains offsets that are not members of that list at all. There
// is no weight to pair with them. The three ways out are all worse than a
// refusal: invent a weight (the caller did not supply it), drop the tap (a
// different filter at a face, silently), or renormalise (a policy nobody asked
// for). So `Kernel::new` refuses the element **at construction**, before a plan
// exists, and says which of the two step origins it wants. Every kernel this
// file can hold therefore reads one set of offsets everywhere, and the gather
// below may read `offsets` directly for the reason `ops::sliding` may.
//
// What is not here
// ----------------
// **Separability.** A caller-supplied *separable* kernel — three one-dimensional
// weight vectors, `n + m + p` taps instead of `n * m * p` — is a different op
// with a different cost declaration, and `ops::smooth` already owns the machine
// for it (`ridge::gaussian_smooth_into_with` takes `&[Vec<f64>; 3]` and is
// public). It is a real gap and it is named as one rather than half-built here.
//
// **A transform-based path** for a large kernel. `docs/ops-survey` records that
// choosing between a direct convolution and a transform-based one is what
// `cost_per_voxel_in` was built for and that the cost model cannot price
// `log n`; nothing here pretends otherwise.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::element::{ElementShape, StepOrigin, StructuringElement};
use super::ridge::Boundary;
use super::shapes_agree;

// --------------------------------------------------------------- the sense --

/// Which of the two linear filters a kernel names.
///
/// No `Default`, deliberately: see the module header for why a silent choice
/// here is wrong for half of any audience.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sense {
    /// `out[v] = sum_k w_k * in[v + o_k]` — the kernel is laid over the image
    /// the way it is written. OpenCV's `filter2D` and
    /// `scipy.ndimage.correlate`.
    Correlate,
    /// `out[v] = sum_k w_k * in[v - o_k]` — the kernel is reflected through the
    /// anchor first. `scipy.ndimage.convolve`, `numpy.convolve`, and what the
    /// word means in the mathematics.
    Convolve,
}

impl Sense {
    /// The index this sense reads for the offset `offset`.
    fn displace(self, offset: [isize; 3]) -> [isize; 3] {
        match self {
            Sense::Correlate => offset,
            Sense::Convolve => [-offset[0], -offset[1], -offset[2]],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Sense::Correlate => "correlate",
            Sense::Convolve => "convolve",
        }
    }
}

// -------------------------------------------------------------- the kernel --

/// A weight per member of a [`StructuringElement`]: the caller's filter.
///
/// **Two objects rather than one dense array**, because the element already
/// carries every question a window has to answer — where the anchor sits, how
/// wide each side is, whether the box is full or a shape inside it, whether the
/// taps are decimated — and re-deriving those from an array plus an anchor index
/// would be a second, disagreeing statement of all of it. `reach_spec` in
/// particular is then the element's own and there is no field that could say
/// otherwise, which is the rule `super`'s header holds every op to.
///
/// The weights are parallel to [`StructuringElement::offsets`], which is
/// ascending lexicographic and is part of that type's contract. That is what
/// makes the pairing well defined, and it is what a `ClippedStart` element
/// breaks — see [`Self::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    element: StructuringElement,
    weights: Vec<f64>,
}

impl Kernel {
    /// One weight per member of `element`, in the element's own offset order.
    ///
    /// Three refusals, each of something that would otherwise be a quiet wrong
    /// answer:
    ///
    /// * **a length that does not match the element**, which would pair weights
    ///   with the wrong offsets from the first mismatch onwards;
    /// * **a non-finite weight**, which makes every voxel of the result
    ///   non-finite regardless of the data and is far more likely to be a
    ///   caller's arithmetic slip than an intention;
    /// * **a [`StepOrigin::ClippedStart`] element**, for the reason the module
    ///   header gives: its offsets are not one list, and a weight per member
    ///   needs them to be. `StructuringElement::from_sides_stepped_at` with
    ///   [`StepOrigin::Anchor`] is the decimated element this can hold.
    pub fn new(element: StructuringElement, weights: Vec<f64>) -> Result<Self> {
        if element.origin() == StepOrigin::ClippedStart {
            return Err(Error::InvalidArgument(
                "a convolution kernel is one weight per member of its element, and an element \
                 whose step counts from the clipped start of the window has a different set of \
                 members at a low face of the volume — offsets this list has no weight for. \
                 `ops::rank` can honour such an element because membership is a fact the \
                 re-phased set answers for itself; a supplied weight is not. Build the element \
                 with `StepOrigin::Anchor`, whose offsets are one list everywhere."
                    .to_string(),
            ));
        }
        if weights.len() != element.len() {
            return Err(Error::InvalidArgument(format!(
                "a convolution kernel needs one weight per member: the element has {} and {} \
                 weight(s) were given. The weights are in the element's own offset order, \
                 ascending on axis 0, then 1, then 2.",
                element.len(),
                weights.len()
            )));
        }
        if let Some(position) = weights.iter().position(|weight| !weight.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "weight {position} is {}, which makes every voxel of the result non-finite \
                 whatever the data holds",
                weights[position]
            )));
        }
        Ok(Self { element, weights })
    }

    /// A dense rectangular kernel: every voxel of the box `lo + hi + 1` is a
    /// member, and `weights` is that box in row-major order.
    ///
    /// The convenience constructor for the ordinary case — a 3x3x3 Laplacian,
    /// a Sobel pair, a binomial — where the caller thinks in a small array
    /// rather than in offsets. [`Self::new`] is the general form.
    pub fn from_sides(lo: [usize; 3], hi: [usize; 3], weights: Vec<f64>) -> Result<Self> {
        Self::new(
            StructuringElement::from_sides(ElementShape::Box, lo, hi),
            weights,
        )
    }

    /// A dense rectangular kernel centred on its anchor, spanning
    /// `2 * radius + 1` on each axis.
    pub fn from_radius(radius: [usize; 3], weights: Vec<f64>) -> Result<Self> {
        Self::from_sides(radius, radius, weights)
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// How many taps one voxel costs.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Never true — [`StructuringElement`] refuses an empty element and the
    /// weights are the same length. Present because `len` is.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The sum of the weights, in the element's own order.
    ///
    /// What a constant field maps to under a unit input, and therefore the
    /// number a caller checks when they meant a kernel to preserve the mean
    /// (`1.0`) or to annihilate a constant (`0.0`, which every derivative
    /// stencil wants). Summed in the order the gather sums in, so it is the
    /// number the op would really produce and not a re-associated approximation
    /// of it.
    pub fn total(&self) -> f64 {
        let mut total = 0.0;
        for &weight in &self.weights {
            total += weight;
        }
        total
    }

    /// The same weights on the negated offsets: the kernel reflected through its
    /// anchor.
    ///
    /// `correlate(k)` and `convolve(k.reflected())` are the same filter, which
    /// is the identity that makes [`Sense`] a naming convention rather than two
    /// algorithms. It is also the sharpest **negative control** available to a
    /// test: the same program with the kernel flipped is plausible output that
    /// has moved.
    ///
    /// The rebuilt element is a bare offset list ([`StructuringElement::from_offsets`]),
    /// so it is sorted afresh and the weights are permuted to follow — the
    /// pairing is by offset and not by position, which is the only way to keep
    /// it through a sort.
    pub fn reflected(&self) -> Result<Kernel> {
        let negated: Vec<[isize; 3]> = self
            .element
            .offsets()
            .iter()
            .map(|offset| [-offset[0], -offset[1], -offset[2]])
            .collect();
        let element = StructuringElement::from_offsets(negated.iter().copied())?;
        let weights = element
            .offsets()
            .iter()
            .map(|offset| {
                let which = negated
                    .iter()
                    .position(|candidate| candidate == offset)
                    .expect("the element was built from exactly these offsets");
                self.weights[which]
            })
            .collect();
        Kernel::new(element, weights)
    }

    /// What this kernel reads, per side and per axis, in the given sense.
    ///
    /// **The two sides swap under [`Sense::Convolve`]**, and that is not a
    /// detail: a kernel with `lo = 1, hi = 4` correlated reads one voxel below
    /// the anchor and four above, and convolved reads four below and one above.
    /// Declaring the correlation's sides for a convolution would fetch three
    /// planes per block that nothing depends on and be three planes short on the
    /// other face — a well-formed wrong volume at every seam.
    pub fn reach_spec(&self, sense: Sense) -> Reach {
        let sides = |axis: usize| {
            let (lo, hi) = self.element.sides(axis);
            match sense {
                Sense::Correlate => (lo, hi),
                Sense::Convolve => (hi, lo),
            }
        };
        Reach::asymmetric([sides(0), sides(1), sides(2)])
    }

    /// The wider side, per axis: the symmetric bound, unchanged by the sense
    /// because a swap does not change a maximum.
    pub fn reach(&self, axis: usize) -> usize {
        self.element.reach(axis)
    }
}

// -------------------------------------------------------------- the gather --

/// `out[v] = sum_k w_k * in[v +- o_k]`, with the edge resolved by `boundary`.
///
/// **`input` is read as the whole array it is**, and the boundary convention is
/// applied at its edge; see the module header for why that is the right reading
/// inside a block as well as outside one, and why this function takes no
/// [`Anchor`].
///
/// The accumulator is `f64` whatever the input holds, and `out` is `f64`: a
/// weighted sum of integers is not an integer, and rounding one back is a
/// decision that belongs to the caller and to [`super::NarrowOp`] rather than to
/// the filter. `ops::smooth` says the same thing about the same question.
///
/// `out` may not alias `input`.
pub fn convolve_into<T>(
    input: ArrayView3<'_, T>,
    kernel: &Kernel,
    sense: Sense,
    boundary: Boundary,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "convolve_into")?;
    let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];
    if extent.iter().any(|&length| length == 0) {
        return Ok(());
    }
    let plan = TapPlan::build(kernel, sense, boundary, extent);
    let weights = &kernel.weights;
    let taps = weights.len();

    // The two outer axes' resolved indices, hoisted out of the loops they do not
    // depend on: a tap's index on axis 0 is a function of `i` alone, so it is
    // computed once per plane rather than once per voxel.
    let mut on_a = vec![0usize; taps];
    let mut on_b = vec![0usize; taps];

    // The contiguous path and the general one, on `VoxelwiseMapOp::apply`'s
    // precedent: the buffers are whole `Array3`s in practice, and an
    // `ArrayView` need not be in standard layout, so a wrong answer for a
    // strided view would be a silent one. **Both sum in the same order** — the
    // element's own — so they agree bit for bit and the choice between them is
    // invisible in the answer.
    let contiguous = input.is_standard_layout() && out.is_standard_layout();
    if contiguous {
        let src = input.as_slice().expect("standard layout is contiguous");
        let dst = out.as_slice_mut().expect("standard layout is contiguous");
        let stride = [extent[1] * extent[2], extent[2]];
        let mut written = 0usize;
        for i in 0..extent[0] {
            for tap in 0..taps {
                on_a[tap] = plan.at(0, tap, i) * stride[0];
            }
            for j in 0..extent[1] {
                for tap in 0..taps {
                    on_b[tap] = on_a[tap] + plan.at(1, tap, j) * stride[1];
                }
                for k in 0..extent[2] {
                    let mut total = 0.0f64;
                    for tap in 0..taps {
                        total += weights[tap] * src[on_b[tap] + plan.at(2, tap, k)].into();
                    }
                    dst[written] = total;
                    written += 1;
                }
            }
        }
        return Ok(());
    }
    for i in 0..extent[0] {
        for tap in 0..taps {
            on_a[tap] = plan.at(0, tap, i);
        }
        for j in 0..extent[1] {
            for tap in 0..taps {
                on_b[tap] = plan.at(1, tap, j);
            }
            for k in 0..extent[2] {
                let mut total = 0.0f64;
                for tap in 0..taps {
                    total +=
                        weights[tap] * input[[on_a[tap], on_b[tap], plan.at(2, tap, k)]].into();
                }
                out[[i, j, k]] = total;
            }
        }
    }
    Ok(())
}

/// Where every tap lands, per axis, for every position on that axis — the
/// boundary convention evaluated once instead of once per voxel per tap.
///
/// **This is a rearrangement and not an approximation.** The index a tap reads
/// is `boundary.index(position + delta, extent)`, which is a function of one
/// axis's position and that tap's displacement on that axis and of nothing else;
/// so it can be tabulated per axis, and the table is `extent` entries per
/// *distinct* displacement rather than per tap. A 3x3x3 kernel has 27 taps and
/// three distinct displacements per axis.
///
/// The tap order is untouched, which is the load-bearing part: the sum is still
/// taken over the element's own ascending offsets, so this changes the
/// arithmetic's speed and not its last bit.
struct TapPlan {
    /// Per axis, `resolved[axis][slot * extent[axis] + position]`.
    resolved: [Vec<usize>; 3],
    /// Per axis, the extent it was tabulated over.
    extent: [usize; 3],
    /// Per tap, which slot of each axis's table it reads.
    slot: Vec<[usize; 3]>,
}

impl TapPlan {
    fn build(kernel: &Kernel, sense: Sense, boundary: Boundary, extent: [usize; 3]) -> Self {
        let displaced: Vec<[isize; 3]> = kernel
            .element
            .offsets()
            .iter()
            .map(|offset| sense.displace(*offset))
            .collect();
        let mut resolved = [Vec::new(), Vec::new(), Vec::new()];
        let mut distinct: [Vec<isize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..3 {
            let mut deltas: Vec<isize> = displaced.iter().map(|tap| tap[axis]).collect();
            deltas.sort_unstable();
            deltas.dedup();
            let mut table = Vec::with_capacity(deltas.len() * extent[axis]);
            for &delta in &deltas {
                for position in 0..extent[axis] {
                    table.push(boundary.index(position as isize + delta, extent[axis]));
                }
            }
            resolved[axis] = table;
            distinct[axis] = deltas;
        }
        let slot = displaced
            .iter()
            .map(|tap| {
                let mut slot = [0usize; 3];
                for axis in 0..3 {
                    slot[axis] = distinct[axis]
                        .binary_search(&tap[axis])
                        .expect("every displacement is in its own axis's list");
                }
                slot
            })
            .collect();
        Self {
            resolved,
            extent,
            slot,
        }
    }

    #[inline]
    fn at(&self, axis: usize, tap: usize, position: usize) -> usize {
        self.resolved[axis][self.slot[tap][axis] * self.extent[axis] + position]
    }
}

// ------------------------------------------------------------------ the op --

/// A linear filter with the caller's kernel, the caller's sense and the
/// caller's boundary convention.
///
/// Every one of the three is a constructor argument and none has a default,
/// which is `README.md`'s first rule applied to the one op in this crate where
/// all three are genuinely open questions.
pub struct ConvolveOp {
    name: &'static str,
    kernel: Kernel,
    sense: Sense,
    boundary: Boundary,
    cost: f64,
}

impl ConvolveOp {
    pub fn new(name: &'static str, kernel: Kernel, sense: Sense, boundary: Boundary) -> Self {
        Self {
            name,
            kernel,
            sense,
            boundary,
            cost: CONVOLVE_COST_PER_TAP,
        }
    }

    /// Override the measured per-tap cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn sense(&self) -> Sense {
        self.sense
    }

    pub fn boundary(&self) -> Boundary {
        self.boundary
    }
}

impl BlockOp for ConvolveOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The kernel's wider side. Derived, with nothing to configure it; the exact
    /// statement is [`Self::reach_spec`] and it is what the plan uses.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.kernel.reach(axis)
    }

    /// The kernel's two sides per axis, **swapped under [`Sense::Convolve`]**.
    /// See [`Kernel::reach_spec`].
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.kernel.reach_spec(self.sense)
    }

    /// Every element type a buffer holds. What it accumulates into is its own
    /// business and is always `f64` — `ops::smooth`'s rule, for its reason.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `f64`, whatever it read. Rounding a weighted sum back to the input's type
    /// is a decision and it is not this op's to make.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    /// **`at` is ignored, and the module header is where that is argued.** The
    /// boundary convention is resolved at the buffer's edge, which is the
    /// volume's face where the block has one and a halo edge otherwise; and no
    /// kernel this op can hold re-phases, because [`Kernel::new`] refuses the
    /// one element that would.
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        macro_rules! direct {
            ($type:ty) => {
                convolve_into(
                    input.view::<$type>()?,
                    &self.kernel,
                    self.sense,
                    self.boundary,
                    out,
                )
            };
        }
        match input.dtype() {
            Dtype::U8 => direct!(u8),
            Dtype::U16 => direct!(u16),
            Dtype::U32 => direct!(u32),
            Dtype::I8 => direct!(i8),
            Dtype::I16 => direct!(i16),
            Dtype::I32 => direct!(i32),
            Dtype::F32 => direct!(f32),
            Dtype::F64 => direct!(f64),
            // No `Into<f64>`, so the same detour `ops::smooth` takes.
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                convolve_into(widened.view(), &self.kernel, self.sense, self.boundary, out)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts \
                 ({})",
                self.name,
                self.sense.label()
            ))),
        }
    }

    /// The weighted sum of the constant, **computed the way the gather computes
    /// it** rather than as `value * total`.
    ///
    /// Every tap of a constant field reads the same value, so the op's inner
    /// loop is exactly `total += weight * value` over the weights in the
    /// element's order — which is what this runs. It is therefore bit-identical
    /// to what a computed block would hold, not approximately equal to it, and
    /// that is the standard a declaration has to meet: a short-circuited block
    /// must produce exactly what a computed one would.
    ///
    /// `value * self.kernel.total()` would be the same number for most kernels
    /// and a different one in the last bit for some, which is why it is not what
    /// this does.
    ///
    /// `None` for a non-finite result, on `ops::background`'s argument: a `NaN`
    /// is not equal to itself, so a declaration of one could not be checked
    /// against the computed block by the standard this crate holds declarations
    /// to.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        let mut total = 0.0f64;
        for &weight in &self.kernel.weights {
            total += weight * value;
        }
        total.is_finite().then_some(total)
    }

    /// Per **tap**, times the number of taps.
    ///
    /// `ops::rank`'s rule: the cost of a windowed op is its window, so a 27-tap
    /// Laplacian and a 1331-tap blur are not one number. It is *not*
    /// `ops::smooth`'s rule, and the difference is the point — a Gaussian is
    /// separable and costs the **sum** of its kernel lengths, while a general
    /// kernel costs their product, because a general kernel is not separable and
    /// this op does not pretend to find out whether it happens to be.
    fn cost_per_voxel(&self) -> f64 {
        self.cost * self.kernel.len() as f64
    }
}

/// Measured, per **tap**; see [`cost_report`], which is runnable and prints the
/// table this was read off.
///
/// Denominated in [`super::MAP_COST`] like every other constant in `super`, and
/// taken beside it on the same machine in the same session so the ratio is a
/// ratio rather than two numbers from two places: `ops::voxelwise::cost_report`
/// put the threshold at **0.72 ns/voxel** there, against the 0.71 the stored
/// unit was read off, so this machine and the one the table came from agree to
/// 1.5%.
///
/// ```text
/// 96 x 64 x 64, --release, one thread, best of 20
/// kernel radius      taps   ns/voxel   ns/tap/voxel
/// [1, 1, 1]            27      50.00         1.8517
/// [2, 2, 2]           125     243.03         1.9442
/// [1, 3, 3]           147     285.35         1.9411
/// ```
///
/// **Flat across a 5.4x range of kernel sizes**, which is what licenses a
/// per-tap constant at all: had the figure fallen with the tap count, the cost
/// would have a fixed part per voxel and a per-tap part, and one number would be
/// wrong at both ends. `1.94 / 0.72` is 2.7, and the constant is that.
///
/// **The flatness survives contention and the absolute figure does not**, which
/// is worth recording because it says which half of the number to trust. The
/// same three rows re-measured with two other builds running on the machine read
/// `2.18 / 2.14 / 2.25` — 12% higher, and still flat to 5% across the range. The
/// per-tap *shape* is a property of the loop; the scale is a property of the
/// afternoon, which is exactly why `crate::statistics` calibrates the model from
/// real runs rather than trusting this.
///
/// **`ops::smooth`'s number is not comparable and must not be read as one.** A
/// Gaussian costs the *sum* of its kernel lengths, not their product, so a
/// radius-`[2, 2, 2]` blur is 15 taps where this is 125. That is the price of a
/// general kernel and it is the reason a caller who wants a Gaussian should ask
/// for the Gaussian.
pub const CONVOLVE_COST_PER_TAP: f64 = 2.7;

/// Time a convolution per tap, and print the table.
///
/// Its own report rather than a row in `ops::cost::measure`, on
/// `ops::voxelwise::cost_report`'s precedent and for its reason: adding cases to
/// that function reshuffles the numbers four other modules' constants were read
/// off, and this op's figure is per tap rather than per voxel so it would need
/// its own denominator there anyway.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let input: Voxels =
        ndarray::Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (i + j + k) as f64
        })
        .into();
    let mut report = String::from("case                          taps   ns/voxel   ns/tap/voxel\n");
    for radius in [[1usize, 1, 1], [2, 2, 2], [1, 3, 3]] {
        let size = (2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1);
        let weights: Vec<f64> = (0..size)
            .map(|which| (which as f64 + 1.0) / size as f64)
            .collect();
        let kernel = Kernel::from_radius(radius, weights).expect("a kernel");
        let taps = kernel.len();
        let op = ConvolveOp::new("convolve", kernel, Sense::Correlate, Boundary::Clamp);
        let mut out = Voxels::zeros(Dtype::F64, shape).expect("a buffer");
        let at = Anchor::whole(shape);
        let mut best = f64::INFINITY;
        for _ in 0..repetitions.max(1) {
            let started = Instant::now();
            op.apply(&input, &mut out, &at).expect("a block");
            let elapsed = started.elapsed().as_secs_f64() * 1e9 / voxels;
            best = best.min(elapsed);
        }
        report.push_str(&format!(
            "{:<28} {:>5} {:>10.2} {:>14.4}\n",
            format!("{radius:?}"),
            taps,
            best,
            best / taps as f64
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// A field with no symmetry of any kind, so that a flip, a shift or a wrong
    /// boundary all move it.
    fn field(shape: [usize; 3]) -> Array3<f64> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (i * 37 + j * 11 + k * 3) as f64 % 13.0 - 4.0 + (i * j) as f64 * 0.25
        })
    }

    /// A kernel that is **not** symmetric under negation: the only kind that can
    /// tell correlation from convolution.
    fn lopsided() -> Kernel {
        Kernel::new(
            StructuringElement::from_offsets([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).unwrap(),
            vec![1.0, -3.0, 0.5, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn a_kernel_needs_one_weight_per_member() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        assert_eq!(element.len(), 3);
        let refusal = Kernel::new(element, vec![1.0, 2.0]).expect_err("a refusal");
        let message = refusal.to_string();
        assert!(message.contains("the element has 3"), "{message}");
        assert!(message.contains("2 weight(s)"), "{message}");
    }

    #[test]
    fn a_non_finite_weight_is_refused_by_position() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let refusal = Kernel::new(element, vec![1.0, f64::NAN, 2.0]).expect_err("a refusal");
        assert!(refusal.to_string().contains("weight 1"), "{refusal}");
    }

    /// The decision this file makes about a re-phasing element, stated where a
    /// caller meets it: at construction, before a plan exists.
    #[test]
    fn a_clipped_start_element_is_refused_by_name() {
        let element = StructuringElement::from_sides_stepped(
            ElementShape::Box,
            [4, 0, 0],
            [4, 0, 0],
            [2, 1, 1],
        )
        .unwrap();
        assert_eq!(element.origin(), StepOrigin::ClippedStart);
        let weights = vec![1.0; element.len()];
        let refusal = Kernel::new(element, weights).expect_err("a refusal");
        let message = refusal.to_string();
        assert!(message.contains("clipped start"), "{message}");
        assert!(message.contains("no weight for"), "{message}");
        assert!(message.contains("StepOrigin::Anchor"), "{message}");

        // …and the same element with the other origin is accepted, so the
        // refusal is about the re-phasing and not about the step.
        let anchored = StructuringElement::from_sides_stepped_at(
            ElementShape::Box,
            [4, 0, 0],
            [4, 0, 0],
            [2, 1, 1],
            StepOrigin::Anchor,
        )
        .unwrap();
        let weights = vec![1.0; anchored.len()];
        Kernel::new(anchored, weights).expect("an anchored element is a kernel");
    }

    /// **The liveness partner for every sense test below.** An asymmetric kernel
    /// separates the two senses; without this the tests that assert they *agree*
    /// somewhere would be vacuous.
    #[test]
    fn correlation_and_convolution_differ_on_an_asymmetric_kernel() {
        let input = field([7, 6, 5]);
        let kernel = lopsided();
        let mut correlated = Array3::zeros(input.raw_dim());
        let mut convolved = Array3::zeros(input.raw_dim());
        convolve_into(
            input.view(),
            &kernel,
            Sense::Correlate,
            Boundary::Clamp,
            correlated.view_mut(),
        )
        .unwrap();
        convolve_into(
            input.view(),
            &kernel,
            Sense::Convolve,
            Boundary::Clamp,
            convolved.view_mut(),
        )
        .unwrap();
        assert_ne!(correlated, convolved);
        // and they differ in most voxels, not in one corner
        let moved = correlated
            .iter()
            .zip(convolved.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > input.len() / 2, "only {moved} voxels moved");
    }

    /// The other half of the pair: a kernel symmetric under negation **cannot**
    /// tell the two apart, so a suite tested only on one would say nothing.
    #[test]
    fn a_symmetric_kernel_cannot_tell_the_two_senses_apart() {
        let input = field([7, 6, 5]);
        let kernel = Kernel::from_radius([1, 1, 0], vec![1.0; 9]).unwrap();
        assert!(kernel.element().is_symmetric());
        let mut correlated = Array3::zeros(input.raw_dim());
        let mut convolved = Array3::zeros(input.raw_dim());
        convolve_into(
            input.view(),
            &kernel,
            Sense::Correlate,
            Boundary::Clamp,
            correlated.view_mut(),
        )
        .unwrap();
        convolve_into(
            input.view(),
            &kernel,
            Sense::Convolve,
            Boundary::Clamp,
            convolved.view_mut(),
        )
        .unwrap();
        assert_eq!(correlated, convolved);
    }

    /// The identity that makes [`Sense`] a naming convention rather than two
    /// algorithms — and the reason `reflected` is the negative control lever.
    #[test]
    fn correlating_is_convolving_with_the_reflected_kernel() {
        let input = field([7, 6, 5]);
        let kernel = lopsided();
        let reflected = kernel.reflected().unwrap();
        let mut correlated = Array3::zeros(input.raw_dim());
        let mut convolved = Array3::zeros(input.raw_dim());
        convolve_into(
            input.view(),
            &kernel,
            Sense::Correlate,
            Boundary::Clamp,
            correlated.view_mut(),
        )
        .unwrap();
        convolve_into(
            input.view(),
            &reflected,
            Sense::Convolve,
            Boundary::Clamp,
            convolved.view_mut(),
        )
        .unwrap();
        assert_eq!(correlated, convolved);
        // reflecting twice is the identity, weights and all
        assert_eq!(reflected.reflected().unwrap(), kernel);
    }

    /// The declaration a plan acts on: the two sides swap under a convolution,
    /// and an op that declared the correlation's sides would be short by three
    /// planes on one face.
    #[test]
    fn the_reach_swaps_sides_under_a_convolution() {
        let kernel = Kernel::from_sides([1, 0, 0], [4, 0, 0], vec![1.0; 6]).unwrap();
        let correlate = kernel.reach_spec(Sense::Correlate);
        let convolve = kernel.reach_spec(Sense::Convolve);
        assert_eq!(correlate.at(0, 3, 20), (1, 4));
        assert_eq!(convolve.at(0, 3, 20), (4, 1));
        // and the symmetric bound is the same for both
        assert_eq!(kernel.reach(0), 4);
    }

    /// A constant block is filled from the declaration rather than computed, so
    /// the declaration has to be what the computation would have written — in
    /// bits, not approximately.
    #[test]
    fn a_constant_field_maps_to_the_declared_value_in_bits() {
        let shape = [5usize, 4, 3];
        let kernel =
            Kernel::from_radius([1, 1, 0], (1..=9).map(|w| w as f64 / 7.0).collect()).unwrap();
        for &sense in &[Sense::Correlate, Sense::Convolve] {
            let op = ConvolveOp::new("convolve", kernel.clone(), sense, Boundary::Clamp);
            for &value in &[0.0f64, 1.0, -3.5, 0.1] {
                let input: Voxels = Array3::from_elem((shape[0], shape[1], shape[2]), value).into();
                let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
                op.apply(&input, &mut out, &Anchor::whole(shape)).unwrap();
                let declared = op.constant_maps_to(value).expect("a declaration");
                let computed = out.view::<f64>().unwrap();
                for &written in computed.iter() {
                    assert_eq!(
                        written.to_bits(),
                        declared.to_bits(),
                        "value {value}, {:?}",
                        sense
                    );
                }
            }
        }
    }

    /// The boundary convention is the one asked for, and the two conventions
    /// really differ here — otherwise the parameter would be decoration.
    #[test]
    fn the_boundary_convention_is_a_parameter_and_the_two_differ() {
        // A one-sided kernel that only ever reads below the anchor, so voxel 0
        // reads outside the array on every tap but the last.
        let kernel = Kernel::from_sides([2, 0, 0], [0, 0, 0], vec![1.0, 10.0, 100.0]).unwrap();
        let input: Array3<f64> =
            Array3::from_shape_vec((4, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut clamped = Array3::zeros(input.raw_dim());
        let mut reflected = Array3::zeros(input.raw_dim());
        convolve_into(
            input.view(),
            &kernel,
            Sense::Correlate,
            Boundary::Clamp,
            clamped.view_mut(),
        )
        .unwrap();
        convolve_into(
            input.view(),
            &kernel,
            Sense::Correlate,
            Boundary::Reflect,
            reflected.view_mut(),
        )
        .unwrap();
        // At voxel 0 the taps are -2, -1, 0. Clamp reads 0, 0, 0 -> 1*1 + 10*1 + 100*1.
        assert_eq!(clamped[[0, 0, 0]], 111.0);
        // Reflect reads 1, 0, 0 -> 1*2 + 10*1 + 100*1.
        assert_eq!(reflected[[0, 0, 0]], 112.0);
        // and they agree in the interior, where nothing leaves the array
        assert_eq!(clamped[[3, 0, 0]], reflected[[3, 0, 0]]);
        assert_eq!(clamped[[3, 0, 0]], 1.0 * 2.0 + 10.0 * 3.0 + 100.0 * 4.0);
    }

    /// The weight sum is the sum the gather does, so it is the number a caller
    /// checks a mean-preserving or a constant-annihilating kernel against.
    #[test]
    fn a_derivative_kernel_annihilates_a_constant_exactly() {
        let kernel = Kernel::from_radius([1, 0, 0], vec![-1.0, 0.0, 1.0]).unwrap();
        assert_eq!(kernel.total(), 0.0);
        let op = ConvolveOp::new("gradient", kernel, Sense::Convolve, Boundary::Clamp);
        assert_eq!(op.constant_maps_to(7.25), Some(0.0));
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
    }

    /// The strided path and the contiguous one are the same function, to the
    /// bit. Both are reachable — `ArrayView3` need not be in standard layout —
    /// and a fallback that differed in the last place would be a wrong answer
    /// that only showed up for some callers.
    #[test]
    fn the_strided_path_is_the_contiguous_one_to_the_bit() {
        let source = field([7, 6, 5]);
        let strided = source.view().reversed_axes();
        assert!(!strided.is_standard_layout());
        let packed = ndarray::Array3::from_shape_fn(
            (strided.shape()[0], strided.shape()[1], strided.shape()[2]),
            |(i, j, k)| strided[[i, j, k]],
        );
        assert!(packed.is_standard_layout());
        let kernel = lopsided();
        let mut through_strides = ndarray::Array3::zeros(strided.raw_dim());
        let mut through_slices = ndarray::Array3::zeros(packed.raw_dim());
        convolve_into(
            strided,
            &kernel,
            Sense::Convolve,
            Boundary::Reflect,
            through_strides.view_mut(),
        )
        .unwrap();
        convolve_into(
            packed.view(),
            &kernel,
            Sense::Convolve,
            Boundary::Reflect,
            through_slices.view_mut(),
        )
        .unwrap();
        assert_eq!(through_strides, through_slices);
        assert!(through_slices.iter().any(|&value| value != 0.0));
    }

    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_a_tap() {
        println!("{}", cost_report([96, 64, 64], 20));
    }
}
