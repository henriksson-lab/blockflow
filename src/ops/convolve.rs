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

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::reach::{AxisReach, Reach};
use crate::voxels::Voxels;

use super::element::{ElementShape, StepOrigin, StructuringElement};
use super::fft::{next_smooth_length, RealTransform3, Spectrum3};
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
    /// **A stencil.** Each output voxel is the weighted sum of a fixed tap list
    /// read at fixed offsets from that voxel, in the element's own order — no
    /// accumulator crosses the block, the output lattice is the input lattice,
    /// and the sum's order is the kernel's rather than the buffer's, which is
    /// what makes a cut leave every bit alone. `super`'s own header says the
    /// order "is a property of the kernel rather than of the block, which is
    /// what makes the answer decomposition-invariant to the bit"; slicing is the
    /// same claim one level down.
    ///
    /// Held to it by `tests/intra_block_slicing.rs`, which is the bar rather
    /// than this declaration.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

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

/// Measured, per unit of `padded * log2(padded) / tile` — the transform's own
/// work, divided by the voxels one tile produces.
///
/// Denominated in [`super::MAP_COST`] like every other constant in `super`, and
/// **cross-denominated through the direct op rather than through a fresh
/// threshold measurement**, because the two paths were timed in the same session
/// on the same fixture and that is what makes the ratio a ratio: the direct op
/// read `3.6 ns/tap/voxel` there against the `1.94` its own table was taken at,
/// so this session ran `1.85x` slow and one `MAP_COST` unit was `1.33 ns`.
///
/// ```text
/// 96 x 64 x 64, --release, one thread, best of 5
/// radius       tile      padded   taps   ns/voxel   direct ns/voxel   ns per unit
/// [1, 1, 1]  [16,16,16]  [18^3]     27      79.49             91.15        4.4626
/// [1, 1, 1]  [32,32,32]  [36^3]     27      69.62             90.98        3.1528
/// [2, 2, 2]  [16,16,16]  [20^3]    125     122.97            445.57        4.8559
/// [2, 2, 2]  [32,32,32]  [36^3]    125      60.22            478.67        2.7268
/// [4, 4, 4]  [16,16,16]  [24^3]    729     130.22           2847.58        2.8051
/// [4, 4, 4]  [32,32,32]  [40^3]    729      94.86           4015.33        3.0419
/// ```
///
/// `3.0 / 1.33` is `2.3`, and the constant is that.
///
/// **This is a coarser constant than [`CONVOLVE_COST_PER_TAP`] and the
/// difference is stated rather than smoothed.** The per-tap figure is flat to
/// 5% across a 5.4x range of kernel sizes; this one spans `2.73` to `4.86`, a
/// factor of `1.8`, across a 27x range of tap counts and two tiles. Two things
/// it does not model: how *smooth* the padded length happens to be — a length
/// that lands on a power of two is cheaper per element than one that lands on a
/// product of threes and fives — and whether the tile's working set stays in
/// cache, which is why every `[32, 32, 32]` row is cheaper per unit than its
/// `[16, 16, 16]` neighbour. A model that claimed 5% here would be claiming
/// something the measurement does not show.
///
/// **What the table is actually evidence of.** The transform beats the direct
/// gather by `7.9x` at radius two and by **`42x`** at radius four, and loses to
/// it — barely, `79` against `91` — at radius one with the smaller tile. That is
/// the crossover a planner would need to see, and nothing consults either
/// number: see [`TransformConvolveOp::cost_per_voxel`] on why that is a question
/// about the search rather than about the coefficient.
pub const TRANSFORM_CONVOLVE_COST: f64 = 2.3;

/// Time a transform convolution, and print the table
/// [`TRANSFORM_CONVOLVE_COST`] is read off.
///
/// Its own report rather than a row in `ops::cost::measure`, on
/// [`cost_report`]'s precedent and for its reason.
pub fn transform_convolve_cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let input: Voxels =
        ndarray::Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (i + j + k) as f64
        })
        .into();
    let at = Anchor::whole(shape);
    let mut report = String::from(
        "radius       tile   padded   taps   ns/voxel  direct ns/voxel   ns per n log n unit\n",
    );
    for radius in [[1usize, 1, 1], [2, 2, 2], [4, 4, 4]] {
        for tile in [[16usize, 16, 16], [32, 32, 32]] {
            let size = (2 * radius[0] + 1) * (2 * radius[1] + 1) * (2 * radius[2] + 1);
            let weights: Vec<f64> = (0..size)
                .map(|which| (which as f64 + 1.0) / size as f64)
                .collect();
            let kernel = Kernel::from_radius(radius, weights).expect("a kernel");
            let taps = kernel.len();
            let op = TransformConvolveOp::new(
                "transform",
                kernel.clone(),
                Sense::Correlate,
                Boundary::Clamp,
                tile,
            )
            .expect("an op");
            let direct = ConvolveOp::new("direct", kernel, Sense::Correlate, Boundary::Clamp);
            let mut out = Voxels::zeros(Dtype::F64, shape).expect("a buffer");
            let mut best = f64::INFINITY;
            let mut best_direct = f64::INFINITY;
            for _ in 0..repetitions.max(1) {
                let started = Instant::now();
                op.apply(&input, &mut out, &at).expect("a block");
                let elapsed = started.elapsed().as_secs_f64() * 1e9 / voxels;
                if elapsed.total_cmp(&best).is_lt() {
                    best = elapsed;
                }
                let started = Instant::now();
                direct.apply(&input, &mut out, &at).expect("a block");
                let elapsed = started.elapsed().as_secs_f64() * 1e9 / voxels;
                if elapsed.total_cmp(&best_direct).is_lt() {
                    best_direct = elapsed;
                }
            }
            let length = op.transform_shape();
            let padded = (length[0] * length[1] * length[2]) as f64;
            let per_voxel_units = padded * padded.log2() / (tile[0] * tile[1] * tile[2]) as f64;
            report.push_str(&format!(
                "{radius:?}  {tile:?}  {length:?}  {taps:5}  {best:9.2}  {best_direct:15.2}  {:19.4}\n",
                best / per_voxel_units
            ));
        }
    }
    report
}

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
            // `f64::total_cmp` and not `f64::min`, which this crate does not use:
            // `f64::min(-0.0, 0.0)` may return either operand.
            if elapsed.total_cmp(&best).is_lt() {
                best = elapsed;
            }
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

// ------------------------------------------------ the same filter, by transform --

/// The same linear filter as [`ConvolveOp`], computed through the Fourier
/// transform instead of by gathering taps.
///
/// **What this is evidence of, and it is the reason it exists.** `ops::fft`'s
/// header says the transform "declines to be an op at all", and the ops survey's
/// G3 row reads that as a missing element type: there is no `Dtype::Complex*`,
/// so — the argument goes — nothing a phase can write, so no frequency-domain
/// operation can ever be a phase. **This is a frequency-domain phase.** It is an
/// ordinary [`BlockOp`] with an ordinary bounded reach, it writes `f64` voxels,
/// and the spectrum never leaves the inside of one `apply`. The element type was
/// never what stood in the way; a *whole-volume* transform's reach was, and a
/// convolution does not have one.
///
/// Overlap-save, and why the tile grid is global
/// ---------------------------------------------
/// A circular convolution of a window long enough to hold the linear one is the
/// linear one — `ops::fft`'s padding rule, with the lag window a kernel's own
/// `[-lo, hi]`. So the volume is cut into **tiles** of [`Self::tile`], each tile
/// is computed from an input window of `tile + lo + hi` padded up to a smooth
/// length, and the tiles are written side by side.
///
/// **The tile grid is anchored at the volume's origin and its stride is this
/// op's own, not the plan's.** That is the whole of what makes this
/// decomposition-invariant: an output voxel's value is a function of its
/// *global* tile and of nothing else, so two runs that cut the volume
/// differently compute every voxel through the identical arithmetic and agree to
/// the bit. A transform over the *block* would be a different transform length
/// per block edge and a different sum per voxel — plausible output, and a
/// different answer at every lattice.
///
/// **What that costs, stated because it is the interesting number.** A block
/// cannot know where the tile boundaries fall inside it, so the halo must cover
/// the worst alignment: `tile - 1 + lo` below and `tile - 1 + hi` above. Against
/// [`ConvolveOp`]'s halo of `lo` and `hi` that is `tile - 1` more per side per
/// axis, and it is paid **even when the plan's blocks happen to be a whole
/// number of tiles**, which is the common case and the one where the true halo
/// is exactly the kernel's. The framework has nowhere to say "cut me on a
/// stride": [`crate::op::BlockConstraint::Extent`] mandates all three extents
/// and gives up the search, and `Constraints` has no per-axis rule at all —
/// which is the ops survey's G9, reached from a new direction. So the price is
/// real, it is a *planning* price and not a correctness one, and a caller who
/// pins the block extent to a multiple of the tile pays a halo it does not need
/// rather than a wrong answer.
///
/// **Corrected: the paragraph above described this file for one pass, and the
/// sentence that stopped being true is "the framework has nowhere to say".** It
/// has one now — [`crate::reach::AxisReach::Aligned`] — and it is a *reach*
/// rather than a constraint, which is a smaller change and a better fit. The
/// halo is still `tile - 1 + lo` and `tile - 1 + hi` to everything that cannot
/// see a lattice; `Reach::in_voxels`, which is handed one and which
/// `decomposition::price_phase` already calls once per candidate grid, discounts
/// it to exactly `lo` and `hi` when the block edge is a whole number of tiles.
/// So the planner *prices* an aligned edge cheaper and prefers it, instead of a
/// constraint refusing everything else — a cost gap answered with a cost, which
/// is the shape it should have had.
///
/// Measured on `1024^3` with a 32-voxel tile and a radius-4 kernel, at the
/// coarse ladder's rungs that are whole tiles: **`30.176x`, `8.309x` and
/// `3.232x`** of read amplification without the discount at edges 32, 64 and
/// 128, against **`1.917x`, `1.394x` and `1.173x`** with it.
///
/// **Those are byte figures and must be quoted as byte figures.** A halo voxel
/// does not cost what a core voxel costs, and the ops survey's G20 row measures
/// by how much: on a cold sequential read the extra bytes ride along on
/// readahead — `3.48x` the bytes for `1.32x` the time — so the amplifications
/// above overstate the *time* the slack costs by roughly `2.6x` cold. Warm they
/// are close to right, and on a chunked store they understate it. The half of
/// this op's case that is not a ratio is unaffected: below, a phase that loses
/// the discount does not merely fetch more, it stops being cuttable at all. In residency, which
/// is the currency a tile-scale stage runs out of, at edge 128 that is **62.1 MB
/// against 20.1 MB per block**, or **2.48 GB against 0.80 GB** at 40-way
/// concurrency.
///
///
/// **And the amplification above understates it, which the planner-level test
/// found and this file did not predict.** `decomposition::cuttable_axes` drops
/// an axis whose `edge + lo + hi` is not less than the extent, and it runs
/// *before* anything is priced. So where the volume is not large against the
/// slack, the axis is not amplified — it is **dropped**, and the phase
/// degenerates to one block reading the whole volume, which is the cost
/// `docs/design/barriers.md` exists to remove, arrived at from the other end.
/// Measured on `96^3` with a 32-voxel tile and a radius-2 kernel: `Greedy` at
/// candidate edge 32 plans **27 blocks** with the discount and **one** without.
/// The two regimes are both real — the amplification is what a volume large
/// against the slack pays, the degeneration is what a volume that is not pays —
/// and `cuttable_axes` resolves the reach against the candidate edge for exactly
/// this reason.
/// **Two things it does not do, and both are stated rather than left to be
/// discovered.** It cannot *demand* an aligned lattice — a caller whose
/// `block_candidates` are all odd multiples of nothing still gets a correct
/// answer at the full halo — and the discount is **lost when this op shares a
/// phase with another**, because `AxisReach::add` and `::max` flatten to the
/// worst case rather than invent a lattice that satisfies two strides. Both are
/// the remaining half of G9 and neither is a correctness question.
///
/// **Corrected: the second of those was measured and it was not a limitation, it
/// was a defect.** A phase's reach is its ops' reaches *added*, so flattening
/// meant that adding a reach of **nothing** was not the identity — fusing this
/// op with a voxelwise map lost the entire discount. Measured on `96^3` at
/// candidate edge 32: **27 blocks alone against one when fused**, which is not a
/// lost discount but a phase reading the whole volume per block. `AxisReach`
/// now carries **both** of its answers and folds each componentwise, so adding
/// nothing is the identity, adding a bounded reach is exact, and two strides
/// take their least common multiple. A multiple past every candidate edge
/// degrades to exactly what flattening gave, so the fold is never dearer than
/// the rule it replaced. `tests/transform_convolution.rs` pins the fused phase
/// end to end.
///
/// **What is still true of the first**: this op cannot demand an aligned
/// lattice, and that half of G9 is untouched.
///
/// The arithmetic this trades for it: a direct gather is one multiply-add per
/// tap per voxel and the tap count is the kernel's product, while this is a
/// fixed transform per tile whose per-voxel cost is `padded log padded / tile`
/// and does **not** grow with the kernel at all. See [`Self::cost_per_voxel`],
/// which is the one place in this crate where an `n log n` is priced — and note
/// that it is priceable precisely *because* `n` is the tile rather than the
/// volume. The survey's G4 is about a cost that grows with the volume; this op
/// has none.
///
/// **It is not bit-identical to [`ConvolveOp`], and must not be sold as if it
/// were.** The two sum the same products in different orders through different
/// arithmetic, so they agree to a rounding and not to a bit. That is why this is
/// a separate op rather than a mode of the other one: a flag that changed the
/// answer in the last place would be a flag no caller could reason about, and
/// this crate's own rule for `ops::fft`'s rejected `f32` axis — "an axis whose
/// other setting cannot meet the bar is not an axis, it is a trap" — applies to
/// an axis whose two settings disagree at all.
///
/// **A transform is not local, and one consequence is measured.** Every sample
/// of a tile's window feeds every one of that tile's outputs, including the
/// samples that only exist to fill positions past a volume face — positions the
/// tile computes and throws away. So the *boundary convention* perturbs the
/// interior of an edge tile in the last place, where a direct gather leaves it
/// bit-identical: at a reach of one voxel [`Boundary::Clamp`] and
/// [`Boundary::Reflect`] resolve every index the answer depends on to the same
/// sample, and `ConvolveOp` therefore gives identical bits under the two while
/// this op does not. `tests/transform_convolution.rs` asserts both halves.
/// **It is not a decomposition hazard** — the window is a function of the global
/// tile and of the volume's faces, and every lattice agrees on both — but it is
/// what "the same function" costs once the arithmetic stops being local.
///
/// **[`crate::op::BlockOp::constant_maps_to`] is therefore `None`.** A uniform
/// block really does map to the weighted sum of the constant, but not to the
/// bit this op would compute, and a short circuit that produces something a
/// computed block would not is the one thing that declaration may never do.
#[derive(Debug, Clone)]
pub struct TransformConvolveOp {
    name: &'static str,
    kernel: Kernel,
    sense: Sense,
    boundary: Boundary,
    tile: [usize; 3],
    lo: [usize; 3],
    hi: [usize; 3],
    length: [usize; 3],
    transform: RealTransform3,
    /// The **conjugated** spectrum of the kernel, laid out at the origin of a
    /// zeroed volume of [`Self::length`].
    ///
    /// Conjugated because the sum this op computes is a *correlation* of the
    /// window against the kernel — `out[i] = sum_m g[m] u[i + m]` — whose
    /// transform is `conj(g_hat) * u_hat`. The sense is already folded into `g`
    /// itself, so there is exactly one convention here and not two.
    kernel_spectrum: Spectrum3,
    cost: f64,
}

impl TransformConvolveOp {
    /// The same three arguments as [`ConvolveOp::new`] plus the tile the
    /// transform runs over.
    ///
    /// Fallible where [`ConvolveOp::new`] is not, because a tile is a shape and
    /// a shape can be empty, and because the plans are built here rather than
    /// per block.
    pub fn new(
        name: &'static str,
        kernel: Kernel,
        sense: Sense,
        boundary: Boundary,
        tile: [usize; 3],
    ) -> Result<Self> {
        if tile.iter().any(|&edge| edge == 0) {
            return Err(Error::InvalidArgument(format!(
                "a transform convolution's tile is the extent it computes at once and must be \
                 non-empty on every axis, got {tile:?}"
            )));
        }
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for axis in 0..3 {
            let (low, high) = kernel.element().sides(axis);
            let (low, high) = match sense {
                Sense::Correlate => (low, high),
                Sense::Convolve => (high, low),
            };
            lo[axis] = low;
            hi[axis] = high;
        }
        let mut length = [0usize; 3];
        for axis in 0..3 {
            let window = tile[axis]
                .checked_add(lo[axis])
                .and_then(|sum| sum.checked_add(hi[axis]))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "a tile of {:?} and a kernel reaching {:?}/{:?} overflow the window \
                         length this transform would need",
                        tile, lo, hi
                    ))
                })?;
            length[axis] = next_smooth_length(window);
        }
        let mut dense =
            Array3::<f64>::zeros((lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1));
        for (offset, &weight) in kernel
            .element()
            .offsets()
            .iter()
            .zip(kernel.weights().iter())
        {
            let placed = sense.displace(*offset);
            let index = [
                (placed[0] + lo[0] as isize) as usize,
                (placed[1] + lo[1] as isize) as usize,
                (placed[2] + lo[2] as isize) as usize,
            ];
            dense[index] = weight;
        }
        let mut transform = RealTransform3::new(length)?;
        let mut kernel_spectrum = transform.spectrum();
        transform.forward_zero_padded(dense.view(), &mut kernel_spectrum)?;
        for value in kernel_spectrum.iter_mut() {
            value.im = -value.im;
        }
        Ok(Self {
            name,
            kernel,
            sense,
            boundary,
            tile,
            lo,
            hi,
            length,
            transform,
            kernel_spectrum,
            cost: TRANSFORM_CONVOLVE_COST,
        })
    }

    /// Override the measured per-`n log n`-unit cost.
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

    /// The extent one transform produces.
    pub fn tile(&self) -> [usize; 3] {
        self.tile
    }

    /// The smooth length one transform actually runs at: the tile plus the
    /// kernel's two sides, rounded up by [`super::fft::next_smooth_length`].
    pub fn transform_shape(&self) -> [usize; 3] {
        self.length
    }

    /// One tile, from a buffer that is already resolved to `f64` on demand.
    #[allow(clippy::too_many_arguments)]
    fn tile_into<T>(
        &self,
        input: &ArrayView3<'_, T>,
        out: &mut ArrayViewMut3<'_, f64>,
        at: &Anchor,
        origin: [usize; 3],
        transform: &mut RealTransform3,
        window: &mut Array3<f64>,
        spectrum: &mut Spectrum3,
        result: &mut Array3<f64>,
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];
        // Where each window sample reads inside the buffer. The boundary
        // convention is resolved at the **buffer's** edge, exactly as
        // `ConvolveOp` resolves it and for the same reason: a sufficient halo
        // makes the buffer's edge a halo edge everywhere but a volume face, and
        // a halo voxel's value is discarded.
        let mut source = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..3 {
            let span = self.tile[axis] + self.lo[axis] + self.hi[axis];
            let mut resolved = Vec::with_capacity(span);
            for step in 0..span {
                let global = origin[axis] as isize - self.lo[axis] as isize + step as isize;
                let local = global - at.offset[axis] as isize;
                resolved.push(self.boundary.index(local, extent[axis]));
            }
            source[axis] = resolved;
        }
        for (a, &i) in source[0].iter().enumerate() {
            for (b, &j) in source[1].iter().enumerate() {
                for (c, &k) in source[2].iter().enumerate() {
                    window[[a, b, c]] = input[[i, j, k]].into();
                }
            }
        }
        transform.forward_zero_padded(window.view(), spectrum)?;
        for (value, factor) in spectrum.iter_mut().zip(self.kernel_spectrum.iter()) {
            *value *= *factor;
        }
        transform.inverse(spectrum, result)?;
        // Where each tile position lands inside the buffer, or nowhere.
        let mut sink = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..3 {
            let mut landing = Vec::with_capacity(self.tile[axis]);
            for step in 0..self.tile[axis] {
                let local = (origin[axis] + step) as isize - at.offset[axis] as isize;
                landing
                    .push((local >= 0 && local < extent[axis] as isize).then_some(local as usize));
            }
            sink[axis] = landing;
        }
        for (a, &i) in sink[0].iter().enumerate() {
            let Some(i) = i else { continue };
            for (b, &j) in sink[1].iter().enumerate() {
                let Some(j) = j else { continue };
                for (c, &k) in sink[2].iter().enumerate() {
                    let Some(k) = k else { continue };
                    out[[i, j, k]] = result[[a, b, c]];
                }
            }
        }
        Ok(())
    }

    fn run<T>(
        &self,
        input: ArrayView3<'_, T>,
        mut out: ArrayViewMut3<'_, f64>,
        at: &Anchor,
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        shapes_agree(input.shape(), out.shape(), "transform convolution")?;
        let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];
        if extent.iter().any(|&length| length == 0) {
            return Ok(());
        }
        let mut transform = self.transform.clone();
        let mut window = Array3::<f64>::zeros((
            self.tile[0] + self.lo[0] + self.hi[0],
            self.tile[1] + self.lo[1] + self.hi[1],
            self.tile[2] + self.lo[2] + self.hi[2],
        ));
        let mut spectrum = transform.spectrum();
        let mut result = Array3::<f64>::zeros((self.length[0], self.length[1], self.length[2]));
        let mut first = [0usize; 3];
        let mut last = [0usize; 3];
        for axis in 0..3 {
            first[axis] = at.offset[axis] / self.tile[axis];
            last[axis] = (at.offset[axis] + extent[axis] - 1) / self.tile[axis];
        }
        for t0 in first[0]..=last[0] {
            for t1 in first[1]..=last[1] {
                for t2 in first[2]..=last[2] {
                    let origin = [t0 * self.tile[0], t1 * self.tile[1], t2 * self.tile[2]];
                    self.tile_into(
                        &input,
                        &mut out,
                        at,
                        origin,
                        &mut transform,
                        &mut window,
                        &mut spectrum,
                        &mut result,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl BlockOp for TransformConvolveOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The tile's own alignment slack on top of the kernel's wider side. The
    /// exact statement is [`Self::reach_spec`]; this is the symmetric bound it
    /// is checked against.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        let (low, high) = self.halo(axis);
        low.max(high)
    }

    /// `tile - 1 + lo` below and `tile - 1 + hi` above, per axis — **unless the
    /// lattice's block edge is a whole number of tiles, and then the kernel's
    /// own two sides.**
    ///
    /// The `tile - 1` is the alignment slack argued in this type's header: a
    /// block's core may begin anywhere inside a tile, so the buffer must reach
    /// back to that tile's start and forward to the end of the last tile the
    /// core touches. It is **not** a property of the kernel.
    ///
    /// It *is* a property of the lattice, and [`AxisReach::Aligned`] is how that
    /// is said. `BlockGrid::cores` builds `start = index * block`, so an edge
    /// that is a multiple of the tile makes every core start tile-aligned and
    /// the slack exactly zero. Everything that cannot see a lattice still gets
    /// the worst case; `Reach::in_voxels`, which is handed one, takes the
    /// discount. Measured on `1024^3` with a 32-voxel tile and a radius-4
    /// kernel, at the ladder rungs that are whole tiles, the discount is
    /// **15.7x, 6.0x and 2.75x** of read amplification at edges 32, 64 and 128.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([self.axis_reach(0), self.axis_reach(1), self.axis_reach(2)])
    }

    /// Every element type a block holds and a transform can widen, **named**
    /// rather than excluded.
    ///
    /// `ConvolveOp` writes this as `dtype != Dtype::F16`, which is the crate's
    /// usual shape and is a liability that costs nothing only while no element
    /// type is ever added: an exclusion says *yes* to every variant a later
    /// change introduces, and an op that accepts what it cannot compute fails in
    /// the executor rather than at plan time — the exact failure `accepts`
    /// exists to remove. This one is a list, so a new variant is refused until
    /// somebody adds it here on purpose.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(
            dtype,
            Dtype::Bool
                | Dtype::U8
                | Dtype::U16
                | Dtype::U32
                | Dtype::U64
                | Dtype::I8
                | Dtype::I16
                | Dtype::I32
                | Dtype::I64
                | Dtype::F32
                | Dtype::F64
        )
    }

    /// `f64`, whatever it read. [`ConvolveOp::produces`]'s reason.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    /// **`at` is load-bearing here**, where `ConvolveOp` ignores it: the tile
    /// grid is anchored to the volume, so a block has to know where it sits to
    /// know which tiles it holds.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        macro_rules! run {
            ($type:ty) => {
                self.run(input.view::<$type>()?, out, at)
            };
        }
        match input.dtype() {
            Dtype::U8 => run!(u8),
            Dtype::U16 => run!(u16),
            Dtype::U32 => run!(u32),
            Dtype::I8 => run!(i8),
            Dtype::I16 => run!(i16),
            Dtype::I32 => run!(i32),
            Dtype::F32 => run!(f32),
            Dtype::F64 => run!(f64),
            // No `Into<f64>`, so the same detour `ConvolveOp::apply` takes.
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                self.run(widened.view(), out, at)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts \
                 ({})",
                self.name,
                self.sense.label()
            ))),
        }
    }

    /// **`None`, and this type's header argues it.** The weighted sum of a
    /// constant is not what this op computes to the bit, and a short circuit
    /// that disagrees with a computed block in the last place is worse than no
    /// short circuit at all.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    /// The transform's own `n log n`, divided by the voxels it produces.
    ///
    /// **Priceable because `n` is the tile.** The ops survey's G4 says the cost
    /// model cannot express a superlinear cost, and it is right about a
    /// transform over the *volume*; a transform over a fixed tile is a constant
    /// per voxel like every other coefficient here, so this op needs nothing the
    /// trait does not already have. What it does need — and what nothing here
    /// provides — is for the planner to be able to *compare* this number against
    /// [`ConvolveOp::cost_per_voxel`] and choose, which is a question about the
    /// search and not about the coefficient.
    fn cost_per_voxel(&self) -> f64 {
        let padded = (self.length[0] as f64) * (self.length[1] as f64) * (self.length[2] as f64);
        let tile = (self.tile[0] as f64) * (self.tile[1] as f64) * (self.tile[2] as f64);
        self.cost * padded * padded.log2() / tile
    }
}

impl TransformConvolveOp {
    /// The declared halo on one axis: the kernel's side plus the tile's
    /// alignment slack.
    fn halo(&self, axis: usize) -> (usize, usize) {
        (
            self.tile[axis] - 1 + self.lo[axis],
            self.tile[axis] - 1 + self.hi[axis],
        )
    }

    /// The same statement as [`Self::halo`], made so a lattice can discount it.
    fn axis_reach(&self, axis: usize) -> AxisReach {
        AxisReach::aligned(self.tile[axis], self.lo[axis], self.hi[axis])
    }
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

    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_a_transform_tile() {
        println!("{}", transform_convolve_cost_report([96, 64, 64], 5));
    }
}
