// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// smooth, differentiate twice, decompose, combine — not adapted from any
// implementation of it.
//
// A multi-scale second-derivative structure filter: the response is large where
// the local intensity forms an **elongated ridge** and small where it forms a
// blob or a sheet.
//
// The construction, in four steps
// -------------------------------
// At one scale, and per voxel:
//
// 1. **Smooth.** A separable Gaussian of a per-axis standard deviation,
//    truncated at a stated multiple of it and normalised to sum to one. Per-axis
//    rather than scalar because a volume's voxels are not required to be cubes,
//    and which sigma belongs on which axis is the caller's knowledge, not this
//    file's.
// 2. **Differentiate twice.** The six components of the Hessian, by the central
//    stencils `[1, -2, 1]` on a diagonal and the four-corner difference off it.
//    Each is scaled by `sigma_a^gamma * sigma_b^gamma` — without that the
//    derivatives of a wider Gaussian are uniformly smaller and step 4's maximum
//    would always choose the narrowest scale, which is not a preference but an
//    artefact.
// 3. **Decompose.** The three eigenvalues of that symmetric 3x3, in closed form.
//    See [`symmetric_eigenvalues`] for what "closed form" costs and what is done
//    about it.
// 4. **Combine.** A [`Response`] folds the eigenvalues into one number. There
//    are two of them and they answer differently, which is why the choice is a
//    parameter rather than a constant:
//    * [`RidgeResponse`] orders the eigenvalues by magnitude and multiplies
//      three saturating terms — the two shape ratios that separate a line from a
//      sheet and from a blob, times a term that suppresses the near-flat
//      background. Bounded by one.
//    * [`RatioResponse`] takes them in algebraic order and carries the largest
//      curvature's **magnitude** through two power-law terms. Unbounded, and in
//      the units of the input's own second derivative. See its documentation for
//      why no setting of the first is the second.
//
// The multi-scale form runs 1-4 at each scale in the list and keeps the largest
// response, which is what makes the filter answer for structures of a range of
// widths rather than one.
//
// The reach, and where its two terms come from
// --------------------------------------------
// `reach = gaussian radius + 1`, per axis, maximised over the scales.
//
// The `+ 1` is the derivative stencil and it is not decoration. To answer at
// voxel `v` the filter needs the *smoothed* field at `v ± 1` on each axis (the
// cross terms need `v ± 1` on two axes at once), and the smoothed field at `u`
// needs the input within the Gaussian radius of `u`. The two compose by
// addition, exactly the way `local.rs`'s lattice distance and window radius do,
// and an implementation that declared only the Gaussian radius would be short by
// one voxel everywhere — the smallest possible understatement and therefore the
// easiest one to not notice.
//
// It is *tight*: the outermost tap of the truncated Gaussian has a non-zero
// weight and the outer coefficient of the second-difference stencil is 1, so the
// answer at `v` genuinely depends on the input at `v ± (radius + 1)`. The
// integration suite understates it by exactly one voxel and requires the output
// to change.
//
// Nothing configures it. The radius is `ceil(truncate * sigma)` computed from
// the same two numbers the kernel is built from, so there is no second statement
// of it that could drift, and no field a caller could set.
//
// Why the intermediates are **not** side outputs
// ----------------------------------------------
// This is the first op here that builds whole intermediate fields — a smoothed
// volume, six Hessian components, three eigenvalues, per scale — and
// `BlockOp::side_outputs` exists for results that need somewhere to go. They are
// nonetheless internal to `apply`, for three reasons, in order of weight:
//
// * **A side output is terminal, and these are not.** The trait's own note says
//   nothing reads a side output back inside the chain; that is why they are
//   declared separately rather than returned from `apply`. The Hessian is the
//   opposite — the primary result is computed *from* it. And `apply_side` is
//   handed the input and the primary and nothing else, so declaring the Hessian
//   would not hand the executor a field that already exists, it would make the
//   op compute it a second time.
// * **The arithmetic.** With `S` scales the declaration is `10 * S` arrays the
//   size of the level. At three scales that is thirty times the primary's bytes,
//   written, for fields that are a pure function of the input and the
//   parameters. `docs/design/BLOCK_OPS.md`'s reason for the machinery was an
//   accounting *shortfall*; declaring these would close no shortfall, because
//   nothing writes them today.
// * **They are not a different shape or rank.** The `Output`/`SideBlock`
//   machinery earns its keep where the second array is a different kind of thing
//   — one row per object, one score per class. Six arrays of the level's own
//   shape are a level's worth of data each and are better recomputed than
//   stored.
//
// What *is* offered as a side output, and what it costs
// -----------------------------------------------------
// One thing, opt-in: **which scale won** ([`RidgeFilterOp::with_scale_map`]).
// That is a result rather than an intermediate — it is not derivable from the
// response, and it is the second half of the answer for a caller who wants to
// know how wide the structure at a voxel is, not only that there is one.
//
// It is opt-in because of a real cost, stated rather than hidden: `apply_side`
// receives the input and the primary, and the argmax is carried by neither, so a
// block that declares the scale map runs the whole multi-scale pass **twice**.
// There is no channel from `apply` to `apply_side` that would let the two share
// it. A caller who does not want the map pays nothing.
//
// Costs
// -----
// Measured; see [`cost_report`], which is runnable and prints the table the two
// constants at the bottom of this file were read off.

use ndarray::{Array3, ArrayD, ArrayView3, ArrayViewMut3, Axis, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Output, SideBlock};
use crate::voxels::Voxels;

use super::shapes_agree;

// ------------------------------------------------------------ smoothing --

/// How far a Gaussian of `sigma` truncated at `truncate` standard deviations
/// reaches, in voxels.
///
/// `ceil`, not `round`: the kernel must contain every tap whose weight is
/// counted, and rounding down would build a narrower kernel than the one this
/// number describes — which is the reach and the arithmetic disagreeing, the
/// failure this crate is arranged against.
///
/// Zero for a zero sigma, which is the **axis this blur does not touch** — see
/// [`gaussian_weights`]. [`ScaleSpace`] still refuses one, because a ridge
/// filter differentiates the smoothed field on every axis and has its own reason
/// to want all three positive.
pub fn gaussian_radius(sigma: f64, truncate: f64) -> usize {
    if !(sigma > 0.0) || !(truncate > 0.0) || !sigma.is_finite() || !truncate.is_finite() {
        return 0;
    }
    (truncate * sigma).ceil() as usize
}

/// The normalised, truncated Gaussian along one axis, lowest offset first.
///
/// The weights are computed from `x * x`, so the kernel is exactly symmetric —
/// the tap at `-k` and the tap at `+k` are the same bits, not merely the same
/// number to within a rounding. That matters for the second-difference stencil
/// above it, whose whole business is cancellation.
///
/// **A sigma of zero is the identity on that axis**, and returns the one-tap
/// kernel `[1.0]`. It is not a degenerate case to be refused: it is how a caller
/// says *this axis is not blurred*, which is what a blur flat on one axis — a
/// plane-by-plane smoothing of a volume — consists of.
///
/// It was refused until a caller needed exactly that and found it could not be
/// **constructed**, while `StructuringElement::from_radius` had always accepted a
/// zero radius for the same meaning. One concept answered two ways is a cost
/// that keeps being paid, so the two now agree.
///
/// The zero case is returned before the loop rather than falling out of it,
/// because the loop divides by `sigma`: at zero the single tap would be
/// `0.0 / 0.0`, and `NaN.exp()` is `NaN`. A special case that produces the
/// obviously right answer is better than arithmetic that produces a quiet one.
pub fn gaussian_weights(sigma: f64, truncate: f64) -> Result<Vec<f64>> {
    if sigma < 0.0 || !sigma.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "a smoothing scale must be a non-negative, finite number of voxels; got {sigma}. \
             Zero is legitimate and means the axis is not blurred."
        )));
    }
    if sigma == 0.0 {
        return Ok(vec![1.0]);
    }
    if !(truncate > 0.0) || !truncate.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "a truncation must be a positive, finite multiple of the scale; got {truncate}"
        )));
    }
    let radius = gaussian_radius(sigma, truncate) as isize;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|step| {
            let ratio = step as f64 / sigma;
            (-0.5 * ratio * ratio).exp()
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) || !total.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "a Gaussian of sigma {sigma} truncated at {truncate} has no usable weight"
        )));
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

/// One separable pass: convolve `src` along `axis` with `kernel` into `dst`.
///
/// **Generic over the element type**, which is as far as the algorithm allows: a
/// convolution accumulates, so it needs a way into `f64` and nothing else. The
/// three passes of a smoothing therefore share one function — the first reads the
/// caller's element type with no copy, the second and third read `f64`.
///
/// The neighbourhood is **clamped** to the array handed in. At a real volume
/// boundary that is the whole story and the whole-volume reference clamps
/// identically; at a block seam it is a truncation the whole volume would not
/// have made, which is what the halo is for and what the guard exists to catch.
fn convolve_axis<T>(
    src: ArrayView3<'_, T>,
    axis: usize,
    kernel: &[f64],
    mut dst: ArrayViewMut3<'_, f64>,
) where
    T: Copy + Into<f64>,
{
    let shape = [src.shape()[0], src.shape()[1], src.shape()[2]];
    let radius = (kernel.len() / 2) as isize;
    let last = shape[axis] as isize - 1;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let here = [i, j, k];
                let centre = here[axis] as isize;
                let mut total = 0.0;
                // Lowest offset first, always: the summation order is part of
                // the answer, and two blocks that summed the same taps in two
                // orders would disagree in the last bit.
                for (step, &weight) in kernel.iter().enumerate() {
                    let mut index = here;
                    index[axis] = (centre + step as isize - radius).clamp(0, last) as usize;
                    total += weight * src[index].into();
                }
                dst[here] = total;
            }
        }
    }
}

/// Smooth `input` with a separable Gaussian, one kernel per axis.
///
/// The passes run in axis order 0, 1, 2. That is a choice and it is fixed here
/// rather than left to the caller, because two callers that ordered them
/// differently would get different last bits from the same parameters.
pub fn gaussian_smooth_into<T>(
    input: ArrayView3<'_, T>,
    kernels: &[Vec<f64>; 3],
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "gaussian_smooth_into")?;
    let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);
    if dim.0 == 0 || dim.1 == 0 || dim.2 == 0 {
        return Ok(());
    }
    let mut scratch = Array3::<f64>::zeros(dim);
    convolve_axis(input, 0, &kernels[0], scratch.view_mut());
    let mut second = Array3::<f64>::zeros(dim);
    convolve_axis(scratch.view(), 1, &kernels[1], second.view_mut());
    convolve_axis(second.view(), 2, &kernels[2], out.view_mut());
    Ok(())
}

// ------------------------------------------------------------- Hessian --

/// The six distinct components of the Hessian at `at`, as
/// `[xx, yy, zz, xy, xz, yz]`.
///
/// Central differences in voxel units: `[1, -2, 1]` on a diagonal component and
/// the quarter four-corner difference off it. Sampling is clamped to `field`,
/// for the smoothing's reason and with the same consequence at a seam: right at
/// a real volume boundary, deliberately wrong short of a sufficient halo.
///
/// **The expression order is load-bearing.** `plus - 2 * mid + minus` is
/// evaluated left to right, so on a field that is constant it is
/// `(c - 2c) + c`, which is `+0.0` exactly in IEEE arithmetic rather than
/// approximately zero. The four-corner form is exactly zero for the same reason.
/// That is what lets [`RidgeFilterOp::constant_maps_to`] be declared at all —
/// see the note there, and the test that checks the bits rather than the theory.
pub fn hessian_at(field: ArrayView3<'_, f64>, at: [usize; 3]) -> [f64; 6] {
    let shape = [field.shape()[0], field.shape()[1], field.shape()[2]];
    let sample = |delta: [isize; 3]| -> f64 {
        let mut index = [0usize; 3];
        for axis in 0..3 {
            let last = shape[axis] as isize - 1;
            index[axis] = (at[axis] as isize + delta[axis]).clamp(0, last) as usize;
        }
        field[index]
    };
    let mid = sample([0, 0, 0]);

    let mut second = [0.0f64; 3];
    for axis in 0..3 {
        let mut plus = [0isize; 3];
        plus[axis] = 1;
        let mut minus = [0isize; 3];
        minus[axis] = -1;
        second[axis] = sample(plus) - 2.0 * mid + sample(minus);
    }

    let mut cross = [0.0f64; 3];
    for (slot, (a, b)) in [(0usize, 1usize), (0, 2), (1, 2)].into_iter().enumerate() {
        let corner = |sa: isize, sb: isize| {
            let mut delta = [0isize; 3];
            delta[a] = sa;
            delta[b] = sb;
            sample(delta)
        };
        cross[slot] = (corner(1, 1) - corner(1, -1) - corner(-1, 1) + corner(-1, -1)) * 0.25;
    }

    [
        second[0], second[1], second[2], cross[0], cross[1], cross[2],
    ]
}

// --------------------------------------------------------- eigenvalues --

/// The three eigenvalues of the symmetric 3x3 `[xx, yy, zz, xy, xz, yz]`, in
/// **descending algebraic order**.
///
/// Closed form, and no dependency: a symmetric 3x3 characteristic polynomial is
/// a cubic whose roots are all real, so the trigonometric solution applies and
/// there is nothing to iterate. Three things are done that the textbook form
/// does not do, and each answers a way it breaks:
///
/// * **The matrix is scaled by its largest magnitude entry first**, and the
///   eigenvalues scaled back at the end. Without it the sum of squares below
///   underflows for a Hessian around `1e-160` — squares of `1e-160` are
///   subnormal and squares of `1e-200` are zero — so `p` comes out zero, the
///   function takes its repeated-root exit and reports three equal eigenvalues
///   for a matrix that has none. A flat region of a smoothed volume produces
///   exactly such a Hessian. There is a test at `1e-200`.
/// * **`acos`'s argument is clamped to `[-1, 1]`.** It is `det(B) / 2` and is
///   *algebraically* in range, but a matrix with a repeated eigenvalue sits
///   exactly at an endpoint and rounding steps outside it — `acos(1 + 1e-16)` is
///   `NaN`, and a `NaN` here becomes a `NaN` voxel. There is a test on a rotated
///   repeated-eigenvalue matrix.
/// * **The middle eigenvalue is recovered from the trace** rather than evaluated
///   as a third cosine, so the three returned always sum to the trace of what was
///   handed in.
///
/// A non-finite component has no eigen-decomposition to report and yields
/// `[0, 0, 0]`, which the response above treats as "no structure". Propagating
/// the `NaN` instead would put it in the output volume, where it is neither a
/// diagnosis nor a value.
///
/// **What it does not fix, stated rather than discovered later.** At a *repeated*
/// eigenvalue the closed form keeps about half the mantissa — `acos` is
/// evaluated at an endpoint of its domain, where its derivative is unbounded, so
/// an error of one ulp in `det(B) / 2` becomes an error of `sqrt(eps)` in the
/// angle. Measured here: a matrix with the exact double root `2` is answered to
/// `1.5e-8`, while its simple root is exact to the last bits. That is inherent to
/// solving a cubic by its trigonometric form and is not something a scaling or a
/// clamp can recover; an iterative Jacobi rotation would, at several times the
/// cost. It is harmless for what this file does with the numbers — the response
/// above is a ratio of magnitudes fed through an exponential, and `1.5e-8` on an
/// eigenvalue is far below the difference between any two structures — but it is
/// the reason `symmetric_eigenvalues` should not be reached for where a
/// well-conditioned decomposition is what is wanted.
pub fn symmetric_eigenvalues(hessian: [f64; 6]) -> [f64; 3] {
    // Stated as an explicit predicate rather than left to the fold below:
    // `f64::max` *ignores* a `NaN` operand, so a fold would quietly report the
    // largest of the finite entries and carry the `NaN` into the arithmetic.
    if !hessian.iter().all(|value| value.is_finite()) {
        return [0.0, 0.0, 0.0];
    }
    let scale = hessian
        .iter()
        .fold(0.0f64, |widest, value| widest.max(value.abs()));
    if scale == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let [xx, yy, zz, xy, xz, yz] = hessian.map(|value| value / scale);

    let third = (xx + yy + zz) / 3.0;
    let (dx, dy, dz) = (xx - third, yy - third, zz - third);
    let spread = (dx * dx + dy * dy + dz * dz + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0;
    let p = spread.sqrt();
    if !(p > 0.0) {
        // A multiple of the identity: every eigenvalue is the mean of the
        // diagonal, and this is the exit that must not be reached by underflow,
        // which is what the scaling above is for.
        let repeated = third * scale;
        return [repeated, repeated, repeated];
    }

    // B = (A - third * I) / p, whose determinant halved is the cosine of three
    // times the angle the roots are spaced by.
    let (bx, by, bz) = (dx / p, dy / p, dz / p);
    let (bxy, bxz, byz) = (xy / p, xz / p, yz / p);
    let determinant =
        bx * (by * bz - byz * byz) - bxy * (bxy * bz - byz * bxz) + bxz * (bxy * byz - by * bxz);
    let phase = (determinant / 2.0).clamp(-1.0, 1.0).acos() / 3.0;

    let high = third + 2.0 * p * phase.cos();
    // `phase` lies in `[0, pi/3]`, so `phase + 2pi/3` lies in `[2pi/3, pi]` and
    // its cosine is the most negative of the three roots' — which is what makes
    // this the smallest without having to sort.
    let low = third + 2.0 * p * (phase + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    let middle = 3.0 * third - high - low;
    [high * scale, middle * scale, low * scale]
}

// ------------------------------------------------------------ response --

/// Which side of its surroundings a structure is on.
///
/// A ridge in the intensity has two negative curvatures across it; a valley has
/// two positive ones. Which of the two a caller is looking for is the caller's
/// knowledge, so it is a parameter and not a convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// Structures **brighter** than what surrounds them.
    Ridge,
    /// Structures **darker** than what surrounds them.
    Valley,
}

/// How the three eigenvalues become one number.
///
/// Ordered by magnitude as `|e1| <= |e2| <= |e3|`, an elongated structure has
/// `|e1|` small — nothing curves along it — and `|e2| ~ |e3|` large. Three terms
/// separate that from everything else, and each has one parameter:
///
/// | term | large when | parameter |
/// |---|---|---|
/// | `1 - exp(-(\|e2\|/\|e3\|)^2 / 2a^2)` | the cross-section is round rather than flat, so a line and not a sheet | `line` |
/// | `exp(-(\|e1\|/sqrt(\|e2 e3\|))^2 / 2b^2)` | there is no curvature along the structure, so a line and not a blob | `blob` |
/// | `1 - exp(-\|e\|^2 / 2c^2)` | there is curvature at all, so structure and not background | `strength` |
///
/// The response is their product, and zero wherever the two large curvatures do
/// not both have [`Polarity`]'s sign.
///
/// **All three are parameters and none has a default here.** `strength` in
/// particular is in the units of the input's own second derivative, so a value
/// that suits one dataset is meaningless for another — baking one in would be
/// precisely the leak `README.md`'s rule 1 describes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgeResponse {
    line: f64,
    blob: f64,
    strength: f64,
    polarity: Polarity,
}

impl RidgeResponse {
    pub fn new(line: f64, blob: f64, strength: f64, polarity: Polarity) -> Result<Self> {
        for (what, value) in [("line", line), ("blob", blob), ("strength", strength)] {
            if !(value > 0.0) || !value.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "the {what} sensitivity divides an exponent and must be positive and \
                     finite; got {value}"
                )));
            }
        }
        Ok(Self {
            line,
            blob,
            strength,
            polarity,
        })
    }

    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Fold eigenvalues in **descending algebraic order** — what
    /// [`symmetric_eigenvalues`] returns — into the response.
    ///
    /// Two exits before the arithmetic, and both are about division rather than
    /// about modelling:
    ///
    /// * the wrong polarity, which is the ordinary "not this kind of structure"
    ///   answer;
    /// * a largest magnitude of exactly zero, which is a Hessian of exactly zero
    ///   and would otherwise make the first ratio `0 / 0`. This is the exit a
    ///   constant block takes, and it returns `0.0` rather than a value that
    ///   merely rounds to it.
    pub fn evaluate(&self, eigenvalues: [f64; 3]) -> f64 {
        let mut ordered = eigenvalues;
        ordered.sort_by(|left, right| left.abs().total_cmp(&right.abs()));
        let [small, middle, large] = ordered;

        let wanted = match self.polarity {
            Polarity::Ridge => -1.0,
            Polarity::Valley => 1.0,
        };
        if middle * wanted < 0.0 || large * wanted < 0.0 {
            return 0.0;
        }
        let (small, middle, large) = (small.abs(), middle.abs(), large.abs());
        if large == 0.0 {
            return 0.0;
        }

        let flatness = middle / large;
        let product = middle * large;
        // `middle` is zero only if `small` is too, and then the flatness term
        // below is already zero — so the value chosen here cannot change the
        // answer, and zero is the one that cannot produce a `NaN`.
        let roundness = if product > 0.0 {
            small / product.sqrt()
        } else {
            0.0
        };
        let energy = small * small + middle * middle + large * large;

        let line = 1.0 - (-(flatness * flatness) / (2.0 * self.line * self.line)).exp();
        let blob = (-(roundness * roundness) / (2.0 * self.blob * self.blob)).exp();
        let present = 1.0 - (-energy / (2.0 * self.strength * self.strength)).exp();
        line * blob * present
    }
}

/// The **second** response, and a different question from the first rather than
/// a re-parameterisation of it.
///
/// Over the same three eigenvalues in the same descending algebraic order, and
/// with `high >= middle >= low`:
///
/// ```text
/// zero unless both middle and low have the polarity's sign
/// |low| * (middle / low)^cross * along^along_power
/// where along = 1 + high / |middle|            if high has the polarity's sign or is zero
///               1 - opposed * |high| / |middle| if it has the opposite sign, and zero if that is negative
/// ```
///
/// **No setting of [`RidgeResponse`]'s three sensitivities is this, and the
/// difference is structural rather than a matter of tuning.** Three ways:
///
/// * **The magnitude is carried, not saturated.** `|low|` appears as a factor,
///   so the response is in the units of the input's own second derivative and is
///   unbounded above. `RidgeResponse`'s corresponding term is
///   `1 - exp(-|e|^2 / 2c^2)`, which is bounded by one and saturates: past a few
///   `strength` it stops distinguishing a strong ridge from an overwhelming one.
///   Which is wanted depends on what reads the number — a comparison against a
///   fixed level wants the first, a product of shape terms wants the second.
/// * **The cross-section term is a power, not a Gaussian.** `(middle / low)` is
///   in `(0, 1]` and is raised to `cross`; `RidgeResponse` puts the same ratio
///   through `1 - exp(-r^2 / 2a^2)`. At `cross = 0` the term is exactly one and
///   the response stops distinguishing a line from a sheet at all, which is a
///   configuration the exponential form cannot reach for any positive `line`.
/// * **The along-axis term is signed and one-sided.** A curvature along the
///   structure that has the *same* sign as the two across it reduces the
///   response linearly towards zero; one of the opposite sign reduces it at
///   `opposed` times the rate and cuts it to zero outright once it dominates.
///   `RidgeResponse` folds the same eigenvalue through `|small| / sqrt(|middle
///   low|)` and an exponential, which is even in the sign it discards.
///
/// The three parameters
/// --------------------
/// | parameter | what it weights |
/// |---|---|
/// | `cross` | the exponent on the ratio of the two curvatures across the structure. `0` ignores the cross-section's shape, larger values punish a flat one harder |
/// | `along_power` | the exponent on the along-axis term |
/// | `opposed` | how hard a curvature of the *wrong* sign along the structure is punished, relative to one of the right sign |
///
/// None has a default here, for [`RidgeResponse`]'s reason: they trade against
/// each other and against the data, and a value baked in would be this crate
/// knowing something about a caller's images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatioResponse {
    cross: f64,
    along_power: f64,
    opposed: f64,
    polarity: Polarity,
}

impl RatioResponse {
    pub fn new(cross: f64, along_power: f64, opposed: f64, polarity: Polarity) -> Result<Self> {
        for (what, value) in [
            ("cross", cross),
            ("along_power", along_power),
            ("opposed", opposed),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "the {what} weight must be finite and not negative; got {value}. The two \
                     exponents are applied to bases in [0, 1], where a negative exponent sends \
                     a base of zero to an infinity; a negative {what} would make a flatter \
                     cross-section or a stronger opposing curvature respond *more*, which \
                     inverts what this response is."
                )));
            }
        }
        Ok(Self {
            cross,
            along_power,
            opposed,
            polarity,
        })
    }

    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Fold eigenvalues in **descending algebraic order** — what
    /// [`symmetric_eigenvalues`] returns — into the response.
    ///
    /// [`Polarity::Valley`] is served by negating the triple, which reverses the
    /// order, and evaluating the ridge form on it. That is exact rather than
    /// approximate: negation of a finite `f64` is a sign-bit flip and every
    /// quantity below is a ratio or a magnitude of the negated values, so a
    /// valley of a volume answers bit-for-bit what a ridge of its negation
    /// does.
    ///
    /// **What the eigenvalues' own conditioning costs here**, since this reads
    /// them as ratios rather than through an exponential. `symmetric_
    /// eigenvalues` keeps only about `sqrt(eps)` at a repeated root — see its
    /// note, which measures it — and `middle / low` is exactly the ratio that is
    /// near one there. So the cross-section term is accurate to about `1e-8`
    /// precisely where it is closest to its maximum, and its *derivative* with
    /// respect to the shape is smallest there too, which is the fortunate
    /// direction: an error of `1e-8` in a ratio that the response is flat in
    /// moves the answer by less than that. The along-axis term divides by
    /// `|middle|`, which is the well-conditioned root of the pair unless all
    /// three coincide — and all three coinciding is a sphere, where the response
    /// is near zero for any parameters. Neither is a case this crate can improve
    /// without a different solver; both are stated so that a caller reading
    /// `1e-8` differences between two runs knows where they came from.
    pub fn evaluate(&self, eigenvalues: [f64; 3]) -> f64 {
        let [high, middle, low] = match self.polarity {
            Polarity::Ridge => eigenvalues,
            Polarity::Valley => [-eigenvalues[2], -eigenvalues[1], -eigenvalues[0]],
        };
        // Both curvatures across the structure must be strictly negative. Not
        // `<= 0`: a zero `low` would make the ratio below `0 / 0`, and a
        // structure with no curvature across it is not this structure.
        if !(middle < 0.0 && low < 0.0) {
            return 0.0;
        }
        let cross = (middle / low).powf(self.cross);
        let along = if high <= 0.0 {
            // `high` is between `middle` and zero, so this is in `[0, 1]` and
            // is one exactly where there is no curvature along the structure.
            1.0 + high / middle.abs()
        } else {
            let opposed = self.opposed * high / middle.abs();
            // Stated as `!(x < 1)` rather than `x >= 1` so that a `NaN` — which
            // this cannot produce from finite eigenvalues, but a caller passing
            // eigenvalues from elsewhere could — takes the zero exit instead of
            // being carried into the product.
            if !(opposed < 1.0) {
                // The opposing curvature dominates. Returned here rather than
                // letting `(1 - opposed)` go negative and be raised to a
                // fractional power, which is a `NaN`.
                return 0.0;
            }
            1.0 - opposed
        };
        low.abs() * cross * along.powf(self.along_power)
    }
}

/// Which of the two responses a filter folds its eigenvalues with.
///
/// An enum rather than a trait object because there are two of them and both are
/// small `Copy` structs: the whole point of [`EigenResponse`] below is that the
/// kernel can be generic and pay nothing, and a `Box<dyn>` per voxel would undo
/// that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Response {
    /// [`RidgeResponse`]: three saturating shape terms multiplied, bounded by
    /// one.
    Saturating(RidgeResponse),
    /// [`RatioResponse`]: the largest curvature's magnitude scaled by two
    /// powers, unbounded.
    Ratio(RatioResponse),
}

impl Response {
    pub fn polarity(&self) -> Polarity {
        match self {
            Response::Saturating(response) => response.polarity(),
            Response::Ratio(response) => response.polarity(),
        }
    }
}

impl From<RidgeResponse> for Response {
    fn from(response: RidgeResponse) -> Self {
        Response::Saturating(response)
    }
}

impl From<RatioResponse> for Response {
    fn from(response: RatioResponse) -> Self {
        Response::Ratio(response)
    }
}

/// Anything that folds three eigenvalues into one number.
///
/// It exists so that [`ridge_response_into`] is generic over the fold rather
/// than naming one of them, which is what lets a second response be added
/// **without touching the first**: `RidgeResponse` gains an impl and loses
/// nothing, and a caller who was passing one keeps passing one.
///
/// Deliberately not a hook for arbitrary caller-supplied folds — it is a
/// vocabulary this file's two implementations share and the [`Response`] enum
/// closes over. A caller who has a third fold is welcome to implement it and
/// call the kernel; what they cannot do is put it in a [`RidgeFilterOp`], which
/// is a limit worth having until there is a third.
pub trait EigenResponse {
    /// The eigenvalues arrive in **descending algebraic order**, which is what
    /// [`symmetric_eigenvalues`] produces.
    fn evaluate(&self, eigenvalues: [f64; 3]) -> f64;
}

impl EigenResponse for RidgeResponse {
    fn evaluate(&self, eigenvalues: [f64; 3]) -> f64 {
        RidgeResponse::evaluate(self, eigenvalues)
    }
}

impl EigenResponse for RatioResponse {
    fn evaluate(&self, eigenvalues: [f64; 3]) -> f64 {
        RatioResponse::evaluate(self, eigenvalues)
    }
}

impl EigenResponse for Response {
    fn evaluate(&self, eigenvalues: [f64; 3]) -> f64 {
        match self {
            Response::Saturating(response) => response.evaluate(eigenvalues),
            Response::Ratio(response) => response.evaluate(eigenvalues),
        }
    }
}

// --------------------------------------------------------- scale space --

/// The scales the filter is evaluated at, and everything the smoothing and the
/// reach are derived from.
///
/// A scale is a **per-axis** sigma. Anisotropic voxels are the ordinary case and
/// a scalar sigma would quietly assume they are cubes; [`Self::isotropic`] is
/// the convenience for when they are.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleSpace {
    scales: Vec<[f64; 3]>,
    truncate: f64,
    gamma: f64,
}

impl ScaleSpace {
    /// `truncate` is how many standard deviations the Gaussian is cut at, and
    /// `gamma` is the exponent that makes responses at different scales
    /// comparable. Both are parameters: the first trades reach against
    /// faithfulness and the second trades the filter's preference between narrow
    /// and wide structures, and neither has an answer that is right for every
    /// caller.
    pub fn new(scales: Vec<[f64; 3]>, truncate: f64, gamma: f64) -> Result<Self> {
        if scales.is_empty() {
            return Err(Error::InvalidArgument(
                "a scale space needs at least one scale; with none there is nothing to \
                 take a maximum over"
                    .to_string(),
            ));
        }
        for sigma in &scales {
            for axis in 0..3 {
                if !(sigma[axis] > 0.0) || !sigma[axis].is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "every axis of a scale must be a positive, finite number of voxels; \
                         got {sigma:?}. An axis of extent one is not an exception — smoothing \
                         it clamps to its single plane and costs nothing."
                    )));
                }
            }
        }
        if !(truncate > 0.0) || !truncate.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "a truncation must be a positive, finite multiple of the scale; got {truncate}"
            )));
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "a scale normalisation exponent must be finite and not negative; got {gamma}"
            )));
        }
        // The radius is `ceil(truncate * sigma)` cast to `usize`, and a cast in
        // Rust saturates rather than wrapping — so a sigma of `1e30` would give
        // a reach of `usize::MAX` and every arithmetic on it downstream would
        // overflow. The bound is generous rather than tuned: nothing that reads
        // a million voxels either side of the one it writes is a local op, and
        // saying so here is better than saying it as a panic in a cost model.
        for sigma in &scales {
            for axis in 0..3 {
                if truncate * sigma[axis] > 1e6 {
                    return Err(Error::InvalidArgument(format!(
                        "a scale of {sigma:?} truncated at {truncate} reaches more than a \
                         million voxels on axis {axis}. That is not a local operation, and a \
                         reach that large overflows every number derived from it."
                    )));
                }
            }
        }
        Ok(Self {
            scales,
            truncate,
            gamma,
        })
    }

    /// The same sigma on every axis, one scale per entry.
    pub fn isotropic(sigmas: &[f64], truncate: f64, gamma: f64) -> Result<Self> {
        Self::new(
            sigmas.iter().map(|&sigma| [sigma; 3]).collect(),
            truncate,
            gamma,
        )
    }

    pub fn scales(&self) -> &[[f64; 3]] {
        &self.scales
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The Gaussian radius of scale `which` on `axis`.
    pub fn radius(&self, which: usize, axis: usize) -> usize {
        gaussian_radius(self.scales[which][axis], self.truncate)
    }

    /// **The reach, derived and not configured**: the widest Gaussian in the
    /// list, plus the one voxel the second-difference stencil adds on top of it.
    ///
    /// The maximum over scales rather than the sum, because the scales are
    /// alternatives folded by a maximum rather than stages applied in turn — the
    /// same distinction `Chain` draws between `Alternative` and `Sequence`,
    /// arising here inside one op.
    pub fn reach(&self, axis: usize) -> usize {
        (0..self.scales.len())
            .map(|which| self.radius(which, axis) + 1)
            .max()
            .unwrap_or(0)
    }

    /// The three separable kernels of scale `which`.
    pub fn kernels(&self, which: usize) -> Result<[Vec<f64>; 3]> {
        let sigma = self.scales[which];
        Ok([
            gaussian_weights(sigma[0], self.truncate)?,
            gaussian_weights(sigma[1], self.truncate)?,
            gaussian_weights(sigma[2], self.truncate)?,
        ])
    }

    /// The factor each Hessian component of scale `which` is multiplied by, as
    /// one `sigma_a^gamma` per axis — a component differentiated twice along `a`
    /// is scaled by the square of it, a cross component by the product of the
    /// two axes it differentiates along.
    ///
    /// A per-axis form rather than a single `sigma^(2 gamma)` because the scales
    /// here are per-axis; collapsing them to one number would have to pick a mean
    /// and would normalise an anisotropic scale wrongly on both of its axes.
    pub fn normalisation(&self, which: usize) -> [f64; 3] {
        let sigma = self.scales[which];
        [
            sigma[0].powf(self.gamma),
            sigma[1].powf(self.gamma),
            sigma[2].powf(self.gamma),
        ]
    }
}

/// Multiply a Hessian by a per-axis normalisation.
fn normalise(hessian: [f64; 6], factor: [f64; 3]) -> [f64; 6] {
    [
        hessian[0] * factor[0] * factor[0],
        hessian[1] * factor[1] * factor[1],
        hessian[2] * factor[2] * factor[2],
        hessian[3] * factor[0] * factor[1],
        hessian[4] * factor[0] * factor[2],
        hessian[5] * factor[1] * factor[2],
    ]
}

// -------------------------------------------------------------- kernel --

/// The whole filter: the largest response over the scales, and optionally which
/// scale gave it.
///
/// Generic over the element type as far as the algorithm allows. Everything past
/// the first convolution is `f64` and has to be — a second difference of a
/// smoothed field is signed and fractional whatever the input was, which is the
/// same reason `local.rs` states for its own accumulator.
///
/// `chosen`, where given, receives the **index** into the scale list rather than
/// a sigma: a scale here is three numbers, and an index is the one answer that
/// does not have to choose which of them to report.
///
/// Ties go to the earlier scale, by comparing strictly. That is a decision and
/// not an accident: it makes the answer a function of the parameters rather than
/// of the iteration order.
pub fn ridge_response_into<T, R>(
    input: ArrayView3<'_, T>,
    scales: &ScaleSpace,
    response: &R,
    mut out: ArrayViewMut3<'_, f64>,
    mut chosen: Option<ArrayViewMut3<'_, f64>>,
) -> Result<()>
where
    T: Copy + Into<f64>,
    R: EigenResponse + ?Sized,
{
    shapes_agree(input.shape(), out.shape(), "ridge_response_into")?;
    if let Some(chosen) = chosen.as_ref() {
        shapes_agree(
            input.shape(),
            chosen.shape(),
            "ridge_response_into (scale map)",
        )?;
    }
    let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);
    if dim.0 == 0 || dim.1 == 0 || dim.2 == 0 {
        return Ok(());
    }

    let mut smoothed = Array3::<f64>::zeros(dim);
    for which in 0..scales.scales().len() {
        let kernels = scales.kernels(which)?;
        gaussian_smooth_into(input, &kernels, smoothed.view_mut())?;
        let factor = scales.normalisation(which);
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let hessian = normalise(hessian_at(smoothed.view(), [i, j, k]), factor);
                    let value = response.evaluate(symmetric_eigenvalues(hessian));
                    if which == 0 || value > out[[i, j, k]] {
                        out[[i, j, k]] = value;
                        if let Some(chosen) = chosen.as_mut() {
                            chosen[[i, j, k]] = which as f64;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// --------------------------------------------------------------- shell --

/// A multi-scale Hessian ridge filter as a [`BlockOp`].
pub struct RidgeFilterOp {
    name: &'static str,
    scales: ScaleSpace,
    response: Response,
    scale_map: Option<&'static str>,
    cost: f64,
}

impl RidgeFilterOp {
    /// `response` is either of the two folds — see [`Response`] — and is taken
    /// as `impl Into<Response>` so that handing it a [`RidgeResponse`] directly,
    /// which is what every caller of this op did before there was a second
    /// fold, still says what it always said.
    pub fn new(name: &'static str, scales: ScaleSpace, response: impl Into<Response>) -> Self {
        let cost = cost_for(&scales);
        Self {
            name,
            scales,
            response: response.into(),
            scale_map: None,
            cost,
        }
    }

    /// Also write, as a side output named `{name}.{suffix}`, the index of the
    /// scale whose response won at each voxel.
    ///
    /// **This doubles the op's compute for a block**, and the reason is
    /// structural rather than an oversight: `apply_side` is handed the input and
    /// the primary result, and the argmax is carried by neither, so the
    /// multi-scale pass runs again to recover it. There is no channel from
    /// `apply` to `apply_side` to carry it across. That is why the map is opt-in
    /// — a caller who does not ask pays nothing at all.
    pub fn with_scale_map(mut self, suffix: &'static str) -> Self {
        self.scale_map = Some(suffix);
        self
    }

    pub fn scales(&self) -> &ScaleSpace {
        &self.scales
    }

    pub fn response(&self) -> &Response {
        &self.response
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The dispatch from the tagged buffer to the generic kernel.
    ///
    /// Three groups, and the middle one is the interesting one. The types with a
    /// lossless `Into<f64>` reach the kernel with **no copy**, which is the whole
    /// point of the kernel stating a standard-library bound. `bool`, `u64` and
    /// `i64` have no such conversion — `bool` has none at all and the two wide
    /// integers have none that is lossless — so the shell widens them through
    /// `Voxels::widened`, which is a copy it is *choosing* rather than a
    /// conversion the kernel would have to invent. `f16` is refused before a run
    /// starts, by `accepts`, because no buffer holds one.
    fn evaluate(
        &self,
        input: &Voxels,
        out: ArrayViewMut3<'_, f64>,
        chosen: Option<ArrayViewMut3<'_, f64>>,
    ) -> Result<()> {
        macro_rules! direct {
            ($type:ty) => {
                ridge_response_into(
                    input.view::<$type>()?,
                    &self.scales,
                    &self.response,
                    out,
                    chosen,
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
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                ridge_response_into(widened.view(), &self.scales, &self.response, out, chosen)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }
}

impl BlockOp for RidgeFilterOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The widest Gaussian radius in the scale list, plus the derivative
    /// stencil's one voxel. Both terms come from the parameters and there is no
    /// field that could set either.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.scales.reach(axis)
    }

    /// Every element type a buffer holds. The filter reads and accumulates; what
    /// it accumulates into is its own business and is always `f64`.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `f64` whatever came in. The response is a product of exponentials of
    /// ratios of second derivatives, which is fractional and signed-in-the-middle
    /// even for a `u8` volume; narrowing it back would be the shell inventing a
    /// quantisation the caller did not ask for.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        self.evaluate(input, out.view_mut::<f64>()?, None)
    }

    /// **Exactly zero, for every constant** — and that is a stronger claim than
    /// the local mean can make, so it is worth saying why it holds here.
    ///
    /// The mean withholds its mapping because `(v + ... + v) / m` is not `v` in
    /// binary floating point. This op never needs that step to be exact. The
    /// smoothing of a constant field produces the *same* value at every voxel —
    /// whatever that value is, and it is not exactly `v` — because every voxel
    /// sums the same taps in the same order, clamping included. The second
    /// difference of a field that is bit-for-bit constant is `(c - 2c) + c`,
    /// which IEEE arithmetic makes exactly `+0.0`; the four-corner difference is
    /// exactly `+0.0` for the same reason. A Hessian of exactly zero has a
    /// largest magnitude of exactly zero, which is the exit
    /// [`RidgeResponse::evaluate`] takes before it divides, and it returns
    /// `0.0`. The maximum over scales of zeroes is zero.
    ///
    /// So the error the mean is protecting against cannot arise: the rounding
    /// happens *before* a cancellation that is exact regardless of what it
    /// cancels. The test `a_constant_block_is_exactly_zero_and_not_nearly_zero`
    /// checks the bits over a range of constants rather than trusting this
    /// paragraph.
    ///
    /// **The last step is asked rather than assumed.** What is exactly true is
    /// that the eigenvalues are `[0, 0, 0]`; what that *becomes* is the
    /// response's business, so the declaration evaluates the response on that
    /// triple instead of writing its answer down. Both folds return zero there
    /// — one takes its "largest magnitude is zero" exit, the other its "no
    /// curvature across the structure" exit — so the number is unchanged, but a
    /// third fold that did something else would be declared correctly rather
    /// than wrongly.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        Some(self.response.evaluate([0.0, 0.0, 0.0]))
    }

    fn side_outputs(&self, volume: [usize; 3]) -> Vec<Output> {
        match self.scale_map {
            None => Vec::new(),
            // `U32` rather than `F64`: the values are scale indices, which are
            // small non-negative integers exactly representable in either, and
            // a level's worth of `f64` for a number below the scale count is
            // eight times the bytes for nothing. A side output carrying its own
            // dtype is what makes that a declaration rather than a cast.
            Some(suffix) => vec![Output::new(
                format!("{}.{suffix}", self.name),
                Dtype::U32,
                &volume,
            )],
        }
    }

    fn apply_side(
        &self,
        input: &Voxels,
        _primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        if self.scale_map.is_none() {
            return Ok(Vec::new());
        }
        let shape = input.shape();
        let dim = (shape[0], shape[1], shape[2]);
        // The second pass. See `with_scale_map` for why there is one.
        let mut response = Array3::<f64>::zeros(dim);
        let mut chosen = Array3::<f64>::zeros(dim);
        self.evaluate(input, response.view_mut(), Some(chosen.view_mut()))?;

        // The trustworthy sub-box of the buffer: a side output is written once
        // per output voxel and the halo is read, not written.
        let mut trusted = chosen.view();
        for (axis, (&start, &len)) in block
            .within
            .start
            .iter()
            .zip(block.within.shape.iter())
            .enumerate()
        {
            trusted.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
        }
        Ok(vec![trusted.to_owned().into_dyn()])
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------------- cost --

/// Measured; see [`cost_report`].
///
/// Two terms per scale, summed over the scales, because that is the shape of the
/// work: a separable smoothing costs one multiply-add per tap and there are
/// `2r + 1` taps on each of three axes, while the Hessian, the eigenvalues and
/// the response are a fixed slab per voxel that no filter width changes. A
/// single number would be right for one scale list only.
pub(super) fn cost_for(scales: &ScaleSpace) -> f64 {
    (0..scales.scales().len())
        .map(|which| {
            let taps: usize = (0..3).map(|axis| 2 * scales.radius(which, axis) + 1).sum();
            SMOOTH_COST_PER_TAP * taps as f64 + DECOMPOSITION_COST
        })
        .sum()
}

/// Measured; see [`cost_report`]. One tap of one separable pass, relative to a
/// voxelwise map.
pub(super) const SMOOTH_COST_PER_TAP: f64 = 0.79;

/// Measured; see [`cost_report`]. The Hessian, the eigenvalues and the response
/// at one voxel at one scale, relative to a voxelwise map. Large, and that is
/// the figure that matters most: a planner pricing this as a neighbourhood
/// filter would be out by more than an order of magnitude, because the work per
/// voxel is a cubic root-finder rather than a window.
pub(super) const DECOMPOSITION_COST: f64 = 41.2;

/// The measurement the two constants above came from, kept as text so that a
/// re-run somewhere else can be compared against it rather than merely replacing
/// it. `--release`, one thread, 48 x 32 x 32, best of 3, on the machine this
/// crate was developed on; the unit is the voxelwise map at 6.05 ns/voxel.
///
/// ```text
/// case                                       ns/voxel     taps   relative
/// voxelwise map (the unit)                       6.05        0       1.00
/// ridge, sigmas [1.0], truncate 3              334.06       21      55.21
/// ridge, sigmas [2.0], truncate 3              391.95       39      64.78
/// ridge, sigmas [4.0], truncate 3              577.82       75      95.50
/// ridge, sigmas [1.0, 2.0], truncate 3         776.15       60     128.28
/// ridge, sigmas [1.0, 2.0, 4.0], truncate 3   1432.04      135     236.69
/// ```
///
/// The two constants are the least-squares fit of `relative = tap * taps +
/// decomposition * scales` over all five rows — two predictors and no intercept,
/// because the model *is* the shape of the work. It reproduces the five rows to
/// within 11% at worst and 2% at the multi-scale end, which is where the
/// planner's decisions are made. `docs/design/BLOCK_OPS.md` §"Simulating
/// strategies" is the standard being met: trust "A beats B", distrust "A takes
/// 40 minutes".
pub const COST_MEASUREMENT: &str = "ops::ridge::cost_report";

/// Retake the measurement the two constants above were read off.
///
/// Same method as `super::cost`: through `BlockOp::apply`, which is the entry
/// point the executor uses; the **best** of several repetitions, because
/// contention on a shared machine is one-sided and a mean over such samples is
/// worthless; expressed relative to the voxelwise map, which is this module's
/// unit of work.
///
/// The table has one row per configuration and prints the tap count, so the two
/// constants can be read off as a straight line: the slope between two rows of
/// different radius is the per-tap cost, and the intercept is the per-voxel
/// decomposition.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let anchor = Anchor::whole(shape);
    let mut input = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    let input: Voxels = input.into();

    let response = RidgeResponse::new(0.5, 0.5, 30.0, Polarity::Ridge).unwrap();
    let mut cases: Vec<(String, Box<dyn BlockOp>, f64)> = vec![(
        "voxelwise map (the unit)".to_string(),
        Box::new(super::voxelwise::VoxelwiseMapOp::threshold(
            "map", 500.0, 1.0, 0.0,
        )),
        0.0,
    )];
    for sigmas in [
        vec![1.0f64],
        vec![2.0],
        vec![4.0],
        vec![1.0, 2.0],
        vec![1.0, 2.0, 4.0],
    ] {
        let scales = ScaleSpace::isotropic(&sigmas, 3.0, 1.0).unwrap();
        let taps: f64 = (0..scales.scales().len())
            .map(|which| {
                (0..3)
                    .map(|axis| 2 * scales.radius(which, axis) + 1)
                    .sum::<usize>() as f64
            })
            .sum();
        cases.push((
            format!("ridge, sigmas {sigmas:?}, truncate 3"),
            Box::new(RidgeFilterOp::new("ridge", scales, response)),
            taps,
        ));
    }

    let mut rows = Vec::new();
    for (name, op, taps) in cases {
        let mut out = Voxels::zeros(op.produces(input.dtype()), op.output_shape(shape)).unwrap();
        // One untimed pass: a freshly allocated output pays a page fault per
        // page on first touch, and that fault is the measurement for the
        // cheapest op here.
        op.apply(&input, &mut out, &anchor).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions.max(1) {
            let started = Instant::now();
            op.apply(&input, &mut out, &anchor).unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e9;
            std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
            best = best.min(elapsed / voxels);
        }
        rows.push((name, best, taps, op.cost_per_voxel()));
    }

    let unit = rows.first().map(|(_, nanos, _, _)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "ridge cost, {}x{}x{}, best of {repetitions}\n{:<40} {:>10} {:>8} {:>10} {:>10}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "taps", "measured", "stored"
    );
    for (name, nanos, taps, stored) in &rows {
        out.push_str(&format!(
            "{name:<40} {nanos:>10.2} {taps:>8.0} {:>10.2} {stored:>10.2}\n",
            nanos / unit
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(dim: (usize, usize, usize)) -> Array3<f64> {
        let mut array = Array3::zeros(dim);
        for (flat, value) in array.iter_mut().enumerate() {
            *value = ((flat * 7919) % 1013) as f64;
        }
        array
    }

    fn scales() -> ScaleSpace {
        ScaleSpace::isotropic(&[1.0, 2.0], 3.0, 1.0).unwrap()
    }

    fn response() -> RidgeResponse {
        RidgeResponse::new(0.5, 0.5, 5.0, Polarity::Ridge).unwrap()
    }

    // ------------------------------------------------------- smoothing --

    #[test]
    fn the_kernel_is_normalised_symmetric_and_as_wide_as_the_radius_says() {
        let weights = gaussian_weights(1.5, 3.0).unwrap();
        let radius = gaussian_radius(1.5, 3.0);
        assert_eq!(radius, 5);
        assert_eq!(weights.len(), 2 * radius + 1);
        // Exactly symmetric, bit for bit, because the weight is built from a
        // square. The second-difference stencil above it lives on cancellation.
        for step in 0..=radius {
            assert_eq!(weights[radius - step], weights[radius + step]);
        }
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // and the outermost tap is not zero, which is what makes the reach tight
        assert!(weights[0] > 0.0);
    }

    #[test]
    fn a_scale_must_be_positive_and_finite_on_every_axis() {
        assert!(ScaleSpace::new(vec![[1.0, 1.0, 0.0]], 3.0, 1.0).is_err());
        assert!(ScaleSpace::new(vec![[1.0, -1.0, 1.0]], 3.0, 1.0).is_err());
        assert!(ScaleSpace::new(vec![[1.0, f64::NAN, 1.0]], 3.0, 1.0).is_err());
        assert!(ScaleSpace::new(Vec::new(), 3.0, 1.0).is_err());
        assert!(ScaleSpace::isotropic(&[1.0], 0.0, 1.0).is_err());
        assert!(ScaleSpace::isotropic(&[1.0], 3.0, -1.0).is_err());
        assert!(ScaleSpace::isotropic(&[1.0], 3.0, 1.0).is_ok());
        // a scale so wide that the derived reach would saturate the cast it is
        // computed through, refused where it is stated rather than overflowing
        // wherever it is next used
        assert!(ScaleSpace::isotropic(&[1e30], 3.0, 1.0).is_err());
    }

    /// The two terms, separately visible: change the sigma and the reach moves
    /// with the radius; the `+ 1` is there at every sigma.
    #[test]
    fn the_reach_is_the_widest_radius_plus_the_derivative_stencil() {
        let space = ScaleSpace::new(vec![[1.0, 2.0, 0.5], [3.0, 0.5, 0.5]], 3.0, 1.0).unwrap();
        // radii: axis 0 -> ceil(3), ceil(9) = 9; axis 1 -> ceil(6), ceil(1.5) = 6;
        // axis 2 -> ceil(1.5), ceil(1.5) = 2
        assert_eq!(space.reach(0), 9 + 1);
        assert_eq!(space.reach(1), 6 + 1);
        assert_eq!(space.reach(2), 2 + 1);

        let op = RidgeFilterOp::new("ridge", space, response());
        assert_eq!(op.reach(0, 10_000), 10);
        assert_eq!(op.reach(1, 10_000), 7);
        assert_eq!(op.reach(2, 10_000), 3);
        // and it does not depend on the volume, which is what makes it a local
        // op rather than a planning barrier
        assert_eq!(op.reach(0, 12), 10);
    }

    // --------------------------------------------------------- Hessian --

    /// The stencil against the analytic answer on a field it is exact for: a
    /// quadratic. `f = 3x^2 + y^2 - 2z^2 + 5xy` has a constant Hessian, and a
    /// central second difference reproduces a quadratic exactly.
    #[test]
    fn the_stencil_reproduces_the_hessian_of_a_quadratic_exactly() {
        let dim = (9, 9, 9);
        let field = Array3::from_shape_fn(dim, |(i, j, k)| {
            let (x, y, z) = (i as f64, j as f64, k as f64);
            3.0 * x * x + y * y - 2.0 * z * z + 5.0 * x * y
        });
        let got = hessian_at(field.view(), [4, 4, 4]);
        assert_eq!(got, [6.0, 2.0, -4.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn a_constant_field_has_a_hessian_of_exactly_zero() {
        for constant in [0.0, 0.1, -7.5, 1e300, 1e-300, f64::MIN_POSITIVE] {
            let field = Array3::from_elem((5, 5, 5), constant);
            for at in [[2, 2, 2], [0, 0, 0], [4, 2, 0]] {
                assert_eq!(
                    hessian_at(field.view(), at),
                    [0.0; 6],
                    "constant {constant}, at {at:?}"
                );
            }
        }
    }

    // ----------------------------------------------------- eigenvalues --

    /// The generic case, checked against what an eigenvalue *is* rather than
    /// against a second implementation: the characteristic polynomial vanishes,
    /// the trace is the sum and the determinant is the product.
    #[test]
    fn the_eigenvalues_satisfy_the_characteristic_polynomial() {
        let cases = [
            [3.0, -1.0, 2.0, 0.5, -0.25, 0.75],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-2.0, -2.0, -9.0, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [5.0, -5.0, 0.0, 3.0, -3.0, 3.0],
            [1e6, 1.0, -1e6, 1.0, 1.0, 1.0],
        ];
        for hessian in cases {
            let [xx, yy, zz, xy, xz, yz] = hessian;
            let got = symmetric_eigenvalues(hessian);
            assert!(got.iter().all(|value| value.is_finite()), "{hessian:?}");
            assert!(
                got[0] >= got[1] && got[1] >= got[2],
                "{hessian:?} -> {got:?}"
            );

            let widest = hessian.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let trace = xx + yy + zz;
            assert!(
                (got.iter().sum::<f64>() - trace).abs() < 1e-9 * widest.max(1.0),
                "{hessian:?} -> {got:?}"
            );
            let determinant =
                xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz);
            assert!(
                (got[0] * got[1] * got[2] - determinant).abs() < 1e-8 * widest.powi(3).max(1.0),
                "{hessian:?} -> {got:?}"
            );
            for &value in &got {
                let residual = (xx - value) * ((yy - value) * (zz - value) - yz * yz)
                    - xy * (xy * (zz - value) - yz * xz)
                    + xz * (xy * yz - (yy - value) * xz);
                assert!(
                    residual.abs() < 1e-8 * widest.powi(3).max(1.0),
                    "{hessian:?} -> {got:?}, residual {residual}"
                );
            }
        }
    }

    /// **Degenerate case 1: nothing at all.** The zero matrix, exactly zero out,
    /// which is the case `constant_maps_to` rests on.
    #[test]
    fn the_zero_matrix_gives_exactly_zero_eigenvalues() {
        assert_eq!(symmetric_eigenvalues([0.0; 6]), [0.0, 0.0, 0.0]);
        assert_eq!(symmetric_eigenvalues([-0.0; 6]), [0.0, 0.0, 0.0]);
    }

    /// **Degenerate case 2: a triple root.** A multiple of the identity has one
    /// eigenvalue three times, and the `p == 0` exit must return it exactly
    /// rather than dividing by zero on the way.
    #[test]
    fn a_multiple_of_the_identity_gives_that_multiple_three_times() {
        for value in [1.0, -3.25, 1e-200, 1e200] {
            let got = symmetric_eigenvalues([value, value, value, 0.0, 0.0, 0.0]);
            assert_eq!(got, [value, value, value], "{value}");
        }
    }

    /// **Degenerate case 3: a double root, off the axes.** `diag(2, 2, 5)`
    /// rotated by 45 degrees about z. The rotation makes `det(B) / 2` land on
    /// `1` up to rounding, which is where an unclamped `acos` returns `NaN` —
    /// the failure this asserts is absent.
    #[test]
    fn a_repeated_eigenvalue_is_stable_and_never_produces_a_nan() {
        // R diag(2, 2, 5) R^T for a 45-degree rotation about z leaves the block
        // untouched, so build a less kind one: diag(5, 2, 2) rotated about x.
        let (c, s) = (
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        // A = R diag(5, 2, 2) R^T with R a rotation about the x axis by 45 deg.
        // The y-z block is 2 * I, so it is invariant; rotate about y instead so
        // the 5 mixes with a 2.
        let (a, b) = (5.0f64, 2.0f64);
        let xx = a * c * c + b * s * s;
        let zz = a * s * s + b * c * c;
        let xz = (a - b) * c * s;
        let got = symmetric_eigenvalues([xx, b, zz, 0.0, xz, 0.0]);
        assert!(got.iter().all(|value| value.is_finite()), "{got:?}");
        // The *simple* root keeps full precision; the double root keeps half of
        // it. That is the closed form's known limit, not a bug in this one — see
        // `symmetric_eigenvalues` — so the two are asserted to their own
        // standards rather than to one loose common tolerance that would hide
        // the difference.
        assert!((got[0] - 5.0).abs() < 1e-14, "{got:?}");
        let half = f64::EPSILON.sqrt();
        assert!((got[1] - 2.0).abs() < 10.0 * half, "{got:?}");
        assert!((got[2] - 2.0).abs() < 10.0 * half, "{got:?}");
        assert!(got[1] >= got[2], "still ordered: {got:?}");

        // and the endpoint itself: a matrix whose `det(B) / 2` is exactly -1.
        let got = symmetric_eigenvalues([2.0, 2.0, 5.0, 0.0, 0.0, 0.0]);
        assert!(got.iter().all(|value| value.is_finite()), "{got:?}");
        assert_eq!(got[0], 5.0);
    }

    /// **Degenerate case 4: a Hessian too small to square.** At `1e-200` the sum
    /// of squares underflows to zero, and an unscaled solver takes its
    /// triple-root exit and reports three equal eigenvalues for a matrix with
    /// three distinct ones. The scaling in `symmetric_eigenvalues` is what makes
    /// this pass, and this test is the reason it is there.
    #[test]
    fn a_near_zero_hessian_keeps_its_eigenvalues_in_proportion() {
        let base = [3.0, -1.0, 2.0, 0.5, -0.25, 0.75];
        let reference = symmetric_eigenvalues(base);
        for tiny in [1e-100, 1e-160, 1e-200, 1e-250] {
            let scaled = base.map(|value| value * tiny);
            let got = symmetric_eigenvalues(scaled);
            for axis in 0..3 {
                let want = reference[axis] * tiny;
                assert!(
                    (got[axis] - want).abs() <= 1e-12 * want.abs(),
                    "at {tiny}: {got:?} is not {reference:?} scaled"
                );
            }
            // the failure mode the scaling removes: three equal eigenvalues
            assert!(got[0] != got[1] && got[1] != got[2], "at {tiny}: {got:?}");
        }
    }

    /// A non-finite Hessian has no decomposition to report, and reports zero
    /// rather than putting a `NaN` in the output volume.
    #[test]
    fn a_non_finite_hessian_yields_no_structure_rather_than_a_nan() {
        assert_eq!(
            symmetric_eigenvalues([f64::NAN, 1.0, 1.0, 0.0, 0.0, 0.0]),
            [0.0; 3]
        );
        assert_eq!(
            symmetric_eigenvalues([f64::INFINITY, 1.0, 1.0, 0.0, 0.0, 0.0]),
            [0.0; 3]
        );
    }

    // -------------------------------------------------------- response --

    /// What the filter is *for*, stated on eigenvalues directly: a line beats a
    /// sheet, a line beats a blob, and the wrong polarity is nothing at all.
    #[test]
    fn the_response_ranks_a_line_above_a_sheet_and_a_blob() {
        let response = RidgeResponse::new(0.5, 0.5, 1.0, Polarity::Ridge).unwrap();
        // Descending algebraic order, as `symmetric_eigenvalues` returns.
        let line = response.evaluate([0.0, -10.0, -10.0]);
        let sheet = response.evaluate([0.0, 0.0, -10.0]);
        let blob = response.evaluate([-10.0, -10.0, -10.0]);
        assert!(line > sheet, "line {line}, sheet {sheet}");
        assert!(line > blob, "line {line}, blob {blob}");
        assert!(
            line > 0.5,
            "a clean line should be a strong response, got {line}"
        );

        // the other polarity of the same structure is not this structure
        assert_eq!(response.evaluate([10.0, 10.0, 0.0]), 0.0);
        let valley = RidgeResponse::new(0.5, 0.5, 1.0, Polarity::Valley).unwrap();
        assert_eq!(valley.evaluate([10.0, 10.0, 0.0]), line);
    }

    #[test]
    fn a_zero_hessian_is_zero_response_and_not_a_nan() {
        let response = response();
        assert_eq!(response.evaluate([0.0, 0.0, 0.0]), 0.0);
        // and the one-non-zero case, where the flatness ratio is 0 / x
        assert_eq!(response.evaluate([0.0, 0.0, -3.0]), 0.0);
    }

    #[test]
    fn the_sensitivities_must_be_positive_and_finite() {
        assert!(RidgeResponse::new(0.0, 0.5, 1.0, Polarity::Ridge).is_err());
        assert!(RidgeResponse::new(0.5, -0.5, 1.0, Polarity::Ridge).is_err());
        assert!(RidgeResponse::new(0.5, 0.5, f64::INFINITY, Polarity::Ridge).is_err());
    }

    // -------------------------------------------------- ratio response --

    fn ratio() -> RatioResponse {
        RatioResponse::new(0.5, 1.0, 0.25, Polarity::Ridge).unwrap()
    }

    #[test]
    fn the_ratio_weights_must_be_finite_and_not_negative() {
        assert!(RatioResponse::new(-1.0, 1.0, 0.25, Polarity::Ridge).is_err());
        assert!(RatioResponse::new(0.5, -1.0, 0.25, Polarity::Ridge).is_err());
        assert!(RatioResponse::new(0.5, 1.0, -0.25, Polarity::Ridge).is_err());
        assert!(RatioResponse::new(f64::NAN, 1.0, 0.25, Polarity::Ridge).is_err());
        // zero is legitimate for all three and is a meaningful setting: it turns
        // the term it weights off
        assert!(RatioResponse::new(0.0, 0.0, 0.0, Polarity::Ridge).is_ok());
    }

    /// The same ranking the first response is asked for, over the same triples,
    /// so that the two are known to be answering the same *question* even though
    /// they are not the same function.
    #[test]
    fn the_ratio_response_also_ranks_a_line_above_a_sheet_and_a_blob() {
        let response = RatioResponse::new(1.0, 1.0, 0.25, Polarity::Ridge).unwrap();
        let line = response.evaluate([0.0, -10.0, -10.0]);
        let sheet = response.evaluate([0.0, 0.0, -10.0]);
        let blob = response.evaluate([-10.0, -10.0, -10.0]);
        assert!(line > sheet, "line {line}, sheet {sheet}");
        assert!(line > blob, "line {line}, blob {blob}");
        assert_eq!(sheet, 0.0, "a sheet has no second negative curvature");
        assert_eq!(
            blob, 0.0,
            "curvature along the structure cancels it exactly"
        );

        // the other polarity of the same structure is not this structure...
        assert_eq!(response.evaluate([10.0, 10.0, 0.0]), 0.0);
        // ...and is exactly this one on the negated triple
        let valley = RatioResponse::new(1.0, 1.0, 0.25, Polarity::Valley).unwrap();
        assert_eq!(valley.evaluate([10.0, 10.0, 0.0]), line);
    }

    /// The two exits that keep a division or a fractional power from producing
    /// a `NaN`, and the one that is a modelling decision rather than a guard.
    #[test]
    fn the_ratio_response_has_no_nan_and_cuts_a_dominating_opposed_curvature() {
        let response = RatioResponse::new(0.5, 0.5, 0.25, Polarity::Ridge).unwrap();
        assert_eq!(response.evaluate([0.0, 0.0, 0.0]), 0.0);
        assert_eq!(response.evaluate([0.0, 0.0, -3.0]), 0.0);
        // opposed * e1 / |e2| >= 1 cuts the response to zero rather than raising
        // a negative base to a fractional power
        assert_eq!(response.evaluate([40.0, -1.0, -10.0]), 0.0);
        // and just short of the cut it is positive, and less than the same
        // structure with no opposing curvature at all
        let just_short = response.evaluate([3.0, -1.0, -10.0]);
        let unopposed = response.evaluate([0.0, -1.0, -10.0]);
        assert!(
            just_short > 0.0 && just_short < unopposed,
            "{just_short} against {unopposed}"
        );
    }

    /// The response is **unbounded**, which is the property no setting of
    /// [`RidgeResponse`] has: it carries the largest curvature's magnitude.
    #[test]
    fn the_ratio_response_carries_the_magnitude_rather_than_saturating_it() {
        let response = RatioResponse::new(1.0, 1.0, 0.25, Polarity::Ridge).unwrap();
        let weak = response.evaluate([0.0, -1.0, -1.0]);
        let strong = response.evaluate([0.0, -1000.0, -1000.0]);
        assert_eq!(weak, 1.0);
        assert_eq!(strong, 1000.0);
        for strength in [0.01f64, 1.0, 1e6] {
            let existing = RidgeResponse::new(0.5, 0.5, strength, Polarity::Ridge).unwrap();
            assert!(existing.evaluate([0.0, -1000.0, -1000.0]) <= 1.0);
        }
    }

    /// A constant block is exactly zero for this fold too, and the op's
    /// declaration is the fold's own answer rather than a number written down
    /// beside it.
    #[test]
    fn a_constant_block_is_exactly_zero_for_the_ratio_response_as_well() {
        let op = RidgeFilterOp::new("ratio", scales(), ratio());
        for constant in [0.0, 0.1, -7.5, 1e12] {
            let input: Voxels = Array3::from_elem((9, 8, 7), constant).into();
            let mut out = Voxels::zeros(Dtype::F64, [9, 8, 7]).unwrap();
            op.apply(&input, &mut out, &Anchor::whole([9, 8, 7]))
                .unwrap();
            assert!(out.view::<f64>().unwrap().iter().all(|&value| value == 0.0));
            assert_eq!(op.constant_maps_to(constant), Some(0.0));
        }
    }

    /// The op takes either fold, and the choice changes the answer rather than
    /// merely the type: on the same volume the two produce different numbers.
    #[test]
    fn the_filter_takes_either_fold_and_the_two_do_not_agree() {
        let dim = (5usize, 15, 15);
        let field = Array3::from_shape_fn(dim, |(_, j, k)| {
            let (dy, dz) = (j as f64 - 7.0, k as f64 - 7.0);
            if dy * dy + dz * dz <= 4.0 {
                1.0
            } else {
                0.0
            }
        });
        let run = |op: RidgeFilterOp| {
            let input: Voxels = field.clone().into();
            let mut out = Voxels::zeros(Dtype::F64, [dim.0, dim.1, dim.2]).unwrap();
            op.apply(&input, &mut out, &Anchor::whole([dim.0, dim.1, dim.2]))
                .unwrap();
            out.view::<f64>().unwrap().to_owned()
        };
        let saturating = run(RidgeFilterOp::new("saturating", scales(), response()));
        let ratios = run(RidgeFilterOp::new("ratio", scales(), ratio()));
        assert!(saturating.iter().all(|&value| value <= 1.0));
        assert!(
            ratios.iter().any(|&value| value > 0.0),
            "the ratio fold answered nothing at all"
        );
        assert_ne!(saturating, ratios);
        // both find the structure, which is what makes them two answers to one
        // question rather than one answer and one accident
        assert!(saturating[[2, 7, 7]] > saturating[[2, 1, 1]]);
        assert!(ratios[[2, 7, 7]] > ratios[[2, 1, 1]]);
    }

    // ---------------------------------------------------------- kernel --

    /// **The claim `constant_maps_to` makes, checked in bits.**
    ///
    /// Not "close to zero": `== 0.0`, over a range of constants including ones
    /// whose Gaussian-weighted sum is emphatically not themselves. This is the
    /// test `local.rs`'s mean would fail, and the reason this op may declare what
    /// the mean may not.
    #[test]
    fn a_constant_block_is_exactly_zero_and_not_nearly_zero() {
        let op = RidgeFilterOp::new("ridge", scales(), response());
        for constant in [0.0, 0.1, 1.0, -7.5, 1e12, 1e-12, 1e300] {
            let input: Voxels = Array3::from_elem((11, 9, 8), constant).into();
            let mut out = Voxels::zeros(Dtype::F64, [11, 9, 8]).unwrap();
            op.apply(&input, &mut out, &Anchor::whole([11, 9, 8]))
                .unwrap();
            let view = out.view::<f64>().unwrap();
            assert!(
                view.iter().all(|&value| value == 0.0),
                "constant {constant}: got {:?}",
                view.iter().find(|&&value| value != 0.0)
            );
            assert_eq!(op.constant_maps_to(constant), Some(0.0));
        }
    }

    /// The kernel on a synthetic line: a bright cylinder along one axis must
    /// respond more strongly than the background it sits in.
    #[test]
    fn a_bright_line_responds_more_than_its_background() {
        let dim = (5usize, 21usize, 21usize);
        let field = Array3::from_shape_fn(dim, |(_, j, k)| {
            let (dy, dz) = (j as f64 - 10.0, k as f64 - 10.0);
            if dy * dy + dz * dz <= 4.0 {
                1.0
            } else {
                0.0
            }
        });
        let space = ScaleSpace::isotropic(&[1.0, 2.0], 3.0, 1.0).unwrap();
        let response = RidgeResponse::new(0.5, 0.5, 0.05, Polarity::Ridge).unwrap();
        let mut out = Array3::zeros(dim);
        ridge_response_into(field.view(), &space, &response, out.view_mut(), None).unwrap();
        let centre = out[[2, 10, 10]];
        let corner = out[[2, 1, 1]];
        assert!(
            centre > 10.0 * corner.max(1e-9),
            "on the line {centre}, off it {corner}"
        );
    }

    /// The scale map records the argmax, and the argmax is a real choice: a
    /// narrow structure and a wide one in the same volume must not both be won
    /// by the same scale.
    #[test]
    fn the_scale_map_records_which_scale_won() {
        let dim = (5usize, 41usize, 21usize);
        let field = Array3::from_shape_fn(dim, |(_, j, k)| {
            let dz = k as f64 - 10.0;
            let narrow = (j as f64 - 10.0).powi(2) + dz * dz <= 1.0;
            let wide = (j as f64 - 30.0).powi(2) + dz * dz <= 16.0;
            if narrow || wide {
                1.0
            } else {
                0.0
            }
        });
        let space = ScaleSpace::isotropic(&[0.8, 3.0], 3.0, 1.0).unwrap();
        let response = RidgeResponse::new(0.5, 0.5, 0.02, Polarity::Ridge).unwrap();
        let mut out = Array3::zeros(dim);
        let mut chosen = Array3::zeros(dim);
        ridge_response_into(
            field.view(),
            &space,
            &response,
            out.view_mut(),
            Some(chosen.view_mut()),
        )
        .unwrap();
        assert_eq!(
            chosen[[2, 10, 10]],
            0.0,
            "the narrow line wants the small scale"
        );
        assert_eq!(
            chosen[[2, 30, 10]],
            1.0,
            "the wide line wants the large scale"
        );
    }

    /// The kernel is generic, and the shell bridges every type a buffer holds.
    /// A `u8` volume reaches it with no copy; `bool` and the wide integers reach
    /// it through the widening the shell chooses.
    #[test]
    fn every_element_type_a_buffer_holds_reaches_the_kernel() {
        let op = RidgeFilterOp::new("ridge", scales(), response());
        let dim = [9usize, 8, 7];
        let reference = {
            let input: Voxels = ramp((dim[0], dim[1], dim[2]))
                .mapv(|v| (v as u8) as f64)
                .into();
            let mut out = Voxels::zeros(Dtype::F64, dim).unwrap();
            op.apply(&input, &mut out, &Anchor::whole(dim)).unwrap();
            out.view::<f64>().unwrap().to_owned()
        };
        let narrowed = ramp((dim[0], dim[1], dim[2])).mapv(|v| v as u8);
        let input: Voxels = narrowed.into();
        let mut out = Voxels::zeros(Dtype::F64, dim).unwrap();
        op.apply(&input, &mut out, &Anchor::whole(dim)).unwrap();
        assert_eq!(out.view::<f64>().unwrap().to_owned(), reference);

        // the widened group, which must at least run and produce finite values
        for input in [
            Voxels::from(ramp((dim[0], dim[1], dim[2])).mapv(|v| v as u64)),
            Voxels::from(ramp((dim[0], dim[1], dim[2])).mapv(|v| v as i64)),
            Voxels::from(ramp((dim[0], dim[1], dim[2])).mapv(|v| v > 500.0)),
        ] {
            assert!(op.accepts(input.dtype()));
            let mut out = Voxels::zeros(Dtype::F64, dim).unwrap();
            op.apply(&input, &mut out, &Anchor::whole(dim)).unwrap();
            assert!(out.view::<f64>().unwrap().iter().all(|v| v.is_finite()));
        }
        assert!(!op.accepts(Dtype::F16));
        assert_eq!(op.produces(Dtype::U8), Dtype::F64);
    }

    /// The stored cost keeps the ordering the measurement found: this op is far
    /// more expensive per voxel than anything else in the module, a wider scale
    /// costs more than a narrow one, and a second scale costs more than one.
    #[test]
    fn the_stored_cost_keeps_the_order_the_measurement_found() {
        let narrow = RidgeFilterOp::new(
            "narrow",
            ScaleSpace::isotropic(&[1.0], 3.0, 1.0).unwrap(),
            response(),
        );
        let wide = RidgeFilterOp::new(
            "wide",
            ScaleSpace::isotropic(&[4.0], 3.0, 1.0).unwrap(),
            response(),
        );
        let both = RidgeFilterOp::new(
            "both",
            ScaleSpace::isotropic(&[1.0, 4.0], 3.0, 1.0).unwrap(),
            response(),
        );
        assert!(wide.cost_per_voxel() > narrow.cost_per_voxel());
        assert!(both.cost_per_voxel() > wide.cost_per_voxel());
        assert!(
            (both.cost_per_voxel() - narrow.cost_per_voxel() - wide.cost_per_voxel()).abs() < 1e-9,
            "the scales are independent passes and the cost is their sum"
        );

        let map = super::super::voxelwise::VoxelwiseMapOp::new("map", |value| value);
        assert!(narrow.cost_per_voxel() > 10.0 * map.cost_per_voxel());
    }

    /// Retaking the measurement. Ignored because timing in a test suite measures
    /// the machine's mood, not the code — but it is here and it is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::ridge
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([48, 32, 32], 3));
    }
}
