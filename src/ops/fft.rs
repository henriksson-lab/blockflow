// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the discrete
// transform and of the correlation theorem, not adapted from any
// implementation of either.
//
// A discrete Fourier transform of a real plane, and the correlation it exists
// to accelerate.
//
// What is here, in the order it is built
// --------------------------------------
// | item | what it is |
// |---|---|
// | [`RealTransform2`] | the transform: a real `rows x cols` plane to a half-spectrum and back, with the plans built once and reused |
// | [`TransformBackend`] | which library does that transform. One variant by default; the `fftw` feature adds a second and this file's last section says what it is worth |
// | [`Correlation2`] | `C(k) = sum_x a[x] b[x-k]` over a window of integer lags, through the correlation theorem |
// | [`SquaredDifference`] | the landscape this was built for: a mean squared difference between two planes at every lag, normalised by the overlapping element count |
// | [`correlate_direct`], [`squared_difference_direct`] | the same two answers computed by walking every lag. The oracle, and the reason the two above can be trusted |
//
// Why this is not a `BlockOp`, and not a decomposable op of any kind
// -----------------------------------------------------------------
// A Fourier coefficient is a sum over **every** element of its input. There is
// no halo that makes one, no block-local form that approaches one, and no
// [`crate::reach::Reach`] short of [`crate::reach::AxisReach::All`] that
// describes one honestly. `ops::deconvolve`'s header says exactly this about
// the frequency-domain form it declined to build, and it is still true: this is
// a **resident kernel**, and the crate says so here rather than dressing it in a
// lattice it cannot honour.
//
// [`crate::ops::watershed`] shows what declaring the barrier looks like when the
// op's *shape* still fits — reach `All`, one block, and the memory cost written
// down. This one does not fit even that, and there are three independent reasons
// rather than one:
//
// 1. **Two inputs of different extents.** A `BlockOp` is handed one
//    [`crate::voxels::Voxels`] and writes one. The landscape is a function of a
//    *pair* of planes, and their shapes need not agree.
// 2. **An output space that is not the input space.** The landscape is indexed
//    by *lag*, and its extent comes from [`ShiftWindow`] rather than from either
//    input. `output_shape` can rescale an axis; it cannot replace the coordinate
//    system with a different one.
// 3. **No element type to hold a spectrum.** `Voxels` has eleven variants and
//    none of them is complex. The intermediate this file's whole method rests on
//    cannot be named in the pipeline's own algebra.
//
// So these are free functions and plain structs, which is the shape
// [`crate::ops::fill`]'s fragment-and-join and [`crate::ops::tabulate`] already
// establish as acceptable for work the block lattice cannot express. Nothing
// here implements `BlockOp`, and the absence is the statement.
//
// **What that absence is not evidence of, and this has been measured since.**
// It says these three *types* cannot be a phase; it was read for a while as
// saying the frequency domain cannot be one, and that reading is wrong. Only
// reason 3 above is about the element type, and it is the weakest of the three:
// an operation whose *answer* is real needs no complex image, because the
// spectrum lives inside one `apply` and dies there.
// [`crate::ops::convolve::TransformConvolveOp`] is exactly that — a linear
// filter through this module's transform, an ordinary `BlockOp` with an
// ordinary bounded reach, byte-identical across lattices — and it is built on
// [`RealTransform3`] below. The ops survey's G3 row carries the argument and the
// three findings against adding a `Dtype::Complex*` at all.
//
// Composition, and why there is no `rayon` inside
// -----------------------------------------------
// The consumer's parallelism is **across planes**: one landscape per plane of a
// stack, and the planes are independent. Every type here is `Send` and `Clone`,
// and cloning a plan clones an `Arc` of the twiddles and allocates fresh
// scratch, so `rayon`'s `map_init(|| plan.clone(), ..)` gives one working set
// per thread over one set of shared plans. Parallelising *within* a transform
// instead would take the same cores and win less, so this module does not.
//
// The normalisation convention, stated once
// -----------------------------------------
// **The forward transform is unnormalised and the inverse carries `1/N`**, with
// `N = rows * cols` the element count of the padded plane. That is NumPy's
// convention and SciPy's, and it is the one under which
// `inverse(forward(x)) == x` rather than `N x`.
//
// The round trip is **not** the identity to the bit and this file does not
// pretend otherwise: it is the identity to a bound. Over a `128 x 1350` plane of
// values in `[0, 1)` the measured worst absolute deviation is `1.4e-15`, and
// over `192 x 2700` it is `1.6e-15` — a few units in the last place of the input
// range, growing like `sqrt(log N)`. [`RealTransform2`]'s tests assert the round
// trip against a tolerance and state the achieved figure; an assertion of exact
// equality there would be a false claim that happens to pass on one machine.
//
// One consequence of the real-input path is worth naming because it looks like a
// fudge and is not. `realfft`'s complex-to-real transform **refuses** an input
// whose zero and Nyquist bins have a non-zero imaginary part, and after the
// inverse pass along the first axis those bins carry rounding noise of order
// `1e-17` where the exact value is real. [`RealTransform2::inverse`] therefore
// clears exactly those two imaginary parts per row before handing the row over.
// That is not discarding information; it is asserting a symmetry the exact
// answer has and the arithmetic lost.
//
// **`f64` and not `f32`, and this is a correctness decision before it is a speed
// one.** `f32` is `1.55x` faster here — the same `128 x 1350` forward, `0.60 ms`
// against `0.93 ms` — and its round trip deviates by `3.0e-7` against `f64`'s
// `8.9e-16`, which is **eight orders of magnitude** worse and five orders above
// the `1e-12` agreement this file is accepted against. There is no version of the
// acceptance test an `f32` landscape passes, so it is not offered as a choice: an
// axis whose other setting cannot meet the bar is not an axis, it is a trap. A
// caller with `f32` data widens it and gets the `f64` answer.
//
// Circular, linear, and the padding rule
// --------------------------------------
// A transform gives a **circular** correlation. The consumer wants a **linear**
// one — no contribution may wrap from the far end of the plane — and the
// difference is invisible in the answer: wrap-around produces a smooth, entirely
// plausible landscape with its minimum in the wrong place.
//
// Write `A` and `B` for the two extents on one axis and `[lo, hi]` for the lags
// the caller asks for. The circular result at index `j` is the sum of the linear
// result over **every** lag congruent to `j`:
//
// ```text
// Ccirc[j] = sum_m C(j + m N)
// ```
//
// and `C` is non-zero only for `-(B-1) <= k <= A-1`. So a read lag `k` is
// uncontaminated exactly when `k + N` is past the top of that range and `k - N`
// is below its bottom, for every `k` in `[lo, hi]`:
//
// ```text
// N >= A - lo    and    N >= B + hi
// ```
//
// together with `N >= A` and `N >= B`, which is only a constraint when the
// window excludes zero. [`minimal_wrap_free_length`] is those four maxima and
// nothing else, and [`Padding::Minimal`] uses it directly.
//
// **This is sharper than "pad to the sum of the extents", and reduces to it.**
// Asking for the whole non-zero range — `lo = -(B-1)`, `hi = A-1` — gives
// `N >= A + B - 1`, the familiar rule. Asking for a narrow window of lags, which
// is what a search over a bounded displacement does, gives very much less: two
// `1304`-long extents with lags in `[-30, 30]` need `1334`, not `2607`.
//
// **The two constraints are one-sided, so where the window *sits* matters as much
// as how wide it is — and an off-centre window is the normal case, not the odd
// one.** `A - lo` is bound by the bottom of the window and `B + hi` by its top,
// and neither is a function of the width alone. Two windows of width `61` over
// two `96`-long extents:
//
// | window | `max(A, B, A - lo, B + hi)` | [`Padding::Smooth`] |
// |---|---|---|
// | `[-30, 30]`, centred | `126` | `128` |
// | `[0, 60]`, off-centre | **`156`** | **`160`** |
//
// The worked example below is the centred one because it is easier to read, and
// this table is here so that nobody mistakes it for the general shape. The first
// consumer of this module has an off-centre row window of exactly `[0, 60]` — its
// two cuts sit `30` apart in global coordinates, so the lags it wants are all
// non-negative — and pads to `160 x 1350` rather than `128 x 1350`. Exploiting
// that is the whole value of this rule over the sum-of-extents one: a rule that
// only knew the width would have to assume the worst side and pad to `160`
// either way.
//
// **And any longer length is also correct**, which is what makes
// [`Padding::Smooth`] free. Rounding each axis up to the next `5`-smooth integer
// — `2^a 3^b 5^c`, the sizes a mixed-radix transform has dedicated kernels for —
// costs a few percent more elements and buys back much more than that:
//
// | padded shape | how it arose | forward `r2c` |
// |---|---|---|
// | `157 x 1335` | the geometry the consumer starts from; `157` is prime and `1335 = 3 . 5 . 89` | `6.41 ms` |
// | `126 x 1334` | [`Padding::Minimal`] for that geometry; `1334 = 2 . 23 . 29` | `2.46 ms` |
// | `128 x 1350` | [`Padding::Smooth`] of the same; `2^7` and `2 . 3^3 . 5^2` | **`1.09 ms`** |
// | `192 x 2700` | [`Padding::Smooth`] of the sum-of-extents rule | `3.37 ms` |
//
// Measured on the machine this was written on, `--release`, one thread, best of
// 50. The last two rows are the price of the coarser padding rule; the first two
// are the price of a hostile length. Choosing the length well is worth **5.9x**
// here, which is more than any available choice of transform library is worth,
// and it is why `Padding::Smooth` is the default.
//
// It survives end to end. A whole [`SquaredDifference::landscape`] over two
// `96 x 1304` planes and a `61 x 61` lag window, best of 20:
//
// | padding | padded shape | per landscape |
// |---|---|---|
// | [`Padding::Smooth`] | `128 x 1350` | **`4.1 ms`** |
// | [`Padding::Minimal`] | `126 x 1334` | `8.2 ms` |
// | `Padding::Exact([157, 1335])` | `157 x 1335` | `18.9 ms` |
//
// against `472 ms` for [`squared_difference_direct`] on the same input — **38x**,
// which is the whole reason this file exists. `tests/fft_correlation.rs`'s
// `the_speed_levers_are_measured` is that table and is runnable, which matters
// because these are best-of-20 on a shared machine and move by a few percent
// between runs. As with the rest of `ops`, trust the **ratios**.
//
// Two other levers, measured in the same test. **Plan reuse is worth `1.5x`**
// (`3.8 ms` reused against `6.0 ms` re-planned per landscape), which is why every
// type here is a plan the caller holds rather than a function that builds one.
// **Threads over independent plane pairs scale `6.4x` on eight**, from 265 to
// 1692 landscapes per second, with one cloned plan each — and that is the
// parallelism to reach for, not a parallel transform.
//
// [`Padding::Exact`] exists so a caller can pin a length, and so a test can pad
// *short* and watch the wrap-around arrive. [`Correlation2::wraps`] reports
// which side of the rule a plan is on rather than leaving it to be inferred.
//
// The three terms of the squared difference, and their supports
// -------------------------------------------------------------
// The quantity [`SquaredDifference`] produces at lag `k` is
//
// ```text
// D(k) = ( sum_{x in O(k)} (a[x] - b[x-k])^2 ) / |O(k)|
// ```
//
// where `O(k)` is the overlap — the `x` that are inside `a` and whose `x - k` is
// inside `b`. Expanding the square gives three terms, and **they are not
// computed the same way, because they should not be**:
//
// | term | support | how |
// |---|---|---|
// | `sum_{O(k)} a[x]^2` | a rectangle in `a`'s frame, `[max(0,k), min(A, B+k))` per axis | **exact rectangle sums**, not a transform |
// | `sum_{O(k)} b[x-k]^2` | a rectangle in `b`'s frame, `[max(0,-k), min(B, A-k))` per axis | **exact rectangle sums**, not a transform |
// | `sum_{O(k)} a[x] b[x-k]` | the overlap itself | [`Correlation2`] — the one term that genuinely needs the theorem |
// | `\|O(k)\|` | the overlap's size | a **product of per-axis overlap lengths**. An exact integer, geometry only |
//
// Every one of the four is over the overlap and only the overlap, and that is
// the sentence a transcription of this gets wrong. The two energy sums are
// **not** sums over the whole plane: they shrink as the lag pushes the planes
// apart, they are what makes `D` a mean rather than a total, and using the whole
// plane's energy instead produces a landscape that is smooth, wrong, and
// minimised in the wrong place.
//
// Computing them as rectangle sums rather than as two more correlations is worth
// saying twice, because the all-transform form is the one an implementation of
// this usually reaches for. It costs **two forward transforms and one inverse**
// per plane pair instead of four and one, and — this matters more — it makes
// three of the four terms **exact up to a single rounding of the accumulator**,
// so the only floating-point error in `D` is the one the cross-correlation
// carries. The rectangle sums go through a compensated (Neumaier) prefix along
// each row, which holds the prefix to a rounding of its own value however long
// the row is, and then a plain sum down the rows.
//
// `|O(k)|` being an exact integer rather than the modulus of an inverse
// transform removes the other place this arithmetic usually loses precision, and
// with it the need for a floor to keep a near-zero divisor from exploding: an
// empty overlap is `0` exactly, and [`SquaredDifference`] reports `INFINITY`
// there and says so, rather than dividing by an epsilon and returning a finite
// number that means nothing.
//
// What the caller should do about the mean
// ----------------------------------------
// `D(k) = Ea(k) + Eb(k) - 2 C(k)`, and if `a` and `b` are non-negative with a
// large common mean, all three terms are large and their combination is small.
// That is cancellation, and no choice of transform fixes it. **Centre the data**
// — subtract each plane's mean before handing it over — and the terms become the
// same size as their difference. This module does not do it for the caller,
// because whether the two planes should be centred together or apart, and
// whether they should also be scaled, is a question about the data rather than
// about the arithmetic. It is measured rather than asserted: on the fixtures in
// `tests/fft_correlation.rs` the agreement against the direct walk is `5.3e-16`
// relative for centred data and `7.1e-12` for the same data with a mean of `50`
// added, which is the whole of the difference this paragraph is about.
//
// The backend, and the feature that swaps it
// -------------------------------------------
// Everything above is arithmetic this file owns. The one thing it does not own
// is the transform itself, and there is a [`TransformBackend`] for that:
// [`TransformBackend::Portable`] is `rustfft` + `realfft` and is always
// compiled, and the crate's optional `fftw` feature adds
// [`TransformBackend::Fftw`], the system's FFTW 3, and makes *it* the default.
// [`RealTransform2::with_backend`], [`Correlation2::with_backend`] and
// [`SquaredDifference::with_backend`] name one explicitly; everything else takes
// [`TransformBackend::default`].
//
// **With the feature off there is one variant and nothing else changes**: the
// package graph is the same 29, no build script appears, and
// `tests/fft_correlation.rs` prints the same agreement figures to the last
// digit. With it on, **both** backends are compiled, which is what lets that
// file measure the agreement *between* them in one process rather than across
// two runs of it.
//
// **What the second backend is worth, and it is worth nothing here.** Same
// protocol as the tables above — `--release`, one thread, best of 50 forward,
// best of 20 landscape, `the_backends_are_measured_against_each_other`, best
// over seven repetitions:
//
// | | forward `128 x 1350` | forward `157 x 1335` | a whole landscape |
// |---|---|---|---|
// | portable | **`0.816 ms`** | **`5.851 ms`** | **`4.097 ms`** |
// | FFTW | `0.841 ms` | `6.834 ms` | `4.262 ms` |
// | | 0.97x | 0.86x | **0.96x** |
//
// **FFTW is behind at every geometry measured**, including the one it was
// expected to win. On a *loaded* machine the first column reads `1.02 ms`
// against `0.87 ms` and FFTW looks `1.17x` ahead, which is what the earlier
// estimate of this was built on; with the machine quiet the portable path drops
// to `0.816 ms` and FFTW does not follow. Ratios taken while eight other builds
// are running are not ratios.
//
// The reason the faster library loses is in `Cargo.toml` and it is structural:
// an FFTW plan may only be executed against buffers of the alignment it was
// planned with, so the half-spectrum is copied between the caller's [`Spectrum`]
// and an `fftw_alloc` buffer three times per landscape at `0.118 ms` each —
// `0.72 ms` of raw transform plus `0.118 ms` of copy is the `0.841 ms` above.
//
// Set against the `5.9x` that choosing the padded length well is worth and the
// `6.4x` that threads over plane pairs are worth — both of which the FFTW
// backend keeps and neither of which it improves — **this feature does not earn
// its place, and the recommendation that ships with it is to leave it off.** It
// is here, properly tested, for a consumer who measures their own geometry and
// finds otherwise; the measurement is one `cargo test` away, and it is the same
// test either way.
//
// Two things that do **not** change with the backend, because they were settled
// on evidence that has nothing to do with the library. The padding rule is one:
// `N >= max(A, B, A - lo, B + hi)` rounded up to a 5-smooth length, whichever
// transform runs. `f64` is the other — the `f32` round trip is `3.0e-7` against
// `f64`'s `8.9e-16`, five orders above the acceptance bar, and FFTW having a
// single-precision half is not a reason to revisit a decision that was never
// about speed.
//
// The dependency
// --------------
// `rustfft` and `realfft`, and `Cargo.toml` records what was measured against
// what before they were chosen — including FFTW, which is where the `fftw`
// feature's `fftw-sys` line and its `system`-not-`source` flag are argued.

use std::sync::Arc;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};

/// The complex scalar the spectra are made of, re-exported so a caller need not
/// depend on `rustfft` to name one.
pub use rustfft::num_complex::Complex;

/// The half-spectrum of a real `rows x cols` plane: `[rows, cols / 2 + 1]`.
///
/// Only half the coefficients are stored along the last axis, because a real
/// input's spectrum is conjugate-symmetric and the other half is redundant. The
/// **first** axis is stored in full — the symmetry relates `S[r][c]` to
/// `S[-r][-c]`, not to anything within one row.
pub type Spectrum = Array2<Complex<f64>>;

/// How many columns of a `cols`-wide real transform are stored.
pub fn spectrum_width(cols: usize) -> usize {
    cols / 2 + 1
}

/// The smallest `5`-smooth integer — `2^a 3^b 5^c` — that is at least `least`.
///
/// The sizes a mixed-radix transform has dedicated kernels for. A prime length
/// falls back to Bluestein's algorithm, which is correct and several times
/// slower; see this module's header for the measurement. `0` maps to `1`.
pub fn next_smooth_length(least: usize) -> usize {
    let least = least.max(1);
    // A power of two is 5-smooth and there is always one in `[least, 2 least)`,
    // so this is a valid answer to improve on rather than a guess.
    let mut best = least.next_power_of_two();
    let mut five = 1usize;
    while five <= best {
        let mut three = five;
        while three <= best {
            let mut candidate = three;
            while candidate < least {
                match candidate.checked_mul(2) {
                    Some(next) => candidate = next,
                    None => break,
                }
            }
            if candidate >= least && candidate < best {
                best = candidate;
            }
            match three.checked_mul(3) {
                Some(next) => three = next,
                None => break,
            }
        }
        match five.checked_mul(5) {
            Some(next) => five = next,
            None => break,
        }
    }
    best
}

// ------------------------------------------------------------- the transform --

/// How many columns are gathered into contiguous scratch at a time when
/// transforming along the first axis.
///
/// The first-axis lanes are strided by a whole row, so gathering one at a time
/// touches a cache line per element and uses sixteen bytes of it. Gathering a
/// block of them uses the whole line. Measured at `128 x 1350`: `1` lane
/// `1.45 ms`, `4` lanes `1.18 ms`, **`8` lanes `1.09 ms`**, `16` lanes
/// `1.16 ms`, `32` lanes `1.26 ms`.
const LANE_BLOCK: usize = 8;

/// Which arithmetic a [`RealTransform2`] runs.
///
/// [`Portable`](Self::Portable) is `rustfft` and `realfft`, is always compiled,
/// and is what every number in this module's header was measured against.
/// `Fftw` exists only when the crate's `fftw` feature is on, links the system's
/// FFTW 3, and is [`Default`] when it is on — swapping the default is the whole
/// of what the feature does. It is a plain code span and not a link because the
/// variant is not there to link to in a build without the feature.
///
/// **The enum has one variant in the default build**, and that is the point
/// rather than an oversight to be tidied away: it lets a test say "run this bar
/// over every backend this build has" and get exactly the old behaviour when
/// there is only one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransformBackend {
    /// `rustfft` and `realfft`: pure Rust, no C toolchain, no build script.
    #[default]
    Portable,
    /// The system's FFTW 3. The `fftw` feature only.
    #[cfg(feature = "fftw")]
    Fftw,
}

#[cfg(not(feature = "fftw"))]
#[cfg(feature = "fftw")]
impl Default for TransformBackend {
    fn default() -> Self {
        Self::Fftw
    }
}

impl TransformBackend {
    /// Every backend this build has, portable first and in a stable order.
    ///
    /// A test that runs its bar over this runs it over one backend by default
    /// and over both with the feature on, without naming either.
    #[cfg(not(feature = "fftw"))]
    pub fn available() -> &'static [TransformBackend] {
        &[TransformBackend::Portable]
    }

    /// Every backend this build has, portable first and in a stable order.
    #[cfg(feature = "fftw")]
    pub fn available() -> &'static [TransformBackend] {
        &[TransformBackend::Portable, TransformBackend::Fftw]
    }

    /// A short name for a message or a printed table.
    pub fn name(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            #[cfg(feature = "fftw")]
            Self::Fftw => "fftw",
        }
    }
}

/// A two-dimensional real-input discrete Fourier transform of a fixed shape,
/// with its plans and its scratch built once.
///
/// **The plans are the point.** Building them computes twiddle factors and picks
/// an algorithm per length, and a consumer transforming hundreds of planes of
/// one shape should pay for that once. Cloning shares the plans through an
/// `Arc` and allocates fresh scratch, which is what makes one plan usable from
/// several threads.
///
/// Which arithmetic runs underneath is a [`TransformBackend`]; the shape checks,
/// the padding rule and the normalisation are this type's and are the same
/// whichever one it is.
///
/// See this module's header for the normalisation convention: **the forward
/// direction is unnormalised and the inverse carries `1/N`**.
pub struct RealTransform2 {
    inner: Inner,
}

/// The transform itself, one variant per backend this build has.
#[derive(Clone)]
enum Inner {
    Portable(Portable),
    #[cfg(feature = "fftw")]
    Fftw(fftw_backend::Transform),
}

/// The pure-Rust transform: `realfft` along each row, then `rustfft` along the
/// first axis a block of lanes at a time.
///
/// Every argument here has already been checked by [`RealTransform2`], which is
/// where the shape errors live and where they stay whatever the backend.
struct Portable {
    rows: usize,
    cols: usize,
    row_forward: Arc<dyn RealToComplex<f64>>,
    row_inverse: Arc<dyn ComplexToReal<f64>>,
    column_forward: Arc<dyn Fft<f64>>,
    column_inverse: Arc<dyn Fft<f64>>,
    row_forward_scratch: Vec<Complex<f64>>,
    row_inverse_scratch: Vec<Complex<f64>>,
    column_scratch: Vec<Complex<f64>>,
    lanes: Vec<Complex<f64>>,
    real_row: Vec<f64>,
}

impl Clone for Portable {
    /// Shares the plans and allocates fresh scratch, so a clone is a second
    /// working set over one set of twiddles rather than a second planning pass.
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            row_forward: Arc::clone(&self.row_forward),
            row_inverse: Arc::clone(&self.row_inverse),
            column_forward: Arc::clone(&self.column_forward),
            column_inverse: Arc::clone(&self.column_inverse),
            row_forward_scratch: vec![Complex::new(0.0, 0.0); self.row_forward_scratch.len()],
            row_inverse_scratch: vec![Complex::new(0.0, 0.0); self.row_inverse_scratch.len()],
            column_scratch: vec![Complex::new(0.0, 0.0); self.column_scratch.len()],
            lanes: vec![Complex::new(0.0, 0.0); self.lanes.len()],
            real_row: vec![0.0; self.real_row.len()],
        }
    }
}

impl Clone for RealTransform2 {
    /// Shares the plans and allocates fresh scratch, whichever backend it is.
    ///
    /// For the FFTW backend that is load-bearing rather than merely thrifty:
    /// FFTW's planner is not thread-safe, and a clone that re-planned would put
    /// a plan creation inside whatever loop the caller cloned for.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for RealTransform2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealTransform2")
            .field("shape", &self.shape())
            .field("backend", &self.backend())
            .finish_non_exhaustive()
    }
}

impl RealTransform2 {
    /// Plan a transform of `shape` on the default backend. Both extents must be
    /// non-zero.
    pub fn new(shape: [usize; 2]) -> Result<Self> {
        Self::with_backend(shape, TransformBackend::default())
    }

    /// Plan a transform of `shape` on a named backend.
    ///
    /// [`Self::new`] is this with [`TransformBackend::default`]. A caller that
    /// wants a particular backend whatever the build's features are — a test
    /// comparing the two, most obviously — says so here.
    pub fn with_backend(shape: [usize; 2], backend: TransformBackend) -> Result<Self> {
        let [rows, cols] = shape;
        if rows == 0 || cols == 0 {
            return Err(Error::invalid(format!(
                "a transform needs a non-empty shape, got {rows} x {cols}"
            )));
        }
        let inner = match backend {
            TransformBackend::Portable => Inner::Portable(Portable::new(rows, cols)),
            #[cfg(feature = "fftw")]
            TransformBackend::Fftw => Inner::Fftw(fftw_backend::Transform::new(rows, cols)?),
        };
        Ok(Self { inner })
    }

    /// Which backend this plan runs on.
    pub fn backend(&self) -> TransformBackend {
        match &self.inner {
            Inner::Portable(_) => TransformBackend::Portable,
            #[cfg(feature = "fftw")]
            Inner::Fftw(_) => TransformBackend::Fftw,
        }
    }

    /// The shape of the real plane this transforms.
    pub fn shape(&self) -> [usize; 2] {
        match &self.inner {
            Inner::Portable(portable) => [portable.rows, portable.cols],
            #[cfg(feature = "fftw")]
            Inner::Fftw(fftw) => fftw.shape(),
        }
    }

    /// The shape of the half-spectrum: `[rows, cols / 2 + 1]`.
    pub fn spectrum_shape(&self) -> [usize; 2] {
        let [rows, cols] = self.shape();
        [rows, spectrum_width(cols)]
    }

    /// A zeroed spectrum of the right shape, for a caller that wants to allocate
    /// once and reuse.
    pub fn spectrum(&self) -> Spectrum {
        let [rows, cols] = self.spectrum_shape();
        Array2::zeros((rows, cols))
    }

    /// Forward transform. `input` must be exactly [`Self::shape`].
    pub fn forward(&mut self, input: ArrayView2<f64>, out: &mut Spectrum) -> Result<()> {
        let [rows, cols] = self.shape();
        if input.dim() != (rows, cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![input.dim().0, input.dim().1],
            });
        }
        self.forward_zero_padded(input, out)
    }

    /// Forward transform of `input` **placed at the origin of a zeroed plane** of
    /// [`Self::shape`].
    ///
    /// The zero padding is the whole reason a correlation can be linear rather
    /// than circular, so it is a named operation rather than a convenience:
    /// `input` may be smaller than the transform on either axis, and everything
    /// beyond it is zero.
    pub fn forward_zero_padded(
        &mut self,
        input: ArrayView2<f64>,
        out: &mut Spectrum,
    ) -> Result<()> {
        let [rows, cols] = self.shape();
        let (input_rows, input_cols) = input.dim();
        if input_rows > rows || input_cols > cols {
            return Err(Error::invalid(format!(
                "an input of {input_rows} x {input_cols} does not fit a {rows} x {cols} transform"
            )));
        }
        let width = spectrum_width(cols);
        if out.dim() != (rows, width) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, width],
                got: vec![out.dim().0, out.dim().1],
            });
        }
        let data = out
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("the spectrum must be contiguous and row-major"))?;
        match &mut self.inner {
            Inner::Portable(portable) => portable.forward_zero_padded(input, data),
            #[cfg(feature = "fftw")]
            Inner::Fftw(fftw) => fftw.forward_zero_padded(input, data),
        }
    }

    /// Inverse transform, carrying the `1/N` this convention puts on this side.
    ///
    /// `spectrum` is **not preserved**: what it holds afterwards is unspecified
    /// and differs between backends — the portable one transforms it in place
    /// along the first axis and consumes it row by row, and FFTW's
    /// complex-to-real transform destroys a copy of it instead. That is stated
    /// rather than hidden behind a copy on the portable side, because the caller
    /// almost always has no further use for it and a copy of a spectrum is not
    /// free.
    pub fn inverse(&mut self, spectrum: &mut Spectrum, out: &mut Array2<f64>) -> Result<()> {
        let [rows, cols] = self.shape();
        let width = spectrum_width(cols);
        if spectrum.dim() != (rows, width) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, width],
                got: vec![spectrum.dim().0, spectrum.dim().1],
            });
        }
        if out.dim() != (rows, cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![out.dim().0, out.dim().1],
            });
        }
        let data = spectrum
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("the spectrum must be contiguous and row-major"))?;
        match &mut self.inner {
            Inner::Portable(portable) => portable.inverse(data, out),
            #[cfg(feature = "fftw")]
            Inner::Fftw(fftw) => fftw.inverse(data, out),
        }
    }
}

impl Portable {
    /// Plan a transform of `rows x cols`, both already known to be non-zero.
    fn new(rows: usize, cols: usize) -> Self {
        let mut real = RealFftPlanner::<f64>::new();
        let row_forward = real.plan_fft_forward(cols);
        let row_inverse = real.plan_fft_inverse(cols);
        let mut complex = FftPlanner::<f64>::new();
        let column_forward = complex.plan_fft_forward(rows);
        let column_inverse = complex.plan_fft_inverse(rows);
        let column_scratch = vec![
            Complex::new(0.0, 0.0);
            column_forward
                .get_inplace_scratch_len()
                .max(column_inverse.get_inplace_scratch_len())
        ];
        Self {
            rows,
            cols,
            row_forward_scratch: vec![Complex::new(0.0, 0.0); row_forward.get_scratch_len()],
            row_inverse_scratch: vec![Complex::new(0.0, 0.0); row_inverse.get_scratch_len()],
            column_scratch,
            lanes: vec![Complex::new(0.0, 0.0); rows * LANE_BLOCK],
            real_row: vec![0.0; cols],
            row_forward,
            row_inverse,
            column_forward,
            column_inverse,
        }
    }

    /// Forward transform of `input` placed at the origin of a zeroed plane, into
    /// the contiguous `rows x (cols / 2 + 1)` spectrum `data`.
    fn forward_zero_padded(
        &mut self,
        input: ArrayView2<f64>,
        data: &mut [Complex<f64>],
    ) -> Result<()> {
        let (input_rows, input_cols) = input.dim();
        let width = spectrum_width(self.cols);
        let Self {
            row_forward,
            row_forward_scratch,
            real_row,
            column_forward,
            column_scratch,
            lanes,
            rows,
            ..
        } = self;
        for row in 0..*rows {
            if row < input_rows {
                let source = input.row(row);
                for (slot, &value) in real_row.iter_mut().zip(source.iter()) {
                    *slot = value;
                }
                real_row[input_cols..].fill(0.0);
            } else {
                real_row.fill(0.0);
            }
            row_forward
                .process_with_scratch(
                    real_row,
                    &mut data[row * width..(row + 1) * width],
                    row_forward_scratch,
                )
                .map_err(|error| Error::invalid(format!("forward row transform: {error}")))?;
        }
        transform_lanes(data, *rows, width, &**column_forward, lanes, column_scratch);
        Ok(())
    }

    /// Inverse transform of the contiguous spectrum `data`, carrying the `1/N`
    /// this convention puts on this side. `data` is transformed in place along
    /// the first axis and then consumed row by row.
    fn inverse(&mut self, data: &mut [Complex<f64>], out: &mut Array2<f64>) -> Result<()> {
        let width = spectrum_width(self.cols);
        let Self {
            row_inverse,
            row_inverse_scratch,
            real_row,
            column_inverse,
            column_scratch,
            lanes,
            rows,
            cols,
            ..
        } = self;
        transform_lanes(data, *rows, width, &**column_inverse, lanes, column_scratch);
        let scale = 1.0 / (*rows as f64 * *cols as f64);
        let even = *cols % 2 == 0;
        for row in 0..*rows {
            // The exact values of these two bins are real; the arithmetic above
            // leaves ~1e-17 of imaginary part on them and `realfft` refuses the
            // row rather than ignoring it. Asserting the symmetry is correct;
            // see this module's header.
            data[row * width].im = 0.0;
            if even {
                data[row * width + width - 1].im = 0.0;
            }
            row_inverse
                .process_with_scratch(
                    &mut data[row * width..(row + 1) * width],
                    real_row,
                    row_inverse_scratch,
                )
                .map_err(|error| Error::invalid(format!("inverse row transform: {error}")))?;
            for (slot, &value) in out.row_mut(row).iter_mut().zip(real_row.iter()) {
                *slot = value * scale;
            }
        }
        Ok(())
    }
}

/// Transform every lane along the first axis of a `rows x cols` row-major
/// complex buffer, a block of [`LANE_BLOCK`] lanes at a time.
fn transform_lanes(
    data: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
    fft: &dyn Fft<f64>,
    lanes: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    let mut base = 0;
    while base < cols {
        let width = LANE_BLOCK.min(cols - base);
        for row in 0..rows {
            let source = &data[row * cols + base..row * cols + base + width];
            for (lane, &value) in source.iter().enumerate() {
                lanes[lane * rows + row] = value;
            }
        }
        for lane in 0..width {
            fft.process_with_scratch(&mut lanes[lane * rows..(lane + 1) * rows], scratch);
        }
        for row in 0..rows {
            let sink = &mut data[row * cols + base..row * cols + base + width];
            for (lane, slot) in sink.iter_mut().enumerate() {
                *slot = lanes[lane * rows + row];
            }
        }
        base += width;
    }
}

// ---------------------------------------------------------- the third axis --

/// The half-spectrum of a real `d0 x d1 x d2` volume: `[d0, d1, d2 / 2 + 1]`.
///
/// [`Spectrum`] one rank up and under the same convention — only the **last**
/// axis is halved. A real input's spectrum is conjugate-symmetric under
/// negation of *all three* indices at once, so halving a second axis would need
/// the sign of the other two carried beside it; one halved axis is the whole of
/// the redundancy that comes for free.
pub type Spectrum3 = Array3<Complex<f64>>;

/// A discrete Fourier transform of a real **volume**.
///
/// [`RealTransform2`] with the axis [`crate::ops::deconvolve`]'s table calls
/// missing — "the same twenty lines, over `Array3`" — and it is the same lines:
/// a real transform along the last axis, then [`transform_lanes`] along axis 1
/// within each plane, then [`transform_lanes`] along axis 0 over the whole
/// buffer, because a `[d0, d1, w]` row-major buffer **is** a `[d0, d1 * w]` one
/// to a first-axis pass. Nothing new is planned for the outermost axis and
/// nothing new is scratch-managed; only the loop bounds differ.
///
/// **The normalisation is this module's**: the forward direction is
/// unnormalised and the inverse carries `1 / (d0 d1 d2)`. The round trip is the
/// identity to a bound and not to the bit, exactly as the two-dimensional one
/// is, and [`RealTransform3`]'s tests state the achieved figure.
///
/// **One backend, and that is deliberate rather than pending.** The `fftw`
/// feature's transform is a two-dimensional `r2c` plan and is not a
/// three-dimensional one; a `TransformBackend` argument here would be a
/// parameter with one legal setting that reads as a choice. [`RealTransform2`]
/// keeps its enum because it really has two; this one says what it is.
///
/// **Clone shares the plans and allocates fresh scratch**, so a clone is a
/// second working set over one set of twiddles. That is what makes a
/// [`crate::op::BlockOp`] able to hold one: `apply` takes `&self` and the
/// transform needs `&mut`, so the op clones its template per block rather than
/// planning per block or locking.
pub struct RealTransform3 {
    shape: [usize; 3],
    row_forward: Arc<dyn RealToComplex<f64>>,
    row_inverse: Arc<dyn ComplexToReal<f64>>,
    plane_forward: Arc<dyn Fft<f64>>,
    plane_inverse: Arc<dyn Fft<f64>>,
    volume_forward: Arc<dyn Fft<f64>>,
    volume_inverse: Arc<dyn Fft<f64>>,
    row_forward_scratch: Vec<Complex<f64>>,
    row_inverse_scratch: Vec<Complex<f64>>,
    plane_scratch: Vec<Complex<f64>>,
    volume_scratch: Vec<Complex<f64>>,
    plane_lanes: Vec<Complex<f64>>,
    volume_lanes: Vec<Complex<f64>>,
    real_row: Vec<f64>,
}

impl Clone for RealTransform3 {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            row_forward: Arc::clone(&self.row_forward),
            row_inverse: Arc::clone(&self.row_inverse),
            plane_forward: Arc::clone(&self.plane_forward),
            plane_inverse: Arc::clone(&self.plane_inverse),
            volume_forward: Arc::clone(&self.volume_forward),
            volume_inverse: Arc::clone(&self.volume_inverse),
            row_forward_scratch: vec![Complex::new(0.0, 0.0); self.row_forward_scratch.len()],
            row_inverse_scratch: vec![Complex::new(0.0, 0.0); self.row_inverse_scratch.len()],
            plane_scratch: vec![Complex::new(0.0, 0.0); self.plane_scratch.len()],
            volume_scratch: vec![Complex::new(0.0, 0.0); self.volume_scratch.len()],
            plane_lanes: vec![Complex::new(0.0, 0.0); self.plane_lanes.len()],
            volume_lanes: vec![Complex::new(0.0, 0.0); self.volume_lanes.len()],
            real_row: vec![0.0; self.real_row.len()],
        }
    }
}

impl std::fmt::Debug for RealTransform3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealTransform3")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl RealTransform3 {
    /// Plan a transform of `shape`. Every extent must be non-zero.
    pub fn new(shape: [usize; 3]) -> Result<Self> {
        let [d0, d1, d2] = shape;
        if d0 == 0 || d1 == 0 || d2 == 0 {
            return Err(Error::invalid(format!(
                "a transform needs a non-empty shape, got {d0} x {d1} x {d2}"
            )));
        }
        let mut real = RealFftPlanner::<f64>::new();
        let row_forward = real.plan_fft_forward(d2);
        let row_inverse = real.plan_fft_inverse(d2);
        let mut complex = FftPlanner::<f64>::new();
        let plane_forward = complex.plan_fft_forward(d1);
        let plane_inverse = complex.plan_fft_inverse(d1);
        let volume_forward = complex.plan_fft_forward(d0);
        let volume_inverse = complex.plan_fft_inverse(d0);
        let plane_scratch = vec![
            Complex::new(0.0, 0.0);
            plane_forward
                .get_inplace_scratch_len()
                .max(plane_inverse.get_inplace_scratch_len())
        ];
        let volume_scratch = vec![
            Complex::new(0.0, 0.0);
            volume_forward
                .get_inplace_scratch_len()
                .max(volume_inverse.get_inplace_scratch_len())
        ];
        Ok(Self {
            shape,
            row_forward_scratch: vec![Complex::new(0.0, 0.0); row_forward.get_scratch_len()],
            row_inverse_scratch: vec![Complex::new(0.0, 0.0); row_inverse.get_scratch_len()],
            plane_scratch,
            volume_scratch,
            plane_lanes: vec![Complex::new(0.0, 0.0); d1 * LANE_BLOCK],
            volume_lanes: vec![Complex::new(0.0, 0.0); d0 * LANE_BLOCK],
            real_row: vec![0.0; d2],
            row_forward,
            row_inverse,
            plane_forward,
            plane_inverse,
            volume_forward,
            volume_inverse,
        })
    }

    /// The real volume's shape.
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// `[d0, d1, d2 / 2 + 1]`.
    pub fn spectrum_shape(&self) -> [usize; 3] {
        [self.shape[0], self.shape[1], spectrum_width(self.shape[2])]
    }

    /// A zeroed spectrum of the right shape, for a caller that wants to reuse
    /// one across many transforms.
    pub fn spectrum(&self) -> Spectrum3 {
        let [a, b, c] = self.spectrum_shape();
        Array3::from_elem((a, b, c), Complex::new(0.0, 0.0))
    }

    /// Forward transform of `input` placed at the origin of a zeroed volume of
    /// [`Self::shape`], into `out`.
    ///
    /// The padding is the caller's whole reason for a transform longer than the
    /// data — see this module's header on wrap-around — so it is done here
    /// rather than asking every caller to allocate and zero the larger volume
    /// itself. An `input` larger than the transform on any axis is refused.
    pub fn forward_zero_padded(
        &mut self,
        input: ArrayView3<f64>,
        out: &mut Spectrum3,
    ) -> Result<()> {
        let [d0, d1, d2] = self.shape;
        let width = spectrum_width(d2);
        let (in0, in1, in2) = input.dim();
        if in0 > d0 || in1 > d1 || in2 > d2 {
            return Err(Error::invalid(format!(
                "a {in0} x {in1} x {in2} volume does not fit in a {d0} x {d1} x {d2} transform"
            )));
        }
        let expected = self.spectrum_shape();
        if out.shape() != expected {
            return Err(Error::invalid(format!(
                "this transform's spectrum is {expected:?} and was given {:?}",
                out.shape()
            )));
        }
        let data = out
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("a spectrum must be contiguous".to_string()))?;
        let Self {
            row_forward,
            row_forward_scratch,
            real_row,
            plane_forward,
            plane_scratch,
            plane_lanes,
            volume_forward,
            volume_scratch,
            volume_lanes,
            ..
        } = self;
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                if i0 < in0 && i1 < in1 {
                    for (k, slot) in real_row.iter_mut().enumerate() {
                        *slot = if k < in2 { input[[i0, i1, k]] } else { 0.0 };
                    }
                } else {
                    real_row.fill(0.0);
                }
                let base = (i0 * d1 + i1) * width;
                row_forward
                    .process_with_scratch(
                        real_row,
                        &mut data[base..base + width],
                        row_forward_scratch,
                    )
                    .map_err(|error| Error::invalid(format!("forward row transform: {error}")))?;
            }
        }
        for i0 in 0..d0 {
            let base = i0 * d1 * width;
            transform_lanes(
                &mut data[base..base + d1 * width],
                d1,
                width,
                &**plane_forward,
                plane_lanes,
                plane_scratch,
            );
        }
        transform_lanes(
            data,
            d0,
            d1 * width,
            &**volume_forward,
            volume_lanes,
            volume_scratch,
        );
        Ok(())
    }

    /// Inverse transform of `spectrum`, carrying the `1 / (d0 d1 d2)` this
    /// convention puts on this side. `spectrum` is consumed — it is transformed
    /// in place — and `out` must be [`Self::shape`].
    pub fn inverse(&mut self, spectrum: &mut Spectrum3, out: &mut Array3<f64>) -> Result<()> {
        let [d0, d1, d2] = self.shape;
        let width = spectrum_width(d2);
        let expected = self.spectrum_shape();
        if spectrum.shape() != expected {
            return Err(Error::invalid(format!(
                "this transform's spectrum is {expected:?} and was given {:?}",
                spectrum.shape()
            )));
        }
        if out.shape() != self.shape {
            return Err(Error::invalid(format!(
                "this transform writes {:?} and was given {:?}",
                self.shape,
                out.shape()
            )));
        }
        let data = spectrum
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("a spectrum must be contiguous".to_string()))?;
        let sink = out
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("an output volume must be contiguous".to_string()))?;
        let Self {
            row_inverse,
            row_inverse_scratch,
            real_row,
            plane_inverse,
            plane_scratch,
            plane_lanes,
            volume_inverse,
            volume_scratch,
            volume_lanes,
            ..
        } = self;
        transform_lanes(
            data,
            d0,
            d1 * width,
            &**volume_inverse,
            volume_lanes,
            volume_scratch,
        );
        for i0 in 0..d0 {
            let base = i0 * d1 * width;
            transform_lanes(
                &mut data[base..base + d1 * width],
                d1,
                width,
                &**plane_inverse,
                plane_lanes,
                plane_scratch,
            );
        }
        let scale = 1.0 / (d0 as f64 * d1 as f64 * d2 as f64);
        let even = d2 % 2 == 0;
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                let base = (i0 * d1 + i1) * width;
                // The exact values of these two bins are real; the passes above
                // leave rounding noise on them and `realfft` refuses the row
                // rather than ignoring it. Asserting the symmetry is correct —
                // this module's header argues it for the two-dimensional case
                // and the argument does not depend on the rank.
                data[base].im = 0.0;
                if even {
                    data[base + width - 1].im = 0.0;
                }
                row_inverse
                    .process_with_scratch(
                        &mut data[base..base + width],
                        real_row,
                        row_inverse_scratch,
                    )
                    .map_err(|error| Error::invalid(format!("inverse row transform: {error}")))?;
                let out_base = (i0 * d1 + i1) * d2;
                for (slot, &value) in sink[out_base..out_base + d2]
                    .iter_mut()
                    .zip(real_row.iter())
                {
                    *slot = value * scale;
                }
            }
        }
        Ok(())
    }
}

// -------------------------------------------------------- the other backend --

/// The `fftw` feature's transform: the system's FFTW 3 through `fftw-sys`.
///
/// **What is different from the portable path, and why.** FFTW plans a whole
/// two-dimensional transform at once, so this is one `r2c` plan and one `c2r`
/// plan rather than a row pass and a hand-blocked lane pass. That is FFTW's own
/// shape and the fair way to ask it for a number; a transcription of the
/// two-pass structure onto FFTW would be measuring this file's loop rather than
/// the library.
///
/// **Three hazards, each handled here rather than hoped about.**
///
/// 1. **The planner is not thread-safe.** FFTW documents *the new-array execute
///    functions* as the only re-entrant entry points it has. Plan creation, plan
///    destruction and `fftw_malloc`/`fftw_free` all touch global state, so every
///    one of them in this module goes through [`planner`], a process-wide lock,
///    and the execute calls — the only ones in the hot path — go through nothing
///    at all. A shared plan executed from several threads at once is exactly
///    what FFTW says is allowed, and it is what [`Transform::clone`] produces.
/// 2. **New-array execute requires the same alignment.** A plan records the
///    alignment of the arrays it was made with and is only valid for arrays that
///    match; hand it something else and the answer is wrong rather than refused.
///    Every buffer here comes from `fftw_alloc_*`, so they all agree by
///    construction — and [`Plans::check`] asserts it anyway, because a silent
///    wrong answer is the one failure mode this whole module must not have.
/// 3. **`FFTW_MEASURE` would break determinism.** It picks an algorithm by
///    timing, so two clones of one plan could disagree in the last place and
///    `a_cloned_plan_computes_the_same_answer_from_another_thread` — which
///    asserts bit-identity — would become flaky. `FFTW_ESTIMATE` is also what the
///    measurements say to use: 0.4-1.1 s of planning for 0-25%, and this crate's
///    parallelism is one plan shared by every thread, so that cost is paid once
///    and cannot be amortised away.
///
/// FFTW's own threading (`fftw_plan_with_nthreads`) is deliberately not used:
/// the consumer's parallelism is across plane pairs, and a parallel transform
/// would take the same cores and win less. That is the same argument this
/// module's header makes for the portable backend, and it does not change with
/// the library.
#[cfg(feature = "fftw")]
mod fftw_backend {
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    use fftw_sys as ffi;
    use ndarray::{Array2, ArrayView2};

    use super::{spectrum_width, Complex};
    use crate::error::{Error, Result};

    /// The planning flag, and see this module's header for why it is this one.
    const FLAGS: u32 = ffi::FFTW_ESTIMATE;

    /// The lock every FFTW call that is **not** an execute goes through.
    ///
    /// Process-wide, because what it guards is FFTW's process-wide state — its
    /// planner, its wisdom and its allocator — rather than anything this crate
    /// owns. Poisoning is recovered from rather than propagated: the data behind
    /// the lock is `()`, the only panic that can happen under it is an allocation
    /// failure that leaves FFTW's state exactly as it found it, and a poisoned
    /// lock would otherwise mean one such panic stops this process from planning
    /// a transform ever again.
    fn planner() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A buffer from `fftw_alloc_*`: SIMD-aligned, and owned.
    ///
    /// Not a `Vec`, and the reason is hazard 2 in this module's header rather
    /// than a preference: FFTW's alignment classes come from its own allocator,
    /// and a `Vec`'s eight-byte alignment is not one a plan made with
    /// `fftw_alloc_*` memory may be executed against.
    struct Aligned<T> {
        data: *mut T,
        len: usize,
    }

    // SAFETY: an `Aligned` owns its allocation exclusively — it is created from
    // a fresh `fftw_alloc_*`, never cloned, never handed out except through
    // `&self`/`&mut self`, and freed exactly once in `Drop`. That makes moving
    // one to another thread no more sharing than moving a `Vec` is.
    unsafe impl<T: Send> Send for Aligned<T> {}

    // SAFETY: shared access to an `Aligned` gives out `&[T]` and nothing else, so
    // two threads holding one can only read. This is not tidiness: the portable
    // backend's plans are `Send + Sync`, and a feature that quietly took `Sync`
    // off `RealTransform2` would be changing the public API rather than the
    // transform behind it. `the_plans_are_send_and_sync_whichever_backend_they_
    // are_on` is where that is held to.
    unsafe impl<T: Sync> Sync for Aligned<T> {}

    impl Aligned<f64> {
        fn real(len: usize) -> Self {
            let _guard = planner();
            // SAFETY: `fftw_alloc_real` is malloc with an alignment guarantee;
            // `len` is non-zero because a transform of a non-empty shape is.
            let data = unsafe { ffi::fftw_alloc_real(len) };
            assert!(!data.is_null(), "fftw_alloc_real({len}) failed");
            // SAFETY: `data` owns `len` freshly allocated `f64`s, and every
            // slice handed out below promises they are initialised.
            unsafe { std::ptr::write_bytes(data, 0, len) };
            Self { data, len }
        }
    }

    impl Aligned<Complex<f64>> {
        fn complex(len: usize) -> Self {
            let _guard = planner();
            // SAFETY: as above. `fftw_complex` is `double[2]`, which is the
            // layout of `num_complex::Complex<f64>` — a `#[repr(C)]` pair of
            // `f64` — so one allocation serves both names for it. The cast is
            // written out rather than relying on the two crates resolving to the
            // same `num-complex`.
            let data = unsafe { ffi::fftw_alloc_complex(len) }.cast::<Complex<f64>>();
            assert!(!data.is_null(), "fftw_alloc_complex({len}) failed");
            // SAFETY: as above.
            unsafe { std::ptr::write_bytes(data, 0, len) };
            Self { data, len }
        }
    }

    impl<T> Aligned<T> {
        fn as_slice(&self) -> &[T] {
            // SAFETY: `data` points at `len` initialised, owned elements.
            unsafe { std::slice::from_raw_parts(self.data, self.len) }
        }

        fn as_mut_slice(&mut self) -> &mut [T] {
            // SAFETY: as above, and `&mut self` makes the borrow exclusive.
            unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
        }
    }

    impl<T> Drop for Aligned<T> {
        fn drop(&mut self) {
            let _guard = planner();
            // SAFETY: `data` came from `fftw_alloc_*` and is freed once.
            unsafe { ffi::fftw_free(self.data.cast::<c_void>()) };
        }
    }

    /// The two plans, and the alignment they were made against.
    struct Plans {
        forward: ffi::fftw_plan,
        inverse: ffi::fftw_plan,
        real_alignment: c_int,
        spectrum_alignment: c_int,
    }

    // SAFETY: the pointers are FFTW plans, and after creation nothing in this
    // module writes through them. The only calls that touch them outside the
    // `planner` lock are `fftw_execute_dft_r2c` and `fftw_execute_dft_c2r`,
    // which FFTW documents as thread-safe on a shared plan — that is the whole
    // reason a plan can be behind an `Arc` here. Creation and destruction take
    // the lock, and destruction happens once, when the last `Arc` goes.
    unsafe impl Send for Plans {}
    unsafe impl Sync for Plans {}

    impl Plans {
        fn new(
            rows: usize,
            cols: usize,
            real: &mut Aligned<f64>,
            spectrum: &mut Aligned<Complex<f64>>,
        ) -> Result<Self> {
            let extent = |value: usize| {
                c_int::try_from(value).map_err(|_| {
                    Error::invalid(format!("an extent of {value} is beyond what FFTW can plan"))
                })
            };
            let (rows_c, cols_c) = (extent(rows)?, extent(cols)?);
            let _guard = planner();
            // SAFETY: both buffers are the sizes the two-dimensional real
            // transform of `rows x cols` needs — `rows * cols` reals and
            // `rows * (cols / 2 + 1)` complexes — and the lock makes this the
            // only planning in flight. `FFTW_ESTIMATE` does not touch either
            // array, so their contents survive.
            let (forward, inverse) = unsafe {
                (
                    ffi::fftw_plan_dft_r2c_2d(
                        rows_c,
                        cols_c,
                        real.data,
                        spectrum.data.cast::<ffi::fftw_complex>(),
                        FLAGS,
                    ),
                    ffi::fftw_plan_dft_c2r_2d(
                        rows_c,
                        cols_c,
                        spectrum.data.cast::<ffi::fftw_complex>(),
                        real.data,
                        FLAGS,
                    ),
                )
            };
            if forward.is_null() || inverse.is_null() {
                // SAFETY: each pointer is either null or a plan this call made,
                // and the lock is still held.
                unsafe {
                    if !forward.is_null() {
                        ffi::fftw_destroy_plan(forward);
                    }
                    if !inverse.is_null() {
                        ffi::fftw_destroy_plan(inverse);
                    }
                }
                return Err(Error::invalid(format!(
                    "FFTW could not plan a {rows} x {cols} real transform"
                )));
            }
            // SAFETY: both pointers are live allocations; `fftw_alignment_of`
            // only reads the address.
            let (real_alignment, spectrum_alignment) = unsafe {
                (
                    ffi::fftw_alignment_of(real.data),
                    ffi::fftw_alignment_of(spectrum.data.cast::<f64>()),
                )
            };
            Ok(Self {
                forward,
                inverse,
                real_alignment,
                spectrum_alignment,
            })
        }

        /// Hazard 2, asserted: a plan may only be executed against arrays whose
        /// alignment matches the ones it was planned with.
        ///
        /// Every buffer here comes from `fftw_alloc_*` so this holds by
        /// construction, which is exactly why it is worth checking — a
        /// construction that quietly stopped holding would produce wrong
        /// numbers, not a crash.
        fn check(&self, real: &Aligned<f64>, spectrum: &Aligned<Complex<f64>>) {
            let (real_alignment, spectrum_alignment) = {
                let _guard = planner();
                // SAFETY: both pointers are live allocations and only their
                // addresses are read. Under the lock like every other FFTW call
                // that is not an execute, so the rule this module states about
                // which calls are serialised has no exceptions to remember.
                unsafe {
                    (
                        ffi::fftw_alignment_of(real.data),
                        ffi::fftw_alignment_of(spectrum.data.cast::<f64>()),
                    )
                }
            };
            assert_eq!(
                real_alignment, self.real_alignment,
                "an FFTW plan may only be executed against arrays of the alignment \
                 it was planned with, and this working set's real buffer is not"
            );
            assert_eq!(
                spectrum_alignment, self.spectrum_alignment,
                "an FFTW plan may only be executed against arrays of the alignment \
                 it was planned with, and this working set's spectrum is not"
            );
        }
    }

    impl Drop for Plans {
        fn drop(&mut self) {
            let _guard = planner();
            // SAFETY: both plans were made by `Plans::new`, this runs once when
            // the last `Arc` goes, and the lock excludes any other FFTW call
            // that is not an execute.
            unsafe {
                ffi::fftw_destroy_plan(self.forward);
                ffi::fftw_destroy_plan(self.inverse);
            }
        }
    }

    /// One working set over a shared pair of plans.
    pub(super) struct Transform {
        rows: usize,
        cols: usize,
        plans: Arc<Plans>,
        real: Aligned<f64>,
        spectrum: Aligned<Complex<f64>>,
    }

    impl Clone for Transform {
        /// The same bargain the portable backend's clone makes: share the plans,
        /// allocate a fresh working set. Here it is also what keeps plan
        /// creation — the part FFTW cannot do from two threads at once — out of
        /// whatever loop the caller cloned for.
        fn clone(&self) -> Self {
            let real = Aligned::real(self.rows * self.cols);
            let spectrum = Aligned::complex(self.rows * spectrum_width(self.cols));
            self.plans.check(&real, &spectrum);
            Self {
                rows: self.rows,
                cols: self.cols,
                plans: Arc::clone(&self.plans),
                real,
                spectrum,
            }
        }
    }

    impl Transform {
        pub(super) fn new(rows: usize, cols: usize) -> Result<Self> {
            let mut real = Aligned::real(rows * cols);
            let mut spectrum = Aligned::complex(rows * spectrum_width(cols));
            let plans = Plans::new(rows, cols, &mut real, &mut spectrum)?;
            Ok(Self {
                rows,
                cols,
                plans: Arc::new(plans),
                real,
                spectrum,
            })
        }

        pub(super) fn shape(&self) -> [usize; 2] {
            [self.rows, self.cols]
        }

        /// Forward transform of `input` placed at the origin of a zeroed plane,
        /// into the contiguous spectrum `data`.
        ///
        /// The zero padding is assembled into this working set's own aligned
        /// buffer — which is what the portable backend does row by row anyway —
        /// and the spectrum is copied out of the aligned one at the end. That
        /// copy is the price of `Spectrum` being an `ndarray` type the caller
        /// owns, and it is inside every measurement quoted for this backend.
        pub(super) fn forward_zero_padded(
            &mut self,
            input: ArrayView2<f64>,
            data: &mut [Complex<f64>],
        ) -> Result<()> {
            let (input_rows, input_cols) = input.dim();
            let cols = self.cols;
            let plane = self.real.as_mut_slice();
            for row in 0..self.rows {
                let target = &mut plane[row * cols..(row + 1) * cols];
                if row < input_rows {
                    for (slot, &value) in target.iter_mut().zip(input.row(row).iter()) {
                        *slot = value;
                    }
                    target[input_cols..].fill(0.0);
                } else {
                    target.fill(0.0);
                }
            }
            // SAFETY: the new-array execute functions are FFTW's documented
            // thread-safe entry point; `&mut self` makes these two buffers
            // exclusively ours; they are the shapes the plan was made for and,
            // by `Plans::check`, of the alignment it was made against.
            unsafe {
                ffi::fftw_execute_dft_r2c(
                    self.plans.forward,
                    self.real.data,
                    self.spectrum.data.cast::<ffi::fftw_complex>(),
                )
            };
            data.copy_from_slice(self.spectrum.as_slice());
            Ok(())
        }

        /// Inverse transform of the contiguous spectrum `data`, carrying the
        /// `1/N`.
        ///
        /// FFTW's multi-dimensional complex-to-real transform destroys its
        /// input, so `data` is copied into this working set's aligned spectrum
        /// and *that* is destroyed. The zero and Nyquist bins need no fixing up
        /// on this side — FFTW reads a half-spectrum as Hermitian by definition
        /// and does not refuse an imaginary part of `1e-17` where the exact value
        /// is real, which is the one thing `realfft` insists on.
        pub(super) fn inverse(
            &mut self,
            data: &mut [Complex<f64>],
            out: &mut Array2<f64>,
        ) -> Result<()> {
            self.spectrum.as_mut_slice().copy_from_slice(data);
            // SAFETY: as in `forward_zero_padded`.
            unsafe {
                ffi::fftw_execute_dft_c2r(
                    self.plans.inverse,
                    self.spectrum.data.cast::<ffi::fftw_complex>(),
                    self.real.data,
                )
            };
            let scale = 1.0 / (self.rows as f64 * self.cols as f64);
            let cols = self.cols;
            let plane = self.real.as_slice();
            for row in 0..self.rows {
                let source = &plane[row * cols..(row + 1) * cols];
                for (slot, &value) in out.row_mut(row).iter_mut().zip(source.iter()) {
                    *slot = value * scale;
                }
            }
            Ok(())
        }
    }
}

// ------------------------------------------------------------------ the lags --

/// The rectangle of integer lags a landscape is computed over, **inclusive at
/// both ends**.
///
/// The convention is stated here and used everywhere below: lag `k` places the
/// second plane's origin at `k` in the first plane's frame, so element `x` of the
/// first is compared with element `x - k` of the second. A sign error here is
/// the single easiest way to produce a landscape that looks right and is
/// reflected, which is why the tests use a window that is **not** symmetric
/// about zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShiftWindow {
    lower: [isize; 2],
    upper: [isize; 2],
}

impl ShiftWindow {
    /// A window from `lower` to `upper` inclusive. Each `upper` must be at least
    /// its `lower`.
    pub fn new(lower: [isize; 2], upper: [isize; 2]) -> Result<Self> {
        for axis in 0..2 {
            if upper[axis] < lower[axis] {
                return Err(Error::invalid(format!(
                    "lag window axis {axis} runs from {} to {}, which is empty",
                    lower[axis], upper[axis]
                )));
            }
        }
        Ok(Self { lower, upper })
    }

    /// A window of `radius` either side of zero on both axes.
    pub fn symmetric(radius: [usize; 2]) -> Self {
        Self {
            lower: [-(radius[0] as isize), -(radius[1] as isize)],
            upper: [radius[0] as isize, radius[1] as isize],
        }
    }

    pub fn lower(&self) -> [isize; 2] {
        self.lower
    }

    /// Inclusive.
    pub fn upper(&self) -> [isize; 2] {
        self.upper
    }

    /// The shape of a landscape over this window.
    pub fn extent(&self) -> [usize; 2] {
        [
            (self.upper[0] - self.lower[0] + 1) as usize,
            (self.upper[1] - self.lower[1] + 1) as usize,
        ]
    }

    /// The lag at a landscape index.
    pub fn shift_at(&self, index: [usize; 2]) -> [isize; 2] {
        [
            self.lower[0] + index[0] as isize,
            self.lower[1] + index[1] as isize,
        ]
    }

    /// The landscape index of a lag, or `None` if it is outside the window.
    pub fn index_of(&self, shift: [isize; 2]) -> Option<[usize; 2]> {
        (0..2)
            .all(|axis| shift[axis] >= self.lower[axis] && shift[axis] <= self.upper[axis])
            .then(|| {
                [
                    (shift[0] - self.lower[0]) as usize,
                    (shift[1] - self.lower[1]) as usize,
                ]
            })
    }

    /// Every lag in the window, first axis slowest.
    pub fn shifts(&self) -> impl Iterator<Item = [isize; 2]> + '_ {
        let [rows, cols] = self.extent();
        (0..rows).flat_map(move |row| (0..cols).map(move |col| self.shift_at([row, col])))
    }
}

/// The overlap of `[0, a)` and `[shift, shift + b)` on one axis, as a half-open
/// range in the **first** array's frame. Empty ranges come back as `(x, x)`.
fn overlap_in_first(a: usize, b: usize, shift: isize) -> (usize, usize) {
    let low = shift.max(0).min(a as isize) as usize;
    let high = (b as isize + shift).clamp(low as isize, a as isize) as usize;
    (low, high)
}

/// The same overlap in the **second** array's frame.
fn overlap_in_second(a: usize, b: usize, shift: isize) -> (usize, usize) {
    let (low, high) = overlap_in_first(a, b, shift);
    (
        (low as isize - shift) as usize,
        (high as isize - shift) as usize,
    )
}

// --------------------------------------------------------------- the padding --

/// How long the padded plane a correlation transforms is.
///
/// An **axis**, not a switch. [`Padding::Smooth`] is the answer; the other two
/// exist to be stepped to, and `Exact` shorter than [`Padding::Minimal`] is the
/// negative control this module's tests use — it reproduces every count and
/// every shape and moves the answer, because the contamination is a wrap rather
/// than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// [`minimal_wrap_free_length`] exactly. Correct, and often a hostile length
    /// for a mixed-radix transform.
    Minimal,
    /// `Minimal` rounded up per axis to the next `5`-smooth integer. Correct for
    /// the same reason — any length at or above the minimum is — and several
    /// times faster. The default.
    #[default]
    Smooth,
    /// A length the caller chose. **Below [`minimal_wrap_free_length`] the
    /// answer wraps**, and [`Correlation2::wraps`] says whether it does.
    Exact([usize; 2]),
}

/// The shortest padded length, per axis, at which every lag in `window` is free
/// of wrap-around.
///
/// `max(A, B, A - lo, B + hi)`; see this module's header for the derivation.
pub fn minimal_wrap_free_length(
    shape_a: [usize; 2],
    shape_b: [usize; 2],
    window: ShiftWindow,
) -> [usize; 2] {
    let mut length = [0usize; 2];
    for axis in 0..2 {
        let a = shape_a[axis] as isize;
        let b = shape_b[axis] as isize;
        let needed = a
            .max(b)
            .max(a - window.lower()[axis])
            .max(b + window.upper()[axis]);
        length[axis] = needed.max(1) as usize;
    }
    length
}

impl Padding {
    /// The padded shape this choice asks for.
    pub fn resolve(
        self,
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
    ) -> [usize; 2] {
        match self {
            Self::Exact(shape) => shape,
            Self::Minimal => minimal_wrap_free_length(shape_a, shape_b, window),
            Self::Smooth => {
                let minimal = minimal_wrap_free_length(shape_a, shape_b, window);
                [
                    next_smooth_length(minimal[0]),
                    next_smooth_length(minimal[1]),
                ]
            }
        }
    }
}

// ----------------------------------------------------------- the correlation --

/// `C(k) = sum_x a[x] b[x-k]` over a window of integer lags, through the
/// correlation theorem.
///
/// Two forward transforms, one conjugate product and one inverse, whatever the
/// number of lags — against one pass over the overlap per lag for the direct
/// form. The plans and every buffer are built once; [`Self::correlate_into`]
/// allocates nothing.
#[derive(Debug, Clone)]
pub struct Correlation2 {
    shape_a: [usize; 2],
    shape_b: [usize; 2],
    window: ShiftWindow,
    padded: [usize; 2],
    minimal: [usize; 2],
    transform: RealTransform2,
    spectrum_a: Spectrum,
    spectrum_b: Spectrum,
    plane: Array2<f64>,
}

impl Correlation2 {
    /// Plan a correlation between planes of `shape_a` and `shape_b` over
    /// `window`, on the default backend.
    pub fn new(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
    ) -> Result<Self> {
        Self::with_backend(
            shape_a,
            shape_b,
            window,
            padding,
            TransformBackend::default(),
        )
    }

    /// The same, on a named [`TransformBackend`].
    ///
    /// Everything except the transform itself — the padding rule, the
    /// conjugation, the lag convention, the wrap-around report — is this type's
    /// and is the same either way.
    pub fn with_backend(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
        backend: TransformBackend,
    ) -> Result<Self> {
        for axis in 0..2 {
            if shape_a[axis] == 0 || shape_b[axis] == 0 {
                return Err(Error::invalid(format!(
                    "a correlation needs non-empty planes, got {shape_a:?} and {shape_b:?}"
                )));
            }
        }
        let minimal = minimal_wrap_free_length(shape_a, shape_b, window);
        let padded = padding.resolve(shape_a, shape_b, window);
        for axis in 0..2 {
            if padded[axis] < shape_a[axis].max(shape_b[axis]) {
                return Err(Error::invalid(format!(
                    "a padded length of {} on axis {axis} cannot hold planes of {} and {}",
                    padded[axis], shape_a[axis], shape_b[axis]
                )));
            }
        }
        let transform = RealTransform2::with_backend(padded, backend)?;
        let spectrum_a = transform.spectrum();
        let spectrum_b = transform.spectrum();
        Ok(Self {
            shape_a,
            shape_b,
            window,
            padded,
            minimal,
            transform,
            spectrum_a,
            spectrum_b,
            plane: Array2::zeros((padded[0], padded[1])),
        })
    }

    /// The shape of the padded plane that is transformed.
    pub fn padded_shape(&self) -> [usize; 2] {
        self.padded
    }

    /// The shortest padded shape that would have been free of wrap-around.
    pub fn minimal_shape(&self) -> [usize; 2] {
        self.minimal
    }

    /// Which backend the transform underneath runs on.
    pub fn backend(&self) -> TransformBackend {
        self.transform.backend()
    }

    /// Whether this plan's padding is short enough for some lag in the window to
    /// pick up a contribution that wrapped round the plane.
    ///
    /// `false` for [`Padding::Minimal`] and [`Padding::Smooth`] by construction.
    /// A plan for which this is `true` is not broken — it is a different
    /// quantity, and one nothing here computes on purpose.
    pub fn wraps(&self) -> bool {
        (0..2).any(|axis| self.padded[axis] < self.minimal[axis])
    }

    pub fn window(&self) -> ShiftWindow {
        self.window
    }

    /// The correlation over the whole window, allocated.
    pub fn correlate(&mut self, a: ArrayView2<f64>, b: ArrayView2<f64>) -> Result<Array2<f64>> {
        let [rows, cols] = self.window.extent();
        let mut out = Array2::zeros((rows, cols));
        self.correlate_into(a, b, &mut out)?;
        Ok(out)
    }

    /// The correlation over the whole window, into a caller's array of
    /// [`ShiftWindow::extent`].
    pub fn correlate_into(
        &mut self,
        a: ArrayView2<f64>,
        b: ArrayView2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<()> {
        if a.dim() != (self.shape_a[0], self.shape_a[1]) {
            return Err(Error::ShapeMismatch {
                expected: self.shape_a.to_vec(),
                got: vec![a.dim().0, a.dim().1],
            });
        }
        if b.dim() != (self.shape_b[0], self.shape_b[1]) {
            return Err(Error::ShapeMismatch {
                expected: self.shape_b.to_vec(),
                got: vec![b.dim().0, b.dim().1],
            });
        }
        let [rows, cols] = self.window.extent();
        if out.dim() != (rows, cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![out.dim().0, out.dim().1],
            });
        }

        self.transform
            .forward_zero_padded(a, &mut self.spectrum_a)?;
        self.transform
            .forward_zero_padded(b, &mut self.spectrum_b)?;
        // `IDFT{ A conj(B) }[k] = sum_x a[x] b[x-k]`. Conjugating the *second*
        // operand is what makes the lag run this way round; conjugating the
        // first reflects the whole landscape, which is a negative control rather
        // than a variant.
        self.spectrum_a
            .zip_mut_with(&self.spectrum_b, |left, right| *left *= right.conj());
        self.transform
            .inverse(&mut self.spectrum_a, &mut self.plane)?;

        let [padded_rows, padded_cols] = self.padded;
        for row in 0..rows {
            let lag_row = self.window.lower()[0] + row as isize;
            let source_row = lag_row.rem_euclid(padded_rows as isize) as usize;
            for col in 0..cols {
                let lag_col = self.window.lower()[1] + col as isize;
                let source_col = lag_col.rem_euclid(padded_cols as isize) as usize;
                out[[row, col]] = self.plane[[source_row, source_col]];
            }
        }
        Ok(())
    }
}

/// `C(k) = sum_x a[x] b[x-k]` by walking every lag's overlap.
///
/// The oracle. `O(lags * overlap)` where [`Correlation2`] is `O(N log N)`, and
/// the only thing that can say whether the fast one is right.
pub fn correlate_direct(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    window: ShiftWindow,
) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let [rows, cols] = window.extent();
    let mut out = Array2::zeros((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let shift = window.shift_at([row, col]);
            let (row_low, row_high) = overlap_in_first(a_rows, b_rows, shift[0]);
            let (col_low, col_high) = overlap_in_first(a_cols, b_cols, shift[1]);
            let mut total = 0.0;
            for x in row_low..row_high {
                let y = (x as isize - shift[0]) as usize;
                for u in col_low..col_high {
                    let v = (u as isize - shift[1]) as usize;
                    total += a[[x, u]] * b[[y, v]];
                }
            }
            out[[row, col]] = total;
        }
    }
    out
}

// ------------------------------------------------------------- the landscape --

/// A mean squared difference at every lag of a window, with the overlap size
/// that produced each one.
#[derive(Debug, Clone, PartialEq)]
pub struct Landscape {
    window: ShiftWindow,
    mean_squared: Array2<f64>,
    overlap: Array2<u64>,
}

impl Landscape {
    /// A landscape over `window` with nothing computed into it yet: every value
    /// `INFINITY`, every overlap `0`.
    ///
    /// The buffer [`SquaredDifference::landscape_into`] fills. It is here rather
    /// than only on the plan so that a caller can allocate one without holding a
    /// plan, and it starts at `INFINITY`/`0` — the module's own encoding of "no
    /// overlap, so no answer" — because a buffer that starts at zero would read
    /// as a landscape whose every lag is a perfect match.
    pub fn empty(window: ShiftWindow) -> Self {
        let [rows, cols] = window.extent();
        Self {
            window,
            mean_squared: Array2::from_elem((rows, cols), f64::INFINITY),
            overlap: Array2::zeros((rows, cols)),
        }
    }

    pub fn window(&self) -> ShiftWindow {
        self.window
    }

    /// The landscape itself: `f64::INFINITY` at any lag whose overlap is empty.
    pub fn mean_squared(&self) -> ArrayView2<'_, f64> {
        self.mean_squared.view()
    }

    /// How many element pairs each lag's mean was taken over. **Exact
    /// integers** — a product of per-axis overlap lengths, not the modulus of a
    /// transform.
    pub fn overlap(&self) -> ArrayView2<'_, u64> {
        self.overlap.view()
    }

    /// The value at one lag, or `None` outside the window.
    pub fn at(&self, shift: [isize; 2]) -> Option<f64> {
        self.window
            .index_of(shift)
            .map(|index| self.mean_squared[[index[0], index[1]]])
    }

    /// The lag with the lowest value, and that value.
    ///
    /// **The tie-breaking rule is part of the contract, not an accident of the
    /// iterator.** Lags with an empty overlap are not candidates at all; among
    /// the rest the comparison is [`f64::total_cmp`], and among exact ties the
    /// **lexicographically smallest lag** wins — lowest first axis, then lowest
    /// second. `None` when no lag in the window overlaps at all.
    ///
    /// `total_cmp` rather than `f64::min` because a landscape's argmin is
    /// precisely a selection, and `f64::max(-0.0, 0.0)` is allowed to return
    /// either operand.
    pub fn argmin(&self) -> Option<([isize; 2], f64)> {
        let [rows, cols] = self.window.extent();
        let mut best: Option<([isize; 2], f64)> = None;
        for row in 0..rows {
            for col in 0..cols {
                if self.overlap[[row, col]] == 0 {
                    continue;
                }
                let value = self.mean_squared[[row, col]];
                let take = match best {
                    None => true,
                    // Strictly less, so the first lag of a tie group — which is
                    // the lexicographically smallest, since this walk is
                    // row-major — is the one that survives.
                    Some((_, incumbent)) => value.total_cmp(&incumbent).is_lt(),
                };
                if take {
                    best = Some((self.window.shift_at([row, col]), value));
                }
            }
        }
        best
    }
}

/// The landscape this module exists for: a mean squared difference between two
/// planes at every lag of a window, computed through one correlation.
///
/// See this module's header for which of the three terms goes through the
/// transform and which do not.
#[derive(Debug, Clone)]
pub struct SquaredDifference {
    correlation: Correlation2,
    shape_a: [usize; 2],
    shape_b: [usize; 2],
    /// Per lag index on each axis, the overlap in the first plane's frame.
    rows_in_a: Vec<(usize, usize)>,
    cols_in_a: Vec<(usize, usize)>,
    /// And in the second's.
    rows_in_b: Vec<(usize, usize)>,
    cols_in_b: Vec<(usize, usize)>,
    overlap: Array2<u64>,
    cross: Array2<f64>,
    /// The two energy landscapes and the two-stage scratch behind them, held by
    /// the plan for the same reason the transform's twiddles are: they are a
    /// function of the geometry, and the geometry does not change between calls.
    /// `parts_a` is the larger of the two at `shape_a[0] x window cols`.
    energy_a: Array2<f64>,
    energy_b: Array2<f64>,
    parts_a: Array2<f64>,
    parts_b: Array2<f64>,
    prefix: Vec<f64>,
}

impl SquaredDifference {
    /// Plan a landscape between planes of `shape_a` and `shape_b` over `window`,
    /// on the default backend.
    pub fn new(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
    ) -> Result<Self> {
        Self::with_backend(
            shape_a,
            shape_b,
            window,
            padding,
            TransformBackend::default(),
        )
    }

    /// The same, on a named [`TransformBackend`].
    ///
    /// Only the one term that goes through a transform changes: the two energy
    /// sums and the overlap counts are rectangle arithmetic and integer
    /// geometry, and are identical to the bit whichever backend this is.
    pub fn with_backend(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
        backend: TransformBackend,
    ) -> Result<Self> {
        let correlation = Correlation2::with_backend(shape_a, shape_b, window, padding, backend)?;
        let [rows, cols] = window.extent();
        let rows_in_a = (0..rows)
            .map(|index| overlap_in_first(shape_a[0], shape_b[0], window.shift_at([index, 0])[0]))
            .collect::<Vec<_>>();
        let cols_in_a = (0..cols)
            .map(|index| overlap_in_first(shape_a[1], shape_b[1], window.shift_at([0, index])[1]))
            .collect::<Vec<_>>();
        let rows_in_b = (0..rows)
            .map(|index| overlap_in_second(shape_a[0], shape_b[0], window.shift_at([index, 0])[0]))
            .collect::<Vec<_>>();
        let cols_in_b = (0..cols)
            .map(|index| overlap_in_second(shape_a[1], shape_b[1], window.shift_at([0, index])[1]))
            .collect::<Vec<_>>();
        let mut overlap = Array2::<u64>::zeros((rows, cols));
        for row in 0..rows {
            let (low, high) = rows_in_a[row];
            for col in 0..cols {
                let (left, right) = cols_in_a[col];
                overlap[[row, col]] = (high - low) as u64 * (right - left) as u64;
            }
        }
        Ok(Self {
            correlation,
            shape_a,
            shape_b,
            rows_in_a,
            cols_in_a,
            rows_in_b,
            cols_in_b,
            overlap,
            cross: Array2::zeros((rows, cols)),
            energy_a: Array2::zeros((rows, cols)),
            energy_b: Array2::zeros((rows, cols)),
            parts_a: Array2::zeros((shape_a[0], cols)),
            parts_b: Array2::zeros((shape_b[0], cols)),
            prefix: vec![0.0; shape_a[1].max(shape_b[1]) + 1],
        })
    }

    pub fn window(&self) -> ShiftWindow {
        self.correlation.window()
    }

    pub fn padded_shape(&self) -> [usize; 2] {
        self.correlation.padded_shape()
    }

    /// Which backend the transform underneath runs on.
    pub fn backend(&self) -> TransformBackend {
        self.correlation.backend()
    }

    /// Whether the padding is short enough to wrap; see [`Correlation2::wraps`].
    pub fn wraps(&self) -> bool {
        self.correlation.wraps()
    }

    /// The overlap sizes, which are a function of the geometry alone and are
    /// therefore known before any data arrives.
    pub fn overlap(&self) -> ArrayView2<'_, u64> {
        self.overlap.view()
    }

    /// A landscape buffer of this plan's shape, for [`Self::landscape_into`].
    pub fn empty_landscape(&self) -> Landscape {
        Landscape::empty(self.window())
    }

    /// The landscape for one pair of planes, allocated.
    ///
    /// [`Self::landscape_into`] with a fresh buffer. Convenient, and the right
    /// call when a caller wants one landscape; a caller in a loop should hold the
    /// buffer, for the same reason it holds the plan.
    pub fn landscape(&mut self, a: ArrayView2<f64>, b: ArrayView2<f64>) -> Result<Landscape> {
        let mut out = self.empty_landscape();
        self.landscape_into(a, b, &mut out)?;
        Ok(out)
    }

    /// The landscape for one pair of planes, into a caller-owned buffer.
    ///
    /// Allocates nothing. `out` must be a [`Landscape::empty`] of this plan's own
    /// [`Self::window`] — every value in it is overwritten, so its previous
    /// contents do not matter, but its **window** does and a mismatch is an error
    /// rather than a reshape: a landscape carries the lag convention its indices
    /// are read against, and silently replacing it would turn a caller's stale
    /// buffer into a wrong answer with the right shape.
    ///
    /// **This exists to match the rest of the module, and it is measured at
    /// `1.003x` — which is to say at nothing.** The plan already holds the
    /// twiddles, the spectra, the padded plane and the two-stage energy scratch,
    /// and a per-call `Landscape` was the one buffer left over; hoisting it saves
    /// two allocations of `window.extent()` against a transform that costs
    /// milliseconds. `tests/fft_correlation.rs`'s `the_speed_levers_are_measured`
    /// prints that ratio beside the ones that are real (plan reuse `1.6x`, the
    /// padded length `5x`), and it prints it precisely so that nobody reads this
    /// signature as a performance feature. Reach for it when a caller is holding
    /// a landscape anyway; `landscape` is not the slow path.
    pub fn landscape_into(
        &mut self,
        a: ArrayView2<f64>,
        b: ArrayView2<f64>,
        out: &mut Landscape,
    ) -> Result<()> {
        if a.dim() != (self.shape_a[0], self.shape_a[1]) {
            return Err(Error::ShapeMismatch {
                expected: self.shape_a.to_vec(),
                got: vec![a.dim().0, a.dim().1],
            });
        }
        if b.dim() != (self.shape_b[0], self.shape_b[1]) {
            return Err(Error::ShapeMismatch {
                expected: self.shape_b.to_vec(),
                got: vec![b.dim().0, b.dim().1],
            });
        }
        if out.window != self.window() {
            return Err(Error::invalid(format!(
                "a landscape buffer over lags {:?}..={:?} cannot hold a plan over {:?}..={:?}",
                out.window.lower(),
                out.window.upper(),
                self.window().lower(),
                self.window().upper()
            )));
        }
        let [rows, cols] = self.window().extent();
        if out.mean_squared.dim() != (rows, cols) || out.overlap.dim() != (rows, cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![out.mean_squared.dim().0, out.mean_squared.dim().1],
            });
        }

        self.correlation.correlate_into(a, b, &mut self.cross)?;
        rectangle_energies_into(
            a,
            &self.rows_in_a,
            &self.cols_in_a,
            &mut self.parts_a,
            &mut self.prefix,
            &mut self.energy_a,
        );
        rectangle_energies_into(
            b,
            &self.rows_in_b,
            &self.cols_in_b,
            &mut self.parts_b,
            &mut self.prefix,
            &mut self.energy_b,
        );

        for row in 0..rows {
            for col in 0..cols {
                let count = self.overlap[[row, col]];
                out.mean_squared[[row, col]] = if count == 0 {
                    // Not `0 / eps`. An empty overlap has no answer, and a large
                    // value keeps it out of an argmin where a floored division
                    // would make it a spurious global minimum.
                    f64::INFINITY
                } else {
                    let total = self.energy_a[[row, col]] + self.energy_b[[row, col]]
                        - 2.0 * self.cross[[row, col]];
                    total / count as f64
                };
            }
        }
        out.overlap.assign(&self.overlap);
        Ok(())
    }
}

/// `sum` of `values[x]^2` over every rectangle `row_ranges[i] x column_ranges[j]`,
/// into buffers the caller owns.
///
/// Two stages, and the split is about precision rather than only speed. Along
/// each row a **compensated (Neumaier) prefix** holds every partial sum to a
/// rounding of its own value however long the row is, so a rectangle's row part
/// is a difference of two well-rounded numbers. Down the rows the parts are
/// summed directly, which is a short accumulation over a bounded count.
///
/// `parts` is `values.rows() x column_ranges.len()`, `prefix` is at least
/// `values.cols() + 1` long, and `out` is
/// `row_ranges.len() x column_ranges.len()`. Every element of all three is
/// written before it is read, so their previous contents do not matter. There is
/// deliberately **no allocating wrapper**: the plan owns these three, and a
/// convenience that allocated them per call is the thing
/// [`SquaredDifference::landscape_into`] exists not to do.
fn rectangle_energies_into(
    values: ArrayView2<f64>,
    row_ranges: &[(usize, usize)],
    column_ranges: &[(usize, usize)],
    parts: &mut Array2<f64>,
    prefix: &mut [f64],
    out: &mut Array2<f64>,
) {
    let (value_rows, _) = values.dim();
    for row in 0..value_rows {
        let source = values.row(row);
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;
        prefix[0] = 0.0;
        for (column, &value) in source.iter().enumerate() {
            let term = value * value;
            let next = sum + term;
            compensation += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
            prefix[column + 1] = sum + compensation;
        }
        for (index, &(low, high)) in column_ranges.iter().enumerate() {
            parts[[row, index]] = prefix[high] - prefix[low];
        }
    }
    for (index, &(low, high)) in row_ranges.iter().enumerate() {
        for column in 0..column_ranges.len() {
            let mut total = 0.0;
            for row in low..high {
                total += parts[[row, column]];
            }
            out[[index, column]] = total;
        }
    }
}

/// The landscape by walking every lag's overlap and summing the squared
/// differences directly.
///
/// The oracle, and the acceptance bar: an FFT landscape that agrees with this
/// one to `1e-12` relative is right, and one that agrees to `1e-3` has a padding
/// or a windowing bug. `tests/fft_correlation.rs` is that comparison.
pub fn squared_difference_direct(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    window: ShiftWindow,
) -> Landscape {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let [rows, cols] = window.extent();
    let mut mean_squared = Array2::<f64>::zeros((rows, cols));
    let mut overlap = Array2::<u64>::zeros((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let shift = window.shift_at([row, col]);
            let (row_low, row_high) = overlap_in_first(a_rows, b_rows, shift[0]);
            let (col_low, col_high) = overlap_in_first(a_cols, b_cols, shift[1]);
            let count = (row_high - row_low) as u64 * (col_high - col_low) as u64;
            overlap[[row, col]] = count;
            if count == 0 {
                mean_squared[[row, col]] = f64::INFINITY;
                continue;
            }
            let mut total = 0.0;
            for x in row_low..row_high {
                let y = (x as isize - shift[0]) as usize;
                for u in col_low..col_high {
                    let v = (u as isize - shift[1]) as usize;
                    let difference = a[[x, u]] - b[[y, v]];
                    total += difference * difference;
                }
            }
            mean_squared[[row, col]] = total / count as f64;
        }
    }
    Landscape {
        window,
        mean_squared,
        overlap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The larger of two floats **by [`f64::total_cmp`]**, because this crate
    /// does not use `f64::max` anywhere: `f64::max(-0.0, 0.0)` is allowed to
    /// return either operand, and a rule that has exceptions for "only a
    /// magnitude" is a rule nobody applies.
    fn larger(left: f64, right: f64) -> f64 {
        if left.total_cmp(&right).is_gt() {
            left
        } else {
            right
        }
    }

    /// A deterministic plane with no symmetry on either axis: an xorshift
    /// sequence with a per-axis ramp on top, so that a transpose, a reflection
    /// and a one-element shift all change it.
    fn plane(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut state = seed | 1;
        Array2::from_shape_fn((rows, cols), |(row, col)| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            noise + 0.01 * row as f64 - 0.003 * col as f64
        })
    }

    #[test]
    fn the_round_trip_is_the_identity_to_a_bound_and_the_bound_is_stated() {
        // Not a power of two, not square, both extents odd, and 47 is prime, so
        // neither axis is a size a radix-2 transform would flatter.
        let source = plane(47, 90, 0x9E37_79B9_7F4A_7C15);
        let mut transform = RealTransform2::new([47, 90]).unwrap();
        let mut spectrum = transform.spectrum();
        transform.forward(source.view(), &mut spectrum).unwrap();
        let mut back = Array2::zeros((47, 90));
        transform.inverse(&mut spectrum, &mut back).unwrap();

        let mut worst = 0.0f64;
        for (&expected, &got) in source.iter().zip(back.iter()) {
            worst = larger(worst, (expected - got).abs());
        }
        println!("round trip worst absolute deviation: {worst:e}");
        // Measured: 1.1e-15 on the machine this was written on. The assertion is
        // two orders looser so it is a bound rather than a fingerprint, and it is
        // an assertion of a *tolerance* because a floating-point round trip is
        // not exact and claiming otherwise would be the worse error.
        assert!(
            worst < 1.0e-13,
            "round trip deviated by {worst:e}, which is not a rounding"
        );
        assert!(
            worst > 0.0,
            "an exactly zero deviation means the round trip \
             was not actually computed — this test would then pass on a pair of \
             no-ops"
        );
    }

    #[test]
    fn the_inverse_carries_the_one_over_n_and_the_forward_does_not() {
        // The convention, asserted rather than described: the zero-frequency
        // coefficient of the *forward* transform is the plain sum of the input,
        // unscaled.
        let source = plane(9, 14, 0xDEAD_BEEF_CAFE_0001);
        let mut transform = RealTransform2::new([9, 14]).unwrap();
        let mut spectrum = transform.spectrum();
        transform.forward(source.view(), &mut spectrum).unwrap();
        let total: f64 = source.iter().sum();
        assert!(
            (spectrum[[0, 0]].re - total).abs() < 1.0e-12,
            "the zero coefficient is {} and the sum is {total}",
            spectrum[[0, 0]].re
        );
        assert!(spectrum[[0, 0]].im.abs() < 1.0e-12);

        // And the negative control for the direction: normalising the forward
        // side instead would leave the round trip short by exactly N^2.
        let scale = (9 * 14) as f64;
        let mut back = Array2::zeros((9, 14));
        let mut wrong = spectrum.mapv(|value| value / scale);
        transform.inverse(&mut wrong, &mut back).unwrap();
        let ratio = source[[3, 5]] / back[[3, 5]];
        assert!(
            (ratio - scale).abs() < 1.0e-6,
            "normalising the wrong side should divide the round trip by N, got {ratio} against {scale}"
        );
    }

    #[test]
    fn a_zero_padded_forward_places_the_input_at_the_origin() {
        let small = plane(5, 7, 0x1234_5678_9ABC_DEF1);
        let mut padded = Array2::<f64>::zeros((11, 15));
        padded.slice_mut(ndarray::s![0..5, 0..7]).assign(&small);

        let mut transform = RealTransform2::new([11, 15]).unwrap();
        let mut from_small = transform.spectrum();
        transform
            .forward_zero_padded(small.view(), &mut from_small)
            .unwrap();
        let mut from_padded = transform.spectrum();
        transform.forward(padded.view(), &mut from_padded).unwrap();
        for (left, right) in from_small.iter().zip(from_padded.iter()) {
            assert!((left - right).norm() < 1.0e-13);
        }
    }

    #[test]
    fn the_minimal_length_reduces_to_the_sum_of_the_extents() {
        // The whole non-zero lag range asks for exactly the familiar rule.
        let window = ShiftWindow::new([-(19 - 1), -(23 - 1)], [13 - 1, 17 - 1]).unwrap();
        assert_eq!(
            minimal_wrap_free_length([13, 17], [19, 23], window),
            [13 + 19 - 1, 17 + 23 - 1]
        );
        // A narrow window asks for very much less.
        let narrow = ShiftWindow::symmetric([30, 30]);
        assert_eq!(
            minimal_wrap_free_length([96, 1304], [96, 1304], narrow),
            [126, 1334]
        );
        assert_eq!(
            Padding::Smooth.resolve([96, 1304], [96, 1304], narrow),
            [128, 1350]
        );
    }

    #[test]
    fn the_minimal_length_depends_on_where_the_window_sits_and_not_only_on_its_width() {
        // Both windows are 61 lags wide on the first axis and they do not ask for
        // the same padding, because `A - lo` and `B + hi` are one-sided. The
        // second is the geometry the module's first consumer actually has: its
        // two cuts sit 30 apart in global coordinates, so every lag it wants is
        // non-negative.
        let centred = ShiftWindow::new([-30, -30], [30, 30]).unwrap();
        let off_centre = ShiftWindow::new([0, -30], [60, 30]).unwrap();
        assert_eq!(centred.extent()[0], off_centre.extent()[0]);
        assert_eq!(
            minimal_wrap_free_length([96, 1304], [96, 1304], centred)[0],
            126
        );
        assert_eq!(
            minimal_wrap_free_length([96, 1304], [96, 1304], off_centre)[0],
            156
        );
        assert_eq!(
            Padding::Smooth.resolve([96, 1304], [96, 1304], off_centre),
            [160, 1350]
        );
        // And a rule that knew only the width would have to assume the worse
        // side, so the sharp rule is what saves the centred case its 32 rows.
        assert!(
            minimal_wrap_free_length([96, 1304], [96, 1304], centred)[0]
                < minimal_wrap_free_length([96, 1304], [96, 1304], off_centre)[0]
        );
    }

    #[test]
    fn the_next_smooth_length_is_smooth_and_least() {
        for length in [1usize, 2, 3, 7, 11, 13, 126, 157, 313, 1334, 2607, 2669] {
            let smooth = next_smooth_length(length);
            assert!(smooth >= length, "{smooth} < {length}");
            let mut residue = smooth;
            for factor in [2usize, 3, 5] {
                while residue % factor == 0 {
                    residue /= factor;
                }
            }
            assert_eq!(residue, 1, "{smooth} is not 5-smooth");
            for candidate in length..smooth {
                let mut residue = candidate;
                for factor in [2usize, 3, 5] {
                    while residue % factor == 0 {
                        residue /= factor;
                    }
                }
                assert_ne!(residue, 1, "{candidate} is smooth and below {smooth}");
            }
        }
        assert_eq!(next_smooth_length(126), 128);
        assert_eq!(next_smooth_length(1334), 1350);
        assert_eq!(next_smooth_length(157), 160);
    }

    #[test]
    fn the_overlap_count_is_the_product_of_the_axis_overlaps() {
        // Asymmetric extents and a lag that hangs off both ends in turn.
        assert_eq!(overlap_in_first(10, 6, 0), (0, 6));
        assert_eq!(overlap_in_first(10, 6, 3), (3, 9));
        assert_eq!(overlap_in_first(10, 6, 7), (7, 10));
        assert_eq!(overlap_in_first(10, 6, 10), (10, 10));
        assert_eq!(overlap_in_first(10, 6, -2), (0, 4));
        assert_eq!(overlap_in_first(10, 6, -6), (0, 0));
        assert_eq!(overlap_in_second(10, 6, 3), (0, 6));
        assert_eq!(overlap_in_second(10, 6, -2), (2, 6));
    }

    #[test]
    fn the_argmin_breaks_exact_ties_at_the_lowest_lag() {
        let window = ShiftWindow::new([-2, -3], [1, 0]).unwrap();
        let [rows, cols] = window.extent();
        let mut mean_squared = Array2::<f64>::from_elem((rows, cols), 5.0);
        let overlap = Array2::<u64>::from_elem((rows, cols), 7);
        // Three exact ties at the minimum, deliberately not adjacent.
        mean_squared[[1, 2]] = 1.0;
        mean_squared[[2, 0]] = 1.0;
        mean_squared[[3, 3]] = 1.0;
        let landscape = Landscape {
            window,
            mean_squared,
            overlap,
        };
        assert_eq!(landscape.argmin(), Some(([-1, -1], 1.0)));

        // And the rule is a *rule*: the same three values in a different order
        // still select the lexicographically smallest lag.
        let mut mean_squared = Array2::<f64>::from_elem((rows, cols), 5.0);
        mean_squared[[3, 3]] = 1.0;
        mean_squared[[2, 0]] = 1.0;
        mean_squared[[1, 2]] = 1.0;
        let landscape = Landscape {
            window,
            mean_squared,
            overlap: Array2::<u64>::from_elem((rows, cols), 7),
        };
        assert_eq!(landscape.argmin(), Some(([-1, -1], 1.0)));
    }

    #[test]
    fn the_argmin_ignores_lags_with_no_overlap_and_says_so_when_there_are_none() {
        let window = ShiftWindow::new([0, 0], [1, 1]).unwrap();
        let mut mean_squared = Array2::<f64>::from_elem((2, 2), 9.0);
        mean_squared[[0, 0]] = f64::INFINITY;
        let mut overlap = Array2::<u64>::from_elem((2, 2), 4);
        overlap[[0, 0]] = 0;
        let landscape = Landscape {
            window,
            mean_squared,
            overlap,
        };
        assert_eq!(landscape.argmin(), Some(([0, 1], 9.0)));

        let empty = Landscape {
            window,
            mean_squared: Array2::from_elem((2, 2), f64::INFINITY),
            overlap: Array2::zeros((2, 2)),
        };
        assert_eq!(empty.argmin(), None);
    }

    #[test]
    fn the_rectangle_energies_are_the_sums_they_claim_to_be() {
        let values = plane(13, 29, 0x0BAD_C0DE_F00D_0002);
        let rows = [(0usize, 13usize), (2, 11), (7, 8), (5, 5)];
        let columns = [(0usize, 29usize), (3, 19), (28, 29), (12, 12)];
        // The buffers a plan would hold, filled with rubbish first: every element
        // is written before it is read, and this is what says so.
        let mut parts = Array2::<f64>::from_elem((13, columns.len()), f64::NAN);
        let mut prefix = vec![f64::NAN; 30];
        let mut energies = Array2::<f64>::from_elem((rows.len(), columns.len()), f64::NAN);
        rectangle_energies_into(
            values.view(),
            &rows,
            &columns,
            &mut parts,
            &mut prefix,
            &mut energies,
        );
        for (index, &(low, high)) in rows.iter().enumerate() {
            for (column, &(left, right)) in columns.iter().enumerate() {
                let mut expected = 0.0;
                for row in low..high {
                    for slot in left..right {
                        expected += values[[row, slot]] * values[[row, slot]];
                    }
                }
                let got = energies[[index, column]];
                assert!(
                    (got - expected).abs() <= 1.0e-13 * larger(expected.abs(), 1.0),
                    "rectangle {low}..{high} x {left}..{right}: {got} against {expected}"
                );
            }
        }
    }

    // -------------------------------------------------------- the third axis --

    /// A deterministic volume with no symmetry on any axis, for the same reason
    /// [`plane`] has none: a transpose, a reflection and a one-voxel shift must
    /// all change it, or a test comparing two transforms of it says nothing.
    fn volume(shape: [usize; 3], seed: u64) -> Array3<f64> {
        let mut state = seed | 1;
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(a, b, c)| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            noise + 0.01 * a as f64 - 0.003 * b as f64 + 0.007 * c as f64
        })
    }

    /// The transform written from its definition: one sum per coefficient over
    /// every voxel. The oracle, and the only thing in this file that does not
    /// go through a library.
    fn dft3_direct(input: &Array3<f64>, index: [usize; 3]) -> Complex<f64> {
        let (d0, d1, d2) = input.dim();
        let mut total = Complex::new(0.0, 0.0);
        for a in 0..d0 {
            for b in 0..d1 {
                for c in 0..d2 {
                    let phase = -2.0
                        * std::f64::consts::PI
                        * ((index[0] * a) as f64 / d0 as f64
                            + (index[1] * b) as f64 / d1 as f64
                            + (index[2] * c) as f64 / d2 as f64);
                    total += input[[a, b, c]] * Complex::new(phase.cos(), phase.sin());
                }
            }
        }
        total
    }

    #[test]
    fn the_volume_transform_is_the_definition_coefficient_by_coefficient() {
        // Odd on two axes and prime on one, so no axis is a size a radix-2
        // transform would flatter and no two axes can be confused for each
        // other.
        let shape = [5usize, 7, 6];
        let source = volume(shape, 0x51ED_270F_A2C1_0003);
        let mut transform = RealTransform3::new(shape).unwrap();
        let mut spectrum = transform.spectrum();
        transform
            .forward_zero_padded(source.view(), &mut spectrum)
            .unwrap();

        let mut worst = 0.0f64;
        for a in 0..shape[0] {
            for b in 0..shape[1] {
                for c in 0..spectrum_width(shape[2]) {
                    let expected = dft3_direct(&source, [a, b, c]);
                    let got = spectrum[[a, b, c]];
                    worst = larger(worst, (got - expected).norm());
                }
            }
        }
        println!("volume transform worst coefficient deviation: {worst:e}");
        assert!(
            worst < 1.0e-11,
            "the transform deviates from its definition by {worst:e}"
        );
        // **Liveness.** The comparison above is only a claim if the oracle can
        // tell two volumes apart at all; a direct transform with a sign error in
        // the phase, or one that ignored its index, would agree with itself.
        let other = volume(shape, 0x51ED_270F_A2C1_0004);
        let mut apart = 0.0f64;
        for a in 0..shape[0] {
            for b in 0..shape[1] {
                for c in 0..spectrum_width(shape[2]) {
                    apart = larger(
                        apart,
                        (dft3_direct(&source, [a, b, c]) - dft3_direct(&other, [a, b, c])).norm(),
                    );
                }
            }
        }
        assert!(
            apart > 1.0,
            "the oracle gives two different volumes the same spectrum, so the \
             agreement above is not evidence"
        );
    }

    #[test]
    fn the_volume_round_trip_is_the_identity_to_a_bound_and_the_bound_is_stated() {
        let shape = [17usize, 23, 30];
        let source = volume(shape, 0x9E37_79B9_7F4A_7C17);
        let mut transform = RealTransform3::new(shape).unwrap();
        let mut spectrum = transform.spectrum();
        transform
            .forward_zero_padded(source.view(), &mut spectrum)
            .unwrap();
        let mut back = Array3::zeros((shape[0], shape[1], shape[2]));
        transform.inverse(&mut spectrum, &mut back).unwrap();

        let mut worst = 0.0f64;
        for (&expected, &got) in source.iter().zip(back.iter()) {
            worst = larger(worst, (expected - got).abs());
        }
        println!("volume round trip worst absolute deviation: {worst:e}");
        // Measured: 1.4e-15 on the machine this was written on, which is the
        // same order the two-dimensional round trip reports over a comparable
        // element count. Two orders looser here so it is a bound and not a
        // fingerprint.
        assert!(
            worst < 1.0e-13,
            "round trip deviated by {worst:e}, which is not a rounding"
        );
        assert!(
            worst > 0.0,
            "an exactly zero deviation means the round trip was not computed at \
             all — this assertion would pass on a pair of no-ops"
        );
    }

    #[test]
    fn the_volume_inverse_carries_the_one_over_n_and_the_forward_does_not() {
        let shape = [4usize, 5, 6];
        // A **positive** field, not the zero-mean fixture the other cases use:
        // a volume whose sum is near zero cannot tell an unnormalised forward
        // transform from a scaled one, which the liveness assertion below
        // enforces and which the plain fixture fails.
        let source = volume(shape, 0xDEAD_BEEF_CAFE_0007).mapv(|value| value + 5.0);
        let mut transform = RealTransform3::new(shape).unwrap();
        let mut spectrum = transform.spectrum();
        transform
            .forward_zero_padded(source.view(), &mut spectrum)
            .unwrap();
        let sum: f64 = source.iter().sum();
        let zero = spectrum[[0, 0, 0]];
        assert!(
            (zero.re - sum).abs() < 1.0e-12 * larger(sum.abs(), 1.0) && zero.im.abs() < 1.0e-12,
            "the forward zero bin is {zero} and the plain sum is {sum}: the \
             forward direction must be unnormalised"
        );
        // **Liveness.** A volume whose sum is zero cannot tell an unnormalised
        // forward transform from one carrying any scale at all, so the fixture
        // above has to have a sum far from zero — asserted, not assumed.
        let count = (shape[0] * shape[1] * shape[2]) as f64;
        assert!(
            sum.abs() > count / 4.0,
            "the fixture's sum is {sum} over {count} voxels, which is too near \
             zero for the zero bin to be evidence of a normalisation"
        );
    }

    #[test]
    fn zero_padding_a_volume_is_the_transform_of_the_padded_volume() {
        // The whole reason `forward_zero_padded` exists: a wrap-free length is
        // longer than the data, and a caller must not have to allocate and zero
        // the larger volume itself.
        let small = [3usize, 4, 5];
        let large = [8usize, 9, 12];
        let source = volume(small, 0x0BAD_F00D_1234_5679);
        let mut padded = Array3::zeros((large[0], large[1], large[2]));
        for a in 0..small[0] {
            for b in 0..small[1] {
                for c in 0..small[2] {
                    padded[[a, b, c]] = source[[a, b, c]];
                }
            }
        }
        let mut transform = RealTransform3::new(large).unwrap();
        let mut from_small = transform.spectrum();
        transform
            .forward_zero_padded(source.view(), &mut from_small)
            .unwrap();
        let mut from_padded = transform.spectrum();
        transform
            .forward_zero_padded(padded.view(), &mut from_padded)
            .unwrap();
        for (&a, &b) in from_small.iter().zip(from_padded.iter()) {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits()),
                "padding inside the transform must be the same arithmetic as \
                 padding outside it, to the bit"
            );
        }
        // **Liveness.** Byte-equality above proves nothing if both spectra are
        // zero, which is what a `forward_zero_padded` that ignored its input
        // would produce.
        let energy: f64 = from_small.iter().map(|value| value.norm_sqr()).sum();
        assert!(
            energy > 1.0,
            "both spectra are ~zero ({energy:e}), so their equality is not evidence"
        );
    }

    #[test]
    fn a_cloned_volume_transform_computes_the_same_answer() {
        // The property the block op below depends on: `apply` takes `&self` and
        // a transform needs `&mut`, so the op clones a template per block. A
        // clone that re-planned, or that shared scratch, would make the answer a
        // function of which thread ran it.
        let shape = [6usize, 6, 8];
        let source = volume(shape, 0x0102_0304_0506_0709);
        let mut original = RealTransform3::new(shape).unwrap();
        let mut copy = original.clone();
        let mut left = original.spectrum();
        let mut right = copy.spectrum();
        original
            .forward_zero_padded(source.view(), &mut left)
            .unwrap();
        copy.forward_zero_padded(source.view(), &mut right).unwrap();
        for (&a, &b) in left.iter().zip(right.iter()) {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits()),
                "a clone must be a second working set over one set of twiddles"
            );
        }
        let energy: f64 = left.iter().map(|value| value.norm_sqr()).sum();
        assert!(
            energy > 1.0,
            "both spectra are ~zero, so equality says nothing"
        );
    }

    #[test]
    fn an_empty_extent_is_refused_and_an_oversized_input_is_refused() {
        assert!(RealTransform3::new([0, 4, 4]).is_err());
        assert!(RealTransform3::new([4, 0, 4]).is_err());
        assert!(RealTransform3::new([4, 4, 0]).is_err());
        let mut transform = RealTransform3::new([4, 4, 4]).unwrap();
        let mut spectrum = transform.spectrum();
        let too_big = Array3::<f64>::zeros((5, 4, 4));
        let message = transform
            .forward_zero_padded(too_big.view(), &mut spectrum)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("does not fit"),
            "an input longer than the transform must be refused by name, got {message}"
        );
        let mut wrong = Array3::from_elem((4, 4, 4), Complex::new(0.0, 0.0));
        let fits = Array3::<f64>::zeros((4, 4, 4));
        assert!(transform
            .forward_zero_padded(fits.view(), &mut wrong)
            .is_err());
    }
}
