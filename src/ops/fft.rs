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
// The dependency
// --------------
// `rustfft` and `realfft`, and `Cargo.toml` records what was measured against
// what before they were chosen.

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
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

/// A two-dimensional real-input discrete Fourier transform of a fixed shape,
/// with its plans and its scratch built once.
///
/// **The plans are the point.** Building them computes twiddle factors and picks
/// an algorithm per length, and a consumer transforming hundreds of planes of
/// one shape should pay for that once. Cloning shares the plans through an
/// `Arc` and allocates fresh scratch, which is what makes one plan usable from
/// several threads.
///
/// See this module's header for the normalisation convention: **the forward
/// direction is unnormalised and the inverse carries `1/N`**.
pub struct RealTransform2 {
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

impl Clone for RealTransform2 {
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

impl std::fmt::Debug for RealTransform2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealTransform2")
            .field("shape", &[self.rows, self.cols])
            .finish_non_exhaustive()
    }
}

impl RealTransform2 {
    /// Plan a transform of `shape`. Both extents must be non-zero.
    pub fn new(shape: [usize; 2]) -> Result<Self> {
        let [rows, cols] = shape;
        if rows == 0 || cols == 0 {
            return Err(Error::invalid(format!(
                "a transform needs a non-empty shape, got {rows} x {cols}"
            )));
        }
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
        Ok(Self {
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
        })
    }

    /// The shape of the real plane this transforms.
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// The shape of the half-spectrum: `[rows, cols / 2 + 1]`.
    pub fn spectrum_shape(&self) -> [usize; 2] {
        [self.rows, spectrum_width(self.cols)]
    }

    /// A zeroed spectrum of the right shape, for a caller that wants to allocate
    /// once and reuse.
    pub fn spectrum(&self) -> Spectrum {
        let [rows, cols] = self.spectrum_shape();
        Array2::zeros((rows, cols))
    }

    /// Forward transform. `input` must be exactly [`Self::shape`].
    pub fn forward(&mut self, input: ArrayView2<f64>, out: &mut Spectrum) -> Result<()> {
        if input.dim() != (self.rows, self.cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![self.rows, self.cols],
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
        let (input_rows, input_cols) = input.dim();
        if input_rows > self.rows || input_cols > self.cols {
            return Err(Error::invalid(format!(
                "an input of {input_rows} x {input_cols} does not fit a {} x {} transform",
                self.rows, self.cols
            )));
        }
        let width = spectrum_width(self.cols);
        if out.dim() != (self.rows, width) {
            return Err(Error::ShapeMismatch {
                expected: vec![self.rows, width],
                got: vec![out.dim().0, out.dim().1],
            });
        }
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
        let data = out
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("the spectrum must be contiguous and row-major"))?;
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

    /// Inverse transform, carrying the `1/N` this convention puts on this side.
    ///
    /// `spectrum` is **clobbered**: it is transformed in place along the first
    /// axis and then consumed row by row. That is stated rather than hidden
    /// behind a copy, because the caller almost always has no further use for it
    /// and a copy of a spectrum is not free.
    pub fn inverse(&mut self, spectrum: &mut Spectrum, out: &mut Array2<f64>) -> Result<()> {
        let width = spectrum_width(self.cols);
        if spectrum.dim() != (self.rows, width) {
            return Err(Error::ShapeMismatch {
                expected: vec![self.rows, width],
                got: vec![spectrum.dim().0, spectrum.dim().1],
            });
        }
        if out.dim() != (self.rows, self.cols) {
            return Err(Error::ShapeMismatch {
                expected: vec![self.rows, self.cols],
                got: vec![out.dim().0, out.dim().1],
            });
        }
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
        let data = spectrum
            .as_slice_mut()
            .ok_or_else(|| Error::invalid("the spectrum must be contiguous and row-major"))?;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// [`minimal_wrap_free_length`] exactly. Correct, and often a hostile length
    /// for a mixed-radix transform.
    Minimal,
    /// `Minimal` rounded up per axis to the next `5`-smooth integer. Correct for
    /// the same reason — any length at or above the minimum is — and several
    /// times faster. The default.
    Smooth,
    /// A length the caller chose. **Below [`minimal_wrap_free_length`] the
    /// answer wraps**, and [`Correlation2::wraps`] says whether it does.
    Exact([usize; 2]),
}

impl Default for Padding {
    fn default() -> Self {
        Self::Smooth
    }
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
    /// `window`.
    pub fn new(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
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
        let transform = RealTransform2::new(padded)?;
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
}

impl SquaredDifference {
    /// Plan a landscape between planes of `shape_a` and `shape_b` over `window`.
    pub fn new(
        shape_a: [usize; 2],
        shape_b: [usize; 2],
        window: ShiftWindow,
        padding: Padding,
    ) -> Result<Self> {
        let correlation = Correlation2::new(shape_a, shape_b, window, padding)?;
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
        })
    }

    pub fn window(&self) -> ShiftWindow {
        self.correlation.window()
    }

    pub fn padded_shape(&self) -> [usize; 2] {
        self.correlation.padded_shape()
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

    /// The landscape for one pair of planes.
    pub fn landscape(&mut self, a: ArrayView2<f64>, b: ArrayView2<f64>) -> Result<Landscape> {
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
        self.correlation.correlate_into(a, b, &mut self.cross)?;
        let energy_a = rectangle_energies(a, &self.rows_in_a, &self.cols_in_a);
        let energy_b = rectangle_energies(b, &self.rows_in_b, &self.cols_in_b);

        let [rows, cols] = self.window().extent();
        let mut mean_squared = Array2::<f64>::zeros((rows, cols));
        for row in 0..rows {
            for col in 0..cols {
                let count = self.overlap[[row, col]];
                mean_squared[[row, col]] = if count == 0 {
                    f64::INFINITY
                } else {
                    let total =
                        energy_a[[row, col]] + energy_b[[row, col]] - 2.0 * self.cross[[row, col]];
                    total / count as f64
                };
            }
        }
        Ok(Landscape {
            window: self.window(),
            mean_squared,
            overlap: self.overlap.clone(),
        })
    }
}

/// `sum` of `values[x]^2` over every rectangle `row_ranges[i] x column_ranges[j]`.
///
/// Two stages, and the split is about precision rather than only speed. Along
/// each row a **compensated (Neumaier) prefix** holds every partial sum to a
/// rounding of its own value however long the row is, so a rectangle's row part
/// is a difference of two well-rounded numbers. Down the rows the parts are
/// summed directly, which is a short accumulation over a bounded count.
fn rectangle_energies(
    values: ArrayView2<f64>,
    row_ranges: &[(usize, usize)],
    column_ranges: &[(usize, usize)],
) -> Array2<f64> {
    let (value_rows, value_cols) = values.dim();
    let mut parts = Array2::<f64>::zeros((value_rows, column_ranges.len()));
    let mut prefix = vec![0.0f64; value_cols + 1];
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
    let mut out = Array2::<f64>::zeros((row_ranges.len(), column_ranges.len()));
    for (index, &(low, high)) in row_ranges.iter().enumerate() {
        for column in 0..column_ranges.len() {
            let mut total = 0.0;
            for row in low..high {
                total += parts[[row, column]];
            }
            out[[index, column]] = total;
        }
    }
    out
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
        // A narrow window asks for very much less, and this is the geometry the
        // consumer actually has.
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
        let energies = rectangle_energies(values.view(), &rows, &columns);
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
}
