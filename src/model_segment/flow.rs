//! **Phase 1 and 2: the flow field, as a function of the pixels alone.**
//!
//! A tiled network is not automatically a blockable operator. Cellpose's is
//! not, and the reason is worth stating precisely because everything here is
//! shaped by it.
//!
//! CP-SAM is a **fixed 256x256 in, 256x256x3 out** function — the positional
//! embeddings are sized for exactly that, and asking for another size fails in
//! the network rather than degrading. So the network itself is a pure function
//! `f` of a window. The only freedom is *where the windows go*, and cellpose
//! chooses:
//!
//! ```text
//! ny     = ceil((1 + 2*overlap) * L / 256)      // L is the buffer's length
//! ystart = linspace(0, L - 256, ny)
//! ```
//!
//! The stride is therefore a **non-integer function of the buffer size** —
//! 199.1 px at `L = 2048`, 182.9 at `L = 1536`, 202.1 at `L = 4096`. Hand the
//! same pixels to a different-sized buffer and every window lands somewhere
//! new. That would only matter at seams if `f` were translation-equivariant,
//! but CP-SAM has global attention and absolute positional embeddings, so the
//! same pixel at window-offset `(3, 7)` gets a different answer than at
//! `(200, 150)`. The disagreement is therefore **uniform across the image**,
//! which is what measurement showed: 42.3% of cells within 96 px of a block
//! seam were missed, against 43.5% overall.
//!
//! # What this module changes
//!
//! [`WindowGrid`] anchors the windows to **absolute image coordinates** with a
//! **fixed integer stride**. Every pixel then sits at the same offset in the
//! same set of windows no matter which block reached it, and since `f` is a
//! pure function, the network's output per pixel becomes a function of the
//! pixels alone.
//!
//! # Why blending is the easy half
//!
//! Cellpose combines overlapping windows with a **weighted mean**, not a
//! winner-take-all:
//!
//! ```text
//! flow[p] = sum_j f_j(p) * w(p - o_j)  /  sum_j w(p - o_j)
//! ```
//!
//! `w` is `transforms::taper_mask`, computed once and identical for every
//! window — a function of the offset *within* a window and of nothing else. So
//! there is no "which tile wins" rule to make deterministic. Both the numerator
//! and the denominator are plain sums, which makes them associative and
//! commutative, which is exactly the accumulate-and-join shape `blockflow` is
//! built to decompose: a block that sees part of the window set contributes a
//! partial `(sum w*f, sum w)` that adds exactly to another block's.
//!
//! Two things still have to be arranged, and both are here:
//!
//! * **a block must see every window covering its core**, or its sums are
//!   short. That is a halo of one window — see [`WindowGrid::halo`] — and it is
//!   the ordinary `blockflow` reach argument;
//! * **summation order**, because `f32` addition does not associate. Windows
//!   are accumulated in ascending absolute origin order, which is a total order
//!   independent of the block, so the sum is bit-identical and not merely
//!   close.

use crate::error::{Error, Result};

/// Where the network's windows go, in absolute image coordinates.
///
/// The grid is a pure function of the **volume** and the two parameters, and of
/// nothing about how the volume is cut into blocks. That is the whole point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowGrid {
    /// The network's input size. Fixed by the model — 256 for CP-SAM.
    window: usize,
    /// Distance between consecutive window origins. An **integer**, unlike
    /// cellpose's own, and independent of the buffer.
    stride: usize,
}

impl WindowGrid {
    /// `stride` must be a positive integer no larger than `window`; a larger
    /// one would leave gaps no window covers, which is a hole rather than a
    /// coarser tiling.
    pub fn new(window: usize, stride: usize) -> Result<Self> {
        if window == 0 || stride == 0 || stride > window {
            return Err(Error::InvalidArgument(format!(
                "a window grid wants 0 < stride <= window and got stride {stride}, window \
                 {window}. A stride wider than the window leaves pixels no window covers, and \
                 their flow would be a division by zero rather than a coarser answer."
            )));
        }
        Ok(Self { window, stride })
    }

    /// Cellpose's own overlap fraction, as a stride. `0.1` gives 230.
    ///
    /// Offered so a caller can ask for the amount of overlap cellpose's
    /// defaults imply, without inheriting the buffer-dependent placement that
    /// comes with them.
    pub fn from_overlap(window: usize, overlap: f64) -> Result<Self> {
        let stride = ((1.0 - overlap) * window as f64).round() as usize;
        Self::new(window, stride.max(1))
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The halo a block needs so that every window covering its core is wholly
    /// inside its buffer.
    ///
    /// A window covering pixel `p` has an origin in `(p - window, p]`, and
    /// needs pixels up to `origin + window`. So the buffer must reach
    /// `window - 1` beyond the core on each side; this rounds to `window`,
    /// which is also what keeps the arithmetic honest at `stride == window`.
    ///
    /// **This is a lower bound for the flow field only.** Turning flows into
    /// masks follows them for `niter` steps, and that travel adds to the halo —
    /// see the module documentation of the phase that does it.
    pub fn halo(&self) -> usize {
        self.window
    }

    /// Every window origin along one axis of a volume of `length`.
    ///
    /// Origins are multiples of the stride, plus — when the last one leaves a
    /// tail uncovered — a final window flush with the volume's end. That final
    /// window is off the regular grid, and deliberately: it is a function of
    /// the **volume**, which every block agrees about, rather than of a buffer,
    /// which they do not. Cellpose's own `linspace` placement makes *every*
    /// window a function of the buffer, which is the defect this replaces.
    pub fn origins(&self, length: usize) -> Vec<usize> {
        if length <= self.window {
            return vec![0];
        }
        let mut origins: Vec<usize> = (0..)
            .map(|k| k * self.stride)
            .take_while(|origin| origin + self.window <= length)
            .collect();
        let last = length - self.window;
        if origins.last().copied() != Some(last) {
            origins.push(last);
        }
        origins
    }

    /// The window origins along one axis that cover any of `start..end`.
    pub fn covering(&self, length: usize, start: usize, end: usize) -> Vec<usize> {
        self.origins(length)
            .into_iter()
            .filter(|origin| *origin < end && origin + self.window > start)
            .collect()
    }

    /// The total blending weight at absolute position `at`, from every window
    /// of the grid that covers it.
    ///
    /// The quantity a decomposed run has to reproduce exactly. It needs no
    /// network, so it is the cheapest possible check that the geometry is
    /// right: if two block sizes disagree here, they will disagree about the
    /// flow field too, and no GPU was needed to find out.
    ///
    /// `taper` is the within-window weight profile along one axis; the 2-D
    /// weight is the product of the two axes', which is how `taper_mask` builds
    /// it.
    pub fn weight_at(&self, volume: [usize; 2], at: [usize; 2], taper: &[f64]) -> f64 {
        let mut total = 0.0;
        for origin_y in self.covering(volume[0], at[0], at[0] + 1) {
            for origin_x in self.covering(volume[1], at[1], at[1] + 1) {
                total += taper[at[0] - origin_y] * taper[at[1] - origin_x];
            }
        }
        total
    }
}

/// Running the network over an absolutely-anchored window grid.
///
/// Separated from [`WindowGrid`] so the geometry — the half that can be proved
/// and is tested above without a GPU — does not need a model to be reasoned
/// about.
#[cfg(feature = "cellpose")]
pub mod net {
    use super::WindowGrid;
    use crate::error::{Error, Result};
    use cellpose::core::InferenceNetwork;
    use ndarray::{Array2, Array3, ArrayView2};

    /// What a block contributes to the flow field: the two sums, unnormalised.
    ///
    /// Unnormalised because that is the form that **adds**. A block holding
    /// `(sum w*f, sum w)` over part of the window set can be combined with
    /// another block's by adding both, and the division happens once at the
    /// end. A block that divided early would be contributing a mean, and means
    /// do not add.
    pub struct Accumulated {
        /// `sum_j f_j * w_j`, one plane per network output channel.
        pub weighted: Array3<f64>,
        /// `sum_j w_j`.
        pub weight: Array2<f64>,
    }

    impl Accumulated {
        /// `weighted / weight`, the blended flow field.
        ///
        /// A pixel no window covered is `NaN` rather than zero — the same
        /// choice cellpose's `average_tiles` makes, and for the same reason: a
        /// zero flow is a legitimate value and would be indistinguishable from
        /// an answer nobody computed.
        pub fn blend(&self) -> Array3<f32> {
            let (channels, height, width) = self.weighted.dim();
            let mut out = Array3::<f32>::zeros((channels, height, width));
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let weight = self.weight[[y, x]];
                        out[[c, y, x]] = if weight > 0.0 {
                            (self.weighted[[c, y, x]] / weight) as f32
                        } else {
                            f32::NAN
                        };
                    }
                }
            }
            out
        }
    }

    /// Run every window of `grid` that lies wholly inside `buffer`, and
    /// accumulate the taper-weighted sums.
    ///
    /// `buffer` is already normalised into the units the network expects.
    /// `origin` is where the buffer sits in the volume, which is what makes the
    /// window grid absolute.
    ///
    /// **Windows are visited in ascending `(origin_y, origin_x)` order**, which
    /// is a total order on the grid and therefore the same order in every
    /// block. `f32` addition does not associate, so a fixed order is the
    /// difference between a sum that is reproducible and one that is merely
    /// close.
    pub fn accumulate(
        network: &dyn InferenceNetwork,
        buffer: ArrayView2<'_, f32>,
        origin: [usize; 2],
        volume: [usize; 2],
        grid: WindowGrid,
        taper: &[f64],
        batch: usize,
    ) -> Result<Accumulated> {
        let (height, width) = buffer.dim();
        let window = grid.window();
        let channels = network.output_channels();

        // The windows this buffer can run: on the absolute grid, and wholly
        // inside the buffer. For a core pixel this is exactly the set covering
        // it, provided the halo is at least `grid.halo()`.
        let mut origins: Vec<[usize; 2]> = Vec::new();
        for oy in grid.origins(volume[0]) {
            if oy < origin[0] || oy + window > origin[0] + height {
                continue;
            }
            for ox in grid.origins(volume[1]) {
                if ox < origin[1] || ox + window > origin[1] + width {
                    continue;
                }
                origins.push([oy, ox]);
            }
        }
        if origins.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "a buffer of {height}x{width} at {origin:?} holds no whole {window}px window. \
                 The halo must be at least {} for a block to run the windows covering its core.",
                grid.halo()
            )));
        }

        let mut weighted = Array3::<f64>::zeros((channels, height, width));
        let mut weight = Array2::<f64>::zeros((height, width));

        for group in origins.chunks(batch.max(1)) {
            // The network takes 3 channels; a grayscale image goes in channel 0
            // with the others left at zero, which is what `convert_image` does.
            // cellpose's `ndarray`, which is 0.16 against this crate's 0.17.
            let mut tiles = ndarray16::Array4::<f32>::zeros((group.len(), 3, window, window));
            for (slot, at) in group.iter().enumerate() {
                for y in 0..window {
                    for x in 0..window {
                        tiles[[slot, 0, y, x]] =
                            buffer[[at[0] - origin[0] + y, at[1] - origin[1] + x]];
                    }
                }
            }

            let out = network
                .forward_tiles(&tiles, None)
                .map_err(|error| Error::backend(format!("cellpose: {error}")))?;

            for (slot, at) in group.iter().enumerate() {
                let (top, left) = (at[0] - origin[0], at[1] - origin[1]);

                // The channel loop is **outermost**, not innermost. Both arrays
                // hold the channel in their slowest axis, so a channel-innermost
                // loop jumps a plane's worth of elements three times per pixel
                // and misses cache on every one of them. This ordering keeps the
                // innermost loop contiguous in both.
                //
                // It changes no answer: a pixel still receives exactly one term
                // per window, and the windows are still visited in ascending
                // origin order, so the per-pixel summation order is untouched.
                for y in 0..window {
                    let wy = taper[y];
                    let mut row = weight.row_mut(top + y);
                    let row = row.as_slice_mut().expect("a contiguous row");
                    for x in 0..window {
                        row[left + x] += wy * taper[x];
                    }
                }
                for c in 0..channels {
                    for y in 0..window {
                        let wy = taper[y];
                        for x in 0..window {
                            weighted[[c, top + y, left + x]] +=
                                out.predictions[[slot, c, y, x]] as f64 * wy * taper[x];
                        }
                    }
                }
            }
        }

        Ok(Accumulated { weighted, weight })
    }

    /// **Is the network a pure function of its window?**
    ///
    /// Everything anchored windows buy rests on `f(window)` being the same
    /// whichever block ran it. That is true of the mathematics and is not
    /// automatically true of the hardware: `forward_tiles` runs a *batch*, and
    /// a different batch shape can make cuBLAS or cuDNN pick a different kernel
    /// with a different reduction order. The default dtype is `bfloat16`, which
    /// has eight bits of mantissa, so any such difference is coarse.
    ///
    /// So it is measured rather than assumed. `window` is run alone and again
    /// at several positions inside larger batches; the answer is the largest
    /// absolute difference from the batch-of-one result, and how many elements
    /// differ at all.
    ///
    /// A zero here means anchored windows give a bit-identical flow field. A
    /// small non-zero means they give one that agrees to that tolerance, which
    /// may still be enough — whether it is depends on whether the difference
    /// survives into the masks, which is a question for the phase that makes
    /// them and not for this one.
    pub fn batch_purity(
        network: &dyn InferenceNetwork,
        window: ArrayView2<'_, f32>,
        batches: &[usize],
    ) -> Result<Vec<(usize, usize, f64, usize)>> {
        let (height, width) = window.dim();
        let channels = network.output_channels();

        let run = |count: usize, slot: usize| -> Result<Vec<f32>> {
            let mut tiles = ndarray16::Array4::<f32>::zeros((count, 3, height, width));
            for other in 0..count {
                // The other members of the batch are deliberately *different*
                // data — a batch of identical tiles would not exercise whatever
                // the batch shape changes.
                let shift = (other as f32) * 0.013;
                for y in 0..height {
                    for x in 0..width {
                        tiles[[other, 0, y, x]] = if other == slot {
                            window[[y, x]]
                        } else {
                            window[[y, x]] + shift
                        };
                    }
                }
            }
            let out = network
                .forward_tiles(&tiles, None)
                .map_err(|error| Error::backend(format!("cellpose: {error}")))?;
            let mut values = Vec::with_capacity(channels * height * width);
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        values.push(out.predictions[[slot, c, y, x]]);
                    }
                }
            }
            Ok(values)
        };

        let alone = run(1, 0)?;
        let mut report = Vec::new();
        for &count in batches {
            for slot in [0usize, count / 2, count - 1] {
                if slot >= count {
                    continue;
                }
                let got = run(count, slot)?;
                let mut worst = 0.0f64;
                let mut differing = 0usize;
                for (a, b) in alone.iter().zip(got.iter()) {
                    if a.to_bits() != b.to_bits() {
                        differing += 1;
                        worst = worst.max((*a as f64 - *b as f64).abs());
                    }
                }
                report.push((count, slot, worst, differing));
            }
        }
        Ok(report)
    }

    /// Cellpose's own taper, along one axis.
    ///
    /// Taken from `transforms::taper_mask` rather than re-derived: the 2-D mask
    /// is the outer product of this with itself, so one axis is all that a
    /// separable accumulation needs.
    pub fn taper_profile(window: usize) -> Vec<f64> {
        let mask = cellpose::transforms::taper_mask(window, window, 7.5);
        // The mask is `m[i] * m[j]`, so its diagonal is `m[i]^2` and its first
        // row is `m[0] * m[j]`. Recovering one axis from the middle row is
        // exact and needs no assumption about normalisation.
        let middle = window / 2;
        let scale = mask[[middle, middle]].sqrt();
        (0..window).map(|i| mask[[middle, i]] / scale).collect()
    }
}

/// **Following flows to their attractors, without the per-sample overhead.**
///
/// The Euler integration is the CPU half of turning a flow field into masks,
/// and it is memory-bound rather than arithmetic-bound: measured on this
/// machine, 1.9 s per megapixel even when the field fits in L3, and 2.4 s per
/// megapixel when it does not (a 2048² buffer is 32 MB of flows against a 27.5
/// MiB cache).
///
/// Cellpose's own inner loop pays four avoidable costs per pixel per iteration:
///
/// * `grid_sample_align_corners_false_2d` is called **twice**, once per flow
///   component, and each call recomputes `h / (h - 1)` — *a division* — and the
///   identical coordinate transform;
/// * the two components live `h * w` floats apart, so each sample touches two
///   unrelated cache-line families instead of one;
/// * every one of the eight corner reads is a bounds-checked three-dimensional
///   index, so the stride arithmetic is redone per access;
/// * the trajectory is clamped into the array on every step, which is a
///   `min`/`max` pair that only the last one needs.
///
/// This does the same arithmetic with the constants hoisted, the transform
/// computed once, the field **interleaved** so one cache line carries `(dy,
/// dx)`, and the corner reads unchecked behind a single explicit bound.
///
/// It is here rather than in `cellpose-rs` because it is a measurement first:
/// [`interleave`] and this function together answer "how much is on the table"
/// without committing anyone to a refactor of somebody else's inner loop. If
/// the number is worth it, this is the shape the change would take.
pub mod steps {
    /// The flow field as `(dy, dx)` pairs, which is the layout the follower
    /// wants and the one `Array3` does not have.
    pub fn interleave(d_p: &ndarray::ArrayView3<'_, f32>) -> Vec<f32> {
        let (_, height, width) = d_p.dim();
        let mut out = vec![0.0f32; 2 * height * width];
        for y in 0..height {
            for x in 0..width {
                out[2 * (y * width + x)] = d_p[[0, y, x]];
                out[2 * (y * width + x) + 1] = d_p[[1, y, x]];
            }
        }
        out
    }

    /// Follow each seed for `niter` steps. Returns the final `(y, x)` of each.
    ///
    /// The arithmetic is cellpose's, including the `align_corners = false`
    /// coordinate transform and the zero padding outside the array, so the
    /// answer is the same to the last bit that the operation order allows.
    pub fn follow(
        field: &[f32],
        height: usize,
        width: usize,
        seeds: &[(usize, usize)],
        niter: usize,
    ) -> Vec<(f32, f32)> {
        use rayon::prelude::*;

        // **`y * h / (h - 1)`, in that order.** Precomputing `h / (h - 1)` and
        // multiplying is one division cheaper and rounds differently; measured
        // over 200 accumulating steps, that alone moved trajectories by 7.4e-3
        // px. The division stays. What is saved is doing it *twice* — cellpose
        // recomputes the whole transform once per flow component, and both
        // components sample at the same point.
        let height_f = height as f32;
        let width_f = width as f32;
        let last_y = height_f - 1.0;
        let last_x = width_f - 1.0;

        seeds
            .par_iter()
            .map(|&(seed_y, seed_x)| {
                let mut y = seed_y as f32;
                let mut x = seed_x as f32;
                for _ in 0..niter {
                    let sample_y = if height > 1 {
                        y * height_f / last_y - 0.5
                    } else {
                        0.0
                    };
                    let sample_x = if width > 1 {
                        x * width_f / last_x - 0.5
                    } else {
                        0.0
                    };

                    let y0 = sample_y.floor();
                    let x0 = sample_x.floor();
                    let fy = sample_y - y0;
                    let fx = sample_x - x0;
                    let y0 = y0 as i64;
                    let x0 = x0 as i64;

                    // One fetch per corner gives both components.
                    let at = |yy: i64, xx: i64| -> (f32, f32) {
                        if yy >= 0 && yy < height as i64 && xx >= 0 && xx < width as i64 {
                            let base = 2 * (yy as usize * width + xx as usize);
                            // SAFETY: `base + 1 < 2 * height * width` follows
                            // from the bound just checked, and the caller's
                            // slice is that long — asserted below.
                            unsafe { (*field.get_unchecked(base), *field.get_unchecked(base + 1)) }
                        } else {
                            (0.0, 0.0)
                        }
                    };

                    let (a_y, a_x) = at(y0, x0);
                    let (b_y, b_x) = at(y0, x0 + 1);
                    let (c_y, c_x) = at(y0 + 1, x0);
                    let (d_y, d_x) = at(y0 + 1, x0 + 1);

                    // **Cellpose's expression, term for term and in its
                    // order.** The obvious factorisation —
                    // `(1-fy)*((1-fx)*a + fx*b) + fy*(...)` — is two
                    // multiplications cheaper and gives a different answer in
                    // the last bits, which over 200 accumulating steps was
                    // measured at 7.4e-3 px. A follower that is faster and
                    // disagrees is a second implementation; one that is faster
                    // and agrees is an optimisation.
                    let top = 1.0 - fy;
                    let left = 1.0 - fx;
                    let dy = a_y * top * left + b_y * top * fx + c_y * fy * left + d_y * fy * fx;
                    let dx = a_x * top * left + b_x * top * fx + c_x * fy * left + d_x * fy * fx;

                    y = (y + dy).clamp(0.0, last_y);
                    x = (x + dx).clamp(0.0, last_x);
                }
                (y, x)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The optimised follower is an optimisation, not a second
    /// implementation.**
    ///
    /// Bit-identical to `cellpose::dynamics::follow_flows` on a field with
    /// structure — smooth flows converging on several centres, which is what a
    /// real one is — over the full default iteration count. Needs no GPU and no
    /// model: both functions take a flow array.
    ///
    /// Two rounding traps were found by this comparison and are what the
    /// implementation is careful about. Neither showed up as anything but a
    /// last-bit difference that grew over 200 accumulating steps to 7.4e-3 px:
    /// the coordinate transform must be `y * h / (h - 1)` in that order, and
    /// the bilinear must be summed as four separate products.
    #[cfg(feature = "cellpose")]
    #[test]
    fn the_fast_follower_agrees_with_cellpose_bit_for_bit() {
        let (height, width) = (96usize, 112);
        let mut d_p = ndarray16::Array3::<f32>::zeros((2, height, width));
        // Flows pointing at three attractors, plus a little noise, so
        // trajectories are long, curved and land in different places.
        let centres = [(20.0f32, 30.0f32), (60.0, 80.0), (75.0, 20.0)];
        for y in 0..height {
            for x in 0..width {
                let (mut dy, mut dx) = (0.0f32, 0.0f32);
                for (cy, cx) in centres {
                    let (vy, vx) = (cy - y as f32, cx - x as f32);
                    let distance = (vy * vy + vx * vx).sqrt().max(1.0);
                    dy += vy / distance / (1.0 + distance / 20.0);
                    dx += vx / distance / (1.0 + distance / 20.0);
                }
                let jitter = ((y * 7 + x * 13) % 11) as f32 / 40.0 - 0.125;
                d_p[[0, y, x]] = dy + jitter;
                d_p[[1, y, x]] = dx - jitter;
            }
        }

        let seeds: Vec<(usize, usize)> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (y, x)))
            .collect();
        let theirs = cellpose::dynamics::follow_flows(&d_p, &seeds, 200);

        let mut ours_input = ndarray::Array3::<f32>::zeros((2, height, width));
        for c in 0..2 {
            for y in 0..height {
                for x in 0..width {
                    ours_input[[c, y, x]] = d_p[[c, y, x]];
                }
            }
        }
        let field = steps::interleave(&ours_input.view());
        let ours = steps::follow(&field, height, width, &seeds, 200);

        for (index, (y, x)) in ours.iter().enumerate() {
            assert_eq!(
                theirs[[index, 0]].to_bits(),
                y.to_bits(),
                "seed {:?}: y is {} against {}",
                seeds[index],
                y,
                theirs[[index, 0]]
            );
            assert_eq!(
                theirs[[index, 1]].to_bits(),
                x.to_bits(),
                "seed {:?}: x",
                seeds[index]
            );
        }
    }

    #[test]
    fn a_stride_wider_than_the_window_is_refused() {
        assert!(WindowGrid::new(256, 257).is_err());
        assert!(WindowGrid::new(256, 0).is_err());
        assert!(WindowGrid::new(256, 256).is_ok());
    }

    #[test]
    fn the_origins_cover_the_volume_and_end_flush_with_it() {
        let grid = WindowGrid::new(256, 224).unwrap();
        for length in [256, 257, 512, 1000, 2048, 4096] {
            let origins = grid.origins(length);
            assert_eq!(origins[0], 0, "length {length}");
            assert_eq!(
                origins.last().copied().unwrap() + 256,
                length.max(256),
                "length {length}: the last window is flush with the end"
            );
            // No gaps: consecutive origins are within a window of each other.
            for pair in origins.windows(2) {
                assert!(pair[1] - pair[0] <= 256, "length {length}: a gap");
            }
        }
    }

    /// **The property the whole design rests on**, checked without a network:
    /// the set of windows covering a pixel is a function of the pixel, not of
    /// the block that reached it.
    #[test]
    fn the_windows_covering_a_pixel_do_not_depend_on_the_block() {
        let grid = WindowGrid::new(256, 224).unwrap();
        let volume = 4096;
        for block in [512usize, 1024, 2048, 4096] {
            for core_start in (0..volume).step_by(block) {
                let core_end = (core_start + block).min(volume);
                // What this block would run: every window wholly inside its
                // buffer, the buffer being the core grown by the halo.
                let low = core_start.saturating_sub(grid.halo());
                let high = (core_end + grid.halo()).min(volume);
                let ran: Vec<usize> = grid
                    .origins(volume)
                    .into_iter()
                    .filter(|origin| *origin >= low && origin + grid.window() <= high)
                    .collect();

                // What each core pixel actually needs.
                for at in [core_start, (core_start + core_end) / 2, core_end - 1] {
                    let needed = grid.covering(volume, at, at + 1);
                    for origin in needed {
                        assert!(
                            ran.contains(&origin),
                            "block {block}, core {core_start}..{core_end}: pixel {at} needs the \
                             window at {origin} and this block does not run it"
                        );
                    }
                }
            }
        }
    }

    /// And therefore the blending denominator is identical at every block size
    /// — computed the way each block would compute it, from the windows that
    /// block runs, against the whole-volume answer.
    ///
    /// This is the arithmetic the flow field is divided by, and it is exact:
    /// the same `f64` terms in the same ascending order, so the comparison is
    /// on bits and not on a tolerance.
    #[test]
    fn the_blending_weight_is_the_same_under_every_cut() {
        let grid = WindowGrid::new(256, 224).unwrap();
        let volume = [2048usize, 2048];
        // The same shape as cellpose's taper: high in the middle, low at the
        // edges. The claim is about the sum being block-independent, which
        // holds for any profile, so the profile itself is not the point.
        let taper: Vec<f64> = (0..256)
            .map(|i| {
                let x = (i as f64 - 127.5).abs();
                1.0 / (1.0 + ((x - 108.0) / 7.5).exp())
            })
            .collect();

        let probe = [
            [0usize, 0],
            [1, 300],
            [1024, 1024],
            [2047, 2047],
            [700, 1500],
        ];
        for at in probe {
            let reference = grid.weight_at(volume, at, &taper);
            assert!(reference > 0.0, "pixel {at:?} is covered by no window");

            for block in [512usize, 1024, 2048] {
                // The block that owns this pixel, and the windows it can run:
                // those wholly inside its buffer.
                let mut total = 0.0;
                let mut buffer = [(0usize, 0usize); 2];
                for axis in 0..2 {
                    let core_start = (at[axis] / block) * block;
                    let core_end = (core_start + block).min(volume[axis]);
                    buffer[axis] = (
                        core_start.saturating_sub(grid.halo()),
                        (core_end + grid.halo()).min(volume[axis]),
                    );
                }
                let runnable = |axis: usize| -> Vec<usize> {
                    grid.origins(volume[axis])
                        .into_iter()
                        .filter(|origin| {
                            *origin >= buffer[axis].0
                                && origin + grid.window() <= buffer[axis].1
                                && *origin <= at[axis]
                                && origin + grid.window() > at[axis]
                        })
                        .collect()
                };
                for origin_y in runnable(0) {
                    for origin_x in runnable(1) {
                        total += taper[at[0] - origin_y] * taper[at[1] - origin_x];
                    }
                }
                assert_eq!(
                    total.to_bits(),
                    reference.to_bits(),
                    "block {block}, pixel {at:?}: this block's own windows give {total} \
                     against the whole volume's {reference}"
                );
            }
        }
    }
}
