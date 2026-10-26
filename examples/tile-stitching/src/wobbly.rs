// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The per-slice ("wobbly") alignment of each overlapping pair: one in-plane
//! displacement per slice along the wobble axis, and the rule that fills in the
//! slices that were not measured.
//!
//! Light-sheet tiles drift in-plane as the stage moves along z, so a single
//! rigid offset per tile leaves seams that come and go along the stack. For
//! each pair, after the rigid step:
//!
//! 1. The overlap along the wobble axis is `max` of the two starts to `min` of
//!    the two ends, and each tile is cut to the in-plane rectangle the other can
//!    reach under the shift range ([`PlaneGeometry`]).
//! 2. The cut stacks are validated for foreground; a stack without signal marks
//!    every slice [`AlignmentStatus::NO_SIGNAL`].
//! 3. Every `slice_step`-th slice is measured, the rest are
//!    [`AlignmentStatus::UNALIGNED`] (a subsampling, not a failure). A measured
//!    slice must pass its own foreground check.
//! 4. The cut stacks are centred and scaled by the mean and norm of the **whole
//!    cut stack** (not per plane, which would rescale each slice's landscape
//!    differently), and each measured slice gets a mean-squared-difference
//!    landscape over every candidate shift.
//! 5. The shift per slice is chosen by tracing: local minima per slice are
//!    linked across slices into trajectories by optimal assignment, and the
//!    longest trajectories are taken greedily. A shift that is a consistent
//!    minimum over many slices beats a slightly deeper one that appears in a
//!    single slice. [`ShiftMethod::Minimization`] takes each slice's global
//!    minimum instead.
//!
//! The recorded displacement is the chosen shift plus the two tiles' rigid
//! offset with the wobble axis dropped. [`Track::resolve_rejected`] then
//! interpolates the unaligned slices between their measured neighbours.

use blockflow::error::{Error, Result};
use blockflow::ops::fft::{Padding, SquaredDifference};
use blockflow::voxels::VoxelElement;
use ndarray::{Array2, ArrayView2, ArrayView3};
use rayon::prelude::*;

use crate::place::{AlignmentSlot, SourceSlab, WobblyLayout};
use crate::plane::PlaneGeometry;
use crate::solve::round_even;
use crate::tiles::{admitted, BoxAt, KeptPair, TileBoxes};

// --------------------------------------------------------------- vocabulary --

/// The status values a slice of an alignment can carry. `status >= VALID` is
/// "usable" everywhere.
pub struct AlignmentStatus;

impl AlignmentStatus {
    /// The stack or this slice failed foreground validation.
    pub const NO_SIGNAL: i32 = -5;
    /// The landscape had no interior local minimum.
    pub const NO_MINIMA: i32 = -4;
    /// Skipped by the slice step: never measured, and the only status the
    /// resolution rule repairs.
    pub const UNALIGNED: i32 = -3;
    /// Measured, but no chosen trajectory covered it.
    pub const UNTRACED: i32 = -2;
    /// The initial value, and what an unresolvable run becomes.
    pub const INVALID: i32 = -1;
    /// The usability threshold; never assigned.
    pub const VALID: i32 = 0;
    /// A landscape was computed. An intermediate that never survives.
    pub const MEASURED: i32 = 1;
    /// A shift was chosen from this slice's own landscape.
    pub const ALIGNED: i32 = 2;
    /// Filled in by the resolution rule.
    pub const FIXED: i32 = 3;

    pub const ALL: [i32; 9] = [-5, -4, -3, -2, -1, 0, 1, 2, 3];
}

/// The status values a placed source slice can carry. A separate vocabulary:
/// `-2` and `2` mean different things here and in [`AlignmentStatus`].
pub struct SourceStatus;

impl SourceStatus {
    /// No usable alignment reached this tile in this slice.
    pub const ISOLATED: i32 = -2;
    /// The placed position was not usable.
    pub const INVALID: i32 = -1;
    /// Placed.
    pub const VALID: i32 = 0;
    /// An isolated run repaired by interpolation.
    pub const FIXED: i32 = 2;

    pub const ALL: [i32; 4] = [-2, -1, 0, 2];
}

pub fn status_name(status: i32) -> &'static str {
    match status {
        AlignmentStatus::NO_SIGNAL => "no signal",
        AlignmentStatus::NO_MINIMA => "no minima",
        AlignmentStatus::UNALIGNED => "unaligned",
        AlignmentStatus::UNTRACED => "untraced",
        AlignmentStatus::INVALID => "invalid",
        AlignmentStatus::VALID => "valid",
        AlignmentStatus::MEASURED => "measured",
        AlignmentStatus::ALIGNED => "aligned",
        AlignmentStatus::FIXED => "fixed",
        _ => "unknown",
    }
}

pub fn source_status_name(status: i32) -> &'static str {
    match status {
        SourceStatus::ISOLATED => "isolated",
        SourceStatus::INVALID => "invalid",
        SourceStatus::VALID => "valid",
        SourceStatus::FIXED => "fixed",
        _ => "unknown",
    }
}

// ------------------------------------------------------------------- track --

/// One alignment's per-slice displacement, quality and status.
///
/// `displacements` are the chosen shift plus the rigid position difference
/// with the wobble axis dropped, which is what the placement consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub displacements: Vec<Vec<isize>>,
    /// `-inf` where nothing was chosen.
    pub qualities: Vec<f64>,
    pub status: Vec<i32>,
}

impl Track {
    pub fn new(
        displacements: Vec<Vec<isize>>,
        qualities: Vec<f64>,
        status: Vec<i32>,
    ) -> Result<Self> {
        if displacements.len() != qualities.len() || displacements.len() != status.len() {
            return Err(Error::invalid(format!(
                "a wobbly track's three arrays disagree: {} displacements, {} qualities, {} status",
                displacements.len(),
                qualities.len(),
                status.len()
            )));
        }
        if let Some(width) = displacements.first().map(Vec::len) {
            if displacements.iter().any(|row| row.len() != width) {
                return Err(Error::invalid("a wobbly track's displacements are ragged"));
            }
        }
        Ok(Self {
            displacements,
            qualities,
            status,
        })
    }

    pub fn len(&self) -> usize {
        self.status.len()
    }

    pub fn is_empty(&self) -> bool {
        self.status.is_empty()
    }

    /// How many slices carry each status value that occurs, lowest first.
    pub fn census(&self) -> Vec<(i32, usize)> {
        let mut seen: Vec<(i32, usize)> = Vec::new();
        for &status in &self.status {
            match seen.iter_mut().find(|(value, _)| *value == status) {
                Some((_, count)) => *count += 1,
                None => seen.push((status, 1)),
            }
        }
        seen.sort_by_key(|(value, _)| *value);
        seen
    }

    /// The slices the placement may use: `status >= VALID` and a quality
    /// strictly above `min_quality`.
    pub fn valids(&self, min_quality: f64) -> Vec<bool> {
        self.status
            .iter()
            .zip(&self.qualities)
            .map(|(&status, &quality)| status >= AlignmentStatus::VALID && quality > min_quality)
            .collect()
    }

    /// Repair the runs of `UNALIGNED` slices in place.
    ///
    /// Only `UNALIGNED` is repaired: a slice that was measured and failed is
    /// left as it is. Within a run, with `left` and `right` the usable slices
    /// either side: both present interpolates (rounded) linearly, one present
    /// holds its value, neither invalidates the run. A track that is unaligned
    /// end to end is invalidated whole.
    pub fn resolve_rejected(&mut self) -> ResolutionReport {
        let n = self.len();
        let mut report = ResolutionReport {
            slices: n,
            unaligned: self
                .status
                .iter()
                .filter(|&&s| s == AlignmentStatus::UNALIGNED)
                .count(),
            ..ResolutionReport::default()
        };
        if report.unaligned == 0 {
            return report;
        }
        let runs = runs_of(&self.status, AlignmentStatus::UNALIGNED);
        if runs.len() == 1 && runs[0] == (0, n) {
            self.status.fill(AlignmentStatus::INVALID);
            report.whole_stack_invalidated = true;
            report.invalidated = n;
            return report;
        }
        for (start, end) in runs {
            let left =
                (start > 0 && self.status[start - 1] >= AlignmentStatus::VALID).then(|| start - 1);
            let right = (end < n && self.status[end] >= AlignmentStatus::VALID).then_some(end);
            let held = match (left, right) {
                (None, None) => {
                    self.status[start..end].fill(AlignmentStatus::INVALID);
                    report.invalidated += end - start;
                    continue;
                }
                (Some(left), None) => {
                    report.held_left += end - start;
                    Some(left)
                }
                (None, Some(right)) => {
                    report.held_right += end - start;
                    Some(right)
                }
                (Some(_), Some(_)) => None,
            };
            match held {
                Some(from) => {
                    let value = self.displacements[from].clone();
                    let quality = self.qualities[from];
                    for slot in start..end {
                        self.displacements[slot] = value.clone();
                        self.qualities[slot] = quality;
                    }
                }
                None => {
                    let (left, right) = (start - 1, end);
                    let span = (end - start + 1) as f64;
                    let (low, high) = (
                        self.displacements[left].clone(),
                        self.displacements[right].clone(),
                    );
                    let (low_quality, high_quality) = (self.qualities[left], self.qualities[right]);
                    for (step, slot) in (start..end).enumerate() {
                        let k = (step + 1) as f64;
                        self.displacements[slot] = low
                            .iter()
                            .zip(&high)
                            .map(|(&low, &high)| {
                                round_even((high - low) as f64 / span * k + low as f64)
                            })
                            .collect();
                        self.qualities[slot] =
                            if low_quality.is_finite() && high_quality.is_finite() {
                                (high_quality - low_quality) / span * k + low_quality
                            } else if high_quality.is_finite() {
                                high_quality
                            } else {
                                low_quality
                            };
                    }
                    report.interpolated += end - start;
                }
            }
            self.status[start..end].fill(AlignmentStatus::FIXED);
        }
        report
    }
}

/// The maximal runs of `value` in `status`, as half-open ranges.
pub(crate) fn runs_of(status: &[i32], value: i32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < status.len() {
        if status[index] == value {
            let start = index;
            while index < status.len() && status[index] == value {
                index += 1;
            }
            runs.push((start, index));
        } else {
            index += 1;
        }
    }
    runs
}

/// Which branches of the resolution rule a track took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolutionReport {
    pub slices: usize,
    /// `UNALIGNED` before the rule ran.
    pub unaligned: usize,
    pub interpolated: usize,
    /// Held at the usable slice below, there being none above.
    pub held_left: usize,
    /// Held at the usable slice above, there being none below.
    pub held_right: usize,
    pub invalidated: usize,
    pub whole_stack_invalidated: bool,
}

// --------------------------------------------------------------- smoothing --

/// The smoothing windows, as NumPy defines them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// A moving average.
    Flat,
    Hanning,
    Hamming,
    Bartlett,
    Blackman,
}

impl Window {
    /// The window's `length` taps, unnormalised.
    pub fn taps(self, length: usize) -> Vec<f64> {
        if length == 0 {
            return Vec::new();
        }
        if length == 1 {
            return vec![1.0];
        }
        let denominator = (length - 1) as f64;
        let pi = std::f64::consts::PI;
        (0..length)
            .map(|index| {
                let n = 2.0 * index as f64 - denominator;
                match self {
                    Self::Flat => 1.0,
                    Self::Bartlett => 1.0 - n.abs() / denominator,
                    Self::Hanning => 0.5 + 0.5 * (pi * n / denominator).cos(),
                    Self::Hamming => 0.54 + 0.46 * (pi * n / denominator).cos(),
                    Self::Blackman => {
                        0.42 + 0.5 * (pi * n / denominator).cos()
                            + 0.08 * (2.0 * pi * n / denominator).cos()
                    }
                }
            })
            .collect()
    }
}

/// How a displacement sequence is smoothed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothParams {
    /// `None` skips the window and goes straight to the binary pass.
    pub window: Option<Window>,
    /// Clamped to the interval's length before the taps are built.
    pub window_length: usize,
    /// [`smooth_binary`]'s width, or `None` to skip it.
    pub binary: Option<usize>,
}

/// Convolve an integer sequence with a normalised window over edge padding,
/// and round.
///
/// A window whose taps sum to zero (`Bartlett` or `Hanning` of length 2) is
/// refused rather than dividing by zero; only a valid interval of exactly two
/// slices can reach it.
pub fn smooth_window(values: &[isize], params: SmoothParams) -> Result<Vec<isize>> {
    let n = values.len();
    let mut smoothed = match params.window {
        None => values.to_vec(),
        Some(_) if n == 0 => Vec::new(),
        Some(window) => {
            let length = params.window_length.clamp(1, n);
            let taps = window.taps(length);
            let total = taps.iter().sum::<f64>();
            if total == 0.0 {
                return Err(Error::invalid(format!(
                    "a {window:?} window of length {length} sums to zero, so it cannot be \
                     normalised; use a longer window or another window type"
                )));
            }
            // The window's centre tap sits on the output sample; edge padding is
            // addressed by clamping rather than built.
            let centre = (length as isize - 1) / 2;
            (0..n)
                .map(|slot| {
                    let mut accumulator = 0.0;
                    for (tap_index, &tap) in taps.iter().enumerate() {
                        let source =
                            (slot as isize + centre - tap_index as isize).clamp(0, n as isize - 1);
                        accumulator += tap / total * values[source as usize] as f64;
                    }
                    round_even(accumulator)
                })
                .collect()
        }
    };
    if let Some(width) = params.binary.filter(|&width| width > 0) {
        smoothed = smooth_binary(&smoothed, width);
    }
    Ok(smoothed)
}

/// Flatten every excursion narrower than `width`: wherever two samples `w`
/// apart agree, everything between them is set to that value, for `w` from
/// `width + 1` down to 2. The two ends are first set to the median of their own
/// `width + 1` values.
pub fn smooth_binary(values: &[isize], width: usize) -> Vec<isize> {
    let n = values.len();
    let mut out = values.to_vec();
    if n == 0 {
        return out;
    }
    let width = (width + 1).min(n);
    let head = median(&out[..width]);
    out[..width].fill(head);
    let tail = median(&out[n - width..]);
    out[n - width..].fill(tail);
    for window in (2..=width).rev() {
        for start in 0..n.saturating_sub(window) {
            let end = start + window;
            if out[start] == out[end] {
                let value = out[start];
                out[start..end].fill(value);
            }
        }
    }
    out
}

/// The median, rounded half to even when it falls between two integers.
fn median(values: &[isize]) -> isize {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n == 0 {
        0
    } else if n % 2 == 1 {
        sorted[n / 2]
    } else {
        round_even((sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0)
    }
}

/// Smooth each column within each maximal run of `valids`, leaving rejected
/// slices untouched.
pub fn smooth_displacements(
    displacements: &[Vec<isize>],
    valids: &[bool],
    params: SmoothParams,
) -> Result<Vec<Vec<isize>>> {
    let mut out = displacements.to_vec();
    let ndim = displacements.first().map(Vec::len).unwrap_or(0);
    let valid_status = valids
        .iter()
        .take(displacements.len())
        .map(|&valid| i32::from(valid))
        .collect::<Vec<_>>();
    for (start, end) in runs_of(&valid_status, 1) {
        for dim in 0..ndim {
            let column = (start..end)
                .map(|slot| displacements[slot][dim])
                .collect::<Vec<_>>();
            for (slot, value) in (start..end).zip(smooth_window(&column, params)?) {
                out[slot][dim] = value;
            }
        }
    }
    Ok(out)
}

// -------------------------------------------------------------- validation --

/// A foreground test: a value counts when it lies in `[low, high]` (either end
/// optional), and an array passes when at least `size` of its values count (or
/// `fraction` of them; with neither, at least one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Foreground {
    pub low: Option<f64>,
    pub high: Option<f64>,
    pub size: Option<f64>,
    /// Takes precedence over `size`.
    pub fraction: Option<f64>,
}

impl Foreground {
    /// The default stack test: some voxel at 200 or above.
    pub fn stack_default() -> Self {
        Self {
            low: Some(200.0),
            high: None,
            size: None,
            fraction: None,
        }
    }

    /// The default slice test: 1500 pixels between 200 and 20000.
    pub fn slice_default() -> Self {
        Self {
            low: Some(200.0),
            high: Some(20_000.0),
            size: Some(1500.0),
            fraction: None,
        }
    }

    pub fn counts(&self, value: f64) -> bool {
        self.low.is_none_or(|low| value >= low) && self.high.is_none_or(|high| value <= high)
    }

    pub fn count(&self, values: impl Iterator<Item = f64>) -> usize {
        values.filter(|&value| self.counts(value)).count()
    }

    pub fn accepts_count(&self, foreground: usize, total: usize) -> bool {
        match self
            .fraction
            .map(|fraction| fraction * total as f64)
            .or(self.size)
        {
            None => foreground > 0,
            Some(size) => foreground as f64 >= size,
        }
    }
}

// ------------------------------------------------------------- preparation --

/// The mean and norm of a whole cut stack, with compensated sums: a naive sum
/// over tens of millions of elements loses enough to move the landscape.
pub fn stack_statistics(planes: &[Array2<f64>]) -> (f64, f64) {
    fn add(total: &mut f64, compensation: &mut f64, term: f64) {
        let sum = *total + term;
        *compensation += if total.abs() >= term.abs() {
            (*total - sum) + term
        } else {
            (term - sum) + *total
        };
        *total = sum;
    }
    let (mut total, mut compensation, mut count) = (0.0, 0.0, 0usize);
    for value in planes.iter().flat_map(|plane| plane.iter()) {
        add(&mut total, &mut compensation, *value);
        count += 1;
    }
    let mean = if count == 0 {
        0.0
    } else {
        (total + compensation) / count as f64
    };
    let (mut energy, mut energy_compensation) = (0.0, 0.0);
    for value in planes.iter().flat_map(|plane| plane.iter()) {
        let centred = value - mean;
        add(&mut energy, &mut energy_compensation, centred * centred);
    }
    (mean, (energy + energy_compensation).sqrt())
}

/// Centre and scale a stack by its own mean and norm. Returns both.
pub fn prepare_normalization(planes: &mut [Array2<f64>]) -> (f64, f64) {
    let (mean, norm) = stack_statistics(planes);
    let scale = if norm == 0.0 { 0.0 } else { 1.0 / norm };
    for plane in planes.iter_mut() {
        plane.mapv_inplace(|value| (value - mean) * scale);
    }
    (mean, norm)
}

// ------------------------------------------------------------------ minima --

/// The local minima of a landscape: points no higher than anything within
/// Chebyshev distance `distance`, excluding the `distance`-wide border, and
/// below the landscape's own maximum. A plateau admits all of its points; a
/// constant landscape has none. Returned deepest first, ties in raster order.
pub fn detect_local_minima(error: ArrayView2<f64>, distance: usize) -> (Vec<[usize; 2]>, Vec<f64>) {
    let (rows, cols) = error.dim();
    let distance = distance.max(1);
    let reach = distance as isize;
    if rows <= 2 * distance || cols <= 2 * distance {
        return (Vec::new(), Vec::new());
    }
    let highest = error.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut found = Vec::new();
    let mut every_point_is_a_peak = true;
    for row in 0..rows {
        for col in 0..cols {
            let value = error[[row, col]];
            let mut lowest = f64::INFINITY;
            for dr in -reach..=reach {
                for dc in -reach..=reach {
                    let r = (row as isize + dr).clamp(0, rows as isize - 1) as usize;
                    let c = (col as isize + dc).clamp(0, cols as isize - 1) as usize;
                    lowest = lowest.min(error[[r, c]]);
                }
            }
            let peak = value == lowest;
            every_point_is_a_peak &= peak;
            let interior = row >= distance
                && row + distance < rows
                && col >= distance
                && col + distance < cols;
            if peak && interior && value < highest {
                found.push(([row, col], value));
            }
        }
    }
    if every_point_is_a_peak {
        return (Vec::new(), Vec::new());
    }
    // Stable, so equal values keep raster order.
    found.sort_by(|left, right| left.1.total_cmp(&right.1));
    found.into_iter().unzip()
}

// ----------------------------------------------------------------- tracker --

/// One trajectory: `(step, particle)` pairs, one per step it is present in.
pub type Trajectory = Vec<(usize, usize)>;

/// A bound on the swap descent. Every accepted move lowers the total cost by
/// more than `1e-10`, so the descent terminates; this turns a pathological
/// input into an error rather than a hang.
const MAX_ASSIGNMENT_MOVES: usize = 100_000;

/// The initial assignment: each row takes its cheapest free column, or the
/// dummy column if that comes first.
fn initial_assignment(cost: &Array2<f64>) -> Vec<Vec<bool>> {
    let rows = cost.nrows() - 1;
    let cols = cost.ncols() - 1;
    let mut assignment = vec![vec![false; cols + 1]; rows + 1];
    let mut taken = vec![false; cols + 1];
    for row in 0..rows {
        let mut order = (0..=cols).collect::<Vec<_>>();
        order.sort_by(|&left, &right| cost[[row, left]].total_cmp(&cost[[row, right]]));
        let column = order
            .iter()
            .copied()
            .find(|&column| column == cols || !taken[column])
            .unwrap_or(cols);
        assignment[row][column] = true;
        taken[column] = true;
    }
    for column in 0..cols {
        if !taken[column] {
            assignment[rows][column] = true;
        }
    }
    assignment[rows][cols] = true;
    assignment
}

/// A swap descent on the association matrix until no single insertion lowers
/// the total cost.
fn optimal_assignment(cost: &Array2<f64>) -> Result<Vec<Vec<bool>>> {
    let rows = cost.nrows() - 1;
    let cols = cost.ncols() - 1;
    let mut assignment = initial_assignment(cost);
    let finite = (0..rows)
        .flat_map(|row| (0..cols).map(move |column| (row, column)))
        .filter(|&(row, column)| cost[[row, column]].is_finite())
        .collect::<Vec<_>>();
    for _ in 0..MAX_ASSIGNMENT_MOVES {
        let mut best: Option<(usize, usize, usize, usize, f64)> = None;
        for &(row, column) in &finite {
            if assignment[row][column] {
                continue;
            }
            let y = (0..=cols)
                .find(|&c| assignment[row][c])
                .ok_or_else(|| Error::invalid("an association row is empty"))?;
            let x = (0..=rows)
                .find(|&r| assignment[r][column])
                .ok_or_else(|| Error::invalid("an association column is empty"))?;
            let reduced = cost[[row, column]] + cost[[x, y]] - cost[[row, y]] - cost[[x, column]];
            if best.is_none_or(|(.., incumbent)| reduced.total_cmp(&incumbent).is_lt()) {
                best = Some((row, column, x, y, reduced));
            }
        }
        match best {
            Some((row, column, x, y, reduced)) if reduced < -1.0e-10 => {
                assignment[row][column] = true;
                assignment[x][column] = false;
                assignment[row][y] = false;
                assignment[x][y] = true;
            }
            _ => return Ok(assignment),
        }
    }
    Err(Error::invalid(format!(
        "the assignment's swap descent did not settle in {MAX_ASSIGNMENT_MOVES} moves"
    )))
}

/// The optimal association between two point sets, with a dummy row and
/// column priced at `new_trajectory_cost`. For each `pre` point, the `post`
/// point it maps to, or `post.len()` when its trajectory ends.
fn match_positions(
    pre: &[[f64; 2]],
    post: &[[f64; 2]],
    new_trajectory_cost: Option<f64>,
    cutoff: Option<f64>,
) -> Result<Vec<usize>> {
    let (rows, cols) = (pre.len(), post.len());
    let mut cost = Array2::<f64>::zeros((rows + 1, cols + 1));
    let mut highest = f64::NEG_INFINITY;
    for (row, a) in pre.iter().enumerate() {
        for (column, b) in post.iter().enumerate() {
            let distance = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            let value = match cutoff {
                Some(cutoff) if distance > cutoff => f64::INFINITY,
                _ => distance,
            };
            highest = highest.max(value);
            cost[[row, column]] = value;
        }
    }
    let dummy = new_trajectory_cost.unwrap_or(highest + 1.0);
    cost.column_mut(cols).fill(dummy);
    cost.row_mut(rows).fill(dummy);
    let assignment = optimal_assignment(&cost)?;
    Ok((0..rows)
        .map(|row| (0..=cols).find(|&c| assignment[row][c]).unwrap_or(cols))
        .collect())
}

/// Link per-step point sets into trajectories by pairwise optimal assignment.
pub fn track_positions(
    positions: &[Vec<[f64; 2]>],
    new_trajectory_cost: Option<f64>,
    cutoff: Option<f64>,
) -> Result<Vec<Trajectory>> {
    let Some(first) = positions.first() else {
        return Ok(Vec::new());
    };
    let mut trajectories = (0..first.len())
        .map(|particle| vec![(0usize, particle)])
        .collect::<Vec<Trajectory>>();
    let mut active = (0..first.len()).collect::<Vec<usize>>();
    for step in 1..positions.len() {
        let mapping = match_positions(
            &positions[step - 1],
            &positions[step],
            new_trajectory_cost,
            cutoff,
        )?;
        let count = positions[step].len();
        let mut landed = vec![false; count];
        let mut still_active = Vec::with_capacity(active.len());
        for &trajectory in &active {
            let particle = trajectories[trajectory].last().expect("a last point").1;
            let target = *mapping
                .get(particle)
                .ok_or_else(|| Error::invalid("an active particle has no match"))?;
            if target < count {
                trajectories[trajectory].push((step, target));
                landed[target] = true;
                still_active.push(trajectory);
            }
        }
        active = still_active;
        for (particle, landed) in landed.into_iter().enumerate() {
            if !landed {
                active.push(trajectories.len());
                trajectories.push(vec![(step, particle)]);
            }
        }
    }
    Ok(trajectories)
}

/// The greedy pick: take the longest trajectory, drop everything that overlaps
/// it in step range, repeat. Between equally long trajectories, the one with
/// the lower summed error wins, each point's error read from the landscape of
/// the slice it is on (`slices[step]`).
pub fn select_trajectories(
    mut trajectories: Vec<Trajectory>,
    positions: &[Vec<[f64; 2]>],
    landscapes: &[Array2<f64>],
    slices: &[usize],
) -> Vec<Trajectory> {
    let error_of = |trajectory: &Trajectory| {
        trajectory
            .iter()
            .map(|&(step, particle)| {
                let at = positions[step][particle];
                landscapes[slices[step]][[at[0] as usize, at[1] as usize]]
            })
            .sum::<f64>()
    };
    let mut optimal: Vec<Trajectory> = Vec::new();
    let mut covered = 0usize;
    while covered < slices.len() && !trajectories.is_empty() {
        let longest = trajectories.iter().map(Vec::len).max().unwrap_or(0);
        let mut chosen: Option<(usize, f64)> = None;
        for (slot, trajectory) in trajectories.iter().enumerate() {
            if trajectory.len() != longest {
                continue;
            }
            let total = error_of(trajectory);
            if chosen.is_none_or(|(_, best)| total.total_cmp(&best).is_lt()) {
                chosen = Some((slot, total));
            }
        }
        let trajectory = trajectories.remove(chosen.expect("a longest trajectory").0);
        covered += trajectory.len();
        let first = trajectory.first().map_or(0, |point| point.0);
        let last = trajectory.last().map_or(first, |point| point.0);
        trajectories.retain(|other| {
            let other_first = other.first().map_or(0, |point| point.0);
            let other_last = other.last().map_or(other_first, |point| point.0);
            other_last < first || other_first > last
        });
        optimal.push(trajectory);
    }
    optimal
}

// ------------------------------------------------------- shift selection --

/// How a shift is chosen per slice from its landscape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftMethod {
    /// Each slice's own global minimum.
    Minimization,
    /// Local minima linked into trajectories across slices; the longest win.
    Tracing,
}

fn shifts_from_minimization(
    landscapes: &[Array2<f64>],
    status: &mut [i32],
) -> (Vec<Vec<isize>>, Vec<f64>) {
    let n = status.len();
    let mut shifts = vec![vec![0isize; 2]; n];
    let mut qualities = vec![f64::NEG_INFINITY; n];
    for (index, state) in status.iter_mut().enumerate() {
        if *state != AlignmentStatus::MEASURED {
            continue;
        }
        let mut best: Option<([usize; 2], f64)> = None;
        for ((row, col), &value) in landscapes[index].indexed_iter() {
            if best.is_none_or(|(_, incumbent)| value.total_cmp(&incumbent).is_lt()) {
                best = Some(([row, col], value));
            }
        }
        if let Some((at, value)) = best {
            shifts[index] = vec![at[0] as isize, at[1] as isize];
            qualities[index] = -value;
            *state = AlignmentStatus::ALIGNED;
        }
    }
    (shifts, qualities)
}

fn shifts_from_tracing(
    landscapes: &[Array2<f64>],
    status: &mut [i32],
    params: &MeasureParams,
) -> Result<(Vec<Vec<isize>>, Vec<f64>)> {
    let n = status.len();
    let mut shifts = vec![vec![0isize; 2]; n];
    let mut qualities = vec![f64::NEG_INFINITY; n];
    let mut minima: Vec<Option<Vec<[f64; 2]>>> = vec![None; n];
    for index in 0..n {
        if status[index] != AlignmentStatus::MEASURED {
            continue;
        }
        let (coordinates, _) =
            detect_local_minima(landscapes[index].view(), params.minima_distance);
        if coordinates.is_empty() {
            status[index] = AlignmentStatus::NO_MINIMA;
        } else {
            minima[index] = Some(
                coordinates
                    .iter()
                    .map(|&[row, col]| [row as f64, col as f64])
                    .collect(),
            );
        }
    }
    // The diagonal of the landscape: farther than any two minima can be apart.
    let new_trajectory_cost = params.new_trajectory_cost.or_else(|| {
        landscapes.first().map(|landscape| {
            let (rows, cols) = landscape.dim();
            ((rows * rows + cols * cols) as f64).sqrt()
        })
    });
    // A stretch is a maximal run of measured or skipped slices: a failed slice
    // breaks the trace, a skipped one does not.
    let traceable = status
        .iter()
        .map(|&s| i32::from(s == AlignmentStatus::MEASURED || s == AlignmentStatus::UNALIGNED))
        .collect::<Vec<_>>();
    for (start, end) in runs_of(&traceable, 1) {
        let slices = (start..end)
            .filter(|&slot| status[slot] == AlignmentStatus::MEASURED)
            .collect::<Vec<_>>();
        if slices.is_empty() {
            continue;
        }
        let positions = slices
            .iter()
            .map(|&slot| minima[slot].clone().expect("minima of a measured slice"))
            .collect::<Vec<_>>();
        let trajectories = track_positions(&positions, new_trajectory_cost, params.cutoff)?;
        let optimal = select_trajectories(trajectories, &positions, landscapes, &slices);
        for &slot in &slices {
            status[slot] = AlignmentStatus::UNTRACED;
        }
        for (step, particle) in optimal.into_iter().flatten() {
            let slot = slices[step];
            let at = positions[step][particle];
            let at = [at[0] as usize, at[1] as usize];
            shifts[slot] = vec![at[0] as isize, at[1] as isize];
            qualities[slot] = -landscapes[slot][at];
            status[slot] = AlignmentStatus::ALIGNED;
        }
    }
    Ok((shifts, qualities))
}

// ------------------------------------------------------- the measurement --

/// The per-slice measurement's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureParams {
    /// The inclusive in-plane search range, per plane axis (the two volume axes
    /// other than the wobble axis, in ascending order: `[y, x]` for a `(z, y,
    /// x)` stack wobbling along z).
    pub max_shifts: [[isize; 2]; 2],
    /// Every `slice_step`-th slice of the overlap is measured; the rest are
    /// interpolated by [`Track::resolve_rejected`].
    pub slice_step: usize,
    pub stack_validation: Option<Foreground>,
    pub slice_validation: Option<Foreground>,
    /// Centre and scale the cut stacks by their own mean and norm.
    pub prepare: bool,
    /// Additionally centre and scale each plane on its own.
    pub prepare_slice: bool,
    /// Replace each plane by the maximum over planes `[local - before, local +
    /// after]` (clamped to the stack) before measuring.
    pub axis_mip: Option<(usize, usize)>,
    pub find_shifts: ShiftMethod,
    /// The largest jump of a minimum between consecutive measured slices that
    /// the tracer will link.
    pub cutoff: Option<f64>,
    pub minima_distance: usize,
    /// The cost of starting a new trajectory; `None` is the landscape diagonal.
    pub new_trajectory_cost: Option<f64>,
}

impl Default for MeasureParams {
    /// The reference's shipped per-slice defaults for a `(z, y, x)` stack
    /// wobbling along z: +/-15 along y and +/-30 along x.
    fn default() -> Self {
        Self {
            max_shifts: [[-15, 15], [-30, 30]],
            slice_step: 3,
            stack_validation: Some(Foreground::stack_default()),
            slice_validation: Some(Foreground::slice_default()),
            prepare: true,
            prepare_slice: false,
            axis_mip: None,
            find_shifts: ShiftMethod::Tracing,
            cutoff: Some(3.0 * std::f64::consts::SQRT_2),
            minima_distance: 1,
            new_trajectory_cost: None,
        }
    }
}

/// One pair's measurement.
#[derive(Debug, Clone)]
pub struct Measurement {
    pub track: Track,
    /// The chosen shifts, before the rigid offset is added.
    pub shifts: Vec<Vec<isize>>,
    /// One landscape per slice; zeros where nothing was measured.
    pub landscapes: Vec<Array2<f64>>,
    /// How many slices a landscape was computed for.
    pub measured: usize,
}

/// A stack of cut planes, one per wobble-axis coordinate of an overlap, read
/// as `f64` plane by plane: held planes, or planes taken from a tile box as
/// they are wanted, so a pair need not hold its stacks in `f64`.
pub trait CutStack: Sync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Plane `index`'s extent.
    fn dim(&self, index: usize) -> (usize, usize);

    /// Plane `index` as stored, before any preparation.
    fn plane(&self, index: usize) -> Array2<f64>;

    /// Every value of plane `index` in [`Self::plane`]'s order (rows, then
    /// columns), without holding the plane.
    fn visit(&self, index: usize, f: &mut dyn FnMut(f64)) {
        self.plane(index).iter().for_each(|&value| f(value));
    }
}

impl CutStack for [Array2<f64>] {
    fn len(&self) -> usize {
        <[Array2<f64>]>::len(self)
    }

    fn dim(&self, index: usize) -> (usize, usize) {
        self[index].dim()
    }

    fn plane(&self, index: usize) -> Array2<f64> {
        self[index].clone()
    }
}

/// The planes of a tile box across `axis`, widened to `f64` when read.
pub struct BoxCuts<'a, T> {
    pub view: ArrayView3<'a, T>,
    pub axis: usize,
}

impl<T: VoxelElement> CutStack for BoxCuts<'_, T> {
    fn len(&self) -> usize {
        self.view.len_of(ndarray::Axis(self.axis))
    }

    fn dim(&self, _index: usize) -> (usize, usize) {
        let kept: Vec<usize> = (0..3)
            .filter(|&a| a != self.axis)
            .map(|a| self.view.len_of(ndarray::Axis(a)))
            .collect();
        (kept[0], kept[1])
    }

    fn plane(&self, index: usize) -> Array2<f64> {
        self.view
            .index_axis(ndarray::Axis(self.axis), index)
            .mapv(|value| value.into_f64())
    }

    fn visit(&self, index: usize, f: &mut dyn FnMut(f64)) {
        self.view
            .index_axis(ndarray::Axis(self.axis), index)
            .iter()
            .for_each(|value| f(value.into_f64()));
    }
}

/// [`stack_statistics`] of a [`CutStack`], reading each plane twice (once per
/// sum) in the same order, so the figures are the same.
fn stack_statistics_of(stack: &(impl CutStack + ?Sized)) -> (f64, f64) {
    fn add(total: &mut f64, compensation: &mut f64, term: f64) {
        let sum = *total + term;
        *compensation += if total.abs() >= term.abs() {
            (*total - sum) + term
        } else {
            (term - sum) + *total
        };
        *total = sum;
    }
    let (mut total, mut compensation, mut count) = (0.0, 0.0, 0usize);
    for index in 0..stack.len() {
        stack.visit(index, &mut |value| {
            add(&mut total, &mut compensation, value);
            count += 1;
        });
    }
    let mean = if count == 0 {
        0.0
    } else {
        (total + compensation) / count as f64
    };
    let (mut energy, mut energy_compensation) = (0.0, 0.0);
    for index in 0..stack.len() {
        stack.visit(index, &mut |value| {
            let centred = value - mean;
            add(&mut energy, &mut energy_compensation, centred * centred);
        });
    }
    (mean, (energy + energy_compensation).sqrt())
}

/// The foreground and total counts of plane `index` of `stack`.
fn plane_counts<S: CutStack + ?Sized>(
    validation: Foreground,
    stack: &S,
    index: usize,
) -> (usize, usize) {
    let (mut foreground, mut total) = (0usize, 0usize);
    stack.visit(index, &mut |value| {
        total += 1;
        foreground += usize::from(validation.counts(value));
    });
    (foreground, total)
}

/// Whether the stack validation refuses a whole stack.
fn stack_rejects<S: CutStack + ?Sized>(validation: Foreground, stack: &S) -> bool {
    let (mut foreground, mut total) = (0usize, 0usize);
    for index in 0..stack.len() {
        let (f, t) = plane_counts(validation, stack, index);
        foreground += f;
        total += t;
    }
    !validation.accepts_count(foreground, total)
}

/// One pair's measurement from its two cut stacks: one plane per wobble-axis
/// coordinate of the overlap, in the rectangles `geometry` chose.
///
/// `plane_offset` is `post.position - pre.position` on the two plane axes; it
/// turns a shift into a displacement.
pub fn measure_wobbly_axis(
    cut_a: Vec<Array2<f64>>,
    cut_b: Vec<Array2<f64>>,
    geometry: &PlaneGeometry,
    plane_offset: [isize; 2],
    params: &MeasureParams,
) -> Result<Measurement> {
    measure_stacks(&cut_a[..], &cut_b[..], geometry, plane_offset, params)
}

/// [`measure_wobbly_axis`] over any [`CutStack`]s: each plane is read when it
/// is wanted (and centred and scaled by its stack's mean and norm then), so
/// only the planes being measured are held in `f64`.
pub fn measure_stacks<A: CutStack + ?Sized, B: CutStack + ?Sized>(
    cut_a: &A,
    cut_b: &B,
    geometry: &PlaneGeometry,
    plane_offset: [isize; 2],
    params: &MeasureParams,
) -> Result<Measurement> {
    let n_slices = cut_a.len();
    if cut_b.len() != n_slices {
        return Err(Error::invalid(format!(
            "the two cut stacks are {n_slices} and {} planes deep",
            cut_b.len()
        )));
    }
    let (shape_a, shape_b) = (geometry.cut_shape_a(), geometry.cut_shape_b());
    for index in 0..n_slices {
        for (which, dim, shape) in [
            ("first", cut_a.dim(index), shape_a),
            ("second", cut_b.dim(index), shape_b),
        ] {
            if dim != (shape[0], shape[1]) {
                return Err(Error::invalid(format!(
                    "cut plane {index} of the {which} tile is {dim:?}, not {shape:?}"
                )));
            }
        }
    }
    let [rows, cols] = geometry.landscape_shape();
    let mut landscapes = vec![Array2::<f64>::zeros((rows, cols)); n_slices];
    let mut status = vec![AlignmentStatus::INVALID; n_slices];
    let displacement_of =
        |shift: &Vec<isize>| vec![shift[0] + plane_offset[0], shift[1] + plane_offset[1]];

    let stack_rejected = params.stack_validation.is_some_and(|validation| {
        stack_rejects(validation, cut_a) || stack_rejects(validation, cut_b)
    });
    if stack_rejected {
        // No signal at all: every slice keeps the rigid offset.
        status.fill(AlignmentStatus::NO_SIGNAL);
        let shifts = vec![vec![0isize; 2]; n_slices];
        return Ok(Measurement {
            track: Track::new(
                shifts.iter().map(displacement_of).collect(),
                vec![f64::NEG_INFINITY; n_slices],
                status,
            )?,
            shifts,
            landscapes,
            measured: 0,
        });
    }

    // Which slices are measured, and the slice validation on the raw planes.
    let step = params.slice_step.max(1);
    let mut to_measure = Vec::new();
    for (index, state) in status.iter_mut().enumerate() {
        if index % step != 0 {
            *state = AlignmentStatus::UNALIGNED;
            continue;
        }
        if let Some(validation) = params.slice_validation {
            let accepts =
                |(foreground, total): (usize, usize)| validation.accepts_count(foreground, total);
            let ok = accepts(plane_counts(validation, cut_a, index))
                && accepts(plane_counts(validation, cut_b, index));
            if !ok {
                *state = AlignmentStatus::NO_SIGNAL;
                continue;
            }
        }
        to_measure.push(index);
    }

    // The stacks' own mean and norm, applied to each plane as it is read.
    let scaling = |stack: &dyn Fn() -> (f64, f64)| {
        params.prepare.then(|| {
            let (mean, norm) = stack();
            (mean, if norm == 0.0 { 0.0 } else { 1.0 / norm })
        })
    };
    let scale_a = scaling(&|| stack_statistics_of(cut_a));
    let scale_b = scaling(&|| stack_statistics_of(cut_b));
    let prepared = |plane: Array2<f64>, scale: Option<(f64, f64)>| match scale {
        None => plane,
        Some((mean, scale)) => {
            let mut plane = plane;
            plane.mapv_inplace(|value| (value - mean) * scale);
            plane
        }
    };
    let projected = |stack: &dyn Fn(usize) -> Array2<f64>, local: usize, slab: (usize, usize)| {
        let start = local.saturating_sub(slab.0);
        let stop = (local + slab.1 + 1).min(n_slices);
        let mut out = stack(start);
        for index in start + 1..stop {
            let plane = stack(index);
            out.zip_mut_with(&plane, |left, &right| *left = left.max(right));
        }
        out
    };

    let plan = SquaredDifference::new(shape_a, shape_b, geometry.window()?, Padding::Smooth)?;
    let measured = to_measure
        .par_iter()
        .map_init(
            || plan.clone(),
            |plan, &index| -> Result<(usize, Array2<f64>)> {
                let plane_a = |i: usize| prepared(cut_a.plane(i), scale_a);
                let plane_b = |i: usize| prepared(cut_b.plane(i), scale_b);
                let (mut a, mut b) = match params.axis_mip {
                    None => (plane_a(index), plane_b(index)),
                    Some(slab) => (
                        projected(&plane_a, index, slab),
                        projected(&plane_b, index, slab),
                    ),
                };
                if params.prepare_slice {
                    prepare_normalization(std::slice::from_mut(&mut a));
                    prepare_normalization(std::slice::from_mut(&mut b));
                }
                let landscape = plan.landscape(a.view(), b.view())?;
                Ok((index, landscape.mean_squared().to_owned()))
            },
        )
        .collect::<Result<Vec<_>>>()?;
    let measured_count = measured.len();
    for (index, landscape) in measured {
        landscapes[index] = landscape;
        status[index] = AlignmentStatus::MEASURED;
    }

    let (mut shifts, qualities) = match params.find_shifts {
        ShiftMethod::Minimization => shifts_from_minimization(&landscapes, &mut status),
        ShiftMethod::Tracing => shifts_from_tracing(&landscapes, &mut status, params)?,
    };
    // A landscape index becomes a shift.
    for shift in &mut shifts {
        shift[0] += geometry.shift_min[0];
        shift[1] += geometry.shift_min[1];
    }
    Ok(Measurement {
        track: Track::new(
            shifts.iter().map(displacement_of).collect(),
            qualities,
            status,
        )?,
        shifts,
        landscapes,
        measured: measured_count,
    })
}

/// Every overlapping pair of a layout, measured, as the [`WobblyLayout`] the
/// placement takes.
///
/// `positions` are the rigid positions. Each pair is cut in its tiles' own
/// local coordinates: the in-plane rectangles from [`PlaneGeometry`], and along
/// the wobble axis the intersection of the two tiles' spans.
pub fn measure_layout<T: VoxelElement>(
    positions: &[[isize; 3]],
    pairs: &[(usize, usize)],
    tiles: &[ArrayView3<'_, T>],
    axis: usize,
    params: &MeasureParams,
) -> Result<WobblyLayout> {
    measure_layout_boxes(positions, pairs, tiles, axis, params)
}

/// One pair's cuts: the plane geometry, the planes along the wobble axis
/// both tiles span, and where they start in each tile.
struct PairCut {
    geometry: PlaneGeometry,
    start: isize,
    stop: isize,
}

fn pair_cut(
    positions: &[[isize; 3]],
    shapes: &[[usize; 3]],
    (pre, post): (usize, usize),
    axis: usize,
    plane: [usize; 2],
    params: &MeasureParams,
) -> Result<PairCut> {
    if pre >= shapes.len() || post >= shapes.len() {
        return Err(Error::invalid(format!(
            "pair ({pre}, {post}) names a tile beyond the {} given",
            shapes.len()
        )));
    }
    let (a_pos, b_pos) = (positions[pre], positions[post]);
    let (a_shape, b_shape) = (shapes[pre], shapes[post]);
    let geometry = PlaneGeometry::new(
        plane.map(|p| a_pos[p]),
        plane.map(|p| a_shape[p]),
        plane.map(|p| b_pos[p]),
        plane.map(|p| b_shape[p]),
        params.max_shifts,
    )?;
    let start = a_pos[axis].max(b_pos[axis]);
    let stop = (a_pos[axis] + a_shape[axis] as isize).min(b_pos[axis] + b_shape[axis] as isize);
    if stop <= start {
        return Err(Error::invalid(format!(
            "tiles {pre} and {post} do not overlap along axis {axis}: {start} to {stop}"
        )));
    }
    Ok(PairCut {
        geometry,
        start,
        stop,
    })
}

/// The plane axes of a wobble axis.
fn plane_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// The box of a tile a cut covers: `ranges` on the plane axes, planes
/// `first..first + n` along the wobble axis.
fn cut_box(
    axis: usize,
    plane: [usize; 2],
    ranges: [(usize, usize); 2],
    first: usize,
    n: usize,
) -> BoxAt {
    let mut start = [0usize; 3];
    let mut shape = [0usize; 3];
    start[axis] = first;
    shape[axis] = n;
    for (d, &p) in plane.iter().enumerate() {
        start[p] = ranges[d].0;
        shape[p] = ranges[d].1 - ranges[d].0;
    }
    (start, shape)
}

/// For each pair, boxes of its two tiles (`pre`, then `post`) holding the cut
/// boxes [`measure_layout_boxes`] reads at any positions whose difference
/// (`post - pre`) is within `margins[pair][axis]` (inclusive) of its
/// difference at `positions`: what to read ahead before the positions are
/// final. `None` for a pair with no such box.
pub fn cut_boxes_within(
    positions: &[[isize; 3]],
    pairs: &[(usize, usize)],
    shapes: &[[usize; 3]],
    axis: usize,
    params: &MeasureParams,
    margins: &[[(isize, isize); 3]],
) -> Vec<Option<[BoxAt; 2]>> {
    if axis > 2 || positions.len() != shapes.len() {
        return vec![None; pairs.len()];
    }
    let plane = plane_axes(axis);
    pairs
        .iter()
        .zip(margins)
        .map(|(&(pre, post), margin)| {
            if pre >= shapes.len() || post >= shapes.len() {
                return None;
            }
            // A move of `post` by `d` is a shift of the search range by `d`.
            let widened: [[isize; 2]; 2] = std::array::from_fn(|d| {
                let p = plane[d];
                [
                    params.max_shifts[d][0] + margin[p].0,
                    params.max_shifts[d][1] + margin[p].1,
                ]
            });
            let (a, b) = (positions[pre], positions[post]);
            let (sa, sb) = (shapes[pre], shapes[post]);
            let geometry = PlaneGeometry::new(
                plane.map(|p| a[p]),
                plane.map(|p| sa[p]),
                plane.map(|p| b[p]),
                plane.map(|p| sb[p]),
                widened,
            )
            .ok()?;
            // Along the wobble axis, the union of the spans both tiles share.
            let (low, high) = margin[axis];
            let start = a[axis].max(b[axis] + low);
            let stop = (a[axis] + sa[axis] as isize).min(b[axis] + sb[axis] as isize + high);
            let start_b = b[axis].max(a[axis] - high);
            let stop_b = (b[axis] + sb[axis] as isize).min(a[axis] + sa[axis] as isize - low);
            if stop <= start || stop_b <= start_b {
                return None;
            }
            Some([
                cut_box(
                    axis,
                    plane,
                    geometry.cut_a,
                    (start - a[axis]) as usize,
                    (stop - start) as usize,
                ),
                cut_box(
                    axis,
                    plane,
                    geometry.cut_b,
                    (start_b - b[axis]) as usize,
                    (stop_b - start_b) as usize,
                ),
            ])
        })
        .collect()
}

/// [`measure_layout`] reading each pair's cuts from `tiles`, pairs in
/// parallel as the memory cap allows ([`crate::tiles::admitted`]): a pair
/// holds its two cut boxes, their `f64` planes and its landscapes while it is
/// measured.
pub fn measure_layout_boxes<T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    positions: &[[isize; 3]],
    pairs: &[(usize, usize)],
    tiles: &B,
    axis: usize,
    params: &MeasureParams,
) -> Result<WobblyLayout> {
    measure_layout_kept(positions, pairs, tiles, axis, params, Vec::new())
}

/// [`measure_layout_boxes`], cutting a pair from the boxes `kept[pair]` (read
/// by an earlier step, [`crate::rigid::align_layout_keeping`]) where they hold
/// its cuts, and reading them otherwise. A kept pair is let go as soon as it
/// is measured, or at once if it does not hold the cuts.
pub fn measure_layout_kept<'a, T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    positions: &[[isize; 3]],
    pairs: &[(usize, usize)],
    tiles: &'a B,
    axis: usize,
    params: &MeasureParams,
    kept: Vec<Option<KeptPair<'a, T>>>,
) -> Result<WobblyLayout> {
    if positions.len() != tiles.len() {
        return Err(Error::invalid(format!(
            "{} positions for {} tiles",
            positions.len(),
            tiles.len()
        )));
    }
    if axis > 2 {
        return Err(Error::invalid(format!(
            "the wobble axis is {axis} and a volume has three"
        )));
    }
    let shapes = (0..tiles.len())
        .map(|tile| tiles.shape(tile))
        .collect::<Vec<_>>();
    let plane = plane_axes(axis);
    // The two cut boxes of a pair, over the planes both tiles span.
    let boxes = |cut: &PairCut, (pre, post): (usize, usize)| -> [BoxAt; 2] {
        let n = (cut.stop - cut.start) as usize;
        [
            cut_box(
                axis,
                plane,
                cut.geometry.cut_a,
                (cut.start - positions[pre][axis]) as usize,
                n,
            ),
            cut_box(
                axis,
                plane,
                cut.geometry.cut_b,
                (cut.start - positions[post][axis]) as usize,
                n,
            ),
        ]
    };
    // Each pair with what was kept for it, if that holds its cuts; the rest
    // is let go now.
    let mut kept = kept.into_iter();
    let items = pairs
        .iter()
        .map(|&pair| {
            let held = kept.next().flatten().filter(|held| {
                pair_cut(positions, &shapes, pair, axis, plane, params).is_ok_and(|cut| {
                    let [a, b] = boxes(&cut, pair);
                    held.pre.covers(a) && held.post.covers(b)
                })
            });
            (pair, held)
        })
        .collect::<Vec<_>>();
    drop(kept);
    let workers = rayon::current_num_threads() as u64;
    let bytes = |((pre, post), held): &((usize, usize), Option<KeptPair<'a, T>>)| -> u64 {
        let Ok(cut) = pair_cut(positions, &shapes, (*pre, *post), axis, plane, params) else {
            return 0;
        };
        let n = (cut.stop - cut.start) as usize;
        let [(_, a), (_, b)] = boxes(&cut, (*pre, *post));
        let plane = |shape: [usize; 3]| (shape.iter().product::<usize>() / n.max(1)) as u64;
        let [rows, cols] = cut.geometry.landscape_shape();
        // The boxes as read (unless they were kept, and leased then), a
        // landscape per plane, and the planes each worker holds in `f64`
        // while it measures one (both tiles', prepared and padded).
        let read = if held.is_some() {
            0
        } else {
            tiles.read_bytes(a) + tiles.read_bytes(b)
        };
        read + 8 * (n * rows * cols) as u64 + workers * 4 * 8 * (plane(a) + plane(b))
    };
    let alignments = admitted(items, bytes, tiles.ceiling(), |((pre, post), held)| {
        let cut = pair_cut(positions, &shapes, (pre, post), axis, plane, params)?;
        let [a, b] = boxes(&cut, (pre, post));
        let read;
        let (view_a, view_b) = match &held {
            Some(held) => (held.pre.view_of(a), held.post.view_of(b)),
            None => {
                read = [tiles.read(pre, a.0, a.1)?, tiles.read(post, b.0, b.1)?];
                (read[0].view(), read[1].view())
            }
        };
        let measurement = measure_stacks(
            &BoxCuts { view: view_a, axis },
            &BoxCuts { view: view_b, axis },
            &cut.geometry,
            plane.map(|p| positions[post][p] - positions[pre][p]),
            params,
        )?;
        Ok(AlignmentSlot {
            pre,
            post,
            track: measurement.track,
        })
    })?;
    Ok(WobblyLayout {
        sources: positions
            .iter()
            .zip(&shapes)
            .map(|(position, shape)| SourceSlab {
                position: position.to_vec(),
                height: shape[axis],
            })
            .collect(),
        alignments,
        axis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bartlett_taps_are_a_triangle() {
        assert_eq!(Window::Bartlett.taps(5), vec![0.0, 0.5, 1.0, 0.5, 0.0]);
        assert_eq!(Window::Bartlett.taps(1), vec![1.0]);
        assert_eq!(Window::Flat.taps(3), vec![1.0; 3]);
    }

    #[test]
    fn a_degenerate_window_is_refused() {
        let params = SmoothParams {
            window: Some(Window::Bartlett),
            window_length: 100,
            binary: None,
        };
        let error = smooth_window(&[3, 5], params).expect_err("a refusal");
        assert!(error.to_string().contains("sums to zero"), "{error}");
        assert!(smooth_window(&[1, 1, 1], params).is_ok());
    }

    #[test]
    fn smoothing_a_constant_is_the_identity() {
        let params = SmoothParams {
            window: Some(Window::Bartlett),
            window_length: 10,
            binary: Some(3),
        };
        assert_eq!(
            smooth_window(&[4; 20], params).expect("smooth"),
            vec![4; 20]
        );
    }

    #[test]
    fn a_step_is_smoothed_symmetrically() {
        let values = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10];
        let smoothed = smooth_window(
            &values,
            SmoothParams {
                window: Some(Window::Flat),
                window_length: 3,
                binary: None,
            },
        )
        .expect("smooth");
        assert_eq!(smoothed, vec![0, 0, 0, 0, 3, 7, 10, 10, 10, 10]);
    }

    #[test]
    fn the_binary_smoothing_removes_narrow_excursions() {
        let values = [
            0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 10, 10, 0, 0, -3, -3, -3, 0, 0, 0,
        ];
        assert_eq!(smooth_binary(&values, 1), values.to_vec());
        assert_eq!(smooth_binary(&values, 3), vec![0; 20]);
    }

    #[test]
    fn the_local_minima_admit_a_plateau_and_refuse_a_constant() {
        let at = |values: &[f64]| {
            let array = Array2::from_shape_vec((5, 5), values.to_vec()).expect("an array");
            detect_local_minima(array.view(), 1)
        };
        let well = [
            9., 9., 9., 9., 9., 9., 5., 4., 5., 9., 9., 4., 1., 4., 9., 9., 5., 4., 5., 9., 9., 9.,
            9., 9., 9.,
        ];
        assert_eq!(at(&well), (vec![[2, 2]], vec![1.0]));
        let plateau = [
            9., 9., 9., 9., 9., 9., 2., 2., 7., 9., 9., 2., 2., 7., 9., 9., 7., 7., 3., 9., 9., 9.,
            9., 9., 9.,
        ];
        assert_eq!(at(&plateau).0, vec![[1, 1], [1, 2], [2, 1], [2, 2]]);
        assert_eq!(at(&[3.0; 25]).0, Vec::<[usize; 2]>::new());
        // A frame of zeroes beats every interior point.
        let frame = [
            0., 0., 0., 0., 0., 0., 1., 5., 1., 0., 0., 5., 5., 5., 0., 0., 1., 5., 2., 0., 0., 0.,
            0., 0., 0.,
        ];
        assert!(at(&frame).0.is_empty());
    }

    #[test]
    fn the_tracker_links_nearby_points() {
        let positions = vec![
            vec![[5.0, 6.0], [10.0, 10.0]],
            vec![[5.0, 7.0], [11.0, 10.0], [30.0, 10.0]],
            vec![[9.0, 10.0], [31.0, 9.0]],
        ];
        let expected = vec![
            vec![(0usize, 0usize), (1, 0)],
            vec![(0, 1), (1, 1), (2, 0)],
            vec![(1, 2), (2, 1)],
        ];
        for (cost, cutoff) in [
            (None, None),
            (Some(5.0), Some(3.0 * std::f64::consts::SQRT_2)),
        ] {
            assert_eq!(
                track_positions(&positions, cost, cutoff).expect("a trace"),
                expected,
                "cost {cost:?} cutoff {cutoff:?}"
            );
        }
    }

    #[test]
    fn the_foreground_test_is_inclusive_at_both_ends() {
        let values = [199.0, 200.0, 201.0, 19_999.0, 20_000.0, 20_001.0];
        assert_eq!(Foreground::slice_default().count(values.iter().copied()), 4);
        assert_eq!(Foreground::stack_default().count(values.iter().copied()), 5);
        assert!(Foreground::stack_default().accepts_count(1, 10));
        assert!(!Foreground::stack_default().accepts_count(0, 10));
        assert!(Foreground::slice_default().accepts_count(1500, 10_000));
        assert!(!Foreground::slice_default().accepts_count(1499, 10_000));
    }

    #[test]
    fn the_preparation_is_over_the_stack_and_not_the_plane() {
        let mut planes = vec![
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("a plane"),
            Array2::from_shape_vec((2, 2), vec![101.0, 102.0, 103.0, 104.0]).expect("a plane"),
        ];
        let (mean, norm) = prepare_normalization(&mut planes);
        assert_eq!(mean, 52.5);
        assert!(norm > 0.0);
        let energy: f64 = planes.iter().flat_map(|p| p.iter()).map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1.0e-12, "the stack has unit energy");
        let first = planes[0].iter().sum::<f64>() / 4.0;
        assert!(
            first < -0.3,
            "each plane keeps its offset from the stack mean"
        );
    }

    #[test]
    fn the_two_status_vocabularies_are_not_one() {
        assert_eq!(AlignmentStatus::UNTRACED, SourceStatus::ISOLATED);
        assert_eq!(status_name(-2), "untraced");
        assert_eq!(source_status_name(-2), "isolated");
        for status in AlignmentStatus::ALL {
            assert_ne!(status_name(status), "unknown");
        }
        for status in SourceStatus::ALL {
            assert_ne!(source_status_name(status), "unknown");
        }
        assert_eq!(status_name(7), "unknown");
    }
}
