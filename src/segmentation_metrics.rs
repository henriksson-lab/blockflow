// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the published definitions of
// the metrics (Meila 2007 for VOI; the SNEMI3D / CREMI challenge definition of
// adapted Rand error; Januszewski et al. 2018 for expected run length), not
// adapted from any implementation.
//
// Scoring a dense instance segmentation against ground truth.
//
// [`crate::agreement`] answers *which* objects split and merged. This module
// answers *how much*, with the numbers the connectomics literature reports, so
// a run here can be put beside a published table:
//
// | metric | question it answers |
// |---|---|
// | [`Voi`] | information-theoretic split / merge, in bits |
// | [`AdaptedRand`] | pair-counting F-score, the SNEMI3D / CREMI headline |
// | [`EditCounts`] | how many split and merge clicks a proofreader needs |
// | [`BoundaryF1`] | per-z-slice boundary precision/recall within a tolerance |
// | [`SkeletonScores`] | skeleton recall, ERL and NERL for thin processes |
//
// Everything but the boundary and skeleton metrics is a function of one
// sparse [`Contingency`] table, which is built in parallel and is **mergeable**:
// tables of disjoint blocks add up to the table of their union, so a volume
// that does not fit can be scored block by block.
//
// The background conventions are where published numbers silently differ, so
// they are explicit parameters ([`ContingencyOptions`]) rather than a choice
// made here. CREMI's is `ignore_truth_zero = true` with produced `0` as
// [`ProducedZero::Singletons`].

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;

use crate::error::{ensure, Error, Result};
use crate::ops::boundaries::find_boundaries_into;
use crate::ops::components::Union;
use crate::ops::distance::{distance_transform, DistanceParams, Unbounded};

// ---------------------------------------------------------------------------
// Contingency table
// ---------------------------------------------------------------------------

/// A multiplicative hasher for `(u32, u32)` keys: SipHash costs more than the
/// rest of the accumulation put together at 1e8 voxels, and these keys are not
/// attacker-controlled.
#[derive(Default, Clone, Copy)]
struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(byte as u64);
        }
    }
    fn write_u32(&mut self, value: u32) {
        self.write_u64(value as u64);
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<PairHasher>>;

/// What a produced voxel labelled `0` means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducedZero {
    /// `0` is one more object, like any other id (e.g. a segmentation whose
    /// background class is a genuine prediction).
    Object,
    /// `0` is *unassigned*: every such voxel is its own singleton object. This
    /// is the CREMI convention — unassigned voxels cost recall (they are split
    /// from everything) but never precision.
    Singletons,
    /// Voxels where the produced label is `0` are dropped from the comparison.
    Ignore,
}

/// The background conventions a [`Contingency`] is built under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContingencyOptions {
    /// Drop voxels whose truth label is `0` (CREMI: truth `0` is "not
    /// annotated", not an object). When `false`, truth `0` is an object.
    pub ignore_truth_zero: bool,
    /// See [`ProducedZero`].
    pub produced_zero: ProducedZero,
}

impl ContingencyOptions {
    /// The CREMI / SNEMI3D convention: ignore truth `0`, produced `0` is
    /// unassigned singletons.
    pub fn cremi() -> Self {
        Self {
            ignore_truth_zero: true,
            produced_zero: ProducedZero::Singletons,
        }
    }

    /// Every id, `0` included, is an object on both sides.
    pub fn all_objects() -> Self {
        Self {
            ignore_truth_zero: false,
            produced_zero: ProducedZero::Object,
        }
    }
}

/// Sparse overlap counts between a truth and a produced labelling.
///
/// `pairs[(t, p)]` is the number of voxels with truth `t` and produced `p`,
/// over the voxels the options keep. Under [`ProducedZero::Singletons`] the
/// produced-`0` voxels are not in `pairs`; they are counted per truth label in
/// [`Self::unassigned`], each standing for its own one-voxel object.
#[derive(Debug, Clone, PartialEq)]
pub struct Contingency {
    options: ContingencyOptions,
    pairs: FastMap<(u32, u32), u64>,
    unassigned: FastMap<u32, u64>,
    total: u64,
}

impl Contingency {
    /// An empty table, ready for [`Self::accumulate`] or [`Self::merge`].
    pub fn new(options: ContingencyOptions) -> Self {
        Self {
            options,
            pairs: FastMap::default(),
            unassigned: FastMap::default(),
            total: 0,
        }
    }

    /// The table of two whole volumes, accumulated in parallel over rows.
    pub fn from_volumes(
        truth: ArrayView3<u32>,
        produced: ArrayView3<u32>,
        options: ContingencyOptions,
    ) -> Result<Self> {
        let mut table = Self::new(options);
        table.accumulate(truth, produced)?;
        Ok(table)
    }

    /// Add the voxels of one block (any shape, the two views aligned).
    ///
    /// Blocks must be disjoint: a voxel accumulated twice counts twice.
    pub fn accumulate(&mut self, truth: ArrayView3<u32>, produced: ArrayView3<u32>) -> Result<()> {
        if truth.shape() != produced.shape() {
            return Err(Error::ShapeMismatch {
                expected: truth.shape().to_vec(),
                got: produced.shape().to_vec(),
            });
        }
        let [nz, ny, _] = [truth.shape()[0], truth.shape()[1], truth.shape()[2]];
        let options = self.options;
        let part = (0..nz * ny)
            .into_par_iter()
            .fold(
                || Self::new(options),
                |mut acc, row| {
                    let (z, y) = (row / ny, row % ny);
                    let t = truth.slice(s![z, y, ..]);
                    let p = produced.slice(s![z, y, ..]);
                    acc.add_row(t.iter().copied().zip(p.iter().copied()));
                    acc
                },
            )
            .reduce(|| Self::new(options), |a, b| a.merged_with(b));
        self.merge(part)
    }

    /// One row, run-length compressed before hashing: a segmentation is mostly
    /// long runs of one pair, so this removes most hash lookups.
    fn add_row(&mut self, row: impl Iterator<Item = (u32, u32)>) {
        let mut current: Option<(u32, u32)> = None;
        let mut run = 0u64;
        for pair in row {
            if current == Some(pair) {
                run += 1;
                continue;
            }
            if let Some(previous) = current {
                self.add(previous, run);
            }
            current = Some(pair);
            run = 1;
        }
        if let Some(previous) = current {
            self.add(previous, run);
        }
    }

    fn add(&mut self, (t, p): (u32, u32), count: u64) {
        if t == 0 && self.options.ignore_truth_zero {
            return;
        }
        if p == 0 {
            match self.options.produced_zero {
                ProducedZero::Ignore => return,
                ProducedZero::Singletons => {
                    *self.unassigned.entry(t).or_default() += count;
                    self.total += count;
                    return;
                }
                ProducedZero::Object => {}
            }
        }
        *self.pairs.entry((t, p)).or_default() += count;
        self.total += count;
    }

    fn merged_with(mut self, other: Self) -> Self {
        // Options are equal by construction inside `accumulate`.
        self.merge_unchecked(other);
        self
    }

    fn merge_unchecked(&mut self, other: Self) {
        let options = self.options;
        let mine = std::mem::replace(self, Self::new(options));
        let (mut big, small) = if mine.pairs.len() >= other.pairs.len() {
            (mine, other)
        } else {
            (other, mine)
        };
        for (key, count) in small.pairs {
            *big.pairs.entry(key).or_default() += count;
        }
        for (key, count) in small.unassigned {
            *big.unassigned.entry(key).or_default() += count;
        }
        big.total += small.total;
        *self = big;
    }

    /// Add another table — of a disjoint block — into this one. Refuses tables
    /// built under different options, since their counts mean different things.
    pub fn merge(&mut self, other: Contingency) -> Result<()> {
        ensure!(
            self.options == other.options,
            "Contingency::merge: options differ ({:?} vs {:?})",
            self.options,
            other.options
        );
        self.merge_unchecked(other);
        Ok(())
    }

    pub fn options(&self) -> ContingencyOptions {
        self.options
    }

    /// Voxels counted, unassigned singletons included.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Overlap count per `(truth, produced)` pair (singletons excluded).
    pub fn pairs(&self) -> impl Iterator<Item = ((u32, u32), u64)> + '_ {
        self.pairs.iter().map(|(&key, &count)| (key, count))
    }

    /// Unassigned (produced-`0` singleton) voxels per truth label; empty unless
    /// [`ProducedZero::Singletons`].
    pub fn unassigned(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.unassigned.iter().map(|(&key, &count)| (key, count))
    }

    /// Size of every truth object over the counted voxels (unassigned included).
    pub fn truth_sizes(&self) -> HashMap<u32, u64> {
        let mut sizes = HashMap::new();
        for (&(t, _), &count) in &self.pairs {
            *sizes.entry(t).or_default() += count;
        }
        for (&t, &count) in &self.unassigned {
            *sizes.entry(t).or_default() += count;
        }
        sizes
    }

    /// Size of every produced object over the counted voxels (singletons are
    /// not listed: each has size 1).
    pub fn produced_sizes(&self) -> HashMap<u32, u64> {
        let mut sizes = HashMap::new();
        for (&(_, p), &count) in &self.pairs {
            *sizes.entry(p).or_default() += count;
        }
        sizes
    }

    fn singleton_count(&self) -> u64 {
        self.unassigned.values().sum()
    }

    /// Variation of information, in bits.
    ///
    /// With `p_tp = n_tp / N` over the counted voxels, `split = H(P | T)` (the
    /// information needed to recover the produced label knowing the truth —
    /// non-zero when truth objects are cut up) and `merge = H(T | P)`. Each
    /// singleton is an object of size 1, so it adds to split only.
    pub fn voi(&self) -> Voi {
        let n = self.total as f64;
        if self.total == 0 {
            return Voi { split: 0.0, merge: 0.0 };
        }
        let xlogx = |x: u64| if x == 0 { 0.0 } else { x as f64 * (x as f64).log2() };
        // Singleton cells have n = 1: x log x = 0, on both the joint and the
        // produced side.
        let joint: f64 = self.pairs.values().map(|&c| xlogx(c)).sum();
        let truth: f64 = self.truth_sizes().values().map(|&c| xlogx(c)).sum();
        let produced: f64 = self.produced_sizes().values().map(|&c| xlogx(c)).sum();
        Voi {
            split: ((truth - joint) / n).max(0.0),
            merge: ((produced - joint) / n).max(0.0),
        }
    }

    /// Adapted Rand error, the SNEMI3D / CREMI definition.
    ///
    /// `precision = Σ n_tp² / Σ b_p²`, `recall = Σ n_tp² / Σ a_t²` (with `a`,
    /// `b` the truth and produced sizes, singletons contributing 1 each to the
    /// joint and produced sums), `error = 1 − 2PR / (P + R)`. On an empty table
    /// both are 1 and the error 0.
    pub fn adapted_rand(&self) -> AdaptedRand {
        let square = |x: u64| (x as u128) * (x as u128);
        let singletons = self.singleton_count() as u128;
        let joint: u128 = self.pairs.values().map(|&c| square(c)).sum::<u128>() + singletons;
        let truth: u128 = self.truth_sizes().values().map(|&c| square(c)).sum();
        let produced: u128 =
            self.produced_sizes().values().map(|&c| square(c)).sum::<u128>() + singletons;
        let ratio = |a: u128, b: u128| if b == 0 { 1.0 } else { a as f64 / b as f64 };
        let precision = ratio(joint, produced);
        let recall = ratio(joint, truth);
        let f = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        AdaptedRand {
            error: 1.0 - f,
            precision,
            recall,
        }
    }

    /// Proofreading edit counts at a minimum-overlap threshold in voxels.
    ///
    /// A `(t, p)` pair is *significant* when it shares at least
    /// `min_overlap_voxels` voxels (so boundary slivers are ignored). Then
    /// `merges = Σ_t max(0, #significant p − 1)` — a truth object in k pieces
    /// takes k − 1 merge operations — and `splits = Σ_p max(0, #significant t −
    /// 1)`. Label `0` on either side is never an object here, whatever the
    /// options, and unassigned singletons need no edit.
    pub fn edit_counts(&self, min_overlap_voxels: u64) -> EditCounts {
        self.edit_counts_relative(min_overlap_voxels, 0.0)
    }

    /// [`Self::edit_counts`] with a second, relative significance rule: a
    /// pair must also share at least `min_fraction` of **the smaller** of the
    /// two objects (the rule `agreement::compare_labels` uses).
    ///
    /// The absolute threshold alone does not scale: on a large cell a
    /// boundary displaced by one voxel shares thousands of voxels with its
    /// neighbour, which no proofreader would call a split. A fraction of the
    /// smaller object ignores that sliver and still sees a small process
    /// glued to a large cell, because the process is the smaller object.
    pub fn edit_counts_relative(&self, min_overlap_voxels: u64, min_fraction: f64) -> EditCounts {
        let truth_sizes = if min_fraction > 0.0 { self.truth_sizes() } else { HashMap::new() };
        let produced_sizes = if min_fraction > 0.0 { self.produced_sizes() } else { HashMap::new() };
        let mut per_truth: HashMap<u32, u64> = HashMap::new();
        let mut per_produced: HashMap<u32, u64> = HashMap::new();
        for (&(t, p), &count) in &self.pairs {
            if t == 0 || p == 0 || count < min_overlap_voxels.max(1) {
                continue;
            }
            if min_fraction > 0.0 {
                let smaller = truth_sizes[&t].min(produced_sizes[&p]) as f64;
                if (count as f64) < min_fraction * smaller {
                    continue;
                }
            }
            *per_truth.entry(t).or_default() += 1;
            *per_produced.entry(p).or_default() += 1;
        }
        let excess = |m: &HashMap<u32, u64>| m.values().map(|&k| k.saturating_sub(1)).sum();
        EditCounts {
            merges: excess(&per_truth),
            splits: excess(&per_produced),
            min_overlap_voxels,
            min_fraction,
        }
    }
}

/// Variation of information, in bits. See [`Contingency::voi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voi {
    /// `H(produced | truth)`: over-segmentation.
    pub split: f64,
    /// `H(truth | produced)`: under-segmentation.
    pub merge: f64,
}

impl Voi {
    pub fn total(&self) -> f64 {
        self.split + self.merge
    }
}

/// Adapted Rand error and its two halves. See [`Contingency::adapted_rand`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptedRand {
    /// `1 − F`; 0 is perfect.
    pub error: f64,
    /// Low when produced objects merge truth objects.
    pub precision: f64,
    /// Low when truth objects are split.
    pub recall: f64,
}

/// Proofreading operations to turn produced into truth. See
/// [`Contingency::edit_counts`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditCounts {
    /// Split operations needed (produced objects covering several truths).
    pub splits: u64,
    /// Merge operations needed (truth objects cut into several pieces).
    pub merges: u64,
    /// The thresholds the counts were taken at.
    pub min_overlap_voxels: u64,
    /// Fraction of the smaller object; 0 when only the absolute rule applied.
    pub min_fraction: f64,
}

// ---------------------------------------------------------------------------
// Boundary F1
// ---------------------------------------------------------------------------

/// Boundary precision / recall on 2D z-slices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryF1 {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// The in-plane tolerance in pixels.
    pub tolerance: f64,
}

/// Boundary F1 over every z-slice, pooled.
///
/// In each slice a pixel is a boundary pixel when its in-plane 4-neighbourhood
/// holds a different label (the predicate of
/// [`find_boundaries_into`], restricted to the plane — both sides of a label
/// change are marked, on truth and produced alike). A produced boundary pixel
/// is a true positive when an in-plane Euclidean distance transform puts a
/// truth boundary pixel within `tolerance` pixels, and symmetrically for
/// recall. Counts are pooled over all slices before the ratios (a slice with
/// more boundary weighs more); an empty side scores 1.
///
/// With `ignore_truth_zero`, pixels where truth is `0` are not *scored* on
/// either side (they may still be the nearest match of a scored pixel).
pub fn boundary_f1(
    truth: ArrayView3<u32>,
    produced: ArrayView3<u32>,
    tolerance: f64,
    ignore_truth_zero: bool,
) -> Result<BoundaryF1> {
    if truth.shape() != produced.shape() {
        return Err(Error::ShapeMismatch {
            expected: truth.shape().to_vec(),
            got: produced.shape().to_vec(),
        });
    }
    ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "boundary_f1: tolerance {tolerance} must be finite and non-negative"
    );
    let params = DistanceParams::default().with_unbounded(Unbounded::Infinite);
    let [_, ny, nx] = [truth.shape()[0], truth.shape()[1], truth.shape()[2]];
    let counts = (0..truth.shape()[0])
        .into_par_iter()
        .map(|z| -> Result<[u64; 4]> {
            let t = truth.slice(s![z..z + 1, .., ..]);
            let p = produced.slice(s![z..z + 1, .., ..]);
            let mut tb = Array3::<bool>::from_elem((1, ny, nx), false);
            let mut pb = Array3::<bool>::from_elem((1, ny, nx), false);
            find_boundaries_into(t, tb.view_mut())?;
            find_boundaries_into(p, pb.view_mut())?;
            let to_truth = distance_transform(tb.mapv(|b| !b).view(), &params)?;
            let to_produced = distance_transform(pb.mapv(|b| !b).view(), &params)?;
            let mut c = [0u64; 4]; // produced total, produced hit, truth total, truth hit
            for ((index, &label), (&is_t, &is_p)) in
                t.indexed_iter().zip(tb.iter().zip(pb.iter()))
            {
                if ignore_truth_zero && label == 0 {
                    continue;
                }
                if is_p {
                    c[0] += 1;
                    c[1] += (to_truth[index] <= tolerance) as u64;
                }
                if is_t {
                    c[2] += 1;
                    c[3] += (to_produced[index] <= tolerance) as u64;
                }
            }
            Ok(c)
        })
        .try_reduce(|| [0u64; 4], |a, b| Ok([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]))?;
    let ratio = |hit: u64, all: u64| if all == 0 { 1.0 } else { hit as f64 / all as f64 };
    let precision = ratio(counts[1], counts[0]);
    let recall = ratio(counts[3], counts[2]);
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok(BoundaryF1 {
        precision,
        recall,
        f1,
        tolerance,
    })
}

// ---------------------------------------------------------------------------
// The one-call summary
// ---------------------------------------------------------------------------

/// What [`score_segmentation`] computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentationOptions {
    pub contingency: ContingencyOptions,
    /// See [`Contingency::edit_counts_relative`].
    pub min_overlap_voxels: u64,
    /// See [`Contingency::edit_counts_relative`]; 0 = absolute rule only.
    pub min_overlap_fraction: f64,
    /// In-plane tolerance in pixels for [`boundary_f1`]; `None` skips it (it
    /// costs two distance transforms per slice).
    pub boundary_tolerance: Option<f64>,
}

/// VOI, adapted Rand, edit counts and optionally boundary F1 of one labelling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentationScores {
    pub voi: Voi,
    pub rand: AdaptedRand,
    pub edits: EditCounts,
    pub boundary: Option<BoundaryF1>,
    /// Voxels scored.
    pub voxels: u64,
}

impl std::fmt::Display for SegmentationScores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VOI {:.4} (split {:.4}, merge {:.4}) | ARAND {:.4} (P {:.4}, R {:.4}) | \
             edits: {} splits, {} merges (>= {} vx)",
            self.voi.total(),
            self.voi.split,
            self.voi.merge,
            self.rand.error,
            self.rand.precision,
            self.rand.recall,
            self.edits.splits,
            self.edits.merges,
            self.edits.min_overlap_voxels
        )?;
        if let Some(b) = &self.boundary {
            write!(
                f,
                " | boundary F1 {:.4} (P {:.4}, R {:.4}, tol {})",
                b.f1, b.precision, b.recall, b.tolerance
            )?;
        }
        Ok(())
    }
}

/// Score a produced labelling against truth. See the individual metrics for the
/// definitions; all of them except boundary F1 share one [`Contingency`].
pub fn score_segmentation(
    truth: ArrayView3<u32>,
    produced: ArrayView3<u32>,
    options: &SegmentationOptions,
) -> Result<SegmentationScores> {
    let table = Contingency::from_volumes(truth, produced, options.contingency)?;
    let boundary = match options.boundary_tolerance {
        Some(tolerance) => Some(boundary_f1(
            truth,
            produced,
            tolerance,
            options.contingency.ignore_truth_zero,
        )?),
        None => None,
    };
    Ok(scores_from_table(
        &table,
        options.min_overlap_voxels,
        options.min_overlap_fraction,
        boundary,
    ))
}

/// The table-derived part of [`SegmentationScores`], for a table accumulated
/// block by block.
pub fn scores_from_table(
    table: &Contingency,
    min_overlap_voxels: u64,
    min_overlap_fraction: f64,
    boundary: Option<BoundaryF1>,
) -> SegmentationScores {
    SegmentationScores {
        voi: table.voi(),
        rand: table.adapted_rand(),
        edits: table.edit_counts_relative(min_overlap_voxels, min_overlap_fraction),
        boundary,
        voxels: table.total(),
    }
}

// ---------------------------------------------------------------------------
// Skeleton metrics
// ---------------------------------------------------------------------------

/// Ground-truth skeletons: a forest of node graphs, one per skeleton id.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Skeletons {
    /// Node positions `[z, y, x]` in voxel coordinates.
    pub nodes: Vec<[f64; 3]>,
    /// Edges as pairs of node indices; both ends must be the same skeleton.
    pub edges: Vec<[usize; 2]>,
    /// Skeleton id of each node (same length as `nodes`).
    pub skeleton_ids: Vec<u64>,
}

/// One skeleton's recall.
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonRecall {
    pub skeleton: u64,
    /// Total edge length, in the units of `spacing`.
    pub length: f64,
    /// The produced label most of its nodes lie in (`0` if none is non-zero).
    pub majority_label: u32,
    /// Fraction of `length` on edges whose both ends are in `majority_label`
    /// (0 for a skeleton without length).
    pub recall: f64,
}

/// Skeleton-based scores. See [`score_skeletons`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonScores {
    /// Per skeleton, ascending by id.
    pub per_skeleton: Vec<SkeletonRecall>,
    /// Length-weighted mean of the per-skeleton recall.
    pub recall: f64,
    /// Expected run length, in the units of `spacing`.
    pub erl: f64,
    /// ERL of the truth itself: `Σ L_s² / Σ L_s`.
    pub max_erl: f64,
    /// `erl / max_erl` (0 when there is no length at all).
    pub nerl: f64,
    /// Produced labels touched by nodes of more than one skeleton, ascending.
    pub merger_labels: Vec<u32>,
    /// Edges whose two ends lie in different labels or on `0`.
    pub split_edges: usize,
}

/// Skeleton recall, ERL and NERL of a produced labelling.
///
/// **Node labels.** Each node takes the produced label of its nearest voxel
/// (coordinates rounded); a node outside the volume is label `0`. Label `0`
/// is never an object.
///
/// **Edge length** is the Euclidean distance between its nodes with each axis
/// scaled by `spacing` (`[z, y, x]`).
///
/// **Skeleton recall** ([`SkeletonRecall`]): the majority label is the
/// non-zero label holding most of the skeleton's nodes (ties to the smaller
/// id); recall is the fraction of the skeleton's length on edges whose both
/// ends carry it. Mergers do not lower it — that is ERL's job.
///
/// **ERL** (Januszewski et al. 2018). A produced label is a *merger* when
/// nodes of more than one skeleton lie in it. An edge is *kept* when both ends
/// carry the same non-zero, non-merger label; every other edge is a break.
/// The kept edges of a skeleton fall into connected *runs* of lengths `r`, and
/// `ERL = Σ_skeletons Σ_runs r² / Σ_skeletons L_s` — the expected length of
/// correctly reconstructed neurite around a point drawn uniformly along the
/// total skeleton length. Mergers contribute zero run length for every
/// skeleton they touch.
///
/// **NERL** = `ERL / (Σ L_s² / Σ L_s)`, the ERL of the truth itself, so 1 is
/// perfect.
pub fn score_skeletons(
    skeletons: &Skeletons,
    produced: ArrayView3<u32>,
    spacing: [f64; 3],
) -> Result<SkeletonScores> {
    let n = skeletons.nodes.len();
    ensure!(
        skeletons.skeleton_ids.len() == n,
        "score_skeletons: {} nodes but {} skeleton ids",
        n,
        skeletons.skeleton_ids.len()
    );
    ensure!(
        spacing.iter().all(|s| s.is_finite() && *s > 0.0),
        "score_skeletons: spacing {spacing:?} must be finite and positive"
    );
    for &[a, b] in &skeletons.edges {
        ensure!(a < n && b < n, "score_skeletons: edge [{a}, {b}] names a missing node");
        ensure!(
            skeletons.skeleton_ids[a] == skeletons.skeleton_ids[b],
            "score_skeletons: edge [{a}, {b}] joins two skeletons"
        );
    }

    let shape = produced.shape();
    let labels: Vec<u32> = skeletons
        .nodes
        .iter()
        .map(|position| {
            let mut index = [0usize; 3];
            for axis in 0..3 {
                let rounded = position[axis].round();
                if !(rounded >= 0.0 && rounded < shape[axis] as f64) {
                    return 0;
                }
                index[axis] = rounded as usize;
            }
            produced[index]
        })
        .collect();

    // Mergers: labels touched by more than one skeleton.
    let mut owner: HashMap<u32, Option<u64>> = HashMap::new();
    for (&label, &id) in labels.iter().zip(&skeletons.skeleton_ids) {
        if label == 0 {
            continue;
        }
        owner
            .entry(label)
            .and_modify(|seen| {
                if *seen != Some(id) {
                    *seen = None;
                }
            })
            .or_insert(Some(id));
    }
    let mut merger_labels: Vec<u32> =
        owner.iter().filter(|(_, o)| o.is_none()).map(|(&l, _)| l).collect();
    merger_labels.sort_unstable();
    let is_merger = |label: u32| matches!(owner.get(&label), Some(None));

    // Majority label per skeleton, by node count.
    let mut votes: BTreeMap<u64, HashMap<u32, usize>> = BTreeMap::new();
    for (&label, &id) in labels.iter().zip(&skeletons.skeleton_ids) {
        let tally = votes.entry(id).or_default();
        if label != 0 {
            *tally.entry(label).or_default() += 1;
        }
    }
    let majority: HashMap<u64, u32> = votes
        .iter()
        .map(|(&id, tally)| {
            let best = tally
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(&l, _)| l)
                .unwrap_or(0);
            (id, best)
        })
        .collect();

    let edge_length = |a: usize, b: usize| {
        let (p, q) = (skeletons.nodes[a], skeletons.nodes[b]);
        (0..3)
            .map(|axis| ((p[axis] - q[axis]) * spacing[axis]).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut length: HashMap<u64, f64> = HashMap::new();
    let mut on_majority: HashMap<u64, f64> = HashMap::new();
    let mut runs = Union::new(n);
    let mut split_edges = 0usize;
    let mut kept = Vec::new();
    for &[a, b] in &skeletons.edges {
        let id = skeletons.skeleton_ids[a];
        let len = edge_length(a, b);
        *length.entry(id).or_default() += len;
        let (la, lb) = (labels[a], labels[b]);
        if la == lb && la != 0 {
            if la == majority[&id] {
                *on_majority.entry(id).or_default() += len;
            }
            if !is_merger(la) {
                runs.union(a, b);
                kept.push((a, len));
            }
        } else {
            split_edges += 1;
        }
    }
    let mut run_length: HashMap<usize, f64> = HashMap::new();
    for (a, len) in kept {
        *run_length.entry(runs.find(a)).or_default() += len;
    }
    let squared_runs: f64 = run_length.values().map(|r| r * r).sum();

    let total: f64 = length.values().sum();
    let per_skeleton: Vec<SkeletonRecall> = votes
        .keys()
        .map(|&id| {
            let l = length.get(&id).copied().unwrap_or(0.0);
            let hit = on_majority.get(&id).copied().unwrap_or(0.0);
            SkeletonRecall {
                skeleton: id,
                length: l,
                majority_label: majority[&id],
                recall: if l > 0.0 { hit / l } else { 0.0 },
            }
        })
        .collect();
    let (recall, erl, max_erl) = if total > 0.0 {
        (
            per_skeleton.iter().map(|s| s.recall * s.length).sum::<f64>() / total,
            squared_runs / total,
            length.values().map(|l| l * l).sum::<f64>() / total,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    Ok(SkeletonScores {
        per_skeleton,
        recall,
        erl,
        max_erl,
        nerl: if max_erl > 0.0 { erl / max_erl } else { 0.0 },
        merger_labels,
        split_edges,
    })
}

impl std::fmt::Display for SkeletonScores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} skeletons | recall {:.4} | ERL {:.3} of {:.3} (NERL {:.4}) | {} mergers, {} split edges",
            self.per_skeleton.len(),
            self.recall,
            self.erl,
            self.max_erl,
            self.nerl,
            self.merger_labels.len(),
            self.split_edges
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const EPS: f64 = 1e-12;

    /// 1 x 2 x 4: left half label 1, right half label 2.
    fn halves() -> Array3<u32> {
        Array3::from_shape_fn((1, 2, 4), |(_, _, x)| if x < 2 { 1 } else { 2 })
    }

    fn scores(t: &Array3<u32>, p: &Array3<u32>, o: ContingencyOptions) -> SegmentationScores {
        let options = SegmentationOptions {
            contingency: o,
            min_overlap_voxels: 1,
            min_overlap_fraction: 0.0,
            boundary_tolerance: Some(0.0),
        };
        score_segmentation(t.view(), p.view(), &options).unwrap()
    }

    #[test]
    fn relative_overlap_ignores_slivers_on_large_objects() {
        // Truth: two 1 x 10 x 10 halves. Produced: the boundary displaced by
        // one column, so produced 1 takes a 10-voxel sliver of truth 2.
        let truth = Array3::from_shape_fn((1, 10, 10), |(_, _, x)| if x < 5 { 1 } else { 2 });
        let produced = Array3::from_shape_fn((1, 10, 10), |(_, _, x)| if x < 6 { 1 } else { 2 });
        let table =
            Contingency::from_volumes(truth.view(), produced.view(), ContingencyOptions::all_objects())
                .unwrap();
        assert_eq!(table.edit_counts(5).splits, 1);
        // 10 shared voxels are 20% of the smaller object (truth 2, 50 voxels).
        assert_eq!(table.edit_counts_relative(5, 0.3).splits, 0);
        assert_eq!(table.edit_counts_relative(5, 0.2).splits, 1);
    }

    #[test]
    fn identical_labelings_score_perfectly() {
        let t = halves();
        let s = scores(&t, &t, ContingencyOptions::all_objects());
        assert!(s.voi.total().abs() < EPS);
        assert!(s.rand.error.abs() < EPS);
        assert_eq!((s.edits.splits, s.edits.merges), (0, 0));
        assert!((s.boundary.unwrap().f1 - 1.0).abs() < EPS);
        assert_eq!(s.voxels, 8);
    }

    #[test]
    fn ids_do_not_matter() {
        let t = halves();
        let p = t.mapv(|l| if l == 1 { 77 } else { 5 });
        let s = scores(&t, &p, ContingencyOptions::all_objects());
        assert!(s.voi.total().abs() < EPS && s.rand.error.abs() < EPS);
        // A split relabelled stays the same split.
        let split = Array3::from_shape_fn((1, 2, 4), |(_, _, x)| x as u32 + 1);
        let a = scores(&t, &split, ContingencyOptions::all_objects());
        let b = scores(&t, &split.mapv(|l| 100 - l), ContingencyOptions::all_objects());
        assert_eq!(a.voi, b.voi);
        assert_eq!(a.rand, b.rand);
        assert_eq!(a.edits, b.edits);
    }

    #[test]
    fn a_known_split() {
        // Truth: one object of 8. Produced: two of 4.
        let t = Array3::from_elem((1, 2, 4), 1u32);
        let p = halves();
        let s = scores(&t, &p, ContingencyOptions::all_objects());
        assert!((s.voi.split - 1.0).abs() < EPS, "{s}");
        assert!(s.voi.merge.abs() < EPS);
        // Σn² = 32, Σa² = 64, Σb² = 32: P = 1, R = 0.5, F = 2/3.
        assert!((s.rand.precision - 1.0).abs() < EPS);
        assert!((s.rand.recall - 0.5).abs() < EPS);
        assert!((s.rand.error - 1.0 / 3.0).abs() < EPS);
        assert_eq!((s.edits.splits, s.edits.merges), (0, 1));
        let b = s.boundary.unwrap();
        assert!((b.recall - 1.0).abs() < EPS && b.precision.abs() < EPS);
    }

    #[test]
    fn a_known_merge() {
        let t = halves();
        let p = Array3::from_elem((1, 2, 4), 3u32);
        let s = scores(&t, &p, ContingencyOptions::all_objects());
        assert!((s.voi.merge - 1.0).abs() < EPS);
        assert!(s.voi.split.abs() < EPS);
        assert!((s.rand.precision - 0.5).abs() < EPS);
        assert!((s.rand.recall - 1.0).abs() < EPS);
        assert_eq!((s.edits.splits, s.edits.merges), (1, 0));
        assert!(s.boundary.unwrap().recall.abs() < EPS);
    }

    #[test]
    fn slivers_are_below_the_edit_threshold() {
        let t = halves();
        let mut p = t.clone();
        p[[0, 0, 2]] = 1; // one voxel of truth 2 goes to produced 1
        let table = Contingency::from_volumes(t.view(), p.view(), ContingencyOptions::all_objects())
            .unwrap();
        assert_eq!(table.edit_counts(1).splits, 1);
        assert_eq!(table.edit_counts(2).splits, 0);
    }

    #[test]
    fn background_conventions() {
        // Truth: 0 on the first column, 1 elsewhere. Produced: 0 on x == 1.
        let t = Array3::from_shape_fn((1, 1, 4), |(_, _, x)| (x > 0) as u32);
        let p = Array3::from_shape_fn((1, 1, 4), |(_, _, x)| if x == 1 { 0 } else { 5 });
        let cremi = Contingency::from_volumes(t.view(), p.view(), ContingencyOptions::cremi())
            .unwrap();
        // Truth 0 dropped: 3 voxels, one unassigned singleton under truth 1.
        assert_eq!(cremi.total(), 3);
        assert_eq!(cremi.unassigned().collect::<Vec<_>>(), vec![(1, 1)]);
        // Σn² = 4 + 1, Σa² = 9, Σb² = 4 + 1: P = 1, R = 5/9.
        let r = cremi.adapted_rand();
        assert!((r.precision - 1.0).abs() < EPS);
        assert!((r.recall - 5.0 / 9.0).abs() < EPS);
        // A singleton only splits.
        let v = cremi.voi();
        assert!(v.merge.abs() < EPS);
        let expected = (3.0 * 3f64.log2() - 2.0 * 2f64.log2()) / 3.0;
        assert!((v.split - expected).abs() < EPS);

        let ignore = Contingency::from_volumes(
            t.view(),
            p.view(),
            ContingencyOptions {
                ignore_truth_zero: true,
                produced_zero: ProducedZero::Ignore,
            },
        )
        .unwrap();
        assert_eq!(ignore.total(), 2);
        assert!(ignore.voi().total().abs() < EPS);
    }

    #[test]
    fn tables_of_blocks_merge_to_the_whole() {
        let t = Array3::from_shape_fn((4, 5, 6), |(z, y, x)| ((z + 2 * y + x) % 4) as u32);
        let p = Array3::from_shape_fn((4, 5, 6), |(z, y, x)| ((z * y + x) % 3) as u32);
        for options in [ContingencyOptions::cremi(), ContingencyOptions::all_objects()] {
            let whole = Contingency::from_volumes(t.view(), p.view(), options).unwrap();
            let mut parts = Contingency::new(options);
            for range in [0..1, 1..4] {
                let block = Contingency::from_volumes(
                    t.slice(s![range.clone(), .., ..]),
                    p.slice(s![range, .., ..]),
                    options,
                )
                .unwrap();
                parts.merge(block).unwrap();
            }
            assert_eq!(whole, parts);
        }
        let mut a = Contingency::new(ContingencyOptions::cremi());
        assert!(a.merge(Contingency::new(ContingencyOptions::all_objects())).is_err());
    }

    #[test]
    fn boundary_tolerance_forgives_a_shift() {
        let t = Array3::from_shape_fn((2, 3, 8), |(_, _, x)| (x >= 4) as u32 + 1);
        let p = Array3::from_shape_fn((2, 3, 8), |(_, _, x)| (x >= 5) as u32 + 1);
        let exact = boundary_f1(t.view(), p.view(), 0.0, false).unwrap();
        // Truth boundary at x = 3,4; produced at 4,5: half of each matches.
        assert!((exact.precision - 0.5).abs() < EPS && (exact.recall - 0.5).abs() < EPS);
        let loose = boundary_f1(t.view(), p.view(), 1.0, false).unwrap();
        assert!((loose.f1 - 1.0).abs() < EPS);
    }

    /// Skeleton 1 is a 4-node line along x at y = 0, skeleton 2 the same at y = 2.
    fn two_lines() -> Skeletons {
        let mut s = Skeletons::default();
        for (id, y) in [(1u64, 0.0), (2, 2.0)] {
            let base = s.nodes.len();
            for x in 0..4 {
                s.nodes.push([0.0, y, x as f64]);
                s.skeleton_ids.push(id);
            }
            for k in 0..3 {
                s.edges.push([base + k, base + k + 1]);
            }
        }
        s
    }

    fn rows(top: [u32; 4], bottom: [u32; 4]) -> Array3<u32> {
        Array3::from_shape_fn((1, 3, 4), |(_, y, x)| match y {
            0 => top[x],
            2 => bottom[x],
            _ => 0,
        })
    }

    #[test]
    fn perfect_skeletons_have_nerl_one() {
        let s = score_skeletons(&two_lines(), rows([1; 4], [2; 4]).view(), [1.0; 3]).unwrap();
        assert!((s.erl - 3.0).abs() < EPS && (s.max_erl - 3.0).abs() < EPS);
        assert!((s.nerl - 1.0).abs() < EPS && (s.recall - 1.0).abs() < EPS);
        assert!(s.merger_labels.is_empty());
    }

    #[test]
    fn a_split_skeleton() {
        // Skeleton 1 split 1,1 | 3,3: runs of 1 and 1. Skeleton 2 intact: 3.
        let s = score_skeletons(&two_lines(), rows([1, 1, 3, 3], [2; 4]).view(), [1.0; 3]).unwrap();
        // ERL = (1 + 1 + 9) / 6; max = (9 + 9) / 6 = 3.
        assert!((s.erl - 11.0 / 6.0).abs() < EPS);
        assert!((s.nerl - 11.0 / 18.0).abs() < EPS);
        assert_eq!(s.split_edges, 1);
        assert_eq!(s.per_skeleton[0].majority_label, 1);
        assert!((s.per_skeleton[0].recall - 1.0 / 3.0).abs() < EPS);
        // Spacing scales lengths, and ERL with them.
        let scaled =
            score_skeletons(&two_lines(), rows([1, 1, 3, 3], [2; 4]).view(), [1.0, 1.0, 2.0])
                .unwrap();
        assert!((scaled.erl - 2.0 * 11.0 / 6.0).abs() < EPS);
    }

    #[test]
    fn a_merged_skeleton_pair_scores_zero_for_both() {
        let s = score_skeletons(&two_lines(), rows([1; 4], [1; 4]).view(), [1.0; 3]).unwrap();
        assert_eq!(s.merger_labels, vec![1]);
        assert!(s.erl.abs() < EPS && s.nerl.abs() < EPS);
        // A merger at one node: skeleton 2's tip in label 1 poisons label 1.
        let s = score_skeletons(&two_lines(), rows([1; 4], [2, 2, 2, 1]).view(), [1.0; 3]).unwrap();
        assert_eq!(s.merger_labels, vec![1]);
        // Skeleton 1 contributes nothing; skeleton 2 keeps a run of 2.
        assert!((s.erl - 4.0 / 6.0).abs() < EPS);
    }

    #[test]
    fn background_and_outside_nodes_break_runs() {
        let mut skeletons = two_lines();
        skeletons.nodes[7] = [0.0, 2.0, 9.0]; // outside the volume
        let s = score_skeletons(&skeletons, rows([1, 1, 0, 1], [2; 4]).view(), [1.0; 3]).unwrap();
        // Skeleton 1: one run of 1 (two edges touch 0). Skeleton 2: edges are
        // 1, 1, 7 long; the last ends outside, so one run of 2.
        assert!((s.erl - (1.0 + 4.0) / (3.0 + 9.0)).abs() < EPS, "{s}");
        assert!(score_skeletons(
            &Skeletons {
                nodes: vec![[0.0; 3]],
                edges: vec![[0, 1]],
                skeleton_ids: vec![1],
            },
            rows([1; 4], [1; 4]).view(),
            [1.0; 3]
        )
        .is_err());
    }
}
