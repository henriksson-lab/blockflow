// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What an operation reads beyond what it writes, and *in what units*.
//
// The degenerate statement — one unsigned integer per axis, the same for every
// block, applied to both sides — is the one this crate shipped with. It is kept
// as a variant rather than replaced, because most operations mean exactly that
// and `src/ops/` derives several of them tight to a single voxel. What it
// cannot say, and what each of the other variants exists for, was measured
// rather than imagined:
//
// | variant | what it says | why |
// |---|---|---|
// | `Bounded { lo, hi }` | asymmetric, uniform | a lattice dependency is one-sided; declaring it on both sides fetched **3.27x** what was needed |
// | `PerBlock` | one `(lo, hi)` per index along the axis | an unevenly *spread* lattice has a uniform reach in index units and a different voxel footprint per block, so no single integer is tight |
// | `All` | the whole axis | "reaches 4096" and "reaches everything" were the same number, so a planning barrier could only be found by comparing an integer against the volume |
//
// And the fourth axis of the problem, which caused most of the loss: the same
// dependency is `2` in one coordinate space and `255` in another. A [`Reach`]
// therefore carries a [`Space`] — whose volume it is measured against, in what
// unit, and in which axis order.
//
// Two rules this module exists to keep
// ------------------------------------
// * **A reach is computed independently of the halo.** That is what keeps the
//   tiling check in `decomposition` able to fail at all: if the granted halo fed
//   the required reach, the guard would compare a number against itself. Nothing
//   here reads a halo; `Reach` is used for both quantities precisely so that the
//   guard is a comparison of two values of one type, derived from two places.
// * **A per-block reach is a function of the block index and nothing else.**
//   Not a closure and not a callback: a table indexed by the block's index along
//   the axis. A `Decomposition` is binding and parity-visible, so a reach that
//   could consult the data would make a plan unreproducible — and a table cannot.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde_json::{json, Value};

use crate::error::{Error, Result};

/// The unit a reach counts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Units {
    /// Voxels of the volume named by the [`Frame`]. The default, and what every
    /// reach in this crate meant before spaces were named.
    #[default]
    Voxels,
    /// Whole blocks of the phase's own lattice: one unit is one block edge.
    ///
    /// This is not hypothetical — `fragment::fragment_phase` already converts a
    /// neighbour fold's reach from blocks to voxels by hand
    /// (`halo[axis] = reach[axis] * edge[axis]`), and `strategy` walks the same
    /// reach in index units to find a task's neighbours. The conversion is done
    /// in [`Reach::in_voxels`], at the one place a grid is known.
    Blocks,
    /// Steps of the **image below's own lattice**, which this phase's geometry
    /// cannot convert into its own voxels.
    ///
    /// The case that costs the most and the one the coordinate space exists for:
    /// a dependency that is *two* in the lattice the data is stored on becomes
    /// `edge - 1` voxels when it is restated in the only space the phase has,
    /// and the fetch grows by **3.27x** for it. A phase reading such a lattice
    /// states each block's fetch region explicitly (`BlockGeometry::source`), and
    /// the dependency lives in that mapping rather than in a halo.
    ///
    /// So this unit is **carried and checked, not converted**. It implies
    /// [`Frame::Source`], it contributes nothing to the read extent — there is
    /// no conversion factor a `BlockGrid` could supply — and a phase that
    /// declares it without per-block fetch regions is refused by name rather
    /// than quietly planned as though it reached nothing. What verifies the
    /// mapping itself is the op's own `BlockConstraint::Regions`, which states
    /// the lattice region by region; the plan records the dependency so that it
    /// is in the fingerprint and on the wire instead of being a zero somebody
    /// wrote to get past the guard.
    SourceIndex,
}

/// Whose volume the numbers are measured against.
///
/// The distinction has no effect on a phase whose output grid is its input
/// grid, which is every phase this crate shipped before `BlockGeometry::source`
/// existed. It matters at exactly one place, and that place is a recorded
/// defect: `BlockGeometry::derive` treats a read clamped at the phase's own
/// volume edge as trustworthy — correct at a real edge of the array, because
/// there is nothing beyond the end to have read, and **wrong for a phase whose
/// edges are not edges of the image below**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Frame {
    /// The phase's own volume: what its cores are cut from and what its valid
    /// regions must tile.
    #[default]
    Phase,
    /// The image below — the space `BlockGeometry::source` is stated in.
    ///
    /// A reach in this frame is **not granted the clamp exception** at the
    /// phase's own boundaries, because a cropping or regridding phase's boundary
    /// is an interior position of the array it reads. The granted halo has to
    /// cover it there like any other seam, or the tiling check fires.
    Source,
}

/// Which volume a reach is measured against, in what unit, in which axis order.
///
/// **The axis order is carried but not yet acted on by the geometry**, and that
/// is deliberate rather than unfinished. A three-branch pass over axis
/// permutations of one volume needs the *lattice*, the read extent, the valid
/// region and the anchor permuted together; a permuted reach is one of the five
/// and the cheapest. Carrying it here means the tag a permuted branch will need
/// exists and is already hashed into the fingerprint, so adding the rest is a
/// change to `BlockGrid` and `Anchor` rather than a change to every op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Space {
    pub frame: Frame,
    pub units: Units,
    /// `axes[i]` is the canonical axis this space's axis `i` refers to. The
    /// identity `[0, 1, 2]` for every space this crate resolves today.
    pub axes: [usize; 3],
}

impl Default for Space {
    fn default() -> Self {
        Self {
            frame: Frame::Phase,
            units: Units::Voxels,
            axes: [0, 1, 2],
        }
    }
}

impl Space {
    /// The space every reach was in before spaces were named: the phase's own
    /// voxels, in the phase's own axis order.
    pub fn phase_voxels() -> Self {
        Self::default()
    }

    /// Voxels of the image below: the same distance, measured against an array
    /// whose edges this phase's edges are not.
    pub fn source_voxels() -> Self {
        Self {
            frame: Frame::Source,
            ..Self::default()
        }
    }

    /// Steps of the image below's own lattice — see [`Units::SourceIndex`].
    pub fn source_index() -> Self {
        Self {
            frame: Frame::Source,
            units: Units::SourceIndex,
            axes: [0, 1, 2],
        }
    }

    /// Whether this space's numbers are distances the phase's geometry can
    /// apply to its own read extent.
    ///
    /// False for [`Units::SourceIndex`], and that is the point: there is no
    /// factor converting a step of somebody else's lattice into a voxel of this
    /// one, so a plan that pretended there was would be inventing geometry.
    pub fn converts_to_voxels(&self) -> bool {
        !matches!(self.units, Units::SourceIndex)
    }

    /// Whole blocks of the phase's own lattice.
    pub fn blocks() -> Self {
        Self {
            units: Units::Blocks,
            ..Self::default()
        }
    }

    pub fn with_frame(mut self, frame: Frame) -> Self {
        self.frame = frame;
        self
    }

    /// State the same reach in a permuted axis order.
    pub fn with_axes(mut self, axes: [usize; 3]) -> Result<Self> {
        let mut seen = [false; 3];
        for &axis in &axes {
            if axis >= 3 || seen[axis] {
                return Err(Error::InvalidArgument(format!(
                    "reach: {axes:?} is not a permutation of three axes. A coordinate space's \
                     axis order names each axis exactly once."
                )));
            }
            seen[axis] = true;
        }
        self.axes = axes;
        Ok(self)
    }

    /// Whether this is the space the geometry works in, so no conversion and no
    /// re-interpretation is needed.
    pub fn is_canonical(&self) -> bool {
        *self == Self::default()
    }

    /// Whether a read clamped at the phase's own volume edge may be trusted.
    ///
    /// True in the phase's own frame — a voxel at a real array boundary saw
    /// everything that exists — and false in the source frame, where the
    /// phase's edge is an interior position of the array being read.
    pub fn clamp_is_an_edge(&self) -> bool {
        matches!(self.frame, Frame::Phase)
    }

    fn label(&self) -> String {
        let frame = match self.frame {
            Frame::Phase => "phase",
            Frame::Source => "source",
        };
        let units = match self.units {
            Units::Voxels => "voxels",
            Units::Blocks => "blocks",
            Units::SourceIndex => "source-index",
        };
        format!("{frame}/{units}{:?}", self.axes)
    }
}

/// What is read beyond what is written, along one axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AxisReach {
    /// `lo` units below the written position and `hi` above, the same for every
    /// block. `Bounded { lo: r, hi: r }` is the degenerate form this crate had.
    Bounded { lo: usize, hi: usize },
    /// One `(lo, hi)` per block index along this axis.
    ///
    /// A **table**, not a rule that could be evaluated against anything but the
    /// index. That is the whole reason it is a `Vec` rather than a closure: a
    /// `Decomposition` is parity-visible and must be reproducible from what it
    /// records.
    PerBlock(Vec<(usize, usize)>),
    /// A bound that **shrinks when the lattice's block edge is a whole number of
    /// `stride`**, stated as the two answers rather than as one plus a rule.
    ///
    /// The one variant whose value is a function of the grid rather than of the
    /// axis. It exists because an op can need a *divisibility* of the block edge
    /// — not an axis left whole, which is [`Self::All`], and not a per-block
    /// table, which needs the lattice to write down.
    ///
    /// **Both answers are carried, and that is what makes it fold.** A single
    /// `(lo, hi)` plus "`stride - 1` when unaligned" cannot survive addition:
    /// two ops of stride 32 fused reach `31 + 31` past their kernels when the
    /// edge is odd, which no `stride - 1` can express, and a fold that tried
    /// would **under-halo by 31**. With both states present the sum is
    /// componentwise on each and is exact.
    ///
    /// **The first payer is `ops::convolve::TransformConvolveOp`.** Its tile grid
    /// is anchored to the volume, so a block whose core starts mid-tile must
    /// reach back to that tile's start; a block edge that is a multiple of the
    /// tile makes every core start tile-aligned, because `BlockGrid::cores`
    /// builds `start = index * block`, and then the true halo is exactly the
    /// kernel's.
    ///
    /// **Everything that does not know the grid answers `unaligned`**, which is
    /// the only safe direction: [`Self::bound`], [`Self::at`] and
    /// [`Self::widest`] all do. [`Reach::in_voxels`] and
    /// [`crate::decomposition::cuttable_axes`] are handed the lattice and are the
    /// two places the discount is taken.
    ///
    /// Build one with [`Self::aligned`], which is the only way to get the
    /// invariant right: `unaligned` is never narrower than `aligned`.
    Aligned {
        stride: usize,
        /// `(lo, hi)` on a lattice whose block edge `stride` divides.
        aligned: (usize, usize),
        /// `(lo, hi)` on one it does not. Never narrower than `aligned`.
        unaligned: (usize, usize),
    },
    /// Everything on this axis, whatever its extent.
    ///
    /// Distinct from `Bounded { lo: n, hi: n }` with `n >= extent` even though
    /// the arithmetic agrees, because a planning barrier is then a property of
    /// the *type* rather than a comparison somebody has to remember to make.
    All,
}

impl Default for AxisReach {
    fn default() -> Self {
        Self::Bounded { lo: 0, hi: 0 }
    }
}

impl AxisReach {
    pub fn symmetric(reach: usize) -> Self {
        Self::Bounded {
            lo: reach,
            hi: reach,
        }
    }

    pub fn none() -> Self {
        Self::Bounded { lo: 0, hi: 0 }
    }

    /// `(lo, hi)` for the block at `index` along this axis.
    ///
    /// `extent` resolves [`AxisReach::All`], which is the only variant that has
    /// to ask how long the axis is.
    pub fn at(&self, index: usize, extent: usize) -> (usize, usize) {
        match self {
            Self::Bounded { lo, hi } => (*lo, *hi),
            // A table shorter than the lattice is refused by `check_lattice`
            // before a plan is built; falling back to the last entry here keeps
            // this total without inventing a zero, which is the one wrong answer
            // that would be silent.
            Self::PerBlock(table) => table
                .get(index)
                .copied()
                .or_else(|| table.last().copied())
                .unwrap_or((0, 0)),
            // No grid here, so the worst alignment: see the variant's own
            // documentation for why this direction and not the other.
            Self::Aligned { .. } => self.worst_case(),
            Self::All => (extent, extent),
        }
    }

    /// The widest `(lo, hi)` over every block.
    pub fn bound(&self, extent: usize) -> (usize, usize) {
        match self {
            Self::Bounded { lo, hi } => (*lo, *hi),
            Self::PerBlock(table) => table
                .iter()
                .fold((0, 0), |(lo, hi), &(l, h)| (lo.max(l), hi.max(h))),
            Self::Aligned { .. } => self.worst_case(),
            Self::All => (extent, extent),
        }
    }

    /// A reach of `(lo, hi)` on any lattice whose block edge `stride` divides,
    /// and `stride - 1` wider on each side on one it does not.
    ///
    /// The only constructor for [`Self::Aligned`], because the invariant that
    /// `unaligned` is never narrower than `aligned` is the whole safety of the
    /// variant and a struct literal could get it backwards. A `stride` of 0 or 1
    /// divides nothing or everything and collapses to a plain [`Self::Bounded`],
    /// which is what it means.
    pub fn aligned(stride: usize, lo: usize, hi: usize) -> Self {
        let slack = stride.saturating_sub(1);
        Self::Aligned {
            stride,
            aligned: (lo, hi),
            unaligned: (slack + lo, slack + hi),
        }
        .normalised()
    }

    /// The unaligned `(lo, hi)` of an [`Self::Aligned`], `(0, 0)` for anything
    /// else. What every question asked without a lattice gets.
    fn worst_case(&self) -> (usize, usize) {
        match self {
            Self::Aligned { unaligned, .. } => *unaligned,
            _ => (0, 0),
        }
    }

    /// An [`Self::Aligned`] whose two answers agree is a [`Self::Bounded`], and
    /// saying so keeps a plan's recorded reach the simplest form of itself.
    fn normalised(self) -> Self {
        match self {
            Self::Aligned {
                aligned, unaligned, ..
            } if aligned == unaligned => Self::Bounded {
                lo: aligned.0,
                hi: aligned.1,
            },
            other => other,
        }
    }

    /// This reach against a lattice whose block edge on this axis is `edge`.
    ///
    /// The identity for every variant but [`Self::Aligned`], which is the point:
    /// the others say the same thing whatever the lattice, and this one does
    /// not. `edge = 0` resolves to the unaligned answer rather than dividing by
    /// it.
    pub fn resolved(&self, edge: usize) -> Self {
        match self {
            Self::Aligned {
                stride,
                aligned,
                unaligned,
            } => {
                let (lo, hi) = if *stride > 0 && edge > 0 && edge % stride == 0 {
                    *aligned
                } else {
                    *unaligned
                };
                Self::Bounded { lo, hi }
            }
            other => other.clone(),
        }
    }

    /// The unaligned answer as a plain [`Self::Bounded`]: what this reach means
    /// to anything that cannot see a lattice, and what a fold falls back to when
    /// two strides have no usable common multiple.
    fn flattened(&self) -> Self {
        match self {
            Self::Aligned { .. } => {
                let (lo, hi) = self.worst_case();
                Self::Bounded { lo, hi }
            }
            other => other.clone(),
        }
    }

    /// The two answers this reach gives, for a fold to combine componentwise.
    /// A reach with no stride gives the same answer on both lattices, which is
    /// exactly why folding one in is exact.
    fn both(&self) -> Option<((usize, usize), (usize, usize), usize)> {
        match self {
            Self::Aligned {
                stride,
                aligned,
                unaligned,
            } => Some((*aligned, *unaligned, *stride)),
            Self::Bounded { lo, hi } => Some(((*lo, *hi), (*lo, *hi), 1)),
            _ => None,
        }
    }

    /// Fold two reaches that both answer per lattice, combining each answer with
    /// `join` and the strides with their least common multiple.
    ///
    /// `None` when either side is a table or `All`, or when the common multiple
    /// overflows — in each case the caller falls back to the unaligned answers,
    /// which is what this variant meant before it could fold and is never wrong,
    /// only dearer.
    fn fold_aligned(&self, other: &Self, join: impl Fn(usize, usize) -> usize) -> Option<Self> {
        let (left_aligned, left_unaligned, left_stride) = self.both()?;
        let (right_aligned, right_unaligned, right_stride) = other.both()?;
        let stride = least_common_multiple(left_stride, right_stride)?;
        Some(
            Self::Aligned {
                stride,
                aligned: (
                    join(left_aligned.0, right_aligned.0),
                    join(left_aligned.1, right_aligned.1),
                ),
                unaligned: (
                    join(left_unaligned.0, right_unaligned.0),
                    join(left_unaligned.1, right_unaligned.1),
                ),
            }
            .normalised(),
        )
    }

    /// The symmetric per-axis integer this used to be: the widest side.
    pub fn widest(&self, extent: usize) -> usize {
        let (lo, hi) = self.bound(extent);
        lo.max(hi)
    }

    /// The **narrowest** side over every block: what is granted everywhere.
    ///
    /// The direction [`Self::widest`] is not. A caller asking "is this halo at
    /// least `n` wide" is asking about the worst block and the worst side, and
    /// answering it with the best would be a guard that passes because somewhere
    /// else was generous.
    pub fn narrowest(&self, extent: usize) -> usize {
        match self {
            Self::Bounded { lo, hi } => (*lo).min(*hi),
            Self::PerBlock(table) => table.iter().map(|&(lo, hi)| lo.min(hi)).min().unwrap_or(0),
            // What is granted *everywhere* is the aligned amount, not the
            // unaligned one: this is the direction the doc above calls the one
            // `widest` is not, and understating here is the safe error.
            Self::Aligned { aligned, .. } => aligned.0.min(aligned.1),
            Self::All => extent,
        }
    }

    /// Does this reach cover the whole of an axis of `extent`?
    ///
    /// The barrier predicate, and the reason [`AxisReach::All`] exists. An axis
    /// of extent 1 is excluded on the argument `decomposition::reaches_whole_axis`
    /// records: every block already spans it, so counting it would make every op
    /// on a flat volume a barrier.
    pub fn is_whole(&self, extent: usize) -> bool {
        extent > 1
            && match self {
                Self::All => true,
                _ => self.widest(extent) >= extent,
            }
    }

    /// Applied one after the other, reaches add.
    pub fn add(&self, other: &Self) -> Self {
        // **An `Aligned` survives a fold, and it has to.** Fusing a transform
        // convolution with a voxelwise map that reaches nothing used to flatten
        // it, which lost the whole discount to the most ordinary fusion there
        // is: measured on `96^3`, **27 blocks alone against one when fused**.
        // Adding zero must be the identity, and the fold below makes it so —
        // exactly for `add`, because both answers combine componentwise.
        if matches!(self, Self::Aligned { .. }) || matches!(other, Self::Aligned { .. }) {
            return match self.fold_aligned(other, |a, b| a + b) {
                Some(folded) => folded,
                // A table, an `All`, or a common multiple that overflows: fall
                // back to the unaligned answers, which is what this variant
                // meant before it could fold. Never wrong, only dearer.
                None => self.flattened().add(&other.flattened()),
            };
        }
        match (self, other) {
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::Bounded { lo: a, hi: b }, Self::Bounded { lo: c, hi: d }) => Self::Bounded {
                lo: a + c,
                hi: b + d,
            },
            _ => Self::PerBlock(zip_tables(self, other, |(a, b), (c, d)| (a + c, b + d))),
        }
    }

    /// Alternatives and concurrent branches take the wider of the two, per side.
    pub fn max(&self, other: &Self) -> Self {
        // Folded like [`Self::add`], and **exact where it matters and generous
        // where it cannot be**: on an edge the common multiple divides, both
        // branches take their aligned answer and the maximum of the two is the
        // truth. On an edge only one stride divides, this reports the maximum of
        // the two *unaligned* answers, which is at least the truth — the safe
        // direction, and the one a halo may err in.
        if matches!(self, Self::Aligned { .. }) || matches!(other, Self::Aligned { .. }) {
            return match self.fold_aligned(other, |a, b| a.max(b)) {
                Some(folded) => folded,
                None => self.flattened().max(&other.flattened()),
            };
        }
        match (self, other) {
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::Bounded { lo: a, hi: b }, Self::Bounded { lo: c, hi: d }) => Self::Bounded {
                lo: (*a).max(*c),
                hi: (*b).max(*d),
            },
            _ => Self::PerBlock(zip_tables(self, other, |(a, b), (c, d)| {
                (a.max(c), b.max(d))
            })),
        }
    }

    /// Voxels, from whole blocks of a lattice with this `edge`.
    fn scaled(&self, edge: usize) -> Self {
        match self {
            Self::All => Self::All,
            Self::Bounded { lo, hi } => Self::Bounded {
                lo: lo * edge,
                hi: hi * edge,
            },
            Self::PerBlock(table) => Self::PerBlock(
                table
                    .iter()
                    .map(|&(lo, hi)| (lo * edge, hi * edge))
                    .collect(),
            ),
            // A reach counted in **blocks** and a stride counted in **voxels**
            // are not the same quantity, and multiplying one by the other would
            // invent a number. The unaligned answer, scaled, is the honest one
            // and no op states both today.
            Self::Aligned { .. } => self.flattened().scaled(edge),
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Bounded { lo: 0, hi: 0 })
    }

    fn to_json(&self) -> Value {
        match self {
            Self::Bounded { lo, hi } if lo == hi => json!(lo),
            Self::Bounded { lo, hi } => json!({ "lo": lo, "hi": hi }),
            Self::All => json!("all"),
            Self::Aligned {
                stride,
                aligned,
                unaligned,
            } => json!({
                "stride": stride,
                "aligned": [aligned.0, aligned.1],
                "unaligned": [unaligned.0, unaligned.1],
            }),
            Self::PerBlock(table) => json!({
                "per_block": table.iter().map(|&(lo, hi)| json!([lo, hi])).collect::<Vec<_>>(),
            }),
        }
    }

    fn from_json(value: &Value) -> Result<Self> {
        if let Some(number) = value.as_u64() {
            return Ok(Self::symmetric(number as usize));
        }
        if let Some(text) = value.as_str() {
            if text == "all" {
                return Ok(Self::All);
            }
            return Err(Error::InvalidArgument(format!(
                "{text:?} is not an axis reach. The only word here is \"all\"."
            )));
        }
        let object = value
            .as_object()
            .ok_or_else(|| Error::InvalidArgument(format!("{value} is not an axis reach")))?;
        if let Some(table) = object.get("per_block") {
            let rows = table.as_array().ok_or_else(|| {
                Error::InvalidArgument("a per-block reach is a list of [lo, hi] pairs".to_string())
            })?;
            let mut pairs = Vec::with_capacity(rows.len());
            for row in rows {
                let pair = row
                    .as_array()
                    .filter(|pair| pair.len() == 2)
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "a per-block reach entry is exactly [lo, hi]".to_string(),
                        )
                    })?;
                pairs.push((number(&pair[0])?, number(&pair[1])?));
            }
            return Ok(Self::PerBlock(pairs));
        }
        // **Before the fall-through below, and that ordering is the whole
        // hazard.** The last arm accepts any object and reads `lo`/`hi` out of
        // it, so an aligned reach that arrived here would round-trip into a
        // `Bounded` carrying the *aligned* sides with the stride dropped — a
        // halo narrower than the op needs, on a lattice nothing checked, in
        // another process. `a_reach_survives_the_wire_in_every_variant` pins it.
        if let Some(stride) = object.get("stride") {
            let pair = |key: &str| -> Result<(usize, usize)> {
                let entry = object.get(key).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "an aligned reach states both of its answers; `{key}` is missing"
                    ))
                })?;
                let values = entry
                    .as_array()
                    .filter(|row| row.len() == 2)
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!("`{key}` of an aligned reach is [lo, hi]"))
                    })?;
                Ok((number(&values[0])?, number(&values[1])?))
            };
            let aligned = pair("aligned")?;
            let unaligned = pair("unaligned")?;
            if unaligned.0 < aligned.0 || unaligned.1 < aligned.1 {
                return Err(Error::InvalidArgument(format!(
                    "an aligned reach's unaligned answer {unaligned:?} is narrower than its \
                     aligned one {aligned:?}. The invariant is the whole safety of the variant: \
                     a plan carrying it inverted would under-halo every block on every lattice \
                     the stride does not divide."
                )));
            }
            return Ok(Self::Aligned {
                stride: number(stride)?,
                aligned,
                unaligned,
            }
            .normalised());
        }
        Ok(Self::Bounded {
            lo: object.get("lo").map(number).transpose()?.unwrap_or(0),
            hi: object.get("hi").map(number).transpose()?.unwrap_or(0),
        })
    }
}

/// The least common multiple, or `None` on overflow.
///
/// **A common multiple beyond every candidate edge is not a failure**, which is
/// why this does not clamp: no edge is then a multiple of it, the reach answers
/// its unaligned form everywhere, and that is exactly the behaviour the fold
/// replaced. The degradation is graceful and it is never wrong, only dearer.
fn least_common_multiple(left: usize, right: usize) -> Option<usize> {
    if left == 0 || right == 0 {
        return None;
    }
    let (mut a, mut b) = (left, right);
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    (left / a).checked_mul(right)
}

fn number(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .map(|number| number as usize)
        .ok_or_else(|| Error::InvalidArgument(format!("{value} is not a count")))
}

/// Fold two axis reaches into a table, whichever forms they are in.
///
/// `All` never reaches here — both callers absorb it first — so the two operands
/// are a table, a uniform pair, or one of each, and a uniform pair is a table
/// that repeats.
fn zip_tables(
    left: &AxisReach,
    right: &AxisReach,
    fold: impl Fn((usize, usize), (usize, usize)) -> (usize, usize),
) -> Vec<(usize, usize)> {
    let len = table_len(left).max(table_len(right)).max(1);
    (0..len)
        .map(|index| fold(left.at(index, 0), right.at(index, 0)))
        .collect()
}

fn table_len(reach: &AxisReach) -> usize {
    match reach {
        AxisReach::PerBlock(table) => table.len(),
        _ => 0,
    }
}

/// What is read beyond what is written, on every axis, in a named space.
///
/// This type is used for **two** quantities that must not be conflated: the
/// reach an operation *requires*, and the halo a plan *grants*. They are the
/// same shape of thing — a one-sided distance per axis, possibly per block —
/// and the guard this crate is built around is the comparison between them,
/// made not by an assertion but by deriving a valid region from one and the
/// fetch from the other and checking that the valid regions still tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reach {
    space: Space,
    axes: [AxisReach; 3],
}

impl Default for Reach {
    fn default() -> Self {
        Self::none()
    }
}

/// A symmetric integer per axis is a `Reach`, so every call site that states
/// one keeps working and every plan built from one is the plan it was.
impl From<[usize; 3]> for Reach {
    fn from(reach: [usize; 3]) -> Self {
        Self::symmetric(reach)
    }
}

/// So that `phase.reach == [4, 0, 0]` still reads as the question it always
/// asked. It answers `false` for anything the triple cannot express, which is
/// the honest answer rather than a lossy one.
impl PartialEq<[usize; 3]> for Reach {
    fn eq(&self, other: &[usize; 3]) -> bool {
        self.space.is_canonical()
            && (0..3).all(|axis| self.axes[axis] == AxisReach::symmetric(other[axis]))
    }
}

/// **The degenerate form hashes exactly as the triple it replaced**, so every
/// fingerprint of every plan that does not use the new forms is unchanged. That
/// is not a convenience: `Decomposition` is parity-visible, and a fingerprint
/// that moved without a plan moving would make every stored figure a mystery.
impl Hash for Reach {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.as_symmetric() {
            Some(triple) => triple.hash(state),
            None => {
                self.space.hash(state);
                self.axes.hash(state);
            }
        }
    }
}

impl Reach {
    /// Reaches nothing: the identity of both folds.
    pub fn none() -> Self {
        Self {
            space: Space::default(),
            axes: [AxisReach::none(), AxisReach::none(), AxisReach::none()],
        }
    }

    /// One integer per axis, applied to both sides — the form this crate had,
    /// kept as a variant because most operations mean exactly it.
    pub fn symmetric(reach: [usize; 3]) -> Self {
        Self {
            space: Space::default(),
            axes: [
                AxisReach::symmetric(reach[0]),
                AxisReach::symmetric(reach[1]),
                AxisReach::symmetric(reach[2]),
            ],
        }
    }

    /// `(lo, hi)` per axis.
    pub fn asymmetric(reach: [(usize, usize); 3]) -> Self {
        Self {
            space: Space::default(),
            axes: [
                AxisReach::Bounded {
                    lo: reach[0].0,
                    hi: reach[0].1,
                },
                AxisReach::Bounded {
                    lo: reach[1].0,
                    hi: reach[1].1,
                },
                AxisReach::Bounded {
                    lo: reach[2].0,
                    hi: reach[2].1,
                },
            ],
        }
    }

    /// The whole of every axis: a planning barrier, said rather than detected.
    pub fn all() -> Self {
        Self {
            space: Space::default(),
            axes: [AxisReach::All, AxisReach::All, AxisReach::All],
        }
    }

    pub fn per_axis(axes: [AxisReach; 3]) -> Self {
        Self {
            space: Space::default(),
            axes,
        }
    }

    /// Restate the same numbers in another coordinate space.
    pub fn in_space(mut self, space: Space) -> Self {
        self.space = space;
        self
    }

    pub fn space(&self) -> Space {
        self.space
    }

    pub fn axis(&self, axis: usize) -> &AxisReach {
        &self.axes[axis]
    }

    /// The triple this used to be, when that is exactly what it is.
    ///
    /// `None` means the reach says something a triple cannot, which is the
    /// question every place that must fall back to the old form is really
    /// asking.
    pub fn as_symmetric(&self) -> Option<[usize; 3]> {
        if !self.space.is_canonical() {
            return None;
        }
        let mut triple = [0usize; 3];
        for axis in 0..3 {
            match &self.axes[axis] {
                AxisReach::Bounded { lo, hi } if lo == hi => triple[axis] = *lo,
                _ => return None,
            }
        }
        Some(triple)
    }

    /// `(lo, hi)` for the block at `index`, on `axis`, over a volume of
    /// `extent`.
    pub fn at(&self, axis: usize, index: usize, extent: usize) -> (usize, usize) {
        self.axes[axis].at(index, extent)
    }

    /// The symmetric per-axis upper bound: what a caller that can only hold one
    /// integer per axis must use.
    pub fn bound(&self, volume: [usize; 3]) -> [usize; 3] {
        [
            self.axes[0].widest(volume[0]),
            self.axes[1].widest(volume[1]),
            self.axes[2].widest(volume[2]),
        ]
    }

    /// The narrowest side granted on every block, per axis: what a caller
    /// checking "is at least this much granted everywhere" must compare against.
    pub fn granted_everywhere(&self, volume: [usize; 3]) -> [usize; 3] {
        [
            self.axes[0].narrowest(volume[0]),
            self.axes[1].narrowest(volume[1]),
            self.axes[2].narrowest(volume[2]),
        ]
    }

    /// Whether this reach spans the whole of `axis`.
    pub fn is_whole_axis(&self, axis: usize, extent: usize) -> bool {
        self.axes[axis].is_whole(extent)
    }

    pub fn is_barrier(&self, volume: [usize; 3]) -> bool {
        (0..3).any(|axis| self.is_whole_axis(axis, volume[axis]))
    }

    /// Nothing is read beyond what is written, anywhere.
    pub fn is_none(&self) -> bool {
        self.axes.iter().all(AxisReach::is_none)
    }

    /// Whether any axis says something the old triple could not.
    pub fn is_degenerate(&self) -> bool {
        self.as_symmetric().is_some()
    }

    /// Sequential reaches add; the spaces must agree.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_same_space(other, "added")?;
        Ok(Self {
            space: self.space,
            axes: [
                self.axes[0].add(&other.axes[0]),
                self.axes[1].add(&other.axes[1]),
                self.axes[2].add(&other.axes[2]),
            ],
        })
    }

    /// Exclusive and concurrent branches take the wider; the spaces must agree.
    pub fn max(&self, other: &Self) -> Result<Self> {
        self.check_same_space(other, "compared")?;
        Ok(Self {
            space: self.space,
            axes: [
                self.axes[0].max(&other.axes[0]),
                self.axes[1].max(&other.axes[1]),
                self.axes[2].max(&other.axes[2]),
            ],
        })
    }

    /// Two reaches in two coordinate spaces cannot be folded without a grid to
    /// convert with, and a phase is where a grid is chosen. So this is refused
    /// rather than guessed, and a planner turns it into "these two ops cannot
    /// share a phase" — the same shape of answer `constraint_for` gives when two
    /// ops mandate different blocks.
    fn check_same_space(&self, other: &Self, what: &str) -> Result<()> {
        if self.space != other.space {
            return Err(Error::InvalidArgument(format!(
                "a reach in {} cannot be {what} with one in {}: the two count in different \
                 coordinate spaces, and converting between them needs the grid a phase has not \
                 chosen yet. Ops that state their reaches in different spaces belong in \
                 different phases.",
                self.space.label(),
                other.space.label()
            )));
        }
        Ok(())
    }

    /// The same reach in voxels of the phase's own volume, given the lattice it
    /// is cut on.
    ///
    /// This is where [`Units::Blocks`] becomes a distance and where a permuted
    /// axis order is put back into the geometry's order. It is done at
    /// `PhaseDecomposition::derive`, which is the first place a grid exists —
    /// before that a reach has to stay symbolic, because the planner compares
    /// candidate grids and the reach must not depend on which one it is looking
    /// at.
    pub fn in_voxels(&self, block: [usize; 3]) -> Self {
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        for stated in 0..3 {
            let canonical = self.space.axes[stated];
            axes[canonical] = match self.space.units {
                // **The one place an `AxisReach::Aligned` is discounted.** This
                // is handed the lattice and is called once per candidate grid by
                // `decomposition::price_phase`, so the planner sees the cheaper
                // halo on an aligned edge and prices it — a *preference* it can
                // act on, rather than a mandate it would have to refuse against.
                // Every other variant is unchanged by this call, which is why it
                // was a `clone` before there was one that was not.
                Units::Voxels => self.axes[stated].resolved(block[canonical]),
                Units::Blocks => self.axes[stated].scaled(block[canonical].max(1)),
                // Nothing. Not zero-because-we-do-not-know: a dependency in the
                // image below's own lattice is satisfied by the fetch region,
                // not by this phase's halo, and inventing a voxel distance for
                // it is the one thing that would be wrong rather than absent.
                Units::SourceIndex => AxisReach::none(),
            };
        }
        Self {
            space: Space {
                frame: self.space.frame,
                units: Units::Voxels,
                axes: [0, 1, 2],
            },
            axes,
        }
    }

    /// A per-block table must have an entry for every block it will be asked
    /// about.
    ///
    /// Checked where a plan is built and again where one is checked, because a
    /// plan may arrive from any strategy or off a wire. A table one entry short
    /// would otherwise resolve to its last row, which is a plausible wrong
    /// answer — the kind this crate refuses to have.
    pub fn check_lattice(&self, blocks: [usize; 3], what: &str) -> Result<()> {
        for axis in 0..3 {
            if let AxisReach::PerBlock(table) = &self.axes[axis] {
                if table.len() != blocks[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "{what}: a per-block reach on axis {axis} has {} entries and the lattice \
                         has {} blocks. A per-block reach is a table indexed by the block index, \
                         so it has to have exactly one row per block — anything else is a rule \
                         nobody can reproduce.",
                        table.len(),
                        blocks[axis]
                    )));
                }
            }
        }
        Ok(())
    }

    /// A halo that hands every block exactly `extent`, whatever its position.
    ///
    /// **This is what separates "the extent I accept" from "the extent I need
    /// around it".** With one symmetric integer they are the same number: a
    /// block at a volume edge has its read clamped and is handed something
    /// narrower than an interior one, so an operation that mandates an input
    /// extent *and* reaches cannot be planned at all
    /// (`tests/op_constraints.rs`). A halo that may differ per block and per
    /// side can slide the window inward at the edges instead of clipping it, and
    /// then every block is handed the extent the operation asked for while the
    /// reach — which is still stated independently — decides what of it may be
    /// trusted.
    ///
    /// `grid` supplies the cores; `extent` is what the operation demands. It
    /// fails when no window can satisfy it, which is a fact about the request:
    /// a window cannot be smaller than the core it must contain, and cannot be
    /// larger than the volume it must lie inside.
    pub fn window(grid: &crate::geometry::BlockGrid, extent: [usize; 3]) -> Result<Self> {
        let volume = grid.volume();
        let block = grid.block();
        let counts = grid.blocks_per_axis();
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        for axis in 0..3 {
            if extent[axis] > volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "a window of {} on axis {axis} does not fit in a volume of {}: an operation \
                     that demands more than exists cannot be handed it by any halo.",
                    extent[axis], volume[axis]
                )));
            }
            if extent[axis] < block[axis] {
                return Err(Error::InvalidArgument(format!(
                    "a window of {} on axis {axis} is narrower than the block of {} it must \
                     contain. The block grid has to be cut to leave room for the window, not the \
                     other way round.",
                    extent[axis], block[axis]
                )));
            }
            let mut table = Vec::with_capacity(counts[axis]);
            for index in 0..counts[axis] {
                let core_lo = index * block[axis];
                let core_hi = (core_lo + block[axis]).min(volume[axis]);
                // Centred where there is room, slid inward at the ends. The
                // clamp that used to shorten the read now moves it.
                let want = core_lo + (core_hi - core_lo) / 2;
                let half = extent[axis] / 2;
                let read_lo = want
                    .saturating_sub(half)
                    .min(volume[axis] - extent[axis])
                    .min(core_lo);
                let read_hi = (read_lo + extent[axis]).max(core_hi);
                let read_lo = read_hi - extent[axis];
                table.push((core_lo - read_lo, read_hi - core_hi));
            }
            axes[axis] = AxisReach::PerBlock(table);
        }
        Ok(Self {
            space: Space::default(),
            axes,
        })
    }

    /// A stable digest of everything this reach says, for a fingerprint that
    /// must move when the plan does.
    pub fn digest(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// The wire form. **A degenerate reach is the triple it always was**, so a
    /// plan that does not use the new forms is the byte-identical document it
    /// was before they existed.
    pub fn to_json(&self) -> Value {
        if let Some(triple) = self.as_symmetric() {
            return json!(triple);
        }
        let mut value = json!({
            "axes": [
                self.axes[0].to_json(),
                self.axes[1].to_json(),
                self.axes[2].to_json(),
            ],
        });
        if self.space.frame != Frame::Phase {
            value["frame"] = json!("source");
        }
        if self.space.units != Units::Voxels {
            value["units"] = json!(match self.space.units {
                Units::Blocks => "blocks",
                _ => "source-index",
            });
        }
        if self.space.axes != [0, 1, 2] {
            value["order"] = json!(self.space.axes);
        }
        value
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        if let Some(triple) = value.as_array() {
            if triple.len() != 3 {
                return Err(Error::InvalidArgument(format!(
                    "{value} is not a reach: a bare list is the symmetric per-axis form and has \
                     three entries."
                )));
            }
            return Ok(Self::symmetric([
                number(&triple[0])?,
                number(&triple[1])?,
                number(&triple[2])?,
            ]));
        }
        let object = value
            .as_object()
            .ok_or_else(|| Error::InvalidArgument(format!("{value} is not a reach")))?;
        let stated = object
            .get("axes")
            .and_then(Value::as_array)
            .filter(|axes| axes.len() == 3)
            .ok_or_else(|| {
                Error::InvalidArgument(format!("{value} is not a reach: it has no three axes"))
            })?;
        let axes = [
            AxisReach::from_json(&stated[0])?,
            AxisReach::from_json(&stated[1])?,
            AxisReach::from_json(&stated[2])?,
        ];
        let frame = match object.get("frame").and_then(Value::as_str) {
            None | Some("phase") => Frame::Phase,
            Some("source") => Frame::Source,
            Some(other) => {
                return Err(Error::InvalidArgument(format!(
                    "{other:?} is not a coordinate frame. It is \"phase\" or \"source\"."
                )))
            }
        };
        let units = match object.get("units").and_then(Value::as_str) {
            None | Some("voxels") => Units::Voxels,
            Some("blocks") => Units::Blocks,
            Some("source-index") => Units::SourceIndex,
            Some(other) => {
                return Err(Error::InvalidArgument(format!(
                    "{other:?} is not a reach unit. It is \"voxels\", \"blocks\" or \
                     \"source-index\"."
                )))
            }
        };
        let mut space = Space {
            frame,
            units,
            axes: [0, 1, 2],
        };
        if let Some(order) = object.get("order").and_then(Value::as_array) {
            if order.len() != 3 {
                return Err(Error::InvalidArgument(
                    "a coordinate space's axis order names three axes".to_string(),
                ));
            }
            space =
                space.with_axes([number(&order[0])?, number(&order[1])?, number(&order[2])?])?;
        }
        Ok(Self { space, axes })
    }
}

impl std::fmt::Display for Reach {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_symmetric() {
            Some(triple) => write!(formatter, "{triple:?}"),
            None => write!(formatter, "{:?} in {}", self.axes, self.space.label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BlockGrid;

    /// The form every op in this crate states, and the promise that it is
    /// exactly what it was.
    #[test]
    fn a_symmetric_triple_survives_the_round_trip_as_itself() {
        let reach = Reach::from([4, 0, 2]);
        assert_eq!(reach.as_symmetric(), Some([4, 0, 2]));
        assert!(reach == [4, 0, 2]);
        assert_eq!(reach.to_json(), json!([4, 0, 2]));
        assert_eq!(Reach::from_json(&reach.to_json()).unwrap(), reach);
        assert_eq!(reach.bound([100, 100, 100]), [4, 0, 2]);
    }

    /// The fingerprint of a plan that says nothing new must not move, and the
    /// hash is where that is decided.
    #[test]
    fn the_degenerate_form_hashes_as_the_triple_it_replaced() {
        for triple in [[0, 0, 0], [4, 0, 0], [7, 3, 1]] {
            let mut expected = DefaultHasher::new();
            triple.hash(&mut expected);
            assert_eq!(Reach::from(triple).digest(), expected.finish());
        }
        // and something the triple cannot say hashes differently
        assert_ne!(
            Reach::asymmetric([(4, 0), (0, 0), (0, 0)]).digest(),
            Reach::from([4, 0, 0]).digest()
        );
    }

    #[test]
    fn asymmetry_is_carried_per_side_and_folds_per_side() {
        let one = Reach::asymmetric([(3, 0), (0, 0), (0, 0)]);
        let two = Reach::asymmetric([(1, 5), (0, 0), (0, 0)]);
        assert_eq!(one.add(&two).unwrap().at(0, 0, 64), (4, 5));
        assert_eq!(one.max(&two).unwrap().at(0, 0, 64), (3, 5));
        // and it is not a triple, so nothing can silently treat it as one
        assert_eq!(one.as_symmetric(), None);
        assert_eq!(one.bound([64, 64, 64]), [3, 0, 0]);
    }

    #[test]
    fn all_absorbs_every_fold_and_is_a_barrier_by_type() {
        let all = Reach::all();
        let small = Reach::from([1, 1, 1]);
        assert_eq!(all.add(&small).unwrap(), all);
        assert_eq!(small.max(&all).unwrap(), all);
        assert!(all.is_barrier([4096, 4096, 4096]));
        // 4095 of 4096 is not a barrier and never becomes one by being large
        assert!(!Reach::from([4095, 0, 0]).is_barrier([4096, 1, 1]));
        assert!(Reach::from([4096, 0, 0]).is_barrier([4096, 1, 1]));
        // an axis of extent 1 is spanned by every block already
        assert!(!Reach::all().is_barrier([1, 1, 1]));
    }

    #[test]
    fn a_per_block_reach_is_a_table_indexed_by_the_block_index() {
        let reach = Reach::per_axis([
            AxisReach::PerBlock(vec![(0, 4), (2, 2), (4, 0)]),
            AxisReach::none(),
            AxisReach::none(),
        ]);
        assert_eq!(reach.at(0, 0, 64), (0, 4));
        assert_eq!(reach.at(0, 2, 64), (4, 0));
        assert_eq!(reach.bound([64, 64, 64]), [4, 0, 0]);
        reach.check_lattice([3, 1, 1], "t").unwrap();
        let message = reach.check_lattice([4, 1, 1], "t").unwrap_err().to_string();
        assert!(
            message.contains("3 entries") && message.contains("4 blocks"),
            "{message}"
        );
    }

    #[test]
    fn blocks_become_voxels_only_where_a_grid_is_known() {
        let reach = Reach::from([2, 0, 0]).in_space(Space::blocks());
        assert_eq!(
            reach.as_symmetric(),
            None,
            "the units are part of the value"
        );
        let voxels = reach.in_voxels([32, 8, 8]);
        assert_eq!(voxels.as_symmetric(), Some([64, 0, 0]));
    }

    /// The tag a permuted branch will need exists and converts the numbers,
    /// which is the half of the problem that lives in the reach.
    #[test]
    fn a_permuted_space_is_put_back_into_the_geometrys_axis_order() {
        let space = Space::default().with_axes([2, 0, 1]).unwrap();
        let reach = Reach::from([5, 0, 0]).in_space(space);
        // stated axis 0 is the geometry's axis 2
        assert_eq!(reach.in_voxels([8, 8, 8]).as_symmetric(), Some([0, 0, 5]));
        assert!(Space::default().with_axes([0, 0, 1]).is_err());
    }

    #[test]
    fn spaces_that_disagree_are_refused_rather_than_guessed() {
        let phase = Reach::from([1, 0, 0]);
        let blocks = Reach::from([1, 0, 0]).in_space(Space::blocks());
        let message = phase.add(&blocks).unwrap_err().to_string();
        assert!(message.contains("different coordinate spaces"), "{message}");
    }

    /// The window that separates "what I accept" from "what I need around it".
    #[test]
    fn a_window_halo_hands_every_block_the_same_extent() {
        let grid = BlockGrid::new([12, 4, 4], [2, 4, 4]).unwrap();
        let halo = Reach::window(&grid, [4, 4, 4]).unwrap();
        for core in grid.cores() {
            let (lo, hi) = halo.at(0, core.index[0], 12);
            let read_lo = core.core.start[0] - lo;
            let read_hi = core.core.start[0] + core.core.shape[0] + hi;
            assert_eq!(read_hi - read_lo, 4, "block {:?}", core.index);
            assert!(read_hi <= 12);
        }
        // and a demand no window can meet is refused rather than approximated
        assert!(Reach::window(&grid, [64, 4, 4]).is_err());
        assert!(Reach::window(&grid, [1, 4, 4]).is_err());
    }

    #[test]
    fn the_rich_forms_survive_the_wire() {
        let cases = [
            Reach::asymmetric([(255, 0), (0, 3), (0, 0)]),
            Reach::all(),
            Reach::per_axis([
                AxisReach::PerBlock(vec![(0, 2), (1, 1)]),
                AxisReach::All,
                AxisReach::none(),
            ]),
            // Added when the variant was: a hand-enumerated case list is
            // exactly the assertion a new variant slips past, and this one
            // would have — see the dedicated test below for what it slips into.
            Reach::per_axis([
                AxisReach::aligned(32, 4, 4),
                AxisReach::aligned(3, 0, 2),
                AxisReach::none(),
            ]),
            Reach::from([1, 2, 3]).in_space(Space::source_voxels()),
            Reach::from([1, 2, 3]).in_space(Space::blocks()),
            Reach::from([1, 2, 3]).in_space(Space::default().with_axes([1, 2, 0]).unwrap()),
        ];
        for reach in cases {
            let rebuilt = Reach::from_json(&reach.to_json()).unwrap();
            assert_eq!(rebuilt, reach, "{}", reach.to_json());
            assert_eq!(rebuilt.digest(), reach.digest());
        }
    }

    /// **An aligned reach must not come back off the wire as a bounded one**,
    /// and the shape of `from_json` is why this needs saying: its last arm
    /// accepts *any* object and reads `lo`/`hi` out of it, so an aligned reach
    /// whose `stride` key were not handled first would round-trip into a
    /// `Bounded` carrying the **aligned** sides with the stride dropped — a halo
    /// narrower than the op needs, on a lattice nothing would then check, in
    /// another process.
    #[test]
    fn an_aligned_reach_does_not_come_back_as_the_halo_it_would_have_on_a_good_lattice() {
        let aligned = AxisReach::aligned(32, 4, 4);
        let rebuilt = AxisReach::from_json(&aligned.to_json()).unwrap();
        assert_eq!(rebuilt, aligned);
        // The specific wrong answer, named so that a regression is recognised
        // rather than merely detected.
        assert_ne!(
            rebuilt,
            AxisReach::Bounded { lo: 4, hi: 4 },
            "an aligned reach that decoded as its own discounted sides would \
             under-halo every block on every unaligned lattice"
        );
        // **Liveness.** The two are only distinguishable because `bound` reports
        // the worst case; a variant whose `bound` had been written as the
        // aligned sides would make the assertion above pass and mean nothing.
        assert_eq!(aligned.bound(1024), (35, 35));
        assert_eq!(AxisReach::Bounded { lo: 4, hi: 4 }.bound(1024), (4, 4));
    }

    /// The discount, and the four ways it is deliberately not taken.
    #[test]
    fn an_aligned_reach_is_discounted_only_against_a_lattice_that_earns_it() {
        let aligned = AxisReach::aligned(32, 4, 4);
        // Taken: the edge is a whole number of tiles.
        for edge in [32, 64, 96, 128, 256] {
            assert_eq!(
                aligned.resolved(edge),
                AxisReach::Bounded { lo: 4, hi: 4 },
                "edge {edge} is a whole number of tiles"
            );
        }
        // Not taken: the edge is not, or there is no edge at all.
        for edge in [0, 1, 16, 24, 48, 80] {
            assert_eq!(
                aligned.resolved(edge),
                AxisReach::Bounded { lo: 35, hi: 35 },
                "edge {edge} is not a whole number of tiles"
            );
        }
        // Not taken: nothing that cannot see a lattice may take it.
        assert_eq!(aligned.bound(1024), (35, 35));
        assert_eq!(aligned.at(7, 1024), (35, 35));
        assert_eq!(aligned.widest(1024), 35);
        // **Inverted, not deleted.** This block used to assert that a fold
        // flattens — `aligned.add(&Bounded { lo: 1, hi: 1 })` was
        // `Bounded { lo: 36, hi: 36 }` and `aligned.add(&aligned)` was
        // `Bounded { lo: 70, hi: 70 }` — on the argument that two ops may state
        // two strides. That was true of the second case and wrong about the
        // first, and the first is the common one: fusing a transform
        // convolution with a voxelwise map that reaches nothing lost the whole
        // discount, **27 blocks against one on `96^3`**. A fold now carries
        // both answers.
        let other = AxisReach::Bounded { lo: 1, hi: 1 };
        assert_eq!(aligned.add(&other), AxisReach::aligned(32, 5, 5));
        assert_eq!(aligned.max(&other), AxisReach::aligned(32, 4, 4));
        assert_eq!(
            aligned.add(&AxisReach::none()),
            aligned,
            "adding nothing is the identity, which is the case the old rule got wrong"
        );
        // `narrowest` is the one direction that reports the aligned amount,
        // because it answers "what is granted everywhere".
        assert_eq!(aligned.narrowest(1024), 4);
        // **Liveness.** Every assertion above would hold for a variant that
        // ignored its stride entirely and always answered the worst case, so
        // the discount has to be shown to happen at all.
        assert_ne!(aligned.resolved(64), aligned.resolved(48));
    }

    /// **Two aligned reaches add on both answers, and the unaligned one is not
    /// `stride - 1`.** The trap this is here for: a variant carrying one
    /// `(lo, hi)` plus the rule "`stride - 1` when unaligned" would fold two
    /// stride-32 reaches into something claiming `31 + lo` off-alignment, where
    /// the truth is `31 + 31 + lo` — an **under-halo of 31 voxels**, silent, on
    /// every lattice the stride does not divide.
    #[test]
    fn two_aligned_reaches_add_on_both_answers_and_not_on_the_stride() {
        let one = AxisReach::aligned(32, 4, 4);
        let sum = one.add(&one);
        assert_eq!(
            sum,
            AxisReach::Aligned {
                stride: 32,
                aligned: (8, 8),
                unaligned: (70, 70),
            }
        );
        // The two answers, each checked against what the phase really reads.
        assert_eq!(sum.resolved(64), AxisReach::Bounded { lo: 8, hi: 8 });
        assert_eq!(sum.resolved(48), AxisReach::Bounded { lo: 70, hi: 70 });
        // And the number the trap would have produced, named so a regression is
        // recognised rather than merely detected.
        assert_ne!(
            sum.bound(1024),
            (35, 35),
            "an unaligned answer of `stride - 1 + lo + lo` would under-halo by 31 a side"
        );
        assert_eq!(sum.bound(1024), (70, 70));
    }

    /// A fold of two different strides uses their least common multiple, and
    /// **degrades to the old behaviour rather than to a wrong one** when that
    /// multiple is past every edge anybody would cut on.
    #[test]
    fn a_fold_of_two_strides_takes_their_common_multiple() {
        let thirty_two = AxisReach::aligned(32, 1, 1);
        let forty_eight = AxisReach::aligned(48, 1, 1);
        let sum = thirty_two.add(&forty_eight);
        assert_eq!(
            sum,
            AxisReach::Aligned {
                stride: 96,
                aligned: (2, 2),
                unaligned: (80, 80),
            }
        );
        // 96 divides 96 and 192 and nothing smaller that anyone would cut on.
        assert_eq!(sum.resolved(96), AxisReach::Bounded { lo: 2, hi: 2 });
        assert_eq!(sum.resolved(64), AxisReach::Bounded { lo: 80, hi: 80 });
        // **The graceful-degradation claim, asserted.** Coprime strides give a
        // multiple past any candidate edge, and the reach is then its unaligned
        // answer everywhere — which is exactly what flattening gave, so the fold
        // is never worse than the rule it replaced.
        let coprime = AxisReach::aligned(31, 1, 1).add(&AxisReach::aligned(32, 1, 1));
        // The unaligned answers the two would have carried alone, added — which
        // is precisely what flattening used to produce.
        let flattened =
            AxisReach::Bounded { lo: 31, hi: 31 }.add(&AxisReach::Bounded { lo: 32, hi: 32 });
        for edge in [16, 32, 48, 64, 96, 128, 256] {
            assert_eq!(
                coprime.resolved(edge),
                flattened,
                "at edge {edge} a coprime fold must cost exactly what flattening cost"
            );
        }
        // Liveness: it is not simply always flattened — its own multiple works.
        assert_eq!(
            coprime.resolved(31 * 32),
            AxisReach::Bounded { lo: 2, hi: 2 }
        );
    }

    /// `max` is exact on a common multiple and **generous** off it, which is the
    /// direction a halo may err in.
    #[test]
    fn max_of_two_aligned_reaches_is_exact_where_it_can_be_and_generous_where_it_cannot() {
        let wide = AxisReach::aligned(32, 6, 6);
        let narrow = AxisReach::aligned(16, 2, 2);
        let joined = wide.max(&narrow);
        assert_eq!(
            joined,
            AxisReach::Aligned {
                stride: 32,
                aligned: (6, 6),
                unaligned: (37, 37),
            }
        );
        // Exact where 32 divides: both branches take their aligned answer.
        assert_eq!(joined.resolved(64), AxisReach::Bounded { lo: 6, hi: 6 });
        // Generous where only 16 does: the truth is `max(37, 2) = 37` for the
        // wide branch and 2 for the narrow, so 37 is exactly right here — the
        // generosity shows where the *narrow* branch is the wider unaligned one.
        assert_eq!(joined.resolved(48), AxisReach::Bounded { lo: 37, hi: 37 });
        // Liveness: a `max` that had taken the minimum, or ignored one operand,
        // would be caught here.
        assert!(
            joined.bound(1024).0 >= wide.bound(1024).0,
            "a join must cover its widest branch"
        );
        assert!(joined.bound(1024).0 >= narrow.bound(1024).0);
    }

    /// The discount is taken **where a lattice exists**, which is
    /// [`Reach::in_voxels`], and nowhere earlier.
    #[test]
    fn a_reach_in_voxels_takes_the_discount_the_lattice_earns() {
        let reach = Reach::per_axis([
            AxisReach::aligned(32, 4, 4),
            AxisReach::aligned(32, 4, 4),
            AxisReach::aligned(32, 4, 4),
        ]);
        assert_eq!(reach.bound([1024, 1024, 1024]), [35, 35, 35]);
        assert_eq!(
            reach.in_voxels([64, 48, 64]).bound([1024, 1024, 1024]),
            [4, 35, 4],
            "one unaligned axis discounts the other two and not itself"
        );
        // A grid's own arithmetic, so the figure is the crate's and not one
        // written here: the discount is worth this much of read amplification.
        let grid = BlockGrid::new([1024, 1024, 1024], [64, 64, 64]).unwrap();
        let worst = grid.mean_read_voxels(&reach);
        let taken = grid.mean_read_voxels(&reach.in_voxels(grid.block()));
        assert!(
            worst > taken * 5.0,
            "the discount is worth {:.3}x at edge 64 and this test claims it is large",
            worst / taken
        );
        // **Liveness.** At one block the two are the same fetch, so a test that
        // had picked that lattice would be asserting nothing.
        let whole = BlockGrid::new([1024, 1024, 1024], [1024, 1024, 1024]).unwrap();
        assert_eq!(
            whole.mean_read_voxels(&reach),
            whole.mean_read_voxels(&reach.in_voxels(whole.block()))
        );
    }

    #[test]
    fn the_source_frame_is_the_one_that_does_not_trust_a_clamp() {
        assert!(Space::phase_voxels().clamp_is_an_edge());
        assert!(!Space::source_voxels().clamp_is_an_edge());
    }
}
