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
            Self::All => (extent, extent),
        }
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
        Ok(Self::Bounded {
            lo: object.get("lo").map(number).transpose()?.unwrap_or(0),
            hi: object.get("hi").map(number).transpose()?.unwrap_or(0),
        })
    }
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
                Units::Voxels => self.axes[stated].clone(),
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

    #[test]
    fn the_source_frame_is_the_one_that_does_not_trust_a_clamp() {
        assert!(Space::phase_voxels().clamp_is_an_edge());
        assert!(!Space::source_voxels().clamp_is_an_edge());
    }
}
