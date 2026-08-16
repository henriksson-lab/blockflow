// SPDX-License-Identifier: MIT
//
// A windowed statistic evaluated on a coarse lattice, and the interpolation
// back to full resolution, as **two ops rather than one**.
//
// `local.rs` already ships the fused form: `LocalStatisticOp` evaluates a
// statistic at every lattice point and interpolates the result back to every
// voxel of its buffer, writing at full resolution. It is correct, it is
// decomposition-invariant, and it is what a caller wants when the spacing is
// small. What it cannot do is make the two dependencies it carries visible
// separately, and at a wide spacing that is the whole cost:
//
// | | reads, per block, per side | writes |
// |---|---|---|
// | fused | `spacing + element radius` fine voxels | fine volume |
// | statistic alone | `element radius` fine voxels | one voxel per sample |
// | interpolation alone | one **coarse** voxel | fine volume |
//
// At a spacing of 25 the fused form fetches 25 fine voxels of halo per side per
// block that neither of the split forms does, and it fetches them at full
// resolution because that is the only space it has. Split, the interpolation's
// halo is a coarse voxel — `spacing` times cheaper by construction, on every
// axis — and the statistic's halo is the element and nothing else.
//
// **The subsampling is not the part that needs splitting off; the
// interpolation is.** Evaluating a filter at every `n`-th position is an output
// grid with fewer points and no halo of its own. Interpolating that grid back
// is what introduces a dependency on neighbouring samples, and a dependency
// that is hidden inside an op that also computes those samples cannot be
// priced, cannot be scheduled against, and cannot be checked against a fetch.
// Two ops, two declarations.
//
// What each phase's geometry rests on
// -----------------------------------
//
// Both ops are **cross-grid**: the space their blocks are cut from is not the
// space they read. `SampleLattice`'s grid half (`lattice_volume`,
// `source_window`, `samples_for_window`, `uniform_gap`) is where that mapping
// lives, and `tests/lattice_grid.rs` is what it is checked by. Two properties
// carry the whole arrangement, and both are already argued there:
//
// * **the statistic's fetch is a constant width, slid inward at the volume's
//   ends rather than clipped.** `BlockOp::output_shape` maps an input *shape* to
//   an output shape and sees nothing else, so a block holding the same number of
//   samples as an interior one must be handed the same number of voxels or its
//   output count could not be derived at all. (The interpolation's fetch is
//   tight instead, because its extent comes from its placement and needs no
//   inversion — see below.)
// * **the blocks are cut on the output grid.** A phase's valid regions must
//   tile what it writes and every chunk of that output must fall inside exactly
//   one of them. Cutting the *input* grid and deriving output counts would put
//   block boundaries wherever the arithmetic landed; cutting the output grid
//   cannot. What varies instead is how much of the other level each block
//   reads, and that is stated per block in `BlockGeometry::source`.
//
// The reach both ops declare is therefore `Reach::none()` in
// `Space::source_index`: **a marked zero, not a silent one.** The dependency is
// real and it is stated per block in the fetch region, in units this phase's
// grid has no factor to convert — which is exactly what `Units::SourceIndex`
// exists to say, and why a phase declaring it without per-block fetch regions
// is refused by name rather than planned as though it reached nothing.
//
// The two halves are not symmetric, and the asymmetry is the whole story
// ---------------------------------------------------------------------
//
// The **statistic** half's fetch is a constant width slid inward at the volume's
// ends, and it can be: its output *is* the lattice, so the number of samples a
// block owns is a function of the fine window it was handed, and the inversion
// exists.
//
// The **interpolation** half has no such inversion, and for a reason no grid
// removes: a lattice's first sample sits half a gap into the volume and the
// volume's extent is not in general a whole number of gaps, so a block at each
// end covers a different span from an interior one while fetching the same
// samples. It is the *placement* that differs, not the shape. That is why this
// half ran in **one block only** until `op::Placement` was folded through the
// chain — migration step 4 in `forme.md`, arriving from the consumer's end.
//
// With the placement in hand nothing has to be inverted. The extent a block
// writes is the block's own read region, which the plan holds; the fetch is then
// simply the samples that block's voxels bracket between, which is *tighter*
// than the constant-width window it replaced. `BlockOp::placed_output_shape` is
// where the extent is taken from the plan and `BlockOp::apply_placed` is where
// both ends of the map arrive together — the coarse offset the buffer was read
// at, and the fine offset the block owns. Neither is derivable from the other,
// which is precisely what an `Anchor` could not say and a `Placement` can.
//
// One check moves with it. Taking the write extent from the plan makes the
// executor's "declared shape against derived read extent" comparison compare the
// plan with itself, so the op owes a check of its own: every voxel it writes
// must bracket between samples the buffer actually holds, refused by name rather
// than clamped to the buffer's edge. That is a check against data instead of
// against a declaration, which is the stronger of the two.

use ndarray::{ArrayView3, ArrayViewMut3};

use super::element::{StructuringElement, Total};
use super::local::{SampleLattice, Sampling, Statistic};
use crate::decomposition::PhaseDecomposition;
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Geometry, InputMap, Placement};
use crate::reach::{Reach, Space};
use crate::region::Region;
use crate::voxels::Voxels;

/// The reach both ops in this module declare, in the space that says what it
/// means: nothing this phase's halo can supply, because the dependency is in
/// the other level's own lattice and is satisfied by the fetch region.
fn lattice_reach() -> Reach {
    Reach::none().in_space(Space::source_index())
}

// ------------------------------------------------------------ statistic --

/// Evaluate a statistic over a window at each point of a lattice, writing **one
/// voxel per sample**.
///
/// The output is the lattice read as a coordinate space of its own: a volume of
/// `SampleLattice::lattice_volume()`, one voxel per sample, in sample order.
/// Nothing is interpolated here, which is the point — see the module header.
///
/// **The lattice is materialised into the op**, and that is a real difference
/// from [`LocalStatisticOp`], which holds a `Sampling` and builds the lattice
/// per call from `Anchor::volume`. That op can: it writes at full resolution, so
/// its output volume is its input volume whatever the lattice turns out to be.
/// This one's output volume *is* the lattice, so the lattice has to exist before
/// the plan does — `BlockOp::geometry` and `BlockOp::output_shape` are both
/// asked for it, and neither is given a volume to derive it from. An op that is
/// bound to one volume is the honest shape for a phase whose output space is a
/// function of that volume.
///
/// [`LocalStatisticOp`]: super::local::LocalStatisticOp
#[derive(Debug)]
pub struct LatticeStatisticOp {
    name: &'static str,
    element: StructuringElement,
    statistic: Statistic,
    lattice: SampleLattice,
    cost: f64,
}

impl LatticeStatisticOp {
    /// Bind a statistic and a window to the lattice `sampling` lays over
    /// `volume`.
    ///
    /// **Refuses an irregular lattice, by name.** `source_window` needs a
    /// constant gap to produce a constant fetch width and `samples_for_window`
    /// needs one to invert it, so an unevenly spread lattice has no output shape
    /// this op could declare. That is a fact about the request rather than a
    /// failure to try, and the caller that gets it has [`LocalStatisticOp`],
    /// which needs no inversion because it writes at full resolution.
    ///
    /// [`LocalStatisticOp`]: super::local::LocalStatisticOp
    pub fn new(
        name: &'static str,
        element: StructuringElement,
        sampling: &Sampling,
        statistic: Statistic,
        volume: [usize; 3],
    ) -> Result<Self> {
        if element.is_empty() {
            return Err(Error::InvalidArgument(
                "a lattice statistic needs a non-empty window".to_string(),
            ));
        }
        let lattice = SampleLattice::of(sampling, volume)?;
        for axis in 0..3 {
            if lattice.uniform_gap(axis).is_none() {
                return Err(Error::InvalidArgument(format!(
                    "op {name:?}: the lattice on axis {axis} is unevenly spread ({:?}), so a block \
                     of samples has no constant fetch width and this op's output shape could not \
                     be inverted from the shape it is handed. An unevenly spread lattice is \
                     expressible by `LocalStatisticOp`, which writes at full resolution and needs \
                     no inversion.",
                    lattice.positions(axis)
                )));
            }
        }
        let cost = super::local::SAMPLE_COST_PER_ELEMENT_VOXEL * element.len() as f64
            + statistic.cost_per_sample(element.len());
        Ok(Self {
            name,
            element,
            statistic,
            lattice,
            cost,
        })
    }

    pub fn lattice(&self) -> &SampleLattice {
        &self.lattice
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// How far each sample's window reaches below and above it, per axis: the
    /// element's own sides and nothing else.
    ///
    /// **The term that is missing is the point of this op.** The fused form's
    /// reach is this plus the lattice's widest gap, because it also has to reach
    /// the samples a voxel interpolates from. Here it does not: the samples are
    /// what this op writes.
    fn window(&self) -> [(usize, usize); 3] {
        [
            self.element.sides(0),
            self.element.sides(1),
            self.element.sides(2),
        ]
    }

    /// The fine voxels a block holding samples `start .. start + count` is
    /// handed, per axis.
    fn fetch_axis(&self, axis: usize, start: usize, count: usize) -> Option<(usize, usize)> {
        let (lo, hi) = self.element.sides(axis);
        self.lattice.source_window(axis, start, count, lo, hi)
    }

    /// The fine region a block reading `read` — stated in **lattice indices** —
    /// must be handed.
    ///
    /// The constant-width form, which is what makes [`Self::output_shape`]
    /// invertible; `SampleLattice::source_region` is the tight one and is what a
    /// coverage guard compares against.
    pub fn fetch_region(&self, read: &Region) -> Result<Region> {
        if read.ndim() != 3 {
            return Err(Error::InvalidArgument(format!(
                "a lattice statistic is 3-D, got a read region of rank {}",
                read.ndim()
            )));
        }
        let mut start = [0usize; 3];
        let mut shape = [0usize; 3];
        for axis in 0..3 {
            let (low, high) = self
                .fetch_axis(axis, read.start[axis], read.shape[axis])
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "op {:?}: a block holding samples {}..{} of axis {axis} has no constant \
                         fetch window — {} sample(s) plus the element's {:?} span more than the \
                         {} voxel(s) the axis holds. The block grid has to be cut small enough on \
                         the lattice to leave room for one window.",
                        self.name,
                        read.start[axis],
                        read.start[axis] + read.shape[axis],
                        read.shape[axis],
                        self.element.sides(axis),
                        self.lattice.volume()[axis]
                    ))
                })?;
            start[axis] = low;
            shape[axis] = high - low;
        }
        Ok(Region::new(&start, &shape))
    }

    /// Which samples a buffer handed to [`BlockOp::apply`] holds, recovered from
    /// where that buffer sits.
    ///
    /// **The inversion this op exists on, and the one place it can fail.** The
    /// executor builds an `Anchor` from the fetch region, so `apply` is told
    /// where it is in the *fine* volume and has to work out which lattice
    /// indices that is. `source_window` slides inward at the volume's ends, so
    /// the map from a sample index to a fetch start is not the affine one a
    /// ratio would give, and it is inverted by searching the candidates rather
    /// than by an arithmetic that would have to reproduce the slide.
    ///
    /// Two starts landing on one fetch is refused rather than resolved, and the
    /// refusal names what would lift it: two blocks handed identical buffers at
    /// identical offsets cannot be told apart by an `Anchor`, which carries
    /// where a buffer was *read* from and not which outputs it owns. `Placement`
    /// is the type that carries both; until the executor passes one, a plan that
    /// would need it is refused at the block rather than computed wrongly.
    /// [`lattice_statistic_phase`] checks the same property once, when the plan
    /// is built, so this is the backstop and not the report.
    fn samples_at(&self, at: &Anchor, extent: [usize; 3]) -> Result<[usize; 3]> {
        let mut start = [0usize; 3];
        for axis in 0..3 {
            let (lo, hi) = self.element.sides(axis);
            let count = self
                .lattice
                .samples_for_window(axis, extent[axis], lo, hi)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "op {:?}: a buffer of {} voxel(s) on axis {axis} is not a whole number of \
                         sample gaps past a window of {}+{}+1, so it holds no whole number of \
                         samples. The fetch a block is handed has to be the one \
                         `LatticeStatisticOp::fetch_region` states.",
                        self.name, extent[axis], lo, hi
                    ))
                })?;
            let mut found: Option<usize> = None;
            for candidate in 0..=self.lattice.count(axis).saturating_sub(count) {
                if self.fetch_axis(axis, candidate, count)
                    == Some((at.offset[axis], at.offset[axis] + extent[axis]))
                {
                    if found.is_some() {
                        return Err(Error::InvalidArgument(format!(
                            "op {:?}: a buffer of {} voxel(s) at {} on axis {axis} is the fetch of \
                             more than one block of {count} sample(s), because the window slid \
                             inward to the same place for both. An `Anchor` says where a buffer \
                             was read from and not which outputs it owns, so the two cannot be \
                             told apart here; `op::Placement` is what carries both, and until the \
                             executor passes one this plan is refused rather than computed for \
                             whichever block asked first.",
                            self.name, extent[axis], at.offset[axis]
                        )));
                    }
                    found = Some(candidate);
                }
            }
            start[axis] = found.ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "op {:?}: a buffer of {} voxel(s) at {} on axis {axis} is the fetch of no \
                     block of {count} sample(s) of this lattice. A block's buffer has to be the \
                     region `LatticeStatisticOp::fetch_region` states for it.",
                    self.name, extent[axis], at.offset[axis]
                ))
            })?;
        }
        Ok(start)
    }
}

impl BlockOp for LatticeStatisticOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing, and see [`Self::reach_spec`] for why that is a statement rather
    /// than an omission.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// Nothing this phase's halo can supply, in the space that says so.
    ///
    /// The phase's own grid is the **lattice**, and a sample's dependency on the
    /// fine voxels of its window is not a distance in lattice steps at all: it
    /// is a region of another level, and it is stated per block in
    /// `BlockGeometry::source`. `Units::SourceIndex` is what carries that
    /// without inventing a conversion, and it is checked rather than trusted —
    /// a phase declaring it with no per-block fetch regions is refused.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        lattice_reach()
    }

    /// The lattice's own volume, and a per-input map that has to be a stencil
    /// carrying a source-index reach rather than the `Table` this really is.
    ///
    /// **A limitation worth stating where it bites.** `InputMap::Table` is the
    /// variant meant for exactly this dependency — one fetch region per block,
    /// materialised — and it cannot be produced here: `geometry` is asked for an
    /// input *volume* and there is no grid yet, so the blocks whose regions the
    /// table would hold do not exist. What is declared instead folds to the same
    /// reach `Table` would (`Chain::reach_of` answers a source-index nothing for
    /// both) and the same output volume, and the regions themselves are attached
    /// where they can be — `PhaseDecomposition::with_sources`, in
    /// [`lattice_statistic_phase`].
    fn geometry(&self, _input_volume: [usize; 3]) -> Geometry {
        Geometry::new(
            self.lattice.lattice_volume(),
            vec![InputMap::Stencil(lattice_reach())],
        )
    }

    /// How many samples fit the constant window a block of this extent is: the
    /// inverse of the fetch width, which is why the fetch has a constant one.
    ///
    /// A shape it cannot invert answers the input's own shape, because this
    /// returns no `Result` and a wrong shape is caught immediately —
    /// `Environment::apply` allocates from it and the executor compares it
    /// against the read extent the plan derived. [`Self::samples_at`] refuses
    /// the same shape by name a moment later, which is where the message is.
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        let mut out = input;
        for axis in 0..3 {
            let (lo, hi) = self.element.sides(axis);
            if let Some(count) = self.lattice.samples_for_window(axis, input[axis], lo, hi) {
                out[axis] = count;
            }
        }
        out
    }

    /// `f64` in and `f64` out, for [`LocalStatisticOp::apply`]'s reason: the
    /// kernel is generic over `T: Copy` with an `f64` accumulator, and this
    /// shell declares what it can actually bridge rather than promising a
    /// conversion it would have to invent.
    ///
    /// [`LocalStatisticOp::apply`]: super::local::LocalStatisticOp
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let out = out.view_mut::<f64>()?;
        if self.lattice.volume() != at.volume {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: the lattice is over {:?} and the anchor says the level being read is \
                 {:?}. This op is bound to one volume — see its own docs for why — so a plan over \
                 a different one is a plan for a different op.",
                self.name,
                self.lattice.volume(),
                at.volume
            )));
        }
        let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let produced = [out.shape()[0], out.shape()[1], out.shape()[2]];
        let start = self.samples_at(at, extent)?;
        if self.output_shape(extent) != produced {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: a buffer of {extent:?} holds {:?} sample(s) and the output block is \
                 {produced:?}",
                self.name,
                self.output_shape(extent)
            )));
        }
        let ordered = input.mapv(Total);
        let full = self.element.len();
        let mut scratch: Vec<f64> = Vec::with_capacity(full);
        let statistic = &self.statistic;
        lattice_statistic_into(
            ordered.view(),
            at.offset,
            &self.element,
            &self.lattice,
            start,
            |window| statistic.reduce_with(window, full, &mut scratch),
            out,
        )
    }

    /// Exactly the statistic's own mapping: a window of one value reduces to
    /// whatever that statistic says, and no interpolation happens here to widen
    /// or narrow the claim.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.statistic.constant_maps_to(value)
    }

    /// Per **sample**, not per fine voxel, which is the arithmetic that makes a
    /// lattice worth having: the gather plus whatever the reducer does after it.
    ///
    /// The fused form divides the same figure by the spacing's cube because it
    /// is charged per voxel of its output. This op's output *is* the samples, so
    /// there is nothing to divide by and no interpolation term to add.
    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Evaluate `reduce` over `element` at the lattice points a block holds.
///
/// `offset` is where `input` sits in the fine volume and `start` is the first
/// lattice index this block owns; `out` is one voxel per sample, in sample
/// order. Nothing is interpolated — the caller that wants full resolution back
/// runs [`LatticeInterpolateOp`] over the result.
///
/// A window is clamped to the buffer. At a real volume boundary that is the
/// global clamp and is right; short of a sufficient fetch it is a truncation the
/// whole volume would not have made, and the values differ — which is how a
/// short fetch is *seen* to matter rather than merely asserted.
pub fn lattice_statistic_into<T, F>(
    input: ArrayView3<'_, T>,
    offset: [usize; 3],
    element: &StructuringElement,
    lattice: &SampleLattice,
    start: [usize; 3],
    mut reduce: F,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy,
    F: FnMut(&mut [T]) -> f64,
{
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let counts = [out.shape()[0], out.shape()[1], out.shape()[2]];
    for axis in 0..3 {
        if start[axis] + counts[axis] > lattice.count(axis) {
            return Err(Error::InvalidArgument(format!(
                "lattice_statistic_into: samples {}..{} of axis {axis} run past the {} this \
                 lattice holds",
                start[axis],
                start[axis] + counts[axis],
                lattice.count(axis)
            )));
        }
    }
    if shape.contains(&0) || counts.contains(&0) {
        return Ok(());
    }
    let mut window: Vec<T> = Vec::with_capacity(element.len());
    for p in 0..counts[0] {
        for q in 0..counts[1] {
            for r in 0..counts[2] {
                let centre = [
                    lattice.centre(0, start[0] + p) as isize,
                    lattice.centre(1, start[1] + q) as isize,
                    lattice.centre(2, start[2] + r) as isize,
                ];
                window.clear();
                for step in element.offsets() {
                    let mut index = [0usize; 3];
                    let mut inside = true;
                    for axis in 0..3 {
                        // Stated in volume coordinates and read out of the
                        // buffer, so this is the global clamp intersected with
                        // what is held — and with a sufficient fetch the
                        // intersection is the global clamp itself.
                        let global = centre[axis] + step[axis];
                        let local = global - offset[axis] as isize;
                        if local < 0 || local >= shape[axis] as isize {
                            inside = false;
                            break;
                        }
                        index[axis] = local as usize;
                    }
                    if inside {
                        window.push(input[index]);
                    }
                }
                out[[p, q, r]] = reduce(&mut window);
            }
        }
    }
    Ok(())
}

/// The most samples a block of `axis` can hold, at or below `wanted`.
///
/// **A lattice phase has a maximum block, which is the surprising half of this
/// module.** A block of `n` samples is handed a window of `(n - 1) * gap +
/// element` fine voxels, and that window has to fit inside the axis — so `n` is
/// bounded by the volume, and the bound bites hardest exactly where a lattice is
/// worth having. A `Sampling::Centred` lattice leaves an unsampled margin
/// narrower than one gap at each end, so the whole lattice fits in one block
/// only when the element is no wider than the spacing; at a spacing of 25 and an
/// element of 200 it takes at least eight blocks per axis, whatever the planner
/// would prefer.
///
/// That is a constraint on the plan and not a defect in it: the same eight
/// fetches are what the fused form would have made too, and it would have made
/// them 25 voxels wider on each side. Stated here so a planner can ask instead
/// of discovering it from [`lattice_statistic_phase`]'s refusal.
///
/// `None` when not even one sample's window fits the axis, which is a fact about
/// the element and the volume.
pub fn statistic_block_edge(op: &LatticeStatisticOp, axis: usize, wanted: usize) -> Option<usize> {
    let (lo, hi) = op.element.sides(axis);
    let gap = op.lattice.uniform_gap(axis)?;
    let extent = op.lattice.volume()[axis];
    let room = extent.checked_sub(lo + hi + 1)?;
    let most = (room / gap.max(1) + 1).min(op.lattice.count(axis));
    Some(wanted.clamp(1, most.max(1)))
}

/// The phase a lattice statistic needs: the grid over the **lattice**, and each
/// block's fetch region in the fine level below.
///
/// Shipped with the op rather than written at the call site, for
/// `resample_phase`'s reason: the fetch mapping is the op's own geometry, and a
/// plan that states it differently from the way the op reads it is a plan whose
/// blocks are silently a window out. There is one derivation and both halves use
/// it.
///
/// `grid` is over the lattice volume — one core is a set of samples — and any
/// block edge is allowed as long as one window fits the axis.
pub fn lattice_statistic_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    op: &LatticeStatisticOp,
    grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    let lattice_volume = op.lattice.lattice_volume();
    if grid.volume() != lattice_volume {
        return Err(Error::InvalidArgument(format!(
            "a lattice statistic's grid is cut from what it writes: this lattice holds \
             {lattice_volume:?} sample(s) and the grid is over {:?}. Cutting the fine grid \
             instead would put block boundaries wherever the sample arithmetic landed, which is \
             what makes two blocks write one chunk.",
            grid.volume()
        )));
    }
    // The halo is nothing, in the phase's own space, because there is nothing
    // for it to be: a sample's answer depends on no other sample. Every block's
    // read is its core and every core is written exactly once.
    let phase = PhaseDecomposition::derive(slots, names, lattice_reach(), Reach::none(), grid);
    let sources = phase
        .blocks
        .iter()
        .map(|block| op.fetch_region(&block.read))
        .collect::<Result<Vec<Region>>>()?;
    check_fetch_covers_samples(op, &phase.blocks, &sources)?;
    check_fetch_is_unambiguous(op, &sources)?;
    Ok(phase.with_sources(|block| sources[block.flat].clone()))
}

/// That each block's fetch contains every voxel its own samples read.
///
/// The tiling guard compares a granted halo against a required reach, both in
/// the phase's own space, and this phase's reach is a marked nothing — so the
/// guard has nothing to catch here at all. What can be wrong instead is the
/// fetch, and this asks the sample geometry directly rather than trusting the
/// derivation that produced it: `SampleLattice::source_span` is the tight
/// interval a block's samples actually read, and the constant-width window has
/// to contain it.
fn check_fetch_covers_samples(
    op: &LatticeStatisticOp,
    blocks: &[crate::geometry::BlockGeometry],
    sources: &[Region],
) -> Result<()> {
    let window = op.window();
    for (block, source) in blocks.iter().zip(sources) {
        let tight = op.lattice.source_region(&block.valid, window)?;
        for axis in 0..3 {
            if block.valid.shape[axis] == 0 {
                continue;
            }
            let fetched = source.start[axis]..source.start[axis] + source.shape[axis];
            let wanted = tight.start[axis]..tight.start[axis] + tight.shape[axis];
            if wanted.start < fetched.start || wanted.end > fetched.end {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}, block {:?}: its samples on axis {axis} read fine voxels {wanted:?} \
                     and the fetch this plan states is {fetched:?}. Nothing else checks that \
                     pair — this phase's reach is a source-index nothing, so the tiling guard has \
                     no distance to compare.",
                    op.name, block.index
                )));
            }
        }
    }
    Ok(())
}

/// That no two blocks are handed the same buffer at the same place.
///
/// The property [`LatticeStatisticOp::samples_at`] needs to invert its anchor,
/// checked once when the plan is built instead of once per block: the fetch
/// window slides inward at the volume's ends, so two blocks near an end can in
/// principle be slid onto each other, and a plan in which they are cannot be
/// executed by an executor that passes an `Anchor`. Reported here, where the
/// message can name the plan.
fn check_fetch_is_unambiguous(op: &LatticeStatisticOp, sources: &[Region]) -> Result<()> {
    for (first, region) in sources.iter().enumerate() {
        for (second, other) in sources.iter().enumerate().skip(first + 1) {
            if region == other {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: blocks {first} and {second} both fetch {region:?}, because the \
                     constant-width window slid inward to the same place for both. They own \
                     different samples and would be handed identical buffers, which an `Anchor` \
                     cannot tell apart — cut the lattice into fewer, larger blocks, or pass an \
                     `op::Placement` once the executor carries one.",
                    op.name
                )));
            }
        }
    }
    Ok(())
}

// -------------------------------------------------------- interpolation --

/// Re-exported at the path it was first published under.
///
/// The type itself sits beside `SampleLattice`, in `local`, because it is a rule
/// about that lattice and both this module and the fused op in `local` ask it
/// the same question. Two callers, one rule: see [`Alignment::bracket`].
pub use super::local::Alignment;

/// Interpolate a lattice-shaped volume back to full resolution.
///
/// The other half of the split: it reads the coarse level
/// [`LatticeStatisticOp`] writes and produces the fine volume the lattice was
/// laid over. **Its halo is one coarse voxel per side**, which is `gap` times
/// cheaper on every axis than the fused form's, and it is declared here rather
/// than hidden inside an op that also computes the samples.
///
/// The interpolation is `a + t * (b - a)`, the same expression
/// `SampleLattice::bracket` is documented against: where the two samples agree
/// it returns their value **exactly**, whatever the weight, which is what makes
/// `constant_maps_to` exactly true rather than nearly true.
///
/// **Which samples a voxel blends is a declared parameter**, not a property of
/// the lattice: see [`Alignment`], and [`Self::with_alignment`] for how
/// a caller states it. It is held here rather than in the `SampleLattice`
/// because it is not a fact about where the samples are — the statistic half
/// shares the same lattice and has no interpolation to have a convention about.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeInterpolateOp {
    name: &'static str,
    lattice: SampleLattice,
    alignment: Alignment,
    cost: f64,
}

impl LatticeInterpolateOp {
    /// Bind to a lattice, at [`Alignment::default`]. The same lattice the
    /// statistic was bound to, or the two phases do not compose — checked in
    /// [`lattice_interpolate_phase`], which is where both volumes are in scope.
    ///
    /// **The evenness this insists on is wider than one convention needs.**
    /// [`Alignment::PinnedEnds`] reads how many samples there are and never
    /// where they sit, so an unevenly spread lattice would interpolate under it
    /// perfectly well. The check stays on the constructor anyway, because a
    /// constructor has one contract and the convention is chosen after it — and
    /// because an uneven lattice is refused by [`interpolate_block_edge`] and by
    /// the statistic half regardless, so accepting it here would buy a caller an
    /// op it could not plan.
    pub fn new(name: &'static str, lattice: SampleLattice) -> Result<Self> {
        for axis in 0..3 {
            if lattice.uniform_gap(axis).is_none() {
                return Err(Error::InvalidArgument(format!(
                    "op {name:?}: the lattice on axis {axis} is unevenly spread ({:?}). A block of \
                     fine voxels then covers a number of samples that depends on where it sits, \
                     and this op's output shape is a function of the shape it is handed and \
                     nothing else.",
                    lattice.positions(axis)
                )));
            }
        }
        Ok(Self {
            name,
            lattice,
            alignment: Alignment::default(),
            cost: INTERPOLATE_COST,
        })
    }

    pub fn lattice(&self) -> &SampleLattice {
        &self.lattice
    }

    /// The convention this op maps a fine voxel back to the lattice by.
    ///
    /// A builder rather than an argument to [`Self::new`], for
    /// `ops::ridge::Boundary`'s reason: it has a default that is right for every
    /// caller who has not been asked to match something else, and putting it in
    /// the constructor would make every existing caller state it. The choice is
    /// part of what this op *is* — two ops differing only in it are not equal,
    /// and they compute different numbers from the same samples — so it is held
    /// in the struct and compared by its `PartialEq` rather than passed per
    /// call, where two blocks of one phase could be given different answers.
    ///
    /// It changes the op's fetch regions, and it changes them everywhere at
    /// once: `fetch_axis` is the single place they are derived, so the plan
    /// [`lattice_interpolate_phase`] builds and the samples
    /// [`BlockOp::apply_placed`] reads cannot come apart.
    ///
    /// It changes **no** output volume, no `reach_spec`, and no cost: the same
    /// trilinear blend over a differently chosen pair.
    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The samples a block writing fine voxels `[start, start + extent)` reads:
    /// exactly the ones its own voxels bracket between, and no others.
    ///
    /// **Tight, where it used to be a constant width slid inward at the ends.**
    /// The constant width was never what the interpolation wanted; it was what
    /// [`BlockOp::output_shape`] forced, because a shape-to-shape map can only
    /// invert a fetch whose width is a function of the extent written. It is
    /// also what the two end blocks could not satisfy: a lattice's first sample
    /// sits half a gap into the volume and the volume's extent is not in general
    /// a whole number of gaps, so an end block's extent is not a whole number of
    /// gaps and has no constant-width fetch at all.
    ///
    /// [`Placement`] removes the requirement rather than working around it — the
    /// extent a block writes comes from the plan and is not derived from what it
    /// fetched — so the fetch can be the set of samples the block actually
    /// reads. `bracket` is monotone in the coordinate, so the endpoints bound
    /// every voxel between them.
    ///
    /// **Which samples those are depends on this op's [`Alignment`]**, and this
    /// is the only place it is asked — so the fetch a plan states and the
    /// samples the kernel reads move together by construction rather than by two
    /// call sites agreeing. The two conventions give genuinely different
    /// answers here for the same block: they read different source positions,
    /// which is the whole of what the parameter does.
    ///
    /// [`Placement`]: crate::op::Placement
    fn fetch_axis(&self, axis: usize, start: usize, extent: usize) -> Option<(usize, usize)> {
        if start >= self.lattice.volume()[axis] {
            return None;
        }
        if extent == 0 {
            let at = self.alignment.bracket(&self.lattice, axis, start).0;
            return Some((at, at));
        }
        let last = (start + extent - 1).min(self.lattice.volume()[axis] - 1);
        let low = self.alignment.bracket(&self.lattice, axis, start).0;
        let high = self.alignment.bracket(&self.lattice, axis, last).1;
        Some((low, high + 1))
    }

    /// The coarse region a block writing `read` — stated in **fine voxels** —
    /// must be handed.
    pub fn fetch_region(&self, read: &Region) -> Result<Region> {
        if read.ndim() != 3 {
            return Err(Error::InvalidArgument(format!(
                "a lattice interpolation is 3-D, got a read region of rank {}",
                read.ndim()
            )));
        }
        let mut start = [0usize; 3];
        let mut shape = [0usize; 3];
        for axis in 0..3 {
            let (low, high) = self
                .fetch_axis(axis, read.start[axis], read.shape[axis])
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "op {:?}: a block writing {} fine voxel(s) from {} on axis {axis} starts \
                         past the {} voxel(s) this lattice is laid over, so it brackets no \
                         samples at all. A grid for this phase is cut from what it writes.",
                        self.name,
                        read.shape[axis],
                        read.start[axis],
                        self.lattice.volume()[axis]
                    ))
                })?;
            start[axis] = low;
            shape[axis] = high - low;
        }
        Ok(Region::new(&start, &shape))
    }
}

impl BlockOp for LatticeInterpolateOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// A source-index nothing, for [`LatticeStatisticOp::reach_spec`]'s reason
    /// and with the direction reversed: the dependency is on the *coarse*
    /// level's own lattice, and a step of it is not a distance in this phase's
    /// fine voxels that any grid could supply.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        lattice_reach()
    }

    /// The fine volume the lattice was laid over.
    fn geometry(&self, _input_volume: [usize; 3]) -> Geometry {
        Geometry::new(
            self.lattice.volume(),
            vec![InputMap::Stencil(lattice_reach())],
        )
    }

    /// The fine volume, for a buffer holding the whole lattice — and **nothing
    /// else**, because there is nothing else to say.
    ///
    /// This is the method the whole of migration step 4 exists because of. It
    /// maps a shape to a shape, so two blocks fetching the same samples must
    /// write the same number of voxels; a block at each end of this lattice does
    /// not, because the first sample sits half a gap in and the volume's extent
    /// is not a whole number of gaps. What differs between those blocks is not
    /// their shape but their *placement*, and the honest answer here is that the
    /// question has no answer — [`Self::placed_output_shape`] is where it is
    /// asked of something that can answer it.
    ///
    /// The input's own shape is returned for anything but the whole lattice.
    /// That is deliberately a shape the executor will reject if it ever reaches
    /// it, rather than an arithmetic that would be right for interior blocks and
    /// quietly wrong for the two at the ends.
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        if input == self.lattice.lattice_volume() {
            return self.lattice.volume();
        }
        input
    }

    /// The extent the **plan** states for this block, which is where this op's
    /// write extent has always lived.
    ///
    /// A [`Placement`] whose output anchor is in this op's own output space is a
    /// placement from an executor running this phase, and its
    /// [`Placement::writes`] is the block's read region — cut from the fine
    /// volume, which is what this phase's grid is cut from. Anything else is a
    /// caller applying the chain by hand, and falls through to
    /// [`Self::output_shape`], which answers for the whole lattice and refuses
    /// the rest a moment later in [`Self::apply`].
    ///
    /// **What this gives up, and what pays for it.** The executor compares the
    /// shape a phase declares against the read extent its plan derived; an op
    /// answering *from* that plan makes the comparison compare the plan with
    /// itself. The replacement is in [`Self::apply_placed`] and is stronger: it
    /// checks the samples the block was actually handed against the samples its
    /// voxels actually bracket, which is a check against a buffer rather than
    /// against a declaration.
    ///
    /// [`Placement`]: crate::op::Placement
    /// [`Placement::writes`]: crate::op::Placement::writes
    fn placed_output_shape(&self, input: [usize; 3], at: &Placement) -> [usize; 3] {
        if at.output.volume == self.lattice.volume() {
            if let Some(extent) = at.writes() {
                return extent;
            }
        }
        self.output_shape(input)
    }

    /// The waiver, stated. This op's two extents are not a function of each
    /// other — see [`Self::placed_output_shape`] — and the check it owes in
    /// exchange is in [`Self::apply_placed`]: every voxel written must bracket
    /// between samples the buffer actually holds.
    ///
    /// Without this the plan is refused by
    /// [`check_output_shapes`](crate::decomposition::check_output_shapes),
    /// which is the point: it is the framework noticing that an op answered out
    /// of the plan, rather than an op being trusted not to.
    fn takes_extent_from_placement(&self) -> bool {
        true
    }

    /// Interpolate the block's own fine voxels from the samples it was handed.
    ///
    /// **Both ends of the map come from the placement**, and neither is derived
    /// from the other: `at.input.offset` is the first sample the buffer holds
    /// and `at.output.offset` is the first fine voxel this block owns. That pair
    /// is exactly what an `Anchor` could not carry and what
    /// [`lattice_interpolate_into`] has always taken separately — see its own
    /// docs, which were written against the day this arrived.
    ///
    /// The check is the one [`Self::placed_output_shape`] traded away: every
    /// voxel written must bracket between samples this buffer holds. `bracket`
    /// is monotone in the coordinate, so the first and last voxel bound the
    /// rest. A short fetch is refused here by name instead of silently clamping
    /// to the buffer's edge and producing a plausible volume.
    fn apply_placed(
        &self,
        input: &Voxels,
        _sources: crate::op::SourceInputs<'_>,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        if at.output.volume != self.lattice.volume() {
            // Not a placement in this op's own output space, so there is no
            // second anchor to use: the anchor path accepts the whole lattice
            // and names what is missing for anything else.
            return self.apply(input, out, &at.input);
        }
        if at.input.volume != self.lattice.lattice_volume() {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: this lattice holds {:?} sample(s) and the placement says the level being \
                 read is {:?}",
                self.name,
                self.lattice.lattice_volume(),
                at.input.volume
            )));
        }
        let written = out.shape();
        let view = input.view::<f64>()?;
        let held = [view.shape()[0], view.shape()[1], view.shape()[2]];
        let offset = at.input.offset;
        let origin = at.output.offset;
        for axis in 0..3 {
            if written[axis] == 0 || held[axis] == 0 {
                continue;
            }
            let low = self.alignment.bracket(&self.lattice, axis, origin[axis]).0;
            let high = self
                .alignment
                .bracket(&self.lattice, axis, origin[axis] + written[axis] - 1)
                .1;
            if low < offset[axis] || high >= offset[axis] + held[axis] {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: a block writing fine voxels {}..{} of axis {axis} brackets samples \
                     {low}..={high} and was handed samples {}..{}. Every voxel this block writes \
                     has to bracket between samples the buffer holds; clamping to the buffer's \
                     edge instead would produce a plausible volume that the whole-volume answer \
                     does not agree with.",
                    self.name,
                    origin[axis],
                    origin[axis] + written[axis],
                    offset[axis],
                    offset[axis] + held[axis]
                )));
            }
        }
        let out = out.view_mut::<f64>()?;
        lattice_interpolate_into_with(view, offset, origin, &self.lattice, self.alignment, out)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let out = out.view_mut::<f64>()?;
        if self.lattice.lattice_volume() != at.volume {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: this lattice holds {:?} sample(s) and the anchor says the level being \
                 read is {:?}",
                self.name,
                self.lattice.lattice_volume(),
                at.volume
            )));
        }
        if at.offset != [0, 0, 0] || at.volume != self.lattice.lattice_volume() {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: this block holds samples from {:?} and an `Anchor` says only where the \
                 coarse buffer was read, not which fine voxels this block owns — so the whole \
                 lattice is the only block it can place. A blocked interpolation goes through \
                 `BlockOp::apply_placed`, which is handed an `op::Placement` carrying both.",
                self.name, at.offset
            )));
        }
        lattice_interpolate_into_with(
            input,
            at.offset,
            [0, 0, 0],
            &self.lattice,
            self.alignment,
            out,
        )
    }

    /// Exactly, and by the arithmetic rather than by an argument: every tap is
    /// `a + t * (b - a)` with `a == b`, which is `a`.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Interpolate a block of samples back to the fine voxels it covers.
///
/// `offset` is the first sample this buffer holds, in lattice indices, and
/// `origin` is the first fine voxel it writes. **Both, and not one derived from
/// the other**, because the derivation does not exist: see
/// [`LatticeInterpolateOp`] for why a block's fine origin is not a function of
/// its coarse fetch, and `op::Placement` for the type that would carry it.
///
/// **Every tap is clamped to the buffer**, which at the lattice's own ends is
/// the clamp the whole-volume answer makes too (a voxel outside the sampled
/// margin reads the nearest sample, and there is no other sample to read), and
/// at a block seam is a truncation the whole volume would not have made. That
/// asymmetry is deliberate: it is what makes a short fetch produce a different
/// number rather than a plausible one.
pub fn lattice_interpolate_into(
    input: ArrayView3<'_, f64>,
    offset: [usize; 3],
    origin: [usize; 3],
    lattice: &SampleLattice,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    lattice_interpolate_into_with(input, offset, origin, lattice, Alignment::default(), out)
}

/// [`lattice_interpolate_into`], with the convention stated.
///
/// The general form; the one above is this at [`Alignment::default`], which
/// is what every caller meant before there was a choice — so nothing moves by
/// adding this and no existing call site has to say anything.
///
/// The convention is applied to a **global** fine coordinate, `origin + step`,
/// and the sample indices it answers are then shifted by `offset` into the
/// buffer. That order is the whole reason this is decomposition-invariant: the
/// coordinate a voxel maps to is a function of the volume and the voxel, and a
/// block never enters the arithmetic.
pub fn lattice_interpolate_into_with(
    input: ArrayView3<'_, f64>,
    offset: [usize; 3],
    origin: [usize; 3],
    lattice: &SampleLattice,
    alignment: Alignment,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    let held = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    for axis in 0..3 {
        if offset[axis] + held[axis] > lattice.count(axis) {
            return Err(Error::InvalidArgument(format!(
                "lattice_interpolate_into: samples {}..{} of axis {axis} run past the {} this \
                 lattice holds",
                offset[axis],
                offset[axis] + held[axis],
                lattice.count(axis)
            )));
        }
    }
    if held.contains(&0) || shape.contains(&0) {
        return Ok(());
    }
    let brackets: Vec<Vec<(usize, usize, f64)>> = (0..3)
        .map(|axis| {
            (0..shape[axis])
                .map(|step| {
                    let (a, b, t) = alignment.bracket(lattice, axis, origin[axis] + step);
                    let last = held[axis] - 1;
                    (
                        a.saturating_sub(offset[axis]).min(last),
                        b.saturating_sub(offset[axis]).min(last),
                        t,
                    )
                })
                .collect()
        })
        .collect();
    for i in 0..shape[0] {
        let (a0, b0, t0) = brackets[0][i];
        for j in 0..shape[1] {
            let (a1, b1, t1) = brackets[1][j];
            for k in 0..shape[2] {
                let (a2, b2, t2) = brackets[2][k];
                let edge = |p: usize, q: usize| lerp(input[[p, q, a2]], input[[p, q, b2]], t2);
                let face = |p: usize| lerp(edge(p, a1), edge(p, b1), t1);
                out[[i, j, k]] = lerp(face(a0), face(b0), t0);
            }
        }
    }
    Ok(())
}

/// `a + t * (b - a)`, which returns `a` exactly where `a == b`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// The block edge on `axis` a lattice interpolation can be planned at, at or
/// below `wanted`.
///
/// **Any edge, now, and the change is the point.** This used to round `wanted`
/// down to a whole number of gaps, because a block's fetched sample count had to
/// be a function of the fine extent it wrote and back again — and it is exactly
/// that requirement which the two end blocks could not meet, since the volume's
/// extent is not in general a whole number of gaps. With the extent coming from
/// the block's placement rather than from its fetch there is nothing left to
/// invert, so the only bound is the axis itself.
///
/// `None` for an unevenly spread lattice, which neither op accepts at all.
pub fn interpolate_block_edge(
    lattice: &SampleLattice,
    axis: usize,
    wanted: usize,
) -> Option<usize> {
    lattice.uniform_gap(axis)?;
    Some(wanted.clamp(1, lattice.volume()[axis]))
}

/// The phase a lattice interpolation needs: the grid over the **fine** volume,
/// and each block's fetch region in the coarse level below.
///
/// `grid` is over the fine volume and **any block edge is allowed**. That is the
/// whole visible effect of migration step 4 here: this used to refuse anything
/// but a single block, because a block's fine extent had to be derivable from
/// its fetched sample count and the two end blocks broke the derivation. The
/// extent now comes from the block's [`Placement`], so the fetch can be the
/// samples the block actually brackets and the derivation is not needed at all.
///
/// Shipped with the op rather than written at the call site, for
/// [`lattice_statistic_phase`]'s reason: one derivation, used by the plan and by
/// the kernel, so a block cannot be handed a window the op does not read from.
///
/// [`Placement`]: crate::op::Placement
pub fn lattice_interpolate_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    op: &LatticeInterpolateOp,
    grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    let fine = op.lattice.volume();
    if grid.volume() != fine {
        return Err(Error::InvalidArgument(format!(
            "a lattice interpolation's grid is cut from what it writes: this lattice is laid over \
             {fine:?} and the grid is over {:?}.",
            grid.volume()
        )));
    }
    // The halo is nothing in the phase's own space, as it is for the statistic
    // half: every fine voxel is written by exactly one block, and what a block
    // needs beyond its own core is in the *coarse* level and is stated as a
    // fetch region rather than as a distance.
    let phase = PhaseDecomposition::derive(slots, names, lattice_reach(), Reach::none(), grid);
    let sources = phase
        .blocks
        .iter()
        .map(|block| op.fetch_region(&block.read))
        .collect::<Result<Vec<Region>>>()?;
    Ok(phase.with_sources(move |block| sources[block.flat].clone()))
}

/// Measured; see `super::COST_MEASUREMENT`. One trilinear blend per fine voxel,
/// which is the same work `ops::resample`'s linear pass does and is charged the
/// same.
pub(super) const INTERPOLATE_COST: f64 = 1.58;
