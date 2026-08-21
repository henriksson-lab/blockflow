// SPDX-License-Identifier: MIT
//
// Original work for this crate. A harness, not a feature.
//
// What this models, and why it is not a made-up shape
// ---------------------------------------------------
// Everything the framework has been proven against so far has one property:
// **input and output live in the same coordinate space**, and a block's read
// extent and its write extent are boxes of the same grid. This harness models a
// real operation that does not have that property, using only noop ops and an
// accounting loader, so that the walls it hits are framework walls rather than
// kernel walls.
//
// Three grids, two transitions:
//
// 1. **spatial -> patch grid.** A fixed-edge patch lattice is laid over the
//    spatial array; the output has explicit *patch-index* axes, so patch
//    `(j, i)` writes slot `[j, i, ..]`. Those slots are disjoint and tile the
//    patch-grid array exactly — **the overlap lives in the mapping between the
//    grids, not in overlapping writes.** The patch edge is fixed by the
//    operation; the planner does not get to choose it.
// 2. **patch grid -> spatial.** A bounded reach in *patch-index* units, with a
//    weighted combine over the patches that cover each output pixel. An
//    ordinary block op in an unusual coordinate space.
//
// The lattice is deliberately the awkward kind
// --------------------------------------------
// [`Lattice`] spreads its patches **evenly** across the extent rather than on a
// fixed stride: `n` patches, first at 0, last at `len - edge`, the rest
// interpolated. That is the shape the motivating operation actually has, and it
// matters here for one reason: *the patch positions are a function of the global
// extent*. Hand the same operation a block instead of the whole array and every
// patch moves, so the blending moves, so the output is decomposition-dependent —
// throughout the block, not only at its seams. `XY_BLOCK_SPLITTING.md` records
// the same defect class for an unanchored sample grid, including the part that
// matters most: **a halo does not fix it.**
//
// So this harness always builds its lattice from the *global* extent, and the
// tests assert that the framework has no way to check that it did.
//
// Everything here counts; nothing computes
// ----------------------------------------
// [`RegridEnvironment`] wraps `AccountingEnvironment`, so the real executor runs
// the real scheduler over a real task graph, and no pixel exists. What the
// wrapper adds is the coordinate mapping the framework cannot express: an image-0
// read arrives in the coordinates of the array being *written* and is served
// from an array in a different space, and the wrapper prices the fetch that
// implies.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use blockflow::env::EnvCounters;
use blockflow::{
    AccountingEnvironment, Anchor, BlockBuf, BlockGrid, BlockOp, Chain, Decomposition, Dtype,
    Environment, Error, Lifecycle, PhaseDecomposition, Placement, Reach, Region, Result, Voxels,
};
use ndarray::ArrayD;

// ------------------------------------------------------------- lattice --

/// A fixed-edge patch lattice over one axis, positioned from the **global**
/// extent.
///
/// `edge` is mandatory, not advisory: the operation this models accepts exactly
/// one input extent, so a decomposition that offers it anything else is not
/// merely slower, it is unrunnable. The framework has no vocabulary for that
/// today, which is one of the things the tests record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice {
    len: usize,
    edge: usize,
    starts: Vec<usize>,
}

impl Lattice {
    /// `count` patches spread evenly over `0 ..= len - edge`.
    ///
    /// The count rule is the application's, not this crate's — what the
    /// framework has to cope with is only that the *positions* depend on `len`.
    pub fn new(len: usize, edge: usize) -> Result<Self> {
        if edge == 0 || len < edge {
            return Err(Error::invalid(format!(
                "lattice: an extent of {len} cannot carry a fixed patch edge of {edge}"
            )));
        }
        let count = (2 * len).div_ceil(edge).max(2);
        let span = len - edge;
        let starts = (0..count)
            .map(|index| {
                // Evenly spread and rounded, so the step is fractional in
                // general: this is what makes the lattice a function of `len`
                // rather than a fixed stride anyone could re-derive locally.
                (index * span + (count - 1) / 2) / (count - 1)
            })
            .collect();
        Ok(Self { len, edge, starts })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn edge(&self) -> usize {
        self.edge
    }

    pub fn count(&self) -> usize {
        self.starts.len()
    }

    pub fn start(&self, patch: usize) -> usize {
        self.starts[patch]
    }

    /// The spatial half-open span patch `patch` covers.
    pub fn span(&self, patch: usize) -> (usize, usize) {
        let start = self.starts[patch];
        (start, start + self.edge)
    }

    /// Inclusive patch-index range covering spatial position `at`.
    pub fn patches_covering(&self, at: usize) -> (usize, usize) {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for (index, &start) in self.starts.iter().enumerate() {
            if start <= at && at < start + self.edge {
                lo = lo.min(index);
                hi = hi.max(index);
            }
        }
        (lo, hi)
    }

    /// Inclusive patch-index range covering the spatial half-open range.
    pub fn patches_over(&self, lo: usize, hi: usize) -> (usize, usize) {
        let (first, _) = self.patches_covering(lo);
        let (_, last) = self.patches_covering(hi.saturating_sub(1).min(self.len - 1));
        (first, last)
    }

    /// The spatial union of a patch-index range.
    pub fn union_span(&self, first: usize, last: usize) -> (usize, usize) {
        (self.starts[first], self.starts[last] + self.edge)
    }

    /// **Reach in patch-index units**: the largest number of patch indices
    /// separating two patches that cover the same output position.
    ///
    /// This is the quantity a combine op has to declare, and it is bounded and
    /// small by construction. Whether it can be *stated* through
    /// `BlockOp::reach(axis, volume_len)` depends entirely on which array the
    /// phase's grid is over; see the tests.
    pub fn index_reach(&self) -> usize {
        (0..self.len)
            .map(|at| {
                let (lo, hi) = self.patches_covering(at);
                hi - lo
            })
            .max()
            .unwrap_or(0)
    }

    /// Every spatial position is covered by at least one patch.
    pub fn covers_everything(&self) -> bool {
        (0..self.len).all(|at| {
            let (lo, _) = self.patches_covering(at);
            lo != usize::MAX
        })
    }
}

/// The whole geometry of the model: one spatial volume, one patch lattice per
/// split axis, and the size of what each patch carries.
///
/// The spatial volume is 3-D with a **z extent of 1**, which is this project's
/// standing spelling of the 2-D case: a degenerate axis, not a second code path.
#[derive(Debug, Clone)]
pub struct PatchGeometry {
    pub volume: [usize; 3],
    pub lattice: [Lattice; 2],
    /// Classes the operation emits per output position.
    pub classes: usize,
}

impl PatchGeometry {
    pub fn new(volume: [usize; 3], edge: usize, classes: usize) -> Result<Self> {
        Ok(Self {
            volume,
            lattice: [
                Lattice::new(volume[1], edge)?,
                Lattice::new(volume[2], edge)?,
            ],
            classes,
        })
    }

    /// What one patch carries: `classes * edge * edge`, times the (degenerate)
    /// z extent.
    ///
    /// **This is a folding, and the folding is forced.** The natural rank of the
    /// patch-grid array is six — `(z, ty, tx, classes, edge, edge)` — and every
    /// shape in the framework's geometry is `[usize; 3]`. Two of the three
    /// available axes are spent on the patch indices, which are the axes that
    /// are split and reached over; the payload takes the third. In the 2-D case
    /// that fits exactly, with nothing left over for z.
    pub fn payload(&self) -> usize {
        self.volume[0] * self.classes * self.lattice[0].edge() * self.lattice[1].edge()
    }

    /// The patch-grid array as the framework must see it: `[ty, tx, payload]`.
    pub fn patch_volume(&self) -> [usize; 3] {
        [
            self.lattice[0].count(),
            self.lattice[1].count(),
            self.payload(),
        ]
    }

    /// The bounded reach of the combine, in patch-index units, per axis of the
    /// patch-grid array.
    pub fn patch_reach(&self) -> [usize; 3] {
        [
            self.lattice[0].index_reach(),
            self.lattice[1].index_reach(),
            0,
        ]
    }

    /// The same bounded reach restated in **spatial** units, which is the only
    /// unit available to an op whose phase is anchored to the spatial array.
    ///
    /// It is an over-declaration in two independent ways, and both are recorded
    /// by the tests: the true dependency is asymmetric (a pixel needs patches
    /// starting up to `edge - 1` *before* it and none after), while `reach` is
    /// one number applied to both sides; and a patch index converted to voxels
    /// is the coarsest possible statement of a dependency that is exact in the
    /// index space.
    pub fn spatial_reach(&self) -> [usize; 3] {
        [
            0,
            self.lattice[0].edge().saturating_sub(1),
            self.lattice[1].edge().saturating_sub(1),
        ]
    }
}

// --------------------------------------------------------------- the ops --

/// A noop op with a declared reach, cost and traversal preference.
///
/// `blockflow::IdentityOp` would do everything this does except that its
/// `apply` refuses an array that is not rank 3, and the harness never allocates
/// one at all. It is repeated here so the harness owns its own probes and the
/// crate's are left alone.
pub struct CountedOp {
    name: &'static str,
    reach: [usize; 3],
    /// What the op really depends on, when that is something the symmetric
    /// triple above cannot say. `reach` stays a bound on it, which the chain
    /// checks.
    spec: Option<Reach>,
    cost: f64,
    order: Option<[usize; 3]>,
}

impl CountedOp {
    pub fn new(name: &'static str, reach: [usize; 3]) -> Self {
        Self {
            name,
            reach,
            spec: None,
            cost: 1.0,
            order: None,
        }
    }

    /// State the dependency exactly, rather than as the widest side of it.
    pub fn with_reach(mut self, spec: Reach) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_order(mut self, order: [usize; 3]) -> Self {
        self.order = Some(order);
        self
    }
}

impl BlockOp for CountedOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach[axis]
    }

    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        let _ = volume;
        match &self.spec {
            Some(spec) => spec.clone(),
            None => Reach::symmetric(self.reach),
        }
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.order
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ------------------------------------------------------- the environment --

/// Which array an image-0 read is really served from.
///
/// The executor asks for a region in the coordinates of the array it is
/// *writing*. When the input lives in another grid, something has to map. The
/// framework offers nowhere to put that mapping, so it goes here — which is the
/// finding, not the fix.
pub enum Inbound {
    /// Nothing to map: input and output share a grid.
    SameGrid,
    /// Reads arrive in patch-index coordinates, served from the spatial array.
    SpatialUnderPatches(PatchGeometry),
    /// Reads arrive in spatial coordinates, served from the patch-grid array.
    PatchesUnderSpatial(PatchGeometry),
}

/// One output beyond the workflow's own region output.
///
/// `Workflow` names a single output of a single dtype, and `Environment::write`
/// writes one buffer to one image, so an operation with several outputs of
/// differing dtype and rank has to declare them somewhere the framework cannot
/// see. Here, that is this list — and the tests measure how far the framework's
/// own byte accounting is out as a result.
#[derive(Debug, Clone)]
pub struct ExtraOutput {
    pub name: &'static str,
    /// Elements written per voxel of the block's valid region.
    pub elements_per_valid_voxel: f64,
    pub dtype: Dtype,
    /// Rank of the array as it would really be stored, before folding.
    pub rank: usize,
}

/// The accounting loader, plus the coordinate mapping and the extra outputs.
pub struct RegridEnvironment {
    inner: AccountingEnvironment,
    inbound: Inbound,
    extras: Vec<ExtraOutput>,
    /// Bytes per block of per-block, non-pixel output. Zero disables it.
    sidecar_bytes: usize,
    sidecar_stream: &'static str,
    final_image: usize,
    /// The grid whose block indices the sidecar and the touch log are keyed by.
    write_grid: BlockGrid,
    foreign_reads: AtomicU64,
    /// Elements fetched from the other grid as one box — what an IO layer moves.
    foreign_union_elements: AtomicU64,
    /// Elements the patches actually carry — what the operation consumes, which
    /// exceeds the union because the patches overlap.
    foreign_gathered_elements: AtomicU64,
    extra_output_bytes: AtomicU64,
    touched: Mutex<BTreeSet<[usize; 3]>>,
}

impl RegridEnvironment {
    pub fn new(volume: [usize; 3], chunk: [usize; 3], dtype: Dtype, write_grid: BlockGrid) -> Self {
        Self {
            inner: AccountingEnvironment::new(volume, chunk, dtype.size_of() as u64),
            inbound: Inbound::SameGrid,
            extras: Vec::new(),
            sidecar_bytes: 0,
            sidecar_stream: "block_summary",
            final_image: 1,
            write_grid,
            foreign_reads: AtomicU64::new(0),
            foreign_union_elements: AtomicU64::new(0),
            foreign_gathered_elements: AtomicU64::new(0),
            extra_output_bytes: AtomicU64::new(0),
            touched: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn with_inbound(mut self, inbound: Inbound) -> Self {
        self.inbound = inbound;
        self
    }

    pub fn with_extras(mut self, extras: Vec<ExtraOutput>) -> Self {
        self.extras = extras;
        self
    }

    /// Emit `bytes` of per-block, non-pixel output for every block written.
    pub fn with_sidecar(mut self, stream: &'static str, bytes: usize) -> Result<Self> {
        self.sidecar_stream = stream;
        self.sidecar_bytes = bytes;
        self.inner
            .declare_sidecar(stream, Lifecycle::DeleteOnExit)?;
        Ok(self)
    }

    pub fn with_final_image(mut self, image: usize) -> Self {
        self.final_image = image;
        self
    }

    pub fn foreign_reads(&self) -> u64 {
        self.foreign_reads.load(Ordering::SeqCst)
    }

    pub fn foreign_union_elements(&self) -> u64 {
        self.foreign_union_elements.load(Ordering::SeqCst)
    }

    pub fn foreign_gathered_elements(&self) -> u64 {
        self.foreign_gathered_elements.load(Ordering::SeqCst)
    }

    pub fn extra_output_bytes(&self) -> u64 {
        self.extra_output_bytes.load(Ordering::SeqCst)
    }

    /// Block indices this environment saw a write for, in index order.
    pub fn touched_blocks(&self) -> Vec<[usize; 3]> {
        self.touched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .copied()
            .collect()
    }

    /// Which block of `write_grid` a valid region belongs to.
    ///
    /// `Environment::write` is handed regions and no block index, so an
    /// environment that must key anything by block has to invert the geometry
    /// itself. Exact here because the write grid is regular; an environment
    /// whose blocks were not a lattice could not do this at all.
    fn block_of(&self, valid: &Region) -> [usize; 3] {
        let edge = self.write_grid.block();
        [
            valid.start[0] / edge[0],
            valid.start[1] / edge[1],
            valid.start[2] / edge[2],
        ]
    }

    /// Elements the other grid must yield to serve `region`, as
    /// `(union box, gathered)`.
    fn foreign_extent(&self, region: &Region) -> Option<(u64, u64)> {
        match &self.inbound {
            Inbound::SameGrid => None,
            Inbound::SpatialUnderPatches(geometry) => {
                // `region` indexes patches; the fetch is spatial.
                let mut union = geometry.volume[0] as u64;
                let mut gathered = geometry.volume[0] as u64;
                for axis in 0..2 {
                    let lattice = &geometry.lattice[axis];
                    let first = region.start[axis];
                    let last = first + region.shape[axis] - 1;
                    let (lo, hi) = lattice.union_span(first, last);
                    union *= (hi - lo) as u64;
                    gathered *= (region.shape[axis] * lattice.edge()) as u64;
                }
                Some((union, gathered))
            }
            Inbound::PatchesUnderSpatial(geometry) => {
                // `region` is spatial; the fetch is whole patch slots.
                let mut patches = 1u64;
                for axis in 0..2 {
                    let lattice = &geometry.lattice[axis];
                    let start = region.start[axis + 1];
                    let end = start + region.shape[axis + 1];
                    let (first, last) = lattice.patches_over(start, end);
                    patches *= (last - first + 1) as u64;
                }
                let elements = patches * geometry.payload() as u64;
                Some((elements, elements))
            }
        }
    }
}

impl Environment for RegridEnvironment {
    fn volume(&self) -> [usize; 3] {
        self.inner.volume()
    }

    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        self.inner.prepare(decomposition)
    }

    fn read(&self, image: usize, region: &Region) -> Result<BlockBuf> {
        if image == 0 {
            if let Some((union, gathered)) = self.foreign_extent(region) {
                self.foreign_reads.fetch_add(1, Ordering::SeqCst);
                self.foreign_union_elements
                    .fetch_add(union, Ordering::SeqCst);
                self.foreign_gathered_elements
                    .fetch_add(gathered, Ordering::SeqCst);
            }
        }
        self.inner.read(image, region)
    }

    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf> {
        self.inner.apply(slot, input, sources, at)
    }

    fn write(&self, image: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        self.inner.write(image, within, valid, buf)?;
        if valid.voxels() == 0 {
            return Ok(());
        }
        let index = self.block_of(valid);
        self.touched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(index);
        if image == self.final_image {
            for extra in &self.extras {
                let elements = valid.voxels() as f64 * extra.elements_per_valid_voxel;
                self.extra_output_bytes.fetch_add(
                    (elements * extra.dtype.size_of() as f64) as u64,
                    Ordering::SeqCst,
                );
            }
            if self.sidecar_bytes > 0 {
                // The op cannot do this: `BlockOp::apply` is handed two arrays
                // and an anchor and never sees the environment. So the fragment
                // is written from the only place that has both a block and a
                // store, which is the environment's own `write`.
                let payload = vec![0u8; self.sidecar_bytes];
                self.write_sidecar(self.sidecar_stream, image - 1, index, &payload)?;
            }
        }
        Ok(())
    }

    fn uniform(&self, buf: &BlockBuf) -> Option<f64> {
        self.inner.uniform(buf)
    }

    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf> {
        self.inner.constant(dtype, region, value)
    }

    fn release(&self, buf: &BlockBuf) {
        self.inner.release(buf)
    }

    fn finish(&self, image: usize) -> Result<()> {
        self.inner.finish(image)
    }

    fn counters(&self) -> &EnvCounters {
        self.inner.counters()
    }

    fn chunk_shape(&self) -> [usize; 3] {
        self.inner.chunk_shape()
    }

    fn sidecars(&self) -> Option<&blockflow::Sidecars> {
        self.inner.sidecars()
    }
}

// ------------------------------------------------------- plan assembly --

/// A one-phase decomposition over an explicit grid.
///
/// Written by hand rather than obtained from a `Strategy`, because no shipped
/// strategy can be *made* to produce a mandated grid: `Constraints` offers a
/// list of scalar block edges and the strategy picks whichever prices best. See
/// the tests.
pub fn one_phase(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: BlockGrid,
    reach: impl Into<Reach>,
    halo: impl Into<Reach>,
) -> Decomposition {
    let slots = chain.slots();
    let names = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = reach.into();
    let chain_reach = reach.bound(volume);
    Decomposition {
        volume,
        dtype,
        phases: vec![PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names,
            reach,
            halo,
            grid,
        )],
        chain_reach,
    }
}

// ------------------------------------------------------- a bespoke plan --

/// A `Strategy` that produces the grid the operation mandates.
///
/// The good news half of the "the planner may choose a shape the op cannot
/// accept" finding: the `Strategy` seam is wide enough to express a mandated
/// grid, because `decompose` returns a `Decomposition` rather than a choice from
/// a menu. What it cannot do is *learn* the grid — no `BlockOp` method reports
/// one, so this is told the lattice at construction, out of band from the chain
/// it is planning for.
pub struct PatchStrategy {
    geometry: PatchGeometry,
}

impl PatchStrategy {
    pub fn new(geometry: PatchGeometry) -> Self {
        Self { geometry }
    }
}

impl blockflow::Strategy for PatchStrategy {
    fn name(&self) -> &'static str {
        "patch"
    }

    fn decompose(
        &self,
        workflow: &blockflow::Workflow,
        _constraints: &blockflow::Constraints,
    ) -> Result<Decomposition> {
        let volume = self.geometry.patch_volume();
        if workflow.shape != volume {
            return Err(Error::invalid(format!(
                "this plan is for the patch grid {volume:?}, not {:?}",
                workflow.shape
            )));
        }
        let grid = BlockGrid::new(volume, [1, 1, self.geometry.payload()])?;
        let reach = workflow.chain.reach3(&volume);
        let decomposition = one_phase(&workflow.chain, volume, workflow.dtype, grid, reach, reach);
        decomposition.check()?;
        Ok(decomposition)
    }
}
