// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// This is the **binding** half of `docs/design/BLOCK_OPS.md` §"The planning
// problem is NP-hard". A `Decomposition` is everything that is parity-visible —
// block sizes per phase, halos, valid regions, seams, and the partition of the
// chain into phases. Changing any of it changes output: the reference pipeline's
// differs by two voxels across pool sizes, localised to a block seam. So it is
// decided **statically, deterministically, from shape and dtype only**, and it
// is hashable so a run can record which decomposition produced its numbers.
//
// Everything else — visit order, concurrency, prefetch depth, placement,
// whether a phase boundary lands in memory or in storage — is in `Hints`, is
// advisory, and is allowed to be greedy and data-dependent. That split is the
// contract; see `strategy.rs`.
//
// What this file is careful *not* to do
// -------------------------------------
// It does not look at data. It cannot: nothing here takes a source. A
// decomposition that peeked at content would seam differently on two datasets
// and no parity figure would transfer between them.
//
// Levels, and why a volume is per phase
// -------------------------------------
// Level 0 is the input; level `p+1` is what phase `p` wrote. Each phase owns
// the volume its lattice is cut from — `PhaseDecomposition::volume` — and reads
// the level below, whose shape is `Decomposition::volume_at(p)`. The two are
// equal for every phase whose output grid is its input grid, which is most of
// them, and the plan is byte-identical there. Where they differ the plan *says
// so*, which is the whole point: a phase that changes shape used to be refused
// by `check`, so the only way to run one was to hide the mapping inside an
// `Environment`, where nothing prices it and nothing checks it.
//
// The guard
// ---------
// `Decomposition::check` runs `block_processing::boxes_tile_exactly` over
// the derived valid regions. That is the *existing* check, unchanged, pointed
// at the right quantity — the design is explicit that a new bespoke halo
// assertion is the wrong answer, because the tiling check already runs, already
// has a message and cannot be forgotten at a new call site. The only change to
// `block_processing.rs` is that the function is now `pub(crate)`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::reach::Reach;
use crate::region::Region;
use crate::tiling::boxes_tile_exactly;

use super::geometry::{region_within, BlockGeometry, BlockGrid};
use super::op::{BlockConstraint, Chain};

/// One fused run of slots: read once with a halo sized to **this phase's**
/// reach, apply the slots back to back, write the valid region.
///
/// That the halo is per-phase rather than per-chain is the whole reason cutting
/// can pay. If every phase still carried the entire chain's reach, a cut would
/// add a materialisation and save nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseDecomposition {
    /// Indices into `chain.slots()`, contiguous and in chain order.
    pub slots: Vec<usize>,
    pub names: Vec<String>,
    /// Sum of the slots' reaches: what must be shrunk off `read` to get `valid`.
    ///
    /// A [`Reach`] rather than a triple, so that the *required* dependency can be
    /// one-sided, per-block, whole-axis or stated in another coordinate space.
    /// `[usize; 3]` converts into one and compares equal to one, so every caller
    /// that means the symmetric form still says it that way.
    pub reach: Reach,
    /// What to actually fetch. A **hint** in the design's sense — the planner
    /// sets it equal to `reach`, and it lives in the binding half only because
    /// a wrong value must be *detectable*, which requires it to be recorded.
    ///
    /// The same type as `reach` because it is the same shape of quantity — how
    /// far outside its core a block reaches, per side, per axis — and the guard
    /// this crate is built on is the comparison between the two. It is *not* the
    /// same number: a granted halo may be wider, and where an operation mandates
    /// an input extent it must differ per block ([`Reach::window`]), which is
    /// what lets "the extent I accept" and "the extent I need around it" stop
    /// being one number.
    pub halo: Reach,
    /// Per-phase block grid. A phase boundary is already a materialisation, so
    /// re-blocking there is free.
    ///
    /// **This grid is the phase's own volume**, and no longer has to be the
    /// decomposition's: see [`PhaseDecomposition::volume`].
    pub grid: BlockGrid,
    /// The element type this phase **writes**, when it is not the one it read.
    ///
    /// `None` means "unchanged", which is every phase this crate shipped before
    /// the field existed, and is why `derive` still takes five arguments — the
    /// alternative was a default element type, and a default is exactly the kind
    /// of plausible wrong answer a binding plan must not contain. Resolve it
    /// with [`Decomposition::dtype_at`], which folds the chain from level 0.
    pub dtype: Option<Dtype>,
    /// Levels this phase reads **besides** the one it is handed, ascending and
    /// without repeats: one per [`Chain::Source`] leaf in its slots.
    ///
    /// **Two different things in this file are called a source, and this is the
    /// one that is a level.** `BlockGeometry::source` is a *region* — where in
    /// the level below a block fetches from — and every phase has one per
    /// block. This is a list of *levels*, and it is empty for every phase that
    /// does not read a second array. They meet in exactly one place: a source
    /// level is read at each block's `source` region, because a source leaf has
    /// reach 0 and therefore reads what the phase already fetches.
    ///
    /// **In the binding half of the plan, unlike `Visibility`.** Which level an
    /// arm reads changes voxels, so it is recorded, fingerprinted and shipped —
    /// the "explicit edges in the binding plan" of `docs/design/BLOCK_OPS.md`
    /// §"Levels are a DAG", in the one shape this crate needs them.
    ///
    /// Derived from the chain by [`Decomposition::declare_source_levels`] and
    /// verified against it by [`check_source_levels`], the same split
    /// `declare_dtypes` and `check_dtypes` have.
    pub source_levels: Vec<usize>,
    /// Cores, read extents and valid regions, derived and recorded.
    pub blocks: Vec<BlockGeometry>,
}

impl PhaseDecomposition {
    /// Derive the geometry for one phase.
    ///
    /// `reach` and `halo` are anything that converts into a [`Reach`], which a
    /// symmetric `[usize; 3]` does — so this keeps the arity and the spelling it
    /// has always had, and a caller with something richer to say passes the
    /// richer value.
    ///
    /// **This is where a coordinate space becomes a distance.** A reach in whole
    /// blocks, or in a permuted axis order, is symbolic until a grid exists: the
    /// planner compares candidate grids, and a reach that changed with the grid
    /// under consideration could not be compared against anything. So the
    /// conversion happens exactly here, once the grid is settled, and the
    /// geometry below sees voxels in the geometry's own axis order.
    pub fn derive(
        slots: Vec<usize>,
        names: Vec<String>,
        reach: impl Into<Reach>,
        halo: impl Into<Reach>,
        grid: BlockGrid,
    ) -> Self {
        let volume = grid.volume();
        let reach = reach.into();
        let halo = halo.into();
        let reach_voxels = reach.in_voxels(grid.block());
        let halo_voxels = halo.in_voxels(grid.block());
        let blocks = grid
            .cores()
            .iter()
            .map(|core| BlockGeometry::derive_with(core, volume, &halo_voxels, &reach_voxels))
            .collect();
        Self {
            slots,
            names,
            reach,
            halo,
            grid,
            dtype: None,
            source_levels: Vec::new(),
            blocks,
        }
    }

    /// The coordinate space **this phase** works in: what its cores are cut
    /// from, what its valid regions must tile, and the shape of the level it
    /// writes.
    ///
    /// It is derived from the grid rather than stored beside it, because two
    /// copies of one number are two numbers that can disagree, and a plan that
    /// disagrees with itself is worse than a plan that cannot express something.
    ///
    /// What it is *not* is the shape of what the phase reads. That is the level
    /// below — `Decomposition::volume_at(phase)` — and the two differ exactly
    /// when the phase changes the array's shape.
    pub fn volume(&self) -> [usize; 3] {
        self.grid.volume()
    }

    /// Say this phase writes a different element type than it read.
    pub fn with_dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Say this phase also reads these levels, through source leaves.
    ///
    /// Normalised on the way in — ascending, no repeats — because the list is
    /// fingerprinted and two plans that read the same levels must hash the same
    /// whatever order the chain was walked in.
    pub fn with_source_levels(mut self, levels: impl IntoIterator<Item = usize>) -> Self {
        let mut levels: Vec<usize> = levels.into_iter().collect();
        levels.sort_unstable();
        levels.dedup();
        self.source_levels = levels;
        self
    }

    /// Give every block a fetch region in the level below's coordinate space.
    ///
    /// `map` is handed each block's geometry — index, core, read extent and all
    /// — and returns the region to fetch for it. It must be a function of the
    /// **block index and the plan**, never of the data: a `Decomposition` is
    /// parity-visible and is built without a source to look at.
    ///
    /// The mapping is applied once, here, and what is recorded is its result, so
    /// the plan carries the regions rather than a closure nobody can hash. What
    /// checks them is [`Decomposition::check`], which is the only place that
    /// knows what volume the level below has.
    pub fn with_sources(mut self, map: impl Fn(&BlockGeometry) -> Region) -> Self {
        self.blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                let source = map(&block);
                block.with_source(source)
            })
            .collect();
        self
    }

    /// Whether any block of this phase fetches something other than its read
    /// extent.
    pub fn reads_across_grids(&self) -> bool {
        self.blocks.iter().any(|block| block.reads_across_grids())
    }

    /// Whether every block's valid region covers its whole core.
    ///
    /// Diagnostic only. The guard is `Decomposition::check`.
    pub fn blocks_missing_valid_core(&self) -> Vec<[usize; 3]> {
        self.blocks
            .iter()
            .filter(|block| !block.valid_covers_core())
            .map(|block| block.index)
            .collect()
    }
}

/// Whether a level survives the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Level 0 and the workflow output. Somebody outside the run reads these,
    /// so they exist when it ends.
    Published,
    /// Written by one phase, read by exactly one phase, then dead.
    ///
    /// The reason this is worth naming: today every level of an `N`-phase plan
    /// is allocated at full volume for the whole run, so a twenty-stage chain
    /// holds twenty-one copies of the data at once. Only ever two of them are
    /// live. Saying which are which is what lets the environment free the rest.
    Internal,
}

/// The binding plan: what must be reproduced exactly for output to match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decomposition {
    /// The shape of **level 0** — the input the first phase reads.
    ///
    /// Not "the volume of the plan": every phase owns its own volume
    /// ([`PhaseDecomposition::volume`]), and level `p+1` is phase `p`'s. Level 0
    /// is the one level that is no phase's output, so it is the one that has to
    /// be stated here. [`Decomposition::volume_at`] is the accessor that reads
    /// either kind, and [`Decomposition::uniform_volume`] is the derived
    /// "everything agrees" answer the single-volume callers want.
    pub volume: [usize; 3],
    /// The element type of **level 0**, on the same argument. A phase that
    /// changes it says so in its own `dtype`; [`Decomposition::dtype_at`] folds
    /// the chain.
    pub dtype: Dtype,
    pub phases: Vec<PhaseDecomposition>,
    /// Reach of the whole chain. Equal to the sum of the phases' reaches: a
    /// phase split does not reduce the total reach, it stops every phase from
    /// paying it.
    pub chain_reach: [usize; 3],
}

impl Decomposition {
    /// Slot indices in execution order, phase by phase. This must equal
    /// `(0..n_slots)` — a decomposition may partition the chain but never
    /// reorder or drop an op.
    pub fn slot_order(&self) -> Vec<usize> {
        self.phases
            .iter()
            .flat_map(|phase| phase.slots.iter().copied())
            .collect()
    }

    pub fn op_names_in_order(&self) -> Vec<String> {
        self.phases
            .iter()
            .flat_map(|phase| phase.names.iter().cloned())
            .collect()
    }

    pub fn n_phases(&self) -> usize {
        self.phases.len()
    }

    pub fn n_tasks(&self) -> usize {
        self.phases.iter().map(|phase| phase.blocks.len()).sum()
    }

    /// The number of levels: level 0 plus one per phase.
    pub fn n_levels(&self) -> usize {
        self.phases.len() + 1
    }

    /// The shape of level `level`: level 0 is the input, level `p+1` is what
    /// phase `p` wrote.
    ///
    /// This is the accessor a caller that used to read `decomposition.volume`
    /// and meant "the space this phase reads" wants. It panics on a level that
    /// does not exist, like an index, because every caller of it holds a phase
    /// number it got from the plan.
    pub fn volume_at(&self, level: usize) -> [usize; 3] {
        match level {
            0 => self.volume,
            _ => self.phases[level - 1].volume(),
        }
    }

    /// The element type of level `level`, folded from level 0: a phase that
    /// declares no `dtype` hands on the one it read.
    pub fn dtype_at(&self, level: usize) -> Dtype {
        let mut dtype = self.dtype;
        for phase in self.phases.iter().take(level) {
            dtype = phase.dtype.unwrap_or(dtype);
        }
        dtype
    }

    /// Every phase that reads `level`, ascending: the phase it is the input of,
    /// plus every later phase naming it in `source_levels`.
    ///
    /// **The refcount.** Before source leaves existed this was always a single
    /// phase — level `p` is phase `p`'s input and nobody else's — and the whole
    /// lifetime rule was written to that special case. A source leaf is a
    /// second reader, so the general statement is the one the design record
    /// asks for: *a level dies after its last reader*. With no source leaf
    /// anywhere this returns `vec![level]` for every level a phase reads, and
    /// [`Self::levels_dead_after`] reduces to exactly what it replaced.
    ///
    /// The last level has no reader at all, which is why it is `Published`.
    pub fn readers_of_level(&self, level: usize) -> Vec<usize> {
        let mut readers = Vec::new();
        if level < self.n_phases() {
            readers.push(level);
        }
        for (phase, decomposition) in self.phases.iter().enumerate() {
            if phase != level && decomposition.source_levels.contains(&level) {
                readers.push(phase);
            }
        }
        readers.sort_unstable();
        readers
    }

    /// The levels whose **last** reader is `phase`: what dies when this phase
    /// finishes every one of its tasks.
    ///
    /// This is the quantity the executor wants, and stating it this way is what
    /// keeps the executor from having to know whether the rule is "one reader"
    /// or "several". A plan with no source leaf answers `[phase]`, which is the
    /// level the phase read — the old rule, unchanged, as an instance of the
    /// new one.
    ///
    /// Whether a level may be freed at all is still [`Self::level_visibility`]'s
    /// question, and pinning is still the caller's; neither is folded in here,
    /// because this is a fact about the plan and those two are policy.
    pub fn levels_dead_after(&self, phase: usize) -> Vec<usize> {
        (0..self.n_levels())
            .filter(|&level| self.readers_of_level(level).last() == Some(&phase))
            .collect()
    }

    /// Whether a level survives the run, or exists only to get from one phase
    /// to the next.
    ///
    /// **Derived, not recorded.** Level 0 is the input and the last level is the
    /// output; everything between them is written by one phase, read by at least
    /// one phase, and then dead. Nothing in the plan needs to say so, and a
    /// field that could disagree with the arithmetic would be a field that
    /// eventually does.
    ///
    /// *Which* phase is the last reader is [`Self::levels_dead_after`]'s
    /// question and has moved; whether the level is somebody else's to keep has
    /// not, because a source leaf is inside the run and cannot make an
    /// intermediate outlive it.
    ///
    /// This is deliberately **not** part of the binding half of the plan.
    /// Discarding an intermediate cannot change a voxel of the output, so
    /// keeping one is a decision about debuggability rather than about the
    /// answer — it belongs in `Hints`, next to every other advisory value, and
    /// the fingerprint is unchanged by it.
    pub fn level_visibility(&self, level: usize) -> Visibility {
        if level == 0 || level + 1 >= self.n_levels() {
            Visibility::Published
        } else {
            Visibility::Internal
        }
    }

    /// The shape of the workflow's output: the last phase's volume.
    pub fn output_volume(&self) -> [usize; 3] {
        self.volume_at(self.n_phases())
    }

    /// The one volume every level is in, when they are all the same.
    ///
    /// **The derived form of what used to be a field.** `None` says the plan
    /// changes shape somewhere, which is precisely the question a caller that
    /// assumed a single volume — an environment holding one array, a chunk grid,
    /// a wire format that sends one triple — has to answer before it can do its
    /// job. Making it `Option` puts that answer at the call site instead of in a
    /// check that refused every such plan for everyone.
    pub fn uniform_volume(&self) -> Option<[usize; 3]> {
        self.phases
            .iter()
            .all(|phase| phase.volume() == self.volume)
            .then_some(self.volume)
    }

    /// The one element type every level is in, when they are all the same.
    pub fn uniform_dtype(&self) -> Option<Dtype> {
        self.phases
            .iter()
            .all(|phase| phase.dtype.is_none_or(|dtype| dtype == self.dtype))
            .then_some(self.dtype)
    }

    /// Record the element type each phase writes, from the ops that write it.
    ///
    /// Consulted by a strategy at the end of `decompose`, so that a shipped
    /// planner produces a plan whose levels are the width its chain needs rather
    /// than one `check_dtypes` will refuse. Separate from `check_dtypes` because
    /// the two answer different questions — this one *derives*, that one
    /// *verifies* a plan that may have come from anywhere.
    ///
    /// A phase whose ops hand the type on unchanged is left declaring nothing,
    /// which is what keeps a plan that does not use the feature fingerprinting
    /// exactly as it did before the feature existed.
    pub fn declare_dtypes(&mut self, chain: &Chain) -> Result<()> {
        let slots = chain.slots();
        let mut current = self.dtype;
        let mut read = self.dtype;
        for phase in &mut self.phases {
            if phase.slots.iter().any(|&slot| slot >= slots.len()) {
                continue;
            }
            for &slot in &phase.slots {
                current = slots[slot].produces(current)?;
            }
            phase.dtype = (current != read).then_some(current);
            read = current;
        }
        Ok(())
    }

    /// Record which levels each phase reads besides its own input, from the
    /// source leaves in its slots.
    ///
    /// The counterpart of [`Self::declare_dtypes`], and separate from
    /// [`check_source_levels`] for the same reason: this one *derives* from a
    /// chain the caller is holding, that one *verifies* a plan that may have
    /// arrived from anywhere.
    ///
    /// A phase with no source leaf is left declaring nothing, which is what
    /// keeps a plan that does not use the feature fingerprinting exactly as it
    /// did before the feature existed.
    pub fn declare_source_levels(&mut self, chain: &Chain) -> Result<()> {
        let slots = chain.slots();
        for phase in &mut self.phases {
            let mut levels = Vec::new();
            for &slot in &phase.slots {
                let Some(node) = slots.get(slot) else {
                    continue;
                };
                levels.extend(node.source_levels());
            }
            levels.sort_unstable();
            levels.dedup();
            phase.source_levels = levels;
        }
        Ok(())
    }

    /// Exact voxels read, per phase, from the real clamped geometry.
    ///
    /// This is the number to compare against what a run actually counted. A
    /// cost model is for *choosing*; this is for checking the choice was
    /// described honestly — "a plan predicting N reads against an execution
    /// performing 3N is a bug worth surfacing".
    ///
    /// It counts `source`, not `read`, because `source` is what the environment
    /// is asked for. Where a phase reads across grids the two differ, and the
    /// whole reason `source` is in the plan is that this figure was silently
    /// wrong by 4x when the mapping lived in an environment instead.
    ///
    /// **Once per level read, and a phase with source leaves reads more than
    /// one.** Each one is fetched at the same region as the input — reach 0 —
    /// so the multiplier is exactly `1 + source_levels.len()`. A figure that
    /// counted only the input would be under by that factor for precisely the
    /// plans this number is most worth checking, and it is compared against a
    /// run's counter to the voxel.
    pub fn exact_read_voxels(&self) -> Vec<usize> {
        self.phases
            .iter()
            .map(|phase| {
                let per_level: usize = phase.blocks.iter().map(|block| block.source.voxels()).sum();
                per_level * (1 + phase.source_levels.len())
            })
            .collect()
    }

    /// A stable identifier, for the manifest.
    ///
    /// Deterministic across runs and processes: it hashes only integers and
    /// `&'static str`, never a pointer, a `HashMap` iteration order or an
    /// `f64`. Two runs with the same decomposition must produce the same value
    /// or the manifest cannot be used to compare them.
    ///
    /// **Per-phase volumes and element types were already covered**: the grid's
    /// volume has always been hashed per phase, so a phase that changes shape
    /// changes the fingerprint without anything being added here. What is new is
    /// hashed *only where it is used* — a phase that declares no `dtype` and a
    /// block that reads its own read extent contribute nothing — so a plan that
    /// does not use the new expressiveness fingerprints exactly as it did before
    /// the new expressiveness existed. That is not tidiness: a fingerprint is
    /// how a parity figure is attached to the plan that produced it, and
    /// renumbering every historical plan to record that two features are unused
    /// would throw that away for nothing.
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.volume.hash(&mut hasher);
        self.dtype.numpy_name().hash(&mut hasher);
        self.chain_reach.hash(&mut hasher);
        self.phases.len().hash(&mut hasher);
        for phase in &self.phases {
            phase.slots.hash(&mut hasher);
            phase.names.hash(&mut hasher);
            phase.reach.hash(&mut hasher);
            phase.halo.hash(&mut hasher);
            phase.grid.volume().hash(&mut hasher);
            phase.grid.block().hash(&mut hasher);
            if let Some(dtype) = phase.dtype {
                dtype.numpy_name().hash(&mut hasher);
            }
            // Hashed only where it is used, on exactly `dtype`'s argument: a
            // phase that reads no second level contributes nothing, so every
            // plan built before source leaves existed fingerprints as it did.
            // Which level an arm reads changes voxels, so a plan that uses one
            // must not collide with a plan that reads another.
            if !phase.source_levels.is_empty() {
                phase.source_levels.hash(&mut hasher);
            }
            for block in &phase.blocks {
                block.index.hash(&mut hasher);
                block.core.start.hash(&mut hasher);
                block.core.shape.hash(&mut hasher);
                block.read.start.hash(&mut hasher);
                block.read.shape.hash(&mut hasher);
                block.valid.start.hash(&mut hasher);
                block.valid.shape.hash(&mut hasher);
                if block.reads_across_grids() {
                    block.source.start.hash(&mut hasher);
                    block.source.shape.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// The guard. Every phase's valid regions must tile **its own** volume
    /// exactly, and every block must fetch from inside the level it reads.
    ///
    /// A short halo makes a phase's valid regions shrink below their cores, the
    /// tiling develops a hole, and this fires. There is no separate
    /// `halo >= reach` assertion anywhere, by design.
    ///
    /// **What used to be here and is not.** A phase whose grid volume differed
    /// from the decomposition's was refused outright, which made
    /// `input grid != output grid` inexpressible — not hard, not unpriced,
    /// *impossible*, for cross-grid pixel ops and for anything that reshapes.
    /// The refusal is replaced by two checks that are strictly stronger where
    /// the old one applied and still say something where it did not:
    ///
    /// * the tiling runs against the phase's own volume, so it is a real check
    ///   for every phase rather than a check of one phase and a shape assertion
    ///   for the rest;
    /// * a block's `source` must lie inside the level it reads, which is the
    ///   part that used to be true by construction and now has to be verified.
    ///   The levels chain — level 0 is `self.volume`, level `p+1` is phase `p`'s
    ///   — so a plan whose phases do not join up is caught here rather than
    ///   becoming two decompositions with no edge between them.
    pub fn check(&self) -> Result<()> {
        if self.phases.is_empty() {
            return Err(Error::InvalidArgument(
                "decomposition: no phases".to_string(),
            ));
        }
        for (index, phase) in self.phases.iter().enumerate() {
            let volume = phase.volume();
            let source_volume = self.volume_at(index);
            // A per-block reach is a table indexed by the block index, so a
            // table that does not match the lattice is a plan nobody can
            // reproduce. Checked here as well as where it is built, because a
            // plan may arrive from any strategy or off a wire.
            let blocks = phase.grid.blocks_per_axis();
            phase
                .reach
                .check_lattice(blocks, &format!("decomposition phase {index} reach"))?;
            phase
                .halo
                .check_lattice(blocks, &format!("decomposition phase {index} halo"))?;
            // A reach in the level below's own lattice is satisfied by the fetch
            // region and by nothing else — there is no factor turning a step of
            // that lattice into a voxel of this one, so it contributes nothing to
            // the halo. A phase that declares one and then fetches its own read
            // extent has declared a dependency it has no way to meet, and that is
            // exactly the shape of the zero somebody writes to get past a guard.
            if !phase.reach.space().converts_to_voxels() && !phase.reads_across_grids() {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {index} ({}) states its reach as {} — steps of the level \
                     below's own lattice — and every block fetches its own read extent. A \
                     dependency in that space is met by where a block reads, not by how wide its \
                     halo is, so the plan has to say where each block reads \
                     (`PhaseDecomposition::with_sources`).",
                    phase.names.join(">"),
                    phase.reach
                )));
            }
            for block in &phase.blocks {
                if !block.reads_across_grids() && source_volume == volume {
                    // The common case: the fetch is the read extent, derived
                    // from this volume and clamped to it by construction.
                    continue;
                }
                region_within(
                    &block.source,
                    &source_volume,
                    &format!(
                        "decomposition phase {index} block {:?}: the region it reads from level \
                         {index}",
                        block.index
                    ),
                )?;
            }
            let boxes = phase
                .blocks
                .iter()
                .map(|block| region_to_ranges(&block.valid))
                .collect::<Vec<_>>();
            boxes_tile_exactly(&boxes, &volume).map_err(|err| {
                let short = phase.blocks_missing_valid_core();
                Error::InvalidArgument(format!(
                    "decomposition phase {index} ({}): {err}. \
                     reach {}, halo {}; {} block(s) lost part of their core, first {:?}. \
                     A halo below the phase reach is the usual cause.",
                    phase.names.join(">"),
                    phase.reach,
                    phase.halo,
                    short.len(),
                    short.first()
                ))
            })?;
        }
        Ok(())
    }

    /// Rebuild every phase with a forced halo, **without** checking.
    ///
    /// The only reason this exists is to provoke the guard: a halo below a
    /// phase's reach must make `check` fail and must make a window-sum op's
    /// values diverge. A guard that has never been seen to fire is not known to
    /// work. It is a method with an unappealing name rather than a constructor
    /// parameter so that a caller reaching for it is visible in a grep.
    ///
    /// The rebuild keeps everything that is not derived from the halo: a phase's
    /// element type, and each block's fetch region, which is a function of the
    /// block rather than of the halo and is copied across by index. Losing them
    /// here would make this an unfaithful copy of the plan, and a provocation
    /// that changes two things at once proves nothing about either.
    pub fn with_forced_halo(&self, halo: impl Into<Reach>) -> Self {
        let halo = halo.into();
        let phases = self
            .phases
            .iter()
            .map(|phase| {
                let mut rebuilt = PhaseDecomposition::derive(
                    phase.slots.clone(),
                    phase.names.clone(),
                    phase.reach.clone(),
                    halo.clone(),
                    phase.grid.clone(),
                );
                rebuilt.dtype = phase.dtype;
                if phase.reads_across_grids() {
                    for (block, original) in rebuilt.blocks.iter_mut().zip(&phase.blocks) {
                        block.source = original.source.clone();
                    }
                }
                rebuilt
            })
            .collect();
        Self {
            volume: self.volume,
            dtype: self.dtype,
            phases,
            chain_reach: self.chain_reach,
        }
    }
}

pub(crate) fn region_to_ranges(region: &Region) -> Vec<(usize, usize)> {
    region
        .start
        .iter()
        .zip(region.shape.iter())
        .map(|(&start, &len)| (start, start + len))
        .collect()
}

// ----------------------------------------------------------------- cost --

/// Weights for the closed-form per-block cost.
///
/// Every number here is a *model*, and the design is blunt about the risk: the
/// search returns the optimum for whatever model it is given, and this project
/// has produced a 181-hour estimate that was really 3.8 and a `perf`
/// attribution that moved wall clock 0-6 %. So the defaults are declared
/// placeholders, not measurements.
///
/// Two consequences of that honesty are built in:
///
/// * `order_conflict_penalty` defaults to **zero**, so an unmeasured intuition
///   about traversal disagreement cannot silently drive a cut. Turn it on when
///   a measurement backs it.
/// * `materialise_cost_per_voxel` is **per phase boundary**, because
///   compressibility spans 2.09x (raw uint16) to 19.7x (`bool` after binarize)
///   and a single constant systematically over-values fusing late stages. The
///   default is the *conservative* end — assume poor compression, which
///   over-values materialisation and biases towards fusing, the cheaper
///   mistake.
///
/// **There is deliberately no residency price.** A phase whose working set is
/// the whole volume and one whose working set is a block carry the same
/// per-voxel weights here, and adding a term to separate them would need a
/// number nobody has measured — the same objection that keeps
/// `order_conflict_penalty` at zero. Residency is already represented, as a
/// *feasibility* constraint rather than a price: `Constraints::budget_bytes`
/// against `PhaseCost::working_set_bytes_per_block`. What was wrong was that
/// figure, not its absence — it was derived from the infinite-grid read and
/// could exceed the volume — so it is now the clamped extent. A planner told
/// memory is unbounded (`budget_bytes: None`) choosing one whole-volume block
/// for a local op is answering the question it was asked; the barrier it must
/// not fuse across is settled structurally, by `is_planning_barrier`, and needs
/// no price at all.
///
/// None of this can produce a wrong voxel. The partition decides where
/// intermediates land, not what is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    pub read_cost_per_voxel: f64,
    pub write_cost_per_voxel: f64,
    pub compute_scale: f64,
    /// Charged per extra distinct `preferred_iteration` inside one phase.
    pub order_conflict_penalty: f64,
    /// Cost of writing a voxel at a **phase boundary**, where the result is an
    /// intermediate rather than the workflow output.
    ///
    /// Separate from `write_cost_per_voxel` because compressibility is
    /// stage-dependent: 2.09x for raw uint16, 19.7x for the `bool` volumes
    /// after binarize. A planner using one number over-values fusing late
    /// stages and under-values fusing early ones. The default assumes poor
    /// compression, which biases towards fusing — the cheaper mistake.
    pub materialise_cost_per_voxel: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            read_cost_per_voxel: 1.0,
            write_cost_per_voxel: 1.0,
            compute_scale: 1.0,
            order_conflict_penalty: 0.0,
            materialise_cost_per_voxel: 1.0,
        }
    }
}

/// What a strategy is allowed to spend.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraints {
    /// Bytes available for in-flight blocks. `None` means unbounded.
    pub budget_bytes: Option<u64>,
    /// How many blocks a run is expected to hold at once. Part of the budget
    /// arithmetic, not of the decomposition: a strategy may run with fewer.
    pub expected_concurrency: usize,
    pub model: CostModel,
    /// Block edges the planner may choose between, per phase. Small by design —
    /// the search is `partitions x candidates^phases`, so this must stay short.
    pub block_candidates: Vec<usize>,
    /// Which axes may be cut. `[2]` is the z-only default that every recorded
    /// parity figure was measured under.
    pub split_axes: Vec<usize>,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![32, 64, 128],
            split_axes: vec![2],
        }
    }
}

// ------------------------------------------------------------- barriers --

/// Does a reach of `reach` cover the whole of an axis of extent `extent`?
///
/// This is the **barrier predicate**, and it is deliberately an exact
/// comparison rather than a threshold. `docs/design/BLOCK_OPS.md` measures why:
/// of seven merge steps in one real chain two reach a single voxel and four are
/// cheap full-reach reductions over streams of 17 kB to 45 MB, so any rule of
/// the form "large means full" would segment the chain in four places that do
/// not want it. "Full" is a property of the reach *relative to the volume*, not
/// a size and not a flag someone sets.
///
/// **Declared where it can be, detected where it cannot.** An op that means
/// "everything" now says so — `AxisReach::All` — and [`Reach::is_whole_axis`]
/// keys off the variant. An op that states a number keeps being compared against
/// the extent, because a number that happens to equal the volume is a full reach
/// whether or not anybody noticed. The two live behind this one name, called
/// from the pricing and from both planners, so there is no scatter of
/// `>= volume[axis]` tests to keep in step.
///
/// An axis of extent 1 is excluded. Reaching across it is trivially true, costs
/// nothing and forbids no blocking — every block already spans it — so counting
/// it would make every op on a flat volume a barrier.
pub fn reaches_whole_axis(reach: usize, extent: usize) -> bool {
    extent > 1 && reach >= extent
}

/// Whether a run of slots is a **planning barrier**: it reaches across the whole
/// of some axis, so no block on that axis is interior.
///
/// `docs/design/BLOCK_OPS.md` §"A full-reach op is a planning barrier": fusion
/// across one is impossible, the infinite-grid costing assumption breaks because
/// per-block cost stops being local, and cache state does not survive it.
/// Planning across one is planning across a reboot. Both planners therefore
/// segment here **structurally** — the cost model gets a vote on where else to
/// cut, not on this.
///
/// A slot whose reaches are stated in two coordinate spaces cannot be folded
/// into one reach at all; it is treated as a barrier, which is the conservative
/// answer and is also the right one — a change of coordinate space is a phase
/// boundary by construction.
pub fn is_planning_barrier(slot: &Chain, volume: [usize; 3]) -> bool {
    match slot.reach_spec(volume) {
        Ok(reach) => reach.is_barrier(volume),
        Err(_) => true,
    }
}

/// The axes a phase with this reach may still be cut on.
///
/// A block that does not span a full-reach axis reads less than the op needs on
/// every one of its voxels: `BlockGeometry::derive` marks it degenerate, its
/// valid region collapses and `Decomposition::check` reports the coverage hole.
/// Such a candidate is not merely expensive, it is unrunnable, so it is removed
/// from the choice rather than left for the cost model to dislike. Every other
/// axis stays cuttable: a reach that is full on one axis says nothing about the
/// others, and dropping them would be the over-firing the design warns against.
pub fn splittable_axes(split_axes: &[usize], reach: &Reach, volume: [usize; 3]) -> Vec<usize> {
    split_axes
        .iter()
        .copied()
        .filter(|&axis| axis >= 3 || !reach.is_whole_axis(axis, volume[axis]))
        .collect()
}

/// The infinite-grid per-block cost of one phase.
///
/// `total = n_blocks x per_block_cost(B, partition)`, with no boundary term.
/// The assumption is **conservative**: at a real volume boundary the halo is
/// clamped, so an edge block reads less and costs less than an interior one.
/// Assuming every block is interior therefore overestimates, which can only
/// make the planner cautious — the same direction of error as a generous halo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseCost {
    /// Read amplification on the infinite grid, charged per axis.
    ///
    /// Not a prediction of bytes moved — `Decomposition::exact_read_voxels` is
    /// that, from the real clamped geometry. This is the quantity the planner
    /// *chooses* on, and on a full-reach axis it states dependency rather than
    /// traffic: every output voxel depends on the whole extent.
    pub redundancy: f64,
    pub read_voxels_per_block: f64,
    pub compute_per_voxel: f64,
    /// Bytes resident while one block is in flight, from the **clamped** read
    /// extent: input plus output. This is what the byte budget is checked
    /// against, so it must be physical — a read can never exceed the volume.
    pub working_set_bytes_per_block: f64,
    pub cost_per_block: f64,
}

/// Price one candidate phase.
///
/// **Which axes are charged.** `BlockGrid` drops an axis from `split_axes` when
/// the block spans the volume, and charging only the split axes was a *clamp
/// discount*: on an uncut axis the read is clamped to the volume, so a bounded
/// reach really does cost nothing extra there and the discount is exact.
///
/// It is not exact when the reach is **full**. There the clamp is not a boundary
/// effect on an otherwise local computation, it is the entire behaviour — every
/// output voxel depends on every input voxel, no block is interior, and the
/// infinite-grid assumption the model is stated under has broken. Applying the
/// discount there priced the single-block full-reach phase, the most expensive
/// configuration there is, at redundancy **1.0**: the model charged nothing for
/// a phase that cannot be blocked, streamed or fused across. So a full-reach
/// axis is charged on the infinite grid whether or not the grid cut it.
///
/// The charge is deliberately kept out of the residency figure. `read_voxels`
/// feeds the *choice*, where over-charging is the design's declared safe
/// direction; `working_set_bytes_per_block` feeds the *budget*, where
/// over-charging invents infeasibility, so it is computed from the clamped read
/// extent and can never exceed the volume.
///
/// **The two sides are charged separately**, which is the whole point of an
/// asymmetric reach reaching the price: a dependency that is one-sided grows the
/// read by `lo + hi` and not by twice the wider of them. For a symmetric reach
/// that is `2r` exactly, so no plan built from one moves.
#[allow(clippy::too_many_arguments)]
pub fn price_phase(
    grid: &BlockGrid,
    reach: &Reach,
    compute_per_voxel: f64,
    distinct_orders: usize,
    is_materialised: bool,
    bytes_per_voxel: f64,
    model: &CostModel,
    materialise_cost_per_voxel: f64,
) -> PhaseCost {
    let core_voxels = grid.core_voxels();
    let block = grid.block();
    let volume = grid.volume();
    let reach = reach.in_voxels(block);
    let mut redundancy = 1.0_f64;
    let mut resident_voxels = 1.0_f64;
    for axis in 0..3 {
        // The widest block, because the price is per block and the model is
        // stated on the infinite grid: a per-block reach is charged at its worst
        // block, the same direction of error a generous halo has.
        let (lo, hi) = reach.axis(axis).bound(volume[axis]);
        let grown = block[axis] as f64 + lo as f64 + hi as f64;
        let charged = grid.split_axes().contains(&axis) || reach.is_whole_axis(axis, volume[axis]);
        if charged {
            redundancy *= grown / block[axis] as f64;
        }
        resident_voxels *= grown.min(volume[axis] as f64);
    }
    let read_voxels = core_voxels * redundancy;
    let conflict = if distinct_orders > 1 {
        model.order_conflict_penalty * core_voxels * (distinct_orders - 1) as f64
    } else {
        0.0
    };
    let write = if is_materialised {
        materialise_cost_per_voxel
    } else {
        model.write_cost_per_voxel
    };
    PhaseCost {
        redundancy,
        read_voxels_per_block: read_voxels,
        compute_per_voxel,
        // input buffer plus output buffer, both over the clamped read extent
        working_set_bytes_per_block: resident_voxels * bytes_per_voxel * 2.0,
        cost_per_block: read_voxels
            * (model.read_cost_per_voxel + model.compute_scale * compute_per_voxel)
            + core_voxels * write
            + conflict,
    }
}

/// Reach, compute and traversal preferences of a contiguous run of slots.
///
/// **Fallible, on the same argument as [`constraint_for`].** Reaches stated in
/// two coordinate spaces cannot be added — converting between them needs the
/// grid this function is called before choosing — so a group containing both is
/// refused, and a planner turns that into "this partition is infeasible" and
/// keeps searching. The alternative, folding them anyway under some assumed
/// conversion, would put a number in the binding half of the plan that nobody
/// stated.
#[allow(clippy::type_complexity)]
pub fn summarise_slots(
    slots: &[&Chain],
    group: &[usize],
    volume: [usize; 3],
) -> Result<(Reach, f64, Vec<String>, Vec<[usize; 3]>)> {
    // The first slot's space is the group's; `Reach::none()` would impose the
    // default one on a group that states another, and then adding the first
    // slot's own reach would be refused for disagreeing with a value nobody
    // wrote.
    let mut reach: Option<Reach> = None;
    let mut compute = 0.0_f64;
    let mut names = Vec::with_capacity(group.len());
    let mut orders: Vec<[usize; 3]> = Vec::new();
    for &slot in group {
        let chain = slots[slot];
        let stated = chain.reach_spec(volume)?;
        reach = Some(match reach {
            Some(so_far) => so_far.add(&stated)?,
            None => stated,
        });
        compute += chain.cost_per_voxel();
        names.push(chain.display_name());
        for order in chain.preferred_iterations() {
            if !orders.contains(&order) {
                orders.push(order);
            }
        }
    }
    Ok((reach.unwrap_or_default(), compute, names, orders))
}

/// What a contiguous run of slots requires of the blocks it is handed.
///
/// The counterpart of [`summarise_slots`] for [`BlockConstraint`], and separate
/// from it because it can *fail*: two ops that mandate different blocks cannot
/// share a phase, and that is a fact about the partition rather than a summary
/// of it. A planner turns the error into "this partition is infeasible" and
/// keeps searching, which is why it is returned rather than raised.
pub fn constraint_for(
    slots: &[&Chain],
    group: &[usize],
    volume: [usize; 3],
) -> Result<Option<BlockConstraint>> {
    let mut found: Option<BlockConstraint> = None;
    for &slot in group {
        let Some(constraint) = slots[slot].block_constraint(volume)? else {
            continue;
        };
        match &found {
            None => found = Some(constraint),
            Some(existing) if existing == &constraint => {}
            Some(existing) => {
                return Err(Error::InvalidArgument(format!(
                    "slot {slot} ({}) mandates {constraint:?} and an earlier slot of the same \
                     phase mandates {existing:?}; one phase hands every one of its ops the \
                     same block, so these two cannot be fused",
                    slots[slot].display_name()
                )))
            }
        }
    }
    Ok(found)
}

/// Every phase of `decomposition` hands its ops blocks they accept.
///
/// **The guard that cannot live in [`Decomposition::check`].** A plan records
/// op *names*, not implementations, so the plan alone cannot answer this; the
/// executor is the first place that holds both. That is also why it is a real
/// guard rather than a formality: `execute` may be handed a decomposition from
/// any strategy, or from a wire, and a plan that satisfied a mandate when it was
/// chosen is not thereby a plan that satisfies it now.
pub fn check_block_constraints(chain: &Chain, decomposition: &Decomposition) -> Result<()> {
    let slots = chain.slots();
    for (index, phase) in decomposition.phases.iter().enumerate() {
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            // A slot index out of range is caught, with a better message, by the
            // executor's own slot-order check. Nothing to say here.
            continue;
        }
        // The **level the phase reads**, not the phase's own volume. A mandate
        // is about the region an op is handed, and that region — `source` — is
        // in the level below's coordinate space. The two are the same for every
        // phase whose output grid is its input grid, which is most of them, and
        // where they differ this is the one that means anything: a lattice laid
        // over an array is a lattice over *that* array's extent.
        let volume = decomposition.volume_at(index);
        let Some(constraint) = constraint_for(&slots, &phase.slots, volume)? else {
            continue;
        };
        constraint.check(
            &phase.blocks,
            &format!("decomposition phase {index} ({})", phase.names.join(">")),
        )?;
    }
    Ok(())
}

/// Every source leaf in `chain` names a level its phase can actually read, and
/// the plan records which ones.
///
/// **The guard that cannot live in [`Decomposition::check`]**, on exactly
/// [`check_block_constraints`]' argument: a plan records op names, not
/// implementations, so the plan alone cannot see the leaves. The executor is
/// the first place holding both halves, and it runs this before the first
/// block — a forward reference is a fact about the plan, and a plan that is not
/// a plan should be refused as one rather than survive until some block asks
/// for a level nothing has written.
///
/// Four things are checked, and each of them is a way for a well-formed,
/// complete, wrong volume to come out otherwise:
///
/// * **the level exists.** An index past the end is not a reference to
///   anything.
/// * **it is not a forward reference.** Phases run `0..n`, so level `s` is
///   written by phase `s - 1` and is only there for a phase that runs after it:
///   `s <= p` for phase `p`, which reads level `p`. Refused *by name*, saying
///   which phase writes the level and which reads it. (`s == p` is the phase's
///   own input read a second time — degenerate, harmless, and not worth a
///   special case that would then have to be right.)
/// * **it is on the same lattice.** A source leaf has reach 0 and reads the
///   block's own fetch region, so the level it reads has to be in the same
///   coordinate space as the level the phase reads. A different volume would
///   make the same integers mean different voxels, which is the failure this
///   crate exists to prevent rather than one to price.
/// * **the declared element type is the level's.** The leaf carries a `Dtype`
///   because `Chain::produces` has nothing else to answer with, and every fold
///   of the chain was built from that answer; here it meets the plan, which is
///   the only thing that knows what the level holds.
///
/// Finally the recorded `source_levels` must be exactly what the slots name —
/// a plan whose record disagrees with its chain would read one level and price
/// another, and the whole reason the field exists is that it is parity-visible.
pub fn check_source_levels(chain: &Chain, decomposition: &Decomposition) -> Result<()> {
    let slots = chain.slots();
    for (phase_index, phase) in decomposition.phases.iter().enumerate() {
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            // Out of range is caught, with a better message, by the executor's
            // own slot-order check.
            continue;
        }
        let mut named: Vec<usize> = phase
            .slots
            .iter()
            .flat_map(|&slot| slots[slot].source_levels())
            .collect();
        named.sort_unstable();
        named.dedup();
        if named != phase.source_levels {
            return Err(Error::InvalidArgument(format!(
                "decomposition phase {phase_index} ({}) records that it also reads level(s) \
                 {:?}, and its slots name {:?}. The recorded list is what the executor reads and \
                 what the fingerprint hashes, so a plan whose record disagrees with its chain \
                 would price one level and read another.",
                phase.names.join(">"),
                phase.source_levels,
                named
            )));
        }
        for &level in &named {
            if level >= decomposition.n_levels() {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads level {level} through a source \
                     leaf, and this plan has {} level(s), numbered 0 to {}.",
                    phase.names.join(">"),
                    decomposition.n_levels(),
                    decomposition.n_levels() - 1
                )));
            }
            if level > phase_index {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads level {level} through a source \
                     leaf, but level {level} is written by phase {}, which runs after it. Phases \
                     run in order, so a source leaf may only name a level at or below the one its \
                     phase is handed — level {phase_index} here.",
                    phase.names.join(">"),
                    level - 1
                )));
            }
            let read = decomposition.volume_at(phase_index);
            let stored = decomposition.volume_at(level);
            if stored != read {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads level {level}, which is \
                     {stored:?}, beside level {phase_index}, which is {read:?}. A source leaf has \
                     reach 0 and is read at the block's own fetch region, so the two levels have \
                     to be in one coordinate space; across grids the same integers would name \
                     different voxels.",
                    phase.names.join(">")
                )));
            }
        }
        for &slot in &phase.slots {
            check_declared_source_dtypes(slots[slot], phase_index, phase, decomposition)?;
        }
    }
    Ok(())
}

/// Walk one slot's source leaves and compare each declaration against the level.
fn check_declared_source_dtypes(
    node: &Chain,
    phase_index: usize,
    phase: &PhaseDecomposition,
    decomposition: &Decomposition,
) -> Result<()> {
    match node {
        Chain::Op(_) => Ok(()),
        Chain::Source { level, dtype } => {
            let held = decomposition.dtype_at(*level);
            if held != *dtype {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) has a source leaf declaring level \
                     {level} holds {}, and the plan folds that level to {}. Every fold of the \
                     chain — what the combine accepts, what the phase writes — was built from \
                     the declaration, so it has to be the level's own element type.",
                    phase.names.join(">"),
                    dtype.numpy_name(),
                    held.numpy_name()
                )));
            }
            Ok(())
        }
        Chain::Sequence(children)
        | Chain::Alternative {
            branches: children, ..
        }
        | Chain::Parallel {
            branches: children, ..
        } => children.iter().try_for_each(|child| {
            check_declared_source_dtypes(child, phase_index, phase, decomposition)
        }),
    }
}

/// Every phase of `decomposition` writes the element type its ops produce.
///
/// **The guard that cannot live in [`Decomposition::check`]**, for exactly the
/// reason [`check_block_constraints`] cannot: a plan records op *names*, not
/// implementations, so the plan alone cannot answer what its ops produce. The
/// executor is the first place that holds both, and a plan may arrive from any
/// strategy or off a wire.
///
/// Two things are refused here, and they fail differently on purpose:
///
/// * an op handed an element type it does not [`accept`](crate::op::BlockOp::accepts)
///   — the message names the op and the type, because that is a chain that
///   cannot run at all rather than a plan that is merely wrong;
/// * a phase whose level is allocated at one type while its ops write another —
///   the message names the level, because that *is* the plan being wrong.
pub fn check_dtypes(
    chain: &Chain,
    decomposition: &Decomposition,
    work: &[crate::fragment::PhaseWork<'_>],
) -> Result<()> {
    let slots = chain.slots();
    let mut current = decomposition.dtype;
    for (index, phase) in decomposition.phases.iter().enumerate() {
        // A fragment phase owns no slot, so the fold above has nothing to fold
        // and the op is the only thing that knows what it writes. Before this
        // arm existed, a `volume -> fragments` op that widened its level — a
        // labelling writing `u32` over a `bool` mask — was refused by a message
        // about ops the phase does not have.
        if let Some(crate::fragment::PhaseWork::Fragments(op)) = work.get(index) {
            if !op.writes_pixels() {
                // Terminal as far as levels go; there is no level to be wrong.
                continue;
            }
            let produced = op.produces(current);
            let declared = decomposition.dtype_at(index + 1);
            if declared != produced {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs fragment op {:?}, which reads {} and writes {}, but \
                     the plan allocates level {} as {}. A phase that changes the element type \
                     says so in its own `dtype`.",
                    op.name(),
                    current.numpy_name(),
                    produced.numpy_name(),
                    index + 1,
                    declared.numpy_name()
                )));
            }
            current = produced;
            continue;
        }
        // An iterative phase owns no slot, so the fold below would silently pass
        // it through and never ask the op whether it can work in this width. It
        // *does* hand the type on unchanged — the running operand of one substage
        // is the previous substage's output, so a step that retyped could not be
        // iterated — and that is asserted rather than assumed by the shared
        // comparison at the end.
        if let Some(crate::fragment::PhaseWork::Iterate(op)) = work.get(index) {
            if !op.accepts(current) {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs iterative op {:?}, which does not accept {}. An op that \
                     cannot work in the width it is handed is refused when the plan is made, not \
                     discovered when a block reaches it.",
                    op.name(),
                    current.numpy_name()
                )));
            }
        }
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            // A slot index out of range is caught, with a better message, by the
            // executor's own slot-order check. Nothing to say here.
            continue;
        }
        for &slot in &phase.slots {
            current = slots[slot].produces(current)?;
        }
        let declared = decomposition.dtype_at(index + 1);
        if declared != current {
            return Err(Error::InvalidArgument(format!(
                "phase {index} ({}) reads {} and its ops write {}, but the plan allocates level \
                 {} as {}. A phase that changes the element type says so in its own `dtype`; a \
                 plan that does not is a plan whose level is the wrong width.",
                phase.names.join(">"),
                decomposition.dtype_at(index).numpy_name(),
                current.numpy_name(),
                index + 1,
                declared.numpy_name()
            )));
        }
    }
    Ok(())
}

/// **Every chunk of a level is written by exactly one task.**
///
/// The third guard that cannot live in [`Decomposition::check`], for the same
/// reason [`check_block_constraints`] and [`check_dtypes`] cannot: a plan says
/// nothing about how a level is chunked, so the plan alone cannot answer this.
/// The chunk shapes come from whatever holds the storage, and the first place
/// that holds both halves is the environment's `prepare`.
///
/// Precisely: for phase `p`, which writes level `p+1`, every chunk of level
/// `p+1` must lie inside exactly one of phase `p`'s **valid regions**. The valid
/// regions already tile the phase's volume exactly — that is what
/// [`Decomposition::check`] asserts — so "exactly one" reduces to "no chunk is
/// cut by a valid-region boundary", which is what this checks, per axis, per
/// block, in linear time.
///
/// **Why the invariant is a mandate rather than an aspiration**, since a reader
/// meeting this function will want to know what it buys:
///
/// * a chunk with two writers is a lost-update hazard in any store whose partial
///   writes are read-modify-write, which is most of them, and is why the Zarr
///   environment carries per-chunk locks at all;
/// * a chunk with one writer has a lifetime and an invalidation point; a chunk
///   with several needs write-combining, cross-task invalidation and a notion of
///   "partly valid" that no cache tier wants to carry.
///
/// **It constrains writes, not reads.** Reads may straddle chunks freely — that
/// is what a halo is — so `chunks[0]` is never looked at: level 0 is nobody's
/// output. It is still taken, so that the argument is "the chunk shape of every
/// level" and a caller does not have to remember an off-by-one.
///
/// A chunk overhanging the volume's far edge is not a violation. It holds no
/// voxel outside the array, so the one valid region that meets it owns every
/// voxel it can hold — which is exactly the ownership this asserts.
pub fn check_chunk_exclusive_writes(
    decomposition: &Decomposition,
    chunks: &[[usize; 3]],
) -> Result<()> {
    if chunks.len() != decomposition.n_levels() {
        return Err(Error::InvalidArgument(format!(
            "chunk-exclusive check: this plan has {} level(s) and {} chunk shape(s) were given. \
             The argument is one shape per level, level 0 included, so that the index is the \
             level number.",
            decomposition.n_levels(),
            chunks.len()
        )));
    }
    for (index, phase) in decomposition.phases.iter().enumerate() {
        let level = index + 1;
        let chunk = chunks[level];
        let volume = phase.volume();
        for axis in 0..3 {
            if chunk[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "chunk-exclusive check: level {level} is chunked {chunk:?}, and a chunk of \
                     zero extent on axis {axis} tiles nothing"
                )));
            }
        }
        for block in &phase.blocks {
            if block.valid.voxels() == 0 {
                // A block with no trustworthy voxel writes nothing and owns
                // nothing. The hole it leaves is `check`'s to report.
                continue;
            }
            for axis in 0..3 {
                let start = block.valid.start[axis];
                let end = start + block.valid.shape[axis];
                // A boundary interior to the volume that is not on the chunk
                // grid is a chunk with a writer on each side of it. The volume's
                // own far edge is not such a boundary: nothing lies beyond it.
                let cut = if start % chunk[axis] != 0 {
                    start
                } else if end % chunk[axis] != 0 && end != volume[axis] {
                    end
                } else {
                    continue;
                };
                let mut chunk_index = [0usize; 3];
                for other in 0..3 {
                    chunk_index[other] = block.valid.start[other] / chunk[other];
                }
                chunk_index[axis] = cut / chunk[axis];
                return Err(shared_chunk_error(
                    index,
                    level,
                    phase,
                    &chunk_index,
                    &chunk,
                    &volume,
                ));
            }
        }
    }
    Ok(())
}

/// The message [`check_chunk_exclusive_writes`] fires with: the phase, the
/// chunk, and the valid regions that share it.
///
/// Naming the sharers costs a scan of the phase's blocks, which is paid once and
/// only on the failure path. It is worth it: "some chunk is shared" sends a
/// reader hunting, and "block A writes this and block B writes that" tells them
/// whether the block grid or the chunk grid is the thing to move.
fn shared_chunk_error(
    index: usize,
    level: usize,
    phase: &PhaseDecomposition,
    chunk_index: &[usize; 3],
    chunk: &[usize; 3],
    volume: &[usize; 3],
) -> Error {
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for axis in 0..3 {
        lo[axis] = chunk_index[axis] * chunk[axis];
        hi[axis] = (lo[axis] + chunk[axis]).min(volume[axis]);
    }
    let sharers: Vec<String> = phase
        .blocks
        .iter()
        .filter(|block| {
            block.valid.voxels() > 0
                && (0..3).all(|axis| {
                    let start = block.valid.start[axis];
                    start < hi[axis] && lo[axis] < start + block.valid.shape[axis]
                })
        })
        .map(|block| {
            format!(
                "block {:?} writes {:?}..{:?}",
                block.index,
                block.valid.start,
                block.valid.end()
            )
        })
        .collect();
    Error::InvalidArgument(format!(
        "decomposition phase {index} ({}) writes level {level}, which is chunked {chunk:?}: chunk \
         {chunk_index:?} spans {lo:?}..{hi:?} and {} of this phase's valid regions land in it — \
         {}. Every chunk of a level must be written by exactly one task: two writers of one chunk \
         lose each other's bytes in any store whose partial write is a read-modify-write, and a \
         chunk with no single owner has no lifetime a cache can hold it by. Either the block grid \
         must be a whole multiple of the chunk shape, or — for a level whose layout nobody outside \
         the run has asked for — the chunk shape should be derived from the block grid, which \
         satisfies this at no cost.",
        phase.names.join(">"),
        sharers.len(),
        sharers.join("; ")
    ))
}

/// Bit `i` of `mask` set means "cut between slot `i` and slot `i+1`".
pub fn groups_for(mask: u32, n_slots: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut current = vec![0usize];
    for slot in 1..n_slots {
        if mask & (1 << (slot - 1)) != 0 {
            groups.push(std::mem::take(&mut current));
        }
        current.push(slot);
    }
    groups.push(current);
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::IdentityOp;

    #[test]
    fn cut_masks_enumerate_every_contiguous_partition() {
        assert_eq!(groups_for(0b00, 3), vec![vec![0, 1, 2]]);
        assert_eq!(groups_for(0b01, 3), vec![vec![0], vec![1, 2]]);
        assert_eq!(groups_for(0b10, 3), vec![vec![0, 1], vec![2]]);
        assert_eq!(groups_for(0b11, 3), vec![vec![0], vec![1], vec![2]]);
        assert_eq!((0..1u32 << 7).map(|m| groups_for(m, 8).len()).count(), 128);
    }

    fn decomposition(halo: [usize; 3], reach: [usize; 3]) -> Decomposition {
        let grid = BlockGrid::new([64, 8, 8], [16, 8, 8]).unwrap();
        Decomposition {
            volume: [64, 8, 8],
            dtype: Dtype::F64,
            phases: vec![PhaseDecomposition::derive(
                vec![0],
                vec!["op".to_string()],
                reach,
                halo,
                grid,
            )],
            chain_reach: reach,
        }
    }

    #[test]
    fn the_tiling_check_passes_when_the_halo_covers_the_reach() {
        decomposition([4, 0, 0], [4, 0, 0]).check().unwrap();
        decomposition([40, 0, 0], [4, 0, 0]).check().unwrap();
    }

    #[test]
    fn the_tiling_check_fires_when_the_halo_is_short() {
        let err = decomposition([2, 0, 0], [5, 0, 0]).check().unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("do not tile the volume exactly"),
            "expected the tiling guard to fire, got: {message}"
        );
        assert!(message.contains("halo [2, 0, 0]"), "got: {message}");
    }

    // ------------------------------------------- chunk-exclusive writing --

    /// A chunk shape that divides the block extent is exclusive; the volume's
    /// far edge is not a boundary and a chunk overhanging it is nobody's second
    /// owner.
    #[test]
    fn a_chunk_grid_the_blocks_are_a_multiple_of_is_exclusive() {
        let plan = decomposition([4, 0, 0], [4, 0, 0]);
        // Blocks are [16, 8, 8] over [64, 8, 8].
        for chunk in [[16, 8, 8], [8, 8, 8], [4, 4, 4], [2, 8, 1], [1, 1, 1]] {
            check_chunk_exclusive_writes(&plan, &[[9, 9, 9], chunk]).unwrap();
        }
        // 5 divides neither 16 nor 8, but on the axes the grid does not cut —
        // one block spans them — there is no interior boundary to land on, so
        // only the overhang past the volume edge is left and that is legal.
        check_chunk_exclusive_writes(&plan, &[[9, 9, 9], [16, 5, 5]]).unwrap();
    }

    /// **The guard, watched firing**, naming the phase, the chunk and both
    /// blocks that would write into it.
    #[test]
    fn a_chunk_two_blocks_would_write_is_refused_and_names_them() {
        let plan = decomposition([4, 0, 0], [4, 0, 0]);
        let err = check_chunk_exclusive_writes(&plan, &[[9, 9, 9], [6, 8, 8]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("phase 0"), "got: {err}");
        // Block 1's valid region starts at 16, which sits inside chunk 2
        // (12..18) — shared with block 0, which writes 0..16.
        assert!(err.contains("chunk [2, 0, 0]"), "got: {err}");
        assert!(err.contains("[12, 0, 0]..[18, 8, 8]"), "got: {err}");
        assert!(err.contains("block [0, 0, 0]") && err.contains("block [1, 0, 0]"));
        assert!(err.contains("exactly one task"), "got: {err}");
    }

    /// One shape per level, level 0 included, so that the index is the level
    /// number — and a caller who gets that wrong is told rather than checked
    /// against the wrong grid.
    #[test]
    fn the_chunk_check_wants_one_shape_per_level() {
        let plan = decomposition([4, 0, 0], [4, 0, 0]);
        let err = check_chunk_exclusive_writes(&plan, &[[8, 8, 8]])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 level(s)") && err.contains("1 chunk shape"),
            "got: {err}"
        );
        // Level 0 is read-only, so whatever it is chunked as says nothing.
        check_chunk_exclusive_writes(&plan, &[[7, 3, 5], [8, 8, 8]]).unwrap();
    }

    #[test]
    fn the_fingerprint_is_stable_and_distinguishes_decompositions() {
        let a = decomposition([4, 0, 0], [4, 0, 0]);
        let b = decomposition([4, 0, 0], [4, 0, 0]);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_ne!(
            a.fingerprint(),
            decomposition([8, 0, 0], [4, 0, 0]).fingerprint()
        );
    }

    #[test]
    fn a_forced_halo_rebuilds_the_geometry_without_touching_the_reach() {
        let good = decomposition([5, 0, 0], [5, 0, 0]);
        good.check().unwrap();
        let bad = good.with_forced_halo([1, 0, 0]);
        assert_eq!(bad.phases[0].reach, [5, 0, 0]);
        assert!(bad.check().is_err());
    }

    /// The adversarial case from the design: a huge reach between two zeros.
    ///
    /// Isolating it wins **in proportion to the work it would otherwise
    /// multiply**. Redundancy is a multiplier on compute, so if the neighbours
    /// are nearly free there is nothing to save and paying two extra
    /// materialisations loses — which the second half of this test records,
    /// because it is a real and easily-missed property of the cost model rather
    /// than a defect in it.
    #[test]
    fn isolating_a_high_reach_op_pays_when_its_neighbours_do_real_work() {
        let grid = BlockGrid::along([1024, 64, 64], &[0], 64).unwrap();
        let model = CostModel::default();
        let price = |reach: [usize; 3], compute: f64, materialised: bool| {
            price_phase(
                &grid,
                &reach.into(),
                compute,
                1,
                materialised,
                8.0,
                &model,
                1.0,
            )
            .cost_per_block
        };

        // neighbours cost 5 units per voxel each
        let fused = price([50, 0, 0], 11.0, false);
        let isolated = price([0, 0, 0], 5.0, true)
            + price([50, 0, 0], 1.0, true)
            + price([0, 0, 0], 5.0, false);
        assert!(
            isolated < fused,
            "isolating cost {isolated}, fusing cost {fused}"
        );

        // neighbours nearly free: the redundancy saved is worth less than the
        // two extra read/write rounds, and fusing is right
        let fused_cheap = price([50, 0, 0], 3.0, false);
        let isolated_cheap = price([0, 0, 0], 1.0, true)
            + price([50, 0, 0], 1.0, true)
            + price([0, 0, 0], 1.0, false);
        assert!(
            fused_cheap < isolated_cheap,
            "fusing cost {fused_cheap}, isolating cost {isolated_cheap}"
        );
    }

    /// Exact, not a threshold. The design measures why: of seven merge steps in
    /// one real chain two reach a single voxel, and a rule that treated "large"
    /// as "full" would segment where nothing wants a segment.
    #[test]
    fn the_barrier_predicate_is_an_exact_comparison_and_ignores_a_flat_axis() {
        assert!(!reaches_whole_axis(4095, 4096));
        assert!(reaches_whole_axis(4096, 4096));
        assert!(reaches_whole_axis(9999, 4096));
        // an axis of extent 1 is spanned by every block already, so reaching
        // across it forbids nothing and must not make every op a barrier
        assert!(!reaches_whole_axis(7, 1));

        let volume = [4096, 4096, 1];
        assert!(!is_planning_barrier(
            &Chain::op(IdentityOp::new("bounded", [4095, 0, 1])),
            volume
        ));
        assert!(is_planning_barrier(
            &Chain::op(IdentityOp::new("global", [4096, 0, 0])),
            volume
        ));
        // a sequence is a barrier if any part of it is — reaches add
        assert!(is_planning_barrier(
            &Chain::sequence(vec![
                Chain::op(IdentityOp::new("a", [2048, 0, 0])),
                Chain::op(IdentityOp::new("b", [2048, 0, 0])),
            ]),
            volume
        ));
    }

    /// A full-reach axis stops being splittable, and only that axis does.
    #[test]
    fn only_the_full_reach_axis_is_removed_from_the_choice_of_cuts() {
        let volume = [1024, 64, 64];
        assert_eq!(
            splittable_axes(&[0, 1, 2], &[8, 8, 8].into(), volume),
            vec![0, 1, 2]
        );
        assert_eq!(
            splittable_axes(&[0, 1, 2], &[8, 64, 8].into(), volume),
            vec![0, 2]
        );
        assert!(splittable_axes(&[0], &[1024, 0, 0].into(), volume).is_empty());
        // an axis nobody asked to cut is not added by being full
        assert_eq!(splittable_axes(&[0], &[0, 64, 64].into(), volume), vec![0]);
    }

    /// The defect this file carried: `split_axes` drops an axis when the block
    /// spans the volume, so the *only* configuration a full-reach op can run in
    /// was priced at redundancy 1.0 — nothing charged for a phase that cannot be
    /// blocked, streamed, or fused across.
    #[test]
    fn a_single_block_full_reach_phase_is_not_priced_as_free() {
        let volume = [32, 4, 4];
        let grid = BlockGrid::whole(volume).unwrap();
        assert!(grid.split_axes().is_empty());
        let model = CostModel::default();
        let cost = price_phase(&grid, &volume.into(), 1.0, 1, false, 8.0, &model, 1.0);
        assert!(
            cost.redundancy > 1.0,
            "a phase whose every voxel depends on the whole volume priced at \
             redundancy {}",
            cost.redundancy
        );

        // and a bounded reach on an axis the block spans is still free, because
        // there the clamp is exact: the read cannot leave the volume
        let bounded = price_phase(&grid, &[4, 0, 0].into(), 1.0, 1, false, 8.0, &model, 1.0);
        assert_eq!(bounded.redundancy, 1.0);
    }

    /// Residency is physical, so it is clamped even though the cost is not.
    ///
    /// Measured consequence: over a `[64, 8, 8]` volume — 32 kB at 8 bytes —
    /// the unclamped figure claimed a phase needed more than 1 MB resident, and
    /// a 1 MB budget refused every partition of a chain that fits eight times
    /// over.
    #[test]
    fn the_working_set_is_the_clamped_read_and_cannot_exceed_the_volume() {
        let volume = [64, 8, 8];
        let grid = BlockGrid::along(volume, &[0], 32).unwrap();
        let model = CostModel::default();
        let cost = price_phase(
            &grid,
            &[512, 512, 512].into(),
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
        );
        let whole_volume_bytes = (volume.iter().product::<usize>() * 8 * 2) as f64;
        assert!(
            cost.working_set_bytes_per_block <= whole_volume_bytes,
            "a block claimed {} bytes resident of a {whole_volume_bytes} byte volume",
            cost.working_set_bytes_per_block
        );
        // the cost, by contrast, is deliberately the un-clamped infinite grid
        assert!(cost.read_voxels_per_block > grid.core_voxels());
    }

    // ------------------------------------------- per-phase volumes --

    /// Two phases over two volumes, joined by a fetch region.
    ///
    /// The plan a single `volume` field made inexpressible: phase 0 writes a
    /// `[16, 4, 4]` level, phase 1 is cut from `[8, 4, 4]` and reads the top half
    /// of the level below.
    fn two_volumes(halo: [usize; 3], reach: [usize; 3]) -> Decomposition {
        let first = PhaseDecomposition::derive(
            vec![0],
            vec!["first".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new([16, 4, 4], [8, 4, 4]).unwrap(),
        );
        let second = PhaseDecomposition::derive(
            vec![1],
            vec!["second".to_string()],
            reach,
            halo,
            BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap(),
        )
        .with_sources(|block| {
            Region::new(
                &[block.read.start[0] + 8, 0, 0],
                &[block.read.shape[0], 4, 4],
            )
        });
        Decomposition {
            volume: [16, 4, 4],
            dtype: Dtype::F64,
            phases: vec![first, second],
            chain_reach: reach,
        }
    }

    #[test]
    fn a_phase_may_be_cut_from_a_different_volume_than_the_one_it_reads() {
        let plan = two_volumes([0, 0, 0], [0, 0, 0]);
        plan.check().unwrap();
        assert_eq!(plan.volume_at(0), [16, 4, 4]);
        assert_eq!(plan.volume_at(1), [16, 4, 4]);
        assert_eq!(plan.volume_at(2), [8, 4, 4]);
        assert_eq!(plan.output_volume(), [8, 4, 4]);
        assert_eq!(plan.uniform_volume(), None);
        // and what it fetches is what the plan says will be read
        assert_eq!(plan.exact_read_voxels(), vec![16 * 4 * 4, 8 * 4 * 4]);
    }

    /// The guard, on the phase that changed the volume.
    ///
    /// It is reached through a path that did not exist before — the tiling now
    /// runs against each phase's own volume rather than one shared one — so it
    /// is provoked here rather than assumed to have come along.
    #[test]
    fn the_tiling_check_fires_on_a_short_halo_in_a_phase_with_its_own_volume() {
        let plan = two_volumes([1, 0, 0], [3, 0, 0]);
        let message = plan.check().unwrap_err().to_string();
        assert!(
            message.contains("phase 1") && message.contains("do not tile the volume exactly"),
            "expected the tiling guard to fire on the second phase, got: {message}"
        );
        assert!(message.contains("halo [1, 0, 0]"), "got: {message}");
    }

    /// The check that replaced "every grid must be over the one volume": a
    /// phase that changes shape and does *not* say where it reads is fetching
    /// coordinates of one array out of another.
    #[test]
    fn a_fetch_outside_the_level_it_reads_is_refused() {
        let mut plan = two_volumes([0, 0, 0], [0, 0, 0]);
        // strip the mapping: the blocks fall back to their own read extents,
        // which are regions of [8, 4, 4] and not of the [16, 4, 4] below
        plan.phases[1] = plan.phases[1].clone().with_sources(|block| {
            Region::new(&[block.read.start[0] + 12, 0, 0], &block.read.shape.clone())
        });
        let message = plan.check().unwrap_err().to_string();
        assert!(
            message.contains("reads from level 1") && message.contains("region axis 0"),
            "{message}"
        );
    }

    /// A plan that uses neither new feature fingerprints exactly as it did
    /// before either existed, so a recorded parity figure still names its plan.
    #[test]
    fn the_fingerprint_is_unchanged_by_features_a_plan_does_not_use() {
        // The literal is what this decomposition hashed to *before* a phase
        // could own an element type and a block could own a fetch region — read
        // off the tree as it was and pinned here, because "unchanged" is a claim
        // about history and cannot be checked against the present.
        let plan = decomposition([4, 0, 0], [4, 0, 0]);
        assert!(plan.phases.iter().all(|phase| phase.dtype.is_none()));
        assert!(!plan.phases[0].reads_across_grids());
        assert_eq!(plan.fingerprint(), 9_837_599_547_069_300_871);

        // and a plan that does use them is a different plan
        let plain = decomposition([4, 0, 0], [4, 0, 0]);
        let mut retyped = plain.clone();
        retyped.phases[0] = retyped.phases[0].clone().with_dtype(Dtype::U16);
        assert_ne!(plain.fingerprint(), retyped.fingerprint());
        let mut moved = plain.clone();
        moved.phases[0] = moved.phases[0]
            .clone()
            .with_sources(|block| Region::new(&[0, 0, 0], &block.read.shape.clone()));
        assert_ne!(plain.fingerprint(), moved.fingerprint());
        // a "mapping" that maps every block to its own read extent is not one
        let same = plain.phases[0]
            .clone()
            .with_sources(|block| block.read.clone());
        assert!(!same.reads_across_grids());
    }

    /// The element type is folded from level 0, so a phase that says nothing
    /// hands on what it read.
    #[test]
    fn the_element_type_is_per_level_and_folded_from_the_input() {
        let mut plan = two_volumes([0, 0, 0], [0, 0, 0]);
        assert_eq!(plan.dtype_at(0), Dtype::F64);
        assert_eq!(plan.dtype_at(2), Dtype::F64);
        assert_eq!(plan.uniform_dtype(), Some(Dtype::F64));
        plan.phases[0] = plan.phases[0].clone().with_dtype(Dtype::U8);
        assert_eq!(plan.dtype_at(0), Dtype::F64);
        assert_eq!(plan.dtype_at(1), Dtype::U8);
        // phase 1 declares nothing, so level 2 is what level 1 was
        assert_eq!(plan.dtype_at(2), Dtype::U8);
        assert_eq!(plan.uniform_dtype(), None);
    }

    /// The provocation must change the halo and nothing else, or it proves
    /// nothing about the halo.
    #[test]
    fn a_forced_halo_keeps_the_element_type_and_the_fetch_regions() {
        let plan = two_volumes([0, 0, 0], [0, 0, 0]);
        let sources: Vec<Region> = plan.phases[1]
            .blocks
            .iter()
            .map(|block| block.source.clone())
            .collect();
        let forced = plan.with_forced_halo([2, 0, 0]);
        assert_eq!(forced.phases[1].halo, [2, 0, 0]);
        assert_eq!(
            forced.phases[1]
                .blocks
                .iter()
                .map(|block| block.source.clone())
                .collect::<Vec<_>>(),
            sources
        );
        let mut typed = plan.clone();
        typed.phases[0] = typed.phases[0].clone().with_dtype(Dtype::U16);
        assert_eq!(
            typed.with_forced_halo([2, 0, 0]).phases[0].dtype,
            Some(Dtype::U16)
        );
    }

    #[test]
    fn summarise_adds_reaches_along_a_run_of_slots() {
        let chain = Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [1, 2, 3])),
            Chain::op(IdentityOp::new("b", [4, 0, 0]).with_cost(2.5)),
        ]);
        let slots = chain.slots();
        let (reach, compute, names, orders) =
            summarise_slots(&slots, &[0, 1], [100, 100, 100]).unwrap();
        assert_eq!(reach, [5, 2, 3]);
        assert_eq!(compute, 3.5);
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
        assert!(orders.is_empty());
    }
}
