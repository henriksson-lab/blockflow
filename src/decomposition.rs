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
// Images, and why a volume is per phase
// -------------------------------------
// Image 0 is the input; image `p+1` is what phase `p` wrote. Each phase owns
// the volume its lattice is cut from — `PhaseDecomposition::volume` — and reads
// the image below, whose shape is `Decomposition::volume_at(p)`. The two are
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
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use crate::assemble::{describe_image, is_supplied_image, ImageId};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::reach::{AxisReach, Frame, Reach};
use crate::region::Region;
use crate::tiling::boxes_tile_exactly;

use super::geometry::{region_within, BlockGeometry, BlockGrid};
use super::op::{Anchor, BlockConstraint, Chain, Placement};

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
    /// with [`Decomposition::dtype_at`], which folds the chain from image 0.
    pub dtype: Option<Dtype>,
    /// Images this phase reads **besides** the one it is handed, ascending and
    /// without repeats: one per [`Chain::Source`] leaf in its slots.
    ///
    /// **Two different things in this file are called a source, and this is the
    /// one that is an image.** `BlockGeometry::source` is a *region* — where in
    /// the image below a block fetches from — and every phase has one per
    /// block. This is a list of *images*, and it is empty for every phase that
    /// does not read a second array. They meet in exactly one place: a source
    /// image is read at each block's `source` region, because a source leaf has
    /// reach 0 and therefore reads what the phase already fetches.
    ///
    /// **In the binding half of the plan, unlike `Visibility`.** Which image an
    /// arm reads changes voxels, so it is recorded, fingerprinted and shipped —
    /// the "explicit edges in the binding plan" of `docs/design/BLOCK_OPS.md`
    /// §"Images are a DAG", in the one shape this crate needs them.
    ///
    /// Derived from the chain by [`Decomposition::declare_source_images`] and
    /// verified against it by [`check_source_images`], the same split
    /// `declare_dtypes` and `check_dtypes` have.
    pub source_images: Vec<usize>,
    /// What each **supplied** image in `source_images` holds, ascending by
    /// image and without repeats.
    ///
    /// Empty for every phase that reads only images the run writes, which is
    /// every phase this crate shipped before a run could be handed a second
    /// array.
    ///
    /// **Why it is recorded here and not derived.** The element type of an image
    /// the run writes is the fold of the chain up to it, and
    /// [`Decomposition::dtype_at`] computes it — there is nothing to store. A
    /// supplied input is produced by no phase, so there is no fold and no
    /// arithmetic that could answer; the only statement of what is in it is the
    /// one its readers make (`Chain::Source`'s `dtype`, `SourceInput::dtype`).
    /// So it is recorded beside the images it belongs to, derived from the chain
    /// by [`Decomposition::declare_source_images`] and verified against it by
    /// [`check_source_images`] — exactly the split `declare_dtypes` and
    /// `check_dtypes` have, and the same one `source_images` itself has.
    ///
    /// Two phases reading one supplied input must agree, and
    /// [`check_source_images`] says so by name if they do not.
    pub supplied_dtypes: Vec<(usize, Dtype)>,
    /// Whether this phase reads the image it is handed — image `p`, the one
    /// below it — at all.
    ///
    /// `true` for every phase that owns a slot, because a chain reads its input
    /// by construction, and `true` is therefore the derived default. It is
    /// `false` only for a fragment phase whose op declares
    /// `FragmentOp::reads_pixels() == false`: such a phase is handed an image and
    /// never touches it, and `fragment::fragment_phase` is what records that
    /// here.
    ///
    /// **It exists for the accounting and for nothing else.**
    /// [`Decomposition::exact_read_voxels`] is compared against a run's counter
    /// to the voxel, and the run — `strategy::run_fragment_task` — skips the
    /// input read entirely when the op declares it reads no pixels. Without this
    /// field the plan had no way to know that and charged for a fetch that never
    /// happened, which is the one failure mode that figure exists to catch, in
    /// the plan rather than in the execution.
    ///
    /// It changes no voxel and constrains nothing: `check` does not consult it,
    /// because what it describes is a property of the op and the op is not in
    /// the plan. A plan that arrives with it wrong is a plan whose read figure
    /// is wrong, which is exactly the fault it is here to make visible.
    pub reads_input_image: bool,
    /// Whether this phase waits for **all** of the previous phase rather than
    /// for the tasks whose valid regions cover what it fetches.
    ///
    /// `false` for every phase this crate shipped before the field existed, and
    /// it is set only by `fragment::fragment_phase`, from
    /// `FragmentOp::barrier`.
    ///
    /// **A declaration, not a hint.** Everything in `Hints` may be ignored,
    /// overridden or recomputed by a strategy and the run is still correct; this
    /// may not. A phase whose answer depends on every block of the phase below
    /// it, and which is run without waiting for every block of the phase below
    /// it, computes a plausible wrong answer whose wrongness depends on the
    /// schedule. So it is in the binding half of the plan: hashed into
    /// [`Decomposition::fingerprint`] when it is set, carried over the
    /// distributed wire, and checked against the op by
    /// `fragment::check_phase_work` — a plan whose record disagrees with its ops
    /// is a plan that fetches one thing and waits for another.
    ///
    /// **What it buys, and it is the reason it exists.** Without it, the only
    /// way to say "every block of the phase below has finished" is to *fetch*
    /// the whole volume, because [`crate::graph::TaskGraph::build`] derives the
    /// edges from the fetch and from nothing else. The fetch is then charged, per
    /// block, and the traffic is the block count times the volume. With it the
    /// ordering and the fetch stop being the same statement — the ordering is
    /// this bool, carried to every scheduler through
    /// [`crate::graph::TaskGraph::barriers`], and the fetch shrinks to what the
    /// op actually reads. See `docs/design/barriers.md`.
    ///
    /// **What it costs.** A barrier declared where it is not needed serialises
    /// two phases that could have overlapped, silently — the same class of
    /// mistake as a halo declared wider than an op needs, and no guard can catch
    /// either, because both are statements about what an op needs and only the
    /// op knows. It is *not* a general loss of pipelining: the phases before it
    /// still pipeline among themselves, the phases after it likewise, and the
    /// blocks within it still run concurrently.
    pub barrier: bool,
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
            source_images: Vec::new(),
            supplied_dtypes: Vec::new(),
            reads_input_image: true,
            barrier: false,
            blocks,
        }
    }

    /// The coordinate space **this phase** works in: what its cores are cut
    /// from, what its valid regions must tile, and the shape of the image it
    /// writes.
    ///
    /// It is derived from the grid rather than stored beside it, because two
    /// copies of one number are two numbers that can disagree, and a plan that
    /// disagrees with itself is worse than a plan that cannot express something.
    ///
    /// What it is *not* is the shape of what the phase reads. That is the image
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

    /// Say this phase also reads these images, through source leaves.
    ///
    /// Normalised on the way in — ascending, no repeats — because the list is
    /// fingerprinted and two plans that read the same images must hash the same
    /// whatever order the chain was walked in.
    pub fn with_source_images(mut self, images: impl IntoIterator<Item = usize>) -> Self {
        let mut images: Vec<usize> = images.into_iter().collect();
        images.sort_unstable();
        images.dedup();
        self.source_images = images;
        self
    }

    /// Say what the supplied inputs this phase reads hold.
    ///
    /// Normalised on the way in — ascending, no repeats — for the reason
    /// [`Self::with_source_images`] is: the list is fingerprinted.
    pub fn with_supplied_dtypes(mut self, held: impl IntoIterator<Item = (usize, Dtype)>) -> Self {
        let mut held: Vec<(usize, Dtype)> = held.into_iter().collect();
        held.sort_by_key(|(image, _)| *image);
        held.dedup();
        self.supplied_dtypes = held;
        self
    }

    /// Say whether this phase reads the image it is handed.
    ///
    /// Only a fragment phase ever passes `false`; see
    /// [`PhaseDecomposition::reads_input_image`].
    pub fn reading_input_image(mut self, reads: bool) -> Self {
        self.reads_input_image = reads;
        self
    }

    /// The images one block of this phase fetches: its own input image if it
    /// reads one, then every entry of [`Self::source_images`].
    ///
    /// **`index` is the phase's own position**, because a phase reads image
    /// `index` and writes image `index + 1`; the type does not carry its own
    /// number and inventing one here would be a second answer to a question the
    /// caller already knows.
    ///
    /// One home for the list, so that a caller counting fetches and a caller
    /// performing them cannot disagree: `phase_traffic` takes its
    /// `images_read` count from `.len()` of this, `strategy` reads exactly these
    /// images at `BlockGeometry::source`, and `simulate` charges exactly these.
    /// Before this existed the simulator charged for one image whatever the
    /// phase read, which under-priced every fused multi-input phase — the
    /// arrangement a scheduler would otherwise prefer.
    pub fn images_read(&self, index: usize) -> Vec<usize> {
        let mut images =
            Vec::with_capacity(usize::from(self.reads_input_image) + self.source_images.len());
        if self.reads_input_image {
            images.push(index);
        }
        images.extend(self.source_images.iter().copied());
        images
    }

    /// Say whether this phase waits for all of the previous phase.
    ///
    /// Only a fragment phase ever passes `true`; see
    /// [`PhaseDecomposition::barrier`].
    pub fn with_barrier(mut self, barrier: bool) -> Self {
        self.barrier = barrier;
        self
    }

    /// Give every block a fetch region in the image below's coordinate space.
    ///
    /// `map` is handed each block's geometry — index, core, read extent and all
    /// — and returns the region to fetch for it. It must be a function of the
    /// **block index and the plan**, never of the data: a `Decomposition` is
    /// parity-visible and is built without a source to look at.
    ///
    /// The mapping is applied once, here, and what is recorded is its result, so
    /// the plan carries the regions rather than a closure nobody can hash. What
    /// checks them is [`Decomposition::check`], which is the only place that
    /// knows what volume the image below has.
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

/// What an image **is**: given to the run, made on the way through, or the
/// answer.
///
/// The distinction [`Visibility`] cannot make, and the one a scheduler needs.
/// `Published` covers both ends of the run on the single shared ground that
/// somebody outside reads them, which gets image 0's *behaviour* right — never
/// freed — for a reason that is not the real one. The real one is the middle
/// column below.
///
/// | | produced by a phase? | recomputable by this run? | must exist when the run ends? |
/// |---|---|---|---|
/// | [`Input`](ImageKind::Input) | no — handed to the run | **never, at any price** | n/a: it existed before the run and is not the run's to free |
/// | [`Intermediate`](ImageKind::Intermediate) | yes | yes | no |
/// | [`Output`](ImageKind::Output) | yes | yes | **yes** |
///
/// An intermediate may be dropped and rebuilt — the memory-for-recomputation
/// trade. An input may not be, because there is no phase that produces it. The
/// two look identical through `Visibility`, and any policy that trades residency
/// for recomputation has to be able to tell them apart.
///
/// "Recomputable" and "must be there at the end" are separate columns on
/// purpose: an output is exactly as recomputable as an intermediate, and what
/// makes it an output is a **materialisation obligation** rather than a property
/// of the computation. Welding the two makes "I may drop this and rebuild it,
/// but I must rebuild it before I finish" unsayable, and that is a real state.
///
/// `docs/design/images-and-phases.md` specifies a rename in which this enum
/// replaces [`Visibility`] outright. The spelling half of that is done — this is
/// `ImageKind` and not `LevelKind` — and the removal is **not**, deliberately:
/// `Visibility` answers the narrower question every one of its callers actually
/// asks — *may this be freed* — and folding it away would make each of them
/// restate `Input | Output` for itself. The two are kept because they are two
/// questions, and this one is derived from that one so they cannot disagree.
///
/// **The argument deliberately does not turn on a count.** Two attempts to
/// state one disagreed — nine, then eight-and-none-in-the-consumer — and both
/// were wrong: `image_visibility` is read at seven sites here and **four in the
/// consumer**, and one of the eight matches a counter found was this file's own
/// definition. A number in a doc comment is a fact about the day it was written;
/// what carries the decision is that the two enums answer different questions,
/// which is true at any count above zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageKind {
    /// Handed to the run. No phase writes it and nothing can rebuild it.
    Input,
    /// Written by one phase, read by its readers, then dead.
    Intermediate,
    /// Written by a phase, and somebody outside the run reads it.
    Output,
}

/// Whether an image survives the run.
///
/// Derived from [`ImageKind`], which is the finer question: see that type for
/// what this one cannot say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// An input and the workflow output. Somebody outside the run reads these,
    /// so they exist when it ends.
    Published,
    /// Written by one phase, read by its readers, then dead.
    ///
    /// *Its readers*, plural, and not "read by exactly one phase": a source leaf
    /// is a second reader, so the general statement is the one
    /// [`Decomposition::readers_of_image`] makes — an image dies after its
    /// **last** reader.
    ///
    /// The reason this is worth naming: today every image of an `N`-phase plan
    /// is allocated at full volume for the whole run, so a twenty-stage chain
    /// holds twenty-one copies of the data at once. Only ever two of them are
    /// live. Saying which are which is what lets the environment free the rest.
    Internal,
}

/// The binding plan: what must be reproduced exactly for output to match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decomposition {
    /// The shape of **image 0** — the input the first phase reads.
    ///
    /// Not "the volume of the plan": every phase owns its own volume
    /// ([`PhaseDecomposition::volume`]), and image `p+1` is phase `p`'s. Image 0
    /// is the one image that is no phase's output, so it is the one that has to
    /// be stated here. [`Decomposition::volume_at`] is the accessor that reads
    /// either kind, and [`Decomposition::uniform_volume`] is the derived
    /// "everything agrees" answer the single-volume callers want.
    pub volume: [usize; 3],
    /// The element type of **image 0**, on the same argument. A phase that
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

    /// The number of images **the run writes into**: image 0 plus one per phase.
    ///
    /// **This is not the number of arrays a run touches**, and stopped being it
    /// when a run could be handed more than one. Supplied inputs are images —
    /// they are read through the same `source_images`, fetched through the same
    /// `Environment::read`, priced by the same byte accounting — but they are
    /// addressed in a disjoint high range ([`ImageId::SUPPLIED_BASE`]) and are not
    /// counted here, because every caller of this counts *what the plan fills
    /// in*: a chunk list, an image table, a bound on an image a phase may name.
    /// [`Self::n_supplied_inputs`] is the other half, and the two are added by
    /// nobody, deliberately.
    ///
    /// [`ImageId::SUPPLIED_BASE`]: ImageId::SUPPLIED_BASE
    pub fn n_images(&self) -> usize {
        self.phases.len() + 1
    }

    /// Every supplied input any phase reads, ascending and without repeats.
    ///
    /// Derived from `source_images` and not recorded: an array nothing reads is
    /// not an image of this plan, whatever the caller handed the environment.
    pub fn supplied_input_images(&self) -> Vec<usize> {
        let mut images: Vec<usize> = self
            .phases
            .iter()
            .flat_map(|phase| phase.source_images.iter().copied())
            .filter(|&image| is_supplied_image(image))
            .collect();
        images.sort_unstable();
        images.dedup();
        images
    }

    /// How many arrays this plan expects to be handed.
    pub fn n_supplied_inputs(&self) -> usize {
        self.supplied_input_images().len()
    }

    /// The shape of image `image`: image 0 is the input, image `p+1` is what
    /// phase `p` wrote, and a supplied input is in image 0's space.
    ///
    /// This is the accessor a caller that used to read `decomposition.volume`
    /// and meant "the space this phase reads" wants. It panics on an image that
    /// does not exist, like an index, because every caller of it holds a phase
    /// number it got from the plan.
    pub fn volume_at(&self, image: usize) -> [usize; 3] {
        match image {
            0 => self.volume,
            // A supplied input is read at the reading block's own fetch region
            // — a source leaf has reach 0 — so it has to be in the coordinate
            // space that fetch is stated in, and that is image 0's. Stated as a
            // rule rather than recorded per input, and it is not an assumption:
            // `check_source_images` compares this against the volume of every
            // phase that reads it and refuses the pair by name, so a plan that
            // reshapes and then reads a supplied input is caught at plan time.
            _ if is_supplied_image(image) => self.volume,
            _ => self.phases[image - 1].volume(),
        }
    }

    /// The element type of image `image`, folded from image 0: a phase that
    /// declares no `dtype` hands on the one it read.
    pub fn dtype_at(&self, image: usize) -> Dtype {
        if is_supplied_image(image) {
            // No phase writes it, so there is no chain to fold: the readers'
            // declaration is the only statement there is, and
            // `declare_source_images` recorded it. Falling back to image 0's
            // type for an unread supplied image is the honest answer to a
            // question about an array this plan does not read — and
            // `check_source_images` is what makes sure an image a phase *does*
            // read was recorded.
            return self
                .phases
                .iter()
                .flat_map(|phase| phase.supplied_dtypes.iter())
                .find(|(named, _)| *named == image)
                .map(|(_, dtype)| *dtype)
                .unwrap_or(self.dtype);
        }
        let mut dtype = self.dtype;
        for phase in self.phases.iter().take(image) {
            dtype = phase.dtype.unwrap_or(dtype);
        }
        dtype
    }

    /// Every phase that reads `image`, ascending.
    ///
    /// One predicate, applied to every phase: **phase `p` reads image `i` when
    /// `i == p` and the phase reads its input image at all, or when the phase's
    /// `source_images` names `i`.** No special case, and the loop is ascending
    /// by construction so nothing is sorted afterwards.
    ///
    /// **The refcount.** Before source leaves existed this was always a single
    /// phase — image `p` is phase `p`'s input and nobody else's — and the whole
    /// lifetime rule was written to that special case. A source leaf is a
    /// second reader, so the general statement is the one the design record
    /// asks for: *an image dies after its last reader*.
    ///
    /// # Why `reads_input_image` and not "phase `p` reads image `p`"
    ///
    /// It used to be the latter, unconditionally, and that is false for exactly
    /// one kind of phase: a fragment phase whose op declares
    /// `reads_pixels() == false` is handed an image and never touches it. The
    /// plan already records the truth in
    /// [`PhaseDecomposition::reads_input_image`], which `fragment_phase` put
    /// there. The over-count was **one image for one phase** — image `p`'s only
    /// credited reader was phase `p` and its writer is phase `p - 1`, so the
    /// claim could be wrong but not wrong by two — and what it cost was one
    /// whole image at the plan's element type, held for one phase: kilobytes at
    /// a test lattice, gigabytes at a real one.
    ///
    /// # The two things that had to move with it
    ///
    /// Narrowing this alone is wrong twice, and both were found by running the
    /// suite against a candidate rather than by reading the call sites.
    ///
    /// * The old `phase != image` guard on the `source_images` loop skipped a
    ///   phase that reads an image through a source leaf *and* is the phase that
    ///   image is the input of. That reader existed only by the unconditional
    ///   push, so narrowing the push without widening the loop drops it, and
    ///   every run in `tests/region_tabulation.rs` fails with *image 1 was
    ///   discarded after phase 0*. The guard is gone and the predicate above is
    ///   what replaces both halves.
    /// * An image with no reader at all would then be named by **no** phase, so
    ///   [`Self::images_dead_after`] would never free it — strictly worse than
    ///   the over-count. That is where the zero-reader rule went.
    ///
    /// The last image has no reader either, which is why it is `Published`;
    /// [`Self::images_dead_after`] now names it after its writer, and
    /// [`Self::image_visibility`] is what keeps it alive.
    pub fn readers_of_image(&self, image: usize) -> Vec<usize> {
        self.phases
            .iter()
            .enumerate()
            .filter(|(phase, decomposition)| {
                (*phase == image && decomposition.reads_input_image)
                    || decomposition.source_images.contains(&image)
            })
            .map(|(phase, _)| phase)
            .collect()
    }

    /// The images whose **last** reader is `phase`: what dies when this phase
    /// finishes every one of its tasks.
    ///
    /// This is the quantity the executor wants, and stating it this way is what
    /// keeps the executor from having to know whether the rule is "one reader"
    /// or "several". A plan with no source leaf answers `[phase]`, which is the
    /// image the phase read — the old rule, unchanged, as an instance of the
    /// new one.
    ///
    /// Whether an image may be freed at all is still [`Self::image_visibility`]'s
    /// question, and pinning is still the caller's; neither is folded in here,
    /// because this is a fact about the plan and those two are policy.
    ///
    /// # An image nothing reads dies as soon as it is written
    ///
    /// The same rule — *after its last reader* — at zero readers, and it is the
    /// half [`Self::peak_image_bytes`] already applied while this one did not.
    /// Without it an unread image is named by no phase at all and is never
    /// freed, which is worse than any over-count; and unread images are not
    /// exotic, because the run's output is one and a fragment phase that reads
    /// no pixels leaves one behind it.
    ///
    /// The writer of image `i` is phase `i - 1`. Image 0 has no writer, is the
    /// run's input and is `Published`, so it is named by nobody when nothing
    /// reads it — which is right: there is no phase after which it becomes dead.
    pub fn images_dead_after(&self, phase: usize) -> Vec<usize> {
        (0..self.n_images())
            .filter(|&image| match self.readers_of_image(image).last() {
                Some(&last) => last == phase,
                None => image > 0 && image - 1 == phase,
            })
            .collect()
    }

    /// Whether policy permits freeing `image` at all — the half
    /// [`Self::images_dead_after`] deliberately leaves out.
    ///
    /// `Internal` **or** released, and `keep_images` wins over both. The first
    /// is the plan's answer — an image that exists only to get from one phase to
    /// the next — and the second is the caller's, for an image the plan calls
    /// [`Visibility::Published`] because of where it sits and that this caller does not want
    /// back. A caller that named the same image in `release_images` and
    /// `keep_images` has contradicted itself, and the reading that cannot lose
    /// data is the one taken.
    ///
    /// **One function because it used to be three.** This predicate was written
    /// out at `strategy`'s discard loop, inside
    /// [`Self::peak_image_bytes_with`]'s residency walk, and again in
    /// `simulate`'s — the middle one describing itself as "word for word the
    /// executor's rule", which is an accurate confession rather than a
    /// reassurance. Two of the three then disagreed with the first about an
    /// image nothing reads; see [`Self::images_freed_after`].
    pub fn image_freeable(
        &self,
        image: usize,
        released: &BTreeSet<ImageId>,
        kept: &BTreeSet<ImageId>,
    ) -> bool {
        let id = ImageId::from(image);
        (self.image_visibility(image) == Visibility::Internal || released.contains(&id))
            && !kept.contains(&id)
    }

    /// The images a run may free once `phase` has finished: dead after it
    /// ([`Self::images_dead_after`]) **and** freeable ([`Self::image_freeable`]).
    ///
    /// The one place the two halves are put together, and therefore the one
    /// statement of what the executor does. `strategy`'s discard loop, this
    /// type's residency walk and `simulate`'s all call it.
    ///
    /// # The reconciliation, stated rather than left to an invariant
    ///
    /// The three former copies did not agree about **an image nothing reads**.
    /// [`Self::images_dead_after`] answers `i - 1 == phase` for one — it dies
    /// after the phase that wrote it — while both residency walks matched on
    /// `readers_of_image(..).last()` and treated `None` as *drop now*. Those are
    /// different rules. They agreed in practice only because an image enters the
    /// live set when its writer starts, which made the difference unreachable —
    /// agreement resting on an invariant stated in another function, which is
    /// the failure mode transcription produces.
    ///
    /// **`images_dead_after`'s answer is the one kept**, because the executor
    /// is what actually runs and that is the rule it followed. Two edge cases
    /// move as a result, both in the safe direction: **image 0** and a
    /// **supplied input**, each unread and named in `release_images`, were
    /// dropped by the residency walks at the first phase boundary and are not
    /// freed by the executor at all — `images_dead_after` excludes image 0 by
    /// construction and never names a supplied input, which `n_images` does not
    /// count. The walks now report those bytes as resident, which is what the
    /// run holds.
    pub fn images_freed_after(
        &self,
        phase: usize,
        released: &BTreeSet<ImageId>,
        kept: &BTreeSet<ImageId>,
    ) -> Vec<usize> {
        self.images_dead_after(phase)
            .into_iter()
            .filter(|&image| self.image_freeable(image, released, kept))
            .collect()
    }

    /// Whether an image survives the run, or exists only to get from one phase
    /// to the next.
    ///
    /// **Derived, not recorded.** Image 0 is the input and the last image is the
    /// output; everything between them is written by one phase, read by at least
    /// one phase, and then dead. Nothing in the plan needs to say so, and a
    /// field that could disagree with the arithmetic would be a field that
    /// eventually does.
    ///
    /// *Which* phase is the last reader is [`Self::images_dead_after`]'s
    /// question and has moved; whether the image is somebody else's to keep has
    /// not, because a source leaf is inside the run and cannot make an
    /// intermediate outlive it.
    ///
    /// This is deliberately **not** part of the binding half of the plan.
    /// Discarding an intermediate cannot change a voxel of the output, so
    /// keeping one is a decision about debuggability rather than about the
    /// answer — it belongs in `Hints`, next to every other advisory value, and
    /// the fingerprint is unchanged by it.
    pub fn image_visibility(&self, image: usize) -> Visibility {
        match self.image_kind(image) {
            ImageKind::Input | ImageKind::Output => Visibility::Published,
            ImageKind::Intermediate => Visibility::Internal,
        }
    }

    /// What `image` **is**: handed to the run, made on the way through, or the
    /// answer.
    ///
    /// **Derived, on the same argument [`Self::image_visibility`] is derived**,
    /// and now with something to say that the arithmetic it replaced could not:
    /// image 0 and every supplied input are `Input` because no phase writes
    /// them, the last image a phase writes is `Output`, and everything between
    /// is `Intermediate`. The old rule collapsed the first and the last into one
    /// answer and had no way to distinguish "must not be freed because nothing
    /// could rebuild it" from "must not be freed because somebody is waiting for
    /// it".
    pub fn image_kind(&self, image: usize) -> ImageKind {
        if image == 0 || is_supplied_image(image) {
            ImageKind::Input
        } else if image + 1 >= self.n_images() {
            ImageKind::Output
        } else {
            ImageKind::Intermediate
        }
    }

    /// The shape of the workflow's output: the last phase's volume.
    pub fn output_volume(&self) -> [usize; 3] {
        self.volume_at(self.n_phases())
    }

    /// The one volume every image is in, when they are all the same.
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

    /// The one element type every image is in, when they are all the same.
    pub fn uniform_dtype(&self) -> Option<Dtype> {
        self.phases
            .iter()
            .all(|phase| phase.dtype.is_none_or(|dtype| dtype == self.dtype))
            .then_some(self.dtype)
    }

    /// Record the element type each phase writes, from the ops that write it.
    ///
    /// Consulted by a strategy at the end of `decompose`, so that a shipped
    /// planner produces a plan whose images are the width its chain needs rather
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

    /// Record which images each phase reads besides its own input, from the
    /// source leaves in its slots.
    ///
    /// The counterpart of [`Self::declare_dtypes`], and separate from
    /// [`check_source_images`] for the same reason: this one *derives* from a
    /// chain the caller is holding, that one *verifies* a plan that may have
    /// arrived from anywhere.
    ///
    /// A phase with no source leaf is left declaring nothing, which is what
    /// keeps a plan that does not use the feature fingerprinting exactly as it
    /// did before the feature existed.
    pub fn declare_source_images(&mut self, chain: &Chain) -> Result<()> {
        let slots = chain.slots();
        // Volumes first, because a source input's reach is stated over the
        // phase's own anchoring volume and `self.phases` is about to be borrowed
        // mutably. Two loops rather than a cell: the quantity is read-only.
        let volumes: Vec<[usize; 3]> = (0..self.phases.len())
            .map(|phase| self.volume_at(phase))
            .collect();
        for (index, phase) in self.phases.iter_mut().enumerate() {
            // **A phase that owns no slot is left alone rather than emptied.**
            // This derives from the chain, and a phase with no slot of the chain
            // has nothing in the chain that could say what it reads — a fragment
            // phase's second image is declared by its op and recorded by
            // `fragment::fragment_phase`, which is the only thing holding the
            // op. Overwriting it here would silently drop it, and the run would
            // then fetch nothing while the op asked for something.
            if phase.slots.is_empty() {
                continue;
            }
            let mut declared: Vec<crate::op::SourceInput> = Vec::new();
            for &slot in &phase.slots {
                let Some(node) = slots.get(slot) else {
                    continue;
                };
                for input in node.source_inputs(volumes[index])? {
                    if !declared.iter().any(|held| held.image == input.image) {
                        declared.push(input);
                    }
                }
            }
            let mut images: Vec<usize> = declared.iter().map(|input| input.image.index()).collect();
            images.sort_unstable();
            images.dedup();
            // Only the supplied ones. An image the run writes has its element
            // type in the fold and recording a second copy of it would be a
            // second number to disagree with the first.
            let mut held: Vec<(usize, Dtype)> = declared
                .iter()
                .filter(|input| input.image.is_supplied())
                .filter_map(|input| input.dtype.map(|dtype| (input.image.index(), dtype)))
                .collect();
            held.sort_by_key(|(image, _)| *image);
            held.dedup();
            phase.source_images = images;
            phase.supplied_dtypes = held;
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
    /// **Once per image read, and a phase with source leaves reads more than
    /// one.** Each one is fetched at the same region as the input — reach 0 —
    /// so the multiplier is the number of images the phase actually reads. A
    /// figure that counted only the input would be under by that factor for
    /// precisely the plans this number is most worth checking, and it is
    /// compared against a run's counter to the voxel.
    ///
    /// **The input image is one of them only when the phase reads it.** A
    /// fragment phase whose op declares `reads_pixels() == false` is handed
    /// image `p` and never fetches it — `strategy::run_fragment_task` skips the
    /// read outright — so the multiplier is `source_images.len()` and not one
    /// more. This used to be `1 + source_images.len()` unconditionally, which
    /// charged such a phase for a fetch that does not happen. It was invisible
    /// while the only non-reading fragment phases had no source images either
    /// and thus no `Environment::read` to be compared against; it stopped being
    /// invisible when phases appeared that read a second array and not their
    /// own, and it was over by one whole halo-inflated image each.
    ///
    /// A phase that reads neither its input nor any source image scores zero,
    /// and that is the right answer rather than a degenerate one:
    /// `fragments -> fragments` merges move no voxels at all, only sidecar
    /// bytes, and this function has never counted those — its own header calls
    /// itself the figure to compare against what a run counted, and a run
    /// counts a fragment gather through the sidecar, not through `read`.
    pub fn exact_read_voxels(&self) -> Vec<usize> {
        self.phases
            .iter()
            .map(|phase| {
                let per_image: usize = phase.blocks.iter().map(|block| block.source.voxels()).sum();
                let images = usize::from(phase.reads_input_image) + phase.source_images.len();
                per_image * images
            })
            .collect()
    }

    /// The whole-array bytes this plan's images occupy at the worst moment of a
    /// run, from the plan alone.
    ///
    /// # What the number is
    ///
    /// Exactly one thing: **the images this plan names, sized by the volumes and
    /// element widths it declares, summed over the phase boundary at which most
    /// of them are simultaneously alive.** An image is alive from the phase that
    /// writes it until its last reader finishes; a
    /// [`Visibility::Published`] image — image 0, the
    /// output, a supplied input — is never discarded, because
    /// `ArrayEnvironment` does not discard them. Every input is an integer the
    /// plan states, so two runs of the same plan give the same answer and two
    /// plans differ here only where they really do name different arrays.
    ///
    /// It is the figure that decides whether a stage is worth *attempting* at a
    /// given size, and it is the right thing to compare two plans on when the
    /// question is about arrays the plan names.
    ///
    /// # What the number is not, measured rather than hedged
    ///
    /// **It is not a bound on a run's resident bytes, in either direction**, and
    /// the two ways it misses were measured rather than reasoned about — on the
    /// same plan at two block sizes, against a counting allocator:
    ///
    /// * **at one block the run saved more than this figure predicted, by 25%.**
    ///   A `Chain::Parallel` allocates a buffer per branch at the block's read
    ///   extent, and narrowing a reach shrinks every one of them. No
    ///   `Decomposition` can see those buffers: they are not images, they have no
    ///   entry here, and nothing in this walk moves when they do.
    /// * **at `32^3` the run saved half what this figure predicted.**
    ///   `ImageStore::pending` allocates an image lazily at first write and frees
    ///   an internal one after its last reader, so a five-image plan never holds
    ///   five and the peak this walk computes is a moment the run does not have.
    ///
    /// So a caller comparing two plans gets a figure that is **directionally
    /// right and quantitatively not** — the plan that names fewer and smaller
    /// arrays does hold less, and by how much is a question only a run answers.
    /// A caller sizing a machine wants a measured `VmHWM`, and
    /// `crate::budget::MemoryBudget` for what a run is *permitted* to hold.
    ///
    /// # Why `work`
    ///
    /// Phase `p` writes image `p + 1` — **unless it does not**. A fragment phase
    /// that writes no pixels writes no image, and a plan whose last phase is one
    /// has an image slot nothing ever fills. Whether a phase writes is the op's
    /// answer and not the plan's ([`crate::fragment::PhaseWork::writes_an_image`]),
    /// which is the same reason [`predicted_cost`] takes this argument, and it is
    /// enforced the same way: `&[]` is fine for an all-pixel plan and a slotless
    /// phase with no entry is **refused** rather than assumed to write.
    ///
    /// The read side needs no argument, because the plan already records it:
    /// [`PhaseDecomposition::reads_input_image`] is what `fragment_phase` put
    /// there and what [`Self::exact_read_voxels`] already counts with.
    ///
    /// # Why this does not call `images_dead_after`, and why it no longer
    /// # disagrees with it
    ///
    /// It used to say something the other could not: [`Self::readers_of_image`]
    /// counted phase `p` as a reader of image `p` unconditionally, and a
    /// fragment phase that reads no pixels is not one, so the walk here was
    /// exact and [`Self::images_dead_after`] was not. The zero-reader rule this
    /// walk applied — *an image nothing reads dies as soon as it is written*,
    /// which is the same rule at zero readers — was likewise only here.
    ///
    /// **Both have moved to where they belong.** `readers_of_image` now consults
    /// [`PhaseDecomposition::reads_input_image`] and `images_dead_after` applies
    /// the zero-reader rule, so the two agree and the executor frees what this
    /// predicts. The `retain` below is still written out rather than delegated,
    /// because it asks the question in the walk's own terms — *is there a reader
    /// after this phase* — and that is one comparison rather than a vector per
    /// image per phase.
    /// **The two halves of what a run is resident in, named together.**
    ///
    /// A budget question has two terms and they behave differently, which is
    /// why one number was never enough and why
    /// [`Constraints::affords_working_set`] deliberately answers only half of
    /// it:
    ///
    /// * [`Residency::working_set_bytes`] — the block tiles in flight. **Scales
    ///   with the block and the concurrency**, so a finer cut reduces it, and a
    ///   refusal on it is actionable.
    /// * [`Residency::image_bytes`] — the whole-volume images alive at the
    ///   worst phase, from [`Self::peak_image_bytes`]. **Invariant to the cut**:
    ///   images are allocated at the phase's volume, so a finer grid is an
    ///   iteration order and not a residency policy. It moves for a narrower
    ///   dtype, an out-of-core environment or `Hints::release_images`, and for
    ///   nothing the partition search can choose.
    ///
    /// Reported rather than enforced, and that is the point. Peak residency is a
    /// **soft boundary**: memory that is held and being read is a run going
    /// fast, and only memory that is held and dead is a defect. A hard gate here
    /// would refuse plans whose residency is entirely working, and — because the
    /// image half answers the same for every candidate — it would refuse them
    /// with no candidate to suggest.
    ///
    /// `work` is the fragment work, as [`Self::peak_image_bytes`] takes it;
    /// `&[]` is correct and exact for an all-pixel plan.
    ///
    /// # Two ways this understates, both stated rather than fixed
    ///
    /// **It is a report, and the gate is elsewhere.** Nothing is admitted on
    /// this figure — [`Constraints::affords_working_set`] decides that, and it
    /// charges `working_set x concurrency` *unclamped*, which over-states. So
    /// the safe direction is preserved where it matters and these two limits
    /// cost a reader accuracy rather than costing a run its memory.
    ///
    /// * **The working-set term is a max over phases, and the executor can have
    ///   two phases in flight.** `strategy::execute` fills a wave from whatever
    ///   is ready, and a phase's successor becomes ready long before the phase
    ///   finishes, so blocks of two phases really do coexist. The true figure is
    ///   bounded by the sum of the two largest phases rather than by the
    ///   largest. Not summed here because "how many of each" is a scheduling
    ///   fact and this type is the binding half of a plan; the simulator is
    ///   where that question has an answer.
    ///
    /// The image term does not have that problem: it is exact, and a consumer
    /// suite pins it against a measured run to the byte — `56.200 GiB` of images
    /// predicted where a process watching `VmHWM` measured `56.200`.
    ///
    /// # Why it takes the chain
    ///
    /// The working-set term is the same quantity
    /// [`PhaseCost::working_set_bytes_per_block`] states, and it used to be
    /// computed here a second time with a hardcoded `x 2.0` — carrying, in that
    /// method's own words, "that field's known gap verbatim: a phase reading
    /// three images is charged as if it read one". When the gap was closed there
    /// the two immediately disagreed, which is the failure mode this crate warns
    /// about everywhere it states a quantity twice.
    ///
    /// So it is stated once. The chain is what the count needs — a phase's
    /// buffers are its slots' — and it is a parameter rather than something read
    /// off the plan because a `PhaseDecomposition` does not hold the chain and a
    /// recorded copy would be a third statement of the same number.
    pub fn residency(
        &self,
        chain: &Chain,
        constraints: &Constraints,
        work: &[crate::fragment::PhaseWork<'_>],
        released: &BTreeSet<ImageId>,
        kept: &BTreeSet<ImageId>,
    ) -> Result<Residency> {
        let workers = constraints.expected_concurrency.max(1);
        // The worst phase's, because residency is a peak and the phases do not
        // overlap. The per-block factor is
        // `PhaseCost::working_set_bytes_per_block`'s own — every block buffer
        // alive at once, not the two a hardcoded `x 2.0` used to assume — and it
        // is derived from the same slots by the same helper, so the two cannot
        // drift.
        let slots = chain.slots();
        let working_set_bytes = self
            .phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                let bytes = phase.dtype.unwrap_or(self.dtype).size_of() as f64;
                // **Clamped to the blocks that exist.** `affords_working_set`
                // multiplies by the concurrency unclamped, which is the safe
                // direction for a budget — over-charging refuses a plan that
                // would have run — but it is simply wrong for a report: at one
                // block it said `1057.889 GiB` for forty workers, forty copies
                // of a block only one worker can hold. A phase runs
                // `min(workers, n_blocks)` blocks at once and no more.
                let in_flight = workers.min(phase.grid.n_blocks()).max(1) as f64;
                let buffers = if phase.slots.is_empty()
                    || phase.slots.iter().any(|&slot| slot >= slots.len())
                {
                    // A fragment or iterative phase owns no slot of the chain,
                    // so the chain cannot speak for what it holds; the phase's
                    // own input and output are all this can claim.
                    phase.images_read(index).len() + 1
                } else {
                    phase.images_read(index).len() + 1 + resident_buffers_of(&slots, &phase.slots)
                };
                resident_voxels_per_block(&phase.grid, &phase.halo)
                    * bytes
                    * buffers as f64
                    * in_flight
            })
            .fold(0.0_f64, f64::max)
            .max(0.0) as u64;
        Ok(Residency {
            image_bytes: self.peak_image_bytes_with(work, released, kept)?,
            working_set_bytes,
            sidecar_bytes: self.gathered_sidecar_bytes(work),
        })
    }

    /// The largest a **barrier gather** gets: every contributing block's
    /// fragment, resident at once, from the declared
    /// [`crate::fragment::SidecarSize`] of each stream.
    ///
    /// # Why this is a residency term at all
    ///
    /// A barrier reduces over the whole phase, so the blob it folds holds one
    /// fragment per block simultaneously. Under `Coverage::EveryBlock` that is
    /// `n_blocks x payload` at one instant — and until this existed, nothing
    /// budgeted it: `exact_read_voxels` says outright that it "has never counted
    /// those", and `Residency` had two terms, both of which **fall** as the cut
    /// gets finer. A fixed per-block payload totals *more* as it gets finer, so
    /// a planner shrinking the working set was free to grow an unbudgeted peak.
    ///
    /// # What an undeclared stream contributes, and why it is zero
    ///
    /// Zero, and that is a statement about the declaration rather than about the
    /// stream: `SidecarSize::Unstated` means nobody said, and inventing a number
    /// here would be the plausible constant this whole mechanism exists to
    /// avoid. `tests/one_freeing_rule.rs` is what keeps a shipped stream from
    /// being in that state.
    ///
    /// # And why it is a bound rather than a measurement
    ///
    /// A budget refuses on it, and a refusal has to rest on something always
    /// true. `Stats::sidecar_bytes_written` is the measurement, and it is what a
    /// *choice* between two plans should read; the two are kept apart on purpose.
    /// See `SidecarSize::is_tight` for the streams whose bound is loose enough
    /// that only the measurement is usable.
    pub fn gathered_sidecar_bytes(&self, work: &[crate::fragment::PhaseWork<'_>]) -> u64 {
        self.phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                let Some(crate::fragment::PhaseWork::Fragments(op)) = work.get(index) else {
                    return 0;
                };
                let outputs = op.outputs();
                phase
                    .blocks
                    .iter()
                    .map(|block| {
                        let core = block.core.shape3();
                        let read = block.read.shape3();
                        outputs
                            .iter()
                            .filter_map(|output| output.size.bytes_at_most(core, read))
                            .sum::<u64>()
                    })
                    .sum()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn peak_image_bytes(&self, work: &[crate::fragment::PhaseWork<'_>]) -> Result<u64> {
        self.peak_image_bytes_with(work, &BTreeSet::new(), &BTreeSet::new())
    }

    /// [`Self::peak_image_bytes`], told what the caller releases and keeps.
    ///
    /// **The plain form predicts a run nobody makes.** The executor frees an
    /// image after its last reader when it is `Internal` **or** named in
    /// `Hints::release_images`, and never when it is named in
    /// `Hints::keep_images`; the walk here applied only the first clause, so it
    /// over-stated any plan whose caller released a `Published` image and
    /// under-stated any plan that kept one.
    ///
    /// The size of that gap on the shipped binarisation at the `404 x 1304 x
    /// 3369` tile over `float64` is **13.223 GiB** — image 0, the run's own
    /// input, which is `Published` because of where it sits rather than because
    /// anything reads it back. Predicting **56.200 GiB** where
    /// `tests/residency_across_cuts.rs` measures **42.977** is not a margin; it
    /// is the walk answering about a different run.
    ///
    /// `released` and `kept` are `Hints::release_images` and
    /// `Hints::keep_images`. They are passed rather than reached for because a
    /// `Decomposition` is the binding half of a plan and a `Hints` is the
    /// advisory half — this crate does not let the first depend on the second,
    /// and a caller holding a `Plan` holds both.
    pub fn peak_image_bytes_with(
        &self,
        work: &[crate::fragment::PhaseWork<'_>],
        released: &BTreeSet<ImageId>,
        kept: &BTreeSet<ImageId>,
    ) -> Result<u64> {
        let bytes_of = |image: usize| -> u64 {
            let volume = self.volume_at(image);
            volume.iter().product::<usize>() as u64 * self.dtype_at(image).size_of() as u64
        };
        // Every image the run holds: the ones the plan fills in, and the
        // supplied inputs, which `n_images` deliberately excludes because every
        // other caller of it counts what the plan *writes*. A supplied array is
        // as resident as image 0 and is `Published` for the same reason.
        let mut live: Vec<(usize, u64)> = vec![(0, bytes_of(0))];
        for image in self.supplied_input_images() {
            live.push((image, bytes_of(image)));
        }
        let mut peak: u64 = live.iter().map(|&(_, bytes)| bytes).sum();
        for (index, phase) in self.phases.iter().enumerate() {
            let writes = phase_traffic(index, phase, work.get(index))?.writes_an_image;
            let written = index + 1;
            if writes && written < self.n_images() {
                live.push((written, bytes_of(written)));
            }
            let now: u64 = live.iter().map(|&(_, bytes)| bytes).sum();
            if now > peak {
                peak = now;
            }
            // Freed after this phase, by the executor's own rule and not by a
            // second statement of it. See `images_freed_after`, which is the
            // only place the two halves — dead after this phase, and freeable
            // at all — are put together.
            let freed = self.images_freed_after(index, released, kept);
            live.retain(|&(image, _)| !freed.contains(&image));
        }
        Ok(peak)
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
            // phase that reads no second image contributes nothing, so every
            // plan built before source leaves existed fingerprints as it did.
            // Which image an arm reads changes voxels, so a plan that uses one
            // must not collide with a plan that reads another.
            if !phase.supplied_dtypes.is_empty() {
                for (image, dtype) in &phase.supplied_dtypes {
                    image.hash(&mut hasher);
                    dtype.numpy_name().hash(&mut hasher);
                }
            }
            if !phase.source_images.is_empty() {
                phase.source_images.hash(&mut hasher);
            }
            // Hashed only when it is set, on `source_images`' argument: a
            // plan with no barrier in it fingerprints exactly as it did before
            // the field existed, so the frozen fingerprints every op is pinned
            // by are unmoved. It *is* hashed when set, unlike
            // `reads_input_image` below, and the difference is the one that
            // matters: this changes which tasks a task waits for, and two plans
            // that wait for different things may answer differently. A plan
            // that arrives off the wire with it dropped would run, produce a
            // schedule-dependent answer, and agree with the sender's
            // fingerprint.
            if phase.barrier {
                phase.barrier.hash(&mut hasher);
            }
            // `reads_input_image` is deliberately **not** hashed. Every other
            // field here can change a voxel: which images an arm reads, what
            // type a phase writes, where a block fetches from. That one cannot
            // — it says a phase does not touch an image it was never going to
            // use, so the same plan with it right and with it wrong produces
            // identical output and differs only in a *predicted* read count.
            // Hashing it would renumber the plans of every fragment op that
            // declines its own image, breaking the frozen fingerprints those
            // ops are pinned by, in exchange for discriminating between two
            // plans that cannot disagree about a voxel.
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
    /// exactly, and every block must fetch from inside the image it reads.
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
    /// * a block's `source` must lie inside the image it reads, which is the
    ///   part that used to be true by construction and now has to be verified.
    ///   The images chain — image 0 is `self.volume`, image `p+1` is phase `p`'s
    ///   — so a plan whose phases do not join up is caught here rather than
    ///   becoming two decompositions with no edge between them.
    ///
    /// **And a whole-axis reach on the image below is checked against the
    /// fetch.** `AxisReach::All` in `Frame::Source` says the op consumes the
    /// whole of that axis of the array it reads. Nothing in the halo arithmetic
    /// can confirm it — the halo is measured in this phase's own volume, and a
    /// phase that collapses the axis has an extent of 1 there — so the only
    /// place the claim can be met is `BlockGeometry::source`, and this is the
    /// only place that knows what the image below is shaped like. Without it the
    /// declaration is decoration: a projection whose fetch covers one plane of
    /// its axis, or half of it, produces a complete, well-formed volume of
    /// exactly the right shape and the wrong numbers, and every other guard
    /// passes it.
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
            // A reach in the image below's own lattice is satisfied by the fetch
            // region and by nothing else — there is no factor turning a step of
            // that lattice into a voxel of this one, so it contributes nothing to
            // the halo. A phase that declares one and then fetches its own read
            // extent has declared a dependency it has no way to meet, and that is
            // exactly the shape of the zero somebody writes to get past a guard.
            if !phase.reach.space().converts_to_voxels() && !phase.reads_across_grids() {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {index} ({}) states its reach as {} — steps of the image \
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
                        "decomposition phase {index} block {:?}: the region it reads from image \
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
            // Last, because it is the only check about what the plan *meant*
            // rather than about whether it is self-consistent: a whole-axis
            // reach in the image below's frame is a claim, and the fetch is the
            // only thing that can meet it.
            if matches!(phase.reach.space().frame, Frame::Source) {
                for axis in 0..3 {
                    if !matches!(phase.reach.axis(axis), AxisReach::All) {
                        continue;
                    }
                    for block in &phase.blocks {
                        let lo = block.source.start[axis];
                        let hi = lo + block.source.shape[axis];
                        if lo == 0 && hi == source_volume[axis] {
                            continue;
                        }
                        return Err(Error::InvalidArgument(format!(
                            "decomposition phase {index} ({}) declares reach {} — the whole of \
                             axis {axis} of image {index} — and block {:?} fetches {lo}..{hi} of \
                             that axis, where the whole of it is 0..{}. A whole-axis reach in the \
                             image below's frame is a claim about what each block reads, and only \
                             the fetch can meet it: the halo is measured in this phase's own \
                             volume, which is {} voxel(s) on axis {axis}, so no halo widens the \
                             fetch. Every block has to fetch 0..{} \
                             (`PhaseDecomposition::with_sources`).",
                            phase.names.join(">"),
                            phase.reach,
                            block.index,
                            source_volume[axis],
                            volume[axis],
                            source_volume[axis],
                        )));
                    }
                }
            }
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
                rebuilt.source_images = phase.source_images.clone();
                rebuilt.supplied_dtypes = phase.supplied_dtypes.clone();
                rebuilt.reads_input_image = phase.reads_input_image;
                rebuilt.barrier = phase.barrier;
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
/// **The defaults are a seed, and they are not apologised for.** There may be
/// no first run — a cold planner has to plan *something* — so `1.0` everywhere
/// is what the planner starts from, and it is the right thing to start from
/// whatever its absolute accuracy: what a cold planner needs is that a median
/// costs more than a map and an opening more than the erosion inside it, and
/// `ops`' constants have that ordering right even where their scale is out by
/// 2.7x. Nobody should "fix" these by chasing precision that measurement will
/// supply anyway. [`crate::statistics`] is where the measurement goes: it
/// accumulates nanoseconds per unit of *declared* cost from real runs and hands
/// back a calibrated `CostModel`, so the seeds are displaced by evidence about
/// the machine that will do the work rather than by a better guess.
///
/// The unit is therefore whatever the caller's weights are denominated in.
/// Untouched, that is "one voxelwise map's worth of work per voxel". Calibrated
/// by a `statistics::Snapshot`, it is **nanoseconds** — a wholesale change of
/// unit, which is why calibration replaces all four weights together rather
/// than any one of them.
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
/// **There is deliberately no per-block overhead price either, and that one is
/// measured rather than argued.** Every weight here is per voxel, so a grid of
/// many small blocks is charged nothing for being many. That looks like the
/// obvious gap whenever the search prefers a finer grid than the machine does,
/// and on the one case where it has been checked it is not the gap. Measured
/// through [`crate::strategy::execute`] over a `256^3` volume with a zero-reach
/// op — so that redundancy is 1 at every edge and the only thing that moves is
/// the block count — the slope between adjacent grids is:
///
/// ```text
/// 256^3 f64, zero-reach map, best of 9, us of wall clock per extra block
/// edge   blocks   1 worker   40 workers
///  128        8          -            -
///   64       64          -       +336.9      (the pool filling up, not overhead)
///   32      512      -33.1         +3.3
///   16     4096      +57.8        +48.3
///    8    32768     +347.7       +291.0      (4 kB blocks; the scheduler, not a grid)
/// ```
///
/// **Zero within noise at any block a planner would choose**, and tens of
/// microseconds even at 32 kB blocks. The case that motivated looking — the
/// thinning over a `404 x 1304 x 3369` tile, where the model preferred a 128
/// grid that measured 1.38x slower than a 256 one — needed about **14.5 ms per
/// block** to flip, which is three orders of magnitude away. The defect was in
/// [`BlockGrid::mean_core_voxels`]'s territory instead, and a per-block constant
/// large enough to have papered over it would have been fitted to that one
/// table rather than measured. If a future case really does want this term,
/// measure it the same way first.
///
/// None of this can produce a wrong voxel. The partition decides where
/// intermediates land, not what is computed.
#[derive(Debug, Clone, PartialEq)]
pub struct CostModel {
    /// Per voxel of the **read extent** — core plus halo — over every image the
    /// phase reads.
    ///
    /// **It charges a halo voxel at core price, and that is a decision now
    /// rather than an accident.** A voxel fetched again because it fell in a
    /// neighbour's halo may cost very much less than one fetched for the first
    /// time, or rather more, and this weight cannot tell them apart. The
    /// measurements are in `docs/ops-survey/README.md`'s **G20** row and in
    /// `tests/halo_cost.rs` and `tests/halo_io_cost.rs`; the summary is:
    ///
    /// | path | halo bytes | halo time | effective weight |
    /// |---|---|---|---|
    /// | flat file, cold | `3.48x` | `1.32x` | `0.38` |
    /// | flat file, warm | `3.48x` | `2.98x` | `0.86` |
    /// | in memory, by region shape | — | — | `0.31`-`0.72` |
    /// | chunked store, gzip, halo 4 | `1.31x` | `5.33x` | **above `4`** |
    ///
    /// So a single weight **over**-charges a cold sequential read — the safe
    /// direction, since it biases against the fine cuts with wide halos that
    /// `docs/design/barriers.md` exists to prevent — is about right warm, and
    /// **under**-charges a chunked store.
    ///
    /// **Why a second weight is not the fix.** The plan is made before the
    /// storage path exists: [`crate::strategy::Strategy::decompose`] takes no
    /// `Environment` and is `data-blind` by contract, so no constant here can be
    /// chosen by path. And the coefficient this model is really missing is not
    /// halo against core but **cold against warm**, measured at `42x` at one
    /// block — twenty times larger, and supplied by nothing the planner can see.
    /// Refining the smaller term under an unmodelled larger one would make the
    /// price look better calibrated than it is.
    ///
    /// [`crate::statistics`] fits this from real runs, which is the only route
    /// by which a halo weight could ever be honest, and it fits one term today.
    pub read_cost_per_voxel: f64,
    pub write_cost_per_voxel: f64,
    pub compute_scale: f64,
    /// **Per-op-family corrections to the declared compute cost**, keyed by the
    /// slot's display name. Absent means `1.0`, and an empty table is
    /// [`Self::compute_scale`] alone — which is what this crate shipped with,
    /// bit for bit.
    ///
    /// # Why one scalar was not enough
    ///
    /// `compute_scale` is one number against a measured **57x** per-phase
    /// spread: the tile run put `combine` at 3.541, `smooth` at 98.329 and
    /// `skeletonize` at 201.397 nanoseconds per voxel. The declared
    /// `cost_per_voxel` each op carries is meant to capture that ratio, and it
    /// is a *seed* — nothing had ever compared it against a measurement, and
    /// `statistics::Term::ComputeOf` has recorded the per-family evidence all
    /// along under a doc that says outright it is "recorded and reported, **not
    /// used**", because there was nowhere in this struct to put it.
    ///
    /// This is that somewhere. `docs/design/planner-gaps.md` carries it as
    /// **G3** and the reason it matters is the shape of the objective:
    /// `strategy::phase_makespan` is a **roofline**, a max of a pool term and a
    /// channel term, so an error on one phase's compute does not scale the
    /// answer — it flips which side of the max binds, and with it which
    /// candidate wins. `assemble.rs` measured exactly that, sweeping declared
    /// compute over `1e-3..1e3` and watching the argmin walk from the coarsest
    /// candidate to the finest at every `workers > 1`.
    ///
    /// # Relative, not absolute, and that is deliberate
    ///
    /// A value here multiplies the slot's **declared** cost, alongside
    /// `compute_scale` rather than instead of it: the charge for a run is
    /// `sum over slots of declared x correction`, all of it times
    /// `compute_scale`. So `compute_scale` stays the single carrier of the
    /// unit — which is what `Term::Compute` means — and this table holds
    /// dimensionless corrections whose neutral value is `1.0`. A family whose
    /// measured rate matches the run-wide one contributes nothing, an absent
    /// family costs nothing to look up, and a model built by hand is unchanged.
    ///
    /// [`crate::statistics::Snapshot::calibrate`] fills it as
    /// `ComputeOf(family) / Compute`, from believable coefficients only — so a
    /// family measured once does not move a plan, on the same argument
    /// `REPRODUCTIONS` makes for every other term.
    ///
    /// A `BTreeMap` because a `Decomposition` must be deterministic and
    /// hashable: a hash over a `HashMap`'s iteration order is not a function of
    /// its contents.
    pub compute_of: BTreeMap<String, f64>,
    /// How much a worker's compute slows for each *other* worker running on its
    /// own machine — the same quantity, the same coefficient and the same
    /// Amdahl form as `simulate::Machine::contention`, and `0.0` is the
    /// perfectly-scaling pool this model assumed before the field existed.
    ///
    /// # Why the planner needs it
    ///
    /// [`crate::strategy::phase_makespan`]'s pool term is `cost_per_block x
    /// ceil(n / workers)`, which says forty workers are forty times one. The
    /// simulator says they are **2.41 times** one — that is the figure
    /// `simulate::MEASURED_CONTENTION` was fitted to — and the difference is not
    /// a scale factor, because the pool term is one side of a `max`. Measured on
    /// `costs/forty-cores`: the model priced the coarsest grid at **3.045** times
    /// its own argmin where the simulator put it at **1.018**, so it chose a grid
    /// with more, smaller blocks on the strength of workers that do not exist.
    /// That was the largest planner error in the whole scenario sweep.
    ///
    /// # Not a coefficient
    ///
    /// It is a property of the machine, like the worker count, and
    /// [`crate::statistics::Snapshot::calibrate`] does not fit it: nothing in an
    /// event stream separates "this block was slow because the machine was busy"
    /// from "this block was slow". It is carried through calibration untouched,
    /// and a caller who knows their machine states it —
    /// `scenario::Scenario::model` sets it from the machine the scenario
    /// describes.
    pub contention: f64,
    /// How many computers the run will use. `1` is one machine, and is what
    /// every figure this crate has recorded was taken on.
    ///
    /// The planner needs it for two terms and neither is a coefficient:
    /// [`Self::contention`] counts the workers of **one** node, and the channel
    /// bound is per node because each computer has its own link to storage. See
    /// [`crate::strategy::phase_makespan`], which is the only place either is
    /// read.
    ///
    /// **What it does not yet buy** is the term the node boundary most obviously
    /// wants: a charge for the chunks two machines both fetch, which
    /// `tests/multiple_computers.rs` measures at 3.26x the bytes on eight nodes
    /// and which nothing here prices. `docs/design/planner-gaps.md` carries that
    /// as the next item.
    pub nodes: usize,
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

impl CostModel {
    /// The correction for one op family: the table's entry, or `1.0`.
    ///
    /// **A method rather than a `get(..).unwrap_or(1.0)` at each call site**,
    /// so that "absent means neutral" is one statement. It is the property that
    /// makes the table safe to add to a shipped model: a family nobody has
    /// measured must cost exactly what it costed before the table existed.
    pub fn correction_for(&self, family: &str) -> f64 {
        self.compute_of.get(family).copied().unwrap_or(1.0)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            read_cost_per_voxel: 1.0,
            write_cost_per_voxel: 1.0,
            compute_scale: 1.0,
            // Empty, so the default model is the one this crate shipped with
            // and every recorded figure taken under it still stands.
            compute_of: BTreeMap::new(),
            // Off, and one machine: the perfectly-scaling single computer this
            // model assumed before either field existed.
            contention: 0.0,
            nodes: 1,
            order_conflict_penalty: 0.0,
            materialise_cost_per_voxel: 1.0,
        }
    }
}

/// What a strategy is allowed to spend.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraints {
    /// Whether a block may be cut into slabs for intra-block threading, and how
    /// many. See [`SlabPolicy`].
    ///
    /// **The default asks for one slab under the default concurrency**, so
    /// adding this field moved no recorded figure — asserted in
    /// `the_default_policy_asks_for_one_slab_under_the_default_constraints`
    /// rather than reasoned.
    pub slab_policy: SlabPolicy,
    /// Bytes available for in-flight blocks. `None` means unbounded.
    pub budget_bytes: Option<u64>,
    /// How many blocks a run is expected to hold at once. Part of the budget
    /// arithmetic, not of the decomposition: a strategy may run with fewer.
    pub expected_concurrency: usize,
    pub model: CostModel,
    /// Block edges the planner may choose between, per phase. Small by design —
    /// the search is `partitions x candidates^phases`, so this must stay short.
    ///
    /// **One consequence of the list being scalars and powers of two that is
    /// worth knowing before changing it.** An op may declare
    /// [`crate::reach::AxisReach::Aligned`] — a halo that shrinks when the block
    /// edge is a whole number of some stride of its own — and the discount is
    /// available exactly on the rungs the stride divides. The default ladder is
    /// `[32, 64, 128]`, so a power-of-two stride up to 32 is aligned on **every**
    /// rung and one of 64 on two of three. A ladder with rungs at odd multiples
    /// would silently deny the discount to every such op, which is a cost rather
    /// than an error and would show up as a phase that suddenly cannot be cut.
    ///
    /// # Why this is a list of scalars and not of `[usize; 3]`
    ///
    /// Because it was measured, and a per-axis candidate list does not earn its
    /// mechanism. The question was raised by strongly anisotropic volumes —
    /// `404 x 1304 x 3369` — on the reasoning that a scalar ladder searches a
    /// diagonal of the space. Two findings, both from sweeping the model over
    /// families of candidates at a fixed byte budget:
    ///
    /// * **A scalar edge does not produce a cubic block.** [`BlockGrid::along`]
    ///   clamps each axis at the volume and [`cuttable_axes`] drops an axis a cut
    ///   would not narrow, so edge 512 on that volume gives `[404, 512, 512]` and
    ///   the `[2]` default gives a z-slab. The diagonal is projected onto the
    ///   volume's own box before it is priced, and the reachable shapes are
    ///   already anisotropic.
    /// * **Where a general per-axis family does win, a finer scalar ladder wins
    ///   by more, and costs nothing but entries here.** Adding a rung at `3/4` of
    ///   each power of two — 24, 48, 96, 192, 384 — takes the working-set step
    ///   between neighbours from 8x to about 2.4x. Swept over two volumes, three
    ///   reaches, three worker counts and five budgets: the finer ladder beat the
    ///   coarse one by up to **2.7x** (a tight budget where the coarse ladder
    ///   drops from a 32-cube to a 16-cube and lands far under the budget, while
    ///   a 24-cube uses it), against at most **1.4x** for full per-axis freedom
    ///   over the same coarse rungs. The general family won a minority of cells
    ///   and never by as much.
    ///
    /// So the gap is real and it is **granularity, not anisotropy** — and it is
    /// reachable today. The same result holds on an isotropic `1024^3`, which is
    /// what says it is not about the volume's aspect ratio.
    ///
    /// **That finer ladder is [`BlockLadder::Refined`]**, computed by
    /// [`refined_ladder`] and selected by [`Constraints::with_ladder`]. It is a
    /// named setting rather than the default, and the reason is not the search
    /// cost the sentence above would suggest — that was measured at `1.06x` and
    /// retired. It is that this project intends to pick planners by running them
    /// against each other, and a context-dependent gain belongs in that
    /// competition rather than in a default. `refined_ladder`'s header carries
    /// the measurement, the decision, and the one guarantee elsewhere that this
    /// ladder changes the wording of.
    ///
    /// # What this does not price, and what that is worth
    ///
    /// **Fragment traffic, which can prefer the opposite lattice.** In-plan
    /// fragment traffic is `(1 + blocks) x F(blocks)` with `F` the total face
    /// area, and `F = sum over axes of (cuts on that axis) x (area of the face
    /// perpendicular to it)`. On a volume whose axis-0 face is 66x its axis-2
    /// face, `F` is very nearly a function of axis-0 cuts alone. Measured on the
    /// volume above: **52.4 GiB at 482 slab blocks against 136.8 GiB at 512 cube
    /// blocks** — near-identical block counts, 2.6x apart in traffic. Nothing in
    /// [`price_phase`] grows with the block *count* per block, so a planner
    /// choosing on pixel cost alone will pick the expensive lattice, and the
    /// direction is knowable in closed form from the volume's aspect ratio before
    /// any sweep is run. That is the case for extending this list; the case for
    /// making its entries per-axis is the one the measurement above declines.
    pub block_candidates: Vec<usize>,
    /// Which axes may be cut. `[2]` is the z-only default that every recorded
    /// parity figure was measured under.
    pub split_axes: Vec<usize>,
}

/// The ladder above with a rung at **three quarters** of each entry: the
/// granularity the sweep in [`Constraints::block_candidates`] measured and
/// nothing had built.
///
/// `[16, 32, 64, 128]` becomes `[16, 24, 32, 48, 64, 96, 128]`. Sorted
/// ascending, de-duplicated, and **the floor is not lowered**: `3/4` of the
/// smallest entry is dropped, because the smallest rung is a caller's statement
/// about how small a block they will accept and quietly going below it is a
/// change they did not ask for. That is also what the sweep's own ladder did —
/// its rungs are `24, 48, 96, 192, 384`, which are `3/4` of `32` upwards on a
/// ladder whose floor is `16`, and it lists no `12`.
///
/// **It is a fixed point**, and that is load-bearing rather than tidy. Three
/// quarters of a power of two is never itself a power of two, so a second
/// application finds nothing to insert. An earlier version refined every entry
/// instead — `3/4` of `48` is `36` — which made it an *operator*: applying it
/// twice gave a third ladder, so "which ladder is this" could not be answered
/// from the value. A setting a planner competition sweeps has to be answerable
/// from its value, and [`BlockLadder`] is that setting.
///
/// # What it buys
///
/// The working-set step between neighbours falls from `8x` to about `2.4x`. The
/// planner admits the largest candidate that fits, so a coarse ladder can drop a
/// whole factor of eight and land far under the budget where a `3/4` rung would
/// have used it. Swept over two volumes, three reaches, three worker counts and
/// five budgets, the finer ladder beat the coarse one by up to **2.7x** —
/// against at most **1.4x** for full per-axis freedom over the same coarse
/// rungs, which is the comparison that says the gap is granularity rather than
/// anisotropy.
///
/// # What it costs: entries, and one restated guarantee
///
/// **Entries.** The search is `partitions x candidates^phases`, so refining a
/// three-rung ladder to six squares the per-phase factor. That is the reason
/// this is a function a caller applies rather than the default: the default
/// stays where every recorded parity figure was measured, and a caller who wants
/// the finer grid says so and pays the search for it.
///
/// **And a guarantee elsewhere that this changes the wording of.**
/// `budget::UNOBSERVED_SHAPE_MARGIN`'s header argues that an admission margin
/// can cost *at most one ladder step*, and calls that arithmetic rather than
/// luck: a ladder of powers of two steps by `8x` in volume, and every margin is
/// under eight — `3.6` alone, and `3.5626 x 2.1 = 7.48` for the worst measured
/// shape. **That wording does not survive this ladder.** A step here is
/// `(4/3)^3 = 2.37x` or `(3/2)^3 = 3.375x`, alternating, so a margin of `3.6`
/// already spans two of them.
///
/// **The arithmetic survives; only the unit it was stated in does not.** Any two
/// consecutive rungs of this ladder span exactly `2.37 x 3.375 = 8.0` in volume
/// — it interleaves one rung into each octave and changes nothing else — so
/// every margin under eight costs **at most two rungs here, which is at most the
/// same `8x` in block volume the one-step bound guaranteed.** The bound that
/// survives any spacing is therefore the volume ratio, and the step count was a
/// proxy that happened to equal it while the ladder was powers of two.
///
/// It is also the *better* bound to state, and not only the more durable one:
/// the planner stops at the largest rung that fits, so on a finer ladder the
/// correction lands **closer**, never further. That is the same fact the `2.7x`
/// win is, seen from the margin's side.
/// [`a_finer_ladder_never_admits_a_smaller_block_than_a_coarse_one`] measures
/// both halves.
///
/// [`a_finer_ladder_never_admits_a_smaller_block_than_a_coarse_one`]: https://docs.rs/blockflow
/// # Should this be the default? The two reasons it is not, measured
///
/// It was made opt-in for two stated reasons. Both have now been tested rather
/// than inherited, and **neither survives as originally stated**. The decision
/// is recorded here rather than acted on, because flipping a default changes
/// every consumer's plan and that is not a change to make from inside the
/// function that would benefit from it.
///
/// **Reason one — the search cost — is retired.** The framing was
/// `partitions x candidates^phases`, which makes refining a three-rung ladder to
/// five look like `81 -> 625` at four phases. Two things are wrong with it.
/// `boxes_tile_exactly` became a linear separating pass, taking
/// `PlanBuilder::finish` from 445 ms to 4.2 ms at 8192 blocks, so one candidate
/// costs a fraction of what it did when that multiplier last looked expensive.
/// And the exponent is not real: [`crate::strategy::PartitionSearch`] is a
/// dynamic program over prefixes, so the objective is additive over phases and
/// the candidate loop runs once per (run, candidate) rather than once per
/// combination. Measured end to end by `tests/ladder_planning_cost.rs`:
/// **`1.06x` to `1.13x` of the coarse ladder's planning time, sub-millisecond
/// throughout, at phase counts where the nominal combination count reaches seven
/// figures.** The measured ratio stays under the bare `5/3` candidate ratio,
/// which is the observable form of "it scales with the count, not the power".
///
/// **Reason two — the recorded parity figures — is narrower than it sounded, and
/// it is the one that still decides.** Surveyed across both crates, splitting
/// what a plan change would move into *answers* (voxel values, digests,
/// component and row counts, reference comparisons) and *plan-shape
/// measurements* (phase counts, chosen edges, fingerprints, read counters):
///
/// * **No answer is exposed.** Nothing that guarantees decomposition invariance
///   reaches the planner through this default. The oracle comparisons build
///   their plans with `Trivial`, which discards the candidate list, or with an
///   explicit list of their own.
/// * **The sibling crate is entirely immune** — it generates its own ladder from
///   the caller's block and never consumes this default.
/// * **The one frozen fingerprint literal is immune**, being built under an
///   explicit four-rung list. Every other fingerprint assertion is relational.
/// * What would move is a handful of **ladder-shape literals** and one recorded
///   paragraph in the ops survey.
///
/// # It was recommended as a default, and the project decided otherwise
///
/// **The recommendation is kept rather than deleted, because a recommendation
/// reversed by a decision should keep its reasoning visible.** On the two
/// reasons above it was: *flip the default, once the volume-bound restatement
/// lands in `budget.rs` and once the opt-in method goes with it.* Both
/// prerequisites were real; the first has since landed.
///
/// **The decision was not to flip it, and the reason is better than the
/// question.** This project intends to run several planners against simulated
/// data and pick between them by what wins in a given context. A `1.06x`
/// planning cost buying a larger block in 15 of 27 cells is exactly the kind of
/// finding that should become a **competitor** rather than a default: the gain
/// is real, it is context-dependent, and it is not large enough to justify
/// moving the ground under every consumer of this crate on an average that
/// nobody can defend in every context.
///
/// So what the measurement above is *for* changed. It is no longer the case for
/// a default; it is the evidence a competition starts from, and the thing that
/// had to change was how a caller **says which ladder** — see [`BlockLadder`],
/// which replaced the opt-in method for three reasons that all bite when
/// planners are compared rather than argued about.
///
/// **What [`BlockLadder::Refined`] buys, and it is modest and real**: a larger
/// admitted block in 15 of 27 (budget, charge) cells at up to `3.38x` in volume,
/// never a smaller one; and over a (chain, budget) sweep, a different plan in 6
/// of 16 cells, two of which planned *fewer phases* — one less materialisation,
/// which is a second-order win nobody was claiming.
///
/// **What it does not buy, and should not be sold as:** feasibility.
/// `the_refined_ladder_never_plans_where_the_coarse_one_cannot` measures it —
/// whether a plan exists at all is decided by the ladder's **floor**, which this
/// refinement does not move, so it changes *which* block is chosen and never
/// *whether* there is one.
pub fn refined_ladder(coarse: &[usize]) -> Vec<usize> {
    let Some(&floor) = coarse.iter().min() else {
        return Vec::new();
    };
    let mut out: Vec<usize> = coarse
        .iter()
        .flat_map(|&edge| {
            // **Three quarters of each power of two**, which is what the sweep
            // named — `24, 48, 96, 192, 384` — and not three quarters of every
            // entry. The distinction is invisible on an octave ladder, where
            // every entry is a power of two, and it is what makes this a **fixed
            // point**: `3k/4` for a power of two `k` is `3 * 2^(k-2)`, never
            // itself a power of two, so a second application finds nothing new
            // to insert.
            //
            // That idempotence is not tidiness. A setting whose value cannot be
            // read back — where "which ladder is this" is unanswerable because
            // applying it twice gives a third thing — is a bad thing for a
            // planner competition to sweep, and [`BlockLadder`] is that
            // competition's first entry.
            [
                edge,
                if edge.is_power_of_two() {
                    edge * 3 / 4
                } else {
                    0
                },
            ]
        })
        .filter(|&edge| edge >= floor)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// **Which ladder of block edges a planner is offered.** A named setting, not a
/// transformation.
///
/// # The shape, and why it is this one
///
/// The precedent is [`crate::strategy::PartitionSearch`], and this is
/// deliberately its twin: a small enumerable choice carried in the constraints,
/// with a default, and with the alternative kept as a **named value** rather
/// than as the absence of a call. That matters for one reason —
/// `docs/design/barriers.md` is full of settings that are choices, and this
/// crate now intends to pick planners by running them against each other rather
/// than by arguing. **A competition needs settings it can put in a list**, and
/// [`Self::ALL`] is that list.
///
/// It replaced a method — `with_refined_ladder()` — which was wrong for three
/// reasons that all bite exactly when planners compete. It named a
/// *transformation*, so there was no domain to enumerate. It could not be swept.
/// And it was not idempotent, so "which ladder is this plan using" was not
/// answerable from the value: applying it twice gave a third ladder. The first
/// two are fixed by being an enum; the third is fixed in [`refined_ladder`],
/// which is now a fixed point.
///
/// # The parameter space this belongs to, which is a second one
///
/// The consumer of this crate keeps a disciplined parameter space for anything
/// that can change **the answer**: an arbitrary choice is an *axis*, a *synonym*
/// cannot change the answer and is not one, and a *transcription decision*
/// changes the answer but no reference parameter reaches it. The block ladder is
/// placed there already, and placed correctly — a consumer's `LADDER_FLOOR`
/// calls it *"arbitrary, and therefore a synonym rather than an axis — it cannot
/// change an answer, only which decomposition computes it."*
///
/// **That is true and it is not the whole placement.** A synonym in the answer's
/// space can be a genuine axis in a second space that has had no name: the
/// **planner's**. Its parameters are the ones that change which decomposition is
/// chosen and cannot change what the decomposition computes —
/// [`crate::strategy::PartitionSearch`] and this are its first two entries.
///
/// **Its acceptance bar is different, and that is the point of naming it.** For
/// the answer's space the bar is "the answer does not move", and here that bar
/// is met *by construction* — the survey behind [`refined_ladder`]'s header
/// found no voxel, digest, component count or reference comparison exposed to
/// this choice at all, because decomposition invariance is what the whole crate
/// guarantees. A bar that is satisfied before the experiment starts is not a
/// bar. The bar in the planner's space is **"which is faster, and in what
/// context"** — a question only measurement answers, and the one a competition
/// between planners over simulated data exists to ask.
///
/// The discipline that space needs is already visible in `PartitionSearch`:
/// `Exhaustive` is kept although `Dp` supersedes it, because *"the DP's licence
/// is the additivity above, and a cost model that breaks it needs a search that
/// never assumed it. It is also the oracle the DP is tested against."* A
/// planner's-space parameter keeps a variant when it is somebody's oracle or
/// somebody's context, not when it wins on average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockLadder {
    /// The rungs as the caller stated them, which for every ladder this crate
    /// and its consumers build is powers of two. **The default**, and the
    /// spacing every recorded parity figure was measured under.
    ///
    /// A step here is `8x` in block volume, which is the spacing
    /// `budget::UNOBSERVED_SHAPE_MARGIN`'s bound was first stated against.
    ///
    /// **It is the identity on the rungs it is given, not a reduction to powers
    /// of two.** A ladder names the *spacing* a caller wants offered, not the
    /// set — so selecting this one never discards a rung a caller deliberately
    /// stated, which a "round back down to octaves" rule would. A sweep
    /// therefore applies each of [`Self::ALL`] to the same **base** list rather
    /// than chaining them, and that is the pattern a competition wants anyway.
    #[default]
    Octave,
    /// [`Octave`](Self::Octave) with a rung at three quarters of each power of
    /// two above the floor: `[32, 64, 128]` becomes `[32, 48, 64, 96, 128]`.
    ///
    /// Steps of `2.37x` and `3.375x` alternating instead of `8x`, so the
    /// planner's correction lands closer. Measured: a larger admitted block in
    /// 15 of 27 (budget, charge) cells at up to `3.38x` in volume and never a
    /// smaller one, for `1.06x` to `1.13x` of the planning time. See
    /// [`refined_ladder`] for the measurement, for what it does **not** buy —
    /// feasibility — and for the guarantee it restates.
    Refined,
}

impl BlockLadder {
    /// Every ladder, for a caller that wants to try them all.
    ///
    /// The reason this is a `const` rather than something a caller writes out:
    /// a competition that enumerates by hand silently stops covering a variant
    /// the day one is added.
    pub const ALL: [BlockLadder; 2] = [BlockLadder::Octave, BlockLadder::Refined];

    /// The rungs this ladder offers over `base`.
    ///
    /// **Idempotent in the ladder and in the value**: applying the same ladder
    /// twice gives the same list, so a caller may set it without knowing what
    /// was set before. That is what makes it a setting rather than an operator.
    pub fn rungs(self, base: &[usize]) -> Vec<usize> {
        match self {
            BlockLadder::Octave => {
                let mut out = base.to_vec();
                out.sort_unstable();
                out.dedup();
                out
            }
            BlockLadder::Refined => refined_ladder(base),
        }
    }

    /// For a message or a report column.
    pub fn as_str(self) -> &'static str {
        match self {
            BlockLadder::Octave => "octave",
            BlockLadder::Refined => "refined",
        }
    }
}

/// How many **slabs** a block is cut into for intra-block threading.
///
/// A planner's-space parameter beside [`BlockLadder`] and
/// `strategy::PartitionSearch`, and for the same reason: it is a decision about
/// how to search, enumerable so a competition covers every variant, with a
/// default, and **it must be possible to turn off**.
///
/// **It is a field of [`Constraints`], and getting it there answered a design
/// question rather than just costing an edit.** Adding one breaks every struct
/// literal of that type — seven in the library and six in the integration tests
/// — and a struct constructed by literal in thirteen places is a struct whose
/// every future field is a tree-wide edit. The case *for* the literal is
/// exhaustiveness: a new field is a compile error at each site, so somebody has
/// to decide rather than inherit a default. That is worth paying **where the
/// struct is built for a real run**, and worth nothing where it is a test
/// fixture stating three knobs it cares about.
///
/// So the sites were read rather than rewritten, and **all thirteen turned out
/// to be test fixtures**: every one sits inside a `#[cfg(test)]` module or a
/// `tests/` file. No production path in this crate constructs `Constraints` by
/// literal — real callers take [`Constraints::default`] and a `with_*`. The
/// exhaustiveness was therefore being paid entirely by fixtures, where a new
/// planner knob is noise and not a decision, so they now end
/// `..Default::default()` and the next field costs them nothing. **The compiler
/// named every site**; no pattern was matched against `Constraints {`, which
/// matches things that are not literals and corrupted four files the first time
/// it was tried. Deliberately not an
/// operator — see [`BlockLadder`]'s own note on why a setting a caller states
/// once beats a method that compounds.
///
/// **What it is for, and the measurement is the whole argument.** Cutting a
/// block into slabs was measured at **`5.39x` at one block and nothing once
/// `n_blocks >= workers`**. That is not a better way to spend threads; it is the
/// only way to spend threads that would otherwise be **parked**. So the rule
/// below is not a tuning curve, it is a description of when there are idle
/// workers: `floor(workers / n_blocks)`, which is `1` exactly when the block
/// lattice already has work for everyone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlabPolicy {
    /// Cut a block into `floor(workers / n_blocks)` slabs, capped at
    /// [`Self::CAP`]. **The default**, and it changes nothing under the default
    /// [`Constraints`], whose `expected_concurrency` is one — asserted below,
    /// because a default that quietly altered every recorded figure would be the
    /// wrong kind of safe.
    #[default]
    FillIdleWorkers,
    /// Never cut. One slab per block, whatever the concurrency.
    ///
    /// The off switch, and it is required rather than a convenience: an
    /// intra-block cut changes which code runs, and a caller reproducing a
    /// recorded run needs to be able to say "not that".
    Off,
}

impl SlabPolicy {
    /// Every policy, so a competition that enumerates cannot silently stop
    /// covering one. [`BlockLadder::ALL`]'s reason.
    pub const ALL: [SlabPolicy; 2] = [SlabPolicy::FillIdleWorkers, SlabPolicy::Off];

    /// **Four**, and the cap is about halo rather than about threads.
    ///
    /// A block already pays a halo on every side; a slab pays another one on the
    /// cut axis, so the redundant read grows with the slab count while the
    /// parallel speed-up saturates against memory bandwidth. It is a constant
    /// rather than a field because a caller who wants a different number wants a
    /// different policy, and adding one is one variant here rather than a knob
    /// everywhere.
    pub const CAP: usize = 4;

    /// How many slabs a block should be cut into, given the pool and the
    /// lattice.
    ///
    /// **Never zero and never more than [`Self::CAP`].** `workers = 0` and
    /// `n_blocks = 0` are both answered `1` rather than refused: they are what an
    /// empty plan and an unset concurrency look like, and one slab is the
    /// behaviour this crate had before slabs existed.
    pub fn slabs_for(self, workers: usize, n_blocks: usize) -> usize {
        match self {
            SlabPolicy::Off => 1,
            SlabPolicy::FillIdleWorkers => {
                if n_blocks == 0 || workers == 0 {
                    return 1;
                }
                (workers / n_blocks).clamp(1, Self::CAP)
            }
        }
    }

    /// For a message or a report column.
    pub fn as_str(self) -> &'static str {
        match self {
            SlabPolicy::FillIdleWorkers => "fill-idle-workers",
            SlabPolicy::Off => "off",
        }
    }
}

impl Constraints {
    /// Whether one candidate phase's **block working set** fits the budget at
    /// the concurrency the caller expects.
    ///
    /// **The one copy.** This predicate existed three times — `assemble::
    /// affordable`, the block-candidate loop in `strategy.rs` and the
    /// partition search's own closure — and the first of them carried a comment
    /// explaining that it had been lifted out of the sweep so that "a second
    /// copy that drifted would let this builder accept a lattice the search
    /// would have refused". There were three. They agreed, which is luck rather
    /// than structure.
    ///
    /// # What it bounds, and what it does not
    ///
    /// It bounds **what the cut controls**: the input and output tiles of one
    /// block in flight, times the number of blocks in flight. Refusing here is
    /// actionable — a smaller block candidate makes the same plan fit.
    ///
    /// It does **not** bound what the run is resident in. A plan also holds its
    /// whole-volume images, and that quantity is
    /// [`Decomposition::peak_image_bytes`], not this. The two are different in
    /// kind and not merely in size:
    ///
    /// * the working set **scales with the block and the concurrency**;
    /// * the image residency is **invariant to both** — images are allocated at
    ///   the phase's volume, so a finer grid is an iteration order and not a
    ///   residency policy, and the image total does not move when the cut does.
    ///
    /// That is why the image term is reported by
    /// [`Decomposition::peak_image_bytes`] rather than added in here. A budget
    /// that charged it would refuse **every** candidate identically, since no
    /// candidate can change it — a feasibility gate with no feasible answer and
    /// no advice to give. The remedies for image residency are a narrower
    /// dtype, an out-of-core environment, or `Hints::release_images`, and all
    /// three live above the partition search.
    ///
    /// A caller that wants one number for both adds them; it must not expect
    /// this to have done it.
    pub fn affords_working_set(&self, cost: &PhaseCost) -> bool {
        self.budget_bytes.is_none_or(|budget| {
            cost.working_set_bytes_per_block * self.expected_concurrency.max(1) as f64
                <= budget as f64
        })
    }

    /// The same constraints with `policy`.
    ///
    /// **The one way to ask for a slab policy**, so a caller that sweeps
    /// [`SlabPolicy::ALL`] and a caller that names one go through the same line.
    /// Idempotent, because the field is replaced rather than combined.
    ///
    /// **What carries it into a run** is `strategy::Strategy::plan`, which
    /// copies it into `strategy::Hints::slab_policy` — the one method that holds
    /// a `Constraints` and a `Hints` at once. A caller who builds a
    /// decomposition and executes it directly is advising the hints themselves
    /// and states the policy there.
    pub fn with_slab_policy(mut self, policy: SlabPolicy) -> Self {
        self.slab_policy = policy;
        self
    }

    /// The same constraints with `ladder`'s rungs.
    ///
    /// **The one way to ask for a ladder**, so that a caller that sweeps
    /// [`BlockLadder::ALL`] and a caller that names one variant go through the
    /// same line. Idempotent, because [`BlockLadder::rungs`] is: setting the
    /// same ladder twice is setting it once, and setting a different one is
    /// answering a different question rather than compounding two.
    ///
    /// **There is one effective list and it is [`Self::block_candidates`].** The
    /// ladder is applied to it here rather than stored beside it, so nothing in
    /// this struct can disagree with anything else in it — a plan is priced
    /// against the rungs the field holds, whichever call put them there.
    pub fn with_ladder(mut self, ladder: BlockLadder) -> Self {
        self.block_candidates = ladder.rungs(&self.block_candidates);
        self
    }
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            slab_policy: SlabPolicy::default(),
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
///
/// **What this predicate is not about.** It decides *segmentation* — whether a
/// slot forces a phase boundary — and the measurement above is an argument about
/// segmentation only. It says nothing about which axes a phase's grid may be cut
/// on, which is [`splittable_axes`] and [`cuttable_axes`]; those two answer a
/// different question with a different consequence, and the same "large is not
/// full" caution does not transfer to them unexamined. See [`cuttable_axes`],
/// which does examine it — including the measurement above, which turns out to
/// leave the cutting rule untouched on the very chain it was taken from.
///
/// The paraphrase above is also a little stronger than its source. `§6.5.1` of
/// `GRAPH_MIGRATION.md` has four of the seven steps *cheap* rather than four
/// wrongly segmented: an exact predicate segments at every whole-volume step, and
/// what a threshold rule would break is the two steps that reach a single voxel,
/// plus the pricing of the cheap ones. The conclusion is unchanged — exact, not a
/// threshold — but the "four" counts cheapness, not segments.
pub fn reaches_whole_axis(reach: usize, extent: usize) -> bool {
    extent > 1 && reach >= extent
}

/// Does the reach's **halo alone** cover an axis of extent `extent`?
///
/// The two-sided form of [`reaches_whole_axis`], and the same kind of statement:
/// exact arithmetic on the declared reach against the extent, no threshold. Where
/// `reaches_whole_axis` asks whether one side already covers the axis, this asks
/// whether the two together do — `lo + hi >= extent`, which is precisely the
/// condition under which `edge + lo + hi >= extent` holds **for every edge**, so
/// no block on the axis can be interior and no cut on it can narrow a read.
///
/// It is what [`cuttable_axes`] is a per-candidate application of, and what
/// [`price_phase`] charges a **single-block** phase on: a phase the reach has
/// left with no cut anywhere is not blocked, not streamed and not interior, so
/// the clamp discount is not exact for it. A phase still cut on some other axis
/// keeps the discount and it is still exact there — the block spans this axis, so
/// the read is the volume, once — which is why that condition carries the block
/// count and this predicate does not.
///
/// An axis of extent 1 is excluded for the reason [`reaches_whole_axis`] excludes
/// it: every block already spans it, so it forbids nothing and costs nothing.
pub fn halo_spans_axis(reach: &Reach, axis: usize, extent: usize) -> bool {
    let (lo, hi) = reach.axis(axis).bound(extent);
    extent > 1 && lo.saturating_add(hi) >= extent
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

/// [`splittable_axes`], with the **floor derived from the reach** applied at one
/// candidate block edge.
///
/// **The planner has a ceiling from memory and needed a floor from reach.** A
/// block reads `min(edge + lo + hi, volume)` on an axis. Once `lo + hi` reaches
/// the extent that is the whole axis *for every edge*, so cutting the axis gives
/// `n` blocks each reading all of it: the total read goes from `volume` to
/// `n x volume` and the resident set does not move, because it was already
/// clamped. A measured case is on record — a 716-offset element whose reach was
/// 15 on a `24 x 20` volume, cut into 336 blocks, running past 15 minutes where
/// one block would have read the volume once. `O(volume)` became
/// `O(blocks x volume)`, and nothing in the plan said so.
///
/// So a cut is admitted **only where it narrows what a block reads**. That is
/// the whole rule, it is exact arithmetic on quantities the plan already holds —
/// the declared reach and the volume — and it contains no threshold to tune. An
/// edge that fails it on some axis does not lose the candidate: the axis is
/// simply not cut, the block spans it, and the read that was going to be the
/// whole axis happens once instead of once per block. That is the strictly
/// dominating plan by both terms at once, which is why this is structural rather
/// than a weight — the same argument [`is_planning_barrier`] and
/// [`splittable_axes`] already make, and this is the sharper form of the second
/// of them: a full reach (`r >= extent`) is the special case of `lo + hi >=
/// extent` where one side already covers the volume, and everything between
/// `extent / 2` and `extent` was being cut for nothing.
///
/// **It cannot make a plan infeasible.** The budget is checked against
/// [`PhaseCost::working_set_bytes_per_block`], which is computed from the
/// *clamped* read extent; on an axis this drops, the clamp was already at the
/// volume, so the resident figure is the number it already was. A candidate that
/// fitted still fits.
///
/// **It cannot change a voxel.** A block grid is not an answer, it is how the
/// answer is cut up; every strategy's output is asserted against `Trivial`'s
/// single block. What moves is which plan a planner offers, and the one it stops
/// offering is one that computes the same volume by reading it `n` times.
///
/// # Wired, and what had to be settled first
///
/// This was landed unwired because it contradicted a stated and tested position
/// of the crate. Both planners call it now, and the contradiction was resolved by
/// **moving the position**, not by weakening its test:
///
/// * `tests::a_large_but_bounded_reach_is_not_a_barrier_and_still_fuses` asserted
///   that a reach of `volume - 1` left its phase *cuttable* — "priced out of
///   fusing, but not **forbidden** from it, which is the whole difference from a
///   barrier". At that reach `lo + hi` is nearly twice the volume, so this rule
///   forbids the cut, and the two could not both hold. The test now asserts the
///   narrower claim that survives, and it is the claim that was doing the work:
///   a bounded reach is not a **barrier** — it forces no phase boundary, its
///   neighbours may still fuse into it, and every other axis stays cuttable —
///   while whether *this* axis is worth cutting is a question about the grid,
///   which the arithmetic here answers and a barrier never asked.
/// * [`reaches_whole_axis`] argues from the other side that the barrier predicate
///   is exact *because* a "large means full" rule would segment a real chain
///   where nothing wants a segment. That argument survives and does not reach
///   here, for two reasons that were checked rather than assumed. It is an
///   argument about **segmentation**, and this rule adds no forced cut and
///   removes no fusion; and on the chain it is measured on it makes no difference
///   at all — the five whole-volume merge steps have already lost those axes to
///   [`splittable_axes`], and the two that reach a single voxel keep every axis
///   here at every candidate edge, since `edge + 1 + 1 < extent` for any edge
///   that cuts anything. (Worth recording while passing: the "four places that do
///   not want it" in [`reaches_whole_axis`]'s own note overstates its source.
///   `GRAPH_MIGRATION.md` §6.5.1 has four of the seven *cheap*, not four wrongly
///   segmented — an exact predicate segments at all five whole-volume steps too,
///   and the objection to a threshold rule is that it would catch the two
///   1-voxel ones and price the cheap ones as expensive.) See
///   `strategy::block_floor_tests::the_floor_changes_nothing_on_the_chain_the_barrier_rule_was_measured_on`.
///   What does *not* transfer is any claim that the grid is therefore untouched
///   by fusion: the grid a phase is given changes its price, and price chooses
///   the partition — which is the next point.
/// * the second-order effect that had to move with it: [`price_phase`] charged an
///   axis on the infinite grid when the grid cut it *or* the reach was whole.
///   Dropping an axis here without extending that condition hands the phase the
///   clamp discount and prices it at redundancy `1.0` — the exact hole
///   `price_phase`'s own doc records having closed for full reaches — and the
///   partition collapses. The condition is now stated over
///   [`halo_spans_axis`] as well, which is this rule's own edge-independent form.
///
/// The number the decision rests on is in
/// `strategy::block_floor_tests::the_floor_takes_the_amplification_from_49_to_1`,
/// and `strategy::block_floor_tests` also asserts through the planners that the
/// partition survives the pricing change.
pub fn cuttable_axes(
    split_axes: &[usize],
    reach: &Reach,
    volume: [usize; 3],
    edge: usize,
) -> Vec<usize> {
    splittable_axes(split_axes, reach, volume)
        .into_iter()
        .filter(|&axis| {
            if axis >= 3 {
                // Out of bounds, and `BlockGrid::along` is where that is said.
                return true;
            }
            // The widest halo over every block, which is what the read extent is
            // sized by. Saturating because a reach may exceed the volume, which
            // is exactly the case this exists for.
            //
            // **`resolved(edge)` before `bound`, and this function is the reason
            // that ordering had to be found rather than assumed.** It is the
            // identity for every reach but [`crate::reach::AxisReach::Aligned`],
            // so no plan that existed before it changes. For an aligned reach it
            // is decisive: the unresolved halo is `stride - 1` wider per side, so
            // an op whose true halo leaves an axis comfortably cuttable is judged
            // on its worst case, the axis is **dropped here**, and the phase
            // degenerates to one block — reading the whole volume per block,
            // which is the cost `docs/design/barriers.md` exists to remove. The
            // discount taken in `price_phase` never gets a chance, because this
            // runs first and decides there is nothing to price. Measured: a
            // 32-voxel-tile transform convolution over `96^3` planned as a
            // *single block* at both edge 32 and edge 48 before this line
            // resolved.
            let (lo, hi) = reach.axis(axis).resolved(edge).bound(volume[axis]);
            edge.saturating_add(lo).saturating_add(hi) < volume[axis]
        })
        .collect()
}

/// What a block of one phase actually moves, as opposed to how it is blocked.
///
/// **The three facts [`price_phase`] cannot get from a grid and a halo.** A grid
/// says how the volume is cut and a halo says how far past a core a block
/// fetches; neither says *how many arrays* are fetched over that extent, nor
/// whether anything is written back. Before this existed the price assumed one
/// read and one write for every phase, and a `fragments -> fragments` phase —
/// which reads no pixel and writes no image — was charged a full traversal of
/// the volume plus a full write of it, per phase, for traffic that
/// [`Decomposition::exact_read_voxels`] correctly reports as **zero**.
///
/// That is the same defect class the halo charge was: the size of the phantom is
/// `mean_core * redundancy * n_blocks`, which varies with the candidate, so it
/// is not a constant a model could absorb and it moves the ranking. See
/// `tests/phase_pricing.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseTraffic {
    /// How many stored images a block reads over its read extent.
    ///
    /// The same count [`Decomposition::exact_read_voxels`] uses — the phase's own
    /// input image if it reads one, plus each entry of
    /// [`PhaseDecomposition::source_images`] — so the price and the exact figure
    /// are counting the same fetches. It is **not** clamped to one: a phase with
    /// a [`Chain::Source`] arm really does traverse two arrays, and charging it
    /// once was an under-charge in the direction the model is not allowed to be
    /// wrong in.
    pub images_read: usize,
    /// Whether the phase writes the image after it.
    ///
    /// [`crate::fragment::PhaseWork::writes_an_image`] is the answer, and it is
    /// the op's rather than the plan's: a fragment phase writes an image only if
    /// its op says it does, and the plan records no such field.
    pub writes_an_image: bool,
    /// How many times the phase's **read and compute** happen, where its write
    /// happens once.
    ///
    /// `1` for everything but an iteration. An iterative phase runs `S`
    /// substages that read and compute, ping-ponging two private buffers, and
    /// `strategy::run_iterative_phase` writes the image **once**, after the
    /// loop. So the shape is `S x (read + compute) + write`, and pricing at
    /// `S == 1` over-weights the write against the rest by a residual that
    /// varies with the block edge — measured at up to **1.125x** the right
    /// choice, and departing from it at counts as ordinary as two and four. See
    /// `crate::iterate::IterativeOp::cost_per_voxel`.
    ///
    /// **Measured, never declared.** `IterativeOp::limit` is a runaway *bound*
    /// and deliberately not an estimate; the count is a fixed point over data,
    /// so no plan holds it and `phase_traffic` answers `1`. A caller with a real
    /// run's `Stats::substages` says so with [`PhaseTraffic::repeating`].
    pub repeats: u64,
    /// Block buffers the chain holds **inside** this phase, from
    /// [`Chain::resident_block_buffers`] over the phase's own slots.
    ///
    /// **Derived where the price is taken rather than recorded on the plan.**
    /// The quantity is a function of the chain, and every caller that prices a
    /// budget holds the chain — `resident_buffers_of` is the helper. A field on
    /// [`PhaseDecomposition`] was the other option and is worse: it has no safe
    /// neutral value, so a `check` would have to refuse every plan that did not
    /// declare it, where `dtype` and `source_images` both have defaults that
    /// most plans genuinely satisfy.
    ///
    /// Zero for the one-op phase [`PhaseCost::working_set_bytes_per_block`]'s
    /// old `x 2.0` was written for, so adding this moved no such phase's price.
    /// It is the term that was missing for everything else, and by 46.5x for a
    /// 91-arm fan-in whose combine cannot fold.
    pub chain_buffers: usize,
}

impl PhaseTraffic {
    /// One image in, one image out, once — every pixel phase with no second
    /// array, which is what the price assumed before the distinctions existed.
    pub fn one_in_one_out() -> Self {
        PhaseTraffic {
            images_read: 1,
            writes_an_image: true,
            repeats: 1,
            chain_buffers: 0,
        }
    }

    /// The same traffic, holding `buffers` block buffers inside its chain.
    pub fn holding(self, buffers: usize) -> Self {
        PhaseTraffic {
            chain_buffers: buffers,
            ..self
        }
    }

    /// The same traffic, with the read-and-compute happening `substages` times.
    ///
    /// `0` is read as `1`: `Stats::substages` reports zero for a phase that is
    /// not an iteration, and charging that phase nothing would be reading "not
    /// an iteration" as "no work".
    pub fn repeating(self, substages: usize) -> Self {
        PhaseTraffic {
            repeats: (substages as u64).max(1),
            ..self
        }
    }
}

/// The infinite-grid per-block cost of one phase.
///
/// `total = n_blocks x per_block_cost(B, partition)`, with no boundary term.
/// The assumption is **conservative**: at a real volume boundary the halo is
/// clamped, so an edge block reads less and costs less than an interior one.
/// Assuming every block is interior therefore overestimates, which can only
/// make the planner cautious — the same direction of error as a generous halo.
/// # The rule this type has now been wrong under twice
///
/// **A term that is wrong by an amount which varies with the candidate is not a
/// conservative approximation. It is a bias.**
///
/// Every figure here is stated on the infinite grid, and the file is full of
/// deliberate over-charges justified by "the error is in the safe direction".
/// That justification is only available to an error whose *size* is a property
/// of the work. An error whose size is a property of the **candidate being
/// priced** does not make the planner cautious — it reorders the candidates, and
/// it reorders them for a reason that has nothing to do with what any plan would
/// do. Two instances have been found in this one expression, and they were found
/// separately because nothing named the pattern:
///
/// * the **core** was charged at `BlockGrid::core_voxels`, the widest block, so
///   each candidate paid for its own grid's padding: `1.253 / 1.588 / 1.404 /
///   1.151` at edges `512 / 256 / 128 / 64` over the pipeline tile, a 38%
///   penalty on 256 against 64 unconnected to the work. Now
///   [`BlockGrid::mean_core_voxels`], which is exact and has no constant to tune.
/// * the **read** was charged at the phase's `reach` rather than its `halo`, so
///   a granted halo the ops did not ask for was counted zero times per block
///   however many blocks there were: `0.0% / -15.8% / -45.2% / -70.4%` at edges
///   `64 / 32 / 16 / 8`. Now the halo. See [`price_phase`].
///
/// A third of the same shape was found beside them and is fixed here too: a
/// phase's *traffic* was assumed to be one image in and one image out, so a
/// `fragments -> fragments` phase was charged a phantom traversal whose size is
/// `mean_core x redundancy x n_blocks`. See [`PhaseTraffic`].
///
/// A fourth was found in the same window, and it is the subtlest: an iterative
/// phase runs `S` substages of read-and-compute but writes its image **once**, so
/// pricing it at `S == 1` over-weights the write by a residual that varies with
/// the block edge. See [`phase_compute_per_voxel`]. A fifth, in
/// [`crate::strategy::phase_makespan`]'s pool, was found by holding
/// `is_materialised` fixed on the argument that the write enters the channel
/// bound as `mean_core x n = volume` identically at every candidate — true of the
/// channel, false of the pool, where it enters as `mean_core x ceil(n / workers)`
/// and a one-block candidate carries the whole write while a forty-block one
/// carries a fortieth.
///
/// # The test to apply
///
/// Not "is this an over-estimate" but **"is the size of the over-estimate the
/// same question for every candidate"**. Where it is not, the term has to be made
/// exact, or kept out of the number the search ranks on — which is what
/// [`Self::working_set_bytes_per_block`] does in the other direction: it is
/// allowed to over-state, because it feeds a budget and never a comparison.
///
/// **The sharper form, which catches the ones that do not look like costs:** a
/// reassurance about a *per-item* cost must name the item count, and the defect
/// is when that count is the thing the caller is free to increase. Every one of
/// the five above reads as a bounded per-block statement and is unbounded in the
/// block count, the array count, the substage count or the worker count.
///
/// And the corollary, because all five looked constant right up until somebody
/// swept them: **"constant across candidates" is a claim to be tested
/// adversarially, not asserted from the algebra.** Two of the five were argued
/// correctly from the algebra of one term while a second term, in the same
/// expression, did the opposite.
///
/// [`BlockGrid::mean_core_voxels`]: crate::geometry::BlockGrid::mean_core_voxels
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
    /// How many times the read-and-compute terms of [`Self::cost_per_block`]
    /// were charged. `1` for everything but a measured iteration; see
    /// [`PhaseTraffic::repeats`].
    pub repeats: u64,
    pub cost_per_block: f64,
}

/// What a run is resident in, in its two terms. See
/// [`Decomposition::residency`], which is the only thing that builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Residency {
    /// Whole-volume images alive at the worst phase. **Invariant to the block.**
    pub image_bytes: u64,
    /// Block tiles in flight: the worst phase's working set times the expected
    /// concurrency. **Scales with the block and the concurrency.**
    pub working_set_bytes: u64,
    /// The worst barrier gather: every contributing block's fragment at once,
    /// from each stream's declared bound. See
    /// [`Decomposition::gathered_sidecar_bytes`].
    ///
    /// **The only term that grows as the cut gets finer**, which is why it had
    /// to be a term rather than a footnote: the other two fall, so a planner
    /// optimising against them alone walks toward a peak nothing was watching.
    pub sidecar_bytes: u64,
}

impl Residency {
    /// Both terms. The number to compare against a machine.
    pub fn total_bytes(self) -> u64 {
        self.image_bytes
            .saturating_add(self.working_set_bytes)
            .saturating_add(self.sidecar_bytes)
    }

    /// [`Self::total_bytes`] in GiB, for a message or a table.
    pub fn total_gib(self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Voxels resident while one block of `grid` is in flight, under `halo`.
///
/// **Extracted rather than copied.** It is the sole definition of the quantity
/// [`PhaseCost::working_set_bytes_per_block`] is bytes of, and
/// [`Decomposition::residency`] needs the same number from a decomposition that
/// no longer holds the `PhaseCost` it was priced with. A second spelling would
/// drift, and the budget predicate three lines from here has already been that
/// bug once — see [`Constraints::affords_working_set`].
///
/// Clamped per axis: the grown block can never exceed the volume, because a
/// read cannot.
fn resident_voxels_per_block(grid: &BlockGrid, halo: &Reach) -> f64 {
    let block = grid.block();
    let volume = grid.volume();
    let halo = halo.in_voxels(block);
    let mut resident = 1.0_f64;
    for axis in 0..3 {
        let (lo, hi) = halo.axis(axis).bound(volume[axis]);
        let grown = block[axis] as f64 + lo as f64 + hi as f64;
        // `total_cmp`, not `f64::min`: the crate does not select between two
        // `f64`s through a partial order, here or anywhere.
        let extent = volume[axis] as f64;
        resident *= if grown.total_cmp(&extent).is_gt() {
            extent
        } else {
            grown
        };
    }
    resident
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
/// It is not exact either **on a single-block phase whose halo spans the axis**
/// ([`halo_spans_axis`]) — which is what [`cuttable_axes`] leaves behind when the
/// reach denies every cut. The same sentence applies word for word: no block is
/// interior, because every output voxel's window covers the extent before the
/// clamp touches it; the clamp is the whole behaviour rather than a boundary
/// effect; and the phase cannot be blocked or streamed at all. Without this the
/// floor is self-defeating — it turns the phase into one block and the discount
/// then prices that block at redundancy **1.0**, cheaper than any phase that is
/// still cut, so the search fuses the chain into it. Measured on a seven-slot
/// chain in `crate::strategy::block_floor_tests`: the discount gives one phase of
/// seven slots in one whole-volume block, and this charge gives back the three
/// phases the chain had before the floor.
///
/// **Two conditions, deliberately not one.** The full-reach clause is per axis
/// and does not care how many blocks the grid has; this one fires only when the
/// grid has a single block. An axis the grid did not cut while it was still
/// cutting some *other* axis is a phase that is blocked, streamed and priced per
/// block, and there the clamp discount is exact whatever the halo does — the read
/// on that axis is the volume, once, because the block spans it. Charging it
/// would invent traffic that no plan can incur, and it changes partitions of
/// chains the floor never touched: measured, it flips
/// `tests::a_memory_budget_forces_cuts_the_cost_model_would_not_choose` from the
/// one fused phase the model prefers to three, on a chain where fusing really is
/// 2.4x cheaper in bytes.
///
/// **Where this charge is known to cost something, stated rather than found
/// later.** When *every* candidate edge is at least the volume, every phase is a
/// single block whatever any rule says, and this clause then charges each of them
/// — so the search prefers to segment a chain whose phases would each have read
/// the volume exactly once either way. Measured on the five-op chain of
/// `crate::strategy::chain_floor_measurement` over a `24 x 20 x 16` volume at a
/// candidate edge of 32: two phases reading `2.0x` become four reading `4.0x`.
/// Nothing is unrunnable and no voxel moves; the model simply buys a
/// materialisation it did not need. It is the same over-charge the full-reach
/// clause has always made in the same situation, in the direction the model is
/// declared safe in, and separating it out needs `price_phase` to know which axes
/// the planner *offered* to cut — which it cannot be told without making the
/// price a function of the search rather than of the plan, and
/// [`predicted_cost`] reads plans back with no search in hand.
///
/// **A bounded reach is still not a barrier, in the price as well as the
/// structure.** The charge on a dropped axis is `(extent + lo + hi) / extent`; a
/// bounded reach has `lo + hi < 2 * extent` by definition, so it is charged
/// strictly under 3, while `AxisReach::All` has `lo + hi = 2 * extent` and is
/// charged exactly 3. The barrier remains the more expensive of the two at the
/// same grid, which is what the distinction costs out to once neither is cut.
///
/// **What is charged is the phase's `halo`, not its `reach`, and the two are
/// only usually the same number.** A phase fetches `core (+) halo` — that is
/// what [`PhaseDecomposition::derive`] hands `BlockGeometry::derive_with` and
/// what [`Decomposition::exact_read_voxels`] adds up — while the *reach* is the
/// narrower thing the halo is required to cover. They part exactly where a halo
/// is granted wider than the ops asked for: a `BlockConstraint` whose mandated
/// extent forces a per-block window ([`BlockConstraint::lattice`]), and a
/// fragment phase whose halo is raised by a fragment input's reach in blocks or
/// by a second array's ([`crate::fragment::fragment_phase`]).
///
/// **And the charge is exact.** Measured on a `64^3` volume cut on all three
/// axes, reach `1`, as the granted halo widens — predicted whole-phase read
/// against `exact_read_voxels`, which is the geometry the run will really fetch.
/// Each cell is *what this function charges* / *what charging the reach instead
/// would charge*, the second kept as the negative control:
///
/// | halo | edge 64 | edge 32 | edge 16 | edge 8 |
/// |---|---|---|---|---|
/// | 1 (= reach) | 0.0% / +0.0% | 0.0% / +0.0% | 0.0% / +0.0% | 0.0% / +0.0% |
/// | 2 | 0.0% / +0.0% | 0.0% / **-8.6%** | 0.0% / **-21.9%** | 0.0% / **-39.1%** |
/// | 4 | 0.0% / +0.0% | 0.0% / **-23.0%** | 0.0% / **-49.7%** | 0.0% / **-72.5%** |
/// | 8 | 0.0% / +0.0% | 0.0% / **-43.8%** | 0.0% / **-75.6%** | 0.0% / **-91.3%** |
///
/// The reach charge fails the test the core charge was fixed under, in the same
/// words: **the size of the error is a property of the candidate.** It is zero at
/// one block and grows without bound as the edge falls, because the ungranted
/// part of the halo is counted zero times per block however many blocks there
/// are — so a price built on it prefers the smallest grid on offer for a reason
/// that has nothing to do with the work.
///
/// The halo column is zero rather than merely small because the charge is
/// [`BlockGrid::mean_read_voxels`] — the exact clamped mean, not the
/// infinite-grid one. That was a second fix of the same shape and its history is
/// on [`PhaseCost`]. `tests/phase_pricing.rs` holds both columns and asserts the
/// first as an **equality**, which is what leaves no width for a
/// candidate-dependent error to hide in.
///
/// [`BlockGrid::mean_read_voxels`]: crate::geometry::BlockGrid::mean_read_voxels
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
    halo: &Reach,
    compute_per_voxel: f64,
    distinct_orders: usize,
    is_materialised: bool,
    bytes_per_voxel: f64,
    model: &CostModel,
    materialise_cost_per_voxel: f64,
    traffic: PhaseTraffic,
) -> PhaseCost {
    // The **average** core, not the widest. Every block is still charged the
    // widest block's *halo* below, which is the declared over-charge; the core
    // it is charged against is the grid's own, because the padding a widest-core
    // price adds is a property of the candidate rather than of the work and it
    // is what inverted this model's ranking. See `BlockGrid::mean_core_voxels`,
    // which carries the measurement.
    let core_voxels = grid.mean_core_voxels();
    let block = grid.block();
    let volume = grid.volume();
    // Before the shadowing below, because the helper takes the `Reach` and does
    // its own `in_voxels`.
    let resident_voxels = resident_voxels_per_block(grid, halo);
    let halo = halo.in_voxels(block);
    let mut read_per_block = 1.0_f64;
    for axis in 0..3 {
        // The widest block's halo, because the model is stated on the infinite
        // grid: a per-block halo is charged at its worst block, the same
        // direction of error a generous halo has. That over-charge is kept —
        // unlike the core's, it is bounded and barely moves with the edge. On
        // the tile of `BlockGrid::mean_core_voxels`'s table it over-states the
        // grid's true read amplification by 7.4% / 3.5% / 1.3% / 2.2% at the
        // four edges, against the core's 1.253 / 1.588 / 1.404 / 1.151.
        let (lo, hi) = halo.axis(axis).bound(volume[axis]);
        let grown = block[axis] as f64 + lo as f64 + hi as f64;
        // The two axes that are charged on the **infinite grid** on purpose. Not
        // a boundary approximation — a statement that the infinite-grid model
        // has broken and the clamp *is* the whole behaviour, so pricing the
        // clamped truth would say the phase is free to fuse across when it
        // cannot be blocked at all. Everything else takes the exact clamped
        // mean, which is what the two paragraphs below used to be arguing about.
        let on_the_infinite_grid = halo.is_whole_axis(axis, volume[axis])
            || (grid.n_blocks() == 1 && halo_spans_axis(&halo, axis, volume[axis]));
        read_per_block *= if on_the_infinite_grid {
            grown
        } else {
            grid.mean_read_extent(axis, lo, hi)
        };
    }
    // One traversal of the read extent per stored image the phase reads. Zero
    // for a phase that reads none — the `fragments -> fragments` case — and the
    // zero is the point: see `PhaseTraffic::images_read`.
    // Reported rather than computed: the ratio the model used to build the read
    // from, kept because callers and tests read it as "how many times over does
    // this grid traverse the volume".
    let redundancy = read_per_block / core_voxels;
    let read_voxels = read_per_block * traffic.images_read as f64;
    // **The voxels the op computes over, which is one traversal and not one per
    // array.** A phase that gathers four operands still evaluates its kernel
    // once per output voxel; charging the compute at `read_voxels` charged it
    // four times, and — the part that made it a defect rather than a scale
    // factor — the multiplier is the *group's* array count, so fusing multiplied
    // the whole chain's compute by the number of `Chain::Source` arms anywhere
    // in it while cutting spread them. That is a pressure to cut that grows with
    // the arms and exists nowhere in any run.
    //
    // It reached a consumer of this crate's planner before it reached a test
    // here: a partition suite over a real chain cut a **single-block, zero-halo**
    // plan into three phases, where there is by construction nothing for a cut
    // to spread. The control that should have caught it lives here now —
    // `tests/phase_pricing.rs`'s
    // `nothing_makes_the_search_cut_a_single_block_plan_with_no_halo`.
    //
    // Zero when the phase reads no array at all: a `fragments -> fragments` op
    // traverses no voxels, so there are none to compute over.
    let compute_voxels = if traffic.images_read == 0 {
        0.0
    } else {
        read_per_block
    };
    let conflict = if distinct_orders > 1 {
        model.order_conflict_penalty * core_voxels * (distinct_orders - 1) as f64
    } else {
        0.0
    };
    let write = if !traffic.writes_an_image {
        0.0
    } else if is_materialised {
        materialise_cost_per_voxel
    } else {
        model.write_cost_per_voxel
    };
    PhaseCost {
        redundancy,
        read_voxels_per_block: read_voxels,
        compute_per_voxel,
        // Input buffer plus output buffer, both over the clamped read extent —
        // and deliberately **not** scaled by `traffic`. This figure feeds the
        // byte budget, where over-charging invents infeasibility but is at least
        // a statement about one grid, while under-charging admits a plan that
        // will not run. A phase reading nothing is charged as if it read one
        // image, which is the safe direction; a phase reading three is charged
        // as if it read one, which is not, and is a known gap recorded here
        // rather than half-fixed: correcting it changes which plans are
        // *affordable*, and that is a budget review with its own measurements.
        // **Every block-sized buffer alive while one block is in flight**, not
        // the two the formula assumed. The phase's own input and output are
        // `images_read + writes_an_image` — which is exactly `2` for the
        // one-in-one-out phase this was written for, so that case did not move —
        // and `chain_buffers` is what the chain holds between them: two for a
        // long sequence, which ping-pongs, and **one per arm** for a fan-in whose
        // combine cannot fold.
        //
        // The old `2.0` was an under-charge for everything else, and
        // `budget.rs`'s header recorded it as a known gap "rather than
        // half-fixed". What made it worth fixing was measuring the size: 93
        // buffers against 2 for the feature stack of
        // `docs/design/pixel-classification.md`, in the direction that admits a
        // plan the run cannot afford. See `tests/working_set_residency.rs`,
        // which holds the shape-derived count to a global allocator as an
        // equality over fourteen shapes.
        working_set_bytes_per_block: resident_voxels
            * bytes_per_voxel
            * (traffic.images_read + usize::from(traffic.writes_an_image) + traffic.chain_buffers)
                as f64,
        repeats: traffic.repeats,
        // **`S x (read + compute) + write`, and the split is the point.**
        //
        // A constant multiplier on the *whole* price would be neutral to an
        // argmin, which is why pricing an iteration at one substage looked free.
        // It is not: the substages read and compute, and the image is written
        // once after the loop, so the residual `(S - 1) x (read + compute)` is a
        // function of the block edge through the read amplification. `iterate`
        // measured the one-substage choice at up to **1.125x** the right one,
        // departing at counts as ordinary as two and four and at every worker
        // count above two.
        //
        // The conflict term is outside the repeat too: it prices *ordering*
        // between slots of a group, which happens once however many substages
        // one of them runs.
        cost_per_block: traffic.repeats as f64
            * (read_voxels * model.read_cost_per_voxel
                + compute_voxels * model.compute_scale * compute_per_voxel)
            + core_voxels * write
            + conflict,
    }
}

/// What a plan *predicts* it will cost, under `model`.
///
/// The same arithmetic the planner searched with — [`price_phase`] per phase,
/// times the phase's block count — pointed at the plan that was chosen rather
/// than at the candidates that were not. It exists because a prediction nobody
/// can read back is a prediction nobody can check: the whole claim of
/// [`crate::statistics`] is that a measured coefficient predicts better than a
/// stale constant, and that claim is only assertable if the prediction is a
/// number.
///
/// The units are the model's. With `CostModel::default` that is voxelwise maps;
/// with a model calibrated from a snapshot it is **nanoseconds**, directly
/// comparable with `statistics::observed_nanos` over a run's log.
///
/// It reads the plan's *stated* reach rather than re-deriving it from the
/// chain, on the principle the file is built on: the decomposition is the
/// binding half, and a second derivation that disagreed with it would be a
/// second planner. The chain is still needed, for what each slot declared it
/// would cost — that is the one thing a plan records only as a name.
///
/// **`work` is required for the same reason the chain is.** A fragment or
/// iterative phase owns no chain slot, so the fold over `phase.slots` visits
/// nothing and every such phase priced at exactly **zero compute** — while
/// `IterativeOp::cost_per_voxel` sat there unread and the phase most likely to
/// be one is a thinning, whose whole cost is compute. The same arrangement
/// [`check_dtypes`] and [`check_output_shapes`] already have: the plan records
/// names, the ops are the only things that know, and a caller holding both is
/// the only place the question can be asked.
///
/// A caller whose plan is all pixels may pass `&[]`. That is not a silent
/// default — [`phase_traffic`] **refuses** a phase that owns no slot and has no
/// `work` entry, by name, because such a phase is precisely the one whose price
/// would otherwise be a fabricated zero.
pub fn predicted_cost(
    chain: &Chain,
    decomposition: &Decomposition,
    work: &[crate::fragment::PhaseWork<'_>],
    model: &CostModel,
) -> Result<f64> {
    let slots = chain.slots();
    let mut total = 0.0_f64;
    for (index, phase) in decomposition.phases.iter().enumerate() {
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            return Err(Error::InvalidArgument(format!(
                "predicted_cost: phase {index} names slot {:?}, and the chain has {}",
                phase.slots.iter().max(),
                slots.len()
            )));
        }
        let volume = decomposition.volume_at(index);
        let (_, _, _, orders) = summarise_slots(&slots, &phase.slots, volume)?;
        // At the grid the plan actually holds, which is the same figure the
        // search priced this candidate with. See `compute_per_voxel`.
        let compute = phase_compute_per_voxel(&slots, phase, work.get(index), model)?;
        let traffic = phase_traffic(index, phase, work.get(index))?;
        // The last phase writes the workflow's output; every other writes an
        // intermediate. Exactly the test the enumeration makes.
        let is_materialised = index + 1 < decomposition.phases.len();
        let cost = price_phase(
            &phase.grid,
            // The halo, because that is the extent a block fetches; see
            // `price_phase`, which carries the measurement.
            &phase.halo,
            compute,
            orders.len(),
            is_materialised,
            decomposition.dtype_at(index).size_of() as f64,
            model,
            model.materialise_cost_per_voxel,
            traffic,
        );
        total += cost.cost_per_block * phase.grid.n_blocks() as f64;
    }
    Ok(total)
}

/// The compute figure [`price_phase`] wants, at one candidate block shape.
///
/// [`summarise_slots`] answers the same question with no block in hand, because
/// it is asked *before* a grid exists — its result is what the reach and the
/// traversal preferences are folded from, and those choose the grid. The compute
/// term is the one quantity in that tuple that a block can move
/// ([`crate::op::BlockOp::cost_per_voxel_in`]), so a planner comparing candidates
/// re-asks it per candidate rather than pricing every grid with the figure from
/// no grid. For every op that takes the default this is the same number by the
/// same route, so no plan built before it existed moves.
/// How many stored arrays a block of a run of pixel slots traverses.
///
/// One for the phase's own input image, plus one for each distinct image a
/// [`Chain::Source`] leaf in the run reads. It is the search's counterpart of the
/// count [`Decomposition::exact_read_voxels`] takes off a finished plan, and it
/// exists because the search prices *before* `declare_source_images` has run —
/// so the plan cannot be asked and the chain is the only thing that knows.
///
/// Charging one where a phase traverses two was an under-charge, which is the
/// direction this model is not permitted to be wrong in, and it is exactly the
/// saving a source arm is supposed to *not* have: `Chain::Source` is cheaper than
/// materialising the second array because it adds nothing to the halo, not
/// because its traversal is free.
pub fn images_read_by(slots: &[&Chain], group: &[usize], volume: [usize; 3]) -> Result<usize> {
    let mut images: Vec<usize> = Vec::new();
    for &slot in group {
        for input in slots[slot].source_inputs(volume)? {
            let index = input.image.index();
            if !images.contains(&index) {
                images.push(index);
            }
        }
    }
    Ok(1 + images.len())
}

/// What one phase costs per voxel, whatever kind of work it runs.
///
/// **The fold over `phase.slots` is only an answer for a pixel phase.** A
/// fragment or iterative phase owns no slot, so that fold returns `0.0` — and it
/// returns it for a phase that has forgotten its `work` entry as readily as for
/// one that genuinely computes nothing. Those two are not the same statement, so
/// they do not get the same answer here: the second is an error naming the phase.
///
/// The iterative case is **one substage's** cost, which is what
/// [`crate::iterate::IterativeOp::cost_per_voxel`] documents itself as and is the
/// only figure available — the substage count is a fixed point over data, so no
/// plan holds it.
///
/// **What that costs was got wrong here once and is worth stating carefully.**
/// This doc claimed the missing count could not move the block edge, because a
/// common factor across the candidates cannot move an argmin. The claim was
/// measured and is false, twice over:
///
/// * the count is not a common factor of the whole price. A substage reads and
///   computes; the image is written **once**, after the loop. So `S` substages
///   are `S x (read + compute) + write`, and pricing at `S == 1` over-weights the
///   write against the rest by a residual that varies with the block edge.
/// * even at a fixed count, scaling the compute moves the edge above one worker.
///   [`crate::strategy::phase_makespan`] is a roofline and compute is in its pool
///   but not in its channel bound, so raising it walks the phase across the knee.
///   Flat at `workers == 1`, where the objective collapses to serial work and the
///   largest candidate always wins; moves at every worker count above it.
///   `tests/phase_pricing.rs` sweeps both.
///
/// The first of those is a correction this function cannot make, because the
/// count is not in the plan. See the note on [`PhaseCost`]: what is wanted is a
/// statement of **which terms of a phase repeat and which happen once**, which is
/// a fact about `strategy::run_iterative_phase` rather than about any builder.
pub(crate) fn phase_compute_per_voxel(
    slots: &[&Chain],
    phase: &PhaseDecomposition,
    work: Option<&crate::fragment::PhaseWork<'_>>,
    model: &CostModel,
) -> Result<f64> {
    match work {
        // A fragment or iterative op is one op and its own family, so the
        // correction is looked up by its own name rather than folded over slots
        // it does not have.
        Some(crate::fragment::PhaseWork::Fragments(op)) => {
            Ok(op.cost_per_voxel() * model.correction_for(op.name()))
        }
        Some(crate::fragment::PhaseWork::Iterate(op)) => {
            Ok(op.cost_per_voxel() * model.correction_for(op.name()))
        }
        Some(crate::fragment::PhaseWork::IterateReduce(op)) => {
            Ok(op.cost_per_voxel() * model.correction_for(op.name()))
        }
        Some(crate::fragment::PhaseWork::Pixels) | None => {
            if phase.slots.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "phase {:?} owns no chain slot and `work` does not say what it runs, so its                      compute would be priced at zero — which is a number nobody stated. A                      fragment or iterative phase must appear in `work`; see                      `fragment::PhaseWork`.",
                    phase.names
                )));
            }
            Ok(compute_charge_per_voxel(
                slots,
                &phase.slots,
                phase.grid.block(),
                model,
            ))
        }
    }
}

/// How many arrays a block of `phase` reads and whether it writes one.
///
/// Both are facts the price needs and the grid cannot supply; see
/// [`PhaseTraffic`]. The read count comes from the plan — it is the same count
/// [`Decomposition::exact_read_voxels`] adds up — and the write comes from the
/// op, because a fragment phase writes an image only if its op says so.
///
/// **A slotless phase with no `work` entry is refused**, on the argument
/// [`phase_compute_per_voxel`] makes: `reads_input_image` defaults true and
/// `writes_an_image` would default true, so such a phase would be priced for a
/// read and a write it may do neither of, silently.
/// [`PhaseTraffic`] for one phase of a plan — what it reads, whether it writes,
/// and how many times its read-and-compute happen.
///
/// Public where `phase_traffic` is `pub(crate)` because a caller pricing a plan
/// needs the answer and has no business constructing one: `images_read` is
/// derived from the phase and `repeats` is `1` until a measurement says
/// otherwise, and both are facts about the plan rather than choices.
pub fn phase_traffic_of(
    index: usize,
    decomposition: &Decomposition,
    work: &[crate::fragment::PhaseWork<'_>],
) -> Result<PhaseTraffic> {
    let phase = decomposition.phases.get(index).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "phase {index} of a decomposition with {}",
            decomposition.n_phases()
        ))
    })?;
    phase_traffic(index, phase, work.get(index))
}

pub(crate) fn phase_traffic(
    index: usize,
    phase: &PhaseDecomposition,
    work: Option<&crate::fragment::PhaseWork<'_>>,
) -> Result<PhaseTraffic> {
    let writes_an_image = match work {
        Some(entry) => entry.writes_an_image(),
        None => {
            if phase.slots.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} owns no chain slot and `work` does not say what it runs, so                      the plan cannot say whether it reads or writes an image at all. A fragment                      or iterative phase must appear in `work`; see `fragment::PhaseWork`."
                )));
            }
            true
        }
    };
    Ok(PhaseTraffic {
        images_read: phase.images_read(index).len(),
        writes_an_image,
        // **Zero, and this is the one path where that is right.** The term is a
        // fact about the *chain*, which a `PhaseDecomposition` does not hold —
        // and every caller of this function either owns no chain slots (a
        // fragment or iterative phase, which holds nothing of its own) or is
        // reporting rather than pricing a budget. The budget is decided in
        // `strategy`, where the slots are in scope and
        // `resident_buffers_of` answers; see `PhaseTraffic::chain_buffers`.
        chain_buffers: 0,
        // One, always, and a caller who has measured otherwise says so with
        // `PhaseTraffic::repeating`. The plan cannot know: an iteration runs to
        // convergence over data, which is why `IterativeOp` has a `limit` and no
        // count.
        repeats: 1,
    })
}

pub fn compute_per_voxel(slots: &[&Chain], group: &[usize], block: [usize; 3]) -> f64 {
    group
        .iter()
        .map(|&slot| slots[slot].cost_per_voxel_in(block))
        .sum()
}

/// [`compute_per_voxel`] with the model's **per-family corrections** applied:
/// `sum over slots of declared x correction`.
///
/// The figure [`price_phase`] wants, and the one that reduces to
/// [`compute_per_voxel`] exactly when [`CostModel::compute_of`] is empty —
/// which is the shipped model, so every plan and every recorded figure taken
/// under it is unchanged.
///
/// **Keyed by the slot's display name**, which is the same key
/// `statistics::Term::ComputeOf` records against and the same one
/// `simulate::phase_rates_from_snapshot` looks a phase's rate up by. Three
/// places, one key, and no translation between them: a family that is spelled
/// differently in any of the three is a family that silently falls back, so
/// they share the string rather than a convention about it.
pub fn compute_charge_per_voxel(
    slots: &[&Chain],
    group: &[usize],
    block: [usize; 3],
    model: &CostModel,
) -> f64 {
    group
        .iter()
        .map(|&slot| {
            let chain = slots[slot];
            chain.cost_per_voxel_in(block) * model.correction_for(&chain.display_name())
        })
        .sum()
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
        // The **image the phase reads**, not the phase's own volume. A mandate
        // is about the region an op is handed, and that region — `source` — is
        // in the image below's coordinate space. The two are the same for every
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

/// Every source leaf in `chain` names an image its phase can actually read, and
/// the plan records which ones.
///
/// **The guard that cannot live in [`Decomposition::check`]**, on exactly
/// [`check_block_constraints`]' argument: a plan records op names, not
/// implementations, so the plan alone cannot see the leaves. The executor is
/// the first place holding both halves, and it runs this before the first
/// block — a forward reference is a fact about the plan, and a plan that is not
/// a plan should be refused as one rather than survive until some block asks
/// for an image nothing has written.
///
/// Four things are checked, and each of them is a way for a well-formed,
/// complete, wrong volume to come out otherwise:
///
/// * **the image exists.** An index past the end is not a reference to
///   anything.
/// * **it is not a forward reference.** Phases run `0..n`, so image `s` is
///   written by phase `s - 1` and is only there for a phase that runs after it:
///   `s <= p` for phase `p`, which reads image `p`. Refused *by name*, saying
///   which phase writes the image and which reads it. (`s == p` is the phase's
///   own input read a second time — degenerate, harmless, and not worth a
///   special case that would then have to be right.) A **supplied** input is
///   written by no phase and existed before the first one ran, so this question
///   does not arise for it: every phase may read it, and that is the whole of
///   what makes it an input rather than an early image.
/// * **a supplied input says what it holds.** There is no fold to ask, so a
///   reader that names one without declaring its element type is refused, and
///   two readers that declare different ones are refused by name.
/// * **it is on the same lattice.** A source leaf has reach 0 and reads the
///   block's own fetch region, so the image it reads has to be in the same
///   coordinate space as the image the phase reads. A different volume would
///   make the same integers mean different voxels, which is the failure this
///   crate exists to prevent rather than one to price.
/// * **the declared element type is the image's.** The leaf carries a `Dtype`
///   because `Chain::produces` has nothing else to answer with, and every fold
///   of the chain was built from that answer; here it meets the plan, which is
///   the only thing that knows what the image holds.
///
/// Finally the recorded `source_images` must be exactly what the slots name —
/// a plan whose record disagrees with its chain would read one image and price
/// another, and the whole reason the field exists is that it is parity-visible.
/// How many block buffers a run of chain slots holds inside itself.
///
/// **A phase is a `Chain::Sequence` of its slots**, and asking the sequence is
/// not the same as asking each slot: a run of four maps holds two intermediates
/// because the walk ping-pongs, where summing the slots would say zero and
/// counting them would say three. [`Chain::resident_block_buffers`] holds that
/// rule and every other, measured against a global allocator; this only has to
/// build the sequence it is a rule about.
///
/// Nothing is cloned and no `Chain` is built: a `Chain` owns boxed trait objects
/// and is not `Clone`, which is why the rule is an associated function taking
/// references rather than a method on a node this would have had to construct.
pub(crate) fn resident_buffers_of(slots: &[&Chain], group: &[usize]) -> usize {
    let children: Vec<&Chain> = group.iter().map(|&slot| slots[slot]).collect();
    Chain::sequence_resident_buffers(&children)
}

pub fn check_source_images(chain: &Chain, decomposition: &Decomposition) -> Result<()> {
    let slots = chain.slots();
    for (phase_index, phase) in decomposition.phases.iter().enumerate() {
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            // Out of range is caught, with a better message, by the executor's
            // own slot-order check.
            continue;
        }
        // **A phase with no slot is not this guard's to speak for.** It folds
        // the chain, and a phase owning no part of the chain has nothing here
        // that could name an image: a fragment op declares its second image on
        // itself, and only the `(plan, work)` pair holds the op. That half is
        // `fragment::check_phase_work`, which makes every assertion below and
        // two more the chain has no way to need. Asserting `source_images` is
        // empty here would refuse exactly the plans that guard exists to check.
        if phase.slots.is_empty() {
            continue;
        }
        let volume = decomposition.volume_at(phase_index);
        let mut declared: Vec<crate::op::SourceInput> = Vec::new();
        for &slot in &phase.slots {
            for input in slots[slot].source_inputs(volume)? {
                match declared.iter_mut().find(|held| held.image == input.image) {
                    Some(held) => held.reach = held.reach.max(&input.reach)?,
                    None => declared.push(input),
                }
            }
        }
        declared.sort_by_key(|input| input.image);

        // **The equal-reach limit, stated where it is checkable.** The executor
        // reads a source image at the block's own fetch region, so an operand
        // wanting *more* than the phase already fetches would be handed a buffer
        // narrower than its kernel walks. Per-input halos are what would lift
        // this, and nothing shipped needs them yet — the masked-window case that
        // motivated per-input reach reads its operand over the same element it
        // reads its input over, so the two are equal by construction. Refused by
        // name rather than planned and discovered as an out-of-bounds read.
        let block = phase.grid.block();
        let granted = phase.halo.in_voxels(block);
        for input in &declared {
            let wanted = input.reach.in_voxels(block);
            for axis in 0..3 {
                let (want_lo, want_hi) = wanted.axis(axis).bound(volume[axis]);
                let (have_lo, have_hi) = granted.axis(axis).bound(volume[axis]);
                if want_lo > have_lo || want_hi > have_hi {
                    return Err(Error::InvalidArgument(format!(
                        "decomposition phase {phase_index} ({}) reads image {} with a reach of \
                         {want_lo}+{want_hi} on axis {axis}, and the phase is granted a halo of \
                         {have_lo}+{have_hi} there. A source image is read at the block's own \
                         fetch region, so an operand reaching further than the phase does would \
                         be handed a buffer its kernel walks past the end of. Widening only this \
                         input needs a per-input halo, which this plan cannot express.",
                        phase.names.join(">"),
                        input.image
                    )));
                }
            }
        }

        // A supplied input has no producing phase, so nothing in the plan can
        // be folded to say what is in it: the readers are the declaration. One
        // that says nothing would leave `dtype_at` guessing, and every fold the
        // chain makes is built on that answer.
        for input in &declared {
            if input.image.is_supplied() && input.dtype.is_none() {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads {}, and nothing says what it \
                     holds. An image the run writes has its element type in the fold of the chain \
                     that wrote it; a supplied input is produced by no phase, so the reader is \
                     the only statement there is — declare it with `SourceInput::holding`, or \
                     read the array through a `Chain::Source` leaf, which carries it.",
                    phase.names.join(">"),
                    describe_image(input.image.index())
                )));
            }
        }
        let mut held: Vec<(usize, Dtype)> = declared
            .iter()
            .filter(|input| input.image.is_supplied())
            .filter_map(|input| input.dtype.map(|dtype| (input.image.index(), dtype)))
            .collect();
        held.sort_by_key(|(image, _)| *image);
        held.dedup();
        if held != phase.supplied_dtypes {
            return Err(Error::InvalidArgument(format!(
                "decomposition phase {phase_index} ({}) records that its supplied inputs hold \
                 {:?}, and its slots declare {:?}. The recorded list is what `dtype_at` answers \
                 with and what the fingerprint hashes, so a plan whose record disagrees with its \
                 chain would allocate one width and read another.",
                phase.names.join(">"),
                phase.supplied_dtypes,
                held
            )));
        }
        for (image, dtype) in &phase.supplied_dtypes {
            let plan = decomposition.dtype_at(*image);
            if plan != *dtype {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads {} as {}, and another phase of \
                     the same plan reads it as {}. One array is handed to the run and every \
                     phase that names it is handed the same bytes, so the two cannot both be \
                     right.",
                    phase.names.join(">"),
                    describe_image(*image),
                    dtype.numpy_name(),
                    plan.numpy_name()
                )));
            }
        }

        let mut named: Vec<usize> = declared.iter().map(|input| input.image.index()).collect();
        named.sort_unstable();
        named.dedup();
        if named != phase.source_images {
            return Err(Error::InvalidArgument(format!(
                "decomposition phase {phase_index} ({}) records that it also reads image(s) \
                 {:?}, and its slots name {:?}. The recorded list is what the executor reads and \
                 what the fingerprint hashes, so a plan whose record disagrees with its chain \
                 would price one image and read another.",
                phase.names.join(">"),
                phase.source_images,
                named
            )));
        }
        for &image in &named {
            if is_supplied_image(image) {
                // No producing phase, so neither the bound nor the forward
                // reference is a question that can be asked of it. What is left
                // is the lattice, which is checked below for every image alike.
                let read = decomposition.volume_at(phase_index);
                let stored = decomposition.volume_at(image);
                if stored != read {
                    return Err(Error::InvalidArgument(format!(
                        "decomposition phase {phase_index} ({}) reads {}, which is {stored:?}, \
                         beside image {phase_index}, which is {read:?}. A supplied input is read \
                         at the block's own fetch region — a source leaf has reach 0 — so it has \
                         to be in the same coordinate space as the image the phase reads, and \
                         that space is image 0's. A phase downstream of one that changes shape \
                         cannot read a supplied array directly.",
                        phase.names.join(">"),
                        describe_image(image)
                    )));
                }
                continue;
            }
            if image >= decomposition.n_images() {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads image {image} through a source \
                     leaf, and this plan has {} image(s), numbered 0 to {}.",
                    phase.names.join(">"),
                    decomposition.n_images(),
                    decomposition.n_images() - 1
                )));
            }
            if image > phase_index {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads image {image} through a source \
                     leaf, but image {image} is written by phase {}, which runs after it. Phases \
                     run in order, so a source leaf may only name an image at or below the one its \
                     phase is handed — image {phase_index} here.",
                    phase.names.join(">"),
                    image - 1
                )));
            }
            let read = decomposition.volume_at(phase_index);
            let stored = decomposition.volume_at(image);
            if stored != read {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) reads image {image}, which is \
                     {stored:?}, beside image {phase_index}, which is {read:?}. A source leaf has \
                     reach 0 and is read at the block's own fetch region, so the two images have \
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

/// Walk one slot's source leaves and compare each declaration against the image.
fn check_declared_source_dtypes(
    node: &Chain,
    phase_index: usize,
    phase: &PhaseDecomposition,
    decomposition: &Decomposition,
) -> Result<()> {
    match node {
        Chain::Op(_) => Ok(()),
        Chain::Source { image, dtype } => {
            let image = image.index();
            let held = decomposition.dtype_at(image);
            if held != *dtype {
                return Err(Error::InvalidArgument(format!(
                    "decomposition phase {phase_index} ({}) has a source leaf declaring image \
                     {image} holds {}, and the plan folds that image to {}. Every fold of the \
                     chain — what the combine accepts, what the phase writes — was built from \
                     the declaration, so it has to be the image's own element type.",
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
/// * a phase whose image is allocated at one type while its ops write another —
///   the message names the image, because that *is* the plan being wrong.
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
        // arm existed, a `volume -> fragments` op that widened its image — a
        // labelling writing `u32` over a `bool` mask — was refused by a message
        // about ops the phase does not have.
        if let Some(crate::fragment::PhaseWork::Fragments(op)) = work.get(index) {
            if !op.writes_pixels() {
                // Terminal as far as images go; there is no image to be wrong.
                continue;
            }
            let produced = op.produces(current);
            let declared = decomposition.dtype_at(index + 1);
            if declared != produced {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs fragment op {:?}, which reads {} and writes {}, but \
                     the plan allocates image {} as {}. A phase that changes the element type \
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
                "phase {index} ({}) reads {} and its ops write {}, but the plan allocates image \
                 {} as {}. A phase that changes the element type says so in its own `dtype`; a \
                 plan that does not is a plan whose image is the wrong width.",
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

/// **An op's write extent must be derivable without the plan, or the op must
/// say that it is not.**
///
/// The fourth guard that cannot live in [`Decomposition::check`], for the reason
/// [`check_block_constraints`] and [`check_dtypes`] cannot: a plan records op
/// *names*, not implementations, so the plan alone cannot ask an op anything.
///
/// **What went missing.** The executor compares what a phase's ops declare they
/// produce against the read extent the plan derived (`strategy::run_task`). That
/// is a real check only while the two sides are derived independently, and
/// [`BlockOp::placed_output_shape`] opened a door out of it: an op whose write
/// extent is not a function of its read extent may take the extent from
/// [`Placement::writes`], which *is* the read extent, and the comparison then
/// compares the plan with itself. One op does that legitimately and pays for it
/// with a check of its own against the buffer it was handed. Nothing required
/// the payment, so any op could take the door and say nothing — which is the
/// hazard `env.rs`' argument for `apply_with` is about, in its second instance.
///
/// **Exact tiling is not the replacement**, and it is worth saying why here
/// rather than leaving the next reader to re-derive it:
///
/// * it already holds, twice. [`Decomposition::check`] runs
///   [`boxes_tile_exactly`] over each phase's valid regions, and `execute_phases`
///   runs it again over the regions the executor *actually wrote* — the same
///   statement about the run rather than about the plan.
/// * it could not catch this even if it did not hold. The region a task writes is
///   [`BlockGeometry::valid`], which comes from the plan, and `Placement::writes`
///   is handed *to* the op by the executor and is the plan's own read extent. An
///   op influences neither, so it cannot make the union of written regions gap or
///   overlap. Taking the plan's extent is a wrong-*value* hazard — a kernel
///   filling a correctly shaped, correctly placed buffer out of a fetch that
///   could not support it — and no statement about coverage can see one.
///
/// **What this asks instead.** [`Placement::writes`] is the only channel by which
/// the plan's answer reaches an op, so a placement carrying none is a question
/// the plan cannot answer on the op's behalf. Every block is therefore priced
/// twice: once with the placement the executor will really pass, and once with
/// `writes` stripped from it.
///
/// * **The two agree.** The op's own arithmetic reproduces the plan's extent, so
///   the executor's per-block comparison is a real check whatever the op does
///   with the placement internally. This is every op that takes the default —
///   which ignores the placement entirely, so the answers cannot differ — and
///   also an op that reads `writes` while keeping the `output_shape` behind it
///   maintained. Nothing is owed.
/// * **They differ.** The op answered out of the plan, the per-block comparison
///   has gone vacuous, and that is allowed — but only if the op
///   [`declares`][`BlockOp::takes_extent_from_placement`] it. An op that has
///   not is refused by name, here, before a block is read.
///
/// The declaration is not a formality: it is what makes the obligation in
/// `placed_output_shape`'s doc — *the op then owes a check of its own, in its
/// kernel* — attach to a greppable set of ops rather than to every op that might
/// one day override the method. It does not verify that the kernel check exists;
/// nothing the framework can ask would. What it removes is the *silence*.
///
/// **A phase whose volume changes is not by itself a case for this.** A crop or
/// a regrid can be stated entirely in the plan — every block fetches a translated
/// region and writes its own read extent — with an op that does not resize at
/// all. Both askings then return the read extent and the phase passes, which is
/// right: nothing was traded away there.
///
/// **Phases with no slots are skipped**, not defaulted: a fragment or iterative
/// phase owns no chain slot, so there is nothing to ask. The same arrangement
/// [`check_dtypes`] has.
///
/// [`BlockOp::placed_output_shape`]: crate::op::BlockOp::placed_output_shape
/// [`BlockOp::takes_extent_from_placement`]: crate::op::BlockOp::takes_extent_from_placement
/// [`Placement::writes`]: crate::op::Placement::writes
/// [`BlockGeometry::valid`]: crate::geometry::BlockGeometry::valid
pub fn check_output_shapes(
    chain: &Chain,
    decomposition: &Decomposition,
    work: &[crate::fragment::PhaseWork<'_>],
) -> Result<()> {
    let slots = chain.slots();
    for (index, phase) in decomposition.phases.iter().enumerate() {
        match work.get(index) {
            None | Some(crate::fragment::PhaseWork::Pixels) => {}
            // Owns no chain slot; see the note above.
            Some(_) => continue,
        }
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            // A slot index out of range is caught, with a better message, by the
            // executor's own slot-order check. Nothing to say here.
            continue;
        }
        let parts: Vec<&Chain> = phase.slots.iter().map(|&slot| slots[slot]).collect();
        // Asked once per phase rather than once per block: the declaration is a
        // property of the op, so a phase that has it is exempt whatever its
        // blocks look like, and a phase that has it not is checked at all of
        // them.
        if parts.iter().any(|part| part.takes_extent_from_placement()) {
            continue;
        }
        let reads = decomposition.volume_at(index);
        let writes = decomposition.volume_at(index + 1);
        for block in &phase.blocks {
            let fetched = block_extent(&block.source);
            let taken = Placement::new(
                Anchor::of_region(&block.source, reads)?,
                Anchor::of_region(&block.read, writes)?,
            )
            .writing(block_extent(&block.read));
            // The same placement with the one field the plan speaks through
            // removed. `place_parts` may still derive an extent for an inner
            // boundary from a member's own `keeps_grid`, and that is wanted: it
            // is the ops' declaration, not the plan's.
            let unaided = Placement::new(taken.input.clone(), taken.output.clone());
            let with_plan = crate::op::parts_output_shape(&parts, &taken, fetched)?;
            let without_plan = crate::op::parts_output_shape(&parts, &unaided, fetched)?;
            if with_plan != without_plan {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} ({}) block {:?} fetches {fetched:?}: asked what it writes with \
                     the plan's extent in hand its ops answer {with_plan:?}, and asked with \
                     `op::Placement::writes` withheld they answer {without_plan:?}. An op is \
                     entitled to take its write extent from the plan — its two extents need not \
                     be a function of each other — but it has to say so, because the executor's \
                     own comparison of the declared shape against the derived read extent then \
                     compares the plan with itself and stops being a check. Declare it with \
                     `BlockOp::takes_extent_from_placement`, and owe the check that replaces it: \
                     against the buffer the block was actually handed, in the kernel.",
                    phase.names.join(">"),
                    block.index
                )));
            }
        }
    }
    Ok(())
}

/// A region's extent as the triple the op traits speak in.
fn block_extent(region: &Region) -> [usize; 3] {
    [region.shape[0], region.shape[1], region.shape[2]]
}

/// **Every chunk of an image is written by exactly one task.**
///
/// The fifth guard that cannot live in [`Decomposition::check`], for the same
/// reason [`check_block_constraints`] and [`check_dtypes`] cannot: a plan says
/// nothing about how an image is chunked, so the plan alone cannot answer this.
/// The chunk shapes come from whatever holds the storage, and the first place
/// that holds both halves is the environment's `prepare`.
///
/// Precisely: for phase `p`, which writes image `p+1`, every chunk of image
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
/// is what a halo is — so `chunks[0]` is never looked at: image 0 is nobody's
/// output. It is still taken, so that the argument is "the chunk shape of every
/// image" and a caller does not have to remember an off-by-one.
///
/// A chunk overhanging the volume's far edge is not a violation. It holds no
/// voxel outside the array, so the one valid region that meets it owns every
/// voxel it can hold — which is exactly the ownership this asserts.
pub fn check_chunk_exclusive_writes(
    decomposition: &Decomposition,
    chunks: &[[usize; 3]],
) -> Result<()> {
    if chunks.len() != decomposition.n_images() {
        return Err(Error::InvalidArgument(format!(
            "chunk-exclusive check: this plan has {} image(s) and {} chunk shape(s) were given. \
             The argument is one shape per image, image 0 included, so that the index is the \
             image number.",
            decomposition.n_images(),
            chunks.len()
        )));
    }
    for (index, phase) in decomposition.phases.iter().enumerate() {
        let image = index + 1;
        let chunk = chunks[image];
        let volume = phase.volume();
        for axis in 0..3 {
            if chunk[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "chunk-exclusive check: image {image} is chunked {chunk:?}, and a chunk of \
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
                    image,
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
    image: usize,
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
        "decomposition phase {index} ({}) writes image {image}, which is chunked {chunk:?}: chunk \
         {chunk_index:?} spans {lo:?}..{hi:?} and {} of this phase's valid regions land in it — \
         {}. Every chunk of an image must be written by exactly one task: two writers of one chunk \
         lose each other's bytes in any store whose partial write is a read-modify-write, and a \
         chunk with no single owner has no lifetime a cache can hold it by. Either the block grid \
         must be a whole multiple of the chunk shape, or — for an image whose layout nobody outside \
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
    use crate::reach::AxisReach;

    #[test]
    fn cut_masks_enumerate_every_contiguous_partition() {
        assert_eq!(groups_for(0b00, 3), vec![vec![0, 1, 2]]);
        assert_eq!(groups_for(0b01, 3), vec![vec![0], vec![1, 2]]);
        assert_eq!(groups_for(0b10, 3), vec![vec![0, 1], vec![2]]);
        assert_eq!(groups_for(0b11, 3), vec![vec![0], vec![1], vec![2]]);
        // **Every mask is a *different* partition**, which is what "enumerate"
        // claims and what the line here used to look like it checked. It was
        // `(0..1 << 7).map(..).count() == 128` — the count of a range, true by
        // construction whatever `groups_for` returned, so a `groups_for` that
        // answered the same partition every time passed it.
        let partitions: std::collections::BTreeSet<Vec<Vec<usize>>> =
            (0..1u32 << 7).map(|mask| groups_for(mask, 8)).collect();
        assert_eq!(
            partitions.len(),
            128,
            "eight slots have 2^7 contiguous partitions and every mask must give a distinct one"
        );
        // And a mask's cut count is its population count: one more group than
        // cuts, which is the other half of "contiguous".
        for mask in 0..1u32 << 7 {
            assert_eq!(
                groups_for(mask, 8).len(),
                mask.count_ones() as usize + 1,
                "mask {mask:#09b}"
            );
        }
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

    // ------------------------------------- taking the plan's write extent --
    //
    // `check_output_shapes`, watched passing and firing, on a pair of ops that
    // differ in exactly one thing: whether the arithmetic behind the extent they
    // take from the plan says the same thing the plan does.

    /// A decimation that answers [`crate::op::BlockOp::placed_output_shape`] out
    /// of [`crate::op::Placement::writes`] — the door
    /// `LatticeInterpolateOp` walks through legitimately.
    ///
    /// `honest` decides whether the `output_shape` behind that answer is
    /// maintained. Both halve axis 0 when they run; the dishonest one *says* it
    /// keeps the extent, which is the shape of the mistake an op makes by
    /// overriding `placed_output_shape` and then never exercising the method it
    /// bypassed.
    #[derive(Debug)]
    struct PlanFedDecimateOp {
        honest: bool,
    }

    impl crate::op::BlockOp for PlanFedDecimateOp {
        fn name(&self) -> &'static str {
            "plan-fed-decimate"
        }

        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }

        fn accepts(&self, _dtype: Dtype) -> bool {
            true
        }

        /// Honestly halving, on **both** ops: the geometry is not where the lie
        /// is. An op that declared it kept its grid would be caught by
        /// `place_parts` propagating the input extent forward, which is a
        /// different mistake and not the one under test.
        fn geometry(&self, input_volume: [usize; 3]) -> crate::op::Geometry {
            crate::op::Geometry::new(
                [input_volume[0] / 2, input_volume[1], input_volume[2]],
                vec![crate::op::InputMap::Stencil(Reach::none())],
            )
        }

        fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
            if self.honest {
                [input[0] / 2, input[1], input[2]]
            } else {
                input
            }
        }

        /// The plan's extent, when the plan states one. Verbatim the pattern the
        /// migration made available to every op.
        fn placed_output_shape(&self, input: [usize; 3], at: &Placement) -> [usize; 3] {
            at.writes().unwrap_or_else(|| self.output_shape(input))
        }

        fn apply(
            &self,
            _input: &crate::voxels::Voxels,
            _out: &mut crate::voxels::Voxels,
            _at: &Anchor,
        ) -> Result<()> {
            Ok(())
        }
    }

    /// The same broken arithmetic, with the waiver declared. Stands in for the
    /// op that legitimately has no inverse.
    #[derive(Debug)]
    struct WaivedDecimateOp;

    impl crate::op::BlockOp for WaivedDecimateOp {
        fn name(&self) -> &'static str {
            "waived-decimate"
        }

        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }

        fn accepts(&self, _dtype: Dtype) -> bool {
            true
        }

        fn geometry(&self, input_volume: [usize; 3]) -> crate::op::Geometry {
            crate::op::Geometry::new(
                [input_volume[0] / 2, input_volume[1], input_volume[2]],
                vec![crate::op::InputMap::Stencil(Reach::none())],
            )
        }

        fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
            input
        }

        fn placed_output_shape(&self, input: [usize; 3], at: &Placement) -> [usize; 3] {
            at.writes().unwrap_or_else(|| self.output_shape(input))
        }

        fn takes_extent_from_placement(&self) -> bool {
            true
        }

        fn apply(
            &self,
            _input: &crate::voxels::Voxels,
            _out: &mut crate::voxels::Voxels,
            _at: &Anchor,
        ) -> Result<()> {
            Ok(())
        }
    }

    /// Image 0 is `[16, 4, 4]`, the phase writes `[8, 4, 4]`, and each block
    /// fetches twice its read extent from the image below.
    fn decimating_plan() -> Decomposition {
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let phase = PhaseDecomposition::derive(
            vec![0],
            vec!["halve".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid,
        )
        .with_sources(|block| {
            Region::new(
                &[block.read.start[0] * 2, 0, 0],
                &[block.read.shape[0] * 2, 4, 4],
            )
        });
        Decomposition {
            volume: [16, 4, 4],
            dtype: Dtype::F64,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        }
    }

    /// **The check that was lost, watched being vacuous.**
    ///
    /// The executor compares what a phase's ops declare against the read extent
    /// the plan derived. Both ops here answer that comparison out of the plan, so
    /// it passes for the one whose arithmetic is wrong exactly as it does for the
    /// one whose arithmetic is right. Nothing about this is a bug in either op —
    /// it is the comparison having become the plan against itself, and it is why
    /// a second, independent question has to be asked somewhere.
    #[test]
    fn the_per_block_comparison_passes_for_a_wrong_op_and_a_right_one_alike() {
        let plan = decimating_plan();
        let block = &plan.phases[0].blocks[0];
        for honest in [true, false] {
            let op = Chain::op(PlanFedDecimateOp { honest });
            let at = Placement::new(
                Anchor::of_region(&block.source, plan.volume_at(0)).unwrap(),
                Anchor::of_region(&block.read, plan.volume_at(1)).unwrap(),
            )
            .writing(block_extent(&block.read));
            let produced =
                crate::op::parts_output_shape(&[&op], &at, block_extent(&block.source)).unwrap();
            assert_eq!(
                produced,
                block_extent(&block.read),
                "the per-block comparison should be satisfied by construction, honest = {honest}"
            );
        }
    }

    /// The guard, on the same pair, asked the one way the plan cannot answer for
    /// them: the same block again with `Placement::writes` withheld.
    ///
    /// The honest op's own arithmetic reproduces the plan's extent, so its two
    /// answers agree and nothing is owed even though it reads `writes`. The
    /// wrong one's do not agree, and it has not declared the waiver.
    #[test]
    fn the_withheld_question_separates_the_wrong_op_from_the_right_one() {
        let plan = decimating_plan();
        let work = vec![crate::fragment::PhaseWork::Pixels];

        let honest = Chain::op(PlanFedDecimateOp { honest: true });
        check_output_shapes(&honest, &plan, &work).unwrap();

        let wrong = Chain::op(PlanFedDecimateOp { honest: false });
        let err = check_output_shapes(&wrong, &plan, &work)
            .unwrap_err()
            .to_string();
        assert!(err.contains("phase 0 (halve)"), "got: {err}");
        // The extent it answered with the plan in hand, and the one it answered
        // without — the two things this compares.
        assert!(
            err.contains("[4, 4, 4]") && err.contains("[8, 4, 4]"),
            "got: {err}"
        );
        assert!(err.contains("withheld"), "got: {err}");
        assert!(
            err.contains("takes_extent_from_placement"),
            "the refusal has to name the way out of it, got: {err}"
        );
    }

    /// And the waiver, taken: the same wrong arithmetic is admitted once the op
    /// says it answers from the plan.
    ///
    /// This is the guard's honest limit, asserted rather than left implicit. It
    /// removes the *silence*, not the possibility: an op that declares the waiver
    /// owes a check against the buffer it was handed, in its kernel, and no
    /// question the framework can ask verifies that the check is there.
    #[test]
    fn a_declared_waiver_is_admitted_and_that_is_the_limit() {
        let plan = decimating_plan();
        let work = vec![crate::fragment::PhaseWork::Pixels];
        let waived = Chain::op(WaivedDecimateOp);
        check_output_shapes(&waived, &plan, &work).unwrap();
    }

    /// And through the executor, which is where a plan off a wire arrives.
    ///
    /// The guard fires before `prepare` and before a block is touched, which is
    /// the whole point of it being a guard on the plan rather than a check inside
    /// a kernel — so the honest arm gets as far as running and the dishonest one
    /// does not get as far as allocating.
    #[test]
    fn the_executor_refuses_a_plan_whose_ops_do_not_add_up_to_it() {
        let plan = decimating_plan();
        let hints = crate::strategy::Hints::default();
        for (honest, expected) in [(true, true), (false, false)] {
            let input = crate::voxels::Voxels::zeros(Dtype::F64, [16, 4, 4]).unwrap();
            let env =
                crate::env::ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).unwrap();
            let workflow = crate::strategy::Workflow::new(
                Chain::op(PlanFedDecimateOp { honest }),
                [16, 4, 4],
                Dtype::F64,
            );
            let outcome = crate::strategy::execute("test", &workflow, &plan, &hints, &env);
            assert_eq!(
                outcome.is_ok(),
                expected,
                "honest = {honest}, got {:?}",
                outcome.err().map(|err| err.to_string())
            );
            if let Err(err) = outcome {
                assert!(err.to_string().contains("withheld"), "got: {err}");
            }
        }
    }

    /// One shape per image, image 0 included, so that the index is the image
    /// number — and a caller who gets that wrong is told rather than checked
    /// against the wrong grid.
    #[test]
    fn the_chunk_check_wants_one_shape_per_image() {
        let plan = decomposition([4, 0, 0], [4, 0, 0]);
        let err = check_chunk_exclusive_writes(&plan, &[[8, 8, 8]])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 image(s)") && err.contains("1 chunk shape"),
            "got: {err}"
        );
        // Image 0 is read-only, so whatever it is chunked as says nothing.
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
                PhaseTraffic::one_in_one_out(),
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
        let cost = price_phase(
            &grid,
            &volume.into(),
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
            PhaseTraffic::one_in_one_out(),
        );
        assert!(
            cost.redundancy > 1.0,
            "a phase whose every voxel depends on the whole volume priced at \
             redundancy {}",
            cost.redundancy
        );

        // and a bounded reach on an axis the block spans is still free, because
        // there the clamp is exact: the read cannot leave the volume
        let bounded = price_phase(
            &grid,
            &[4, 0, 0].into(),
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
            PhaseTraffic::one_in_one_out(),
        );
        assert_eq!(bounded.redundancy, 1.0);
    }

    /// The same defect, in the form [`cuttable_axes`] creates: a phase the reach
    /// has left with no cut anywhere is not an interior block either.
    ///
    /// And the line the second condition holds. An axis is charged for it only
    /// when the grid has a *single block*; an uncut axis beside a cut one is a
    /// phase that is still blocked and streamed, and there the clamp discount is
    /// exact whatever the halo does. Charging that would invent traffic no plan
    /// can incur — measured, it moves partitions of chains the floor never
    /// touches.
    #[test]
    fn a_single_block_phase_whose_halo_spans_an_axis_is_not_priced_as_free_either() {
        let volume = [4096, 4, 4];
        let model = CostModel::default();
        let nearly: Reach = [4095, 0, 0].into();
        assert!(
            !nearly.is_whole_axis(0, volume[0]),
            "bounded, not a barrier"
        );
        assert!(halo_spans_axis(&nearly, 0, volume[0]), "lo + hi >= extent");

        let single = BlockGrid::whole(volume).unwrap();
        let charged = price_phase(
            &single,
            &nearly,
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
            PhaseTraffic::one_in_one_out(),
        )
        .redundancy;
        assert!(charged > 1.0, "priced at the clamp discount: {charged}");
        // strictly under a barrier's charge at the same grid: `lo + hi` is under
        // `2 * extent` for a bounded reach and exactly `2 * extent` for `All`
        let barrier = price_phase(
            &single,
            &Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()]),
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
            PhaseTraffic::one_in_one_out(),
        )
        .redundancy;
        assert_eq!(barrier, 3.0);
        assert!(charged < barrier, "{charged} against {barrier}");

        // an uncut axis beside a cut one keeps the discount, and it is exact:
        // the block spans the axis, so the read is the volume, once
        let volume = [1024, 32, 32];
        let cut = BlockGrid::along(volume, &[0], 128).unwrap();
        let spanning: Reach = [24, 24, 24].into();
        assert!(halo_spans_axis(&spanning, 1, volume[1]));
        let cost = price_phase(
            &cut,
            &spanning,
            1.0,
            1,
            false,
            8.0,
            &model,
            1.0,
            PhaseTraffic::one_in_one_out(),
        )
        .redundancy;
        // **Re-measured when the read charge stopped being an infinite-grid
        // one.** This used to be `(128 + 48) / 128 = 1.375`, every block paying
        // a full halo on both sides. Axis 0 has eight blocks of 128 over 1024
        // and a halo of 24 a side: six interior blocks read `24 + 128 + 24`, and
        // the two on the boundary read `128 + 24`, because there is nothing
        // outside the volume to read. The mean is `1360 / 8 = 170`.
        //
        // The number is derived here rather than pasted, so that a change to the
        // grid or the halo moves both sides together and this stays a statement
        // about *which axes are charged* — which is what it is for, and which is
        // unchanged: axes 1 and 2 are spanned by the block, so they contribute
        // exactly 1 and the whole ratio is axis 0's.
        let interior = 24.0 + 128.0 + 24.0;
        let boundary = 128.0 + 24.0;
        let mean_read = (6.0 * interior + 2.0 * boundary) / 8.0;
        assert_eq!(mean_read, 170.0);
        assert_eq!(cost, mean_read / 128.0, "axis 0 only, and no other");
        assert_eq!(cost, 1.328125);
    }

    /// The two-sided form of the barrier predicate, and exact like it.
    #[test]
    fn the_halo_spans_an_axis_when_its_two_sides_together_cover_the_extent() {
        let bounded: Reach = [2048, 0, 0].into();
        assert!(halo_spans_axis(&bounded, 0, 4096), "2048 + 2048 == 4096");
        assert!(!halo_spans_axis(&bounded, 0, 4097));
        // a one-sided reach is measured on the side it has, not on twice it
        let one_sided = Reach::asymmetric([(4096, 0), (0, 0), (0, 0)]);
        assert!(halo_spans_axis(&one_sided, 0, 4096));
        assert!(!halo_spans_axis(&one_sided, 0, 4097));
        // and an axis of extent 1 is excluded, for the reason
        // `reaches_whole_axis` excludes it: every block already spans it
        assert!(!halo_spans_axis(&bounded, 0, 1));
        // a full reach is the case where one side already suffices
        assert!(halo_spans_axis(
            &Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()]),
            0,
            4096
        ));
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
            PhaseTraffic::one_in_one_out(),
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
    /// `[16, 4, 4]` image, phase 1 is cut from `[8, 4, 4]` and reads the top half
    /// of the image below.
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

    /// The read figure counts images the phase reads, and no others.
    ///
    /// Four cases, and the third is the one this test was written for: a phase
    /// that reads a second array and *not* the image it was handed — a fragment
    /// op declaring `reads_pixels() == false` alongside `source_inputs` — used
    /// to be charged for two arrays and reads one. The over-count was a whole
    /// halo-inflated image, on exactly the plans the figure is checked against a
    /// run for.
    #[test]
    fn a_phase_is_charged_for_the_images_it_reads_and_not_for_the_one_it_declines() {
        let base = PhaseDecomposition::derive(
            vec![0],
            vec!["only".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new([16, 4, 4], [8, 4, 4]).unwrap(),
        );
        let voxels = 16 * 4 * 4;
        let plan = |phase: PhaseDecomposition| Decomposition {
            volume: [16, 4, 4],
            dtype: Dtype::F64,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        };

        // its own image, which is every chain phase
        assert!(base.reads_input_image);
        assert_eq!(plan(base.clone()).exact_read_voxels(), vec![voxels]);
        // its own image and one more
        let with_source = base.clone().with_source_images([0]);
        assert_eq!(
            plan(with_source.clone()).exact_read_voxels(),
            vec![2 * voxels]
        );
        // one other image and not its own
        let source_only = with_source.reading_input_image(false);
        assert_eq!(plan(source_only).exact_read_voxels(), vec![voxels]);
        // and neither: a fragments-to-fragments phase moves no voxels at all
        let neither = base.reading_input_image(false);
        assert_eq!(plan(neither).exact_read_voxels(), vec![0]);
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
    fn a_fetch_outside_the_image_it_reads_is_refused() {
        let mut plan = two_volumes([0, 0, 0], [0, 0, 0]);
        // strip the mapping: the blocks fall back to their own read extents,
        // which are regions of [8, 4, 4] and not of the [16, 4, 4] below
        plan.phases[1] = plan.phases[1].clone().with_sources(|block| {
            Region::new(&[block.read.start[0] + 12, 0, 0], &block.read.shape.clone())
        });
        let message = plan.check().unwrap_err().to_string();
        assert!(
            message.contains("reads from image 1") && message.contains("region axis 0"),
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

    /// The element type is folded from image 0, so a phase that says nothing
    /// hands on what it read.
    #[test]
    fn the_element_type_is_per_image_and_folded_from_the_input() {
        let mut plan = two_volumes([0, 0, 0], [0, 0, 0]);
        assert_eq!(plan.dtype_at(0), Dtype::F64);
        assert_eq!(plan.dtype_at(2), Dtype::F64);
        assert_eq!(plan.uniform_dtype(), Some(Dtype::F64));
        plan.phases[0] = plan.phases[0].clone().with_dtype(Dtype::U8);
        assert_eq!(plan.dtype_at(0), Dtype::F64);
        assert_eq!(plan.dtype_at(1), Dtype::U8);
        // phase 1 declares nothing, so image 2 is what image 1 was
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

    /// **The rule, and the boundary it is a description of.**
    ///
    /// Intra-block slicing was measured at `5.39x` at one block and nothing once
    /// `n_blocks >= workers`, so the rule is not a curve to tune — it is a
    /// statement of when a worker would otherwise be parked.
    #[test]
    fn the_slab_rule_asks_for_threads_only_where_the_lattice_left_some_idle() {
        use super::SlabPolicy::{FillIdleWorkers as Fill, Off};

        // The whole point: a lattice with work for everyone gets one slab.
        for (workers, blocks) in [(40, 40), (40, 41), (40, 400), (8, 8), (2, 3)] {
            assert_eq!(
                Fill.slabs_for(workers, blocks),
                1,
                "{workers} workers over {blocks} blocks have nothing spare"
            );
        }
        // And one that does not gets the spare threads, up to the cap.
        assert_eq!(Fill.slabs_for(40, 20), 2);
        assert_eq!(Fill.slabs_for(40, 14), 2);
        assert_eq!(Fill.slabs_for(40, 10), 4);
        assert_eq!(Fill.slabs_for(40, 1), SlabPolicy::CAP, "capped, not forty");
        assert_eq!(Fill.slabs_for(1, 1), 1);
        // Degenerate inputs answer one rather than zero or a panic.
        assert_eq!(Fill.slabs_for(0, 0), 1);
        assert_eq!(Fill.slabs_for(0, 8), 1);
        assert_eq!(Fill.slabs_for(8, 0), 1);
        // The off switch is off at every input, which is what makes it an off
        // switch rather than a hint.
        for (workers, blocks) in [(40, 1), (40, 10), (1, 1), (0, 0)] {
            assert_eq!(Off.slabs_for(workers, blocks), 1);
        }
        // **Liveness.** Every assertion above would hold for a policy that
        // answered one unconditionally, so the two must be shown to differ
        // somewhere.
        assert_ne!(Fill.slabs_for(40, 1), Off.slabs_for(40, 1));
        assert_eq!(SlabPolicy::ALL.len(), 2, "a sweep must cover both");
    }

    /// **The default changes nothing**, which is what makes it safe to be the
    /// default: `Constraints::default()`'s concurrency is one, so the rule
    /// answers one slab and no recorded figure moves.
    #[test]
    fn the_default_policy_asks_for_one_slab_under_the_default_constraints() {
        let constraints = Constraints::default();
        assert_eq!(constraints.expected_concurrency, 1);
        assert_eq!(
            SlabPolicy::default().slabs_for(constraints.expected_concurrency, 1),
            1,
            "the default policy at the default concurrency must not cut"
        );
        // And it is the filling one rather than the off one, so a caller who
        // raises the concurrency gets the behaviour without asking twice.
        assert_eq!(SlabPolicy::default(), SlabPolicy::FillIdleWorkers);
    }
}
