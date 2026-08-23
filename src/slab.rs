// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Cutting one block into slabs, and running them on separate threads.**
//
// The framework's unit of parallelism is the block: at one block a phase has one
// task, and a pool of forty workers runs it on one thread with thirty-nine
// parked. This module is the second unit, below that one. It is *not* a policy —
// nothing here decides when to slice, or into how many pieces. It is the
// mechanism a planner would call, and the arithmetic that says what a cut costs.
//
// `docs/design/intra-block.md` is the measurement this exists because of. Three
// of its results shape the code below and are worth having in front of you:
//
// * **It pays in exactly one regime.** At a fixed thread budget, block-level
//   parallelism beats slicing whenever there are enough blocks to spend the
//   budget on — by 1.19x at a 160^3 block and 1.50x at 256^3, with the
//   arithmetic held identical. Slicing is what you do with threads that would
//   otherwise be parked, and it bought 4.6-5.4x on a one-block phase.
// * **Zero-copy is not a precondition, and it is not free either.** A
//   per-thread clone of the *whole* block cost at most 10% on the op that
//   measurement was taken on, because its arithmetic outweighed the copy 400 to
//   1. So this module copies: each slab gets an owned buffer of its halo-grown
//   extent, which is a fraction of a block. Where the arithmetic does *not*
//   outweigh it — a voxelwise chain, priced at about a nanosecond a voxel —
//   the copies are the whole cost, and §13.3.1 is the measurement of that.
// * **No `unsafe`, and none is needed for this shape.** Each slab writes its own
//   disjoint slice of the output through a [`crate::voxels::VoxelsMut`], and the
//   compiler is what says the slices do not overlap. This used to be one serial
//   pass at the end instead; it was the only part of a cut that did not
//   parallelise, it measured at 0.09 s for an 89.9 MB block, and it is gone. The
//   only shape that would need a raw pointer is an op that reads and writes one
//   buffer, and no op in this crate does.
//
// What makes a cut exact
// ----------------------
// A slab's *core* is what it writes; its *extent* is the core grown by the
// chain's own reach and clamped to the block. That is `decomposition.rs`'s block
// grid one level down, and it is exact for the same reason: a voxel with the
// full reach of margin around it sums exactly the values it would have summed
// over the uncut block, in the same order, so no floating-point sum is
// reassociated and the answer is **bit-identical** rather than close.
//
// It is exact at the block edge too, and by a different argument: an edge slab's
// extent clamps to the block, so the op sees the same block boundary it would
// have seen uncut. Clamping is not an approximation here, it is the identity.
//
// What this refuses, and why refusing is the whole point
// -----------------------------------------------------
// Every refusal below is a case where slicing would return a complete, well
// formed, wrong volume. `Slicing` is the declaration that gates the first and
// largest of them; see [`crate::op::Slicing`] for why it cannot be derived from
// a reach.
//
// Two entry points, and one `plan_cut` under both
// -----------------------------------------------
// `apply_sliced` is a caller *asking* for a cut, where being unable to give one
// is an error. `apply_at_most` is a planner *offering* threads to a chain it has
// not looked at — the count comes from the worker count and the block count and
// from nothing else — where an offer that cannot be taken is declined and the
// block runs uncut, exactly as it did before this module existed. They share the
// decision so that they cannot come to disagree about what is sliceable, which
// would show as a plan reporting a cut it did not make.

use crate::error::{Error, Result};
use crate::op::{Anchor, Chain, Placement, SourceInputs};
use crate::reach::Reach;
use crate::region::Region;
use crate::voxels::{Voxels, VoxelsMut};

/// One piece of a cut block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slab {
    /// What this slab writes, in the block's own coordinates.
    pub core: Region,
    /// What it must read to write that: `core` grown by the chain's reach and
    /// clamped to the block.
    pub extent: Region,
    /// Where `core` sits inside a buffer holding `extent` — the slice taken out
    /// of the slab's answer before it is placed.
    pub core_within_extent: Region,
}

/// A block cut into slabs along one axis.
///
/// One axis rather than three deliberately. A three-dimensional cut multiplies
/// the halo on every axis at once, and the measurement says the halo is what
/// ends the scaling: at thirty-two slabs of a 160-plane block the halo is
/// already triple the arithmetic on **one** axis. A caller that wants more
/// pieces than one axis affords should ask for a bigger block, not a second cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlabCut {
    axis: usize,
    block: [usize; 3],
    slabs: Vec<Slab>,
}

impl SlabCut {
    /// Cut `block` into `pieces` slabs along `axis`, each grown by `reach`.
    ///
    /// `volume` is the extent of the axis in the space the reach is stated over,
    /// which is what [`crate::reach::AxisReach::bound`] needs to answer for a
    /// reach that is `All`.
    pub fn plan(
        block: [usize; 3],
        axis: usize,
        pieces: usize,
        reach: &Reach,
        volume: [usize; 3],
    ) -> Result<Self> {
        if axis >= 3 {
            return Err(Error::InvalidArgument(format!(
                "a block is 3-D and cannot be cut along axis {axis}"
            )));
        }
        let len = block[axis];
        if pieces == 0 || pieces > len {
            return Err(Error::InvalidArgument(format!(
                "cannot cut an axis of {len} voxels into {pieces} slabs: every slab must write at \
                 least one voxel, so the piece count is bounded by the extent."
            )));
        }
        // **`All` is refused rather than clamped.** A slab whose extent is the
        // whole block does the whole block's arithmetic, so a cut into `n` of
        // them is `n` times the work for at best `n` threads — never a win, and
        // silently so. An op that reaches an entire axis is not sliceable along
        // it, and saying that here is cheaper than discovering it in a timing.
        if reach.is_whole_axis(axis, volume[axis]) {
            return Err(Error::InvalidArgument(format!(
                "the chain reaches the whole of axis {axis}, so every slab of a cut along it \
                 would read the entire block and do the entire block's arithmetic. Cutting a \
                 whole-axis reach is `n` times the work for `n` threads. Cut another axis or \
                 do not cut."
            )));
        }
        let (lo, hi) = reach.axis(axis).bound(volume[axis]);
        let slabs = (0..pieces)
            .map(|index| {
                let core_lo = index * len / pieces;
                let core_hi = (index + 1) * len / pieces;
                let ext_lo = core_lo.saturating_sub(lo);
                let ext_hi = (core_hi + hi).min(len);
                Slab {
                    core: axis_region(block, axis, core_lo, core_hi),
                    extent: axis_region(block, axis, ext_lo, ext_hi),
                    core_within_extent: axis_region(
                        extent_shape(block, axis, ext_lo, ext_hi),
                        axis,
                        core_lo - ext_lo,
                        core_hi - ext_lo,
                    ),
                }
            })
            .collect();
        Ok(Self { axis, block, slabs })
    }

    /// Cut into `pieces` along whichever axis has the most extent.
    ///
    /// Ties go to the **lowest** axis, which for this crate's C-ordered arrays
    /// is the one whose slabs are contiguous in memory. That is a locality
    /// preference and not a correctness one — `docs/design/intra-block.md` §4
    /// measured two assignments of identical arithmetic and found no difference
    /// in CPU-seconds — so it is stated as a tie-break rather than defended.
    pub fn plan_longest(
        block: [usize; 3],
        pieces: usize,
        reach: &Reach,
        volume: [usize; 3],
    ) -> Result<Self> {
        let mut best = 0usize;
        for axis in 1..3 {
            if block[axis] > block[best] {
                best = axis;
            }
        }
        Self::plan(block, best, pieces, reach, volume)
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn slabs(&self) -> &[Slab] {
        &self.slabs
    }

    pub fn len(&self) -> usize {
        self.slabs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    /// Voxels put through the kernel, against voxels written.
    ///
    /// Arithmetic rather than measurement, and it is the number a planner has to
    /// weigh against the threads a cut buys: `1.0` is a free cut and `3.0` says
    /// three threads will produce one thread's worth of progress.
    pub fn amplification(&self) -> f64 {
        let written: usize = self.block.iter().product();
        let computed: usize = self.slabs.iter().map(|slab| slab.extent.voxels()).sum();
        computed as f64 / written as f64
    }
}

fn extent_shape(block: [usize; 3], axis: usize, lo: usize, hi: usize) -> [usize; 3] {
    let mut shape = block;
    shape[axis] = hi - lo;
    shape
}

fn axis_region(block: [usize; 3], axis: usize, lo: usize, hi: usize) -> Region {
    let mut start = [0usize; 3];
    let mut shape = block;
    start[axis] = lo;
    shape[axis] = hi - lo;
    Region::new(&start, &shape)
}

/// Why this chain cannot be cut into `pieces` slabs here, or the cut itself.
///
/// **Total: every refusal is a sentence rather than an [`Error`], and that is
/// what lets there be one of these rather than two.** Two callers ask this
/// question and want opposite things done with the answer. [`apply_sliced`] was
/// *asked* to cut, so a refusal is an error there. [`apply_at_most`] was
/// *offered* threads by a planner that cannot know what the chain holds, so a
/// refusal there is simply not cutting. Deriving both from one function is what
/// keeps them from coming to disagree about what is sliceable — which would show
/// up as a plan that reports a cut it did not make, or the reverse.
///
/// Every branch below is one of the four refusals the module header lists, in
/// the order they become answerable.
fn plan_cut(
    chain: &Chain,
    input: &Voxels,
    out: &Voxels,
    at: &Placement,
    pieces: usize,
) -> std::result::Result<SlabCut, String> {
    if let Some(why) = chain.slicing().refusal() {
        return Err(format!(
            "this chain is not sliceable, so its block cannot be cut into {pieces} slabs: {why}"
        ));
    }
    // **A source leaf is refused, and this is a scope boundary rather than an
    // impossibility.** Each source buffer holds the block's whole fetch extent,
    // and a slab needs the sub-extent that *its* arm reads — which is the
    // combine's reach around the slab core, not the chain's. `fold_reach_spec`
    // folds a chain to one reach and does not expose the per-arm halo a source
    // buffer would have to be narrowed by, so slicing one today would mean
    // guessing at that number. Handing an arm too little is a wrong answer with
    // no diagnostic, so it is refused until the per-arm fold exists.
    if chain_reads_an_image(chain) {
        return Err(
            "this chain reads a stored image through a source leaf. Narrowing that buffer to a \
             slab needs the reach of the arm the leaf sits in, which `Chain::reach_spec` folds \
             away, so slicing it would have to guess at the halo. Refused rather than guessed."
                .to_string(),
        );
    }
    let block = input.shape();
    let volume = at.input.volume;
    // **The output lattice must be the input lattice.** A slab's core is the
    // same index range in both buffers, and that is only true when the op does
    // not resample. Checked against what the chain says it produces rather than
    // assumed from `Slicing::Stencil`, because two statements of one quantity
    // are two statements that can drift.
    let produced = chain.placed_output_shape(block, at).map_err(|err| {
        format!("slicing cannot ask the chain what it produces from {block:?}: {err}")
    })?;
    if produced != block || out.shape() != block {
        return Err(format!(
            "slicing needs the output lattice to be the input lattice: the block is {block:?}, \
             the chain turns that into {produced:?}, and the output buffer is {:?}. A slab's \
             core is the same index range in both buffers and that is only true when the two \
             agree.",
            out.shape()
        ));
    }
    let reach = chain
        .reach_spec(volume)
        .map_err(|err| format!("slicing cannot ask the chain for its reach: {err}"))?;
    // The whole-axis refusal and "more slabs than the axis has voxels" both live
    // in `plan_longest`, and both are refusals in exactly this sense.
    SlabCut::plan_longest(block, pieces, &reach, volume).map_err(|err| format!("{err}"))
}

/// [`Chain::apply_placed`], with the block cut into `threads` slabs run
/// concurrently.
///
/// **Bit-identical to `apply_placed` on the same arguments**, at every thread
/// count, which is the acceptance bar and is what `tests/intra_block_slicing.rs`
/// holds it to. `threads <= 1` is not a special case that skips the machinery —
/// it takes the uncut path outright, so the identity is trivially true there and
/// the test is measuring the cut for every count above it.
///
/// Everything this refuses is listed in the module header and each refusal names
/// itself. In particular it refuses a chain the ops have not **declared**
/// sliceable, and that declaration is never inferred; see [`crate::op::Slicing`].
///
/// A caller that would rather *fall back* than fail wants [`apply_at_most`];
/// the two share one private `plan_cut` so that they cannot disagree about which chains
/// are sliceable.
pub fn apply_sliced(
    chain: &Chain,
    input: &Voxels,
    sources: SourceInputs<'_>,
    out: &mut Voxels,
    at: &Placement,
    threads: usize,
) -> Result<()> {
    if threads <= 1 {
        return chain.apply_placed(input, sources, out, at);
    }
    let cut = plan_cut(chain, input, out, at, threads).map_err(Error::InvalidArgument)?;
    run_cut(chain, input, out, at, &cut)
}

/// [`apply_sliced`], where every refusal is **declining to cut** rather than
/// failing, and the answer is how many slabs actually ran.
///
/// `1` is the uncut path — [`Chain::apply_placed`], on this thread, exactly as
/// this crate ran a block before slabs existed — and it is what every chain that
/// has not declared itself a stencil gets.
///
/// **This is the entry point a planner uses, and the fallback is the whole
/// difference.** [`apply_sliced`] is a caller *asking* for a cut, where being
/// unable to give one is a failure. A slab count from
/// [`crate::decomposition::SlabPolicy`] is an *offer* made by a planner that
/// has not looked at the chain: it is derived from the worker count and the
/// block count and from nothing else. An offer that failed the run every time a
/// phase held an undeclared op would fail every plan this crate has today, so an
/// offer that cannot be taken is declined.
///
/// **Only the decision to cut is swallowed.** Once a cut is planned, everything
/// the ops do propagates: a slab that errors errors the block, and a slab that
/// panics is reported as one. A fallback that also caught those would turn a
/// broken op into a slow one.
pub fn apply_at_most(
    chain: &Chain,
    input: &Voxels,
    sources: SourceInputs<'_>,
    out: &mut Voxels,
    at: &Placement,
    slabs: usize,
) -> Result<usize> {
    if slabs <= 1 {
        chain.apply_placed(input, sources, out, at)?;
        return Ok(1);
    }
    match plan_cut(chain, input, out, at, slabs) {
        Err(_) => {
            chain.apply_placed(input, sources, out, at)?;
            Ok(1)
        }
        Ok(cut) => {
            let ran = cut.len();
            run_cut(chain, input, out, at, &cut)?;
            Ok(ran)
        }
    }
}

/// Run a planned cut: one thread per slab, each placing its own core.
///
/// Split from the two entry points above so that the *decision* to cut and the
/// *execution* of one are separately readable, and so that neither entry point
/// can grow a second copy of the thread scope.
fn run_cut(
    chain: &Chain,
    input: &Voxels,
    out: &mut Voxels,
    at: &Placement,
    cut: &SlabCut,
) -> Result<()> {
    let dtype = out.dtype();
    let axis = cut.axis();

    // **One mutable view of `out` per slab, disjoint by construction, peeled off
    // before any thread starts.**
    //
    // This is what removed the join. It used to be one serial pass at the end,
    // copying every slab's core into `out` on the thread that ran the cut — the
    // only part of a cut that did not parallelise, and therefore the whole of
    // its Amdahl term. Measured against itself, interleaved, with the two arms
    // asserted to write the same volume, threading it is **2.9-3.6x** on the
    // pass itself, and the pass grows steeply with the block: about **6 ms at
    // `160^3` against 90 ms at `224^3`**, 2.7x the bytes for 15x the time across
    // a 27.5 MiB L3. So what this buys rises with the block, which is the
    // opposite of how the halo behaves.
    //
    // The cores tile the cut axis exactly, so `VoxelsMut::split_at` can hand
    // each slab the part of `out` it writes and nothing else, and each thread
    // places its own answer. No `unsafe`: the compiler is what says the halves
    // do not overlap.
    //
    // `peeled` counts what has already been split away, because each split is
    // taken from the *remainder* and the slab's core is in the block's
    // coordinates. The last split takes the whole of what is left and answers an
    // empty second half, which is a view of nothing rather than a special case.
    let mut cores: Vec<VoxelsMut<'_>> = Vec::with_capacity(cut.len());
    {
        let mut rest = VoxelsMut::of(out);
        let mut peeled = 0usize;
        for slab in cut.slabs() {
            let end = slab.core.start[axis] + slab.core.shape[axis];
            let (mine, tail) = rest.split_at(axis, end - peeled)?;
            cores.push(mine);
            rest = tail;
            peeled = end;
        }
    }

    // Each slab computes into a buffer of its own extent, then places its core.
    // The *answer* is owned because an op writes the whole buffer it is handed
    // and a slab's extent is wider than its core — that asymmetry is arithmetic,
    // not a gap in the types, and no borrowed variant removes it. What the
    // borrowed variant removes is the second copy: the core goes straight into
    // `out`, on this slab's own thread.
    let answers: Vec<Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = cut
            .slabs()
            .iter()
            .zip(cores)
            .map(|(slab, mut core)| {
                scope.spawn(move || -> Result<()> {
                    let piece = input.slice_region(&slab.extent)?;
                    let mut answer = Voxels::zeros(dtype, piece.shape())?;
                    // The slab is told where it really sits, so an op that is a
                    // function of position answers for the voxels it is writing
                    // rather than for the block's corner.
                    let placement = slab_placement(at, slab, axis);
                    chain.apply_placed(&piece, SourceInputs::none(), &mut answer, &placement)?;
                    core.assign_from(&answer, &slab.core_within_extent)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or_else(|_| {
                    Err(Error::InvalidArgument(
                        "a slab panicked while computing its part of the block".to_string(),
                    ))
                })
            })
            .collect()
    });

    // **The first failure, and every thread is joined before it is looked at.**
    // A slab that failed has still written nothing outside its own core, so the
    // buffer a caller sees on an error is partial rather than crossed.
    for answer in answers {
        answer?;
    }
    Ok(())
}

/// Where one slab sits, in both of the block's spaces.
///
/// The slab's buffer starts at the block's own offset plus the slab extent's
/// start, on the cut axis and nowhere else.
fn slab_placement(at: &Placement, slab: &Slab, axis: usize) -> Placement {
    let shift = |anchor: &Anchor| {
        let mut offset = anchor.offset;
        offset[axis] += slab.extent.start[axis];
        Anchor::new(offset, anchor.volume)
    };
    Placement::new(shift(&at.input), shift(&at.output))
}

fn chain_reads_an_image(chain: &Chain) -> bool {
    match chain {
        Chain::Source { .. } => true,
        Chain::Op(_) => false,
        Chain::Sequence(children)
        | Chain::Alternative {
            branches: children, ..
        } => children.iter().any(chain_reads_an_image),
        Chain::Parallel { branches, .. } => branches.iter().any(chain_reads_an_image),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reach::Reach;

    fn reach(n: usize) -> Reach {
        Reach::symmetric([n, n, n])
    }

    #[test]
    fn a_cut_covers_the_block_exactly_once() {
        let block = [17usize, 5, 5];
        for pieces in 1..=17 {
            let cut = SlabCut::plan(block, 0, pieces, &reach(2), block).expect("cut");
            assert_eq!(cut.len(), pieces);
            let mut covered = vec![0usize; block[0]];
            for slab in cut.slabs() {
                for index in slab.core.start[0]..slab.core.start[0] + slab.core.shape[0] {
                    covered[index] += 1;
                }
            }
            assert!(
                covered.iter().all(|count| *count == 1),
                "{pieces} slabs did not tile the axis exactly once: {covered:?}"
            );
        }
    }

    #[test]
    fn the_extent_contains_the_core_and_the_offset_says_where() {
        let block = [20usize, 4, 4];
        let cut = SlabCut::plan(block, 0, 4, &reach(3), block).expect("cut");
        for slab in cut.slabs() {
            let ext_lo = slab.extent.start[0];
            let ext_hi = ext_lo + slab.extent.shape[0];
            assert!(ext_lo <= slab.core.start[0]);
            assert!(ext_hi >= slab.core.start[0] + slab.core.shape[0]);
            assert_eq!(
                slab.core_within_extent.start[0],
                slab.core.start[0] - ext_lo,
                "the core's offset inside the extent must be where it really is"
            );
            assert_eq!(slab.core_within_extent.shape, slab.core.shape);
        }
    }

    #[test]
    fn the_interior_margin_is_the_whole_reach_and_the_edges_clamp() {
        let block = [20usize, 4, 4];
        let halo = 3;
        let cut = SlabCut::plan(block, 0, 4, &reach(halo), block).expect("cut");
        let slabs = cut.slabs();
        // The first slab starts at the block edge and cannot grow below it; the
        // last ends at it. Every other side has the full reach.
        assert_eq!(slabs[0].extent.start[0], 0);
        assert_eq!(slabs[0].core_within_extent.start[0], 0);
        for slab in &slabs[1..] {
            assert_eq!(
                slab.core_within_extent.start[0], halo,
                "an interior slab must carry the whole reach below its core"
            );
        }
        let last = slabs.last().expect("slabs");
        assert_eq!(last.extent.start[0] + last.extent.shape[0], block[0]);
    }

    #[test]
    fn amplification_is_one_for_an_uncut_block_and_grows_with_the_cut() {
        let block = [64usize, 8, 8];
        let one = SlabCut::plan(block, 0, 1, &reach(2), block).expect("cut");
        assert!((one.amplification() - 1.0).abs() < 1e-12);
        let mut previous = one.amplification();
        for pieces in [2usize, 4, 8, 16, 32] {
            let cut = SlabCut::plan(block, 0, pieces, &reach(2), block).expect("cut");
            let now = cut.amplification();
            assert!(
                now > previous,
                "a finer cut must recompute more: {pieces} slabs gave {now} after {previous}"
            );
            previous = now;
        }
    }

    #[test]
    fn a_reach_of_nothing_costs_nothing_to_cut() {
        let block = [32usize, 4, 4];
        for pieces in [1usize, 2, 8, 32] {
            let cut = SlabCut::plan(block, 0, pieces, &Reach::none(), block).expect("cut");
            assert!(
                (cut.amplification() - 1.0).abs() < 1e-12,
                "a voxelwise chain has no halo, so cutting it must be free"
            );
        }
    }

    #[test]
    fn a_whole_axis_reach_is_refused_rather_than_cut() {
        let block = [32usize, 4, 4];
        let whole = Reach::symmetric([64, 0, 0]);
        let error = SlabCut::plan(block, 0, 4, &whole, block).expect_err("must refuse");
        assert!(
            format!("{error}").contains("entire block"),
            "the refusal must say why: {error}"
        );
        // The other axes are unaffected, which is what makes this per-axis.
        assert!(SlabCut::plan(block, 1, 4, &whole, block).is_ok());
    }

    #[test]
    fn more_slabs_than_voxels_is_refused() {
        let block = [4usize, 4, 4];
        assert!(SlabCut::plan(block, 0, 4, &reach(1), block).is_ok());
        assert!(SlabCut::plan(block, 0, 5, &reach(1), block).is_err());
        assert!(SlabCut::plan(block, 0, 0, &reach(1), block).is_err());
    }

    #[test]
    fn the_longest_axis_is_chosen_and_ties_go_to_the_lowest() {
        let cut = SlabCut::plan_longest([4, 9, 4], 3, &reach(1), [4, 9, 4]).expect("cut");
        assert_eq!(cut.axis(), 1);
        let tied = SlabCut::plan_longest([8, 8, 8], 2, &reach(1), [8, 8, 8]).expect("cut");
        assert_eq!(tied.axis(), 0, "a tie takes the contiguous axis");
    }
}
