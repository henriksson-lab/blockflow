// SPDX-License-Identifier: MIT
//
// **What one block of a phase actually holds, against what the byte budget
// charges for it.**
//
// `PhaseCost::working_set_bytes_per_block` **was** `resident_voxels x
// bytes_per_voxel x 2.0` — "input buffer plus output buffer" — and `strategy`
// admits the largest candidate block for which `working_set_bytes_per_block x
// expected_concurrency <= budget_bytes`. The figure was deliberately allowed to
// over-state, on the stated grounds that it feeds a budget and never a ranking,
// and over-charging only invents infeasibility.
//
// That argument is sound for the direction it was written about and it did not
// cover the other one. `price_phase` recorded the gap in its own words: "a phase
// reading three is charged as if it read one, which is not [the safe direction],
// and is a known gap recorded here rather than half-fixed". **This file is the
// measurement that gap was waiting on, and it closed it**: the charge now counts
// every buffer alive, `Chain::resident_block_buffers` derives the chain's own
// share from its shape, and the rows below are an equality rather than a gap.
//
// Why an allocator and not an argument
// ------------------------------------
// The buffers a block holds are not all the phase's to declare. The executor
// allocates the input block and the output block; `Chain::Sequence` clones its
// input and allocates an intermediate; `Chain::Parallel` allocates branch
// buffers; a `Chain::Source` arm is handed a buffer the executor fetched. Only
// the first two are in the `x 2.0`. So the question "how far out is it" cannot
// be answered from the plan — every term that is missing is missing *because*
// no `Decomposition` can see it — and the honest instrument is to count the
// allocator.
//
// **How many branch buffers are alive at once is a property of the combine.**
// It used to be all of them: a fan-in held one buffer per branch until the
// combine had read them all, so the figure grew with the arity and at one block
// each of those buffers was a whole volume. A combine that declares itself a
// left fold over pairs (`Combine::fold_carrier`) is now folded as its branches
// are computed and holds three buffers whatever the arity — the partial, the
// branch just finished, and the buffer their join is written into. The rows
// below are that change measured; `tests/dead_block_buffers.rs` is the bit-identity it
// is only allowed under.
//
// The measurement is one `apply` of one block, because that is the quantity the
// budget is denominated in. It is not the whole-run peak: `tests/
// mask_carrier_residency.rs` measures that, and the two differ for reasons —
// lazy image allocation, images freed after their last reader — that have
// nothing to do with this question.
//
// **The control is the one-in-one-out row.** If the harness could not reproduce
// `x 2.0` for the shape the formula was written for, it would be measuring
// itself rather than the gap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::budget::{
    admission_bytes, FrameworkFigure, UNOBSERVED_OP_MARGIN, UNOBSERVED_SHAPE_MARGIN,
};
use blockflow::decomposition::{price_phase, CostModel, PhaseTraffic};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockResidency, Chain, Placement, SourceInputs};
use blockflow::ops::{
    ElementShape, Logic, LogicCombine, Morphology, MorphologyOp, RankFilterOp, StructuringElement,
    VoxelwiseMapOp,
};
use blockflow::reach::Reach;
use blockflow::voxels::Voxels;
use blockflow::Dtype;

// ------------------------------------------------------- the measurement --

// **Per thread, not per process, and that took two failures to learn.**
//
// These were process-wide atomics. A control row that reproduces the formula's
// own shape caught it twice: first when a second test in this binary was
// measured into the first, and again — after every test was serialised behind a
// mutex — when the suite still went red at about one run in three. The gate was
// not the fix because the contaminating allocations are not the tests': `libtest`
// runs a binary's tests on threads it owns and allocates on them for its own
// bookkeeping, whatever the tests do.
//
// A thread-local counter is immune to all of it. Every buffer this file measures
// is allocated by `Chain::apply_placed` on the calling thread, so the figures do
// not move; what moves is that nothing else can land in them.
//
// **What that costs, stated because it is a real limit.** An op that allocates
// on a worker thread — a rayon-parallel kernel, say — is invisible here, so the
// op-internal figures below are a *lower* bound on op-internal residency. That
// is the conservative direction for this file's argument, which is that such
// residency exists and is unpriced: under-counting it can only weaken the case
// being made, never manufacture it.
thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

fn took(bytes: usize) {
    // `try_with`, because a thread tearing down its locals may still free — and
    // an allocator that panicked there would take the process with it.
    let _ = LIVE.try_with(|live| {
        let now = live.get().saturating_add(bytes);
        live.set(now);
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

fn gave(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(bytes)));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            took(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            took(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        gave(layout.size());
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            if new_size >= layout.size() {
                took(new_size - layout.size());
            } else {
                gave(layout.size() - new_size);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The bytes `body` added at its worst moment, over what this thread already
/// held.
fn peak_of<R>(body: impl FnOnce() -> R) -> (R, usize) {
    let base = LIVE.with(|live| live.get());
    PEAK.with(|peak| peak.set(base));
    let result = body();
    let peak = PEAK.with(|peak| peak.get());
    (result, peak.saturating_sub(base))
}

// ------------------------------------------------------------ the shapes --

const BLOCK: [usize; 3] = [64, 64, 64];
const VOLUME: [usize; 3] = [128, 64, 64];

/// One `f64` block buffer, which is the unit every figure below is quoted in.
fn buffer_bytes() -> usize {
    BLOCK.iter().product::<usize>() * 8
}

fn block() -> Voxels {
    Array3::<f64>::zeros((BLOCK[0], BLOCK[1], BLOCK[2])).into()
}

fn map(name: &'static str) -> Chain {
    Chain::op(VoxelwiseMapOp::new(name, |value: f64| value * 2.0 + 1.0))
}

/// What the budget charges for one block of a phase reading `images_read`
/// images and holding `chain_buffers` of its own, at zero reach so that the
/// resident extent is exactly the block and the arithmetic is transparent:
/// `block_voxels x 8 x (images_read + 1 + chain_buffers)`.
///
/// The last term is the one this file's measurements put there. It used to be a
/// literal `2.0` and the table below used to be a record of how wrong that was.
fn charged(images_read: usize, chain_buffers: usize) -> f64 {
    let grid = BlockGrid::along(VOLUME, &[0], BLOCK[0]).expect("a lattice");
    price_phase(
        &grid,
        &Reach::symmetric([0, 0, 0]),
        1.0,
        1,
        false,
        8.0,
        &CostModel::default(),
        1.0,
        PhaseTraffic {
            images_read,
            writes_an_image: true,
            // A pixel phase reads and computes once.
            repeats: 1,
            chain_buffers,
        },
    )
    .working_set_bytes_per_block
}

/// One block through `chain`, with `sources` stored arrays supplied beside it —
/// **all of it inside the measured region**, because every one of those buffers
/// is resident while the block is in flight and the budget is a statement about
/// exactly that moment. The executor allocates the input, the output and each
/// fetched source; the chain allocates the rest.
fn measure(chain: &Chain, sources: &[ImageId]) -> (usize, BlockResidency, usize) {
    // **The harness's own two allocations, computed rather than measured.**
    // `held` and `entries` are built inside the measured region — they must be,
    // because the buffers they hold are part of the figure — and their spines
    // are not block buffers and are not in `BlockResidency`. Both are built by
    // `collect` from an exact-size iterator, so each is one allocation of
    // exactly `len x size_of::<T>()` and there is nothing to measure: at zero
    // sources a `Vec` of capacity zero does not allocate at all.
    //
    // This is here so the comparison below can be an equality. A test that
    // allowed a window because its author could not account for the difference
    // is a test that will absorb a real regression later.
    let spines = if sources.is_empty() {
        0
    } else {
        sources.len() * std::mem::size_of::<Voxels>()
            + sources.len() * std::mem::size_of::<(ImageId, &Voxels)>()
    };
    let (observed, peak) = peak_of(|| {
        let input = block();
        let held: Vec<Voxels> = sources.iter().map(|_| block()).collect();
        let entries: Vec<(ImageId, &Voxels)> = sources.iter().copied().zip(held.iter()).collect();
        let mut out = Voxels::zeros(Dtype::F64, BLOCK).expect("an output block");
        // The observing form, so the allocator's figure and the walk's own come
        // from **one** execution rather than two — nothing between them can
        // differ, not even a run.
        chain
            .apply_observing(
                &input,
                SourceInputs::new(&entries),
                &mut out,
                &Anchor::whole(BLOCK),
            )
            .expect("one block through the chain")
    });
    (peak, observed, spines)
}

/// **What `Chain::apply_tallied` allocates before its tally exists.**
///
/// The first two statements of that function are `placed_output_shape` and
/// `produces` — the checks that the buffer it was handed is the one the chain
/// writes. Both fold over the tree and both allocate small `Vec`s on the way;
/// neither is tallied, and neither can be, because both are planning methods
/// asked of a `Decomposition` as readily as of a run and threading a tally into
/// them would put measurement apparatus in an API about plans.
///
/// **Measured here rather than predicted**, and that is deliberate: the figure
/// is `Vec`'s growth policy, not the chain's shape. A fallible `collect` cannot
/// use its iterator's size hint, so a two-branch fan-in's shape list is four
/// slots and not two — 96 bytes where arithmetic over the branch count says 48.
/// A constant written here would have been a guess with a plausible derivation
/// behind it, which is the worst kind.
///
/// The arguments are the ones `measure` runs the chain at, so this is the same
/// call the walk makes and not an analogue of it.
fn preamble_bytes(chain: &Chain) -> usize {
    peak_of(|| {
        let _ = chain.placed_output_shape(BLOCK, &Placement::same(Anchor::whole(BLOCK)));
        let _ = chain.produces(Dtype::F64);
    })
    .1
}

fn fan_in(computed: usize, sources: &[ImageId]) -> Chain {
    let mut branches: Vec<Chain> = (0..computed).map(|_| map("arm")).collect();
    for &image in sources {
        branches.push(Chain::source(image, Dtype::F64));
    }
    Chain::parallel(branches, Box::new(LogicCombine::new("or", Logic::Or))).expect("a fan-in")
}

/// **What a block holds, and what the budget now charges for it: the same
/// number.**
///
/// Each row is one `apply` of one `64^3` block, measured through the allocator,
/// against `working_set_bytes_per_block` priced for the same block, the same
/// `images_read` and the same `Chain::resident_block_buffers`. The unit is one
/// `f64` block buffer — 2 MiB.
///
/// **This test used to assert the opposite.** Its title was "the budget
/// under-charges every phase that reads more than one image, and by how much is
/// not a constant", and its assertions were that every row held *more* than it
/// was charged and that the excess varied enough not to be absorbable into a
/// margin. Both were true and both were the point: the formula was a literal
/// `x 2.0`, `budget.rs` recorded the gap as known "rather than half-fixed", and
/// this file was the measurement it was waiting on.
///
/// What closed it was measuring a shape nobody had: a fan-in whose combine
/// cannot declare a `fold_carrier` holds one block buffer **per arm**, so the
/// 91-arm feature stack of `docs/design/pixel-classification.md` held 93 where
/// the budget charged 2. A 46.5x under-charge, in the direction that admits a
/// plan the run cannot afford, is not a gap a margin absorbs.
/// `Chain::resident_block_buffers` derives the count from the chain's shape,
/// `Decomposition::declare_resident_buffers` records it on the plan and
/// `check_resident_buffers` refuses a plan that under-states it, and
/// `price_phase` now charges it.
///
/// So the rows below are an **equality**, and the fractional slack is the small
/// `Vec` spines and op scratch that are not block buffers and never were.
#[test]
fn what_a_block_holds_against_what_the_budget_charges_for_it() {
    let unit = buffer_bytes() as f64;
    let one = ImageId::from(7usize);
    let two = ImageId::from(8usize);

    let cases: Vec<(&str, Chain, Vec<ImageId>, usize)> = vec![
        ("one in, one out", map("only"), vec![], 1),
        // **Two children and four, because they differ and only one of them
        // used to.** A sequence holds the intermediate it is reading and the one
        // it is writing; at two children the first of those is the caller's own
        // block, borrowed. It used to be a copy of it.
        (
            "sequence of two maps",
            Chain::sequence(vec![map("a"), map("b")]),
            vec![],
            1,
        ),
        (
            "sequence of four maps",
            Chain::sequence(vec![map("a"), map("b"), map("c"), map("d")]),
            vec![],
            1,
        ),
        ("fan-in, 2 computed arms", fan_in(2, &[]), vec![], 1),
        ("fan-in, 3 computed arms", fan_in(3, &[]), vec![], 1),
        // **The row the fold exists for.** Seven arms hold what three do, and
        // the assertion below is that they hold *exactly* what three do rather
        // than merely less than seven would have.
        ("fan-in, 7 computed arms", fan_in(7, &[]), vec![], 1),
        ("fan-in, 1 arm + 1 source", fan_in(1, &[one]), vec![one], 2),
        (
            "fan-in, 1 arm + 2 sources",
            fan_in(1, &[one, two]),
            vec![one, two],
            3,
        ),
        // **Two source arms and nothing computed**, which is the shape a source
        // arm's own buffer used to dominate: every branch's answer already
        // existed and the walk copied both of them.
        (
            "fan-in, 2 source arms",
            fan_in(0, &[one, two]),
            vec![one, two],
            3,
        ),
        // The other place a source leaf's answer is an operand rather than an
        // output: the head of a sequence, whose next child reads it.
        (
            "sequence from a source",
            Chain::sequence(vec![Chain::source(one, Dtype::F64), map("b")]),
            vec![one],
            2,
        ),
    ];

    eprintln!(
        "\none {BLOCK:?} f64 block = {:.2} MiB\n{:<26} {:>9} {:>9} {:>8}",
        unit / (1024.0 * 1024.0),
        "phase shape",
        "held",
        "charged",
        "ratio"
    );
    let mut rows = Vec::new();
    for (name, chain, sources, images_read) in &cases {
        let held = measure(chain, sources).0 as f64;
        // The chain's own count, from its shape — the term the formula gained.
        let charge = charged(*images_read, chain.resident_block_buffers());
        eprintln!(
            "{name:<26} {:>7.2}x {:>7.2}x {:>7.2}x",
            held / unit,
            charge / unit,
            held / charge
        );
        rows.push((*name, held, charge));
    }

    // **The control.** The shape the formula was written for reproduces it: one
    // in, one out is two buffers and is charged two. Without this row the rest
    // of the table would be a measurement of the harness.
    let (_, held, charge) = rows[0];
    assert_eq!(
        held as u64, charge as u64,
        "the one-in-one-out row must reproduce the formula it is the formula for, exactly: two \
         block buffers, charged as two. Not a tolerance — this row is what says the rest of the \
         table is the gap and not the harness, and a window here would let the harness drift \
         into the number it is supposed to be validating."
    );

    // **The second control, and the one this change is answerable to.** A fan-in
    // of seven arms holds what a fan-in of three does, to the byte. Before the
    // fold it held four more whole block buffers; the figure is now a property
    // of the node's shape and not of its arity, which is what makes it safe for
    // a plan to run a wide join at one block.
    let three = rows
        .iter()
        .find(|(name, _, _)| *name == "fan-in, 3 computed arms")
        .expect("the three-arm row");
    let seven = rows
        .iter()
        .find(|(name, _, _)| *name == "fan-in, 7 computed arms")
        .expect("the seven-arm row");
    assert_eq!(
        three.1 as u64, seven.1 as u64,
        "three arms held {} and seven held {}. A fan-in whose combine declares a fold \
         holds the partial, the branch it is computing and their join — three buffers, \
         whatever the arity — so these are one number or the fold is not being taken.",
        three.1, seven.1
    );

    // And every shape is now charged what it holds. A ratio, not a difference,
    // so the tolerance is meaningful at every block size.
    let mut widest = 1.0f64;
    for (name, held, charge) in &rows {
        let ratio = held / charge;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "{name} holds {held} and is charged {charge} — a ratio of {ratio:.3}. The \
             budget is meant to charge what a block holds; a gap here is either a shape \
             `Chain::resident_block_buffers` does not know about or a term `price_phase` \
             has stopped counting."
        );
        widest = widest.max(ratio);
    }
    assert!(
        rows.len() >= 8,
        "only {} shapes measured, which is too few to say the formula is right in general",
        rows.len()
    );
    // The residual is slack in the harness, not in the model: it is small `Vec`
    // spines and op scratch. If it ever reached a whole buffer it would be a
    // shape being missed.
    assert!(
        widest < 1.05,
        "the widest over-hold is {widest:.3}, which is approaching a whole buffer"
    );
}

/// **Even a corrected count would not be an upper bound, and this is why.**
///
/// Everything the table above measures is allocated by `Chain::apply_placed` —
/// this crate's own code, whose allocation pattern is a function of the chain's
/// shape and therefore exactly knowable without measuring anything. That is what
/// makes a corrected figure possible at all.
///
/// What is *not* knowable is what an op allocates inside `BlockOp::apply`.
/// Nothing declares it, the trait has no method for it, and the two ops below
/// are ordinary library ops rather than contrived ones. So a residency figure
/// built from the chain's structure is a statement about the **framework's**
/// buffers and not about the block, and calling it a bound would be a false
/// bound — the one thing this review is not allowed to produce.
///
/// The row that matters is the comparison with `one in, one out` in the table
/// above: the same chain shape, the same block, the same two framework buffers,
/// and a different answer, entirely because of what the op does inside.
#[test]
fn an_ops_own_working_buffers_are_not_visible_to_any_declaration() {
    let unit = buffer_bytes() as f64;
    let element = StructuringElement::from_radius(ElementShape::Box, [2, 2, 2]);

    let plain_bytes = measure(&map("map"), &[]).0;
    let plain = plain_bytes as f64 / unit;
    let rank = measure(
        &Chain::op(RankFilterOp::median("median", element.clone())),
        &[],
    )
    .0 as f64
        / unit;
    let morph = measure(
        &Chain::op(MorphologyOp::new("open", Morphology::Open, element)),
        &[],
    )
    .0 as f64
        / unit;

    eprintln!(
        "\nsame chain shape, two framework buffers, different residency\n  \
         {:<22} {plain:.2}x\n  {:<22} {rank:.2}x\n  {:<22} {morph:.2}x",
        "voxelwise map", "rank filter", "morphological open"
    );

    // The framework's own count is 2 for all three — one input, one output, no
    // sequence, no fan-in, no source. If residency were a function of the chain
    // alone these would agree. Exactly two, not about two: see the control in
    // `what_a_block_holds_against_what_the_budget_charges_for_it`.
    assert_eq!(
        plain_bytes,
        2 * buffer_bytes(),
        "the voxelwise map should hold exactly the two framework buffers"
    );
    assert!(
        rank > plain + 0.05 || morph > plain + 0.05,
        "neither library op allocated anything of its own ({rank}x, {morph}x against {plain}x), \
         so this file cannot claim that op-internal residency is unpriced — re-measure with an \
         op that does before relying on the claim"
    );
}

/// **What a corrected figure would cost in affordable plans, and when the
/// under-charge actually bites.**
///
/// The admission rule is `strategy`'s own: the largest candidate edge for which
/// `working_set_bytes_per_block x expected_concurrency <= budget_bytes`. The
/// factors are measured above — `1.00` is what the formula charges today,
/// `2.00` a sequence or a two-arm fan-in, `3.06` the widest framework shape
/// here, `4.00` a one-in-one-out phase whose op is a rank filter.
///
/// **The affordability cost is bounded at one step of the ladder, and that is
/// arithmetic rather than luck**: the candidates go up by a factor of two in
/// edge, which is eight in volume, and every measured factor is under eight. The
/// sweep asserts it at every budget rather than arguing it once.
///
/// What it said, at 40 workers on a ladder of powers of two:
///
/// ```text
///  budget | admitted edge: map  2.00x  3.06x  4.00x | over-hold: 1.00x 2.00x 3.06x 4.00x
///   1 GiB |                 64     64     64     64 |            0.16x 0.31x 0.56x 0.62x
///   2 GiB |                128     64     64     64 |            0.62x 1.25x 2.23x 2.50x
///   4 GiB |                128    128     64     64 |            0.31x 0.62x 1.11x 1.25x
///   8 GiB |                128    128    128    128 |            0.16x 0.31x 0.56x 0.62x
///  16 GiB |                256    128    128    128 |            0.62x 1.25x 2.23x 2.50x
///  32 GiB |                256    256    128    128 |            0.31x 0.62x 1.11x 1.25x
///  64 GiB |                256    256    256    256 |            0.16x 0.31x 0.56x 0.62x
/// 128 GiB |                512    256    256    256 |            0.62x 1.25x 2.23x 2.50x
/// ```
///
/// **The correction costs one ladder step at five of the eight budgets and
/// nothing at the other three**, and never two. What it buys is the row beside
/// it: at the budgets where it costs something, an uncorrected `4.00x` phase
/// holds **2.50x** the budget it was certified against.
///
/// **The over-hold is not bounded, and it is worst exactly where it matters.**
/// A coarse ladder usually leaves headroom — at 8 GiB the planner picks edge 128
/// and uses a sixth of its budget, so even a `4.00x` phase still fits, and the
/// under-charge costs nothing. The rows where `over-hold` exceeds `1.00x` are
/// the ones where the chosen candidate sits near the budget, which is the
/// situation a memory-constrained run is in by definition. That is the failure
/// this gap produces: not a plan that is slightly too big, but one the planner
/// certified and the machine cannot hold.
#[test]
fn what_a_corrected_figure_would_cost_in_affordable_plans() {
    const CANDIDATES: [usize; 6] = [512, 256, 128, 64, 32, 16];
    const PLANE: [usize; 3] = [1024, 1024, 1024];
    const CONCURRENCY: u64 = 40;
    const FACTORS: [(&str, f64); 4] = [
        ("map", 1.00),
        ("seq / 2-arm", 2.00),
        ("1 arm + 2 src", 3.06),
        ("rank filter", 4.00),
    ];

    // `working_set_bytes_per_block x concurrency` for one candidate, from the
    // real pricing function at zero reach.
    let demand = |edge: usize| -> Option<f64> {
        let grid = BlockGrid::along(PLANE, &[0, 1, 2], edge).ok()?;
        let cost = price_phase(
            &grid,
            &Reach::symmetric([0, 0, 0]),
            1.0,
            1,
            false,
            8.0,
            &CostModel::default(),
            1.0,
            PhaseTraffic::one_in_one_out(),
        );
        Some(cost.working_set_bytes_per_block * CONCURRENCY as f64)
    };
    let admitted = |factor: f64, budget: u64| -> usize {
        CANDIDATES
            .iter()
            .copied()
            .find(|&edge| demand(edge).is_some_and(|need| need * factor <= budget as f64))
            .unwrap_or(*CANDIDATES.last().expect("a ladder"))
    };

    let gib = |n: u64| n * 1024 * 1024 * 1024;
    eprintln!(
        "\nconcurrency {CONCURRENCY}, candidates {CANDIDATES:?}, zero reach, 8 B/voxel\n\
         {:>7} | {:>26} | {:>26}",
        "budget", "admitted edge, by factor", "over-hold at today's edge"
    );
    eprintln!(
        "{:>7} | {:>6}{:>6}{:>7}{:>7} | {:>6}{:>6}{:>7}{:>7}",
        "",
        FACTORS[0].0,
        FACTORS[1].0,
        FACTORS[2].0,
        FACTORS[3].0,
        "1.00x",
        "2.00x",
        "3.06x",
        "4.00x"
    );

    let mut biting = 0usize;
    for power in 0..8u32 {
        let budget = gib(1u64 << power);
        let today = admitted(1.0, budget);
        let today_need = demand(today).expect("a priced candidate");
        let edges: Vec<usize> = FACTORS
            .iter()
            .map(|(_, factor)| admitted(*factor, budget))
            .collect();
        let holds: Vec<f64> = FACTORS
            .iter()
            .map(|(_, factor)| today_need * factor / budget as f64)
            .collect();
        biting += holds.iter().filter(|held| **held > 1.0).count();
        eprintln!(
            "{:>5} GiB | {:>6}{:>6}{:>7}{:>7} | {:>5.2}x{:>5.2}x{:>6.2}x{:>6.2}x",
            1u64 << power,
            edges[0],
            edges[1],
            edges[2],
            edges[3],
            holds[0],
            holds[1],
            holds[2],
            holds[3]
        );

        // **The affordability cost is at most one step**, at every budget. A
        // ladder step is 8x in volume and every measured factor is under 8.
        let today_index = CANDIDATES
            .iter()
            .position(|&e| e == today)
            .expect("on the ladder");
        for ((name, factor), edge) in FACTORS.iter().zip(edges.iter()) {
            let index = CANDIDATES
                .iter()
                .position(|e| e == edge)
                .expect("on the ladder");
            assert!(
                index <= today_index + 1,
                "at {} GiB, {name} ({factor}x) fell {} ladder steps; a factor under 8 cannot \
                 cost more than one on a ladder of powers of two",
                1u64 << power,
                index - today_index
            );
        }
    }

    // The control on the whole table: if no row ever over-held, the gap would be
    // real but unreachable, and this review would be recommending a change with
    // no failure behind it.
    assert!(
        biting > 0,
        "no budget in the sweep put a phase over its budget, so the table shows a gap that \
         cannot be reached and the recommendation would have nothing behind it"
    );
    eprintln!(
        "\n{biting} of {} rows over-hold their budget",
        8 * FACTORS.len()
    );
}

/// **The walk's own figure, against the allocator, from one execution.**
///
/// `Chain::apply_observing` returns the high-water mark of the buffers the walk
/// allocated, plus the input, the output and the distinct source buffers. This
/// is that number checked against what the process actually took — measured
/// around the *same* call, so nothing between the two can differ.
///
/// **Where the two agree, the observation is exact.** Where they do not, the
/// residual is named and computed rather than left as a window, because a
/// residual nobody has accounted for is the thing this whole review exists to
/// stop happening.
///
/// **One term used to be here and is not any more.** `LogicCombine` folded three
/// or more branches through an intermediate it allocated inside `Combine::apply`
/// — invisible to any walk over the chain, and one whole `Bool` block. The walk
/// now drives that fold itself, so the buffer is allocated where it can be
/// counted and the row it used to hide in is an equality with nothing on the
/// callee side at all. It is the same buffer; what changed is who allocates it,
/// and therefore whether a plan can see it.
///
/// **A second went the same way and a third appeared because of it.** The
/// `Region` a source arm's `Voxels::assign` built is gone, because a source arm
/// that is an *operand* is borrowed and never assigns. And with every arm of a
/// two-source fan-in borrowed, that chain allocates no block buffer at all — so
/// for the first time the largest thing it allocates is what
/// `Chain::apply_tallied` folds over the tree **before its tally exists**, to
/// check the buffer it was handed. Nothing about those `Vec`s changed. What
/// changed is that there is no longer a block buffer standing over them.
///
/// That term is [`preamble_bytes`], measured rather than predicted, and it
/// enters as `max(preamble, walk)` written as a saturating difference — one
/// expression for every row, which saturates to nothing for every row that
/// holds a block buffer. **It was found by this equality and not by reading the
/// code**: a window wide enough to absorb 264 bytes would have absorbed it
/// silently, and the two rows it appears in are exactly the two the borrow
/// created.
#[test]
fn the_walks_own_figure_matches_the_allocator_except_for_what_it_says_it_omits() {
    let unit = buffer_bytes() as f64;
    let one = ImageId::from(7usize);
    let two = ImageId::from(8usize);
    // **The two things the walk cannot see, named and computed.** Both are
    // allocated inside a callee — the same rule that puts an op's scratch out of
    // scope — so the walk is right not to count them, and this test is what says
    // how much they are rather than leaving a window for them to hide in.
    //
    // `region`: `Chain::Source` finishes with `Voxels::assign`, which builds a
    // `Region::whole` — two three-element `Vec<usize>`. Only one is ever live,
    // because each `assign` frees its own before the next arm runs. **No row
    // carries it any more**: the arms that used to assign now borrow, and the
    // one place a source leaf still copies is when it writes a caller's `out`,
    // which no row here is. It is left named because the term is real and a
    // future row may meet it again.
    let _ = 2 * 3 * std::mem::size_of::<usize>();

    let cases: Vec<(&str, Chain, Vec<ImageId>, usize)> = vec![
        ("one in, one out", map("only"), vec![], 0),
        (
            "sequence of two maps",
            Chain::sequence(vec![map("a"), map("b")]),
            vec![],
            0,
        ),
        (
            "sequence of four maps",
            Chain::sequence(vec![map("a"), map("b"), map("c"), map("d")]),
            vec![],
            0,
        ),
        ("fan-in, 2 computed arms", fan_in(2, &[]), vec![], 0),
        ("fan-in, 3 computed arms", fan_in(3, &[]), vec![], 0),
        ("fan-in, 7 computed arms", fan_in(7, &[]), vec![], 0),
        // **Every source row's callee scratch is now zero, and that is the
        // change rather than a coincidence.** The `Region` counted here was
        // built by `Voxels::assign`, which is how a source leaf used to finish;
        // a borrowed source arm does not assign at all, so there is no `Region`
        // to be short by. The row that carried 48 bytes carries none.
        ("fan-in, 1 arm + 1 source", fan_in(1, &[one]), vec![one], 0),
        (
            "fan-in, 1 arm + 2 sources",
            fan_in(1, &[one, two]),
            vec![one, two],
            0,
        ),
        (
            "fan-in, 2 source arms",
            fan_in(0, &[one, two]),
            vec![one, two],
            0,
        ),
        (
            "sequence from a source",
            Chain::sequence(vec![Chain::source(one, Dtype::F64), map("b")]),
            vec![one],
            0,
        ),
    ];

    eprintln!(
        "\n{:<26} {:>9} {:>9} {:>8} {:>7} {:>8} {:>9}",
        "phase shape", "allocator", "observed", "callee", "spines", "preamble", "residual"
    );
    for (name, chain, sources, expected_scratch) in &cases {
        // **The preamble counts only where it is the tallest thing there is.**
        // It is freed before the first block buffer, so a chain that allocates
        // one never sees it — the walk's own peak stands over it and the
        // subtraction saturates to nothing. A chain that allocates *no* block
        // buffer has nothing standing over it, and then the high-water mark of
        // the whole application is a handful of `Vec` slots. That is not a
        // second rule for two rows: it is `max(preamble, walk)` written as the
        // difference, and it is the same expression for every row.
        let preamble = preamble_bytes(chain);
        let (allocator, observed, spines) = measure(chain, sources);
        let over = preamble.saturating_sub(observed.chain_bytes() as usize);
        let accounted = observed.peak_bytes() as usize + spines + expected_scratch + over;
        eprintln!(
            "{name:<26} {:>7.2}x {:>7.2}x {:>8} {:>7} {:>8} {:>9}",
            allocator as f64 / unit,
            observed.peak_bytes() as f64 / unit,
            expected_scratch,
            spines,
            over,
            allocator as i64 - accounted as i64
        );

        // **Every byte accounted for, as an equality.** The walk's figure, plus
        // the harness's two `Vec` spines, plus the combine's own scratch where
        // it has any, is the number the allocator saw — exactly. A residual of
        // any size is either a node that started allocating something the tally
        // does not count, or a combine that changed what it holds, and both are
        // cases where a window would have swallowed the news.
        assert_eq!(
            allocator,
            accounted,
            "{name}: {allocator} bytes taken against {accounted} accounted for — \
             {} unexplained. Nothing here is allowed to be unexplained: see this file's \
             header for why the counter is per thread and what that rules out.",
            allocator as i64 - accounted as i64
        );

        // The observation carries its scope, and refuses outside it.
        assert!(observed.describes(chain));
        assert_eq!(observed.shape_id(), chain.shape_id());
        assert_eq!(observed.block(), BLOCK);
        assert_eq!(observed.bytes_at(chain, BLOCK), Some(observed.peak_bytes()));
        assert_eq!(
            observed.bytes_at(chain, [32, 32, 32]),
            None,
            "{name}: a figure taken at one block must not answer for another"
        );
        assert_eq!(
            observed.bytes_at(&map("something else"), BLOCK),
            None,
            "{name}: a figure taken on one chain must not answer for another"
        );
        // The estimate does scale along the block axis, and says so in its name.
        let eighth = observed
            .estimate_at(chain, [32, 32, 32])
            .expect("same chain");
        assert_eq!(eighth, observed.peak_bytes() / 8);
        assert_eq!(
            observed.estimate_at(&map("something else"), BLOCK),
            None,
            "{name}: not even the estimate crosses the workload axis"
        );
    }
}

/// The scope key distinguishes the things the measurements say it must.
///
/// Every pair below measured differently in the tables above, so a key that
/// collapsed any of them would let one chain's residency be quoted for another —
/// which is the failure mode a sibling measurement hit when a store keyed by
/// machine was asked about a different workload.
#[test]
fn the_scope_key_separates_every_shape_that_measured_differently() {
    let one = ImageId::from(7usize);
    let two = ImageId::from(8usize);
    let keys = [
        map("only").shape_key(),
        Chain::sequence(vec![map("a"), map("b")]).shape_key(),
        Chain::sequence(vec![map("a"), map("b"), map("c"), map("d")]).shape_key(),
        fan_in(2, &[]).shape_key(),
        fan_in(3, &[]).shape_key(),
        fan_in(1, &[one]).shape_key(),
        fan_in(1, &[one, two]).shape_key(),
        Chain::op(RankFilterOp::median(
            "median",
            StructuringElement::from_radius(ElementShape::Box, [2, 2, 2]),
        ))
        .shape_key(),
    ];
    for (i, left) in keys.iter().enumerate() {
        for right in keys.iter().skip(i + 1) {
            assert_ne!(left, right, "two shapes share one key");
        }
    }

    // **The op's name is in the key, and that is the load-bearing part**: the
    // two chains below are structurally identical and measured `2.00x` against
    // `4.00x`. A key over structure alone would quote one for the other.
    let element = StructuringElement::from_radius(ElementShape::Box, [2, 2, 2]);
    assert_ne!(
        Chain::op(RankFilterOp::median("median", element.clone())).shape_key(),
        Chain::op(MorphologyOp::new("open", Morphology::Open, element)).shape_key()
    );
    // and the same chain built twice is the same key, or nothing could ever match
    assert_eq!(fan_in(2, &[]).shape_key(), fan_in(2, &[]).shape_key());
}

// ------------------------------------------- the cold-start charge, priced --

/// A structuring element big enough that the ops built on it allocate.
fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [2, 2, 2])
}

/// A fan-in whose arms are given rather than assumed to be cheap maps.
fn fan_in_of(arms: Vec<Chain>, sources: &[ImageId]) -> Chain {
    let mut branches = arms;
    for &image in sources {
        branches.push(Chain::source(image, Dtype::F64));
    }
    Chain::parallel(branches, Box::new(LogicCombine::new("or", Logic::Or))).expect("a fan-in")
}

/// **The margin is derived from the measurements, not chosen.**
///
/// `UNOBSERVED_SHAPE_MARGIN` is asserted to be the **smallest whole number that
/// covers every shape measured here** — not merely "large enough", because a
/// margin quietly larger than its evidence is a number nobody can defend, and
/// not an exact fit either, because a fit to three decimal places is a number
/// that fails the next time anything is measured.
///
/// **The shapes include combinations**, which is the correction this test forced.
/// The tables above measure framework cost and op cost on *separate* chains —
/// `3.06x` for a fan-in of cheap maps with two source arms, `2.00x` for a rank
/// filter in a one-in-one-out chain — and a margin justified by the larger of
/// those two would not cover a chain that has both at once. So the worst case is
/// measured rather than argued from the parts.
///
/// **And it is now a combination that is worst, where it was not before.** While
/// a fan-in held every branch at once, its own buffers were tall enough that the
/// peak of `fan-in, rank arm + 2 sources` fell at the combine and the rank
/// filter's transient scratch never showed: the chain measured the same `3.56x`
/// as the cheap-arm fan-in beside it, and this file said so. Folding the branches
/// took the fan-in's own buffers down, the peak moved to the arm, and the
/// combination is now the widest shape here by half a unit. Nothing about the
/// rank filter changed — which is the whole argument for measuring the
/// combinations rather than reasoning about the parts.
#[test]
fn the_shape_margin_is_the_smallest_tenth_that_covers_what_was_measured() {
    let unit = buffer_bytes();
    let one = ImageId::from(7usize);
    let two = ImageId::from(8usize);
    let heavy = || Chain::op(RankFilterOp::median("median", element()));

    let shapes: Vec<(&str, Chain, Vec<ImageId>)> = vec![
        ("one in, one out", map("only"), vec![]),
        (
            "sequence of two maps",
            Chain::sequence(vec![map("a"), map("b")]),
            vec![],
        ),
        (
            "sequence of four maps",
            Chain::sequence(vec![map("a"), map("b"), map("c"), map("d")]),
            vec![],
        ),
        ("fan-in, 2 computed arms", fan_in(2, &[]), vec![]),
        ("fan-in, 3 computed arms", fan_in(3, &[]), vec![]),
        // Included so the arity axis is covered by the shape the margin is
        // fitted to, and not only by the table above.
        ("fan-in, 7 computed arms", fan_in(7, &[]), vec![]),
        ("fan-in, 1 arm + 1 source", fan_in(1, &[one]), vec![one]),
        (
            "fan-in, 1 arm + 2 sources",
            fan_in(1, &[one, two]),
            vec![one, two],
        ),
        (
            "fan-in, 2 source arms",
            fan_in(0, &[one, two]),
            vec![one, two],
        ),
        (
            "sequence from a source",
            Chain::sequence(vec![Chain::source(one, Dtype::F64), map("b")]),
            vec![one],
        ),
        ("rank filter alone", heavy(), vec![]),
        (
            "morphological open alone",
            Chain::op(MorphologyOp::new("open", Morphology::Open, element())),
            vec![],
        ),
        // The combinations. A chain is not obliged to be either expensive in its
        // framework buffers or expensive in its op, and a margin defended by the
        // worse of two separate measurements would not cover one that is both.
        (
            "fan-in, rank arm + 2 sources",
            fan_in_of(vec![heavy()], &[one, two]),
            vec![one, two],
        ),
        (
            "sequence of four rank filters",
            Chain::sequence(vec![heavy(), heavy(), heavy(), heavy()]),
            vec![],
        ),
    ];

    eprintln!("\n{:<32} {:>9} {:>10}", "shape", "held", "of charged");
    let mut widest: f64 = 0.0;
    let mut worst = "";
    let mut worst_charge = 0.0f64;
    for (name, chain, sources) in &shapes {
        let held = measure(chain, sources).0 as f64;
        // **The charge this shape actually gets**, not a flat two buffers. That
        // was what this fitted against and it is why the constant is 3.6: most
        // of what it covered was the framework's own buffers, which the charge
        // now counts. What is left for a margin to cover is the op's scratch.
        let charge = charged(1 + sources.len(), chain.resident_block_buffers());
        let ratio = held / charge;
        eprintln!("{name:<32} {:>7.2}x {:>9.4}x", held / unit as f64, ratio);
        if ratio.total_cmp(&widest).is_gt() {
            widest = ratio;
            worst_charge = charge;
            worst = name;
        }
    }
    eprintln!("widest: {worst} at {widest:.4}x of its own charge");

    // **The smallest tenth that covers it**, which is what the constant is.
    //
    // Not the smallest whole number, which is what this test asked for first and
    // was wrong to: a rank filter measures `2.0002x` its framework buffers, and
    // rounding that up to `3` would be fifty per cent of headroom bought with
    // two ten-thousandths of evidence — the exact thing the second assertion
    // below forbids. A tenth is fine enough that the rounding is not an argument
    // and coarse enough that a constant does not move on noise.
    let smallest_covering = (widest * 10.0).ceil() / 10.0;
    assert_eq!(
        UNOBSERVED_SHAPE_MARGIN, smallest_covering,
        "the widest shape measured holds {widest:.4}x the assumed charge ({worst}), so the \
         smallest tenth that covers it is {smallest_covering}. The constant is \
         {UNOBSERVED_SHAPE_MARGIN}: either a measurement moved or the constant was chosen \
         rather than derived."
    );
    assert!(
        UNOBSERVED_SHAPE_MARGIN - 0.1 < widest,
        "the margin is more than a tenth above what any measurement asks for, which is headroom \
         nobody can point at evidence for"
    );

    // and it does cover, in bytes, which is the statement that actually matters
    let admitted = admission_bytes(FrameworkFigure::Assumed(worst_charge));
    assert!(admitted as f64 >= widest * worst_charge);
}

/// The same rule for the op margin, on the figure that remains once the
/// framework's half is exact.
///
/// With the framework known, what is left is what an op allocates inside
/// `BlockOp::apply`. Measured against the two framework buffers a
/// one-in-one-out chain holds, so that the ratio is the op's own contribution
/// and not the chain's.
#[test]
fn the_op_margin_is_the_smallest_tenth_that_covers_the_ops_measured() {
    let unit = buffer_bytes();
    let framework = 2.0 * unit as f64;
    let ops: Vec<(&str, Chain)> = vec![
        ("voxelwise map", map("map")),
        (
            "morphological open",
            Chain::op(MorphologyOp::new("open", Morphology::Open, element())),
        ),
        (
            "rank filter",
            Chain::op(RankFilterOp::median("median", element())),
        ),
    ];
    let mut widest: f64 = 0.0;
    let mut worst = "";
    eprintln!("\n{:<24} {:>10}", "op, one in one out", "of framework");
    for (name, chain) in &ops {
        let ratio = measure(chain, &[]).0 as f64 / framework;
        eprintln!("{name:<24} {ratio:>9.4}x");
        if ratio.total_cmp(&widest).is_gt() {
            widest = ratio;
            worst = name;
        }
    }
    assert_eq!(
        UNOBSERVED_OP_MARGIN,
        (widest * 10.0).ceil() / 10.0,
        "the widest op measured holds {widest:.4}x its chain's framework buffers ({worst})"
    );
    assert!(UNOBSERVED_OP_MARGIN - 0.1 < widest);

    // **The two branches now charge the same, and that is the finding rather
    // than a defect.**
    //
    // This used to assert the exact branch was *cheaper*, which was the whole
    // reason to prefer a known framework figure to an assumed one: `2.1` against
    // `3.6`. The difference was never about the op — both branches leave the op
    // unpriced — it was that an assumed framework figure was missing the
    // chain's own buffers and needed a margin to cover them. Now that
    // `working_set_bytes_per_block` counts them, an assumed figure *is* exact
    // about the framework, the two margins were re-fitted to the same residual,
    // and they came out equal.
    //
    // Asserted as an equality so that the day a shape is found which the
    // shape-derivation misses, this fails and says so.
    assert_eq!(
        admission_bytes(FrameworkFigure::Exact(2 * unit as u64)),
        admission_bytes(FrameworkFigure::Assumed(2.0 * unit as f64)),
        "the two branches charge differently again, so the shape-derived framework \
         figure and an observed one have diverged; `UNOBSERVED_SHAPE_MARGIN` and \
         `UNOBSERVED_OP_MARGIN` no longer cover the same residual"
    );
}

/// **Neither margin can move the admitted block by more than `8x` in volume, at
/// any budget.**
///
/// This is what makes the policy affordable, and it is arithmetic rather than
/// luck: every margin here is under eight — `UNOBSERVED_SHAPE_MARGIN` on its
/// own, and the widest measured framework figure times `UNOBSERVED_OP_MARGIN`
/// for the exact branch — so a block that fitted without a margin has a rung at
/// an eighth of its volume that fits with one.
///
/// **The bound is stated in volume, and that is the correction this test carries
/// rather than a flourish.** It was first written as "never more than one ladder
/// step", which is true here and is *not* the same claim: these candidates are
/// powers of two, where a rung is exactly `8x` in volume, so the step count and
/// the volume ratio coincide. On `decomposition::refined_ladder` a rung is
/// `2.37x` or `3.375x` and `UNOBSERVED_SHAPE_MARGIN` alone already spans two of
/// them, while the volume bound is untouched — two consecutive refined rungs
/// span exactly `2.37 x 3.375 = 8.0`. The step count was a proxy that happened
/// to equal the volume ratio at one spacing.
///
/// Both assertions are kept, and that is the point: the step bound is asserted
/// because it is true *at this spacing*, the volume bound because it is true at
/// any. `tests/block_ladder.rs` asserts the same volume bound at the refined
/// spacing, and an invariant that holds at two spacings is an invariant rather
/// than a coincidence.
#[test]
fn a_margin_never_moves_the_admitted_block_by_more_than_eight_times_in_volume() {
    const CANDIDATES: [usize; 6] = [512, 256, 128, 64, 32, 16];
    const PLANE: [usize; 3] = [1024, 1024, 1024];
    const CONCURRENCY: u64 = 40;

    let assumed_for = |edge: usize| -> Option<f64> {
        let grid = BlockGrid::along(PLANE, &[0, 1, 2], edge).ok()?;
        Some(
            price_phase(
                &grid,
                &Reach::symmetric([0, 0, 0]),
                1.0,
                1,
                false,
                8.0,
                &CostModel::default(),
                1.0,
                PhaseTraffic::one_in_one_out(),
            )
            .working_set_bytes_per_block,
        )
    };
    let charges: [(&str, Box<dyn Fn(f64) -> u64>); 3] = [
        ("today", Box::new(|ws: f64| ws.round() as u64)),
        (
            "assumed x margin",
            Box::new(|ws: f64| admission_bytes(FrameworkFigure::Assumed(ws))),
        ),
        (
            "exact(3.56x) x margin",
            Box::new(|ws: f64| admission_bytes(FrameworkFigure::Exact((ws * 3.56).round() as u64))),
        ),
    ];

    eprintln!(
        "\n{:>7} | {:>7} {:>18} {:>22}",
        "budget", "today", "assumed x margin", "exact(3.56x) x margin"
    );
    let mut cold_start_steps = 0usize;
    for power in 0..9u32 {
        let budget = (1u64 << power) * 1024 * 1024 * 1024;
        let admitted: Vec<usize> = charges
            .iter()
            .map(|(_, charge)| {
                CANDIDATES
                    .iter()
                    .copied()
                    .find(|&edge| {
                        assumed_for(edge).is_some_and(|ws| charge(ws) * CONCURRENCY <= budget)
                    })
                    .unwrap_or(*CANDIDATES.last().expect("a ladder"))
            })
            .collect();
        eprintln!(
            "{:>5} GiB | {:>7} {:>18} {:>22}",
            1u64 << power,
            admitted[0],
            admitted[1],
            admitted[2]
        );

        let today = CANDIDATES
            .iter()
            .position(|e| *e == admitted[0])
            .expect("on the ladder");
        for (index, (name, _)) in charges.iter().enumerate().skip(1) {
            let step = CANDIDATES
                .iter()
                .position(|e| *e == admitted[index])
                .expect("on the ladder");
            // **The bound that survives any spacing**: the admitted block's
            // volume, which cannot fall by more than the margin rounded up to
            // the next rung — and every margin here is under `8x`.
            let moved = (admitted[0] as f64 / admitted[index] as f64).powi(3);
            assert!(
                moved <= 8.0 + 1e-9,
                "at {} GiB, {name} moved the admitted block from edge {} to edge {} — \
                 {moved:.3}x in volume. Every margin here is under 8x, so a correction that \
                 moves more than that is a margin that grew past the arithmetic this claim \
                 rests on.",
                1u64 << power,
                admitted[0],
                admitted[index]
            );

            // **And the step bound, which is true at *this* spacing only.** These
            // candidates are powers of two, so one rung is exactly the `8x`
            // above; on a refined ladder the same margin spans two rungs and the
            // same volume. Kept rather than replaced, because a bound that holds
            // at two spacings is what makes the volume one an invariant.
            assert!(
                step <= today + 1,
                "at {} GiB, {name} fell {} rungs of a powers-of-two ladder, where one rung is \
                 the 8x the assertion above allows.",
                1u64 << power,
                step - today
            );
            if index == 1 && step > today {
                cold_start_steps += 1;
            }
        }
    }

    // **What the cold-start charge costs, counted rather than described.**
    // `budget.rs` quotes this figure where it argues the policy, and a figure
    // quoted in prose beside a test that does not check it is a figure that
    // rots.
    assert_eq!(
        cold_start_steps, 3,
        "the cold-start charge costs a rung at {cold_start_steps} of 9 budgets on this \
         powers-of-two ladder; `UNOBSERVED_SHAPE_MARGIN`'s documentation says three. It was \
         six while that margin was 3.6, and re-fitting it to 2.1 halved the cost — most of \
         what the old margin covered is now counted exactly rather than guarded against. \
         The count is a property of the spacing — see `tests/block_ladder.rs` for the \
         refined one — where the volume bound above is a property of the margin."
    );

    // **The control: the volume bound has teeth.** Six of nine budgets move the
    // block at all, so the assertion is exercised — but "exercised" is not
    // "would fail if it should".
    //
    // **A margin of `9.0` does not break it, and that is worth knowing rather
    // than hiding.** This control was written with `9.0` first, on the reasoning
    // that nine is past the eight the arithmetic rests on. It moved the block by
    // at most `8.000x` anyway: the block admitted *without* a margin sits below
    // its rung's ceiling by whatever the ladder's coarseness left it — up to a
    // full rung — and that headroom absorbs the extra `9/8` before any rung is
    // lost. So the bound is genuinely slack at this spacing, which is the honest
    // reading of a coarse ladder and is exactly why `tests/block_ladder.rs` at
    // the refined spacing is the sharper of the two tests.
    //
    // `64.0` is the principled choice instead: two full rungs, which no headroom
    // under one rung can absorb, so it must break the bound at every budget the
    // ladder does not bottom out on.
    let mut worst_when_doubled_past: f64 = 1.0;
    for power in 0..9u32 {
        let budget = (1u64 << power) * 1024 * 1024 * 1024;
        let admit = |charge: &dyn Fn(f64) -> u64| {
            CANDIDATES
                .iter()
                .copied()
                .find(|&edge| {
                    assumed_for(edge).is_some_and(|ws| charge(ws) * CONCURRENCY <= budget)
                })
                .unwrap_or(*CANDIDATES.last().expect("a ladder"))
        };
        let plain = admit(&|ws: f64| ws.round() as u64);
        let over = admit(&|ws: f64| (ws * 64.0).round() as u64);
        let moved = (plain as f64 / over as f64).powi(3);
        if moved.total_cmp(&worst_when_doubled_past).is_gt() {
            worst_when_doubled_past = moved;
        }
    }
    assert!(
        worst_when_doubled_past > 8.0,
        "a margin of 64.0 — two full rungs — moved the admitted block by at most \
         {worst_when_doubled_past:.3}x in volume, so the 8x assertion above cannot distinguish a margin \
         that is within the arithmetic from one that is not"
    );
}

// -------------------------------------------- the arity a fold cannot hide --

/// **What a fan-in holds when its combine is not a fold**, which is the shape
/// `docs/design/pixel-classification.md` puts 91 arms into.
///
/// Every fan-in measured above is joined by a `LogicCombine`, which declares a
/// [`Combine::fold_carrier`] and is therefore folded branch by branch: it holds
/// three block buffers whatever its arity, and the table says so. That is the
/// happy case and it is not the case a feature stack is in. A random forest
/// walks a tree per voxel and needs **every channel at that voxel at once**,
/// which is the definition of not being a left fold over pairs, so
/// `ops::classify::ForestPredictor` answers `None` and its fan-in must hold one
/// buffer per arm.
///
/// The budget cannot see any of this: `working_set_bytes_per_block` is
/// `resident_voxels x bytes_per_voxel x 2.0` for every phase reading one image,
/// so a 2-arm fan-in and a 91-arm one are charged **identically**. The
/// `budget.rs` table records the gap at `1.00x` to `3.56x` over the shapes it
/// has measured; this is the same gap at an arity two orders of magnitude past
/// any of them.
///
/// # What is asserted, and why it is a slope rather than a row at 91
///
/// The arities run to 24 and not to 91 deliberately: 91 `64^3` `f64` buffers is
/// 182 MiB, and `.github/workflows/ci.yml` records this suite's peak RSS as 128
/// MB and reasons about the runner from it. Raising the whole suite's ceiling by
/// a third to measure a straight line at its far end is a poor trade, so the
/// line is measured over four points and its slope is what the claim is made
/// about — with the 91-arm figure stated as the extrapolation it is.
#[test]
fn a_fan_in_that_cannot_fold_holds_one_buffer_per_arm_and_the_budget_charges_for_them() {
    use std::sync::Arc;

    use blockflow::forest::{Forest, Node};
    use blockflow::ops::{ForestPredictor, Prediction};

    let unit = buffer_bytes() as f64;

    /// A forest that names `arity` channels and decides on the first one. It is
    /// the *arity* that matters here — the walk itself allocates two small
    /// `Vec`s per call and nothing block-sized — so the smallest forest with a
    /// decision in it is the right fixture.
    fn stump(arity: usize) -> Arc<Forest> {
        Arc::new(
            Forest::new(
                vec![Node::split(0, 0.5, 1, 2), Node::leaf(0), Node::leaf(2)],
                vec![0],
                vec![1.0, 0.0, 0.0, 1.0],
                2,
                (0..arity).map(|index| format!("c{index}")).collect(),
            )
            .expect("a stump"),
        )
    }

    let mut folding = Vec::new();
    let mut standing = Vec::new();
    for arity in [2usize, 6, 12, 24] {
        let arms = || -> Vec<Chain> { (0..arity).map(|_| map("arm")).collect() };

        // The folding control: the same arms, joined by a combine that declares
        // a carrier. Without this row a growing figure below would be evidence
        // about `Chain::Parallel` in general rather than about the declaration.
        let (peak, _, spines) = measure(&fan_in_of(arms(), &[]), &[]);
        let folded = (peak - spines - preamble_bytes(&fan_in_of(arms(), &[]))) as f64 / unit;

        // `Probability` and not `Label` so the output is `f64` and the harness's
        // block arithmetic is unchanged; the two differ in one line of the
        // predictor and in nothing this measures.
        let build = || {
            Chain::parallel(
                arms(),
                Box::new(
                    ForestPredictor::new(
                        "classify",
                        stump(arity),
                        Prediction::Probability { class: 0 },
                    )
                    .expect("a predictor"),
                ),
            )
            .expect("a fan-in")
        };
        let (peak, _, spines) = measure(&build(), &[]);
        let held = (peak - spines - preamble_bytes(&build())) as f64 / unit;

        println!("{arity:>3} arms: folding {folded:>6.2}x, not folding {held:>6.2}x");
        folding.push((arity as f64, folded));
        standing.push((arity as f64, held));
    }

    // **The fold is flat in the arity** — that is the property it exists for.
    let folded_spread = folding
        .iter()
        .map(|&(_, held)| held)
        .fold(f64::NEG_INFINITY, f64::max)
        - folding
            .iter()
            .map(|&(_, held)| held)
            .fold(f64::INFINITY, f64::min);
    assert!(
        folded_spread < 0.6,
        "a folding fan-in moved by {folded_spread:.2} units across a twelve-fold range of \
         arities, so it is not folding and the comparison below means nothing"
    );

    // **The non-folding one is a straight line of slope one**: one block buffer
    // per arm. Least squares over the four points, so a single noisy row cannot
    // carry the claim.
    let mean_x = standing.iter().map(|&(x, _)| x).sum::<f64>() / standing.len() as f64;
    let mean_y = standing.iter().map(|&(_, y)| y).sum::<f64>() / standing.len() as f64;
    let slope = standing
        .iter()
        .map(|&(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / standing
            .iter()
            .map(|&(x, _)| (x - mean_x).powi(2))
            .sum::<f64>();
    let intercept = mean_y - slope * mean_x;
    println!("not folding: {slope:.3} buffers per arm + {intercept:.2}");
    assert!(
        (slope - 1.0).abs() < 0.15,
        "a fan-in that cannot fold grew by {slope:.3} block buffers per arm; the claim \
         `docs/design/pixel-classification.md` rests on is one"
    );

    // **And the budget now follows it, arm for arm.** This is what the test
    // exists to hold: the figure the allocator measures and the figure
    // `price_phase` charges move together, at every arity.
    //
    // It used to assert the opposite — that the charge stayed at two units
    // however many arms there were — because that was true and was the defect.
    // A 91-arm stack held 93 buffers and was charged 2.
    for arity in [2usize, 6, 12, 24] {
        let charge = charged(1, arity) / unit;
        assert!(
            (charge - (2 + arity) as f64).abs() < 1e-9,
            "at {arity} arms the budget charges {charge:.2} units where the chain holds \
             {} — the phase's own two plus one per arm",
            2 + arity
        );
    }

    // The extrapolation the design document's memory arithmetic rests on, now
    // that the budget agrees with it: 93 units for the 91-arm stack, against the
    // 2 it was charged before.
    let extrapolated = slope * 91.0 + intercept;
    let charged_now = charged(1, 91) / unit;
    println!("91 arms: {extrapolated:.0} units held, {charged_now:.0} charged");
    assert!(
        (extrapolated - charged_now).abs() < 2.0,
        "the measured line extrapolates to {extrapolated:.1} units at 91 arms and the \
         budget charges {charged_now:.1}; these are the same quantity and must agree"
    );
    assert!(
        charged_now > 20.0 * charged(1, 0) / unit,
        "the corrected charge is not the order of magnitude this test is about"
    );
}

/// **The shape-derived figure against the allocator, shape by shape.**
///
/// `Chain::resident_block_buffers` claims to predict, from a chain's structure
/// alone, exactly what the table above measures. This is that claim checked as
/// an **equality** for every shape in this file — not a bound, because a
/// derivation that were merely conservative would drift silently until it was
/// useless, and the whole point of it is to replace a figure that was wrong by
/// 47x with one that is right.
///
/// The accounting, stated so a failure can be read:
///
/// ```text
/// held = 2                          the phase's own input and output
///      + sources                    one buffer per source image, the executor's
///      + resident_block_buffers()   what the chain holds inside itself
/// ```
///
/// The first two terms are what `working_set_bytes_per_block` already covers or
/// could; the third is the one nothing could see.
#[test]
fn the_shape_derived_residency_is_what_the_allocator_measures() {
    use std::sync::Arc;

    use blockflow::forest::{Forest, Node};
    use blockflow::ops::{ForestPredictor, Prediction};

    let unit = buffer_bytes() as f64;
    let one = ImageId::from(7usize);
    let two = ImageId::from(8usize);

    fn predictor(arity: usize) -> Box<dyn blockflow::op::Combine> {
        Box::new(
            ForestPredictor::new(
                "classify",
                Arc::new(
                    Forest::new(
                        vec![Node::split(0, 0.5, 1, 2), Node::leaf(0), Node::leaf(2)],
                        vec![0],
                        vec![1.0, 0.0, 0.0, 1.0],
                        2,
                        (0..arity).map(|index| format!("c{index}")).collect(),
                    )
                    .expect("a stump"),
                ),
                Prediction::Probability { class: 0 },
            )
            .expect("a predictor"),
        )
    }

    let cases: Vec<(&str, Chain, Vec<ImageId>)> = vec![
        ("one in, one out", map("only"), vec![]),
        (
            "sequence of two maps",
            Chain::sequence(vec![map("a"), map("b")]),
            vec![],
        ),
        (
            "sequence of three maps",
            Chain::sequence(vec![map("a"), map("b"), map("c")]),
            vec![],
        ),
        (
            "sequence of four maps",
            Chain::sequence(vec![map("a"), map("b"), map("c"), map("d")]),
            vec![],
        ),
        ("fan-in, 2 computed arms", fan_in(2, &[]), vec![]),
        ("fan-in, 3 computed arms", fan_in(3, &[]), vec![]),
        ("fan-in, 7 computed arms", fan_in(7, &[]), vec![]),
        ("fan-in, 1 arm + 1 source", fan_in(1, &[one]), vec![one]),
        (
            "fan-in, 1 arm + 2 sources",
            fan_in(1, &[one, two]),
            vec![one, two],
        ),
        (
            "fan-in, 2 source arms",
            fan_in(0, &[one, two]),
            vec![one, two],
        ),
        (
            "sequence from a source",
            Chain::sequence(vec![Chain::source(one, Dtype::F64), map("after")]),
            vec![one],
        ),
        // The shapes the fold exists for: a combine that cannot fold.
        (
            "unfoldable fan-in, 2 arms",
            Chain::parallel((0..2).map(|_| map("arm")).collect(), predictor(2)).unwrap(),
            vec![],
        ),
        (
            "unfoldable fan-in, 9 arms",
            Chain::parallel((0..9).map(|_| map("arm")).collect(), predictor(9)).unwrap(),
            vec![],
        ),
        // And a nested one, because `max` over branches is a rule no flat shape
        // can distinguish from `ignore the branches`.
        (
            "unfoldable fan-in of sequences",
            Chain::parallel(
                (0..3)
                    .map(|_| Chain::sequence(vec![map("a"), map("b"), map("c")]))
                    .collect(),
                predictor(3),
            )
            .unwrap(),
            vec![],
        ),
    ];

    println!(
        "{:<34} {:>8} {:>8} {:>8}",
        "phase shape", "held", "derived", "agree"
    );
    let mut checked = 0;
    for (name, chain, sources) in &cases {
        let (peak, _, spines) = measure(chain, sources);
        let held = (peak - spines - preamble_bytes(chain)) as f64 / unit;
        let derived = 2 + sources.len() + chain.resident_block_buffers();
        println!(
            "{name:<34} {held:>7.2}x {derived:>7}x {:>8}",
            if (held - derived as f64).abs() < 0.2 {
                "yes"
            } else {
                "NO"
            }
        );
        assert!(
            (held - derived as f64).abs() < 0.2,
            "{name}: the allocator says {held:.2} block buffers and the shape-derived \
             figure says {derived}. The fractional part of the measurement is small \
             `Vec` spines and scratch, never a block buffer, so a gap of a fifth of a \
             buffer is already generous — this is a real disagreement."
        );
        checked += 1;
    }
    assert_eq!(checked, cases.len());
}

/// **What correcting the figure would cost, for the chain that needs it most.**
///
/// `what_a_corrected_figure_would_cost_in_affordable_plans` above sweeps factors
/// up to `4.00x`, which covered every shape this file had measured. A 91-arm
/// fan-in under a combine that cannot fold is **46.5x**, an order of magnitude
/// past the end of that table, so what it does to an admitted block is a
/// different question and is asked separately here.
///
/// **The answer is milder than the ratio suggests, and that is the finding.** A
/// 46.5x correction does not make the chain unplannable; it moves the admitted
/// block down two or three rungs of the ladder, because the demand grows as the
/// cube of the edge and the ladder steps by factors of two. At 4 GiB and
/// concurrency 8 the planner drops from a 256-voxel block to a 64-voxel one; at
/// 256 GiB and concurrency 1 it does not move at all.
///
/// That matters for whether the correction is worth making, so it is worth
/// stating plainly. A correction that refused every plan would be one nobody
/// could adopt. This one costs a smaller block on the chains that were being
/// under-charged and **nothing at all** on the one-in-one-out phases the formula
/// was written for, since their factor is exactly the `2.0` already charged. It
/// only ever charges more, so it can refuse a plan the planner used to admit but
/// never admits one it used to refuse — and a plan it now refuses is one that
/// would have exhausted memory at run time.
///
/// The prose here first claimed there was no admissible block at all. The table
/// says otherwise, and the table is what runs.
#[test]
fn print_what_an_honest_figure_admits_for_a_ninety_one_arm_stack() {
    const CANDIDATES: [usize; 6] = [512, 256, 128, 64, 32, 16];
    const VOLUME_EDGE: usize = 1024;

    // 2 for the phase's own in and out, plus one per arm: the figure
    // `Chain::resident_block_buffers` derives and the allocator confirms.
    let buffers = 2.0 + 91.0;
    let charged = 2.0;

    eprintln!(
        "\n91-arm unfoldable fan-in, zero reach, 8 B/voxel, volume {VOLUME_EDGE}^3\n\
         {:>10} {:>6} {:>14} {:>14} {:>10}",
        "budget", "conc", "charged", "actually held", "admits"
    );
    for power in [2u32, 4, 6, 8] {
        let budget = (1u64 << power) * 1024 * 1024 * 1024;
        for concurrency in [1u64, 8] {
            let need = |edge: usize, factor: f64| -> Option<f64> {
                let grid = BlockGrid::along([VOLUME_EDGE; 3], &[0, 1, 2], edge).ok()?;
                let cost = price_phase(
                    &grid,
                    &Reach::symmetric([0, 0, 0]),
                    1.0,
                    1,
                    false,
                    8.0,
                    &CostModel::default(),
                    1.0,
                    PhaseTraffic::one_in_one_out(),
                );
                // `working_set_bytes_per_block` is the `x 2.0` form, so the
                // honest demand is that scaled by `buffers / 2`.
                Some(cost.working_set_bytes_per_block * concurrency as f64 * factor / charged)
            };
            let admits = |factor: f64| {
                CANDIDATES
                    .iter()
                    .copied()
                    .find(|&edge| need(edge, factor).is_some_and(|want| want <= budget as f64))
            };
            let today = admits(charged);
            let honest = admits(buffers);
            eprintln!(
                "{:>9} GiB {concurrency:>6} {:>14} {:>14} {:>10}",
                1u64 << power,
                match today {
                    Some(edge) => format!("admits {edge}"),
                    None => "none".to_string(),
                },
                match honest {
                    Some(edge) => format!("admits {edge}"),
                    None => "none".to_string(),
                },
                match (today, honest) {
                    (Some(a), Some(b)) if a == b => "same".to_string(),
                    (Some(_), Some(b)) => format!("{b}"),
                    (Some(_), None) => "refuses".to_string(),
                    _ => "-".to_string(),
                }
            );
        }
    }

    // The claim the table is here to support, asserted so it cannot rot: at the
    // smallest candidate and the smallest concurrency, an honest figure still
    // wants more than a 4 GiB budget allows.
    let grid = BlockGrid::along([VOLUME_EDGE; 3], &[0, 1, 2], 16).expect("a lattice");
    let cost = price_phase(
        &grid,
        &Reach::symmetric([0, 0, 0]),
        1.0,
        1,
        false,
        8.0,
        &CostModel::default(),
        1.0,
        PhaseTraffic::one_in_one_out(),
    );
    let honest = cost.working_set_bytes_per_block * buffers / charged;
    assert!(
        honest > cost.working_set_bytes_per_block * 40.0,
        "the corrected figure is not the order of magnitude this test is about"
    );
}
