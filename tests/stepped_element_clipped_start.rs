// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A decimated element whose stride is counted **from the clipped start of the
// window** — `StepOrigin::ClippedStart` — which is the other of the two
// conventions `tests/stepped_element.rs` holds the first one to.
//
// The difference, in one line
// ---------------------------
// `a[max(0, c - lo) : min(c + hi + 1, n) : step]`. The stride starts where the
// *clipped* range starts, so an anchor closer to the volume's low face than
// `lo` re-phases the decimation and reads a different residue class of the
// volume than the anchored lattice names. Everywhere else — every interior
// anchor, and every high face, where the clip takes members off the far end
// without moving the ones that remain — the two conventions are the same set.
//
// **Where a window is wider than the volume, "everywhere else" is nowhere.**
// Every anchor on that axis is inside `lo` of the low face, so every anchor
// re-phases, and the two conventions disagree at all of them but the ones where
// the shortfall happens to be a multiple of the step. That is not an edge case
// to be swept: it is the ordinary situation for the large flat windows a
// decimation exists for.
//
// What is asserted, and in what order the claims depend on each other
// -------------------------------------------------------------------
// 1. **The offsets, hand-computed, anchor by anchor** on a seven-voxel axis
//    under a nine-voxel window stepped by two — a case chosen so the two
//    conventions give *different* sets at two of the seven anchors and the same
//    set at the other five. A window that fits inside the volume with room to
//    spare, or a stride that divides the shortfall, agrees under both and would
//    prove nothing.
// 2. **The same case through the op**, as numbers rather than as offsets: the
//    values are powers of two, so a window's mean names the set that produced it
//    and the expected column can be written down by hand.
// 3. **The reach is what the widest phase reaches**, which is a *different*
//    number from the widest interior offset — the assertion
//    `tests/stepped_element.rs` makes for the anchored element, and the one it
//    would be wrong to carry over unchanged.
// 4. **Decomposition invariance**, byte for byte, over every block size and
//    split — the property this crate exists for, on an element whose membership
//    depends on where it is evaluated.
// 5. **The phase is keyed on the volume's face and not on a block's**, which is
//    the subtle half of (4). Asserted as the negative control it needs: the same
//    arithmetic handed a block's extent instead of the volume's answers
//    differently, so the wrong answer is reachable and (4) is not a tautology.
// 6. **The two origins are different filters here**, so (4) is not an invariance
//    sweep over an element that quietly behaves like the anchored one.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::{
    ElementShape, LocalStatistic, LocalStatisticOp, Sampling, Statistic, StepOrigin,
    StructuringElement,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

// ------------------------------------------------------- the hand case --

/// Seven voxels on the axis under test and one on each of the others, so that
/// the whole element is one axis and every number below can be written out.
const HAND: [usize; 3] = [7, 1, 1];

/// Nine voxels of window on a seven-voxel axis: `lo = 4`, `hi = 4`, and **every**
/// anchor is inside `lo` of the low face. This is the consumer's situation
/// — a window wider than the volume — at a size a reader can check.
const HAND_SIZE: [usize; 3] = [9, 1, 1];
const HAND_STEP: [usize; 3] = [2, 1, 1];

fn clipped(shape: ElementShape, size: [usize; 3], step: [usize; 3]) -> StructuringElement {
    StructuringElement::from_size_stepped_at(shape, size, step, StepOrigin::ClippedStart).unwrap()
}

fn anchored(shape: ElementShape, size: [usize; 3], step: [usize; 3]) -> StructuringElement {
    StructuringElement::from_size_stepped_at(shape, size, step, StepOrigin::Anchor).unwrap()
}

/// The offsets along axis 0 at `anchor`, which is all there is to these
/// elements: the other two axes hold offset zero and nothing else.
fn along(element: &StructuringElement, anchor: isize, volume: [usize; 3]) -> Vec<isize> {
    let mut scratch = Vec::new();
    element
        .offsets_at([anchor, 0, 0], volume, &mut scratch)
        .iter()
        .map(|offset| {
            assert_eq!([offset[1], offset[2]], [0, 0], "a flat element");
            offset[0]
        })
        .collect()
}

/// The anchored element's members that still land inside the volume at `anchor`
/// — what the *other* convention reads there, computed here from its own
/// definition rather than asked of anything that could share a bug with the
/// implementation under test.
fn anchored_survivors(
    element: &StructuringElement,
    anchor: isize,
    volume: [usize; 3],
) -> Vec<isize> {
    element
        .offsets()
        .iter()
        .map(|offset| offset[0])
        .filter(|offset| {
            let coordinate = anchor + offset;
            coordinate >= 0 && coordinate < volume[0] as isize
        })
        .collect()
}

/// **The table, computed by hand.** `n = 7`, `lo = hi = 4`, `step = 2`, so the
/// window is `[max(0, c - 4), min(c + 5, 7))` strided by two from its own start:
///
/// | anchor | clipped range | coordinates read | offsets |
/// |---:|---|---|---|
/// | 0 | `[0, 5)` | 0, 2, 4 | 0, 2, 4 |
/// | 1 | `[0, 6)` | 0, 2, 4 | -1, 1, 3 |
/// | 2 | `[0, 7)` | 0, 2, 4, 6 | -2, 0, 2, 4 |
/// | 3 | `[0, 7)` | 0, 2, 4, 6 | -3, -1, 1, 3 |
/// | 4 | `[0, 7)` | 0, 2, 4, 6 | -4, -2, 0, 2 |
/// | 5 | `[1, 7)` | 1, 3, 5 | -4, -2, 0 |
/// | 6 | `[2, 7)` | 2, 4, 6 | -4, -2, 0 |
///
/// The anchored element reads `{-4, -2, 0, 2, 4}` wherever they land inside, so
/// its coordinates are `c`'s own parity throughout: 0, 2, 4 at `c = 0`; **1, 3,
/// 5** at `c = 1`; 0, 2, 4, 6 at `c = 2`; **1, 3, 5** at `c = 3`; and the same
/// as the clipped rule at 4, 5 and 6.
///
/// So the two conventions differ at exactly `c = 1` and `c = 3` — the anchors
/// whose shortfall `lo - c` is odd — and agree at the other five. Both halves
/// are asserted: a case that only differed would not show that the rules share
/// an interior, and one that only agreed would not discriminate at all.
#[test]
fn the_offsets_are_the_clipped_ranges_stride_anchor_by_anchor() {
    let element = clipped(ElementShape::Box, HAND_SIZE, HAND_STEP);
    let other = anchored(ElementShape::Box, HAND_SIZE, HAND_STEP);

    let by_hand: [(isize, &[isize]); 7] = [
        (0, &[0, 2, 4]),
        (1, &[-1, 1, 3]),
        (2, &[-2, 0, 2, 4]),
        (3, &[-3, -1, 1, 3]),
        (4, &[-4, -2, 0, 2]),
        (5, &[-4, -2, 0]),
        (6, &[-4, -2, 0]),
    ];
    let mut differed = Vec::new();
    for (anchor, want) in by_hand {
        assert_eq!(along(&element, anchor, HAND), want, "anchor {anchor}");
        // and the coordinates really are inside the volume, which is the half
        // of the rule the offsets alone do not show
        for offset in want {
            let coordinate = anchor + offset;
            assert!(
                (0..HAND[0] as isize).contains(&coordinate),
                "anchor {anchor} offset {offset} reads {coordinate}"
            );
        }
        if along(&element, anchor, HAND) != anchored_survivors(&other, anchor, HAND) {
            differed.push(anchor);
        }
    }
    assert_eq!(
        differed,
        vec![1, 3],
        "the two conventions must differ at the odd shortfalls and nowhere else"
    );
    // Disjoint, not merely unequal, at those two: the whole window moved to the
    // other parity of the volume rather than gaining or losing an end. Compared
    // as **voxels**, since that is what the two rules disagree about.
    for anchor in [1, 3] {
        let ours: Vec<isize> = along(&element, anchor, HAND)
            .iter()
            .map(|offset| anchor + offset)
            .collect();
        let theirs: Vec<isize> = anchored_survivors(&other, anchor, HAND)
            .iter()
            .map(|offset| anchor + offset)
            .collect();
        assert!(
            ours.iter().all(|coordinate| !theirs.contains(coordinate)),
            "anchor {anchor}: {ours:?} and {theirs:?} share a voxel"
        );
    }

    // The interior of a *wider* volume is the anchored element exactly, which is
    // what makes this a boundary rule and not a different filter everywhere.
    let wide = [40usize, 1, 1];
    for anchor in 4..36 {
        assert_eq!(
            along(&element, anchor, wide),
            along(&other, anchor, wide),
            "anchor {anchor} of a volume that clips nothing"
        );
    }
}

// --------------------------------------------------- the same, as values --

/// Powers of two along axis 0, so that the mean of a window names the set of
/// voxels that produced it and nothing else could have.
fn powers() -> Array3<f64> {
    Array3::from_shape_fn((HAND[0], HAND[1], HAND[2]), |(i, _, _)| (1u64 << i) as f64)
}

fn mean_statistic(element: StructuringElement) -> LocalStatistic {
    // A spacing of one is the no-lattice case: a sample at every voxel and an
    // interpolation that is the identity, so what comes out is the window's own
    // statistic at that voxel and not a blend of two.
    LocalStatistic::sampled(element, Sampling::every([1, 1, 1]), Statistic::Mean).unwrap()
}

fn evaluate(element: StructuringElement, volume: [usize; 3], input: &Array3<f64>) -> Vec<f64> {
    let mut out = Array3::<f64>::zeros(input.raw_dim());
    mean_statistic(element)
        .evaluate_into(input.view(), &Anchor::whole(volume), out.view_mut())
        .expect("the statistic must run");
    (0..volume[0]).map(|i| out[[i, 0, 0]]).collect()
}

/// The table above, arrived at through the op rather than through the element.
///
/// With `v[i] = 2^i` the mean is `sum / count`, and every value here is exact in
/// `f64`:
///
/// | anchor | clipped | sum / count | anchored | sum / count |
/// |---:|---|---:|---|---:|
/// | 0 | 1 + 4 + 16 | 7 | same | 7 |
/// | 1 | 1 + 4 + 16 | 7 | 2 + 8 + 32 | 14 |
/// | 2 | 1 + 4 + 16 + 64 | 21.25 | same | 21.25 |
/// | 3 | 1 + 4 + 16 + 64 | 21.25 | 2 + 8 + 32 | 14 |
/// | 4 | 1 + 4 + 16 + 64 | 21.25 | same | 21.25 |
/// | 5 | 2 + 8 + 32 | 14 | same | 14 |
/// | 6 | 4 + 16 + 64 | 28 | same | 28 |
#[test]
fn the_op_reads_the_hand_computed_window() {
    let input = powers();
    let ours = evaluate(
        clipped(ElementShape::Box, HAND_SIZE, HAND_STEP),
        HAND,
        &input,
    );
    let theirs = evaluate(
        anchored(ElementShape::Box, HAND_SIZE, HAND_STEP),
        HAND,
        &input,
    );
    assert_eq!(ours, vec![7.0, 7.0, 21.25, 21.25, 21.25, 14.0, 28.0]);
    assert_eq!(theirs, vec![7.0, 14.0, 21.25, 14.0, 21.25, 14.0, 28.0]);
    // stated as a difference too, so that a change to either column has to be
    // deliberate about which convention it is changing
    let differing: Vec<usize> = (0..HAND[0]).filter(|&i| ours[i] != theirs[i]).collect();
    assert_eq!(differing, vec![1, 3]);
}

// ------------------------------------------------------------- the reach --

/// The reach is **what the widest phase reaches**, which is not the widest
/// interior offset.
///
/// An eight-wide axis anchors at `(4, 3)` and the interior window strides from
/// `-4` to `2`, stranding the far pole — the fact `tests/stepped_element.rs`
/// pins for the anchored element. Under this convention the far pole is not
/// stranded: at `c = 3` the window strides from the clipped start `0` and its
/// last coordinate is `c + 3`, an offset of `+3`. An element that carried the
/// anchored `(4, 2)` here would understate its own dependency by a plane, which
/// is the one direction a derived reach must never be wrong in.
#[test]
fn the_reach_is_what_the_widest_phase_reaches_and_not_the_interior_span() {
    let element = clipped(ElementShape::Box, [8, 8, 1], [2, 2, 1]);
    let other = anchored(ElementShape::Box, [8, 8, 1], [2, 2, 1]);
    assert_eq!(other.sides(0), (4, 2), "the anchored element, unchanged");
    assert_eq!(element.sides(0), (4, 3));
    assert_eq!(element.sides(1), (4, 3));
    assert_eq!(element.sides(2), (0, 0));
    assert_eq!(element.size(), [8, 8, 1], "the box it was asked for");
    // the interior offsets are the anchored ones, so `len` and every cost built
    // on it are unchanged
    assert_eq!(element.len(), other.len());
    assert_eq!(element.offsets(), other.offsets());

    // brute force, over every anchor of a volume wide enough to clip nothing at
    // the far end: `+3` is really reached, and nothing wider is
    let volume = [64usize, 64, 1];
    let mut widest_above = 0isize;
    let mut widest_below = 0isize;
    let mut reached_three = false;
    let mut scratch = Vec::new();
    for anchor in 0..volume[0] as isize {
        for offset in element.offsets_at([anchor, 32, 0], volume, &mut scratch) {
            widest_above = widest_above.max(offset[0]);
            widest_below = widest_below.min(offset[0]);
            reached_three |= offset[0] == 3;
        }
    }
    assert!(reached_three, "the far pole must be reached at some anchor");
    assert_eq!((widest_below, widest_above), (-4, 3));

    // and the shape the consumer asks for: a 200-wide axis stepped by two
    // reaches 99 above the anchor where the anchored lattice stops at 98
    let wide = clipped(ElementShape::Box, [200, 200, 1], [2, 2, 1]);
    assert_eq!(wide.sides(0), (100, 99));
    assert_eq!(
        anchored(ElementShape::Box, [200, 200, 1], [2, 2, 1]).sides(0),
        (100, 98)
    );
    assert_eq!(wide.len(), 100 * 100, "a quarter of the box, as before");

    // a step of one is the unstepped element under either origin, value and all
    assert_eq!(
        clipped(ElementShape::Box, [9, 7, 1], [1, 1, 1]),
        StructuringElement::from_size(ElementShape::Box, [9, 7, 1]).unwrap()
    );
    assert_eq!(
        clipped(ElementShape::Box, [9, 7, 1], [1, 1, 1]).origin(),
        StepOrigin::Anchor,
        "with nothing to re-phase the two origins are one origin"
    );
}

// --------------------------------------------- decomposition invariance --

/// Long enough on axis 0 for several blocks, and **narrower there than the
/// window below**, so that every anchor on that axis re-phases and no block can
/// contain only interior samples.
const VOLUME: [usize; 3] = [24, 6, 4];
const ELEMENT: [usize; 3] = [11, 3, 1];
const STEP: [usize; 3] = [2, 2, 1];
/// Coarse enough that the lattice term of the reach is real, and not a divisor
/// of the volume, so the samples do not sit at the same place in every block.
const SPACING: [usize; 3] = [5, 3, 2];

fn image() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        // deterministic, not smooth, and different on each axis, so a window
        // that moved by one voxel cannot give the same statistic by accident
        let value = (i * 37 + j * 11 + k * 5) % 29;
        value as f64 + (i % 2) as f64 * 0.5
    })
}

fn op(element: StructuringElement) -> LocalStatisticOp {
    LocalStatisticOp::new("clipped-start", sampled(element))
}

fn sampled(element: StructuringElement) -> LocalStatistic {
    LocalStatistic::sampled(element, Sampling::every(SPACING), Statistic::Mean).unwrap()
}

fn whole_volume(element: StructuringElement, input: &Array3<f64>) -> Array3<f64> {
    let chain = Chain::op(op(element));
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn blocked(element: StructuringElement, grid: &BlockGrid, input: &Array3<f64>) -> Array3<f64> {
    let chain = Chain::op(op(element));
    let slots = chain.slots();
    let reach = chain.reach_spec(VOLUME).expect("a foldable reach");
    let phase = PhaseDecomposition::derive(
        (0..slots.len()).collect(),
        slots.iter().map(|slot| slot.display_name()).collect(),
        reach.clone(),
        reach,
        grid.clone(),
    );
    let decomposition = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: chain.reach3(&VOLUME),
    };
    decomposition.check().expect("an honest plan must tile");
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    execute(
        "clipped-start",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<f64>().unwrap().to_owned()
}

fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0], 7).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[1], 2).unwrap(),
        BlockGrid::along(VOLUME, &[2], 3).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 3).unwrap(),
    ]
}

/// Byte-identical to the whole-volume answer under every decomposition, on an
/// element whose membership depends on where it is evaluated.
#[test]
fn every_decomposition_gives_the_whole_volume_answer() {
    let input = image();
    let element = clipped(ElementShape::Box, ELEMENT, STEP);
    let want = whole_volume(element.clone(), &input);
    let mut checked = 0;
    for grid in grids() {
        let got = blocked(element.clone(), &grid, &input);
        let differing = got
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "block {:?} disagreed with the whole-volume answer at {differing} voxels",
            grid.block()
        );
        checked += 1;
    }
    assert_eq!(checked, 8);
}

/// **The phase is the volume's, not the block's** — the negative control the
/// sweep above needs.
///
/// The re-phasing rule is keyed on the distance to a face, and a block has faces
/// the volume does not. Handing the arithmetic a block's extent instead of the
/// volume's is therefore a reachable mistake and not a hypothetical one, and
/// this asserts that it is reachable: the same anchor, expressed in a block's
/// own coordinates against the block's own extent, produces a *different* set of
/// offsets. The sweep above then says that the op does not make it.
///
/// The plan check is what keeps the two from differing in practice as well as in
/// principle: a halo at least the element's low side means a window clipped at a
/// buffer's low edge is a window clipped at the volume's, for every sample whose
/// value a core voxel reads. That is an argument, and the sweep is the
/// measurement — which is the right way round, because the argument holds only
/// while the halo does.
#[test]
fn the_phase_is_keyed_on_the_volumes_face_and_not_a_blocks() {
    let element = clipped(ElementShape::Box, HAND_SIZE, HAND_STEP);
    // A block of `[3, 7)` inside a `24`-voxel volume, and an anchor at volume
    // coordinate 5 — comfortably interior, so the volume's rule does not clip at
    // all.
    let volume = [24usize, 1, 1];
    let block_offset = 3isize;
    let block = [4usize, 1, 1];
    let anchor = 5isize;

    let by_the_volume = along(&element, anchor, volume);
    assert_eq!(
        by_the_volume,
        vec![-4, -2, 0, 2, 4],
        "an interior anchor is the anchored lattice"
    );

    // The same call with the block's extent and the block's own coordinate for
    // the anchor: what an implementation that mistook the buffer for the volume
    // would compute.
    // Its window is `[max(0, 2 - 4), min(2 + 5, 4)) = [0, 4)` strided by two, so
    // it reads the block's voxels 0 and 2 and calls them offsets -2 and 0.
    let by_the_block = along(&element, anchor - block_offset, block);
    assert_eq!(by_the_block, vec![-2, 0], "the mistake, computed");
    assert_ne!(
        by_the_block, by_the_volume,
        "if these agreed there would be nothing for the sweep to be right about"
    );

    // And it is not only the flat hand element: the sweep's own element, at an
    // anchor a block genuinely holds, moves too.
    let sweep = clipped(ElementShape::Box, ELEMENT, STEP);
    let mut theirs = Vec::new();
    let mut ours = Vec::new();
    assert_ne!(
        sweep.offsets_at([9, 3, 1], [8, 6, 4], &mut theirs).to_vec(),
        sweep.offsets_at([9, 3, 1], VOLUME, &mut ours).to_vec()
    );
}

/// The sweep is not an invariance test of the anchored element wearing another
/// name: the two origins compute **different volumes** here, at a large fraction
/// of the voxels.
#[test]
fn the_two_origins_are_different_filters_on_this_volume() {
    let input = image();
    let ours = whole_volume(clipped(ElementShape::Box, ELEMENT, STEP), &input);
    let theirs = whole_volume(anchored(ElementShape::Box, ELEMENT, STEP), &input);
    let differing = ours
        .iter()
        .zip(theirs.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let total = VOLUME[0] * VOLUME[1] * VOLUME[2];
    assert!(
        differing * 4 > total,
        "the two origins must be materially different filters here, differed at \
         {differing} of {total}"
    );
}
